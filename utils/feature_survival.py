"""Training-free, geometry-calibrated feature survival diagnostics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class TranslationControlSet:
    component_mask: np.ndarray
    all_target_mask: np.ndarray
    guarded_target_mask: np.ndarray
    translated_masks: tuple[np.ndarray, ...]
    sample_key: str


CONTEXT_DESCRIPTOR_NAMES = (
    "median",
    "mad",
    "gradient_energy",
    "laplacian_energy",
    "entropy",
)


@dataclass(frozen=True)
class ContextControlMatch:
    """One selected context control with an auditable ranking record."""

    source_index: int
    component_mask: np.ndarray
    mask_digest: str
    descriptor: tuple[float, ...]
    mahalanobis_distance: float
    ring_pixels: int
    stencil_pixels: int
    ring_coverage: float


@dataclass(frozen=True)
class ContextMatchedControlSelection:
    """Fail-closed result of target-exterior context matching."""

    available: bool
    reason: str | None
    control_set: TranslationControlSet | None
    descriptor_names: tuple[str, ...]
    target_descriptor: tuple[float, ...] | None
    descriptor_center: tuple[float, ...] | None
    descriptor_scale: tuple[float, ...] | None
    active_descriptor_mask: tuple[bool, ...] | None
    regularized_covariance: np.ndarray | None
    covariance_condition_number: float | None
    context_distance_caliper: float | None
    target_ring_pixels: int
    target_stencil_pixels: int
    target_ring_coverage: float | None
    eligible_candidate_count: int
    selected: tuple[ContextControlMatch, ...]
    rejected_candidate_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ProjectedFootprint:
    occupancy: np.ndarray
    background_flat_indices: np.ndarray


@dataclass(frozen=True)
class ProjectedGeometry:
    output_shape: tuple[int, int]
    target: ProjectedFootprint
    controls: tuple[ProjectedFootprint, ...]
    target_effective_cells: float
    target_max_occupancy: float


@dataclass(frozen=True)
class FeatureSurvivalResult:
    available: bool
    reason: str | None
    state: str
    rank: float | None
    robust_effect: float | None
    observed_score: float | None
    null_q05: float | None
    null_median: float | None
    null_q95: float | None
    null_max: float | None
    num_controls: int
    target_effective_cells: float
    target_max_occupancy: float
    target_background_cells: int
    directional_auc: float | None
    target_peak: float | None
    target_peak_margin: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _binary_mask(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not array.size:
        raise ValueError("%s must be a non-empty 2-D mask" % name)
    if not np.logical_or(array == 0, array == 1).all():
        raise ValueError("%s must be binary" % name)
    return array.astype(bool, copy=False)


def _hash_order(key: str, *values: int) -> bytes:
    payload = "%s\0%s" % (key, "\0".join(str(value) for value in values))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _translate_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray | None:
    coordinates = np.argwhere(mask)
    moved = coordinates + np.asarray((dy, dx), dtype=np.int64)
    height, width = mask.shape
    if (
        np.any(moved[:, 0] < 0)
        or np.any(moved[:, 0] >= height)
        or np.any(moved[:, 1] < 0)
        or np.any(moved[:, 1] >= width)
    ):
        return None
    result = np.zeros_like(mask)
    result[moved[:, 0], moved[:, 1]] = True
    return result


def build_translation_control_set(
    component_mask,
    all_target_mask,
    *,
    sample_key: str,
    guard_radius: float = 3.0,
    min_translation_radius: float = 8.0,
    max_translation_radius: float = 96.0,
    max_candidate_controls: int = 256,
) -> TranslationControlSet:
    """Create deterministic same-shape background translations in image space."""

    component = _binary_mask(component_mask, name="component_mask")
    all_targets = _binary_mask(all_target_mask, name="all_target_mask")
    if component.shape != all_targets.shape:
        raise ValueError("component and all-target masks must share a shape")
    if not component.any() or np.any(component & ~all_targets):
        raise ValueError("component must be a non-empty subset of all targets")
    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("sample_key must be a non-empty string")
    numeric = (guard_radius, min_translation_radius, max_translation_radius)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in numeric):
        raise ValueError("radii must be finite and non-negative")
    if not 0 <= min_translation_radius < max_translation_radius:
        raise ValueError("translation radii must satisfy 0 <= min < max")
    if not isinstance(max_candidate_controls, int) or max_candidate_controls < 1:
        raise ValueError("max_candidate_controls must be a positive integer")

    distance_to_targets = distance_transform_edt(~all_targets)
    guarded_targets = distance_to_targets <= float(guard_radius)
    extent = int(math.ceil(max_translation_radius))
    min_squared = float(min_translation_radius) ** 2
    max_squared = float(max_translation_radius) ** 2
    offsets = [
        (dy, dx)
        for dy in range(-extent, extent + 1)
        for dx in range(-extent, extent + 1)
        if (dy != 0 or dx != 0)
        and min_squared <= dy * dy + dx * dx <= max_squared
    ]
    offsets.sort(key=lambda item: _hash_order(sample_key, item[0], item[1]))

    translated = []
    for dy, dx in offsets:
        candidate = _translate_mask(component, dy, dx)
        if candidate is None or np.any(candidate & guarded_targets):
            continue
        translated.append(candidate)
        if len(translated) == max_candidate_controls:
            break
    return TranslationControlSet(
        component_mask=component.copy(),
        all_target_mask=all_targets.copy(),
        guarded_target_mask=guarded_targets,
        translated_masks=tuple(translated),
        sample_key=sample_key,
    )


@dataclass(frozen=True)
class _ContextRingStatistics:
    descriptor_without_entropy: tuple[float, float, float, float]
    values: np.ndarray
    ring_pixels: int
    stencil_pixels: int
    coverage: float


def _context_ring_statistics(
    image: np.ndarray,
    footprint: np.ndarray,
    protection_mask: np.ndarray,
    *,
    inner_radius: float,
    ring_width: float,
    minimum_ring_pixels: int,
    minimum_stencil_pixels: int,
    minimum_ring_coverage: float,
) -> _ContextRingStatistics | None:
    """Measure context without accessing the footprint or protected pixels."""

    distance = distance_transform_edt(~footprint)
    geometric_ring = (distance > inner_radius) & (
        distance <= inner_radius + ring_width
    )
    geometric_count = int(geometric_ring.sum())
    if geometric_count == 0:
        return None
    ring = geometric_ring & ~footprint & ~protection_mask
    ring_count = int(ring.sum())
    coverage = ring_count / geometric_count
    if ring_count < minimum_ring_pixels or coverage < minimum_ring_coverage:
        return None

    # Central differences and the four-neighbour Laplacian are evaluated only
    # where their complete stencils are exterior to both the footprint and the
    # protection mask. This makes the result invariant to every protected
    # intensity, including the target itself.
    exterior = ~footprint & ~protection_mask
    stencil = np.zeros_like(exterior)
    stencil[1:-1, 1:-1] = (
        ring[1:-1, 1:-1]
        & exterior[:-2, 1:-1]
        & exterior[2:, 1:-1]
        & exterior[1:-1, :-2]
        & exterior[1:-1, 2:]
    )
    coordinates = np.argwhere(stencil)
    if coordinates.shape[0] < minimum_stencil_pixels:
        return None
    rows = coordinates[:, 0]
    columns = coordinates[:, 1]
    center_values = image[rows, columns]
    north = image[rows - 1, columns]
    south = image[rows + 1, columns]
    west = image[rows, columns - 1]
    east = image[rows, columns + 1]
    gradient_y = 0.5 * (south - north)
    gradient_x = 0.5 * (east - west)
    laplacian = north + south + west + east - 4.0 * center_values

    values = image[ring].astype(np.float64, copy=True)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    gradient_energy = float(np.mean(gradient_x**2 + gradient_y**2))
    laplacian_energy = float(np.mean(laplacian**2))
    statistics = np.asarray(
        (median, mad, gradient_energy, laplacian_energy),
        dtype=np.float64,
    )
    if not np.isfinite(statistics).all():
        raise RuntimeError("context descriptor produced non-finite statistics")
    return _ContextRingStatistics(
        descriptor_without_entropy=tuple(float(value) for value in statistics),
        values=values,
        ring_pixels=ring_count,
        stencil_pixels=int(coordinates.shape[0]),
        coverage=float(coverage),
    )


def _histogram_entropy(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    bins: int,
) -> float:
    if not upper > lower:
        return 0.0
    clipped = np.clip(values, lower, upper)
    counts = np.histogram(clipped, bins=bins, range=(lower, upper))[0]
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    entropy = -np.sum(probabilities * np.log(probabilities)) / math.log(bins)
    return float(entropy)


def _context_mask_digest(mask: np.ndarray, *, selection_key: str) -> str:
    digest = hashlib.sha256()
    digest.update(selection_key.encode("utf-8"))
    digest.update(b"\0context-control-v1\0")
    digest.update(np.asarray(mask.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(mask.reshape(-1)).tobytes())
    return digest.hexdigest()


def _is_exact_translation(reference: np.ndarray, candidate: np.ndarray) -> bool:
    """Return whether ``candidate`` is a rigid integer translation of ``reference``."""

    reference_coordinates = np.argwhere(reference)
    candidate_coordinates = np.argwhere(candidate)
    if reference_coordinates.shape != candidate_coordinates.shape:
        return False
    offsets = candidate_coordinates - reference_coordinates
    return bool(np.all(offsets == offsets[0]))


def select_context_matched_controls(
    image,
    controls: TranslationControlSet,
    *,
    protection_mask=None,
    context_inner_radius: float = 3.0,
    context_ring_width: float = 8.0,
    num_controls: int = 64,
    minimum_ring_pixels: int = 32,
    minimum_stencil_pixels: int = 16,
    minimum_ring_coverage: float = 0.8,
    histogram_bins: int = 16,
    covariance_shrinkage: float = 0.2,
    covariance_eigenvalue_floor: float = 1e-6,
    maximum_covariance_condition: float = 1e8,
    minimum_covariance_candidates: int = 12,
    candidate_support_quantile: float = 0.95,
    maximum_selected_iou: float = 0.0,
) -> ContextMatchedControlSelection:
    """Select deterministic context-matched same-shape background controls.

    Only rings exterior to the target/control footprint and to every protected
    pixel contribute to the descriptors. Descriptor location and scale are
    estimated from eligible background candidates only. A shrunk, eigenvalue-
    floored covariance defines the Mahalanobis ranking. The target footprint's
    intensities cannot affect candidate eligibility, descriptors, scaling, or
    ranking.

    The returned selection is separate from ``controls`` so callers can retain
    the original geometry-control cohort alongside this matched subset.
    """

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or not array.size:
        raise ValueError("image must be a non-empty 2-D array")
    if not np.isfinite(array).all():
        raise ValueError("image must contain only finite values")
    if not isinstance(controls, TranslationControlSet):
        raise TypeError("controls must be a TranslationControlSet")

    component = _binary_mask(controls.component_mask, name="component_mask")
    all_targets = _binary_mask(controls.all_target_mask, name="all_target_mask")
    guarded_targets = _binary_mask(
        controls.guarded_target_mask,
        name="guarded_target_mask",
    )
    if not (
        component.shape
        == all_targets.shape
        == guarded_targets.shape
        == array.shape
    ):
        raise ValueError("image and control masks must share a shape")
    if not component.any() or np.any(component & ~all_targets):
        raise ValueError("component must be a non-empty subset of all targets")
    if np.any(all_targets & ~guarded_targets):
        raise ValueError("guarded target mask must protect every target pixel")
    if not isinstance(controls.sample_key, str) or not controls.sample_key:
        raise ValueError("control sample_key must be a non-empty string")

    if protection_mask is None:
        protection = guarded_targets.copy()
    else:
        protection = _binary_mask(protection_mask, name="protection_mask")
        if protection.shape != array.shape:
            raise ValueError("protection mask and image must share a shape")
        if np.any(all_targets & ~protection):
            raise ValueError("protection mask must include every target pixel")
        protection = protection | guarded_targets

    radii = (context_inner_radius, context_ring_width)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in radii):
        raise ValueError("context radii must be finite and non-negative")
    if context_ring_width <= 0:
        raise ValueError("context_ring_width must be positive")
    integer_parameters = (
        ("num_controls", num_controls, 1),
        ("minimum_ring_pixels", minimum_ring_pixels, 2),
        ("minimum_stencil_pixels", minimum_stencil_pixels, 1),
        ("histogram_bins", histogram_bins, 2),
        ("minimum_covariance_candidates", minimum_covariance_candidates, 2),
    )
    for name, value, minimum in integer_parameters:
        if not isinstance(value, int) or value < minimum:
            raise ValueError("%s must be an integer >= %d" % (name, minimum))
    if not 0 < minimum_ring_coverage <= 1:
        raise ValueError("minimum_ring_coverage must lie in (0,1]")
    if not 0 < covariance_shrinkage <= 1:
        raise ValueError("covariance_shrinkage must lie in (0,1]")
    if (
        not math.isfinite(float(covariance_eigenvalue_floor))
        or covariance_eigenvalue_floor <= 0
    ):
        raise ValueError("covariance_eigenvalue_floor must be finite and positive")
    if (
        not math.isfinite(float(maximum_covariance_condition))
        or maximum_covariance_condition <= 1
    ):
        raise ValueError("maximum_covariance_condition must be finite and > 1")
    if not 0 <= maximum_selected_iou < 1:
        raise ValueError("maximum_selected_iou must lie in [0,1)")
    if not 0.5 <= candidate_support_quantile < 1:
        raise ValueError("candidate_support_quantile must lie in [0.5,1)")

    rejection_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    def unavailable(
        reason: str,
        *,
        target_statistics: _ContextRingStatistics | None = None,
        eligible_count: int = 0,
        target_descriptor: tuple[float, ...] | None = None,
        center: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        active: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
        condition: float | None = None,
        caliper: float | None = None,
    ) -> ContextMatchedControlSelection:
        return ContextMatchedControlSelection(
            available=False,
            reason=reason,
            control_set=None,
            descriptor_names=CONTEXT_DESCRIPTOR_NAMES,
            target_descriptor=target_descriptor,
            descriptor_center=(
                tuple(float(value) for value in center) if center is not None else None
            ),
            descriptor_scale=(
                tuple(float(value) for value in scale) if scale is not None else None
            ),
            active_descriptor_mask=(
                tuple(bool(value) for value in active) if active is not None else None
            ),
            regularized_covariance=(
                covariance.copy() if covariance is not None else None
            ),
            covariance_condition_number=condition,
            context_distance_caliper=caliper,
            target_ring_pixels=(
                target_statistics.ring_pixels if target_statistics is not None else 0
            ),
            target_stencil_pixels=(
                target_statistics.stencil_pixels if target_statistics is not None else 0
            ),
            target_ring_coverage=(
                target_statistics.coverage if target_statistics is not None else None
            ),
            eligible_candidate_count=eligible_count,
            selected=(),
            rejected_candidate_counts=tuple(sorted(rejection_counts.items())),
        )

    target_statistics = _context_ring_statistics(
        array,
        component,
        protection,
        inner_radius=float(context_inner_radius),
        ring_width=float(context_ring_width),
        minimum_ring_pixels=minimum_ring_pixels,
        minimum_stencil_pixels=minimum_stencil_pixels,
        minimum_ring_coverage=float(minimum_ring_coverage),
    )
    if target_statistics is None:
        return unavailable("insufficient_target_exterior_context")

    candidates = []
    seen_digests = set()
    for source_index, raw_candidate in enumerate(controls.translated_masks):
        candidate = _binary_mask(raw_candidate, name="candidate_control")
        if candidate.shape != array.shape:
            raise ValueError("candidate controls and image must share a shape")
        # TranslationControlSet is intentionally a plain data carrier. Validate
        # its masks here as well, so a hand-built or deserialized set cannot
        # silently turn a same-area deformation into a geometry control.
        if not _is_exact_translation(component, candidate):
            reject("geometry_mismatch")
            continue
        if np.any(candidate & protection):
            reject("protection_overlap")
            continue
        digest = _context_mask_digest(candidate, selection_key=controls.sample_key)
        if digest in seen_digests:
            reject("duplicate_mask")
            continue
        seen_digests.add(digest)
        statistics = _context_ring_statistics(
            array,
            candidate,
            protection,
            inner_radius=float(context_inner_radius),
            ring_width=float(context_ring_width),
            minimum_ring_pixels=minimum_ring_pixels,
            minimum_stencil_pixels=minimum_stencil_pixels,
            minimum_ring_coverage=float(minimum_ring_coverage),
        )
        if statistics is None:
            reject("insufficient_exterior_context")
            continue
        candidates.append((source_index, candidate.copy(), digest, statistics))

    required_candidates = max(num_controls, minimum_covariance_candidates)
    if len(candidates) < required_candidates:
        return unavailable(
            "insufficient_eligible_context_controls",
            target_statistics=target_statistics,
            eligible_count=len(candidates),
        )

    # Entropy uses common candidate-only bin edges. Neither the target footprint
    # nor its exterior ring can alter the quantizer fitted to background controls.
    candidate_context_values = np.concatenate(
        [item[3].values for item in candidates]
    )
    entropy_lower, entropy_upper = np.quantile(
        candidate_context_values,
        (0.005, 0.995),
    )

    def descriptor(statistics: _ContextRingStatistics) -> tuple[float, ...]:
        entropy = _histogram_entropy(
            statistics.values,
            lower=float(entropy_lower),
            upper=float(entropy_upper),
            bins=histogram_bins,
        )
        return (*statistics.descriptor_without_entropy, entropy)

    target_descriptor = descriptor(target_statistics)
    candidate_descriptors = np.asarray(
        [descriptor(item[3]) for item in candidates],
        dtype=np.float64,
    )
    target_descriptor_array = np.asarray(target_descriptor, dtype=np.float64)
    if not (
        np.isfinite(candidate_descriptors).all()
        and np.isfinite(target_descriptor_array).all()
    ):
        raise RuntimeError("context descriptors must be finite")

    # Robust per-descriptor scaling prevents intensity, gradient, Laplacian and
    # entropy units from defining the ranking. Candidate-only scaling prevents
    # the target context from setting its own distance metric.
    descriptor_center = np.median(candidate_descriptors, axis=0)
    deviations = candidate_descriptors - descriptor_center[None, :]
    absolute_deviations = np.abs(deviations)
    mad_scale = 1.4826 * np.median(absolute_deviations, axis=0)
    lower_quartile, upper_quartile = np.quantile(
        candidate_descriptors,
        (0.25, 0.75),
        axis=0,
    )
    iqr_scale = (upper_quartile - lower_quartile) / 1.349
    lower_decile, upper_decile = np.quantile(
        candidate_descriptors,
        (0.1, 0.9),
        axis=0,
    )
    central_eighty_scale = (upper_decile - lower_decile) / 2.563
    # All fallbacks remain quantile based. A rare extreme candidate therefore
    # cannot activate an otherwise unidentified descriptor dimension or set
    # the normalization scale through an unbounded RMS estimate.
    scale_signal = np.maximum.reduce(
        (mad_scale, iqr_scale, central_eighty_scale)
    )
    maximum_magnitude = np.maximum(
        np.max(np.abs(candidate_descriptors), axis=0),
        np.abs(descriptor_center),
    )
    numeric_tolerance = 256.0 * np.finfo(np.float64).eps * np.maximum(
        maximum_magnitude,
        np.finfo(np.float64).tiny,
    )
    active_descriptors = scale_signal > numeric_tolerance
    inactive_target_delta = np.abs(
        target_descriptor_array - descriptor_center
    ) > numeric_tolerance
    if np.any(inactive_target_delta & ~active_descriptors):
        return unavailable(
            "unidentified_descriptor_scale",
            target_statistics=target_statistics,
            eligible_count=len(candidates),
            target_descriptor=target_descriptor,
            center=descriptor_center,
            active=active_descriptors,
        )
    descriptor_scale = np.where(active_descriptors, scale_signal, 1.0)
    standardized_candidates = deviations / descriptor_scale[None, :]
    standardized_target = (
        target_descriptor_array - descriptor_center
    ) / descriptor_scale
    standardized_candidates[:, ~active_descriptors] = 0.0
    standardized_target[~active_descriptors] = 0.0

    covariance = np.cov(
        standardized_candidates,
        rowvar=False,
        ddof=1,
    )
    covariance = np.atleast_2d(np.asarray(covariance, dtype=np.float64))
    descriptor_count = len(CONTEXT_DESCRIPTOR_NAMES)
    if covariance.shape != (descriptor_count, descriptor_count):
        raise RuntimeError("context covariance has an unexpected shape")
    covariance = 0.5 * (covariance + covariance.T)
    regularized = (
        (1.0 - covariance_shrinkage) * covariance
        + covariance_shrinkage * np.eye(descriptor_count, dtype=np.float64)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    eigenvalues = np.maximum(eigenvalues, covariance_eigenvalue_floor)
    regularized = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    regularized = 0.5 * (regularized + regularized.T)
    condition_number = float(eigenvalues.max() / eigenvalues.min())
    if (
        not np.isfinite(regularized).all()
        or not math.isfinite(condition_number)
        or condition_number > maximum_covariance_condition
    ):
        return unavailable(
            "ill_conditioned_context_covariance",
            target_statistics=target_statistics,
            eligible_count=len(candidates),
            target_descriptor=target_descriptor,
            center=descriptor_center,
            scale=descriptor_scale,
            active=active_descriptors,
            covariance=regularized,
            condition=condition_number,
        )
    inverse_covariance = (
        eigenvectors * (1.0 / eigenvalues)[None, :]
    ) @ eigenvectors.T

    # A target must lie inside background descriptor support; merely returning
    # the least-bad controls would silently call an unmatched context "matched".
    # The caliper is the requested-control-count nearest-neighbour radius among
    # background candidates, evaluated at a conservative candidate quantile.
    pairwise_difference = (
        standardized_candidates[:, None, :]
        - standardized_candidates[None, :, :]
    )
    pairwise_squared_distance = np.einsum(
        "...i,ij,...j->...",
        pairwise_difference,
        inverse_covariance,
        pairwise_difference,
    )
    pairwise_squared_distance = np.maximum(pairwise_squared_distance, 0.0)
    np.fill_diagonal(pairwise_squared_distance, np.inf)
    support_neighbour = min(num_controls, len(candidates) - 1)
    candidate_support_radii = np.sqrt(
        np.partition(
            pairwise_squared_distance,
            support_neighbour - 1,
            axis=1,
        )[:, support_neighbour - 1]
    )
    context_distance_caliper = float(
        np.quantile(candidate_support_radii, candidate_support_quantile)
    )
    if not math.isfinite(context_distance_caliper):
        return unavailable(
            "unidentified_candidate_context_support",
            target_statistics=target_statistics,
            eligible_count=len(candidates),
            target_descriptor=target_descriptor,
            center=descriptor_center,
            scale=descriptor_scale,
            active=active_descriptors,
            covariance=regularized,
            condition=condition_number,
        )

    ranked = []
    for item, candidate_descriptor, standardized in zip(
        candidates,
        candidate_descriptors,
        standardized_candidates,
    ):
        difference = standardized - standardized_target
        squared_distance = float(
            difference @ inverse_covariance @ difference
        )
        if squared_distance < 0 and squared_distance > -1e-10:
            squared_distance = 0.0
        if squared_distance < 0 or not math.isfinite(squared_distance):
            raise RuntimeError("Mahalanobis distance must be finite and non-negative")
        ranked.append(
            (
                math.sqrt(squared_distance),
                item[2],
                item,
                tuple(float(value) for value in candidate_descriptor),
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]))

    selected_records = []
    selected_masks = []
    within_caliper_count = sum(
        distance <= context_distance_caliper
        for distance, _, _, _ in ranked
    )
    for distance, digest, item, candidate_descriptor in ranked:
        if distance > context_distance_caliper:
            reject("context_distance_caliper")
            continue
        source_index, candidate, _, statistics = item
        independent = True
        for previous in selected_masks:
            intersection = int(np.logical_and(candidate, previous).sum())
            union = int(np.logical_or(candidate, previous).sum())
            if intersection / union > maximum_selected_iou:
                independent = False
                break
        if not independent:
            reject("selected_control_overlap")
            continue
        selected_masks.append(candidate.copy())
        selected_records.append(
            ContextControlMatch(
                source_index=int(source_index),
                component_mask=candidate.copy(),
                mask_digest=digest,
                descriptor=candidate_descriptor,
                mahalanobis_distance=float(distance),
                ring_pixels=statistics.ring_pixels,
                stencil_pixels=statistics.stencil_pixels,
                ring_coverage=statistics.coverage,
            )
        )
        if len(selected_records) == num_controls:
            break
    if len(selected_records) < num_controls:
        return unavailable(
            (
                "target_context_out_of_candidate_support"
                if within_caliper_count < num_controls
                else "insufficient_independent_context_controls"
            ),
            target_statistics=target_statistics,
            eligible_count=len(candidates),
            target_descriptor=target_descriptor,
            center=descriptor_center,
            scale=descriptor_scale,
            active=active_descriptors,
            covariance=regularized,
            condition=condition_number,
            caliper=context_distance_caliper,
        )

    matched_control_set = TranslationControlSet(
        component_mask=component.copy(),
        all_target_mask=all_targets.copy(),
        guarded_target_mask=protection.copy(),
        translated_masks=tuple(mask.copy() for mask in selected_masks),
        sample_key=controls.sample_key + "\0context-matched-v1",
    )
    return ContextMatchedControlSelection(
        available=True,
        reason=None,
        control_set=matched_control_set,
        descriptor_names=CONTEXT_DESCRIPTOR_NAMES,
        target_descriptor=target_descriptor,
        descriptor_center=tuple(float(value) for value in descriptor_center),
        descriptor_scale=tuple(float(value) for value in descriptor_scale),
        active_descriptor_mask=tuple(bool(value) for value in active_descriptors),
        regularized_covariance=regularized.copy(),
        covariance_condition_number=condition_number,
        context_distance_caliper=context_distance_caliper,
        target_ring_pixels=target_statistics.ring_pixels,
        target_stencil_pixels=target_statistics.stencil_pixels,
        target_ring_coverage=target_statistics.coverage,
        eligible_candidate_count=len(candidates),
        selected=tuple(selected_records),
        rejected_candidate_counts=tuple(sorted(rejection_counts.items())),
    )


def fractional_project(mask, output_shape: tuple[int, int]) -> np.ndarray:
    """Area-project a binary mask while retaining sub-cell occupancy."""

    binary = _binary_mask(mask, name="mask")
    if (
        len(output_shape) != 2
        or any(not isinstance(value, int) or value < 1 for value in output_shape)
    ):
        raise ValueError("output_shape must contain two positive integers")
    if output_shape[0] > binary.shape[0] or output_shape[1] > binary.shape[1]:
        raise ValueError("fractional projection only supports downsampling")
    tensor = torch.from_numpy(binary.astype(np.float32))[None, None]
    projected = F.interpolate(tensor, size=output_shape, mode="area")
    array = projected[0, 0].numpy().astype(np.float64, copy=False)
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise RuntimeError("area projection produced invalid occupancy")
    return array


def _fractional_project_many(
    masks: tuple[np.ndarray, ...],
    output_shape: tuple[int, int],
) -> np.ndarray:
    if not masks:
        return np.empty((0, *output_shape), dtype=np.float64)
    stacked = np.stack(
        [_binary_mask(mask, name="mask") for mask in masks], axis=0
    ).astype(np.float32)
    tensor = torch.from_numpy(stacked)[:, None]
    projected = F.interpolate(tensor, size=output_shape, mode="area")[:, 0]
    array = projected.numpy().astype(np.float64, copy=False)
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise RuntimeError("batched area projection produced invalid occupancy")
    return array


def _select_background_indices(
    support: np.ndarray,
    guarded_targets: np.ndarray,
    *,
    physical_stride: float,
    guard_radius: float,
    background_radii: tuple[float, ...],
    minimum_background_cells: int,
    maximum_background_cells: int,
    selection_key: str,
) -> np.ndarray | None:
    distance = distance_transform_edt(~support)
    chosen = None
    for physical_radius in background_radii:
        radius = float(physical_radius) / physical_stride
        guard = float(guard_radius) / physical_stride
        ring = (
            (distance > guard)
            & (distance <= radius)
            & ~guarded_targets
            & ~support
        )
        indices = np.flatnonzero(ring.reshape(-1))
        if indices.size >= minimum_background_cells:
            chosen = indices
            break
    if chosen is None:
        return None
    if chosen.size > maximum_background_cells:
        ordered = sorted(
            (int(index) for index in chosen),
            key=lambda index: _hash_order(selection_key, index),
        )
        chosen = np.asarray(ordered[:maximum_background_cells], dtype=np.int64)
    else:
        chosen = chosen.astype(np.int64, copy=False)
    return chosen


def project_geometry_controls(
    controls: TranslationControlSet,
    output_shape: tuple[int, int],
    *,
    required_controls: int = 64,
    guard_radius: float = 3.0,
    background_radii: tuple[float, ...] = (12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0),
    minimum_background_cells: int = 16,
    maximum_background_cells: int = 128,
) -> ProjectedGeometry | None:
    """Project target/control footprints and construct local stage backgrounds."""

    if not isinstance(required_controls, int) or required_controls < 1:
        raise ValueError("required_controls must be a positive integer")
    if minimum_background_cells < 2:
        raise ValueError("minimum_background_cells must be at least two")
    if maximum_background_cells < minimum_background_cells:
        raise ValueError("maximum background count must cover the minimum")
    if not background_radii or any(
        not math.isfinite(float(radius)) or radius <= guard_radius
        for radius in background_radii
    ):
        raise ValueError("background radii must be finite and exceed the guard")
    if tuple(sorted(background_radii)) != tuple(background_radii):
        raise ValueError("background radii must be sorted")

    full_height, full_width = controls.component_mask.shape
    output_height, output_width = output_shape
    stride_y = full_height / output_height
    stride_x = full_width / output_width
    if not math.isclose(stride_y, stride_x, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("feature projection requires equal spatial strides")
    projected_guard = fractional_project(
        controls.guarded_target_mask, output_shape
    ) > 0

    projected_masks = _fractional_project_many(
        (controls.component_mask, *controls.translated_masks),
        output_shape,
    )

    def make_footprint(
        occupancy: np.ndarray,
        *,
        key: str,
    ) -> ProjectedFootprint | None:
        support = occupancy > 0
        if not support.any():
            return None
        background_indices = _select_background_indices(
            support,
            projected_guard,
            physical_stride=stride_y,
            guard_radius=guard_radius,
            background_radii=background_radii,
            minimum_background_cells=minimum_background_cells,
            maximum_background_cells=maximum_background_cells,
            selection_key=key,
        )
        if background_indices is None:
            return None
        return ProjectedFootprint(
            occupancy=occupancy,
            background_flat_indices=background_indices,
        )

    target = make_footprint(
        projected_masks[0],
        key=controls.sample_key + "\0target",
    )
    if target is None:
        return None

    projected_controls = []
    seen = set()
    target_area = float(target.occupancy.sum())
    for control_index, occupancy in enumerate(projected_masks[1:]):
        footprint = make_footprint(
            occupancy,
            key="%s\0control\0%d" % (controls.sample_key, control_index),
        )
        if footprint is None or not math.isclose(
            float(footprint.occupancy.sum()),
            target_area,
            rel_tol=1e-5,
            abs_tol=1e-8,
        ):
            continue
        identity = footprint.occupancy.astype(np.float32).tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        projected_controls.append(footprint)
        if len(projected_controls) == required_controls:
            break
    if len(projected_controls) < required_controls:
        return None

    weights = target.occupancy[target.occupancy > 0]
    effective_cells = float(weights.sum() ** 2 / np.sum(weights**2))
    return ProjectedGeometry(
        output_shape=output_shape,
        target=target,
        controls=tuple(projected_controls),
        target_effective_cells=effective_cells,
        target_max_occupancy=float(weights.max()),
    )


def _feature_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or any(size < 1 for size in array.shape):
        raise ValueError("feature must have shape [C,H,W]")
    if not np.isfinite(array).all():
        raise ValueError("feature must contain only finite values")
    return array


def _contrast_score(
    feature_flat: np.ndarray,
    footprint: ProjectedFootprint,
    channel_floor: np.ndarray,
) -> tuple[float, float | None, float | None]:
    occupancy = footprint.occupancy.reshape(-1)
    target_indices = np.flatnonzero(occupancy > 0)
    target_weights = occupancy[target_indices]
    target_weights = target_weights / target_weights.sum()
    target_values = feature_flat[:, target_indices]
    background_values = feature_flat[:, footprint.background_flat_indices]

    center = np.median(background_values, axis=1)
    deviations = background_values - center[:, None]
    mad = 1.4826 * np.median(np.abs(deviations), axis=1)
    rms = np.sqrt(np.mean(deviations**2, axis=1))
    scale = np.maximum.reduce((mad, 0.1 * rms, channel_floor))
    target_mean = np.sum(target_values * target_weights[None, :], axis=1)
    standardized = (target_mean - center) / scale
    score = float(np.sqrt(np.mean(standardized**2)))

    directional_auc = None
    target_peak = None
    if feature_flat.shape[0] == 1:
        target_scalar = target_values[0]
        background_scalar = background_values[0]
        comparisons = target_scalar[:, None] - background_scalar[None, :]
        pair_scores = (comparisons > 0).astype(np.float64)
        pair_scores += 0.5 * (comparisons == 0)
        directional_auc = float(
            np.sum(pair_scores * target_weights[:, None])
            / background_scalar.size
        )
        target_peak = float(np.max(target_scalar))
    return score, directional_auc, target_peak


def evaluate_feature_survival(
    feature,
    geometry: ProjectedGeometry | None,
    *,
    distinct_rank: float = 0.95,
    background_like_rank: float = 0.5,
    scalar_threshold: float | None = None,
) -> FeatureSurvivalResult:
    """Rank target contrast against geometry-matched translated controls."""

    array = _feature_array(feature)
    if geometry is None:
        return FeatureSurvivalResult(
            available=False,
            reason="insufficient_geometry_controls",
            state="undefined",
            rank=None,
            robust_effect=None,
            observed_score=None,
            null_q05=None,
            null_median=None,
            null_q95=None,
            null_max=None,
            num_controls=0,
            target_effective_cells=0.0,
            target_max_occupancy=0.0,
            target_background_cells=0,
            directional_auc=None,
            target_peak=None,
            target_peak_margin=None,
        )
    if tuple(array.shape[1:]) != geometry.output_shape:
        raise ValueError("feature and projected geometry spatial shapes differ")
    if not 0.5 < distinct_rank <= 1.0:
        raise ValueError("distinct_rank must lie in (0.5,1]")
    if not 0.0 <= background_like_rank < distinct_rank:
        raise ValueError("background_like_rank must lie below distinct_rank")
    if scalar_threshold is not None and (
        array.shape[0] != 1 or not math.isfinite(float(scalar_threshold))
    ):
        raise ValueError("scalar_threshold requires a finite scalar feature")

    feature_flat = array.reshape(array.shape[0], -1)
    global_center = np.median(feature_flat, axis=1)
    global_deviation = feature_flat - global_center[:, None]
    global_mad = 1.4826 * np.median(np.abs(global_deviation), axis=1)
    global_rms = np.sqrt(np.mean(global_deviation**2, axis=1))
    channel_floor = np.maximum(1e-8, 1e-3 * np.maximum(global_mad, global_rms))

    observed, directional_auc, target_peak = _contrast_score(
        feature_flat,
        geometry.target,
        channel_floor,
    )
    null_scores = np.asarray(
        [
            _contrast_score(feature_flat, control, channel_floor)[0]
            for control in geometry.controls
        ],
        dtype=np.float64,
    )
    if not np.isfinite(null_scores).all() or not math.isfinite(observed):
        raise RuntimeError("feature contrast produced non-finite scores")
    rank = float((1 + np.sum(null_scores < observed)) / (len(null_scores) + 1))
    null_median = float(np.median(null_scores))
    null_deviation = null_scores - null_median
    null_mad = float(1.4826 * np.median(np.abs(null_deviation)))
    null_rms = float(np.sqrt(np.mean(null_deviation**2)))
    effect_scale = max(null_mad, 0.1 * null_rms, 1e-8)
    robust_effect = float((observed - null_median) / effect_scale)
    state = (
        "distinct"
        if rank >= distinct_rank
        else "background_like"
        if rank <= background_like_rank
        else "uncertain"
    )
    quantiles = np.quantile(null_scores, (0.05, 0.95))
    target_peak_margin = (
        target_peak - float(scalar_threshold)
        if target_peak is not None and scalar_threshold is not None
        else None
    )
    return FeatureSurvivalResult(
        available=True,
        reason=None,
        state=state,
        rank=rank,
        robust_effect=robust_effect,
        observed_score=observed,
        null_q05=float(quantiles[0]),
        null_median=null_median,
        null_q95=float(quantiles[1]),
        null_max=float(np.max(null_scores)),
        num_controls=len(null_scores),
        target_effective_cells=geometry.target_effective_cells,
        target_max_occupancy=geometry.target_max_occupancy,
        target_background_cells=int(
            geometry.target.background_flat_indices.size
        ),
        directional_auc=directional_auc,
        target_peak=target_peak,
        target_peak_margin=target_peak_margin,
    )


__all__ = [
    "CONTEXT_DESCRIPTOR_NAMES",
    "ContextControlMatch",
    "ContextMatchedControlSelection",
    "FeatureSurvivalResult",
    "ProjectedFootprint",
    "ProjectedGeometry",
    "TranslationControlSet",
    "build_translation_control_set",
    "evaluate_feature_survival",
    "fractional_project",
    "project_geometry_controls",
    "select_context_matched_controls",
]
