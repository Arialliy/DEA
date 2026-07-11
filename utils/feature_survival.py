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
    "FeatureSurvivalResult",
    "ProjectedFootprint",
    "ProjectedGeometry",
    "TranslationControlSet",
    "build_translation_control_set",
    "evaluate_feature_survival",
    "fractional_project",
    "project_geometry_controls",
]
