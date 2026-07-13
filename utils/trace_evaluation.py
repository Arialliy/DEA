"""Leakage-resistant paper evaluation for TRACE and dense controls.

The evaluator deliberately separates development-set calibration from locked
test-set evaluation.  Threshold candidates are the *exact* sorted unique
finite scores pooled over the development bundle.  No quantile grid,
subsampling, interpolation, or test score is allowed to influence threshold
selection.  Every binary prediction uses the repository convention
``score > threshold``.

The official/primary matcher is the historical target-order matcher in
``utils.metric``.  The Hungarian matcher is reported as a mechanism audit at
the very same threshold selected by the primary matcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from utils.metric import match_components_hungarian, match_connected_components


TRACE_EVALUATION_SCHEMA_VERSION = "trace_paper_evaluation_v1"
DEFAULT_FA_BUDGETS_PER_MILLION_PIXELS = (1.0, 5.0, 10.0, 20.0)
DEFAULT_MAX_UNIQUE_SCORES = 100_000
EMPTY_GT_NIOU_POLICY = (
    "Per-image nIoU is intersection/union.  An empty-GT image scores 1 only "
    "when its prediction is also empty; it scores 0 when any pixel is "
    "predicted.  All images, including empty-GT images, enter the arithmetic "
    "mean with equal weight."
)
GLOBAL_IOU_EMPTY_POLICY = (
    "Global foreground IoU is pooled intersection divided by pooled union; "
    "it is defined as 1 when the pooled union is empty."
)


class TraceEvaluationError(ValueError):
    """Base class for fail-closed TRACE evaluation errors."""


class TooManyUniqueScoresError(TraceEvaluationError):
    """The exact development threshold set exceeds its declared safety cap."""


class InfeasibleFABudgetError(TraceEvaluationError):
    """No eligible development operating point satisfies an FA budget."""


@dataclass(frozen=True)
class TraceBundle:
    """Validated score maps, binary targets, identifiers, and input hashes."""

    scores: tuple[np.ndarray, ...]
    targets: tuple[np.ndarray, ...]
    sample_ids: tuple[str, ...]
    content_sha256: str
    source_path: str | None = None
    file_sha256: str | None = None
    score_key: str | None = None
    target_key: str | None = None

    @property
    def sample_count(self) -> int:
        return len(self.scores)

    @property
    def total_pixels(self) -> int:
        return sum(int(array.size) for array in self.scores)

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "file_sha256": self.file_sha256,
            "content_sha256": self.content_sha256,
            "score_key": self.score_key,
            "target_key": self.target_key,
            "sample_count": self.sample_count,
            "total_pixels": self.total_pixels,
            "sample_ids_sha256": _hash_json(list(self.sample_ids)),
            "sample_shapes": [list(array.shape) for array in self.scores],
        }


@dataclass(frozen=True)
class TraceOperatingPoint:
    """All aggregate metrics for one population, threshold, and matcher."""

    threshold: float
    matching: str
    sample_count: int
    total_pixels: int
    target_components: int
    prediction_components: int
    matched_components: int
    missed_target_components: int
    false_component_count: int
    false_component_area_pixels: int
    pd: float | None
    achieved_fa_per_million_pixels: float
    global_foreground_iou: float
    per_image_niou: float
    foreground_intersection_pixels: int
    foreground_union_pixels: int
    empty_gt_image_count: int
    empty_gt_and_empty_prediction_count: int
    empty_gt_with_prediction_count: int

    def to_dict(self, *, requested_fa: float | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "threshold": self.threshold,
            "threshold_operator": "score > threshold",
            "matching": self.matching,
            "sample_count": self.sample_count,
            "total_pixels": self.total_pixels,
            "target_components": self.target_components,
            "prediction_components": self.prediction_components,
            "matched_components": self.matched_components,
            "missed_target_components": self.missed_target_components,
            "pd": self.pd,
            "achieved_fa_per_million_pixels": (
                self.achieved_fa_per_million_pixels
            ),
            "false_component_count": self.false_component_count,
            "false_component_area_pixels": self.false_component_area_pixels,
            "false_component_area_fraction": (
                float(self.false_component_area_pixels) / float(self.total_pixels)
            ),
            "false_components_per_million_pixels": (
                float(self.false_component_count) / float(self.total_pixels) * 1e6
            ),
            "global_foreground_iou": self.global_foreground_iou,
            "per_image_niou": self.per_image_niou,
            "foreground_intersection_pixels": (
                self.foreground_intersection_pixels
            ),
            "foreground_union_pixels": self.foreground_union_pixels,
            "empty_gt_image_count": self.empty_gt_image_count,
            "empty_gt_and_empty_prediction_count": (
                self.empty_gt_and_empty_prediction_count
            ),
            "empty_gt_with_prediction_count": (
                self.empty_gt_with_prediction_count
            ),
        }
        if requested_fa is not None:
            result["requested_fa_per_million_pixels"] = float(requested_fa)
        return result


@dataclass(frozen=True)
class LockedFASelection:
    """One primary legacy development choice for a requested FA budget."""

    requested_fa_per_million_pixels: float
    dev_primary_operating_point: TraceOperatingPoint

    @property
    def threshold(self) -> float:
        return self.dev_primary_operating_point.threshold


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_array_hash(
    scores: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    sample_ids: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"trace_score_target_bundle_v1\0")
    for index, (score, target, sample_id) in enumerate(
        zip(scores, targets, sample_ids)
    ):
        digest.update(index.to_bytes(8, byteorder="little", signed=False))
        identifier = sample_id.encode("utf-8")
        digest.update(len(identifier).to_bytes(8, byteorder="little", signed=False))
        digest.update(identifier)
        for label, array in (
            (b"scores", np.asarray(score, dtype="<f8")),
            (b"targets", np.asarray(target, dtype=np.uint8)),
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(label + b"\0")
            digest.update(
                json.dumps(list(contiguous.shape), separators=(",", ":")).encode(
                    "ascii"
                )
            )
            digest.update(b"\0")
            digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _maps_from_value(value: Any, *, name: str) -> tuple[Any, ...]:
    """Split a dense N(HW)/N1HW array or retain an explicit sample sequence."""

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TraceEvaluationError(f"{name} must not use object dtype")
        if value.ndim == 2:
            return (value,)
        if value.ndim == 3:
            return tuple(value[index] for index in range(value.shape[0]))
        if value.ndim == 4 and value.shape[1] == 1:
            return tuple(value[index, 0] for index in range(value.shape[0]))
        raise TraceEvaluationError(
            f"{name} must have shape HxW, NxHxW, or Nx1xHxW"
        )
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TraceEvaluationError(
            f"{name} must be an array or a sequence of 2-D maps"
        ) from exc
    return result


def _validated_score(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not all(int(size) > 0 for size in array.shape):
        raise TraceEvaluationError(f"{name} must be a non-empty 2-D map")
    if array.dtype.hasobject or array.dtype.kind not in {"b", "i", "u", "f"}:
        raise TraceEvaluationError(f"{name} must contain real numeric scores")
    if array.dtype.kind in {"i", "u"} and array.size:
        # Larger integers can collapse when converted to the common float64
        # threshold domain.  Reject instead of silently changing score ties.
        maximum = int(np.max(array))
        minimum = int(np.min(array))
        if maximum > 2**53 or minimum < -(2**53):
            raise TraceEvaluationError(
                f"{name} contains integers not exactly representable as float64"
            )
    if array.dtype.kind == "f" and array.dtype.itemsize > 8:
        raise TraceEvaluationError(
            f"{name} has precision wider than float64 and cannot be converted exactly"
        )
    converted = np.asarray(array, dtype=np.float64)
    if not bool(np.all(np.isfinite(converted))):
        raise TraceEvaluationError(
            f"{name} contains non-finite scores; non-finite inputs are rejected"
        )
    return np.ascontiguousarray(converted)


def _validated_target(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not all(int(size) > 0 for size in array.shape):
        raise TraceEvaluationError(f"{name} must be a non-empty 2-D map")
    if array.dtype.hasobject:
        raise TraceEvaluationError(f"{name} must not use object dtype")
    try:
        binary = (array == 0) | (array == 1)
    except (TypeError, ValueError) as exc:
        raise TraceEvaluationError(f"{name} must be binary") from exc
    if not bool(np.all(binary)):
        raise TraceEvaluationError(f"{name} must be exactly binary (0/1 or bool)")
    return np.ascontiguousarray(array.astype(bool, copy=False))


def make_trace_bundle(
    scores: Any,
    targets: Any,
    *,
    sample_ids: Sequence[Any] | np.ndarray | None = None,
    source_path: str | None = None,
    file_sha256: str | None = None,
    score_key: str | None = None,
    target_key: str | None = None,
) -> TraceBundle:
    """Validate in-memory samples and construct deterministic content hashes."""

    score_values = _maps_from_value(scores, name="scores")
    target_values = _maps_from_value(targets, name="targets")
    if not score_values:
        raise TraceEvaluationError("at least one score/target sample is required")
    if len(score_values) != len(target_values):
        raise TraceEvaluationError("scores and targets must have equal sample counts")

    validated_scores: list[np.ndarray] = []
    validated_targets: list[np.ndarray] = []
    for index, (score_value, target_value) in enumerate(
        zip(score_values, target_values)
    ):
        score = _validated_score(score_value, name=f"scores[{index}]")
        target = _validated_target(target_value, name=f"targets[{index}]")
        if score.shape != target.shape:
            raise TraceEvaluationError(
                f"scores[{index}] and targets[{index}] shapes must match"
            )
        validated_scores.append(score)
        validated_targets.append(target)

    if sample_ids is None:
        identifiers = tuple(f"sample_{index:06d}" for index in range(len(score_values)))
    else:
        identifier_values = np.asarray(sample_ids)
        if identifier_values.ndim != 1:
            raise TraceEvaluationError("sample_ids must be a one-dimensional sequence")
        if len(identifier_values) != len(score_values):
            raise TraceEvaluationError(
                "sample_ids must have the same length as scores and targets"
            )
        identifiers = tuple(str(value) for value in identifier_values.tolist())
        if any(not value for value in identifiers):
            raise TraceEvaluationError("sample_ids must not contain empty identifiers")
        if len(set(identifiers)) != len(identifiers):
            raise TraceEvaluationError("sample_ids must be unique")

    scores_tuple = tuple(validated_scores)
    targets_tuple = tuple(validated_targets)
    return TraceBundle(
        scores=scores_tuple,
        targets=targets_tuple,
        sample_ids=identifiers,
        content_sha256=_canonical_array_hash(
            scores_tuple, targets_tuple, identifiers
        ),
        source_path=source_path,
        file_sha256=file_sha256,
        score_key=score_key,
        target_key=target_key,
    )


def load_npz_bundle(
    path: str | Path,
    *,
    score_key: str = "scores",
    target_key: str = "targets",
    sample_id_key: str | None = "sample_ids",
) -> TraceBundle:
    """Load a non-pickled ``.npz`` score/target bundle with byte provenance.

    The default contract is ``scores`` plus ``targets`` with shape ``NxHxW``
    (``Nx1xHxW`` and a single ``HxW`` sample are also accepted).  Optional
    Unicode/integer ``sample_ids`` are used when present.  Object arrays are
    rejected; ``allow_pickle`` is never enabled.
    """

    bundle_path = Path(path).expanduser().resolve()
    if not bundle_path.is_file():
        raise TraceEvaluationError(f"NPZ bundle does not exist: {bundle_path}")
    try:
        # Hash and load through the same open descriptor.  This avoids holding
        # a potentially multi-gigabyte compressed bundle in an extra bytes
        # object while keeping the provenance tied to the opened input.
        with bundle_path.open("rb") as bundle_file:
            digest = hashlib.sha256()
            while True:
                chunk = bundle_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            file_sha256 = digest.hexdigest()
            bundle_file.seek(0)
            with np.load(bundle_file, allow_pickle=False) as archive:
                available = tuple(archive.files)
                if score_key not in archive:
                    raise TraceEvaluationError(
                        f"score key {score_key!r} is missing from {bundle_path}; "
                        f"available keys: {available}"
                    )
                if target_key not in archive:
                    raise TraceEvaluationError(
                        f"target key {target_key!r} is missing from {bundle_path}; "
                        f"available keys: {available}"
                    )
                scores = archive[score_key]
                targets = archive[target_key]
                sample_ids = (
                    archive[sample_id_key]
                    if sample_id_key is not None and sample_id_key in archive
                    else None
                )
    except TraceEvaluationError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise TraceEvaluationError(
            f"failed to load safe NPZ bundle {bundle_path}: {exc}"
        ) from exc

    return make_trace_bundle(
        scores,
        targets,
        sample_ids=sample_ids,
        source_path=str(bundle_path),
        file_sha256=file_sha256,
        score_key=score_key,
        target_key=target_key,
    )


# The longer name is useful at call sites where the storage format matters.
load_trace_npz_bundle = load_npz_bundle


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TraceEvaluationError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TraceEvaluationError(f"{name} must be a positive integer") from exc
    try:
        if float(integer) != float(value) or integer <= 0:
            raise TraceEvaluationError(f"{name} must be a positive integer")
    except (TypeError, ValueError, OverflowError) as exc:
        raise TraceEvaluationError(f"{name} must be a positive integer") from exc
    return integer


def build_dev_threshold_candidates(
    dev_scores: Iterable[Any] | np.ndarray,
    *,
    max_unique_scores: int = DEFAULT_MAX_UNIQUE_SCORES,
) -> np.ndarray:
    """Return exact sorted unique development scores, or fail at the cap.

    Each input map must already be finite.  The implementation merges exact
    per-map unique arrays and checks the declared cap after every merge.  It
    never quantizes, samples, or substitutes a grid.
    """

    maximum = _positive_integer(max_unique_scores, name="max_unique_scores")
    score_values = _maps_from_value(dev_scores, name="dev_scores")
    if not score_values:
        raise TraceEvaluationError("at least one development score map is required")

    pooled_unique = np.empty(0, dtype=np.float64)
    # Bound the temporary ``np.unique`` allocation as well as the final set.
    # A single very large, high-cardinality map must not defeat the declared
    # safety cap before we get a chance to check it.
    chunk_size = max(65_536, min(1_000_000, maximum + 1))
    for index, value in enumerate(score_values):
        scores = _validated_score(value, name=f"dev_scores[{index}]")
        flat_scores = scores.reshape(-1)
        for start in range(0, int(flat_scores.size), chunk_size):
            chunk_unique = np.unique(flat_scores[start : start + chunk_size])
            pooled_unique = np.union1d(pooled_unique, chunk_unique)
            if int(pooled_unique.size) > maximum:
                raise TooManyUniqueScoresError(
                    "exact pooled development threshold count exceeds "
                    f"max_unique_scores={maximum} in sample {index}; "
                    "evaluation stopped without quantization or subsampling"
                )
    if pooled_unique.size == 0:
        raise TraceEvaluationError("development scores contain no threshold candidates")
    return np.ascontiguousarray(pooled_unique, dtype=np.float64)


# Backwards-readable alias for callers that emphasize exactness.
build_exact_dev_thresholds = build_dev_threshold_candidates


def _validated_matching_parameters(
    *, centroid_radius: Any, connectivity: Any
) -> tuple[float, int]:
    try:
        radius = float(centroid_radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TraceEvaluationError(
            "centroid_radius must be a finite positive number"
        ) from exc
    if not np.isfinite(radius) or radius <= 0.0:
        raise TraceEvaluationError(
            "centroid_radius must be a finite positive number"
        )
    connectedness = _positive_integer(connectivity, name="connectivity")
    if connectedness != 2:
        raise TraceEvaluationError(
            "connectivity must be 2 (the repository's 8-connected convention)"
        )
    return radius, connectedness


def evaluate_operating_point(
    scores: Any,
    targets: Any,
    threshold: Any,
    *,
    matching: str,
    centroid_radius: float = 3.0,
    connectivity: int = 2,
) -> TraceOperatingPoint:
    """Evaluate one strict threshold on one population and one matcher."""

    if matching not in {"legacy", "hungarian"}:
        raise TraceEvaluationError("matching must be 'legacy' or 'hungarian'")
    radius, connectedness = _validated_matching_parameters(
        centroid_radius=centroid_radius, connectivity=connectivity
    )
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TraceEvaluationError("threshold must be finite") from exc
    if not np.isfinite(threshold_value):
        raise TraceEvaluationError("threshold must be finite")
    bundle = make_trace_bundle(scores, targets)

    return _evaluate_validated_bundle(
        bundle,
        threshold_value,
        matching=matching,
        centroid_radius=radius,
        connectivity=connectedness,
    )


def _evaluate_validated_bundle(
    bundle: TraceBundle,
    threshold_value: float,
    *,
    matching: str,
    centroid_radius: float,
    connectivity: int,
) -> TraceOperatingPoint:
    """Internal hot path; all inputs have already passed public validation."""

    total_pixels = bundle.total_pixels
    target_components = 0
    prediction_components = 0
    matched_components = 0
    missed_target_components = 0
    false_component_count = 0
    false_component_area_pixels = 0
    foreground_intersection_pixels = 0
    foreground_union_pixels = 0
    per_image_ious: list[float] = []
    empty_gt_image_count = 0
    empty_gt_and_empty_prediction_count = 0
    empty_gt_with_prediction_count = 0

    for score_map, target in zip(bundle.scores, bundle.targets):
        # Do not change this to >=.  Equality must remain inactive throughout
        # calibration and every locked evaluation split/matcher.
        prediction = score_map > threshold_value
        if matching == "legacy":
            match = match_connected_components(
                prediction,
                target,
                max_centroid_distance=centroid_radius,
                connectivity=connectivity,
            )
        else:
            match = match_components_hungarian(
                prediction,
                target,
                centroid_radius=centroid_radius,
                connectivity=connectivity,
            )

        target_components += len(match.target_regions)
        prediction_components += len(match.prediction_regions)
        matched_components += len(match.matches)
        missed_target_components += len(match.unmatched_target_indices)
        false_component_count += len(match.unmatched_prediction_indices)
        false_component_area_pixels += int(
            sum(
                match.prediction_regions[index].area
                for index in match.unmatched_prediction_indices
            )
        )

        intersection = int(np.count_nonzero(prediction & target))
        union = int(np.count_nonzero(prediction | target))
        foreground_intersection_pixels += intersection
        foreground_union_pixels += union
        per_image_ious.append(float(intersection) / float(union) if union else 1.0)

        if not bool(np.any(target)):
            empty_gt_image_count += 1
            if bool(np.any(prediction)):
                empty_gt_with_prediction_count += 1
            else:
                empty_gt_and_empty_prediction_count += 1

    pd = (
        float(matched_components) / float(target_components)
        if target_components
        else None
    )
    global_iou = (
        float(foreground_intersection_pixels) / float(foreground_union_pixels)
        if foreground_union_pixels
        else 1.0
    )
    return TraceOperatingPoint(
        threshold=threshold_value,
        matching=matching,
        sample_count=bundle.sample_count,
        total_pixels=total_pixels,
        target_components=target_components,
        prediction_components=prediction_components,
        matched_components=matched_components,
        missed_target_components=missed_target_components,
        false_component_count=false_component_count,
        false_component_area_pixels=false_component_area_pixels,
        pd=pd,
        achieved_fa_per_million_pixels=(
            float(false_component_area_pixels) / float(total_pixels) * 1e6
        ),
        global_foreground_iou=global_iou,
        per_image_niou=float(np.mean(per_image_ious)),
        foreground_intersection_pixels=foreground_intersection_pixels,
        foreground_union_pixels=foreground_union_pixels,
        empty_gt_image_count=empty_gt_image_count,
        empty_gt_and_empty_prediction_count=(
            empty_gt_and_empty_prediction_count
        ),
        empty_gt_with_prediction_count=empty_gt_with_prediction_count,
    )


def _validated_budgets(values: Iterable[Any]) -> tuple[float, ...]:
    budgets: list[float] = []
    for value in values:
        try:
            budget = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TraceEvaluationError(
                "FA budgets must be finite and non-negative"
            ) from exc
        if not np.isfinite(budget) or budget < 0.0:
            raise TraceEvaluationError("FA budgets must be finite and non-negative")
        budgets.append(budget)
    if not budgets:
        raise TraceEvaluationError("at least one FA budget is required")
    if len(set(budgets)) != len(budgets):
        raise TraceEvaluationError("FA budgets must be unique")
    return tuple(budgets)


def select_dev_operating_points(
    dev_primary_curve: Iterable[TraceOperatingPoint],
    fa_budgets_per_million_pixels: Iterable[Any],
) -> tuple[LockedFASelection, ...]:
    """Choose legacy dev thresholds by a declared deterministic rule.

    Feasibility is ``achieved FA/Mpix <= requested FA/Mpix`` with no numeric
    tolerance.  Among feasible thresholds, maximize matched target count,
    then minimize false prediction area, then take the higher threshold.
    Test metrics cannot enter this API.
    """

    points = tuple(dev_primary_curve)
    if not points:
        raise TraceEvaluationError("dev_primary_curve must be non-empty")
    if any(point.matching != "legacy" for point in points):
        raise TraceEvaluationError(
            "primary development threshold selection requires legacy matching"
        )
    populations = {
        (point.sample_count, point.total_pixels, point.target_components)
        for point in points
    }
    if len(populations) != 1:
        raise TraceEvaluationError(
            "all primary development curve points must describe one population"
        )
    if points[0].target_components == 0:
        raise InfeasibleFABudgetError(
            "development population contains no target component, so Pd-based "
            "FA-budget threshold selection is undefined"
        )
    if len({point.threshold for point in points}) != len(points):
        raise TraceEvaluationError("development curve thresholds must be unique")

    budgets = _validated_budgets(fa_budgets_per_million_pixels)
    selections: list[LockedFASelection] = []
    for budget in budgets:
        feasible = tuple(
            point
            for point in points
            if (
                point.false_component_area_pixels * 1_000_000.0
                <= budget * point.total_pixels
            )
        )
        if not feasible:
            minimum = min(
                point.achieved_fa_per_million_pixels for point in points
            )
            raise InfeasibleFABudgetError(
                f"no exact development threshold satisfies FA/Mpix <= {budget}; "
                f"minimum evaluated FA/Mpix is {minimum}"
            )
        selected = max(
            feasible,
            key=lambda point: (
                point.matched_components,
                -point.false_component_area_pixels,
                point.threshold,
            ),
        )
        selections.append(
            LockedFASelection(
                requested_fa_per_million_pixels=budget,
                dev_primary_operating_point=selected,
            )
        )
    return tuple(selections)


def _threshold_sha256(thresholds: np.ndarray) -> str:
    array = np.ascontiguousarray(thresholds, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _curve_sha256(points: Sequence[TraceOperatingPoint]) -> str:
    selection_fields = [
        {
            "threshold": point.threshold,
            "matched_components": point.matched_components,
            "target_components": point.target_components,
            "false_component_count": point.false_component_count,
            "false_component_area_pixels": point.false_component_area_pixels,
            "total_pixels": point.total_pixels,
        }
        for point in points
    ]
    return _hash_json(selection_fields)


def _evaluate_bundle_at_threshold(
    bundle: TraceBundle,
    threshold: float,
    *,
    matching: str,
    centroid_radius: float,
    connectivity: int,
) -> TraceOperatingPoint:
    return _evaluate_validated_bundle(
        bundle,
        threshold,
        matching=matching,
        centroid_radius=centroid_radius,
        connectivity=connectivity,
    )


def evaluate_trace_bundles(
    dev_bundle: TraceBundle,
    test_bundle: TraceBundle,
    *,
    fa_budgets_per_million_pixels: Iterable[Any] = (
        DEFAULT_FA_BUDGETS_PER_MILLION_PIXELS
    ),
    max_unique_scores: int = DEFAULT_MAX_UNIQUE_SCORES,
    centroid_radius: float = 3.0,
    connectivity: int = 2,
    run_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate on dev only, then evaluate locked thresholds on dev and test."""

    if not isinstance(dev_bundle, TraceBundle) or not isinstance(
        test_bundle, TraceBundle
    ):
        raise TraceEvaluationError(
            "dev_bundle and test_bundle must be validated TraceBundle instances"
        )
    budgets = _validated_budgets(fa_budgets_per_million_pixels)
    maximum = _positive_integer(max_unique_scores, name="max_unique_scores")
    radius, connectedness = _validated_matching_parameters(
        centroid_radius=centroid_radius, connectivity=connectivity
    )
    normalized_run_provenance: dict[str, Any] | None = None
    if run_provenance is not None:
        if not isinstance(run_provenance, Mapping):
            raise TraceEvaluationError("run_provenance must be a JSON object")
        try:
            provenance_payload = json.dumps(
                run_provenance,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise TraceEvaluationError(
                "run_provenance must be finite JSON-serializable data"
            ) from exc
        normalized_run_provenance = json.loads(provenance_payload)

    # This is the only candidate-construction call in the protocol.  It sees
    # development scores and nothing from the test bundle.
    thresholds = build_dev_threshold_candidates(
        dev_bundle.scores, max_unique_scores=maximum
    )
    dev_primary_curve = tuple(
        _evaluate_bundle_at_threshold(
            dev_bundle,
            float(threshold),
            matching="legacy",
            centroid_radius=radius,
            connectivity=connectedness,
        )
        for threshold in thresholds
    )
    selections = select_dev_operating_points(dev_primary_curve, budgets)

    operating_points: list[dict[str, Any]] = []
    dev_hungarian_cache: dict[float, TraceOperatingPoint] = {}
    test_legacy_cache: dict[float, TraceOperatingPoint] = {}
    test_hungarian_cache: dict[float, TraceOperatingPoint] = {}
    for selection in selections:
        threshold = selection.threshold
        requested = selection.requested_fa_per_million_pixels
        dev_legacy = selection.dev_primary_operating_point
        if threshold not in dev_hungarian_cache:
            dev_hungarian_cache[threshold] = _evaluate_bundle_at_threshold(
                dev_bundle,
                threshold,
                matching="hungarian",
                centroid_radius=radius,
                connectivity=connectedness,
            )
            test_legacy_cache[threshold] = _evaluate_bundle_at_threshold(
                test_bundle,
                threshold,
                matching="legacy",
                centroid_radius=radius,
                connectivity=connectedness,
            )
            test_hungarian_cache[threshold] = _evaluate_bundle_at_threshold(
                test_bundle,
                threshold,
                matching="hungarian",
                centroid_radius=radius,
                connectivity=connectedness,
            )
        dev_hungarian = dev_hungarian_cache[threshold]
        test_legacy = test_legacy_cache[threshold]
        test_hungarian = test_hungarian_cache[threshold]
        evaluated = (dev_legacy, dev_hungarian, test_legacy, test_hungarian)
        if any(point.threshold != threshold for point in evaluated):
            raise RuntimeError(
                "internal error: matchers did not share a locked threshold"
            )

        operating_points.append(
            {
                "requested_fa_per_million_pixels": requested,
                "locked_threshold": threshold,
                "selected_on": "dev",
                "selected_with_matcher": "legacy",
                "dev": {
                    "legacy": dev_legacy.to_dict(requested_fa=requested),
                    "hungarian": dev_hungarian.to_dict(requested_fa=requested),
                },
                "test": {
                    "legacy": test_legacy.to_dict(requested_fa=requested),
                    "hungarian": test_hungarian.to_dict(requested_fa=requested),
                },
            }
        )

    report: dict[str, Any] = {
        "schema_version": TRACE_EVALUATION_SCHEMA_VERSION,
        "protocol": {
            "name": "paper_protocol_v1_exact_dev_unique_scores",
            "threshold_operator": "score > threshold",
            "threshold_domain": (
                "input score domain (no sigmoid or calibration transform)"
            ),
            "primary_matcher": "legacy",
            "audit_matcher": "hungarian",
            "connectivity": connectedness,
            "centroid_radius_pixels": radius,
            "fa_definition": (
                "unmatched prediction component area / all evaluated pixels * 1e6"
            ),
            "pd_definition": (
                "matched target components / all target components"
            ),
            "global_foreground_iou_definition": GLOBAL_IOU_EMPTY_POLICY,
            "per_image_niou_definition": EMPTY_GT_NIOU_POLICY,
        },
        "inputs": {
            "dev": dev_bundle.provenance_dict(),
            "test": test_bundle.provenance_dict(),
        },
        "selection_provenance": {
            "selection_split": "dev",
            "primary_matcher": "legacy",
            "test_used_for_candidate_construction": False,
            "test_used_for_threshold_selection": False,
            "candidate_source": (
                "sorted exact unique finite scores pooled over dev score maps only"
            ),
            "candidate_quantization": None,
            "candidate_subsampling": None,
            "candidate_count": int(thresholds.size),
            "max_unique_scores": maximum,
            "candidate_min": float(thresholds[0]),
            "candidate_max": float(thresholds[-1]),
            "candidate_sha256": _threshold_sha256(thresholds),
            "dev_primary_curve_sha256": _curve_sha256(dev_primary_curve),
            "feasibility_rule": "achieved FA/Mpix <= requested FA/Mpix",
            "selection_rule": (
                "among feasible exact dev thresholds: maximize matched target "
                "count, then minimize unmatched prediction area, then choose "
                "the higher threshold"
            ),
            "requested_fa_budgets_per_million_pixels": list(budgets),
            "locked_thresholds": [selection.threshold for selection in selections],
        },
        "operating_points": operating_points,
        "run_provenance": normalized_run_provenance,
    }
    # Computing this last makes the report self-identifying without a timestamp
    # or host-dependent value.  The digest excludes only itself.
    report["report_payload_sha256"] = _hash_json(report)
    return report


def evaluate_trace_protocol(
    dev_scores: Any,
    dev_targets: Any,
    test_scores: Any,
    test_targets: Any,
    *,
    dev_sample_ids: Sequence[Any] | np.ndarray | None = None,
    test_sample_ids: Sequence[Any] | np.ndarray | None = None,
    fa_budgets_per_million_pixels: Iterable[Any] = (
        DEFAULT_FA_BUDGETS_PER_MILLION_PIXELS
    ),
    max_unique_scores: int = DEFAULT_MAX_UNIQUE_SCORES,
    centroid_radius: float = 3.0,
    connectivity: int = 2,
    run_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """In-memory convenience wrapper around :func:`evaluate_trace_bundles`."""

    dev_bundle = make_trace_bundle(
        dev_scores, dev_targets, sample_ids=dev_sample_ids
    )
    test_bundle = make_trace_bundle(
        test_scores, test_targets, sample_ids=test_sample_ids
    )
    return evaluate_trace_bundles(
        dev_bundle,
        test_bundle,
        fa_budgets_per_million_pixels=fa_budgets_per_million_pixels,
        max_unique_scores=max_unique_scores,
        centroid_radius=centroid_radius,
        connectivity=connectivity,
        run_provenance=run_provenance,
    )


def dump_trace_evaluation_json(
    report: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a finite JSON report and return its resolved path."""

    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise TraceEvaluationError(
            f"output already exists (pass overwrite=True explicitly): {output}"
        )
    try:
        payload = json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise TraceEvaluationError(
            "evaluation report is not finite JSON-serializable data"
        ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    return output


__all__ = [
    "DEFAULT_FA_BUDGETS_PER_MILLION_PIXELS",
    "DEFAULT_MAX_UNIQUE_SCORES",
    "EMPTY_GT_NIOU_POLICY",
    "GLOBAL_IOU_EMPTY_POLICY",
    "InfeasibleFABudgetError",
    "LockedFASelection",
    "TRACE_EVALUATION_SCHEMA_VERSION",
    "TooManyUniqueScoresError",
    "TraceBundle",
    "TraceEvaluationError",
    "TraceOperatingPoint",
    "build_dev_threshold_candidates",
    "build_exact_dev_thresholds",
    "dump_trace_evaluation_json",
    "evaluate_operating_point",
    "evaluate_trace_bundles",
    "evaluate_trace_protocol",
    "load_npz_bundle",
    "load_trace_npz_bundle",
    "make_trace_bundle",
    "select_dev_operating_points",
]
