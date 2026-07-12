"""Read-only component operating-point diagnostics for finite logit maps.

This module deliberately keeps threshold calibration separate from model
training.  Every mask uses the repository convention ``logit > threshold``
and every detection uses :func:`utils.metric.match_components_hungarian`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from utils.metric import match_components_hungarian, match_connected_components


DEFAULT_TAIL_QUANTILES = tuple(
    sorted(
        {
            0.5,
            0.75,
            0.85,
            1.0,
            *(
                float(1.0 - value)
                for value in np.geomspace(1e-1, 1e-7, num=49)
            ),
        }
    )
)


@dataclass(frozen=True)
class ComponentOperatingPoint:
    """Dataset-level component counts at one strict logit threshold."""

    threshold: float
    sample_count: int
    total_pixels: int
    matched_components: int
    target_components: int
    prediction_components: int
    unmatched_target_components: int
    unmatched_prediction_components: int
    unmatched_prediction_area: int
    pd: float | None
    fa_per_million_pixels: float


@dataclass(frozen=True)
class FixedFASelection:
    """The deterministic calibration choice for one FA/Mpix budget."""

    fa_budget_per_million_pixels: float
    operating_point: ComponentOperatingPoint

    @property
    def threshold(self) -> float:
        return self.operating_point.threshold


@dataclass(frozen=True)
class ComponentCalibrationResult:
    """Threshold grid, calibration curve, and fixed-FA choices."""

    threshold_grid: tuple[float, ...]
    curve: tuple[ComponentOperatingPoint, ...]
    selections: tuple[FixedFASelection, ...]


@dataclass(frozen=True)
class TargetOperatingStatus:
    """Exact component match and an independent local-peak diagnostic.

    ``matched`` is the component-Pd event produced by Hungarian matching.
    ``neighborhood_peak_above_threshold`` only asks whether any pixel at
    strict Euclidean support distance ``< neighborhood_radius`` exceeds the
    threshold.  Neither Boolean implies the other, so the peak must never be
    reported as component Pd.
    """

    target_index: int
    threshold: float
    target_component_area: int
    target_component_centroid: tuple[float, float]
    matched: bool
    matched_prediction_index: int | None
    matched_centroid_distance: float | None
    neighborhood_radius: float
    neighborhood_pixel_count: int
    neighborhood_peak: float
    neighborhood_margin: float
    neighborhood_peak_above_threshold: bool


class InfeasibleFABudgetError(ValueError):
    """Raised when no evaluated threshold satisfies an FA/Mpix budget."""


def _finite_2d(value: object, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not bool(np.all(finite)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary_target(value: object, *, name: str) -> np.ndarray:
    array = _finite_2d(value, name=name)
    if not bool(np.all((array == 0) | (array == 1))):
        raise ValueError(f"{name} must be binary (0/1 or boolean)")
    return array.astype(bool, copy=False)


def _finite_scalar(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_samples(
    logit_samples: Iterable[object],
    target_samples: Iterable[object],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    logits = tuple(logit_samples)
    targets = tuple(target_samples)
    if not logits:
        raise ValueError("at least one logit/target sample is required")
    if len(logits) != len(targets):
        raise ValueError("logit_samples and target_samples must have equal length")

    validated = []
    for index, (logit, target) in enumerate(zip(logits, targets)):
        score_array = _finite_2d(logit, name=f"logit_samples[{index}]")
        target_array = _binary_target(target, name=f"target_samples[{index}]")
        if score_array.shape != target_array.shape:
            raise ValueError(
                f"logit_samples[{index}] and target_samples[{index}] shapes must match"
            )
        validated.append((score_array, target_array))
    return tuple(validated)


def _thresholds(values: Iterable[object], *, name: str) -> tuple[float, ...]:
    result = tuple(_finite_scalar(value, name=name) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return tuple(sorted(set(result)))


def build_logit_threshold_grid(
    logit_samples: Iterable[object],
    *,
    fixed_thresholds: Sequence[float] = (0.0,),
    tail_quantiles: Sequence[float] = DEFAULT_TAIL_QUANTILES,
) -> tuple[float, ...]:
    """Build a deterministic finite grid from fixed and pooled-tail points.

    Quantiles are computed over the pooled calibration logits with NumPy's
    deterministic linear definition.  Tail quantiles are restricted to
    ``[0.5, 1]``.  Including quantile ``1`` is useful: because thresholding is
    strict, the maximum-score threshold produces the all-off prediction.
    Returned values are unique and ascending.
    """

    arrays = tuple(
        _finite_2d(value, name=f"logit_samples[{index}]")
        for index, value in enumerate(logit_samples)
    )
    if not arrays:
        raise ValueError("at least one logit sample is required")
    fixed = tuple(
        _finite_scalar(value, name="fixed_thresholds")
        for value in fixed_thresholds
    )
    quantiles = tuple(
        _finite_scalar(value, name="tail_quantiles")
        for value in tail_quantiles
    )
    if not fixed and not quantiles:
        raise ValueError("fixed_thresholds and tail_quantiles cannot both be empty")
    if any(value < 0.5 or value > 1.0 for value in quantiles):
        raise ValueError("tail_quantiles must lie in [0.5, 1]")

    pooled = np.concatenate([array.reshape(-1) for array in arrays])
    quantile_thresholds = (
        tuple(
            float(value)
            for value in np.quantile(pooled, quantiles, method="linear")
        )
        if quantiles
        else ()
    )
    return tuple(sorted(set(fixed + quantile_thresholds)))


def evaluate_component_operating_points(
    logit_samples: Iterable[object],
    target_samples: Iterable[object],
    thresholds: Iterable[float],
    *,
    matching: str = "hungarian",
    centroid_radius: float = 3.0,
    connectivity: int = 2,
) -> tuple[ComponentOperatingPoint, ...]:
    """Aggregate component Pd and unmatched-prediction-area FA/Mpix.

    Pd is ``sum(matches) / sum(target components)``.  It is ``None`` when the
    evaluated population has no target component.  FA/Mpix is
    ``sum(unmatched prediction area) / sum(image pixels) * 1e6``.
    """

    if matching not in {"legacy", "hungarian"}:
        raise ValueError("matching must be 'legacy' or 'hungarian'")
    samples = _validate_samples(logit_samples, target_samples)
    threshold_grid = _thresholds(thresholds, name="thresholds")
    total_pixels = sum(int(scores.size) for scores, _ in samples)
    results = []
    for threshold in threshold_grid:
        matched_components = 0
        target_components = 0
        prediction_components = 0
        unmatched_target_components = 0
        unmatched_prediction_components = 0
        unmatched_prediction_area = 0
        for scores, target in samples:
            if matching == "legacy":
                component_match = match_connected_components(
                    scores > threshold,
                    target,
                    max_centroid_distance=centroid_radius,
                    connectivity=connectivity,
                )
            else:
                component_match = match_components_hungarian(
                    scores > threshold,
                    target,
                    centroid_radius=centroid_radius,
                    connectivity=connectivity,
                )
            matched_components += len(component_match.matches)
            target_components += len(component_match.target_regions)
            prediction_components += len(component_match.prediction_regions)
            unmatched_target_components += len(
                component_match.unmatched_target_indices
            )
            unmatched_prediction_components += len(
                component_match.unmatched_prediction_indices
            )
            unmatched_prediction_area += int(
                sum(
                    component_match.prediction_regions[index].area
                    for index in component_match.unmatched_prediction_indices
                )
            )
        pd = (
            float(matched_components) / float(target_components)
            if target_components
            else None
        )
        results.append(
            ComponentOperatingPoint(
                threshold=threshold,
                sample_count=len(samples),
                total_pixels=total_pixels,
                matched_components=matched_components,
                target_components=target_components,
                prediction_components=prediction_components,
                unmatched_target_components=unmatched_target_components,
                unmatched_prediction_components=unmatched_prediction_components,
                unmatched_prediction_area=unmatched_prediction_area,
                pd=pd,
                fa_per_million_pixels=(
                    float(unmatched_prediction_area) / float(total_pixels) * 1e6
                ),
            )
        )
    return tuple(results)


def select_fixed_fa_operating_points(
    curve: Iterable[ComponentOperatingPoint],
    fa_budgets_per_million_pixels: Iterable[float],
) -> tuple[FixedFASelection, ...]:
    """Select a calibration threshold for each FA/Mpix budget.

    Among feasible thresholds, selection is lexicographic: maximize matched
    target count (Pd), minimize unmatched prediction area (FA), then choose
    the highest threshold.  The final tie-break is conservative and makes the
    result independent of input curve order.  A population with no target or
    a budget with no feasible evaluated threshold raises instead of silently
    substituting an operating point.
    """

    points = tuple(curve)
    if not points:
        raise ValueError("curve must be non-empty")
    population = {
        (point.sample_count, point.total_pixels, point.target_components)
        for point in points
    }
    if len(population) != 1:
        raise ValueError("all curve points must describe the same population")
    if points[0].target_components == 0:
        raise ValueError("fixed-FA Pd selection requires at least one target component")
    if len({point.threshold for point in points}) != len(points):
        raise ValueError("curve thresholds must be unique")

    budgets = tuple(
        _finite_scalar(value, name="fa_budgets_per_million_pixels")
        for value in fa_budgets_per_million_pixels
    )
    if not budgets:
        raise ValueError("fa_budgets_per_million_pixels must be non-empty")
    if any(value < 0.0 for value in budgets):
        raise ValueError("FA/Mpix budgets must be non-negative")

    selections = []
    for budget in budgets:
        feasible = tuple(
            point for point in points if point.fa_per_million_pixels <= budget
        )
        if not feasible:
            minimum = min(point.fa_per_million_pixels for point in points)
            raise InfeasibleFABudgetError(
                f"no evaluated threshold satisfies FA/Mpix <= {budget}; "
                f"minimum evaluated FA/Mpix is {minimum}"
            )
        chosen = max(
            feasible,
            key=lambda point: (
                point.matched_components,
                -point.unmatched_prediction_area,
                point.threshold,
            ),
        )
        selections.append(
            FixedFASelection(
                fa_budget_per_million_pixels=budget,
                operating_point=chosen,
            )
        )
    return tuple(selections)


def calibrate_component_operating_points(
    logit_samples: Iterable[object],
    target_samples: Iterable[object],
    fa_budgets_per_million_pixels: Iterable[float],
    *,
    matching: str = "hungarian",
    fixed_thresholds: Sequence[float] = (0.0,),
    tail_quantiles: Sequence[float] = DEFAULT_TAIL_QUANTILES,
    centroid_radius: float = 3.0,
    connectivity: int = 2,
) -> ComponentCalibrationResult:
    """Build, evaluate, and select fixed-FA points on one calibration set."""

    logits = tuple(logit_samples)
    targets = tuple(target_samples)
    # Validation occurs in evaluation; retaining tuples prevents generators
    # from being consumed once for the grid and then appearing empty.
    threshold_grid = build_logit_threshold_grid(
        logits,
        fixed_thresholds=fixed_thresholds,
        tail_quantiles=tail_quantiles,
    )
    curve = evaluate_component_operating_points(
        logits,
        targets,
        threshold_grid,
        matching=matching,
        centroid_radius=centroid_radius,
        connectivity=connectivity,
    )
    selections = select_fixed_fa_operating_points(
        curve,
        fa_budgets_per_million_pixels,
    )
    return ComponentCalibrationResult(
        threshold_grid=threshold_grid,
        curve=curve,
        selections=selections,
    )


def evaluate_target_operating_status(
    logits: object,
    target: object,
    *,
    target_index: int,
    threshold: float,
    matching: str = "hungarian",
    centroid_radius: float = 3.0,
    neighborhood_radius: float = 3.0,
    connectivity: int = 2,
) -> TargetOperatingStatus:
    """Evaluate one target's exact match and strict local logit peak."""

    scores = _finite_2d(logits, name="logits")
    target_array = _binary_target(target, name="target")
    if scores.shape != target_array.shape:
        raise ValueError("logits and target shapes must match")
    threshold_value = _finite_scalar(threshold, name="threshold")
    radius = _finite_scalar(neighborhood_radius, name="neighborhood_radius")
    if radius <= 0.0:
        raise ValueError("neighborhood_radius must be positive")
    if isinstance(target_index, bool) or not isinstance(target_index, (int, np.integer)):
        raise ValueError("target_index must be an integer")
    target_index = int(target_index)

    if matching not in {"legacy", "hungarian"}:
        raise ValueError("matching must be 'legacy' or 'hungarian'")
    if matching == "legacy":
        component_match = match_connected_components(
            scores > threshold_value,
            target_array,
            max_centroid_distance=centroid_radius,
            connectivity=connectivity,
        )
    else:
        component_match = match_components_hungarian(
            scores > threshold_value,
            target_array,
            centroid_radius=centroid_radius,
            connectivity=connectivity,
        )
    if target_index < 0 or target_index >= len(component_match.target_regions):
        raise ValueError(
            f"target_index must be in [0, {len(component_match.target_regions)})"
        )
    target_region = component_match.target_regions[target_index]
    component_mask = component_match.target_label_map == target_region.label
    support_distance = distance_transform_edt(~component_mask)
    neighborhood = support_distance < radius
    neighborhood_peak = float(np.max(scores[neighborhood]))

    matched_prediction_index = None
    matched_centroid_distance = None
    for match_target, match_prediction, distance in component_match.matches:
        if match_target == target_index:
            matched_prediction_index = int(match_prediction)
            matched_centroid_distance = float(distance)
            break
    return TargetOperatingStatus(
        target_index=target_index,
        threshold=threshold_value,
        target_component_area=int(target_region.area),
        target_component_centroid=tuple(float(value) for value in target_region.centroid),
        matched=matched_prediction_index is not None,
        matched_prediction_index=matched_prediction_index,
        matched_centroid_distance=matched_centroid_distance,
        neighborhood_radius=radius,
        neighborhood_pixel_count=int(np.count_nonzero(neighborhood)),
        neighborhood_peak=neighborhood_peak,
        neighborhood_margin=neighborhood_peak - threshold_value,
        neighborhood_peak_above_threshold=neighborhood_peak > threshold_value,
    )
