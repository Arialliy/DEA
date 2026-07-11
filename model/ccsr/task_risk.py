from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage import measure


@dataclass(frozen=True)
class ComponentMatch:
    prediction_index: int
    target_index: int
    centroid_distance: float
    missing_risk: float
    excess_risk: float

    @property
    def cost(self) -> float:
        return self.missing_risk + self.excess_risk


@dataclass(frozen=True)
class ComponentEditRiskResult:
    """Exact minimum-risk one-to-one assignment for a fixed frontier."""

    risk: float
    matches: tuple[ComponentMatch, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_target_indices: tuple[int, ...]
    matched_missing_risk: float
    matched_excess_risk: float
    miss_risk: float
    clutter_risk: float
    unmatched_base_risk: float
    matching_credit: float
    num_pixels: int


def _as_numpy_2d(value, *, name: str, binary: bool = True) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("%s must be a 2-D array" % name)
    if not np.isfinite(array).all():
        raise ValueError("%s contains NaN or Inf" % name)
    if binary and not np.logical_or(array == 0, array == 1).all():
        raise ValueError("%s must be binary" % name)
    return array.astype(bool, copy=False) if binary else array


def build_components_from_binary(
    mask,
    *,
    connectivity: int = 2,
) -> tuple[torch.Tensor, ...]:
    """Build deterministic CPU boolean masks from one binary 2-D mask."""
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2")
    binary = _as_numpy_2d(mask, name="mask")
    labels = measure.label(binary, connectivity=connectivity)
    return tuple(
        torch.from_numpy(labels == component_id)
        for component_id in range(1, int(labels.max()) + 1)
    )


def build_gt_components(
    target,
    instance_labels,
    *,
    connectivity: int = 2,
) -> tuple[torch.Tensor, ...]:
    """Validate worker labels and return one boolean mask per GT instance."""
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2")
    target_array = _as_numpy_2d(target, name="target")
    labels = _as_numpy_2d(
        instance_labels,
        name="instance_labels",
        binary=False,
    )
    if labels.shape != target_array.shape:
        raise ValueError("target and instance_labels shapes must match")
    if not np.equal(labels, np.floor(labels)).all() or (labels < 0).any():
        raise ValueError("instance_labels must contain non-negative integers")
    labels = labels.astype(np.int64, copy=False)
    if not np.array_equal(labels > 0, target_array):
        raise ValueError("instance_labels support must equal target support")

    components = []
    for component_id in sorted(int(item) for item in np.unique(labels) if item > 0):
        component = labels == component_id
        if int(measure.label(component, connectivity=connectivity).max()) != 1:
            raise ValueError(
                "instance label %d is not one connected component" % component_id
            )
        components.append(torch.from_numpy(component))
    return tuple(components)


def _validate_components(
    components: Sequence,
    *,
    name: str,
    expected_shape: tuple[int, int] | None = None,
    connectivity: int = 2,
) -> tuple[tuple[np.ndarray, ...], tuple[int, int] | None]:
    arrays = tuple(
        _as_numpy_2d(component, name="%s[%d]" % (name, index))
        for index, component in enumerate(components)
    )
    shape = expected_shape
    occupied = None
    for index, component in enumerate(arrays):
        if shape is None:
            shape = component.shape
            occupied = np.zeros(shape, dtype=bool)
        if component.shape != shape:
            raise ValueError("all prediction and target components must share a shape")
        if not component.any():
            raise ValueError("%s[%d] must be non-empty" % (name, index))
        if int(measure.label(component, connectivity=connectivity).max()) != 1:
            raise ValueError("%s[%d] must be connected" % (name, index))
        if occupied is None:
            occupied = np.zeros(shape, dtype=bool)
        if np.logical_and(occupied, component).any():
            raise ValueError("%s components must be disjoint" % name)
        occupied |= component
    return arrays, shape


def _centroid(component: np.ndarray) -> np.ndarray:
    return np.argwhere(component).mean(axis=0)


def _pair_terms(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    num_gt: int,
    num_pixels: int,
    centroid_radius: float,
) -> tuple[float, float, float]:
    distance = float(np.linalg.norm(_centroid(prediction) - _centroid(target)))
    if not distance < centroid_radius:
        return float("inf"), float("inf"), distance
    target_area = int(target.sum())
    missing = int(np.logical_and(target, ~prediction).sum()) / (
        float(num_gt) * float(target_area)
    )
    excess = int(np.logical_and(prediction, ~target).sum()) / float(num_pixels)
    return float(missing), float(excess), distance


def component_pair_cost(
    pred_component,
    gt_component,
    *,
    num_gt: int,
    num_pixels: int,
    centroid_radius: float = 3.0,
) -> float:
    """Metric-derived matched edit cost, or ``inf`` for an illegal edge."""
    if int(num_gt) <= 0:
        raise ValueError("num_gt must be positive for a matched pair")
    if int(num_pixels) <= 0:
        raise ValueError("num_pixels must be positive")
    if not np.isfinite(centroid_radius) or float(centroid_radius) <= 0:
        raise ValueError("centroid_radius must be finite and positive")
    prediction = _as_numpy_2d(pred_component, name="pred_component")
    target = _as_numpy_2d(gt_component, name="gt_component")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target component shapes must match")
    if prediction.size != int(num_pixels):
        raise ValueError("num_pixels must equal the component canvas area")
    if not prediction.any() or not target.any():
        raise ValueError("matched components must be non-empty")
    missing, excess, _ = _pair_terms(
        prediction,
        target,
        num_gt=int(num_gt),
        num_pixels=int(num_pixels),
        centroid_radius=float(centroid_radius),
    )
    return missing + excess


def exact_component_edit_risk(
    pred_components: Sequence,
    gt_components: Sequence,
    *,
    centroid_radius: float = 3.0,
    image_shape: tuple[int, int] | None = None,
    connectivity: int = 2,
) -> ComponentEditRiskResult:
    """Minimize the fixed-frontier component edit risk over assignments.

    Assignment is an *inner minimization*.  It must not be exposed as a free
    state to a loss-augmented outer maximization; doing so makes even perfect
    predictions admit a positive, zero-gradient adversarial matching.
    """
    if connectivity != 2:
        raise ValueError(
            "connectivity must be 2 (8-connectivity for CCSR)"
        )
    if not np.isfinite(centroid_radius) or float(centroid_radius) <= 0:
        raise ValueError("centroid_radius must be finite and positive")

    shape = tuple(image_shape) if image_shape is not None else None
    if shape is not None and (
        len(shape) != 2
        or any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) <= 0
            for value in shape
        )
    ):
        raise ValueError("image_shape must contain two positive integers")
    if shape is not None:
        shape = (int(shape[0]), int(shape[1]))
    prediction, shape = _validate_components(
        pred_components,
        name="pred_components",
        expected_shape=shape,
        connectivity=connectivity,
    )
    targets, shape = _validate_components(
        gt_components,
        name="gt_components",
        expected_shape=shape,
        connectivity=connectivity,
    )
    if shape is None:
        raise ValueError("image_shape is required when both component sets are empty")

    num_pixels = int(shape[0]) * int(shape[1])
    num_predictions = len(prediction)
    num_gt = len(targets)

    if num_gt == 0:
        clutter = sum(int(component.sum()) for component in prediction) / float(num_pixels)
        return ComponentEditRiskResult(
            risk=float(clutter),
            matches=(),
            unmatched_prediction_indices=tuple(range(num_predictions)),
            unmatched_target_indices=(),
            matched_missing_risk=0.0,
            matched_excess_risk=0.0,
            miss_risk=0.0,
            clutter_risk=float(clutter),
            unmatched_base_risk=float(clutter),
            matching_credit=0.0,
            num_pixels=num_pixels,
        )
    if num_predictions == 0:
        return ComponentEditRiskResult(
            risk=1.0,
            matches=(),
            unmatched_prediction_indices=(),
            unmatched_target_indices=tuple(range(num_gt)),
            matched_missing_risk=0.0,
            matched_excess_risk=0.0,
            miss_risk=1.0,
            clutter_risk=0.0,
            unmatched_base_risk=1.0,
            matching_credit=0.0,
            num_pixels=num_pixels,
        )

    # Rows: real predictions, then one miss row per GT.
    # Cols: real GTs, then one private clutter column per prediction.
    size = num_predictions + num_gt
    invalid = 1e6
    costs = np.full((size, size), invalid, dtype=np.float64)
    pair_terms: dict[tuple[int, int], tuple[float, float, float]] = {}
    for prediction_index, pred_component in enumerate(prediction):
        for target_index, gt_component in enumerate(targets):
            missing, excess, distance = _pair_terms(
                pred_component,
                gt_component,
                num_gt=num_gt,
                num_pixels=num_pixels,
                centroid_radius=float(centroid_radius),
            )
            pair_terms[(prediction_index, target_index)] = (
                missing,
                excess,
                distance,
            )
            if np.isfinite(missing):
                costs[prediction_index, target_index] = missing + excess
        costs[prediction_index, num_gt + prediction_index] = (
            int(pred_component.sum()) / float(num_pixels)
        )
    for target_index in range(num_gt):
        costs[num_predictions + target_index, target_index] = 1.0 / float(num_gt)
    costs[num_predictions:, num_gt:] = 0.0

    row_indices, column_indices = linear_sum_assignment(costs)
    if (costs[row_indices, column_indices] >= invalid).any():
        raise RuntimeError("component assignment matrix has no valid completion")

    matches = []
    unmatched_predictions = []
    unmatched_targets = []
    for row, column in zip(row_indices.tolist(), column_indices.tolist()):
        if row < num_predictions and column < num_gt:
            missing, excess, distance = pair_terms[(row, column)]
            matches.append(ComponentMatch(row, column, distance, missing, excess))
        elif row < num_predictions:
            if column != num_gt + row:
                raise RuntimeError("invalid clutter assignment selected")
            unmatched_predictions.append(row)
        elif column < num_gt:
            if row != num_predictions + column:
                raise RuntimeError("invalid miss assignment selected")
            unmatched_targets.append(column)

    matches.sort(key=lambda item: (item.prediction_index, item.target_index))
    unmatched_predictions.sort()
    unmatched_targets.sort()
    matched_missing = sum(item.missing_risk for item in matches)
    matched_excess = sum(item.excess_risk for item in matches)
    miss = len(unmatched_targets) / float(num_gt)
    clutter = sum(
        int(prediction[index].sum()) / float(num_pixels)
        for index in unmatched_predictions
    )
    risk = matched_missing + matched_excess + miss + clutter
    unmatched_base = 1.0 + sum(
        int(component.sum()) for component in prediction
    ) / float(num_pixels)
    matching_credit = 0.0
    for item in matches:
        intersection = int(
            np.logical_and(
                prediction[item.prediction_index],
                targets[item.target_index],
            ).sum()
        )
        target_area = int(targets[item.target_index].sum())
        matching_credit += intersection * (
            1.0 / (float(num_gt) * float(target_area))
            + 1.0 / float(num_pixels)
        )
    if not np.isclose(
        risk,
        unmatched_base - matching_credit,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("component-risk matching-credit identity failed")
    return ComponentEditRiskResult(
        risk=float(risk),
        matches=tuple(matches),
        unmatched_prediction_indices=tuple(unmatched_predictions),
        unmatched_target_indices=tuple(unmatched_targets),
        matched_missing_risk=float(matched_missing),
        matched_excess_risk=float(matched_excess),
        miss_risk=float(miss),
        clutter_risk=float(clutter),
        unmatched_base_risk=float(unmatched_base),
        matching_credit=float(matching_credit),
        num_pixels=num_pixels,
    )


def exact_component_edit_risk_from_masks(
    prediction,
    target,
    *,
    centroid_radius: float = 3.0,
    connectivity: int = 2,
) -> ComponentEditRiskResult:
    prediction_array = _as_numpy_2d(prediction, name="prediction")
    target_array = _as_numpy_2d(target, name="target")
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target shapes must match")
    return exact_component_edit_risk(
        build_components_from_binary(prediction_array, connectivity=connectivity),
        build_components_from_binary(target_array, connectivity=connectivity),
        centroid_radius=centroid_radius,
        image_shape=prediction_array.shape,
        connectivity=connectivity,
    )
