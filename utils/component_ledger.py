"""Per-sample connected-component diagnostics for CCSR Gate C1.

The ledger is deliberately read-only evaluation code.  It does not alter the
historical ``PD_FA`` implementation and it does not provide a training loss.
All component relations use 8-connectivity and the same strict 3-pixel
centroid rule as the CCSR matching definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import distance_transform_edt

from model.ccsr.task_risk import exact_component_edit_risk_from_masks
from utils.metric import (
    _binary_2d_array,
    _finite_2d_array,
    _validate_new_component_arguments,
    match_components_hungarian,
    match_connected_components,
)


@dataclass(frozen=True)
class ComponentLedger:
    """One sample at one threshold, with mutually interpretable counts."""

    num_gt: int
    num_pred_components: int
    legacy_matches: int
    hungarian_matches: int
    unmatched_gt: int
    unmatched_pred_components: int
    unmatched_pred_area: int
    no_response_gt: int
    centroid_miss_gt: int
    merged_gt_count: int
    split_prediction_count: int
    multi_gt_per_pred_component: tuple[int, ...]
    pred_components_per_gt: tuple[int, ...]
    bridge_candidate_count: int
    mean_bridge_saddle_margin: float | None
    raw_component_edit_risk: float
    no_response_target_indices: tuple[int, ...]
    centroid_miss_target_indices: tuple[int, ...]
    bridge_candidate_prediction_indices: tuple[int, ...]
    threshold: float | None = None
    input_semantics: str = "mask"

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready record without changing tuple identities."""

        return asdict(self)


def _component_relations(component_match, *, centroid_radius: float):
    """Return support-neighborhood and centroid-neighborhood relations."""

    target_regions = component_match.target_regions
    prediction_regions = component_match.prediction_regions
    num_target = len(target_regions)
    num_prediction = len(prediction_regions)
    support_near = np.zeros((num_target, num_prediction), dtype=bool)
    target_centroid_in_prediction_neighborhood = np.zeros_like(support_near)
    centroid_legal = np.zeros_like(support_near)

    for target_index, target_region in enumerate(target_regions):
        target_centroid = np.asarray(target_region.centroid, dtype=np.float64)
        target_component = (
            component_match.target_label_map == target_region.label
        )
        distance_to_target = distance_transform_edt(~target_component)
        for prediction_index, prediction_region in enumerate(prediction_regions):
            prediction_coords = np.asarray(
                prediction_region.coords, dtype=np.float64
            )
            # The distance transform avoids materializing a potentially large
            # all-pairs pixel-distance matrix while retaining exact Euclidean
            # distances on the image grid.
            min_support_distance = float(
                np.min(
                    distance_to_target[
                        prediction_region.coords[:, 0],
                        prediction_region.coords[:, 1],
                    ]
                )
            )
            support_near[target_index, prediction_index] = (
                min_support_distance < centroid_radius
            )

            target_to_prediction = prediction_coords - target_centroid[None, :]
            target_centroid_distance = float(
                np.sqrt(np.min(np.sum(target_to_prediction**2, axis=1)))
            )
            target_centroid_in_prediction_neighborhood[
                target_index, prediction_index
            ] = target_centroid_distance < centroid_radius

            prediction_centroid = np.asarray(
                prediction_region.centroid, dtype=np.float64
            )
            centroid_legal[target_index, prediction_index] = (
                float(np.linalg.norm(target_centroid - prediction_centroid))
                < centroid_radius
            )
    return support_near, target_centroid_in_prediction_neighborhood, centroid_legal


def compute_component_ledger(
    pred_mask,
    target_mask,
    *,
    centroid_radius: float = 3.0,
    connectivity: int = 2,
    threshold: float | None = None,
    input_semantics: str = "mask",
    score_array=None,
) -> ComponentLedger:
    """Compute a component ledger from an already thresholded binary mask.

    ``merged_gt_count`` and ``split_prediction_count`` count *excess* atomic
    units: a single prediction joining two GTs contributes one merge, and two
    predictions around one GT contribute one split.  ``no_response_gt`` and
    ``centroid_miss_gt`` are disjoint subsets of unmatched Hungarian targets.

    ``bridge_candidate_count`` is a Gate-C1 observable proxy: an active
    prediction component whose 3-pixel support neighborhood contains at least
    two GT centroids.  Exact max-tree branch/saddle validation belongs to Gate
    C3.  When scores are supplied by :func:`build_component_ledger`, the
    reported margin is the conservative minimum active-support margin, not a
    claim of exact max-tree saddle recovery.
    """

    prediction_array, target_array, radius = _validate_new_component_arguments(
        pred_mask,
        target_mask,
        centroid_radius=centroid_radius,
        connectivity=connectivity,
    )
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("threshold must be finite when provided") from exc
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite when provided")
    if input_semantics not in {"mask", "logits", "probabilities"}:
        raise ValueError(
            "input_semantics must be 'mask', 'logits', or 'probabilities'"
        )

    hungarian = match_components_hungarian(
        prediction_array,
        target_array,
        centroid_radius=radius,
        connectivity=connectivity,
    )
    legacy = match_connected_components(
        prediction_array,
        target_array,
        max_centroid_distance=radius,
        connectivity=connectivity,
    )
    support_near, centroid_near, centroid_legal = _component_relations(
        hungarian, centroid_radius=radius
    )

    pred_components_per_gt = tuple(
        int(value) for value in np.sum(support_near, axis=1)
    )
    multi_gt_per_pred_component = tuple(
        int(value) for value in np.sum(centroid_near, axis=0)
    )
    merged_gt_count = sum(
        max(0, count - 1) for count in multi_gt_per_pred_component
    )
    split_prediction_count = sum(
        max(0, count - 1) for count in pred_components_per_gt
    )
    bridge_indices = tuple(
        index
        for index, count in enumerate(multi_gt_per_pred_component)
        if count >= 2
    )

    unmatched_targets = set(hungarian.unmatched_target_indices)
    no_response_indices = tuple(
        target_index
        for target_index in hungarian.unmatched_target_indices
        if pred_components_per_gt[target_index] == 0
    )
    no_response_set = set(no_response_indices)
    centroid_miss_indices = tuple(
        target_index
        for target_index in hungarian.unmatched_target_indices
        if target_index not in no_response_set
        and not bool(np.any(centroid_legal[target_index]))
    )
    if not set(centroid_miss_indices).issubset(unmatched_targets):
        raise RuntimeError("centroid-miss classification escaped unmatched targets")

    mean_bridge_margin = None
    if score_array is not None:
        scores = _finite_2d_array(score_array, name="score_array")
        if scores.shape != prediction_array.shape:
            raise ValueError("score_array and pred_mask shapes must match")
        if threshold is None:
            raise ValueError("threshold is required when score_array is provided")
        bridge_margins = []
        for prediction_index in bridge_indices:
            region = hungarian.prediction_regions[prediction_index]
            values = scores[
                region.coords[:, 0],
                region.coords[:, 1],
            ]
            bridge_margins.append(float(np.min(values) - threshold))
        if bridge_margins:
            mean_bridge_margin = float(np.mean(bridge_margins))

    unmatched_pred_area = int(
        sum(
            hungarian.prediction_regions[index].area
            for index in hungarian.unmatched_prediction_indices
        )
    )
    return ComponentLedger(
        num_gt=len(hungarian.target_regions),
        num_pred_components=len(hungarian.prediction_regions),
        legacy_matches=len(legacy.matches),
        hungarian_matches=len(hungarian.matches),
        unmatched_gt=len(hungarian.unmatched_target_indices),
        unmatched_pred_components=len(hungarian.unmatched_prediction_indices),
        unmatched_pred_area=unmatched_pred_area,
        no_response_gt=len(no_response_indices),
        centroid_miss_gt=len(centroid_miss_indices),
        merged_gt_count=int(merged_gt_count),
        split_prediction_count=int(split_prediction_count),
        multi_gt_per_pred_component=multi_gt_per_pred_component,
        pred_components_per_gt=pred_components_per_gt,
        bridge_candidate_count=len(bridge_indices),
        mean_bridge_saddle_margin=mean_bridge_margin,
        # This is a separate assignment from the headline Hungarian metric:
        # component edit risk minimizes its full matched/miss/clutter cost,
        # whereas the metric matcher maximizes cardinality then minimizes
        # centroid distance.  Reusing the latter assignment is not exact.
        raw_component_edit_risk=exact_component_edit_risk_from_masks(
            prediction_array,
            target_array,
            centroid_radius=radius,
            connectivity=connectivity,
        ).risk,
        no_response_target_indices=no_response_indices,
        centroid_miss_target_indices=centroid_miss_indices,
        bridge_candidate_prediction_indices=bridge_indices,
        threshold=threshold,
        input_semantics=input_semantics,
    )


def build_component_ledger(
    logits_or_probs,
    target,
    *,
    threshold: float,
    input_semantics: Literal["logits", "probabilities"],
    centroid_radius: float = 3.0,
    connectivity: int = 2,
) -> ComponentLedger:
    """Threshold logits/probabilities explicitly, then build one ledger."""

    scores = _finite_2d_array(logits_or_probs, name="logits_or_probs")
    target_array = _binary_2d_array(target, name="target")
    if scores.shape != target_array.shape:
        raise ValueError("logits_or_probs and target shapes must match")
    if input_semantics not in {"logits", "probabilities"}:
        raise ValueError("input_semantics must be 'logits' or 'probabilities'")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be finite") from exc
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite")
    if input_semantics == "probabilities":
        if not bool(np.all((scores >= 0.0) & (scores <= 1.0))):
            raise ValueError("probability inputs must lie in [0, 1]")
        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError("probability threshold must lie in [0, 1]")

    prediction_mask = scores > threshold_value
    return compute_component_ledger(
        prediction_mask,
        target_array,
        centroid_radius=centroid_radius,
        connectivity=connectivity,
        threshold=threshold_value,
        input_semantics=input_semantics,
        score_array=scores,
    )


__all__ = [
    "ComponentLedger",
    "build_component_ledger",
    "compute_component_ledger",
]
