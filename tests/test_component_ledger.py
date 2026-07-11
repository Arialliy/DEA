from __future__ import annotations

import numpy as np
import pytest

from utils.component_ledger import (
    build_component_ledger,
    compute_component_ledger,
)


def test_bridge_ledger_records_one_atomic_merge_candidate() -> None:
    target = np.zeros((15, 15), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[7, 3] = 1
    target[7, 11] = 1
    prediction[7, 3:12] = 1

    ledger = compute_component_ledger(prediction, target)

    assert ledger.num_gt == 2
    assert ledger.num_pred_components == 1
    assert ledger.multi_gt_per_pred_component == (2,)
    assert ledger.merged_gt_count == 1
    assert ledger.bridge_candidate_count == 1
    assert ledger.bridge_candidate_prediction_indices == (0,)
    assert ledger.split_prediction_count == 0
    assert ledger.no_response_gt == 0
    assert ledger.centroid_miss_gt == 2


def test_split_ledger_counts_excess_prediction_once() -> None:
    target = np.zeros((15, 15), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[7, 7] = 1
    prediction[7, 5] = 1
    prediction[7, 9] = 1

    ledger = compute_component_ledger(prediction, target)

    assert ledger.pred_components_per_gt == (2,)
    assert ledger.split_prediction_count == 1
    assert ledger.hungarian_matches == 1
    assert ledger.unmatched_pred_components == 1
    assert ledger.unmatched_pred_area == 1
    assert ledger.no_response_gt == 0
    assert ledger.centroid_miss_gt == 0


def test_no_response_and_centroid_miss_are_disjoint() -> None:
    target = np.zeros((16, 16), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[2, 2] = 1
    prediction[12, 12] = 1

    no_response = compute_component_ledger(prediction, target)
    assert no_response.no_response_gt == 1
    assert no_response.no_response_target_indices == (0,)
    assert no_response.centroid_miss_gt == 0

    target.fill(0)
    prediction.fill(0)
    target[7, 3] = 1
    prediction[7, 3:11] = 1  # response overlaps; component centroid is 3.5 px away
    centroid_miss = compute_component_ledger(prediction, target)
    assert centroid_miss.no_response_gt == 0
    assert centroid_miss.centroid_miss_gt == 1
    assert centroid_miss.centroid_miss_target_indices == (0,)
    assert not (
        set(centroid_miss.no_response_target_indices)
        & set(centroid_miss.centroid_miss_target_indices)
    )


def test_empty_ledgers_have_defined_counts_and_component_risk() -> None:
    target = np.zeros((8, 8), dtype=np.uint8)
    prediction = np.zeros_like(target)

    both_empty = compute_component_ledger(prediction, target)
    assert both_empty.num_gt == 0
    assert both_empty.num_pred_components == 0
    assert both_empty.raw_component_edit_risk == 0.0

    prediction[3, 3] = 1
    clutter = compute_component_ledger(prediction, target)
    assert clutter.num_gt == 0
    assert clutter.num_pred_components == 1
    assert clutter.unmatched_pred_area == 1
    assert clutter.raw_component_edit_risk == pytest.approx(1 / 64)

    prediction.fill(0)
    target[3, 3] = 1
    missed = compute_component_ledger(prediction, target)
    assert missed.num_gt == 1
    assert missed.unmatched_gt == 1
    assert missed.no_response_gt == 1
    assert missed.raw_component_edit_risk == 1.0


def test_score_builder_records_semantics_threshold_and_bridge_margin() -> None:
    target = np.zeros((12, 12), dtype=np.uint8)
    target[6, 2] = 1
    target[6, 9] = 1
    logits = np.full((12, 12), -2.0)
    logits[6, 2:10] = 0.25

    ledger = build_component_ledger(
        logits,
        target,
        threshold=0.0,
        input_semantics="logits",
    )

    assert ledger.input_semantics == "logits"
    assert ledger.threshold == 0.0
    assert ledger.bridge_candidate_count == 1
    assert ledger.mean_bridge_saddle_margin == pytest.approx(0.25)
    assert ledger.as_dict()["bridge_candidate_count"] == 1


@pytest.mark.parametrize(
    ("scores", "target", "threshold", "semantics"),
    [
        (np.zeros((2, 2, 1)), np.zeros((2, 2)), 0.0, "logits"),
        (np.zeros((2, 2)), np.zeros((3, 2)), 0.0, "logits"),
        (np.full((2, 2), np.nan), np.zeros((2, 2)), 0.0, "logits"),
        (np.full((2, 2), 1.5), np.zeros((2, 2)), 0.5, "probabilities"),
        (np.zeros((2, 2)), np.zeros((2, 2)), np.inf, "logits"),
        (np.zeros((2, 2)), np.zeros((2, 2)), 1.5, "probabilities"),
        (np.zeros((2, 2)), np.zeros((2, 2)), 0.0, "scores"),
    ],
)
def test_score_ledger_fails_closed(scores, target, threshold, semantics) -> None:
    with pytest.raises(ValueError):
        build_component_ledger(
            scores,
            target,
            threshold=threshold,
            input_semantics=semantics,
        )

