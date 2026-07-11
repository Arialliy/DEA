from __future__ import annotations

import numpy as np
import pytest

from utils.metric import (
    evaluate_component_curve,
    match_components_hungarian,
)


def _greedy_trap_masks() -> tuple[np.ndarray, np.ndarray]:
    """Two legal matches exist, but target-order greedy returns only one."""

    prediction = np.zeros((16, 16), dtype=np.uint8)
    target = np.zeros_like(prediction)
    target[6, 5] = 1
    target[6, 9] = 1
    # Centroid x=2.5: legal only for the first target (distance 2.5).
    prediction[6, 2:4] = 1
    # Centroid x=7: closer to the first target, but also the only legal
    # prediction for the second target (both distances 2).
    prediction[6, 7] = 1
    return prediction, target


def test_hungarian_maximizes_legal_cardinality_before_distance() -> None:
    prediction, target = _greedy_trap_masks()

    result = match_components_hungarian(prediction, target)

    assert len(result.matches) == 2
    assert result.unmatched_prediction_indices == ()
    assert result.unmatched_target_indices == ()
    assert {(target_id, prediction_id) for target_id, prediction_id, _ in result.matches} == {
        (0, 0),
        (1, 1),
    }


def test_hungarian_minimizes_distance_at_maximum_cardinality() -> None:
    prediction = np.zeros((12, 12), dtype=np.uint8)
    target = np.zeros_like(prediction)
    target[6, 5] = 1
    target[6, 7] = 1
    prediction[5, 5] = 1
    prediction[5, 7] = 1

    result = match_components_hungarian(prediction, target)

    assert [(target_id, prediction_id) for target_id, prediction_id, _ in result.matches] == [
        (0, 0),
        (1, 1),
    ]
    assert sum(distance for _, _, distance in result.matches) == pytest.approx(2.0)


def test_hungarian_strict_radius_and_eight_connectivity() -> None:
    prediction = np.zeros((10, 10), dtype=np.uint8)
    target = np.zeros_like(prediction)
    target[4, 1] = 1
    prediction[4, 4] = 1  # centroid distance is exactly three
    result = match_components_hungarian(prediction, target)
    assert result.matches == ()

    prediction.fill(0)
    target.fill(0)
    prediction[3, 3] = 1
    prediction[4, 4] = 1
    target[3, 3] = 1
    diagonal = match_components_hungarian(prediction, target)
    assert len(diagonal.prediction_regions) == 1


def test_hungarian_metrics_are_invariant_under_spatial_reordering() -> None:
    prediction, target = _greedy_trap_masks()
    original = match_components_hungarian(prediction, target)
    reflected = match_components_hungarian(
        np.fliplr(prediction), np.fliplr(target)
    )

    assert len(original.matches) == len(reflected.matches) == 2
    assert sorted(distance for _, _, distance in original.matches) == pytest.approx(
        sorted(distance for _, _, distance in reflected.matches)
    )


@pytest.mark.parametrize(
    ("prediction", "target", "kwargs"),
    [
        (np.zeros((2, 2, 1)), np.zeros((2, 2, 1)), {}),
        (np.zeros((2, 2)), np.zeros((3, 2)), {}),
        (np.array([[np.nan]]), np.zeros((1, 1)), {}),
        (np.array([[0.5]]), np.zeros((1, 1)), {}),
        (np.zeros((2, 2)), np.zeros((2, 2)), {"centroid_radius": 0}),
        (np.zeros((2, 2)), np.zeros((2, 2)), {"connectivity": 1}),
    ],
)
def test_hungarian_fails_closed(prediction, target, kwargs) -> None:
    with pytest.raises(ValueError):
        match_components_hungarian(prediction, target, **kwargs)


def test_component_curve_has_explicit_same_domain_threshold_semantics() -> None:
    logits = np.full((8, 8), -2.0)
    target = np.zeros((8, 8), dtype=np.uint8)
    logits[2, 2] = 2.0
    logits[6, 6] = 1.0
    target[2, 2] = 1
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    logit_curve = evaluate_component_curve(
        logits,
        target,
        [0.0],
        input_semantics="logits",
        matching="hungarian",
    )
    probability_curve = evaluate_component_curve(
        probabilities,
        target,
        [0.5],
        input_semantics="probabilities",
        matching="hungarian",
    )

    assert logit_curve[0]["pd"] == probability_curve[0]["pd"] == 1.0
    assert logit_curve[0]["fa"] == probability_curve[0]["fa"] == 1 / 64
    assert logit_curve[0]["prediction_components"] == 2


def test_curve_exposes_legacy_and_hungarian_difference() -> None:
    prediction, target = _greedy_trap_masks()
    scores = np.where(prediction, 1.0, 0.0)

    legacy = evaluate_component_curve(
        scores,
        target,
        [0.5],
        input_semantics="probabilities",
        matching="legacy",
    )
    hungarian = evaluate_component_curve(
        scores,
        target,
        [0.5],
        input_semantics="probabilities",
        matching="hungarian",
    )

    assert legacy[0]["matched_components"] == 1
    assert hungarian[0]["matched_components"] == 2


@pytest.mark.parametrize(
    ("scores", "target", "thresholds", "kwargs"),
    [
        (np.zeros((2, 2, 1)), np.zeros((2, 2)), [0.5], {"input_semantics": "probabilities"}),
        (np.zeros((2, 2)), np.zeros((3, 2)), [0.5], {"input_semantics": "probabilities"}),
        (np.full((2, 2), np.inf), np.zeros((2, 2)), [0.5], {"input_semantics": "probabilities"}),
        (np.full((2, 2), 1.1), np.zeros((2, 2)), [0.5], {"input_semantics": "probabilities"}),
        (np.zeros((2, 2)), np.zeros((2, 2)), [1.1], {"input_semantics": "probabilities"}),
        (np.zeros((2, 2)), np.zeros((2, 2)), [np.nan], {"input_semantics": "logits"}),
        (np.zeros((2, 2)), np.zeros((2, 2)), [], {"input_semantics": "logits"}),
        (np.zeros((2, 2)), np.zeros((2, 2)), [0.0], {"input_semantics": "scores"}),
        (np.zeros((2, 2)), np.zeros((2, 2)), [0.0], {"input_semantics": "logits", "matching": "nearest"}),
    ],
)
def test_component_curve_fails_closed(scores, target, thresholds, kwargs) -> None:
    with pytest.raises(ValueError):
        evaluate_component_curve(scores, target, thresholds, **kwargs)
