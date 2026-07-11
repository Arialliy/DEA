from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.component_operating_point import (
    ComponentOperatingPoint,
    InfeasibleFABudgetError,
    build_logit_threshold_grid,
    calibrate_component_operating_points,
    evaluate_component_operating_points,
    evaluate_target_operating_status,
    select_fixed_fa_operating_points,
)


def test_operating_curve_uses_strict_threshold_and_aggregates_area() -> None:
    logits_a = np.full((4, 4), -2.0)
    target_a = np.zeros((4, 4), dtype=np.uint8)
    target_a[0, 0] = 1
    logits_a[0, 0] = 0.0  # Equality is off under the strict convention.
    logits_a[3, 3] = 1.0

    logits_b = np.full((2, 2), -2.0)
    target_b = np.zeros((2, 2), dtype=np.uint8)
    target_b[0, 0] = 1
    logits_b[0, 0] = 1.0

    point = evaluate_component_operating_points(
        [logits_a, logits_b],
        [target_a, target_b],
        [0.0],
    )[0]

    assert point.sample_count == 2
    assert point.total_pixels == 20
    assert point.matched_components == 1
    assert point.target_components == 2
    assert point.pd == 0.5
    assert point.unmatched_prediction_components == 1
    assert point.unmatched_prediction_area == 1
    assert point.fa_per_million_pixels == pytest.approx(50_000.0)


def test_operating_curve_uses_hungarian_maximum_cardinality() -> None:
    logits = np.full((16, 16), -1.0)
    target = np.zeros((16, 16), dtype=np.uint8)
    target[6, 5] = 1
    target[6, 9] = 1
    logits[6, 2:4] = 1.0
    logits[6, 7] = 1.0

    point = evaluate_component_operating_points([logits], [target], [0.0])[0]

    assert point.matched_components == 2
    assert point.pd == 1.0
    assert point.unmatched_prediction_area == 0


def test_threshold_grid_is_deterministic_unique_and_includes_all_off() -> None:
    logits = [
        np.asarray([[0.0, 1.0], [2.0, 3.0]]),
        np.asarray([[4.0, 5.0]]),
    ]

    first = build_logit_threshold_grid(
        logits,
        fixed_thresholds=(0.0, 2.0, 2.0),
        tail_quantiles=(0.5, 0.9, 1.0),
    )
    second = build_logit_threshold_grid(
        logits,
        fixed_thresholds=(2.0, 0.0),
        tail_quantiles=(1.0, 0.9, 0.5),
    )

    assert first == second
    assert first == tuple(sorted(set(first)))
    assert 0.0 in first and 2.0 in first
    assert first[-1] == 5.0
    # The maximum is a finite all-off threshold under score > threshold.
    target = np.zeros((2, 2), dtype=np.uint8)
    target[0, 0] = 1
    all_off = evaluate_component_operating_points(
        [np.asarray([[0.0, 1.0], [2.0, 5.0]])],
        [target],
        [first[-1]],
    )[0]
    assert all_off.prediction_components == 0


def _point(
    threshold: float,
    *,
    matches: int,
    false_area: int,
) -> ComponentOperatingPoint:
    return ComponentOperatingPoint(
        threshold=threshold,
        sample_count=1,
        total_pixels=100_000,
        matched_components=matches,
        target_components=10,
        prediction_components=matches + (false_area > 0),
        unmatched_target_components=10 - matches,
        unmatched_prediction_components=int(false_area > 0),
        unmatched_prediction_area=false_area,
        pd=matches / 10,
        fa_per_million_pixels=false_area * 10.0,
    )


def test_fixed_fa_selection_has_explicit_lexicographic_ties() -> None:
    curve = [
        _point(-1.0, matches=8, false_area=10),
        _point(0.0, matches=8, false_area=5),
        _point(1.0, matches=8, false_area=5),
        _point(2.0, matches=7, false_area=0),
    ]

    selections = select_fixed_fa_operating_points(curve, [100.0, 0.0])

    # First maximize matches, then minimize FA, then take the higher threshold.
    assert selections[0].threshold == 1.0
    assert selections[1].threshold == 2.0


def test_fixed_fa_selection_fails_closed_when_grid_is_infeasible() -> None:
    curve = [_point(0.0, matches=8, false_area=1)]

    with pytest.raises(InfeasibleFABudgetError, match="no evaluated threshold"):
        select_fixed_fa_operating_points(curve, [0.0])

    no_targets = replace(
        curve[0],
        target_components=0,
        matched_components=0,
        unmatched_target_components=0,
        pd=None,
    )
    with pytest.raises(ValueError, match="at least one target"):
        select_fixed_fa_operating_points([no_targets], [10.0])


def test_calibration_convenience_keeps_grid_curve_and_selections_together() -> None:
    logits = np.full((5, 5), -1.0)
    target = np.zeros((5, 5), dtype=np.uint8)
    logits[2, 2] = 2.0
    target[2, 2] = 1

    result = calibrate_component_operating_points(
        [logits],
        [target],
        [0.0],
        fixed_thresholds=(0.0,),
        tail_quantiles=(1.0,),
    )

    assert result.threshold_grid == (0.0, 2.0)
    assert tuple(point.threshold for point in result.curve) == result.threshold_grid
    assert result.selections[0].threshold == 0.0
    assert result.selections[0].operating_point.pd == 1.0


def test_local_peak_can_be_positive_without_component_match() -> None:
    logits = np.full((15, 15), -1.0)
    target = np.zeros((15, 15), dtype=np.uint8)
    target[7, 2] = 1
    # The active component touches the target but its long tail moves its
    # centroid six pixels away, outside the strict component matching radius.
    logits[7, 2:15] = 1.0

    status = evaluate_target_operating_status(
        logits,
        target,
        target_index=0,
        threshold=0.0,
    )

    assert status.neighborhood_peak_above_threshold
    assert status.neighborhood_peak == 1.0
    assert status.neighborhood_margin == 1.0
    assert not status.matched


def test_component_match_can_hold_without_strict_neighborhood_peak() -> None:
    logits = np.full((11, 11), -1.0)
    target = np.zeros((11, 11), dtype=np.uint8)
    target[5, 5] = 1
    # A square perimeter has the same centroid as the target, while every
    # active pixel is at Euclidean distance >= 3 from its support.
    logits[2, 2:9] = 1.0
    logits[8, 2:9] = 1.0
    logits[2:9, 2] = 1.0
    logits[2:9, 8] = 1.0

    status = evaluate_target_operating_status(
        logits,
        target,
        target_index=0,
        threshold=0.0,
    )

    assert status.matched
    assert status.matched_centroid_distance == pytest.approx(0.0)
    assert not status.neighborhood_peak_above_threshold
    assert status.neighborhood_peak == -1.0


@pytest.mark.parametrize(
    ("logits", "target", "thresholds"),
    [
        ([np.zeros((2, 2, 1))], [np.zeros((2, 2))], [0.0]),
        ([np.zeros((2, 2))], [np.zeros((3, 2))], [0.0]),
        ([np.full((2, 2), np.nan)], [np.zeros((2, 2))], [0.0]),
        ([np.zeros((2, 2))], [np.full((2, 2), 0.5)], [0.0]),
        ([np.zeros((2, 2))], [np.zeros((2, 2))], [np.inf]),
    ],
)
def test_operating_curve_rejects_invalid_samples(
    logits: list[np.ndarray],
    target: list[np.ndarray],
    thresholds: list[float],
) -> None:
    with pytest.raises(ValueError):
        evaluate_component_operating_points(logits, target, thresholds)


def test_grid_and_target_status_fail_closed_on_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="tail_quantiles"):
        build_logit_threshold_grid(
            [np.zeros((2, 2))],
            fixed_thresholds=(),
            tail_quantiles=(0.49,),
        )
    with pytest.raises(ValueError, match="both be empty"):
        build_logit_threshold_grid(
            [np.zeros((2, 2))],
            fixed_thresholds=(),
            tail_quantiles=(),
        )
    with pytest.raises(ValueError, match="target_index"):
        evaluate_target_operating_status(
            np.zeros((2, 2)),
            np.asarray([[1, 0], [0, 0]]),
            target_index=1,
            threshold=0.0,
        )
