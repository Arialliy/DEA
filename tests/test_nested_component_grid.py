from __future__ import annotations

import json

import numpy as np
import pytest

import utils.nested_component_grid as nested_grid
from utils.component_operating_point import (
    DEFAULT_TAIL_QUANTILES,
    ComponentOperatingPoint,
    build_logit_threshold_grid,
)
from utils.nested_component_grid import (
    MATCHER_IMPLEMENTATIONS,
    NestedComponentGridError,
    build_nested_quantile_probability_grids,
    evaluate_nested_component_grids,
    exact_budget_feasible,
    select_exact_budget_point,
    summarize_nested_component_grids,
)


def _sample() -> tuple[np.ndarray, np.ndarray]:
    logits = np.full((8, 8), -3.0)
    target = np.zeros((8, 8), dtype=np.uint8)
    target[2, 2] = 1
    logits[2, 2] = 3.0
    logits[6, 6] = 2.0
    return logits, target


def _point(
    threshold: float,
    *,
    matches: int,
    false_area: int,
    stored_fa: float,
) -> ComponentOperatingPoint:
    return ComponentOperatingPoint(
        threshold=threshold,
        sample_count=1,
        total_pixels=100_000,
        matched_components=matches,
        target_components=10,
        prediction_components=matches + int(false_area > 0),
        unmatched_target_components=10 - matches,
        unmatched_prediction_components=int(false_area > 0),
        unmatched_prediction_area=false_area,
        pd=matches / 10.0,
        fa_per_million_pixels=stored_fa,
    )


def test_default_probability_grids_are_nested_53_105_209() -> None:
    grids = build_nested_quantile_probability_grids()
    q0, q1, q2 = (grid.probabilities for grid in grids)

    assert tuple(grid.level for grid in grids) == ("Q0", "Q1", "Q2")
    assert q0 == DEFAULT_TAIL_QUANTILES
    assert (len(q0), len(q1), len(q2)) == (53, 105, 209)
    assert set(q0) < set(q1) < set(q2)
    assert all(left < right for left, right in zip(q2, q2[1:]))


def test_midpoint_refinement_has_frozen_arithmetic_definition() -> None:
    grids = build_nested_quantile_probability_grids((0.5, 0.75, 1.0))

    assert grids[0].probabilities == (0.5, 0.75, 1.0)
    assert grids[1].probabilities == (0.5, 0.625, 0.75, 0.875, 1.0)
    assert grids[2].probabilities == (
        0.5,
        0.5625,
        0.625,
        0.6875,
        0.75,
        0.8125,
        0.875,
        0.9375,
        1.0,
    )


def test_q0_thresholds_exactly_equal_existing_default_builder() -> None:
    logits, target = _sample()
    expected = build_logit_threshold_grid(
        [logits],
        fixed_thresholds=(0.0,),
        tail_quantiles=DEFAULT_TAIL_QUANTILES,
    )

    result = evaluate_nested_component_grids([logits], [target], [0, 10])

    for matcher, _ in MATCHER_IMPLEMENTATIONS:
        assert result.matcher(matcher).level("Q0").threshold_grid == expected


def test_only_q2_curve_is_evaluated_once_per_matcher(monkeypatch) -> None:
    logits, target = _sample()
    actual_evaluator = nested_grid.evaluate_component_operating_points
    calls: list[tuple[str, tuple[float, ...]]] = []

    def recording_evaluator(
        logit_samples,
        target_samples,
        thresholds,
        *,
        matching,
        centroid_radius,
        connectivity,
    ):
        threshold_tuple = tuple(thresholds)
        calls.append((matching, threshold_tuple))
        return actual_evaluator(
            logit_samples,
            target_samples,
            threshold_tuple,
            matching=matching,
            centroid_radius=centroid_radius,
            connectivity=connectivity,
        )

    monkeypatch.setattr(
        nested_grid,
        "evaluate_component_operating_points",
        recording_evaluator,
    )
    result = evaluate_nested_component_grids(
        [logits],
        [target],
        [0],
        base_quantiles=(0.5, 0.75, 1.0),
    )

    assert [matching for matching, _ in calls] == ["legacy", "hungarian"]
    for (matching, thresholds), (_, implementation) in zip(
        calls, MATCHER_IMPLEMENTATIONS
    ):
        assert matching == implementation
        matcher = next(
            item
            for item in result.matchers
            if item.matching_implementation == implementation
        )
        assert thresholds == matcher.level("Q2").threshold_grid


def test_coarse_curves_are_exact_membership_projections_of_q2() -> None:
    logits, target = _sample()
    result = evaluate_nested_component_grids(
        [logits],
        [target],
        [0, 20],
        base_quantiles=(0.5, 0.75, 1.0),
    )

    for matcher in result.matchers:
        q0 = matcher.level("Q0")
        q1 = matcher.level("Q1")
        q2 = matcher.level("Q2")
        assert set(q0.threshold_grid).issubset(q1.threshold_grid)
        assert set(q1.threshold_grid).issubset(q2.threshold_grid)
        q2_points = {point.threshold: point for point in q2.curve}
        for level in (q0, q1):
            assert level.curve == tuple(
                q2_points[threshold] for threshold in level.threshold_grid
            )


def test_exact_selector_ignores_stored_float_fa_and_uses_frozen_ties() -> None:
    curve = (
        # Infeasible at budget 10 despite its deliberately false stored FA.
        _point(0.0, matches=9, false_area=2, stored_fa=0.0),
        # Both points below are exactly feasible: 1e6 <= 10 * 100000.
        _point(1.0, matches=8, false_area=1, stored_fa=1e12),
        _point(2.0, matches=8, false_area=1, stored_fa=1e12),
        _point(3.0, matches=7, false_area=0, stored_fa=0.0),
    )

    selection = select_exact_budget_point(curve, 10)
    zero_budget = select_exact_budget_point(curve, 0)

    assert selection.threshold == 2.0
    assert selection.integer_margin == 0
    assert zero_budget.threshold == 3.0
    assert exact_budget_feasible(1, 100_000, 10)
    assert not exact_budget_feasible(2, 100_000, 10)


@pytest.mark.parametrize(
    "base_quantiles",
    [
        (),
        (0.5,),
        (0.5, 0.5, 1.0),
        (0.75, 0.5, 1.0),
        (0.49, 1.0),
        (0.5, 1.01),
        (0.5, float("nan")),
        (0.5, True),
    ],
)
def test_probability_grid_rejects_invalid_protocol(
    base_quantiles: tuple[object, ...],
) -> None:
    with pytest.raises(NestedComponentGridError):
        build_nested_quantile_probability_grids(base_quantiles)


@pytest.mark.parametrize("budget", [True, 1.0, -1])
def test_exact_budget_rejects_non_integer_or_negative_budget(budget: object) -> None:
    curve = (_point(0.0, matches=1, false_area=0, stored_fa=0.0),)
    with pytest.raises(NestedComponentGridError, match="non-negative integer"):
        select_exact_budget_point(curve, budget)  # type: ignore[arg-type]


def test_nested_evaluation_rejects_bad_sample_and_budget_inputs() -> None:
    logits, target = _sample()
    with pytest.raises(NestedComponentGridError, match="equal length"):
        evaluate_nested_component_grids([logits], [], [1])
    with pytest.raises(NestedComponentGridError, match="non-empty"):
        evaluate_nested_component_grids([logits], [target], [])
    with pytest.raises(NestedComponentGridError, match="unique"):
        evaluate_nested_component_grids([logits], [target], [1, 1])

    no_target = np.zeros_like(target)
    with pytest.raises(NestedComponentGridError, match="target component"):
        evaluate_nested_component_grids(
            [logits],
            [no_target],
            [1],
            base_quantiles=(0.5, 1.0),
        )


def test_summary_is_json_serializable_and_preserves_exact_budget_evidence() -> None:
    logits, target = _sample()
    result = evaluate_nested_component_grids(
        [logits],
        [target],
        [0, 20],
        base_quantiles=(0.5, 0.75, 1.0),
    )

    summary = summarize_nested_component_grids(result)
    encoded = json.dumps(summary, sort_keys=True, allow_nan=False)

    assert encoded
    assert summary["probability_grids_are_nested"]
    assert summary["threshold_grids_are_nested"]
    assert summary["level_sizes"]["Q2"]["quantile_probability_count"] == 9
    assert set(summary["matchers"]) == {
        "official_legacy",
        "audit_hungarian",
    }
    for matcher in summary["matchers"].values():
        for level in matcher["levels"].values():
            assert all(
                selection["budget_feasible_exact"]
                for selection in level["budget_selections"].values()
            )


def test_summary_rejects_wrong_input_type() -> None:
    with pytest.raises(NestedComponentGridError, match="NestedComponentGridResult"):
        summarize_nested_component_grids({})  # type: ignore[arg-type]
