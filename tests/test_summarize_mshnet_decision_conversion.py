import pytest

from tools.summarize_mshnet_decision_conversion import (
    _pool_component_points,
    summarize_outcome,
    summarize_pairs,
)


BUDGETS = ("1", "10", "20")


def _conversion(*, margin, utilization, availability=2.0, sensitivity=3.0, stable=True):
    return {
        "available": True,
        "mean_logit_margin": margin,
        "utilization_cosine": utilization,
        "mean_margin_availability": availability,
        "normalized_mean_margin_availability": availability / 2.0,
        "head_sensitivity": sensitivity,
        "absolute_scale_floor_active_channels": 0 if stable else 1,
        "reparameterization_stable": stable,
    }


def _status(*, matched, local_peak, margin):
    return {
        "matched": matched,
        "neighborhood_peak_above_threshold": local_peak,
        "neighborhood_margin": margin,
    }


def _record(
    *,
    outcome,
    sample_name,
    target_index,
    final_margin,
    final_utilization,
    geometry_state,
    context_available=True,
    context_state="distinct",
    paired_no_response_id=None,
):
    fixed_detected = outcome == "matched_control"
    fixed_status = _status(
        matched=fixed_detected,
        local_peak=fixed_detected,
        margin=4.0 if fixed_detected else -4.0,
    )
    conversions = {
        stage: _conversion(margin=1.0, utilization=0.25)
        for stage in ("d0", "d1", "d2", "d3")
    }
    conversions["final"] = _conversion(
        margin=final_margin,
        utilization=final_utilization,
        stable=outcome == "matched_control",
    )
    record = {
        "_run_key": "dataset/seed_1",
        "outcome": outcome,
        "sample_name": sample_name,
        "target_index": target_index,
        "conversion": conversions,
        "controls": {
            "geometry": {
                "available": True,
                "survival": {
                    "d0": {"available": True, "state": geometry_state}
                },
            },
            "context_matched": {
                "available": context_available,
                "survival": {
                    "d0": {
                        "available": context_available,
                        "state": context_state if context_available else "undefined",
                    }
                },
            },
        },
        "contribution_margins": {
            "available": True,
            "has_sign_cancellation": outcome == "no_response",
            "positive_sum": 2.0,
            "negative_sum": -1.0 if outcome == "no_response" else 0.0,
            "final_direct": final_margin,
        },
        "operating_point": {
            "fixed_threshold": fixed_status,
            "cross_fitted_fixed_fa": {
                budget: {
                    "status": _status(
                        matched=fixed_detected and budget != "1",
                        local_peak=fixed_detected and budget != "1",
                        margin=float(budget) - 5.0,
                    )
                }
                for budget in BUDGETS
            },
        },
    }
    if paired_no_response_id is not None:
        record["paired_no_response_id"] = paired_no_response_id
        record["pair_index"] = 0
    return record


def test_outcome_summary_keeps_explicit_denominators_and_stage_signs():
    records = [
        _record(
            outcome="no_response",
            sample_name="miss_a",
            target_index=0,
            final_margin=-2.0,
            final_utilization=-0.2,
            geometry_state="distinct",
            context_available=True,
            context_state="background_like",
        ),
        _record(
            outcome="no_response",
            sample_name="miss_b",
            target_index=1,
            final_margin=0.0,
            final_utilization=0.0,
            geometry_state="uncertain",
            context_available=False,
        ),
    ]

    summary = summarize_outcome(records, BUDGETS)

    geometry = summary["controls"]["geometry"]
    context = summary["controls"]["context_matched"]
    assert geometry["d0_distinct"] == 1
    assert geometry["d0_distinct_denominator"] == 2
    assert context["d0_distinct"] == 0
    assert context["d0_distinct_denominator"] == 1
    final = summary["stages"]["final"]
    assert final["mean_margin"] == {
        "defined": 2,
        "positive": 0,
        "negative": 1,
        "zero": 1,
        "median": -1.0,
    }
    assert final["utilization"]["negative"] == 1
    assert final["utilization"]["zero"] == 1
    assert final["availability"]["median"] == 2.0
    assert final["head_sensitivity"]["median"] == 3.0
    assert final["floor_stability"] == {
        "eligible": 2,
        "stable": 0,
        "rate": 0.0,
    }
    assert summary["final_contribution_signs"]["sign_cancellation"] == 2
    assert summary["fixed_threshold"]["exact_component_detected"] == 0
    assert summary["cross_fitted_fixed_fa"]["20"][
        "local_peak_above_threshold"
    ] == 0


def test_paired_final_difference_is_miss_minus_control():
    miss = _record(
        outcome="no_response",
        sample_name="miss",
        target_index=2,
        final_margin=-2.0,
        final_utilization=-0.2,
        geometry_state="distinct",
    )
    control = _record(
        outcome="matched_control",
        sample_name="control",
        target_index=1,
        final_margin=4.0,
        final_utilization=0.4,
        geometry_state="distinct",
        paired_no_response_id="miss:2",
    )

    summary = summarize_pairs([miss, control])

    assert summary["difference_orientation"] == "no_response_minus_matched_control"
    assert summary["candidate_pairs"] == 1
    assert summary["final_utilization"]["median_difference"] == pytest.approx(-0.6)
    assert summary["final_mean_logit_margin"]["median_difference"] == -6.0


def test_component_points_are_pooled_from_raw_sufficient_statistics():
    points = [
        {
            "sample_count": 1,
            "total_pixels": 100,
            "target_components": 2,
            "prediction_components": 2,
            "matched_components": 1,
            "unmatched_target_components": 1,
            "unmatched_prediction_components": 1,
            "unmatched_prediction_area": 10,
        },
        {
            "sample_count": 9,
            "total_pixels": 900,
            "target_components": 8,
            "prediction_components": 8,
            "matched_components": 7,
            "unmatched_target_components": 1,
            "unmatched_prediction_components": 1,
            "unmatched_prediction_area": 0,
        },
    ]

    pooled = _pool_component_points(points)

    assert pooled["actual_evaluation_pd"] == 0.8
    assert pooled["actual_evaluation_fa_per_million_pixels"] == 10_000.0
    assert pooled["sample_count"] == 10
