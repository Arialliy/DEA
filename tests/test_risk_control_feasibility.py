from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.audit_gate_f0_risk_control_feasibility import (
    _candidate_grid_size,
    _fold_sizes,
    _validate_equal_image_area,
    write_bundle,
)

from utils.risk_control_feasibility import (
    RiskControlFeasibilityError,
    analyze_risk_control_feasibility,
    hb_ltt_zero_loss_pvalue,
    minimum_crc_sample_count_for_budget,
    minimum_zero_loss_sample_count,
    standard_crc_unit_bound_floor,
)


def test_hb_zero_loss_formula_and_minimum_sample_count() -> None:
    alpha = 20 / 1_000_000
    assert hb_ltt_zero_loss_pvalue(82, alpha) == pytest.approx((1 - alpha) ** 82)
    assert minimum_zero_loss_sample_count(alpha, 0.1) == 115_129
    required = minimum_zero_loss_sample_count(alpha, 0.1 / 54)
    assert required == 314_576
    assert hb_ltt_zero_loss_pvalue(required, alpha) <= 0.1 / 54
    assert hb_ltt_zero_loss_pvalue(required - 1, alpha) > 0.1 / 54


def test_standard_crc_unit_bound_floor_is_far_above_low_fa_budget() -> None:
    assert standard_crc_unit_bound_floor(82) == pytest.approx(1 / 83)
    assert standard_crc_unit_bound_floor(82) > 20 / 1_000_000
    assert minimum_crc_sample_count_for_budget(20) == 49_999
    assert 1 / (49_999 + 1) <= 20 / 1_000_000
    assert 1 / 49_999 > 20 / 1_000_000


def test_feasibility_summary_keeps_crossfit_and_optimistic_scopes_separate() -> None:
    rows, summary = analyze_risk_control_feasibility(
        {"D": {0: 82, 1: 78}},
        budgets_fa_per_mpix=(20,),
        candidate_grid_size=54,
        confidence_deltas=(0.1, 0.05),
    )

    assert {row["sample_count"] for row in rows} == {78, 82, 160}
    assert sum(row["deployable_split"] for row in rows) == 2
    budget = summary["by_budget"]["20"]
    assert budget["maximum_crossfit_calibration_image_count"] == 82
    assert budget["maximum_full_development_image_count"] == 160
    assert budget["any_crossfit_fixed_sequence_certifiable"] is False
    assert budget["any_full_development_fixed_sequence_certifiable"] is False
    assert summary["pre_gate"]["full_threshold_inference_needed_for_this_precheck"] is False


@pytest.mark.parametrize(
    ("sample_count", "target_risk", "message"),
    [
        (0, 0.1, "sample_count"),
        (1, 0.0, "target_risk"),
        (1, 1.0, "target_risk"),
        (1, math.nan, "target_risk"),
    ],
)
def test_hb_inputs_fail_closed(
    sample_count: int, target_risk: float, message: str
) -> None:
    with pytest.raises(RiskControlFeasibilityError, match=message):
        hb_ltt_zero_loss_pvalue(sample_count, target_risk)


def test_feasibility_rejects_invalid_fold_or_candidate_contract() -> None:
    with pytest.raises(RiskControlFeasibilityError, match="two image folds"):
        analyze_risk_control_feasibility(
            {"D": {0: 2}},
            budgets_fa_per_mpix=(20,),
            candidate_grid_size=54,
        )
    with pytest.raises(RiskControlFeasibilityError, match="candidate_grid_size"):
        analyze_risk_control_feasibility(
            {"D": {0: 2, 1: 2}},
            budgets_fa_per_mpix=(20,),
            candidate_grid_size=0,
        )


def test_tool_validates_grid_folds_and_equal_image_area() -> None:
    folds = {"D": {"a": 0, "b": 1}}
    assert _fold_sizes(folds, fold_count=2) == {"D": {0: 1, 1: 1}}
    calibration = [
        {
            "schema_version": "dea.gate_e.low_fa_calibration.v1",
            "threshold_grid": [0.0, 1.0],
            "selections": {"20": {}},
        }
    ]
    assert _candidate_grid_size(calibration, budgets=(20,)) == 2
    pixels, counts = _validate_equal_image_area(
        [
            {"dataset": "D", "image_name": "a", "total_pixels": 16},
            {"dataset": "D", "image_name": "b", "total_pixels": 16},
        ],
        fold_mappings=folds,
    )
    assert pixels == 16
    assert counts == {"D": 2}
    with pytest.raises(RiskControlFeasibilityError, match="unequal image areas"):
        _validate_equal_image_area(
            [
                {"dataset": "D", "image_name": "a", "total_pixels": 16},
                {"dataset": "D", "image_name": "b", "total_pixels": 25},
            ],
            fold_mappings=folds,
        )


def test_risk_feasibility_bundle_is_atomic_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    summary = {
        "analysis_scope": "test",
        "protocol": {"budgets_fa_per_mpix": []},
        "by_budget": {},
        "pre_gate": {"decision": "test"},
        "scope_limit": "test",
    }
    write_bundle(
        output,
        rows=[],
        summary=summary,
        provenance={"schema_version": "test"},
    )
    assert {path.name for path in output.iterdir()} == {
        "risk_feasibility.jsonl",
        "risk_feasibility_summary.json",
        "risk_feasibility_summary.md",
        "provenance.json",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_bundle(output, rows=[], summary=summary, provenance={})
