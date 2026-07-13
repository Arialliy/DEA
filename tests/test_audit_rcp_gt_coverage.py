import json

import pytest

from tools.audit_rcp_gt_coverage import (
    RCPGTCoverageError,
    formal_hard_core_ids,
    summarize_areas,
)


def test_formal_hard_core_requires_three_exact_seeds_and_sixteen_targets(tmp_path) -> None:
    path = tmp_path / "targets.jsonl"
    rows = []
    for target_index in range(16):
        for seed in (20260711, 20260712, 20260713):
            rows.append(
                {
                    "grid_level": "Q2",
                    "nominal_budget_fa_per_mpix": 20,
                    "category_core": "no_feasible_local_peak_activation",
                    "dataset": "D",
                    "stable_target_id": f"T{target_index}",
                    "seed": seed,
                }
            )
    rows.append(
        {
            "grid_level": "Q1",
            "nominal_budget_fa_per_mpix": 20,
            "category_core": "no_feasible_local_peak_activation",
            "dataset": "D",
            "stable_target_id": "ignored",
            "seed": 20260711,
        }
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    panel = formal_hard_core_ids(path)
    assert len(panel) == 16
    assert ("D", "T0") in panel

    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-4]))
    with pytest.raises(RCPGTCoverageError, match="16 targets"):
        formal_hard_core_ids(path)


def test_summarize_areas_reports_exact_cap_coverage() -> None:
    rows = [
        {
            "dataset": "D",
            "split": "fit",
            "image_name": f"I{index}",
            "target_area": area,
            "exact_roundtrip": True,
        }
        for index, area in enumerate((1, 4, 9, 16))
    ]
    summary = summarize_areas(rows, caps=(4, 8, 16))
    assert summary["target_count"] == 4
    assert summary["maximum_area"] == 16
    assert summary["exact_roundtrip_targets"] == 4
    assert summary["node_cap_coverage"]["4"] == {
        "covered_targets": 2,
        "coverage": 0.5,
        "uncovered_targets": 2,
    }
    assert summary["node_cap_coverage"]["16"]["coverage"] == 1.0


def test_summarize_areas_fails_on_empty_or_invalid_caps() -> None:
    with pytest.raises(RCPGTCoverageError, match="positive target areas"):
        summarize_areas([])
    row = {
        "dataset": "D",
        "split": "fit",
        "image_name": "I",
        "target_area": 1,
        "exact_roundtrip": True,
    }
    with pytest.raises(RCPGTCoverageError, match="caps"):
        summarize_areas([row], caps=(8, 4))
