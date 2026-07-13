import json

import numpy as np
import pytest

from tools.audit_phase_survival_gate import (
    PhaseSurvivalAuditError,
    _hard_core_panel,
    build_summary,
    oracle_summary,
)
from utils.cross_fitted_low_fa import BUDGETS, MATCHERS


def test_hard_core_panel_requires_three_seed_consensus_and_16_targets(tmp_path) -> None:
    path = tmp_path / "targets.jsonl"
    rows = []
    for target_index in range(16):
        for seed in (11, 12, 13):
            rows.append(
                {
                    "grid_level": "Q2",
                    "nominal_budget_fa_per_mpix": 20,
                    "category_core": "no_feasible_local_peak_activation",
                    "dataset": "D%d" % (target_index % 3),
                    "stable_target_id": "target-%02d" % target_index,
                    "image_name": "image-%02d" % target_index,
                    "target_area": target_index + 1,
                    "seed": seed,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    panel = _hard_core_panel(path)
    assert len(panel) == 16
    assert all(row["source_seeds"] == [11, 12, 13] for row in panel)

    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8"
    )
    with pytest.raises(PhaseSurvivalAuditError, match="must contain 16"):
        _hard_core_panel(path)


def test_oracle_summary_reports_both_matchers_and_all_budgets() -> None:
    targets = []
    scores = []
    for row, column in ((2, 2), (5, 5)):
        target = np.zeros((8, 8), dtype=bool)
        target[row, column] = True
        score = np.full((8, 8), -2.0, dtype=np.float64)
        score[row, column] = 2.0
        targets.append(target)
        scores.append(score)
    summary = oracle_summary(scores, targets)
    assert summary["threshold_count"] > 1
    for matcher in MATCHERS:
        assert set(summary["by_matcher"][matcher]) == {str(value) for value in BUDGETS}
        for budget in BUDGETS:
            point = summary["by_matcher"][matcher][str(budget)]
            assert point["matched_components"] == 2
            assert point["unmatched_prediction_area"] == 0


def test_build_summary_keeps_sentinel_and_full_seed_semantics_separate() -> None:
    datasets = ("A", "B")
    sentinel = build_summary(
        [
            {"dataset": "A", "seed": 11, "job_pass": True},
            {"dataset": "B", "seed": 11, "job_pass": False},
        ],
        requested_datasets=datasets,
        requested_seeds=(11,),
    )
    assert sentinel["sentinel_mode"] is True
    assert sentinel["phase_survival_gate_pass"] is False
    assert sentinel["method_training_authorization"] is False

    full = build_summary(
        [
            {"dataset": dataset, "seed": seed, "job_pass": seed != 13}
            for dataset in datasets
            for seed in (11, 12, 13)
        ],
        requested_datasets=datasets,
        requested_seeds=(11, 12, 13),
    )
    assert full["sentinel_mode"] is False
    assert full["phase_survival_gate_pass"] is True
    assert all(row["required_passing_seed_count"] == 2 for row in full["by_dataset"].values())


def test_build_summary_fails_closed_on_missing_job() -> None:
    with pytest.raises(PhaseSurvivalAuditError, match="requested seed grid"):
        build_summary(
            [{"dataset": "A", "seed": 11, "job_pass": True}],
            requested_datasets=("A",),
            requested_seeds=(11, 12),
        )
