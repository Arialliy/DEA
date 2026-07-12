from __future__ import annotations

import copy

import numpy as np
import pytest

from tools.audit_gate_e_low_fa_bridge import write_bundle as write_low_fa_bundle

from utils.cross_fitted_low_fa import (
    BUDGETS,
    MATCHERS,
    LowFABridgeError,
    budget_feasible_exact,
    cross_fit_job,
    image_fold,
    summarize_low_fa_bridge,
    validate_hungarian_fixed_alignment,
)
from utils.target_identity import build_stable_target_set


def _names_in_both_folds() -> tuple[str, str]:
    by_fold: dict[int, str] = {}
    for index in range(1000):
        name = f"image-{index}"
        by_fold.setdefault(image_fold(name), name)
        if set(by_fold) == {0, 1}:
            return by_fold[0], by_fold[1]
    raise AssertionError("could not construct fold fixture")


def test_image_fold_is_deterministic_and_name_only() -> None:
    assert image_fold("x") == image_fold("x")
    with pytest.raises(LowFABridgeError, match="image_name"):
        image_fold("")


def test_budget_feasibility_uses_exact_integer_cross_multiplication() -> None:
    assert budget_feasible_exact(1, 1_000_000, 1)
    assert not budget_feasible_exact(2, 1_000_000, 1)
    assert budget_feasible_exact(0, 7, 0)
    with pytest.raises(LowFABridgeError, match="positive int"):
        budget_feasible_exact(0, 0, 1)


def test_cross_fit_grid_uses_only_calibration_logits_and_keeps_target_free_images() -> None:
    fold0, fold1 = _names_in_both_folds()
    names = (fold0, fold1)
    target0 = np.zeros((8, 8), dtype=np.uint8)
    target1 = np.zeros((8, 8), dtype=np.uint8)
    target0[2, 2] = 1
    target1[5, 5] = 1
    logits0 = np.full((8, 8), -3.0)
    logits1 = np.full((8, 8), -3.0)
    logits0[2, 2] = 100.0
    logits1[5, 5] = 2.0
    registry = {
        fold0: build_stable_target_set(
            target0, dataset="D", image_name=fold0, connectivity=2
        ),
        fold1: build_stable_target_set(
            target1, dataset="D", image_name=fold1, connectivity=2
        ),
    }

    target_rows, image_rows, calibration = cross_fit_job(
        (logits0, logits1),
        (target0, target1),
        names,
        dataset="D",
        seed=1,
        registry=registry,
        checkpoint={"policy": "fixed_epoch", "epoch": 399},
    )

    eval_fold0 = [
        row
        for row in calibration
        if row["evaluation_fold"] == 0 and row["matcher"] == "audit_hungarian"
    ][0]
    assert max(eval_fold0["threshold_grid"]) == 2.0
    assert 100.0 not in eval_fold0["threshold_grid"]
    assert len(target_rows) == 2 * len(MATCHERS) * len(BUDGETS)
    assert len(image_rows) == 2 * len(MATCHERS) * len(BUDGETS)
    assert all(row["held_out_fold_aggregate"]["total_pixels"] == 64 for row in image_rows)
    assert all(
        row["dataset_seed_aggregate"]["total_pixels"] == 128 for row in image_rows
    )
    assert all(
        row["dataset_seed_aggregate"]["aggregation"]
        == "integer counts pooled across both held-out folds"
        for row in image_rows
    )


def _bridge_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    target_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    for dataset in ("A", "B"):
        for seed in (1, 2, 3):
            for matcher in MATCHERS:
                for budget in BUDGETS:
                    aggregate = {
                        "budget_feasible_zero_overshoot": True,
                        "matched_components": 20,
                    }
                    image_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "matcher": matcher,
                            "nominal_budget_fa_per_mpix": budget,
                            "dataset_seed_aggregate": aggregate,
                        }
                    )
                    for index in range(24):
                        bridge = index < 6
                        target_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "matcher": matcher,
                                "nominal_budget_fa_per_mpix": budget,
                                "stable_target_id": f"{dataset}-{index}",
                                "fixed_logit0_matched": not bridge,
                                "low_fa_matched": not bridge,
                            }
                        )
    return target_rows, image_rows


def test_joint_bridge_requires_both_matchers_overlap_controls_and_non_all_off() -> None:
    target_rows, image_rows = _bridge_fixture()
    summary = summarize_low_fa_bridge(target_rows, image_rows)

    assert summary["joint_gate"]["pass"]
    assert summary["joint_gate"]["by_budget"]["1"][
        "joint_fixed0_and_low_fa_repeated_miss_target_count"
    ] == 12
    assert summary["joint_gate"]["by_budget"]["1"][
        "joint_stable_control_target_count"
    ] == 36

    all_off = copy.deepcopy(image_rows)
    for row in all_off:
        if (
            row["dataset"] == "A"
            and row["seed"] == 1
            and row["matcher"] == "official_legacy"
        ):
            row["dataset_seed_aggregate"]["matched_components"] = 0
    blocked = summarize_low_fa_bridge(target_rows, all_off)
    assert blocked["joint_gate"]["pass"] is False


def test_hungarian_fixed_alignment_rejects_status_drift() -> None:
    low = []
    fixed = []
    for seed, matched in ((1, True), (2, False), (3, True)):
        fixed.append(
            {
                "row_kind": "target",
                "stable_target_id": "k",
                "seed": seed,
                "matched": matched,
            }
        )
        for budget in BUDGETS:
            low.append(
                {
                    "matcher": "audit_hungarian",
                    "stable_target_id": "k",
                    "seed": seed,
                    "fixed_logit0_matched": matched,
                    "nominal_budget_fa_per_mpix": budget,
                }
            )
    validate_hungarian_fixed_alignment(low, fixed)
    low[0]["fixed_logit0_matched"] = False
    with pytest.raises(LowFABridgeError, match="varies across budgets"):
        validate_hungarian_fixed_alignment(low, fixed)


def test_low_fa_bundle_is_atomic_and_refuses_overwrite(tmp_path) -> None:
    target_rows, image_rows = _bridge_fixture()
    summary = summarize_low_fa_bridge(target_rows, image_rows)
    output = tmp_path / "bridge"
    write_low_fa_bundle(
        output,
        target_rows=target_rows,
        image_rows=image_rows,
        calibration=[],
        summary=summary,
        provenance={"schema_version": "test"},
    )
    assert {path.name for path in output.iterdir()} == {
        "target_low_fa.jsonl",
        "image_low_fa.jsonl",
        "calibration.json",
        "low_fa_bridge_summary.json",
        "low_fa_bridge_summary.md",
        "provenance.json",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_low_fa_bundle(
            output,
            target_rows=[],
            image_rows=[],
            calibration=[],
            summary=summary,
            provenance={"schema_version": "test"},
        )
