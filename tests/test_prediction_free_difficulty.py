from __future__ import annotations

import copy

import numpy as np
import pytest

from tools.audit_gate_e_prediction_free_difficulty import (
    write_bundle as write_difficulty_bundle,
)

from utils.prediction_free_difficulty import (
    COVARIATES,
    DifficultyAuditError,
    _average_precision,
    _rank_auc,
    _separation_status,
    compute_prediction_free_covariates,
    join_fixed_outcomes,
    lodo_analysis,
    summarize_difficulty,
)


def test_covariates_follow_rgb_luminance_ring_and_border_contract() -> None:
    rgb = np.full((32, 32, 3), 10, dtype=np.uint8)
    target = np.zeros((32, 32), dtype=np.uint8)
    target[15:17, 15:17] = 1
    rgb[15:17, 15:17] = 30

    rows = compute_prediction_free_covariates(
        rgb, target, dataset="D", image_name="image"
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["area"] == 4
    assert row["border_distance"] == 15.0
    assert row["ring_pixel_count"] >= 16
    assert row["ring_available"]
    assert row["local_ring_robust_dispersion"] == 0.0
    assert row["local_robust_scr"] == pytest.approx((20 / 255) / 1e-6)


def test_covariates_fail_closed_on_non_uint8_and_mark_small_ring_unavailable() -> None:
    target = np.ones((2, 2), dtype=np.uint8)
    with pytest.raises(DifficultyAuditError, match="uint8"):
        compute_prediction_free_covariates(
            np.zeros((2, 2, 3), dtype=np.float32),
            target,
            dataset="D",
            image_name="x",
        )

    rows = compute_prediction_free_covariates(
        np.zeros((2, 2, 3), dtype=np.uint8),
        target,
        dataset="D",
        image_name="x",
    )
    assert rows[0]["ring_available"] is False
    assert rows[0]["local_robust_scr"] is None
    assert rows[0]["local_ring_robust_dispersion"] is None


def _feature(key: str = "k") -> dict[str, object]:
    return {
        "stable_target_id": key,
        "dataset": "D",
        "image_name": "i",
        "area": 1,
        "component_mask_sha256": "a" * 64,
        "label_mask_sha256": "b" * 64,
        **{name: 1.0 for name in COVARIATES},
    }


def _ledger(seed: int, matched: bool, key: str = "k") -> dict[str, object]:
    return {
        "row_kind": "target",
        "stable_target_id": key,
        "dataset": "D",
        "image_name": "i",
        "area": 1,
        "component_mask_sha256": "a" * 64,
        "label_mask_sha256": "b" * 64,
        "seed": seed,
        "matched": matched,
        "miss_count": 2,
        "checkpoint": {"policy": "fixed_epoch"},
    }


def test_join_fixed_outcomes_requires_three_seeds_and_recomputes_response() -> None:
    ledger = [_ledger(1, False), _ledger(2, False), _ledger(3, True)]

    joined = join_fixed_outcomes([_feature()], ledger)

    assert joined[0]["miss_count"] == 2
    assert joined[0]["miss_any_seed"] == 1
    assert joined[0]["miss_three_of_three"] == 0
    assert joined[0]["miss_seed_ids"] == [1, 2]
    with pytest.raises(DifficultyAuditError, match="exactly three"):
        join_fixed_outcomes([_feature()], ledger[:2])
    drift = copy.deepcopy(ledger)
    drift[0]["checkpoint"]["policy"] = "best_iou"
    with pytest.raises(DifficultyAuditError, match="fixed_epoch"):
        join_fixed_outcomes([_feature()], drift)


def test_auc_and_average_precision_are_tie_deterministic() -> None:
    y = np.asarray([0, 1, 0, 1], dtype=np.float64)
    assert _rank_auc(y, np.asarray([0.0, 1.0, 0.0, 1.0])) == 1.0
    assert _average_precision(y, np.asarray([0.0, 1.0, 0.0, 1.0])) == 1.0
    assert _rank_auc(y, np.zeros(4)) == 0.5


def test_separation_detector_distinguishes_complete_and_overlap() -> None:
    complete_design = np.column_stack((np.ones(4), [-2.0, -1.0, 1.0, 2.0]))
    complete_y = np.asarray([0, 0, 1, 1], dtype=np.float64)
    assert _separation_status(complete_design, complete_y) == "complete_separation"

    overlap_design = np.column_stack((np.ones(4), [-1.0, 1.0, -1.0, 1.0]))
    overlap_y = np.asarray([0, 0, 1, 1], dtype=np.float64)
    assert _separation_status(overlap_design, overlap_y) is None


def _synthetic_rows() -> list[dict[str, object]]:
    rows = []
    for dataset_index, dataset in enumerate(("A", "B", "C")):
        for index in range(24):
            positive = index < 12
            value = float(positive) + dataset_index * 0.01 + index * 1e-4
            rows.append(
                {
                    "dataset": dataset,
                    "image_name": f"{dataset}-{index}",
                    "miss_any_seed": int(positive),
                    "miss_three_of_three": int(positive and index < 11),
                    "log1p_area": value,
                    "border_distance": value * 2 + index * 0.001,
                    "local_robust_scr": value * 3 + index * 0.002,
                    "local_ring_robust_dispersion": value * 4 + index * 0.003,
                }
            )
    return rows


def test_lodo_uses_image_cluster_support_and_fixed_logistic() -> None:
    result = lodo_analysis(_synthetic_rows(), response="miss_any_seed")

    assert result["eligible_fold_count"] == 3
    assert all(fold["eligible"] for fold in result["folds"])
    assert all(fold["auroc"] == 1.0 for fold in result["folds"])
    assert all(fold["model"]["C"] == 1.0 for fold in result["folds"])


def test_summary_routes_highly_predictable_primary_to_no_go() -> None:
    summary = summarize_difficulty(_synthetic_rows())

    assert summary["availability"]["unresolved"] is False
    assert summary["routing"]["eligible_fold_count"] == 3
    assert summary["routing"]["near_complete_explanation"] is True
    assert (
        summary["routing"]["decision"]
        == "E0_NO_GO_PREDICTION_FREE_DIFFICULTY"
    )


def test_difficulty_bundle_is_atomic_and_refuses_overwrite(tmp_path) -> None:
    summary = summarize_difficulty(_synthetic_rows())
    output = tmp_path / "difficulty"
    write_difficulty_bundle(
        output,
        rows=_synthetic_rows(),
        summary=summary,
        provenance={"schema_version": "test"},
    )
    assert {path.name for path in output.iterdir()} == {
        "target_difficulty.jsonl",
        "difficulty_summary.json",
        "difficulty_summary.md",
        "provenance.json",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_difficulty_bundle(
            output,
            rows=[],
            summary=summary,
            provenance={"schema_version": "test"},
        )
