from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.audit_gate_f_nested_grid_sensitivity import (
    NestedGridAuditError,
    _key_calibration,
    _key_image,
    _key_target,
    _unique_index,
    analyze_job_nested_grids,
    summarize_nested_grid_sensitivity,
    write_bundle,
)
from utils.cross_fitted_low_fa import cross_fit_job, image_fold
from utils.target_identity import build_stable_target_set


def _two_fold_names() -> tuple[str, ...]:
    by_fold = {0: [], 1: []}
    candidate = 0
    while min(len(values) for values in by_fold.values()) < 2:
        name = f"synthetic_{candidate:03d}"
        fold = image_fold(name)
        if len(by_fold[fold]) < 2:
            by_fold[fold].append(name)
        candidate += 1
    return tuple(by_fold[0] + by_fold[1])


def _formal_synthetic_inputs():
    names = _two_fold_names()
    logits = []
    targets = []
    registry = {}
    for index, name in enumerate(names):
        target = np.zeros((8, 8), dtype=bool)
        target[1 + index, 1 + index] = True
        scores = np.full((8, 8), -4.0, dtype=np.float64)
        scores[target] = 5.0 + index
        scores[7, 7 - index] = 1.0 + index / 10.0
        targets.append(target)
        logits.append(scores)
        registry[name] = build_stable_target_set(
            target,
            dataset="D",
            image_name=name,
            connectivity=2,
        )
    checkpoint = {
        "policy": "fixed_epoch",
        "epoch": 1,
        "path": "/tmp/frozen.pth.tar",
        "sha256": "0" * 64,
        "job_id": "synthetic",
        "run_config_sha256": "1" * 64,
        "validation_split_sha256": "2" * 64,
    }
    formal_targets, formal_images, formal_calibration = cross_fit_job(
        logits,
        targets,
        names,
        dataset="D",
        seed=7,
        registry=registry,
        checkpoint=checkpoint,
    )
    return (
        tuple(logits),
        tuple(targets),
        names,
        registry,
        checkpoint,
        formal_targets,
        formal_images,
        formal_calibration,
    )


def test_job_audit_exactly_replays_q0_and_builds_all_ledgers() -> None:
    (
        logits,
        targets,
        names,
        registry,
        checkpoint,
        formal_targets,
        formal_images,
        formal_calibration,
    ) = _formal_synthetic_inputs()

    selections, images, target_rows, pairs, event = analyze_job_nested_grids(
        logits,
        targets,
        names,
        dataset="D",
        seed=7,
        registry=registry,
        checkpoint=checkpoint,
        frozen_calibration=_unique_index(
            formal_calibration, _key_calibration, label="calibration"
        ),
        frozen_images=_unique_index(formal_images, _key_image, label="image"),
        frozen_targets=_unique_index(
            formal_targets, _key_target, label="target"
        ),
    )

    assert len(selections) == 2 * 2 * 3 * 4
    assert len(images) == len(names) * 2 * 3 * 4
    assert len(target_rows) == len(names) * 2 * 3 * 4
    assert len(pairs) == 2 * 3 * 4
    assert event["checkpoint_image_forward_pairs"] == len(names)
    assert event["total_pixel_scores"] == len(names) * 64
    assert {row["grid_level"] for row in images} == {"Q0", "Q1", "Q2"}


def test_job_audit_fails_closed_when_frozen_q0_curve_is_tampered() -> None:
    (
        logits,
        targets,
        names,
        registry,
        checkpoint,
        formal_targets,
        formal_images,
        formal_calibration,
    ) = _formal_synthetic_inputs()
    formal_calibration[0] = {
        **formal_calibration[0],
        "threshold_grid": [999.0],
    }

    with pytest.raises(NestedGridAuditError, match="Q0 replay mismatch"):
        analyze_job_nested_grids(
            logits,
            targets,
            names,
            dataset="D",
            seed=7,
            registry=registry,
            checkpoint=checkpoint,
            frozen_calibration=_unique_index(
                formal_calibration, _key_calibration, label="calibration"
            ),
            frozen_images=_unique_index(
                formal_images, _key_image, label="image"
            ),
            frozen_targets=_unique_index(
                formal_targets, _key_target, label="target"
            ),
        )


def _summary_fixture():
    pair_rows = []
    target_rows = []
    for dataset_index, dataset in enumerate(("A", "B", "C")):
        for seed in (1, 2, 3):
            for matcher in ("official_legacy", "audit_hungarian"):
                for budget in (1, 5, 10, 20):
                    for level in ("Q0", "Q1", "Q2"):
                        q2_gain = level == "Q2" and dataset_index < 2 and seed < 3
                        calibration_matches = 50 + (3 if q2_gain else 0)
                        heldout_feasible = not (
                            level == "Q0"
                            and dataset_index < 2
                            and seed == 1
                            and budget == 20
                        )
                        pair_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "matcher": matcher,
                                "grid_level": level,
                                "nominal_budget_fa_per_mpix": budget,
                                "calibration_pooled": {
                                    "matched_components": calibration_matches,
                                    "pd": calibration_matches / 60.0,
                                },
                                "held_out_pooled": {
                                    "matched_components": 40,
                                    "pd": 40 / 60.0,
                                    "unmatched_prediction_area": 0,
                                    "budget_feasible_zero_overshoot": heldout_feasible,
                                },
                                "selected_thresholds_by_evaluation_fold": {
                                    "0": float(level[-1]),
                                    "1": float(level[-1]),
                                },
                            }
                        )
                    for target_id in ("t0", "t1"):
                        for level in ("Q0", "Q2"):
                            target_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "matcher": matcher,
                                    "grid_level": level,
                                    "nominal_budget_fa_per_mpix": budget,
                                    "stable_target_id": (
                                        f"{dataset}:{target_id}"
                                    ),
                                    "matched": level == "Q2",
                                }
                            )
    event_rows = [
        {
            "checkpoint_image_forward_pairs": 1,
            "total_pixel_scores": 64,
            "sum_image_local_unique_float32_score_groups": 60,
        }
        for _ in range(9)
    ]
    return pair_rows, target_rows, event_rows


def test_summary_applies_pre_registered_cross_dataset_trigger() -> None:
    pair_rows, target_rows, event_rows = _summary_fixture()
    summary = summarize_nested_grid_sensitivity(
        pair_rows, target_rows, event_rows
    )

    assert summary["targeted_exact_interval_gate"]["pass"]
    assert 20 in summary["targeted_exact_interval_gate"]["passing_budgets"]
    assert summary["event_scale"]["checkpoint_image_forward_pairs"] == 9
    assert summary["event_scale"]["image_local_unique_fraction"] == pytest.approx(
        60 / 64
    )


def test_bundle_writer_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    pair_rows, target_rows, event_rows = _summary_fixture()
    summary = summarize_nested_grid_sensitivity(
        pair_rows, target_rows, event_rows
    )
    output = tmp_path / "bundle"
    write_bundle(
        output,
        selection_rows=[],
        image_rows=[],
        target_rows=target_rows,
        pair_rows=pair_rows,
        event_rows=event_rows,
        summary=summary,
        provenance={"schema_version": "test"},
    )
    assert {path.name for path in output.iterdir()} == {
        "selection_sensitivity.jsonl",
        "image_sensitivity.jsonl",
        "target_sensitivity.jsonl",
        "pair_sensitivity.jsonl",
        "event_scale.jsonl",
        "nested_grid_summary.json",
        "nested_grid_summary.md",
        "provenance.json",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_bundle(
            output,
            selection_rows=[],
            image_rows=[],
            target_rows=[],
            pair_rows=[],
            event_rows=[],
            summary=summary,
            provenance={},
        )
