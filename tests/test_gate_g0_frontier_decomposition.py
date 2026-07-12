from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.audit_gate_f_nested_grid_sensitivity import (
    _key_calibration,
    _key_image,
    _key_target,
    _unique_index,
    analyze_job_nested_grids,
)
from tools.audit_gate_g0_frontier_decomposition import (
    _index_unique,
    _pair_key,
    _selection_key,
    _target_key,
    analyze_job_frontier,
    summarize_frontier_decomposition,
    write_bundle,
)
from utils.component_frontier_decomposition import (
    CATEGORY_COMPONENT_CONVERSION,
    CATEGORY_PEAK_ORDER,
)
from utils.cross_fitted_low_fa import BUDGETS, cross_fit_job, image_fold
from utils.target_identity import build_stable_target_set


def _synthetic_job():
    by_fold = {0: [], 1: []}
    candidate = 0
    while min(len(values) for values in by_fold.values()) < 2:
        name = f"g0_{candidate:03d}"
        fold = image_fold(name)
        if len(by_fold[fold]) < 2:
            by_fold[fold].append(name)
        candidate += 1
    names = tuple(by_fold[0] + by_fold[1])
    logits = []
    targets = []
    registry = {}
    for index, name in enumerate(names):
        target = np.zeros((8, 8), dtype=bool)
        target[1 + index, 1 + index] = True
        scores = np.full((8, 8), -5.0, dtype=np.float64)
        scores[target] = 6.0 + index
        scores[7, 7 - index] = 1.0 + index / 10.0
        logits.append(scores)
        targets.append(target)
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
    selections, images, nested_targets, pairs, _ = analyze_job_nested_grids(
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
    return (
        tuple(logits),
        tuple(targets),
        names,
        registry,
        checkpoint,
        selections,
        nested_targets,
        pairs,
    )


def test_job_frontier_replays_selected_targets_and_builds_joint_rows() -> None:
    (
        logits,
        targets,
        names,
        registry,
        checkpoint,
        selections,
        nested_targets,
        pairs,
    ) = _synthetic_job()
    curves, target_rows, pair_rows = analyze_job_frontier(
        logits,
        targets,
        names,
        dataset="D",
        seed=7,
        registry=registry,
        checkpoint=checkpoint,
        selection_index=_index_unique(
            selections, _selection_key, label="selection"
        ),
        selected_target_index=_index_unique(
            nested_targets, _target_key, label="target"
        ),
        selected_pair_index=_index_unique(pairs, _pair_key, label="pair"),
    )

    assert curves
    assert len(target_rows) == len(names) * 2 * len(BUDGETS)
    assert len(pair_rows) == 2 * len(BUDGETS)
    assert {row["grid_level"] for row in target_rows} == {"Q1", "Q2"}
    assert all("analysis_semantics" in row for row in target_rows)
    assert all(row["joint_feasible_pair_count"] >= 0 for row in pair_rows)


def _summary_fixture():
    target_rows = []
    pair_rows = []
    for dataset in ("A", "B", "C"):
        for seed in (1, 2, 3):
            for level in ("Q1", "Q2"):
                for budget in BUDGETS:
                    pair_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "grid_level": level,
                            "nominal_budget_fa_per_mpix": budget,
                            "no_feasible_finite_pair": False,
                            "oracle_comparator_status": (
                                "comparable_same_pair_and_held_out_budget_feasible"
                            ),
                            "oracle_delta_joint_matches_vs_selected": (
                                4 if dataset in {"A", "B"} else 0
                            ),
                            "oracle_delta_joint_pd_vs_selected": (
                                0.1 if dataset in {"A", "B"} else 0.0
                            ),
                        }
                    )
                    for target_index in range(12):
                        category = (
                            CATEGORY_COMPONENT_CONVERSION
                            if dataset in {"A", "B"}
                            else CATEGORY_PEAK_ORDER
                        )
                        target_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "grid_level": level,
                                "nominal_budget_fa_per_mpix": budget,
                                "stable_target_id": f"{dataset}:t{target_index}",
                                "image_name": f"{dataset}:i{target_index % 8}",
                                "category_support": category,
                                "category_core": category,
                            }
                        )
    return target_rows, pair_rows


def test_summary_enforces_persistent_breadth_and_adjacent_budget_gate() -> None:
    target_rows, pair_rows = _summary_fixture()
    summary = summarize_frontier_decomposition(target_rows, pair_rows)

    gate = summary["component_conversion_direction_gate"]
    assert gate["pass"]
    assert gate["passing_budgets"] == list(BUDGETS)
    assert summary["joint_fold_pair_oracle_comparable_gain_gate"]["10"][
        "pass"
    ]
    record = gate["by_budget"]["10"]["by_dataset"]["A"]
    assert record["primary_pass"]
    assert record["q2_core_persistent"][
        "persistent_conversion_target_count"
    ] == 12
    assert record["q2_core_persistent"]["persistent_conversion_image_count"] == 8


def test_oracle_gap_requires_same_budget_comparator() -> None:
    target_rows, pair_rows = _summary_fixture()
    invalid_rows = []
    for row in pair_rows:
        copied = dict(row)
        if copied["dataset"] in {"A", "B"}:
            copied["oracle_comparator_status"] = "not_same_budget_comparable"
        invalid_rows.append(copied)
    summary = summarize_frontier_decomposition(target_rows, invalid_rows)

    assert summary["joint_fold_pair_oracle_raw_gap_diagnostic"]["10"][
        "crossdataset_pattern_count"
    ] == 2
    assert not summary["joint_fold_pair_oracle_raw_gap_diagnostic"]["10"][
        "decision_valid"
    ]
    assert not summary["joint_fold_pair_oracle_comparable_gain_gate"]["10"][
        "pass"
    ]


def test_gate_g0_bundle_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    target_rows, pair_rows = _summary_fixture()
    summary = summarize_frontier_decomposition(target_rows, pair_rows)
    output = tmp_path / "bundle"
    write_bundle(
        output,
        curve_rows=[],
        target_rows=target_rows,
        pair_rows=pair_rows,
        summary=summary,
        provenance={"schema_version": "test"},
    )
    assert {path.name for path in output.iterdir()} == {
        "fold_curve.jsonl",
        "target_decomposition.jsonl",
        "joint_pair_oracle.jsonl",
        "frontier_decomposition_summary.json",
        "frontier_decomposition_summary.md",
        "provenance.json",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_bundle(
            output,
            curve_rows=[],
            target_rows=[],
            pair_rows=[],
            summary=summary,
            provenance={},
        )
