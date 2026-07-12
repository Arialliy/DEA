#!/usr/bin/env python3
"""Gate F-1a nested-grid sensitivity for frozen MSHNet checkpoints.

This audit refines only calibration-derived quantile probabilities.  Q0 is
the immutable Gate E-1c grid, Q1 and Q2 are deterministic midpoint
refinements, and selected thresholds are applied once to the held-out fold.
The output is an alternative-grid sensitivity bundle; it never re-decides
the formal E-1c gate and never opens the official test split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_gate_f_operating_transport import (  # noqa: E402
    _read_json,
    _read_jsonl,
    _verify_input_bundle,
    sha256_file,
)
from tools.audit_cross_seed_failure_persistence import (  # noqa: E402
    build_authoritative_registries_before_checkpoints,
    git_worktree_provenance,
    load_validated_jobs,
)
from tools.audit_gate_e_low_fa_bridge import (  # noqa: E402
    _source_hashes as gate_e_source_hashes,
    collect_job_predictions,
)
from utils.cross_fitted_low_fa import (  # noqa: E402
    BUDGETS,
    FOLD_COUNT,
    MATCHERS,
    LowFABridgeError,
    _image_counts,
    _match,
    image_fold,
)
from utils.nested_component_grid import (  # noqa: E402
    GRID_LEVELS,
    NestedComponentGridError,
    evaluate_nested_component_grids,
    exact_budget_feasible,
)
from utils.target_identity import StableTargetSet  # noqa: E402


SELECTION_SCHEMA = "dea.gate_f.nested_grid_selection.v1"
IMAGE_SCHEMA = "dea.gate_f.nested_grid_image.v1"
TARGET_SCHEMA = "dea.gate_f.nested_grid_target.v1"
PAIR_SCHEMA = "dea.gate_f.nested_grid_pair.v1"
EVENT_SCALE_SCHEMA = "dea.gate_f.nested_grid_event_scale.v1"
SUMMARY_SCHEMA = "dea.gate_f.nested_grid_summary.v1"
PROVENANCE_SCHEMA = "dea.gate_f.nested_grid_provenance.v1"
COUNT_FIELDS = (
    "total_pixels",
    "target_components",
    "matched_components",
    "prediction_components",
    "unmatched_prediction_components",
    "unmatched_prediction_area",
)
OUTPUT_FILES = (
    "selection_sensitivity.jsonl",
    "image_sensitivity.jsonl",
    "target_sensitivity.jsonl",
    "pair_sensitivity.jsonl",
    "event_scale.jsonl",
    "nested_grid_summary.json",
    "nested_grid_summary.md",
    "provenance.json",
)


class NestedGridAuditError(RuntimeError):
    """Raised when a Gate F-1a input, replay, or output invariant fails."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Gate E-1c Q0 and compare deterministic Q1/Q2 nested "
            "calibration grids without training or opening official test."
        )
    )
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument(
        "--input-dir",
        default="repro_runs/gate_e/persistence_v2/low_fa_bridge",
    )
    parser.add_argument(
        "--output-dir",
        default="repro_runs/gate_f/nested_grid_sensitivity_v1",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise NestedGridAuditError(f"invalid device {value!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise NestedGridAuditError("CUDA was requested but is unavailable")
    return device


def _key_calibration(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    try:
        return (
            str(row["dataset"]),
            int(row["seed"]),
            str(row["matcher"]),
            int(row["evaluation_fold"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NestedGridAuditError("invalid E-1c calibration key") from exc


def _key_image(row: Mapping[str, Any]) -> tuple[str, int, str, str, int]:
    try:
        return (
            str(row["dataset"]),
            int(row["seed"]),
            str(row["image_name"]),
            str(row["matcher"]),
            int(row["nominal_budget_fa_per_mpix"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NestedGridAuditError("invalid E-1c image key") from exc


def _key_target(row: Mapping[str, Any]) -> tuple[str, int, str, str, int]:
    try:
        return (
            str(row["stable_target_id"]),
            int(row["seed"]),
            str(row["image_name"]),
            str(row["matcher"]),
            int(row["nominal_budget_fa_per_mpix"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NestedGridAuditError("invalid E-1c target key") from exc


def _unique_index(
    rows: Sequence[Mapping[str, Any]],
    key_function,
    *,
    label: str,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = key_function(row)
        if key in result:
            raise NestedGridAuditError(f"duplicate {label} key: {key}")
        result[key] = row
    return result


def _sum_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    if not rows:
        raise NestedGridAuditError("cannot aggregate an empty count group")
    result = {field: 0 for field in COUNT_FIELDS}
    for row in rows:
        for field in COUNT_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise NestedGridAuditError(f"invalid count field {field}")
            result[field] += value
    return result


def _metric_payload(counts: Mapping[str, int], *, budget: int) -> dict[str, Any]:
    total_pixels = int(counts["total_pixels"])
    targets = int(counts["target_components"])
    matched = int(counts["matched_components"])
    area = int(counts["unmatched_prediction_area"])
    if total_pixels <= 0 or targets <= 0 or matched > targets:
        raise NestedGridAuditError("invalid metric population")
    feasible = exact_budget_feasible(area, total_pixels, budget)
    return {
        **{field: int(counts[field]) for field in COUNT_FIELDS},
        "pd": float(matched) / float(targets),
        # Preserve the frozen E-1c reporting order exactly.  Feasibility is
        # still decided independently by integer cross multiplication.
        "fa_per_mpix": float(area) / float(total_pixels) * 1_000_000.0,
        "budget_feasible_zero_overshoot": feasible,
        "budget_integer_margin": budget * total_pixels - area * 1_000_000,
    }


def _assert_exact_equal(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise NestedGridAuditError(f"Q0 replay mismatch: {label}")


def _replay_calibration_q0(
    level,
    frozen: Mapping[str, Any],
) -> None:
    _assert_exact_equal(
        [float(value) for value in level.threshold_grid],
        frozen.get("threshold_grid"),
        label="calibration threshold_grid",
    )
    _assert_exact_equal(
        [asdict(point) for point in level.curve],
        frozen.get("curve"),
        label="calibration curve",
    )
    selections = {
        str(selection.budget_fa_per_million_pixels): asdict(
            selection.operating_point
        )
        for selection in level.selections
    }
    _assert_exact_equal(
        selections,
        frozen.get("selections"),
        label="calibration selections",
    )


def _source_to_identity(
    target_set: StableTargetSet,
    target_component_count: int,
) -> dict[int, Any]:
    mapping = {
        identity.source_component_index: identity
        for identity in target_set.targets
    }
    if set(mapping) != set(range(target_component_count)):
        raise NestedGridAuditError("target source-component indices drifted")
    return mapping


def analyze_job_nested_grids(
    logits: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    image_names: Sequence[str],
    *,
    dataset: str,
    seed: int,
    registry: Mapping[str, StableTargetSet],
    checkpoint: Mapping[str, Any],
    frozen_calibration: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    frozen_images: Mapping[tuple[str, int, str, str, int], Mapping[str, Any]],
    frozen_targets: Mapping[tuple[str, int, str, str, int], Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Evaluate one dataset/seed job and fail closed on the Q0 replay."""

    if not (len(logits) == len(targets) == len(image_names) == len(registry)):
        raise NestedGridAuditError("job samples, names, and registry do not align")
    if tuple(registry) != tuple(image_names) or len(set(image_names)) != len(
        image_names
    ):
        raise NestedGridAuditError("job image universe/order differs from authority")
    score_arrays = tuple(np.asarray(value) for value in logits)
    target_arrays = tuple(np.asarray(value, dtype=bool) for value in targets)
    for scores, target in zip(score_arrays, target_arrays):
        if scores.ndim != 2 or scores.shape != target.shape or not np.isfinite(
            scores
        ).all():
            raise NestedGridAuditError("invalid job logit/target sample")
    folds = tuple(image_fold(name) for name in image_names)
    if set(folds) != set(range(FOLD_COUNT)):
        raise NestedGridAuditError("deterministic two-fold split lacks a fold")

    image_unique_events = [
        int(np.unique(scores.astype(np.float32, copy=False)).size)
        for scores in score_arrays
    ]
    event_record = {
        "schema_version": EVENT_SCALE_SCHEMA,
        "dataset": dataset,
        "seed": int(seed),
        "checkpoint": dict(checkpoint),
        "checkpoint_image_forward_pairs": len(score_arrays),
        "total_pixel_scores": int(sum(value.size for value in score_arrays)),
        "sum_image_local_unique_float32_score_groups": int(
            sum(image_unique_events)
        ),
        "minimum_image_local_unique_score_groups": min(image_unique_events),
        "median_image_local_unique_score_groups": float(
            np.median(image_unique_events)
        ),
        "maximum_image_local_unique_score_groups": max(image_unique_events),
    }
    event_record["image_local_unique_fraction"] = (
        event_record["sum_image_local_unique_float32_score_groups"]
        / event_record["total_pixel_scores"]
    )

    selection_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    match_cache: dict[tuple[str, float, int], Any] = {}

    for evaluation_fold in range(FOLD_COUNT):
        evaluation_indices = [
            index for index, fold in enumerate(folds) if fold == evaluation_fold
        ]
        calibration_indices = [
            index for index, fold in enumerate(folds) if fold != evaluation_fold
        ]
        try:
            nested = evaluate_nested_component_grids(
                [score_arrays[index] for index in calibration_indices],
                [target_arrays[index] for index in calibration_indices],
                BUDGETS,
            )
        except (ValueError, NestedComponentGridError) as exc:
            raise NestedGridAuditError(str(exc)) from exc
        for matcher in MATCHERS:
            matcher_result = nested.matcher(matcher)
            frozen_key = (dataset, int(seed), matcher, evaluation_fold)
            if frozen_key not in frozen_calibration:
                raise NestedGridAuditError(
                    f"missing frozen calibration record {frozen_key}"
                )
            frozen_record = frozen_calibration[frozen_key]
            _assert_exact_equal(
                frozen_record.get("evaluation_image_names"),
                [image_names[index] for index in evaluation_indices],
                label="evaluation image names",
            )
            _assert_exact_equal(
                frozen_record.get("calibration_image_names"),
                [image_names[index] for index in calibration_indices],
                label="calibration image names",
            )
            _replay_calibration_q0(
                matcher_result.level("Q0"), frozen_record
            )

            for level in matcher_result.levels:
                if tuple(selection.budget_fa_per_million_pixels for selection in level.selections) != tuple(BUDGETS):
                    raise NestedGridAuditError("nested selection budget order drifted")
                for selection in level.selections:
                    budget = int(selection.budget_fa_per_million_pixels)
                    point = selection.operating_point
                    selection_rows.append(
                        {
                            "schema_version": SELECTION_SCHEMA,
                            "dataset": dataset,
                            "seed": int(seed),
                            "matcher": matcher,
                            "matching_implementation": matcher_result.matching_implementation,
                            "evaluation_fold": evaluation_fold,
                            "calibration_fold": 1 - evaluation_fold,
                            "grid_level": level.level,
                            "quantile_probability_count": len(level.probabilities),
                            "unique_threshold_count": len(level.threshold_grid),
                            "nominal_budget_fa_per_mpix": budget,
                            "selection": asdict(point),
                            "budget_integer_margin": selection.integer_margin,
                            "checkpoint": dict(checkpoint),
                        }
                    )
                    threshold = float(point.threshold)
                    pending_images: list[dict[str, Any]] = []
                    pending_targets: list[dict[str, Any]] = []
                    for image_index in evaluation_indices:
                        image_name = image_names[image_index]
                        cache_key = (matcher, threshold, image_index)
                        if cache_key not in match_cache:
                            match_cache[cache_key] = _match(
                                score_arrays[image_index],
                                target_arrays[image_index],
                                threshold=threshold,
                                matcher=matcher,
                            )
                        component_match = match_cache[cache_key]
                        target_set = registry[image_name]
                        if len(component_match.target_regions) != len(
                            target_set.targets
                        ):
                            raise NestedGridAuditError(
                                "target count disagrees with authority"
                            )
                        counts = _image_counts(
                            component_match, score_arrays[image_index].size
                        )
                        image_row = {
                            "schema_version": IMAGE_SCHEMA,
                            "dataset": dataset,
                            "seed": int(seed),
                            "image_name": image_name,
                            "image_index": image_index,
                            "evaluation_fold": evaluation_fold,
                            "matcher": matcher,
                            "grid_level": level.level,
                            "nominal_budget_fa_per_mpix": budget,
                            "calibration_threshold": threshold,
                            "target_free_image": not bool(target_set.targets),
                            "checkpoint": dict(checkpoint),
                            **counts,
                        }
                        pending_images.append(image_row)
                        matched_targets = {
                            int(target_index)
                            for target_index, _, _ in component_match.matches
                        }
                        identities = _source_to_identity(
                            target_set, len(component_match.target_regions)
                        )
                        for source_index, identity in identities.items():
                            pending_targets.append(
                                {
                                    "schema_version": TARGET_SCHEMA,
                                    "dataset": dataset,
                                    "seed": int(seed),
                                    "image_name": image_name,
                                    "image_index": image_index,
                                    "evaluation_fold": evaluation_fold,
                                    "stable_target_id": identity.stable_key,
                                    "component_mask_sha256": identity.component_mask_sha256,
                                    "label_mask_sha256": identity.label_mask_sha256,
                                    "component_index": identity.component_index,
                                    "source_component_index": identity.source_component_index,
                                    "area": identity.area,
                                    "matcher": matcher,
                                    "grid_level": level.level,
                                    "nominal_budget_fa_per_mpix": budget,
                                    "calibration_threshold": threshold,
                                    "matched": source_index in matched_targets,
                                    "checkpoint": dict(checkpoint),
                                }
                            )
                    fold_counts = _sum_counts(pending_images)
                    fold_payload = _metric_payload(fold_counts, budget=budget)
                    fold_payload["aggregation"] = "held-out fold integer counts"
                    for row in pending_images:
                        row["held_out_fold_aggregate"] = fold_payload
                    for row in pending_targets:
                        row["held_out_fold_aggregate"] = fold_payload
                    image_rows.extend(pending_images)
                    target_rows.extend(pending_targets)

    pair_rows: list[dict[str, Any]] = []
    for matcher in MATCHERS:
        for level in GRID_LEVELS:
            for budget in BUDGETS:
                selected_images = [
                    row
                    for row in image_rows
                    if row["matcher"] == matcher
                    and row["grid_level"] == level
                    and row["nominal_budget_fa_per_mpix"] == budget
                ]
                selected_targets = [
                    row
                    for row in target_rows
                    if row["matcher"] == matcher
                    and row["grid_level"] == level
                    and row["nominal_budget_fa_per_mpix"] == budget
                ]
                selected_calibration = [
                    row
                    for row in selection_rows
                    if row["matcher"] == matcher
                    and row["grid_level"] == level
                    and row["nominal_budget_fa_per_mpix"] == budget
                ]
                if len(selected_calibration) != FOLD_COUNT:
                    raise NestedGridAuditError("pair lacks two calibration folds")
                heldout = _metric_payload(
                    _sum_counts(selected_images), budget=budget
                )
                calibration_counts = {
                    field: sum(
                        int(row["selection"][field])
                        for row in selected_calibration
                    )
                    for field in COUNT_FIELDS
                }
                calibration = _metric_payload(
                    calibration_counts, budget=budget
                )
                pooled = {
                    **heldout,
                    "aggregation": (
                        "integer counts pooled across both held-out folds"
                    ),
                }
                for row in (*selected_images, *selected_targets):
                    row["dataset_seed_aggregate"] = pooled
                pair_rows.append(
                    {
                        "schema_version": PAIR_SCHEMA,
                        "dataset": dataset,
                        "seed": int(seed),
                        "matcher": matcher,
                        "grid_level": level,
                        "nominal_budget_fa_per_mpix": budget,
                        "calibration_pooled": calibration,
                        "held_out_pooled": heldout,
                        "selected_thresholds_by_evaluation_fold": {
                            str(row["evaluation_fold"]): float(
                                row["selection"]["threshold"]
                            )
                            for row in selected_calibration
                        },
                        "checkpoint": dict(checkpoint),
                    }
                )

    _replay_q0_heldout(
        image_rows,
        target_rows,
        frozen_images=frozen_images,
        frozen_targets=frozen_targets,
        dataset=dataset,
        seed=seed,
    )
    return selection_rows, image_rows, target_rows, pair_rows, event_record


def _replay_q0_heldout(
    image_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    *,
    frozen_images: Mapping[tuple[str, int, str, str, int], Mapping[str, Any]],
    frozen_targets: Mapping[tuple[str, int, str, str, int], Mapping[str, Any]],
    dataset: str,
    seed: int,
) -> None:
    def replay_aggregate(
        observed: Mapping[str, Any],
        frozen: Mapping[str, Any],
        *,
        label: str,
    ) -> None:
        for count_field in COUNT_FIELDS:
            _assert_exact_equal(
                observed.get(count_field),
                frozen.get(count_field),
                label=f"{label}.{count_field}",
            )
        _assert_exact_equal(
            observed.get("pd"),
            frozen.get("achieved_pd"),
            label=f"{label}.pd",
        )
        _assert_exact_equal(
            observed.get("fa_per_mpix"),
            frozen.get("achieved_fa_per_mpix"),
            label=f"{label}.fa",
        )
        _assert_exact_equal(
            observed.get("budget_feasible_zero_overshoot"),
            frozen.get("budget_feasible_zero_overshoot"),
            label=f"{label}.feasible",
        )

    q0_images = [row for row in image_rows if row["grid_level"] == "Q0"]
    q0_targets = [row for row in target_rows if row["grid_level"] == "Q0"]
    for row in q0_images:
        key = _key_image(row)
        if key not in frozen_images:
            raise NestedGridAuditError(f"missing frozen Q0 image {key}")
        frozen = frozen_images[key]
        for field in (
            "image_index",
            "evaluation_fold",
            "calibration_threshold",
            "target_free_image",
            *COUNT_FIELDS,
        ):
            _assert_exact_equal(row.get(field), frozen.get(field), label=f"image.{field}")
        replay_aggregate(
            row["held_out_fold_aggregate"],
            frozen["held_out_fold_aggregate"],
            label="image.held_out_fold_aggregate",
        )
        replay_aggregate(
            row["dataset_seed_aggregate"],
            frozen["dataset_seed_aggregate"],
            label="image.dataset_seed_aggregate",
        )
    for row in q0_targets:
        key = _key_target(row)
        if key not in frozen_targets:
            raise NestedGridAuditError(f"missing frozen Q0 target {key}")
        frozen = frozen_targets[key]
        for field in (
            "image_index",
            "evaluation_fold",
            "component_mask_sha256",
            "label_mask_sha256",
            "component_index",
            "source_component_index",
            "area",
            "calibration_threshold",
        ):
            _assert_exact_equal(row.get(field), frozen.get(field), label=f"target.{field}")
        replay_aggregate(
            row["held_out_fold_aggregate"],
            frozen["held_out_fold_aggregate"],
            label="target.held_out_fold_aggregate",
        )
        replay_aggregate(
            row["dataset_seed_aggregate"],
            frozen["dataset_seed_aggregate"],
            label="target.dataset_seed_aggregate",
        )
        _assert_exact_equal(
            row.get("matched"), frozen.get("low_fa_matched"), label="target.matched"
        )
    expected_image_count = sum(
        key[0] == dataset and key[1] == int(seed) for key in frozen_images
    )
    expected_target_count = sum(
        str(row.get("dataset")) == dataset and int(row.get("seed")) == int(seed)
        for row in frozen_targets.values()
    )
    if len(q0_images) != expected_image_count or len(q0_targets) != expected_target_count:
        raise NestedGridAuditError("Q0 held-out replay cardinality drifted")


def _pair_index(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str, int, str], Mapping[str, Any]]:
    result = {}
    for row in pair_rows:
        key = (
            str(row["dataset"]),
            int(row["seed"]),
            str(row["matcher"]),
            int(row["nominal_budget_fa_per_mpix"]),
            str(row["grid_level"]),
        )
        if key in result:
            raise NestedGridAuditError(f"duplicate pair row {key}")
        result[key] = row
    return result


def summarize_nested_grid_sensitivity(
    pair_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the pre-registered Gate F-1a decision from pooled integer rows."""

    index = _pair_index(pair_rows)
    datasets = sorted({str(row["dataset"]) for row in pair_rows})
    seeds = sorted({int(row["seed"]) for row in pair_rows})
    if len(datasets) != 3 or len(seeds) != 3:
        raise NestedGridAuditError("formal sensitivity requires 3 datasets x 3 seeds")
    comparisons: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_seeds = sorted(
            {int(row["seed"]) for row in pair_rows if row["dataset"] == dataset}
        )
        if len(dataset_seeds) != 3:
            raise NestedGridAuditError("each dataset requires exactly three seeds")
        for seed in dataset_seeds:
            for matcher in MATCHERS:
                for budget in BUDGETS:
                    levels = {
                        level: index[(dataset, seed, matcher, budget, level)]
                        for level in GRID_LEVELS
                    }
                    q0 = levels["Q0"]
                    q1 = levels["Q1"]
                    q2 = levels["Q2"]
                    calibration_delta_matches = int(
                        q2["calibration_pooled"]["matched_components"]
                    ) - int(q0["calibration_pooled"]["matched_components"])
                    calibration_delta_pd = float(
                        q2["calibration_pooled"]["pd"]
                    ) - float(q0["calibration_pooled"]["pd"])
                    q0_feasible = bool(
                        q0["held_out_pooled"]["budget_feasible_zero_overshoot"]
                    )
                    q2_feasible = bool(
                        q2["held_out_pooled"]["budget_feasible_zero_overshoot"]
                    )
                    flip_direction = None
                    if q0_feasible != q2_feasible:
                        flip_direction = (
                            "fail_to_pass" if q2_feasible else "pass_to_fail"
                        )
                    target_groups = defaultdict(dict)
                    for row in target_rows:
                        if (
                            row["dataset"] == dataset
                            and int(row["seed"]) == seed
                            and row["matcher"] == matcher
                            and int(row["nominal_budget_fa_per_mpix"]) == budget
                            and row["grid_level"] in {"Q0", "Q2"}
                        ):
                            target_groups[str(row["stable_target_id"])][
                                str(row["grid_level"])
                            ] = bool(row["matched"])
                    if any(set(value) != {"Q0", "Q2"} for value in target_groups.values()):
                        raise NestedGridAuditError("target Q0/Q2 pairing drifted")
                    gained = sum(
                        not values["Q0"] and values["Q2"]
                        for values in target_groups.values()
                    )
                    lost = sum(
                        values["Q0"] and not values["Q2"]
                        for values in target_groups.values()
                    )
                    comparisons.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "matcher": matcher,
                            "nominal_budget_fa_per_mpix": budget,
                            "q0_to_q2_calibration_delta_matches": calibration_delta_matches,
                            "q0_to_q2_calibration_delta_pd": calibration_delta_pd,
                            "q0_to_q2_heldout_delta_matches": int(
                                q2["held_out_pooled"]["matched_components"]
                            )
                            - int(q0["held_out_pooled"]["matched_components"]),
                            "q0_to_q2_heldout_delta_pd": float(
                                q2["held_out_pooled"]["pd"]
                            )
                            - float(q0["held_out_pooled"]["pd"]),
                            "q0_to_q2_heldout_delta_unmatched_area": int(
                                q2["held_out_pooled"]["unmatched_prediction_area"]
                            )
                            - int(q0["held_out_pooled"]["unmatched_prediction_area"]),
                            "q0_heldout_feasible": q0_feasible,
                            "q2_heldout_feasible": q2_feasible,
                            "feasibility_flip_direction": flip_direction,
                            "q2_non_all_off": int(
                                q2["held_out_pooled"]["matched_components"]
                            )
                            > 0,
                            "q1_to_q2_same_selected_threshold_pair": (
                                q1["selected_thresholds_by_evaluation_fold"]
                                == q2["selected_thresholds_by_evaluation_fold"]
                            ),
                            "heldout_targets_gained_q0_to_q2": gained,
                            "heldout_targets_lost_q0_to_q2": lost,
                        }
                    )

    by_budget: dict[str, Any] = {}
    passing_budgets = []
    for budget in BUDGETS:
        budget_records = [
            row
            for row in comparisons
            if row["nominal_budget_fa_per_mpix"] == budget
        ]
        joint_seed_records = []
        for dataset in datasets:
            dataset_seeds = sorted(
                {row["seed"] for row in budget_records if row["dataset"] == dataset}
            )
            for seed in dataset_seeds:
                records = [
                    row
                    for row in budget_records
                    if row["dataset"] == dataset and row["seed"] == seed
                ]
                if {row["matcher"] for row in records} != set(MATCHERS):
                    raise NestedGridAuditError("joint seed lacks both matchers")
                directions = {row["feasibility_flip_direction"] for row in records}
                joint_direction = (
                    next(iter(directions))
                    if len(directions) == 1 and None not in directions
                    else None
                )
                joint_seed_records.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "joint_feasibility_flip_direction": joint_direction,
                        "both_matchers_q2_non_all_off": all(
                            row["q2_non_all_off"] for row in records
                        ),
                        "both_matchers_material_calibration_gain": all(
                            row["q0_to_q2_calibration_delta_matches"] >= 3
                            and row["q0_to_q2_calibration_delta_pd"] >= 0.05
                            for row in records
                        ),
                    }
                )
        flips_by_direction = {}
        flip_gate = False
        for direction in ("fail_to_pass", "pass_to_fail"):
            qualifying = [
                row
                for row in joint_seed_records
                if row["joint_feasibility_flip_direction"] == direction
                and row["both_matchers_q2_non_all_off"]
            ]
            dataset_count = len({row["dataset"] for row in qualifying})
            passed = len(qualifying) >= 2 and dataset_count >= 2
            flip_gate = flip_gate or passed
            flips_by_direction[direction] = {
                "dataset_seed_count": len(qualifying),
                "dataset_count": dataset_count,
                "dataset_seeds": [
                    f"{row['dataset']}:{row['seed']}" for row in qualifying
                ],
                "pass": passed,
            }
        material_gain_by_dataset = {}
        for dataset in datasets:
            qualifying_seeds = [
                row["seed"]
                for row in joint_seed_records
                if row["dataset"] == dataset
                and row["both_matchers_material_calibration_gain"]
            ]
            material_gain_by_dataset[dataset] = {
                "qualifying_seed_count": len(qualifying_seeds),
                "qualifying_seeds": qualifying_seeds,
                "pass": len(qualifying_seeds) >= 2,
            }
        gain_gate = sum(
            record["pass"] for record in material_gain_by_dataset.values()
        ) >= 2
        passed = flip_gate or gain_gate
        if passed:
            passing_budgets.append(budget)
        by_budget[str(budget)] = {
            "feasibility_flip_trigger": {
                "by_direction": flips_by_direction,
                "pass": flip_gate,
            },
            "material_calibration_gain_trigger": {
                "by_dataset": material_gain_by_dataset,
                "pass": gain_gate,
            },
            "q1_to_q2_same_threshold_pair_count": sum(
                row["q1_to_q2_same_selected_threshold_pair"]
                for row in budget_records
            ),
            "matcher_record_count": len(budget_records),
            "pass": passed,
        }

    total_pixels = sum(int(row["total_pixel_scores"]) for row in event_rows)
    total_events = sum(
        int(row["sum_image_local_unique_float32_score_groups"])
        for row in event_rows
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "analysis_scope": (
            "fixed-epoch development holdout alternative-grid sensitivity; "
            "official test sealed and formal E-1c decision unchanged"
        ),
        "protocol": {
            "grid_levels": list(GRID_LEVELS),
            "budgets_fa_per_mpix": list(BUDGETS),
            "matchers": list(MATCHERS),
            "strict_threshold_operator": ">",
            "q0_replay_required": True,
            "candidate_construction": "calibration logits only",
            "integer_budget_test": (
                "unmatched_area*1000000 <= budget*total_pixels"
            ),
        },
        "event_scale": {
            "checkpoint_image_forward_pairs": sum(
                int(row["checkpoint_image_forward_pairs"])
                for row in event_rows
            ),
            "total_pixel_scores": total_pixels,
            "sum_image_local_unique_float32_score_groups": total_events,
            "image_local_unique_fraction": total_events / total_pixels,
            "naive_two_matcher_event_evaluations": 2 * total_events,
            "unit_note": (
                "sum of image-local unique groups, not a global union of thresholds"
            ),
        },
        "comparisons": comparisons,
        "targeted_exact_interval_gate": {
            "by_budget": by_budget,
            "passing_budgets": passing_budgets,
            "pass": bool(passing_budgets),
            "authorization_if_pass": (
                "targeted unique-logit sweep only; no method, training, "
                "formal-gate revision, or risk-control theorem"
            ),
        },
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    gate = summary["targeted_exact_interval_gate"]
    event = summary["event_scale"]
    lines = [
        "# Gate F−1a nested-grid sensitivity",
        "",
        str(summary["analysis_scope"]),
        "",
        f"- Q0 replay: required and passed",
        f"- Targeted exact interval authorized: {gate['pass']}",
        f"- Passing budgets: {gate['passing_budgets']}",
        (
            "- Image-local unique float32 score groups: "
            f"{event['sum_image_local_unique_float32_score_groups']} / "
            f"{event['total_pixel_scores']} "
            f"({event['image_local_unique_fraction']:.6%})"
        ),
        "",
        "| FA/Mpix | flip trigger | calibration-gain trigger | Q1=Q2 threshold pairs | pass |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget in summary["protocol"]["budgets_fa_per_mpix"]:
        row = gate["by_budget"][str(budget)]
        lines.append(
            "| %s | %s | %s | %s/%s | %s |"
            % (
                budget,
                row["feasibility_flip_trigger"]["pass"],
                row["material_calibration_gain_trigger"]["pass"],
                row["q1_to_q2_same_threshold_pair_count"],
                row["matcher_record_count"],
                row["pass"],
            )
        )
    lines.extend(
        [
            "",
            "A flip is an alternative-grid sensitivity, not a revision of Gate E−1c.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def write_bundle(
    output_dir: Path,
    *,
    selection_rows: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        row_groups = (
            selection_rows,
            image_rows,
            target_rows,
            pair_rows,
            event_rows,
        )
        artifact_paths = []
        for name, rows in zip(OUTPUT_FILES[:5], row_groups):
            path = temporary / name
            _write_jsonl(path, rows)
            artifact_paths.append(path)
        summary_path = temporary / OUTPUT_FILES[5]
        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        markdown_path = temporary / OUTPUT_FILES[6]
        markdown_path.write_text(build_markdown(summary), encoding="utf-8")
        artifact_paths.extend((summary_path, markdown_path))
        complete_provenance = {
            **dict(provenance),
            "artifact_sha256": {
                path.name: sha256_file(path) for path in artifact_paths
            },
        }
        (temporary / OUTPUT_FILES[7]).write_text(
            json.dumps(
                complete_provenance,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if {path.name for path in temporary.iterdir()} != set(OUTPUT_FILES):
            raise NestedGridAuditError("temporary bundle inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_hashes() -> dict[str, str]:
    paths = {
        "tool": Path(__file__).resolve(),
        "nested_component_grid": ROOT / "utils" / "nested_component_grid.py",
        "component_operating_point": ROOT / "utils" / "component_operating_point.py",
        "cross_fitted_low_fa": ROOT / "utils" / "cross_fitted_low_fa.py",
        "metric": ROOT / "utils" / "metric.py",
        "target_identity": ROOT / "utils" / "target_identity.py",
        "gate_e_tool": ROOT / "tools" / "audit_gate_e_low_fa_bridge.py",
        "dataset": ROOT / "utils" / "data.py",
        "mshnet": ROOT / "model" / "MSHNet.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.batch_size != 8:
        raise NestedGridAuditError(
            "Q0 byte replay requires the frozen E-1c batch size 8"
        )
    if args.num_workers < 0:
        raise NestedGridAuditError("num_workers must be non-negative")
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise NestedGridAuditError("output cannot be inside immutable E-1c input")
    input_provenance, formal_summary = _verify_input_bundle(input_dir)
    if formal_summary.get("joint_gate", {}).get("pass") is not False:
        raise NestedGridAuditError("Gate F-1a requires the frozen failed E-1c bundle")
    if gate_e_source_hashes() != input_provenance.get("source_sha256"):
        raise NestedGridAuditError("current Gate E inference sources drifted")
    source_hashes = _source_hashes()
    input_provenance_sha256 = sha256_file(input_dir / "provenance.json")
    calibration_values = _read_json(input_dir / "calibration.json")
    if not isinstance(calibration_values, list):
        raise NestedGridAuditError("calibration.json must contain a list")
    frozen_calibration = _unique_index(
        calibration_values,
        _key_calibration,
        label="calibration",
    )
    frozen_images = _unique_index(
        _read_jsonl(input_dir / "image_low_fa.jsonl"),
        _key_image,
        label="image",
    )
    frozen_targets = _unique_index(
        _read_jsonl(input_dir / "target_low_fa.jsonl"),
        _key_target,
        label="target",
    )

    batch_dir = ROOT / "repro_runs" / "clean" / args.batch_id
    authoritative, authority_records, registry_order = (
        build_authoritative_registries_before_checkpoints(
            batch_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )
    jobs, batch_provenance = load_validated_jobs(
        batch_dir, policy="fixed_epoch"
    )
    device = _resolve_device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    selection_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    checkpoint_records = []
    inference_records = []
    for job in jobs:
        dataset = str(job["dataset"])
        logits, targets, names, record = collect_job_predictions(
            job,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            expected_registry=authoritative[dataset],
        )
        values = analyze_job_nested_grids(
            logits,
            targets,
            names,
            dataset=dataset,
            seed=int(job["seed"]),
            registry=authoritative[dataset],
            checkpoint=record["checkpoint"],
            frozen_calibration=frozen_calibration,
            frozen_images=frozen_images,
            frozen_targets=frozen_targets,
        )
        job_selections, job_images, job_targets, job_pairs, job_events = values
        selection_rows.extend(job_selections)
        image_rows.extend(job_images)
        target_rows.extend(job_targets)
        pair_rows.extend(job_pairs)
        event_rows.append(job_events)
        checkpoint_records.append(record["checkpoint"])
        inference_records.append(record["inference"])

    expected_images = sum(len(value) for value in authoritative.values()) * 3
    expected_targets = (
        sum(
            len(target_set.targets)
            for registry in authoritative.values()
            for target_set in registry.values()
        )
        * 3
    )
    if len(selection_rows) != 9 * FOLD_COUNT * len(MATCHERS) * len(GRID_LEVELS) * len(BUDGETS):
        raise NestedGridAuditError("selection ledger cardinality drifted")
    if len(image_rows) != expected_images * len(MATCHERS) * len(GRID_LEVELS) * len(BUDGETS):
        raise NestedGridAuditError("image ledger cardinality drifted")
    if len(target_rows) != expected_targets * len(MATCHERS) * len(GRID_LEVELS) * len(BUDGETS):
        raise NestedGridAuditError("target ledger cardinality drifted")
    if len(pair_rows) != 9 * len(MATCHERS) * len(GRID_LEVELS) * len(BUDGETS):
        raise NestedGridAuditError("pair ledger cardinality drifted")
    if len(event_rows) != 9:
        raise NestedGridAuditError("event-scale ledger cardinality drifted")

    _assert_exact_equal(
        checkpoint_records,
        input_provenance.get("jobs"),
        label="checkpoint records",
    )
    _assert_exact_equal(
        inference_records,
        input_provenance.get("inference"),
        label="inference records",
    )
    _assert_exact_equal(
        batch_provenance,
        input_provenance.get("batch"),
        label="batch provenance",
    )
    _assert_exact_equal(
        authority_records,
        input_provenance.get("authoritative_registry_construction"),
        label="authority records",
    )
    _assert_exact_equal(
        registry_order,
        input_provenance.get("registry_precheckpoint_order"),
        label="registry order",
    )

    summary = summarize_nested_grid_sensitivity(
        pair_rows, target_rows, event_rows
    )
    rechecked_provenance, _ = _verify_input_bundle(input_dir)
    if sha256_file(input_dir / "provenance.json") != input_provenance_sha256:
        raise NestedGridAuditError("E-1c provenance changed during execution")
    if rechecked_provenance != input_provenance:
        raise NestedGridAuditError("E-1c input changed during execution")
    if _source_hashes() != source_hashes:
        raise NestedGridAuditError("audit sources changed during execution")
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "git": git_worktree_provenance(),
        "source_sha256": source_hashes,
        "formal_e1c_input": {
            "directory": str(input_dir),
            "provenance_sha256": input_provenance_sha256,
            "artifact_sha256": dict(input_provenance["artifact_sha256"]),
            "formal_joint_gate_pass": False,
            "formal_decision_changed": False,
        },
        "batch": batch_provenance,
        "registry_precheckpoint_order": registry_order,
        "authoritative_registry_construction": authority_records,
        "jobs": checkpoint_records,
        "inference": inference_records,
        "protocol": {
            **summary["protocol"],
            "fold_count": FOLD_COUNT,
            "fold_mappings": input_provenance["protocol"]["fold_mappings"],
            "fold_mappings_sha256": input_provenance["protocol"][
                "fold_mappings_sha256"
            ],
            "official_test_policy": "sealed and never opened",
            "sensitivity_name": "alternative-grid sensitivity",
        },
        "event_scale": summary["event_scale"],
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit_image": importlib.metadata.version("scikit-image"),
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    }
    write_bundle(
        output_dir,
        selection_rows=selection_rows,
        image_rows=image_rows,
        target_rows=target_rows,
        pair_rows=pair_rows,
        event_rows=event_rows,
        summary=summary,
        provenance=provenance,
    )
    return summary, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary, output_dir = run(parse_args(argv))
    except (
        FileExistsError,
        LowFABridgeError,
        NestedComponentGridError,
        NestedGridAuditError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "targeted_exact_interval_gate": summary[
                    "targeted_exact_interval_gate"
                ],
                "event_scale": summary["event_scale"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
