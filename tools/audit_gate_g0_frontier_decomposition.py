#!/usr/bin/env python3
"""Gate G0 post-hoc direction audit on finite Q1/Q2 component frontiers.

The audit uses calibration-derived finite threshold grids and evaluation-fold
labels to separate Q2-selected misses into neutral operational phenotypes.
It is not a deployable threshold estimate, a causal explanation, or a method
authorization.  Official test data remain sealed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import datetime as dt
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_gate_f_operating_transport import (  # noqa: E402
    _read_json,
    _read_jsonl,
    sha256_file,
)
from tools.audit_cross_seed_failure_persistence import (  # noqa: E402
    build_authoritative_registries_before_checkpoints,
    git_worktree_provenance,
    load_validated_jobs,
)
from tools.audit_gate_e_low_fa_bridge import (  # noqa: E402
    collect_job_predictions,
)
from tools.audit_gate_f_nested_grid_sensitivity import (  # noqa: E402
    _resolve_device,
)
from utils.component_frontier_decomposition import (  # noqa: E402
    CATEGORY_COMPONENT_CONVERSION,
    CATEGORY_MATCHER_SENSITIVE,
    CATEGORY_PEAK_ORDER,
    CATEGORY_SELECTED_HIT,
    CATEGORY_SELECTION_SENSITIVE,
    FoldCandidateState,
    JointGlobalOraclePair,
    classify_joint_matcher_targets,
    joint_feasible_pairs,
    select_joint_global_oracle_pair,
)
from utils.component_operating_point import (  # noqa: E402
    build_logit_threshold_grid,
)
from utils.cross_fitted_low_fa import (  # noqa: E402
    BUDGETS,
    FOLD_COUNT,
    MATCHERS,
    _image_counts,
    _match,
    image_fold,
)
from utils.nested_component_grid import (  # noqa: E402
    build_nested_quantile_probability_grids,
)
from utils.target_identity import StableTargetSet  # noqa: E402


INPUT_PROVENANCE_SCHEMA = "dea.gate_f.nested_grid_provenance.v1"
INPUT_SUMMARY_SCHEMA = "dea.gate_f.nested_grid_summary.v1"
CURVE_SCHEMA = "dea.gate_g0.frontier_fold_curve.v2"
TARGET_SCHEMA = "dea.gate_g0.frontier_target.v2"
PAIR_SCHEMA = "dea.gate_g0.frontier_joint_pair.v2"
SUMMARY_SCHEMA = "dea.gate_g0.frontier_summary.v2"
PROVENANCE_SCHEMA = "dea.gate_g0.frontier_provenance.v2"
GRID_LEVELS = ("Q1", "Q2")
COUNT_FIELDS = (
    "total_pixels",
    "target_components",
    "matched_components",
    "prediction_components",
    "unmatched_prediction_components",
    "unmatched_prediction_area",
)
OUTPUT_FILES = (
    "fold_curve.jsonl",
    "target_decomposition.jsonl",
    "joint_pair_oracle.jsonl",
    "frontier_decomposition_summary.json",
    "frontier_decomposition_summary.md",
    "provenance.json",
)


class GateG0AuditError(RuntimeError):
    """Raised when a Gate G0 input, replay, or direction gate drifts."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose finite-Q-grid selected misses under matcher-joint "
            "development-only post-hoc threshold-pair oracles."
        )
    )
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument(
        "--input-dir",
        default="repro_runs/gate_f/nested_grid_sensitivity_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="repro_runs/gate_g/frontier_decomposition_v2",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=1,
        help="CPU job-level workers; inference remains sequential and frozen",
    )
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _verify_input_bundle(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = _read_json(input_dir / "provenance.json")
    summary = _read_json(input_dir / "nested_grid_summary.json")
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != INPUT_PROVENANCE_SCHEMA
        or not isinstance(summary, dict)
        or summary.get("schema_version") != INPUT_SUMMARY_SCHEMA
    ):
        raise GateG0AuditError("input is not the formal Gate F-1a bundle")
    hashes = provenance.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise GateG0AuditError("input provenance lacks artifact hashes")
    for name, expected in hashes.items():
        path = input_dir / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise GateG0AuditError(f"input artifact hash mismatch: {name}")
    gate = summary.get("targeted_exact_interval_gate")
    if not isinstance(gate, Mapping) or gate.get("pass") is not False:
        raise GateG0AuditError("Gate G0 requires the frozen failed Gate F-1a")
    formal_input = provenance.get("formal_e1c_input")
    if (
        not isinstance(formal_input, Mapping)
        or formal_input.get("formal_decision_changed") is not False
    ):
        raise GateG0AuditError("Gate F-1a did not preserve the formal decision")
    return provenance, summary


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    key_function,
    *,
    label: str,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = key_function(row)
        if key in result:
            raise GateG0AuditError(f"duplicate {label} key: {key}")
        result[key] = row
    return result


def _selection_key(row: Mapping[str, Any]) -> tuple[str, int, str, str, int, int]:
    return (
        str(row["dataset"]),
        int(row["seed"]),
        str(row["matcher"]),
        str(row["grid_level"]),
        int(row["nominal_budget_fa_per_mpix"]),
        int(row["evaluation_fold"]),
    )


def _target_key(row: Mapping[str, Any]) -> tuple[str, int, str, str, int]:
    return (
        str(row["stable_target_id"]),
        int(row["seed"]),
        str(row["matcher"]),
        str(row["grid_level"]),
        int(row["nominal_budget_fa_per_mpix"]),
    )


def _pair_key(row: Mapping[str, Any]) -> tuple[str, int, str, str, int]:
    return (
        str(row["dataset"]),
        int(row["seed"]),
        str(row["matcher"]),
        str(row["grid_level"]),
        int(row["nominal_budget_fa_per_mpix"]),
    )


def _target_peak_records(
    scores: np.ndarray,
    target: np.ndarray,
    target_set: StableTargetSet,
    *,
    neighborhood_radius: float = 3.0,
) -> dict[str, dict[str, Any]]:
    target_map = measure.label(target.astype(bool), connectivity=2)
    regions = tuple(measure.regionprops(target_map))
    identities = {
        identity.source_component_index: identity
        for identity in target_set.targets
    }
    if set(identities) != set(range(len(regions))):
        raise GateG0AuditError("target identity/source order drifted")
    result = {}
    for source_index, region in enumerate(regions):
        identity = identities[source_index]
        component = target_map == region.label
        support_distance = distance_transform_edt(~component)
        local_support = support_distance < neighborhood_radius
        core_peak = float(np.max(scores[component]))
        support_peak = float(np.max(scores[local_support]))
        if not np.isfinite(core_peak) or not np.isfinite(support_peak):
            raise GateG0AuditError("target peak is not finite")
        if core_peak > support_peak:
            raise GateG0AuditError("core peak exceeds enclosing support peak")
        result[identity.stable_key] = {
            "stable_target_id": identity.stable_key,
            "image_name": identity.image_name,
            "area": identity.area,
            "component_mask_sha256": identity.component_mask_sha256,
            "label_mask_sha256": identity.label_mask_sha256,
            "core_peak": core_peak,
            "local_support_peak": support_peak,
            "local_support_definition": (
                "pixels with strict Euclidean distance <3 from fixed GT component"
            ),
        }
    return result


def _evaluate_fold_states(
    score_arrays: Sequence[np.ndarray],
    target_arrays: Sequence[np.ndarray],
    image_names: Sequence[str],
    evaluation_indices: Sequence[int],
    thresholds: Sequence[float],
    *,
    matcher: str,
    registry: Mapping[str, StableTargetSet],
    peak_records: Mapping[str, Mapping[str, Any]],
    dataset: str,
    seed: int,
    evaluation_fold: int,
    grid_level: str,
) -> tuple[tuple[FoldCandidateState, ...], list[dict[str, Any]]]:
    fold_target_ids = {
        target.stable_key
        for index in evaluation_indices
        for target in registry[image_names[index]].targets
    }
    if fold_target_ids != {
        stable_id
        for stable_id, record in peak_records.items()
        if record["image_name"] in {image_names[index] for index in evaluation_indices}
    }:
        raise GateG0AuditError("fold peak/target universe drifted")
    total_pixels = sum(score_arrays[index].size for index in evaluation_indices)
    states = []
    rows = []
    for threshold in thresholds:
        threshold_value = float(threshold)
        counts = {field: 0 for field in COUNT_FIELDS}
        matched_ids = set()
        for image_index in evaluation_indices:
            component_match = _match(
                score_arrays[image_index],
                target_arrays[image_index],
                threshold=threshold_value,
                matcher=matcher,
            )
            image_counts = _image_counts(
                component_match, score_arrays[image_index].size
            )
            for field in COUNT_FIELDS:
                counts[field] += int(image_counts[field])
            identities = {
                identity.source_component_index: identity
                for identity in registry[image_names[image_index]].targets
            }
            if set(identities) != set(range(len(component_match.target_regions))):
                raise GateG0AuditError("component target identity drifted")
            matched_ids.update(
                identities[int(target_index)].stable_key
                for target_index, _, _ in component_match.matches
            )
        support_active = frozenset(
            stable_id
            for stable_id in fold_target_ids
            if float(peak_records[stable_id]["local_support_peak"])
            > threshold_value
        )
        core_active = frozenset(
            stable_id
            for stable_id in fold_target_ids
            if float(peak_records[stable_id]["core_peak"]) > threshold_value
        )
        state = FoldCandidateState(
            threshold=threshold_value,
            all_off_sentinel=False,
            total_pixels=counts["total_pixels"],
            target_components=counts["target_components"],
            matched_components=counts["matched_components"],
            prediction_components=counts["prediction_components"],
            unmatched_prediction_components=counts[
                "unmatched_prediction_components"
            ],
            unmatched_prediction_area=counts["unmatched_prediction_area"],
            matched_target_ids=frozenset(matched_ids),
            support_active_target_ids=support_active,
            core_active_target_ids=core_active,
        )
        states.append(state)
        rows.append(
            {
                "schema_version": CURVE_SCHEMA,
                "dataset": dataset,
                "seed": int(seed),
                "matcher": matcher,
                "grid_level": grid_level,
                "evaluation_fold": evaluation_fold,
                "threshold_index": len(states) - 1,
                "threshold": threshold_value,
                **counts,
                "pd": counts["matched_components"] / counts["target_components"],
                "fa_per_mpix": (
                    counts["unmatched_prediction_area"]
                    / counts["total_pixels"]
                    * 1_000_000.0
                ),
                "support_active_target_count": len(support_active),
                "core_active_target_count": len(core_active),
            }
        )
    if any(state.total_pixels != total_pixels for state in states):
        raise GateG0AuditError("fold state pixel denominator drifted")
    return tuple(states), rows


def _project_states(
    q2_states: Sequence[FoldCandidateState],
    q1_thresholds: Sequence[float],
) -> tuple[FoldCandidateState, ...]:
    by_threshold = {state.threshold: state for state in q2_states}
    try:
        return tuple(by_threshold[float(value)] for value in q1_thresholds)
    except KeyError as exc:
        raise GateG0AuditError("Q1 threshold is absent from Q2 states") from exc


def _selected_target_map(
    target_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    target_ids: Sequence[str],
    *,
    seed: int,
    matcher: str,
    grid_level: str,
    budget: int,
) -> dict[str, bool]:
    result = {}
    for stable_id in target_ids:
        key = (stable_id, int(seed), matcher, grid_level, int(budget))
        if key not in target_index:
            raise GateG0AuditError(f"missing selected target row {key}")
        result[stable_id] = bool(target_index[key]["matched"])
    return result


def _oracle_payload(
    oracle: JointGlobalOraclePair | None,
    fold_families: Mapping[int, Mapping[str, Sequence[FoldCandidateState]]],
) -> dict[str, Any]:
    if oracle is None:
        return {
            "exists": False,
            "joint_matched_components": 0,
            "joint_matched_target_ids": [],
            "thresholds_by_fold": None,
            "matcher_unmatched_prediction_area": {},
            "total_pixels": sum(
                next(iter(fold_families[fold].values()))[0].total_pixels
                for fold in range(FOLD_COUNT)
            ),
        }
    state0 = next(iter(fold_families[0].values()))[oracle.fold0_index]
    state1 = next(iter(fold_families[1].values()))[oracle.fold1_index]
    return {
        "exists": True,
        "joint_matched_components": len(oracle.joint_matched_target_ids),
        "joint_matched_target_ids": sorted(oracle.joint_matched_target_ids),
        "thresholds_by_fold": {
            "0": state0.threshold,
            "1": state1.threshold,
        },
        "matcher_unmatched_prediction_area": dict(
            oracle.matcher_unmatched_prediction_area
        ),
        "total_pixels": oracle.total_pixels,
    }


def analyze_job_frontier(
    logits: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    image_names: Sequence[str],
    *,
    dataset: str,
    seed: int,
    registry: Mapping[str, StableTargetSet],
    checkpoint: Mapping[str, Any],
    selection_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    selected_target_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    selected_pair_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate Q2 held-out curves once and derive Q1/Q2 joint phenotypes."""

    if not (len(logits) == len(targets) == len(image_names) == len(registry)):
        raise GateG0AuditError("job samples and registry do not align")
    if tuple(registry) != tuple(image_names):
        raise GateG0AuditError("image order differs from registry")
    score_arrays = tuple(np.asarray(value) for value in logits)
    target_arrays = tuple(np.asarray(value, dtype=bool) for value in targets)
    folds = tuple(image_fold(name) for name in image_names)
    if set(folds) != set(range(FOLD_COUNT)):
        raise GateG0AuditError("job lacks a deterministic fold")
    peak_records = {}
    for scores, target, name in zip(score_arrays, target_arrays, image_names):
        peak_records.update(
            _target_peak_records(scores, target, registry[name])
        )
    if len(peak_records) != sum(
        len(target_set.targets) for target_set in registry.values()
    ):
        raise GateG0AuditError("job target peak cardinality drifted")

    probability_grids = {
        grid.level: grid.probabilities
        for grid in build_nested_quantile_probability_grids()
    }
    q_thresholds: dict[int, dict[str, tuple[float, ...]]] = {}
    fold_indices = {}
    for evaluation_fold in range(FOLD_COUNT):
        evaluation_indices = tuple(
            index for index, fold in enumerate(folds) if fold == evaluation_fold
        )
        calibration_indices = tuple(
            index for index, fold in enumerate(folds) if fold != evaluation_fold
        )
        fold_indices[evaluation_fold] = evaluation_indices
        q_thresholds[evaluation_fold] = {
            level: build_logit_threshold_grid(
                [score_arrays[index] for index in calibration_indices],
                fixed_thresholds=(0.0,),
                tail_quantiles=probability_grids[level],
            )
            for level in GRID_LEVELS
        }
        if not set(q_thresholds[evaluation_fold]["Q1"]).issubset(
            q_thresholds[evaluation_fold]["Q2"]
        ):
            raise GateG0AuditError("Q1 threshold grid is not nested in Q2")

    curve_rows = []
    states: dict[
        int, dict[str, dict[str, tuple[FoldCandidateState, ...]]]
    ] = {fold: {matcher: {} for matcher in MATCHERS} for fold in range(FOLD_COUNT)}
    for evaluation_fold in range(FOLD_COUNT):
        for matcher in MATCHERS:
            q2_states, q2_rows = _evaluate_fold_states(
                score_arrays,
                target_arrays,
                image_names,
                fold_indices[evaluation_fold],
                q_thresholds[evaluation_fold]["Q2"],
                matcher=matcher,
                registry=registry,
                peak_records=peak_records,
                dataset=dataset,
                seed=seed,
                evaluation_fold=evaluation_fold,
                grid_level="Q2",
            )
            states[evaluation_fold][matcher]["Q2"] = q2_states
            q1_states = _project_states(
                q2_states, q_thresholds[evaluation_fold]["Q1"]
            )
            states[evaluation_fold][matcher]["Q1"] = q1_states
            q2_row_index = {row["threshold"]: row for row in q2_rows}
            q1_rows = []
            for index, threshold in enumerate(q_thresholds[evaluation_fold]["Q1"]):
                row = dict(q2_row_index[float(threshold)])
                row["grid_level"] = "Q1"
                row["threshold_index"] = index
                q1_rows.append(row)
            curve_rows.extend(q1_rows)
            curve_rows.extend(q2_rows)

    target_rows = []
    pair_rows = []
    all_target_ids_by_fold = {
        fold: sorted(
            target.stable_key
            for index in fold_indices[fold]
            for target in registry[image_names[index]].targets
        )
        for fold in range(FOLD_COUNT)
    }
    target_metadata = {
        stable_id: record for stable_id, record in peak_records.items()
    }
    for level in GRID_LEVELS:
        fold_families = {
            fold: {
                matcher: states[fold][matcher][level] for matcher in MATCHERS
            }
            for fold in range(FOLD_COUNT)
        }
        for budget in BUDGETS:
            feasible_pairs = joint_feasible_pairs(
                fold_families[0],
                fold_families[1],
                budget_fa_per_million_pixels=budget,
            )
            oracle = select_joint_global_oracle_pair(
                fold_families[0],
                fold_families[1],
                budget_fa_per_million_pixels=budget,
            )
            selected_joint_ids = None
            selected_pair_by_matcher = {}
            selected_id_sets = []
            for matcher in MATCHERS:
                pair_key = (dataset, int(seed), matcher, level, budget)
                if pair_key not in selected_pair_index:
                    raise GateG0AuditError(f"missing selected pair {pair_key}")
                selected_pair_by_matcher[matcher] = selected_pair_index[pair_key]
                selected_id_sets.append(
                    {
                        stable_id
                        for stable_id in peak_records
                        if bool(
                            selected_target_index[
                                (stable_id, int(seed), matcher, level, budget)
                            ]["matched"]
                        )
                    }
                )
            selected_joint_ids = set.intersection(*selected_id_sets)
            oracle_payload = _oracle_payload(oracle, fold_families)
            target_count = len(peak_records)
            selected_joint_count = len(selected_joint_ids)
            oracle_joint_count = int(oracle_payload["joint_matched_components"])
            selected_threshold_pairs = {
                matcher: selected_pair_by_matcher[matcher][
                    "selected_thresholds_by_evaluation_fold"
                ]
                for matcher in MATCHERS
            }
            selected_common_pair = all(
                thresholds == selected_threshold_pairs[MATCHERS[0]]
                for thresholds in selected_threshold_pairs.values()
            )
            selected_budget_feasible_both = all(
                bool(
                    selected_pair_by_matcher[matcher]["held_out_pooled"][
                        "budget_feasible_zero_overshoot"
                    ]
                )
                for matcher in MATCHERS
            )
            comparator_status = (
                "comparable_same_pair_and_held_out_budget_feasible"
                if selected_common_pair and selected_budget_feasible_both
                else "not_same_budget_comparable"
            )
            pair_rows.append(
                {
                    "schema_version": PAIR_SCHEMA,
                    "dataset": dataset,
                    "seed": int(seed),
                    "grid_level": level,
                    "nominal_budget_fa_per_mpix": budget,
                    "joint_feasible_pair_count": len(feasible_pairs),
                    "no_feasible_finite_pair": oracle is None,
                    "feasible_zero_joint_hit": (
                        oracle is not None and oracle_joint_count == 0
                    ),
                    "selected_common_pair_across_matchers": selected_common_pair,
                    "selected_held_out_budget_feasible_both_matchers": (
                        selected_budget_feasible_both
                    ),
                    "oracle_comparator_status": comparator_status,
                    "selected_joint_matched_components": selected_joint_count,
                    "target_components": target_count,
                    "selected_joint_pd": selected_joint_count / target_count,
                    "joint_fold_pair_oracle": oracle_payload,
                    "oracle_joint_pd": oracle_joint_count / target_count,
                    "oracle_delta_joint_matches_vs_selected": (
                        oracle_joint_count - selected_joint_count
                    ),
                    "oracle_delta_joint_pd_vs_selected": (
                        oracle_joint_count - selected_joint_count
                    )
                    / target_count,
                    "selected_matcher_pairs": {
                        matcher: {
                            "thresholds_by_fold": selected_pair_by_matcher[matcher][
                                "selected_thresholds_by_evaluation_fold"
                            ],
                            "held_out_pooled": selected_pair_by_matcher[matcher][
                                "held_out_pooled"
                            ],
                        }
                        for matcher in MATCHERS
                    },
                    "checkpoint": dict(checkpoint),
                }
            )
            for fold in range(FOLD_COUNT):
                selected_by_matcher = {
                    matcher: _selected_target_map(
                        selected_target_index,
                        all_target_ids_by_fold[fold],
                        seed=seed,
                        matcher=matcher,
                        grid_level=level,
                        budget=budget,
                    )
                    for matcher in MATCHERS
                }
                classified = classify_joint_matcher_targets(
                    fold,
                    fold_families[0],
                    fold_families[1],
                    budget_fa_per_million_pixels=budget,
                    selected_by_matcher=selected_by_matcher,
                    joint_global_oracle=oracle,
                )
                for status in classified:
                    metadata = target_metadata[status.stable_target_id]
                    target_rows.append(
                        {
                            "schema_version": TARGET_SCHEMA,
                            "dataset": dataset,
                            "seed": int(seed),
                            "image_name": metadata["image_name"],
                            "evaluation_fold": fold,
                            "grid_level": level,
                            "nominal_budget_fa_per_mpix": budget,
                            **asdict(status),
                            "target_area": metadata["area"],
                            "component_mask_sha256": metadata[
                                "component_mask_sha256"
                            ],
                            "label_mask_sha256": metadata["label_mask_sha256"],
                            "core_peak": metadata["core_peak"],
                            "local_support_peak": metadata[
                                "local_support_peak"
                            ],
                            "local_support_definition": metadata[
                                "local_support_definition"
                            ],
                            "analysis_semantics": (
                                "development-only post-hoc Q-grid phenotype; "
                                "target-wise witnesses cannot be summed as Pd"
                            ),
                            "checkpoint": dict(checkpoint),
                        }
                    )

    _replay_selected_states(
        dataset,
        seed,
        states,
        selection_index=selection_index,
        selected_target_index=selected_target_index,
        image_names=image_names,
        registry=registry,
    )
    return curve_rows, target_rows, pair_rows


def _replay_selected_states(
    dataset: str,
    seed: int,
    states: Mapping[int, Mapping[str, Mapping[str, Sequence[FoldCandidateState]]]],
    *,
    selection_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    selected_target_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    image_names: Sequence[str],
    registry: Mapping[str, StableTargetSet],
) -> None:
    for fold in range(FOLD_COUNT):
        fold_target_ids = [
            target.stable_key
            for name in image_names
            if image_fold(name) == fold
            for target in registry[name].targets
        ]
        for matcher in MATCHERS:
            for level in GRID_LEVELS:
                by_threshold = {
                    state.threshold: state for state in states[fold][matcher][level]
                }
                for budget in BUDGETS:
                    key = (dataset, int(seed), matcher, level, budget, fold)
                    if key not in selection_index:
                        raise GateG0AuditError(f"missing selection replay row {key}")
                    frozen = selection_index[key]
                    threshold = float(frozen["selection"]["threshold"])
                    if threshold not in by_threshold:
                        raise GateG0AuditError("selected threshold is absent from curve")
                    state = by_threshold[threshold]
                    for field in COUNT_FIELDS:
                        if int(frozen["selection"][field]) < 0:
                            raise GateG0AuditError("invalid calibration selection count")
                    selected_ids = {
                        stable_id
                        for stable_id in fold_target_ids
                        if bool(
                            selected_target_index[
                                (stable_id, int(seed), matcher, level, budget)
                            ]["matched"]
                        )
                    }
                    if selected_ids != set(state.matched_target_ids):
                        raise GateG0AuditError("selected target replay mismatch")


def _persistent_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    level: str,
    budget: int,
    category_field: str,
) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        if (
            row["dataset"] == dataset
            and row["grid_level"] == level
            and int(row["nominal_budget_fa_per_mpix"]) == budget
        ):
            grouped[str(row["stable_target_id"])].append(row)
    if not grouped or any(len(group) != 3 for group in grouped.values()):
        raise GateG0AuditError("persistent target grouping requires three seeds")
    persistent_miss = []
    persistent_conversion = []
    for stable_id, group in grouped.items():
        miss_count = sum(
            row[category_field]
            not in {CATEGORY_SELECTED_HIT, CATEGORY_MATCHER_SENSITIVE}
            for row in group
        )
        conversion_count = sum(
            row[category_field] == CATEGORY_COMPONENT_CONVERSION
            for row in group
        )
        if miss_count >= 2:
            persistent_miss.append(stable_id)
        if conversion_count >= 2:
            persistent_conversion.append(stable_id)
    images = [
        next(
            row["image_name"]
            for row in grouped[stable_id]
            if row[category_field] == CATEGORY_COMPONENT_CONVERSION
        )
        for stable_id in persistent_conversion
    ]
    image_counts = Counter(images)
    return {
        "persistent_joint_miss_target_count": len(persistent_miss),
        "persistent_conversion_target_count": len(persistent_conversion),
        "persistent_conversion_coverage": (
            len(persistent_conversion) / len(persistent_miss)
            if persistent_miss
            else 0.0
        ),
        "persistent_conversion_image_count": len(image_counts),
        "maximum_single_image_share": (
            max(image_counts.values()) / len(persistent_conversion)
            if persistent_conversion
            else 0.0
        ),
        "persistent_conversion_target_ids": sorted(persistent_conversion),
    }


def summarize_frontier_decomposition(
    target_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in target_rows})
    if len(datasets) != 3:
        raise GateG0AuditError("Gate G0 requires exactly three datasets")
    category_counts = defaultdict(Counter)
    seed_rows = {}
    for dataset in datasets:
        seeds = sorted(
            {int(row["seed"]) for row in target_rows if row["dataset"] == dataset}
        )
        if len(seeds) != 3:
            raise GateG0AuditError("each dataset requires exactly three seeds")
        for seed in seeds:
            for level in GRID_LEVELS:
                for budget in BUDGETS:
                    group = [
                        row
                        for row in target_rows
                        if row["dataset"] == dataset
                        and int(row["seed"]) == seed
                        and row["grid_level"] == level
                        and int(row["nominal_budget_fa_per_mpix"]) == budget
                    ]
                    counts = Counter(row["category_support"] for row in group)
                    core_counts = Counter(row["category_core"] for row in group)
                    miss_denominator = sum(
                        count
                        for category, count in counts.items()
                        if category
                        not in {CATEGORY_SELECTED_HIT, CATEGORY_MATCHER_SENSITIVE}
                    )
                    key = (dataset, seed, level, budget)
                    seed_rows[key] = {
                        "dataset": dataset,
                        "seed": seed,
                        "grid_level": level,
                        "nominal_budget_fa_per_mpix": budget,
                        "category_counts_support": dict(counts),
                        "category_counts_core": dict(core_counts),
                        "joint_selected_miss_count": miss_denominator,
                        "support_conversion_count": counts[
                            CATEGORY_COMPONENT_CONVERSION
                        ],
                        "core_conversion_count": core_counts[
                            CATEGORY_COMPONENT_CONVERSION
                        ],
                        "support_conversion_share_of_joint_misses": (
                            counts[CATEGORY_COMPONENT_CONVERSION] / miss_denominator
                            if miss_denominator
                            else 0.0
                        ),
                        "core_conversion_share_of_joint_misses": (
                            core_counts[CATEGORY_COMPONENT_CONVERSION]
                            / miss_denominator
                            if miss_denominator
                            else 0.0
                        ),
                    }
                    category_counts[(level, budget)].update(counts)

    persistent = {
        level: {
            str(budget): {
                dataset: {
                    "support": _persistent_counts(
                        target_rows,
                        dataset=dataset,
                        level=level,
                        budget=budget,
                        category_field="category_support",
                    ),
                    "core": _persistent_counts(
                        target_rows,
                        dataset=dataset,
                        level=level,
                        budget=budget,
                        category_field="category_core",
                    ),
                }
                for dataset in datasets
            }
            for budget in BUDGETS
        }
        for level in GRID_LEVELS
    }

    pair_index = {
        (
            str(row["dataset"]),
            int(row["seed"]),
            str(row["grid_level"]),
            int(row["nominal_budget_fa_per_mpix"]),
        ): row
        for row in pair_rows
    }
    by_budget = {}
    passing_budgets = []
    for budget_index, budget in enumerate(BUDGETS):
        dataset_records = {}
        primary_datasets = []
        for dataset in datasets:
            seeds = sorted(
                {key[1] for key in seed_rows if key[0] == dataset}
            )
            finite_pair_seeds = [
                seed
                for seed in seeds
                if not bool(
                    pair_index[(dataset, seed, "Q2", budget)][
                        "no_feasible_finite_pair"
                    ]
                )
            ]
            seed_qualifiers = [
                seed
                for seed in finite_pair_seeds
                if seed_rows[(dataset, seed, "Q2", budget)][
                    "core_conversion_count"
                ]
                >= 6
                and seed_rows[(dataset, seed, "Q2", budget)][
                    "core_conversion_share_of_joint_misses"
                ]
                >= 0.40
            ]
            q2 = persistent["Q2"][str(budget)][dataset]["core"]
            q1 = persistent["Q1"][str(budget)][dataset]["core"]
            grid_stable = abs(
                q2["persistent_conversion_coverage"]
                - q1["persistent_conversion_coverage"]
            ) <= 0.10
            primary = (
                len(finite_pair_seeds) >= 2
                and len(seed_qualifiers) >= 2
                and q2["persistent_conversion_target_count"] >= 12
                and q2["persistent_conversion_coverage"] >= 0.40
                and q2["persistent_conversion_image_count"] >= 8
                and q2["maximum_single_image_share"] <= 0.25
                and grid_stable
            )
            if primary:
                primary_datasets.append(dataset)
            dataset_records[dataset] = {
                "finite_pair_seed_count": len(finite_pair_seeds),
                "seed_level_qualifying_count": len(seed_qualifiers),
                "seed_level_qualifying_seeds": seed_qualifiers,
                "q2_core_persistent": q2,
                "q1_core_persistent": q1,
                "q1_q2_coverage_difference": (
                    q2["persistent_conversion_coverage"]
                    - q1["persistent_conversion_coverage"]
                ),
                "q1_q2_stable_within_10pp": grid_stable,
                "primary_pass": primary,
            }
        adjacent_budgets = [
            value
            for index, value in enumerate(BUDGETS)
            if abs(index - budget_index) == 1
        ]
        adjacent_support = {}
        for adjacent in adjacent_budgets:
            supported = [
                dataset
                for dataset in primary_datasets
                if persistent["Q2"][str(adjacent)][dataset]["core"][
                    "persistent_conversion_target_count"
                ]
                >= 8
                and persistent["Q2"][str(adjacent)][dataset]["core"][
                    "persistent_conversion_coverage"
                ]
                >= 0.25
            ]
            adjacent_support[str(adjacent)] = supported
        common_adjacent = [
            adjacent
            for adjacent, supported in adjacent_support.items()
            if len(set(supported)) >= 2
        ]
        passed = len(primary_datasets) >= 2 and bool(common_adjacent)
        if passed:
            passing_budgets.append(budget)
        by_budget[str(budget)] = {
            "by_dataset": dataset_records,
            "primary_passing_datasets": primary_datasets,
            "adjacent_budget_support": adjacent_support,
            "common_adjacent_passing_budgets": common_adjacent,
            "pass": passed,
        }

    q1q2_transitions = Counter()
    q1_index = {
        (
            str(row["stable_target_id"]),
            int(row["seed"]),
            int(row["nominal_budget_fa_per_mpix"]),
        ): str(row["category_core"])
        for row in target_rows
        if row["grid_level"] == "Q1"
    }
    for row in target_rows:
        if row["grid_level"] != "Q2":
            continue
        key = (
            str(row["stable_target_id"]),
            int(row["seed"]),
            int(row["nominal_budget_fa_per_mpix"]),
        )
        q1q2_transitions[(q1_index[key], str(row["category_core"]))] += 1

    raw_pair_gap_diagnostic = {}
    comparable_pair_gain_gate = {}
    for budget in BUDGETS:
        raw_by_dataset = {}
        comparable_by_dataset = {}
        for dataset in datasets:
            raw_qualifying = [
                int(row["seed"])
                for row in pair_rows
                if row["dataset"] == dataset
                and row["grid_level"] == "Q2"
                and int(row["nominal_budget_fa_per_mpix"]) == budget
                and not row["no_feasible_finite_pair"]
                and int(row["oracle_delta_joint_matches_vs_selected"]) >= 3
                and float(row["oracle_delta_joint_pd_vs_selected"]) >= 0.05
            ]
            comparable_qualifying = [
                int(row["seed"])
                for row in pair_rows
                if row["dataset"] == dataset
                and row["grid_level"] == "Q2"
                and int(row["nominal_budget_fa_per_mpix"]) == budget
                and row["oracle_comparator_status"]
                == "comparable_same_pair_and_held_out_budget_feasible"
                and not row["no_feasible_finite_pair"]
                and int(row["oracle_delta_joint_matches_vs_selected"]) >= 3
                and float(row["oracle_delta_joint_pd_vs_selected"]) >= 0.05
            ]
            raw_by_dataset[dataset] = {
                "qualifying_seeds": raw_qualifying,
                "cross_seed_pattern": len(raw_qualifying) >= 2,
            }
            comparable_by_dataset[dataset] = {
                "qualifying_seeds": comparable_qualifying,
                "pass": len(comparable_qualifying) >= 2,
            }
        raw_pair_gap_diagnostic[str(budget)] = {
            "by_dataset": raw_by_dataset,
            "crossdataset_pattern_count": sum(
                value["cross_seed_pattern"] for value in raw_by_dataset.values()
            ),
            "decision_valid": False,
            "reason": (
                "includes selected operating points that may overshoot the "
                "held-out FA budget"
            ),
        }
        comparable_pair_gain_gate[str(budget)] = {
            "by_dataset": comparable_by_dataset,
            "passing_dataset_count": sum(
                value["pass"] for value in comparable_by_dataset.values()
            ),
            "pass": sum(
                value["pass"] for value in comparable_by_dataset.values()
            )
            >= 2,
        }

    return {
        "schema_version": SUMMARY_SCHEMA,
        "analysis_scope": (
            "development-only post-hoc finite Q1/Q2 fold-pair oracle; "
            "official test sealed; no deployable/generalization/causal claim"
        ),
        "protocol": {
            "grid_levels": list(GRID_LEVELS),
            "budgets_fa_per_mpix": list(BUDGETS),
            "matchers": list(MATCHERS),
            "selected_cohort": "Q1 or Q2 selected pair, evaluated per level",
            "primary_direction_level": "Q2",
            "strict_threshold_operator": ">",
            "joint_feasibility": (
                "same finite threshold pair feasible by integer pooled FA "
                "under both matchers"
            ),
            "no_feasible_finite_pair": (
                "no calibration-derived finite threshold pair is pooled-budget "
                "feasible under both matchers; zero joint hits alone are not "
                "a budget collapse"
            ),
            "oracle_comparator": (
                "decision-valid comparison requires the selected thresholds to "
                "be the same pair for both matchers and achieved held-out FA to "
                "satisfy the nominal budget under both matchers"
            ),
            "targetwise_warning": (
                "each target may have a different witness pair; counts are "
                "phenotypes and cannot be summed into achievable Pd"
            ),
            "local_support": (
                "fixed pixels at strict Euclidean distance <3 from the GT component"
            ),
        },
        "category_counts_support": {
            f"{level}:FA{budget}": dict(counts)
            for (level, budget), counts in sorted(category_counts.items())
        },
        "category_counts_core": {
            f"{level}:FA{budget}": dict(
                Counter(
                    row["category_core"]
                    for row in target_rows
                    if row["grid_level"] == level
                    and int(row["nominal_budget_fa_per_mpix"]) == budget
                )
            )
            for level in GRID_LEVELS
            for budget in BUDGETS
        },
        "seed_records": [seed_rows[key] for key in sorted(seed_rows)],
        "persistent_records": persistent,
        "q1_to_q2_core_transition_counts": {
            f"{source} -> {target}": count
            for (source, target), count in sorted(q1q2_transitions.items())
        },
        "joint_fold_pair_oracle_raw_gap_diagnostic": raw_pair_gap_diagnostic,
        "joint_fold_pair_oracle_comparable_gain_gate": comparable_pair_gain_gate,
        "component_conversion_direction_gate": {
            "by_budget": by_budget,
            "passing_budgets": passing_budgets,
            "pass": bool(passing_budgets),
            "authorization_if_pass": (
                "non-topological representation-mechanism search only; "
                "no loss, training, official test, PH/persistence/Betti/" 
                "component-tree method"
            ),
        },
        "method_authorization": False,
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    direction = summary["component_conversion_direction_gate"]
    comparable = summary["joint_fold_pair_oracle_comparable_gain_gate"]
    raw_gap = summary["joint_fold_pair_oracle_raw_gap_diagnostic"]
    lines = [
        "# Gate G0 finite-frontier failure decomposition",
        "",
        str(summary["analysis_scope"]),
        "",
        f"- Component-conversion direction pass: {direction['pass']}",
        f"- Passing budgets: {direction['passing_budgets']}",
        "- Comparable joint-oracle gain pass: %s"
        % any(value["pass"] for value in comparable.values()),
        "- Method/training authorization: False",
        "",
        "| FA/Mpix | primary passing datasets | adjacent support | pass |",
        "|---:|---|---|---:|",
    ]
    for budget in summary["protocol"]["budgets_fa_per_mpix"]:
        row = direction["by_budget"][str(budget)]
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                budget,
                row["primary_passing_datasets"],
                row["common_adjacent_passing_budgets"],
                row["pass"],
            )
        )
    lines.extend(
        [
            "",
            "| FA/Mpix | comparable gain datasets | raw post-hoc pattern datasets | comparable pass |",
            "|---:|---:|---:|---:|",
        ]
    )
    for budget in summary["protocol"]["budgets_fa_per_mpix"]:
        comparable_row = comparable[str(budget)]
        raw_row = raw_gap[str(budget)]
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                budget,
                comparable_row["passing_dataset_count"],
                raw_row["crossdataset_pattern_count"],
                comparable_row["pass"],
            )
        )
    lines.extend(
        [
            "",
            "Raw oracle gaps are descriptive only: rows whose selected point overshoots the held-out budget are not same-budget comparisons.",
            "A zero-hit feasible family is classified by peak activation, not as budget collapse.",
            "",
            "Target-wise witnesses are not a simultaneously achievable Pd.",
            "All categories are finite-Q-grid operational phenotypes.",
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
    curve_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
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
        artifacts = []
        for name, rows in zip(OUTPUT_FILES[:3], (curve_rows, target_rows, pair_rows)):
            path = temporary / name
            _write_jsonl(path, rows)
            artifacts.append(path)
        summary_path = temporary / OUTPUT_FILES[3]
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
        markdown_path = temporary / OUTPUT_FILES[4]
        markdown_path.write_text(build_markdown(summary), encoding="utf-8")
        artifacts.extend((summary_path, markdown_path))
        complete_provenance = {
            **dict(provenance),
            "artifact_sha256": {
                path.name: sha256_file(path) for path in artifacts
            },
        }
        (temporary / OUTPUT_FILES[5]).write_text(
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
            raise GateG0AuditError("temporary Gate G0 inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_hashes() -> dict[str, str]:
    paths = {
        "tool": Path(__file__).resolve(),
        "frontier_utility": ROOT / "utils" / "component_frontier_decomposition.py",
        "nested_grid_utility": ROOT / "utils" / "nested_component_grid.py",
        "component_operating_point": ROOT / "utils" / "component_operating_point.py",
        "cross_fitted_low_fa": ROOT / "utils" / "cross_fitted_low_fa.py",
        "metric": ROOT / "utils" / "metric.py",
        "target_identity": ROOT / "utils" / "target_identity.py",
        "gate_e_tool": ROOT / "tools" / "audit_gate_e_low_fa_bridge.py",
        "gate_f_tool": ROOT / "tools" / "audit_gate_f_nested_grid_sensitivity.py",
        "dataset": ROOT / "utils" / "data.py",
        "mshnet": ROOT / "model" / "MSHNet.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.batch_size != 8:
        raise GateG0AuditError("frozen inference replay requires batch size 8")
    if args.num_workers < 0:
        raise GateG0AuditError("num_workers must be non-negative")
    if args.analysis_workers < 1 or args.analysis_workers > 9:
        raise GateG0AuditError("analysis_workers must lie in [1, 9]")
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    input_provenance, input_summary = _verify_input_bundle(input_dir)
    input_provenance_sha256 = sha256_file(input_dir / "provenance.json")
    source_hashes = _source_hashes()
    selection_index = _index_unique(
        _read_jsonl(input_dir / "selection_sensitivity.jsonl"),
        _selection_key,
        label="selection",
    )
    selected_target_index = _index_unique(
        _read_jsonl(input_dir / "target_sensitivity.jsonl"),
        _target_key,
        label="selected target",
    )
    selected_pair_index = _index_unique(
        _read_jsonl(input_dir / "pair_sensitivity.jsonl"),
        _pair_key,
        label="selected pair",
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

    curve_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    checkpoint_records = []
    inference_records = []
    job_results: dict[
        int,
        tuple[
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, Any]],
        ],
    ] = {}
    executor = (
        ProcessPoolExecutor(
            max_workers=args.analysis_workers,
            mp_context=mp.get_context("spawn"),
        )
        if args.analysis_workers > 1
        else None
    )
    futures = {}
    try:
        for job_index, job in enumerate(jobs):
            dataset = str(job["dataset"])
            seed = int(job["seed"])
            logits, targets, names, record = collect_job_predictions(
                job,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                expected_registry=authoritative[dataset],
            )
            job_selection_index = {
                key: value
                for key, value in selection_index.items()
                if key[0] == dataset and key[1] == seed
            }
            job_target_index = {
                key: value
                for key, value in selected_target_index.items()
                if str(value["dataset"]) == dataset and int(value["seed"]) == seed
            }
            job_pair_index = {
                key: value
                for key, value in selected_pair_index.items()
                if key[0] == dataset and key[1] == seed
            }
            arguments = (
                logits,
                targets,
                names,
            )
            keywords = {
                "dataset": dataset,
                "seed": seed,
                "registry": authoritative[dataset],
                "checkpoint": record["checkpoint"],
                "selection_index": job_selection_index,
                "selected_target_index": job_target_index,
                "selected_pair_index": job_pair_index,
            }
            if executor is None:
                job_results[job_index] = analyze_job_frontier(
                    *arguments, **keywords
                )
            else:
                future = executor.submit(
                    analyze_job_frontier, *arguments, **keywords
                )
                futures[future] = job_index
            checkpoint_records.append(record["checkpoint"])
            inference_records.append(record["inference"])
        for future in as_completed(futures):
            job_results[futures[future]] = future.result()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if set(job_results) != set(range(len(jobs))):
        raise GateG0AuditError("parallel job result inventory drifted")
    for job_index in range(len(jobs)):
        job_curves, job_targets, job_pairs = job_results[job_index]
        curve_rows.extend(job_curves)
        target_rows.extend(job_targets)
        pair_rows.extend(job_pairs)

    expected_target_instances = (
        sum(
            len(target_set.targets)
            for registry in authoritative.values()
            for target_set in registry.values()
        )
        * 3
    )
    if len(target_rows) != expected_target_instances * len(GRID_LEVELS) * len(BUDGETS):
        raise GateG0AuditError("target decomposition cardinality drifted")
    if len(pair_rows) != 9 * len(GRID_LEVELS) * len(BUDGETS):
        raise GateG0AuditError("joint pair cardinality drifted")
    if not curve_rows:
        raise GateG0AuditError("fold curve ledger is empty")
    if checkpoint_records != input_provenance.get("jobs"):
        raise GateG0AuditError("checkpoint replay differs from Gate F-1a")
    if inference_records != input_provenance.get("inference"):
        raise GateG0AuditError("inference replay differs from Gate F-1a")
    if batch_provenance != input_provenance.get("batch"):
        raise GateG0AuditError("batch provenance differs from Gate F-1a")
    if authority_records != input_provenance.get(
        "authoritative_registry_construction"
    ) or registry_order != input_provenance.get("registry_precheckpoint_order"):
        raise GateG0AuditError("target authority differs from Gate F-1a")

    summary = summarize_frontier_decomposition(target_rows, pair_rows)
    rechecked_provenance, rechecked_summary = _verify_input_bundle(input_dir)
    if (
        rechecked_provenance != input_provenance
        or rechecked_summary != input_summary
        or sha256_file(input_dir / "provenance.json") != input_provenance_sha256
    ):
        raise GateG0AuditError("Gate F-1a input changed during execution")
    if _source_hashes() != source_hashes:
        raise GateG0AuditError("Gate G0 sources changed during execution")
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "git": git_worktree_provenance(),
        "source_sha256": source_hashes,
        "gate_f_input": {
            "directory": str(input_dir),
            "provenance_sha256": input_provenance_sha256,
            "artifact_sha256": dict(input_provenance["artifact_sha256"]),
            "targeted_exact_interval_gate_pass": False,
        },
        "batch": batch_provenance,
        "registry_precheckpoint_order": registry_order,
        "authoritative_registry_construction": authority_records,
        "jobs": checkpoint_records,
        "inference": inference_records,
        "protocol": summary["protocol"],
        "official_test_policy": "sealed and never opened",
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit_image": importlib.metadata.version("scikit-image"),
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "analysis_workers": args.analysis_workers,
        },
    }
    write_bundle(
        output_dir,
        curve_rows=curve_rows,
        target_rows=target_rows,
        pair_rows=pair_rows,
        summary=summary,
        provenance=provenance,
    )
    return summary, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary, output_dir = run(parse_args(argv))
    except (FileExistsError, GateG0AuditError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "component_conversion_direction_gate": summary[
                    "component_conversion_direction_gate"
                ],
                "joint_fold_pair_oracle_comparable_gain_gate": summary[
                    "joint_fold_pair_oracle_comparable_gain_gate"
                ],
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
