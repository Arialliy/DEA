#!/usr/bin/env python3
"""Frozen MSHNet phase-survival falsification gate.

This tool is deliberately a read-only development audit.  It asks whether
integer input-lattice residue changes the target/clutter ordering enough to
alter the low component-FA frontier.  It is not a test-time-augmentation method,
does not authorize training, and never opens the official test split.

Four score fields are compared for every fixed checkpoint:

``canonical``
    The original MSHNet full-graph logit.
``unit_phase_max``
    Max of aligned logits from offsets (0,0), (0,1), (1,0), (1,1).
``lattice_control_max``
    Same-cost max using offsets that preserve the deepest stride residue.
``lattice_plus_unit_max``
    Same-cost max using offsets one pixel beyond the deepest stride.  This is
    the magnitude-matched residue-changing partner of ``lattice_control_max``.

Both controls are essential.  A unit-vs-stride comparison alone confounds
phase with displacement magnitude; the adjacent stride/stride+1 pair is the
causal residue check.  A gain shared by the residue-preserving and
residue-changing ensembles is generic translation/TTA evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import datetime as dt
import hashlib
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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.MSHNet import MSHNet  # noqa: E402
from tools.audit_cross_seed_failure_persistence import (  # noqa: E402
    CANONICAL_SIZE,
    DATASET_NAMES,
    PersistenceAuditError,
    _normalize_state_dict,
    build_authoritative_registries_before_checkpoints,
    git_worktree_provenance,
    load_validated_jobs,
    sha256_file,
    sha256_json,
)
from tools.audit_gate_f_nested_grid_sensitivity import _resolve_device  # noqa: E402
from tools.finalize_clean_baselines import (  # noqa: E402
    FinalizationError,
    load_checkpoint_cpu,
)
from utils.component_operating_point import build_logit_threshold_grid  # noqa: E402
from utils.cross_fitted_low_fa import (  # noqa: E402
    BUDGETS,
    MATCHERS,
    _match,
    cross_fit_job,
)
from utils.data import IRSTD_Dataset  # noqa: E402
from utils.nested_component_grid import (  # noqa: E402
    build_nested_quantile_probability_grids,
    evaluate_nested_component_grids,
)
from utils.phase_intervention import (  # noqa: E402
    UNIT_PHASE_OFFSETS,
    aggregate_aligned_scores,
    phase_preserving_offsets,
    residue_shifted_offsets,
    translate_reflect,
)
from utils.target_identity import (  # noqa: E402
    StableTargetSet,
    assert_same_target_set,
    build_stable_target_set,
)


VARIANTS = (
    "canonical",
    "unit_phase_max",
    "lattice_control_max",
    "lattice_plus_unit_max",
)
SUMMARY_SCHEMA = "dea.phase_survival.summary.v1"
PROVENANCE_SCHEMA = "dea.phase_survival.provenance.v1"
TARGET_SCHEMA = "dea.phase_survival.target_crossfit.v1"
IMAGE_SCHEMA = "dea.phase_survival.image_crossfit.v1"
JOB_SCHEMA = "dea.phase_survival.job_summary.v1"
OUTPUT_FILES = (
    "target_crossfit.jsonl",
    "image_crossfit.jsonl",
    "calibration.json",
    "job_summary.jsonl",
    "phase_survival_summary.json",
    "phase_survival_summary.md",
    "provenance.json",
)


class PhaseSurvivalAuditError(RuntimeError):
    """Raised when the phase-survival audit cannot preserve its contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument(
        "--output-dir",
        default="repro_runs/gate_h/phase_survival_sentinel_v1",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_NAMES),
        help="comma-separated subset of the three clean datasets",
    )
    parser.add_argument(
        "--seeds",
        default="20260711",
        help="comma-separated fixed checkpoint seeds; sentinel defaults to one",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--deepest-stride", type=int, default=16)
    parser.add_argument(
        "--hard-core-source",
        default=(
            "repro_runs/gate_g/frontier_decomposition_v2/"
            "target_decomposition.jsonl"
        ),
    )
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _csv_strings(value: str, *, name: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise PhaseSurvivalAuditError(f"{name} must be non-empty and unique")
    return result


def _csv_ints(value: str, *, name: str) -> tuple[int, ...]:
    raw = _csv_strings(value, name=name)
    try:
        result = tuple(int(item) for item in raw)
    except ValueError as exc:
        raise PhaseSurvivalAuditError(f"{name} must contain integers") from exc
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PhaseSurvivalAuditError(
                        f"{path}:{line_number} is not a JSON object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseSurvivalAuditError(f"cannot read {path}: {exc}") from exc
    return rows


def _hard_core_panel(path: Path) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in _jsonl(path):
        if (
            row.get("grid_level") == "Q2"
            and int(row.get("nominal_budget_fa_per_mpix", -1)) == 20
            and row.get("category_core") == "no_feasible_local_peak_activation"
        ):
            grouped[(str(row["dataset"]), str(row["stable_target_id"]))].append(row)
    panel = []
    for (dataset, stable_target_id), rows in grouped.items():
        seeds = sorted({int(row["seed"]) for row in rows})
        if len(seeds) != 3:
            continue
        first = rows[0]
        panel.append(
            {
                "dataset": dataset,
                "stable_target_id": stable_target_id,
                "image_name": str(first["image_name"]),
                "target_area": int(first["target_area"]),
                "source_seeds": seeds,
            }
        )
    panel.sort(key=lambda row: (row["dataset"], row["image_name"], row["stable_target_id"]))
    if len(panel) != 16:
        raise PhaseSurvivalAuditError(
            f"formal FA20 Q2 hard-core panel must contain 16 targets, got {len(panel)}"
        )
    return tuple(panel)


def _model_logits(model: MSHNet, images: torch.Tensor) -> torch.Tensor:
    output = model(images, True)
    if not isinstance(output, tuple) or len(output) != 2:
        raise PhaseSurvivalAuditError("MSHNet full graph returned an invalid output")
    logits = output[1]
    if (
        not torch.is_tensor(logits)
        or logits.ndim != 4
        or tuple(logits.shape[1:]) != (1, CANONICAL_SIZE, CANONICAL_SIZE)
        or not bool(torch.isfinite(logits).all())
    ):
        raise PhaseSurvivalAuditError("MSHNet returned invalid canonical logits")
    return logits


def collect_phase_predictions(
    job: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    deepest_stride: int,
    expected_registry: Mapping[str, StableTargetSet],
) -> tuple[
    dict[str, tuple[np.ndarray, ...]],
    dict[tuple[int, int], tuple[np.ndarray, ...]],
    tuple[np.ndarray, ...],
    tuple[str, ...],
    dict[str, Any],
]:
    stored_args = job.get("stored_args")
    if not isinstance(stored_args, dict):
        raise PhaseSurvivalAuditError("validated job lacks stored args")
    dataset = IRSTD_Dataset(argparse.Namespace(**stored_args), mode="val")
    if dataset.split_sha256 != job["split_hashes"]["validation"]:
        raise PhaseSurvivalAuditError("validation split hash drifted")
    if dataset.base_size != CANONICAL_SIZE or dataset.crop_size != CANONICAL_SIZE:
        raise PhaseSurvivalAuditError("phase audit requires canonical 256x256 input")
    if tuple(expected_registry) != tuple(dataset.names):
        raise PhaseSurvivalAuditError("validation image universe disagrees with authority")

    checkpoint_path = Path(str(job["checkpoint"])).resolve()
    if sha256_file(checkpoint_path) != job["checkpoint_sha256"]:
        raise PhaseSurvivalAuditError("checkpoint hash drifted")
    checkpoint_value = load_checkpoint_cpu(checkpoint_path)
    state = checkpoint_value.get("net")
    if not isinstance(state, Mapping) or not state:
        raise PhaseSurvivalAuditError("checkpoint has no valid network state")
    model = MSHNet(3)
    model.load_state_dict(_normalize_state_dict(state), strict=True)
    del state, checkpoint_value
    model.requires_grad_(False).to(device).eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    unit_offsets = UNIT_PHASE_OFFSETS
    control_offsets = phase_preserving_offsets(deepest_stride)
    paired_phase_offsets = residue_shifted_offsets(deepest_stride)
    unique_offsets = tuple(
        dict.fromkeys((*unit_offsets, *control_offsets, *paired_phase_offsets))
    )
    variant_scores: dict[str, list[np.ndarray]] = {name: [] for name in VARIANTS}
    aligned_unit_scores: dict[tuple[int, int], list[np.ndarray]] = {
        offset: [] for offset in unit_offsets
    }
    targets: list[np.ndarray] = []
    cursor = 0
    repeat_max_abs_difference = 0.0
    forward_calls = 0

    with torch.inference_mode():
        for images, batch_targets in loader:
            images = images.to(device, non_blocking=True)
            raw_by_offset: dict[tuple[int, int], torch.Tensor] = {}
            for offset in unique_offsets:
                translated = images if offset == (0, 0) else translate_reflect(images, offset)
                raw_by_offset[offset] = _model_logits(model, translated)
                forward_calls += 1
            repeated = _model_logits(model, images)
            forward_calls += 1
            repeat_max_abs_difference = max(
                repeat_max_abs_difference,
                float((raw_by_offset[(0, 0)] - repeated).abs().max().item()),
            )
            if not torch.equal(raw_by_offset[(0, 0)], repeated):
                raise PhaseSurvivalAuditError("repeated canonical inference is not bitwise stable")

            unit_aggregate, unit_stack, _ = aggregate_aligned_scores(
                [raw_by_offset[offset] for offset in unit_offsets], unit_offsets
            )
            control_aggregate, _, control_validity = aggregate_aligned_scores(
                [raw_by_offset[offset] for offset in control_offsets], control_offsets
            )
            paired_phase_aggregate, _, paired_phase_validity = aggregate_aligned_scores(
                [raw_by_offset[offset] for offset in paired_phase_offsets],
                paired_phase_offsets,
            )
            common_pair_valid = torch.logical_and(
                control_validity.all(dim=0), paired_phase_validity.all(dim=0)
            )
            canonical = raw_by_offset[(0, 0)]
            control_aggregate = torch.where(
                common_pair_valid, control_aggregate, canonical
            )
            paired_phase_aggregate = torch.where(
                common_pair_valid, paired_phase_aggregate, canonical
            )
            batch_variants = {
                "canonical": canonical,
                "unit_phase_max": unit_aggregate,
                "lattice_control_max": control_aggregate,
                "lattice_plus_unit_max": paired_phase_aggregate,
            }
            batch_target_arrays = (batch_targets[:, 0] > 0.5).numpy().astype(bool, copy=False)
            for batch_index, target in enumerate(batch_target_arrays):
                image_index = cursor + batch_index
                image_name = dataset.names[image_index]
                observed = build_stable_target_set(
                    target,
                    dataset=str(job["dataset"]),
                    image_name=image_name,
                    connectivity=2,
                )
                try:
                    assert_same_target_set(expected_registry[image_name], observed)
                except Exception as exc:
                    raise PhaseSurvivalAuditError(
                        f"inference target differs from authority: {exc}"
                    ) from exc
                targets.append(target.astype(bool, copy=False))
                for variant, values in batch_variants.items():
                    variant_scores[variant].append(
                        values[batch_index, 0].detach().float().cpu().numpy()
                    )
                for offset_index, offset in enumerate(unit_offsets):
                    aligned_unit_scores[offset].append(
                        unit_stack[offset_index, batch_index, 0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
            cursor += int(batch_targets.shape[0])
    if cursor != len(dataset):
        raise PhaseSurvivalAuditError("phase inference image accounting drifted")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint_record = {
        "policy": "fixed_epoch",
        "epoch": int(job["checkpoint_summary"]["epoch"]),
        "path": str(checkpoint_path),
        "sha256": job["checkpoint_sha256"],
        "job_id": job["job_id"],
        "run_config_sha256": job["run_config_sha256"],
        "validation_split_sha256": job["split_hashes"]["validation"],
    }
    record = {
        "checkpoint": checkpoint_record,
        "inference": {
            "job_id": job["job_id"],
            "dataset": job["dataset"],
            "seed": int(job["seed"]),
            "image_count": len(dataset),
            "target_count": sum(len(value.targets) for value in expected_registry.values()),
            "forward_calls": forward_calls,
            "unique_offsets": [list(offset) for offset in unique_offsets],
            "unit_offsets": [list(offset) for offset in unit_offsets],
            "control_offsets": [list(offset) for offset in control_offsets],
            "paired_phase_offsets": [list(offset) for offset in paired_phase_offsets],
            "paired_common_valid_pixels_per_image": int(
                common_pair_valid.sum().item()
            ),
            "paired_common_valid_fraction": float(common_pair_valid.float().mean().item()),
            "repeat_max_abs_difference": repeat_max_abs_difference,
            "post_preprocessing_intervention": True,
        },
    }
    return (
        {name: tuple(values) for name, values in variant_scores.items()},
        {offset: tuple(values) for offset, values in aligned_unit_scores.items()},
        tuple(targets),
        tuple(dataset.names),
        record,
    )


def _pixel_iou(scores: Sequence[np.ndarray], targets: Sequence[np.ndarray], threshold: float) -> float:
    intersection = 0
    union = 0
    for score, target in zip(scores, targets):
        prediction = np.asarray(score) > threshold
        truth = np.asarray(target, dtype=bool)
        intersection += int(np.logical_and(prediction, truth).sum())
        union += int(np.logical_or(prediction, truth).sum())
    return float(intersection / union) if union else 1.0


def oracle_summary(
    scores: Sequence[np.ndarray], targets: Sequence[np.ndarray]
) -> dict[str, Any]:
    result = evaluate_nested_component_grids(
        scores,
        targets,
        BUDGETS,
        fixed_thresholds=(0.0,),
    )
    by_matcher: dict[str, Any] = {}
    q2_thresholds: tuple[float, ...] | None = None
    for matcher in MATCHERS:
        level = result.matcher(matcher).level("Q2")
        if q2_thresholds is None:
            q2_thresholds = level.threshold_grid
        elif q2_thresholds != level.threshold_grid:
            raise PhaseSurvivalAuditError("matcher Q2 grids unexpectedly differ")
        by_matcher[matcher] = {
            str(selection.budget_fa_per_million_pixels): {
                **asdict(selection.operating_point),
                "budget_fa_per_million_pixels": selection.budget_fa_per_million_pixels,
            }
            for selection in level.selections
        }
    assert q2_thresholds is not None
    iou_values = [
        {"threshold": float(threshold), "iou": _pixel_iou(scores, targets, threshold)}
        for threshold in q2_thresholds
    ]
    best_iou = max(iou_values, key=lambda row: (row["iou"], row["threshold"]))
    return {
        "semantics": "same-development-set post-hoc Q2 oracle; not deployable performance",
        "by_matcher": by_matcher,
        "max_pooled_pixel_iou": best_iou,
        "threshold_count": len(q2_thresholds),
    }


def _pooled_crossfit(image_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not image_rows:
        raise PhaseSurvivalAuditError("cross-fit image ledger is empty")
    aggregate = image_rows[0].get("dataset_seed_aggregate")
    if not isinstance(aggregate, Mapping):
        raise PhaseSurvivalAuditError("cross-fit ledger lacks pooled aggregate")
    if any(row.get("dataset_seed_aggregate") != aggregate for row in image_rows):
        raise PhaseSurvivalAuditError("cross-fit pooled aggregate drifted within a group")
    return dict(aggregate)


def _crossfit_summary(
    image_rows: Sequence[Mapping[str, Any]],
    *,
    scores: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for matcher in MATCHERS:
        result[matcher] = {}
        for budget in BUDGETS:
            selected = [
                row
                for row in image_rows
                if row["matcher"] == matcher
                and int(row["nominal_budget_fa_per_mpix"]) == budget
            ]
            if len(selected) != len(scores):
                raise PhaseSurvivalAuditError("cross-fit summary image coverage drifted")
            by_index = {int(row["image_index"]): row for row in selected}
            if set(by_index) != set(range(len(scores))):
                raise PhaseSurvivalAuditError("cross-fit summary image indices drifted")
            fold_records = {}
            intersection = 0
            union = 0
            for image_index, (score, target) in enumerate(zip(scores, targets)):
                row = by_index[image_index]
                fold = int(row["evaluation_fold"])
                held_out = dict(row["held_out_fold_aggregate"])
                if fold in fold_records and fold_records[fold] != held_out:
                    raise PhaseSurvivalAuditError("held-out fold aggregate drifted")
                fold_records[fold] = held_out
                prediction = np.asarray(score) > float(row["calibration_threshold"])
                truth = np.asarray(target, dtype=bool)
                intersection += int(np.logical_and(prediction, truth).sum())
                union += int(np.logical_or(prediction, truth).sum())
            if set(fold_records) != {0, 1}:
                raise PhaseSurvivalAuditError("cross-fit summary lacks a held-out fold")
            pooled = _pooled_crossfit(selected)
            result[matcher][str(budget)] = {
                **pooled,
                "all_held_out_folds_feasible": all(
                    bool(value["budget_feasible_zero_overshoot"])
                    for value in fold_records.values()
                ),
                "held_out_folds": {
                    str(fold): fold_records[fold] for fold in sorted(fold_records)
                },
                "calibration_thresholds": {
                    str(fold): float(
                        next(
                            row["calibration_threshold"]
                            for row in selected
                            if int(row["evaluation_fold"]) == fold
                        )
                    )
                    for fold in sorted(fold_records)
                },
                "crossfit_pooled_pixel_iou": (
                    float(intersection / union) if union else 1.0
                ),
            }
    return result


def _attach_phase_attribution(
    target_rows: list[dict[str, Any]],
    *,
    aligned_unit_scores: Mapping[tuple[int, int], Sequence[np.ndarray]],
    targets: Sequence[np.ndarray],
) -> None:
    cache: dict[tuple[int, str, float, tuple[int, int]], set[int]] = {}
    for row in target_rows:
        image_index = int(row["image_index"])
        matcher = str(row["matcher"])
        threshold = float(row["calibration_threshold"])
        source_index = int(row["source_component_index"])
        matched_offsets = []
        for offset in UNIT_PHASE_OFFSETS:
            key = (image_index, matcher, threshold, offset)
            if key not in cache:
                match = _match(
                    np.asarray(aligned_unit_scores[offset][image_index]),
                    np.asarray(targets[image_index], dtype=bool),
                    threshold=threshold,
                    matcher=matcher,
                )
                cache[key] = {int(target_index) for target_index, _, _ in match.matches}
            if source_index in cache[key]:
                matched_offsets.append(list(offset))
        canonical_matched = [0, 0] in matched_offsets
        nonzero_matched_offsets = [
            offset for offset in matched_offsets if offset != [0, 0]
        ]
        row["unit_phase_individually_matched_offsets"] = matched_offsets
        row["unit_phase_individual_match_exists"] = bool(matched_offsets)
        row["unit_phase_canonical_view_matched"] = canonical_matched
        row["unit_phase_nonzero_matched_offsets"] = nonzero_matched_offsets
        row["unit_phase_exclusive_nonzero_match"] = bool(nonzero_matched_offsets) and not (
            canonical_matched
        )


def _material_gain(target_count: int) -> int:
    return max(2, int(np.ceil(0.02 * target_count)))


def build_job_summary(
    *,
    dataset: str,
    seed: int,
    variant_oracles: Mapping[str, Mapping[str, Any]],
    variant_crossfit: Mapping[str, Mapping[str, Any]],
    variant_target_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    hard_core_ids: set[str],
) -> dict[str, Any]:
    target_count = int(
        variant_oracles["canonical"]["by_matcher"][MATCHERS[0]][str(BUDGETS[0])][
            "target_components"
        ]
    )
    minimum_gain = _material_gain(target_count)
    matcher_results: dict[str, Any] = {}
    for matcher in MATCHERS:
        oracle_comparisons = {}
        crossfit_comparisons = {}
        for budget in BUDGETS:
            key = str(budget)
            oracle_points = {
                variant: variant_oracles[variant]["by_matcher"][matcher][key]
                for variant in VARIANTS
            }
            crossfit_points = {
                variant: variant_crossfit[variant][matcher][key]
                for variant in VARIANTS
            }
            oracle_comparisons[key] = {
                "unit_delta_matched": int(
                    oracle_points["unit_phase_max"]["matched_components"]
                    - oracle_points["canonical"]["matched_components"]
                ),
                "control_delta_matched": int(
                    oracle_points["lattice_control_max"]["matched_components"]
                    - oracle_points["canonical"]["matched_components"]
                ),
                "points": oracle_points,
            }
            all_feasible = all(
                bool(point["budget_feasible_zero_overshoot"])
                for point in crossfit_points.values()
            )
            crossfit_comparisons[key] = {
                "all_variants_zero_overshoot": all_feasible,
                "unit_delta_matched": int(
                    crossfit_points["unit_phase_max"]["matched_components"]
                    - crossfit_points["canonical"]["matched_components"]
                ),
                "control_delta_matched": int(
                    crossfit_points["lattice_control_max"]["matched_components"]
                    - crossfit_points["canonical"]["matched_components"]
                ),
                "points": crossfit_points,
            }

        oracle_unit = [row["unit_delta_matched"] for row in oracle_comparisons.values()]
        oracle_control = [row["control_delta_matched"] for row in oracle_comparisons.values()]
        oracle_pass = (
            sum(value > 0 for value in oracle_unit) >= 3
            and sum(value >= minimum_gain for value in oracle_unit) >= 2
            and min(oracle_unit) >= 0
            and sum(oracle_unit) > max(sum(oracle_control), 0)
        )
        comparable = [
            row for row in crossfit_comparisons.values() if row["all_variants_zero_overshoot"]
        ]
        crossfit_pass = (
            len(comparable) >= 2
            and all(row["unit_delta_matched"] >= 0 for row in comparable)
            and sum(row["unit_delta_matched"] for row in comparable)
            > max(sum(row["control_delta_matched"] for row in comparable), 0)
        )

        unit_rows = [
            row
            for row in variant_target_rows["unit_phase_max"]
            if row["matcher"] == matcher
        ]
        canonical_index = {
            (str(row["stable_target_id"]), int(row["nominal_budget_fa_per_mpix"])): row
            for row in variant_target_rows["canonical"]
            if row["matcher"] == matcher
        }
        recovered = []
        for row in unit_rows:
            key = (str(row["stable_target_id"]), int(row["nominal_budget_fa_per_mpix"]))
            baseline = canonical_index[key]
            if bool(row["low_fa_matched"]) and not bool(baseline["low_fa_matched"]):
                recovered.append(row)
        attributable = sum(bool(row["unit_phase_individual_match_exists"]) for row in recovered)
        attribution_fraction = float(attributable / len(recovered)) if recovered else 0.0
        attribution_pass = bool(recovered) and attribution_fraction >= 0.70

        hard_core = {
            variant: sum(
                bool(row["low_fa_matched"])
                for row in variant_target_rows[variant]
                if row["matcher"] == matcher
                and int(row["nominal_budget_fa_per_mpix"]) == 20
                and str(row["stable_target_id"]) in hard_core_ids
            )
            for variant in VARIANTS
        }
        matcher_results[matcher] = {
            "oracle_comparisons": oracle_comparisons,
            "crossfit_comparisons": crossfit_comparisons,
            "minimum_material_matched_gain": minimum_gain,
            "oracle_direction_pass": oracle_pass,
            "comparable_crossfit_budget_count": len(comparable),
            "crossfit_direction_pass": crossfit_pass,
            "unit_recovered_target_budget_observations": len(recovered),
            "unit_recoveries_individually_attributable": attributable,
            "individual_attribution_fraction": attribution_fraction,
            "individual_attribution_pass": attribution_pass,
            "hard_core_fa20_matched_observations": hard_core,
        }

    iou_delta = float(
        variant_oracles["unit_phase_max"]["max_pooled_pixel_iou"]["iou"]
        - variant_oracles["canonical"]["max_pooled_pixel_iou"]["iou"]
    )
    iou_pass = iou_delta >= -0.005
    job_pass = iou_pass and all(
        result["oracle_direction_pass"]
        and result["crossfit_direction_pass"]
        and result["individual_attribution_pass"]
        for result in matcher_results.values()
    )
    return {
        "schema_version": JOB_SCHEMA,
        "dataset": dataset,
        "seed": int(seed),
        "target_count": target_count,
        "max_iou_unit_minus_canonical": iou_delta,
        "max_iou_non_regression_pass": iou_pass,
        "matchers": matcher_results,
        "job_pass": job_pass,
    }


def build_summary(
    job_summaries: Sequence[Mapping[str, Any]],
    *,
    requested_datasets: Sequence[str],
    requested_seeds: Sequence[int],
) -> dict[str, Any]:
    by_dataset = {}
    for dataset in requested_datasets:
        rows = [row for row in job_summaries if row["dataset"] == dataset]
        seeds = sorted(int(row["seed"]) for row in rows)
        if seeds != sorted(requested_seeds):
            raise PhaseSurvivalAuditError("job summaries do not cover requested seed grid")
        passing = [int(row["seed"]) for row in rows if bool(row["job_pass"])]
        by_dataset[dataset] = {
            "seeds": seeds,
            "passing_seeds": passing,
            "passing_seed_count": len(passing),
            "required_passing_seed_count": 1 if len(seeds) == 1 else 2,
            "pass": len(passing) >= (1 if len(seeds) == 1 else 2),
        }
    sentinel = len(requested_seeds) == 1
    return {
        "schema_version": SUMMARY_SCHEMA,
        "analysis_scope": (
            "development-only frozen post-preprocessing integer-shift intervention; "
            "official test sealed; no method/training/generalization claim"
        ),
        "requested_datasets": list(requested_datasets),
        "requested_seeds": list(requested_seeds),
        "sentinel_mode": sentinel,
        "by_dataset": by_dataset,
        "phase_survival_gate_pass": all(row["pass"] for row in by_dataset.values()),
        "method_training_authorization": False,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen phase-survival gate",
        "",
        str(summary["analysis_scope"]),
        "",
        f"- Sentinel mode: {summary['sentinel_mode']}",
        f"- Phase-survival gate pass: {summary['phase_survival_gate_pass']}",
        "- Method/training authorization: False",
        "",
        "| Dataset | seeds | passing seeds | pass |",
        "|---|---|---|---:|",
    ]
    for dataset, row in summary["by_dataset"].items():
        lines.append(
            f"| {dataset} | {row['seeds']} | {row['passing_seeds']} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "A unit-phase gain must exceed the same-cost stride-preserving control,",
            "remain directionally consistent under both component matchers, and",
            "survive cross-fitted zero-overshoot and pixel-IoU checks.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            )


def _atomic_write_bundle(
    output_dir: Path,
    *,
    target_rows: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    job_summaries: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        _write_jsonl(temporary / OUTPUT_FILES[0], target_rows)
        _write_jsonl(temporary / OUTPUT_FILES[1], image_rows)
        (temporary / OUTPUT_FILES[2]).write_text(
            json.dumps(list(calibration), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_jsonl(temporary / OUTPUT_FILES[3], job_summaries)
        (temporary / OUTPUT_FILES[4]).write_text(
            json.dumps(dict(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temporary / OUTPUT_FILES[5]).write_text(_markdown(summary), encoding="utf-8")
        artifact_paths = [temporary / name for name in OUTPUT_FILES[:6]]
        complete_provenance = {
            **dict(provenance),
            "artifact_sha256": {path.name: sha256_file(path) for path in artifact_paths},
        }
        (temporary / OUTPUT_FILES[6]).write_text(
            json.dumps(complete_provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if {path.name for path in temporary.iterdir()} != set(OUTPUT_FILES):
            raise PhaseSurvivalAuditError("temporary bundle inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_hashes() -> dict[str, str]:
    paths = {
        "tool": Path(__file__).resolve(),
        "phase_intervention": ROOT / "utils" / "phase_intervention.py",
        "cross_fitted_low_fa": ROOT / "utils" / "cross_fitted_low_fa.py",
        "nested_component_grid": ROOT / "utils" / "nested_component_grid.py",
        "metric": ROOT / "utils" / "metric.py",
        "dataset": ROOT / "utils" / "data.py",
        "mshnet": ROOT / "model" / "MSHNet.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.batch_size < 1 or args.num_workers < 0 or args.deepest_stride < 1:
        raise PhaseSurvivalAuditError("batch/worker/stride arguments are invalid")
    requested_datasets = _csv_strings(args.datasets, name="datasets")
    if any(dataset not in DATASET_NAMES for dataset in requested_datasets):
        raise PhaseSurvivalAuditError("datasets contain an unknown clean dataset")
    requested_seeds = _csv_ints(args.seeds, name="seeds")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    hard_core_source = _resolve(args.hard_core_source)
    hard_core_panel = _hard_core_panel(hard_core_source)
    hard_core_ids_by_dataset = {
        dataset: {
            str(row["stable_target_id"])
            for row in hard_core_panel
            if row["dataset"] == dataset
        }
        for dataset in requested_datasets
    }

    batch_dir = ROOT / "repro_runs" / "clean" / args.batch_id
    authoritative, authority_records, registry_order = (
        build_authoritative_registries_before_checkpoints(
            batch_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )
    jobs, batch_provenance = load_validated_jobs(batch_dir, policy="fixed_epoch")
    jobs = [
        job
        for job in jobs
        if str(job["dataset"]) in requested_datasets
        and int(job["seed"]) in requested_seeds
    ]
    expected_pairs = {
        (dataset, seed) for dataset in requested_datasets for seed in requested_seeds
    }
    observed_pairs = {(str(job["dataset"]), int(job["seed"])) for job in jobs}
    if observed_pairs != expected_pairs or len(jobs) != len(expected_pairs):
        raise PhaseSurvivalAuditError("requested dataset/seed grid is not exactly available")

    device = _resolve_device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    q2_quantiles = build_nested_quantile_probability_grids()[-1].probabilities
    source_hashes = _source_hashes()
    git_before = git_worktree_provenance()

    all_target_rows: list[dict[str, Any]] = []
    all_image_rows: list[dict[str, Any]] = []
    all_calibration: list[dict[str, Any]] = []
    job_summaries: list[dict[str, Any]] = []
    checkpoint_records = []
    inference_records = []
    for job in sorted(jobs, key=lambda row: (str(row["dataset"]), int(row["seed"]))):
        dataset = str(job["dataset"])
        seed = int(job["seed"])
        scores, aligned_unit, targets, names, record = collect_phase_predictions(
            job,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            deepest_stride=args.deepest_stride,
            expected_registry=authoritative[dataset],
        )
        variant_oracles = {
            variant: oracle_summary(scores[variant], targets) for variant in VARIANTS
        }
        variant_crossfit = {}
        variant_target_rows: dict[str, list[dict[str, Any]]] = {}
        for variant in VARIANTS:
            target_rows, image_rows, calibration = cross_fit_job(
                scores[variant],
                targets,
                names,
                dataset=dataset,
                seed=seed,
                registry=authoritative[dataset],
                checkpoint=record["checkpoint"],
                tail_quantiles=q2_quantiles,
            )
            for row in target_rows:
                row["schema_version"] = TARGET_SCHEMA
                row["phase_variant"] = variant
            for row in image_rows:
                row["schema_version"] = IMAGE_SCHEMA
                row["phase_variant"] = variant
            for row in calibration:
                row["phase_variant"] = variant
                row["grid_level"] = "Q2"
            if variant == "unit_phase_max":
                _attach_phase_attribution(
                    target_rows,
                    aligned_unit_scores=aligned_unit,
                    targets=targets,
                )
            variant_target_rows[variant] = target_rows
            variant_crossfit[variant] = _crossfit_summary(image_rows)
            all_target_rows.extend(target_rows)
            all_image_rows.extend(image_rows)
            all_calibration.extend(calibration)
        job_summaries.append(
            build_job_summary(
                dataset=dataset,
                seed=seed,
                variant_oracles=variant_oracles,
                variant_crossfit=variant_crossfit,
                variant_target_rows=variant_target_rows,
                hard_core_ids=hard_core_ids_by_dataset[dataset],
            )
        )
        checkpoint_records.append(record["checkpoint"])
        inference_records.append(record["inference"])
        del scores, aligned_unit, targets

    summary = build_summary(
        job_summaries,
        requested_datasets=requested_datasets,
        requested_seeds=requested_seeds,
    )
    if _source_hashes() != source_hashes or git_worktree_provenance() != git_before:
        raise PhaseSurvivalAuditError("source/Git state changed during phase audit")
    protocol = {
        "variants": list(VARIANTS),
        "unit_phase_offsets": [list(offset) for offset in UNIT_PHASE_OFFSETS],
        "lattice_control_offsets": [
            list(offset) for offset in phase_preserving_offsets(args.deepest_stride)
        ],
        "deepest_stride": args.deepest_stride,
        "aggregation": "aligned elementwise maximum with invalid values=-inf",
        "intervention_location": "after canonical resize/normalization",
        "budgets_fa_per_mpix": list(BUDGETS),
        "matchers": list(MATCHERS),
        "grid_level": "Q2",
        "tail_quantiles": [float(value) for value in q2_quantiles],
        "strict_threshold_operator": ">",
        "official_test_policy": "sealed and never opened",
        "sentinel_gate_is_method_authorization": False,
    }
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "git": git_before,
        "source_sha256": source_hashes,
        "batch": batch_provenance,
        "registry_precheckpoint_order": registry_order,
        "authoritative_registry_construction": authority_records,
        "jobs": checkpoint_records,
        "inference": inference_records,
        "hard_core_panel": {
            "source": str(hard_core_source),
            "source_sha256": sha256_file(hard_core_source),
            "records": list(hard_core_panel),
        },
        "protocol": protocol,
        "protocol_sha256": sha256_json(protocol),
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
    _atomic_write_bundle(
        output_dir,
        target_rows=all_target_rows,
        image_rows=all_image_rows,
        calibration=all_calibration,
        job_summaries=job_summaries,
        summary=summary,
        provenance=provenance,
    )
    return summary, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    summary, output_dir = run(parse_args(argv))
    print(
        "phase-survival gate pass: %s; method authorization=%s"
        % (
            summary["phase_survival_gate_pass"],
            summary["method_training_authorization"],
        )
    )
    print(f"wrote immutable phase-survival bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        PhaseSurvivalAuditError,
        PersistenceAuditError,
        FinalizationError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(f"phase-survival audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
