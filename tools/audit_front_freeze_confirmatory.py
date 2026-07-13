#!/usr/bin/env python3
"""Fail-closed confirmatory audit of where formal hard-core evidence disappears.

The audit is read-only with respect to models and datasets.  It evaluates only
the frozen Gate-G Q2/FA20 16-target hard-core panel and one same-dataset,
same-seed area/border-matched successful target for every formal observation.
The successful controls must be matched by both Gate-G component matchers.

The multichannel survival statistic is unsigned.  Consequently, this bundle
can localize an operational distinct/non-distinct transition, but cannot show
that a feature channel points in the target-positive direction or that a stage
caused the final miss.  No official test sample is constructed or iterated.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from skimage import measure
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.MSHNet import MSHNet  # noqa: E402
from model.mshnet_stage_evidence_view import (  # noqa: E402
    forward_mshnet_stage_evidence,
)
from tools.audit_cross_seed_failure_persistence import (  # noqa: E402
    CANONICAL_SIZE,
    DATASET_NAMES,
    _normalize_state_dict,
    build_authoritative_registries_before_checkpoints,
    git_worktree_provenance,
    load_validated_jobs,
    sha256_file,
    sha256_json,
)
from tools.audit_gate_f_nested_grid_sensitivity import _resolve_device  # noqa: E402
from tools.audit_phase_survival_gate import _hard_core_panel  # noqa: E402
from tools.finalize_clean_baselines import load_checkpoint_cpu  # noqa: E402
from utils.data import IRSTD_Dataset  # noqa: E402
from utils.feature_survival import (  # noqa: E402
    build_translation_control_set,
    evaluate_feature_survival,
    project_geometry_controls,
)
from utils.target_identity import (  # noqa: E402
    StableTargetId,
    StableTargetSet,
    assert_same_target_set,
    build_stable_target_set,
    canonical_mask_sha256,
)


SCHEMA = "dea.front_freeze_confirmatory.v1"
OBSERVATION_SCHEMA = "dea.front_freeze_observation.v1"
PROVENANCE_SCHEMA = "dea.front_freeze_provenance.v1"
SEEDS = (20260711, 20260712, 20260713)
GRID_LEVEL = "Q2"
FA_BUDGET = 20
FORMAL_OUTCOME = "formal_hard_core"
CONTROL_OUTCOME = "matched_success_control"
OUTCOMES = (FORMAL_OUTCOME, CONTROL_OUTCOME)
MAIN_PATH = (
    "input",
    "stem",
    "e0",
    "p0",
    "e1",
    "p1",
    "e2",
    "p2",
    "e3",
    "p3",
    "m",
    "j3",
    "d3",
    "j2",
    "d2",
    "j1",
    "d1",
    "j0",
    "d0",
)
REPORT_EDGES = (
    *("%s_to_%s" % edge for edge in zip(MAIN_PATH[:-1], MAIN_PATH[1:])),
    "d0_to_mask0",
    "d0_to_z",
)
OUTPUT_FILES = (
    "observations.jsonl",
    "front_freeze_summary.json",
    "front_freeze_summary.md",
    "provenance.json",
)


class FrontFreezeAuditError(RuntimeError):
    """Raised when the confirmatory scope cannot be preserved exactly."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument(
        "--hard-core-source",
        default=(
            "repro_runs/gate_g/frontier_decomposition_v2/"
            "target_decomposition.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="repro_runs/gate_i/front_freeze_confirmatory_v1",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--required-controls", type=int, default=64)
    parser.add_argument("--max-candidate-controls", type=int, default=256)
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FrontFreezeAuditError(
                        f"{path}:{line_number} is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontFreezeAuditError(f"cannot read {path}: {exc}") from exc
    return rows


def _stable_lookup(
    registry: Mapping[str, StableTargetSet],
) -> dict[str, StableTargetId]:
    result: dict[str, StableTargetId] = {}
    for target_set in registry.values():
        for target in target_set.targets:
            if target.stable_key in result:
                raise FrontFreezeAuditError("authoritative stable target is duplicated")
            result[target.stable_key] = target
    return result


def _border_distance(target: StableTargetId) -> float:
    return float(
        min(
            target.centroid_y,
            target.centroid_x,
            target.height - 1 - target.centroid_y,
            target.width - 1 - target.centroid_x,
        )
    )


def _cohort_record(
    target: StableTargetId,
    *,
    dataset: str,
    seed: int,
    outcome: str,
) -> dict[str, Any]:
    if target.dataset != dataset:
        raise FrontFreezeAuditError("target/dataset mismatch in cohort construction")
    return {
        "dataset": dataset,
        "seed": int(seed),
        "outcome": outcome,
        "stable_target_id": target.stable_key,
        "image_name": target.image_name,
        "component_index": int(target.component_index),
        "source_component_index": int(target.source_component_index),
        "source_label": int(target.source_label),
        "target_area": int(target.area),
        "target_bbox": list(target.bbox),
        "target_centroid": [float(target.centroid_y), float(target.centroid_x)],
        "border_distance": _border_distance(target),
        "component_mask_sha256": target.component_mask_sha256,
        "label_mask_sha256": target.label_mask_sha256,
    }


def _match_distance(
    formal: Mapping[str, Any], control: Mapping[str, Any]
) -> tuple[float, float, float]:
    area = abs(
        math.log1p(float(control["target_area"]))
        - math.log1p(float(formal["target_area"]))
    )
    border = abs(
        float(control["border_distance"])
        - float(formal["border_distance"])
    ) / CANONICAL_SIZE
    return float(area + 0.1 * border), float(area), float(border)


def select_success_controls(
    formal: Sequence[Mapping[str, Any]],
    successful: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Choose one unique same-job area/border-matched success per formal target."""

    if not formal:
        raise FrontFreezeAuditError("a job has no formal observations")
    job_keys = {
        (str(row["dataset"]), int(row["seed"]))
        for row in (*formal, *successful)
    }
    if len(job_keys) != 1:
        raise FrontFreezeAuditError("control matching crossed dataset/seed jobs")
    formal_ids = {str(row["stable_target_id"]) for row in formal}
    available = [
        dict(row)
        for row in successful
        if str(row["stable_target_id"]) not in formal_ids
    ]
    if len(available) < len(formal):
        raise FrontFreezeAuditError("not enough unique successful target controls")
    selected: list[dict[str, Any]] = []
    for miss in sorted(
        formal,
        key=lambda row: (str(row["image_name"]), str(row["stable_target_id"])),
    ):
        ranked = []
        for candidate in available:
            distance, area_distance, border_distance = _match_distance(miss, candidate)
            ranked.append(
                (
                    distance,
                    str(candidate["image_name"]),
                    str(candidate["stable_target_id"]),
                    area_distance,
                    border_distance,
                    candidate,
                )
            )
        chosen = min(ranked)
        candidate = chosen[-1]
        available.remove(candidate)
        annotated = dict(candidate)
        annotated.update(
            {
                "paired_formal_target_id": str(miss["stable_target_id"]),
                "paired_formal_image_name": str(miss["image_name"]),
                "pair_distance": float(chosen[0]),
                "pair_log_area_distance": float(chosen[3]),
                "pair_normalized_border_distance": float(chosen[4]),
            }
        )
        selected.append(annotated)
    return selected


def _validate_gate_g_rows(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    panel: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    authoritative: Mapping[str, Mapping[str, StableTargetSet]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    panel_ids = {
        (str(row["dataset"]), str(row["stable_target_id"])) for row in panel
    }
    if len(panel_ids) != 16:
        raise FrontFreezeAuditError("formal panel identities are not exactly 16")
    job_lookup = {
        (str(job["dataset"]), int(job["seed"])): job for job in jobs
    }
    expected_job_keys = {
        (dataset, seed) for dataset in DATASET_NAMES for seed in SEEDS
    }
    if set(job_lookup) != expected_job_keys or len(jobs) != 9:
        raise FrontFreezeAuditError("clean fixed-epoch job grid is not exactly 3x3")

    filtered: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for raw in source_rows:
        if (
            raw.get("grid_level") == GRID_LEVEL
            and int(raw.get("nominal_budget_fa_per_mpix", -1)) == FA_BUDGET
        ):
            dataset = str(raw.get("dataset"))
            seed = int(raw.get("seed", -1))
            key = (dataset, seed)
            if key not in expected_job_keys:
                raise FrontFreezeAuditError("Gate-G row has an unexpected job key")
            filtered[key].append(dict(raw))

    for key, job in job_lookup.items():
        dataset, seed = key
        rows = filtered.get(key, [])
        target_lookup = _stable_lookup(authoritative[dataset])
        if len(rows) != len(target_lookup):
            raise FrontFreezeAuditError(
                f"Gate-G target coverage drifted for {dataset}/seed{seed}"
            )
        row_lookup: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            stable_id = str(row.get("stable_target_id"))
            if stable_id in row_lookup or stable_id not in target_lookup:
                raise FrontFreezeAuditError("Gate-G target identity is invalid/duplicated")
            row_lookup[stable_id] = row
            target = target_lookup[stable_id]
            checkpoint = row.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise FrontFreezeAuditError("Gate-G row lacks checkpoint provenance")
            checkpoint_assertions = {
                "job_id": job["job_id"],
                "path": str(Path(str(job["checkpoint"])).resolve()),
                "sha256": job["checkpoint_sha256"],
                "epoch": int(job["checkpoint_summary"]["epoch"]),
                "validation_split_sha256": job["split_hashes"]["validation"],
            }
            for field, expected in checkpoint_assertions.items():
                observed = checkpoint.get(field)
                if field == "path":
                    observed = str(Path(str(observed)).resolve())
                if observed != expected:
                    raise FrontFreezeAuditError(
                        f"Gate-G checkpoint {field} drifted for {dataset}/seed{seed}"
                    )
            identity_assertions = {
                "dataset": dataset,
                "image_name": target.image_name,
                "target_area": int(target.area),
                "component_mask_sha256": target.component_mask_sha256,
                "label_mask_sha256": target.label_mask_sha256,
            }
            for field, expected in identity_assertions.items():
                if row.get(field) != expected:
                    raise FrontFreezeAuditError(
                        f"Gate-G identity assertion {field} drifted"
                    )
        if set(row_lookup) != set(target_lookup):
            raise FrontFreezeAuditError("Gate-G identities do not cover authority")
        for panel_dataset, stable_id in panel_ids:
            if panel_dataset != dataset:
                continue
            row = row_lookup[stable_id]
            if (
                row.get("category_core") != "no_feasible_local_peak_activation"
                or row.get("selected_hungarian_matched") is not False
                or row.get("selected_legacy_matched") is not False
            ):
                raise FrontFreezeAuditError("formal hard-core outcome drifted")
    if set(filtered) != expected_job_keys:
        raise FrontFreezeAuditError("Gate-G Q2/FA20 source lacks a clean job")
    return dict(filtered)


def _build_cohorts(
    *,
    panel: Sequence[Mapping[str, Any]],
    gate_rows: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    authoritative: Mapping[str, Mapping[str, StableTargetSet]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    panel_by_dataset = {
        dataset: {
            str(row["stable_target_id"])
            for row in panel
            if str(row["dataset"]) == dataset
        }
        for dataset in DATASET_NAMES
    }
    result: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for dataset in DATASET_NAMES:
        targets = _stable_lookup(authoritative[dataset])
        for seed in SEEDS:
            formal = [
                _cohort_record(
                    targets[stable_id],
                    dataset=dataset,
                    seed=seed,
                    outcome=FORMAL_OUTCOME,
                )
                for stable_id in sorted(panel_by_dataset[dataset])
            ]
            successful = []
            for row in gate_rows[(dataset, seed)]:
                if (
                    row.get("selected_hungarian_matched") is True
                    and row.get("selected_legacy_matched") is True
                ):
                    successful.append(
                        _cohort_record(
                            targets[str(row["stable_target_id"])],
                            dataset=dataset,
                            seed=seed,
                            outcome=CONTROL_OUTCOME,
                        )
                    )
            controls = select_success_controls(formal, successful)
            result[(dataset, seed)] = formal + controls
    formal_count = sum(
        row["outcome"] == FORMAL_OUTCOME
        for rows in result.values()
        for row in rows
    )
    control_count = sum(
        row["outcome"] == CONTROL_OUTCOME
        for rows in result.values()
        for row in rows
    )
    if formal_count != 48 or control_count != 48:
        raise FrontFreezeAuditError(
            f"confirmatory cohort must be 48+48, got {formal_count}+{control_count}"
        )
    return result


def _transition(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if not bool(left.get("available")) or not bool(right.get("available")):
        return "undefined_endpoint"
    left_distinct = left.get("state") == "distinct"
    right_distinct = right.get("state") == "distinct"
    if left_distinct and not right_distinct:
        return "distinct_drop"
    if not left_distinct and right_distinct:
        return "distinct_recovery"
    return "distinct_retained" if left_distinct else "non_distinct_retained"


def build_transition_ledger(stages: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    missing = [stage for stage in (*MAIN_PATH, "mask0", "z") if stage not in stages]
    if missing:
        raise FrontFreezeAuditError("stage ledger lacks: " + ", ".join(missing))
    edges = {
        "%s_to_%s" % (left, right): _transition(stages[left], stages[right])
        for left, right in zip(MAIN_PATH[:-1], MAIN_PATH[1:])
    }
    edges["d0_to_mask0"] = _transition(stages["d0"], stages["mask0"])
    edges["d0_to_z"] = _transition(stages["d0"], stages["z"])
    if tuple(edges) != REPORT_EDGES:
        raise FrontFreezeAuditError("transition edge order drifted")
    return edges


def _audit_job(
    job: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    *,
    authority: Mapping[str, StableTargetSet],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    required_controls: int,
    max_candidate_controls: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_name = str(job["dataset"])
    seed = int(job["seed"])
    if {
        (str(row["dataset"]), int(row["seed"])) for row in cohort
    } != {(dataset_name, seed)}:
        raise FrontFreezeAuditError("cohort/job mismatch")
    dataset = IRSTD_Dataset(Namespace(**job["stored_args"]), mode="val")
    if dataset.split_sha256 != job["split_hashes"]["validation"]:
        raise FrontFreezeAuditError("validation split hash drifted")
    if dataset.base_size != CANONICAL_SIZE or dataset.crop_size != CANONICAL_SIZE:
        raise FrontFreezeAuditError("audit requires canonical 256x256 development input")
    if tuple(dataset.names) != tuple(authority):
        raise FrontFreezeAuditError("development image universe drifted")

    checkpoint_path = Path(str(job["checkpoint"])).resolve()
    if checkpoint_path.name != "checkpoint.pkl":
        raise FrontFreezeAuditError("confirmatory audit requires fixed checkpoint.pkl")
    if sha256_file(checkpoint_path) != job["checkpoint_sha256"]:
        raise FrontFreezeAuditError("checkpoint hash drifted")
    checkpoint = load_checkpoint_cpu(checkpoint_path)
    state = checkpoint.get("net")
    if not isinstance(state, Mapping) or not state:
        raise FrontFreezeAuditError("checkpoint has no network state")
    model = MSHNet(3)
    model.load_state_dict(_normalize_state_dict(state), strict=True)
    del checkpoint, state
    model.requires_grad_(False).to(device).eval()

    image_to_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cohort:
        image_to_rows[str(row["image_name"])].append(row)
    name_to_index = {name: index for index, name in enumerate(dataset.names)}
    if any(name not in name_to_index for name in image_to_rows):
        raise FrontFreezeAuditError("selected image is outside development split")
    indices = sorted(name_to_index[name] for name in image_to_rows)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    observations: list[dict[str, Any]] = []
    reconstruction_max_abs_error = 0.0
    seen_images: list[str] = []
    with torch.inference_mode():
        cursor = 0
        for images, targets in loader:
            if images.shape[0] != targets.shape[0]:
                raise FrontFreezeAuditError("image/target batch size mismatch")
            images_device = images.to(device, non_blocking=True)
            evidence = forward_mshnet_stage_evidence(
                model, images_device, detach=True
            )
            reconstruction_error = float(
                (evidence["z_reconstructed"] - evidence["pred"]).abs().max().item()
            )
            reconstruction_max_abs_error = max(
                reconstruction_max_abs_error, reconstruction_error
            )
            if not torch.allclose(
                evidence["z_reconstructed"],
                evidence["pred"],
                atol=1e-6,
                rtol=1e-5,
            ):
                raise FrontFreezeAuditError("exact final fusion reconstruction failed")
            for batch_index in range(int(images.shape[0])):
                dataset_index = indices[cursor + batch_index]
                image_name = dataset.names[dataset_index]
                seen_images.append(image_name)
                target_mask = (
                    targets[batch_index, 0].detach().cpu().numpy() > 0.5
                )
                observed = build_stable_target_set(
                    target_mask,
                    dataset=dataset_name,
                    image_name=image_name,
                    connectivity=2,
                )
                try:
                    assert_same_target_set(authority[image_name], observed)
                except Exception as exc:
                    raise FrontFreezeAuditError(
                        f"development target authority drifted: {exc}"
                    ) from exc
                labels = measure.label(target_mask, connectivity=2)
                stage_arrays = {
                    stage: tensor[batch_index].detach().float().cpu().numpy()
                    for stage, tensor in evidence["path"].items()
                }
                stage_arrays["mask0"] = (
                    evidence["native_sides"]["mask0"][batch_index]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                stage_arrays["z"] = (
                    evidence["pred"][batch_index]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                for raw_record in image_to_rows[image_name]:
                    record = dict(raw_record)
                    source_label = int(record["source_label"])
                    component = labels == source_label
                    if not component.any():
                        raise FrontFreezeAuditError("selected target component is empty")
                    if canonical_mask_sha256(component) != record[
                        "component_mask_sha256"
                    ]:
                        raise FrontFreezeAuditError("selected component digest drifted")
                    controls = build_translation_control_set(
                        component,
                        target_mask,
                        sample_key=(
                            f"{SCHEMA}:{dataset_name}:{seed}:"
                            f"{record['stable_target_id']}"
                        ),
                        max_candidate_controls=max_candidate_controls,
                    )
                    geometry_by_shape: dict[
                        tuple[int, int], Any
                    ] = {}
                    stage_results: dict[str, dict[str, Any]] = {}
                    for stage, feature in stage_arrays.items():
                        shape = tuple(int(value) for value in feature.shape[-2:])
                        if shape not in geometry_by_shape:
                            geometry_by_shape[shape] = project_geometry_controls(
                                controls,
                                shape,
                                required_controls=required_controls,
                            )
                        stage_results[stage] = evaluate_feature_survival(
                            feature,
                            geometry_by_shape[shape],
                            scalar_threshold=(0.0 if stage in {"mask0", "z"} else None),
                        ).as_dict()
                    record.update(
                        {
                            "schema": OBSERVATION_SCHEMA,
                            "checkpoint_job_id": job["job_id"],
                            "checkpoint_sha256": job["checkpoint_sha256"],
                            "validation_split_sha256": dataset.split_sha256,
                            "geometry_candidate_count": len(
                                controls.translated_masks
                            ),
                            "stages": stage_results,
                            "transitions": build_transition_ledger(stage_results),
                        }
                    )
                    observations.append(record)
            cursor += int(images.shape[0])
    if cursor != len(indices) or seen_images != [dataset.names[index] for index in indices]:
        raise FrontFreezeAuditError("selected development image accounting drifted")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    expected = Counter(str(row["stable_target_id"]) for row in cohort)
    observed = Counter(str(row["stable_target_id"]) for row in observations)
    if observed != expected or len(observations) != len(cohort):
        raise FrontFreezeAuditError("job observation coverage drifted")
    return observations, {
        "dataset": dataset_name,
        "seed": seed,
        "job_id": job["job_id"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": job["checkpoint_sha256"],
        "checkpoint_epoch": int(job["checkpoint_summary"]["epoch"]),
        "validation_split_sha256": dataset.split_sha256,
        "official_test_iterated": False,
        "selected_image_count": len(indices),
        "formal_observation_count": sum(
            row["outcome"] == FORMAL_OUTCOME for row in observations
        ),
        "successful_control_count": sum(
            row["outcome"] == CONTROL_OUTCOME for row in observations
        ),
        "fusion_reconstruction_max_abs_error": reconstruction_max_abs_error,
    }


def _stage_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in (*MAIN_PATH, "mask0", "z"):
        values = [row["stages"][stage] for row in records]
        available = [value for value in values if bool(value["available"])]
        counts = Counter(str(value["state"]) for value in values)
        directional = [
            float(value["directional_auc"])
            for value in available
            if value["directional_auc"] is not None
        ]
        result[stage] = {
            "observations": len(values),
            "available": len(available),
            "unavailable": len(values) - len(available),
            "state_counts": {
                state: int(counts.get(state, 0))
                for state in ("distinct", "uncertain", "background_like", "undefined")
            },
            "distinct_rate_all": (
                counts.get("distinct", 0) / len(values) if values else None
            ),
            "distinct_rate_available": (
                counts.get("distinct", 0) / len(available) if available else None
            ),
            "median_rank": (
                float(np.median([value["rank"] for value in available]))
                if available
                else None
            ),
            "median_robust_effect": (
                float(np.median([value["robust_effect"] for value in available]))
                if available
                else None
            ),
            "median_directional_auc": (
                float(np.median(directional)) if directional else None
            ),
        }
    return result


def _transition_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    states = (
        "distinct_drop",
        "distinct_recovery",
        "distinct_retained",
        "non_distinct_retained",
        "undefined_endpoint",
    )
    for edge in REPORT_EDGES:
        counts = Counter(str(row["transitions"][edge]) for row in records)
        if sum(counts.values()) != len(records) or any(key not in states for key in counts):
            raise FrontFreezeAuditError("transition summary encountered invalid state")
        result[edge] = {
            "observations": len(records),
            "counts": {state: int(counts.get(state, 0)) for state in states},
            "distinct_drop_rate_all": (
                counts.get("distinct_drop", 0) / len(records) if records else None
            ),
            "distinct_recovery_rate_all": (
                counts.get("distinct_recovery", 0) / len(records) if records else None
            ),
        }
    return result


def _scope_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_outcome: dict[str, Any] = {}
    for outcome in OUTCOMES:
        selected = [row for row in records if row["outcome"] == outcome]
        by_outcome[outcome] = {
            "observation_count": len(selected),
            "unique_target_count": len(
                {str(row["stable_target_id"]) for row in selected}
            ),
            "stage_summary": _stage_summary(selected),
            "transition_summary": _transition_summary(selected),
        }
    return by_outcome


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != 96:
        raise FrontFreezeAuditError("pooled bundle must contain 96 observations")
    formal = [row for row in records if row["outcome"] == FORMAL_OUTCOME]
    controls = [row for row in records if row["outcome"] == CONTROL_OUTCOME]
    if len(formal) != 48 or len(controls) != 48:
        raise FrontFreezeAuditError("pooled cohort must contain 48 formal + 48 controls")
    formal_per_target = Counter(str(row["stable_target_id"]) for row in formal)
    if len(formal_per_target) != 16 or set(formal_per_target.values()) != {3}:
        raise FrontFreezeAuditError("formal targets do not have exact three-seed coverage")
    paired_ids = Counter(str(row["paired_formal_target_id"]) for row in controls)
    if paired_ids != formal_per_target:
        raise FrontFreezeAuditError("successful control pairing does not cover formal panel")

    by_dataset = {}
    for dataset in DATASET_NAMES:
        selected = [row for row in records if row["dataset"] == dataset]
        by_dataset[dataset] = _scope_summary(selected)
    by_seed = {}
    for seed in SEEDS:
        selected = [row for row in records if int(row["seed"]) == seed]
        by_seed[str(seed)] = _scope_summary(selected)
    formal_stage = _stage_summary(formal)
    formal_transition = _transition_summary(formal)
    control_stage = _stage_summary(controls)
    control_transition = _transition_summary(controls)
    d0_consensus = Counter()
    for stable_id in formal_per_target:
        distinct_seeds = sum(
            row["stages"]["d0"]["state"] == "distinct"
            for row in formal
            if str(row["stable_target_id"]) == stable_id
        )
        d0_consensus[distinct_seeds] += 1
    dataset_unique_counts = {
        dataset: len(
            {
                str(row["stable_target_id"])
                for row in formal
                if row["dataset"] == dataset
            }
        )
        for dataset in DATASET_NAMES
    }
    dataset_observation_counts = Counter(str(row["dataset"]) for row in formal)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_boundary": {
            "confirmatory_scope": (
                "frozen Gate-G Q2/FA20 16-target hard-core panel, three fixed "
                "seeds, and predeclared same-job area/border target controls"
            ),
            "survival_statistic": (
                "unsigned multichannel target-vs-translated-background contrast rank"
            ),
            "causal_claim_authorized": False,
            "performance_claim_authorized": False,
            "front_freeze_claim_authorized": (
                "only as an engineering routing decision; not as a paper result"
            ),
            "limitations": [
                "the same clean development split supplied Gate-G phenotypes and this diagnostic",
                "three seeds are repeated measurements of 16 targets, not 48 independent targets",
                "geometry translations are not guaranteed to match target context",
                "multichannel contrast magnitude does not identify target-positive direction",
                "distinct drop/recovery is operational state change, not stage causality",
            ],
        },
        "scope": {
            "grid_level": GRID_LEVEL,
            "nominal_budget_fa_per_mpix": FA_BUDGET,
            "formal_target_count": 16,
            "formal_observation_count": 48,
            "matched_success_control_count": 48,
            "seeds": list(SEEDS),
            "dataset_formal_unique_target_counts": dataset_unique_counts,
            "dataset_formal_observation_counts": dict(
                sorted(dataset_observation_counts.items())
            ),
        },
        "engineering_routing": {
            "recommendation": "freeze_input_through_d0_for_next_structural_iteration",
            "recommended_first_mutable_boundary": "after_d0_prediction_conversion",
            "post_hoc_engineering_decision_only": True,
            "not_a_scientific_gate": True,
            "evidence": {
                "formal_d0_distinct_observations": formal_stage["d0"][
                    "state_counts"
                ]["distinct"],
                "formal_mask0_distinct_observations": formal_stage["mask0"][
                    "state_counts"
                ]["distinct"],
                "formal_z_distinct_observations": formal_stage["z"][
                    "state_counts"
                ]["distinct"],
                "formal_d0_to_mask0_distinct_drops": formal_transition[
                    "d0_to_mask0"
                ]["counts"]["distinct_drop"],
                "formal_d0_to_z_distinct_drops": formal_transition["d0_to_z"][
                    "counts"
                ]["distinct_drop"],
                "successful_control_d0_distinct_observations": control_stage["d0"][
                    "state_counts"
                ]["distinct"],
                "successful_control_mask0_distinct_observations": control_stage[
                    "mask0"
                ]["state_counts"]["distinct"],
                "successful_control_z_distinct_observations": control_stage["z"][
                    "state_counts"
                ]["distinct"],
                "successful_control_d0_to_mask0_distinct_drops": control_transition[
                    "d0_to_mask0"
                ]["counts"]["distinct_drop"],
                "successful_control_d0_to_z_distinct_drops": control_transition[
                    "d0_to_z"
                ]["counts"]["distinct_drop"],
                "formal_e0_to_p0_distinct_drops": formal_transition["e0_to_p0"][
                    "counts"
                ]["distinct_drop"],
                "formal_targets_by_d0_distinct_seed_count": {
                    str(seed_count): int(d0_consensus.get(seed_count, 0))
                    for seed_count in range(4)
                },
            },
            "exception": (
                "one formal target is non-distinct at d0 in all three seeds; "
                "the recommendation prioritizes the dominant conversion failure and "
                "does not prove that an output-only redesign can rescue every target"
            ),
        },
        "pooled": _scope_summary(records),
        "by_dataset": by_dataset,
        "by_seed": by_seed,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    formal = summary["pooled"][FORMAL_OUTCOME]
    control = summary["pooled"][CONTROL_OUTCOME]

    def stage_line(stage: str) -> str:
        left = formal["stage_summary"][stage]
        right = control["stage_summary"][stage]
        return (
            f"| {stage} | {left['available']}/{left['observations']} | "
            f"{left['state_counts']['distinct']} | {left['median_rank']} | "
            f"{right['available']}/{right['observations']} | "
            f"{right['state_counts']['distinct']} | {right['median_rank']} |"
        )

    def edge_line(edge: str) -> str:
        left = formal["transition_summary"][edge]["counts"]
        right = control["transition_summary"][edge]["counts"]
        return (
            f"| {edge} | {left['distinct_drop']} | {left['distinct_recovery']} | "
            f"{left['undefined_endpoint']} | {right['distinct_drop']} | "
            f"{right['distinct_recovery']} | {right['undefined_endpoint']} |"
        )

    lines = [
        "# Gate I: front-freeze confirmatory feature-survival audit",
        "",
        "## Scope and boundary",
        "",
        "- Formal scope: Gate-G Q2/FA20 hard-core panel, 16 targets × 3 fixed seeds = 48 observations.",
        "- Control scope: one unique same-dataset/same-seed area/border-matched target per formal observation; both Hungarian and legacy Gate-G matching had to succeed.",
        "- Dataset routing correction: the panel is IRSTD-1K 11 + NUDT-SIRST 4 + NUAA-SIRST 1, so the audit used each target's own dataset checkpoint (9 checkpoints total). Applying IRSTD checkpoints across datasets was explicitly rejected.",
        "- No official test dataset was constructed or iterated.",
        "- The survival statistic is unsigned and development-only. It does not establish target-positive direction, causal responsibility, generalization, or performance improvement.",
        "",
        "## Stage survival",
        "",
        "| Stage | Formal available | Formal distinct | Formal median rank | Success available | Success distinct | Success median rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(stage_line(stage) for stage in (*MAIN_PATH, "mask0", "z"))
    lines.extend(
        [
            "",
            "## Distinct-state drops and recoveries",
            "",
            "| Edge | Formal drop | Formal recovery | Formal undefined | Success drop | Success recovery | Success undefined |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(edge_line(edge) for edge in REPORT_EDGES)
    routing = summary["engineering_routing"]
    evidence = routing["evidence"]
    lines.extend(
        [
            "",
            "The e0→p0 row isolates the first pooling boundary. The complete encoder/decoder path and d0→mask0/z rows are reported without post-hoc stage selection.",
            "",
            "## Engineering routing decision",
            "",
            "For the next structural iteration, freeze the existing input→d0 graph and make the first mutable boundary the prediction conversion after d0. This is a post-hoc engineering routing decision, not a scientific gate or paper claim.",
            "",
            f"- Formal distinct counts: d0 {evidence['formal_d0_distinct_observations']}/48, mask0 {evidence['formal_mask0_distinct_observations']}/48, z {evidence['formal_z_distinct_observations']}/48.",
            f"- Formal drops: e0→p0 {evidence['formal_e0_to_p0_distinct_drops']}; d0→mask0 {evidence['formal_d0_to_mask0_distinct_drops']}; d0→z {evidence['formal_d0_to_z_distinct_drops']}.",
            f"- Successful-control distinct counts: d0/mask0/z = {evidence['successful_control_d0_distinct_observations']}/{evidence['successful_control_mask0_distinct_observations']}/{evidence['successful_control_z_distinct_observations']} of 48, with d0→mask0/z drops = {evidence['successful_control_d0_to_mask0_distinct_drops']}/{evidence['successful_control_d0_to_z_distinct_drops']}.",
            "- Exception: one formal target is non-distinct at d0 in all three seeds, so this audit does not prove that an output-only redesign can recover every target.",
            "",
        ]
    )
    return "\n".join(lines)


def _source_hashes(hard_core_source: Path) -> dict[str, str]:
    paths = {
        "tool": Path(__file__).resolve(),
        "mshnet": ROOT / "model" / "MSHNet.py",
        "stage_evidence": ROOT / "model" / "mshnet_stage_evidence_view.py",
        "evidence": ROOT / "model" / "mshnet_evidence_view.py",
        "feature_survival": ROOT / "utils" / "feature_survival.py",
        "target_identity": ROOT / "utils" / "target_identity.py",
        "dataset": ROOT / "utils" / "data.py",
        "hard_core_source": hard_core_source,
    }
    for path in paths.values():
        if not path.is_file():
            raise FrontFreezeAuditError(f"missing frozen source: {path}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _write_bundle(
    output_dir: Path,
    *,
    observations: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        with (temporary / OUTPUT_FILES[0]).open("w", encoding="utf-8") as handle:
            for row in observations:
                handle.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
        (temporary / OUTPUT_FILES[1]).write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        (temporary / OUTPUT_FILES[2]).write_text(
            _markdown(summary), encoding="utf-8"
        )
        artifact_hashes = {
            name: sha256_file(temporary / name) for name in OUTPUT_FILES[:3]
        }
        complete_provenance = {
            **dict(provenance),
            "artifact_sha256": artifact_hashes,
        }
        (temporary / OUTPUT_FILES[3]).write_text(
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
            raise FrontFreezeAuditError("temporary bundle inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if (
        args.batch_size < 1
        or args.num_workers < 0
        or args.required_controls < 1
        or args.max_candidate_controls < args.required_controls
    ):
        raise FrontFreezeAuditError("invalid batch/worker/control arguments")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    hard_core_source = _resolve(args.hard_core_source)
    panel = _hard_core_panel(hard_core_source)
    if {int(seed) for row in panel for seed in row["source_seeds"]} != set(SEEDS):
        raise FrontFreezeAuditError("formal panel seed scope drifted")
    dataset_counts = Counter(str(row["dataset"]) for row in panel)
    if dataset_counts != Counter({"IRSTD-1K": 11, "NUDT-SIRST": 4, "NUAA-SIRST": 1}):
        raise FrontFreezeAuditError("formal panel dataset composition drifted")

    source_hashes = _source_hashes(hard_core_source)
    git_before = git_worktree_provenance()
    batch_dir = ROOT / "repro_runs" / "clean" / args.batch_id
    authoritative, authority_records, registry_order = (
        build_authoritative_registries_before_checkpoints(
            batch_dir,
            batch_size=max(1, args.batch_size),
            num_workers=args.num_workers,
        )
    )
    jobs, batch_provenance = load_validated_jobs(batch_dir, policy="fixed_epoch")
    if len(jobs) != 9:
        raise FrontFreezeAuditError("validated fixed checkpoint grid is not exactly nine")
    gate_rows = _validate_gate_g_rows(
        _jsonl(hard_core_source),
        panel=panel,
        jobs=jobs,
        authoritative=authoritative,
    )
    cohorts = _build_cohorts(
        panel=panel,
        gate_rows=gate_rows,
        authoritative=authoritative,
    )
    device = _resolve_device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    observations: list[dict[str, Any]] = []
    inference_records = []
    for job in sorted(jobs, key=lambda row: (str(row["dataset"]), int(row["seed"]))):
        key = (str(job["dataset"]), int(job["seed"]))
        job_rows, inference = _audit_job(
            job,
            cohorts[key],
            authority=authoritative[key[0]],
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            required_controls=args.required_controls,
            max_candidate_controls=args.max_candidate_controls,
        )
        observations.extend(job_rows)
        inference_records.append(inference)

    observations.sort(
        key=lambda row: (
            str(row["dataset"]),
            int(row["seed"]),
            0 if row["outcome"] == FORMAL_OUTCOME else 1,
            str(row["image_name"]),
            str(row["stable_target_id"]),
        )
    )
    summary = _summarize(observations)
    if _source_hashes(hard_core_source) != source_hashes:
        raise FrontFreezeAuditError("audit source changed during execution")
    git_after_inference = git_worktree_provenance()
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "explicit_scope_correction": {
            "incorrect_initial_assumption": (
                "three IRSTD-1K checkpoints could evaluate all 16 formal targets"
            ),
            "observed_panel_composition": dict(sorted(dataset_counts.items())),
            "correction": (
                "route every target to its own dataset's clean fixed checkpoint; "
                "3 datasets x 3 seeds = 9 checkpoints and 16 x 3 = 48 formal observations"
            ),
            "cross_dataset_checkpoint_application_rejected": True,
        },
        "data_access": {
            "split": "clean_internal_validation_only",
            "official_test_dataset_constructed": False,
            "official_test_sample_iterated": False,
            "prediction_writes": False,
            "checkpoint_writes": False,
        },
        "hard_core_panel": {
            "source": str(hard_core_source),
            "source_sha256": sha256_file(hard_core_source),
            "records": list(panel),
        },
        "batch": batch_provenance,
        "authoritative_registry_construction": authority_records,
        "registry_order": registry_order,
        "source_sha256": source_hashes,
        "git": {
            "before_inference": git_before,
            "after_inference": git_after_inference,
            "unrelated_concurrent_worktree_change_observed": (
                git_after_inference != git_before
            ),
            "source_specific_hashes_unchanged": True,
            "policy": (
                "concurrent unrelated worktree edits are recorded; every audit "
                "source listed in source_sha256 must remain byte-identical"
            ),
        },
        "device": str(device),
        "required_geometry_controls": args.required_controls,
        "max_candidate_geometry_controls": args.max_candidate_controls,
        "control_selection": {
            "source": "same Gate-G Q2/FA20 dataset/seed job",
            "success_requirement": (
                "selected_hungarian_matched=true and selected_legacy_matched=true"
            ),
            "matching_distance": (
                "abs(log1p(area_control)-log1p(area_formal)) + "
                "0.1*abs(border_control-border_formal)/256"
            ),
            "without_replacement_within_job": True,
        },
        "inference": inference_records,
        "observation_sha256": sha256_json(observations),
    }
    _write_bundle(
        output_dir,
        observations=observations,
        summary=summary,
        provenance=provenance,
    )
    return summary, output_dir


def main() -> None:
    try:
        summary, output_dir = run(parse_args())
    except Exception as exc:
        print(f"front-freeze audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    formal = summary["pooled"][FORMAL_OUTCOME]
    e0 = formal["stage_summary"]["e0"]
    p0 = formal["stage_summary"]["p0"]
    edge = formal["transition_summary"]["e0_to_p0"]["counts"]
    print(
        "formal e0 distinct=%d/%d; p0 distinct=%d/%d; e0->p0 drops=%d; wrote %s"
        % (
            e0["state_counts"]["distinct"],
            e0["observations"],
            p0["state_counts"]["distinct"],
            p0["observations"],
            edge["distinct_drop"],
            output_dir,
        )
    )


if __name__ == "__main__":
    main()
