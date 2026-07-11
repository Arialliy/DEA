#!/usr/bin/env python3
"""Export a frozen MSHNet mean-anchor mechanism audit (no training)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

# Keep direct ``python tools/export_clean_mechanism_audit.py`` invocation valid.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.dea_scale_interaction_exchange import ScaleInteractionExchangeMSHNet
from tools.finalize_clean_baselines import (
    EXPECTED_EPOCHS,
    OUTPUT_JSON as BASELINE_SUMMARY_JSON,
    DATASET_NAMES,
    FinalizationError,
    load_checkpoint_cpu,
    parse_metrics,
    read_json,
    require_mapping,
    validate_checkpoint,
    validate_manifest,
    validate_result,
)
from utils.component_evidence import candidate_label_map, generate_prediction_candidates
from utils.data import IRSTD_Dataset
from utils.metric import match_connected_components


SCHEMA = "dea.clean_mechanism_audit.v1"
ROLE_FILES = {
    "latest": "checkpoint.pkl",
    "best_iou": "checkpoint_best_iou.pkl",
    "pd_fa_best": "checkpoint_pd_fa_best.pkl",
}
PROB, LOGIT, CONNECTIVITY, MAX_DISTANCE = 0.5, 0.0, 2, 3.0
FIXED_EPS = 1e-6
FIXED_CANDIDATE_THRESHOLDS = (0.5, 0.3, 0.2, 0.1)
SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
AUDIT_SOURCE_FILES = {
    "exporter": ROOT / "tools" / "export_clean_mechanism_audit.py",
    "baseline_finalizer": ROOT / "tools" / "finalize_clean_baselines.py",
    "mshnet": ROOT / "model" / "MSHNet.py",
    "mean_anchor_probe": ROOT / "model" / "dea_scale_interaction_exchange.py",
    "component_candidates": ROOT / "utils" / "component_evidence.py",
    "dataset": ROOT / "utils" / "data.py",
    "metrics": ROOT / "utils" / "metric.py",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-role", choices=ROLE_FILES, default="best_iou")
    p.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-dir", default="datasets/NUAA-SIRST")
    p.add_argument("--mode", choices=("val",), default="val")
    p.add_argument("--train-split-file", default="")
    p.add_argument("--val-split-file", default="")
    p.add_argument("--test-split-file", default="")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=20260711)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--base-size", type=int, default=256)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--input-channels", type=int, default=3)
    p.add_argument("--eps", type=float, default=FIXED_EPS)
    p.add_argument(
        "--candidate-thresholds", type=float, nargs="+",
        default=FIXED_CANDIDATE_THRESHOLDS,
    )
    p.add_argument("--device", default="auto")
    return p.parse_args()


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(path: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    # The formal baseline finalizer already provides a weights-only loader with
    # the narrow NumPy scalar allowlist required by these local checkpoints.
    # Do not deserialize an untrusted replacement pickle before provenance is
    # checked.
    ckpt = load_checkpoint_cpu(path)
    if not isinstance(ckpt, dict) or "net" not in ckpt or "method_meta" not in ckpt:
        raise RuntimeError("use metadata-bearing checkpoint*.pkl, never raw weight.pkl")
    if "epoch" not in ckpt or not isinstance(ckpt["method_meta"], dict):
        raise RuntimeError("checkpoint epoch/method_meta missing")
    state = ckpt["net"]
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise RuntimeError("checkpoint net is not a state dict")
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state, ckpt


def validate_checkpoint_metadata(
    checkpoint: dict[str, Any], *, seed: int, val_hash: str, role: str, split_hash: str
) -> dict[str, Any]:
    meta = checkpoint.get("method_meta")
    if not isinstance(meta, dict):
        raise RuntimeError("checkpoint method_meta missing")
    expected = {
        "method": "MSHNet", "model_type": "mshnet", "seed": seed,
        "val_split_sha256": val_hash,
    }
    if role == "test":
        expected["test_split_sha256"] = split_hash
    errors = [
        f"{k}: checkpoint={meta.get(k, '<missing>')!r} expected={v!r}"
        for k, v in expected.items() if meta.get(k) != v
    ]
    if errors:
        raise RuntimeError("checkpoint audit identity mismatch: " + "; ".join(errors))
    return meta


def project_coalitions(
    model: ScaleInteractionExchangeMSHNet, output: dict[str, Any]
) -> dict[str, Tensor]:
    terms, features = output["sied"]["stage_terms"][0], output["decoder_features"]
    coarse = (
        model.up(model.output_1(features[1])),
        model.up_4(model.output_2(features[2])),
        model.up_8(model.output_3(features[3])),
    )
    result = {}
    for q in ("q11", "q10", "q01", "q00"):
        scales = torch.cat((model.output_0(terms[q]),) + coarse, dim=1)
        result["z" + q[1:]] = model.final(scales)
        if q == "q11":
            result["scale_logits"] = scales
    return result


def compute_conflict_maps(
    z11: Tensor, z10: Tensor, z01: Tensor, z00: Tensor,
    current_main: Tensor, interaction: Tensor, *, eps: float
) -> dict[str, Tensor]:
    """Apply the frozen D0 feature-ratio/projected-direction definition."""
    if not math.isfinite(eps) or eps <= 0 or len({tuple(x.shape) for x in (z11,z10,z01,z00)}) != 1:
        raise ValueError("positive eps and shape-matched coalition logits required")
    if (
        z11.ndim != 4
        or z11.shape[1] != 1
        or any(not bool(torch.isfinite(x).all()) for x in (z11, z10, z01, z00))
    ):
        raise ValueError("coalition logits must be finite B1HW tensors")
    if (
        current_main.ndim != 4
        or current_main.shape != interaction.shape
        or current_main.shape[0] != z11.shape[0]
        or current_main.shape[-2:] != z11.shape[-2:]
    ):
        raise ValueError("D0 current-main/interaction feature shapes must match")
    if not bool(torch.isfinite(current_main).all() and torch.isfinite(interaction).all()):
        raise ValueError("D0 mechanism features must be finite")
    p = z10 - z00
    j = z11 - z10 - z01 + z00
    # Preserve the preregistered mean-anchor index: r is measured on D0
    # features; projected p_z/j_z determine whether the final decision moves
    # in opposing directions.
    pr = current_main.square().mean(1, keepdim=True).sqrt()
    jr = interaction.square().mean(1, keepdim=True).sqrt()
    r = jr / (pr + eps)
    opposite = p * j < 0
    score = torch.log1p(r) * opposite.to(r.dtype)
    # This is the preregistered projected-logit support test.  Requiring each
    # term separately to exceed eps would silently define a stricter mask.
    support = p.abs() + j.abs() > eps
    mask = (r >= 1) & opposite & support
    if not bool(torch.isfinite(r).all() and torch.isfinite(score).all()):
        raise RuntimeError("non-finite conflict index")
    return {"p_z": p, "j_z": j, "p_feature_rms": pr, "j_feature_rms": jr,
            "r": r, "conflict_score": score, "conflict_mask": mask}


def _geometry(region: Any) -> dict[str, Any]:
    return {"label": int(region.label), "area": int(region.area),
            "centroid": [float(x) for x in region.centroid],
            "bbox": [int(x) for x in region.bbox]}


def analyse_image_components(
    *, image_id: str, prediction: np.ndarray, target: np.ndarray,
    z_base: np.ndarray, scale_logits: np.ndarray,
    candidate_thresholds=(0.5, 0.3, 0.2, 0.1),
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Label component errors and prediction-only recoverable FN targets."""
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    match = match_connected_components(
        prediction, target, max_centroid_distance=MAX_DISTANCE, connectivity=CONNECTIVITY
    )
    candidates = generate_prediction_candidates(
        z_base, scale_logits, probability_thresholds=candidate_thresholds,
        connectivity=CONNECTIVITY,
    )
    t2p = {t: (p, d) for t, p, d in match.matches}
    p2t = {p: (t, d) for t, p, d in match.matches}
    recoverable: dict[int, list[tuple[int, float]]] = {}
    for tid in match.unmatched_target_indices:
        center = np.asarray(match.target_regions[tid].centroid)
        near = sorted(
            (c.candidate_id, float(np.linalg.norm(np.asarray(c.centroid) - center)))
            for c in candidates
            if np.linalg.norm(np.asarray(c.centroid) - center) < MAX_DISTANCE
        )
        if near:
            recoverable[tid] = near

    candidate_targets: dict[int, list[tuple[int, float]]] = {}
    for tid, near in recoverable.items():
        for candidate_id, distance in near:
            candidate_targets.setdefault(candidate_id, []).append((tid, distance))

    rows = []
    for tid, region in enumerate(match.target_regions):
        pair, near = t2p.get(tid), recoverable.get(tid, [])
        rows.append({
            "image_id": image_id, "domain": "target",
            "role": "tp_target" if pair else "fn_target",
            "recoverable": bool(near), "component_index": tid,
            "component_id": int(region.label), **_geometry(region),
            "match_component_id": (
                int(match.prediction_regions[pair[0]].label) if pair else None
            ),
            "match_distance": pair[1] if pair else None,
            "recoverable_candidate_ids": [x[0] + 1 for x in near],
        })
    for pid, region in enumerate(match.prediction_regions):
        pair = p2t.get(pid)
        rows.append({
            "image_id": image_id, "domain": "prediction",
            "role": "matched_pred" if pair else "fp_pred", "component_index": pid,
            "component_id": int(region.label), **_geometry(region),
            "match_component_id": int(match.target_regions[pair[0]].label) if pair else None,
            "match_distance": pair[1] if pair else None,
        })
    for candidate in candidates:
        supported = sorted(candidate_targets.get(candidate.candidate_id, []))
        rows.append({
            "image_id": image_id, "domain": "candidate", "role": "candidate",
            "component_index": candidate.candidate_id,
            "component_id": candidate.candidate_id + 1,
            "label": candidate.candidate_id + 1, "area": candidate.area,
            "centroid": [float(x) for x in candidate.centroid],
            "bbox": [int(x) for x in candidate.bbox],
            "source": candidate.source,
            "probability_threshold": candidate.probability_threshold,
            "supports_recoverable_fn": bool(supported),
            "recoverable_fn_target_ids": [
                int(match.target_regions[x[0]].label) for x in supported
            ],
            "recoverable_fn_target_distances": [x[1] for x in supported],
        })

    tids, pids = list(range(len(match.target_regions))), list(range(len(match.prediction_regions)))
    tp, fn, mp, fp, rfn = sorted(t2p), list(match.unmatched_target_indices), sorted(p2t), list(match.unmatched_prediction_indices), sorted(recoverable)
    area = lambda regions, ids: int(sum(regions[i].area for i in ids))
    labels = lambda regions, ids: [int(regions[i].label) for i in ids]
    fields = {
        "target_component_count": len(tids), "true_positive_component_count": len(tp),
        "false_negative_component_count": len(fn), "prediction_component_count": len(pids),
        "matched_prediction_component_count": len(mp), "false_positive_component_count": len(fp),
        "recoverable_fn_component_count": len(rfn), "candidate_component_count": len(candidates),
        "target_component_area": area(match.target_regions, tids),
        "true_positive_target_component_area": area(match.target_regions, tp),
        "false_negative_target_component_area": area(match.target_regions, fn),
        "prediction_component_area": area(match.prediction_regions, pids),
        "matched_prediction_component_area": area(match.prediction_regions, mp),
        "false_positive_component_area": area(match.prediction_regions, fp),
        "recoverable_fn_target_component_area": area(match.target_regions, rfn),
        "target_component_ids": labels(match.target_regions, tids),
        "true_positive_target_component_ids": labels(match.target_regions, tp),
        "false_negative_target_component_ids": labels(match.target_regions, fn),
        "prediction_component_ids": labels(match.prediction_regions, pids),
        "matched_prediction_component_ids": labels(match.prediction_regions, mp),
        "false_positive_prediction_component_ids": labels(match.prediction_regions, fp),
        "recoverable_fn_target_component_ids": labels(match.target_regions, rfn),
        "candidate_component_ids": [candidate.candidate_id + 1 for candidate in candidates],
    }
    rlabels = [match.target_regions[i].label for i in rfn]
    maps = {
        "target_component_labels": match.target_label_map.astype(np.int32),
        "prediction_component_labels": match.prediction_label_map.astype(np.int32),
        "candidate_component_labels": candidate_label_map(candidates, prediction.shape).astype(np.int32),
        "recoverable_fn_mask": np.isin(match.target_label_map, rlabels).astype(np.uint8),
    }
    return fields, rows, maps


def enrich_component_mechanism_fields(
    rows: list[dict[str, Any]],
    maps: dict[str, np.ndarray],
    *,
    p_z: np.ndarray,
    j_z: np.ndarray,
    ratio: np.ndarray,
    conflict_score: np.ndarray,
    conflict_mask: np.ndarray,
    prediction_logit: np.ndarray,
) -> None:
    """Attach auditable mechanism statistics to every exported component row."""
    label_maps = {
        "target": maps["target_component_labels"],
        "prediction": maps["prediction_component_labels"],
        "candidate": maps["candidate_component_labels"],
    }
    arrays = (p_z, j_z, ratio, conflict_score, conflict_mask, prediction_logit)
    shape = label_maps["target"].shape
    if any(np.asarray(value).shape != shape for value in arrays):
        raise ValueError("component mechanism arrays must share the label-map shape")
    for row in rows:
        labels = label_maps[row["domain"]]
        mask = labels == int(row["component_id"])
        count = int(mask.sum())
        if count != int(row["area"]) or count <= 0:
            raise RuntimeError("component ID/label-map area mismatch")
        row.update({
            "p_z_mean": float(np.asarray(p_z)[mask].mean()),
            "j_z_mean": float(np.asarray(j_z)[mask].mean()),
            "interaction_ratio_mean": float(np.asarray(ratio)[mask].mean()),
            "interaction_ratio_p95": float(np.percentile(np.asarray(ratio)[mask], 95)),
            "mean_anchor_score_mean": float(np.asarray(conflict_score)[mask].mean()),
            "conflict_pixels": int(np.asarray(conflict_mask, bool)[mask].sum()),
            "conflict_fraction": float(np.asarray(conflict_mask, bool)[mask].mean()),
            "prediction_logit_mean": float(np.asarray(prediction_logit)[mask].mean()),
        })


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def _git() -> tuple[str, bool | None]:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
            check=True, capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def validate_recomputed_checkpoint_metrics(
    checkpoint: dict[str, Any], summary: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Require the exported factual prediction to reproduce checkpoint metrics."""
    pairs = {
        "iou": "pooled_iou",
        "pd": "pd",
        "fa": "fa_per_million",
    }
    validated: dict[str, dict[str, float]] = {}
    for checkpoint_key, summary_key in pairs.items():
        if checkpoint_key not in checkpoint:
            raise RuntimeError(f"checkpoint metric missing: {checkpoint_key}")
        expected = float(checkpoint[checkpoint_key])
        recomputed = float(summary[summary_key])
        if not math.isfinite(expected) or not math.isfinite(recomputed):
            raise RuntimeError(f"non-finite checkpoint/recomputed metric: {checkpoint_key}")
        if not math.isclose(expected, recomputed, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(
                "factual audit prediction does not reproduce checkpoint metric: "
                f"{checkpoint_key} checkpoint={expected!r} recomputed={recomputed!r}"
            )
        validated[checkpoint_key] = {
            "checkpoint": expected,
            "recomputed": recomputed,
        }
    return validated


def validate_finalized_baseline_provenance(
    *,
    batch_id: str,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    dataset_dir: Path,
    seed: int,
    audit_config: dict[str, Any],
) -> dict[str, Any]:
    """Bind an audit to one completed, fail-closed baseline-grid job."""
    if (
        not batch_id
        or batch_id in {".", ".."}
        or Path(batch_id).name != batch_id
        or SAFE_IMAGE_ID.fullmatch(batch_id) is None
    ):
        raise RuntimeError("batch_id must be one directory name")
    batch_dir = ROOT / "repro_runs" / "clean" / batch_id
    manifest = require_mapping(
        read_json(batch_dir / "manifest.json", "clean baseline manifest"),
        "clean baseline manifest",
    )
    seeds, datasets_meta = validate_manifest(manifest, batch_dir)
    summary = require_mapping(
        read_json(batch_dir / BASELINE_SUMMARY_JSON, "finalized baseline summary"),
        "finalized baseline summary",
    )
    expected_summary_header = {
        "schema_version": 1,
        "batch_id": batch_id,
        "status": "complete_and_validated",
        "method": "mshnet",
        "model_type": "mshnet",
        "official_test_status": "untouched; not evaluated by this finalizer",
        "not_for_official_test_or_main_table_claims": True,
        "epochs_per_run": EXPECTED_EPOCHS,
        "seeds": seeds,
    }
    if any(summary.get(key) != value for key, value in expected_summary_header.items()):
        raise RuntimeError("baseline summary is not a completed development-only finalization")
    summary_datasets = require_mapping(summary.get("datasets"), "baseline summary.datasets")
    if set(summary_datasets) != set(DATASET_NAMES):
        raise RuntimeError("finalized summary does not contain the exact three-dataset grid")

    manifest_args = require_mapping(manifest["args"], "manifest.args")
    frozen_manifest_recipe = {
        "epochs": EXPECTED_EPOCHS,
        "batch_size": 4,
        "num_workers": 4,
        "lr": 0.05,
        "warm_epoch": 5,
        "val_fraction": 0.2,
        "split_seed": 20260711,
        "deterministic": "true",
    }
    manifest_mismatches = [
        f"{key}: manifest={manifest_args.get(key)!r} expected={value!r}"
        for key, value in frozen_manifest_recipe.items()
        if manifest_args.get(key) != value
    ]
    if manifest_mismatches:
        raise RuntimeError("clean baseline manifest recipe mismatch: " + "; ".join(manifest_mismatches))

    jobs_by_pair = {
        (job["dataset"], job["seed"]): job for job in manifest["jobs"]
    }
    dataset_name = dataset_dir.resolve().name
    selected: dict[str, Any] | None = None
    validated_jobs = 0
    for grid_dataset in DATASET_NAMES:
        dataset_meta = require_mapping(
            datasets_meta[grid_dataset], f"manifest dataset {grid_dataset}"
        )
        dataset_summary = require_mapping(
            summary_datasets[grid_dataset], f"baseline summary dataset {grid_dataset}"
        )
        expected_split_hashes = {
            "fit": dataset_meta["fit_sha256"],
            "validation": dataset_meta["val_sha256"],
            "official_test_audit_only": dataset_meta["official_test_sha256"],
        }
        if dataset_summary.get("split_hashes") != expected_split_hashes:
            raise RuntimeError(f"finalized summary split hashes disagree for {grid_dataset}")
        runs = dataset_summary.get("runs")
        if not isinstance(runs, list) or len(runs) != len(seeds):
            raise RuntimeError(f"finalized summary run grid incomplete for {grid_dataset}")
        runs_by_seed = {
            run.get("seed"): require_mapping(run, "finalized summary run") for run in runs
        }
        if set(runs_by_seed) != set(seeds) or len(runs_by_seed) != len(seeds):
            raise RuntimeError(f"finalized summary seeds invalid for {grid_dataset}")

        for grid_seed in seeds:
            job = require_mapping(
                jobs_by_pair[(grid_dataset, grid_seed)],
                f"manifest job {grid_dataset}/{grid_seed}",
            )
            result = require_mapping(
                read_json(Path(job["result_file"]), "completed job result"),
                "completed job result",
            )
            validate_result(result, job)
            run_dir = Path(job["run_dir"]).resolve()
            rows = parse_metrics(run_dir / "epoch_metric.log")
            if len(rows) != EXPECTED_EPOCHS or [row["epoch"] for row in rows] != list(
                range(EXPECTED_EPOCHS)
            ):
                raise RuntimeError(
                    f"{job['job_id']} does not contain exactly epochs 0..{EXPECTED_EPOCHS - 1}"
                )
            expected_checkpoint = run_dir / ROLE_FILES["best_iou"]
            is_selected = grid_dataset == dataset_name and grid_seed == seed
            grid_checkpoint = checkpoint if is_selected else load_checkpoint_cpu(expected_checkpoint)
            try:
                job_summary = validate_checkpoint(
                    grid_checkpoint, job, dataset_meta, manifest_args, rows
                )
            finally:
                if not is_selected:
                    del grid_checkpoint

            summary_run = runs_by_seed[grid_seed]
            if Path(summary_run.get("checkpoint", "")).resolve() != expected_checkpoint:
                raise RuntimeError(f"finalized summary checkpoint path mismatch for {job['job_id']}")
            for key in ("best_epoch", "iou", "pd", "fa"):
                if not math.isclose(
                    float(summary_run[key]), float(job_summary[key]),
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        f"finalized summary disagrees with checkpoint {key} for {job['job_id']}"
                    )

            run_config = require_mapping(
                read_json(run_dir / "run_config.json", "training run_config"),
                "training run_config",
            )
            training_args = require_mapping(run_config.get("args"), "run_config.args")
            expected_training = {
                "mode": "train",
                "model_type": "mshnet",
                "dataset_dir": str(Path(job["dataset_dir"]).resolve()),
                "train_split_file": job["train_file"],
                "val_split_file": "",
                "test_split_file": job["test_file"],
                "val_fraction": manifest_args["val_fraction"],
                "split_seed": manifest_args["split_seed"],
                "seed": grid_seed,
                "base_size": 256,
                "crop_size": 256,
                "batch_size": manifest_args["batch_size"],
                "num_workers": manifest_args["num_workers"],
                "lr": manifest_args["lr"],
                "warm_epoch": manifest_args["warm_epoch"],
                "epochs": manifest_args["epochs"],
                "deterministic": True,
                "pin_memory": True,
                "run_label": job["job_id"],
                "run_dir": str(run_dir),
            }
            recipe_errors = [
                f"{key}: training={training_args.get(key)!r} expected={value!r}"
                for key, value in expected_training.items()
                if training_args.get(key) != value
            ]
            if recipe_errors:
                raise RuntimeError(
                    f"frozen training recipe mismatch for {job['job_id']}: "
                    + "; ".join(recipe_errors)
                )
            if is_selected:
                expected_audit = {
                    "train_split_file": training_args["train_split_file"],
                    "val_split_file": training_args["val_split_file"],
                    "test_split_file": training_args["test_split_file"],
                    "val_fraction": training_args["val_fraction"],
                    "split_seed": training_args["split_seed"],
                    "base_size": training_args["base_size"],
                    "crop_size": training_args["crop_size"],
                    "batch_size": training_args["batch_size"],
                    "num_workers": training_args["num_workers"],
                }
                audit_errors = [
                    f"{key}: audit={audit_config.get(key)!r} expected={value!r}"
                    for key, value in expected_audit.items()
                    if audit_config.get(key) != value
                ]
                if audit_errors:
                    raise RuntimeError(
                        "audit/training configuration mismatch: " + "; ".join(audit_errors)
                    )
                if checkpoint_path.resolve() != expected_checkpoint:
                    raise RuntimeError(
                        "formal audit requires the exact frozen best-IoU checkpoint path"
                    )
                if dataset_dir.resolve() != Path(job["dataset_dir"]).resolve():
                    raise RuntimeError("audit dataset directory does not match frozen batch job")
                selected = {"job_id": job["job_id"]}
            validated_jobs += 1

    if validated_jobs != len(DATASET_NAMES) * len(seeds) or selected is None:
        raise RuntimeError("completed baseline provenance grid is incomplete")
    return {
        "batch_id": batch_id,
        "job_id": selected["job_id"],
        "batch_manifest": str((batch_dir / "manifest.json").resolve()),
        "baseline_summary": str((batch_dir / BASELINE_SUMMARY_JSON).resolve()),
        "completion": "all_3x3_jobs_400_epochs_returncode_0_and_finalizer_validated",
    }


def main() -> int:
    a = parse_args()
    source_sha256 = {name: sha256(path) for name, path in AUDIT_SOURCE_FILES.items()}
    if a.eps <= 0 or not math.isfinite(a.eps) or a.batch_size < 1 or a.num_workers < 0:
        raise ValueError("invalid eps/batch/worker setting")
    thresholds = tuple(float(x) for x in a.candidate_thresholds)
    if not thresholds or any(not 0 < x < 1 for x in thresholds):
        raise ValueError("candidate thresholds must be in (0,1)")
    if a.eps != FIXED_EPS or thresholds != FIXED_CANDIDATE_THRESHOLDS:
        raise RuntimeError(
            "formal mechanism audit locks eps and candidate thresholds; no sweep is allowed"
        )

    ckpt_path = Path(a.checkpoint).resolve()
    if ckpt_path.name != ROLE_FILES[a.checkpoint_role]:
        raise RuntimeError("checkpoint role/path mismatch")
    if a.checkpoint_role != "best_iou":
        raise RuntimeError("development mechanism audit is locked to best_iou checkpoints")
    checkpoint_sha256 = sha256(ckpt_path)
    state, ckpt = load_checkpoint(ckpt_path)
    valset = IRSTD_Dataset(a, "val")
    dataset = valset
    meta = validate_checkpoint_metadata(
        ckpt, seed=a.seed, val_hash=valset.split_sha256,
        role=a.mode, split_hash=dataset.split_sha256,
    )
    baseline_provenance = validate_finalized_baseline_provenance(
        batch_id=a.batch_id,
        checkpoint_path=ckpt_path,
        checkpoint=ckpt,
        dataset_dir=Path(a.dataset_dir),
        seed=a.seed,
        audit_config={
            "train_split_file": a.train_split_file,
            "val_split_file": a.val_split_file,
            "test_split_file": a.test_split_file,
            "val_fraction": a.val_fraction,
            "split_seed": a.split_seed,
            "base_size": a.base_size,
            "crop_size": a.crop_size,
            "batch_size": a.batch_size,
            "num_workers": a.num_workers,
        },
    )
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if a.device == "auto" else a.device))
    model = ScaleInteractionExchangeMSHNet(
        a.input_channels, alpha=1, active_stages=(0,), anchor_mode="mean",
        freeze_bn_statistics=True,
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).to(device).eval()

    out = Path(a.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    arrays_dir = out / "arrays"
    arrays_dir.mkdir()
    image_tmp, comp_tmp = out / "images.jsonl.tmp", out / "components.jsonl.tmp"
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False,
        num_workers=a.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=a.num_workers > 0)
    sum_keys = ("intersection_pixels", "union_pixels", "ground_truth_positive_pixels",
        "predicted_positive_pixels", "false_positive_pixels", "false_negative_pixels",
        "target_component_count", "true_positive_component_count", "false_negative_component_count",
        "prediction_component_count", "matched_prediction_component_count",
        "false_positive_component_count", "false_positive_component_area",
        "recoverable_fn_component_count", "recoverable_fn_target_component_area",
        "candidate_component_count", "conflict_on_true_positive_pixels",
        "conflict_on_false_positive_pixels", "conflict_on_false_negative_pixels")
    summary = {k: 0 for k in sum_keys}
    summary.update(images=0, pixels=0, conflict_pixels=0, p_rms_sum=0., j_rms_sum=0.,
                   score_sum=0., mean_anchor_score_sum_true_positive=0.,
                   mean_anchor_score_sum_false_positive=0.,
                   mean_anchor_score_sum_false_negative=0.)
    cursor, decomposition_error = 0, 0.0
    array_records: list[dict[str, Any]] = []
    seen_image_ids: set[str] = set()

    with image_tmp.open("w") as fi, comp_tmp.open("w") as fc, torch.inference_mode():
        for images, labels in loader:
            output = model(images.to(device), True, return_dict=True, alpha=1)
            terms = output["sied"]["stage_terms"][0]
            reconstructed = terms["q00"] + terms["current_main"] + terms["inherited_main"] + terms["interaction"]
            decomposition_error = max(decomposition_error, float((reconstructed-terms["q11"]).abs().max()))
            z = project_coalitions(model, output)
            mech = compute_conflict_maps(
                z["z11"], z["z10"], z["z01"], z["z00"],
                terms["current_main"], terms["interaction"], eps=a.eps,
            )
            for b in range(images.shape[0]):
                image_id = dataset.names[cursor]
                zn = {k: z[k][b,0].float().cpu().numpy() for k in ("z11","z10","z01","z00")}
                mn = {k: mech[k][b,0].float().cpu().numpy() for k in
                      ("p_z","j_z","p_feature_rms","j_feature_rms","r","conflict_score")}
                cmask = mech["conflict_mask"][b,0].cpu().numpy().astype(bool)
                target = labels[b,0].numpy() > .5
                prediction = zn["z11"] > LOGIT
                pred_probability = torch.sigmoid(z["z11"][b,0]).float().cpu().numpy()
                scales = z["scale_logits"][b].float().cpu().numpy()
                fields, rows, maps = analyse_image_components(
                    image_id=image_id, prediction=prediction, target=target,
                    z_base=zn["z11"], scale_logits=scales,
                    candidate_thresholds=thresholds,
                )
                enrich_component_mechanism_fields(
                    rows, maps,
                    p_z=mn["p_z"], j_z=mn["j_z"], ratio=mn["r"],
                    conflict_score=mn["conflict_score"], conflict_mask=cmask,
                    prediction_logit=zn["z11"],
                )
                inter, union = int((prediction & target).sum()), int((prediction | target).sum())
                tp_mask = prediction & target
                fp_mask = prediction & ~target
                fn_mask = ~prediction & target
                pix = {
                    "intersection_pixels": inter, "union_pixels": union,
                    "ground_truth_positive_pixels": int(target.sum()),
                    "predicted_positive_pixels": int(prediction.sum()),
                    "false_positive_pixels": int((prediction & ~target).sum()),
                    "false_negative_pixels": int((~prediction & target).sum()),
                    "conflict_on_true_positive_pixels": int((cmask & tp_mask).sum()),
                    "conflict_on_false_positive_pixels": int((cmask & fp_mask).sum()),
                    "conflict_on_false_negative_pixels": int((cmask & fn_mask).sum()),
                    "mean_anchor_score_sum_true_positive": float(mn["conflict_score"][tp_mask].sum()),
                    "mean_anchor_score_sum_false_positive": float(mn["conflict_score"][fp_mask].sum()),
                    "mean_anchor_score_sum_false_negative": float(mn["conflict_score"][fn_mask].sum()),
                }
                if (
                    SAFE_IMAGE_ID.fullmatch(image_id) is None
                    or image_id in {".", ".."}
                    or image_id in seen_image_ids
                ):
                    raise RuntimeError(f"unsafe or duplicate image_id for artifact path: {image_id!r}")
                seen_image_ids.add(image_id)
                rel = Path("arrays") / f"{image_id}.npz"
                mechanism_arrays = {
                    k: v.astype(np.float32) for k, v in mn.items() if k != "r"
                }
                mechanism_arrays["ratio"] = mn["r"].astype(np.float32)
                _save_npz(out/rel, {**{k:v.astype(np.float32) for k,v in zn.items()},
                    **mechanism_arrays,
                    "conflict_mask": cmask.astype(np.uint8),
                    "pred_logit": zn["z11"].astype(np.float32),
                    "pred_probability": pred_probability.astype(np.float32),
                    "prediction_mask": prediction.astype(np.uint8), "target_mask": target.astype(np.uint8),
                    "scale_logits": scales.astype(np.float32), **maps})
                array_hash = sha256(out / rel)
                array_size = (out / rel).stat().st_size
                array_records.append({
                    "image_id": image_id,
                    "path": str(rel),
                    "sha256": array_hash,
                    "bytes": array_size,
                })
                ratio, score = mn["r"].astype(float), mn["conflict_score"].astype(float)
                row = {"image_index": cursor, "image_id": image_id, **pix,
                    "iou": float(inter/max(1,union)), **fields,
                    "mean_anchor_index": float(score.mean()),
                    "interaction_ratio_mean": float(ratio.mean()),
                    "interaction_ratio_p95": float(np.percentile(ratio,95)),
                    "conflict_pixels": int(cmask.sum()),
                    "conflict_fraction": float(cmask.mean()), "array_path": str(rel),
                    "array_sha256": array_hash, "array_bytes": array_size}
                fi.write(json.dumps(row, allow_nan=False, sort_keys=True)+"\n")
                for component in rows:
                    fc.write(json.dumps(component, allow_nan=False, sort_keys=True)+"\n")
                summary["images"] += 1; summary["pixels"] += target.size
                for k in sum_keys: summary[k] += row[k]
                summary["conflict_pixels"] += int(cmask.sum())
                summary["p_rms_sum"] += float(mn["p_feature_rms"].astype(float).sum())
                summary["j_rms_sum"] += float(mn["j_feature_rms"].astype(float).sum())
                summary["score_sum"] += float(score.sum())
                for key in (
                    "mean_anchor_score_sum_true_positive",
                    "mean_anchor_score_sum_false_positive",
                    "mean_anchor_score_sum_false_negative",
                ):
                    summary[key] += pix[key]
                cursor += 1
    if cursor != len(dataset):
        raise RuntimeError("dataloader/split count mismatch")
    pixels = max(1, summary["pixels"])
    summary.update(
        pooled_iou=float(summary["intersection_pixels"]/max(1,summary["union_pixels"])),
        pd=float(summary["true_positive_component_count"]/max(1,summary["target_component_count"])),
        fa_per_million=float(summary["false_positive_component_area"]/pixels*1e6),
        recoverable_fn_fraction=float(summary["recoverable_fn_component_count"]/max(1,summary["false_negative_component_count"])),
        conflict_fraction=float(summary["conflict_pixels"]/pixels),
        mean_anchor_index=float(summary["score_sum"]/pixels),
        global_r_ratio_of_sums=float(summary["j_rms_sum"]/(summary["p_rms_sum"]+a.eps)),
        conflict_true_positive_coverage=float(
            summary["conflict_on_true_positive_pixels"]
            / max(1, summary["intersection_pixels"])
        ),
        conflict_false_positive_coverage=float(
            summary["conflict_on_false_positive_pixels"]
            / max(1, summary["false_positive_pixels"])
        ),
        conflict_false_negative_coverage=float(
            summary["conflict_on_false_negative_pixels"]
            / max(1, summary["false_negative_pixels"])
        ),
    )
    metric_recomputation = validate_recomputed_checkpoint_metrics(ckpt, summary)
    if len(array_records) != cursor or len({item["path"] for item in array_records}) != cursor:
        raise RuntimeError("array artifact count/path uniqueness mismatch")
    for item in array_records:
        array_path = out / item["path"]
        if not array_path.is_file() or sha256(array_path) != item["sha256"]:
            raise RuntimeError(f"array artifact integrity mismatch: {item['path']}")
    array_inventory = json.dumps(
        array_records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    array_inventory_sha256 = hashlib.sha256(array_inventory).hexdigest()
    if sha256(ckpt_path) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed while mechanism audit was running")
    if {name: sha256(path) for name, path in AUDIT_SOURCE_FILES.items()} != source_sha256:
        raise RuntimeError("audit source code changed while mechanism audit was running")
    image_path, comp_path = out/"images.jsonl", out/"components.jsonl"
    image_tmp.replace(image_path); comp_tmp.replace(comp_path)
    commit, dirty = _git()
    scalar = lambda x: x.item() if isinstance(x, np.generic) else x
    manifest = {
        "schema_version": SCHEMA, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": Path(a.dataset_dir).resolve().name, "dataset_dir": str(Path(a.dataset_dir).resolve()),
        "split_role": a.mode, "split_sha256": dataset.split_sha256,
        "validation_split_sha256": valset.split_sha256, "seed": a.seed,
        "method": meta["method"], "model_type": meta["model_type"],
        "checkpoint": {"role": a.checkpoint_role, "path": str(ckpt_path),
            "sha256": checkpoint_sha256,
            "epoch": int(ckpt["epoch"]), "metrics": {k:scalar(ckpt[k]) for k in ("iou","pd","fa","best_iou") if k in ckpt}},
        "threshold_probability": PROB, "threshold_logit": LOGIT, "connectivity": CONNECTIVITY,
        "max_centroid_distance": MAX_DISTANCE, "base_size": a.base_size,
        "crop_size": a.crop_size, "anchor_mode": "mean",
        "batch_size": a.batch_size, "num_workers": a.num_workers,
        "deterministic": True,
        "active_stage": 0, "eps": a.eps, "candidate_probability_thresholds": list(thresholds),
        "recoverable_fn_definition": "unmatched target with prediction-only final/side candidate centroid distance < 3",
        "conflict_definition": "p=D0 current_main; j=D0 interaction; r=RMS_channel(j)/(RMS_channel(p)+eps); p_z=z10-z00; j_z=z11-z10-z01+z00; conflict_score=log1p(r)*1[p_z*j_z<0]; mask=1[r>=1 and p_z*j_z<0 and |p_z|+|j_z|>eps]; mean_anchor_index=mean(score)",
        "global_ratio_of_sums_definition": "sum_pixel(RMS_channel(D0 interaction))/(sum_pixel(RMS_channel(D0 current_main))+eps)",
        "git_commit": commit, "git_dirty": dirty,
        "source_sha256": source_sha256,
        "official_test_status": "sealed; this exporter accepts development validation only",
        "baseline_provenance": baseline_provenance,
        "checkpoint_validation": {"model_seed_val_hash": "matched", "strict_state_dict": True,
            "frozen": True, "recomputed_metrics": metric_recomputation},
        "artifacts": {"images_jsonl": "images.jsonl", "images_sha256": sha256(image_path),
            "components_jsonl": "components.jsonl", "components_sha256": sha256(comp_path),
            "arrays_dir": "arrays", "array_count": cursor,
            "array_inventory_sha256": array_inventory_sha256,
            "array_total_bytes": sum(item["bytes"] for item in array_records)},
        "max_mobius_reconstruction_abs_error": decomposition_error, "summary": summary,
    }
    tmp = out/"manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")
    tmp.replace(out/"manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FinalizationError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"mechanism audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
