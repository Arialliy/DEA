#!/usr/bin/env python3
"""Test-selected signed-readout diagnostic on a frozen full-train MSHNet.

This entry point is deliberately a diagnostic, not a candidate paper model.
It trains four parameter-matched 17-parameter readouts on every image in the
canonical ``img_idx/train_<dataset>.txt`` manifest and evaluates them on every
image in the canonical test manifest.  The frozen MSHNet checkpoint is the
best-test-IoU checkpoint selected by the repository's every-ten-epoch
full-train protocol.  Consequently every number produced here is explicitly
test selected; there is no validation split and no untouched-test claim.

The purpose is narrow: decide whether the first frozen decoder tensor ``d0``
already contains signed target-vs-local-background evidence.  Passing this
diagnostic may freeze a coordinate convention for the next structural stage,
but the readout itself is never a method contribution and must never be
attached to MSHNet as another module.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections.abc import Mapping, Sequence
import datetime as dt
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
from tools import finalize_test_selected_baselines as baseline_finalizer  # noqa: E402
from tools import run_signed_readout_probe as legacy_probe  # noqa: E402
from tools.audit_cross_seed_failure_persistence import _normalize_state_dict  # noqa: E402
from tools.finalize_clean_baselines import load_checkpoint_cpu  # noqa: E402
from utils.data import IRSTD_Dataset  # noqa: E402
from utils.full_train_test_protocol import audit_canonical_dataset  # noqa: E402
from utils.target_identity import (  # noqa: E402
    StableTargetSet,
    build_stable_target_set,
)


SCHEMA = "dea.gate_k.full_train_test_signed_readout_probe.v1"
PROVENANCE_SCHEMA = "dea.gate_k.full_train_test_signed_readout_provenance.v1"
AUTHORITY_SCHEMA = "dea.gate_k.canonical_test_target_authority.v1"
NATIVE_REPLAY_SCHEMA = "dea.gate_k.native_selected_checkpoint_replay.v1"
DEFAULT_BATCH_ID = "mshnet_test_selected_full_train_interval_v1"
FORMAL_EPOCHS = legacy_probe.PROBE_FORMAL_EPOCHS
SMOKE_EPOCHS = legacy_probe.PROBE_SMOKE_EPOCHS
VARIANT_ORDER = legacy_probe.ALL_VARIANTS
LOCKED_BASE_SIZE = 256
LOCKED_CROP_SIZE = 256
NATIVE_REPLAY_ABS_TOL = 1e-12

BUNDLE_FILES = (
    "summary.json",
    "training_history.jsonl",
    "oracle_targets.jsonl",
    "crossfit_targets.jsonl",
    "crossfit_images.jsonl",
    "crossfit_calibration.jsonl",
    "test_logits.npz",
    "probe_heads.pkl",
    "provenance.json",
)


class FullTrainTestProbeError(RuntimeError):
    """The requested diagnostic violates its fail-closed contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument(
        "--front-freeze-dir",
        default="repro_runs/gate_i/front_freeze_confirmatory_v1",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--protocol",
        choices=("formal", "smoke"),
        default="formal",
        help=(
            "formal uses the pre-registered 20 epochs; smoke uses the same "
            "historical 10-epoch engineering duration and is allowed only "
            "for NUAA-SIRST/20260711"
        ),
    )
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def protocol_epochs(protocol: str, dataset: str, seed: int) -> int:
    if protocol == "formal":
        return FORMAL_EPOCHS
    if protocol == "smoke" and dataset == "NUAA-SIRST" and seed == 20260711:
        return SMOKE_EPOCHS
    raise FullTrainTestProbeError(
        "smoke is predeclared only for NUAA-SIRST/20260711"
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullTrainTestProbeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FullTrainTestProbeError(f"{label} is not a JSON object: {path}")
    return value


def _plain_file_record(path: Path, label: str) -> dict[str, str]:
    """Bind one external input to a plain absolute path and byte hash."""

    path = path.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if path.is_symlink() or not path.is_file():
        raise FullTrainTestProbeError(f"missing/non-plain {label}: {path}")
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": legacy_probe.sha256_file(resolved),
    }


def _require_unchanged(label: str, before: Any, after: Any) -> None:
    if after != before:
        raise FullTrainTestProbeError(f"{label} drifted during probe execution")


def _canonicalize_rng_roles(
    raw: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    """Replace the inherited development-loader label with its true test role."""

    value = dict(raw)
    old_key = "dev_dataloader_generator_seed"
    new_key = "test_dataloader_generator_seed"
    if old_key in value and new_key in value:
        raise FullTrainTestProbeError("RNG record contains both dev and test roles")
    observed = value.pop(old_key, value.get(new_key))
    if isinstance(observed, bool) or observed != seed + 1:
        raise FullTrainTestProbeError("test DataLoader RNG seed drifted")
    value[new_key] = int(observed)
    if old_key in value:
        raise AssertionError("development RNG role survived canonicalization")
    return value


def _read_locked_run_config(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the selected run configuration and enforce the 256x256 contract."""

    path = run_dir.resolve() / "run_config.json"
    record = _plain_file_record(path, "baseline run_config")
    run_config = _read_json(path, "baseline run_config")
    stored = run_config.get("args")
    if not isinstance(stored, dict):
        raise FullTrainTestProbeError("baseline run_config has no args mapping")
    for key, expected in (
        ("base_size", LOCKED_BASE_SIZE),
        ("crop_size", LOCKED_CROP_SIZE),
    ):
        observed = stored.get(key)
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise FullTrainTestProbeError(
                f"baseline run_config.args.{key} must be integer {expected}"
            )
        if observed != expected:
            raise FullTrainTestProbeError(
                f"baseline run_config.args.{key} must equal {expected}, got {observed}"
            )
    return run_config, {
        **record,
        "base_size": LOCKED_BASE_SIZE,
        "crop_size": LOCKED_CROP_SIZE,
    }


def _baseline_artifact_records(
    batch_dir: Path, job: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """Hash every baseline control artifact that must stay immutable."""

    batch_dir = batch_dir.resolve()
    job_id = str(job.get("job_id", ""))
    run_dir = Path(str(job.get("run_dir", ""))).resolve()
    expected_job = (batch_dir / "jobs" / f"{job_id}.json").resolve()
    declared_job = Path(str(job.get("result_file", ""))).resolve()
    if not job_id or declared_job != expected_job:
        raise FullTrainTestProbeError("selected baseline job-result identity drifted")
    return {
        "manifest": _plain_file_record(batch_dir / "manifest.json", "manifest"),
        "job_result": _plain_file_record(expected_job, "job result"),
        "run_config": _plain_file_record(
            run_dir / "run_config.json", "baseline run_config"
        ),
        "protocol_summary": _plain_file_record(
            run_dir / "protocol_summary.json", "baseline protocol summary"
        ),
        "persisted_train_split": _plain_file_record(
            run_dir / "split_train.txt", "persisted train split"
        ),
        "persisted_test_split": _plain_file_record(
            run_dir / "split_test.txt", "persisted test split"
        ),
    }


def _source_paths() -> dict[str, Path]:
    """Inventory all local sources that can change the produced evidence."""

    return {
        "tool": Path(__file__).resolve(),
        "legacy_probe_primitives": ROOT / "tools" / "run_signed_readout_probe.py",
        "mshnet": ROOT / "model" / "MSHNet.py",
        "mshnet_checkpoint": ROOT / "model" / "mshnet_checkpoint.py",
        "signed_local_reference": ROOT / "model" / "signed_local_reference.py",
        "dataset": ROOT / "utils" / "data.py",
        "full_train_test_protocol": ROOT / "utils" / "full_train_test_protocol.py",
        "baseline_finalizer": ROOT / "tools" / "finalize_test_selected_baselines.py",
        "baseline_scheduler": ROOT / "tools" / "run_test_selected_baselines.py",
        "cross_fitted_low_fa": ROOT / "utils" / "cross_fitted_low_fa.py",
        "component_operating_point": ROOT / "utils" / "component_operating_point.py",
        "nested_component_grid": ROOT / "utils" / "nested_component_grid.py",
        "metric": ROOT / "utils" / "metric.py",
        "target_identity": ROOT / "utils" / "target_identity.py",
        "checkpoint_normalizer": (
            ROOT / "tools" / "audit_cross_seed_failure_persistence.py"
        ),
        "checkpoint_loader": ROOT / "tools" / "finalize_clean_baselines.py",
        "front_freeze_validator": ROOT / "tools" / "audit_rcp_gt_coverage.py",
    }


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for name, path in _source_paths().items():
        if path.is_symlink() or not path.is_file():
            raise FullTrainTestProbeError(f"missing/non-plain source {name}: {path}")
        output[name] = legacy_probe.sha256_file(path)
    return output


def validate_selected_baseline(
    batch_dir: Path,
    dataset: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Strictly validate one complete job from the frozen 3x3 baseline grid."""

    batch_dir = batch_dir.resolve()
    if batch_dir.is_symlink() or not batch_dir.is_dir():
        raise FullTrainTestProbeError(f"missing/non-plain baseline batch: {batch_dir}")
    manifest_path = batch_dir / "manifest.json"
    manifest = baseline_finalizer.require_mapping(
        baseline_finalizer.read_json(manifest_path, "manifest"), "manifest"
    )
    try:
        _, manifest_args, jobs = baseline_finalizer._validate_manifest(
            manifest, batch_dir
        )
        baseline_finalizer._validate_sources(
            manifest,
            {
                "scheduler": ROOT / "tools" / "run_test_selected_baselines.py",
                "training_entry": ROOT / "tools" / "train_test_selected_full_train.py",
            },
        )
        split_audits = baseline_finalizer._validate_dataset_splits(
            manifest,
            baseline_finalizer.CANONICAL_DATASETS,
            ROOT / "datasets",
        )
    except baseline_finalizer.FinalizationError as exc:
        raise FullTrainTestProbeError(
            f"baseline batch contract is invalid: {exc}"
        ) from exc
    selected = [
        job
        for job in jobs
        if str(job.get("dataset")) == dataset and int(job.get("seed", -1)) == seed
    ]
    if len(selected) != 1:
        raise FullTrainTestProbeError(
            f"baseline grid must contain exactly one job for {dataset}/{seed}"
        )
    job = selected[0]
    manifest_datasets = baseline_finalizer.require_mapping(
        manifest.get("datasets"), "manifest.datasets"
    )
    try:
        validated = baseline_finalizer._validate_job(
            job=job,
            batch_dir=batch_dir,
            manifest_args=manifest_args,
            manifest_dataset=baseline_finalizer.require_mapping(
                manifest_datasets[dataset], f"manifest.datasets.{dataset}"
            ),
            dataset_audit=split_audits[dataset],
            checkpoint_loader=load_checkpoint_cpu,
        )
    except baseline_finalizer.FinalizationError as exc:
        raise FullTrainTestProbeError(
            f"selected baseline job is incomplete/invalid: {exc}"
        ) from exc
    best = validated.get("best_iou")
    if not isinstance(best, dict) or best.get("status") != "found":
        raise FullTrainTestProbeError("selected job has no validated best-IoU checkpoint")
    checkpoint_path = Path(str(best.get("checkpoint", ""))).resolve()
    if (
        checkpoint_path.name != "checkpoint_best_iou.pkl"
        or checkpoint_path.parent != Path(str(job["run_dir"])).resolve()
        or legacy_probe.sha256_file(checkpoint_path) != best.get("checkpoint_sha256")
    ):
        raise FullTrainTestProbeError("validated best-IoU checkpoint identity drifted")
    manifest_record = {
        "path": str(manifest_path.resolve()),
        "sha256": legacy_probe.sha256_file(manifest_path),
        "immutable_contract_sha256": manifest["immutable_contract_sha256"],
        "protocol": manifest["protocol"],
        "batch_id": manifest["batch_id"],
    }
    return job, validated, manifest_record


def _canonical_data_record(audit: Any) -> dict[str, Any]:
    return {
        "dataset": audit.dataset_name,
        "dataset_dir": audit.dataset_dir,
        "train_path": audit.train.path,
        "train_count": audit.train.count,
        "train_raw_sha256": audit.train.raw_sha256,
        "train_normalized_sha256": audit.train.normalized_sha256,
        "test_path": audit.test.path,
        "test_count": audit.test.count,
        "test_raw_sha256": audit.test.raw_sha256,
        "test_normalized_sha256": audit.test.normalized_sha256,
        "train_test_overlap_count": 0,
        "validation_count": 0,
        "constructor_compatibility_alias": (
            "canonical test manifest was read by the legacy train constructor; "
            "no validation dataset was constructed or iterated"
        ),
    }


def build_full_train_test_datasets(
    job: Mapping[str, Any],
) -> tuple[IRSTD_Dataset, IRSTD_Dataset, dict[str, Any], dict[str, Any]]:
    """Construct exactly canonical full train and test datasets, with no val set."""

    audit = audit_canonical_dataset(str(job["dataset_dir"]))
    if audit.dataset_name != str(job["dataset"]):
        raise FullTrainTestProbeError("canonical dataset identity drifted")
    run_config, run_config_record = _read_locked_run_config(
        Path(str(job["run_dir"]))
    )
    stored = run_config.get("args")
    if not isinstance(stored, dict):  # guarded by _read_locked_run_config
        raise AssertionError("unreachable invalid run_config args")
    args = Namespace(**stored)
    args.dataset_dir = audit.dataset_dir
    args.train_split_file = audit.train.path
    args.test_split_file = audit.test.path
    args.val_fraction = 0.0
    args.return_instance_labels = False

    # IRSTD_Dataset's legacy train constructor requires a non-empty
    # val_split_file even when no train names are removed.  Supplying the
    # canonical test path here is a constructor compatibility alias only: no
    # mode='val' object is constructed, and the resulting train names remain
    # exactly the complete canonical train manifest.
    args.val_split_file = audit.test.path
    train_dataset = IRSTD_Dataset(args, mode="train")
    test_dataset = IRSTD_Dataset(args, mode="test")
    args.val_split_file = ""

    if (
        tuple(train_dataset.names) != audit.train.names
        or tuple(test_dataset.names) != audit.test.names
        or train_dataset.split_sha256 != audit.train.normalized_sha256
        or test_dataset.split_sha256 != audit.test.normalized_sha256
        or set(train_dataset.names).intersection(test_dataset.names)
    ):
        raise FullTrainTestProbeError("constructed full train/test manifests drifted")
    if train_dataset.mode != "train" or test_dataset.mode != "test":
        raise FullTrainTestProbeError("dataset roles drifted")
    data_record = _canonical_data_record(audit)
    return train_dataset, test_dataset, data_record, run_config_record


def _loader(
    dataset: IRSTD_Dataset,
    *,
    training: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": legacy_probe._seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        # The frozen checkpoint was selected with the canonical evaluator's
        # batch-size-one test loader.  Retaining that execution shape here is
        # part of the native-replay contract: otherwise CUDA convolution
        # kernels can differ at threshold-adjacent pixels even though the
        # weights and images are identical.
        batch_size=legacy_probe.PROBE_BATCH_SIZE if training else 1,
        shuffle=training,
        drop_last=False,
        **kwargs,
    )
    if loader.drop_last:
        raise FullTrainTestProbeError("full-train/test probe may not drop samples")
    return loader


def build_test_authority(
    dataset: IRSTD_Dataset,
    *,
    dataset_name: str,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, StableTargetSet], dict[str, Any]]:
    """Build the canonical test target identity registry independently of logits."""

    loader = _loader(
        dataset,
        training=False,
        num_workers=num_workers,
        device=torch.device("cpu"),
        seed=seed,
    )
    registry: dict[str, StableTargetSet] = {}
    cursor = 0
    for _, masks in loader:
        if masks.ndim != 4 or masks.shape[1] != 1 or not bool(torch.isfinite(masks).all()):
            raise FullTrainTestProbeError("canonical test target tensor is invalid")
        arrays = (masks[:, 0] > 0.5).numpy().astype(bool, copy=False)
        for batch_index, target in enumerate(arrays):
            image_name = dataset.names[cursor + batch_index]
            if image_name in registry:
                raise FullTrainTestProbeError("duplicate canonical test image identity")
            registry[image_name] = build_stable_target_set(
                target,
                dataset=dataset_name,
                image_name=image_name,
                connectivity=2,
            )
        cursor += int(arrays.shape[0])
    if cursor != len(dataset) or tuple(registry) != tuple(dataset.names):
        raise FullTrainTestProbeError("test target authority coverage drifted")
    payload = [registry[name].as_dict() for name in dataset.names]
    return registry, {
        "schema": AUTHORITY_SCHEMA,
        "dataset": dataset_name,
        "split": "canonical_test",
        "image_count": len(dataset),
        "target_count": sum(len(value.targets) for value in registry.values()),
        "image_order_sha256": legacy_probe._sha256_json(list(dataset.names)),
        "registry_sha256": legacy_probe._sha256_json(payload),
        "connectivity": 2,
        "constructed_independently_of_logits": True,
    }


def build_native_selected_checkpoint_replay(
    scores: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    checkpoint_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Exactly replay MSHNet's native logit-zero operating point.

    The replay intentionally uses integer pooled counts and the repository's
    historical target-ordered component matcher.  The recomputed metrics are
    the arithmetic authority and must reproduce the independently selected
    checkpoint metrics to the predeclared absolute tolerance.
    """

    if not scores or len(scores) != len(targets):
        raise FullTrainTestProbeError("native replay requires aligned samples")
    required_checkpoint = {
        "policy": "test_selected_best_iou",
        "selection_split": "canonical_test",
    }
    for key, expected in required_checkpoint.items():
        if checkpoint_record.get(key) != expected:
            raise FullTrainTestProbeError(
                f"native replay checkpoint binding {key} drifted"
            )
    checkpoint_path = Path(str(checkpoint_record.get("path", "")))
    checkpoint_file = _plain_file_record(checkpoint_path, "selected checkpoint")
    checkpoint_sha = checkpoint_record.get("sha256")
    if checkpoint_file["sha256"] != checkpoint_sha:
        raise FullTrainTestProbeError("native replay selected checkpoint hash drifted")
    for key in ("epoch_zero_based", "completed_epoch"):
        value = checkpoint_record.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise FullTrainTestProbeError(
                f"native replay checkpoint {key} must be an integer"
            )
    if checkpoint_record["completed_epoch"] != checkpoint_record["epoch_zero_based"] + 1:
        raise FullTrainTestProbeError("native replay checkpoint epoch binding drifted")
    for key in ("selected_iou", "selected_pd", "selected_fa_per_mpix"):
        value = checkpoint_record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FullTrainTestProbeError(
                f"native replay checkpoint {key} must be numeric"
            )
        if not np.isfinite(float(value)):
            raise FullTrainTestProbeError(
                f"native replay checkpoint {key} must be finite"
            )

    normalized_scores: list[np.ndarray] = []
    normalized_targets: list[np.ndarray] = []
    expected_shape = (LOCKED_CROP_SIZE, LOCKED_CROP_SIZE)
    for raw_score, raw_target in zip(scores, targets):
        score = np.asarray(raw_score)
        target = np.asarray(raw_target)
        try:
            finite_score = bool(np.isfinite(score).all())
            finite_target = bool(np.isfinite(target).all())
        except TypeError as exc:
            raise FullTrainTestProbeError("native replay arrays must be numeric") from exc
        if (
            score.shape != expected_shape
            or target.shape != expected_shape
            or not finite_score
            or not finite_target
            or not bool(np.all((target == 0) | (target == 1)))
        ):
            raise FullTrainTestProbeError(
                "native replay requires finite binary-target 256x256 samples"
            )
        normalized_scores.append(score)
        normalized_targets.append(target.astype(bool, copy=False))

    try:
        pixel = legacy_probe._pooled_pixel_iou(
            normalized_scores,
            normalized_targets,
            {index: 0.0 for index in range(len(normalized_scores))},
        )
    except Exception as exc:
        raise FullTrainTestProbeError(f"native pixel replay failed: {exc}") from exc

    matched_components = 0
    target_components = 0
    prediction_components = 0
    unmatched_prediction_components = 0
    unmatched_target_components = 0
    unmatched_prediction_area = 0
    for score, target in zip(normalized_scores, normalized_targets):
        match = legacy_probe.match_connected_components(
            score > 0.0,
            target,
            max_centroid_distance=3.0,
            connectivity=2,
        )
        matched_components += len(match.matches)
        target_components += len(match.target_regions)
        prediction_components += len(match.prediction_regions)
        unmatched_prediction_components += len(match.unmatched_prediction_indices)
        unmatched_target_components += len(match.unmatched_target_indices)
        for index in match.unmatched_prediction_indices:
            area = float(match.prediction_regions[index].area)
            if not area.is_integer():
                raise FullTrainTestProbeError(
                    "official legacy matcher returned non-integer component area"
                )
            unmatched_prediction_area += int(area)

    if matched_components + unmatched_target_components != target_components:
        raise FullTrainTestProbeError("native replay target accounting drifted")
    total_pixels = len(normalized_scores) * LOCKED_BASE_SIZE * LOCKED_BASE_SIZE
    pd = (
        float(matched_components / target_components)
        if target_components
        else 0.0
    )
    fa_fraction = float(unmatched_prediction_area / total_pixels)
    fa_per_mpix = fa_fraction * 1_000_000.0
    binding = {
        "policy": checkpoint_record["policy"],
        "selection_split": checkpoint_record["selection_split"],
        "job_id": checkpoint_record.get("job_id"),
        "path": checkpoint_file["path"],
        "sha256": checkpoint_file["sha256"],
        "epoch_zero_based": int(checkpoint_record["epoch_zero_based"]),
        "completed_epoch": int(checkpoint_record["completed_epoch"]),
    }
    reported = {
        "iou": float(checkpoint_record["selected_iou"]),
        "pd": float(checkpoint_record["selected_pd"]),
        "fa_per_mpix": float(checkpoint_record["selected_fa_per_mpix"]),
    }
    replayed = {
        "iou": float(pixel["iou"]),
        "pd": pd,
        "fa_per_mpix": fa_per_mpix,
    }
    differences = {
        key: replayed[key] - reported[key]
        for key in ("iou", "pd", "fa_per_mpix")
    }
    for metric, difference in differences.items():
        if abs(difference) > NATIVE_REPLAY_ABS_TOL:
            raise FullTrainTestProbeError(
                "native original_final_z replay does not reproduce selected "
                f"checkpoint {metric} within absolute tolerance "
                f"{NATIVE_REPLAY_ABS_TOL}: delta={difference}"
            )
    return {
        "schema": NATIVE_REPLAY_SCHEMA,
        "source_variant": "original_final_z",
        "strict_prediction_rule": "original_final_z logit > 0",
        "equivalent_probability_rule": "sigmoid(logit) > 0.5",
        "threshold": 0.0,
        "sample_count": len(normalized_scores),
        "spatial_shape": [LOCKED_CROP_SIZE, LOCKED_CROP_SIZE],
        "selected_checkpoint": binding,
        "selected_checkpoint_binding_sha256": legacy_probe._sha256_json(binding),
        "integer_pixel_counts": {
            key: int(pixel[key])
            for key in (
                "intersection_pixels",
                "union_pixels",
                "prediction_pixels",
                "target_pixels",
            )
        },
        "iou": float(pixel["iou"]),
        "official_legacy": {
            "matcher": (
                "target-ordered nearest unmatched prediction; centroid distance < 3; "
                "8-connectivity"
            ),
            "matched_components": matched_components,
            "target_components": target_components,
            "prediction_components": prediction_components,
            "unmatched_target_components": unmatched_target_components,
            "unmatched_prediction_components": unmatched_prediction_components,
            "unmatched_prediction_area": unmatched_prediction_area,
            "total_pixels": total_pixels,
            "pd": pd,
            "fa_fraction": fa_fraction,
            "fa_per_mpix": fa_per_mpix,
        },
        "checkpoint_reported_metrics": reported,
        "checkpoint_metric_binding": {
            "status": "passed",
            "absolute_tolerance": NATIVE_REPLAY_ABS_TOL,
            "relative_tolerance": 0.0,
            "metrics": ["iou", "pd", "fa_per_mpix"],
        },
        "replay_minus_checkpoint_reported": differences,
        "metric_authority": (
            "integer replay from saved original_final_z logits and canonical test "
            "targets; exact agreement binds it to the selected checkpoint"
        ),
    }


def _initial_parameter_counts(
    model: MSHNet, heads: torch.nn.ModuleDict
) -> dict[str, dict[str, int]]:
    return {
        "original_final_z": {
            "reported_head_parameters": int(
                sum(
                    module.weight.numel()
                    + (module.bias.numel() if module.bias is not None else 0)
                    for module in (
                        model.output_0,
                        model.output_1,
                        model.output_2,
                        model.output_3,
                        model.final,
                    )
                )
            ),
            "trained_parameters": 0,
        },
        "original_output0": {
            "reported_head_parameters": sum(
                value.numel() for value in model.output_0.parameters()
            ),
            "trained_parameters": 0,
        },
        **{
            name: {
                "reported_head_parameters": sum(
                    value.numel() for value in head.parameters()
                ),
                "trained_parameters": sum(
                    value.numel()
                    for value in head.parameters()
                    if value.requires_grad
                ),
            }
            for name, head in heads.items()
        },
    }


def _jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _write_bundle(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    logits: Mapping[str, Sequence[np.ndarray]],
    image_names: Sequence[str],
    head_payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        (temporary / "summary.json").write_text(
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
        _jsonl_write(temporary / "training_history.jsonl", history)
        _jsonl_write(temporary / "oracle_targets.jsonl", oracle_rows)
        _jsonl_write(temporary / "crossfit_targets.jsonl", target_rows)
        _jsonl_write(temporary / "crossfit_images.jsonl", image_rows)
        _jsonl_write(temporary / "crossfit_calibration.jsonl", calibration_rows)
        arrays: dict[str, Any] = {
            "image_names": np.asarray(tuple(image_names), dtype=np.str_),
        }
        for name in VARIANT_ORDER:
            arrays[name] = np.stack(logits[name]).astype(np.float32, copy=False)
        np.savez_compressed(temporary / "test_logits.npz", **arrays)
        torch.save(dict(head_payload), temporary / "probe_heads.pkl")
        artifact_hashes = {
            name: legacy_probe.sha256_file(temporary / name)
            for name in BUNDLE_FILES[:-1]
        }
        complete_provenance = {**dict(provenance), "artifact_sha256": artifact_hashes}
        (temporary / "provenance.json").write_text(
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
        observed = {path.name for path in temporary.iterdir()}
        if observed != set(BUNDLE_FILES):
            raise FullTrainTestProbeError(
                f"temporary bundle inventory drifted: {sorted(observed)}"
            )
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def revalidate_external_contract(
    *,
    batch_dir: Path,
    dataset: str,
    seed: int,
    front_freeze_dir: Path,
    sources_before: Mapping[str, str],
    front_before: Mapping[str, Any],
    job_before: Mapping[str, Any],
    validated_job_before: Mapping[str, Any],
    manifest_before: Mapping[str, Any],
    baseline_artifacts_before: Mapping[str, Mapping[str, str]],
    run_config_before: Mapping[str, Any],
    canonical_data_before: Mapping[str, Any],
    checkpoint_before: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed if any evidence authority changed during the long run."""

    sources_after = _source_hashes()
    _require_unchanged("probe source hashes", dict(sources_before), sources_after)

    front_after = legacy_probe.validate_front_freeze_bundle(front_freeze_dir)
    _require_unchanged("front-freeze authority", dict(front_before), front_after)

    job_after, validated_job_after, manifest_after = validate_selected_baseline(
        batch_dir, dataset, seed
    )
    _require_unchanged("baseline manifest", dict(manifest_before), manifest_after)
    _require_unchanged("baseline job", dict(job_before), job_after)
    _require_unchanged(
        "validated baseline selection",
        dict(validated_job_before),
        validated_job_after,
    )

    baseline_artifacts_after = _baseline_artifact_records(batch_dir, job_after)
    _require_unchanged(
        "baseline artifact hashes",
        dict(baseline_artifacts_before),
        baseline_artifacts_after,
    )
    _, run_config_after = _read_locked_run_config(
        Path(str(job_after["run_dir"]))
    )
    _require_unchanged(
        "baseline run_config",
        dict(run_config_before),
        run_config_after,
    )

    canonical_data_after = _canonical_data_record(
        audit_canonical_dataset(str(job_after["dataset_dir"]))
    )
    _require_unchanged(
        "canonical train/test splits",
        dict(canonical_data_before),
        canonical_data_after,
    )

    selected_after = validated_job_after.get("best_iou")
    if not isinstance(selected_after, dict) or selected_after.get("status") != "found":
        raise FullTrainTestProbeError(
            "ending baseline validation lost the best-IoU selection"
        )
    checkpoint_after = _plain_file_record(
        Path(str(selected_after.get("checkpoint", ""))), "selected checkpoint"
    )
    expected_checkpoint = {
        "path": str(Path(str(checkpoint_before["path"])).resolve()),
        "sha256": str(checkpoint_before["sha256"]),
    }
    _require_unchanged(
        "selected checkpoint", expected_checkpoint, checkpoint_after
    )

    return {
        "status": "passed",
        "timing": "immediately_before_atomic_bundle_write",
        "checks": [
            "manifest_path_and_hash",
            "job_result_path_and_hash",
            "run_config_path_hash_and_locked_256_sizes",
            "protocol_summary_path_and_hash",
            "front_freeze_validation_and_hashes",
            "canonical_and_persisted_train_test_split_hashes",
            "selected_checkpoint_path_hash_and_selection_binding",
            "all_source_hashes",
        ],
        "manifest": dict(manifest_after),
        "baseline_artifact_sha256": {
            name: value["sha256"]
            for name, value in baseline_artifacts_after.items()
        },
        "run_config": run_config_after,
        "checkpoint": checkpoint_after,
        "front_freeze": {
            key: front_after[key]
            for key in (
                "summary_sha256",
                "provenance_sha256",
                "hard_core_source_sha256",
            )
        },
        "canonical_split_sha256": {
            "train_raw": canonical_data_after["train_raw_sha256"],
            "train_normalized": canonical_data_after[
                "train_normalized_sha256"
            ],
            "test_raw": canonical_data_after["test_raw_sha256"],
            "test_normalized": canonical_data_after[
                "test_normalized_sha256"
            ],
        },
        "source_sha256": sources_after,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.num_workers < 0:
        raise FullTrainTestProbeError("num_workers must be non-negative")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as exc:
        raise FullTrainTestProbeError(f"invalid device: {args.device}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FullTrainTestProbeError("CUDA requested but unavailable")
    epochs = protocol_epochs(args.protocol, args.dataset, int(args.seed))
    sources_before = _source_hashes()
    rng = _canonicalize_rng_roles(
        legacy_probe.seed_everything(int(args.seed)), seed=int(args.seed)
    )

    front_freeze_dir = _resolve(args.front_freeze_dir)
    front = legacy_probe.validate_front_freeze_bundle(front_freeze_dir)
    batch_dir = ROOT / "repro_runs" / "test_selected" / args.batch_id
    job, validated_job, manifest_record = validate_selected_baseline(
        batch_dir, args.dataset, int(args.seed)
    )
    baseline_artifacts = _baseline_artifact_records(batch_dir, job)
    train_dataset, test_dataset, data_record, run_config_record = (
        build_full_train_test_datasets(job)
    )
    if {
        key: run_config_record[key] for key in ("path", "sha256")
    } != baseline_artifacts["run_config"]:
        raise FullTrainTestProbeError(
            "locked run_config record disagrees with baseline artifact inventory"
        )
    authority, authority_record = build_test_authority(
        test_dataset,
        dataset_name=args.dataset,
        num_workers=args.num_workers,
        seed=int(args.seed) + 1,
    )

    selection = validated_job["best_iou"]
    checkpoint_path = Path(str(selection["checkpoint"])).resolve()
    checkpoint = load_checkpoint_cpu(checkpoint_path)
    state = checkpoint.get("net")
    if not isinstance(state, Mapping) or not state:
        raise FullTrainTestProbeError("selected checkpoint has no network state")
    model = MSHNet(3)
    model.load_state_dict(_normalize_state_dict(state), strict=True)
    del checkpoint, state
    model.requires_grad_(False).to(device).eval()
    backbone_before = legacy_probe._module_state_sha256(model)
    bn_before, bn_count = legacy_probe._batchnorm_state_sha256(model)

    heads = legacy_probe.build_probe_heads(int(args.seed), device)
    parameter_counts = _initial_parameter_counts(model, heads)
    original_output0_hash = legacy_probe._module_state_sha256(model.output_0)
    original_final_hash = legacy_probe._state_sha256(
        {
            f"{module_name}.{key}": value
            for module_name in ("output_0", "output_1", "output_2", "output_3", "final")
            for key, value in getattr(model, module_name).state_dict().items()
        }
    )

    train_loader = _loader(
        train_dataset,
        training=True,
        num_workers=args.num_workers,
        device=device,
        seed=int(args.seed),
    )
    history, training_protocol = legacy_probe.train_probe_heads(
        model,
        heads,
        train_loader,
        device=device,
        epochs=epochs,
    )
    del train_loader
    if any(int(row["images"]) != len(train_dataset) for row in history):
        raise FullTrainTestProbeError("an epoch did not expose every train image")
    training_protocol = {
        **training_protocol,
        "drop_last": False,
        "complete_train_manifest_exposed_each_epoch": True,
        "train_images_per_epoch": len(train_dataset),
        "augmentation": (
            "IRSTD_Dataset(mode=train) on complete canonical train manifest; "
            "identical shared batch across controls"
        ),
    }
    backbone_after_training = legacy_probe._module_state_sha256(model)
    bn_after_training, _ = legacy_probe._batchnorm_state_sha256(model)
    if backbone_after_training != backbone_before or bn_after_training != bn_before:
        raise FullTrainTestProbeError("frozen MSHNet/BatchNorm changed during fit")

    test_loader = _loader(
        test_dataset,
        training=False,
        num_workers=args.num_workers,
        device=device,
        seed=int(args.seed) + 1,
    )
    logits, targets = legacy_probe.infer_development(
        model,
        heads,
        test_loader,
        test_dataset,
        authority,
        dataset_name=args.dataset,
        device=device,
    )
    del test_loader
    backbone_after_inference = legacy_probe._module_state_sha256(model)
    bn_after_inference, _ = legacy_probe._batchnorm_state_sha256(model)
    if backbone_after_inference != backbone_before or bn_after_inference != bn_before:
        raise FullTrainTestProbeError("frozen MSHNet/BatchNorm changed during test")

    checkpoint_record = {
        "policy": "test_selected_best_iou",
        "selection_split": "canonical_test",
        "test_interval_completed_epochs": 10,
        "path": str(checkpoint_path),
        "sha256": selection["checkpoint_sha256"],
        "epoch_zero_based": int(selection["epoch_zero_based"]),
        "completed_epoch": int(selection["completed_epoch"]),
        "job_id": job["job_id"],
        "selected_iou": float(selection["iou"]),
        "selected_pd": float(selection["pd"]),
        "selected_fa_per_mpix": float(selection["fa"]),
    }
    evaluations: dict[str, Any] = {}
    oracle_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    names = tuple(test_dataset.names)
    for variant in VARIANT_ORDER:
        evaluation, variant_oracle, variant_targets, variant_images, variant_calibration = (
            legacy_probe.evaluate_variant(
                logits[variant],
                targets,
                names,
                authority,
                dataset=args.dataset,
                seed=int(args.seed),
                variant=variant,
                checkpoint_record=checkpoint_record,
            )
        )
        # The reusable evaluator predates this test-selected protocol and calls
        # the evaluated set "development" in two descriptive fields.  Persist
        # truthful role labels at the new bundle boundary.
        evaluation["canonical_test_logit_sha256"] = evaluation.pop(
            "development_logit_sha256"
        )
        for row in variant_oracle:
            row["semantics"] = (
                "same-canonical-test-set post-hoc Q2 oracle; test-selected diagnostic"
            )
            row["evaluation_split"] = "canonical_test"
            row["test_selected"] = True
        for rows in (variant_targets, variant_images, variant_calibration):
            for row in rows:
                row["evaluation_split"] = "canonical_test"
                row["test_selected"] = True
        evaluations[variant] = evaluation
        oracle_rows.extend(variant_oracle)
        target_rows.extend(variant_targets)
        image_rows.extend(variant_images)
        calibration_rows.extend(variant_calibration)

    trained_hashes = {
        name: legacy_probe._module_state_sha256(head)
        for name, head in heads.items()
    }
    if trained_hashes != training_protocol["final_state_sha256"]:
        raise FullTrainTestProbeError("trained probe state drifted before save")
    if legacy_probe.sha256_file(checkpoint_path) != selection["checkpoint_sha256"]:
        raise FullTrainTestProbeError("baseline checkpoint changed during probe")

    native_replay = build_native_selected_checkpoint_replay(
        logits["original_final_z"], targets, checkpoint_record
    )
    fixed_native = evaluations["original_final_z"]["fixed_logit0_pixel"]
    replay_pixel = native_replay["integer_pixel_counts"]
    for key in (
        "intersection_pixels",
        "union_pixels",
        "prediction_pixels",
        "target_pixels",
    ):
        if int(fixed_native[key]) != int(replay_pixel[key]):
            raise FullTrainTestProbeError(
                "native replay disagrees with original_final_z fixed-threshold counts"
            )
    if float(fixed_native["iou"]) != float(native_replay["iou"]):
        raise FullTrainTestProbeError(
            "native replay disagrees with original_final_z fixed-threshold IoU"
        )

    variants = {
        name: {
            **parameter_counts[name],
            "training": (
                "none"
                if name in legacy_probe.FROZEN_VARIANTS
                else "common_full_train_protocol"
            ),
            "state_sha256": (
                original_final_hash
                if name == "original_final_z"
                else (
                    original_output0_hash
                    if name == "original_output0"
                    else trained_hashes[name]
                )
            ),
            **evaluations[name],
        }
        for name in VARIANT_ORDER
    }
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": (
            "complete canonical train probe fit; canonical test evaluation and "
            "calibration; no validation split; explicitly test selected"
        ),
        "dataset": args.dataset,
        "seed": int(args.seed),
        "protocol": args.protocol,
        "test_selected": True,
        "untouched_test_claim": False,
        "checkpoint": checkpoint_record,
        "train_images": len(train_dataset),
        "validation_images": 0,
        "test_images": len(test_dataset),
        "test_targets": authority_record["target_count"],
        "spatial_contract": {
            "base_size": LOCKED_BASE_SIZE,
            "crop_size": LOCKED_CROP_SIZE,
            "run_config": run_config_record,
        },
        "native_selected_checkpoint_replay": native_replay,
        "variant_order": list(VARIANT_ORDER),
        "variants": variants,
        "training_protocol": training_protocol,
        "scientific_boundary": {
            "diagnostic_only": True,
            "readout_is_not_a_candidate_module": True,
            "readout_is_not_a_paper_contribution": True,
            "same_test_q2_oracle_is_not_deployable_performance": True,
            "crossfit_is_test_selected": True,
            "may_only_authorize_or_reject_a_shared_coordinate_convention": True,
        },
    }
    head_payload = {
        "schema": SCHEMA,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "checkpoint": checkpoint_record,
        "training_protocol": training_protocol,
        "variant_order": list(VARIANT_ORDER),
        "state_dict": {
            name: {
                key: value.detach().cpu()
                for key, value in head.state_dict().items()
            }
            for name, head in heads.items()
        },
    }

    end_revalidation = revalidate_external_contract(
        batch_dir=batch_dir,
        dataset=args.dataset,
        seed=int(args.seed),
        front_freeze_dir=front_freeze_dir,
        sources_before=sources_before,
        front_before=front,
        job_before=job,
        validated_job_before=validated_job,
        manifest_before=manifest_record,
        baseline_artifacts_before=baseline_artifacts,
        run_config_before=run_config_record,
        canonical_data_before=data_record,
        checkpoint_before=checkpoint_record,
    )
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "protocol": args.protocol,
        "device": str(device),
        "runtime": legacy_probe._runtime_versions(),
        "rng": rng,
        "baseline_manifest": manifest_record,
        "baseline_job": job,
        "baseline_artifacts": baseline_artifacts,
        "baseline_run_config": run_config_record,
        "validated_baseline_selection": validated_job,
        "checkpoint": checkpoint_record,
        "native_selected_checkpoint_replay": native_replay,
        "canonical_data": data_record,
        "test_target_authority": authority_record,
        "front_freeze": front,
        "source_sha256": sources_before,
        "source_specific_hashes_unchanged": True,
        "end_revalidation": end_revalidation,
        "variant_order": list(VARIANT_ORDER),
        "freeze_audit": {
            "model_eval_for_all_train_and_test_forwards": True,
            "model_requires_grad_false": True,
            "d0_extracted_under_no_grad": True,
            "shared_d0_once_per_train_batch": True,
            "backbone_state_sha256_before": backbone_before,
            "backbone_state_sha256_after_training": backbone_after_training,
            "backbone_state_sha256_after_inference": backbone_after_inference,
            "batchnorm_module_count": bn_count,
            "batchnorm_state_sha256_before": bn_before,
            "batchnorm_state_sha256_after_training": bn_after_training,
            "batchnorm_state_sha256_after_inference": bn_after_inference,
        },
        "data_access": {
            "complete_canonical_train_dataset_constructed_and_iterated": True,
            "validation_dataset_constructed": False,
            "validation_sample_iterated": False,
            "complete_canonical_test_dataset_constructed_and_iterated": True,
            "test_used_for_checkpoint_selection": True,
            "test_used_for_diagnostic_calibration_and_evaluation": True,
            "untouched_test_claim": False,
        },
        "test_logit_shapes": {
            name: list(np.stack(values).shape) for name, values in logits.items()
        },
    }
    _write_bundle(
        output_dir,
        summary=summary,
        history=history,
        oracle_rows=oracle_rows,
        target_rows=target_rows,
        image_rows=image_rows,
        calibration_rows=calibration_rows,
        logits=logits,
        image_names=names,
        head_payload=head_payload,
        provenance=provenance,
    )
    del heads, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary, output_dir = run(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "dataset": summary["dataset"],
                "seed": summary["seed"],
                "test_selected": summary["test_selected"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
