#!/usr/bin/env python3
"""Build the fail-closed Gate E-1 cross-seed target persistence ledger.

The primary policy is ``fixed_epoch``: every clean MSHNet run is evaluated at
epoch 399 from ``checkpoint.pkl``.  ``best_iou`` is an explicitly separate,
retrospective sensitivity policy.  Both policies use the canonical 256 x 256
validation masks, strict ``logit > 0`` decisions, 8-connected components, and
maximum-cardinality/minimum-distance Hungarian matching with a strict
three-pixel centroid radius.

The program performs one full-graph inference pass per run and emits one row
for every ground-truth component, including matched targets.  It never opens
or evaluates an official test split.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import re
from argparse import Namespace
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.MSHNet import MSHNet  # noqa: E402
from model.mshnet_checkpoint import strip_legacy_dea_lite_head  # noqa: E402
from tools.finalize_clean_baselines import (  # noqa: E402
    DATASET_NAMES,
    EXPECTED_EPOCHS,
    FinalizationError,
    load_checkpoint_cpu,
    normalized_path,
    parse_metrics,
    read_json,
    require_mapping,
    validate_manifest,
    validate_result,
)
from utils.component_ledger import build_component_ledger  # noqa: E402
from utils.cross_seed_persistence import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    EXPECTED_SEED_COUNT,
    summarize_cross_seed_persistence,
)
from utils.data import IRSTD_Dataset  # noqa: E402
from utils.metric import (  # noqa: E402
    match_components_hungarian,
    match_connected_components,
)
from utils.target_identity import (  # noqa: E402
    StableTargetSet,
    assert_same_target_set,
    build_stable_target_set,
)


SCHEMA = "dea.gate_e.cross_seed_failure_persistence.v2"
PROVENANCE_SCHEMA = "dea.gate_e.cross_seed_failure_persistence.provenance.v2"
POLICIES = ("fixed_epoch", "best_iou")
POLICY_FILES = {
    "fixed_epoch": "checkpoint.pkl",
    "best_iou": "checkpoint_best_iou.pkl",
}
THRESHOLD_LOGIT = 0.0
THRESHOLD_OPERATOR = ">"
CENTROID_RADIUS = 3.0
CONNECTIVITY = 2
CANONICAL_SIZE = 256
OFFICIAL_TEST_POLICY = (
    "sealed: paths and hashes are provenance-only; official test images and "
    "masks are never opened or iterated"
)
EXPECTED_RECIPE = {
    "epochs": EXPECTED_EPOCHS,
    "batch_size": 4,
    "num_workers": 4,
    "lr": 0.05,
    "warm_epoch": 5,
    "val_fraction": 0.2,
    "split_seed": 20260711,
    "deterministic": "true",
}
OUTPUT_FILES = (
    "target_persistence.jsonl",
    "target_persistence_summary.json",
    "target_persistence_summary.md",
    "provenance.json",
)
SAFE_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PROTOCOL_DOCUMENTS = {
    "gate_e_protocol": ROOT
    / "MSHNet_Gate_D_NoGo_and_Gate_E_Training_Credit_Audit_Plan.md",
    "north_star_positioning": ROOT
    / "MSHNet_North_Star_Objective_and_Gate_E_Positioning.md",
}


class PersistenceAuditError(RuntimeError):
    """Raised when any frozen input or output invariant fails."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit target-level miss recurrence across the frozen 3x3 clean "
            "MSHNet development-holdout grid."
        )
    )
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument(
        "--checkpoint-policy",
        choices=POLICIES,
        default="fixed_epoch",
        help=(
            "fixed_epoch is the primary epoch-399 analysis; best_iou is a "
            "retrospective sensitivity analysis and is never pooled with it"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "default: repro_runs/gate_e/persistence_v2/<checkpoint-policy>; "
            "an existing path is always refused"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    return parser.parse_args(argv)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_document_fingerprints() -> dict[str, dict[str, Any]]:
    """Capture the pilot-visible protocol freeze before model inference."""

    records: dict[str, dict[str, Any]] = {}
    for name, path in PROTOCOL_DOCUMENTS.items():
        if not path.is_file():
            raise PersistenceAuditError(f"missing protocol document: {path}")
        stat = path.stat()
        records[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "mtime_ns": int(stat.st_mtime_ns),
            "mtime_utc": dt.datetime.fromtimestamp(
                stat.st_mtime, tz=dt.timezone.utc
            ).isoformat(),
        }
    return records


def validate_protocol_documents_unchanged(
    frozen: Mapping[str, Mapping[str, Any]],
) -> None:
    """Refuse output if either protocol changed after the run-start freeze."""

    observed = protocol_document_fingerprints()
    if set(frozen) != set(observed):
        raise PersistenceAuditError("protocol document inventory changed during audit")
    for name in sorted(observed):
        for field in ("path", "sha256", "mtime_ns"):
            if frozen[name].get(field) != observed[name].get(field):
                raise PersistenceAuditError(
                    f"protocol document changed during audit: {name}.{field}"
                )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PersistenceAuditError(f"{label} must be a JSON/object mapping")
    return value


def _command_option(command: Any, option: str) -> str | None:
    """Return one exact CLI option value from a recorded job command."""

    if not isinstance(command, list) or any(
        not isinstance(item, str) for item in command
    ):
        raise PersistenceAuditError("job result.command must be a list of strings")
    positions = [index for index, item in enumerate(command) if item == option]
    if len(positions) > 1:
        raise PersistenceAuditError(f"job command repeats {option}")
    if not positions:
        return None
    position = positions[0]
    if position + 1 >= len(command) or command[position + 1].startswith("--"):
        raise PersistenceAuditError(f"job command option {option} has no value")
    return command[position + 1]


def resume_evidence_from_training_log(
    path: str | Path,
    *,
    resume_requested: bool,
) -> dict[str, Any]:
    """Seal log evidence for the two known resumed completion commands."""

    log_path = Path(path).expanduser().resolve()
    if not log_path.is_file():
        raise PersistenceAuditError(f"missing training log: {log_path}")
    evidence: dict[str, Any] = {
        "training_log": str(log_path),
        "training_log_sha256": sha256_file(log_path),
        "resume_requested_by_recorded_command": bool(resume_requested),
        "resume_epoch": None,
        "resume_from_completed_checkpoint_epoch": None,
        "evidence": (
            "recorded completion command did not request resume"
            if not resume_requested
            else "pending training-log boundary validation"
        ),
    }
    if not resume_requested:
        return evidence
    try:
        text = log_path.read_text(encoding="utf-8", errors="strict").replace(
            "\r", "\n"
        )
    except (OSError, UnicodeError) as exc:
        raise PersistenceAuditError(f"cannot read resumed training log: {exc}") from exc
    starts = list(re.finditer(r"split train:", text))
    if len(starts) != 2:
        raise PersistenceAuditError(
            f"resumed training log must contain exactly two process starts, got {len(starts)}"
        )
    resumed_text = text[starts[1].start() :]
    first_epoch = re.search(r"Epoch ([0-9]+), loss ", resumed_text)
    if first_epoch is None:
        raise PersistenceAuditError("resumed training log lacks first resumed epoch")
    resume_epoch = int(first_epoch.group(1))
    if not 1 <= resume_epoch < EXPECTED_EPOCHS:
        raise PersistenceAuditError("resumed epoch is outside the expected training range")
    evidence.update(
        {
            "process_start_marker_count": len(starts),
            "resume_epoch": resume_epoch,
            "resume_from_completed_checkpoint_epoch": resume_epoch - 1,
            "evidence": (
                "recorded command requested resume and the appended second process "
                f"starts training at epoch {resume_epoch}"
            ),
        }
    )
    return evidence


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersistenceAuditError(f"{label} must be an integer")
    return int(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PersistenceAuditError(f"{label} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PersistenceAuditError(f"{label} must be finite numeric") from exc
    if not math.isfinite(result):
        raise PersistenceAuditError(f"{label} must be finite numeric")
    return result


def _same_logged_metric(left: Any, right: Any) -> bool:
    try:
        return f"{float(left):.4f}" == f"{float(right):.4f}"
    except (TypeError, ValueError):
        return False


def _require_policy(policy: str) -> Literal["fixed_epoch", "best_iou"]:
    if policy not in POLICIES:
        raise PersistenceAuditError(
            f"checkpoint policy must be one of {POLICIES}, got {policy!r}"
        )
    return policy  # type: ignore[return-value]


def validate_adagrad_optimizer_state(value: Any) -> dict[str, Any]:
    """Validate and summarize the exact dense Adagrad checkpoint state."""

    optimizer = _mapping(value, "checkpoint.optimizer")
    state = optimizer.get("state")
    groups = optimizer.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(groups, list):
        raise PersistenceAuditError(
            "checkpoint optimizer must contain state mapping and param_groups list"
        )
    if len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise PersistenceAuditError("baseline Adagrad must have exactly one param group")
    group = groups[0]
    expected_hparams = {
        "lr": 0.05,
        "lr_decay": 0.0,
        "eps": 1e-10,
        "weight_decay": 0.0,
        "initial_accumulator_value": 0.0,
        "maximize": False,
        "differentiable": False,
    }
    for name, expected in expected_hparams.items():
        observed = group.get(name)
        if isinstance(expected, bool):
            if observed is not expected:
                raise PersistenceAuditError(
                    f"Adagrad param-group {name}={observed!r}, expected {expected!r}"
                )
        elif not math.isclose(
            _finite(observed, f"Adagrad param-group {name}"),
            expected,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise PersistenceAuditError(
                f"Adagrad param-group {name}={observed!r}, expected {expected!r}"
            )
    if group.get("foreach") is not None or group.get("fused") is not None:
        raise PersistenceAuditError("baseline Adagrad foreach/fused settings drifted")
    params = group.get("params")
    if not isinstance(params, list) or not params or any(
        isinstance(item, bool) or not isinstance(item, int) for item in params
    ):
        raise PersistenceAuditError("Adagrad param-group params must be integer ids")
    if len(set(params)) != len(params):
        raise PersistenceAuditError("Adagrad param ids are duplicated")
    if set(state) != set(params):
        raise PersistenceAuditError("Adagrad state keys do not equal param-group ids")

    step_counts: Counter[int] = Counter()
    accumulator_dtypes: Counter[str] = Counter()
    for param_id in params:
        record = state[param_id]
        if not isinstance(record, Mapping) or set(record) != {"step", "sum"}:
            raise PersistenceAuditError(
                f"Adagrad state {param_id} must contain exactly step and sum"
            )
        step = record["step"]
        accumulator = record["sum"]
        if not torch.is_tensor(step) or step.numel() != 1 or not torch.isfinite(step):
            raise PersistenceAuditError(f"Adagrad state {param_id}.step is invalid")
        step_value = float(step.item())
        if step_value < 0.0 or not step_value.is_integer():
            raise PersistenceAuditError(f"Adagrad state {param_id}.step is invalid")
        if not torch.is_tensor(accumulator) or accumulator.numel() < 1:
            raise PersistenceAuditError(f"Adagrad state {param_id}.sum is invalid")
        if not bool(torch.isfinite(accumulator).all()) or bool(
            (accumulator < 0).any()
        ):
            raise PersistenceAuditError(
                f"Adagrad state {param_id}.sum must be finite and non-negative"
            )
        step_counts[int(step_value)] += 1
        accumulator_dtypes[str(accumulator.dtype)] += 1
    return {
        "optimizer": "Adagrad",
        "param_group_count": 1,
        "parameter_state_count": len(params),
        "hyperparameters": {
            name: group.get(name) for name in expected_hparams
        },
        "foreach": group.get("foreach"),
        "fused": group.get("fused"),
        "state_fields": ["step", "sum"],
        "step_distribution": {
            str(step): count for step, count in sorted(step_counts.items())
        },
        "accumulator_dtype_counts": dict(sorted(accumulator_dtypes.items())),
    }


def validate_checkpoint_for_policy(
    checkpoint: Mapping[str, Any],
    *,
    policy: Literal["fixed_epoch", "best_iou"],
    job: Mapping[str, Any],
    dataset_meta: Mapping[str, Any],
    manifest_args: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one metadata checkpoint without deserializing another file.

    This function is deliberately independent of CUDA and filesystem access so
    policy invariants can be fixture-tested.  Logged metrics are compared at
    their recorded four-decimal precision.
    """

    policy = _require_policy(policy)
    if not isinstance(checkpoint, Mapping):
        raise PersistenceAuditError("checkpoint must be a mapping")
    if len(metric_rows) != EXPECTED_EPOCHS:
        raise PersistenceAuditError(
            f"metric log must contain exactly {EXPECTED_EPOCHS} rows"
        )
    epochs = []
    for index, row in enumerate(metric_rows):
        if not isinstance(row, Mapping):
            raise PersistenceAuditError(f"metric row {index} must be a mapping")
        epochs.append(_exact_int(row.get("epoch"), f"metric row {index}.epoch"))
    if epochs != list(range(EXPECTED_EPOCHS)):
        raise PersistenceAuditError(
            f"metric epochs must be exactly 0..{EXPECTED_EPOCHS - 1}"
        )

    metadata = _mapping(checkpoint.get("method_meta"), "checkpoint.method_meta")
    expected_metadata = {
        "model_type": "mshnet",
        "seed": job.get("seed"),
        "run_label": job.get("job_id"),
        "split_seed": manifest_args.get("split_seed"),
        "train_split_sha256": dataset_meta.get("fit_sha256"),
        "val_split_sha256": dataset_meta.get("val_sha256"),
        "test_split_sha256": dataset_meta.get("official_test_sha256"),
    }
    errors = [
        f"{key}: checkpoint={metadata.get(key)!r} expected={expected!r}"
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    ]
    method = metadata.get("method")
    if not isinstance(method, str) or method.casefold() != "mshnet":
        errors.append(f"method must be MSHNet, got {method!r}")
    if metadata.get("deterministic") is not True:
        errors.append("deterministic metadata is not true")

    epoch = _exact_int(checkpoint.get("epoch"), "checkpoint.epoch")
    if policy == "fixed_epoch":
        if epoch != EXPECTED_EPOCHS - 1:
            errors.append(
                f"fixed_epoch requires epoch {EXPECTED_EPOCHS - 1}, got {epoch}"
            )
    elif not 0 <= epoch < EXPECTED_EPOCHS:
        errors.append(f"best_iou epoch is outside 0..{EXPECTED_EPOCHS - 1}")

    if not 0 <= epoch < len(metric_rows):
        raise PersistenceAuditError("checkpoint epoch cannot index metric rows")
    logged = metric_rows[epoch]
    metrics = {
        name: _finite(checkpoint.get(name), f"checkpoint.{name}")
        for name in ("iou", "pd", "fa")
    }
    if not 0.0 <= metrics["iou"] <= 1.0:
        errors.append("checkpoint IoU is outside [0,1]")
    if not 0.0 <= metrics["pd"] <= 1.0:
        errors.append("checkpoint Pd is outside [0,1]")
    if metrics["fa"] < 0.0:
        errors.append("checkpoint FA is negative")
    for name, value in metrics.items():
        if not _same_logged_metric(value, logged.get(name)):
            errors.append(
                f"checkpoint {name} disagrees with metric row at epoch {epoch}"
            )

    best_iou = _finite(checkpoint.get("best_iou"), "checkpoint.best_iou")
    logged_best = max(_finite(row.get("iou"), "metric row.iou") for row in metric_rows)
    if not _same_logged_metric(best_iou, logged_best):
        errors.append("checkpoint best_iou disagrees with the 400-row maximum")
    if policy == "best_iou" and not _same_logged_metric(metrics["iou"], logged_best):
        errors.append("best_iou checkpoint is not at the logged best IoU")

    state = checkpoint.get("net")
    if not isinstance(state, Mapping) or not state:
        errors.append("checkpoint net is not a non-empty state dict")
    elif not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state.items()
    ):
        errors.append("checkpoint net contains a non-string key or non-tensor value")
    optimizer_summary = validate_adagrad_optimizer_state(
        checkpoint.get("optimizer")
    )
    if errors:
        raise PersistenceAuditError(
            f"checkpoint policy validation failed for {job.get('job_id')}: "
            + "; ".join(errors)
        )
    return {
        "policy": policy,
        "filename": POLICY_FILES[policy],
        "epoch": epoch,
        "metrics": metrics,
        "best_iou_over_run": best_iou,
        "optimizer": optimizer_summary,
        "selection_scope": (
            "fixed_training_duration_primary"
            if policy == "fixed_epoch"
            else "retrospective_best_validation_iou_sensitivity"
        ),
    }


def validate_grid_cardinality(
    jobs: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = DATASET_NAMES,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
) -> tuple[int, ...]:
    """Fail unless jobs form exactly datasets x three identical seeds."""

    if isinstance(jobs, (str, bytes)) or not isinstance(jobs, Sequence):
        raise PersistenceAuditError("jobs must be a sequence")
    pairs: list[tuple[str, int]] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            raise PersistenceAuditError(f"job {index} must be a mapping")
        dataset = job.get("dataset")
        if dataset not in datasets:
            raise PersistenceAuditError(f"job {index} has unknown dataset {dataset!r}")
        seed = _exact_int(job.get("seed"), f"job {index}.seed")
        pairs.append((str(dataset), seed))
    if len(pairs) != len(set(pairs)):
        raise PersistenceAuditError("duplicate dataset/seed job")
    seeds = tuple(sorted({seed for _, seed in pairs}))
    if len(seeds) != expected_seed_count:
        raise PersistenceAuditError(
            f"exactly {expected_seed_count} distinct seeds are required"
        )
    expected = {(dataset, seed) for dataset in datasets for seed in seeds}
    if set(pairs) != expected or len(pairs) != len(expected):
        raise PersistenceAuditError(
            "jobs do not form the complete 3-dataset x 3-seed grid"
        )
    return seeds


def _validate_run_config(
    config: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    dataset_meta: Mapping[str, Any],
    manifest_args: Mapping[str, Any],
) -> dict[str, Any]:
    args = _mapping(config.get("args"), f"run_config {job['job_id']}.args")
    expected = {
        "mode": "train",
        "model_type": "mshnet",
        "deterministic": True,
        "pin_memory": True,
        "epochs": EXPECTED_EPOCHS,
        "base_size": CANONICAL_SIZE,
        "crop_size": CANONICAL_SIZE,
        "batch_size": manifest_args.get("batch_size"),
        "num_workers": manifest_args.get("num_workers"),
        "lr": manifest_args.get("lr"),
        "warm_epoch": manifest_args.get("warm_epoch"),
        "seed": job.get("seed"),
        "run_label": job.get("job_id"),
        "split_seed": manifest_args.get("split_seed"),
        "train_split_file": job.get("train_file"),
        "test_split_file": job.get("test_file"),
        "val_split_file": "",
        "train_split_sha256": dataset_meta.get("fit_sha256"),
        "val_split_sha256": dataset_meta.get("val_sha256"),
        "test_split_sha256": dataset_meta.get("official_test_sha256"),
    }
    mismatches = [
        f"{key}: config={args.get(key)!r} expected={value!r}"
        for key, value in expected.items()
        if args.get(key) != value
    ]
    try:
        config_dataset = normalized_path(
            args.get("dataset_dir"), f"run_config {job['job_id']}.dataset_dir"
        )
        job_dataset = normalized_path(
            job.get("dataset_dir"), f"manifest {job['job_id']}.dataset_dir"
        )
        config_run = normalized_path(
            args.get("run_dir"), f"run_config {job['job_id']}.run_dir"
        )
        job_run = normalized_path(
            job.get("run_dir"), f"manifest {job['job_id']}.run_dir"
        )
    except FinalizationError as exc:
        raise PersistenceAuditError(str(exc)) from exc
    if config_dataset != job_dataset:
        mismatches.append("dataset_dir differs from manifest")
    if config_run != job_run:
        mismatches.append("run_dir differs from manifest")
    if mismatches:
        raise PersistenceAuditError(
            f"run_config mismatch for {job['job_id']}: " + "; ".join(mismatches)
        )
    return dict(args)


def load_validated_jobs(
    batch_dir: Path,
    *,
    policy: Literal["fixed_epoch", "best_iou"],
    checkpoint_loader: Callable[[Path], Mapping[str, Any]] = load_checkpoint_cpu,
    hash_file: Callable[[str | Path], str] = sha256_file,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate all frozen artifacts and return the exact nine audit jobs."""

    policy = _require_policy(policy)
    batch_dir = batch_dir.expanduser().resolve()
    manifest_path = batch_dir / "manifest.json"
    try:
        manifest = require_mapping(read_json(manifest_path, "clean manifest"), "manifest")
        seeds, datasets_meta = validate_manifest(manifest, batch_dir)
    except FinalizationError as exc:
        raise PersistenceAuditError(str(exc)) from exc
    if len(seeds) != EXPECTED_SEED_COUNT:
        raise PersistenceAuditError("clean manifest does not contain exactly three seeds")
    manifest_args = _mapping(manifest.get("args"), "manifest.args")
    recipe_errors = [
        f"{key}: manifest={manifest_args.get(key)!r} expected={expected!r}"
        for key, expected in EXPECTED_RECIPE.items()
        if manifest_args.get(key) != expected
    ]
    if recipe_errors:
        raise PersistenceAuditError(
            "clean baseline recipe drifted: " + "; ".join(recipe_errors)
        )

    jobs: list[dict[str, Any]] = []
    for raw_job in manifest["jobs"]:
        job = _mapping(raw_job, "manifest job")
        dataset = str(job["dataset"])
        dataset_meta = _mapping(
            datasets_meta[dataset], f"manifest.datasets.{dataset}"
        )
        try:
            job_dataset_dir = normalized_path(
                job.get("dataset_dir"), f"manifest {job['job_id']}.dataset_dir"
            )
            metadata_dataset_dir = normalized_path(
                dataset_meta.get("dataset_dir"),
                f"manifest.datasets.{dataset}.dataset_dir",
            )
        except FinalizationError as exc:
            raise PersistenceAuditError(str(exc)) from exc
        if job_dataset_dir != metadata_dataset_dir:
            raise PersistenceAuditError(
                f"manifest job/dataset directory mismatch for {job['job_id']}"
            )
        for job_field, metadata_field in (
            ("train_file", "train_file"),
            ("test_file", "test_file"),
        ):
            if job.get(job_field) != dataset_meta.get(metadata_field):
                raise PersistenceAuditError(
                    f"manifest job/dataset {job_field} mismatch for {job['job_id']}"
                )
        result_path = normalized_path(
            job.get("result_file"), f"manifest {job['job_id']}.result_file"
        )
        try:
            result = require_mapping(read_json(result_path, "job result"), "job result")
            validate_result(result, job)
        except FinalizationError as exc:
            raise PersistenceAuditError(str(exc)) from exc
        if _exact_int(result.get("returncode"), "job result.returncode") != 0:
            raise PersistenceAuditError(f"{job['job_id']} did not return zero")
        resume_value = _command_option(result.get("command"), "--if-checkpoint")
        if resume_value not in {None, "true", "false"}:
            raise PersistenceAuditError(
                f"{job['job_id']} has invalid --if-checkpoint value"
            )
        resume_requested = resume_value == "true"
        training_log = normalized_path(
            job.get("log_file"), f"manifest {job['job_id']}.log_file"
        )
        result_training_log = normalized_path(
            result.get("log_file"), f"job result {job['job_id']}.log_file"
        )
        if training_log != result_training_log:
            raise PersistenceAuditError(
                f"manifest/result training log mismatch for {job['job_id']}"
            )
        resume_evidence = resume_evidence_from_training_log(
            training_log,
            resume_requested=resume_requested,
        )

        run_dir = normalized_path(job.get("run_dir"), f"manifest {job['job_id']}.run_dir")
        try:
            metric_rows = parse_metrics(run_dir / "epoch_metric.log")
        except FinalizationError as exc:
            raise PersistenceAuditError(str(exc)) from exc
        config_path = run_dir / "run_config.json"
        try:
            run_config = require_mapping(
                read_json(config_path, "run_config"), "run_config"
            )
        except FinalizationError as exc:
            raise PersistenceAuditError(str(exc)) from exc
        stored_args = _validate_run_config(
            run_config,
            job=job,
            dataset_meta=dataset_meta,
            manifest_args=manifest_args,
        )

        checkpoint_path = run_dir / POLICY_FILES[policy]
        if checkpoint_path.name != POLICY_FILES[policy]:
            raise PersistenceAuditError("checkpoint policy/path mismatch")
        try:
            checkpoint = checkpoint_loader(checkpoint_path)
        except (FinalizationError, OSError, RuntimeError) as exc:
            raise PersistenceAuditError(
                f"cannot safely load checkpoint {checkpoint_path}: {exc}"
            ) from exc
        checkpoint_summary = validate_checkpoint_for_policy(
            checkpoint,
            policy=policy,
            job=job,
            dataset_meta=dataset_meta,
            manifest_args=manifest_args,
            metric_rows=metric_rows,
        )
        del checkpoint
        jobs.append(
            {
                "batch_id": batch_dir.name,
                "job_id": job["job_id"],
                "dataset": dataset,
                "seed": int(job["seed"]),
                "dataset_dir": str(normalized_path(job["dataset_dir"], "dataset_dir")),
                "run_dir": str(run_dir),
                "result_file": str(result_path),
                "result_sha256": hash_file(result_path),
                "metric_log": str((run_dir / "epoch_metric.log").resolve()),
                "metric_log_sha256": hash_file(run_dir / "epoch_metric.log"),
                "run_config": str(config_path.resolve()),
                "run_config_sha256": hash_file(config_path),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": hash_file(checkpoint_path),
                "checkpoint_summary": checkpoint_summary,
                "resume_requested_by_recorded_command": resume_requested,
                "resume_evidence": resume_evidence,
                "stored_args": stored_args,
                "split_hashes": {
                    "fit": dataset_meta["fit_sha256"],
                    "validation": dataset_meta["val_sha256"],
                    "official_test_provenance_only": dataset_meta[
                        "official_test_sha256"
                    ],
                },
            }
        )
    validated_seeds = validate_grid_cardinality(jobs)
    if tuple(sorted(seeds)) != validated_seeds:
        raise PersistenceAuditError("validated job seeds disagree with manifest seeds")
    jobs.sort(
        key=lambda item: (DATASET_NAMES.index(item["dataset"]), item["seed"])
    )
    provenance = {
        "batch_id": batch_dir.name,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hash_file(manifest_path),
        "datasets": list(DATASET_NAMES),
        "seeds": list(validated_seeds),
        "policy": policy,
    }
    return jobs, provenance


def _region_record(region: Any) -> dict[str, Any]:
    return {
        "index": None,
        "label": int(region.label),
        "area": int(region.area),
        "bbox": [int(value) for value in region.bbox],
        "centroid_y": float(region.centroid[0]),
        "centroid_x": float(region.centroid[1]),
    }


def build_image_envelope_row(
    target_set: StableTargetSet,
    *,
    seed: int,
    image_index: int,
    checkpoint: Mapping[str, Any],
    operating_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the registry-verifiable row retained for target-free images."""

    row = {
        "schema_version": SCHEMA,
        "row_kind": "image",
        "dataset": target_set.dataset,
        "image_name": target_set.image_name,
        "image_index": int(image_index),
        "seed": int(seed),
        "height": target_set.height,
        "width": target_set.width,
        "pixel_connectivity": target_set.pixel_connectivity,
        "skimage_connectivity": target_set.skimage_connectivity,
        "label_mask_sha256": target_set.label_mask_sha256,
        "target_count": len(target_set.targets),
        "checkpoint": {
            "policy": checkpoint["policy"],
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "epoch": checkpoint["epoch"],
        },
        "run": dict(_mapping(checkpoint.get("run"), "checkpoint.run")),
    }
    if operating_metrics is not None:
        row["operating_point"] = dict(operating_metrics)
    return row


def compute_image_operating_metrics(
    logits: object,
    target_mask: object,
) -> dict[str, Any]:
    """Return achieved logit-zero FA under both frozen matchers."""

    scores = np.asarray(logits)
    target = np.asarray(target_mask)
    if scores.shape != (CANONICAL_SIZE, CANONICAL_SIZE) or target.shape != scores.shape:
        raise PersistenceAuditError("operating metrics require aligned canonical arrays")
    if not bool(np.all(np.isfinite(scores))) or not bool(
        np.all((target == 0) | (target == 1))
    ):
        raise PersistenceAuditError("operating metric arrays are non-finite/non-binary")
    prediction = scores > THRESHOLD_LOGIT
    hungarian = match_components_hungarian(
        prediction,
        target,
        centroid_radius=CENTROID_RADIUS,
        connectivity=CONNECTIVITY,
    )
    legacy = match_connected_components(
        prediction,
        target,
        max_centroid_distance=CENTROID_RADIUS,
        connectivity=CONNECTIVITY,
    )
    pixels = int(scores.size)

    def matcher_record(component_match: Any) -> dict[str, Any]:
        unmatched_area = int(
            sum(
                component_match.prediction_regions[index].area
                for index in component_match.unmatched_prediction_indices
            )
        )
        return {
            "matched_target_components": len(component_match.matches),
            "unmatched_target_components": len(
                component_match.unmatched_target_indices
            ),
            "unmatched_prediction_components": len(
                component_match.unmatched_prediction_indices
            ),
            "unmatched_prediction_area": unmatched_area,
            "fa_per_million_pixels": unmatched_area / pixels * 1_000_000.0,
        }

    return {
        "threshold_logit": THRESHOLD_LOGIT,
        "threshold_operator": THRESHOLD_OPERATOR,
        "image_pixels": pixels,
        "target_components": len(hungarian.target_regions),
        "prediction_components": len(hungarian.prediction_regions),
        "hungarian": matcher_record(hungarian),
        "legacy": matcher_record(legacy),
    }


def build_image_target_rows(
    logits: object,
    target_mask: object,
    *,
    dataset: str,
    image_name: str,
    seed: int,
    image_index: int,
    checkpoint: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], StableTargetSet]:
    """Create the complete component ledger for one already inferred image."""

    scores = np.asarray(logits)
    target = np.asarray(target_mask)
    if scores.ndim != 2 or target.ndim != 2 or scores.shape != target.shape:
        raise PersistenceAuditError("logits and target must be aligned 2-D arrays")
    if scores.shape != (CANONICAL_SIZE, CANONICAL_SIZE):
        raise PersistenceAuditError(
            f"canonical audit requires {CANONICAL_SIZE}x{CANONICAL_SIZE} arrays"
        )
    if not bool(np.all(np.isfinite(scores))):
        raise PersistenceAuditError("logits contain non-finite values")
    if not bool(np.all((target == 0) | (target == 1))):
        raise PersistenceAuditError("canonical target mask must be exactly binary")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise PersistenceAuditError("seed must be an integer")
    if isinstance(image_index, bool) or not isinstance(image_index, (int, np.integer)):
        raise PersistenceAuditError("image_index must be an integer")

    target_bool = target.astype(bool, copy=False)
    target_set = build_stable_target_set(
        target_bool,
        dataset=dataset,
        image_name=image_name,
        connectivity=CONNECTIVITY,
    )
    prediction = scores > THRESHOLD_LOGIT
    component_match = match_components_hungarian(
        prediction,
        target_bool,
        centroid_radius=CENTROID_RADIUS,
        connectivity=CONNECTIVITY,
    )
    ledger = build_component_ledger(
        scores,
        target_bool,
        threshold=THRESHOLD_LOGIT,
        input_semantics="logits",
        centroid_radius=CENTROID_RADIUS,
        connectivity=CONNECTIVITY,
    )
    if len(component_match.target_regions) != len(target_set.targets):
        raise PersistenceAuditError("stable identity/metric target count mismatch")
    if tuple(component_match.unmatched_target_indices) != tuple(
        sorted(component_match.unmatched_target_indices)
    ):
        raise PersistenceAuditError("metric unmatched target indices are not ordered")
    if tuple(component_match.unmatched_target_indices) != tuple(
        index
        for index in range(len(component_match.target_regions))
        if index not in {match[0] for match in component_match.matches}
    ):
        raise PersistenceAuditError("metric matched/unmatched target partition drifted")

    identity_by_source = {
        target_id.source_component_index: target_id for target_id in target_set.targets
    }
    if set(identity_by_source) != set(range(len(component_match.target_regions))):
        raise PersistenceAuditError("stable identity source-index mapping is incomplete")
    match_by_target = {
        int(target_index): (int(prediction_index), float(distance))
        for target_index, prediction_index, distance in component_match.matches
    }
    no_response = set(ledger.no_response_target_indices)
    centroid_miss = set(ledger.centroid_miss_target_indices)
    unmatched = set(component_match.unmatched_target_indices)
    if no_response & centroid_miss or not (no_response | centroid_miss).issubset(unmatched):
        raise PersistenceAuditError("component miss taxonomy is not disjoint/subset")
    assignment_residual = unmatched - no_response - centroid_miss
    if not (
        no_response.isdisjoint(centroid_miss)
        and no_response.isdisjoint(assignment_residual)
        and centroid_miss.isdisjoint(assignment_residual)
        and no_response | centroid_miss | assignment_residual == unmatched
    ):
        raise PersistenceAuditError("miss subtype partition is not mutually exclusive/exhaustive")
    prediction_centroids = [
        np.asarray(region.centroid, dtype=np.float64)
        for region in component_match.prediction_regions
    ]
    for target_index, region in enumerate(component_match.target_regions):
        legal_edge = any(
            float(
                np.linalg.norm(
                    np.asarray(region.centroid, dtype=np.float64)
                    - prediction_centroid
                )
            )
            < CENTROID_RADIUS
            for prediction_centroid in prediction_centroids
        )
        support_near_count = int(ledger.pred_components_per_gt[target_index])
        # Frozen priority partition: lack of support-near response takes
        # precedence even in the pathological hollow-component case whose
        # centroid alone forms a legal edge.
        if target_index in no_response and support_near_count != 0:
            raise PersistenceAuditError("no_response subtype definition drifted")
        if target_index in centroid_miss and (
            support_near_count == 0 or legal_edge
        ):
            raise PersistenceAuditError("centroid_miss subtype definition drifted")
        if target_index in assignment_residual and (
            support_near_count == 0 or not legal_edge
        ):
            raise PersistenceAuditError("assignment_residual subtype definition drifted")

    rows: list[dict[str, Any]] = []
    for source_index, region in enumerate(component_match.target_regions):
        identity = identity_by_source[source_index]
        region_bbox = tuple(int(value) for value in region.bbox)
        if (
            int(region.area) != identity.area
            or region_bbox != identity.bbox
            or not math.isclose(
                float(region.centroid[0]), identity.centroid_y, rel_tol=0.0, abs_tol=1e-12
            )
            or not math.isclose(
                float(region.centroid[1]), identity.centroid_x, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise PersistenceAuditError("metric target geometry disagrees with identity")

        if source_index in match_by_target:
            subtype = "matched"
            prediction_index, distance = match_by_target[source_index]
            prediction_region = component_match.prediction_regions[prediction_index]
            predicted_component = _region_record(prediction_region)
            predicted_component["index"] = prediction_index
            matched = True
        else:
            prediction_index = None
            distance = None
            predicted_component = None
            matched = False
            if source_index in no_response:
                subtype = "no_response"
            elif source_index in centroid_miss:
                subtype = "centroid_miss"
            elif source_index in assignment_residual:
                subtype = "assignment_residual"
            else:
                raise PersistenceAuditError("unmatched target escaped taxonomy")

        rows.append(
            {
                "schema_version": SCHEMA,
                "row_kind": "target",
                "dataset": dataset,
                "image_name": image_name,
                "image_index": int(image_index),
                "seed": int(seed),
                "height": target_set.height,
                "width": target_set.width,
                "pixel_connectivity": target_set.pixel_connectivity,
                "skimage_connectivity": target_set.skimage_connectivity,
                "stable_target_id": identity.stable_key,
                "component_index": identity.component_index,
                "source_component_index": identity.source_component_index,
                "source_label": identity.source_label,
                "bbox": identity.bbox,
                "area": identity.area,
                "centroid_y": identity.centroid_y,
                "centroid_x": identity.centroid_x,
                "component_mask_sha256": identity.component_mask_sha256,
                "target_identity": identity.as_dict(),
                "label_mask_sha256": target_set.label_mask_sha256,
                "matched": matched,
                "unmatched": not matched,
                "outcome_subtype": subtype,
                "matched_prediction_index": prediction_index,
                "match_centroid_distance": distance,
                "matched_prediction_component": predicted_component,
                "decision": {
                    "input_semantics": "logits",
                    "threshold": THRESHOLD_LOGIT,
                    "operator": THRESHOLD_OPERATOR,
                    "connectivity": CONNECTIVITY,
                    "pixel_connectivity": 8,
                    "matching": "hungarian_max_cardinality_min_centroid_distance",
                    "centroid_radius": CENTROID_RADIUS,
                    "centroid_radius_operator": "<",
                },
                "checkpoint": {
                    "policy": checkpoint["policy"],
                    "path": checkpoint["path"],
                    "sha256": checkpoint["sha256"],
                    "epoch": checkpoint["epoch"],
                },
                "run": dict(_mapping(checkpoint.get("run"), "checkpoint.run")),
            }
        )
    if len(rows) != len(target_set.targets):
        raise PersistenceAuditError("not every target received exactly one row")
    return rows, target_set


def _target_registry_digest(
    image_names: Sequence[str], target_sets: Mapping[str, StableTargetSet]
) -> str:
    if set(image_names) != set(target_sets) or len(image_names) != len(target_sets):
        raise PersistenceAuditError("target registry does not cover every image exactly once")
    payload = [target_sets[name].as_dict() for name in image_names]
    return sha256_json(payload)


def _normalize_state_dict(state: Mapping[str, Any]) -> Mapping[str, Any]:
    state = strip_legacy_dea_lite_head(state)
    if state and all(key.startswith("module.") for key in state):
        return {key[7:]: value for key, value in state.items()}
    return state


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise PersistenceAuditError(f"invalid device {value!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PersistenceAuditError("CUDA was requested but is unavailable")
    return device


def build_authoritative_target_registry(
    job: Mapping[str, Any],
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, StableTargetSet], dict[str, Any]]:
    """Enumerate the canonical validation-mask universe independently of logits.

    This registry is the authority for image names, target-free images, full
    label-mask hashes, shapes, component-mask hashes, and assertion metadata.
    Inference rows are checked against it; agreement among seeds alone is not
    considered evidence that the target universe is correct.
    """

    stored_args = _mapping(job.get("stored_args"), "validated job.stored_args")
    dataset = IRSTD_Dataset(Namespace(**stored_args), mode="val")
    expected_split_hash = job["split_hashes"]["validation"]
    if dataset.split_sha256 != expected_split_hash:
        raise PersistenceAuditError(
            f"authoritative validation split hash mismatch for {job['job_id']}"
        )
    if dataset.base_size != CANONICAL_SIZE or dataset.crop_size != CANONICAL_SIZE:
        raise PersistenceAuditError("authoritative masks are not canonical 256x256")
    if len(dataset.names) != len(set(dataset.names)):
        raise PersistenceAuditError("authoritative validation names are not unique")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    registry: dict[str, StableTargetSet] = {}
    cursor = 0
    for _, targets in loader:
        if (
            targets.ndim != 4
            or tuple(targets.shape[1:]) != (1, CANONICAL_SIZE, CANONICAL_SIZE)
            or not bool(torch.isfinite(targets).all())
        ):
            raise PersistenceAuditError("authoritative target tensor is invalid")
        target_arrays = (targets[:, 0] > 0.5).numpy().astype(bool, copy=False)
        for batch_index, target in enumerate(target_arrays):
            image_index = cursor + batch_index
            image_name = dataset.names[image_index]
            if image_name in registry:
                raise PersistenceAuditError("duplicate authoritative image name")
            registry[image_name] = build_stable_target_set(
                target,
                dataset=str(job["dataset"]),
                image_name=image_name,
                connectivity=CONNECTIVITY,
            )
        cursor += int(target_arrays.shape[0])
    if cursor != len(dataset) or tuple(registry) != tuple(dataset.names):
        raise PersistenceAuditError(
            "authoritative registry did not cover validation exactly once"
        )
    digest = _target_registry_digest(dataset.names, registry)
    return registry, {
        "dataset": job["dataset"],
        "source_job_id": job["job_id"],
        "source_seed": job["seed"],
        "validation_split_sha256": dataset.split_sha256,
        "image_count": len(registry),
        "target_count": sum(len(item.targets) for item in registry.values()),
        "target_free_image_count": sum(not item.targets for item in registry.values()),
        "target_registry_sha256": digest,
        "authority": "build_stable_target_set_from_canonical_validation_masks",
    }


def build_authoritative_registries_before_checkpoints(
    batch_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[
    dict[str, dict[str, StableTargetSet]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Build the canonical target authority before any checkpoint is loaded."""

    batch_dir = batch_dir.expanduser().resolve()
    manifest_path = batch_dir / "manifest.json"
    try:
        manifest = require_mapping(read_json(manifest_path, "clean manifest"), "manifest")
        seeds, datasets_meta = validate_manifest(manifest, batch_dir)
    except FinalizationError as exc:
        raise PersistenceAuditError(str(exc)) from exc
    if len(seeds) != EXPECTED_SEED_COUNT:
        raise PersistenceAuditError("clean manifest does not contain exactly three seeds")
    raw_jobs = [_mapping(value, "manifest job") for value in manifest["jobs"]]
    validate_grid_cardinality(raw_jobs)
    manifest_args = _mapping(manifest.get("args"), "manifest.args")
    recipe_errors = [
        f"{key}: manifest={manifest_args.get(key)!r} expected={expected!r}"
        for key, expected in EXPECTED_RECIPE.items()
        if manifest_args.get(key) != expected
    ]
    if recipe_errors:
        raise PersistenceAuditError(
            "clean baseline recipe drifted: " + "; ".join(recipe_errors)
        )

    registries: dict[str, dict[str, StableTargetSet]] = {}
    records: dict[str, dict[str, Any]] = {}
    source_jobs: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        dataset_meta = _mapping(
            datasets_meta[dataset], f"manifest.datasets.{dataset}"
        )
        candidates = sorted(
            (job for job in raw_jobs if str(job["dataset"]) == dataset),
            key=lambda value: int(value["seed"]),
        )
        if len(candidates) != EXPECTED_SEED_COUNT:
            raise PersistenceAuditError(
                f"authoritative registry source grid incomplete for {dataset}"
            )
        job = candidates[0]
        if normalized_path(job["dataset_dir"], "registry dataset_dir") != normalized_path(
            dataset_meta["dataset_dir"], "registry metadata dataset_dir"
        ):
            raise PersistenceAuditError("registry source dataset directory drifted")
        for field in ("train_file", "test_file"):
            if job.get(field) != dataset_meta.get(field):
                raise PersistenceAuditError(
                    f"registry source {field} disagrees with dataset metadata"
                )
        run_dir = normalized_path(job["run_dir"], "registry source run_dir")
        config_path = run_dir / "run_config.json"
        try:
            config = require_mapping(
                read_json(config_path, "registry source run_config"),
                "registry source run_config",
            )
        except FinalizationError as exc:
            raise PersistenceAuditError(str(exc)) from exc
        stored_args = _validate_run_config(
            config,
            job=job,
            dataset_meta=dataset_meta,
            manifest_args=manifest_args,
        )
        source_job = {
            "batch_id": batch_dir.name,
            "job_id": job["job_id"],
            "dataset": dataset,
            "seed": int(job["seed"]),
            "stored_args": stored_args,
            "split_hashes": {
                "fit": dataset_meta["fit_sha256"],
                "validation": dataset_meta["val_sha256"],
                "official_test_provenance_only": dataset_meta[
                    "official_test_sha256"
                ],
            },
        }
        registry, record = build_authoritative_target_registry(
            source_job,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        registries[dataset] = registry
        records[dataset] = {
            **record,
            "constructed_before_any_checkpoint_load": True,
            "source_run_config": str(config_path.resolve()),
            "source_run_config_sha256": sha256_file(config_path),
        }
        source_jobs.append(source_job)
    return registries, records, {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "datasets": list(DATASET_NAMES),
        "seeds": list(sorted(seeds)),
        "checkpoint_files_opened_before_registry_complete": 0,
        "source_jobs": source_jobs,
    }


def infer_job_ledger(
    job: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    expected_registry: Mapping[str, StableTargetSet],
) -> tuple[list[dict[str, Any]], dict[str, StableTargetSet], dict[str, Any]]:
    """Run one and only one full-graph validation inference pass."""

    stored_args = _mapping(job.get("stored_args"), "validated job.stored_args")
    dataset = IRSTD_Dataset(Namespace(**stored_args), mode="val")
    expected_split_hash = job["split_hashes"]["validation"]
    if dataset.split_sha256 != expected_split_hash:
        raise PersistenceAuditError(
            f"validation split hash mismatch for {job['job_id']}"
        )
    if dataset.base_size != CANONICAL_SIZE or dataset.crop_size != CANONICAL_SIZE:
        raise PersistenceAuditError("validation transform is not canonical 256x256")
    if len(dataset.names) != len(set(dataset.names)):
        raise PersistenceAuditError("validation image names are not unique")
    if tuple(expected_registry) != tuple(dataset.names):
        raise PersistenceAuditError(
            f"inference image universe disagrees with authority for {job['job_id']}"
        )

    checkpoint_path = Path(str(job["checkpoint"])).resolve()
    if sha256_file(checkpoint_path) != job["checkpoint_sha256"]:
        raise PersistenceAuditError(
            f"checkpoint changed since validation: {checkpoint_path}"
        )
    try:
        checkpoint = load_checkpoint_cpu(checkpoint_path)
    except FinalizationError as exc:
        raise PersistenceAuditError(str(exc)) from exc
    state = checkpoint.get("net")
    if not isinstance(state, Mapping) or not state:
        raise PersistenceAuditError("checkpoint net is not a state dict")
    model = MSHNet(3)
    model.load_state_dict(_normalize_state_dict(state), strict=True)
    del checkpoint, state
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
    checkpoint_row = {
        "policy": job["checkpoint_summary"]["policy"],
        "path": str(checkpoint_path),
        "sha256": job["checkpoint_sha256"],
        "epoch": job["checkpoint_summary"]["epoch"],
        "run": {
            "batch_id": job["batch_id"],
            "job_id": job["job_id"],
            "dataset": job["dataset"],
            "seed": job["seed"],
            "result_file": job["result_file"],
            "result_sha256": job["result_sha256"],
            "metric_log": job["metric_log"],
            "metric_log_sha256": job["metric_log_sha256"],
            "run_config": job["run_config"],
            "run_config_sha256": job["run_config_sha256"],
            "split_hashes": dict(job["split_hashes"]),
            "resume_evidence": dict(job["resume_evidence"]),
        },
    }
    rows: list[dict[str, Any]] = []
    target_sets: dict[str, StableTargetSet] = {}
    cursor = 0
    forward_calls = 0
    with torch.inference_mode():
        for images, targets in loader:
            output = model(images.to(device, non_blocking=True), True)
            forward_calls += 1
            if not isinstance(output, tuple) or len(output) != 2:
                raise PersistenceAuditError("MSHNet full graph returned an invalid output")
            logits = output[1]
            if (
                not torch.is_tensor(logits)
                or logits.ndim != 4
                or logits.shape[1:] != (1, CANONICAL_SIZE, CANONICAL_SIZE)
            ):
                raise PersistenceAuditError("MSHNet full graph logits have invalid shape")
            if not bool(torch.isfinite(logits).all()):
                raise PersistenceAuditError("MSHNet produced non-finite logits")
            logit_arrays = logits.detach().float().cpu().numpy()[:, 0]
            target_arrays = (targets[:, 0] > 0.5).numpy().astype(bool, copy=False)
            for batch_index in range(logit_arrays.shape[0]):
                image_index = cursor + batch_index
                image_name = dataset.names[image_index]
                image_rows, target_set = build_image_target_rows(
                    logit_arrays[batch_index],
                    target_arrays[batch_index],
                    dataset=str(job["dataset"]),
                    image_name=image_name,
                    seed=int(job["seed"]),
                    image_index=image_index,
                    checkpoint=checkpoint_row,
                )
                operating_metrics = compute_image_operating_metrics(
                    logit_arrays[batch_index],
                    target_arrays[batch_index],
                )
                if operating_metrics["target_components"] != len(target_set.targets):
                    raise PersistenceAuditError(
                        "operating metric target count disagrees with authority"
                    )
                if image_name in target_sets:
                    raise PersistenceAuditError("duplicate image in target registry")
                try:
                    assert_same_target_set(expected_registry[image_name], target_set)
                except Exception as exc:
                    raise PersistenceAuditError(
                        "inference target mask disagrees with authoritative registry: "
                        f"{job['dataset']}/{image_name}: {exc}"
                    ) from exc
                target_sets[image_name] = target_set
                rows.append(
                    build_image_envelope_row(
                        target_set,
                        seed=int(job["seed"]),
                        image_index=image_index,
                        checkpoint=checkpoint_row,
                        operating_metrics=operating_metrics,
                    )
                )
                rows.extend(image_rows)
            cursor += int(logit_arrays.shape[0])
    if cursor != len(dataset) or set(target_sets) != set(dataset.names):
        raise PersistenceAuditError("inference did not cover validation exactly once")
    if forward_calls != len(loader):
        raise PersistenceAuditError("inference forward-call accounting drifted")
    registry_digest = _target_registry_digest(dataset.names, target_sets)
    expected_digest = _target_registry_digest(dataset.names, expected_registry)
    if registry_digest != expected_digest:
        raise PersistenceAuditError("inference target registry digest drifted")
    inference = {
        "job_id": job["job_id"],
        "dataset": job["dataset"],
        "seed": job["seed"],
        "validation_images": len(dataset),
        "artifact_rows": len(rows),
        "image_envelope_rows": sum(row["row_kind"] == "image" for row in rows),
        "target_rows": sum(row["row_kind"] == "target" for row in rows),
        "target_registry_sha256": registry_digest,
        "authoritative_target_registry_sha256": expected_digest,
        "validation_split_sha256": dataset.split_sha256,
        "forward_calls": forward_calls,
        "full_graph_warm_flag": True,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, target_sets, inference


def validate_cross_run_target_registries(
    registry_by_job: Mapping[str, Mapping[str, StableTargetSet]],
    jobs: Sequence[Mapping[str, Any]],
    authoritative_by_dataset: Mapping[str, Mapping[str, StableTargetSet]],
) -> dict[str, dict[str, Any]]:
    """Assert exact image/mask/component identity across all three seeds."""

    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for job in jobs:
        by_dataset[str(job["dataset"])].append(job)
    summary: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_NAMES:
        if dataset not in authoritative_by_dataset:
            raise PersistenceAuditError(
                f"missing authoritative target registry for {dataset}"
            )
        authority = authoritative_by_dataset[dataset]
        dataset_jobs = sorted(by_dataset[dataset], key=lambda item: int(item["seed"]))
        if len(dataset_jobs) != EXPECTED_SEED_COUNT:
            raise PersistenceAuditError(f"target registry grid incomplete for {dataset}")
        image_names = tuple(authority)
        for job in dataset_jobs:
            observed = registry_by_job[job["job_id"]]
            if tuple(observed) != image_names:
                raise PersistenceAuditError(
                    f"validation image identity/order differs across seeds for {dataset}"
                )
            for image_name in image_names:
                try:
                    assert_same_target_set(authority[image_name], observed[image_name])
                except Exception as exc:
                    raise PersistenceAuditError(
                        "target identity differs from canonical-mask authority: "
                        f"{dataset}/{image_name}: {exc}"
                    ) from exc
        target_count = sum(len(item.targets) for item in authority.values())
        summary[dataset] = {
            "image_count": len(authority),
            "target_count": target_count,
            "target_free_image_count": sum(not item.targets for item in authority.values()),
            "target_registry_sha256": _target_registry_digest(image_names, authority),
            "authority": "build_stable_target_set_from_canonical_validation_masks",
            "all_seeds_exactly_equal_to_authority": True,
        }
    return summary


def _taxonomy_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    allowed = {"matched", "no_response", "centroid_miss", "assignment_residual"}
    overall = Counter()
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_run: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, str, int, str]] = set()
    for index, row in enumerate(rows):
        if row.get("row_kind") == "image":
            continue
        if row.get("row_kind") != "target":
            raise PersistenceAuditError(f"row {index} has invalid row_kind")
        subtype = row.get("outcome_subtype")
        if subtype not in allowed:
            raise PersistenceAuditError(f"row {index} has invalid outcome subtype")
        key = (
            str(row["dataset"]),
            str(row["image_name"]),
            int(row["seed"]),
            str(row["stable_target_id"]),
        )
        if key in seen:
            raise PersistenceAuditError("duplicate complete-ledger target row")
        seen.add(key)
        if bool(row["matched"]) != (subtype == "matched"):
            raise PersistenceAuditError("matched status disagrees with outcome subtype")
        overall[str(subtype)] += 1
        by_dataset[str(row["dataset"])][str(subtype)] += 1
        by_run[(str(row["dataset"]), int(row["seed"]))][str(subtype)] += 1

    def record(counter: Counter[str]) -> dict[str, int]:
        return {name: int(counter.get(name, 0)) for name in sorted(allowed)}

    return {
        "overall": record(overall),
        "by_dataset": {
            dataset: record(by_dataset[dataset]) for dataset in DATASET_NAMES
        },
        "by_run": [
            {"dataset": dataset, "seed": seed, **record(by_run[(dataset, seed)])}
            for dataset, seed in sorted(
                by_run, key=lambda item: (DATASET_NAMES.index(item[0]), item[1])
            )
        ],
    }


def _achieved_fa_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("row_kind") != "image":
            continue
        operating = row.get("operating_point")
        if not isinstance(operating, Mapping):
            raise PersistenceAuditError(
                f"image row {index} lacks achieved operating-point metrics"
            )
        operating_threshold = _finite(
            operating.get("threshold_logit"), "operating threshold_logit"
        )
        if (
            operating_threshold != THRESHOLD_LOGIT
            or operating.get("threshold_operator") != THRESHOLD_OPERATOR
        ):
            raise PersistenceAuditError("image operating threshold drifted")
        pixels = _exact_int(operating.get("image_pixels"), "operating image_pixels")
        if pixels != CANONICAL_SIZE * CANONICAL_SIZE:
            raise PersistenceAuditError("image operating pixel count is not canonical")
        record: dict[str, Any] = {
            "dataset": str(row["dataset"]),
            "seed": _exact_int(row["seed"], "image row.seed"),
            "image_pixels": pixels,
            "target_components": _exact_int(
                operating.get("target_components"), "operating target_components"
            ),
        }
        for matcher in ("hungarian", "legacy"):
            matcher_record = operating.get(matcher)
            if not isinstance(matcher_record, Mapping):
                raise PersistenceAuditError(f"missing {matcher} operating metrics")
            area = _exact_int(
                matcher_record.get("unmatched_prediction_area"),
                f"{matcher}.unmatched_prediction_area",
            )
            if not 0 <= area <= pixels:
                raise PersistenceAuditError(f"{matcher} unmatched area is invalid")
            matched = _exact_int(
                matcher_record.get("matched_target_components"),
                f"{matcher}.matched_target_components",
            )
            unmatched_targets = _exact_int(
                matcher_record.get("unmatched_target_components"),
                f"{matcher}.unmatched_target_components",
            )
            if (
                matched < 0
                or unmatched_targets < 0
                or matched + unmatched_targets != record["target_components"]
            ):
                raise PersistenceAuditError(
                    f"{matcher} matched/unmatched target counts are invalid"
                )
            achieved = _finite(
                matcher_record.get("fa_per_million_pixels"),
                f"{matcher}.fa_per_million_pixels",
            )
            recomputed = area / pixels * 1_000_000.0
            if not math.isclose(achieved, recomputed, rel_tol=0.0, abs_tol=1e-9):
                raise PersistenceAuditError(
                    f"{matcher} achieved FA/Mpix disagrees with unmatched area"
                )
            record[f"{matcher}_unmatched_prediction_area"] = area
            record[f"{matcher}_matched_target_components"] = matched
        images.append(record)
    if not images:
        raise PersistenceAuditError("achieved FA summary has no image rows")

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not selected:
            raise PersistenceAuditError("cannot summarize an empty achieved-FA group")
        pixels = sum(int(row["image_pixels"]) for row in selected)
        result: dict[str, Any] = {
            "image_count": len(selected),
            "image_pixels": pixels,
            "target_components": sum(
                int(row["target_components"]) for row in selected
            ),
        }
        for matcher in ("hungarian", "legacy"):
            area = sum(
                int(row[f"{matcher}_unmatched_prediction_area"])
                for row in selected
            )
            matched = sum(
                int(row[f"{matcher}_matched_target_components"])
                for row in selected
            )
            result[matcher] = {
                "matched_target_components": matched,
                "pd": (
                    matched / result["target_components"]
                    if result["target_components"]
                    else None
                ),
                "unmatched_prediction_area": area,
                "achieved_fa_per_million_pixels": area / pixels * 1_000_000.0,
            }
        result["matcher_delta_hungarian_minus_legacy"] = {
            "matched_target_components": (
                result["hungarian"]["matched_target_components"]
                - result["legacy"]["matched_target_components"]
            ),
            "pd": (
                result["hungarian"]["pd"] - result["legacy"]["pd"]
                if result["hungarian"]["pd"] is not None
                and result["legacy"]["pd"] is not None
                else None
            ),
            "fa_per_million_pixels": (
                result["hungarian"]["achieved_fa_per_million_pixels"]
                - result["legacy"]["achieved_fa_per_million_pixels"]
            ),
        }
        return result

    by_dataset = {
        dataset: summarize(
            [row for row in images if row["dataset"] == dataset]
        )
        for dataset in DATASET_NAMES
    }
    run_keys = sorted(
        {(str(row["dataset"]), int(row["seed"])) for row in images},
        key=lambda item: (DATASET_NAMES.index(item[0]), item[1]),
    )
    return {
        "threshold_logit": THRESHOLD_LOGIT,
        "threshold_operator": THRESHOLD_OPERATOR,
        "not_a_budget_matched_operating_point": True,
        "overall": summarize(images),
        "by_dataset": by_dataset,
        "by_run": [
            {
                "dataset": dataset,
                "seed": seed,
                **summarize(
                    [
                        row
                        for row in images
                        if row["dataset"] == dataset and int(row["seed"]) == seed
                    ]
                ),
            }
            for dataset, seed in run_keys
        ],
    }


def annotate_target_recurrence(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach exact three-seed miss/subtype counts to every target row.

    The repeated fields make fixed-epoch and best-IoU bundles independently
    joinable by ``stable_target_id`` for missed-set Jaccard and recurrence
    transition analyses.  Image-envelope rows remain untouched.
    """

    seeds = tuple(
        sorted(
            {
                int(row["seed"])
                for row in rows
                if row.get("row_kind") in {"image", "target"}
            }
        )
    )
    if len(seeds) != EXPECTED_SEED_COUNT:
        raise PersistenceAuditError(
            f"recurrence annotation requires exactly {EXPECTED_SEED_COUNT} seeds"
        )
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("row_kind") == "image":
            continue
        if row.get("row_kind") != "target":
            raise PersistenceAuditError(f"row {index} has invalid row_kind")
        groups[
            (
                str(row["dataset"]),
                str(row["image_name"]),
                str(row["stable_target_id"]),
            )
        ].append(row)
    if not groups:
        raise PersistenceAuditError("cannot annotate an empty target universe")

    annotations: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, target_rows in groups.items():
        target_seeds = tuple(sorted(int(row["seed"]) for row in target_rows))
        if target_seeds != seeds or len(target_rows) != EXPECTED_SEED_COUNT:
            raise PersistenceAuditError(
                f"target does not have exactly one row per seed: {key!r}"
            )
        miss_seeds = sorted(
            int(row["seed"]) for row in target_rows if bool(row["unmatched"])
        )
        no_response_seeds = sorted(
            int(row["seed"])
            for row in target_rows
            if row["outcome_subtype"] == "no_response"
        )
        annotations[key] = {
            "miss_count": len(miss_seeds),
            "miss_seed_ids": miss_seeds,
            "no_response_count": len(no_response_seeds),
            "no_response_seed_ids": no_response_seeds,
            "observed_three_of_three_miss": len(miss_seeds) == EXPECTED_SEED_COUNT,
            "observed_two_or_more_miss": len(miss_seeds) >= 2,
        }

    annotated: list[dict[str, Any]] = []
    for row in rows:
        copy_row = dict(row)
        if row.get("row_kind") == "target":
            key = (
                str(row["dataset"]),
                str(row["image_name"]),
                str(row["stable_target_id"]),
            )
            copy_row.update(annotations[key])
        annotated.append(copy_row)
    return annotated


def _binary_status(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return bool(value)
    raise PersistenceAuditError(f"{label} must be binary bool/0/1")


def _policy_target_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_policy: Literal["fixed_epoch", "best_iou"],
) -> dict[str, dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise PersistenceAuditError("policy ledger rows must be a non-empty sequence")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    all_seeds: set[int] = set()
    signature_fields = (
        "dataset",
        "image_name",
        "stable_target_id",
        "height",
        "width",
        "pixel_connectivity",
        "skimage_connectivity",
        "label_mask_sha256",
        "component_index",
        "source_component_index",
        "source_label",
        "bbox",
        "area",
        "centroid_y",
        "centroid_x",
        "component_mask_sha256",
    )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PersistenceAuditError(f"policy ledger row {index} is not a mapping")
        if row.get("row_kind") == "image":
            continue
        if row.get("row_kind") != "target":
            raise PersistenceAuditError(f"policy ledger row {index} has invalid row_kind")
        missing = [
            field
            for field in (*signature_fields, "seed", "unmatched", "miss_count", "checkpoint")
            if field not in row
        ]
        if missing:
            raise PersistenceAuditError(
                f"policy ledger row {index} missing fields: {missing}"
            )
        checkpoint = row["checkpoint"]
        if not isinstance(checkpoint, Mapping) or checkpoint.get("policy") != expected_policy:
            raise PersistenceAuditError(
                f"policy ledger row {index} is not {expected_policy}"
            )
        target_id = row["stable_target_id"]
        if not isinstance(target_id, str) or not target_id:
            raise PersistenceAuditError(
                f"policy ledger row {index} has invalid stable_target_id"
            )
        seed = _exact_int(row["seed"], f"policy ledger row {index}.seed")
        all_seeds.add(seed)
        grouped[target_id].append(row)
    if len(all_seeds) != EXPECTED_SEED_COUNT:
        raise PersistenceAuditError(
            f"policy ledger requires exactly {EXPECTED_SEED_COUNT} seeds"
        )
    if not grouped:
        raise PersistenceAuditError("policy ledger contains no target rows")

    expected_seeds = tuple(sorted(all_seeds))
    table: dict[str, dict[str, Any]] = {}
    for target_id, target_rows in grouped.items():
        seeds = tuple(sorted(_exact_int(row["seed"], "target seed") for row in target_rows))
        if seeds != expected_seeds or len(target_rows) != EXPECTED_SEED_COUNT:
            raise PersistenceAuditError(
                f"target {target_id!r} does not have one row per seed"
            )
        signatures = [
            sha256_json({field: row[field] for field in signature_fields})
            for row in target_rows
        ]
        if len(set(signatures)) != 1:
            raise PersistenceAuditError(
                f"target assertion metadata differs within {expected_policy}: {target_id!r}"
            )
        counts = {
            _exact_int(row["miss_count"], f"{target_id}.miss_count")
            for row in target_rows
        }
        if len(counts) != 1:
            raise PersistenceAuditError(
                f"repeated miss_count differs for {target_id!r}"
            )
        miss_count = counts.pop()
        if not 0 <= miss_count <= EXPECTED_SEED_COUNT:
            raise PersistenceAuditError(
                f"miss_count outside 0..{EXPECTED_SEED_COUNT} for {target_id!r}"
            )
        observed_count = sum(
            _binary_status(row["unmatched"], label=f"{target_id}.unmatched")
            for row in target_rows
        )
        if observed_count != miss_count:
            raise PersistenceAuditError(
                f"miss_count disagrees with per-seed statuses for {target_id!r}"
            )
        first = target_rows[0]
        table[target_id] = {
            "dataset": str(first["dataset"]),
            "image_name": str(first["image_name"]),
            "stable_target_id": target_id,
            "miss_count": miss_count,
            "seed_ids": list(expected_seeds),
            "assertion_sha256": signatures[0],
        }
    return table


def _policy_transition_scope(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    matrix = [[0 for _ in range(4)] for _ in range(4)]
    fixed_missed: set[str] = set()
    best_missed: set[str] = set()
    fixed_persistent = 0
    fixed_persistent_retained = 0
    for fixed, best in pairs:
        fixed_count = int(fixed["miss_count"])
        best_count = int(best["miss_count"])
        matrix[fixed_count][best_count] += 1
        target_id = str(fixed["stable_target_id"])
        if fixed_count >= 1:
            fixed_missed.add(target_id)
        if best_count >= 1:
            best_missed.add(target_id)
        if fixed_count == 3:
            fixed_persistent += 1
            if best_count >= 2:
                fixed_persistent_retained += 1

    intersection = fixed_missed & best_missed
    union = fixed_missed | best_missed
    jaccard = len(intersection) / len(union) if union else None
    retention = (
        fixed_persistent_retained / fixed_persistent
        if fixed_persistent
        else None
    )

    def recurrence(index: int) -> dict[str, Any]:
        counts = [0, 0, 0, 0]
        for pair in pairs:
            counts[int(pair[index]["miss_count"])] += 1
        events = counts[1] + 2 * counts[2] + 3 * counts[3]
        return {
            "target_count": len(pairs),
            "N0": counts[0],
            "N1": counts[1],
            "N2": counts[2],
            "N3": counts[3],
            "N3_over_N": counts[3] / len(pairs) if pairs else None,
            "event_count": events,
            "persistent_event_share": 3 * counts[3] / events if events else None,
        }

    return {
        "target_count": len(pairs),
        "transition_matrix": {
            "row_axis": "fixed_epoch_miss_count_c_fixed",
            "column_axis": "best_iou_miss_count_c_best",
            "levels": [0, 1, 2, 3],
            "counts": matrix,
            "cells": [
                {
                    "c_fixed": c_fixed,
                    "c_best": c_best,
                    "target_count": matrix[c_fixed][c_best],
                }
                for c_fixed in range(4)
                for c_best in range(4)
            ],
        },
        "missed_set_jaccard": {
            "definition": "J({targets:c_fixed>=1},{targets:c_best>=1})",
            "fixed_missed_target_count": len(fixed_missed),
            "best_missed_target_count": len(best_missed),
            "intersection_target_count": len(intersection),
            "union_target_count": len(union),
            "defined": jaccard is not None,
            "value": jaccard,
            "undefined_reason": None if jaccard is not None else "both_missed_sets_empty",
        },
        "fixed_c3_to_best_c_ge2_retention": {
            "definition": "P(c_best>=2 | c_fixed=3) over canonical targets",
            "denominator_fixed_c3": fixed_persistent,
            "numerator_best_c_ge2": fixed_persistent_retained,
            "defined": retention is not None,
            "value": retention,
            "undefined_reason": None if retention is not None else "no_fixed_c3_targets",
        },
        "fixed_epoch_recurrence": recurrence(0),
        "best_iou_recurrence": recurrence(1),
    }


def compare_policy_target_recurrence(
    fixed_rows: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare immutable fixed-epoch and best-IoU target ledgers exactly."""

    fixed = _policy_target_table(fixed_rows, expected_policy="fixed_epoch")
    best = _policy_target_table(best_rows, expected_policy="best_iou")
    if set(fixed) != set(best):
        raise PersistenceAuditError(
            "fixed_epoch and best_iou target universes differ: "
            f"missing_best={len(set(fixed) - set(best))}, "
            f"extra_best={len(set(best) - set(fixed))}"
        )
    fixed_seed_ids = {tuple(record["seed_ids"]) for record in fixed.values()}
    best_seed_ids = {tuple(record["seed_ids"]) for record in best.values()}
    if len(fixed_seed_ids) != 1 or len(best_seed_ids) != 1:
        raise PersistenceAuditError("policy ledger seed metadata is internally inconsistent")
    if fixed_seed_ids != best_seed_ids:
        raise PersistenceAuditError("fixed_epoch and best_iou seed sets differ")
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    by_dataset: dict[
        str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = defaultdict(list)
    for target_id in sorted(fixed):
        fixed_target = fixed[target_id]
        best_target = best[target_id]
        if fixed_target["assertion_sha256"] != best_target["assertion_sha256"]:
            raise PersistenceAuditError(
                f"fixed/best target assertion metadata differs: {target_id!r}"
            )
        pair = (fixed_target, best_target)
        pairs.append(pair)
        by_dataset[str(fixed_target["dataset"])].append(pair)
    return {
        "schema_version": "dea.gate_e.checkpoint_policy_transition.v2",
        "fixed_policy": "fixed_epoch",
        "best_policy": "best_iou",
        "target_universe_exactly_equal": True,
        "overall": _policy_transition_scope(pairs),
        "by_dataset": {
            dataset: _policy_transition_scope(by_dataset[dataset])
            for dataset in sorted(by_dataset)
        },
    }


def read_policy_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Read one immutable policy JSONL for :func:`compare_policy_target_recurrence`."""

    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "target_persistence.jsonl"
    if not path.is_file():
        raise PersistenceAuditError(f"missing policy ledger: {path}")
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PersistenceAuditError(f"cannot read policy ledger {path}: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise PersistenceAuditError("policy ledger must be non-empty JSONL without blank rows")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PersistenceAuditError(
                f"invalid policy ledger JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise PersistenceAuditError(
                f"policy ledger line {line_number} is not an object"
            )
        records.append(value)
    return records


def _failure_unit_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target_rows = [row for row in rows if row.get("row_kind") == "target"]
    grouped: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in target_rows:
        key = (
            str(row["dataset"]),
            str(row["image_name"]),
            str(row["stable_target_id"]),
        )
        if key in grouped:
            if int(grouped[key]["miss_count"]) != int(row["miss_count"]):
                raise PersistenceAuditError("repeated target miss_count disagrees")
        else:
            grouped[key] = row

    def summarize(dataset: str | None) -> dict[str, int]:
        selected = [
            (key, row)
            for key, row in grouped.items()
            if dataset is None or key[0] == dataset
        ]
        missed = [(key, row) for key, row in selected if int(row["miss_count"]) >= 1]
        persistent = [
            (key, row)
            for key, row in selected
            if int(row["miss_count"]) == EXPECTED_SEED_COUNT
        ]
        no_response = [
            (key, row)
            for key, row in selected
            if int(row["no_response_count"]) >= 1
        ]
        return {
            "unique_targets": len(selected),
            "unique_missed_targets": len(missed),
            "unique_images_with_at_least_one_miss": len(
                {(key[0], key[1]) for key, _ in missed}
            ),
            "observed_three_of_three_target_count": len(persistent),
            "observed_three_of_three_image_count": len(
                {(key[0], key[1]) for key, _ in persistent}
            ),
            "unique_no_response_targets": len(no_response),
            "unique_images_with_at_least_one_no_response": len(
                {(key[0], key[1]) for key, _ in no_response}
            ),
        }

    return {
        "overall": summarize(None),
        "by_dataset": {dataset: summarize(dataset) for dataset in DATASET_NAMES},
        "observed_three_of_three_definition": (
            "an image is counted when it contains at least one canonical target "
            "missed in all three frozen seeds"
        ),
    }


def _no_response_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    for row in rows:
        copy_row = dict(row)
        if row.get("row_kind") == "target":
            absent = row.get("outcome_subtype") == "no_response"
            copy_row["matched"] = not absent
            copy_row["unmatched"] = absent
        derived.append(copy_row)
    return derived


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Literal["fixed_epoch", "best_iou"],
    registry_summary: Mapping[str, Any],
    expected_registry: Sequence[StableTargetSet],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    recurrence = summarize_cross_seed_persistence(
        rows,
        expected_registry=expected_registry,
        status_field="matched",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    no_response_recurrence = summarize_cross_seed_persistence(
        _no_response_rows(rows),
        expected_registry=expected_registry,
        status_field="matched",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    expected_target_rows = sum(
        int(registry_summary[dataset]["target_count"]) * EXPECTED_SEED_COUNT
        for dataset in DATASET_NAMES
    )
    expected_image_rows = sum(
        int(registry_summary[dataset]["image_count"]) * EXPECTED_SEED_COUNT
        for dataset in DATASET_NAMES
    )
    target_row_count = sum(row.get("row_kind") == "target" for row in rows)
    image_row_count = sum(row.get("row_kind") == "image" for row in rows)
    if target_row_count != expected_target_rows or image_row_count != expected_image_rows:
        raise PersistenceAuditError(
            "ledger row universe differs from authoritative registry: "
            f"target={target_row_count}/{expected_target_rows}, "
            f"image={image_row_count}/{expected_image_rows}"
        )
    return {
        "schema_version": SCHEMA,
        "status": "complete_and_validated",
        "stage": "Gate_E_minus_1_failure_persistence_audit",
        "checkpoint_policy": policy,
        "selection_scope": (
            "fixed_epoch_399_primary"
            if policy == "fixed_epoch"
            else "retrospective_best_validation_iou_sensitivity_only"
        ),
        "evaluation_scope": "official-training-set internal development holdout",
        "official_test_policy": OFFICIAL_TEST_POLICY,
        "decision_rule": {
            "score": "final full-graph MSHNet logits",
            "threshold": THRESHOLD_LOGIT,
            "operator": THRESHOLD_OPERATOR,
            "matching": "hungarian_max_cardinality_min_centroid_distance",
            "centroid_radius": CENTROID_RADIUS,
            "centroid_radius_operator": "<",
            "connectivity": CONNECTIVITY,
        },
        "artifact_row_count": len(rows),
        "ledger_row_count": target_row_count,
        "image_envelope_row_count": image_row_count,
        "target_registry": dict(registry_summary),
        "achieved_operating_point": _achieved_fa_summary(rows),
        "miss_taxonomy": _taxonomy_summary(rows),
        "recurrence": recurrence,
        "no_response_recurrence": no_response_recurrence,
        "failure_units": _failure_unit_summary(rows),
        "interpretation_guard": (
            "N3 means observed miss in all three frozen seeds; it is not a "
            "claim of seed-general persistence. Bootstrap intervals resample "
            "images only and exclude training-seed and dataset-domain uncertainty. "
            "Logit zero is a fixed decision rule, not a matched low-FA budget; "
            "the achieved Hungarian and legacy FA/Mpix values must be reported."
        ),
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    recurrence = summary["recurrence"]
    lines = [
        "# Gate E−1 cross-seed target-failure persistence",
        "",
        f"- Checkpoint policy: `{summary['checkpoint_policy']}`",
        f"- Selection scope: `{summary['selection_scope']}`",
        "- Decision: strict final logit `> 0`, 8-connected components",
        "- Matching: maximum-cardinality/minimum-distance Hungarian, centroid distance `< 3` pixels",
        "- Operating-point guard: logit zero is fixed, not budget-matched; achieved FA/Mpix is reported below",
        "- Evaluation: internal development holdout only; official tests remain sealed",
        "",
        "## Achieved matcher metrics at the fixed logit-zero rule",
        "",
        "| Dataset | Seed | Hungarian Pd | Legacy Pd | Hungarian FA/Mpix | Legacy FA/Mpix |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in summary["achieved_operating_point"]["by_run"]:
        lines.append(
            f"| {record['dataset']} | {record['seed']} | "
            f"{record['hungarian']['pd']:.4f} | "
            f"{record['legacy']['pd']:.4f} | "
            f"{record['hungarian']['achieved_fa_per_million_pixels']:.4f} | "
            f"{record['legacy']['achieved_fa_per_million_pixels']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Miss recurrence",
            "",
            "| Dataset | Targets | N0 | N1 | N2 | N3 (observed 3/3) | N3/N | Persistent miss-event share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    records = [("Overall", recurrence["overall"]["target_micro"])] + [
        (dataset, recurrence["by_dataset"][dataset]["target_micro"])
        for dataset in DATASET_NAMES
    ]
    for label, record in records:
        event_share = record["persistent_event_share"]
        event_text = "undefined" if event_share is None else f"{event_share:.4f}"
        lines.append(
            f"| {label} | {record['target_count']} | {record['N0']} | "
            f"{record['N1']} | {record['N2']} | {record['N3']} | "
            f"{record['N3_over_N']:.4f} | {event_text} |"
        )
    lines.extend(
        [
            "",
            "## No-response subtype recurrence",
            "",
            "| Dataset | N0 | N1 | N2 | N3 (observed 3/3 no-response) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    no_response = summary["no_response_recurrence"]
    no_response_records = [
        ("Overall", no_response["overall"]["target_micro"])
    ] + [
        (dataset, no_response["by_dataset"][dataset]["target_micro"])
        for dataset in DATASET_NAMES
    ]
    for label, record in no_response_records:
        lines.append(
            f"| {label} | {record['N0']} | {record['N1']} | "
            f"{record['N2']} | {record['N3']} |"
        )
    lines.extend(
        [
            "",
            "## Unique failure units",
            "",
            "| Dataset | Unique missed targets | Images with any miss | Observed 3/3 targets | Images containing observed 3/3 target |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    units = [("Overall", summary["failure_units"]["overall"])] + [
        (dataset, summary["failure_units"]["by_dataset"][dataset])
        for dataset in DATASET_NAMES
    ]
    for label, record in units:
        lines.append(
            f"| {label} | {record['unique_missed_targets']} | "
            f"{record['unique_images_with_at_least_one_miss']} | "
            f"{record['observed_three_of_three_target_count']} | "
            f"{record['observed_three_of_three_image_count']} |"
        )
    lines.extend(
        [
            "",
            "## Outcome taxonomy",
            "",
            "| Dataset | Matched | No response | Centroid miss | Assignment residual |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    taxonomy = summary["miss_taxonomy"]["by_dataset"]
    for dataset in DATASET_NAMES:
        record = taxonomy[dataset]
        lines.append(
            f"| {dataset} | {record['matched']} | {record['no_response']} | "
            f"{record['centroid_miss']} | {record['assignment_residual']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            summary["interpretation_guard"],
            "",
            "This audit establishes recurrence and subtype only. It does not attribute any "
            "failure to training examples, gradients, optimizer credit, or a causal mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_output_bundle(
    output_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Write the four artifacts through an atomic directory rename."""

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        jsonl_path = temporary / OUTPUT_FILES[0]
        with jsonl_path.open("w", encoding="utf-8") as handle:
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
        summary_path = temporary / OUTPUT_FILES[1]
        _write_text(
            summary_path,
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
        )
        markdown_path = temporary / OUTPUT_FILES[2]
        _write_text(markdown_path, build_markdown(summary))
        artifact_hashes = {
            path.name: sha256_file(path)
            for path in (jsonl_path, summary_path, markdown_path)
        }
        complete_provenance = {
            **dict(provenance),
            "artifact_sha256": artifact_hashes,
        }
        provenance_path = temporary / OUTPUT_FILES[3]
        _write_text(
            provenance_path,
            json.dumps(
                complete_provenance,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
        )
        if set(path.name for path in temporary.iterdir()) != set(OUTPUT_FILES):
            raise PersistenceAuditError("temporary output inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_hashes() -> dict[str, str]:
    files = {
        "audit_tool": Path(__file__).resolve(),
        "mshnet": ROOT / "model" / "MSHNet.py",
        "checkpoint_compatibility": ROOT / "model" / "mshnet_checkpoint.py",
        "dataset": ROOT / "utils" / "data.py",
        "metric": ROOT / "utils" / "metric.py",
        "component_ledger": ROOT / "utils" / "component_ledger.py",
        "target_identity": ROOT / "utils" / "target_identity.py",
        "cross_seed_persistence": ROOT / "utils" / "cross_seed_persistence.py",
        "clean_baseline_finalizer": ROOT / "tools" / "finalize_clean_baselines.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def validate_source_files_unchanged(frozen: Mapping[str, str]) -> None:
    observed = _source_hashes()
    if dict(frozen) != observed:
        changed = sorted(
            set(frozen) | set(observed),
            key=str,
        )
        changed = [name for name in changed if frozen.get(name) != observed.get(name)]
        raise PersistenceAuditError(
            f"audit source changed during run: {changed}"
        )


def git_worktree_provenance() -> dict[str, Any]:
    def run_git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise PersistenceAuditError(
                f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
            )
        return result.stdout.rstrip("\n")

    status = run_git("status", "--short", "--untracked-files=all").splitlines()
    return {
        "head": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status,
    }


def run_audit(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if (
        not isinstance(args.batch_id, str)
        or SAFE_BATCH_ID_RE.fullmatch(args.batch_id) is None
        or args.batch_id in {".", ".."}
    ):
        raise PersistenceAuditError("--batch-id must be one safe directory name")
    policy = _require_policy(args.checkpoint_policy)
    if args.batch_size < 1 or args.num_workers < 0:
        raise PersistenceAuditError("batch size must be positive and workers non-negative")
    if args.bootstrap_replicates < 1:
        raise PersistenceAuditError("bootstrap replicates must be positive")
    batch_dir = ROOT / "repro_runs" / "clean" / args.batch_id
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "repro_runs" / "gate_e" / "persistence_v2" / policy
    ).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    source_freeze = _source_hashes()
    git_freeze = git_worktree_provenance()
    protocol_freeze = {
        "freeze_type": "pilot_visible_prospective_protocol_freeze",
        "captured_before_manifest_validation_and_model_inference": True,
        "pilot_outcomes_visible_before_freeze": True,
        "rejected_fixed_epoch_preflight_outcome_visible": "N3=18/493",
        "formal_fixed_epoch_bundle_visible_before_this_run": policy == "best_iou",
        "outcome_blind_preregistration": False,
        "interpretation": (
            "The formal Gate E-1 protocol and North-Star positioning were "
            "hashed before this audit run, after earlier pilot evidence was "
            "already visible. This is a prospective freeze for the formal "
            "run, not an outcome-blind preregistration."
        ),
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "documents": protocol_document_fingerprints(),
    }
    (
        authoritative_registries,
        authority_records,
        registry_precheckpoint,
    ) = build_authoritative_registries_before_checkpoints(
        batch_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    jobs, batch_provenance = load_validated_jobs(batch_dir, policy=policy)
    device = _resolve_device(args.device)
    torch.manual_seed(args.bootstrap_seed)
    np.random.seed(args.bootstrap_seed % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.bootstrap_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    all_rows: list[dict[str, Any]] = []
    registries: dict[str, dict[str, StableTargetSet]] = {}
    inference_records: list[dict[str, Any]] = []
    for job in jobs:
        rows, target_sets, inference = infer_job_ledger(
            job,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            expected_registry=authoritative_registries[job["dataset"]],
        )
        all_rows.extend(rows)
        registries[job["job_id"]] = target_sets
        inference_records.append(inference)

    registry_summary = validate_cross_run_target_registries(
        registries,
        jobs,
        authoritative_registries,
    )
    all_rows = annotate_target_recurrence(all_rows)
    expected_registry = tuple(
        target_set
        for dataset in DATASET_NAMES
        for target_set in authoritative_registries[dataset].values()
    )
    summary = build_summary(
        all_rows,
        policy=policy,
        registry_summary=registry_summary,
        expected_registry=expected_registry,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    validate_protocol_documents_unchanged(protocol_freeze["documents"])
    validate_source_files_unchanged(source_freeze)
    if git_worktree_provenance() != git_freeze:
        raise PersistenceAuditError("Git HEAD/worktree status changed during audit")
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "checkpoint_policy": policy,
        "protocol_freeze": protocol_freeze,
        "policy_isolation": (
            "fixed_epoch and best_iou write to distinct immutable output bundles "
            "and are never pooled"
        ),
        "batch": batch_provenance,
        "git": git_freeze,
        "source_freeze": {
            "captured_before_registry_and_checkpoint_loading": True,
            "sha256": source_freeze,
            "validated_unchanged_before_output": True,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit_image": importlib.metadata.version("scikit-image"),
            "pillow": importlib.metadata.version("Pillow"),
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "protocol": {
            "split_role": "validation",
            "canonical_mask": "nearest-neighbor resize to 256x256 then tensor > 0.5",
            "full_graph": True,
            "threshold_logit": THRESHOLD_LOGIT,
            "threshold_operator": THRESHOLD_OPERATOR,
            "connectivity": CONNECTIVITY,
            "centroid_radius": CENTROID_RADIUS,
            "centroid_radius_operator": "<",
            "official_test_policy": OFFICIAL_TEST_POLICY,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_uncertainty_scope": "image sampling only",
        },
        "jobs": [
            {
                key: job[key]
                for key in (
                    "job_id",
                    "dataset",
                    "seed",
                    "result_file",
                    "result_sha256",
                    "metric_log",
                    "metric_log_sha256",
                    "run_config",
                    "run_config_sha256",
                    "checkpoint",
                    "checkpoint_sha256",
                    "checkpoint_summary",
                    "resume_requested_by_recorded_command",
                    "resume_evidence",
                    "split_hashes",
                )
            }
            for job in jobs
        ],
        "resume_ledger": [
            {
                "dataset": job["dataset"],
                "seed": job["seed"],
                "resume_requested_by_recorded_command": job[
                    "resume_requested_by_recorded_command"
                ],
                **job["resume_evidence"],
                "rng_state_limitation": (
                    "checkpoint did not store RNG/DataLoader state; a resumed run is "
                    "not an uninterrupted bitwise-reproducible trajectory"
                    if job["resume_requested_by_recorded_command"]
                    else None
                ),
            }
            for job in jobs
        ],
        "inference": inference_records,
        "authoritative_registry_construction": authority_records,
        "registry_precheckpoint_order": registry_precheckpoint,
        "target_registry": registry_summary,
    }
    validate_protocol_documents_unchanged(protocol_freeze["documents"])
    validate_source_files_unchanged(source_freeze)
    if git_worktree_provenance() != git_freeze:
        raise PersistenceAuditError("Git HEAD/worktree status changed before output")
    write_output_bundle(
        output_dir,
        rows=all_rows,
        summary=summary,
        provenance=provenance,
    )
    return summary, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary, output_dir = run_audit(args)
    recurrence = summary["recurrence"]["overall"]
    target_micro = recurrence["target_micro"]
    print(
        f"validated {summary['ledger_row_count']} target-run rows; "
        f"N3={target_micro['N3']}/{target_micro['target_count']} under "
        f"{summary['checkpoint_policy']}"
    )
    print(f"wrote immutable Gate E-1 bundle: {output_dir}")
    print("scope: development holdout only; official test remains sealed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PersistenceAuditError, FinalizationError, FileExistsError, OSError) as exc:
        print(f"Gate E-1 audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
