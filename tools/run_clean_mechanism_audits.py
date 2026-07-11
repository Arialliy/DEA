#!/usr/bin/env python3
"""Run the frozen clean-baseline mechanism audits on the complete 3x3 grid.

The runner is deliberately fail closed.  It starts only after the clean
development-holdout summary exists and the summary, scheduler manifest, job
results, metric logs, checkpoints, and run configs agree.  Audit jobs always
    use validation mode; the official-test split name is propagated solely so the
    exporter can check frozen training provenance and is never opened or iterated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.finalize_clean_baselines import (  # noqa: E402
    DATASET_NAMES,
    EXPECTED_EPOCHS,
    OUTPUT_JSON as BASELINE_SUMMARY_JSON,
    FinalizationError,
    load_checkpoint_cpu,
    normalized_path,
    parse_metrics,
    read_json,
    require_mapping,
    validate_checkpoint,
    validate_manifest,
    validate_result,
)


AUDIT_SCHEMA = "dea.clean_mechanism_audit.v1"
BATCH_SCHEMA = "dea.clean_mechanism_audit_batch.v1"
RESULT_SCHEMA = "dea.clean_mechanism_audit_job.v1"
AUDIT_DIR_NAME = "mechanism_audits"
AUDIT_MANIFEST = "batch_manifest.json"
FINALIZED_AUDIT_OUTPUTS = {
    "clean_mechanism_audit_evidence_summary.json",
    "clean_mechanism_audit_evidence_summary.md",
}
FIXED_GPUS = (2, 3)
CHECKPOINT_NAME = "checkpoint_best_iou.pkl"
OFFICIAL_TEST_STATUS = "sealed; this exporter accepts development validation only"
OFFICIAL_TEST_POLICY = (
    "validation mode only; official-test split path is propagated for frozen "
    "provenance checking and is never opened or iterated"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
AUDIT_SOURCE_FILES = {
    "exporter": PROJECT_DIR / "tools" / "export_clean_mechanism_audit.py",
    "baseline_finalizer": PROJECT_DIR / "tools" / "finalize_clean_baselines.py",
    "mshnet": PROJECT_DIR / "model" / "MSHNet.py",
    "mean_anchor_probe": PROJECT_DIR / "model" / "dea_scale_interaction_exchange.py",
    "component_candidates": PROJECT_DIR / "utils" / "component_evidence.py",
    "dataset": PROJECT_DIR / "utils" / "data.py",
    "metrics": PROJECT_DIR / "utils" / "metric.py",
}
FROZEN_BASELINE_RECIPE = {
    "epochs": EXPECTED_EPOCHS,
    "batch_size": 4,
    "num_workers": 4,
    "lr": 0.05,
    "warm_epoch": 5,
    "val_fraction": 0.2,
    "split_seed": 20260711,
    "deterministic": "true",
}
GRID_COMPLETION = "all_3x3_jobs_400_epochs_returncode_0_and_finalizer_validated"


class AuditBatchError(RuntimeError):
    """Raised when an audit batch cannot be started or safely resumed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schedule the finalized clean-baseline mechanism audits on GPUs 2/3."
    )
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only completed jobs whose result and every audit artifact verify.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all baseline inputs and print commands without writing artifacts.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit_source_sha256() -> dict[str, str]:
    return {name: sha256(path) for name, path in AUDIT_SOURCE_FILES.items()}


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditBatchError(f"{label} must be an integer, got {value!r}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AuditBatchError(f"{label} must be numeric, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditBatchError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise AuditBatchError(f"{label} must be finite, got {value!r}")
    return number


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AuditBatchError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _resolved(value: Any, label: str) -> Path:
    try:
        return normalized_path(value, label)
    except FinalizationError as exc:
        raise AuditBatchError(str(exc)) from exc


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        return require_mapping(read_json(path, label), label)
    except FinalizationError as exc:
        raise AuditBatchError(str(exc)) from exc


def _validate_summary_header(
    summary: dict[str, Any], *, batch_id: str, seeds: list[int]
) -> dict[str, Any]:
    expected = {
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
    mismatches = [
        f"{key}: summary={summary.get(key)!r} expected={value!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if mismatches:
        raise AuditBatchError(
            "baseline summary is not the required completed development grid: "
            + "; ".join(mismatches)
        )
    datasets = require_mapping(summary.get("datasets"), "baseline summary.datasets")
    if set(datasets) != set(DATASET_NAMES) or len(datasets) != len(DATASET_NAMES):
        raise AuditBatchError(f"baseline summary datasets must be exactly {DATASET_NAMES}")
    return datasets


def _validate_run_config(
    run_config: dict[str, Any],
    *,
    job: dict[str, Any],
    dataset_meta: dict[str, Any],
    manifest_args: dict[str, Any],
) -> dict[str, Any]:
    label = f"run_config {job['job_id']}"
    args = require_mapping(run_config.get("args"), f"{label}.args")
    required_exact = {
        "mode": "train",
        "model_type": "mshnet",
        "deterministic": True,
        "pin_memory": True,
        "epochs": EXPECTED_EPOCHS,
        "base_size": 256,
        "crop_size": 256,
        "batch_size": manifest_args.get("batch_size"),
        "num_workers": manifest_args.get("num_workers"),
        "lr": manifest_args.get("lr"),
        "warm_epoch": manifest_args.get("warm_epoch"),
        "seed": job["seed"],
        "run_label": job["job_id"],
        "split_seed": manifest_args.get("split_seed"),
        "train_split_file": job.get("train_file"),
        "test_split_file": job.get("test_file"),
        "val_split_file": "",
        "train_split_sha256": dataset_meta.get("fit_sha256"),
        "val_split_sha256": dataset_meta.get("val_sha256"),
        "test_split_sha256": dataset_meta.get("official_test_sha256"),
    }
    mismatches = [
        f"{key}: run_config={args.get(key)!r} expected={value!r}"
        for key, value in required_exact.items()
        if args.get(key) != value
    ]
    if _resolved(args.get("dataset_dir"), f"{label}.args.dataset_dir") != _resolved(
        job.get("dataset_dir"), f"manifest job {job['job_id']}.dataset_dir"
    ):
        mismatches.append("dataset_dir does not match manifest job")
    if _resolved(args.get("run_dir"), f"{label}.args.run_dir") != _resolved(
        job.get("run_dir"), f"manifest job {job['job_id']}.run_dir"
    ):
        mismatches.append("run_dir does not match manifest job")
    if mismatches:
        raise AuditBatchError(f"{label} identity mismatch: " + "; ".join(mismatches))

    config: dict[str, Any] = {}
    for key in ("base_size", "crop_size", "batch_size", "num_workers"):
        config[key] = _exact_int(args.get(key), f"{label}.args.{key}")
    if config["base_size"] < 1 or config["crop_size"] < 1 or config["batch_size"] < 1:
        raise AuditBatchError(f"{label} contains non-positive size/batch settings")
    if config["num_workers"] < 0:
        raise AuditBatchError(f"{label} contains a negative num_workers")
    config["val_fraction"] = _finite_number(args.get("val_fraction"), f"{label}.val_fraction")
    if not 0.0 < config["val_fraction"] < 1.0:
        raise AuditBatchError(f"{label}.val_fraction must be in (0, 1)")
    config.update(
        dataset_dir=str(_resolved(args["dataset_dir"], f"{label}.dataset_dir")),
        train_split_file=args["train_split_file"],
        val_split_file=args["val_split_file"],
        test_split_file=args["test_split_file"],
        train_split_sha256=args["train_split_sha256"],
        val_split_sha256=args["val_split_sha256"],
        test_split_sha256=args["test_split_sha256"],
        split_seed=args["split_seed"],
    )
    return config


def load_validated_baseline_jobs(
    batch_dir: Path,
    *,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = load_checkpoint_cpu,
    hash_file: Callable[[str | Path], str] = sha256,
) -> list[dict[str, Any]]:
    """Revalidate the finalized baseline and return the exact nine audit inputs."""
    batch_dir = batch_dir.expanduser().resolve()
    manifest = _read_mapping(batch_dir / "manifest.json", "clean baseline manifest")
    try:
        seeds, datasets_meta = validate_manifest(manifest, batch_dir)
    except FinalizationError as exc:
        raise AuditBatchError(str(exc)) from exc
    summary_path = batch_dir / BASELINE_SUMMARY_JSON
    summary = _read_mapping(summary_path, "completed clean baseline summary")
    summary_datasets = _validate_summary_header(
        summary, batch_id=batch_dir.name, seeds=seeds
    )
    manifest_args = require_mapping(manifest.get("args"), "manifest.args")
    recipe_mismatches = [
        f"{key}: manifest={manifest_args.get(key)!r} expected={value!r}"
        for key, value in FROZEN_BASELINE_RECIPE.items()
        if manifest_args.get(key) != value
    ]
    if recipe_mismatches:
        raise AuditBatchError(
            "clean baseline manifest recipe is not frozen: " + "; ".join(recipe_mismatches)
        )

    manifest_jobs = {
        (job["dataset"], job["seed"]): job for job in manifest["jobs"]
    }
    validated_jobs: list[dict[str, Any]] = []
    for dataset_name in DATASET_NAMES:
        dataset_meta = require_mapping(
            datasets_meta[dataset_name], f"manifest.datasets.{dataset_name}"
        )
        summary_dataset = require_mapping(
            summary_datasets[dataset_name], f"summary.datasets.{dataset_name}"
        )
        expected_hashes = {
            "fit": dataset_meta.get("fit_sha256"),
            "validation": dataset_meta.get("val_sha256"),
            "official_test_audit_only": dataset_meta.get("official_test_sha256"),
        }
        if summary_dataset.get("split_hashes") != expected_hashes:
            raise AuditBatchError(f"summary split hashes disagree for {dataset_name}")
        runs = summary_dataset.get("runs")
        if not isinstance(runs, list) or len(runs) != len(seeds):
            raise AuditBatchError(
                f"summary {dataset_name} must contain exactly {len(seeds)} runs"
            )
        run_by_seed: dict[int, dict[str, Any]] = {}
        for index, raw_run in enumerate(runs):
            run = require_mapping(raw_run, f"summary {dataset_name}.runs[{index}]")
            seed = _exact_int(run.get("seed"), f"summary {dataset_name}.runs[{index}].seed")
            if seed not in seeds or seed in run_by_seed:
                raise AuditBatchError(f"unexpected/duplicate summary run for {dataset_name}/{seed}")
            run_by_seed[seed] = run
        if set(run_by_seed) != set(seeds):
            raise AuditBatchError(f"summary seed grid is incomplete for {dataset_name}")

        for seed in seeds:
            job = require_mapping(
                manifest_jobs[(dataset_name, seed)],
                f"manifest job {dataset_name}/{seed}",
            )
            result = _read_mapping(Path(job["result_file"]), f"result {job['job_id']}")
            try:
                validate_result(result, job)
            except FinalizationError as exc:
                raise AuditBatchError(str(exc)) from exc
            run_dir = _resolved(job["run_dir"], f"manifest {job['job_id']}.run_dir")
            try:
                rows = parse_metrics(run_dir / "epoch_metric.log")
            except FinalizationError as exc:
                raise AuditBatchError(str(exc)) from exc
            if len(rows) != EXPECTED_EPOCHS or [row["epoch"] for row in rows] != list(
                range(EXPECTED_EPOCHS)
            ):
                raise AuditBatchError(
                    f"{job['job_id']} must contain exactly epochs 0..{EXPECTED_EPOCHS - 1}"
                )
            checkpoint_path = run_dir / CHECKPOINT_NAME
            checkpoint = checkpoint_loader(checkpoint_path)
            try:
                checkpoint_summary = validate_checkpoint(
                    checkpoint, job, dataset_meta, manifest_args, rows
                )
            except FinalizationError as exc:
                raise AuditBatchError(str(exc)) from exc
            finally:
                del checkpoint

            summary_run = run_by_seed[seed]
            expected_checkpoint = _resolved(
                summary_run.get("checkpoint"),
                f"summary {dataset_name}/{seed}.checkpoint",
            )
            if expected_checkpoint != checkpoint_path.resolve():
                raise AuditBatchError(
                    f"summary checkpoint path mismatch for {dataset_name}/{seed}"
                )
            for key in ("best_epoch", "iou", "pd", "fa"):
                if not _same_number(summary_run.get(key), checkpoint_summary.get(key)):
                    raise AuditBatchError(
                        f"summary/checkpoint {key} mismatch for {dataset_name}/{seed}"
                    )
            config = _validate_run_config(
                _read_mapping(run_dir / "run_config.json", f"run_config {job['job_id']}"),
                job=job,
                dataset_meta=dataset_meta,
                manifest_args=manifest_args,
            )
            validated_jobs.append(
                {
                    "batch_id": batch_dir.name,
                    "baseline_manifest": str((batch_dir / "manifest.json").resolve()),
                    "baseline_summary": str(summary_path.resolve()),
                    "baseline_job_id": job["job_id"],
                    "dataset": dataset_name,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": hash_file(checkpoint_path),
                    "baseline_metrics": {
                        key: summary_run[key]
                        for key in ("best_epoch", "iou", "pd", "fa")
                    },
                    "config": config,
                }
            )
    if len(validated_jobs) != len(DATASET_NAMES) * len(seeds):
        raise AuditBatchError("internal error: baseline audit grid is not exactly 3 x 3")
    return validated_jobs


def _argument(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def build_audit_jobs(
    baseline_jobs: list[dict[str, Any]],
    *,
    batch_id: str,
    audit_root: Path,
    python_executable: str | Path = sys.executable,
) -> list[dict[str, Any]]:
    audit_root = audit_root.resolve()
    frozen_sources = audit_source_sha256()
    jobs: list[dict[str, Any]] = []
    for source in baseline_jobs:
        dataset, seed = source["dataset"], source["seed"]
        job_id = f"mean_anchor__{dataset.lower()}__seed_{seed}"
        output_dir = audit_root / "artifacts" / dataset / f"seed_{seed}"
        log_file = audit_root / "logs" / f"{job_id}.log"
        result_file = audit_root / "jobs" / f"{job_id}.json"
        config = source["config"]
        command = [
            str(Path(python_executable).expanduser().resolve()),
            str((PROJECT_DIR / "tools" / "export_clean_mechanism_audit.py").resolve()),
            "--checkpoint", source["checkpoint"],
            "--checkpoint-role", "best_iou",
            "--batch-id", batch_id,
            "--output-dir", str(output_dir),
            "--dataset-dir", config["dataset_dir"],
            "--mode", "val",
            "--train-split-file", config["train_split_file"],
            "--val-split-file", config["val_split_file"],
            "--test-split-file", config["test_split_file"],
            "--val-fraction", _argument(config["val_fraction"]),
            "--split-seed", _argument(config["split_seed"]),
            "--seed", _argument(seed),
            "--base-size", _argument(config["base_size"]),
            "--crop-size", _argument(config["crop_size"]),
            "--batch-size", _argument(config["batch_size"]),
            "--num-workers", _argument(config["num_workers"]),
            "--input-channels", "3",
            "--eps", "1e-6",
            "--candidate-thresholds", "0.5", "0.3", "0.2", "0.1",
            "--device", "cuda",
        ]
        jobs.append(
            {
                "job_id": job_id,
                "batch_id": source["batch_id"],
                "baseline_manifest": source["baseline_manifest"],
                "baseline_summary": source["baseline_summary"],
                "baseline_job_id": source["baseline_job_id"],
                "dataset": dataset,
                "seed": seed,
                "checkpoint": source["checkpoint"],
                "checkpoint_sha256": source["checkpoint_sha256"],
                "source_sha256": frozen_sources,
                "baseline_metrics": source["baseline_metrics"],
                "config": config,
                "output_dir": str(output_dir),
                "log_file": str(log_file),
                "result_file": str(result_file),
                "command": command,
            }
        )
    if len(jobs) != 9:
        raise AuditBatchError(f"expected exactly 9 audit jobs, found {len(jobs)}")
    for field in ("output_dir", "log_file", "result_file"):
        values = [job[field] for job in jobs]
        if len(values) != len(set(values)):
            raise AuditBatchError(f"audit jobs do not have unique {field} paths")
    return jobs


def build_batch_spec(batch_dir: Path, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_manifest = (batch_dir / "manifest.json").resolve()
    baseline_summary = (batch_dir / BASELINE_SUMMARY_JSON).resolve()
    return {
        "schema_version": BATCH_SCHEMA,
        "batch_id": batch_dir.name,
        "stage": "development_holdout_mechanism_audit",
        "official_test_policy": OFFICIAL_TEST_POLICY,
        "gpu_ids": list(FIXED_GPUS),
        "max_processes_per_gpu": 1,
        "source_sha256": jobs[0]["source_sha256"],
        "baseline_manifest": str(baseline_manifest),
        "baseline_manifest_sha256": sha256(baseline_manifest),
        "baseline_summary": str(baseline_summary),
        "baseline_summary_sha256": sha256(baseline_summary),
        "jobs": jobs,
    }


def validate_batch_manifest(path: Path, spec: dict[str, Any]) -> None:
    manifest = _read_mapping(path, "mechanism-audit batch manifest")
    allowed = set(spec) | {"created_at_utc"}
    if set(manifest) != allowed or not isinstance(manifest.get("created_at_utc"), str):
        raise AuditBatchError("existing mechanism-audit batch manifest has an invalid shape")
    mismatches = [key for key, value in spec.items() if manifest.get(key) != value]
    if mismatches:
        raise AuditBatchError(
            "existing mechanism-audit batch manifest disagrees with frozen inputs: "
            + ", ".join(mismatches)
        )


def validate_root_inventory(audit_root: Path, jobs: list[dict[str, Any]]) -> None:
    """Reject files that are not part of the frozen scheduler layout."""
    # The audit finalizer writes these two immutable summaries only after the
    # 3x3 grid completes.  Their presence must not make a later verified
    # `--resume` look like foreign scheduler state.
    allowed_top = {
        AUDIT_MANIFEST,
        "artifacts",
        "logs",
        "jobs",
        *FINALIZED_AUDIT_OUTPUTS,
    }
    actual_top = {path.name for path in audit_root.iterdir()}
    unexpected = actual_top.difference(allowed_top)
    if unexpected:
        raise AuditBatchError(
            "unexpected entries in mechanism audit root: " + ", ".join(sorted(unexpected))
        )
    expected_logs = {Path(job["log_file"]).name for job in jobs}
    expected_results = {Path(job["result_file"]).name for job in jobs}
    for directory_name, expected_names in (
        ("logs", expected_logs),
        ("jobs", expected_results),
    ):
        directory = audit_root / directory_name
        if directory.exists() and not directory.is_dir():
            raise AuditBatchError(f"{directory} must be a directory")
        if directory.is_dir():
            bad = {
                path.name
                for path in directory.iterdir()
                if path.name not in expected_names or not path.is_file()
            }
            if bad:
                raise AuditBatchError(
                    f"unexpected entries in {directory_name}: " + ", ".join(sorted(bad))
                )
    artifacts = audit_root / "artifacts"
    if artifacts.exists() and not artifacts.is_dir():
        raise AuditBatchError(f"{artifacts} must be a directory")
    if artifacts.is_dir():
        expected_seeds = {
            dataset: {
                Path(job["output_dir"]).name
                for job in jobs
                if job["dataset"] == dataset
            }
            for dataset in DATASET_NAMES
        }
        for entry in artifacts.iterdir():
            if entry.name not in expected_seeds or not entry.is_dir():
                raise AuditBatchError(f"unexpected artifact dataset entry: {entry}")
            bad = {
                path.name
                for path in entry.iterdir()
                if path.name not in expected_seeds[entry.name] or not path.is_dir()
            }
            if bad:
                raise AuditBatchError(
                    f"unexpected artifact runs for {entry.name}: " + ", ".join(sorted(bad))
                )


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditBatchError(f"missing {label}: {path}")
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AuditBatchError(f"cannot read {label}: {path}: {exc}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise AuditBatchError(f"{label} must contain non-blank JSONL records")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditBatchError(
                f"invalid {label} JSON at line {line_number}: {exc}"
            ) from exc
        records.append(require_mapping(value, f"{label} line {line_number}"))
    return records


def _confined_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AuditBatchError(f"{label} must be a non-empty relative path")
    rel = Path(relative)
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise AuditBatchError(f"unsafe {label}: {relative!r}")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AuditBatchError(f"{label} escapes audit output: {relative!r}") from exc
    if not candidate.is_file():
        raise AuditBatchError(f"missing {label}: {candidate}")
    return candidate


def verify_audit_output(job: dict[str, Any]) -> dict[str, Any]:
    """Verify a completed exporter directory, including every per-image array."""
    output_dir = Path(job["output_dir"]).resolve()
    manifest = _read_mapping(output_dir / "manifest.json", f"audit {job['job_id']} manifest")
    expected_header = {
        "schema_version": AUDIT_SCHEMA,
        "dataset": job["dataset"],
        "dataset_dir": job["config"]["dataset_dir"],
        "split_role": "val",
        "split_sha256": job["config"]["val_split_sha256"],
        "validation_split_sha256": job["config"]["val_split_sha256"],
        "seed": job["seed"],
        "method": "MSHNet",
        "model_type": "mshnet",
        "base_size": job["config"]["base_size"],
        "crop_size": job["config"]["crop_size"],
        "batch_size": job["config"]["batch_size"],
        "num_workers": job["config"]["num_workers"],
        "deterministic": True,
        "threshold_probability": 0.5,
        "threshold_logit": 0.0,
        "connectivity": 2,
        "max_centroid_distance": 3.0,
        "anchor_mode": "mean",
        "active_stage": 0,
        "eps": 1e-6,
        "candidate_probability_thresholds": [0.5, 0.3, 0.2, 0.1],
        "official_test_status": OFFICIAL_TEST_STATUS,
    }
    mismatches = [
        f"{key}: audit={manifest.get(key)!r} expected={value!r}"
        for key, value in expected_header.items()
        if manifest.get(key) != value
    ]
    if manifest.get("split_sha256") != manifest.get("validation_split_sha256"):
        mismatches.append("split_sha256 is not the validation split hash")
    checkpoint = require_mapping(manifest.get("checkpoint"), "audit checkpoint")
    if checkpoint.get("role") != "best_iou":
        mismatches.append("checkpoint role is not best_iou")
    if _resolved(checkpoint.get("path"), "audit checkpoint.path") != Path(
        job["checkpoint"]
    ).resolve():
        mismatches.append("checkpoint path mismatch")
    expected_checkpoint_hash = _require_sha256(
        job["checkpoint_sha256"], "job checkpoint_sha256"
    )
    if checkpoint.get("sha256") != expected_checkpoint_hash:
        mismatches.append("checkpoint hash mismatch")
    if sha256(job["checkpoint"]) != expected_checkpoint_hash:
        mismatches.append("checkpoint changed since batch validation")
    source_hashes = require_mapping(manifest.get("source_sha256"), "audit source_sha256")
    for name, expected in job["source_sha256"].items():
        if source_hashes.get(name) != _require_sha256(expected, f"job source hash {name}"):
            mismatches.append(f"audit source hash mismatch: {name}")
    if set(source_hashes) != set(job["source_sha256"]):
        mismatches.append("audit source hash keys mismatch")
    if audit_source_sha256() != job["source_sha256"]:
        mismatches.append("audit source files changed since batch validation")
    metrics = require_mapping(checkpoint.get("metrics"), "audit checkpoint.metrics")
    for key in ("iou", "pd", "fa"):
        if not _same_number(metrics.get(key), job["baseline_metrics"][key]):
            mismatches.append(f"checkpoint metric mismatch: {key}")
    if not _same_number(checkpoint.get("epoch"), job["baseline_metrics"]["best_epoch"]):
        mismatches.append("checkpoint epoch mismatch")

    provenance = require_mapping(manifest.get("baseline_provenance"), "baseline_provenance")
    expected_provenance = {
        "batch_id": job["batch_id"],
        "job_id": job["baseline_job_id"],
        "batch_manifest": job["baseline_manifest"],
        "baseline_summary": job["baseline_summary"],
        "completion": GRID_COMPLETION,
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            mismatches.append(f"baseline provenance mismatch: {key}")
    validation = require_mapping(manifest.get("checkpoint_validation"), "checkpoint_validation")
    if validation.get("model_seed_val_hash") != "matched":
        mismatches.append("checkpoint model/seed/validation hash not certified")
    if validation.get("strict_state_dict") is not True or validation.get("frozen") is not True:
        mismatches.append("checkpoint was not certified strict and frozen")
    recomputed = require_mapping(validation.get("recomputed_metrics"), "recomputed_metrics")
    for key in ("iou", "pd", "fa"):
        pair = require_mapping(recomputed.get(key), f"recomputed_metrics.{key}")
        if not _same_number(pair.get("checkpoint"), job["baseline_metrics"][key]):
            mismatches.append(f"recomputed checkpoint metric mismatch: {key}")
        if not _same_number(pair.get("checkpoint"), pair.get("recomputed")):
            mismatches.append(f"factual audit did not reproduce metric: {key}")
    if mismatches:
        raise AuditBatchError(
            f"completed audit identity mismatch for {job['job_id']}: " + "; ".join(mismatches)
        )

    artifacts = require_mapping(manifest.get("artifacts"), "audit artifacts")
    if artifacts.get("arrays_dir") != "arrays":
        raise AuditBatchError(f"unexpected arrays_dir for {job['job_id']}")
    images_path = _confined_file(output_dir, artifacts.get("images_jsonl"), "images_jsonl")
    components_path = _confined_file(
        output_dir, artifacts.get("components_jsonl"), "components_jsonl"
    )
    if sha256(images_path) != _require_sha256(
        artifacts.get("images_sha256"), "artifacts.images_sha256"
    ):
        raise AuditBatchError(f"images.jsonl hash mismatch for {job['job_id']}")
    if sha256(components_path) != _require_sha256(
        artifacts.get("components_sha256"), "artifacts.components_sha256"
    ):
        raise AuditBatchError(f"components.jsonl hash mismatch for {job['job_id']}")
    image_rows = _read_jsonl(images_path, "audit images")
    _read_jsonl(components_path, "audit components")
    array_count = _exact_int(artifacts.get("array_count"), "artifacts.array_count")
    if len(image_rows) != array_count:
        raise AuditBatchError(f"array/image count mismatch for {job['job_id']}")
    summary = require_mapping(manifest.get("summary"), "audit summary")
    if _exact_int(summary.get("images"), "audit summary.images") != array_count:
        raise AuditBatchError(f"summary/image count mismatch for {job['job_id']}")
    for summary_key, metric_key in (
        ("pooled_iou", "iou"),
        ("pd", "pd"),
        ("fa_per_million", "fa"),
    ):
        if not _same_number(summary.get(summary_key), job["baseline_metrics"][metric_key]):
            raise AuditBatchError(
                f"audit summary does not reproduce {metric_key} for {job['job_id']}"
            )
    reconstruction_error = _finite_number(
        manifest.get("max_mobius_reconstruction_abs_error"),
        "max_mobius_reconstruction_abs_error",
    )
    if reconstruction_error < 0:
        raise AuditBatchError(f"negative reconstruction error for {job['job_id']}")
    inventory: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, row in enumerate(image_rows):
        image_id = row.get("image_id")
        relative = row.get("array_path")
        if not isinstance(image_id, str) or not image_id or image_id in seen_ids:
            raise AuditBatchError(f"unsafe/duplicate audit image id at row {index}")
        if not isinstance(relative, str) or relative in seen_paths:
            raise AuditBatchError(f"duplicate/invalid array path at row {index}")
        array_path = _confined_file(output_dir, relative, f"array row {index}")
        digest = _require_sha256(row.get("array_sha256"), f"array row {index} sha256")
        size = _exact_int(row.get("array_bytes"), f"array row {index} bytes")
        if array_path.stat().st_size != size or sha256(array_path) != digest:
            raise AuditBatchError(f"array integrity mismatch: {relative}")
        inventory.append(
            {"image_id": image_id, "path": relative, "sha256": digest, "bytes": size}
        )
        seen_ids.add(image_id)
        seen_paths.add(relative)
    inventory_bytes = json.dumps(
        inventory, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if hashlib.sha256(inventory_bytes).hexdigest() != _require_sha256(
        artifacts.get("array_inventory_sha256"), "artifacts.array_inventory_sha256"
    ):
        raise AuditBatchError(f"array inventory hash mismatch for {job['job_id']}")
    if sum(item["bytes"] for item in inventory) != _exact_int(
        artifacts.get("array_total_bytes"), "artifacts.array_total_bytes"
    ):
        raise AuditBatchError(f"array byte total mismatch for {job['job_id']}")
    arrays_dir = output_dir / "arrays"
    if not arrays_dir.is_dir():
        raise AuditBatchError(f"missing arrays directory for {job['job_id']}")
    actual_arrays = {
        str(path.relative_to(output_dir)) for path in arrays_dir.iterdir() if path.is_file()
    }
    if actual_arrays != seen_paths or any(path.is_dir() for path in arrays_dir.iterdir()):
        raise AuditBatchError(f"unexpected/missing array entries for {job['job_id']}")
    allowed_top = {"manifest.json", "images.jsonl", "components.jsonl", "arrays"}
    if {path.name for path in output_dir.iterdir()} != allowed_top:
        raise AuditBatchError(f"unexpected or temporary files in {job['job_id']} output")
    return manifest


def validate_completed_result(job: dict[str, Any]) -> dict[str, Any]:
    result_path, log_path = Path(job["result_file"]), Path(job["log_file"])
    result = _read_mapping(result_path, f"audit result {job['job_id']}")
    expected = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_verified",
        "job_id": job["job_id"],
        "dataset": job["dataset"],
        "seed": job["seed"],
        "returncode": 0,
        "command": job["command"],
        "output_dir": job["output_dir"],
        "log_file": job["log_file"],
        "checkpoint": job["checkpoint"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "source_sha256": job["source_sha256"],
    }
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    if mismatches:
        raise AuditBatchError(
            f"completed result mismatch for {job['job_id']}: " + ", ".join(mismatches)
        )
    if _exact_int(result.get("returncode"), f"result {job['job_id']}.returncode") != 0:
        raise AuditBatchError(f"completed result returncode is not zero: {job['job_id']}")
    if _exact_int(result.get("seed"), f"result {job['job_id']}.seed") != job["seed"]:
        raise AuditBatchError(f"completed result seed mismatch: {job['job_id']}")
    if _exact_int(result.get("gpu"), f"result {job['job_id']}.gpu") not in FIXED_GPUS:
        raise AuditBatchError(f"completed result used an unauthorized GPU: {job['job_id']}")
    if _exact_int(result.get("pid"), f"result {job['job_id']}.pid") < 1:
        raise AuditBatchError(f"completed result has an invalid pid: {job['job_id']}")
    elapsed = _finite_number(
        result.get("elapsed_seconds"), f"result {job['job_id']}.elapsed_seconds"
    )
    if elapsed < 0:
        raise AuditBatchError(f"completed result has negative elapsed time: {job['job_id']}")
    for field in ("started_at", "finished_at"):
        if not isinstance(result.get(field), str) or not result[field]:
            raise AuditBatchError(f"completed result has invalid {field}: {job['job_id']}")
    if not log_path.is_file() or log_path.stat().st_size == 0:
        raise AuditBatchError(f"missing completed audit log: {log_path}")
    verify_audit_output(job)
    return result


def classify_jobs(jobs: list[dict[str, Any]], *, resume: bool) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for job in jobs:
        output_dir = Path(job["output_dir"])
        output_nonempty = output_dir.exists() and (
            not output_dir.is_dir() or any(output_dir.iterdir())
        )
        result_exists = Path(job["result_file"]).exists()
        log_exists = Path(job["log_file"]).exists()
        if output_nonempty or result_exists or log_exists:
            if not resume:
                raise AuditBatchError(
                    f"refusing non-empty audit outputs for {job['job_id']}; "
                    "pass --resume only for a completed verified audit"
                )
            if not (output_nonempty and result_exists and log_exists):
                raise AuditBatchError(
                    f"partial audit artifacts cannot be resumed safely: {job['job_id']}"
                )
            validate_completed_result(job)
            print(f"skip completed verified {job['job_id']}", flush=True)
            continue
        if output_dir.exists() and not output_dir.is_dir():
            raise AuditBatchError(f"audit output path is not a directory: {output_dir}")
        pending.append(job)
    return pending


def prepare_root(
    audit_root: Path,
    *,
    spec: dict[str, Any],
    jobs: list[dict[str, Any]],
    resume: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    audit_root = audit_root.resolve()
    nonempty = audit_root.exists() and (
        not audit_root.is_dir() or any(audit_root.iterdir())
    )
    manifest_path = audit_root / AUDIT_MANIFEST
    if nonempty:
        if not resume:
            raise AuditBatchError(
                f"mechanism audit root is non-empty: {audit_root}; "
                "only --resume of completed verified jobs is allowed"
            )
        validate_batch_manifest(manifest_path, spec)
        validate_root_inventory(audit_root, jobs)
    elif audit_root.exists() and not audit_root.is_dir():
        raise AuditBatchError(f"mechanism audit root is not a directory: {audit_root}")

    if not dry_run and not nonempty:
        audit_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            manifest_path,
            {
                **spec,
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
    pending = classify_jobs(jobs, resume=resume)
    if not dry_run:
        (audit_root / "logs").mkdir(parents=True, exist_ok=True)
        (audit_root / "jobs").mkdir(parents=True, exist_ok=True)
        for dataset in DATASET_NAMES:
            (audit_root / "artifacts" / dataset).mkdir(parents=True, exist_ok=True)
    return pending


def _result_payload(
    *,
    job: dict[str, Any],
    gpu: int,
    pid: int | None,
    started_at: str,
    elapsed_seconds: float,
    returncode: int | None,
    status: str,
    verification_error: str | None = None,
    launch_error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "job_id": job["job_id"],
        "dataset": job["dataset"],
        "seed": job["seed"],
        "gpu": gpu,
        "pid": pid,
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "returncode": returncode,
        "command": job["command"],
        "output_dir": job["output_dir"],
        "log_file": job["log_file"],
        "checkpoint": job["checkpoint"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "source_sha256": job["source_sha256"],
    }
    if verification_error is not None:
        payload["verification_error"] = verification_error
    if launch_error is not None:
        payload["launch_error"] = launch_error
    return payload


def run_jobs(
    jobs: list[dict[str, Any]],
    *,
    poll_seconds: float = 2.0,
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
) -> list[str]:
    if not math.isfinite(poll_seconds) or poll_seconds < 0:
        raise AuditBatchError("--poll-seconds must be finite and non-negative")
    pending = list(jobs)
    active: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    while pending or active:
        for gpu in FIXED_GPUS:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            output_dir = Path(job["output_dir"])
            if output_dir.exists() and any(output_dir.iterdir()):
                raise AuditBatchError(f"refusing non-empty output at launch: {output_dir}")
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            log_path = Path(job["log_file"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            started_at = dt.datetime.now(dt.timezone.utc).isoformat()
            started_monotonic = time.monotonic()
            log_handle = log_path.open("x", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            try:
                process = popen(
                    job["command"],
                    cwd=PROJECT_DIR,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                log_handle.close()
                payload = _result_payload(
                    job=job,
                    gpu=gpu,
                    pid=None,
                    started_at=started_at,
                    elapsed_seconds=time.monotonic() - started_monotonic,
                    returncode=None,
                    status="launch_failed",
                    launch_error=f"{type(exc).__name__}: {exc}",
                )
                write_json_atomic(Path(job["result_file"]), payload)
                failures.append(job["job_id"])
                print(f"launch failed gpu={gpu} job={job['job_id']}: {exc}", file=sys.stderr)
                continue
            active[gpu] = {
                "job": job,
                "process": process,
                "log_handle": log_handle,
                "started_at": started_at,
                "started_monotonic": started_monotonic,
            }
            print(f"start gpu={gpu} pid={process.pid} job={job['job_id']}", flush=True)

        if active and poll_seconds:
            time.sleep(poll_seconds)
        for gpu, state in list(active.items()):
            process = state["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            state["log_handle"].close()
            job = state["job"]
            verification_error = None
            status = "process_failed"
            if returncode == 0:
                try:
                    verify_audit_output(job)
                except (AuditBatchError, FinalizationError, OSError, ValueError) as exc:
                    verification_error = f"{type(exc).__name__}: {exc}"
                    status = "verification_failed"
                else:
                    status = "completed_verified"
            payload = _result_payload(
                job=job,
                gpu=gpu,
                pid=process.pid,
                started_at=state["started_at"],
                elapsed_seconds=time.monotonic() - state["started_monotonic"],
                returncode=returncode,
                status=status,
                verification_error=verification_error,
            )
            write_json_atomic(Path(job["result_file"]), payload)
            print(
                f"finish gpu={gpu} rc={returncode} status={status} job={job['job_id']} "
                f"elapsed={payload['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if status != "completed_verified":
                failures.append(job["job_id"])
            del active[gpu]
    return failures


def main() -> int:
    args = parse_args()
    if (
        not args.batch_id
        or args.batch_id in {".", ".."}
        or Path(args.batch_id).name != args.batch_id
        or SAFE_BATCH_ID_RE.fullmatch(args.batch_id) is None
    ):
        raise AuditBatchError("--batch-id must be one directory name, not a path")
    batch_dir = PROJECT_DIR / "repro_runs" / "clean" / args.batch_id
    baseline_jobs = load_validated_baseline_jobs(batch_dir)
    audit_root = batch_dir / AUDIT_DIR_NAME
    jobs = build_audit_jobs(
        baseline_jobs,
        batch_id=args.batch_id,
        audit_root=audit_root,
    )
    spec = build_batch_spec(batch_dir, jobs)
    pending = prepare_root(
        audit_root,
        spec=spec,
        jobs=jobs,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = FIXED_GPUS[index % len(FIXED_GPUS)]
            print(f"GPU {gpu}: " + " ".join(job["command"]))
        print(
            f"validated finalized baseline; {len(pending)} pending / "
            f"{len(jobs) - len(pending)} completed verified; official test sealed"
        )
        return 0
    failures = run_jobs(pending, poll_seconds=args.poll_seconds)
    if failures:
        print("failed mechanism audits: " + ", ".join(failures), file=sys.stderr)
        return 1
    # Recheck both newly completed and resume-skipped jobs before declaring the grid done.
    for job in jobs:
        validate_completed_result(job)
    print(f"all {len(jobs)} clean mechanism audits completed and verified", flush=True)
    print("scope: development validation only; official test remains sealed", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditBatchError, FinalizationError, FileExistsError, OSError, ValueError) as exc:
        print(f"mechanism audit batch refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
