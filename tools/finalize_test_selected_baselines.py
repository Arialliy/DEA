#!/usr/bin/env python3
"""Fail-closed finalizer for the canonical test-selected MSHNet baselines.

The protocol summarized here trains on the complete canonical ``train`` list,
evaluates the canonical ``test`` list every ten completed epochs, and selects
checkpoints with those test measurements.  The resulting numbers are therefore
*test-selected* and are not estimates on an untouched test set.

This tool is deliberately read-only until every artifact has passed validation.
It cross-checks the frozen batch manifest, source and split hashes, the 3 x 3
job grid, run configuration, protocol summary, 40 scheduled metric rows, and
selected checkpoint contents.  Only then are summary files written atomically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL = "test_selected_full_train_interval_v1"
EXPECTED_STAGE = "full_canonical_train_periodic_test_selection"
DATASET_NAMES = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
EXPECTED_SEED_COUNT = 3
EXPECTED_EPOCHS = 400
TEST_INTERVAL = 10
EVALUATION_COMPLETED_EPOCHS = tuple(range(10, EXPECTED_EPOCHS + 1, 10))
EVALUATION_ZERO_BASED_EPOCHS = tuple(epoch - 1 for epoch in EVALUATION_COMPLETED_EPOCHS)
SELECTION_THRESHOLD = 0.5
PD_FA_MIN_PD = 0.93
PD_FA_MIN_IOU = 0.655
PAIRED_BASELINE_IOU = 0.0
TIE_BREAK = "earliest_epoch"
BEST_IOU_RULE = "strictly_greater_iou; ties_keep_earliest_epoch"
PD_FA_RULE = (
    "pd>=0.93 and iou>=0.655 then strictly_minimum_fa; "
    "ties_keep_earliest_epoch"
)
EVALUATION_EPOCH_RULE = (
    "(epoch_zero_based + 1) % test_interval == 0 or "
    "epoch_zero_based == total_epochs - 1"
)

OUTPUT_JSON = "test_selected_baseline_summary.json"
OUTPUT_MARKDOWN = "test_selected_baseline_summary.md"

FLOAT_PATTERN = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
METRIC_RE = re.compile(
    rf"^\S+\s+-\s+(?P<epoch>\d+)\s+-\s+IoU\s+"
    rf"(?P<iou>{FLOAT_PATTERN})\s+-\s+PD\s+(?P<pd>{FLOAT_PATTERN})"
    rf"\s+-\s+FA\s+(?P<fa>{FLOAT_PATTERN})\s*$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# Independent copy of the split contract frozen by the scheduler.  Supplying a
# different contract is supported only by the programmatic API for synthetic
# tests; the CLI always uses this canonical table.
CANONICAL_DATASETS: dict[str, dict[str, dict[str, Any]]] = {
    "NUAA-SIRST": {
        "train": {
            "count": 213,
            "raw_sha256": "324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4",
            "ordered_names_sha256": "815dcca749f087f27f5dad4b447015aee70bd7ae7779d6fdd7d6efa6d5c6943f",
        },
        "test": {
            "count": 214,
            "raw_sha256": "e49023203a323c247306b314f23c8b3b917093a26984067792355adff7a8386e",
            "ordered_names_sha256": "395eecd6bf0ed2a59f531de9145688597632c68f9d0933359aadcb93ec1a60b5",
        },
    },
    "NUDT-SIRST": {
        "train": {
            "count": 663,
            "raw_sha256": "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3",
            "ordered_names_sha256": "dc555df66b62dd1ea98d119ace8fe8ae86de94f3e4833d8d81e90c0e1f287922",
        },
        "test": {
            "count": 664,
            "raw_sha256": "a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e",
            "ordered_names_sha256": "cec44220c69d89a5b3fd245b8ee911404e959fef80bd96b32b6b74f28bb32af0",
        },
    },
    "IRSTD-1K": {
        "train": {
            "count": 800,
            "raw_sha256": "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744",
            "ordered_names_sha256": "b698d2d9dbe9e26e1875978d23450e1e6ec45fd71d56d31415007f56c40bba88",
        },
        "test": {
            "count": 201,
            "raw_sha256": "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579",
            "ordered_names_sha256": "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579",
        },
    },
}


class FinalizationError(RuntimeError):
    """Raised when a required artifact or invariant is missing or inconsistent."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize a complete 3-dataset x 3-seed MSHNet "
            "batch whose checkpoints were selected on periodic test evaluations."
        )
    )
    parser.add_argument(
        "--batch-id",
        default="mshnet_test_selected_full_train_interval_v1",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace prior finalizer outputs, but only after validation succeeds.",
    )
    return parser.parse_args(argv)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    _require_plain_file(path, "hashed file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names: Iterable[str]) -> str:
    return sha256_bytes(("\n".join(names) + "\n").encode("utf-8"))


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _require_plain_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FinalizationError(f"missing or non-plain {label}: {path}")


def read_json(path: Path, label: str) -> Any:
    _require_plain_file(path, label)
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"invalid {label}: {path}: {exc}") from exc


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} must be a JSON/object mapping")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FinalizationError(f"{label} must be a JSON list")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FinalizationError(f"{label} must be an integer, got {value!r}")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FinalizationError(f"{label} must be numeric, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise FinalizationError(f"{label} must be finite, got {value!r}")
    return result


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FinalizationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise FinalizationError(
            f"{label} mismatch: actual={actual!r}, expected={expected!r}"
        )


def normalized_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FinalizationError(f"{label} must be a non-empty path string")
    return Path(value).expanduser().resolve()


def parse_csv(value: Any, cast: Callable[[str], Any], label: str) -> list[Any]:
    if not isinstance(value, str):
        raise FinalizationError(f"{label} must be a comma-separated string")
    raw = [item.strip() for item in value.split(",") if item.strip()]
    try:
        result = [cast(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"invalid {label}: {value!r}") from exc
    if not result or len(result) != len(set(result)):
        raise FinalizationError(f"{label} must contain unique non-empty values")
    return result


def command_value(command: Any, flag: str, label: str) -> str:
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise FinalizationError(f"{label}.command must be a list of strings")
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise FinalizationError(f"{label}.command must contain exactly one {flag}")
    return command[positions[0] + 1]


def _read_split(path: Path, label: str) -> tuple[list[str], str, str, bytes]:
    _require_plain_file(path, label)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise FinalizationError(f"{label} is not strict UTF-8: {path}") from exc
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise FinalizationError(f"{label} must contain unique non-empty names")
    for name in names:
        if Path(name).name != name or Path(name).suffix or "hcval" in name.casefold():
            raise FinalizationError(f"unsafe sample name in {label}: {name!r}")
    return names, sha256_bytes(raw), names_sha256(names), raw


def _validate_sources(
    manifest: dict[str, Any], expected_source_paths: Mapping[str, Path]
) -> dict[str, str]:
    sources = require_mapping(manifest.get("sources"), "manifest.sources")
    immutable = require_mapping(
        manifest.get("immutable_contract"), "manifest.immutable_contract"
    )
    frozen_sources = require_mapping(
        immutable.get("source_sha256"), "immutable_contract.source_sha256"
    )
    output: dict[str, str] = {}
    specification = {
        "scheduler": ("scheduler", "scheduler_sha256"),
        "training_entry": ("training_entry", "training_entry_sha256"),
    }
    for role, (path_key, hash_key) in specification.items():
        path = normalized_path(sources.get(path_key), f"manifest.sources.{path_key}")
        expected_name = (
            "run_test_selected_baselines.py"
            if role == "scheduler"
            else "train_test_selected_full_train.py"
        )
        if path.name != expected_name:
            raise FinalizationError(
                f"manifest source {role} has unexpected basename: {path}"
            )
        if role not in expected_source_paths:
            raise FinalizationError(f"missing expected source path for {role}")
        require_equal(
            path,
            Path(expected_source_paths[role]).resolve(),
            f"manifest source path for {role}",
        )
        declared = require_sha256(
            sources.get(hash_key), f"manifest.sources.{hash_key}"
        )
        require_equal(
            frozen_sources.get(role), declared, f"immutable source hash for {role}"
        )
        actual = sha256_file(path)
        require_equal(actual, declared, f"current source hash for {role}")
        output[role] = actual
    return output


def _validate_dataset_splits(
    manifest: dict[str, Any],
    expected_datasets: Mapping[str, Mapping[str, Mapping[str, Any]]],
    expected_dataset_root: Path,
) -> dict[str, dict[str, Any]]:
    data_protocol = require_mapping(
        manifest.get("data_protocol"), "manifest.data_protocol"
    )
    dataset_root = normalized_path(
        data_protocol.get("dataset_root"), "manifest.data_protocol.dataset_root"
    )
    require_equal(
        dataset_root,
        expected_dataset_root.resolve(),
        "manifest canonical dataset root",
    )
    require_equal(data_protocol.get("validation_role"), None, "validation_role")
    require_equal(data_protocol.get("validation_fraction"), 0.0, "validation_fraction")
    require_equal(data_protocol.get("hcval_policy"), "prohibited_not_read", "hcval_policy")
    require_equal(
        data_protocol.get("fit_role"),
        "complete canonical img_idx/train_<dataset>.txt",
        "data_protocol.fit_role",
    )
    datasets = require_mapping(manifest.get("datasets"), "manifest.datasets")
    if set(datasets) != set(DATASET_NAMES) or set(expected_datasets) != set(DATASET_NAMES):
        raise FinalizationError(f"dataset contract must be exactly {DATASET_NAMES}")

    audited: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_NAMES:
        meta = require_mapping(datasets[dataset], f"manifest.datasets.{dataset}")
        require_equal(meta.get("dataset"), dataset, f"{dataset}.dataset")
        expected_dir = (dataset_root / dataset).resolve()
        require_equal(
            normalized_path(meta.get("dataset_dir"), f"{dataset}.dataset_dir"),
            expected_dir,
            f"{dataset}.dataset_dir",
        )
        require_equal(meta.get("validation_split"), None, f"{dataset}.validation_split")
        require_equal(meta.get("hcval_policy"), "prohibited_not_read", f"{dataset}.hcval_policy")
        require_equal(meta.get("train_test_overlap_count"), 0, f"{dataset}.overlap")

        role_output: dict[str, Any] = {}
        seen: dict[str, set[str]] = {}
        for role in ("train", "test"):
            frozen = expected_datasets[dataset][role]
            path = expected_dir / "img_idx" / f"{role}_{dataset}.txt"
            names, raw_hash, ordered_hash, raw = _read_split(path, f"canonical {dataset} {role}")
            expected_count = require_int(frozen.get("count"), f"frozen {dataset}.{role}.count")
            expected_raw = require_sha256(
                frozen.get("raw_sha256"), f"frozen {dataset}.{role}.raw_sha256"
            )
            expected_ordered = require_sha256(
                frozen.get("ordered_names_sha256"),
                f"frozen {dataset}.{role}.ordered_names_sha256",
            )
            require_equal(len(names), expected_count, f"{dataset} {role} count")
            require_equal(raw_hash, expected_raw, f"{dataset} {role} raw hash")
            require_equal(ordered_hash, expected_ordered, f"{dataset} {role} ordered hash")
            require_equal(meta.get(f"{role}_count"), expected_count, f"manifest {dataset} {role} count")
            require_equal(
                meta.get(f"{role}_raw_sha256"), expected_raw, f"manifest {dataset} {role} raw hash"
            )
            require_equal(
                meta.get(f"{role}_ordered_names_sha256"),
                expected_ordered,
                f"manifest {dataset} {role} ordered hash",
            )
            require_equal(
                normalized_path(meta.get(f"{role}_split_file"), f"{dataset}.{role}_split_file"),
                path.resolve(),
                f"manifest {dataset} {role} path",
            )
            seen[role] = set(names)
            role_output[role] = {
                "path": str(path.resolve()),
                "count": expected_count,
                "raw_sha256": raw_hash,
                "ordered_names_sha256": ordered_hash,
                "normalized_payload": ("\n".join(names) + "\n").encode("utf-8"),
                "raw_payload": raw,
            }
        if seen["train"] & seen["test"]:
            raise FinalizationError(f"canonical train/test overlap for {dataset}")
        audited[dataset] = role_output
    return audited


def parse_metrics(path: Path) -> list[dict[str, float | int]]:
    _require_plain_file(path, "epoch_metric.log")
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise FinalizationError(f"cannot read {path}: {exc}") from exc
    if len(lines) != len(EVALUATION_ZERO_BASED_EPOCHS):
        raise FinalizationError(
            f"{path} must contain exactly {len(EVALUATION_ZERO_BASED_EPOCHS)} "
            f"scheduled test rows, got {len(lines)}"
        )
    rows: list[dict[str, float | int]] = []
    for line_number, (line, expected_epoch) in enumerate(
        zip(lines, EVALUATION_ZERO_BASED_EPOCHS), start=1
    ):
        match = METRIC_RE.fullmatch(line)
        if match is None:
            raise FinalizationError(f"unparseable metric row {path}:{line_number}: {line!r}")
        row: dict[str, float | int] = {
            "epoch_zero_based": int(match.group("epoch")),
            "completed_epoch": int(match.group("epoch")) + 1,
            "iou": float(match.group("iou")),
            "pd": float(match.group("pd")),
            "fa": float(match.group("fa")),
        }
        require_equal(row["epoch_zero_based"], expected_epoch, f"metric schedule {path}:{line_number}")
        if not 0.0 <= float(row["iou"]) <= 1.0:
            raise FinalizationError(f"IoU outside [0,1] at {path}:{line_number}")
        if not 0.0 <= float(row["pd"]) <= 1.0:
            raise FinalizationError(f"PD outside [0,1] at {path}:{line_number}")
        if not math.isfinite(float(row["fa"])) or float(row["fa"]) < 0.0:
            raise FinalizationError(f"invalid FA at {path}:{line_number}")
        rows.append(row)
    return rows


def load_checkpoint_cpu(path: Path) -> dict[str, Any]:
    _require_plain_file(path, "selected checkpoint")
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise FinalizationError(
            "PyTorch and NumPy are required to inspect checkpoints"
        ) from exc
    safe_types = [
        np._core.multiarray.scalar,  # type: ignore[attr-defined]
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.int32)),
        type(np.dtype(np.int64)),
    ]
    try:
        with torch.serialization.safe_globals(safe_types):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise FinalizationError(f"cannot safely load checkpoint {path}: {exc}") from exc
    return require_mapping(payload, f"checkpoint {path}")


def _expected_method_metadata(
    job: dict[str, Any], dataset_meta: dict[str, Any], manifest_args: dict[str, Any]
) -> dict[str, Any]:
    dataset_dir = normalized_path(job["dataset_dir"], f"{job['job_id']}.dataset_dir")
    return {
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL,
        "method": "MSHNet",
        "model_type": "mshnet",
        "mshnet_objective": "sls",
        "mshnet_side_supervision": "canonical",
        "mshnet_train_graph": "canonical_warm",
        "location_loss": "legacy",
        "side_location_loss": "same",
        "lambda_location": 1.0,
        "crwd_lambda": 0.0,
        "crwd_enabled": False,
        "dea_lite_enabled": False,
        "dataset_name": job["dataset"],
        "dataset_dir": str(dataset_dir),
        "canonical_datasets_root": str(dataset_dir.parent),
        "train_split_file": str(dataset_dir / job["train_split_arg"]),
        "test_split_file": str(dataset_dir / job["test_split_arg"]),
        "val_split_file": "",
        "val_split_sha256": "",
        "val_split_count": 0,
        "val_fraction": 0.0,
        "train_split_count": dataset_meta["train_count"],
        "test_split_count": dataset_meta["test_count"],
        "train_split_raw_sha256": job["train_split_raw_sha256"],
        "test_split_raw_sha256": job["test_split_raw_sha256"],
        "train_split_normalized_sha256": job["train_split_sha256"],
        "test_split_normalized_sha256": job["test_split_sha256"],
        "train_split_sha256": job["train_split_sha256"],
        "test_split_sha256": job["test_split_sha256"],
        "seed": job["seed"],
        "deterministic": True,
        "run_label": job["job_id"],
        "selection_split": "test",
        "evaluation_split": "test",
        "evaluation_alias": "val_loader_is_complete_canonical_test",
        "no_internal_holdout": True,
        "test_interval": TEST_INTERVAL,
        "evaluation_epoch_rule": EVALUATION_EPOCH_RULE,
        "selection_threshold": SELECTION_THRESHOLD,
        "selection_prediction_rule": "sigmoid(logit) > 0.5",
        "selection_tie_break": TIE_BREAK,
        "selection_best_iou_rule": BEST_IOU_RULE,
        "selection_pd_fa_rule": PD_FA_RULE,
        "selection_pd_fa_min_pd": PD_FA_MIN_PD,
        "selection_pd_fa_min_iou": PD_FA_MIN_IOU,
        "selection_paired_baseline_iou": PAIRED_BASELINE_IOU,
        "train_loader_drop_last": False,
        "warm_epoch": manifest_args["warm_epoch"],
        "init_from_baseline": "",
        "init_checkpoint_sha256": "",
    }


def _validate_method_metadata(
    metadata: Any,
    job: dict[str, Any],
    dataset_meta: dict[str, Any],
    manifest_args: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    metadata = require_mapping(metadata, label)
    expected = _expected_method_metadata(job, dataset_meta, manifest_args)
    for key, value in expected.items():
        require_equal(metadata.get(key), value, f"{label}.{key}")
    for key in ("dea_lambda_single", "dea_lambda_dec", "dea_lambda_empty"):
        require_equal(metadata.get(key), 0.0, f"{label}.{key}")
    return metadata


def _stable_metadata_view(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only process-local resume flags before cross-artifact comparison."""

    return {
        key: value
        for key, value in metadata.items()
        if key not in {"resume", "resumed_process"}
    }


def _validate_run_config(
    run_dir: Path,
    job: dict[str, Any],
    dataset_meta: dict[str, Any],
    manifest_args: dict[str, Any],
) -> dict[str, Any]:
    config = require_mapping(read_json(run_dir / "run_config.json", "run_config.json"), "run_config")
    args = require_mapping(config.get("args"), "run_config.args")
    expected_args = {
        "mode": "train",
        "model_type": "mshnet",
        "mshnet_objective": "sls",
        "mshnet_side_supervision": "canonical",
        "mshnet_train_graph": "canonical_warm",
        "location_loss": "legacy",
        "side_location_loss": "same",
        "lambda_location": 1.0,
        "crwd_lambda": 0.0,
        "epochs": EXPECTED_EPOCHS,
        "test_interval": TEST_INTERVAL,
        "threshold": SELECTION_THRESHOLD,
        "pd_fa_min_pd": PD_FA_MIN_PD,
        "pd_fa_min_iou": PD_FA_MIN_IOU,
        "paired_baseline_iou": PAIRED_BASELINE_IOU,
        "seed": job["seed"],
        "deterministic": True,
        "run_label": job["job_id"],
        "run_dir": str(run_dir),
        "dataset_dir": job["dataset_dir"],
        "train_split_file": str(Path(job["dataset_dir"]) / job["train_split_arg"]),
        "test_split_file": str(Path(job["dataset_dir"]) / job["test_split_arg"]),
        "val_split_file": "",
        "val_fraction": 0.0,
        "warm_epoch": manifest_args["warm_epoch"],
        "batch_size": manifest_args["batch_size"],
        "num_workers": manifest_args["num_workers"],
        "lr": manifest_args["lr"],
        "init_from_baseline": "",
    }
    for key, value in expected_args.items():
        require_equal(args.get(key), value, f"run_config.args.{key}")
    _validate_method_metadata(
        config.get("method_meta"), job, dataset_meta, manifest_args, "run_config.method_meta"
    )
    return config


def _validate_persisted_splits(
    run_dir: Path, dataset_audit: dict[str, Any]
) -> None:
    stale_val = run_dir / "split_val.txt"
    if stale_val.exists() or stale_val.is_symlink():
        raise FinalizationError(f"validation split artifact is forbidden: {stale_val}")
    for role in ("train", "test"):
        path = run_dir / f"split_{role}.txt"
        _require_plain_file(path, f"persisted {role} split")
        actual = path.read_bytes()
        expected = dataset_audit[role]["normalized_payload"]
        require_equal(actual, expected, f"persisted {role} split bytes in {run_dir}")


def _validate_result(
    result: dict[str, Any], job: dict[str, Any], manifest_args: dict[str, Any]
) -> None:
    label = f"job result {job['job_id']}"
    require_equal(result.get("protocol"), PROTOCOL, f"{label}.protocol")
    require_equal(result.get("job_id"), job["job_id"], f"{label}.job_id")
    require_equal(result.get("dataset"), job["dataset"], f"{label}.dataset")
    require_equal(result.get("seed"), job["seed"], f"{label}.seed")
    require_equal(require_int(result.get("returncode"), f"{label}.returncode"), 0, f"{label}.returncode")
    require_equal(result.get("total_epochs"), EXPECTED_EPOCHS, f"{label}.total_epochs")
    require_equal(result.get("test_interval"), TEST_INTERVAL, f"{label}.test_interval")
    require_equal(result.get("test_evaluation_epochs"), list(EVALUATION_COMPLETED_EPOCHS), f"{label}.evaluation epochs")
    require_equal(result.get("test_selected"), True, f"{label}.test_selected")
    require_equal(normalized_path(result.get("run_dir"), f"{label}.run_dir"), Path(job["run_dir"]).resolve(), f"{label}.run_dir")
    command = result.get("command")
    expected_flags = {
        "--mode": "train",
        "--model-type": "mshnet",
        "--dataset-dir": job["dataset_dir"],
        "--train-split-file": job["train_split_arg"],
        "--test-split-file": job["test_split_arg"],
        "--val-fraction": "0",
        "--seed": str(job["seed"]),
        "--deterministic": "true",
        "--epochs": str(EXPECTED_EPOCHS),
        "--test-interval": str(TEST_INTERVAL),
        "--batch-size": str(manifest_args["batch_size"]),
        "--num-workers": str(manifest_args["num_workers"]),
        "--lr": str(manifest_args["lr"]),
        "--warm-epoch": str(manifest_args["warm_epoch"]),
        "--threshold": str(SELECTION_THRESHOLD),
        "--pd-fa-min-pd": str(PD_FA_MIN_PD),
        "--pd-fa-min-iou": str(PD_FA_MIN_IOU),
        "--paired-baseline-iou": str(PAIRED_BASELINE_IOU),
        "--selection-tie-break": TIE_BREAK,
        "--run-dir": job["run_dir"],
        "--run-label": job["job_id"],
    }
    for flag, expected in expected_flags.items():
        require_equal(command_value(command, flag, label), expected, f"{label}.command {flag}")
    if isinstance(command, list) and (
        "--val-split-file" in command or "--init-from-baseline" in command
    ):
        raise FinalizationError(f"{label}.command contains a forbidden split/init flag")


def _metric_matches_checkpoint(checkpoint_value: float, logged_value: float) -> bool:
    return f"{checkpoint_value:.4f}" == f"{logged_value:.4f}"


def _validate_selection_checkpoint(
    *,
    selector: str,
    selection: dict[str, Any],
    checkpoint_path: Path,
    expected_row: dict[str, float | int],
    job: dict[str, Any],
    dataset_meta: dict[str, Any],
    manifest_args: dict[str, Any],
    stable_method_meta: dict[str, Any],
    checkpoint_loader: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    label = f"{selector} selection for {job['job_id']}"
    require_equal(selection.get("status"), "found", f"{label}.status")
    expected_filename = (
        "checkpoint_best_iou.pkl"
        if selector == "best_iou"
        else "checkpoint_pd_fa_best.pkl"
    )
    require_equal(selection.get("file"), expected_filename, f"{label}.file")
    declared_hash = require_sha256(selection.get("sha256"), f"{label}.sha256")
    require_equal(sha256_file(checkpoint_path), declared_hash, f"{label} checkpoint hash")
    checkpoint = checkpoint_loader(checkpoint_path)
    checkpoint = require_mapping(checkpoint, f"checkpoint {checkpoint_path}")
    checkpoint_meta = _validate_method_metadata(
        checkpoint.get("method_meta"), job, dataset_meta, manifest_args, f"{label}.method_meta"
    )
    require_equal(
        _stable_metadata_view(checkpoint_meta),
        stable_method_meta,
        f"{label}.method_meta cross-artifact identity",
    )
    epoch = require_int(checkpoint.get("epoch"), f"{label}.checkpoint epoch")
    require_equal(epoch, expected_row["epoch_zero_based"], f"{label}.checkpoint epoch")
    require_equal(selection.get("epoch_zero_based"), epoch, f"{label}.summary epoch")
    metrics: dict[str, float] = {}
    for metric in ("iou", "pd", "fa"):
        value = require_number(checkpoint.get(metric), f"{label}.checkpoint {metric}")
        summary_value = require_number(selection.get(metric), f"{label}.summary {metric}")
        require_equal(summary_value, value, f"{label}.{metric} summary/checkpoint")
        if not _metric_matches_checkpoint(value, float(expected_row[metric])):
            raise FinalizationError(
                f"{label}.{metric} disagrees with epoch_metric.log at persisted precision"
            )
        metrics[metric] = value
    if not 0.0 <= metrics["iou"] <= 1.0 or not 0.0 <= metrics["pd"] <= 1.0 or metrics["fa"] < 0.0:
        raise FinalizationError(f"{label} has out-of-range metrics")
    if selector == "best_iou":
        require_equal(
            require_number(checkpoint.get("best_iou"), f"{label}.best_iou"),
            metrics["iou"],
            f"{label}.best_iou",
        )
    else:
        expected_state = {
            "best_pd_fa": metrics["fa"],
            "best_pd_fa_iou": metrics["iou"],
            "best_pd_fa_pd": metrics["pd"],
            "best_pd_fa_epoch": epoch,
        }
        for key, value in expected_state.items():
            actual = checkpoint.get(key)
            actual = require_int(actual, f"{label}.{key}") if key.endswith("epoch") else require_number(actual, f"{label}.{key}")
            require_equal(actual, value, f"{label}.{key}")
    return {
        "status": "found",
        "seed": job["seed"],
        "epoch_zero_based": epoch,
        "completed_epoch": epoch + 1,
        **metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": declared_hash,
    }


def _validate_protocol_summary(
    run_dir: Path,
    job: dict[str, Any],
    dataset_meta: dict[str, Any],
    manifest_args: dict[str, Any],
    rows: list[dict[str, float | int]],
    run_config_method_meta: dict[str, Any],
    checkpoint_loader: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    summary = require_mapping(
        read_json(run_dir / "protocol_summary.json", "protocol_summary.json"),
        "protocol_summary",
    )
    label = f"protocol summary {job['job_id']}"
    exact = {
        "protocol": PROTOCOL,
        "status": "complete",
        "dataset": job["dataset"],
        "run_dir": str(run_dir),
        "selection_split": "test",
        "no_internal_holdout": True,
        "test_interval": TEST_INTERVAL,
        "evaluation_epoch_rule": EVALUATION_EPOCH_RULE,
        "total_epochs": EXPECTED_EPOCHS,
        "executed_evaluation_epochs_zero_based": list(EVALUATION_ZERO_BASED_EPOCHS),
        "last_completed_epoch_zero_based": EXPECTED_EPOCHS - 1,
    }
    for key, value in exact.items():
        require_equal(summary.get(key), value, f"{label}.{key}")
    start_epoch = require_int(summary.get("start_epoch"), f"{label}.start_epoch")
    if start_epoch < 0 or start_epoch > EXPECTED_EPOCHS:
        raise FinalizationError(f"{label}.start_epoch outside [0,{EXPECTED_EPOCHS}]")
    expected_current = [epoch for epoch in EVALUATION_ZERO_BASED_EPOCHS if epoch >= start_epoch]
    require_equal(summary.get("planned_evaluation_epochs_zero_based"), expected_current, f"{label}.planned epochs")
    require_equal(summary.get("current_process_evaluation_epochs_zero_based"), expected_current, f"{label}.current epochs")
    summary_meta = _validate_method_metadata(
        summary.get("method_meta"), job, dataset_meta, manifest_args, f"{label}.method_meta"
    )
    stable_method_meta = _stable_metadata_view(run_config_method_meta)
    require_equal(
        _stable_metadata_view(summary_meta),
        stable_method_meta,
        f"{label}.method_meta cross-artifact identity",
    )

    selections = require_mapping(summary.get("checkpoint_selection"), f"{label}.checkpoint_selection")
    best_row = max(rows, key=lambda row: float(row["iou"]))
    best = _validate_selection_checkpoint(
        selector="best_iou",
        selection=require_mapping(selections.get("best_iou"), f"{label}.best_iou"),
        checkpoint_path=run_dir / "checkpoint_best_iou.pkl",
        expected_row=best_row,
        job=job,
        dataset_meta=dataset_meta,
        manifest_args=manifest_args,
        stable_method_meta=stable_method_meta,
        checkpoint_loader=checkpoint_loader,
    )

    eligible = [
        row
        for row in rows
        if float(row["pd"]) >= PD_FA_MIN_PD
        and float(row["iou"]) >= max(PD_FA_MIN_IOU, PAIRED_BASELINE_IOU)
    ]
    constrained_summary = require_mapping(
        selections.get("constrained_min_fa"), f"{label}.constrained_min_fa"
    )
    constrained_path = run_dir / "checkpoint_pd_fa_best.pkl"
    if not eligible:
        require_equal(constrained_summary.get("status"), "not_found", f"{label}.constrained status")
        require_equal(constrained_summary.get("file"), None, f"{label}.constrained file")
        require_equal(constrained_summary.get("reason"), "no_eligible_epoch", f"{label}.constrained reason")
        if constrained_path.exists() or constrained_path.is_symlink():
            raise FinalizationError(
                f"stale constrained checkpoint exists despite no eligible epoch: {constrained_path}"
            )
        constrained = {
            "status": "not_found",
            "seed": job["seed"],
            "reason": "no_eligible_epoch",
            "epoch_zero_based": None,
            "completed_epoch": None,
            "iou": None,
            "pd": None,
            "fa": None,
            "checkpoint": None,
            "checkpoint_sha256": None,
        }
    else:
        constrained_row = min(eligible, key=lambda row: float(row["fa"]))
        constrained = _validate_selection_checkpoint(
            selector="constrained_min_fa",
            selection=constrained_summary,
            checkpoint_path=constrained_path,
            expected_row=constrained_row,
            job=job,
            dataset_meta=dataset_meta,
            manifest_args=manifest_args,
            stable_method_meta=stable_method_meta,
            checkpoint_loader=checkpoint_loader,
        )
        if constrained["pd"] < PD_FA_MIN_PD or constrained["iou"] < PD_FA_MIN_IOU:
            raise FinalizationError(f"{label} constrained checkpoint is ineligible")
    return {"best_iou": best, "constrained_min_fa": constrained}


def _stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) != EXPECTED_SEED_COUNT:
        raise FinalizationError("sample statistics require exactly three seed runs")
    means: dict[str, float] = {}
    sample_sd: dict[str, float] = {}
    for metric in ("completed_epoch", "iou", "pd", "fa"):
        values = [float(run[metric]) for run in runs]
        means[metric] = statistics.mean(values)
        sample_sd[metric] = statistics.stdev(values)
    return {"n": len(runs), "mean": means, "sample_sd": sample_sd}


def _best_seed(runs: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    # Python min/max retain the first item on ties; runs are in manifest seed order.
    chosen = (
        max(runs, key=lambda run: float(run["iou"]))
        if selector == "best_iou"
        else min(runs, key=lambda run: float(run["fa"]))
    )
    return dict(chosen)


def _validate_manifest(
    manifest: dict[str, Any], batch_dir: Path
) -> tuple[list[int], dict[str, Any], list[dict[str, Any]]]:
    require_equal(manifest.get("protocol"), PROTOCOL, "manifest.protocol")
    require_equal(manifest.get("batch_id"), batch_dir.name, "manifest.batch_id")
    require_equal(manifest.get("stage"), EXPECTED_STAGE, "manifest.stage")
    require_equal(manifest.get("model"), "MSHNet baseline", "manifest.model")
    args = require_mapping(manifest.get("args"), "manifest.args")
    exact_args = {
        "batch_id": batch_dir.name,
        "epochs": EXPECTED_EPOCHS,
        "test_interval": TEST_INTERVAL,
        "threshold": SELECTION_THRESHOLD,
        "pd_fa_min_pd": PD_FA_MIN_PD,
        "pd_fa_min_iou": PD_FA_MIN_IOU,
        "paired_baseline_iou": PAIRED_BASELINE_IOU,
        "val_fraction": 0.0,
        "deterministic": "true",
    }
    for key, value in exact_args.items():
        require_equal(args.get(key), value, f"manifest.args.{key}")
    batch_size = require_int(args.get("batch_size"), "manifest.args.batch_size")
    num_workers = require_int(args.get("num_workers"), "manifest.args.num_workers")
    warm_epoch = require_int(args.get("warm_epoch"), "manifest.args.warm_epoch")
    learning_rate = require_number(args.get("lr"), "manifest.args.lr")
    if batch_size < 1 or num_workers < 0 or warm_epoch < 0 or learning_rate <= 0.0:
        raise FinalizationError("manifest contains invalid training hyperparameters")
    datasets = parse_csv(args.get("datasets"), str, "manifest.args.datasets")
    if tuple(datasets) != DATASET_NAMES:
        raise FinalizationError(f"manifest dataset order must be {DATASET_NAMES}")
    seeds = parse_csv(args.get("seeds"), int, "manifest.args.seeds")
    if len(seeds) != EXPECTED_SEED_COUNT:
        raise FinalizationError("manifest must contain exactly three unique seeds")
    selection = require_mapping(manifest.get("selection_policy"), "selection_policy")
    expected_selection = {
        "test_selected": True,
        "test_is_untouched": False,
        "test_interval_completed_epochs": TEST_INTERVAL,
        "test_evaluation_epochs": list(EVALUATION_COMPLETED_EPOCHS),
        "eligible_checkpoint_epochs": list(EVALUATION_COMPLETED_EPOCHS),
        "metric_threshold": SELECTION_THRESHOLD,
        "pd_fa_min_pd": PD_FA_MIN_PD,
        "pd_fa_min_iou": PD_FA_MIN_IOU,
        "paired_baseline_iou": PAIRED_BASELINE_IOU,
        "strict_tie_break": TIE_BREAK,
    }
    for key, value in expected_selection.items():
        require_equal(selection.get(key), value, f"selection_policy.{key}")

    immutable = require_mapping(manifest.get("immutable_contract"), "immutable_contract")
    declared_contract_hash = require_sha256(
        manifest.get("immutable_contract_sha256"), "immutable_contract_sha256"
    )
    require_equal(canonical_json_sha256(immutable), declared_contract_hash, "immutable contract hash")
    require_equal(immutable.get("protocol"), PROTOCOL, "immutable_contract.protocol")
    require_equal(immutable.get("model"), "MSHNet baseline", "immutable_contract.model")
    require_equal(immutable.get("datasets"), manifest.get("datasets"), "immutable datasets")
    expected_immutable_args = {key: value for key, value in args.items() if key not in {"resume", "dry_run"}}
    require_equal(
        immutable.get("args_excluding_resume_and_dry_run"),
        expected_immutable_args,
        "immutable args",
    )
    require_equal(
        immutable.get("checkpoint_selector"),
        {
            "threshold": SELECTION_THRESHOLD,
            "pd_fa_min_pd": PD_FA_MIN_PD,
            "pd_fa_min_iou": PD_FA_MIN_IOU,
            "paired_baseline_iou": PAIRED_BASELINE_IOU,
            "tie_break": TIE_BREAK,
        },
        "immutable checkpoint selector",
    )

    jobs = require_list(manifest.get("jobs"), "manifest.jobs")
    if len(jobs) != len(DATASET_NAMES) * EXPECTED_SEED_COUNT:
        raise FinalizationError("manifest must contain exactly nine jobs")
    expected_pairs = [(seed, dataset) for seed in seeds for dataset in DATASET_NAMES]
    actual_pairs: list[tuple[int, str]] = []
    job_ids: set[str] = set()
    run_dirs: set[Path] = set()
    for index, raw_job in enumerate(jobs):
        job = require_mapping(raw_job, f"manifest.jobs[{index}]")
        dataset = job.get("dataset")
        seed = require_int(job.get("seed"), f"job[{index}].seed")
        actual_pairs.append((seed, dataset))
        expected_id = f"mshnet__{str(dataset).lower()}__seed_{seed}"
        require_equal(job.get("job_id"), expected_id, f"job[{index}].job_id")
        if expected_id in job_ids:
            raise FinalizationError(f"duplicate job_id {expected_id}")
        job_ids.add(expected_id)
        run_dir = normalized_path(job.get("run_dir"), f"job {expected_id}.run_dir")
        if run_dir in run_dirs:
            raise FinalizationError(f"duplicate run_dir {run_dir}")
        run_dirs.add(run_dir)
        require_equal(job.get("protocol"), PROTOCOL, f"job {expected_id}.protocol")
        require_equal(job.get("total_epochs"), EXPECTED_EPOCHS, f"job {expected_id}.total_epochs")
        require_equal(job.get("test_interval"), TEST_INTERVAL, f"job {expected_id}.test_interval")
        require_equal(job.get("test_evaluation_epochs"), list(EVALUATION_COMPLETED_EPOCHS), f"job {expected_id}.evaluation epochs")
        require_equal(job.get("deterministic"), True, f"job {expected_id}.deterministic")
    require_equal(actual_pairs, expected_pairs, "3x3 dataset/seed job grid and order")

    immutable_jobs = require_list(immutable.get("jobs"), "immutable_contract.jobs")
    expected_immutable_jobs = [
        {
            key: job[key]
            for key in (
                "job_id", "dataset", "seed", "dataset_dir", "train_split_arg",
                "test_split_arg", "train_split_sha256", "test_split_sha256",
                "train_split_raw_sha256", "test_split_raw_sha256", "deterministic",
                "run_dir", "total_epochs", "test_interval", "test_evaluation_epochs",
            )
        }
        for job in jobs
    ]
    require_equal(immutable_jobs, expected_immutable_jobs, "immutable job grid")
    return seeds, args, jobs


def _validate_job(
    *,
    job: dict[str, Any],
    batch_dir: Path,
    manifest_args: dict[str, Any],
    manifest_dataset: dict[str, Any],
    dataset_audit: dict[str, Any],
    checkpoint_loader: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    job_id = job["job_id"]
    expected_result = (batch_dir / "jobs" / f"{job_id}.json").resolve()
    require_equal(
        normalized_path(job.get("result_file"), f"manifest {job_id}.result_file"),
        expected_result,
        f"manifest {job_id}.result_file",
    )
    result = require_mapping(read_json(expected_result, f"job result {job_id}"), f"job result {job_id}")
    _validate_result(result, job, manifest_args)
    run_dir = normalized_path(job["run_dir"], f"{job_id}.run_dir")
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise FinalizationError(f"missing or non-plain run directory: {run_dir}")
    _validate_persisted_splits(run_dir, dataset_audit)
    run_config = _validate_run_config(
        run_dir, job, manifest_dataset, manifest_args
    )
    run_config_method_meta = require_mapping(
        run_config.get("method_meta"), "run_config.method_meta"
    )
    rows = parse_metrics(run_dir / "epoch_metric.log")
    selections = _validate_protocol_summary(
        run_dir,
        job,
        manifest_dataset,
        manifest_args,
        rows,
        run_config_method_meta,
        checkpoint_loader,
    )
    return {
        "job_id": job_id,
        "dataset": job["dataset"],
        "seed": job["seed"],
        "run_dir": str(run_dir),
        **selections,
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Test-selected MSHNet baseline summary",
        "",
        "> **Warning:** the canonical test split was evaluated every 10 completed epochs and used to select both checkpoint types. These are test-selected results, not performance on an untouched test set.",
        "",
        f"Overall validation status: `{summary['status']}`  ",
        f"Best-IoU selector: `{summary['selector_readiness']['best_iou']}`  ",
        f"Constrained PD/FA selector: `{summary['selector_readiness']['constrained_min_fa']}`",
        "",
        "Protocol: complete canonical `img_idx/train_<dataset>.txt`; no validation set; canonical `img_idx/test_<dataset>.txt` evaluated at completed epochs 10, 20, ..., 400.",
        "",
        "## Best test-IoU checkpoint",
        "",
        "| Dataset | Seed | Completed epoch | IoU | Pd | FA/Mpix |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_NAMES:
        for run in summary["datasets"][dataset]["best_iou"]["per_seed"]:
            lines.append(
                f"| {dataset} | {run['seed']} | {run['completed_epoch']} | "
                f"{run['iou']:.6f} | {run['pd']:.6f} | {run['fa']:.6f} |"
            )
    lines.extend([
        "",
        "| Dataset | IoU mean ± sample SD | Pd mean ± sample SD | FA mean ± sample SD | Best seed |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset in DATASET_NAMES:
        block = summary["datasets"][dataset]["best_iou"]
        mean, sd = block["aggregate"]["mean"], block["aggregate"]["sample_sd"]
        lines.append(
            f"| {dataset} | {mean['iou']:.6f} ± {sd['iou']:.6f} | "
            f"{mean['pd']:.6f} ± {sd['pd']:.6f} | {mean['fa']:.6f} ± {sd['fa']:.6f} | "
            f"{block['best_seed']['seed']} |"
        )
    lines.extend([
        "",
        "## Constrained minimum-test-FA checkpoint",
        "",
        f"Eligibility: Pd ≥ {PD_FA_MIN_PD:.2f} and IoU ≥ {PD_FA_MIN_IOU:.3f}; minimum FA, earliest epoch on an exact tie.",
        "",
        "| Dataset | Seed | Status | Completed epoch | IoU | Pd | FA/Mpix |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for dataset in DATASET_NAMES:
        for run in summary["datasets"][dataset]["constrained_min_fa"]["per_seed"]:
            if run["status"] == "found":
                values = (
                    str(run["completed_epoch"]), f"{run['iou']:.6f}",
                    f"{run['pd']:.6f}", f"{run['fa']:.6f}",
                )
            else:
                values = ("NA", "NA", "NA", "NA")
            lines.append(
                f"| {dataset} | {run['seed']} | {run['status']} | {values[0]} | "
                f"{values[1]} | {values[2]} | {values[3]} |"
            )
    lines.extend([
        "",
        "| Dataset | Status | IoU mean ± sample SD | Pd mean ± sample SD | FA mean ± sample SD | Best seed |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for dataset in DATASET_NAMES:
        block = summary["datasets"][dataset]["constrained_min_fa"]
        if block["status"] == "ready":
            mean, sd = block["aggregate"]["mean"], block["aggregate"]["sample_sd"]
            aggregate_values = (
                f"{mean['iou']:.6f} ± {sd['iou']:.6f}",
                f"{mean['pd']:.6f} ± {sd['pd']:.6f}",
                f"{mean['fa']:.6f} ± {sd['fa']:.6f}",
                str(block["best_seed"]["seed"]),
            )
        else:
            aggregate_values = ("NA", "NA", "NA", "NA")
        lines.append(
            f"| {dataset} | {block['status']} | {aggregate_values[0]} | "
            f"{aggregate_values[1]} | {aggregate_values[2]} | {aggregate_values[3]} |"
        )
    lines.extend([
        "",
        "A dataset-level constrained-PD/FA mean ± sample SD and best seed are reported only when all three seeds have eligible checkpoints. Missing selectors remain `NA`; available seeds are never pooled into a nominal three-seed result.",
        "",
    ])
    return "\n".join(lines)


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def finalize_batch(
    batch_dir: Path,
    *,
    force: bool = False,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = load_checkpoint_cpu,
    expected_datasets: Mapping[str, Mapping[str, Mapping[str, Any]]] = CANONICAL_DATASETS,
    expected_dataset_root: Path = PROJECT_DIR / "datasets",
    expected_source_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()
    if batch_dir.is_symlink() or not batch_dir.is_dir():
        raise FinalizationError(f"missing or non-plain batch directory: {batch_dir}")
    output_json = batch_dir / OUTPUT_JSON
    output_markdown = batch_dir / OUTPUT_MARKDOWN
    if not force and (output_json.exists() or output_markdown.exists()):
        raise FileExistsError(
            "refusing to overwrite an existing finalizer output; pass force=True"
        )

    manifest = require_mapping(read_json(batch_dir / "manifest.json", "manifest"), "manifest")
    seeds, manifest_args, jobs = _validate_manifest(manifest, batch_dir)
    if expected_source_paths is None:
        expected_source_paths = {
            "scheduler": PROJECT_DIR / "tools" / "run_test_selected_baselines.py",
            "training_entry": PROJECT_DIR / "tools" / "train_test_selected_full_train.py",
        }
    source_hashes = _validate_sources(manifest, expected_source_paths)
    split_audits = _validate_dataset_splits(
        manifest, expected_datasets, expected_dataset_root
    )
    manifest_datasets = require_mapping(manifest["datasets"], "manifest.datasets")

    validated_jobs: list[dict[str, Any]] = []
    for job in jobs:
        dataset = job["dataset"]
        validated_jobs.append(
            _validate_job(
                job=job,
                batch_dir=batch_dir,
                manifest_args=manifest_args,
                manifest_dataset=require_mapping(manifest_datasets[dataset], f"manifest.datasets.{dataset}"),
                dataset_audit=split_audits[dataset],
                checkpoint_loader=checkpoint_loader,
            )
        )

    dataset_results: dict[str, Any] = {}
    missing_constrained: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        dataset_jobs = [run for run in validated_jobs if run["dataset"] == dataset]
        require_equal([run["seed"] for run in dataset_jobs], seeds, f"validated seed order for {dataset}")
        best_runs = [run["best_iou"] for run in dataset_jobs]
        constrained_runs = [run["constrained_min_fa"] for run in dataset_jobs]
        found_constrained = [run for run in constrained_runs if run["status"] == "found"]
        missing_constrained.extend(
            {"dataset": dataset, "seed": run["seed"], "reason": run["reason"]}
            for run in constrained_runs
            if run["status"] != "found"
        )
        constrained_ready = len(found_constrained) == EXPECTED_SEED_COUNT
        dataset_results[dataset] = {
            "best_iou": {
                "status": "ready",
                "per_seed": best_runs,
                "aggregate": _stats(best_runs),
                "best_seed_rule": "maximum selected test IoU; manifest seed order breaks exact ties",
                "best_seed": _best_seed(best_runs, "best_iou"),
            },
            "constrained_min_fa": {
                "status": "ready" if constrained_ready else "not_ready",
                "n_found": len(found_constrained),
                "n_required": EXPECTED_SEED_COUNT,
                "per_seed": constrained_runs,
                "aggregate": _stats(found_constrained) if constrained_ready else None,
                "best_seed_rule": (
                    "minimum selected test FA; manifest seed order breaks exact ties"
                    if constrained_ready else None
                ),
                "best_seed": (
                    _best_seed(found_constrained, "constrained_min_fa")
                    if constrained_ready else None
                ),
            },
        }

    constrained_ready = not missing_constrained
    status = (
        "complete_and_validated"
        if constrained_ready
        else "best_iou_complete__constrained_min_fa_not_ready"
    )
    split_hash_summary = {
        dataset: {
            role: {
                "path": split_audits[dataset][role]["path"],
                "count": split_audits[dataset][role]["count"],
                "raw_sha256": split_audits[dataset][role]["raw_sha256"],
                "ordered_names_sha256": split_audits[dataset][role][
                    "ordered_names_sha256"
                ],
            }
            for role in ("train", "test")
        }
        for dataset in DATASET_NAMES
    }
    summary = {
        "status": status,
        "protocol": PROTOCOL,
        "batch_id": batch_dir.name,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reporting_warning": (
            "The canonical test split was evaluated every 10 completed epochs "
            "and used for checkpoint selection. These are test-selected results, "
            "not performance on an untouched test set."
        ),
        "selector_readiness": {
            "best_iou": "ready",
            "constrained_min_fa": "ready" if constrained_ready else "not_ready",
        },
        "missing_constrained_min_fa": missing_constrained,
        "grid": {
            "datasets": list(DATASET_NAMES),
            "seeds": seeds,
            "job_count": len(validated_jobs),
            "epochs": EXPECTED_EPOCHS,
            "test_interval": TEST_INTERVAL,
            "test_evaluation_completed_epochs": list(EVALUATION_COMPLETED_EPOCHS),
        },
        "integrity": {
            "immutable_contract_sha256": manifest["immutable_contract_sha256"],
            "source_sha256": source_hashes,
            "canonical_split_hashes_revalidated": True,
            "canonical_splits": split_hash_summary,
            "run_split_copies_byte_matched": True,
            "checkpoint_sha256_and_metadata_revalidated": True,
            "selector_recomputed_from_epoch_metric_log_at_persisted_precision": True,
        },
        "datasets": dataset_results,
    }

    json_payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    markdown_payload = _build_markdown(summary)
    # All validation and rendering completes before either destination changes.
    _atomic_write(output_json, json_payload)
    _atomic_write(output_markdown, markdown_payload)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.batch_id or Path(args.batch_id).name != args.batch_id:
        raise FinalizationError("--batch-id must be one safe path component")
    batch_dir = PROJECT_DIR / "repro_runs" / "test_selected" / args.batch_id
    summary = finalize_batch(batch_dir, force=args.force)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalizationError, FileExistsError) as exc:
        print(f"finalization failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
