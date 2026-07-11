#!/usr/bin/env python3
"""Fail-closed finalizer for the clean MSHNet mechanism-audit grid.

The finalizer is deliberately read-only with respect to source evidence.  It
validates the completed 3-dataset x 3-seed development-holdout audit artifacts
and writes a separate JSON/Markdown evidence index.  It never opens an
official-test split and it never estimates or claims a DEA improvement.
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
import sys
from typing import Any, Callable, Iterable
import zipfile

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.finalize_clean_baselines import (
    DATASET_NAMES,
    EXPECTED_EPOCHS,
    FinalizationError,
    load_checkpoint_cpu,
    normalized_path,
    parse_metrics,
    read_json,
    require_exact_int,
    require_mapping,
    require_number,
    require_sha256,
    validate_checkpoint,
    validate_manifest,
    validate_result,
)


AUDIT_SCHEMA = "dea.clean_mechanism_audit.v1"
OUTPUT_JSON = "clean_mechanism_audit_evidence_summary.json"
OUTPUT_MARKDOWN = "clean_mechanism_audit_evidence_summary.md"
BASELINE_SUMMARY_JSON = "clean_baseline_holdout_summary.json"
EXPECTED_OFFICIAL_TEST_STATUS = (
    "sealed; this exporter accepts development validation only"
)
EXPECTED_BASELINE_COMPLETION = (
    "all_3x3_jobs_400_epochs_returncode_0_and_finalizer_validated"
)
EXPECTED_CHECKPOINT_ROLE = "best_iou"
EXPECTED_SPLIT_ROLE = "val"
EXPECTED_METHOD = "mshnet"
EXPECTED_ANCHOR_MODE = "mean"
EXPECTED_ACTIVE_STAGE = 0
EXPECTED_THRESHOLD_PROBABILITY = 0.5
EXPECTED_THRESHOLD_LOGIT = 0.0
EXPECTED_CONNECTIVITY = 2
EXPECTED_MAX_DISTANCE = 3.0
EXPECTED_AUDIT_COUNT = 9
SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_BATCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

INTEGER_SUM_FIELDS = (
    "images",
    "pixels",
    "intersection_pixels",
    "union_pixels",
    "ground_truth_positive_pixels",
    "predicted_positive_pixels",
    "false_positive_pixels",
    "false_negative_pixels",
    "target_component_count",
    "true_positive_component_count",
    "false_negative_component_count",
    "prediction_component_count",
    "matched_prediction_component_count",
    "false_positive_component_count",
    "false_positive_component_area",
    "recoverable_fn_component_count",
    "recoverable_fn_target_component_area",
    "candidate_component_count",
    "conflict_pixels",
    "conflict_on_true_positive_pixels",
    "conflict_on_false_positive_pixels",
    "conflict_on_false_negative_pixels",
)
FLOAT_SUM_FIELDS = (
    "p_rms_sum",
    "j_rms_sum",
    "score_sum",
    "mean_anchor_score_sum_true_positive",
    "mean_anchor_score_sum_false_positive",
    "mean_anchor_score_sum_false_negative",
)
IMAGE_INCIDENCE_FIELDS = (
    "images_with_false_positive_pixels",
    "images_with_false_negative_pixels",
    "images_with_false_positive_components",
    "images_with_false_negative_components",
    "images_with_recoverable_fn_components",
    "images_with_conflict",
    "images_with_conflict_on_true_positive",
    "images_with_conflict_on_false_positive",
    "images_with_conflict_on_false_negative",
)
IMAGE_INTEGER_FIELDS = tuple(
    field for field in INTEGER_SUM_FIELDS if field not in {"images", "pixels"}
)
REQUIRED_NPZ_KEYS = frozenset(
    {
        "z11",
        "z10",
        "z01",
        "z00",
        "p_z",
        "j_z",
        "p_feature_rms",
        "j_feature_rms",
        "ratio",
        "conflict_score",
        "conflict_mask",
        "pred_logit",
        "pred_probability",
        "prediction_mask",
        "target_mask",
        "scale_logits",
        "target_component_labels",
        "prediction_component_labels",
        "candidate_component_labels",
        "recoverable_fn_mask",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize all nine clean MSHNet development-holdout "
            "mechanism audits without touching official test data."
        )
    )
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument(
        "--audit-root",
        default="",
        help=(
            "Mechanism-audit root. Defaults to "
            "repro_runs/clean/<batch-id>/mechanism_audits."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing final JSON/Markdown only after full validation succeeds.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FinalizationError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _same_number(actual: Any, expected: Any, label: str, *, tolerance: float = 1e-12) -> None:
    actual_number = require_number(actual, label)
    expected_number = require_number(expected, f"expected {label}")
    if not math.isclose(actual_number, expected_number, rel_tol=1e-9, abs_tol=tolerance):
        raise FinalizationError(
            f"{label} mismatch: artifact={actual_number!r}, expected={expected_number!r}"
        )


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise FinalizationError(f"{label} must be a boolean, got {value!r}")
    return value


def _safe_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FinalizationError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FinalizationError(f"{label} must be a normalized relative path: {value!r}")
    return path


def _artifact_path(audit_dir: Path, value: Any, label: str) -> Path:
    relative = _safe_relative_path(value, label)
    path = audit_dir.joinpath(relative)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FinalizationError(f"missing {label}: {path}") from exc
    if audit_dir.resolve() not in resolved.parents:
        raise FinalizationError(f"{label} escapes audit directory: {value!r}")
    if path.is_symlink() or not resolved.is_file():
        raise FinalizationError(f"{label} must be a regular non-symlink file: {path}")
    return resolved


def _directory_path(audit_dir: Path, value: Any, label: str) -> Path:
    relative = _safe_relative_path(value, label)
    path = audit_dir.joinpath(relative)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FinalizationError(f"missing {label}: {path}") from exc
    if audit_dir.resolve() not in resolved.parents:
        raise FinalizationError(f"{label} escapes audit directory: {value!r}")
    if path.is_symlink() or not resolved.is_dir():
        raise FinalizationError(f"{label} must be a regular non-symlink directory: {path}")
    return resolved


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise FinalizationError(f"cannot read {label} {path}: {exc}") from exc
    if not lines:
        raise FinalizationError(f"{label} is empty: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise FinalizationError(f"blank line in {label} {path} at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FinalizationError(
                f"invalid JSON in {label} {path} at line {line_number}: {exc}"
            ) from exc
        rows.append(require_mapping(value, f"{label} line {line_number}"))
    return rows


def _require_nonnegative_int(value: Any, label: str) -> int:
    result = require_exact_int(value, label)
    if result < 0:
        raise FinalizationError(f"{label} must be non-negative")
    return result


def _require_fraction(value: Any, label: str) -> float:
    result = require_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise FinalizationError(f"{label} must be in [0, 1], got {result!r}")
    return result


def _validate_npz_index(path: Path, label: str) -> None:
    """Validate the NPZ central directory without materializing large arrays."""
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise FinalizationError(f"duplicate member in {label}: {path}")
            keys = {
                name[:-4]
                for name in names
                if name.endswith(".npy") and "/" not in name and "\\" not in name
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise FinalizationError(f"invalid NPZ artifact {path}: {exc}") from exc
    missing = REQUIRED_NPZ_KEYS - keys
    if missing:
        raise FinalizationError(f"{label} missing NPZ arrays: {sorted(missing)}")


def _derive_metrics(totals: dict[str, int | float], eps: float) -> dict[str, float]:
    def ratio(numerator: str, denominator: str) -> float:
        return float(totals[numerator]) / max(1, int(totals[denominator]))

    pixels = max(1, int(totals["pixels"]))
    return {
        "pooled_iou": ratio("intersection_pixels", "union_pixels"),
        "pd": ratio("true_positive_component_count", "target_component_count"),
        "fa_per_million": float(totals["false_positive_component_area"]) / pixels * 1e6,
        "recoverable_fn_fraction": ratio(
            "recoverable_fn_component_count", "false_negative_component_count"
        ),
        "conflict_fraction": float(totals["conflict_pixels"]) / pixels,
        "mean_anchor_index": float(totals["score_sum"]) / pixels,
        "global_r_ratio_of_sums": float(totals["j_rms_sum"]) / (
            float(totals["p_rms_sum"]) + eps
        ),
        "conflict_true_positive_coverage": ratio(
            "conflict_on_true_positive_pixels", "intersection_pixels"
        ),
        "conflict_false_positive_coverage": ratio(
            "conflict_on_false_positive_pixels", "false_positive_pixels"
        ),
        "conflict_false_negative_coverage": ratio(
            "conflict_on_false_negative_pixels", "false_negative_pixels"
        ),
    }


def _empty_totals() -> dict[str, int | float]:
    return {
        **{field: 0 for field in INTEGER_SUM_FIELDS},
        **{field: 0.0 for field in FLOAT_SUM_FIELDS},
        **{field: 0 for field in IMAGE_INCIDENCE_FIELDS},
    }


def _sum_totals(items: Iterable[dict[str, Any]]) -> dict[str, int | float]:
    result = _empty_totals()
    for item in items:
        counts = require_mapping(item.get("counts"), "run counts")
        for field in INTEGER_SUM_FIELDS + IMAGE_INCIDENCE_FIELDS:
            result[field] = int(result[field]) + _require_nonnegative_int(
                counts.get(field), f"run counts.{field}"
            )
        for field in FLOAT_SUM_FIELDS:
            result[field] = float(result[field]) + require_number(
                counts.get(field), f"run counts.{field}"
            )
    return result


def _validate_image_rows(
    rows: list[dict[str, Any]],
    *,
    audit_dir: Path,
    arrays_dir: Path,
    artifacts: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    expected_images = _require_nonnegative_int(summary.get("images"), "summary.images")
    if len(rows) != expected_images or expected_images <= 0:
        raise FinalizationError(
            f"images.jsonl row count {len(rows)} does not match positive summary.images "
            f"{expected_images}"
        )
    image_ids: set[str] = set()
    array_paths: set[Path] = set()
    inventory: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    summed = {field: 0 for field in IMAGE_INTEGER_FIELDS}
    incidence = {field: 0 for field in IMAGE_INCIDENCE_FIELDS}
    expected_indices = list(range(expected_images))
    actual_indices: list[int] = []
    array_dir_relative = arrays_dir.relative_to(audit_dir.resolve())

    for row_number, row in enumerate(rows, start=1):
        label = f"images.jsonl line {row_number}"
        index = _require_nonnegative_int(row.get("image_index"), f"{label}.image_index")
        actual_indices.append(index)
        image_id = row.get("image_id")
        if (
            not isinstance(image_id, str)
            or SAFE_IMAGE_ID.fullmatch(image_id) is None
            or image_id in {".", ".."}
            or image_id in image_ids
        ):
            raise FinalizationError(f"unsafe or duplicate {label}.image_id: {image_id!r}")
        image_ids.add(image_id)
        by_id[image_id] = row

        for field in IMAGE_INTEGER_FIELDS:
            summed[field] += _require_nonnegative_int(row.get(field), f"{label}.{field}")
        _require_fraction(row.get("iou"), f"{label}.iou")
        _require_fraction(row.get("conflict_fraction"), f"{label}.conflict_fraction")
        require_number(row.get("mean_anchor_index"), f"{label}.mean_anchor_index")
        require_number(row.get("interaction_ratio_mean"), f"{label}.interaction_ratio_mean")
        require_number(row.get("interaction_ratio_p95"), f"{label}.interaction_ratio_p95")

        predicates = {
            "images_with_false_positive_pixels": row["false_positive_pixels"] > 0,
            "images_with_false_negative_pixels": row["false_negative_pixels"] > 0,
            "images_with_false_positive_components": row["false_positive_component_count"] > 0,
            "images_with_false_negative_components": row["false_negative_component_count"] > 0,
            "images_with_recoverable_fn_components": row["recoverable_fn_component_count"] > 0,
            "images_with_conflict": row["conflict_pixels"] > 0,
            "images_with_conflict_on_true_positive": row["conflict_on_true_positive_pixels"] > 0,
            "images_with_conflict_on_false_positive": row["conflict_on_false_positive_pixels"] > 0,
            "images_with_conflict_on_false_negative": row["conflict_on_false_negative_pixels"] > 0,
        }
        for field, present in predicates.items():
            incidence[field] += int(present)

        relative_array = _safe_relative_path(row.get("array_path"), f"{label}.array_path")
        expected_relative = array_dir_relative / f"{image_id}.npz"
        if relative_array != expected_relative:
            raise FinalizationError(
                f"{label}.array_path must be {expected_relative}, got {relative_array}"
            )
        array_path = _artifact_path(audit_dir, str(relative_array), f"{label}.array_path")
        if array_path in array_paths:
            raise FinalizationError(f"duplicate array artifact path: {array_path}")
        array_paths.add(array_path)
        expected_bytes = _require_nonnegative_int(row.get("array_bytes"), f"{label}.array_bytes")
        if expected_bytes <= 0 or array_path.stat().st_size != expected_bytes:
            raise FinalizationError(f"array byte-size mismatch: {array_path}")
        expected_hash = require_sha256(row.get("array_sha256"), f"{label}.array_sha256")
        if sha256(array_path) != expected_hash:
            raise FinalizationError(f"array SHA-256 mismatch: {array_path}")
        _validate_npz_index(array_path, f"array for {image_id}")
        inventory.append(
            {
                "image_id": image_id,
                "path": str(relative_array),
                "sha256": expected_hash,
                "bytes": expected_bytes,
            }
        )

    if actual_indices != expected_indices:
        raise FinalizationError("images.jsonl image_index values must be exactly 0..N-1 in order")
    declared_count = _require_nonnegative_int(artifacts.get("array_count"), "artifacts.array_count")
    if declared_count != expected_images:
        raise FinalizationError("artifacts.array_count does not match images.jsonl")
    actual_files = {
        path.resolve()
        for path in arrays_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if any(path.is_dir() or path.is_symlink() for path in arrays_dir.iterdir()):
        raise FinalizationError("arrays directory must contain only regular array files")
    if actual_files != array_paths:
        raise FinalizationError("arrays directory does not exactly match images.jsonl inventory")
    inventory_bytes = json.dumps(
        inventory, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    inventory_hash = hashlib.sha256(inventory_bytes).hexdigest()
    if inventory_hash != require_sha256(
        artifacts.get("array_inventory_sha256"), "artifacts.array_inventory_sha256"
    ):
        raise FinalizationError("array inventory SHA-256 mismatch")
    total_bytes = sum(item["bytes"] for item in inventory)
    if total_bytes != _require_nonnegative_int(
        artifacts.get("array_total_bytes"), "artifacts.array_total_bytes"
    ):
        raise FinalizationError("artifacts.array_total_bytes mismatch")
    return {**summed, **incidence}, inventory, by_id


def _validate_component_rows(
    rows: list[dict[str, Any]], image_rows: dict[str, dict[str, Any]]
) -> None:
    counts: dict[str, dict[str, int]] = {
        image_id: {
            "target_component_count": 0,
            "true_positive_component_count": 0,
            "false_negative_component_count": 0,
            "prediction_component_count": 0,
            "matched_prediction_component_count": 0,
            "false_positive_component_count": 0,
            "false_positive_component_area": 0,
            "recoverable_fn_component_count": 0,
            "recoverable_fn_target_component_area": 0,
            "candidate_component_count": 0,
        }
        for image_id in image_rows
    }
    seen: set[tuple[str, str, int]] = set()
    valid_roles = {
        "target": {"tp_target", "fn_target"},
        "prediction": {"matched_pred", "fp_pred"},
        "candidate": {"candidate"},
    }
    for row_number, row in enumerate(rows, start=1):
        label = f"components.jsonl line {row_number}"
        image_id = row.get("image_id")
        if image_id not in image_rows:
            raise FinalizationError(f"{label}.image_id absent from images.jsonl")
        domain = row.get("domain")
        role = row.get("role")
        if domain not in valid_roles or role not in valid_roles[domain]:
            raise FinalizationError(f"invalid {label} domain/role: {(domain, role)!r}")
        component_id = _require_nonnegative_int(row.get("component_id"), f"{label}.component_id")
        component_index = _require_nonnegative_int(
            row.get("component_index"), f"{label}.component_index"
        )
        if component_id < 1:
            raise FinalizationError(f"{label}.component_id must be one-based")
        key = (str(image_id), str(domain), component_id)
        if key in seen:
            raise FinalizationError(f"duplicate component identity: {key!r}")
        seen.add(key)
        area = _require_nonnegative_int(row.get("area"), f"{label}.area")
        if area <= 0:
            raise FinalizationError(f"{label}.area must be positive")
        conflict_pixels = _require_nonnegative_int(
            row.get("conflict_pixels"), f"{label}.conflict_pixels"
        )
        if conflict_pixels > area:
            raise FinalizationError(f"{label}.conflict_pixels exceeds area")
        conflict_fraction = _require_fraction(
            row.get("conflict_fraction"), f"{label}.conflict_fraction"
        )
        _same_number(conflict_fraction, conflict_pixels / area, f"{label}.conflict_fraction")
        for field in (
            "p_z_mean",
            "j_z_mean",
            "interaction_ratio_mean",
            "interaction_ratio_p95",
            "mean_anchor_score_mean",
            "prediction_logit_mean",
        ):
            require_number(row.get(field), f"{label}.{field}")

        current = counts[str(image_id)]
        if domain == "target":
            current["target_component_count"] += 1
            if role == "tp_target":
                current["true_positive_component_count"] += 1
            else:
                current["false_negative_component_count"] += 1
                recoverable = _require_bool(row.get("recoverable"), f"{label}.recoverable")
                if recoverable:
                    current["recoverable_fn_component_count"] += 1
                    current["recoverable_fn_target_component_area"] += area
        elif domain == "prediction":
            current["prediction_component_count"] += 1
            if role == "matched_pred":
                current["matched_prediction_component_count"] += 1
            else:
                current["false_positive_component_count"] += 1
                current["false_positive_component_area"] += area
        else:
            current["candidate_component_count"] += 1
        # The index is retained as a zero-based exporter identity, but it need
        # not equal component_id-1 after arbitrary connected-component labels.
        _ = component_index

    for image_id, expected in image_rows.items():
        for field, actual in counts[image_id].items():
            declared = _require_nonnegative_int(
                expected.get(field), f"image {image_id}.{field}"
            )
            if actual != declared:
                raise FinalizationError(
                    f"component rows disagree with image {image_id}.{field}: "
                    f"components={actual}, image={declared}"
                )


def _validate_summary(
    manifest_summary: dict[str, Any],
    image_counts: dict[str, int],
    *,
    eps: float,
) -> tuple[dict[str, int | float], dict[str, float]]:
    totals = _empty_totals()
    for field in INTEGER_SUM_FIELDS:
        value = _require_nonnegative_int(manifest_summary.get(field), f"summary.{field}")
        totals[field] = value
        if field == "images":
            continue
        if field == "pixels":
            continue
        if image_counts[field] != value:
            raise FinalizationError(
                f"images.jsonl sum disagrees with summary.{field}: "
                f"images={image_counts[field]}, summary={value}"
            )
    for field in FLOAT_SUM_FIELDS:
        totals[field] = require_number(manifest_summary.get(field), f"summary.{field}")
    for field in IMAGE_INCIDENCE_FIELDS:
        totals[field] = image_counts[field]

    if int(totals["pixels"]) <= 0:
        raise FinalizationError("summary.pixels must be positive")
    if int(totals["intersection_pixels"]) > int(totals["union_pixels"]):
        raise FinalizationError("summary intersection exceeds union")
    if int(totals["true_positive_component_count"]) + int(
        totals["false_negative_component_count"]
    ) != int(totals["target_component_count"]):
        raise FinalizationError("summary target component partition is inconsistent")
    if int(totals["matched_prediction_component_count"]) + int(
        totals["false_positive_component_count"]
    ) != int(totals["prediction_component_count"]):
        raise FinalizationError("summary prediction component partition is inconsistent")
    metrics = _derive_metrics(totals, eps)
    for field, expected in metrics.items():
        _same_number(manifest_summary.get(field), expected, f"summary.{field}")
    return totals, metrics


def _validate_baseline(
    batch_dir: Path,
    *,
    checkpoint_loader: Callable[[Path], dict[str, Any]],
) -> tuple[
    list[int],
    dict[str, Any],
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
]:
    """Revalidate the frozen baseline grid and return trusted pair records."""
    batch_manifest = require_mapping(
        read_json(batch_dir / "manifest.json", "clean baseline manifest"),
        "clean baseline manifest",
    )
    seeds, datasets_meta = validate_manifest(batch_manifest, batch_dir)
    baseline_summary_path = batch_dir / BASELINE_SUMMARY_JSON
    baseline_summary = require_mapping(
        read_json(baseline_summary_path, "clean baseline summary"),
        "clean baseline summary",
    )
    expected_header = {
        "batch_id": batch_dir.name,
        "status": "complete_and_validated",
        "method": "mshnet",
        "model_type": "mshnet",
        "official_test_status": "untouched; not evaluated by this finalizer",
        "not_for_official_test_or_main_table_claims": True,
        "epochs_per_run": EXPECTED_EPOCHS,
        "seeds": seeds,
    }
    mismatches = [
        f"{key}: summary={baseline_summary.get(key)!r}, expected={value!r}"
        for key, value in expected_header.items()
        if baseline_summary.get(key) != value
    ]
    if mismatches:
        raise FinalizationError(
            "baseline summary is not a completed sealed development grid: "
            + "; ".join(mismatches)
        )
    summary_datasets = require_mapping(
        baseline_summary.get("datasets"), "baseline summary.datasets"
    )
    if set(summary_datasets) != set(DATASET_NAMES):
        raise FinalizationError(f"baseline summary datasets must be exactly {DATASET_NAMES}")

    jobs = {
        (str(job["dataset"]), int(job["seed"])): job for job in batch_manifest["jobs"]
    }
    trusted: dict[tuple[str, int], dict[str, Any]] = {}
    args = require_mapping(batch_manifest.get("args"), "baseline manifest.args")
    for dataset in DATASET_NAMES:
        dataset_meta = require_mapping(
            datasets_meta[dataset], f"baseline manifest.datasets.{dataset}"
        )
        dataset_summary = require_mapping(
            summary_datasets.get(dataset), f"baseline summary.datasets.{dataset}"
        )
        expected_split_hashes = {
            "fit": dataset_meta["fit_sha256"],
            "validation": dataset_meta["val_sha256"],
            "official_test_audit_only": dataset_meta["official_test_sha256"],
        }
        if dataset_summary.get("split_hashes") != expected_split_hashes:
            raise FinalizationError(f"baseline summary split hashes disagree for {dataset}")
        summary_runs = dataset_summary.get("runs")
        if not isinstance(summary_runs, list) or len(summary_runs) != len(seeds):
            raise FinalizationError(
                f"baseline summary {dataset} must contain exactly {len(seeds)} runs"
            )
        by_seed: dict[int, dict[str, Any]] = {}
        for index, raw_run in enumerate(summary_runs):
            run = require_mapping(raw_run, f"baseline summary {dataset}.runs[{index}]")
            seed = require_exact_int(run.get("seed"), f"baseline summary {dataset} seed")
            if seed not in seeds or seed in by_seed:
                raise FinalizationError(f"unexpected/duplicate baseline summary seed {dataset}/{seed}")
            by_seed[seed] = run
        if set(by_seed) != set(seeds):
            raise FinalizationError(f"baseline summary seed grid incomplete for {dataset}")

        for seed in seeds:
            pair = (dataset, seed)
            job = require_mapping(jobs[pair], f"baseline job {dataset}/{seed}")
            result = require_mapping(
                read_json(Path(job["result_file"]), f"baseline result {dataset}/{seed}"),
                f"baseline result {dataset}/{seed}",
            )
            validate_result(result, job)
            run_dir = normalized_path(job.get("run_dir"), f"baseline job {dataset}/{seed}.run_dir")
            rows = parse_metrics(run_dir / "epoch_metric.log")
            if len(rows) != EXPECTED_EPOCHS or [row["epoch"] for row in rows] != list(
                range(EXPECTED_EPOCHS)
            ):
                raise FinalizationError(
                    f"baseline {dataset}/{seed} must contain exactly epochs "
                    f"0..{EXPECTED_EPOCHS - 1}"
                )
            checkpoint_path = run_dir / "checkpoint_best_iou.pkl"
            checkpoint = checkpoint_loader(checkpoint_path)
            validated = validate_checkpoint(checkpoint, job, dataset_meta, args, rows)
            summary_run = by_seed[seed]
            if normalized_path(
                summary_run.get("checkpoint"),
                f"baseline summary {dataset}/{seed}.checkpoint",
            ) != checkpoint_path.resolve():
                raise FinalizationError(f"baseline summary checkpoint mismatch for {dataset}/{seed}")
            for key in ("best_epoch", "iou", "pd", "fa"):
                _same_number(
                    summary_run.get(key), validated.get(key),
                    f"baseline summary {dataset}/{seed}.{key}",
                )
            trusted[pair] = {
                "job": job,
                "checkpoint_path": checkpoint_path.resolve(),
                "checkpoint_sha256": sha256(checkpoint_path),
                "metrics": {
                    "best_epoch": int(validated["best_epoch"]),
                    "iou": float(validated["iou"]),
                    "pd": float(validated["pd"]),
                    "fa": float(validated["fa"]),
                },
            }
    if len(trusted) != EXPECTED_AUDIT_COUNT:
        raise FinalizationError("baseline grid is not exactly 3 datasets x 3 seeds")
    return seeds, datasets_meta, jobs, trusted


def _validate_audit_batch_manifest(
    audit_root: Path,
    batch_dir: Path,
    *,
    seeds: list[int],
    trusted: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    path = audit_root / "batch_manifest.json"
    manifest = require_mapping(
        read_json(path, "mechanism-audit batch manifest"),
        "mechanism-audit batch manifest",
    )
    expected_header = {
        "schema_version": "dea.clean_mechanism_audit_batch.v1",
        "batch_id": batch_dir.name,
        "stage": "development_holdout_mechanism_audit",
        "max_processes_per_gpu": 1,
    }
    mismatches = [
        key for key, expected in expected_header.items() if manifest.get(key) != expected
    ]
    policy = manifest.get("official_test_policy")
    if not isinstance(policy, str) or "never opened or iterated" not in policy:
        mismatches.append("official_test_policy")
    baseline_manifest = batch_dir / "manifest.json"
    baseline_summary = batch_dir / BASELINE_SUMMARY_JSON
    path_expectations = {
        "baseline_manifest": baseline_manifest.resolve(),
        "baseline_summary": baseline_summary.resolve(),
    }
    for key, expected in path_expectations.items():
        if normalized_path(manifest.get(key), f"audit batch.{key}") != expected:
            mismatches.append(key)
        if require_sha256(
            manifest.get(f"{key}_sha256"), f"audit batch.{key}_sha256"
        ) != sha256(expected):
            mismatches.append(f"{key}_sha256")
    if mismatches:
        raise FinalizationError(
            "mechanism-audit batch manifest identity mismatch: "
            + ", ".join(sorted(set(mismatches)))
        )

    raw_jobs = manifest.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_AUDIT_COUNT:
        raise FinalizationError("audit batch must declare exactly nine jobs")
    jobs: dict[tuple[str, int], dict[str, Any]] = {}
    for index, raw_job in enumerate(raw_jobs):
        job = require_mapping(raw_job, f"audit batch.jobs[{index}]")
        dataset = job.get("dataset")
        seed = require_exact_int(job.get("seed"), f"audit batch.jobs[{index}].seed")
        pair = (dataset, seed)
        if pair not in trusted or pair in jobs:
            raise FinalizationError(f"unexpected/duplicate audit batch pair: {pair!r}")
        expected_id = f"mean_anchor__{dataset.lower()}__seed_{seed}"
        expected_output = audit_root / "artifacts" / dataset / f"seed_{seed}"
        expected_checkpoint = trusted[pair]["checkpoint_path"]
        if job.get("job_id") != expected_id:
            raise FinalizationError(f"audit job_id mismatch for {pair!r}")
        if normalized_path(job.get("output_dir"), f"audit job {expected_id}.output_dir") != expected_output.resolve():
            raise FinalizationError(f"audit output_dir mismatch for {pair!r}")
        if normalized_path(job.get("checkpoint"), f"audit job {expected_id}.checkpoint") != expected_checkpoint:
            raise FinalizationError(f"audit checkpoint path mismatch for {pair!r}")
        if require_sha256(job.get("checkpoint_sha256"), f"audit job {expected_id}.checkpoint_sha256") != trusted[pair]["checkpoint_sha256"]:
            raise FinalizationError(f"audit checkpoint hash mismatch for {pair!r}")
        baseline_metrics = require_mapping(
            job.get("baseline_metrics"), f"audit job {expected_id}.baseline_metrics"
        )
        for key, expected in trusted[pair]["metrics"].items():
            _same_number(baseline_metrics.get(key), expected, f"audit job {expected_id}.{key}")
        jobs[pair] = job
    expected_pairs = {(dataset, seed) for dataset in DATASET_NAMES for seed in seeds}
    if set(jobs) != expected_pairs:
        raise FinalizationError("audit batch jobs do not form the complete dataset x seed grid")
    return manifest, jobs


def _validate_runner_result(job: dict[str, Any], pair: tuple[str, int]) -> None:
    job_id = str(job["job_id"])
    result_path = normalized_path(job.get("result_file"), f"audit job {job_id}.result_file")
    log_path = normalized_path(job.get("log_file"), f"audit job {job_id}.log_file")
    if not log_path.is_file():
        raise FinalizationError(f"missing audit job log: {log_path}")
    result = require_mapping(read_json(result_path, f"audit result {job_id}"), f"audit result {job_id}")
    expected = {
        "schema_version": "dea.clean_mechanism_audit_job.v1",
        "status": "completed_verified",
        "job_id": job_id,
        "dataset": pair[0],
        "seed": pair[1],
        "returncode": 0,
        "output_dir": job["output_dir"],
        "checkpoint": job["checkpoint"],
        "checkpoint_sha256": job["checkpoint_sha256"],
    }
    differences = [key for key, value in expected.items() if result.get(key) != value]
    if differences:
        raise FinalizationError(
            f"audit result identity mismatch for {job_id}: {', '.join(differences)}"
        )


def _validate_audit_manifest(
    manifest_path: Path,
    *,
    batch_dir: Path,
    audit_root: Path,
    pair: tuple[str, int],
    dataset_meta: dict[str, Any],
    trusted: dict[str, Any],
    batch_job: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    dataset, seed = pair
    audit_dir = manifest_path.parent.resolve()
    expected_dir = (audit_root / "artifacts" / dataset / f"seed_{seed}").resolve()
    if audit_dir != expected_dir:
        raise FinalizationError(
            f"audit manifest path mismatch for {dataset}/{seed}: {audit_dir} != {expected_dir}"
        )
    manifest = require_mapping(read_json(manifest_path, "audit manifest"), "audit manifest")
    config = require_mapping(batch_job.get("config"), f"audit job {dataset}/{seed}.config")
    expected_header = {
        "schema_version": AUDIT_SCHEMA,
        "dataset": dataset,
        "dataset_dir": str(normalized_path(trusted["job"].get("dataset_dir"), "baseline dataset_dir")),
        "split_role": EXPECTED_SPLIT_ROLE,
        "split_sha256": dataset_meta["val_sha256"],
        "validation_split_sha256": dataset_meta["val_sha256"],
        "seed": seed,
        "method": "MSHNet",
        "model_type": EXPECTED_METHOD,
        "threshold_probability": EXPECTED_THRESHOLD_PROBABILITY,
        "threshold_logit": EXPECTED_THRESHOLD_LOGIT,
        "connectivity": EXPECTED_CONNECTIVITY,
        "max_centroid_distance": EXPECTED_MAX_DISTANCE,
        "anchor_mode": EXPECTED_ANCHOR_MODE,
        "active_stage": EXPECTED_ACTIVE_STAGE,
        "official_test_status": EXPECTED_OFFICIAL_TEST_STATUS,
        "deterministic": True,
    }
    for field in ("base_size", "crop_size", "batch_size", "num_workers"):
        expected_header[field] = config.get(field)
    mismatches = [
        f"{key}: audit={manifest.get(key)!r}, expected={expected!r}"
        for key, expected in expected_header.items()
        if manifest.get(key) != expected
    ]
    if mismatches:
        raise FinalizationError(
            f"audit manifest identity mismatch for {dataset}/{seed}: " + "; ".join(mismatches)
        )
    eps = require_number(manifest.get("eps"), f"audit {dataset}/{seed}.eps")
    if not math.isclose(eps, 1e-6, rel_tol=0.0, abs_tol=0.0):
        raise FinalizationError(f"audit {dataset}/{seed} must use frozen eps=1e-6")
    thresholds = manifest.get("candidate_probability_thresholds")
    if thresholds != [0.5, 0.3, 0.2, 0.1]:
        raise FinalizationError(f"audit {dataset}/{seed} candidate thresholds are not frozen")
    reconstruction_error = require_number(
        manifest.get("max_mobius_reconstruction_abs_error"),
        f"audit {dataset}/{seed}.max_mobius_reconstruction_abs_error",
    )
    if reconstruction_error < 0 or reconstruction_error > 1e-5:
        raise FinalizationError(f"audit {dataset}/{seed} has excessive decomposition error")

    source_hashes = require_mapping(
        manifest.get("source_sha256"), f"audit {dataset}/{seed}.source_sha256"
    )
    expected_source_keys = {
        "exporter",
        "baseline_finalizer",
        "mshnet",
        "mean_anchor_probe",
        "component_candidates",
        "dataset",
        "metrics",
    }
    if set(source_hashes) != expected_source_keys:
        raise FinalizationError(f"audit {dataset}/{seed} source hash mapping is incomplete")
    validated_source_hashes = {
        key: require_sha256(value, f"audit {dataset}/{seed}.source_sha256.{key}")
        for key, value in source_hashes.items()
    }

    checkpoint = require_mapping(manifest.get("checkpoint"), f"audit {dataset}/{seed}.checkpoint")
    if checkpoint.get("role") != EXPECTED_CHECKPOINT_ROLE:
        raise FinalizationError(f"audit {dataset}/{seed} checkpoint role is not best_iou")
    if normalized_path(checkpoint.get("path"), f"audit {dataset}/{seed}.checkpoint.path") != trusted["checkpoint_path"]:
        raise FinalizationError(f"audit {dataset}/{seed} checkpoint path mismatch")
    digest = require_sha256(checkpoint.get("sha256"), f"audit {dataset}/{seed}.checkpoint.sha256")
    if digest != trusted["checkpoint_sha256"] or sha256(trusted["checkpoint_path"]) != digest:
        raise FinalizationError(f"audit {dataset}/{seed} checkpoint SHA-256 mismatch")
    if require_exact_int(checkpoint.get("epoch"), f"audit {dataset}/{seed}.checkpoint.epoch") != trusted["metrics"]["best_epoch"]:
        raise FinalizationError(f"audit {dataset}/{seed} checkpoint epoch mismatch")
    checkpoint_metrics = require_mapping(
        checkpoint.get("metrics"), f"audit {dataset}/{seed}.checkpoint.metrics"
    )
    for key in ("iou", "pd", "fa"):
        _same_number(checkpoint_metrics.get(key), trusted["metrics"][key], f"audit {dataset}/{seed}.checkpoint.{key}")
    _same_number(checkpoint_metrics.get("best_iou"), trusted["metrics"]["iou"], f"audit {dataset}/{seed}.checkpoint.best_iou")

    provenance = require_mapping(
        manifest.get("baseline_provenance"), f"audit {dataset}/{seed}.baseline_provenance"
    )
    expected_provenance = {
        "batch_id": batch_dir.name,
        "job_id": trusted["job"]["job_id"],
        "batch_manifest": str((batch_dir / "manifest.json").resolve()),
        "baseline_summary": str((batch_dir / BASELINE_SUMMARY_JSON).resolve()),
        "completion": EXPECTED_BASELINE_COMPLETION,
    }
    differences = [key for key, value in expected_provenance.items() if provenance.get(key) != value]
    if differences:
        raise FinalizationError(
            f"audit {dataset}/{seed} baseline provenance mismatch: {', '.join(differences)}"
        )
    validation = require_mapping(
        manifest.get("checkpoint_validation"),
        f"audit {dataset}/{seed}.checkpoint_validation",
    )
    if (
        validation.get("model_seed_val_hash") != "matched"
        or validation.get("strict_state_dict") is not True
        or validation.get("frozen") is not True
    ):
        raise FinalizationError(f"audit {dataset}/{seed} checkpoint validation is not strict/frozen")

    artifacts = require_mapping(manifest.get("artifacts"), f"audit {dataset}/{seed}.artifacts")
    images_path = _artifact_path(audit_dir, artifacts.get("images_jsonl"), "images_jsonl")
    components_path = _artifact_path(
        audit_dir, artifacts.get("components_jsonl"), "components_jsonl"
    )
    if sha256(images_path) != require_sha256(
        artifacts.get("images_sha256"), f"audit {dataset}/{seed}.artifacts.images_sha256"
    ):
        raise FinalizationError(f"audit {dataset}/{seed} images.jsonl SHA-256 mismatch")
    if sha256(components_path) != require_sha256(
        artifacts.get("components_sha256"),
        f"audit {dataset}/{seed}.artifacts.components_sha256",
    ):
        raise FinalizationError(f"audit {dataset}/{seed} components.jsonl SHA-256 mismatch")
    arrays_dir = _directory_path(audit_dir, artifacts.get("arrays_dir"), "arrays_dir")
    image_rows = _read_jsonl(images_path, "images.jsonl")
    component_rows = _read_jsonl(components_path, "components.jsonl")
    summary = require_mapping(manifest.get("summary"), f"audit {dataset}/{seed}.summary")
    image_counts, inventory, image_by_id = _validate_image_rows(
        image_rows,
        audit_dir=audit_dir,
        arrays_dir=arrays_dir,
        artifacts=artifacts,
        summary=summary,
    )
    _validate_component_rows(component_rows, image_by_id)
    if int(summary["pixels"]) != len(image_rows) * int(manifest["base_size"]) * int(manifest["crop_size"]):
        raise FinalizationError(f"audit {dataset}/{seed} pixel count disagrees with image geometry")
    counts, metrics = _validate_summary(summary, image_counts, eps=eps)

    recomputed = require_mapping(
        validation.get("recomputed_metrics"),
        f"audit {dataset}/{seed}.checkpoint_validation.recomputed_metrics",
    )
    metric_names = {"iou": "pooled_iou", "pd": "pd", "fa": "fa_per_million"}
    for checkpoint_name, summary_name in metric_names.items():
        pair_validation = require_mapping(
            recomputed.get(checkpoint_name),
            f"audit {dataset}/{seed}.recomputed_metrics.{checkpoint_name}",
        )
        expected = trusted["metrics"][checkpoint_name]
        _same_number(pair_validation.get("checkpoint"), expected, f"audit {dataset}/{seed} recomputed checkpoint {checkpoint_name}")
        _same_number(pair_validation.get("recomputed"), metrics[summary_name], f"audit {dataset}/{seed} recomputed factual {checkpoint_name}")

    run = {
        "dataset": dataset,
        "seed": seed,
        "audit_manifest": str(manifest_path.resolve()),
        "checkpoint": str(trusted["checkpoint_path"]),
        "checkpoint_sha256": digest,
        "baseline_metrics": {
            "iou": metrics["pooled_iou"],
            "pd": metrics["pd"],
            "fa_per_million": metrics["fa_per_million"],
        },
        "counts": counts,
        "derived_metrics": metrics,
        "artifact_counts": {
            "image_rows": len(image_rows),
            "component_rows": len(component_rows),
            "array_files": len(inventory),
            "array_bytes": sum(record["bytes"] for record in inventory),
        },
    }
    return run, manifest, validated_source_hashes


def _aggregate_runs(runs: list[dict[str, Any]], *, eps: float) -> dict[str, Any]:
    if not runs:
        raise FinalizationError("cannot aggregate an empty run collection")
    counts = _sum_totals(runs)
    return {
        "run_count": len(runs),
        "counts": counts,
        "ratio_of_sums_metrics": _derive_metrics(counts, eps),
        "artifact_counts": {
            field: sum(int(run["artifact_counts"][field]) for run in runs)
            for field in ("image_rows", "component_rows", "array_files", "array_bytes")
        },
    }


def _format_metrics_table_row(label: str, aggregate: dict[str, Any]) -> str:
    metrics = aggregate["ratio_of_sums_metrics"]
    counts = aggregate["counts"]
    return (
        f"| {label} | {aggregate['run_count']} | {metrics['pooled_iou']:.6f} | "
        f"{metrics['pd']:.6f} | {metrics['fa_per_million']:.3f} | "
        f"{counts['false_positive_component_count']} | "
        f"{counts['false_negative_component_count']} | "
        f"{counts['recoverable_fn_component_count']} | "
        f"{counts['conflict_pixels']} | "
        f"{metrics['conflict_false_positive_coverage']:.6f} |"
    )


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Clean MSHNet mechanism-audit evidence summary",
        "",
        "> **Scope guard:** This is a validated, descriptive summary of frozen MSHNet "
        "development-holdout errors and mean-anchor interaction measurements. The official "
        "test sets remain sealed and were not opened or evaluated. No DEA model was evaluated, "
        "so this artifact contains no DEA gain, causal-mechanism, or paper-performance claim.",
        "",
        f"- Baseline batch: `{summary['batch_id']}`",
        f"- Validated evidence grid: {len(DATASET_NAMES)} datasets × {len(summary['seeds'])} seeds",
        f"- Audit schema: `{AUDIT_SCHEMA}`",
        "- Aggregation: raw counts are summed; IoU, PD, FA/M, recoverable-FN rate, conflict "
        "coverage and interaction statistics are recomputed as ratios of summed numerators "
        "and denominators (not averages of per-run ratios).",
        "- Selection/evaluation scope: best-IoU checkpoint on the fixed internal validation "
        "split; design-used evidence only.",
        "",
        "## Per-run evidence ledger",
        "",
        "| Dataset | Seed | IoU | PD | FA/M | FP comp. | FN comp. | Recoverable FN | "
        "Conflict px | FP-conflict coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        metrics = run["baseline_metrics"]
        derived = run["derived_metrics"]
        counts = run["counts"]
        lines.append(
            f"| {run['dataset']} | {run['seed']} | {metrics['iou']:.6f} | "
            f"{metrics['pd']:.6f} | {metrics['fa_per_million']:.3f} | "
            f"{counts['false_positive_component_count']} | "
            f"{counts['false_negative_component_count']} | "
            f"{counts['recoverable_fn_component_count']} | "
            f"{counts['conflict_pixels']} | "
            f"{derived['conflict_false_positive_coverage']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Dataset-level ratio-of-sums evidence",
            "",
            "| Dataset | Runs | IoU | PD | FA/M | FP comp. | FN comp. | Recoverable FN | "
            "Conflict px | FP-conflict coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASET_NAMES:
        lines.append(_format_metrics_table_row(dataset, summary["by_dataset"][dataset]))

    lines.extend(
        [
            "",
            "## Seed-level ratio-of-sums evidence",
            "",
            "| Seed | Runs | IoU | PD | FA/M | FP comp. | FN comp. | Recoverable FN | "
            "Conflict px | FP-conflict coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in summary["seeds"]:
        lines.append(_format_metrics_table_row(str(seed), summary["by_seed"][str(seed)]))

    overall = summary["overall"]
    metrics, counts = overall["ratio_of_sums_metrics"], overall["counts"]
    lines.extend(
        [
            "",
            "## Cross-grid evidence inventory",
            "",
            f"- Baseline FP components: {counts['false_positive_component_count']}",
            f"- Baseline FN components: {counts['false_negative_component_count']}",
            f"- Prediction-only recoverable FN components: {counts['recoverable_fn_component_count']} "
            f"({metrics['recoverable_fn_fraction']:.6f} of baseline FN components)",
            f"- Conflict pixels: {counts['conflict_pixels']} "
            f"({metrics['conflict_fraction']:.6f} of audited pixels)",
            f"- Conflict coverage on TP / FP / FN pixels: "
            f"{metrics['conflict_true_positive_coverage']:.6f} / "
            f"{metrics['conflict_false_positive_coverage']:.6f} / "
            f"{metrics['conflict_false_negative_coverage']:.6f}",
            f"- Mean-anchor index (ratio of summed score to pixels): "
            f"{metrics['mean_anchor_index']:.9f}",
            f"- Global interaction RMS ratio of sums: "
            f"{metrics['global_r_ratio_of_sums']:.9f}",
            "",
            "## Interpretation boundary for the next design stage",
            "",
            "These measurements may be used as inputs to a later problem/gap/root-cause "
            "analysis. They do **not** by themselves show that interaction conflict causes an "
            "error, that a proposed DEA will remove an FP or recover an FN, that TP will be "
            "preserved, or that mean-anchor measurements predict treatment benefit. Those "
            "claims require preregistered paired DEA/control results on evidence not used to "
            "invent the model. No official-test or final-paper result is present here.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(json_path: Path, markdown_path: Path, summary: dict[str, Any]) -> None:
    json_text = json.dumps(summary, indent=2, sort_keys=False, allow_nan=False) + "\n"
    markdown_text = build_markdown(summary)
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    markdown_tmp = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    try:
        json_tmp.write_text(json_text, encoding="utf-8")
        markdown_tmp.write_text(markdown_text, encoding="utf-8")
        os.replace(json_tmp, json_path)
        os.replace(markdown_tmp, markdown_path)
    finally:
        for temporary in (json_tmp, markdown_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def finalize_audits(
    batch_dir: Path,
    *,
    audit_root: Path | None = None,
    force: bool = False,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = load_checkpoint_cpu,
) -> dict[str, Any]:
    """Validate all nine audits and write a descriptive evidence summary."""
    batch_dir = batch_dir.expanduser().resolve()
    if audit_root is None:
        audit_root = batch_dir / "mechanism_audits"
    audit_root = audit_root.expanduser().resolve()
    if not audit_root.is_dir() or audit_root.is_symlink():
        raise FinalizationError(f"missing regular mechanism-audit root: {audit_root}")
    json_path = audit_root / OUTPUT_JSON
    markdown_path = audit_root / OUTPUT_MARKDOWN
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing mechanism evidence summary: "
            + ", ".join(str(path) for path in existing)
            + "; pass --force only when intentional"
        )

    seeds, datasets_meta, _, trusted = _validate_baseline(
        batch_dir, checkpoint_loader=checkpoint_loader
    )
    batch_manifest, batch_jobs = _validate_audit_batch_manifest(
        audit_root, batch_dir, seeds=seeds, trusted=trusted
    )
    batch_sources = require_mapping(
        batch_manifest.get("source_sha256"), "audit batch.source_sha256"
    )
    expected_source_keys = {
        "exporter",
        "baseline_finalizer",
        "mshnet",
        "mean_anchor_probe",
        "component_candidates",
        "dataset",
        "metrics",
    }
    if set(batch_sources) != expected_source_keys:
        raise FinalizationError("audit batch source hash mapping is incomplete")
    frozen_source_hashes = {
        key: require_sha256(value, f"audit batch.source_sha256.{key}")
        for key, value in batch_sources.items()
    }

    artifacts_root = audit_root / "artifacts"
    if not artifacts_root.is_dir() or artifacts_root.is_symlink():
        raise FinalizationError(f"missing regular artifacts directory: {artifacts_root}")
    discovered = set(artifacts_root.rglob("manifest.json"))
    expected = {
        audit_root / "artifacts" / dataset / f"seed_{seed}" / "manifest.json"
        for dataset in DATASET_NAMES
        for seed in seeds
    }
    if discovered != expected or len(discovered) != EXPECTED_AUDIT_COUNT:
        missing = sorted(str(path) for path in expected - discovered)
        extra = sorted(str(path) for path in discovered - expected)
        raise FinalizationError(
            "audit manifest inventory is not the exact 3x3 grid; "
            f"missing={missing}, extra={extra}"
        )

    runs: list[dict[str, Any]] = []
    common_definitions: dict[str, Any] | None = None
    for dataset in DATASET_NAMES:
        dataset_meta = require_mapping(
            datasets_meta[dataset], f"baseline dataset metadata {dataset}"
        )
        for seed in seeds:
            pair = (dataset, seed)
            batch_job = batch_jobs[pair]
            _validate_runner_result(batch_job, pair)
            path = audit_root / "artifacts" / dataset / f"seed_{seed}" / "manifest.json"
            run, manifest, source_hashes = _validate_audit_manifest(
                path,
                batch_dir=batch_dir,
                audit_root=audit_root,
                pair=pair,
                dataset_meta=dataset_meta,
                trusted=trusted[pair],
                batch_job=batch_job,
            )
            if source_hashes != frozen_source_hashes:
                raise FinalizationError(
                    f"audit {dataset}/{seed} source hashes disagree with frozen batch mapping"
                )
            if int(run["counts"]["images"]) != require_exact_int(
                dataset_meta.get("val_count"), f"baseline metadata {dataset}.val_count"
            ):
                raise FinalizationError(f"audit {dataset}/{seed} does not cover full validation split")
            definitions = {
                key: manifest.get(key)
                for key in (
                    "recoverable_fn_definition",
                    "conflict_definition",
                    "global_ratio_of_sums_definition",
                )
            }
            if any(not isinstance(value, str) or not value for value in definitions.values()):
                raise FinalizationError(f"audit {dataset}/{seed} has missing mechanism definitions")
            if common_definitions is None:
                common_definitions = definitions
            elif definitions != common_definitions:
                raise FinalizationError("mechanism definitions differ across audit runs")
            run["audit_manifest_sha256"] = sha256(path)
            runs.append(run)

    if len(runs) != EXPECTED_AUDIT_COUNT or common_definitions is None:
        raise FinalizationError("validated audit run count is not exactly nine")
    eps = 1e-6
    by_dataset = {
        dataset: _aggregate_runs(
            [run for run in runs if run["dataset"] == dataset], eps=eps
        )
        for dataset in DATASET_NAMES
    }
    by_seed = {
        str(seed): _aggregate_runs([run for run in runs if run["seed"] == seed], eps=eps)
        for seed in seeds
    }
    if any(item["run_count"] != len(seeds) for item in by_dataset.values()):
        raise FinalizationError("internal dataset aggregation grid error")
    if any(item["run_count"] != len(DATASET_NAMES) for item in by_seed.values()):
        raise FinalizationError("internal seed aggregation grid error")
    overall = _aggregate_runs(runs, eps=eps)

    summary: dict[str, Any] = {
        "schema_version": "dea.clean_mechanism_audit_evidence_summary.v1",
        "batch_id": batch_dir.name,
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete_and_validated",
        "evaluation_scope": "official-training-set internal development holdouts only",
        "official_test_status": "sealed; not opened or evaluated by exporter/finalizer",
        "not_for_official_test_or_main_table_claims": True,
        "dea_evaluated": False,
        "dea_gain_claimed": False,
        "causal_mechanism_claimed": False,
        "aggregation": (
            "raw numerator/denominator counts summed before ratios; no mean of per-run ratios"
        ),
        "seeds": seeds,
        "datasets": list(DATASET_NAMES),
        "validated_grid": {
            "dataset_count": len(DATASET_NAMES),
            "seed_count": len(seeds),
            "run_count": len(runs),
        },
        "provenance": {
            "baseline_manifest": str((batch_dir / "manifest.json").resolve()),
            "baseline_manifest_sha256": sha256(batch_dir / "manifest.json"),
            "baseline_summary": str((batch_dir / BASELINE_SUMMARY_JSON).resolve()),
            "baseline_summary_sha256": sha256(batch_dir / BASELINE_SUMMARY_JSON),
            "audit_batch_manifest": str((audit_root / "batch_manifest.json").resolve()),
            "audit_batch_manifest_sha256": sha256(audit_root / "batch_manifest.json"),
            "source_sha256": frozen_source_hashes,
        },
        "mechanism_definitions": common_definitions,
        "metric_definitions": {
            "pooled_iou": "sum(intersection pixels) / sum(union pixels)",
            "pd": "sum(matched target components) / sum(target components)",
            "fa_per_million": "sum(unmatched prediction-component area) / sum(image pixels) * 1e6",
            "recoverable_fn_fraction": "sum(prediction-only recoverable FN components) / sum(FN components)",
            "conflict_coverage": "sum(conflict pixels within region class) / sum(region-class pixels)",
            "mean_anchor_index": "sum(mean-anchor conflict score) / sum(image pixels)",
            "global_r_ratio_of_sums": "sum(interaction feature RMS) / (sum(current-main feature RMS) + eps)",
        },
        "runs": runs,
        "by_dataset": by_dataset,
        "by_seed": by_seed,
        "overall": overall,
        "interpretation_boundary": {
            "descriptive_baseline_evidence_only": True,
            "does_not_establish_error_causation": True,
            "does_not_establish_dea_benefit": True,
            "does_not_establish_mean_anchor_predictiveness": True,
            "requires_later_paired_dea_and_control_evidence": True,
        },
    }
    _write_outputs(json_path, markdown_path, summary)
    return summary


def main() -> int:
    args = parse_args()
    if (
        SAFE_BATCH_ID.fullmatch(args.batch_id or "") is None
        or args.batch_id in {".", ".."}
        or Path(args.batch_id).name != args.batch_id
    ):
        raise FinalizationError("--batch-id must be one safe directory name, not a path")
    batch_dir = PROJECT_DIR / "repro_runs" / "clean" / args.batch_id
    audit_root = (
        Path(args.audit_root).expanduser().resolve()
        if args.audit_root
        else batch_dir / "mechanism_audits"
    )
    summary = finalize_audits(
        batch_dir, audit_root=audit_root, force=args.force
    )
    print(
        f"validated {len(DATASET_NAMES)} datasets x {len(summary['seeds'])} seeds; "
        f"wrote {audit_root / OUTPUT_JSON} and {audit_root / OUTPUT_MARKDOWN}"
    )
    print("scope: descriptive development-holdout baseline evidence; official test remains sealed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalizationError, FileExistsError) as exc:
        print(f"mechanism-audit finalization refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
