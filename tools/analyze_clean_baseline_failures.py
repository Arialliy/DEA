#!/usr/bin/env python3
"""Analyze validated clean-MSHNet development-holdout failure evidence.

This tool is intentionally downstream of ``finalize_clean_mechanism_audits``.
It consumes that finalizer's JSON summary plus the hash-bound image/component
ledgers, and produces descriptive baseline evidence only.  It never reads an
official-test split, evaluates a DEA model, estimates a DEA gain, or treats the
mean-anchor index as a treatment-benefit predictor.

Repeated predictions of the same dataset/image under different training seeds
are one inferential cluster.  Every confidence interval therefore resamples
``(dataset, image_id)`` clusters (stratified by dataset), keeping all seed
observations for a sampled image together.  Global IoU is always recomputed as
``sum(intersection) / sum(union)``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SUMMARY_SCHEMA = "dea.clean_mechanism_audit_evidence_summary.v1"
AUDIT_SCHEMA = "dea.clean_mechanism_audit.v1"
OUTPUT_SCHEMA = "dea.clean_baseline_failure_analysis.v1"
OUTPUT_JSON = "clean_baseline_failure_analysis.json"
OUTPUT_MARKDOWN = "clean_baseline_failure_analysis.md"
DATASET_NAMES = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
EXPECTED_EVALUATION_SCOPE = "official-training-set internal development holdouts only"
EXPECTED_OFFICIAL_TEST_STATUS = (
    "sealed; not opened or evaluated by exporter/finalizer"
)
EXPECTED_AUDIT_OFFICIAL_TEST_STATUS = (
    "sealed; this exporter accepts development validation only"
)
DEFAULT_BOOTSTRAP_REPLICATES = 5000
DEFAULT_BOOTSTRAP_SEED = 20260711

TAXONOMY_ORDER = (
    "pixel_perfect",
    "matched_component_localization",
    "fp_only",
    "fn_only",
    "mixed_fp_fn",
)
TAXONOMY_DEFINITIONS = {
    "pixel_perfect": "zero false-positive pixels and zero false-negative pixels",
    "matched_component_localization": (
        "pixel disagreement exists, but no unmatched prediction or target component exists"
    ),
    "fp_only": "one or more unmatched prediction components and no unmatched target component",
    "fn_only": "one or more unmatched target components and no unmatched prediction component",
    "mixed_fp_fn": "both unmatched prediction and unmatched target components occur",
}

IMAGE_INTEGER_FIELDS = (
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
IMAGE_FLOAT_FIELDS = (
    "iou",
    "mean_anchor_index",
    "interaction_ratio_mean",
    "interaction_ratio_p95",
    "conflict_fraction",
    "mean_anchor_score_sum_true_positive",
    "mean_anchor_score_sum_false_positive",
    "mean_anchor_score_sum_false_negative",
)
COMPONENT_FLOAT_FIELDS = (
    "p_z_mean",
    "j_z_mean",
    "interaction_ratio_mean",
    "interaction_ratio_p95",
    "mean_anchor_score_mean",
    "prediction_logit_mean",
    "conflict_fraction",
)


class AnalysisError(RuntimeError):
    """Raised when evidence is not the finalized development-only artifact."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clustered cross-dataset/seed taxonomy from the finalized "
            "clean-MSHNet development-holdout mechanism audits."
        )
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to clean_mechanism_audit_evidence_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory; defaults to the summary directory.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing analysis JSON/Markdown after all validation succeeds.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {label} {path}: {exc}") from exc
    return _mapping(value, label)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"cannot read {label} {path}: {exc}") from exc
    if not lines:
        raise AnalysisError(f"{label} is empty: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise AnalysisError(f"blank line in {label} at line {number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"invalid {label} JSON at line {number}: {exc}") from exc
        rows.append(_mapping(value, f"{label} line {number}"))
    return rows


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must be a JSON object")
    return value


def _integer(value: Any, label: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{label} must be an integer, got {value!r}")
    if nonnegative and value < 0:
        raise AnalysisError(f"{label} must be non-negative")
    return value


def _number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{label} must be finite")
    if nonnegative and result < 0:
        raise AnalysisError(f"{label} must be non-negative")
    return result


def _sha256_value(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AnalysisError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _artifact_path(directory: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise AnalysisError(f"{label} must be a normalized relative path")
    path = directory / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AnalysisError(f"missing {label}: {path}") from exc
    if directory.resolve() not in resolved.parents:
        raise AnalysisError(f"{label} escapes its audit directory")
    if path.is_symlink() or not resolved.is_file():
        raise AnalysisError(f"{label} must be a regular non-symlink file")
    return resolved


def _same_float(actual: float, expected: float, label: str, *, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=tolerance):
        raise AnalysisError(f"{label} mismatch: {actual!r} != {expected!r}")


def classify_failure(image: Mapping[str, Any]) -> str:
    """Return the preregistered mutually exclusive image failure category."""
    fp_components = int(image["false_positive_component_count"])
    fn_components = int(image["false_negative_component_count"])
    fp_pixels = int(image["false_positive_pixels"])
    fn_pixels = int(image["false_negative_pixels"])
    if fp_components > 0 and fn_components > 0:
        return "mixed_fp_fn"
    if fp_components > 0:
        return "fp_only"
    if fn_components > 0:
        return "fn_only"
    if fp_pixels > 0 or fn_pixels > 0:
        return "matched_component_localization"
    return "pixel_perfect"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _validated_vectors(
    first: Sequence[float], second: Sequence[float], label: str
) -> tuple[np.ndarray, np.ndarray] | tuple[None, str]:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        return None, f"{label}_length_or_shape_mismatch"
    if left.size < 2:
        return None, "fewer_than_two_image_clusters"
    if not bool(np.isfinite(left).all() and np.isfinite(right).all()):
        return None, "non_finite_input"
    return left, right


def auroc(scores: Sequence[float], labels: Sequence[int | bool]) -> dict[str, Any]:
    """Tie-correct AUROC with an explicit undefined result."""
    validated = _validated_vectors(scores, labels, "auroc")
    if validated[0] is None:
        return {"defined": False, "estimate": None, "undefined_reason": validated[1]}
    score_values, label_values = validated
    if not bool(np.isin(label_values, (0.0, 1.0)).all()):
        return {"defined": False, "estimate": None, "undefined_reason": "labels_not_binary"}
    positives = int(label_values.sum())
    negatives = int(label_values.size - positives)
    if positives == 0 or negatives == 0:
        return {
            "defined": False,
            "estimate": None,
            "undefined_reason": "single_class_outcome",
            "positive_image_clusters": positives,
            "negative_image_clusters": negatives,
        }
    ranks = _average_ranks(score_values)
    value = float(
        (ranks[label_values == 1].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )
    return {
        "defined": True,
        "estimate": value,
        "undefined_reason": None,
        "positive_image_clusters": positives,
        "negative_image_clusters": negatives,
    }


def spearman_correlation(
    first: Sequence[float], second: Sequence[float]
) -> dict[str, Any]:
    """Tie-correct Spearman correlation with explicit undefined handling."""
    validated = _validated_vectors(first, second, "spearman")
    if validated[0] is None:
        return {"defined": False, "estimate": None, "undefined_reason": validated[1]}
    left, right = validated
    left_rank, right_rank = _average_ranks(left), _average_ranks(right)
    if np.ptp(left_rank) == 0:
        return {"defined": False, "estimate": None, "undefined_reason": "constant_predictor"}
    if np.ptp(right_rank) == 0:
        return {"defined": False, "estimate": None, "undefined_reason": "constant_outcome"}
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return {"defined": True, "estimate": value, "undefined_reason": None}


def _key_getter(key: str | Sequence[str] | Callable[[Mapping[str, Any]], Any]):
    if callable(key):
        return key
    if isinstance(key, str):
        return lambda row: row[key]
    fields = tuple(key)
    return lambda row: tuple(row[field] for field in fields)


def clustered_bootstrap_ci(
    records: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float | None],
    *,
    cluster_key: str | Sequence[str] | Callable[[Mapping[str, Any]], Any],
    strata_key: str | Callable[[Mapping[str, Any]], Any] | None = None,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Generic deterministic cluster bootstrap, retaining whole clusters.

    This public helper is deliberately simple for independent audit tests.  The
    main analysis uses the vectorized ``_ClusterBootstrap`` below.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    cluster_get = _key_getter(cluster_key)
    strata_get = _key_getter(strata_key) if strata_key is not None else lambda row: "all"
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    cluster_strata: dict[Any, Any] = {}
    for record in records:
        cluster = cluster_get(record)
        stratum = strata_get(record)
        if cluster in cluster_strata and cluster_strata[cluster] != stratum:
            raise ValueError("one cluster cannot cross bootstrap strata")
        cluster_strata[cluster] = stratum
        grouped[cluster].append(record)
    if not grouped:
        return {
            "defined": False,
            "estimate": None,
            "undefined_reason": "no_image_clusters",
            "ci95": None,
            "bootstrap": {
                "cluster_count": 0,
                "observation_count": 0,
                "replicates_requested": replicates,
                "replicates_defined": 0,
                "seed": seed,
            },
        }
    strata: dict[Any, list[Any]] = defaultdict(list)
    for cluster, stratum in cluster_strata.items():
        strata[stratum].append(cluster)
    for values in strata.values():
        values.sort(key=repr)
    point = statistic(list(records))
    point_value = None if point is None else float(point)
    point_defined = point_value is not None and math.isfinite(point_value)
    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    for _ in range(replicates):
        sample: list[Mapping[str, Any]] = []
        for stratum in sorted(strata, key=repr):
            clusters = strata[stratum]
            selected = rng.choice(len(clusters), size=len(clusters), replace=True)
            for index in selected:
                sample.extend(grouped[clusters[int(index)]])
        value = statistic(sample)
        if value is not None and math.isfinite(float(value)):
            bootstrap_values.append(float(value))
    ci = None
    if bootstrap_values:
        low, high = np.quantile(np.asarray(bootstrap_values), (0.025, 0.975))
        ci = {"low": float(low), "high": float(high)}
    return {
        "defined": point_defined,
        "estimate": point_value if point_defined else None,
        "undefined_reason": None if point_defined else "statistic_undefined_on_original_sample",
        "ci95": ci,
        "bootstrap": {
            "method": "deterministic_stratified_image_cluster_bootstrap",
            "cluster_count": len(grouped),
            "observation_count": len(records),
            "replicates_requested": replicates,
            "replicates_defined": len(bootstrap_values),
            "seed": seed,
        },
    }


@dataclass(frozen=True)
class _Evidence:
    summary: dict[str, Any]
    summary_path: Path
    observations: list[dict[str, Any]]
    components: list[dict[str, Any]]
    manifests: list[dict[str, Any]]


def _validate_image_row(row: dict[str, Any], *, label: str, pixels: int) -> None:
    for field in IMAGE_INTEGER_FIELDS:
        row[field] = _integer(row.get(field), f"{label}.{field}")
    for field in IMAGE_FLOAT_FIELDS:
        row[field] = _number(
            row.get(field),
            f"{label}.{field}",
            nonnegative=field not in {"interaction_ratio_mean", "interaction_ratio_p95"},
        )
    if row["interaction_ratio_mean"] < 0 or row["interaction_ratio_p95"] < 0:
        raise AnalysisError(f"{label} interaction ratios must be non-negative")
    if row["union_pixels"] > pixels:
        raise AnalysisError(f"{label}.union_pixels exceeds image geometry")
    if row["intersection_pixels"] + row["false_positive_pixels"] != row["predicted_positive_pixels"]:
        raise AnalysisError(f"{label} predicted-positive pixel partition is inconsistent")
    if row["intersection_pixels"] + row["false_negative_pixels"] != row["ground_truth_positive_pixels"]:
        raise AnalysisError(f"{label} ground-truth-positive pixel partition is inconsistent")
    if (
        row["intersection_pixels"]
        + row["false_positive_pixels"]
        + row["false_negative_pixels"]
        != row["union_pixels"]
    ):
        raise AnalysisError(f"{label} union pixel partition is inconsistent")
    if row["true_positive_component_count"] + row["false_negative_component_count"] != row["target_component_count"]:
        raise AnalysisError(f"{label} target component partition is inconsistent")
    if row["matched_prediction_component_count"] + row["false_positive_component_count"] != row["prediction_component_count"]:
        raise AnalysisError(f"{label} prediction component partition is inconsistent")
    _same_float(
        row["iou"],
        row["intersection_pixels"] / max(1, row["union_pixels"]),
        f"{label}.iou",
    )
    _same_float(row["conflict_fraction"], row["conflict_pixels"] / pixels, f"{label}.conflict_fraction")
    conflict_partition = (
        row["conflict_on_true_positive_pixels"]
        + row["conflict_on_false_positive_pixels"]
        + row["conflict_on_false_negative_pixels"]
    )
    if conflict_partition > row["conflict_pixels"]:
        raise AnalysisError(f"{label} class-specific conflict pixels exceed total conflict")
    if row["conflict_on_true_positive_pixels"] > row["intersection_pixels"]:
        raise AnalysisError(f"{label} conflict-on-TP exceeds TP pixels")
    if row["conflict_on_false_positive_pixels"] > row["false_positive_pixels"]:
        raise AnalysisError(f"{label} conflict-on-FP exceeds FP pixels")
    if row["conflict_on_false_negative_pixels"] > row["false_negative_pixels"]:
        raise AnalysisError(f"{label} conflict-on-FN exceeds FN pixels")
    score_sum = row["mean_anchor_index"] * pixels
    class_score_sum = sum(
        row[field]
        for field in (
            "mean_anchor_score_sum_true_positive",
            "mean_anchor_score_sum_false_positive",
            "mean_anchor_score_sum_false_negative",
        )
    )
    if class_score_sum > score_sum + max(1e-6, abs(score_sum) * 1e-8):
        raise AnalysisError(f"{label} class-specific mean-anchor score exceeds total score")


def _validate_components(
    rows: list[dict[str, Any]], image_by_id: dict[str, dict[str, Any]], *, label: str
) -> None:
    valid_roles = {
        "target": {"tp_target", "fn_target"},
        "prediction": {"matched_pred", "fp_pred"},
        "candidate": {"candidate"},
    }
    counts: dict[str, dict[str, int]] = {
        image_id: defaultdict(int) for image_id in image_by_id
    }
    seen: set[tuple[str, str, int]] = set()
    for index, row in enumerate(rows):
        row_label = f"{label}[{index}]"
        image_id = row.get("image_id")
        if image_id not in image_by_id:
            raise AnalysisError(f"{row_label}.image_id is absent from images ledger")
        domain, role = row.get("domain"), row.get("role")
        if domain not in valid_roles or role not in valid_roles[domain]:
            raise AnalysisError(f"{row_label} has invalid domain/role")
        component_id = _integer(row.get("component_id"), f"{row_label}.component_id")
        if component_id < 1:
            raise AnalysisError(f"{row_label}.component_id must be one-based")
        identity = (str(image_id), str(domain), component_id)
        if identity in seen:
            raise AnalysisError(f"duplicate component identity: {identity!r}")
        seen.add(identity)
        area = _integer(row.get("area"), f"{row_label}.area")
        conflict_pixels = _integer(row.get("conflict_pixels"), f"{row_label}.conflict_pixels")
        if area <= 0 or conflict_pixels > area:
            raise AnalysisError(f"{row_label} has invalid area/conflict pixels")
        for field in COMPONENT_FLOAT_FIELDS:
            row[field] = _number(row.get(field), f"{row_label}.{field}")
        _same_float(row["conflict_fraction"], conflict_pixels / area, f"{row_label}.conflict_fraction")
        current = counts[str(image_id)]
        if domain == "target":
            current["target_component_count"] += 1
            if role == "tp_target":
                current["true_positive_component_count"] += 1
            else:
                current["false_negative_component_count"] += 1
                recoverable = row.get("recoverable")
                if not isinstance(recoverable, bool):
                    raise AnalysisError(f"{row_label}.recoverable must be boolean")
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
    check_fields = (
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
    )
    for image_id, image in image_by_id.items():
        for field in check_fields:
            if int(counts[image_id][field]) != int(image[field]):
                raise AnalysisError(
                    f"component ledger disagrees with image {image_id}.{field}: "
                    f"{counts[image_id][field]} != {image[field]}"
                )


def load_validated_evidence(summary_path: str | Path) -> _Evidence:
    """Load and fail-closed validate the final summary and hash-bound ledgers."""
    path = Path(summary_path).expanduser()
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise AnalysisError(f"missing finalized audit summary: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise AnalysisError("finalized audit summary must be a regular non-symlink file")
    summary = _read_json(path, "finalized mechanism-audit summary")
    expected = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete_and_validated",
        "evaluation_scope": EXPECTED_EVALUATION_SCOPE,
        "official_test_status": EXPECTED_OFFICIAL_TEST_STATUS,
        "not_for_official_test_or_main_table_claims": True,
        "dea_evaluated": False,
        "dea_gain_claimed": False,
        "causal_mechanism_claimed": False,
    }
    mismatches = [
        f"{key}={summary.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if mismatches:
        raise AnalysisError(
            "input is not the finalized development-only baseline evidence: "
            + "; ".join(mismatches)
        )
    boundary = _mapping(summary.get("interpretation_boundary"), "interpretation_boundary")
    required_boundary = (
        "descriptive_baseline_evidence_only",
        "does_not_establish_error_causation",
        "does_not_establish_dea_benefit",
        "does_not_establish_mean_anchor_predictiveness",
        "requires_later_paired_dea_and_control_evidence",
    )
    if any(boundary.get(field) is not True for field in required_boundary):
        raise AnalysisError("input interpretation boundary is incomplete")
    datasets = summary.get("datasets")
    seeds = summary.get("seeds")
    if datasets != list(DATASET_NAMES):
        raise AnalysisError(f"datasets must be exactly {list(DATASET_NAMES)!r}")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise AnalysisError("validated evidence must contain exactly three distinct integer seeds")
    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) != len(DATASET_NAMES) * len(seeds):
        raise AnalysisError("validated evidence run grid must be exactly 3 datasets x 3 seeds")
    by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for index, value in enumerate(runs):
        run = _mapping(value, f"runs[{index}]")
        pair = (run.get("dataset"), run.get("seed"))
        if pair in by_pair or pair[0] not in DATASET_NAMES or pair[1] not in seeds:
            raise AnalysisError(f"invalid or duplicate audit run identity: {pair!r}")
        by_pair[pair] = run
    expected_pairs = {(dataset, seed) for dataset in DATASET_NAMES for seed in seeds}
    if set(by_pair) != expected_pairs:
        raise AnalysisError("validated evidence run grid is incomplete")

    observations: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    image_ids_by_pair: dict[tuple[str, int], set[str]] = {}
    for dataset in DATASET_NAMES:
        for seed in seeds:
            run = by_pair[(dataset, seed)]
            manifest_value = run.get("audit_manifest")
            if not isinstance(manifest_value, str) or not manifest_value:
                raise AnalysisError(f"audit manifest path missing for {dataset}/{seed}")
            try:
                manifest_path = Path(manifest_value).expanduser().resolve(strict=True)
            except OSError as exc:
                raise AnalysisError(f"missing audit manifest for {dataset}/{seed}") from exc
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise AnalysisError("audit manifest must be a regular non-symlink file")
            expected_manifest_hash = _sha256_value(
                run.get("audit_manifest_sha256"),
                f"run {dataset}/{seed}.audit_manifest_sha256",
            )
            if _sha256(manifest_path) != expected_manifest_hash:
                raise AnalysisError(f"audit manifest hash mismatch for {dataset}/{seed}")
            manifest = _read_json(manifest_path, f"audit manifest {dataset}/{seed}")
            expected_header = {
                "schema_version": AUDIT_SCHEMA,
                "dataset": dataset,
                "seed": seed,
                "split_role": "val",
                "method": "MSHNet",
                "model_type": "mshnet",
                "anchor_mode": "mean",
                "active_stage": 0,
                "official_test_status": EXPECTED_AUDIT_OFFICIAL_TEST_STATUS,
            }
            errors = [
                key for key, value in expected_header.items() if manifest.get(key) != value
            ]
            if errors:
                raise AnalysisError(
                    f"audit {dataset}/{seed} is not the frozen validation audit: {errors}"
                )
            checkpoint = _mapping(manifest.get("checkpoint"), "audit checkpoint")
            if checkpoint.get("role") != "best_iou":
                raise AnalysisError(f"audit {dataset}/{seed} does not use best_iou checkpoint")
            base_size = _integer(manifest.get("base_size"), "audit.base_size")
            crop_size = _integer(manifest.get("crop_size"), "audit.crop_size")
            if base_size <= 0 or crop_size <= 0 or base_size != crop_size:
                raise AnalysisError("audit image geometry must be positive and square")
            pixels = base_size * crop_size
            artifacts = _mapping(manifest.get("artifacts"), "audit.artifacts")
            audit_dir = manifest_path.parent.resolve()
            images_path = _artifact_path(audit_dir, artifacts.get("images_jsonl"), "images_jsonl")
            component_path = _artifact_path(
                audit_dir, artifacts.get("components_jsonl"), "components_jsonl"
            )
            if _sha256(images_path) != _sha256_value(
                artifacts.get("images_sha256"), "artifacts.images_sha256"
            ):
                raise AnalysisError(f"images ledger hash mismatch for {dataset}/{seed}")
            if _sha256(component_path) != _sha256_value(
                artifacts.get("components_sha256"), "artifacts.components_sha256"
            ):
                raise AnalysisError(f"component ledger hash mismatch for {dataset}/{seed}")
            image_rows = _read_jsonl(images_path, f"images ledger {dataset}/{seed}")
            component_rows = _read_jsonl(component_path, f"components ledger {dataset}/{seed}")
            image_by_id: dict[str, dict[str, Any]] = {}
            for index, row in enumerate(image_rows):
                image_id = row.get("image_id")
                if not isinstance(image_id, str) or not image_id or image_id in image_by_id:
                    raise AnalysisError(f"unsafe or duplicate image ID in {dataset}/{seed}")
                _validate_image_row(
                    row,
                    label=f"{dataset}/{seed}/images[{index}]",
                    pixels=pixels,
                )
                row.update(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "pixels": pixels,
                        "cluster_id": f"{dataset}\x1f{image_id}",
                        "failure_category": classify_failure(row),
                    }
                )
                image_by_id[image_id] = row
            _validate_components(
                component_rows,
                image_by_id,
                label=f"{dataset}/{seed}/components",
            )
            for row in component_rows:
                row.update(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "cluster_id": f"{dataset}\x1f{row['image_id']}",
                    }
                )
            manifest_summary = _mapping(manifest.get("summary"), "audit.summary")
            if _integer(manifest_summary.get("images"), "audit.summary.images") != len(image_rows):
                raise AnalysisError(f"audit summary image count mismatch for {dataset}/{seed}")
            if _integer(manifest_summary.get("pixels"), "audit.summary.pixels") != pixels * len(image_rows):
                raise AnalysisError(f"audit summary pixel count mismatch for {dataset}/{seed}")
            artifact_counts = _mapping(run.get("artifact_counts"), "run.artifact_counts")
            if (
                _integer(artifact_counts.get("image_rows"), "artifact_counts.image_rows")
                != len(image_rows)
                or _integer(
                    artifact_counts.get("component_rows"), "artifact_counts.component_rows"
                )
                != len(component_rows)
            ):
                raise AnalysisError(f"final summary artifact counts disagree for {dataset}/{seed}")
            image_ids_by_pair[(dataset, seed)] = set(image_by_id)
            observations.extend(image_rows)
            components.extend(component_rows)
            manifests.append(manifest)

    for dataset in DATASET_NAMES:
        identities = [image_ids_by_pair[(dataset, seed)] for seed in seeds]
        if any(value != identities[0] for value in identities[1:]):
            raise AnalysisError(
                f"seed audits for {dataset} do not contain the identical validation images"
            )
    return _Evidence(summary, path, observations, components, manifests)


def _cluster_evidence(evidence: _Evidence) -> list[dict[str, Any]]:
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence.observations:
        observations[row["cluster_id"]].append(row)
    for row in evidence.components:
        components[row["cluster_id"]].append(row)
    clusters: list[dict[str, Any]] = []
    for cluster_id in sorted(observations):
        rows = sorted(observations[cluster_id], key=lambda row: row["seed"])
        seeds = [row["seed"] for row in rows]
        if len(seeds) != len(set(seeds)):
            raise AnalysisError(f"duplicate seed observation in cluster {cluster_id!r}")
        sums = {
            field: float(sum(row[field] for row in rows))
            for field in IMAGE_INTEGER_FIELDS
        }
        sums["pixels"] = float(sum(row["pixels"] for row in rows))
        sums["score_sum"] = float(
            sum(row["mean_anchor_index"] * row["pixels"] for row in rows)
        )
        for region in ("true_positive", "false_positive", "false_negative"):
            field = f"mean_anchor_score_sum_{region}"
            sums[field] = float(sum(row[field] for row in rows))
        clusters.append(
            {
                "cluster_id": cluster_id,
                "dataset": rows[0]["dataset"],
                "image_id": rows[0]["image_id"],
                "seed_count": len(rows),
                "observations": rows,
                "components": components.get(cluster_id, []),
                "sums": sums,
            }
        )
    return clusters


class _ClusterBootstrap:
    def __init__(self, clusters: list[dict[str, Any]], *, replicates: int, seed: int):
        if replicates < 1:
            raise ValueError("bootstrap replicates must be positive")
        if not clusters:
            raise AnalysisError("no image clusters available")
        self.clusters = clusters
        self.replicates = replicates
        self.seed = seed
        self.weights = np.zeros((replicates, len(clusters)), dtype=np.int16)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, cluster in enumerate(clusters):
            groups[cluster["dataset"]].append(index)
        rng = np.random.default_rng(seed)
        for dataset in sorted(groups):
            indices = np.asarray(groups[dataset], dtype=int)
            count = len(indices)
            draws = rng.multinomial(count, np.full(count, 1.0 / count), size=replicates)
            self.weights[:, indices] = draws.astype(np.int16)

    def indices(self, dataset: str | None = None) -> np.ndarray:
        if dataset is None:
            return np.arange(len(self.clusters), dtype=int)
        return np.asarray(
            [index for index, cluster in enumerate(self.clusters) if cluster["dataset"] == dataset],
            dtype=int,
        )

    def metadata(self, indices: np.ndarray, defined: int) -> dict[str, Any]:
        return {
            "method": "deterministic_stratified_image_cluster_bootstrap",
            "cluster_unit": "dataset + image_id; all seed observations retained together",
            "stratified_by_dataset": True,
            "image_clusters": int(indices.size),
            "replicates_requested": self.replicates,
            "replicates_defined": int(defined),
            "seed": self.seed,
        }


def _finish_metric(
    point: float | None,
    bootstrap_values: np.ndarray,
    bootstrap: _ClusterBootstrap,
    indices: np.ndarray,
    *,
    undefined_reason: str = "zero_denominator",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = np.asarray(bootstrap_values, dtype=float)
    values = values[np.isfinite(values)]
    defined = point is not None and math.isfinite(float(point))
    ci = None
    if values.size:
        low, high = np.quantile(values, (0.025, 0.975))
        ci = {"low": float(low), "high": float(high)}
    result: dict[str, Any] = {
        "defined": defined,
        "estimate": float(point) if defined else None,
        "undefined_reason": None if defined else undefined_reason,
        "ci95": ci,
        "bootstrap": bootstrap.metadata(indices, int(values.size)),
    }
    if extra:
        result.update(extra)
    return result


def _array(clusters: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([cluster["sums"][field] for cluster in clusters], dtype=float)


def _ratio_metric(
    bootstrap: _ClusterBootstrap,
    numerator: np.ndarray,
    denominator: np.ndarray,
    indices: np.ndarray,
    *,
    scale: float = 1.0,
    undefined_reason: str = "zero_denominator",
) -> dict[str, Any]:
    numerator = np.asarray(numerator, dtype=float)[indices]
    denominator = np.asarray(denominator, dtype=float)[indices]
    point_denominator = float(denominator.sum())
    point = None
    if point_denominator > 0:
        point = float(numerator.sum() / point_denominator * scale)
    weights = bootstrap.weights[:, indices].astype(float)
    boot_numerator = weights @ numerator
    boot_denominator = weights @ denominator
    values = np.full(bootstrap.replicates, np.nan)
    valid = boot_denominator > 0
    values[valid] = boot_numerator[valid] / boot_denominator[valid] * scale
    return _finish_metric(
        point,
        values,
        bootstrap,
        indices,
        undefined_reason=undefined_reason,
    )


def _enrichment_metric(
    bootstrap: _ClusterBootstrap,
    selected_numerator: np.ndarray,
    selected_denominator: np.ndarray,
    reference_numerator: np.ndarray,
    reference_denominator: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    sn = np.asarray(selected_numerator, float)[indices]
    sd = np.asarray(selected_denominator, float)[indices]
    rn = np.asarray(reference_numerator, float)[indices]
    rd = np.asarray(reference_denominator, float)[indices]
    point = None
    if sd.sum() > 0 and rd.sum() > 0 and rn.sum() > 0:
        point = float((sn.sum() / sd.sum()) / (rn.sum() / rd.sum()))
    weights = bootstrap.weights[:, indices].astype(float)
    bsn, bsd = weights @ sn, weights @ sd
    brn, brd = weights @ rn, weights @ rd
    valid = (bsd > 0) & (brd > 0) & (brn > 0)
    values = np.full(bootstrap.replicates, np.nan)
    values[valid] = (bsn[valid] / bsd[valid]) / (brn[valid] / brd[valid])
    return _finish_metric(
        point,
        values,
        bootstrap,
        indices,
        undefined_reason="selected_or_reference_coverage_undefined_or_zero",
    )


def _raw_count(clusters: list[dict[str, Any]], values: np.ndarray, indices: np.ndarray) -> int:
    return int(np.asarray(values, dtype=float)[indices].sum())


def _baseline_metrics(
    clusters: list[dict[str, Any]], bootstrap: _ClusterBootstrap, indices: np.ndarray
) -> dict[str, Any]:
    intersection = _array(clusters, "intersection_pixels")
    union = _array(clusters, "union_pixels")
    tp = _array(clusters, "true_positive_component_count")
    targets = _array(clusters, "target_component_count")
    fp_area = _array(clusters, "false_positive_component_area")
    pixels = _array(clusters, "pixels")
    fn = _array(clusters, "false_negative_component_count")
    recoverable = _array(clusters, "recoverable_fn_component_count")
    return {
        "image_clusters": int(indices.size),
        "seed_image_observations": _raw_count(
            clusters,
            np.asarray([cluster["seed_count"] for cluster in clusters]),
            indices,
        ),
        "global_iou_ratio_of_sums": _ratio_metric(
            bootstrap, intersection, union, indices
        ),
        "pd_component_ratio_of_sums": _ratio_metric(bootstrap, tp, targets, indices),
        "fa_area_per_million_ratio_of_sums": _ratio_metric(
            bootstrap, fp_area, pixels, indices, scale=1e6
        ),
        "fn_target_component_fraction": _ratio_metric(bootstrap, fn, targets, indices),
        "recoverable_fn_fraction": _ratio_metric(
            bootstrap,
            recoverable,
            fn,
            indices,
            undefined_reason="no_false_negative_components",
        ),
    }


def _taxonomy_scope(
    clusters: list[dict[str, Any]], bootstrap: _ClusterBootstrap, indices: np.ndarray
) -> dict[str, Any]:
    seed_counts = np.asarray([cluster["seed_count"] for cluster in clusters], dtype=float)
    categories: dict[str, Any] = {}
    for category in TAXONOMY_ORDER:
        counts = np.asarray(
            [
                sum(row["failure_category"] == category for row in cluster["observations"])
                for cluster in clusters
            ],
            dtype=float,
        )
        categories[category] = {
            "definition": TAXONOMY_DEFINITIONS[category],
            "seed_image_observations": _raw_count(clusters, counts, indices),
            "fraction_of_seed_image_observations": _ratio_metric(
                bootstrap, counts, seed_counts, indices
            ),
            "image_clusters_ever_observed": int(np.sum(counts[indices] > 0)),
            "image_clusters_persistent_across_all_seeds": int(
                np.sum(counts[indices] == seed_counts[indices])
            ),
        }
    stable = 0
    variable = 0
    error_persistence = {"never": 0, "intermittent": 0, "persistent": 0}
    recoverable_persistence = {"never": 0, "intermittent": 0, "persistent": 0}
    for index in indices:
        cluster = clusters[int(index)]
        observed = {row["failure_category"] for row in cluster["observations"]}
        stable += int(len(observed) == 1)
        variable += int(len(observed) > 1)
        component_flags = [
            row["false_positive_component_count"] > 0
            or row["false_negative_component_count"] > 0
            for row in cluster["observations"]
        ]
        recoverable_flags = [
            row["recoverable_fn_component_count"] > 0 for row in cluster["observations"]
        ]
        for flags, ledger in (
            (component_flags, error_persistence),
            (recoverable_flags, recoverable_persistence),
        ):
            if all(flags):
                ledger["persistent"] += 1
            elif any(flags):
                ledger["intermittent"] += 1
            else:
                ledger["never"] += 1
    return {
        "image_clusters": int(indices.size),
        "seed_image_observations": int(seed_counts[indices].sum()),
        "categories": categories,
        "seed_stability": {
            "stable_exclusive_category_image_clusters": stable,
            "seed_variable_exclusive_category_image_clusters": variable,
            "component_error_persistence": error_persistence,
            "recoverable_fn_persistence": recoverable_persistence,
        },
    }


def _component_cluster_arrays(
    clusters: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]
) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {
        "count": [],
        "area": [],
        "conflicted_count": [],
        "conflict_pixels": [],
        "score_sum": [],
        "interaction_ratio_sum": [],
    }
    for cluster in clusters:
        selected = [row for row in cluster["components"] if selector(row)]
        fields["count"].append(float(len(selected)))
        fields["area"].append(float(sum(row["area"] for row in selected)))
        fields["conflicted_count"].append(
            float(sum(row["conflict_pixels"] > 0 for row in selected))
        )
        fields["conflict_pixels"].append(
            float(sum(row["conflict_pixels"] for row in selected))
        )
        fields["score_sum"].append(
            float(sum(row["mean_anchor_score_mean"] * row["area"] for row in selected))
        )
        fields["interaction_ratio_sum"].append(
            float(sum(row["interaction_ratio_mean"] * row["area"] for row in selected))
        )
    return {key: np.asarray(value, dtype=float) for key, value in fields.items()}


COMPONENT_SELECTORS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "tp_target": lambda row: row["domain"] == "target" and row["role"] == "tp_target",
    "fp_prediction": lambda row: row["domain"] == "prediction" and row["role"] == "fp_pred",
    "fn_target": lambda row: row["domain"] == "target" and row["role"] == "fn_target",
    "recoverable_fn_target": lambda row: (
        row["domain"] == "target"
        and row["role"] == "fn_target"
        and row.get("recoverable") is True
    ),
}


def _component_scope(
    clusters: list[dict[str, Any]],
    bootstrap: _ClusterBootstrap,
    indices: np.ndarray,
) -> dict[str, Any]:
    observation_count = np.asarray([cluster["seed_count"] for cluster in clusters], float)
    pixels = _array(clusters, "pixels")
    result: dict[str, Any] = {}
    for name, selector in COMPONENT_SELECTORS.items():
        arrays = _component_cluster_arrays(clusters, selector)
        selected_rows = [
            row
            for index in indices
            for row in clusters[int(index)]["components"]
            if selector(row)
        ]
        areas = np.asarray([row["area"] for row in selected_rows], dtype=float)
        result[name] = {
            "component_observations_across_seeds": int(arrays["count"][indices].sum()),
            "image_clusters_with_component": int(np.sum(arrays["count"][indices] > 0)),
            "area_pixels_across_seed_observations": int(arrays["area"][indices].sum()),
            "area_distribution_descriptive_only": {
                "median": float(np.median(areas)) if areas.size else None,
                "p95": float(np.percentile(areas, 95)) if areas.size else None,
                "repeated_seed_components_are_not_independent": True,
            },
            "components_per_seed_image_observation": _ratio_metric(
                bootstrap, arrays["count"], observation_count, indices
            ),
            "component_area_per_million_image_pixels": _ratio_metric(
                bootstrap, arrays["area"], pixels, indices, scale=1e6
            ),
            "component_conflict_incidence": _ratio_metric(
                bootstrap,
                arrays["conflicted_count"],
                arrays["count"],
                indices,
                undefined_reason="no_components_of_this_role",
            ),
            "component_conflict_pixel_coverage": _ratio_metric(
                bootstrap,
                arrays["conflict_pixels"],
                arrays["area"],
                indices,
                undefined_reason="no_component_area_of_this_role",
            ),
            "area_weighted_mean_anchor_score": _ratio_metric(
                bootstrap,
                arrays["score_sum"],
                arrays["area"],
                indices,
                undefined_reason="no_component_area_of_this_role",
            ),
            "area_weighted_interaction_ratio": _ratio_metric(
                bootstrap,
                arrays["interaction_ratio_sum"],
                arrays["area"],
                indices,
                undefined_reason="no_component_area_of_this_role",
            ),
        }
    return result


def _conflict_scope(
    clusters: list[dict[str, Any]], bootstrap: _ClusterBootstrap, indices: np.ndarray
) -> dict[str, Any]:
    pixels = _array(clusters, "pixels")
    tp = _array(clusters, "intersection_pixels")
    fp = _array(clusters, "false_positive_pixels")
    fn = _array(clusters, "false_negative_pixels")
    error = fp + fn
    correct = pixels - error
    tn = correct - tp
    conflict = _array(clusters, "conflict_pixels")
    conflict_tp = _array(clusters, "conflict_on_true_positive_pixels")
    conflict_fp = _array(clusters, "conflict_on_false_positive_pixels")
    conflict_fn = _array(clusters, "conflict_on_false_negative_pixels")
    conflict_error = conflict_fp + conflict_fn
    conflict_correct = conflict - conflict_error
    conflict_tn = conflict_correct - conflict_tp
    if np.any(tn < 0) or np.any(conflict_tn < -1e-9):
        raise AnalysisError("derived true-negative/conflict counts are negative")
    conflict_tn = np.maximum(conflict_tn, 0)
    localization = {
        "all_pixels": _ratio_metric(bootstrap, conflict, pixels, indices),
        "error_pixels_fp_plus_fn": _ratio_metric(
            bootstrap, conflict_error, error, indices, undefined_reason="no_error_pixels"
        ),
        "correct_pixels_tp_plus_tn": _ratio_metric(
            bootstrap, conflict_correct, correct, indices
        ),
        "true_positive_pixels": _ratio_metric(
            bootstrap, conflict_tp, tp, indices, undefined_reason="no_true_positive_pixels"
        ),
        "false_positive_pixels": _ratio_metric(
            bootstrap, conflict_fp, fp, indices, undefined_reason="no_false_positive_pixels"
        ),
        "false_negative_pixels": _ratio_metric(
            bootstrap, conflict_fn, fn, indices, undefined_reason="no_false_negative_pixels"
        ),
        "true_negative_pixels": _ratio_metric(
            bootstrap, conflict_tn, tn, indices, undefined_reason="no_true_negative_pixels"
        ),
        "fraction_of_all_conflict_localized_to_error_pixels": _ratio_metric(
            bootstrap, conflict_error, conflict, indices, undefined_reason="no_conflict_pixels"
        ),
    }
    enrichment = {
        "error_vs_correct_conflict_coverage": _enrichment_metric(
            bootstrap, conflict_error, error, conflict_correct, correct, indices
        ),
        "fp_vs_correct_conflict_coverage": _enrichment_metric(
            bootstrap, conflict_fp, fp, conflict_correct, correct, indices
        ),
        "fn_vs_correct_conflict_coverage": _enrichment_metric(
            bootstrap, conflict_fn, fn, conflict_correct, correct, indices
        ),
    }
    component_arrays = {
        name: _component_cluster_arrays(clusters, selector)
        for name, selector in COMPONENT_SELECTORS.items()
    }
    tp_components = component_arrays["tp_target"]
    for name in ("fp_prediction", "fn_target", "recoverable_fn_target"):
        selected = component_arrays[name]
        enrichment[f"{name}_vs_tp_component_conflict_coverage"] = _enrichment_metric(
            bootstrap,
            selected["conflict_pixels"],
            selected["area"],
            tp_components["conflict_pixels"],
            tp_components["area"],
            indices,
        )
    return {"pixel_localization": localization, "enrichment": enrichment}


def _predictive_metric(
    bootstrap: _ClusterBootstrap,
    scores: np.ndarray,
    outcomes: np.ndarray,
    indices: np.ndarray,
    *,
    metric: str,
) -> dict[str, Any]:
    point_result = (
        auroc(scores[indices], outcomes[indices])
        if metric == "auroc"
        else spearman_correlation(scores[indices], outcomes[indices])
    )
    values: list[float] = []
    weights = bootstrap.weights[:, indices]
    for bootstrap_weights in weights:
        repeated = np.repeat(np.arange(indices.size), bootstrap_weights.astype(int))
        if repeated.size < 2:
            continue
        sampled_scores = scores[indices][repeated]
        sampled_outcomes = outcomes[indices][repeated]
        result = (
            auroc(sampled_scores, sampled_outcomes)
            if metric == "auroc"
            else spearman_correlation(sampled_scores, sampled_outcomes)
        )
        if result["defined"]:
            values.append(float(result["estimate"]))
    result = _finish_metric(
        float(point_result["estimate"]) if point_result["defined"] else None,
        np.asarray(values, dtype=float),
        bootstrap,
        indices,
        undefined_reason=str(point_result.get("undefined_reason")),
        extra={
            key: value
            for key, value in point_result.items()
            if key not in {"defined", "estimate", "undefined_reason"}
        },
    )
    return result


def _risk_scope(
    clusters: list[dict[str, Any]], bootstrap: _ClusterBootstrap, indices: np.ndarray
) -> dict[str, Any]:
    scores = np.asarray(
        [cluster["sums"]["score_sum"] / cluster["sums"]["pixels"] for cluster in clusters],
        dtype=float,
    )
    iou_loss = np.asarray(
        [
            1.0 - cluster["sums"]["intersection_pixels"] / cluster["sums"]["union_pixels"]
            if cluster["sums"]["union_pixels"] > 0
            else 0.0
            for cluster in clusters
        ],
        dtype=float,
    )
    majority_component_error = np.asarray(
        [
            2
            * sum(
                row["false_positive_component_count"] > 0
                or row["false_negative_component_count"] > 0
                for row in cluster["observations"]
            )
            > cluster["seed_count"]
            for cluster in clusters
        ],
        dtype=float,
    )
    majority_pixel_error = np.asarray(
        [
            2
            * sum(
                row["false_positive_pixels"] > 0 or row["false_negative_pixels"] > 0
                for row in cluster["observations"]
            )
            > cluster["seed_count"]
            for cluster in clusters
        ],
        dtype=float,
    )
    return {
        "image_clusters": int(indices.size),
        "predictor": {
            "name": "mean_anchor_index",
            "uses_ground_truth": False,
            "cluster_aggregation": "sum(conflict score across seeds) / sum(image pixels across seeds)",
        },
        "outcomes": {
            "use_ground_truth_for_evaluation": True,
            "primary_binary": (
                "strict-majority-seed component error: at least one unmatched FP or FN component"
            ),
            "secondary_binary": "strict-majority-seed nonzero FP or FN pixels",
            "continuous": "1 - pooled per-image-across-seeds IoU (ratio of sums)",
        },
        "auroc_primary_majority_component_error": _predictive_metric(
            bootstrap,
            scores,
            majority_component_error,
            indices,
            metric="auroc",
        ),
        "auroc_secondary_majority_pixel_error": _predictive_metric(
            bootstrap, scores, majority_pixel_error, indices, metric="auroc"
        ),
        "spearman_with_pooled_iou_loss": _predictive_metric(
            bootstrap, scores, iou_loss, indices, metric="spearman"
        ),
        "interpretation": (
            "baseline-risk association only; this analysis contains no post-treatment "
            "DEA outcome and therefore cannot estimate or claim future DEA benefit prediction"
        ),
    }


def _ci_lower(metric: Mapping[str, Any]) -> float | None:
    ci = metric.get("ci95")
    if not isinstance(ci, dict):
        return None
    value = ci.get("low")
    return float(value) if isinstance(value, (int, float)) else None


def _evidence_gate(
    analysis_by_dataset: dict[str, dict[str, Any]],
    overall_conflict: dict[str, Any],
    overall_risk: dict[str, Any],
    clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    component_error_datasets = 0
    recoverable_datasets = 0
    conflict_positive_datasets = 0
    risk_direction_datasets = 0
    for dataset in DATASET_NAMES:
        dataset_clusters = [cluster for cluster in clusters if cluster["dataset"] == dataset]
        if sum(
            cluster["sums"]["false_positive_component_count"]
            + cluster["sums"]["false_negative_component_count"]
            for cluster in dataset_clusters
        ) > 0:
            component_error_datasets += 1
        if sum(
            cluster["sums"]["recoverable_fn_component_count"] for cluster in dataset_clusters
        ) > 0:
            recoverable_datasets += 1
        enrichment = analysis_by_dataset[dataset]["conflict"]["enrichment"][
            "fp_prediction_vs_tp_component_conflict_coverage"
        ]
        if enrichment["defined"] and enrichment["estimate"] > 1.0:
            conflict_positive_datasets += 1
        risk = analysis_by_dataset[dataset]["baseline_risk_predictor"]
        auc = risk["auroc_primary_majority_component_error"]
        rho = risk["spearman_with_pooled_iou_loss"]
        if (
            auc["defined"]
            and rho["defined"]
            and auc["estimate"] >= 0.5
            and rho["estimate"] >= 0.0
        ):
            risk_direction_datasets += 1
    overall_enrichment = overall_conflict["enrichment"][
        "fp_prediction_vs_tp_component_conflict_coverage"
    ]
    overall_auc = overall_risk["auroc_primary_majority_component_error"]
    overall_rho = overall_risk["spearman_with_pooled_iou_loss"]
    criteria = [
        {
            "id": "component_errors_cross_dataset",
            "rule": "at least one unmatched FP/FN component is observed in every dataset",
            "observed_dataset_count": component_error_datasets,
            "required_dataset_count": len(DATASET_NAMES),
            "pass": component_error_datasets == len(DATASET_NAMES),
        },
        {
            "id": "recoverable_fn_cross_dataset",
            "rule": "prediction-only recoverable FN is observed in at least two datasets",
            "observed_dataset_count": recoverable_datasets,
            "required_dataset_count": 2,
            "pass": recoverable_datasets >= 2,
        },
        {
            "id": "fp_component_conflict_enrichment_overall_ci",
            "rule": "FP-vs-TP component conflict-coverage enrichment 95% CI lower bound > 1",
            "observed_ci_lower": _ci_lower(overall_enrichment),
            "pass": (
                _ci_lower(overall_enrichment) is not None
                and _ci_lower(overall_enrichment) > 1.0
            ),
        },
        {
            "id": "fp_component_conflict_enrichment_cross_dataset_direction",
            "rule": "FP-vs-TP component conflict enrichment point estimate > 1 in all datasets",
            "observed_dataset_count": conflict_positive_datasets,
            "required_dataset_count": len(DATASET_NAMES),
            "pass": conflict_positive_datasets == len(DATASET_NAMES),
        },
        {
            "id": "mean_anchor_baseline_risk_auroc",
            "rule": "primary baseline-risk AUROC 95% CI lower bound > 0.5",
            "observed_ci_lower": _ci_lower(overall_auc),
            "pass": _ci_lower(overall_auc) is not None and _ci_lower(overall_auc) > 0.5,
        },
        {
            "id": "mean_anchor_baseline_risk_spearman",
            "rule": "IoU-loss Spearman 95% CI lower bound > 0",
            "observed_ci_lower": _ci_lower(overall_rho),
            "pass": _ci_lower(overall_rho) is not None and _ci_lower(overall_rho) > 0.0,
        },
        {
            "id": "mean_anchor_cross_dataset_direction",
            "rule": "AUROC >= 0.5 and Spearman >= 0 in at least two evaluable datasets",
            "observed_dataset_count": risk_direction_datasets,
            "required_dataset_count": 2,
            "pass": risk_direction_datasets >= 2,
        },
    ]
    passed = all(item["pass"] for item in criteria)
    return {
        "name": "conservative_baseline_decoder_interaction_problem_evidence_gate",
        "status": (
            "BASELINE_PROBLEM_EVIDENCE_PASS"
            if passed
            else "NO_GO_FOR_DECODER_INTERACTION_ROOT_CAUSE"
        ),
        "pass": passed,
        "no_go": not passed,
        "criteria": criteria,
        "no_go_reasons": [item["id"] for item in criteria if not item["pass"]],
        "scope": (
            "qualifies or rejects one baseline problem hypothesis only; it is not a DEA "
            "architecture, performance, causality, or treatment-benefit gate"
        ),
    }


def analyze_failures(
    summary_path: str | Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return a JSON-serializable baseline failure analysis."""
    evidence = load_validated_evidence(summary_path)
    clusters = _cluster_evidence(evidence)
    bootstrap = _ClusterBootstrap(
        clusters, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    overall_indices = bootstrap.indices()
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_NAMES:
        indices = bootstrap.indices(dataset)
        by_dataset[dataset] = {
            "baseline_metrics": _baseline_metrics(clusters, bootstrap, indices),
            "failure_taxonomy": _taxonomy_scope(clusters, bootstrap, indices),
            "component_statistics": _component_scope(clusters, bootstrap, indices),
            "conflict": _conflict_scope(clusters, bootstrap, indices),
            "baseline_risk_predictor": _risk_scope(clusters, bootstrap, indices),
        }
    overall = {
        "baseline_metrics": _baseline_metrics(clusters, bootstrap, overall_indices),
        "failure_taxonomy": _taxonomy_scope(clusters, bootstrap, overall_indices),
        "component_statistics": _component_scope(clusters, bootstrap, overall_indices),
        "conflict": _conflict_scope(clusters, bootstrap, overall_indices),
        "baseline_risk_predictor": _risk_scope(clusters, bootstrap, overall_indices),
    }
    gate = _evidence_gate(
        by_dataset,
        overall["conflict"],
        overall["baseline_risk_predictor"],
        clusters,
    )
    return {
        "schema_version": OUTPUT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete_from_validated_development_holdout_evidence",
        "scope": EXPECTED_EVALUATION_SCOPE,
        "official_test_status": "sealed; not read by this analyzer",
        "not_for_official_test_or_main_table_claims": True,
        "input": {
            "summary": str(evidence.summary_path),
            "summary_sha256": _sha256(evidence.summary_path),
            "batch_id": evidence.summary.get("batch_id"),
            "validated_audit_runs": len(evidence.manifests),
            "seed_image_observations": len(evidence.observations),
            "component_observations": len(evidence.components),
            "unique_dataset_image_clusters": len(clusters),
            "datasets": list(DATASET_NAMES),
            "seeds": evidence.summary["seeds"],
        },
        "analysis_protocol": {
            "global_iou": "ratio of sums: sum(intersection_pixels) / sum(union_pixels)",
            "inferential_unit": "dataset + image_id cluster",
            "repeated_seeds": "retained together; never treated as independent images",
            "bootstrap": {
                "method": "deterministic stratified image-cluster bootstrap",
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "confidence_level": 0.95,
            },
            "risk_predictor_scope": "MSHNet baseline risk only",
        },
        "failure_taxonomy_definitions": TAXONOMY_DEFINITIONS,
        "overall": overall,
        "by_dataset": by_dataset,
        "evidence_decision": {
            "baseline_problem_gate": gate,
            "dea_model_or_gain_gate": {
                "status": "NOT_EVALUATED_BASELINE_ONLY",
                "pass": False,
                "no_go": None,
                "dea_evaluated": False,
                "dea_gain_estimated": False,
                "future_benefit_prediction_evaluated": False,
                "reason": (
                    "no paired DEA/control outcomes are present; baseline-risk association "
                    "cannot be relabeled as future treatment-benefit prediction"
                ),
            },
        },
        "interpretation_boundary": {
            "descriptive_baseline_evidence_only": True,
            "does_not_establish_error_causation": True,
            "does_not_establish_dea_benefit": True,
            "does_not_establish_future_benefit_prediction": True,
            "does_not_establish_structural_specificity": True,
            "requires_later_paired_dea_and_parameter_matched_control_evidence": True,
        },
    }


def _metric_text(metric: Mapping[str, Any], digits: int = 4) -> str:
    if not metric.get("defined"):
        return f"NA ({metric.get('undefined_reason', 'undefined')})"
    estimate = float(metric["estimate"])
    ci = metric.get("ci95")
    if isinstance(ci, dict):
        return f"{estimate:.{digits}f} [{float(ci['low']):.{digits}f}, {float(ci['high']):.{digits}f}]"
    return f"{estimate:.{digits}f} [CI undefined]"


def build_markdown(analysis: Mapping[str, Any]) -> str:
    overall = analysis["overall"]
    lines = [
        "# Clean MSHNet baseline failure analysis",
        "",
        "> **Scope guard:** validated internal development holdouts only. This report "
        "contains no official-test result, no DEA evaluation or gain, no causal claim, "
        "and no future-benefit prediction claim.",
        "",
        f"- Dataset/image clusters: {analysis['input']['unique_dataset_image_clusters']}",
        f"- Seed-image observations: {analysis['input']['seed_image_observations']}",
        f"- Bootstrap: {analysis['analysis_protocol']['bootstrap']['replicates']} deterministic "
        f"replicates, seed {analysis['analysis_protocol']['bootstrap']['seed']}; stratified "
        "by dataset; every image's seed observations move together.",
        "- Global IoU is the ratio of summed intersections to summed unions, never the "
        "mean of per-image IoUs.",
        "",
        "## Baseline metrics",
        "",
        "| Scope | Global IoU | PD | FA/M | FN fraction | Recoverable-FN fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    scopes = [("Overall", overall)] + [
        (dataset, analysis["by_dataset"][dataset]) for dataset in DATASET_NAMES
    ]
    for label, scope in scopes:
        metrics = scope["baseline_metrics"]
        lines.append(
            f"| {label} | {_metric_text(metrics['global_iou_ratio_of_sums'])} | "
            f"{_metric_text(metrics['pd_component_ratio_of_sums'])} | "
            f"{_metric_text(metrics['fa_area_per_million_ratio_of_sums'], 2)} | "
            f"{_metric_text(metrics['fn_target_component_fraction'])} | "
            f"{_metric_text(metrics['recoverable_fn_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Cross-seed failure taxonomy",
            "",
            "Counts below are seed-image observations; inferential intervals use image clusters.",
            "",
            "| Category | Count | Fraction (95% clustered CI) | Ever-observed image clusters | Persistent clusters |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category in TAXONOMY_ORDER:
        item = overall["failure_taxonomy"]["categories"][category]
        lines.append(
            f"| {category} | {item['seed_image_observations']} | "
            f"{_metric_text(item['fraction_of_seed_image_observations'])} | "
            f"{item['image_clusters_ever_observed']} | "
            f"{item['image_clusters_persistent_across_all_seeds']} |"
        )
    lines.extend(
        [
            "",
            "## Component evidence",
            "",
            "| Role | Component observations | Image clusters | Area px | Conflict incidence | Conflict coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for role in COMPONENT_SELECTORS:
        item = overall["component_statistics"][role]
        lines.append(
            f"| {role} | {item['component_observations_across_seeds']} | "
            f"{item['image_clusters_with_component']} | "
            f"{item['area_pixels_across_seed_observations']} | "
            f"{_metric_text(item['component_conflict_incidence'])} | "
            f"{_metric_text(item['component_conflict_pixel_coverage'])} |"
        )
    localization = overall["conflict"]["pixel_localization"]
    enrichment = overall["conflict"]["enrichment"]
    lines.extend(
        [
            "",
            "## Conflict localization and enrichment",
            "",
            f"- Conflict coverage on error pixels: {_metric_text(localization['error_pixels_fp_plus_fn'])}.",
            f"- Conflict coverage on correct pixels: {_metric_text(localization['correct_pixels_tp_plus_tn'])}.",
            f"- Error/correct coverage enrichment: {_metric_text(enrichment['error_vs_correct_conflict_coverage'])}.",
            f"- **Primary gate, FP/TP component coverage enrichment:** "
            f"{_metric_text(enrichment['fp_prediction_vs_tp_component_conflict_coverage'])}.",
            f"- FP/correct coverage enrichment: {_metric_text(enrichment['fp_vs_correct_conflict_coverage'])}.",
            f"- FN/correct coverage enrichment: {_metric_text(enrichment['fn_vs_correct_conflict_coverage'])}.",
            "",
            "## Mean-anchor index as a baseline-risk predictor",
            "",
        ]
    )
    risk = overall["baseline_risk_predictor"]
    lines.extend(
        [
            f"- AUROC for strict-majority-seed component error: "
            f"{_metric_text(risk['auroc_primary_majority_component_error'])}.",
            f"- Spearman with pooled per-image IoU loss: "
            f"{_metric_text(risk['spearman_with_pooled_iou_loss'])}.",
            "- The predictor itself does not read GT; GT is used only to define baseline-risk "
            "outcomes for evaluation.",
            "- This is not a future DEA-benefit analysis: no post-treatment outcome exists in "
            "the input.",
            "",
            "## Conservative evidence decision",
            "",
        ]
    )
    gate = analysis["evidence_decision"]["baseline_problem_gate"]
    lines.append(f"**{gate['status']}**")
    lines.extend(
        [
            "",
            "| Criterion | Result |",
            "|---|---:|",
        ]
    )
    for criterion in gate["criteria"]:
        lines.append(
            f"| {criterion['id']} | {'PASS' if criterion['pass'] else 'FAIL / undefined'} |"
        )
    lines.extend(
        [
            "",
            "Regardless of this baseline-problem gate, the DEA model/gain gate remains "
            "**NOT_EVALUATED_BASELINE_ONLY**. Paired DEA and parameter-matched residual, "
            "attention, and final-fusion controls are required before any mechanism or "
            "performance claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    analysis: Mapping[str, Any], output_dir: str | Path, *, force: bool = False
) -> tuple[Path, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = directory / OUTPUT_JSON, directory / OUTPUT_MARKDOWN
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing failure analysis: "
            + ", ".join(str(path) for path in existing)
        )
    json_text = json.dumps(analysis, indent=2, allow_nan=False) + "\n"
    markdown_text = build_markdown(analysis)
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
    return json_path, markdown_path


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else summary_path.parent
    )
    analysis = analyze_failures(
        summary_path,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    json_path, markdown_path = write_outputs(analysis, output_dir, force=args.force)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(
        "scope: validated MSHNet development-holdout baseline risk only; "
        "official test sealed; DEA not evaluated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
