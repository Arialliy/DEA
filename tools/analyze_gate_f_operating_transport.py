#!/usr/bin/env python3
"""Read-only Gate F decomposition of cross-fitted component operating points.

This tool consumes the immutable Gate E-1c bundle.  It never recomputes model
predictions and never changes the formal E-1c decision.  Its outputs are an
exploratory diagnosis of calibration-to-held-out transport and image-level
unmatched-area concentration.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_CALIBRATION_SCHEMA = "dea.gate_e.low_fa_calibration.v1"
INPUT_IMAGE_SCHEMA = "dea.gate_e.low_fa_image.v1"
INPUT_PROVENANCE_SCHEMA = "dea.gate_e.low_fa_bridge_provenance.v1"
FOLD_ROW_SCHEMA = "dea.gate_f.operating_transport_fold.v1"
PAIR_ROW_SCHEMA = "dea.gate_f.operating_transport_pair.v1"
SUMMARY_SCHEMA = "dea.gate_f.operating_transport_summary.v1"
PROVENANCE_SCHEMA = "dea.gate_f.operating_transport_provenance.v1"
COUNT_FIELDS = (
    "total_pixels",
    "target_components",
    "matched_components",
    "prediction_components",
    "unmatched_prediction_components",
    "unmatched_prediction_area",
)
TOP_K_VALUES = (1, 3, 5)
OUTPUT_FILES = (
    "fold_transport.jsonl",
    "pair_transport.jsonl",
    "operating_transport_summary.json",
    "operating_transport_summary.md",
    "provenance.json",
)


class GateFTransportError(RuntimeError):
    """Raised when an input or derived Gate F transport invariant is invalid."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose immutable Gate E-1c calibration-to-held-out component "
            "operating-point transport without rerunning inference."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="repro_runs/gate_e/persistence_v2/low_fa_bridge",
    )
    parser.add_argument(
        "--output-dir",
        default="repro_runs/gate_f/operating_transport_v1",
    )
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateFTransportError(f"cannot read {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise GateFTransportError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateFTransportError(f"cannot read {path}: {exc}") from exc
    return rows


def _as_nonnegative_int(value: Any, *, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise GateFTransportError(f"{label} must be a non-negative integer")
    return value


def _as_positive_int(value: Any, *, label: str) -> int:
    result = _as_nonnegative_int(value, label=label)
    if result == 0:
        raise GateFTransportError(f"{label} must be positive")
    return result


def _as_finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateFTransportError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GateFTransportError(f"{label} must be finite")
    return result


def _fa_per_mpix(area: int, total_pixels: int) -> float:
    return float(area) * 1_000_000.0 / float(total_pixels)


def _pd(matched: int, targets: int) -> float:
    if targets <= 0:
        raise GateFTransportError("target component denominator must be positive")
    return float(matched) / float(targets)


def _budget_feasible(area: int, total_pixels: int, budget: int) -> bool:
    return int(area) * 1_000_000 <= int(budget) * int(total_pixels)


def _metric_counts(value: Mapping[str, Any], *, label: str) -> dict[str, int]:
    result = {
        field: _as_nonnegative_int(value.get(field), label=f"{label}.{field}")
        for field in COUNT_FIELDS
    }
    if result["total_pixels"] <= 0:
        raise GateFTransportError(f"{label}.total_pixels must be positive")
    if result["matched_components"] > result["target_components"]:
        raise GateFTransportError(f"{label} has more matches than targets")
    if result["matched_components"] > result["prediction_components"]:
        raise GateFTransportError(f"{label} has more matches than predictions")
    if result["unmatched_prediction_components"] > result["prediction_components"]:
        raise GateFTransportError(f"{label} has invalid unmatched predictions")
    if result["unmatched_prediction_area"] > result["total_pixels"]:
        raise GateFTransportError(f"{label} has unmatched area beyond its pixels")
    return result


def _sum_image_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    if not rows:
        raise GateFTransportError("cannot aggregate an empty image group")
    result = {field: 0 for field in COUNT_FIELDS}
    for index, row in enumerate(rows):
        counts = _metric_counts(row, label=f"image_rows[{index}]")
        for field in COUNT_FIELDS:
            result[field] += counts[field]
    return result


def _metric_payload(counts: Mapping[str, int], *, budget: int) -> dict[str, Any]:
    total_pixels = int(counts["total_pixels"])
    targets = int(counts["target_components"])
    matched = int(counts["matched_components"])
    area = int(counts["unmatched_prediction_area"])
    if targets <= 0:
        raise GateFTransportError("formal transport group lacks target components")
    feasible = _budget_feasible(area, total_pixels, budget)
    integer_margin = int(budget) * total_pixels - area * 1_000_000
    return {
        **{field: int(counts[field]) for field in COUNT_FIELDS},
        "fa_per_mpix": _fa_per_mpix(area, total_pixels),
        "pd": _pd(matched, targets),
        "budget_feasible_zero_overshoot": feasible,
        "budget_integer_margin": integer_margin,
        "budget_overshoot_fa_per_mpix": max(
            0.0, _fa_per_mpix(area, total_pixels) - float(budget)
        ),
    }


def _assert_close(observed: Any, expected: float, *, label: str) -> None:
    value = _as_finite_float(observed, label=label)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise GateFTransportError(
            f"{label} mismatch: observed={value}, recomputed={expected}"
        )


def _validate_stored_aggregate(
    stored: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for field in COUNT_FIELDS:
        observed = _as_nonnegative_int(stored.get(field), label=f"{label}.{field}")
        if observed != expected[field]:
            raise GateFTransportError(
                f"{label}.{field} mismatch: {observed} != {expected[field]}"
            )
    _assert_close(
        stored.get("achieved_fa_per_mpix"),
        float(expected["fa_per_mpix"]),
        label=f"{label}.achieved_fa_per_mpix",
    )
    _assert_close(
        stored.get("achieved_pd"),
        float(expected["pd"]),
        label=f"{label}.achieved_pd",
    )
    observed_feasible = stored.get("budget_feasible_zero_overshoot")
    if not isinstance(observed_feasible, bool):
        raise GateFTransportError(f"{label}.budget_feasible must be boolean")
    if observed_feasible != expected["budget_feasible_zero_overshoot"]:
        raise GateFTransportError(f"{label}.budget feasibility mismatch")


def _validate_calibration_selection(
    selection: Mapping[str, Any],
    *,
    budget: int,
    image_count: int,
    label: str,
) -> dict[str, Any]:
    sample_count = _as_positive_int(
        selection.get("sample_count"), label=f"{label}.sample_count"
    )
    if sample_count != image_count:
        raise GateFTransportError(f"{label}.sample_count disagrees with image names")
    counts = {
        "total_pixels": _as_positive_int(
            selection.get("total_pixels"), label=f"{label}.total_pixels"
        ),
        "target_components": _as_positive_int(
            selection.get("target_components"),
            label=f"{label}.target_components",
        ),
        "matched_components": _as_nonnegative_int(
            selection.get("matched_components"),
            label=f"{label}.matched_components",
        ),
        "prediction_components": _as_nonnegative_int(
            selection.get("prediction_components"),
            label=f"{label}.prediction_components",
        ),
        "unmatched_prediction_components": _as_nonnegative_int(
            selection.get("unmatched_prediction_components"),
            label=f"{label}.unmatched_prediction_components",
        ),
        "unmatched_prediction_area": _as_nonnegative_int(
            selection.get("unmatched_prediction_area"),
            label=f"{label}.unmatched_prediction_area",
        ),
    }
    payload = _metric_payload(counts, budget=budget)
    _assert_close(
        selection.get("fa_per_million_pixels"),
        payload["fa_per_mpix"],
        label=f"{label}.fa_per_million_pixels",
    )
    _assert_close(selection.get("pd"), payload["pd"], label=f"{label}.pd")
    if not payload["budget_feasible_zero_overshoot"]:
        raise GateFTransportError(f"{label} was not calibration-budget feasible")
    threshold = _as_finite_float(selection.get("threshold"), label=f"{label}.threshold")
    return {
        **payload,
        "sample_count": sample_count,
        "threshold": threshold,
        "all_off": counts["prediction_components"] == 0,
        "no_true_positive": counts["matched_components"] == 0,
    }


def _validate_calibration_curve(
    record: Mapping[str, Any],
    *,
    budgets: Sequence[int],
    label: str,
) -> None:
    curve = record.get("curve")
    threshold_grid = record.get("threshold_grid")
    selections = record.get("selections")
    if (
        not isinstance(curve, list)
        or not curve
        or not isinstance(threshold_grid, list)
        or len(curve) != len(threshold_grid)
        or not isinstance(selections, Mapping)
    ):
        raise GateFTransportError(f"{label} curve/grid is invalid")
    thresholds = [
        _as_finite_float(value, label=f"{label}.threshold_grid")
        for value in threshold_grid
    ]
    if thresholds != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
        raise GateFTransportError(f"{label} threshold grid is not strictly ordered")
    population: tuple[int, int, int] | None = None
    validated_curve: list[Mapping[str, Any]] = []
    for index, (point, threshold) in enumerate(zip(curve, thresholds)):
        if not isinstance(point, Mapping):
            raise GateFTransportError(f"{label}.curve[{index}] is not an object")
        point_threshold = _as_finite_float(
            point.get("threshold"), label=f"{label}.curve[{index}].threshold"
        )
        if point_threshold != threshold:
            raise GateFTransportError(f"{label} curve/grid threshold mismatch")
        sample_count = _as_positive_int(
            point.get("sample_count"), label=f"{label}.curve[{index}].sample_count"
        )
        total_pixels = _as_positive_int(
            point.get("total_pixels"), label=f"{label}.curve[{index}].total_pixels"
        )
        targets = _as_positive_int(
            point.get("target_components"),
            label=f"{label}.curve[{index}].target_components",
        )
        matched = _as_nonnegative_int(
            point.get("matched_components"),
            label=f"{label}.curve[{index}].matched_components",
        )
        predictions = _as_nonnegative_int(
            point.get("prediction_components"),
            label=f"{label}.curve[{index}].prediction_components",
        )
        unmatched_predictions = _as_nonnegative_int(
            point.get("unmatched_prediction_components"),
            label=f"{label}.curve[{index}].unmatched_prediction_components",
        )
        unmatched_targets = _as_nonnegative_int(
            point.get("unmatched_target_components"),
            label=f"{label}.curve[{index}].unmatched_target_components",
        )
        unmatched_area = _as_nonnegative_int(
            point.get("unmatched_prediction_area"),
            label=f"{label}.curve[{index}].unmatched_prediction_area",
        )
        if matched + unmatched_targets != targets:
            raise GateFTransportError(f"{label} target component identity failed")
        if matched + unmatched_predictions != predictions:
            raise GateFTransportError(f"{label} prediction component identity failed")
        if unmatched_area > total_pixels:
            raise GateFTransportError(f"{label} unmatched area exceeds pixels")
        point_population = (sample_count, total_pixels, targets)
        if population is None:
            population = point_population
        elif point_population != population:
            raise GateFTransportError(f"{label} curve population is not constant")
        _assert_close(
            point.get("fa_per_million_pixels"),
            _fa_per_mpix(unmatched_area, total_pixels),
            label=f"{label}.curve[{index}].fa_per_million_pixels",
        )
        _assert_close(
            point.get("pd"),
            _pd(matched, targets),
            label=f"{label}.curve[{index}].pd",
        )
        validated_curve.append(point)
    for budget in budgets:
        selection = selections.get(str(budget))
        if not isinstance(selection, Mapping):
            raise GateFTransportError(f"{label} lacks selection for budget {budget}")
        matching_points = [point for point in validated_curve if dict(point) == dict(selection)]
        if len(matching_points) != 1:
            raise GateFTransportError(
                f"{label} selection for budget {budget} is not one curve point"
            )
        feasible = [
            point
            for point in validated_curve
            if _budget_feasible(
                int(point["unmatched_prediction_area"]),
                int(point["total_pixels"]),
                budget,
            )
        ]
        if not feasible:
            raise GateFTransportError(f"{label} has no feasible point for budget {budget}")
        expected = max(
            feasible,
            key=lambda point: (
                int(point["matched_components"]),
                -int(point["unmatched_prediction_area"]),
                float(point["threshold"]),
            ),
        )
        if dict(selection) != dict(expected):
            raise GateFTransportError(
                f"{label} selection for budget {budget} violates frozen tie-break"
            )


def _curve_structure(record: Mapping[str, Any]) -> dict[str, Any]:
    curve = record["curve"]
    areas = [int(point["unmatched_prediction_area"]) for point in curve]
    increases = [
        {
            "lower_threshold": float(curve[index - 1]["threshold"]),
            "higher_threshold": float(curve[index]["threshold"]),
            "area_before": areas[index - 1],
            "area_after": areas[index],
            "absolute_increase": areas[index] - areas[index - 1],
        }
        for index in range(1, len(areas))
        if areas[index] > areas[index - 1]
    ]
    return {
        "unmatched_area_nonmonotone_in_threshold": bool(increases),
        "unmatched_area_increasing_step_count": len(increases),
        "maximum_single_step_unmatched_area_increase": max(
            (value["absolute_increase"] for value in increases), default=0
        ),
        "largest_increasing_steps": sorted(
            increases,
            key=lambda value: (
                -int(value["absolute_increase"]),
                float(value["lower_threshold"]),
            ),
        )[:5],
    }


def _concentration(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> dict[str, Any]:
    totals = _sum_image_counts(rows)
    total_area = totals["unmatched_prediction_area"]
    total_pixels = totals["total_pixels"]
    nonzero = [
        row
        for row in rows
        if _as_nonnegative_int(
            row.get("unmatched_prediction_area"), label="unmatched_prediction_area"
        )
        > 0
    ]
    ordered = sorted(
        nonzero,
        key=lambda row: (
            -int(row["unmatched_prediction_area"]),
            str(row.get("image_name", "")),
        ),
    )
    target_free_area = sum(
        int(row["unmatched_prediction_area"])
        for row in rows
        if row.get("target_free_image") is True
    )
    top_contributors = [
        {
            "image_name": str(row.get("image_name", "")),
            "unmatched_prediction_area": int(row["unmatched_prediction_area"]),
            "unmatched_prediction_components": int(
                row["unmatched_prediction_components"]
            ),
            "total_pixels": int(row["total_pixels"]),
            "target_free_image": bool(row.get("target_free_image", False)),
        }
        for row in ordered[: max(TOP_K_VALUES)]
    ]
    top_k: dict[str, Any] = {}
    original_feasible = _budget_feasible(total_area, total_pixels, budget)
    integer_excess = max(0, total_area * 1_000_000 - budget * total_pixels)
    for requested_k in TOP_K_VALUES:
        removed = ordered[:requested_k]
        removed_area = sum(int(row["unmatched_prediction_area"]) for row in removed)
        removed_pixels = sum(int(row["total_pixels"]) for row in removed)
        remaining_pixels = total_pixels - removed_pixels
        remaining_area = total_area - removed_area
        leave_one_group_fa = (
            _fa_per_mpix(remaining_area, remaining_pixels)
            if remaining_pixels > 0
            else None
        )
        leave_one_group_feasible = (
            _budget_feasible(remaining_area, remaining_pixels, budget)
            if remaining_pixels > 0
            else None
        )
        top_k[str(requested_k)] = {
            "actual_removed_image_count": len(removed),
            "unmatched_area": removed_area,
            "unmatched_area_share": (
                float(removed_area) / float(total_area) if total_area > 0 else None
            ),
            "fixed_denominator_remaining_fa_per_mpix": _fa_per_mpix(
                remaining_area, total_pixels
            ),
            "leave_images_out_remaining_fa_per_mpix": leave_one_group_fa,
            "leave_images_out_budget_feasible": leave_one_group_feasible,
            "repairs_original_overshoot": bool(
                not original_feasible
                and leave_one_group_feasible is True
            ),
            "covers_fixed_denominator_integer_excess": bool(
                removed_area * 1_000_000 >= integer_excess
            ),
        }
    if total_area > 0:
        shares = [float(row["unmatched_prediction_area"]) / total_area for row in ordered]
        hhi = sum(value * value for value in shares)
        effective_count = 1.0 / hhi
    else:
        hhi = None
        effective_count = None
    return {
        "image_count": len(rows),
        "nonzero_unmatched_area_image_count": len(nonzero),
        "nonzero_unmatched_area_image_fraction": float(len(nonzero)) / len(rows),
        "unmatched_area_share_defined": total_area > 0,
        "unmatched_area_hhi": hhi,
        "effective_unmatched_area_contributor_count": effective_count,
        "target_free_unmatched_area": target_free_area,
        "target_free_unmatched_area_share": (
            float(target_free_area) / total_area if total_area > 0 else None
        ),
        "exact_allowed_unmatched_area_floor": budget * total_pixels // 1_000_000,
        "budget_integer_excess": integer_excess,
        "top_contributors": top_contributors,
        "top_k": top_k,
    }


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _canonical_key(
    dataset: Any,
    seed: Any,
    matcher: Any,
    budget: Any,
    evaluation_fold: Any,
) -> tuple[str, int, str, int, int]:
    if not isinstance(dataset, str) or not dataset:
        raise GateFTransportError("dataset must be a non-empty string")
    seed_value = _as_nonnegative_int(seed, label="seed")
    if not isinstance(matcher, str) or not matcher:
        raise GateFTransportError("matcher must be a non-empty string")
    budget_value = _as_positive_int(budget, label="budget")
    fold_value = _as_nonnegative_int(evaluation_fold, label="evaluation_fold")
    return dataset, seed_value, matcher, budget_value, fold_value


def analyze_transport(
    calibration_records: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int],
    matchers: Sequence[str],
    fold_count: int,
    fold_mappings: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate and decompose a frozen E-1c calibration/image ledger."""

    budget_values = tuple(_as_positive_int(value, label="budget") for value in budgets)
    matcher_values = tuple(str(value) for value in matchers)
    if len(set(budget_values)) != len(budget_values) or not budget_values:
        raise GateFTransportError("budgets must be unique and non-empty")
    if (
        len(set(matcher_values)) != len(matcher_values)
        or not matcher_values
        or any(not value for value in matcher_values)
    ):
        raise GateFTransportError("matchers must be unique and non-empty")
    fold_count = _as_positive_int(fold_count, label="fold_count")
    normalized_fold_mappings: dict[str, dict[str, int]] | None = None
    if fold_mappings is not None:
        normalized_fold_mappings = {}
        for dataset, mapping in fold_mappings.items():
            if not isinstance(dataset, str) or not dataset or not isinstance(
                mapping, Mapping
            ):
                raise GateFTransportError("protocol fold mapping is invalid")
            normalized: dict[str, int] = {}
            for image_name, fold in mapping.items():
                if not isinstance(image_name, str) or not image_name:
                    raise GateFTransportError("fold mapping image name is invalid")
                fold_value = _as_nonnegative_int(fold, label="fold mapping value")
                if fold_value >= fold_count:
                    raise GateFTransportError("fold mapping value is outside protocol")
                normalized[image_name] = fold_value
            if not normalized or set(normalized.values()) != set(range(fold_count)):
                raise GateFTransportError("dataset fold mapping lacks a protocol fold")
            normalized_fold_mappings[dataset] = normalized

    calibration_by_key: dict[tuple[str, int, str, int, int], dict[str, Any]] = {}
    calibration_record_by_base: dict[tuple[str, int, str, int], Mapping[str, Any]] = {}
    for record_index, record in enumerate(calibration_records):
        if record.get("schema_version") != INPUT_CALIBRATION_SCHEMA:
            raise GateFTransportError("unexpected calibration schema")
        dataset = record.get("dataset")
        seed = record.get("seed")
        matcher = record.get("matcher")
        evaluation_fold = record.get("evaluation_fold")
        base_key = _canonical_key(dataset, seed, matcher, 1, evaluation_fold)
        dataset_value, seed_value, matcher_value, _, fold_value = base_key
        if matcher_value not in matcher_values or fold_value >= fold_count:
            raise GateFTransportError("calibration matcher/fold is outside protocol")
        calibration_fold = _as_nonnegative_int(
            record.get("calibration_fold"), label="calibration_fold"
        )
        if calibration_fold >= fold_count or calibration_fold == fold_value:
            raise GateFTransportError("calibration/evaluation fold assignment is invalid")
        evaluation_names = record.get("evaluation_image_names")
        calibration_names = record.get("calibration_image_names")
        if (
            not isinstance(evaluation_names, list)
            or not isinstance(calibration_names, list)
            or not evaluation_names
            or not calibration_names
            or any(not isinstance(value, str) or not value for value in evaluation_names)
            or any(not isinstance(value, str) or not value for value in calibration_names)
            or len(set(evaluation_names)) != len(evaluation_names)
            or len(set(calibration_names)) != len(calibration_names)
            or set(evaluation_names) & set(calibration_names)
        ):
            raise GateFTransportError("calibration image partitions are invalid")
        selections = record.get("selections")
        if not isinstance(selections, Mapping):
            raise GateFTransportError("calibration selections must be an object")
        expected_budget_keys = {str(value) for value in budget_values}
        if set(selections) != expected_budget_keys:
            raise GateFTransportError("calibration budget selections drifted")
        _validate_calibration_curve(
            record,
            budgets=budget_values,
            label=f"calibration[{record_index}]",
        )
        if normalized_fold_mappings is not None:
            mapping = normalized_fold_mappings.get(dataset_value)
            if mapping is None:
                raise GateFTransportError("calibration dataset lacks a fold mapping")
            expected_evaluation = {
                image_name
                for image_name, mapped_fold in mapping.items()
                if mapped_fold == fold_value
            }
            expected_calibration = set(mapping) - expected_evaluation
            if set(evaluation_names) != expected_evaluation or set(
                calibration_names
            ) != expected_calibration:
                raise GateFTransportError(
                    "calibration image partitions disagree with protocol mapping"
                )
        record_base = (dataset_value, seed_value, matcher_value, fold_value)
        if record_base in calibration_record_by_base:
            raise GateFTransportError("duplicate calibration record")
        calibration_record_by_base[record_base] = record
        curve_structure = _curve_structure(record)
        for budget in budget_values:
            selection = selections[str(budget)]
            if not isinstance(selection, Mapping):
                raise GateFTransportError("calibration selection must be an object")
            key = (dataset_value, seed_value, matcher_value, budget, fold_value)
            calibration_by_key[key] = {
                **_validate_calibration_selection(
                selection,
                budget=budget,
                image_count=len(calibration_names),
                label=f"calibration[{record_index}].selections[{budget}]",
                ),
                "curve_structure": curve_structure,
            }

    image_groups: dict[
        tuple[str, int, str, int, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    image_identity: set[tuple[str, int, str, int, str]] = set()
    for row in image_rows:
        if row.get("schema_version") != INPUT_IMAGE_SCHEMA:
            raise GateFTransportError("unexpected image ledger schema")
        key = _canonical_key(
            row.get("dataset"),
            row.get("seed"),
            row.get("matcher"),
            row.get("nominal_budget_fa_per_mpix"),
            row.get("evaluation_fold"),
        )
        if key[2] not in matcher_values or key[3] not in budget_values or key[4] >= fold_count:
            raise GateFTransportError("image row key is outside protocol")
        image_name = row.get("image_name")
        if not isinstance(image_name, str) or not image_name:
            raise GateFTransportError("image row lacks a valid image name")
        identity = (*key[:4], image_name)
        if identity in image_identity:
            raise GateFTransportError("duplicate image row within dataset/seed/matcher/budget")
        image_identity.add(identity)
        _metric_counts(row, label=f"image[{image_name}]")
        if not isinstance(row.get("target_free_image"), bool):
            raise GateFTransportError("target_free_image must be boolean")
        if bool(row["target_free_image"]) != (int(row["target_components"]) == 0):
            raise GateFTransportError("target-free flag disagrees with target count")
        if int(row["matched_components"]) + int(
            row["unmatched_prediction_components"]
        ) != int(row["prediction_components"]):
            raise GateFTransportError("image prediction component identity failed")
        image_groups[key].append(row)

    if set(image_groups) != set(calibration_by_key):
        missing_images = set(calibration_by_key) - set(image_groups)
        missing_calibration = set(image_groups) - set(calibration_by_key)
        raise GateFTransportError(
            "calibration/image key universe differs: "
            f"no_images={len(missing_images)}, no_calibration={len(missing_calibration)}"
        )

    image_metadata: dict[tuple[str, int, str], tuple[int, int, bool]] = {}
    for row in image_rows:
        metadata_key = (str(row["dataset"]), int(row["seed"]), str(row["image_name"]))
        metadata = (
            int(row["total_pixels"]),
            int(row["target_components"]),
            bool(row["target_free_image"]),
        )
        previous = image_metadata.setdefault(metadata_key, metadata)
        if previous != metadata:
            raise GateFTransportError("image population metadata varies by matcher/budget")
    for base_key, record in calibration_record_by_base.items():
        dataset, seed, matcher, evaluation_fold = base_key
        del matcher, evaluation_fold
        calibration_names = [str(value) for value in record["calibration_image_names"]]
        try:
            metadata = [
                image_metadata[(dataset, seed, image_name)]
                for image_name in calibration_names
            ]
        except KeyError as exc:
            raise GateFTransportError(
                "calibration population image is absent from held-out metadata"
            ) from exc
        for budget in budget_values:
            selection = record["selections"][str(budget)]
            if (
                int(selection["sample_count"]) != len(metadata)
                or int(selection["total_pixels"])
                != sum(value[0] for value in metadata)
                or int(selection["target_components"])
                != sum(value[1] for value in metadata)
            ):
                raise GateFTransportError(
                    "calibration population disagrees with immutable image metadata"
                )

    fold_rows: list[dict[str, Any]] = []
    for key in sorted(image_groups):
        dataset, seed, matcher, budget, evaluation_fold = key
        rows = image_groups[key]
        record = calibration_record_by_base[(dataset, seed, matcher, evaluation_fold)]
        expected_names = tuple(record["evaluation_image_names"])
        observed_names = tuple(str(row["image_name"]) for row in rows)
        if observed_names != expected_names:
            raise GateFTransportError("held-out image order/universe differs from calibration record")
        calibration = calibration_by_key[key]
        for row in rows:
            threshold = _as_finite_float(
                row.get("calibration_threshold"), label="calibration_threshold"
            )
            if threshold != calibration["threshold"]:
                raise GateFTransportError("image row threshold differs from calibration selection")
        held_counts = _sum_image_counts(rows)
        held = _metric_payload(held_counts, budget=budget)
        for row in rows:
            stored = row.get("held_out_fold_aggregate")
            if not isinstance(stored, Mapping):
                raise GateFTransportError("image row lacks held-out fold aggregate")
            _validate_stored_aggregate(stored, held, label="held_out_fold_aggregate")
        fa_ratio = (
            held["fa_per_mpix"] / calibration["fa_per_mpix"]
            if calibration["fa_per_mpix"] > 0
            else None
        )
        fold_rows.append(
            {
                "schema_version": FOLD_ROW_SCHEMA,
                "dataset": dataset,
                "seed": seed,
                "matcher": matcher,
                "nominal_budget_fa_per_mpix": budget,
                "evaluation_fold": evaluation_fold,
                "calibration_fold": int(record["calibration_fold"]),
                "calibration": calibration,
                "held_out": held,
                "transport": {
                    "fa_delta_per_mpix": held["fa_per_mpix"]
                    - calibration["fa_per_mpix"],
                    "fa_ratio": fa_ratio,
                    "fa_ratio_defined": fa_ratio is not None,
                    "pd_delta": held["pd"] - calibration["pd"],
                    "calibration_feasible_but_held_out_overshoots": not held[
                        "budget_feasible_zero_overshoot"
                    ],
                },
                "held_out_unmatched_area_concentration": _concentration(
                    rows, budget=budget
                ),
            }
        )

    fold_rows_by_pair: dict[
        tuple[str, int, str, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    image_rows_by_pair: dict[
        tuple[str, int, str, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in fold_rows:
        pair_key = (
            str(row["dataset"]),
            int(row["seed"]),
            str(row["matcher"]),
            int(row["nominal_budget_fa_per_mpix"]),
        )
        fold_rows_by_pair[pair_key].append(row)
    for key, rows in image_groups.items():
        image_rows_by_pair[key[:4]].extend(rows)

    pair_rows: list[dict[str, Any]] = []
    for pair_key in sorted(fold_rows_by_pair):
        dataset, seed, matcher, budget = pair_key
        folds = sorted(
            fold_rows_by_pair[pair_key], key=lambda row: int(row["evaluation_fold"])
        )
        if len(folds) != fold_count or {
            int(row["evaluation_fold"]) for row in folds
        } != set(range(fold_count)):
            raise GateFTransportError("dataset/seed/matcher/budget lacks complete folds")
        if fold_count == 2:
            base0 = (dataset, seed, matcher, 0)
            base1 = (dataset, seed, matcher, 1)
            record0 = calibration_record_by_base[base0]
            record1 = calibration_record_by_base[base1]
            if set(record0["evaluation_image_names"]) != set(
                record1["calibration_image_names"]
            ) or set(record1["evaluation_image_names"]) != set(
                record0["calibration_image_names"]
            ):
                raise GateFTransportError("two-fold calibration/evaluation complements drifted")
        pair_images = image_rows_by_pair[pair_key]
        if len({str(row["image_name"]) for row in pair_images}) != len(pair_images):
            raise GateFTransportError("pooled held-out image universe contains duplicates")
        pooled_counts = _sum_image_counts(pair_images)
        pooled = _metric_payload(pooled_counts, budget=budget)
        for row in pair_images:
            stored = row.get("dataset_seed_aggregate")
            if not isinstance(stored, Mapping):
                raise GateFTransportError("image row lacks dataset/seed aggregate")
            _validate_stored_aggregate(stored, pooled, label="dataset_seed_aggregate")
        thresholds = [float(row["calibration"]["threshold"]) for row in folds]
        held_fa = [float(row["held_out"]["fa_per_mpix"]) for row in folds]
        held_pd = [float(row["held_out"]["pd"]) for row in folds]
        concentration = _concentration(pair_images, budget=budget)
        pair_rows.append(
            {
                "schema_version": PAIR_ROW_SCHEMA,
                "dataset": dataset,
                "seed": seed,
                "matcher": matcher,
                "nominal_budget_fa_per_mpix": budget,
                "fold_count": fold_count,
                "fold_records": [
                    {
                        "evaluation_fold": int(row["evaluation_fold"]),
                        "calibration_threshold": float(
                            row["calibration"]["threshold"]
                        ),
                        "calibration_fa_per_mpix": float(
                            row["calibration"]["fa_per_mpix"]
                        ),
                        "calibration_pd": float(row["calibration"]["pd"]),
                        "calibration_all_off": bool(row["calibration"]["all_off"]),
                        "calibration_no_true_positive": bool(
                            row["calibration"]["no_true_positive"]
                        ),
                        "held_out_fa_per_mpix": float(row["held_out"]["fa_per_mpix"]),
                        "held_out_pd": float(row["held_out"]["pd"]),
                        "held_out_budget_feasible": bool(
                            row["held_out"]["budget_feasible_zero_overshoot"]
                        ),
                    }
                    for row in folds
                ],
                "threshold_transport": {
                    "minimum": min(thresholds),
                    "maximum": max(thresholds),
                    "absolute_span": max(thresholds) - min(thresholds),
                    "selected_thresholds_equal": len(set(thresholds)) == 1,
                },
                "held_out_fold_variation": {
                    "fa_min_per_mpix": min(held_fa),
                    "fa_max_per_mpix": max(held_fa),
                    "fa_absolute_span_per_mpix": max(held_fa) - min(held_fa),
                    "pd_min": min(held_pd),
                    "pd_max": max(held_pd),
                    "pd_absolute_span": max(held_pd) - min(held_pd),
                },
                "calibration_degeneracy": {
                    "all_off_fold_count": sum(
                        bool(row["calibration"]["all_off"]) for row in folds
                    ),
                    "no_true_positive_fold_count": sum(
                        bool(row["calibration"]["no_true_positive"]) for row in folds
                    ),
                },
                "held_out_overshooting_fold_count": sum(
                    not bool(row["held_out"]["budget_feasible_zero_overshoot"])
                    for row in folds
                ),
                "pooled_held_out": pooled,
                "pooled_unmatched_area_concentration": concentration,
            }
        )

    dataset_seed_matchers = {
        (str(row["dataset"]), int(row["seed"]), str(row["matcher"]))
        for row in pair_rows
    }
    expected_pair_count = len(dataset_seed_matchers) * len(budget_values)
    if len(pair_rows) != expected_pair_count:
        raise GateFTransportError("pair ledger cardinality differs across budgets")

    def matcher_independent_signature(row: Mapping[str, Any]) -> str:
        payload = {
            key: value
            for key, value in row.items()
            if key not in {"schema_version", "matcher"}
        }
        return json.dumps(payload, sort_keys=True, allow_nan=False)

    fold_matcher_groups: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in fold_rows:
        fold_matcher_groups[
            (
                str(row["dataset"]),
                int(row["seed"]),
                int(row["nominal_budget_fa_per_mpix"]),
                int(row["evaluation_fold"]),
            )
        ].append(row)
    pair_matcher_groups: dict[tuple[str, int, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in pair_rows:
        pair_matcher_groups[
            (
                str(row["dataset"]),
                int(row["seed"]),
                int(row["nominal_budget_fa_per_mpix"]),
            )
        ].append(row)
    fold_matcher_equal = sum(
        len(rows) == len(matcher_values)
        and len({matcher_independent_signature(row) for row in rows}) == 1
        for rows in fold_matcher_groups.values()
    )
    pair_matcher_equal = sum(
        len(rows) == len(matcher_values)
        and len({matcher_independent_signature(row) for row in rows}) == 1
        for rows in pair_matcher_groups.values()
    )

    by_budget: dict[str, Any] = {}
    for budget in budget_values:
        rows = [
            row
            for row in pair_rows
            if int(row["nominal_budget_fa_per_mpix"]) == budget
        ]
        infeasible = [
            row
            for row in rows
            if not row["pooled_held_out"]["budget_feasible_zero_overshoot"]
        ]
        dataset_seed_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = (
            defaultdict(list)
        )
        for row in rows:
            dataset_seed_groups[(str(row["dataset"]), int(row["seed"]))].append(row)
        all_matcher_infeasible_groups = [
            group
            for group in dataset_seed_groups.values()
            if len(group) == len(matcher_values)
            and all(
                not row["pooled_held_out"]["budget_feasible_zero_overshoot"]
                for row in group
            )
        ]
        overshooting_dataset_seed_folds = {
            (str(row["dataset"]), int(row["seed"]), int(fold["evaluation_fold"]))
            for row in rows
            for fold in row["fold_records"]
            if not bool(fold["held_out_budget_feasible"])
        }
        by_budget[str(budget)] = {
            "dataset_seed_matcher_count": len(rows),
            "pooled_feasible_count": len(rows) - len(infeasible),
            "pooled_infeasible_count": len(infeasible),
            "matcher_collapsed_dataset_seed_count": len(dataset_seed_groups),
            "matcher_collapsed_all_matcher_infeasible_dataset_seed_count": len(
                all_matcher_infeasible_groups
            ),
            "matcher_collapsed_overshooting_dataset_seed_fold_count": len(
                overshooting_dataset_seed_folds
            ),
            "matcher_collapsed_infeasible_dataset_seed_top1_leave_out_repair_count": sum(
                all(
                    bool(
                        row["pooled_unmatched_area_concentration"]["top_k"]["1"][
                            "repairs_original_overshoot"
                        ]
                    )
                    for row in group
                )
                for group in all_matcher_infeasible_groups
            ),
            "held_out_overshooting_fold_count": sum(
                int(row["held_out_overshooting_fold_count"]) for row in rows
            ),
            "calibration_all_off_fold_count": sum(
                int(row["calibration_degeneracy"]["all_off_fold_count"])
                for row in rows
            ),
            "infeasible_pair_top1_leave_out_repair_count": sum(
                bool(
                    row["pooled_unmatched_area_concentration"]["top_k"]["1"][
                        "repairs_original_overshoot"
                    ]
                )
                for row in infeasible
            ),
            "infeasible_pair_top1_area_share_median": _median(
                [
                    float(
                        row["pooled_unmatched_area_concentration"]["top_k"]["1"][
                            "unmatched_area_share"
                        ]
                    )
                    for row in infeasible
                    if row["pooled_unmatched_area_concentration"]["top_k"]["1"][
                        "unmatched_area_share"
                    ]
                    is not None
                ]
            ),
            "threshold_absolute_span_median": _median(
                [float(row["threshold_transport"]["absolute_span"]) for row in rows]
            ),
            "threshold_absolute_span_maximum": max(
                float(row["threshold_transport"]["absolute_span"]) for row in rows
            ),
            "infeasible_pairs": [
                {
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "matcher": row["matcher"],
                    "achieved_fa_per_mpix": row["pooled_held_out"]["fa_per_mpix"],
                    "achieved_pd": row["pooled_held_out"]["pd"],
                    "calibration_thresholds": [
                        value["calibration_threshold"] for value in row["fold_records"]
                    ],
                    "held_out_fold_fa_per_mpix": [
                        value["held_out_fa_per_mpix"] for value in row["fold_records"]
                    ],
                    "top1_unmatched_area_share": row[
                        "pooled_unmatched_area_concentration"
                    ]["top_k"]["1"]["unmatched_area_share"],
                    "top1_leave_out_repairs": row[
                        "pooled_unmatched_area_concentration"
                    ]["top_k"]["1"]["repairs_original_overshoot"],
                }
                for row in infeasible
            ],
        }

    datasets = sorted({str(row["dataset"]) for row in pair_rows})
    by_dataset_budget: dict[str, Any] = {}
    for dataset in datasets:
        by_dataset_budget[dataset] = {}
        for budget in budget_values:
            rows = [
                row
                for row in pair_rows
                if row["dataset"] == dataset
                and int(row["nominal_budget_fa_per_mpix"]) == budget
            ]
            expected = {
                (int(row["seed"]), str(row["matcher"])) for row in rows
            }
            seeds = {seed for seed, _ in expected}
            complete = expected == {
                (seed, matcher) for seed in seeds for matcher in matcher_values
            }
            by_dataset_budget[dataset][str(budget)] = {
                "seed_count": len(seeds),
                "matcher_count": len({matcher for _, matcher in expected}),
                "complete_seed_matcher_grid": complete,
                "all_seed_both_matcher_zero_overshoot": bool(
                    complete
                    and rows
                    and all(
                        row["pooled_held_out"]["budget_feasible_zero_overshoot"]
                        for row in rows
                    )
                ),
                "pooled_fa_per_mpix_by_seed_matcher": [
                    {
                        "seed": row["seed"],
                        "matcher": row["matcher"],
                        "fa_per_mpix": row["pooled_held_out"]["fa_per_mpix"],
                        "pd": row["pooled_held_out"]["pd"],
                        "feasible": row["pooled_held_out"][
                            "budget_feasible_zero_overshoot"
                        ],
                    }
                    for row in rows
                ],
            }

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "analysis_scope": (
            "exploratory read-only decomposition of the immutable Gate E-1c "
            "development-holdout bundle; no inference rerun"
        ),
        "formal_gate_effect": "none; Gate E-1c remains FAIL and Gate E0 remains NO-GO",
        "diagnostic_caveat": (
            "top-k leave-image-out values are outcome-visible concentration "
            "sensitivities, not exclusions, estimators, or revised gate rules"
        ),
        "counts": {
            "calibration_record_count": len(calibration_records),
            "image_row_count": len(image_rows),
            "fold_transport_row_count": len(fold_rows),
            "pair_transport_row_count": len(pair_rows),
            "dataset_count": len(datasets),
            "dataset_seed_matcher_count": len(dataset_seed_matchers),
        },
        "protocol": {
            "budgets_fa_per_mpix": list(budget_values),
            "matchers": list(matcher_values),
            "fold_count": fold_count,
            "budget_test": "unmatched_area*1000000 <= budget*total_pixels",
            "top_k_image_sensitivities": list(TOP_K_VALUES),
        },
        "matcher_sensitivity": {
            "matcher_outputs_were_computed_independently": True,
            "fold_group_count": len(fold_matcher_groups),
            "fold_groups_identical_across_matchers": fold_matcher_equal,
            "pair_group_count": len(pair_matcher_groups),
            "pair_groups_identical_across_matchers": pair_matcher_equal,
            "all_selected_outputs_identical_across_matchers": bool(
                fold_matcher_equal == len(fold_matcher_groups)
                and pair_matcher_equal == len(pair_matcher_groups)
            ),
            "interpretation": (
                "empirical equality in this bundle only; matcher paths remain "
                "separate and are not merged in the formal records"
            ),
        },
        "calibration_curve_structure": {
            "record_count": len(calibration_records),
            "nonmonotone_unmatched_area_record_count": sum(
                bool(_curve_structure(record)["unmatched_area_nonmonotone_in_threshold"])
                for record in calibration_records
            ),
            "maximum_single_step_unmatched_area_increase": max(
                (
                    int(
                        _curve_structure(record)[
                            "maximum_single_step_unmatched_area_increase"
                        ]
                    )
                    for record in calibration_records
                ),
                default=0,
            ),
            "interpretation": (
                "matching-defined unmatched component area is not assumed monotone; "
                "full frozen candidate curves and the original tie-break were validated"
            ),
        },
        "by_budget": by_budget,
        "by_dataset_budget": by_dataset_budget,
    }
    return fold_rows, pair_rows, summary


def build_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gate F operating-point transport decomposition",
        "",
        str(summary["analysis_scope"]),
        "",
        f"- Formal gate effect: {summary['formal_gate_effect']}",
        f"- Diagnostic caveat: {summary['diagnostic_caveat']}",
        "",
        "## Budget-level transport",
        "",
        "| budget | feasible matcher-pairs | infeasible matcher-pairs | infeasible dataset-seeds | overshooting dataset-seed folds | top-1 repair dataset-seeds | median threshold span |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in summary["protocol"]["budgets_fa_per_mpix"]:
        record = summary["by_budget"][str(budget)]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                budget,
                record["pooled_feasible_count"],
                record["pooled_infeasible_count"],
                record[
                    "matcher_collapsed_all_matcher_infeasible_dataset_seed_count"
                ],
                record[
                    "matcher_collapsed_overshooting_dataset_seed_fold_count"
                ],
                record[
                    "matcher_collapsed_infeasible_dataset_seed_top1_leave_out_repair_count"
                ],
                record["threshold_absolute_span_median"],
            )
        )
    lines.extend(
        [
            "",
            "## Dataset-level exact feasibility",
            "",
            "| dataset | budget | all seeds and both matchers feasible |",
            "|---|---:|:---:|",
        ]
    )
    for dataset, by_budget in summary["by_dataset_budget"].items():
        for budget in summary["protocol"]["budgets_fa_per_mpix"]:
            value = by_budget[str(budget)][
                "all_seed_both_matcher_zero_overshoot"
            ]
            lines.append(f"| {dataset} | {budget} | {value} |")
    lines.extend(
        [
            "",
            "Top-k leave-image-out quantities are exploratory sensitivity checks only.",
            "They do not authorize image removal or alter the frozen E-1c decision.",
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
    fold_rows: Sequence[Mapping[str, Any]],
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
        fold_path = temporary / OUTPUT_FILES[0]
        pair_path = temporary / OUTPUT_FILES[1]
        summary_path = temporary / OUTPUT_FILES[2]
        markdown_path = temporary / OUTPUT_FILES[3]
        _write_jsonl(fold_path, fold_rows)
        _write_jsonl(pair_path, pair_rows)
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
        markdown_path.write_text(build_markdown(summary), encoding="utf-8")
        artifact_paths = (fold_path, pair_path, summary_path, markdown_path)
        complete_provenance = {
            **dict(provenance),
            "artifact_sha256": {
                path.name: sha256_file(path) for path in artifact_paths
            },
        }
        (temporary / OUTPUT_FILES[4]).write_text(
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
            raise GateFTransportError("temporary Gate F inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_input_bundle(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance_path = input_dir / "provenance.json"
    provenance = _read_json(provenance_path)
    if not isinstance(provenance, dict) or provenance.get(
        "schema_version"
    ) != INPUT_PROVENANCE_SCHEMA:
        raise GateFTransportError("input is not the formal Gate E-1c bundle")
    artifact_hashes = provenance.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        raise GateFTransportError("input provenance lacks artifact hashes")
    for name, expected in artifact_hashes.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise GateFTransportError("input artifact hash registry is invalid")
        path = input_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise GateFTransportError(f"input artifact hash mismatch: {name}")
    formal_summary = _read_json(input_dir / "low_fa_bridge_summary.json")
    if not isinstance(formal_summary, dict) or not isinstance(
        formal_summary.get("joint_gate"), Mapping
    ):
        raise GateFTransportError("formal E-1c summary is invalid")
    return provenance, formal_summary


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise GateFTransportError("Gate F output cannot be inside the immutable E-1c input")
    input_provenance, formal_summary = _verify_input_bundle(input_dir)
    input_provenance_sha256 = sha256_file(input_dir / "provenance.json")
    source_sha256 = sha256_file(Path(__file__).resolve())
    joint_gate = formal_summary["joint_gate"]
    if joint_gate.get("pass") is not False or joint_gate.get(
        "passing_budgets"
    ) != []:
        raise GateFTransportError(
            "this Gate F workflow requires the frozen failed E-1c decision"
        )
    protocol = input_provenance.get("protocol")
    if not isinstance(protocol, Mapping):
        raise GateFTransportError("input provenance lacks the frozen protocol")
    budgets = protocol.get("budgets_fa_per_mpix")
    matchers = protocol.get("matchers")
    fold_count = protocol.get("fold_count")
    if not isinstance(budgets, list) or not isinstance(matchers, list):
        raise GateFTransportError("input protocol budgets/matchers are invalid")
    calibration = _read_json(input_dir / "calibration.json")
    if not isinstance(calibration, list):
        raise GateFTransportError("calibration.json must contain a list")
    image_rows = _read_jsonl(input_dir / "image_low_fa.jsonl")
    fold_rows, pair_rows, summary = analyze_transport(
        calibration,
        image_rows,
        budgets=budgets,
        matchers=matchers,
        fold_count=fold_count,
        fold_mappings=protocol.get("fold_mappings"),
    )
    rechecked_provenance, rechecked_summary = _verify_input_bundle(input_dir)
    if (
        sha256_file(input_dir / "provenance.json") != input_provenance_sha256
        or rechecked_provenance != input_provenance
        or rechecked_summary != formal_summary
        or sha256_file(Path(__file__).resolve()) != source_sha256
    ):
        raise GateFTransportError("Gate E-1c input or Gate F source changed during analysis")
    provenance_path = input_dir / "provenance.json"
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "source_bundle": {
            "path": str(input_dir),
            "provenance_sha256": sha256_file(provenance_path),
            "artifact_sha256": dict(input_provenance["artifact_sha256"]),
            "formal_joint_gate_pass": bool(formal_summary["joint_gate"]["pass"]),
            "formal_passing_budgets": list(
                formal_summary["joint_gate"]["passing_budgets"]
            ),
        },
        "source_sha256": {"tool": source_sha256},
        "analysis_contract": {
            "read_only": True,
            "model_inference_rerun": False,
            "formal_gate_unchanged": True,
            "leave_image_out_is_exploratory_only": True,
        },
        "runtime": {"python": sys.version},
    }
    write_bundle(
        output_dir,
        fold_rows=fold_rows,
        pair_rows=pair_rows,
        summary=summary,
        provenance=provenance,
    )
    try:
        rechecked_provenance, rechecked_summary = _verify_input_bundle(input_dir)
        if (
            sha256_file(input_dir / "provenance.json") != input_provenance_sha256
            or rechecked_provenance != input_provenance
            or rechecked_summary != formal_summary
            or sha256_file(Path(__file__).resolve()) != source_sha256
        ):
            raise GateFTransportError(
                "Gate E-1c input or Gate F source changed before handoff"
            )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return summary, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    summary, output_dir = run(parse_args(argv))
    print(
        "Gate F transport rows: folds=%s, pairs=%s"
        % (
            summary["counts"]["fold_transport_row_count"],
            summary["counts"]["pair_transport_row_count"],
        )
    )
    print(f"wrote exploratory Gate F bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateFTransportError, FileExistsError, OSError) as exc:
        print(f"Gate F transport analysis refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
