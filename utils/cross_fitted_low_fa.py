"""Pure Gate E-1c cross-fitted component-frontier utilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
from typing import Any

import numpy as np

from utils.component_operating_point import (
    DEFAULT_TAIL_QUANTILES,
    ComponentOperatingPoint,
    build_logit_threshold_grid,
    evaluate_component_operating_points,
)
from utils.metric import match_components_hungarian, match_connected_components
from utils.target_identity import StableTargetSet


FOLD_COUNT = 2
FOLD_NAMESPACE = "mshnet-decision-operating-fold-v1"
BUDGETS = (1, 5, 10, 20)
MATCHERS = ("official_legacy", "audit_hungarian")
MATCHING_IMPLEMENTATION = {
    "official_legacy": "legacy",
    "audit_hungarian": "hungarian",
}
MIN_BRIDGE_DATASETS = 2
MIN_BRIDGE_TARGETS = 12
MIN_CONTROL_TARGETS = 12


class LowFABridgeError(RuntimeError):
    """Raised when the frozen cross-fitted bridge contract is violated."""


def image_fold(image_name: str) -> int:
    if not isinstance(image_name, str) or not image_name:
        raise LowFABridgeError("image_name must be a non-empty string")
    digest = hashlib.sha256(
        (FOLD_NAMESPACE + "\0" + image_name).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") % FOLD_COUNT


def budget_feasible_exact(
    unmatched_prediction_area: int,
    total_pixels: int,
    budget: int,
) -> bool:
    if (
        isinstance(unmatched_prediction_area, bool)
        or int(unmatched_prediction_area) != unmatched_prediction_area
        or unmatched_prediction_area < 0
    ):
        raise LowFABridgeError("unmatched prediction area must be a non-negative int")
    if (
        isinstance(total_pixels, bool)
        or int(total_pixels) != total_pixels
        or total_pixels <= 0
    ):
        raise LowFABridgeError("total pixels must be a positive int")
    if isinstance(budget, bool) or int(budget) != budget or budget < 0:
        raise LowFABridgeError("budget must be a non-negative int")
    return int(unmatched_prediction_area) * 1_000_000 <= int(budget) * int(
        total_pixels
    )


def select_exact_budget_point(
    curve: Sequence[ComponentOperatingPoint], budget: int
) -> ComponentOperatingPoint:
    if not curve:
        raise LowFABridgeError("calibration curve cannot be empty")
    populations = {
        (point.sample_count, point.total_pixels, point.target_components)
        for point in curve
    }
    if len(populations) != 1 or curve[0].target_components <= 0:
        raise LowFABridgeError("calibration curve population is invalid")
    feasible = [
        point
        for point in curve
        if budget_feasible_exact(
            point.unmatched_prediction_area, point.total_pixels, budget
        )
    ]
    if not feasible:
        raise LowFABridgeError(f"no calibration point is feasible for budget {budget}")
    return max(
        feasible,
        key=lambda point: (
            point.matched_components,
            -point.unmatched_prediction_area,
            point.threshold,
        ),
    )


def _match(
    scores: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    matcher: str,
):
    if matcher == "official_legacy":
        return match_connected_components(
            scores > threshold,
            target,
            max_centroid_distance=3.0,
            connectivity=2,
        )
    if matcher == "audit_hungarian":
        return match_components_hungarian(
            scores > threshold,
            target,
            centroid_radius=3.0,
            connectivity=2,
        )
    raise LowFABridgeError(f"unknown matcher {matcher}")


def _image_counts(component_match: Any, total_pixels: int) -> dict[str, Any]:
    unmatched_area = int(
        sum(
            component_match.prediction_regions[index].area
            for index in component_match.unmatched_prediction_indices
        )
    )
    return {
        "total_pixels": int(total_pixels),
        "target_components": len(component_match.target_regions),
        "matched_components": len(component_match.matches),
        "prediction_components": len(component_match.prediction_regions),
        "unmatched_prediction_components": len(
            component_match.unmatched_prediction_indices
        ),
        "unmatched_prediction_area": unmatched_area,
    }


def cross_fit_job(
    logits: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    image_names: Sequence[str],
    *,
    dataset: str,
    seed: int,
    registry: Mapping[str, StableTargetSet],
    checkpoint: Mapping[str, Any],
    tail_quantiles: Sequence[float] = DEFAULT_TAIL_QUANTILES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Cross-fit both matchers for one frozen dataset/seed checkpoint."""

    if not (len(logits) == len(targets) == len(image_names) == len(registry)):
        raise LowFABridgeError("job samples, names, and registry do not align")
    if tuple(registry) != tuple(image_names) or len(set(image_names)) != len(image_names):
        raise LowFABridgeError("job image universe/order differs from authority")
    score_arrays = tuple(np.asarray(value) for value in logits)
    target_arrays = tuple(np.asarray(value, dtype=bool) for value in targets)
    for scores, target in zip(score_arrays, target_arrays):
        if (
            scores.shape != target.shape
            or scores.ndim != 2
            or not bool(np.isfinite(scores).all())
        ):
            raise LowFABridgeError("invalid job logit/target sample")
    folds = tuple(image_fold(name) for name in image_names)
    if set(folds) != set(range(FOLD_COUNT)):
        raise LowFABridgeError("deterministic two-fold split lacks a fold")

    target_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    for evaluation_fold in range(FOLD_COUNT):
        evaluation_indices = [
            index for index, fold in enumerate(folds) if fold == evaluation_fold
        ]
        calibration_indices = [
            index for index, fold in enumerate(folds) if fold != evaluation_fold
        ]
        calibration_logits = [score_arrays[index] for index in calibration_indices]
        calibration_targets = [target_arrays[index] for index in calibration_indices]
        quantiles = tuple(float(value) for value in tail_quantiles)
        threshold_grid = build_logit_threshold_grid(
            calibration_logits,
            fixed_thresholds=(0.0,),
            tail_quantiles=quantiles,
        )
        if max(float(np.max(value)) for value in calibration_logits) not in threshold_grid:
            raise LowFABridgeError("calibration grid lacks its all-off candidate")
        for matcher in MATCHERS:
            curve = evaluate_component_operating_points(
                calibration_logits,
                calibration_targets,
                threshold_grid,
                matching=MATCHING_IMPLEMENTATION[matcher],
                centroid_radius=3.0,
                connectivity=2,
            )
            selections = {
                budget: select_exact_budget_point(curve, budget) for budget in BUDGETS
            }
            calibration_records.append(
                {
                    "schema_version": "dea.gate_e.low_fa_calibration.v1",
                    "dataset": dataset,
                    "seed": int(seed),
                    "matcher": matcher,
                    "evaluation_fold": evaluation_fold,
                    "calibration_fold": 1 - evaluation_fold,
                    "calibration_image_names": [
                        image_names[index] for index in calibration_indices
                    ],
                    "evaluation_image_names": [
                        image_names[index] for index in evaluation_indices
                    ],
                    "threshold_grid": [float(value) for value in threshold_grid],
                    "tail_quantiles": list(quantiles),
                    "curve": [asdict(point) for point in curve],
                    "selections": {
                        str(budget): asdict(point)
                        for budget, point in selections.items()
                    },
                }
            )

            fixed_matches: dict[int, Any] = {
                index: _match(
                    score_arrays[index],
                    target_arrays[index],
                    threshold=0.0,
                    matcher=matcher,
                )
                for index in evaluation_indices
            }
            for budget, calibration_point in selections.items():
                threshold = float(calibration_point.threshold)
                pending_target_rows: list[dict[str, Any]] = []
                pending_image_rows: list[dict[str, Any]] = []
                for image_index in evaluation_indices:
                    image_name = image_names[image_index]
                    low_fa_match = _match(
                        score_arrays[image_index],
                        target_arrays[image_index],
                        threshold=threshold,
                        matcher=matcher,
                    )
                    fixed_match = fixed_matches[image_index]
                    target_set = registry[image_name]
                    if len(low_fa_match.target_regions) != len(target_set.targets):
                        raise LowFABridgeError("target count disagrees with authority")
                    low_fa_matched = {
                        int(target_index)
                        for target_index, _, _ in low_fa_match.matches
                    }
                    fixed_matched = {
                        int(target_index)
                        for target_index, _, _ in fixed_match.matches
                    }
                    counts = _image_counts(low_fa_match, score_arrays[image_index].size)
                    pending_image_rows.append(
                        {
                            "schema_version": "dea.gate_e.low_fa_image.v1",
                            "dataset": dataset,
                            "seed": int(seed),
                            "image_name": image_name,
                            "image_index": image_index,
                            "evaluation_fold": evaluation_fold,
                            "matcher": matcher,
                            "nominal_budget_fa_per_mpix": budget,
                            "calibration_threshold": threshold,
                            "target_free_image": not bool(target_set.targets),
                            "checkpoint": dict(checkpoint),
                            **counts,
                        }
                    )
                    source_to_identity = {
                        identity.source_component_index: identity
                        for identity in target_set.targets
                    }
                    if set(source_to_identity) != set(
                        range(len(low_fa_match.target_regions))
                    ):
                        raise LowFABridgeError("source component indices drifted")
                    for source_index in range(len(low_fa_match.target_regions)):
                        identity = source_to_identity[source_index]
                        pending_target_rows.append(
                            {
                                "schema_version": "dea.gate_e.low_fa_target.v1",
                                "dataset": dataset,
                                "seed": int(seed),
                                "image_name": image_name,
                                "image_index": image_index,
                                "evaluation_fold": evaluation_fold,
                                "stable_target_id": identity.stable_key,
                                "component_mask_sha256": identity.component_mask_sha256,
                                "label_mask_sha256": identity.label_mask_sha256,
                                "component_index": identity.component_index,
                                "source_component_index": identity.source_component_index,
                                "area": identity.area,
                                "matcher": matcher,
                                "nominal_budget_fa_per_mpix": budget,
                                "calibration_threshold": threshold,
                                "fixed_logit0_matched": source_index in fixed_matched,
                                "low_fa_matched": source_index in low_fa_matched,
                                "checkpoint": dict(checkpoint),
                            }
                        )
                aggregate = {
                    name: sum(int(row[name]) for row in pending_image_rows)
                    for name in (
                        "total_pixels",
                        "target_components",
                        "matched_components",
                        "prediction_components",
                        "unmatched_prediction_components",
                        "unmatched_prediction_area",
                    )
                }
                feasible = budget_feasible_exact(
                    aggregate["unmatched_prediction_area"],
                    aggregate["total_pixels"],
                    budget,
                )
                achieved_fa = (
                    aggregate["unmatched_prediction_area"]
                    / aggregate["total_pixels"]
                    * 1_000_000.0
                )
                achieved_pd = (
                    aggregate["matched_components"] / aggregate["target_components"]
                    if aggregate["target_components"]
                    else None
                )
                fold_aggregate_record = {
                    **aggregate,
                    "achieved_fa_per_mpix": achieved_fa,
                    "achieved_pd": achieved_pd,
                    "budget_feasible_zero_overshoot": feasible,
                    "budget_feasibility_integer_test": (
                        f"{aggregate['unmatched_prediction_area']}*1000000 "
                        f"<= {budget}*{aggregate['total_pixels']}"
                    ),
                }
                for row in pending_image_rows:
                    row["held_out_fold_aggregate"] = fold_aggregate_record
                for row in pending_target_rows:
                    row["held_out_fold_aggregate"] = fold_aggregate_record
                image_rows.extend(pending_image_rows)
                target_rows.extend(pending_target_rows)
    for matcher in MATCHERS:
        for budget in BUDGETS:
            selected_images = [
                row
                for row in image_rows
                if row["matcher"] == matcher
                and row["nominal_budget_fa_per_mpix"] == budget
            ]
            selected_targets = [
                row
                for row in target_rows
                if row["matcher"] == matcher
                and row["nominal_budget_fa_per_mpix"] == budget
            ]
            if len({int(row["evaluation_fold"]) for row in selected_images}) != FOLD_COUNT:
                raise LowFABridgeError("dataset/seed pool does not contain both folds")
            aggregate = {
                name: sum(int(row[name]) for row in selected_images)
                for name in (
                    "total_pixels",
                    "target_components",
                    "matched_components",
                    "prediction_components",
                    "unmatched_prediction_components",
                    "unmatched_prediction_area",
                )
            }
            pooled = {
                **aggregate,
                "achieved_fa_per_mpix": (
                    aggregate["unmatched_prediction_area"]
                    / aggregate["total_pixels"]
                    * 1_000_000.0
                ),
                "achieved_pd": (
                    aggregate["matched_components"] / aggregate["target_components"]
                    if aggregate["target_components"]
                    else None
                ),
                "budget_feasible_zero_overshoot": budget_feasible_exact(
                    aggregate["unmatched_prediction_area"],
                    aggregate["total_pixels"],
                    budget,
                ),
                "budget_feasibility_integer_test": (
                    f"{aggregate['unmatched_prediction_area']}*1000000 "
                    f"<= {budget}*{aggregate['total_pixels']}"
                ),
                "aggregation": "integer counts pooled across both held-out folds",
            }
            for row in (*selected_images, *selected_targets):
                row["dataset_seed_aggregate"] = pooled
    expected_target_rows = len(
        [target for target_set in registry.values() for target in target_set.targets]
    ) * len(MATCHERS) * len(BUDGETS)
    expected_image_rows = len(registry) * len(MATCHERS) * len(BUDGETS)
    if len(target_rows) != expected_target_rows or len(image_rows) != expected_image_rows:
        raise LowFABridgeError("cross-fitted job ledger cardinality drifted")
    return target_rows, image_rows, calibration_records


def validate_hungarian_fixed_alignment(
    low_fa_rows: Sequence[Mapping[str, Any]],
    fixed_rows: Sequence[Mapping[str, Any]],
) -> None:
    expected: dict[tuple[str, int], bool] = {}
    for row in fixed_rows:
        if row.get("row_kind") != "target":
            continue
        key = (str(row.get("stable_target_id")), int(row.get("seed")))
        if key in expected:
            raise LowFABridgeError("duplicate fixed target/seed status")
        expected[key] = bool(row.get("matched"))
    observed: dict[tuple[str, int], bool] = {}
    for row in low_fa_rows:
        if row.get("matcher") != "audit_hungarian":
            continue
        key = (str(row.get("stable_target_id")), int(row.get("seed")))
        status = bool(row.get("fixed_logit0_matched"))
        if key in observed and observed[key] != status:
            raise LowFABridgeError("Hungarian fixed status varies across budgets")
        observed[key] = status
    if expected != observed:
        raise LowFABridgeError("E-1c fixed Hungarian statuses differ from E-1a")


def _recurrence_table(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], dict[str, int]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["matcher"]),
            int(row["nominal_budget_fa_per_mpix"]),
            str(row["stable_target_id"]),
        )
        grouped[key].append(row)
    result: dict[tuple[str, int, str], dict[str, int]] = {}
    for key, group in grouped.items():
        if len(group) != 3 or len({int(row["seed"]) for row in group}) != 3:
            raise LowFABridgeError("each matcher/budget target needs exactly three seeds")
        datasets = {str(row["dataset"]) for row in group}
        if len(datasets) != 1:
            raise LowFABridgeError("stable target crossed datasets")
        result[key] = {
            "dataset": next(iter(datasets)),
            "fixed_miss_count": sum(
                not bool(row["fixed_logit0_matched"]) for row in group
            ),
            "low_fa_miss_count": sum(not bool(row["low_fa_matched"]) for row in group),
        }
    return result


def summarize_low_fa_bridge(
    target_rows: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    recurrence = _recurrence_table(target_rows)
    aggregate_keys: dict[tuple[str, int, str, int], Mapping[str, Any]] = {}
    for row in image_rows:
        key = (
            str(row["dataset"]),
            int(row["seed"]),
            str(row["matcher"]),
            int(row["nominal_budget_fa_per_mpix"]),
        )
        aggregate = row["dataset_seed_aggregate"]
        if key in aggregate_keys and dict(aggregate_keys[key]) != dict(aggregate):
            raise LowFABridgeError("dataset/seed aggregate varies across images")
        aggregate_keys[key] = aggregate

    matcher_summaries: dict[str, Any] = {}
    operating_points = []
    for (dataset, seed, matcher, budget), aggregate in sorted(
        aggregate_keys.items()
    ):
        operating_points.append(
            {
                "dataset": dataset,
                "seed": seed,
                "matcher": matcher,
                "nominal_budget_fa_per_mpix": budget,
                **dict(aggregate),
            }
        )
    for matcher in MATCHERS:
        matcher_summaries[matcher] = {}
        for budget in BUDGETS:
            by_dataset = {}
            for dataset in sorted({str(row["dataset"]) for row in target_rows}):
                keys = [
                    key
                    for key, value in recurrence.items()
                    if key[0] == matcher
                    and key[1] == budget
                    and value["dataset"] == dataset
                ]
                dataset_seeds = sorted(
                    {
                        key[1]
                        for key in aggregate_keys
                        if key[0] == dataset
                        and key[2] == matcher
                        and key[3] == budget
                    }
                )
                if len(dataset_seeds) != 3:
                    raise LowFABridgeError(
                        "each dataset/matcher/budget requires exactly three seeds"
                    )
                all_seed_feasible = all(
                    bool(aggregate_keys[(dataset, seed, matcher, budget)][
                        "budget_feasible_zero_overshoot"
                    ])
                    for seed in dataset_seeds
                )
                no_all_off_seed = all(
                    int(aggregate_keys[(dataset, seed, matcher, budget)][
                        "matched_components"
                    ])
                    > 0
                    for seed in dataset_seeds
                )
                by_dataset[dataset] = {
                    "target_count": len(keys),
                    "all_three_seeds_budget_feasible": all_seed_feasible,
                    "every_seed_has_at_least_one_match": no_all_off_seed,
                    "low_fa_miss_at_least_two_count": sum(
                        recurrence[key]["low_fa_miss_count"] >= 2 for key in keys
                    ),
                    "fixed0_and_low_fa_miss_at_least_two_count": sum(
                        recurrence[key]["fixed_miss_count"] >= 2
                        and recurrence[key]["low_fa_miss_count"] >= 2
                        for key in keys
                    ),
                    "stable_control_count": sum(
                        recurrence[key]["fixed_miss_count"] <= 1
                        and recurrence[key]["low_fa_miss_count"] <= 1
                        for key in keys
                    ),
                }
            matcher_summaries[matcher][str(budget)] = {"by_dataset": by_dataset}

    joint_budget_records = {}
    datasets = sorted({str(row["dataset"]) for row in target_rows})
    for budget in BUDGETS:
        eligible_datasets = [
            dataset
            for dataset in datasets
            if all(
                matcher_summaries[matcher][str(budget)]["by_dataset"][dataset][
                    "all_three_seeds_budget_feasible"
                ]
                and matcher_summaries[matcher][str(budget)]["by_dataset"][dataset][
                    "every_seed_has_at_least_one_match"
                ]
                for matcher in MATCHERS
            )
        ]
        stable_ids = sorted(
            {
                key[2]
                for key, value in recurrence.items()
                if key[1] == budget and value["dataset"] in eligible_datasets
            }
        )
        bridge_targets = []
        control_targets = []
        for stable_id in stable_ids:
            records = [recurrence[(matcher, budget, stable_id)] for matcher in MATCHERS]
            if all(
                record["fixed_miss_count"] >= 2
                and record["low_fa_miss_count"] >= 2
                for record in records
            ):
                bridge_targets.append(stable_id)
            if all(
                record["fixed_miss_count"] <= 1
                and record["low_fa_miss_count"] <= 1
                for record in records
            ):
                control_targets.append(stable_id)
        passed = (
            len(eligible_datasets) >= MIN_BRIDGE_DATASETS
            and len(bridge_targets) >= MIN_BRIDGE_TARGETS
            and len(control_targets) >= MIN_CONTROL_TARGETS
        )
        joint_budget_records[str(budget)] = {
            "eligible_datasets": eligible_datasets,
            "eligible_dataset_count": len(eligible_datasets),
            "joint_fixed0_and_low_fa_repeated_miss_target_count": len(
                bridge_targets
            ),
            "joint_stable_control_target_count": len(control_targets),
            "bridge_target_ids": bridge_targets,
            "control_target_ids": control_targets,
            "pass": passed,
        }
    passing_budgets = [
        budget for budget in BUDGETS if joint_budget_records[str(budget)]["pass"]
    ]
    return {
        "schema_version": "dea.gate_e.low_fa_bridge_summary.v1",
        "budgets_fa_per_mpix": list(BUDGETS),
        "matchers": list(MATCHERS),
        "operating_points": operating_points,
        "matcher_summaries": matcher_summaries,
        "joint_gate": {
            "same_budget_required": True,
            "minimum_eligible_datasets": MIN_BRIDGE_DATASETS,
            "minimum_joint_bridge_targets": MIN_BRIDGE_TARGETS,
            "minimum_joint_stable_controls": MIN_CONTROL_TARGETS,
            "all_off_veto": "every eligible dataset/seed/matcher must match >=1 target",
            "bridge_target_definition": (
                "fixed-logit0 miss>=2/3 AND cross-fitted low-FA miss>=2/3 "
                "under both official_legacy and audit_hungarian"
            ),
            "by_budget": joint_budget_records,
            "passing_budgets": passing_budgets,
            "selected_lowest_passing_budget": (
                min(passing_budgets) if passing_budgets else None
            ),
            "pass": bool(passing_budgets),
        },
    }
