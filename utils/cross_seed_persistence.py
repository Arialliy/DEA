"""Registry-anchored aggregation of target misses across exactly three seeds.

The authoritative image and target universes come from canonical
``StableTargetSet`` records, never from the observed status rows.  Every seed
must contain one image-envelope row for every registry image (including
target-free images) and one target-status row for every registry target.

Bootstrap intervals resample whole images within datasets.  They quantify
image-sampling uncertainty only; training seeds and dataset domains remain
fixed.  Event-share intervals are explicitly conditional because a bootstrap
replicate can contain no miss events.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from numbers import Integral, Real
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from utils.target_identity import (
    PIXEL_CONNECTIVITY,
    SKIMAGE_CONNECTIVITY,
    TARGET_IDENTITY_SCHEMA_VERSION,
    StableTargetId,
    StableTargetSet,
    validate_stable_target_set,
)


DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260712
EXPECTED_SEED_COUNT = 3
SUPPORTED_STATUS_FIELDS = ("matched", "unmatched")
SUPPORTED_ROW_KINDS = ("image", "target")

_ENVELOPE_FIELDS = (
    "height",
    "width",
    "pixel_connectivity",
    "skimage_connectivity",
    "label_mask_sha256",
)
_TARGET_ASSERTION_FIELDS = (
    "component_index",
    "source_component_index",
    "source_label",
    "bbox",
    "area",
    "centroid_y",
    "centroid_x",
    "component_mask_sha256",
)


def _nonempty_string(value: Any, *, field: str, row_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"row {row_index} field {field!r} must be a non-empty string"
        )
    return value


def _integer(value: Any, *, field: str, row_index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"row {row_index} field {field!r} must be an integer")
    return int(value)


def _finite_number(value: Any, *, field: str, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"row {row_index} field {field!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"row {row_index} field {field!r} must be finite")
    return result


def _binary(value: Any, *, field: str, row_index: int) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral) and not isinstance(value, bool):
        integer = int(value)
        if integer in (0, 1):
            return bool(integer)
    raise ValueError(
        f"row {row_index} field {field!r} must be binary bool/0/1"
    )


def _registry_values(
    expected_registry: Sequence[StableTargetSet]
    | Mapping[tuple[str, str], StableTargetSet],
) -> tuple[StableTargetSet, ...]:
    if isinstance(expected_registry, Mapping):
        values: list[StableTargetSet] = []
        for key, target_set in expected_registry.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(
                    "registry mapping keys must be (dataset, image_name) tuples"
                )
            if not isinstance(target_set, StableTargetSet):
                raise TypeError("registry values must be StableTargetSet records")
            if key != (target_set.dataset, target_set.image_name):
                raise ValueError("registry mapping key disagrees with target set")
            values.append(target_set)
    else:
        if isinstance(expected_registry, (str, bytes)) or not isinstance(
            expected_registry, Sequence
        ):
            raise TypeError(
                "expected_registry must be a StableTargetSet sequence or mapping"
            )
        values = list(expected_registry)
    if not values:
        raise ValueError("expected_registry must be non-empty")

    validated: list[StableTargetSet] = []
    seen_images: set[tuple[str, str]] = set()
    seen_targets: set[str] = set()
    targets_per_dataset: dict[str, int] = defaultdict(int)
    for target_set in values:
        target_set = validate_stable_target_set(target_set)
        image_key = (target_set.dataset, target_set.image_name)
        if image_key in seen_images:
            raise ValueError(f"duplicate registry image: {image_key!r}")
        seen_images.add(image_key)
        for target in target_set.targets:
            if target.stable_key in seen_targets:
                raise ValueError("duplicate registry stable target key")
            seen_targets.add(target.stable_key)
            targets_per_dataset[target_set.dataset] += 1
        validated.append(target_set)

    datasets = {target_set.dataset for target_set in validated}
    empty_target_datasets = sorted(
        dataset for dataset in datasets if targets_per_dataset[dataset] == 0
    )
    if empty_target_datasets:
        raise ValueError(
            "target-persistence statistics require at least one target per dataset: "
            f"{empty_target_datasets}"
        )
    return tuple(sorted(validated, key=lambda value: (value.dataset, value.image_name)))


def _parse_stable_key(
    value: Any,
    *,
    dataset: str,
    image_name: str,
    component_digest: str,
    row_index: int,
) -> str:
    stable_key = _nonempty_string(
        value, field="stable_target_id", row_index=row_index
    )
    try:
        decoded = json.loads(stable_key)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"row {row_index} stable_target_id is not canonical JSON"
        ) from exc
    expected = [
        TARGET_IDENTITY_SCHEMA_VERSION,
        dataset,
        image_name,
        component_digest,
    ]
    if decoded != expected:
        raise ValueError(
            f"row {row_index} stable_target_id namespace/digest disagrees with row"
        )
    return stable_key


def _require_fields(
    row: Mapping[str, Any], fields: Sequence[str], *, row_index: int
) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"row {row_index} missing required fields: {missing}")


def _validate_envelope(
    row: Mapping[str, Any],
    target_set: StableTargetSet,
    *,
    row_index: int,
) -> None:
    observed = {
        "height": _integer(row["height"], field="height", row_index=row_index),
        "width": _integer(row["width"], field="width", row_index=row_index),
        "pixel_connectivity": _integer(
            row["pixel_connectivity"],
            field="pixel_connectivity",
            row_index=row_index,
        ),
        "skimage_connectivity": _integer(
            row["skimage_connectivity"],
            field="skimage_connectivity",
            row_index=row_index,
        ),
        "label_mask_sha256": _nonempty_string(
            row["label_mask_sha256"],
            field="label_mask_sha256",
            row_index=row_index,
        ),
    }
    for field, value in observed.items():
        if value != getattr(target_set, field):
            raise ValueError(
                f"row {row_index} image envelope disagrees with registry on {field}"
            )
    if observed["pixel_connectivity"] != PIXEL_CONNECTIVITY:
        raise ValueError(f"row {row_index} pixel_connectivity must be 8")
    if observed["skimage_connectivity"] != SKIMAGE_CONNECTIVITY:
        raise ValueError(f"row {row_index} skimage_connectivity must be 2")


def _bbox(value: Any, *, row_index: int) -> tuple[int, int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"row {row_index} field 'bbox' must contain four integers")
    if len(value) != 4:
        raise ValueError(f"row {row_index} field 'bbox' must contain four integers")
    return tuple(
        _integer(item, field="bbox", row_index=row_index) for item in value
    )


def _validate_target_assertions(
    row: Mapping[str, Any],
    target: StableTargetId,
    *,
    row_index: int,
) -> None:
    observed: dict[str, Any] = {
        "component_index": _integer(
            row["component_index"], field="component_index", row_index=row_index
        ),
        "source_component_index": _integer(
            row["source_component_index"],
            field="source_component_index",
            row_index=row_index,
        ),
        "source_label": _integer(
            row["source_label"], field="source_label", row_index=row_index
        ),
        "bbox": _bbox(row["bbox"], row_index=row_index),
        "area": _integer(row["area"], field="area", row_index=row_index),
        "centroid_y": _finite_number(
            row["centroid_y"], field="centroid_y", row_index=row_index
        ),
        "centroid_x": _finite_number(
            row["centroid_x"], field="centroid_x", row_index=row_index
        ),
        "component_mask_sha256": _nonempty_string(
            row["component_mask_sha256"],
            field="component_mask_sha256",
            row_index=row_index,
        ),
    }
    for field, value in observed.items():
        if value != getattr(target, field):
            raise ValueError(
                f"row {row_index} target assertion disagrees with registry on {field}"
            )


def _count_summary(targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = [0, 0, 0, 0]
    for target in targets:
        miss_count = int(target["miss_count"])
        if miss_count < 0 or miss_count > EXPECTED_SEED_COUNT:
            raise RuntimeError("validated target escaped the 0..3 miss range")
        counts[miss_count] += 1
    if not targets:
        raise RuntimeError("cannot summarize an empty target collection")
    event_count = counts[1] + 2 * counts[2] + 3 * counts[3]
    return {
        "target_count": len(targets),
        "N0": counts[0],
        "N1": counts[1],
        "N2": counts[2],
        "N3": counts[3],
        "N3_over_N": counts[3] / len(targets),
        "event_count": event_count,
        "persistent_event_share": (
            3 * counts[3] / event_count if event_count else None
        ),
    }


def _dataset_macro(
    by_dataset: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    n3_rates = [float(value["N3_over_N"]) for value in by_dataset.values()]
    event_shares = [
        float(value["persistent_event_share"])
        for value in by_dataset.values()
        if value["persistent_event_share"] is not None
    ]
    dataset_count = len(by_dataset)
    return {
        "dataset_count": dataset_count,
        "N3_over_N": float(np.mean(n3_rates)),
        "N3_over_N_definition": "equal_weight_mean_over_all_datasets",
        "persistent_event_share": (
            float(np.mean(event_shares)) if event_shares else None
        ),
        "persistent_event_share_definition": (
            "equal_weight_mean_over_datasets_with_event_count_gt_0"
        ),
        "persistent_event_share_datasets_defined": len(event_shares),
        "persistent_event_share_datasets_undefined": (
            dataset_count - len(event_shares)
        ),
    }


def _point_summaries(
    targets: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for target in targets:
        grouped[str(target["dataset"])].append(target)
    by_dataset = {
        dataset: _count_summary(grouped[dataset]) for dataset in sorted(grouped)
    }
    return _count_summary(targets), by_dataset, _dataset_macro(by_dataset)


def _percentile_interval(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    low, high = np.quantile(np.asarray(values, dtype=np.float64), (0.025, 0.975))
    return {"low": float(low), "high": float(high)}


def _rate_bootstrap_metric(
    estimate: float, values: Sequence[float], *, replicates: int
) -> dict[str, Any]:
    if len(values) != replicates:
        raise RuntimeError("an always-defined bootstrap rate became undefined")
    return {
        "defined": True,
        "estimate": float(estimate),
        "ci95": _percentile_interval(values),
        "replicates_defined": len(values),
        "replicates_undefined": 0,
        "undefined_reason": None,
    }


def _event_bootstrap_metric(
    estimate: float | None,
    values: Sequence[float],
    *,
    replicates: int,
    conditioning: str,
) -> dict[str, Any]:
    return {
        "defined": estimate is not None,
        "estimate": float(estimate) if estimate is not None else None,
        "conditional_ci95": _percentile_interval(values),
        "replicates_defined": len(values),
        "replicates_undefined": replicates - len(values),
        "conditioning": conditioning,
        "undefined_reason": None if estimate is not None else "no_miss_events",
    }


def _bootstrap_joint(
    targets: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if isinstance(replicates, bool) or not isinstance(replicates, Integral):
        raise ValueError("bootstrap_replicates must be an integer")
    replicates = int(replicates)
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("bootstrap_seed must be an integer")
    seed = int(seed)

    clusters: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for target in targets:
        clusters[(str(target["dataset"]), str(target["image_name"]))].append(
            target
        )
    strata: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cluster in clusters:
        strata[cluster[0]].append(cluster)
    for dataset in strata:
        strata[dataset].sort()

    micro, _, macro = _point_summaries(targets)
    micro_n3: list[float] = []
    micro_events: list[float] = []
    macro_n3: list[float] = []
    macro_events: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled_targets: list[Mapping[str, Any]] = []
        for dataset in sorted(strata):
            dataset_clusters = strata[dataset]
            selected = rng.integers(
                0, len(dataset_clusters), size=len(dataset_clusters)
            )
            for cluster_index in selected:
                sampled_targets.extend(
                    clusters[dataset_clusters[int(cluster_index)]]
                )
        sampled_micro, _, sampled_macro = _point_summaries(sampled_targets)
        micro_n3.append(float(sampled_micro["N3_over_N"]))
        if sampled_micro["persistent_event_share"] is not None:
            micro_events.append(float(sampled_micro["persistent_event_share"]))
        macro_n3.append(float(sampled_macro["N3_over_N"]))
        if sampled_macro["persistent_event_share"] is not None:
            macro_events.append(float(sampled_macro["persistent_event_share"]))

    images_per_dataset = {
        dataset: len(strata[dataset]) for dataset in sorted(strata)
    }
    return {
        "method": "dataset_stratified_image_cluster_percentile_bootstrap",
        "sampling_unit": ["dataset", "image_name"],
        "stratified_by": "dataset",
        "confidence_level": 0.95,
        "replicates_requested": replicates,
        "seed": seed,
        "image_cluster_count": len(clusters),
        "target_count": len(targets),
        "images_per_dataset": images_per_dataset,
        "cluster_population": "registry_images_with_at_least_one_target",
        "estimand": "target_level_recurrence_conditional_on_a_registry_target",
        "target_free_images": (
            "validated in every seed but absent from the target-level "
            "bootstrap observation population"
        ),
        "seed_statuses_per_target": EXPECTED_SEED_COUNT,
        "training_seeds_resampled": False,
        "uncertainty_scope": "image_sampling_only",
        "excluded_uncertainty": [
            "training_seed_sampling",
            "dataset_domain_sampling",
        ],
        "metrics": {
            "target_micro": {
                "N3_over_N": _rate_bootstrap_metric(
                    float(micro["N3_over_N"]), micro_n3, replicates=replicates
                ),
                "persistent_event_share": _event_bootstrap_metric(
                    micro["persistent_event_share"],
                    micro_events,
                    replicates=replicates,
                    conditioning="replicate event_count > 0",
                ),
            },
            "dataset_macro": {
                "N3_over_N": _rate_bootstrap_metric(
                    float(macro["N3_over_N"]), macro_n3, replicates=replicates
                ),
                "persistent_event_share": _event_bootstrap_metric(
                    macro["persistent_event_share"],
                    macro_events,
                    replicates=replicates,
                    conditioning=(
                        "at least one dataset has replicate event_count > 0; "
                        "the replicate macro averages defined datasets only"
                    ),
                ),
            },
        },
    }


def _validate_and_aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_registry: Sequence[StableTargetSet]
    | Mapping[tuple[str, str], StableTargetSet],
    status_field: Literal["matched", "unmatched"],
) -> tuple[
    list[dict[str, Any]],
    tuple[int, int, int],
    tuple[StableTargetSet, ...],
]:
    if status_field not in SUPPORTED_STATUS_FIELDS:
        raise ValueError("status_field must be exactly 'matched' or 'unmatched'")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("rows must be a non-empty sequence of mappings")
    if not rows:
        raise ValueError("rows must be non-empty")
    registry = _registry_values(expected_registry)
    images = {(item.dataset, item.image_name): item for item in registry}
    targets = {
        target.stable_key: target
        for target_set in registry
        for target in target_set.targets
    }
    expected_images = set(images)
    expected_targets = set(targets)

    seen_image_rows: set[tuple[int, str, str]] = set()
    seen_target_rows: set[tuple[int, str]] = set()
    observed_misses: dict[tuple[int, str], bool] = {}
    observed_seeds: set[int] = set()

    base_fields = (
        "row_kind",
        "dataset",
        "image_name",
        "seed",
        *_ENVELOPE_FIELDS,
    )
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_index} must be a mapping")
        _require_fields(row, base_fields, row_index=row_index)
        row_kind = _nonempty_string(
            row["row_kind"], field="row_kind", row_index=row_index
        )
        if row_kind not in SUPPORTED_ROW_KINDS:
            raise ValueError(f"row {row_index} row_kind must be 'image' or 'target'")
        dataset = _nonempty_string(
            row["dataset"], field="dataset", row_index=row_index
        )
        image_name = _nonempty_string(
            row["image_name"], field="image_name", row_index=row_index
        )
        seed = _integer(row["seed"], field="seed", row_index=row_index)
        observed_seeds.add(seed)
        image_key = (dataset, image_name)
        if image_key not in images:
            raise ValueError(f"row {row_index} image is absent from registry")
        target_set = images[image_key]
        _validate_envelope(row, target_set, row_index=row_index)

        if row_kind == "image":
            forbidden = {
                *SUPPORTED_STATUS_FIELDS,
                "stable_target_id",
                *_TARGET_ASSERTION_FIELDS,
            }
            present = sorted(field for field in forbidden if field in row)
            if present:
                raise ValueError(
                    f"row {row_index} image row contains target-only fields: {present}"
                )
            observation_key = (seed, dataset, image_name)
            if observation_key in seen_image_rows:
                raise ValueError(
                    f"duplicate seed/image envelope row: {observation_key!r}"
                )
            seen_image_rows.add(observation_key)
            continue

        target_fields = (
            "stable_target_id",
            *_TARGET_ASSERTION_FIELDS,
            status_field,
        )
        _require_fields(row, target_fields, row_index=row_index)
        component_digest = _nonempty_string(
            row["component_mask_sha256"],
            field="component_mask_sha256",
            row_index=row_index,
        )
        stable_key = _parse_stable_key(
            row["stable_target_id"],
            dataset=dataset,
            image_name=image_name,
            component_digest=component_digest,
            row_index=row_index,
        )
        if stable_key not in targets:
            raise ValueError(f"row {row_index} target is absent from registry")
        target = targets[stable_key]
        if (target.dataset, target.image_name) != image_key:
            raise ValueError(f"row {row_index} target namespace disagrees with image")
        _validate_target_assertions(row, target, row_index=row_index)

        status = _binary(row[status_field], field=status_field, row_index=row_index)
        other_status_field = "unmatched" if status_field == "matched" else "matched"
        if other_status_field in row:
            other_status = _binary(
                row[other_status_field],
                field=other_status_field,
                row_index=row_index,
            )
            if other_status == status:
                raise ValueError(
                    f"row {row_index} matched and unmatched fields must be complements"
                )
        target_observation_key = (seed, stable_key)
        if target_observation_key in seen_target_rows:
            raise ValueError(
                f"duplicate seed/stable_target_id row: {target_observation_key!r}"
            )
        seen_target_rows.add(target_observation_key)
        observed_misses[target_observation_key] = (
            not status if status_field == "matched" else status
        )

    seeds = tuple(sorted(observed_seeds))
    if len(seeds) != EXPECTED_SEED_COUNT:
        raise ValueError(
            f"exactly {EXPECTED_SEED_COUNT} distinct seeds are required; "
            f"got {len(seeds)}"
        )
    for seed in seeds:
        seed_images = {
            (dataset, image_name)
            for observed_seed, dataset, image_name in seen_image_rows
            if observed_seed == seed
        }
        if seed_images != expected_images:
            raise ValueError(
                "image universe differs from authoritative registry: "
                f"seed={seed}, missing={len(expected_images - seed_images)}, "
                f"extra={len(seed_images - expected_images)}"
            )
        seed_targets = {
            stable_key
            for observed_seed, stable_key in seen_target_rows
            if observed_seed == seed
        }
        if seed_targets != expected_targets:
            raise ValueError(
                "target universe differs from authoritative registry: "
                f"seed={seed}, missing={len(expected_targets - seed_targets)}, "
                f"extra={len(seed_targets - expected_targets)}"
            )

    aggregated: list[dict[str, Any]] = []
    for target_set in registry:
        for target in target_set.targets:
            aggregated.append(
                {
                    "dataset": target.dataset,
                    "image_name": target.image_name,
                    "stable_target_id": target.stable_key,
                    "miss_count": sum(
                        observed_misses[(seed, target.stable_key)] for seed in seeds
                    ),
                }
            )
    return aggregated, seeds, registry


def summarize_cross_seed_persistence(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_registry: Sequence[StableTargetSet]
    | Mapping[tuple[str, str], StableTargetSet],
    status_field: Literal["matched", "unmatched"] = "matched",
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate registry-complete rows and summarize three-seed recurrence.

    ``row_kind="image"`` records prove each seed's image universe, including
    target-free images.  ``row_kind="target"`` records add the canonical
    component identity assertions and one binary match status.
    """

    targets, seeds, registry = _validate_and_aggregate(
        rows,
        expected_registry=expected_registry,
        status_field=status_field,
    )
    micro, dataset_points, macro = _point_summaries(targets)
    overall = {
        "target_micro": micro,
        "dataset_macro": macro,
        "bootstrap": _bootstrap_joint(
            targets,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
    }
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted(dataset_points):
        dataset_targets = [
            target for target in targets if target["dataset"] == dataset
        ]
        by_dataset[dataset] = {
            "target_micro": dataset_points[dataset],
            "bootstrap": _bootstrap_joint(
                dataset_targets,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
        }

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        seed_target_rows = [
            row
            for row in rows
            if row["row_kind"] == "target" and int(row["seed"]) == seed
        ]
        miss_by_dataset: dict[str, int] = defaultdict(int)
        for row_index, row in enumerate(seed_target_rows):
            status = _binary(
                row[status_field], field=status_field, row_index=row_index
            )
            missed = not status if status_field == "matched" else status
            if missed:
                miss_by_dataset[str(row["dataset"])] += 1
        per_seed.append(
            {
                "seed": seed,
                "image_count": len(registry),
                "target_count": len(targets),
                "miss_count": sum(miss_by_dataset.values()),
                "miss_count_by_dataset": {
                    dataset: miss_by_dataset.get(dataset, 0)
                    for dataset in sorted(dataset_points)
                },
            }
        )

    image_count_by_dataset: dict[str, int] = defaultdict(int)
    target_free_count_by_dataset: dict[str, int] = defaultdict(int)
    for target_set in registry:
        image_count_by_dataset[target_set.dataset] += 1
        if not target_set.targets:
            target_free_count_by_dataset[target_set.dataset] += 1
    return {
        "schema_version": "dea.cross_seed_target_persistence_summary.v2",
        "status_field": status_field,
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "registry": {
            "image_count": len(registry),
            "target_count": len(targets),
            "target_free_image_count": sum(
                target_free_count_by_dataset.values()
            ),
            "image_count_by_dataset": {
                dataset: image_count_by_dataset[dataset]
                for dataset in sorted(image_count_by_dataset)
            },
            "target_free_image_count_by_dataset": {
                dataset: target_free_count_by_dataset.get(dataset, 0)
                for dataset in sorted(image_count_by_dataset)
            },
            "identity_authority": "canonical_stable_target_set_registry",
            "pixel_connectivity": PIXEL_CONNECTIVITY,
            "skimage_connectivity": SKIMAGE_CONNECTIVITY,
        },
        "overall": overall,
        "by_dataset": by_dataset,
        "per_seed": per_seed,
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EXPECTED_SEED_COUNT",
    "summarize_cross_seed_persistence",
]
