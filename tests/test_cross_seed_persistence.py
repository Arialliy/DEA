from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest

from utils.cross_seed_persistence import summarize_cross_seed_persistence
from utils.target_identity import (
    build_stable_target_set,
    canonical_mask_sha256,
)


SEEDS = (11, 22, 33)


def _mask(shape, points):
    result = np.zeros(shape, dtype=np.uint8)
    for y, x in points:
        result[y, x] = 1
    return result


def _four_target_registry():
    return [
        build_stable_target_set(
            _mask((7, 8), [(1, 1), (5, 6)]),
            dataset="A",
            image_name="a0",
        ),
        build_stable_target_set(
            _mask((7, 8), [(3, 3)]),
            dataset="A",
            image_name="a1",
        ),
        build_stable_target_set(
            _mask((7, 8), []),
            dataset="A",
            image_name="a_empty",
        ),
        build_stable_target_set(
            _mask((6, 6), [(2, 4)]),
            dataset="B",
            image_name="b0",
        ),
    ]


def _target_rows(registry):
    return [target for target_set in registry for target in target_set.targets]


def _rows(registry, missed=None, *, status_field="matched"):
    missed = missed or {}
    rows = []
    for seed in SEEDS:
        for target_set in registry:
            envelope = {
                "dataset": target_set.dataset,
                "image_name": target_set.image_name,
                "seed": seed,
                "height": target_set.height,
                "width": target_set.width,
                "pixel_connectivity": target_set.pixel_connectivity,
                "skimage_connectivity": target_set.skimage_connectivity,
                "label_mask_sha256": target_set.label_mask_sha256,
            }
            rows.append({"row_kind": "image", **envelope})
            for target in target_set.targets:
                is_missed = seed in missed.get(target.stable_key, set())
                status = not is_missed if status_field == "matched" else is_missed
                rows.append(
                    {
                        "row_kind": "target",
                        **envelope,
                        "stable_target_id": target.stable_key,
                        "component_index": target.component_index,
                        "source_component_index": target.source_component_index,
                        "source_label": target.source_label,
                        "bbox": list(target.bbox),
                        "area": target.area,
                        "centroid_y": target.centroid_y,
                        "centroid_x": target.centroid_x,
                        "component_mask_sha256": target.component_mask_sha256,
                        status_field: status,
                    }
                )
    return rows


def _recurrence_fixture():
    registry = _four_target_registry()
    targets = _target_rows(registry)
    missed = {
        targets[0].stable_key: set(),
        targets[1].stable_key: {11},
        targets[2].stable_key: {11, 22},
        targets[3].stable_key: {11, 22, 33},
    }
    return registry, _rows(registry, missed), targets


def test_registry_anchored_counts_micro_macro_and_seed_counts() -> None:
    registry, rows, _ = _recurrence_fixture()

    result = summarize_cross_seed_persistence(
        rows,
        expected_registry=registry,
        bootstrap_replicates=40,
        bootstrap_seed=7,
    )

    assert result["seeds"] == [11, 22, 33]
    assert result["registry"] == {
        "image_count": 4,
        "target_count": 4,
        "target_free_image_count": 1,
        "image_count_by_dataset": {"A": 3, "B": 1},
        "target_free_image_count_by_dataset": {"A": 1, "B": 0},
        "identity_authority": "canonical_stable_target_set_registry",
        "pixel_connectivity": 8,
        "skimage_connectivity": 2,
    }
    micro = result["overall"]["target_micro"]
    assert [micro[f"N{index}"] for index in range(4)] == [1, 1, 1, 1]
    assert micro["N3_over_N"] == pytest.approx(0.25)
    assert micro["event_count"] == 6
    assert micro["persistent_event_share"] == pytest.approx(0.5)

    assert result["by_dataset"]["A"]["target_micro"]["N3"] == 0
    assert result["by_dataset"]["B"]["target_micro"]["N3"] == 1
    macro = result["overall"]["dataset_macro"]
    assert macro["N3_over_N"] == pytest.approx(0.5)
    assert macro["persistent_event_share"] == pytest.approx(0.5)
    assert macro["persistent_event_share_datasets_defined"] == 2
    assert macro["persistent_event_share_datasets_undefined"] == 0
    assert [item["miss_count"] for item in result["per_seed"]] == [3, 2, 1]
    assert result["per_seed"][0]["miss_count_by_dataset"] == {"A": 2, "B": 1}


def test_macro_event_share_reports_zero_event_datasets_as_undefined() -> None:
    registry = _four_target_registry()
    targets = _target_rows(registry)
    missed = {targets[-1].stable_key: set(SEEDS)}
    result = summarize_cross_seed_persistence(
        _rows(registry, missed),
        expected_registry=registry,
        bootstrap_replicates=20,
        bootstrap_seed=4,
    )

    macro = result["overall"]["dataset_macro"]
    assert macro["persistent_event_share"] == pytest.approx(1.0)
    assert macro["persistent_event_share_datasets_defined"] == 1
    assert macro["persistent_event_share_datasets_undefined"] == 1
    assert macro["persistent_event_share_definition"] == (
        "equal_weight_mean_over_datasets_with_event_count_gt_0"
    )


def test_unmatched_schema_and_no_event_conditioning_are_explicit() -> None:
    registry = _four_target_registry()
    result = summarize_cross_seed_persistence(
        _rows(registry, status_field="unmatched"),
        expected_registry=registry,
        status_field="unmatched",
        bootstrap_replicates=20,
        bootstrap_seed=3,
    )

    assert result["overall"]["target_micro"]["N0"] == 4
    metric = result["overall"]["bootstrap"]["metrics"]["target_micro"][
        "persistent_event_share"
    ]
    assert "ci95" not in metric
    assert metric == {
        "defined": False,
        "estimate": None,
        "conditional_ci95": None,
        "replicates_defined": 0,
        "replicates_undefined": 20,
        "conditioning": "replicate event_count > 0",
        "undefined_reason": "no_miss_events",
    }


def test_event_bootstrap_reports_partially_undefined_replicates_conditionally() -> None:
    registry = [
        build_stable_target_set(
            _mask((5, 5), [(1, 1)]), dataset="A", image_name="easy"
        ),
        build_stable_target_set(
            _mask((5, 5), [(3, 3)]), dataset="A", image_name="hard"
        ),
    ]
    hard = registry[1].targets[0]
    rows = _rows(registry, {hard.stable_key: set(SEEDS)})

    result = summarize_cross_seed_persistence(
        rows,
        expected_registry=registry,
        bootstrap_replicates=200,
        bootstrap_seed=19,
    )
    metric = result["overall"]["bootstrap"]["metrics"]["target_micro"][
        "persistent_event_share"
    ]

    assert "ci95" not in metric
    assert metric["conditional_ci95"] == {"low": 1.0, "high": 1.0}
    assert 0 < metric["replicates_undefined"] < 200
    assert metric["replicates_defined"] + metric["replicates_undefined"] == 200
    assert metric["conditioning"] == "replicate event_count > 0"


def test_joint_stratified_bootstrap_reports_micro_and_macro() -> None:
    registry, rows, _ = _recurrence_fixture()
    result = summarize_cross_seed_persistence(
        rows,
        expected_registry=registry,
        bootstrap_replicates=80,
        bootstrap_seed=29,
    )
    bootstrap = result["overall"]["bootstrap"]

    assert bootstrap["sampling_unit"] == ["dataset", "image_name"]
    assert bootstrap["stratified_by"] == "dataset"
    assert bootstrap["images_per_dataset"] == {"A": 2, "B": 1}
    assert bootstrap["cluster_population"] == (
        "registry_images_with_at_least_one_target"
    )
    assert set(bootstrap["metrics"]) == {"target_micro", "dataset_macro"}
    assert bootstrap["metrics"]["target_micro"]["N3_over_N"][
        "estimate"
    ] == pytest.approx(0.25)
    assert bootstrap["metrics"]["dataset_macro"]["N3_over_N"][
        "estimate"
    ] == pytest.approx(0.5)
    assert bootstrap["excluded_uncertainty"] == [
        "training_seed_sampling",
        "dataset_domain_sampling",
    ]


def test_shuffle_does_not_change_validation_counts_or_bootstrap() -> None:
    registry, rows, _ = _recurrence_fixture()
    shuffled = copy.deepcopy(rows)
    random.Random(73).shuffle(shuffled)

    first = summarize_cross_seed_persistence(
        rows,
        expected_registry=registry,
        bootstrap_replicates=60,
        bootstrap_seed=17,
    )
    second = summarize_cross_seed_persistence(
        shuffled,
        expected_registry={
            (item.dataset, item.image_name): item for item in reversed(registry)
        },
        bootstrap_replicates=60,
        bootstrap_seed=17,
    )
    assert first == second


def test_unequal_targets_per_image_remain_whole_image_clusters() -> None:
    registry = [
        build_stable_target_set(
            _mask((7, 7), [(1, 1)]), dataset="A", image_name="one"
        ),
        build_stable_target_set(
            _mask((7, 7), [(1, 1), (1, 5), (5, 3)]),
            dataset="A",
            image_name="three",
        ),
    ]
    missed = {
        target.stable_key: set(SEEDS) for target in registry[1].targets
    }
    result = summarize_cross_seed_persistence(
        _rows(registry, missed),
        expected_registry=registry,
        bootstrap_replicates=7,
        bootstrap_seed=6,
    )
    bootstrap = result["overall"]["bootstrap"]

    assert bootstrap["image_cluster_count"] == 2
    assert bootstrap["target_count"] == 4
    # For this frozen RNG stream, all seven two-image draws select one copy of
    # each image.  Keeping the three hard targets together therefore gives
    # exactly 3/4 in every replicate; target-row resampling would not preserve
    # that result.
    assert bootstrap["metrics"]["target_micro"]["N3_over_N"]["ci95"] == {
        "low": 0.75,
        "high": 0.75,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda row: row.__setitem__("label_mask_sha256", "0" * 64),
            "label_mask_sha256",
        ),
        (lambda row: row.__setitem__("height", row["height"] + 1), "height"),
        (lambda row: row.__setitem__("width", row["width"] + 1), "width"),
        (lambda row: row.__setitem__("pixel_connectivity", 4), "pixel_connectivity"),
        (
            lambda row: row.__setitem__("skimage_connectivity", 1),
            "skimage_connectivity",
        ),
        (lambda row: row.__setitem__("bbox", [0, 0, 1, 1]), "bbox"),
        (lambda row: row.__setitem__("area", row["area"] + 1), "area"),
        (lambda row: row.__setitem__("centroid_y", 0.25), "centroid_y"),
        (
            lambda row: row.__setitem__("component_index", row["component_index"] + 1),
            "component_index",
        ),
        (
            lambda row: row.__setitem__(
                "source_component_index", row["source_component_index"] + 1
            ),
            "source_component_index",
        ),
        (
            lambda row: row.__setitem__("source_label", row["source_label"] + 1),
            "source_label",
        ),
    ],
)
def test_registry_metadata_drift_fails_closed(mutate, message) -> None:
    registry = _four_target_registry()
    rows = _rows(registry)
    target_row = next(row for row in rows if row["row_kind"] == "target")
    mutate(target_row)

    with pytest.raises(ValueError, match=message):
        summarize_cross_seed_persistence(
            rows, expected_registry=registry, bootstrap_replicates=5
        )


def test_forged_stable_key_namespace_or_digest_fails_closed() -> None:
    registry = _four_target_registry()
    rows = _rows(registry)
    target_row = next(row for row in rows if row["row_kind"] == "target")
    decoded = json.loads(target_row["stable_target_id"])
    decoded[1] = "forged-dataset"
    target_row["stable_target_id"] = json.dumps(decoded, separators=(",", ":"))

    with pytest.raises(ValueError, match="namespace/digest"):
        summarize_cross_seed_persistence(
            rows, expected_registry=registry, bootstrap_replicates=5
        )


@pytest.mark.parametrize("field", ["source_component_index", "source_label"])
def test_missing_source_index_assertion_fails_closed(field) -> None:
    registry = _four_target_registry()
    rows = _rows(registry)
    target_row = next(row for row in rows if row["row_kind"] == "target")
    del target_row[field]

    with pytest.raises(ValueError, match="missing required fields"):
        summarize_cross_seed_persistence(
            rows, expected_registry=registry, bootstrap_replicates=5
        )


def test_component_digest_drift_fails_before_registry_merge() -> None:
    registry = _four_target_registry()
    rows = _rows(registry)
    target_row = next(row for row in rows if row["row_kind"] == "target")
    target_row["component_mask_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="namespace/digest"):
        summarize_cross_seed_persistence(
            rows, expected_registry=registry, bootstrap_replicates=5
        )


def test_three_seeds_cannot_jointly_omit_one_registry_target() -> None:
    registry = _four_target_registry()
    rows = _rows(registry)
    omitted_key = _target_rows(registry)[0].stable_key
    rows = [
        row
        for row in rows
        if not (
            row["row_kind"] == "target"
            and row["stable_target_id"] == omitted_key
        )
    ]

    with pytest.raises(ValueError, match="target universe"):
        summarize_cross_seed_persistence(
            rows, expected_registry=registry, bootstrap_replicates=5
        )


def test_three_seeds_cannot_jointly_omit_a_whole_image() -> None:
    registry = _four_target_registry()
    rows = [row for row in _rows(registry) if row["image_name"] != "a1"]

    with pytest.raises(ValueError, match="image universe"):
        summarize_cross_seed_persistence(
            rows, expected_registry=registry, bootstrap_replicates=5
        )


def test_target_free_registry_image_must_have_an_envelope_in_every_seed() -> None:
    registry = _four_target_registry()
    rows = _rows(registry)
    rows = [
        row
        for row in rows
        if not (row["image_name"] == "a_empty" and row["seed"] == 22)
    ]

    with pytest.raises(ValueError, match="image universe"):
        summarize_cross_seed_persistence(
            rows, expected_registry=registry, bootstrap_replicates=5
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate seed/image"),
        (
            lambda rows: rows.__setitem__(
                next(i for i, row in enumerate(rows) if row["row_kind"] == "target"),
                {
                    **next(row for row in rows if row["row_kind"] == "target"),
                    "matched": 0.5,
                },
            ),
            "binary",
        ),
    ],
)
def test_duplicate_or_nonbinary_rows_fail_closed(mutation, message) -> None:
    registry = _four_target_registry()
    rows = _rows(registry)
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        summarize_cross_seed_persistence(
            rows, expected_registry=registry, bootstrap_replicates=5
        )


def test_registry_mapping_key_must_match_envelope() -> None:
    registry = _four_target_registry()
    with pytest.raises(ValueError, match="mapping key disagrees"):
        summarize_cross_seed_persistence(
            _rows(registry),
            expected_registry={("wrong", "key"): registry[0]},
            bootstrap_replicates=5,
        )


@pytest.mark.parametrize("replicates", [0, -1, 1.5, True])
def test_invalid_bootstrap_replicates_fail_closed(replicates) -> None:
    registry = _four_target_registry()
    with pytest.raises(ValueError, match="bootstrap_replicates"):
        summarize_cross_seed_persistence(
            _rows(registry),
            expected_registry=registry,
            bootstrap_replicates=replicates,
        )


def test_mask_digest_encoding_golden_vector() -> None:
    mask = np.asarray([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    assert canonical_mask_sha256(mask) == (
        "635eae71a8fedfab1b97f5123f14882114d16e44178551733c8a0e3bdf20cbdd"
    )
