from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from utils.target_identity import (
    PIXEL_CONNECTIVITY,
    SKIMAGE_CONNECTIVITY,
    TARGET_IDENTITY_SCHEMA_VERSION,
    TargetIdentityMismatchError,
    assert_same_target_identities,
    assert_same_target_set,
    build_stable_target_set,
    canonical_binary_target_mask,
    canonical_mask_sha256,
    enumerate_stable_targets,
    validate_stable_target_set,
    validate_target_identities,
)


def test_diagonal_pixels_form_one_eight_connected_target() -> None:
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[2, 3] = 1
    mask[3, 4] = 1

    target_set = build_stable_target_set(
        mask, dataset="NUAA-SIRST", image_name="sample.png"
    )

    assert target_set.pixel_connectivity == PIXEL_CONNECTIVITY == 8
    assert target_set.skimage_connectivity == SKIMAGE_CONNECTIVITY == 2
    assert len(target_set.targets) == 1
    target = target_set.targets[0]
    assert target.area == 2
    assert target.bbox == (2, 3, 4, 5)
    assert target.centroid_y == pytest.approx(2.5)
    assert target.centroid_x == pytest.approx(3.5)


def test_canonical_order_and_geometry_are_assertion_metadata() -> None:
    mask = np.zeros((12, 13), dtype=bool)
    mask[7:9, 8:11] = True
    mask[1, 9] = True
    mask[1:4, 2] = True

    targets = enumerate_stable_targets(
        mask, dataset="IRSTD-1K", image_name="X0001.png"
    )

    assert [target.component_index for target in targets] == [0, 1, 2]
    assert [target.bbox for target in targets] == [
        (1, 2, 4, 3),
        (1, 9, 2, 10),
        (7, 8, 9, 11),
    ]
    assert [target.area for target in targets] == [3, 1, 6]
    assert [target.source_component_index for target in targets] == [0, 1, 2]
    assert len({target.stable_key for target in targets}) == 3
    assert len({target.component_mask_sha256 for target in targets}) == 3
    assert len({target.label_mask_sha256 for target in targets}) == 1


def test_component_digest_uses_full_size_mask_and_shape_encoding() -> None:
    upper = np.zeros((2, 2), dtype=np.uint8)
    lower = np.zeros_like(upper)
    upper[0, 0] = 1
    lower[1, 0] = 1
    flat = np.asarray([[1, 0, 0, 0]], dtype=np.uint8)

    assert canonical_mask_sha256(upper) != canonical_mask_sha256(lower)
    assert canonical_mask_sha256(upper) != canonical_mask_sha256(flat)
    assert canonical_mask_sha256(upper) == canonical_mask_sha256(upper.astype(bool))


def test_component_digest_has_a_versioned_golden_vector() -> None:
    mask = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)

    assert canonical_mask_sha256(mask) == (
        "0a4ba7676f49bb665ecce4efdb3f7dae4919801d993744ef9fb995dfc2f8f3ae"
    )


def test_stable_key_contains_namespace_and_digest_but_not_component_index() -> None:
    base = np.zeros((8, 8), dtype=bool)
    base[5, 5] = True
    base_target = enumerate_stable_targets(
        base, dataset="D", image_name="I"
    )[0]

    with_earlier_component = base.copy()
    with_earlier_component[1, 1] = True
    shifted_target = enumerate_stable_targets(
        with_earlier_component, dataset="D", image_name="I"
    )[1]

    assert shifted_target.component_index == 1
    assert shifted_target.component_mask_sha256 == base_target.component_mask_sha256
    assert shifted_target.stable_key == base_target.stable_key
    decoded_key = json.loads(base_target.stable_key)
    assert decoded_key == [
        TARGET_IDENTITY_SCHEMA_VERSION,
        "D",
        "I",
        base_target.component_mask_sha256,
    ]
    # The full label digest is an envelope assertion, not part of the stable key.
    assert shifted_target.label_mask_sha256 != base_target.label_mask_sha256


def test_target_set_carries_empty_mask_identity_and_shape() -> None:
    target_set = build_stable_target_set(
        np.zeros((5, 9), dtype=bool), dataset="D", image_name="empty"
    )

    assert target_set.targets == ()
    assert target_set.height == 5
    assert target_set.width == 9
    assert len(target_set.label_mask_sha256) == 64
    assert target_set.as_dict()["targets"] == []
    assert_same_target_set(target_set, target_set)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((2, 2, 1)),
        np.zeros((0, 2)),
        np.asarray([[0.0, np.nan]]),
        np.asarray([[0.0, np.inf]]),
        np.asarray([[0.0, 0.25]]),
        np.asarray([["0", "1"]]),
        np.asarray([[0.0 + 0.0j, 1.0 + 0.0j]]),
    ],
)
def test_invalid_masks_fail_closed(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        canonical_binary_target_mask(mask)


@pytest.mark.parametrize(
    ("dataset", "image_name", "connectivity"),
    [
        ("", "image", 2),
        (None, "image", 2),
        ("dataset", "", 2),
        ("dataset", None, 2),
        ("dataset", "image", 1),
        ("dataset", "image", 8),
    ],
)
def test_invalid_namespace_or_connectivity_fails_closed(
    dataset: object, image_name: object, connectivity: int
) -> None:
    with pytest.raises(ValueError):
        build_stable_target_set(
            np.zeros((2, 2), dtype=bool),
            dataset=dataset,
            image_name=image_name,
            connectivity=connectivity,
        )


def test_exact_sets_match_and_mask_drift_is_rejected() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 3] = True
    reference = build_stable_target_set(mask, dataset="D", image_name="I")
    observed = build_stable_target_set(mask.copy(), dataset="D", image_name="I")
    assert_same_target_set(reference, observed)
    assert_same_target_identities(reference.targets, observed.targets)

    changed = mask.copy()
    changed[3, 4] = True
    drifted = build_stable_target_set(changed, dataset="D", image_name="I")
    with pytest.raises(TargetIdentityMismatchError):
        assert_same_target_set(reference, drifted)


def test_shape_drift_is_rejected_even_for_target_free_masks() -> None:
    reference = build_stable_target_set(
        np.zeros((2, 6), dtype=bool), dataset="D", image_name="I"
    )
    observed = build_stable_target_set(
        np.zeros((3, 4), dtype=bool), dataset="D", image_name="I"
    )

    with pytest.raises(TargetIdentityMismatchError, match="height"):
        assert_same_target_set(reference, observed)


def test_duplicate_and_tampered_records_fail_closed() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[1, 1] = True
    mask[5, 5] = True
    targets = enumerate_stable_targets(mask, dataset="D", image_name="I")

    duplicate = replace(targets[0], component_index=1)
    with pytest.raises(TargetIdentityMismatchError, match="duplicate stable"):
        validate_target_identities([targets[0], duplicate])

    tampered = replace(targets[0], bbox=(0, 0, 1, 1))
    with pytest.raises(TargetIdentityMismatchError):
        assert_same_target_identities([targets[0]], [tampered])

    bad_key = replace(targets[0], stable_key="not-the-derived-key")
    with pytest.raises(TargetIdentityMismatchError, match="stable key"):
        validate_target_identities([bad_key])


def test_tampered_envelope_fails_even_when_both_arguments_match() -> None:
    target_set = build_stable_target_set(
        np.zeros((3, 3), dtype=bool), dataset="D", image_name="I"
    )
    tampered = replace(target_set, pixel_connectivity=4)

    with pytest.raises(TargetIdentityMismatchError, match="connectivity"):
        validate_stable_target_set(tampered)
    with pytest.raises(TargetIdentityMismatchError, match="connectivity"):
        assert_same_target_set(tampered, tampered)
