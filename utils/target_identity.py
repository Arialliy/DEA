"""Canonical target identities for cross-run component audits.

The stable identity of a target is the SHA256 digest of its *full-image*
component mask, namespaced by the dataset and image name.  Component order,
bounds, area, and centroid are retained as assertions and audit metadata; none
of them is trusted as the primary identity.  This distinction matters for
small-target masks, where a one-pixel resize change can otherwise be hidden by
rounded geometry.

All connected components use 8-neighbour connectivity.  In scikit-image's
2-D convention this is ``connectivity=2``; every other value is rejected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import struct
from typing import Iterable, Sequence

import numpy as np
from skimage import measure


TARGET_IDENTITY_SCHEMA_VERSION = "dea-gate-e-target-identity-v1"
TARGET_MASK_ENCODING_VERSION = "dea-gate-e-target-mask-v1"
SKIMAGE_CONNECTIVITY = 2
PIXEL_CONNECTIVITY = 8

# The digest preimage is deliberately specified here rather than relying on
# NumPy's .npy format (whose metadata representation may change by version).
# Dimensions are unsigned, big-endian 64-bit integers followed by row-major
# uint8 pixels.  The versioned ASCII prefix separates this encoding from any
# future representation.
_MASK_DIGEST_PREFIX = (
    TARGET_MASK_ENCODING_VERSION.encode("ascii")
    + b"\0rank=2\0dtype=uint8\0order=C\0shape=uint64be\0"
)


class TargetIdentityMismatchError(RuntimeError):
    """Raised when target identities or their assertion metadata disagree."""


@dataclass(frozen=True)
class StableTargetId:
    """One canonical 8-connected target component.

    ``component_index`` is the zero-based position after canonical sorting.
    ``source_component_index`` and ``source_label`` retain scikit-image's
    raster-order bookkeeping only for assertions.  The stable key is derived
    exclusively from ``dataset``, ``image_name``, and
    ``component_mask_sha256`` under a versioned schema.
    """

    schema_version: str
    dataset: str
    image_name: str
    component_index: int
    source_component_index: int
    source_label: int
    height: int
    width: int
    bbox: tuple[int, int, int, int]
    area: int
    centroid_y: float
    centroid_x: float
    component_mask_sha256: str
    label_mask_sha256: str
    stable_key: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable audit record."""

        return asdict(self)


@dataclass(frozen=True)
class StableTargetSet:
    """Image-level identity envelope, including target-free masks."""

    schema_version: str
    mask_encoding_version: str
    dataset: str
    image_name: str
    height: int
    width: int
    pixel_connectivity: int
    skimage_connectivity: int
    label_mask_sha256: str
    targets: tuple[StableTargetId, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable image-level record."""

        payload = asdict(self)
        payload["targets"] = [target.as_dict() for target in self.targets]
        return payload


def _validated_name(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def canonical_binary_target_mask(mask: object) -> np.ndarray:
    """Return a private, read-only, C-contiguous 2-D boolean mask.

    Inputs must already be exactly binary.  Thresholding probabilities here
    would make the identity depend on an implicit policy, so non-binary,
    non-finite, empty, or non-numeric inputs fail closed.
    """

    if hasattr(mask, "detach") and hasattr(mask, "cpu"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("mask must have non-zero height and width")
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError("mask must contain real numeric or boolean values")
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise ValueError("mask must contain real numeric or boolean values") from exc
    if not bool(np.all(finite)):
        raise ValueError("mask must contain only finite values")
    if not bool(np.all((array == 0) | (array == 1))):
        raise ValueError("mask must be binary (0/1 or boolean)")

    canonical = np.array(array, dtype=np.bool_, order="C", copy=True)
    canonical.setflags(write=False)
    return canonical


def canonical_mask_sha256(mask: object) -> str:
    """Digest a binary mask using the versioned, shape-aware encoding."""

    canonical = canonical_binary_target_mask(mask)
    height, width = (int(value) for value in canonical.shape)
    shape_bytes = struct.pack(">QQ", height, width)
    pixel_bytes = np.asarray(canonical, dtype=np.uint8).tobytes(order="C")
    return hashlib.sha256(
        _MASK_DIGEST_PREFIX + shape_bytes + pixel_bytes
    ).hexdigest()


def _stable_key(dataset: str, image_name: str, component_digest: str) -> str:
    # A canonical JSON array makes the three fields recoverable and avoids
    # delimiter collisions without hashing away the required key contents.
    return json.dumps(
        [
            TARGET_IDENTITY_SCHEMA_VERSION,
            dataset,
            image_name,
            component_digest,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _canonical_sort_key(target: StableTargetId) -> tuple[object, ...]:
    return (
        *target.bbox,
        target.area,
        target.component_mask_sha256,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_target_identities(
    targets: Iterable[StableTargetId],
) -> tuple[StableTargetId, ...]:
    """Validate identity uniqueness, ordering, and assertion metadata.

    This cannot reconstruct a component mask from metadata alone.  Digest-to-
    mask verification is therefore performed by :func:`build_stable_target_set`;
    this validator protects serialized or merged identity records from silent
    duplication and metadata drift.
    """

    result = tuple(targets)
    if any(not isinstance(target, StableTargetId) for target in result):
        raise TypeError("targets must contain only StableTargetId records")
    if not result:
        return result

    first = result[0]
    common_fields = (
        "schema_version",
        "dataset",
        "image_name",
        "height",
        "width",
        "label_mask_sha256",
    )
    for target in result:
        for field in common_fields:
            if getattr(target, field) != getattr(first, field):
                raise TargetIdentityMismatchError(
                    f"target set disagrees on {field}"
                )
        if target.schema_version != TARGET_IDENTITY_SCHEMA_VERSION:
            raise TargetIdentityMismatchError("unknown target identity schema")
        if target.height <= 0 or target.width <= 0:
            raise TargetIdentityMismatchError("target shape must be positive")
        if not _is_sha256(target.component_mask_sha256):
            raise TargetIdentityMismatchError(
                "component_mask_sha256 is not a lowercase SHA256 digest"
            )
        if not _is_sha256(target.label_mask_sha256):
            raise TargetIdentityMismatchError(
                "label_mask_sha256 is not a lowercase SHA256 digest"
            )
        if target.stable_key != _stable_key(
            target.dataset,
            target.image_name,
            target.component_mask_sha256,
        ):
            raise TargetIdentityMismatchError("stable key disagrees with identity")
        if target.area <= 0:
            raise TargetIdentityMismatchError("component area must be positive")
        top, left, bottom, right = target.bbox
        if not (
            0 <= top < bottom <= target.height
            and 0 <= left < right <= target.width
        ):
            raise TargetIdentityMismatchError("component bbox is outside its mask")
        if target.area > (bottom - top) * (right - left):
            raise TargetIdentityMismatchError("component area exceeds its bbox")
        if not (
            top <= target.centroid_y < bottom
            and left <= target.centroid_x < right
        ):
            raise TargetIdentityMismatchError(
                "component centroid is outside its bbox"
            )
        if target.source_component_index < 0 or target.source_label <= 0:
            raise TargetIdentityMismatchError(
                "source component metadata must be non-negative/positive"
            )

    expected_indices = tuple(range(len(result)))
    actual_indices = tuple(target.component_index for target in result)
    if actual_indices != expected_indices:
        raise TargetIdentityMismatchError(
            "canonical component indices must be contiguous and ordered"
        )
    if tuple(_canonical_sort_key(target) for target in result) != tuple(
        sorted(_canonical_sort_key(target) for target in result)
    ):
        raise TargetIdentityMismatchError("targets are not canonically sorted")

    stable_keys = [target.stable_key for target in result]
    component_digests = [target.component_mask_sha256 for target in result]
    source_indices = [target.source_component_index for target in result]
    source_labels = [target.source_label for target in result]
    if len(set(stable_keys)) != len(stable_keys):
        raise TargetIdentityMismatchError("duplicate stable target key")
    if len(set(component_digests)) != len(component_digests):
        raise TargetIdentityMismatchError("duplicate component-mask digest")
    if len(set(source_indices)) != len(source_indices):
        raise TargetIdentityMismatchError("duplicate source component index")
    if len(set(source_labels)) != len(source_labels):
        raise TargetIdentityMismatchError("duplicate source component label")
    return result


def validate_stable_target_set(target_set: StableTargetSet) -> StableTargetSet:
    """Validate an image envelope and its component-level assertions."""

    if not isinstance(target_set, StableTargetSet):
        raise TypeError("target_set must be a StableTargetSet record")
    if target_set.schema_version != TARGET_IDENTITY_SCHEMA_VERSION:
        raise TargetIdentityMismatchError("unknown target identity schema")
    if target_set.mask_encoding_version != TARGET_MASK_ENCODING_VERSION:
        raise TargetIdentityMismatchError("unknown target-mask encoding")
    _validated_name(target_set.dataset, name="dataset")
    _validated_name(target_set.image_name, name="image_name")
    if target_set.height <= 0 or target_set.width <= 0:
        raise TargetIdentityMismatchError("target-set shape must be positive")
    if target_set.pixel_connectivity != PIXEL_CONNECTIVITY:
        raise TargetIdentityMismatchError("pixel connectivity must be 8")
    if target_set.skimage_connectivity != SKIMAGE_CONNECTIVITY:
        raise TargetIdentityMismatchError("scikit-image connectivity must be 2")
    if not _is_sha256(target_set.label_mask_sha256):
        raise TargetIdentityMismatchError(
            "label_mask_sha256 is not a lowercase SHA256 digest"
        )

    targets = validate_target_identities(target_set.targets)
    for target in targets:
        envelope_assertions = {
            "schema_version": target_set.schema_version,
            "dataset": target_set.dataset,
            "image_name": target_set.image_name,
            "height": target_set.height,
            "width": target_set.width,
            "label_mask_sha256": target_set.label_mask_sha256,
        }
        for field, expected in envelope_assertions.items():
            if getattr(target, field) != expected:
                raise TargetIdentityMismatchError(
                    f"target disagrees with its envelope on {field}"
                )
    return target_set


def build_stable_target_set(
    mask: object,
    *,
    dataset: str,
    image_name: str,
    connectivity: int = SKIMAGE_CONNECTIVITY,
) -> StableTargetSet:
    """Build and fully validate the canonical identities for one label mask."""

    dataset = _validated_name(dataset, name="dataset")
    image_name = _validated_name(image_name, name="image_name")
    if connectivity != SKIMAGE_CONNECTIVITY:
        raise ValueError(
            "connectivity must be 2 (8-connectivity for a 2-D image)"
        )

    canonical = canonical_binary_target_mask(mask)
    height, width = (int(value) for value in canonical.shape)
    label_digest = canonical_mask_sha256(canonical)
    label_map = measure.label(canonical, connectivity=SKIMAGE_CONNECTIVITY)
    regions = tuple(measure.regionprops(label_map))

    provisional: list[StableTargetId] = []
    recovered = np.zeros_like(canonical, dtype=bool)
    for source_index, region in enumerate(regions):
        component = np.asarray(label_map == region.label, dtype=bool)
        if bool(np.any(recovered & component)):
            raise TargetIdentityMismatchError("connected components overlap")
        recovered |= component

        coords = np.argwhere(component)
        area = int(coords.shape[0])
        if area != int(region.area):
            raise TargetIdentityMismatchError("component area assertion failed")
        bbox = tuple(int(value) for value in region.bbox)
        if len(bbox) != 4:
            raise TargetIdentityMismatchError("component bbox must be 2-D")
        centroid_y, centroid_x = (
            float(value) for value in np.mean(coords, axis=0, dtype=np.float64)
        )
        if not np.allclose(
            (centroid_y, centroid_x),
            region.centroid,
            rtol=0.0,
            atol=1e-12,
        ):
            raise TargetIdentityMismatchError("component centroid assertion failed")

        component_digest = canonical_mask_sha256(component)
        provisional.append(
            StableTargetId(
                schema_version=TARGET_IDENTITY_SCHEMA_VERSION,
                dataset=dataset,
                image_name=image_name,
                component_index=-1,
                source_component_index=int(source_index),
                source_label=int(region.label),
                height=height,
                width=width,
                bbox=bbox,
                area=area,
                centroid_y=centroid_y,
                centroid_x=centroid_x,
                component_mask_sha256=component_digest,
                label_mask_sha256=label_digest,
                stable_key=_stable_key(dataset, image_name, component_digest),
            )
        )

    if not np.array_equal(recovered, canonical):
        raise TargetIdentityMismatchError(
            "connected components do not exactly reconstruct the label mask"
        )
    component_digests = [
        target.component_mask_sha256 for target in provisional
    ]
    if len(set(component_digests)) != len(component_digests):
        raise TargetIdentityMismatchError("duplicate component-mask digest")

    ordered = sorted(provisional, key=_canonical_sort_key)
    targets = tuple(
        StableTargetId(
            **{
                **target.as_dict(),
                "component_index": component_index,
            }
        )
        for component_index, target in enumerate(ordered)
    )
    targets = validate_target_identities(targets)
    target_set = StableTargetSet(
        schema_version=TARGET_IDENTITY_SCHEMA_VERSION,
        mask_encoding_version=TARGET_MASK_ENCODING_VERSION,
        dataset=dataset,
        image_name=image_name,
        height=height,
        width=width,
        pixel_connectivity=PIXEL_CONNECTIVITY,
        skimage_connectivity=SKIMAGE_CONNECTIVITY,
        label_mask_sha256=label_digest,
        targets=targets,
    )
    return validate_stable_target_set(target_set)


def enumerate_stable_targets(
    mask: object,
    *,
    dataset: str,
    image_name: str,
    connectivity: int = SKIMAGE_CONNECTIVITY,
) -> list[StableTargetId]:
    """Return canonical target records in deterministic component order."""

    return list(
        build_stable_target_set(
            mask,
            dataset=dataset,
            image_name=image_name,
            connectivity=connectivity,
        ).targets
    )


def assert_same_target_set(
    reference: StableTargetSet,
    observed: StableTargetSet,
) -> None:
    """Fail closed unless two image-level target sets are exactly identical."""

    if not isinstance(reference, StableTargetSet) or not isinstance(
        observed, StableTargetSet
    ):
        raise TypeError("reference and observed must be StableTargetSet records")
    reference = validate_stable_target_set(reference)
    observed = validate_stable_target_set(observed)
    reference_targets = reference.targets
    observed_targets = observed.targets
    envelope_fields = (
        "schema_version",
        "mask_encoding_version",
        "dataset",
        "image_name",
        "height",
        "width",
        "pixel_connectivity",
        "skimage_connectivity",
        "label_mask_sha256",
    )
    for field in envelope_fields:
        if getattr(reference, field) != getattr(observed, field):
            raise TargetIdentityMismatchError(
                f"target-set envelope disagrees on {field}"
            )
    if reference_targets != observed_targets:
        raise TargetIdentityMismatchError(
            "target identities or assertion metadata disagree"
        )


def assert_same_target_identities(
    reference: Sequence[StableTargetId],
    observed: Sequence[StableTargetId],
) -> None:
    """Fail closed unless two non-envelope identity sequences agree exactly.

    Prefer :func:`assert_same_target_set` when target-free images are possible,
    because an empty component sequence cannot carry image shape or label hash.
    """

    reference_targets = validate_target_identities(reference)
    observed_targets = validate_target_identities(observed)
    if reference_targets != observed_targets:
        raise TargetIdentityMismatchError(
            "target identities or assertion metadata disagree"
        )
