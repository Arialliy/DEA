from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from utils.trace_data import (
    SPLIT_ALGORITHM,
    TraceDataContractError,
    audit_disjoint_splits,
    build_trace_data,
)


TRAIN_NAMES = tuple("sample_%02d" % index for index in range(12))
TEST_NAMES = ("heldout_a", "heldout_b")


def test_holdout_ranking_is_byte_exact_with_clean_mshnet_protocol(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    bundle = _build(dataset)
    ranked = sorted(
        TRAIN_NAMES,
        key=lambda name: hashlib.sha256(("7\0%s" % name).encode("utf-8")).digest(),
    )
    expected_dev = set(ranked[:3])
    assert tuple(name for name in TRAIN_NAMES if name in expected_dev) == bundle.dev.names
    assert tuple(name for name in TRAIN_NAMES if name not in expected_dev) == bundle.train.names
    assert SPLIT_ALGORITHM == "sha256_seed_nul_name_rank_clean_mshnet_v1"


def _write_manifest(path: Path, names: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def _write_pair(root: Path, name: str, *, offset: int = 0) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    height, width = 5, 9
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = (np.arange(width, dtype=np.uint8)[None, :] * 21 + offset) % 255
    image[..., 1] = (17 + offset) % 255
    image[..., 2] = np.arange(height, dtype=np.uint8)[:, None] * 31
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[1:4, 0:2] = 1
    mask[2, 5] = 127
    mask[4, 8] = 255
    Image.fromarray(image, mode="RGB").save(root / "images" / (name + ".png"))
    Image.fromarray(mask, mode="L").save(root / "masks" / (name + ".png"))


def _make_dataset(
    root: Path,
    *,
    train_names: tuple[str, ...] = TRAIN_NAMES,
    test_names: tuple[str, ...] = TEST_NAMES,
    write_test_assets: bool = True,
) -> Path:
    dataset = root / "Toy-SIRST"
    _write_manifest(dataset / "img_idx" / "train_Toy-SIRST.txt", train_names)
    _write_manifest(dataset / "img_idx" / "test_Toy-SIRST.txt", test_names)
    for index, name in enumerate(train_names):
        _write_pair(dataset, name, offset=index)
    if write_test_assets:
        for index, name in enumerate(test_names):
            _write_pair(dataset, name, offset=100 + index)
    return dataset


def _build(
    dataset: Path,
    *,
    include_test: bool = False,
    train_horizontal_flip: bool = True,
):
    return build_trace_data(
        dataset,
        image_size=(7, 11),
        seed=7,
        dev_fraction=0.25,
        include_test=include_test,
        train_horizontal_flip=train_horizontal_flip,
    )


def _find_epoch_change(dataset) -> tuple[str, int, int]:
    for name in dataset.names:
        dataset.set_epoch(0)
        initial = dataset.should_flip(name)
        for epoch in range(1, 64):
            dataset.set_epoch(epoch)
            if dataset.should_flip(name) != initial:
                return name, 0, epoch
    raise AssertionError("fixture unexpectedly found no SHA256 flip transition")


def test_sha256_split_is_cross_instance_deterministic_disjoint_and_path_free(
    tmp_path: Path,
) -> None:
    # Missing test assets are deliberate: training reads only test names.
    dataset = _make_dataset(tmp_path, write_test_assets=False)
    first = _build(dataset)
    second = _build(dataset)

    assert first.train.names == second.train.names
    assert first.dev.names == second.dev.names
    assert set(first.train.names).isdisjoint(first.dev.names)
    assert set(first.train.names).union(first.dev.names) == set(TRAIN_NAMES)
    assert set(first.train.names).isdisjoint(first.test_names)
    assert set(first.dev.names).isdisjoint(first.test_names)
    assert len(first.dev) == 3
    assert first.test is None

    provenance = first.provenance()
    assert provenance == second.provenance()
    assert provenance["test_assets_included"] is False
    assert provenance["asset_content_sha256"]["test"] is None
    encoded = json.dumps(provenance, sort_keys=True)
    assert str(tmp_path.resolve()) not in encoded
    assert provenance["canonical_train_manifest"]["locator"] == (
        "img_idx/train_Toy-SIRST.txt"
    )


def test_train_flip_is_shared_across_methods_instances_order_and_resume(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    baseline_data = _build(dataset)
    trace_data = _build(dataset)
    name, first_epoch, second_epoch = _find_epoch_change(baseline_data.train)
    index = baseline_data.train.names.index(name)
    assert trace_data.train.names[index] == name

    baseline_data.set_epoch(first_epoch)
    trace_data.set_epoch(first_epoch)
    image_first, mask_first, returned_name = baseline_data.train[index]
    # Access in an unrelated order; transformation has no process/order RNG.
    _ = trace_data.train[(index + 1) % len(trace_data.train)]
    image_same, mask_same, _ = trace_data.train[index]
    assert returned_name == name
    assert torch.equal(image_first, image_same)
    assert torch.equal(mask_first, mask_same)

    baseline_data.set_epoch(second_epoch)
    trace_data.set_epoch(second_epoch)
    image_second, mask_second, _ = baseline_data.train[index]
    image_second_peer, mask_second_peer, _ = trace_data.train[index]
    assert torch.equal(image_second, image_second_peer)
    assert torch.equal(mask_second, mask_second_peer)
    assert torch.equal(image_second, torch.flip(image_first, dims=(-1,)))
    assert torch.equal(mask_second, torch.flip(mask_first, dims=(-1,)))

    # A resumed method can jump directly to the recorded epoch and reproduces
    # the uninterrupted method exactly.
    resumed_data = _build(dataset)
    resumed_data.set_epoch(second_epoch)
    resumed_image, resumed_mask, _ = resumed_data.train[index]
    assert torch.equal(resumed_image, image_second)
    assert torch.equal(resumed_mask, mask_second)


def test_strict_default_disables_geometry_changing_train_augmentation(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    bundle = _build(dataset, train_horizontal_flip=False)
    bundle.set_epoch(0)
    first = [bundle.train.should_flip(name) for name in bundle.train.names]
    bundle.set_epoch(37)
    later = [bundle.train.should_flip(name) for name in bundle.train.names]
    assert not any(first) and not any(later)
    assert bundle.provenance()["train_augmentation"] == {
        "policy": "none",
        "reason": "preserve the exact train-only T0-A geometry contract",
    }


def test_persistent_worker_observes_shared_set_epoch(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    bundle = _build(dataset)
    name = bundle.train.names[0]
    _, first_epoch, second_epoch = _find_epoch_change(bundle.train)
    assert bundle.train.names[0] == name
    loader = DataLoader(
        bundle.train,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
    )
    try:
        bundle.set_epoch(first_epoch)
        first_image, first_mask, first_name = next(iter(loader))
        bundle.set_epoch(second_epoch)
        second_image, second_mask, second_name = next(iter(loader))
        assert first_name == second_name == (name,)
        assert torch.equal(second_image, torch.flip(first_image, dims=(-1,)))
        assert torch.equal(second_mask, torch.flip(first_mask, dims=(-1,)))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_fixed_resize_binary_mask_and_no_dev_or_test_augmentation(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    bundle = _build(dataset, include_test=True)
    image, mask, name = bundle.train[0]
    assert image.shape == (3, 7, 11)
    assert image.dtype == torch.float32
    assert mask.shape == (1, 7, 11)
    assert mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
    assert name == bundle.train.names[0]

    for evaluation_set in (bundle.dev, bundle.test):
        assert evaluation_set is not None
        evaluation_set.set_epoch(0)
        image_zero, mask_zero, _ = evaluation_set[0]
        evaluation_set.set_epoch(37)
        image_later, mask_later, _ = evaluation_set[0]
        assert torch.equal(image_zero, image_later)
        assert torch.equal(mask_zero, mask_later)
        assert evaluation_set.should_flip(evaluation_set.names[0]) is False


@pytest.mark.parametrize(
    ("train", "dev", "test"),
    [
        (("a", "shared"), ("shared", "b"), ("c",)),
        (("a", "shared"), ("b",), ("shared", "c")),
        (("a",), ("b", "shared"), ("shared", "c")),
    ],
)
def test_explicit_split_audit_rejects_every_overlap(
    train: tuple[str, ...],
    dev: tuple[str, ...],
    test: tuple[str, ...],
) -> None:
    with pytest.raises(TraceDataContractError, match="split leakage"):
        audit_disjoint_splits(train, dev, test)


def test_manifest_contract_rejects_duplicate_empty_test_swap_and_overlap(
    tmp_path: Path,
) -> None:
    duplicate = _make_dataset(tmp_path / "duplicate")
    train_file = duplicate / "img_idx" / "train_Toy-SIRST.txt"
    _write_manifest(train_file, ("sample_00", "sample_00"))
    with pytest.raises(TraceDataContractError, match="duplicate sample names"):
        _build(duplicate)

    empty = _make_dataset(tmp_path / "empty")
    (empty / "img_idx" / "train_Toy-SIRST.txt").write_bytes(b"")
    with pytest.raises(TraceDataContractError, match="empty split manifest"):
        _build(empty)

    swapped = _make_dataset(tmp_path / "swapped")
    with pytest.raises(TraceDataContractError, match="test manifest cannot"):
        build_trace_data(
            swapped,
            train_manifest="img_idx/test_Toy-SIRST.txt",
            test_manifest="img_idx/train_Toy-SIRST.txt",
            image_size=(7, 11),
            seed=7,
        )

    overlap = _make_dataset(tmp_path / "overlap")
    _write_manifest(
        overlap / "img_idx" / "test_Toy-SIRST.txt",
        (TRAIN_NAMES[0], "heldout_a"),
    )
    with pytest.raises(TraceDataContractError, match="split leakage"):
        _build(overlap)


def test_missing_assets_fail_closed_only_when_split_is_included(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path, write_test_assets=False)
    # Training must not inspect locked test contents.
    training_bundle = _build(dataset, include_test=False)
    assert training_bundle.test is None
    with pytest.raises(TraceDataContractError, match="missing image asset"):
        _build(dataset, include_test=True)

    missing_train_mask = dataset / "masks" / (training_bundle.train.names[0] + ".png")
    missing_train_mask.unlink()
    with pytest.raises(TraceDataContractError, match="missing mask asset"):
        _build(dataset, include_test=False)


def test_asset_tampering_changes_content_hash_but_not_split_hash(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    before = _build(dataset, include_test=True).provenance()
    target = dataset / "images" / (TRAIN_NAMES[0] + ".png")
    changed = np.full((5, 9, 3), 231, dtype=np.uint8)
    Image.fromarray(changed, mode="RGB").save(target)
    after = _build(dataset, include_test=True).provenance()

    assert before["split_assignment_sha256"] == after["split_assignment_sha256"]
    assert before["asset_content_sha256"]["all_included"] != (
        after["asset_content_sha256"]["all_included"]
    )
    assert before["canonical_train_manifest"] == after["canonical_train_manifest"]
