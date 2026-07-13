from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch

from utils.trace_provenance import (
    TraceProvenanceError,
    capture_rng_state,
    load_clean_mshnet_checkpoint,
    normalize_state_dict_keys,
    restore_rng_state,
    sha256_file,
    state_dict_sha256,
)


DATASET = "NUAA-SIRST"
SEED = 17
TRAIN_SPLIT_SHA256 = "1" * 64
VAL_SPLIT_SHA256 = "2" * 64
FRONT_STATE_KEYS = (
    "encoder.conv.weight",
    "encoder.bn.weight",
    "encoder.bn.bias",
    "encoder.bn.running_mean",
    "encoder.bn.running_var",
    "encoder.bn.num_batches_tracked",
)
BN_BUFFER_KEYS = (
    "encoder.bn.running_mean",
    "encoder.bn.running_var",
    "encoder.bn.num_batches_tracked",
)


def _state_dict() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (
            ("encoder.conv.weight", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
            ("encoder.bn.weight", torch.tensor([1.0, 1.5], dtype=torch.float32)),
            ("encoder.bn.bias", torch.tensor([-0.25, 0.5], dtype=torch.float32)),
            ("encoder.bn.running_mean", torch.tensor([0.125, -0.75])),
            ("encoder.bn.running_var", torch.tensor([1.25, 0.875])),
            ("encoder.bn.num_batches_tracked", torch.tensor(11, dtype=torch.int64)),
            ("middle.weight", torch.tensor([[3.0]], dtype=torch.float32)),
        )
    )


def _method_meta(**overrides: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "method": "MSHNet",
        "model_type": "mshnet",
        "dataset_dir": f"/synthetic/datasets/{DATASET}",
        "seed": SEED,
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "val_split_sha256": VAL_SPLIT_SHA256,
        "selection_split": "validation",
        "evaluation_split": "validation",
        "protocol": "internal_dev_holdout_v1",
    }
    metadata.update(overrides)
    return metadata


def _write_checkpoint(
    path: Path,
    *,
    state_dict: OrderedDict[str, torch.Tensor] | None = None,
    method_meta: dict[str, Any] | None = None,
) -> Path:
    torch.save(
        {
            "net": _state_dict() if state_dict is None else state_dict,
            "method_meta": _method_meta() if method_meta is None else method_meta,
            "epoch": 23,
            "iou": 0.731,
        },
        path,
    )
    return path


def _load(path: Path):
    return load_clean_mshnet_checkpoint(
        path,
        front_state_keys=FRONT_STATE_KEYS,
        expected_dataset=DATASET,
        expected_seed=SEED,
        expected_train_split_sha256=TRAIN_SPLIT_SHA256,
        expected_val_split_sha256=VAL_SPLIT_SHA256,
    )


def _clone_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value.clone()) for key, value in state_dict.items())


def test_clean_full_checkpoint_is_accepted_and_audited(tmp_path: Path) -> None:
    state = _state_dict()
    path = _write_checkpoint(tmp_path / "clean_mshnet.pkl", state_dict=state)

    normalized, provenance, method_meta = _load(path)

    assert tuple(normalized) == tuple(state)
    for key in state:
        assert torch.equal(normalized[key], state[key])
    assert method_meta == _method_meta()
    assert provenance.checkpoint_sha256 == sha256_file(path)
    assert provenance.dataset == DATASET
    assert provenance.seed == SEED
    assert provenance.train_split_sha256 == TRAIN_SPLIT_SHA256
    assert provenance.val_split_sha256 == VAL_SPLIT_SHA256
    assert provenance.source_epoch == 23
    assert provenance.source_iou == pytest.approx(0.731)
    assert provenance.source_method == "MSHNet"
    assert provenance.source_model_type == "mshnet"
    assert provenance.front_tensor_sha256 == state_dict_sha256(
        normalized, keys=FRONT_STATE_KEYS
    )
    assert provenance.bn_buffer_sha256 == state_dict_sha256(
        normalized, keys=BN_BUFFER_KEYS
    )


def test_raw_weights_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "raw_weights.pkl"
    torch.save(_state_dict(), path)

    with pytest.raises(TraceProvenanceError, match="full baseline checkpoint"):
        _load(path)


@pytest.mark.parametrize("metadata_key", ["selection_split", "evaluation_split"])
def test_test_selected_or_test_evaluated_checkpoint_is_rejected(
    tmp_path: Path, metadata_key: str
) -> None:
    path = _write_checkpoint(
        tmp_path / f"forbidden_{metadata_key}.pkl",
        method_meta=_method_meta(**{metadata_key: "TEST"}),
    )

    with pytest.raises(TraceProvenanceError, match="test-selected/evaluated"):
        _load(path)


@pytest.mark.parametrize(
    ("expected_override", "message"),
    [
        ({"expected_dataset": "IRSTD-1k"}, "dataset mismatch"),
        ({"expected_seed": SEED + 1}, "seed mismatch"),
        (
            {"expected_train_split_sha256": "a" * 64},
            "train split hash does not match",
        ),
        (
            {"expected_val_split_sha256": "b" * 64},
            "validation split hash does not match",
        ),
    ],
)
def test_paired_protocol_identity_mismatch_is_rejected(
    tmp_path: Path, expected_override: dict[str, Any], message: str
) -> None:
    path = _write_checkpoint(tmp_path / "identity_mismatch.pkl")
    expected: dict[str, Any] = {
        "expected_dataset": DATASET,
        "expected_seed": SEED,
        "expected_train_split_sha256": TRAIN_SPLIT_SHA256,
        "expected_val_split_sha256": VAL_SPLIT_SHA256,
    }
    expected.update(expected_override)

    with pytest.raises(TraceProvenanceError, match=message):
        load_clean_mshnet_checkpoint(
            path,
            front_state_keys=FRONT_STATE_KEYS,
            **expected,
        )


def test_uniform_module_prefix_is_normalized_and_loadable(tmp_path: Path) -> None:
    state = _state_dict()
    prefixed = OrderedDict((f"module.{key}", value) for key, value in state.items())

    normalized_direct = normalize_state_dict_keys(prefixed)
    assert tuple(normalized_direct) == tuple(state)
    assert all(normalized_direct[key] is state[key] for key in state)

    path = _write_checkpoint(tmp_path / "data_parallel.pkl", state_dict=prefixed)
    normalized_loaded, provenance, _ = _load(path)
    assert tuple(normalized_loaded) == tuple(state)
    assert provenance.front_tensor_sha256 == state_dict_sha256(
        state, keys=FRONT_STATE_KEYS
    )


def test_mixed_module_prefix_is_rejected() -> None:
    mixed = OrderedDict(
        (
            ("module.encoder.conv.weight", torch.ones(1)),
            ("encoder.bn.weight", torch.ones(1)),
        )
    )

    with pytest.raises(TraceProvenanceError, match="mixes prefixed and unprefixed"):
        normalize_state_dict_keys(mixed)


def test_front_and_bn_hashes_are_stable_and_detect_tensor_changes(
    tmp_path: Path,
) -> None:
    base = _state_dict()
    reordered = OrderedDict(reversed(tuple(_clone_state_dict(base).items())))
    front_changed = _clone_state_dict(base)
    front_changed["encoder.conv.weight"][0, 0] += 0.5
    bn_changed = _clone_state_dict(base)
    bn_changed["encoder.bn.running_mean"][1] -= 0.25

    _, base_provenance, _ = _load(
        _write_checkpoint(tmp_path / "base.pkl", state_dict=base)
    )
    _, reordered_provenance, _ = _load(
        _write_checkpoint(tmp_path / "reordered.pkl", state_dict=reordered)
    )
    _, front_changed_provenance, _ = _load(
        _write_checkpoint(tmp_path / "front_changed.pkl", state_dict=front_changed)
    )
    _, bn_changed_provenance, _ = _load(
        _write_checkpoint(tmp_path / "bn_changed.pkl", state_dict=bn_changed)
    )

    assert base_provenance.front_tensor_sha256 == (
        reordered_provenance.front_tensor_sha256
    )
    assert base_provenance.bn_buffer_sha256 == reordered_provenance.bn_buffer_sha256

    assert (
        front_changed_provenance.front_tensor_sha256
        != base_provenance.front_tensor_sha256
    )
    assert (
        front_changed_provenance.bn_buffer_sha256
        == base_provenance.bn_buffer_sha256
    )

    assert (
        bn_changed_provenance.front_tensor_sha256
        != base_provenance.front_tensor_sha256
    )
    assert bn_changed_provenance.bn_buffer_sha256 != base_provenance.bn_buffer_sha256


def test_rng_capture_and_restore_replays_all_cpu_streams_exactly() -> None:
    original = capture_rng_state()
    try:
        random.seed(20260713)
        np.random.seed(20260713)
        torch.manual_seed(20260713)
        captured = capture_rng_state()

        expected_python = [random.random() for _ in range(5)]
        expected_numpy = np.random.random(5)
        expected_torch = torch.rand(5)

        for _ in range(13):
            random.random()
        np.random.random(13)
        torch.rand(13)

        restore_rng_state(captured)

        assert [random.random() for _ in range(5)] == expected_python
        np.testing.assert_array_equal(np.random.random(5), expected_numpy)
        assert torch.equal(torch.rand(5), expected_torch)
    finally:
        restore_rng_state(original)


def test_rng_restore_rejects_incomplete_capture() -> None:
    incomplete = capture_rng_state()
    incomplete.pop("numpy")

    with pytest.raises(TraceProvenanceError, match="missing.*numpy"):
        restore_rng_state(incomplete)
