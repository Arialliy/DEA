from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from model.MSHNet import MSHNet
from model.mshnet_d0_backbone import MSHNetD0Backbone
from model.trace_front import FrozenMSHNetD0
from utils.trace_provenance import TraceProvenanceError, state_dict_sha256


DATASET = "NUAA-SIRST"
SEED = 17
TRAIN_SPLIT_SHA256 = "1" * 64
VAL_SPLIT_SHA256 = "2" * 64
FORBIDDEN_HEADS = {"output_0", "output_1", "output_2", "output_3", "final"}


@dataclass(frozen=True)
class _SyntheticCheckpoint:
    path: Path
    state: OrderedDict[str, torch.Tensor]


def _constant_expanded_like(
    reference: torch.Tensor,
    *,
    key: str,
    index: int,
) -> torch.Tensor:
    """Create a correctly shaped tensor backed by only one scalar of storage."""

    if not reference.dtype.is_floating_point:
        value: float | int = 0
    elif key.endswith("running_var"):
        value = 1.0
    elif key.endswith("running_mean"):
        value = 0.0
    elif key.endswith(".weight") and (
        ".bn1." in key or ".bn2." in key or ".shortcut.1." in key
    ):
        value = 1.0
    elif key.endswith(".weight"):
        value = (1 + index % 3) * 1.0e-4
    else:
        value = (index % 5 - 2) * 1.0e-4
    scalar = torch.tensor(value, dtype=reference.dtype)
    return scalar.expand(reference.shape)


def _method_meta() -> dict[str, object]:
    return {
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


@pytest.fixture(scope="module")
def synthetic_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> _SyntheticCheckpoint:
    # MSHNet is instantiated only to obtain the exact canonical key/shape contract.
    # Every serialized tensor is a zero-stride scalar expansion, so this remains a
    # full checkpoint without writing a real multi-megabyte parameter payload.
    with torch.random.fork_rng(devices=[]):
        canonical_state = MSHNet(3).state_dict()
    state = OrderedDict(
        (
            key,
            _constant_expanded_like(value, key=key, index=index),
        )
        for index, (key, value) in enumerate(canonical_state.items())
    )
    path = tmp_path_factory.mktemp("trace_front") / "synthetic_mshnet.pt"
    torch.save(
        {
            "net": state,
            "method_meta": _method_meta(),
            "epoch": 23,
            "iou": 0.731,
        },
        path,
    )

    assert all(f"{name}.weight" in state for name in FORBIDDEN_HEADS)
    assert path.stat().st_size < 512 * 1024
    return _SyntheticCheckpoint(path=path, state=state)


def _load_front(checkpoint: _SyntheticCheckpoint) -> FrozenMSHNetD0:
    return FrozenMSHNetD0(
        checkpoint.path,
        expected_dataset=DATASET,
        expected_seed=SEED,
        expected_train_split_sha256=TRAIN_SPLIT_SHA256,
        expected_val_split_sha256=VAL_SPLIT_SHA256,
    )


def _image(*, requires_grad: bool = False) -> torch.Tensor:
    image = torch.linspace(-0.5, 0.5, 3 * 16 * 16).reshape(1, 3, 16, 16)
    return image.requires_grad_(requires_grad)


def test_strict_checkpoint_load_reproduces_independent_d0_bitwise(
    synthetic_checkpoint: _SyntheticCheckpoint,
) -> None:
    front = _load_front(synthetic_checkpoint)
    independent = MSHNetD0Backbone(3).eval()
    independent.load_mshnet_front_state_dict(synthetic_checkpoint.state)

    image = _image()
    with torch.no_grad():
        expected = independent(image)
        observed = front(image)

    assert torch.equal(observed, expected)
    assert front.provenance.dataset == DATASET
    assert front.provenance.seed == SEED
    assert front.provenance.source_method == "MSHNet"
    assert front.provenance.source_model_type == "mshnet"
    assert front.loaded_state_sha256 == state_dict_sha256(
        independent.state_dict(), keys=independent.front_state_keys
    )


def test_loaded_front_physically_contains_no_side_or_fusion_head(
    synthetic_checkpoint: _SyntheticCheckpoint,
) -> None:
    front = _load_front(synthetic_checkpoint)

    for name, _module in front.named_modules():
        assert FORBIDDEN_HEADS.isdisjoint(name.split("."))
    for name in front.state_dict():
        assert FORBIDDEN_HEADS.isdisjoint(name.split("."))


class _OuterTrainingModel(nn.Module):
    def __init__(self, front: FrozenMSHNetD0) -> None:
        super().__init__()
        self.front = front

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.front(image)


def test_outer_train_keeps_front_eval_frozen_and_forward_graph_free(
    synthetic_checkpoint: _SyntheticCheckpoint,
) -> None:
    front = _load_front(synthetic_checkpoint)
    outer = _OuterTrainingModel(front)

    outer.train()
    assert outer.training
    assert not front.training
    assert all(not module.training for module in front.modules())
    assert sum(
        parameter.numel()
        for parameter in front.parameters()
        if parameter.requires_grad
    ) == 0

    image = _image(requires_grad=True)
    assert torch.is_grad_enabled()
    d0 = outer(image)
    assert torch.is_grad_enabled()
    assert not d0.requires_grad
    assert d0.grad_fn is None
    assert image.grad is None


def test_anchor_is_stable_across_repeated_calls_and_outer_train(
    synthetic_checkpoint: _SyntheticCheckpoint,
) -> None:
    front = _load_front(synthetic_checkpoint)
    image = _image()

    first = front.anchor(image)
    _OuterTrainingModel(front).train()
    second = front.anchor(image.clone())

    assert second == first
    assert first["input_shape"] == [1, 3, 16, 16]
    assert first["d0_shape"] == [1, 16, 16, 16]
    assert first["front_state_sha256"] == front.loaded_state_sha256
    assert len(str(first["input_sha256"])) == 64
    assert len(str(first["d0_sha256"])) == 64


@pytest.mark.parametrize("tamper", ["parameter", "bn_buffer"])
def test_integrity_rejects_parameter_or_bn_buffer_tampering(
    synthetic_checkpoint: _SyntheticCheckpoint,
    tamper: str,
) -> None:
    front = _load_front(synthetic_checkpoint)
    assert front.assert_integrity() == front.loaded_state_sha256

    if tamper == "parameter":
        tensor = next(front.backbone.parameters())
    else:
        tensor = dict(front.backbone.named_buffers())[
            "encoder_0.0.bn1.running_mean"
        ]
    with torch.no_grad():
        tensor.reshape(-1)[0].add_(0.25)

    with pytest.raises(
        TraceProvenanceError,
        match="parameters or BN buffers changed",
    ):
        front.assert_integrity()
