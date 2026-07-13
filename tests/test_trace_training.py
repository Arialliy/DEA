from __future__ import annotations

from argparse import Namespace
import copy
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from tools import train_trace
from utils.trace_gates import TraceGateError
from utils.trace_provenance import canonical_json_sha256
from utils.trace_geometry import TraceGeometrySpec
from utils.trace_training import (
    TraceTrainingError,
    atomic_torch_save,
    build_training_checkpoint,
    load_training_checkpoint,
    make_paired_loaders,
    nested_state_sha256,
    restore_training_checkpoint,
    seed_everything,
)


class TinyCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.potential_map = nn.Linear(2, 1)

    @property
    def head(self) -> nn.Module:
        return self.potential_map

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.potential_map(value)


TINY_GEOMETRY = TraceGeometrySpec(
    image_height=2,
    image_width=2,
    cell_size=1,
    max_down=1,
    max_left=0,
    max_right=0,
    margin=0,
)


def _checkpoint_fixture():
    seed_everything(19)
    model = TinyCheckpointModel()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01)
    loss = model(torch.tensor([[1.0, -1.0]])).square().mean()
    loss.backward()
    optimizer.step()
    train_generator = torch.Generator().manual_seed(31)
    dev_generator = torch.Generator().manual_seed(37)
    run_config = {"method": "dense_bernoulli", "epochs": 3}
    data = {"test_assets_included": False, "split_assignment_sha256": "a" * 64}
    gates = {"t0_a": {"report_sha256": "b" * 64}}
    sources = {"tools/train_trace.py": "c" * 64}
    payload = build_training_checkpoint(
        method="dense_bernoulli",
        epoch=0,
        best_dev_loss=0.25,
        best_epoch=0,
        checkpoint_role="latest",
        model=model,
        head=model.head,
        optimizer=optimizer,
        train_generator=train_generator,
        dev_generator=dev_generator,
        run_config=run_config,
        data_provenance=data,
        gates=gates,
        geometry=TINY_GEOMETRY.to_dict(),
        geometry_sha256=TINY_GEOMETRY.sha256,
        logk_cache_sha256=None,
        front_provenance={"checkpoint_sha256": "e" * 64},
        front_state_sha256="f" * 64,
        sources=sources,
        git={"commit": "0" * 40, "dirty": False, "dirty_paths": []},
        runtime={"torch": torch.__version__},
    )
    return (
        model,
        optimizer,
        train_generator,
        dev_generator,
        run_config,
        data,
        gates,
        sources,
        payload,
    )


def test_nested_state_hash_is_order_stable_and_tensor_sensitive() -> None:
    left = {"b": [torch.tensor([1.0, 2.0]), 3], "a": {4: np.array([5, 6])}}
    right = {"a": {4: np.array([5, 6])}, "b": [torch.tensor([1.0, 2.0]), 3]}
    changed = copy.deepcopy(right)
    changed["b"][0][1] += 1.0
    assert nested_state_sha256(left) == nested_state_sha256(right)
    assert nested_state_sha256(left) != nested_state_sha256(changed)


def test_paired_loaders_never_drop_last_and_shuffle_replays_from_seed() -> None:
    dataset = TensorDataset(torch.arange(5))
    first, dev, _, _ = make_paired_loaders(
        dataset, dataset, batch_size=2, num_workers=0, seed=11
    )
    second, _, _, _ = make_paired_loaders(
        dataset, dataset, batch_size=2, num_workers=0, seed=11
    )
    first_order = torch.cat([batch[0] for batch in first]).tolist()
    second_order = torch.cat([batch[0] for batch in second]).tolist()
    assert first_order == second_order
    assert sorted(first_order) == list(range(5))
    assert [batch[0].numel() for batch in dev] == [2, 2, 1]
    assert first.drop_last is dev.drop_last is False


def test_complete_checkpoint_restores_model_optimizer_rng_and_loader_streams(
    tmp_path: Path,
) -> None:
    (
        model,
        optimizer,
        train_generator,
        dev_generator,
        run_config,
        data,
        gates,
        sources,
        payload,
    ) = _checkpoint_fixture()
    checkpoint_path = atomic_torch_save(payload, tmp_path / "checkpoint.pt")
    payload = load_training_checkpoint(checkpoint_path)
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    expected_train = torch.rand(3, generator=train_generator)
    expected_dev = torch.rand(3, generator=dev_generator)
    expected_model_hash = payload["model_state_sha256"]

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
    optimizer.param_groups[0]["lr"] = 9.0
    random.random()
    np.random.random()
    torch.rand(7)
    torch.rand(7, generator=train_generator)
    torch.rand(7, generator=dev_generator)

    start, best_loss, best_epoch = restore_training_checkpoint(
        payload,
        expected_method="dense_bernoulli",
        expected_run_config_sha256=canonical_json_sha256(run_config),
        expected_data_provenance_sha256=canonical_json_sha256(data),
        expected_gates=gates,
        expected_geometry_sha256=TINY_GEOMETRY.sha256,
        expected_logk_cache_sha256=None,
        expected_front_state_sha256="f" * 64,
        expected_sources=sources,
        model=model,
        head=model.head,
        optimizer=optimizer,
        train_generator=train_generator,
        dev_generator=dev_generator,
    )
    assert (start, best_loss, best_epoch) == (1, 0.25, 0)
    assert nested_state_sha256(model.state_dict()) == expected_model_hash
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert torch.equal(torch.rand(3, generator=train_generator), expected_train)
    assert torch.equal(torch.rand(3, generator=dev_generator), expected_dev)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("method", "trace"), "manifest"),
        (
            lambda payload: payload["model_state"]["potential_map.weight"].add_(1.0),
            "model_state authentication",
        ),
    ],
)
def test_checkpoint_rejects_metadata_or_tensor_tampering(mutation, message) -> None:
    (
        model,
        optimizer,
        train_generator,
        dev_generator,
        run_config,
        data,
        gates,
        sources,
        original,
    ) = _checkpoint_fixture()
    payload = copy.deepcopy(original)
    mutation(payload)
    with pytest.raises(TraceTrainingError, match=message):
        restore_training_checkpoint(
            payload,
            expected_method="dense_bernoulli",
            expected_run_config_sha256=canonical_json_sha256(run_config),
            expected_data_provenance_sha256=canonical_json_sha256(data),
            expected_gates=gates,
            expected_geometry_sha256=TINY_GEOMETRY.sha256,
            expected_logk_cache_sha256=None,
            expected_front_state_sha256="f" * 64,
            expected_sources=sources,
            model=model,
            head=model.head,
            optimizer=optimizer,
            train_generator=train_generator,
            dev_generator=dev_generator,
        )


class TinyTripleDataset(Dataset):
    def __init__(self) -> None:
        self.images = torch.zeros((5, 3, 4, 4))
        self.masks = torch.zeros((5, 1, 4, 4))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], self.masks[index], "sample_%d" % index


class TinyDenseControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.potential_map = nn.Conv2d(3, 1, 1)
        nn.init.zeros_(self.potential_map.weight)
        nn.init.zeros_(self.potential_map.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.potential_map(image)

    @staticmethod
    def exact_bernoulli_nll(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

    def assert_front_integrity(self) -> str:
        return "synthetic_front_unchanged"


def test_tiny_dense_training_uses_all_samples_and_proper_empty_bce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_trace, "MatchedDenseMSHNet", TinyDenseControl)
    dataset = TinyTripleDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
    model = TinyDenseControl()
    optimizer = torch.optim.SGD(model.potential_map.parameters(), lr=0.25)
    before = model.potential_map.bias.detach().clone()
    train_loss = train_trace._train_epoch(
        model,
        "dense_bernoulli",
        loader,
        optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=0.0,
    )
    dev_loss = train_trace._dev_epoch(
        model,
        "dense_bernoulli",
        loader,
        device=torch.device("cpu"),
    )
    assert math.isfinite(train_loss)
    assert math.isfinite(dev_loss)
    assert model.potential_map.bias.item() < before.item()


def test_current_formal_no_go_stops_before_dataset_checkpoint_or_t0_b_access() -> None:
    report = (
        Path(__file__).resolve().parents[1]
        / "repro_runs/trace/t0_a/IRSTD-1K_fit_seed_20260711.json"
    )
    assert report.is_file()
    args = Namespace(
        dataset_dir="/definitely/not-opened/IRSTD-1K",
        t0_a_report=str(report),
        t0_b_dp_report="/definitely/not-opened/dp.json",
        baseline_checkpoint="/definitely/not-opened/baseline.pt",
        t0_b_integration_report="/definitely/not-opened/integration.json",
    )
    with pytest.raises(TraceGateError, match="T0-A status.*NO-GO"):
        train_trace._gate_first_contract(args)
