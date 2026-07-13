"""Deterministic, authenticated training-state utilities for TRACE.

The helpers in this module are deliberately method-agnostic: TRACE's exact
likelihood and the matched dense Bernoulli control share the same loader,
checkpoint, resume, and provenance contract.  No helper exposes a test
dataset or contains a permissive checkpoint fallback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.trace_provenance import (
    TraceProvenanceError,
    canonical_json_sha256,
    capture_rng_state,
    restore_rng_state,
    sha256_file,
)
from utils.trace_geometry import TraceGeometrySpec


TRACE_CHECKPOINT_SCHEMA_VERSION = "trace_training_checkpoint_v1"
TRACE_METHODS = ("trace", "dense_bernoulli")


class TraceTrainingError(TraceProvenanceError):
    """A deterministic training or checkpoint invariant was violated."""


def seed_everything(seed: int) -> None:
    """Seed all process RNGs and request deterministic PyTorch algorithms."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TraceTrainingError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_loader_worker(_worker_id: int) -> None:
    """Bind Python/NumPy worker RNGs to PyTorch's audited worker seed."""

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_paired_loaders(
    train_dataset: Dataset,
    dev_dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, torch.Generator, torch.Generator]:
    """Construct paired train/dev loaders with explicit ``drop_last=False``."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise TraceTrainingError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise TraceTrainingError("num_workers must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TraceTrainingError("seed must be a non-negative integer")
    train_generator = torch.Generator(device="cpu")
    dev_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(seed)
    dev_generator.manual_seed(seed ^ 0x5EED5EED)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": False,
        "worker_init_fn": seed_loader_worker,
        "persistent_workers": False,
        "pin_memory": False,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        generator=dev_generator,
        **common,
    )
    if train_loader.drop_last or dev_loader.drop_last:  # pragma: no cover
        raise TraceTrainingError("paired loaders must never drop the final batch")
    return train_loader, dev_loader, train_generator, dev_generator


def _frame(digest: "hashlib._Hash", payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _update_nested_hash(digest: "hashlib._Hash", value: Any) -> None:
    """Canonical recursive hash for optimizer, RNG, and tensor state."""

    if value is None:
        _frame(digest, b"none")
    elif isinstance(value, bool):
        _frame(digest, b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        _frame(digest, ("int:%d" % value).encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise TraceTrainingError("checkpoint state contains a non-finite float")
        _frame(digest, ("float:%s" % value.hex()).encode("ascii"))
    elif isinstance(value, str):
        _frame(digest, b"str:" + value.encode("utf-8"))
    elif isinstance(value, bytes):
        _frame(digest, b"bytes:" + value)
    elif torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        _frame(digest, ("tensor:%s:%s" % (tensor.dtype, list(tensor.shape))).encode("ascii"))
        _frame(digest, tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TraceTrainingError("checkpoint state contains an object array")
        _frame(digest, ("ndarray:%s:%s" % (array.dtype.str, list(array.shape))).encode("ascii"))
        _frame(digest, array.tobytes(order="C"))
    elif isinstance(value, np.dtype):
        _frame(digest, b"numpy_dtype:" + value.str.encode("ascii"))
    elif isinstance(value, Mapping):
        _frame(digest, b"mapping")
        keys = sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        for key in keys:
            _update_nested_hash(digest, key)
            _update_nested_hash(digest, value[key])
    elif isinstance(value, tuple):
        _frame(digest, b"tuple")
        for item in value:
            _update_nested_hash(digest, item)
    elif isinstance(value, list):
        _frame(digest, b"list")
        for item in value:
            _update_nested_hash(digest, item)
    else:
        raise TraceTrainingError(
            "unsupported checkpoint state type: %s" % type(value).__name__
        )


def nested_state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_nested_hash(digest, value)
    return digest.hexdigest()


def source_inventory(repository_root: str | Path, locators: Sequence[str]) -> dict[str, str]:
    """Hash an exact, repository-relative source set."""

    root = Path(repository_root).resolve()
    if isinstance(locators, (str, bytes)) or not locators:
        raise TraceTrainingError("source inventory must be a non-empty sequence")
    result: dict[str, str] = {}
    for locator in sorted(locators):
        if (
            not isinstance(locator, str)
            or not locator
            or Path(locator).is_absolute()
            or ".." in Path(locator).parts
        ):
            raise TraceTrainingError("source inventory contains an unsafe locator")
        source = (root / locator).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:  # pragma: no cover - guarded above
            raise TraceTrainingError("source inventory escapes repository") from exc
        if not source.is_file():
            raise TraceTrainingError("missing training source: %s" % locator)
        result[locator] = sha256_file(source)
    return result


def _checkpoint_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the non-recursive authenticated manifest for a checkpoint."""

    return {
        "schema_version": payload.get("schema_version"),
        "method": payload.get("method"),
        "epoch": payload.get("epoch"),
        "best_dev_loss": payload.get("best_dev_loss"),
        "best_epoch": payload.get("best_epoch"),
        "checkpoint_role": payload.get("checkpoint_role"),
        "run_config_sha256": payload.get("run_config_sha256"),
        "data_provenance_sha256": payload.get("data_provenance_sha256"),
        "geometry_sha256": payload.get("geometry_sha256"),
        "logk_cache_sha256": payload.get("logk_cache_sha256"),
        "front_state_sha256": payload.get("front_state_sha256"),
        "front_provenance_sha256": payload.get("front_provenance_sha256"),
        "gate_provenance_sha256": canonical_json_sha256(payload.get("gates")),
        "source_sha256": payload.get("source_sha256"),
        "model_state_sha256": payload.get("model_state_sha256"),
        "frozen_state_sha256": payload.get("frozen_state_sha256"),
        "head_state_sha256": payload.get("head_state_sha256"),
        "optimizer_state_sha256": payload.get("optimizer_state_sha256"),
        "rng_state_sha256": payload.get("rng_state_sha256"),
        "loader_generator_state_sha256": payload.get(
            "loader_generator_state_sha256"
        ),
        "git_sha256": payload.get("git_sha256"),
        "runtime_sha256": payload.get("runtime_sha256"),
    }


def _frozen_model_state(model_state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model_state.items()
        if not str(key).startswith("potential_map.")
    }


def build_training_checkpoint(
    *,
    method: str,
    epoch: int,
    best_dev_loss: float,
    best_epoch: int,
    checkpoint_role: str,
    model: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    dev_generator: torch.Generator,
    run_config: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    gates: Mapping[str, Any],
    geometry: Mapping[str, Any],
    geometry_sha256: str,
    logk_cache_sha256: str | None,
    front_provenance: Mapping[str, Any],
    front_state_sha256: str,
    sources: Mapping[str, str],
    git: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture a complete restart point after one fully evaluated epoch."""

    if method not in TRACE_METHODS:
        raise TraceTrainingError("unsupported training method")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise TraceTrainingError("checkpoint epoch must be non-negative")
    if checkpoint_role not in {"latest", "best_dev"}:
        raise TraceTrainingError("checkpoint role must be latest or best_dev")
    if not math.isfinite(float(best_dev_loss)) or best_epoch < 0:
        raise TraceTrainingError("best development state must be finite and non-negative")
    # ``state_dict`` tensors alias live module/optimizer storage.  Clone the
    # entire nested state before hashing so an in-memory payload cannot change
    # underneath its authenticated manifest prior to the atomic write.
    model_state = copy.deepcopy(model.state_dict())
    head_state = copy.deepcopy(head.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    rng_state = capture_rng_state()
    generator_state = {
        "train": train_generator.get_state(),
        "dev": dev_generator.get_state(),
    }
    payload: dict[str, Any] = {
        "schema_version": TRACE_CHECKPOINT_SCHEMA_VERSION,
        "method": method,
        "epoch": epoch,
        "best_dev_loss": float(best_dev_loss),
        "best_epoch": int(best_epoch),
        "checkpoint_role": checkpoint_role,
        "selection": {
            "split": "dev",
            "criterion": "mean_proper_nll",
            "direction": "minimize",
            "tie_break": "earliest_epoch",
            "test_access": False,
        },
        "run_config": dict(run_config),
        "run_config_sha256": canonical_json_sha256(run_config),
        "data_provenance": dict(data_provenance),
        "data_provenance_sha256": canonical_json_sha256(data_provenance),
        "gates": dict(gates),
        "geometry": dict(geometry),
        "geometry_sha256": geometry_sha256,
        "logk_cache_sha256": logk_cache_sha256,
        "front_provenance": dict(front_provenance),
        "front_provenance_sha256": canonical_json_sha256(front_provenance),
        "front_state_sha256": front_state_sha256,
        "source_sha256": dict(sources),
        "git": dict(git),
        "git_sha256": canonical_json_sha256(git),
        "runtime": dict(runtime),
        "runtime_sha256": canonical_json_sha256(runtime),
        "model_state": model_state,
        "head_state": head_state,
        "optimizer_state": optimizer_state,
        "rng_state": rng_state,
        "loader_generator_state": generator_state,
        "model_state_sha256": nested_state_sha256(model_state),
        "frozen_state_sha256": nested_state_sha256(_frozen_model_state(model_state)),
        "head_state_sha256": nested_state_sha256(head_state),
        "optimizer_state_sha256": nested_state_sha256(optimizer_state),
        "rng_state_sha256": nested_state_sha256(rng_state),
        "loader_generator_state_sha256": nested_state_sha256(generator_state),
    }
    payload["checkpoint_manifest_sha256"] = canonical_json_sha256(
        _checkpoint_manifest(payload)
    )
    return payload


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write one checkpoint atomically without creating random filenames."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older supported PyTorch
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise TraceTrainingError("training checkpoint root must be a mapping")
    return payload


def restore_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_method: str,
    expected_run_config_sha256: str,
    expected_data_provenance_sha256: str,
    expected_gates: Mapping[str, Any],
    expected_geometry_sha256: str,
    expected_logk_cache_sha256: str | None,
    expected_front_state_sha256: str,
    expected_sources: Mapping[str, str],
    model: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    dev_generator: torch.Generator,
) -> tuple[int, float, int]:
    """Authenticate, load, and exactly restore one complete restart point."""

    required = {
        "schema_version",
        "method",
        "epoch",
        "best_dev_loss",
        "best_epoch",
        "checkpoint_role",
        "selection",
        "run_config",
        "run_config_sha256",
        "data_provenance",
        "data_provenance_sha256",
        "gates",
        "geometry",
        "geometry_sha256",
        "logk_cache_sha256",
        "front_provenance",
        "front_provenance_sha256",
        "front_state_sha256",
        "source_sha256",
        "model_state",
        "head_state",
        "optimizer_state",
        "rng_state",
        "loader_generator_state",
        "model_state_sha256",
        "frozen_state_sha256",
        "head_state_sha256",
        "optimizer_state_sha256",
        "rng_state_sha256",
        "loader_generator_state_sha256",
        "checkpoint_manifest_sha256",
        "git",
        "git_sha256",
        "runtime",
        "runtime_sha256",
    }
    missing = required.difference(payload)
    if missing:
        raise TraceTrainingError("checkpoint is incomplete: missing %s" % sorted(missing))
    if payload.get("schema_version") != TRACE_CHECKPOINT_SCHEMA_VERSION:
        raise TraceTrainingError("unsupported TRACE training checkpoint schema")
    declared_manifest = payload.get("checkpoint_manifest_sha256")
    if declared_manifest != canonical_json_sha256(_checkpoint_manifest(payload)):
        raise TraceTrainingError("checkpoint manifest authentication failed")
    comparisons = {
        "method": (payload.get("method"), expected_method),
        "run config": (payload.get("run_config_sha256"), expected_run_config_sha256),
        "data provenance": (
            payload.get("data_provenance_sha256"),
            expected_data_provenance_sha256,
        ),
        "gates": (
            canonical_json_sha256(payload.get("gates")),
            canonical_json_sha256(expected_gates),
        ),
        "geometry": (payload.get("geometry_sha256"), expected_geometry_sha256),
        "logK cache": (
            payload.get("logk_cache_sha256"),
            expected_logk_cache_sha256,
        ),
        "frozen front": (
            payload.get("front_state_sha256"),
            expected_front_state_sha256,
        ),
        "source inventory": (dict(payload.get("source_sha256", {})), dict(expected_sources)),
    }
    for label, (actual, expected) in comparisons.items():
        if actual != expected:
            raise TraceTrainingError("resume %s mismatch" % label)
    if payload.get("selection") != {
        "split": "dev",
        "criterion": "mean_proper_nll",
        "direction": "minimize",
        "tie_break": "earliest_epoch",
        "test_access": False,
    }:
        raise TraceTrainingError("checkpoint selection protocol is not dev-only")
    if payload["run_config"].get("method") != expected_method:
        raise TraceTrainingError("checkpoint run config method mismatch")
    if payload["data_provenance"].get("test_assets_included") is not False:
        raise TraceTrainingError("training checkpoint contains test-asset access")
    hash_checks = (
        ("model_state", "model_state_sha256"),
        ("head_state", "head_state_sha256"),
        ("optimizer_state", "optimizer_state_sha256"),
        ("rng_state", "rng_state_sha256"),
        ("loader_generator_state", "loader_generator_state_sha256"),
    )
    for state_key, hash_key in hash_checks:
        if nested_state_sha256(payload[state_key]) != payload[hash_key]:
            raise TraceTrainingError("checkpoint %s authentication failed" % state_key)
    if canonical_json_sha256(payload["run_config"]) != payload["run_config_sha256"]:
        raise TraceTrainingError("checkpoint run config content changed")
    if canonical_json_sha256(payload["data_provenance"]) != payload["data_provenance_sha256"]:
        raise TraceTrainingError("checkpoint data provenance content changed")
    try:
        checkpoint_geometry = TraceGeometrySpec.from_dict(payload["geometry"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TraceTrainingError("checkpoint geometry content is malformed") from exc
    if checkpoint_geometry.sha256 != payload["geometry_sha256"]:
        raise TraceTrainingError("checkpoint geometry content changed")
    for content_key, hash_key in (
        ("front_provenance", "front_provenance_sha256"),
        ("git", "git_sha256"),
        ("runtime", "runtime_sha256"),
    ):
        if canonical_json_sha256(payload[content_key]) != payload[hash_key]:
            raise TraceTrainingError("checkpoint %s content changed" % content_key)

    expected_frozen_hash = nested_state_sha256(
        _frozen_model_state(model.state_dict())
    )
    checkpoint_frozen_hash = nested_state_sha256(
        _frozen_model_state(payload["model_state"])
    )
    if (
        payload["frozen_state_sha256"] != checkpoint_frozen_hash
        or checkpoint_frozen_hash != expected_frozen_hash
    ):
        raise TraceTrainingError("checkpoint frozen front/geometry/logK state mismatch")

    model.load_state_dict(payload["model_state"], strict=True)
    if nested_state_sha256(head.state_dict()) != payload["head_state_sha256"]:
        raise TraceTrainingError("full model and separately authenticated head disagree")
    optimizer.load_state_dict(payload["optimizer_state"])
    generators = payload["loader_generator_state"]
    if not isinstance(generators, Mapping) or set(generators) != {"train", "dev"}:
        raise TraceTrainingError("checkpoint loader generator state is malformed")
    train_generator.set_state(generators["train"])
    dev_generator.set_state(generators["dev"])
    restore_rng_state(payload["rng_state"])

    integrity_check = getattr(model, "assert_front_integrity", None)
    if callable(integrity_check) and integrity_check() != expected_front_state_sha256:
        raise TraceTrainingError("restored frozen front failed its integrity anchor")

    epoch = payload.get("epoch")
    best_epoch = payload.get("best_epoch")
    best_dev_loss = payload.get("best_dev_loss")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch < 0
        or not math.isfinite(float(best_dev_loss))
    ):
        raise TraceTrainingError("checkpoint epoch/development state is malformed")
    if best_epoch > epoch or (
        payload.get("checkpoint_role") == "best_dev" and best_epoch != epoch
    ):
        raise TraceTrainingError("checkpoint best-development epoch is inconsistent")
    return epoch + 1, float(best_dev_loss), best_epoch


__all__ = [
    "TRACE_CHECKPOINT_SCHEMA_VERSION",
    "TRACE_METHODS",
    "TraceTrainingError",
    "atomic_torch_save",
    "build_training_checkpoint",
    "load_training_checkpoint",
    "make_paired_loaders",
    "nested_state_sha256",
    "restore_training_checkpoint",
    "seed_everything",
    "seed_loader_worker",
    "source_inventory",
]
