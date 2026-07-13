"""Fail-closed checkpoint and provenance utilities for TRACE experiments."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TraceProvenanceError(RuntimeError):
    """A required provenance invariant is absent or contradicted."""


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def provenance_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return "repo:" + resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "external:" + resolved.name


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    # ``view(dtype)`` rejects a zero-dimensional tensor when element sizes
    # differ (notably BatchNorm's scalar ``num_batches_tracked``).  Flattening
    # only the byte serialization view preserves the original shape in the
    # authenticated header while supporting every dense scalar/buffer tensor.
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return header + b"\0" + raw


def state_dict_sha256(
    state_dict: Mapping[str, torch.Tensor],
    *,
    keys: list[str] | tuple[str, ...] | None = None,
) -> str:
    selected = sorted(state_dict) if keys is None else list(keys)
    digest = hashlib.sha256()
    for key in selected:
        if key not in state_dict:
            raise TraceProvenanceError(f"state_dict is missing required tensor {key!r}")
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise TraceProvenanceError(f"state_dict value {key!r} is not a tensor")
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(value))
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Remove one uniform DataParallel prefix and reject ambiguous variants."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TraceProvenanceError("checkpoint net must be a non-empty mapping")
    keys = tuple(state_dict)
    if not all(isinstance(key, str) for key in keys):
        raise TraceProvenanceError("checkpoint net keys must be strings")
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise TraceProvenanceError("checkpoint mixes prefixed and unprefixed tensor keys")
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        target = key[7:] if all(prefixed) else key
        if target in normalized:
            raise TraceProvenanceError(f"duplicate normalized tensor key: {target}")
        if not torch.is_tensor(value):
            raise TraceProvenanceError(f"checkpoint net value {key!r} is not a tensor")
        normalized[target] = value
    return normalized


@dataclass(frozen=True)
class BaselineFrontProvenance:
    checkpoint_path: str
    checkpoint_sha256: str
    dataset: str
    seed: int
    train_split_sha256: str
    val_split_sha256: str
    source_epoch: int
    source_iou: float | None
    source_method: str
    source_model_type: str
    front_state_keys_sha256: str
    front_tensor_sha256: str
    bn_buffer_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_nonempty_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TraceProvenanceError(f"baseline method_meta requires non-empty {key}")
    return value


def load_clean_mshnet_checkpoint(
    checkpoint_path: str | Path,
    *,
    front_state_keys: tuple[str, ...],
    expected_dataset: str | None = None,
    expected_seed: int | None = None,
    expected_train_split_sha256: str | None = None,
    expected_val_split_sha256: str | None = None,
) -> tuple[OrderedDict[str, torch.Tensor], BaselineFrontProvenance, dict[str, Any]]:
    """Load a dev-selected canonical MSHNet and reject unsafe initialization."""

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "net" not in payload or "method_meta" not in payload:
        raise TraceProvenanceError(
            "TRACE requires a full baseline checkpoint with net and method_meta; raw weights are rejected"
        )
    meta = payload["method_meta"]
    if not isinstance(meta, Mapping):
        raise TraceProvenanceError("baseline method_meta must be a mapping")
    method = _require_nonempty_text(meta, "method")
    model_type = _require_nonempty_text(meta, "model_type")
    if method != "MSHNet" or model_type != "mshnet":
        raise TraceProvenanceError(
            f"baseline must be canonical MSHNet/mshnet, got {method}/{model_type}"
        )
    forbidden_semantics = {
        "selection_split": meta.get("selection_split"),
        "evaluation_split": meta.get("evaluation_split"),
    }
    if any(str(value).lower() == "test" for value in forbidden_semantics.values() if value):
        raise TraceProvenanceError(
            f"test-selected/evaluated baseline checkpoint is forbidden: {forbidden_semantics}"
        )
    protocol = str(meta.get("protocol", meta.get("protocol_version", ""))).lower()
    if "test_selected" in protocol or bool(meta.get("no_internal_holdout", False)):
        raise TraceProvenanceError("baseline protocol does not provide an internal dev holdout")

    dataset_dir = _require_nonempty_text(meta, "dataset_dir")
    dataset = Path(dataset_dir).name
    seed = int(meta.get("seed"))
    train_hash = _require_nonempty_text(meta, "train_split_sha256")
    val_hash = _require_nonempty_text(meta, "val_split_sha256")
    if expected_dataset is not None and dataset != expected_dataset:
        raise TraceProvenanceError(
            f"baseline dataset mismatch: expected {expected_dataset}, got {dataset}"
        )
    if expected_seed is not None and seed != int(expected_seed):
        raise TraceProvenanceError(
            f"baseline seed mismatch: expected {expected_seed}, got {seed}"
        )
    if expected_train_split_sha256 and train_hash != expected_train_split_sha256:
        raise TraceProvenanceError("baseline train split hash does not match paired protocol")
    if expected_val_split_sha256 and val_hash != expected_val_split_sha256:
        raise TraceProvenanceError("baseline validation split hash does not match paired protocol")

    normalized = normalize_state_dict_keys(payload["net"])
    missing = [key for key in front_state_keys if key not in normalized]
    if missing:
        raise TraceProvenanceError(
            f"baseline checkpoint lacks {len(missing)} front tensors, e.g. {missing[:3]}"
        )
    key_hash = hashlib.sha256(
        ("\n".join(front_state_keys) + "\n").encode("utf-8")
    ).hexdigest()
    bn_keys = tuple(
        key
        for key in front_state_keys
        if key.endswith("running_mean")
        or key.endswith("running_var")
        or key.endswith("num_batches_tracked")
    )
    provenance = BaselineFrontProvenance(
        checkpoint_path=provenance_path(path),
        checkpoint_sha256=sha256_file(path),
        dataset=dataset,
        seed=seed,
        train_split_sha256=train_hash,
        val_split_sha256=val_hash,
        source_epoch=int(payload.get("epoch", -1)),
        source_iou=(float(payload["iou"]) if payload.get("iou") is not None else None),
        source_method=method,
        source_model_type=model_type,
        front_state_keys_sha256=key_hash,
        front_tensor_sha256=state_dict_sha256(normalized, keys=front_state_keys),
        bn_buffer_sha256=state_dict_sha256(normalized, keys=bn_keys),
    )
    return normalized, provenance, dict(meta)


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream needed for an exact uninterrupted resume."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(payload)
    if missing:
        raise TraceProvenanceError(f"RNG checkpoint is missing {sorted(missing)}")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available():
        states = payload["torch_cuda"]
        if len(states) != torch.cuda.device_count():
            raise TraceProvenanceError("CUDA RNG state count differs from visible GPU count")
        torch.cuda.set_rng_state_all(states)


def git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty_lines = run("status", "--short").splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TraceProvenanceError(f"unable to audit git state: {exc}") from exc
    return {
        "commit": commit,
        "dirty": bool(dirty_lines),
        "dirty_paths": [line[3:] if len(line) > 3 else line for line in dirty_lines],
    }


def runtime_environment() -> dict[str, Any]:
    cuda_name = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "numpy": np.__version__,
    }


__all__ = [
    "BaselineFrontProvenance",
    "TraceProvenanceError",
    "canonical_json_sha256",
    "capture_rng_state",
    "git_state",
    "load_clean_mshnet_checkpoint",
    "normalize_state_dict_keys",
    "provenance_path",
    "restore_rng_state",
    "runtime_environment",
    "sha256_file",
    "state_dict_sha256",
]
