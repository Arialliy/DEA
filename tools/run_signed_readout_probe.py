#!/usr/bin/env python3
"""Gate K: fit-only signed readout probes on a frozen MSHNet d0 tensor.

This is a diagnostic gate, not a paper method.  One invocation evaluates one
clean fixed-epoch dataset/seed checkpoint.  MSHNet is kept in evaluation mode
and is executed under ``no_grad``; four tiny heads see the same d0 tensor in
every fit batch and are optimized with one pre-registered class-balanced BCE
protocol.
The official test split is neither constructed nor iterated.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import datetime as dt
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.MSHNet import MSHNet  # noqa: E402
from model.signed_local_reference import (  # noqa: E402
    CenteredLocalReferenceProbe,
    RawUnitLinearProbe,
    SignedStandardizedLocalReferenceProbe,
    UnsignedStandardizedProjectionControl,
)
from tools.audit_cross_seed_failure_persistence import (  # noqa: E402
    _normalize_state_dict,
    build_authoritative_target_registry,
    load_validated_jobs,
    sha256_file,
)
from tools.audit_rcp_gt_coverage import formal_hard_core_ids  # noqa: E402
from tools.finalize_clean_baselines import load_checkpoint_cpu  # noqa: E402
from utils.cross_fitted_low_fa import (  # noqa: E402
    BUDGETS,
    MATCHERS,
    cross_fit_job,
)
from utils.data import IRSTD_Dataset  # noqa: E402
from utils.metric import (  # noqa: E402
    match_components_hungarian,
    match_connected_components,
)
from utils.nested_component_grid import (  # noqa: E402
    build_nested_quantile_probability_grids,
    evaluate_nested_component_grids,
)
from utils.target_identity import (  # noqa: E402
    StableTargetSet,
    assert_same_target_set,
    build_stable_target_set,
)


SCHEMA = "dea.gate_k.signed_readout_probe.v1"
PROVENANCE_SCHEMA = "dea.gate_k.signed_readout_probe_provenance.v1"
TRAINING_SCHEMA = "dea.gate_k.signed_readout_training.v1"
ORACLE_TARGET_SCHEMA = "dea.gate_k.signed_readout_oracle_target.v1"
HARD_CORE_SCHEMA = "dea.gate_k.signed_readout_hard_core.v1"

# Pre-registered common protocol.  These are intentionally not CLI options.
PROBE_FORMAL_EPOCHS = 20
PROBE_SMOKE_EPOCHS = 10
PROBE_LR = 0.05
PROBE_BATCH_SIZE = 4
PROBE_OPTIMIZER = "torch.optim.Adagrad"
PROBE_LOSS = "per-image class-balanced binary_cross_entropy_with_logits"
ANNULUS_OUTER_SIZE = 9
ANNULUS_INNER_SIZE = 3
VARIANCE_FLOOR_SCALE = 1e-4

FROZEN_VARIANTS = ("original_final_z", "original_output0")
TRAINABLE_VARIANTS = (
    "refit_raw",
    "refit_annulus_centered",
    "refit_signed_standardized",
    "refit_unsigned_standardized_projection",
)
ALL_VARIANTS = (*FROZEN_VARIANTS, *TRAINABLE_VARIANTS)

BUNDLE_FILES = (
    "summary.json",
    "training_history.jsonl",
    "oracle_targets.jsonl",
    "crossfit_targets.jsonl",
    "crossfit_images.jsonl",
    "crossfit_calibration.jsonl",
    "hard_core_matching.jsonl",
    "dev_logits.npz",
    "probe_heads.pkl",
    "provenance.json",
)


class SignedReadoutProbeError(RuntimeError):
    """Raised when the frozen-readout diagnostic contract is violated."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument(
        "--front-freeze-dir",
        default="repro_runs/gate_i/front_freeze_confirmatory_v1",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--protocol",
        choices=("formal", "smoke"),
        default="formal",
        help=(
            "formal is the fixed 20-epoch protocol; smoke is the predeclared "
            "10-epoch NUAA-SIRST/20260711 engineering check"
        ),
    )
    return parser.parse_args(argv)


def protocol_epochs(protocol: str, dataset: str, seed: int) -> int:
    if protocol == "formal":
        return PROBE_FORMAL_EPOCHS
    if protocol == "smoke":
        if dataset != "NUAA-SIRST" or int(seed) != 20260711:
            raise SignedReadoutProbeError(
                "smoke protocol is predeclared only for NUAA-SIRST/20260711"
            )
        return PROBE_SMOKE_EPOCHS
    raise SignedReadoutProbeError(f"unknown probe protocol: {protocol}")


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SignedReadoutProbeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SignedReadoutProbeError(f"JSON artifact is not an object: {path}")
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not torch.is_tensor(tensor):
            raise SignedReadoutProbeError("state mapping contains a non-tensor value")
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _module_state_sha256(module: torch.nn.Module) -> str:
    return _state_sha256(module.state_dict())


def _array_sequence_sha256(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for index, raw in enumerate(values):
        value = np.asarray(raw, dtype=np.float32, order="C")
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _source_paths() -> dict[str, Path]:
    return {
        "tool": Path(__file__).resolve(),
        "mshnet": ROOT / "model" / "MSHNet.py",
        "signed_local_reference": ROOT / "model" / "signed_local_reference.py",
        "dataset": ROOT / "utils" / "data.py",
        "cross_fitted_low_fa": ROOT / "utils" / "cross_fitted_low_fa.py",
        "nested_component_grid": ROOT / "utils" / "nested_component_grid.py",
        "metric": ROOT / "utils" / "metric.py",
        "target_identity": ROOT / "utils" / "target_identity.py",
        "clean_job_validator": (
            ROOT / "tools" / "audit_cross_seed_failure_persistence.py"
        ),
        "clean_checkpoint_loader": ROOT / "tools" / "finalize_clean_baselines.py",
        "hard_core_validator": ROOT / "tools" / "audit_rcp_gt_coverage.py",
    }


def _source_hashes() -> dict[str, str]:
    paths = _source_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SignedReadoutProbeError(
            "Gate K source inventory is incomplete: " + ", ".join(missing)
        )
    return {name: sha256_file(path) for name, path in paths.items()}


def _runtime_versions() -> dict[str, Any]:
    def package_version(name: str) -> str:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cudnn": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "numpy": np.__version__,
        "scikit_image": package_version("scikit-image"),
        "scipy": package_version("scipy"),
        "torchvision": package_version("torchvision"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _batchnorm_state_sha256(model: torch.nn.Module) -> tuple[str, int]:
    state: dict[str, torch.Tensor] = {}
    count = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            count += 1
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, name, None)
                if value is not None:
                    state[f"{module_name}.{name}"] = value
    if not state or count == 0:
        raise SignedReadoutProbeError("MSHNet exposes no BatchNorm state")
    return _state_sha256(state), count


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seed_everything(seed: int) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SignedReadoutProbeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return {
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_cpu_seed": seed,
        "torch_cuda_all_seed": seed if cuda_available else None,
        "cuda_available": cuda_available,
        "pythonhashseed_environment_set_at_runtime": str(seed),
        "python_hash_randomization_current_process_not_reseedable": True,
        "fit_dataloader_generator_seed": seed,
        "dev_dataloader_generator_seed": seed + 1,
        "worker_seed_rule": "torch.initial_seed() % 2**32",
        "probe_initialization_seed": seed,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "torch_deterministic_algorithms": True,
    }


def _loader(
    dataset: IRSTD_Dataset,
    *,
    training: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=PROBE_BATCH_SIZE if training else PROBE_BATCH_SIZE,
        shuffle=training,
        drop_last=training,
        **kwargs,
    )


def validate_front_freeze_bundle(front_dir: Path) -> dict[str, Any]:
    """Verify the frozen routing artifact and return its formal target panel."""

    summary_path = front_dir / "front_freeze_summary.json"
    provenance_path = front_dir / "provenance.json"
    summary = _read_json(summary_path)
    provenance = _read_json(provenance_path)
    if summary.get("schema") != "dea.front_freeze_confirmatory.v1":
        raise SignedReadoutProbeError("front-freeze summary schema drifted")
    if summary.get("status") != "complete":
        raise SignedReadoutProbeError("front-freeze artifact is not complete")
    routing = summary.get("engineering_routing")
    if not isinstance(routing, dict) or (
        routing.get("recommendation")
        != "freeze_input_through_d0_for_next_structural_iteration"
        or routing.get("recommended_first_mutable_boundary")
        != "after_d0_prediction_conversion"
        or routing.get("not_a_scientific_gate") is not True
    ):
        raise SignedReadoutProbeError("front-freeze routing authority drifted")
    if provenance.get("schema") != "dea.front_freeze_provenance.v1":
        raise SignedReadoutProbeError("front-freeze provenance schema drifted")
    artifact_hashes = provenance.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise SignedReadoutProbeError("front-freeze artifact hashes are missing")
    for name, expected in artifact_hashes.items():
        path = front_dir / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise SignedReadoutProbeError(f"front-freeze artifact hash drifted: {name}")
    source_hashes = provenance.get("source_sha256")
    if not isinstance(source_hashes, dict) or (
        source_hashes.get("mshnet") != sha256_file(ROOT / "model" / "MSHNet.py")
    ):
        raise SignedReadoutProbeError("MSHNet differs from front-freeze authority")
    hard = provenance.get("hard_core_panel")
    if not isinstance(hard, dict) or not isinstance(hard.get("records"), list):
        raise SignedReadoutProbeError("front-freeze hard-core panel is missing")
    records = hard["records"]
    if len(records) != 16:
        raise SignedReadoutProbeError("front-freeze hard-core panel is not 16 targets")
    keys = {(str(row["dataset"]), str(row["stable_target_id"])) for row in records}
    if len(keys) != 16 or Counter(str(row["dataset"]) for row in records) != Counter(
        {"IRSTD-1K": 11, "NUDT-SIRST": 4, "NUAA-SIRST": 1}
    ):
        raise SignedReadoutProbeError("front-freeze hard-core identity scope drifted")
    if any(
        sorted(int(value) for value in row["source_seeds"])
        != [20260711, 20260712, 20260713]
        for row in records
    ):
        raise SignedReadoutProbeError("hard-core source seed scope drifted")
    hard_source = Path(str(hard.get("source", ""))).resolve()
    if (
        not hard_source.is_file()
        or sha256_file(hard_source) != hard.get("source_sha256")
        or source_hashes.get("hard_core_source") != hard.get("source_sha256")
    ):
        raise SignedReadoutProbeError("hard-core source hash drifted")
    if keys != set(formal_hard_core_ids(hard_source)):
        raise SignedReadoutProbeError(
            "front-freeze hard-core records disagree with their frozen source"
        )
    return {
        "directory": str(front_dir),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "provenance": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "hard_core_source": str(hard_source),
        "hard_core_source_sha256": hard["source_sha256"],
        "records": records,
        "routing": routing,
    }


def select_clean_job(batch_id: str, dataset: str, seed: int) -> tuple[dict, dict]:
    batch_dir = ROOT / "repro_runs" / "clean" / batch_id
    jobs, provenance = load_validated_jobs(batch_dir, policy="fixed_epoch")
    selected = [
        job
        for job in jobs
        if str(job["dataset"]) == dataset and int(job["seed"]) == seed
    ]
    if len(selected) != 1:
        raise SignedReadoutProbeError(
            f"clean fixed job must exist exactly once for {dataset}/{seed}"
        )
    job = selected[0]
    checkpoint = Path(str(job["checkpoint"]))
    if (
        checkpoint.name != "checkpoint.pkl"
        or job["checkpoint_summary"]["policy"] != "fixed_epoch"
        or sha256_file(checkpoint) != job["checkpoint_sha256"]
    ):
        raise SignedReadoutProbeError("selected checkpoint is not the clean fixed artifact")
    return job, provenance


def validate_gate_g_job_authority(
    front: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    """Bind this one-job replay to its exact frozen Gate-G source rows."""

    panel_ids = {
        str(row["stable_target_id"])
        for row in front["records"]
        if str(row["dataset"]) == dataset
    }
    rows = []
    source = Path(str(front["hard_core_source"]))
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SignedReadoutProbeError(
                        f"Gate-G source row {line_number} is not an object"
                    )
                if (
                    value.get("grid_level") == "Q2"
                    and int(value.get("nominal_budget_fa_per_mpix", -1)) == 20
                    and int(value.get("seed", -1)) == seed
                    and str(value.get("dataset")) == dataset
                    and str(value.get("stable_target_id")) in panel_ids
                ):
                    rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SignedReadoutProbeError(f"cannot validate Gate-G job rows: {exc}") from exc
    if len(rows) != len(panel_ids) or {
        str(row["stable_target_id"]) for row in rows
    } != panel_ids:
        raise SignedReadoutProbeError("Gate-G one-job hard-core coverage drifted")
    for row in rows:
        checkpoint = row.get("checkpoint")
        if not isinstance(checkpoint, dict) or (
            checkpoint.get("sha256") != job["checkpoint_sha256"]
            or checkpoint.get("job_id") != job["job_id"]
            or checkpoint.get("validation_split_sha256")
            != job["split_hashes"]["validation"]
            or checkpoint.get("policy") != "fixed_epoch"
            or int(checkpoint.get("epoch", -1))
            != int(job["checkpoint_summary"]["epoch"])
        ):
            raise SignedReadoutProbeError(
                "Gate-G row does not reference the selected clean fixed checkpoint"
            )
        if (
            row.get("category_core") != "no_feasible_local_peak_activation"
            or bool(row.get("joint_global_oracle_matched"))
            or bool(row.get("selected_legacy_matched"))
            or bool(row.get("selected_hungarian_matched"))
        ):
            raise SignedReadoutProbeError("Gate-G formal hard-core phenotype drifted")
    rows.sort(key=lambda row: str(row["stable_target_id"]))
    return {
        "source": str(source),
        "source_sha256": front["hard_core_source_sha256"],
        "dataset": dataset,
        "seed": seed,
        "row_count": len(rows),
        "stable_target_ids": sorted(panel_ids),
        "rows_sha256": _sha256_json(rows),
        "checkpoint_sha256": job["checkpoint_sha256"],
        "validation_split_sha256": job["split_hashes"]["validation"],
    }


def build_probe_heads(seed: int, device: torch.device) -> torch.nn.ModuleDict:
    heads = torch.nn.ModuleDict(
        {
            "refit_raw": RawUnitLinearProbe(16, initialization_seed=seed),
            "refit_annulus_centered": CenteredLocalReferenceProbe(
                16,
                outer_size=ANNULUS_OUTER_SIZE,
                inner_size=ANNULUS_INNER_SIZE,
                initialization_seed=seed,
            ),
            "refit_signed_standardized": SignedStandardizedLocalReferenceProbe(
                16,
                outer_size=ANNULUS_OUTER_SIZE,
                inner_size=ANNULUS_INNER_SIZE,
                variance_floor_scale=VARIANCE_FLOOR_SCALE,
                initialization_seed=seed,
            ),
            "refit_unsigned_standardized_projection": UnsignedStandardizedProjectionControl(
                16,
                outer_size=ANNULUS_OUTER_SIZE,
                inner_size=ANNULUS_INNER_SIZE,
                variance_floor_scale=VARIANCE_FLOOR_SCALE,
                initialization_seed=seed,
            ),
        }
    ).to(device)
    if tuple(heads) != TRAINABLE_VARIANTS:
        raise SignedReadoutProbeError("probe variant order drifted")
    expected = {
        "refit_raw": 17,
        "refit_annulus_centered": 17,
        "refit_signed_standardized": 17,
        "refit_unsigned_standardized_projection": 17,
    }
    observed = {
        name: sum(parameter.numel() for parameter in head.parameters())
        for name, head in heads.items()
    }
    if observed != expected or any(
        not parameter.requires_grad for parameter in heads.parameters()
    ):
        raise SignedReadoutProbeError(f"probe parameter budget drifted: {observed}")
    return heads


def frozen_d0_and_native_logits(
    model: MSHNet,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the actual warm graph once and expose d0, output_0, and final z."""

    if model.training or any(module.training for module in model.modules()):
        raise SignedReadoutProbeError("frozen MSHNet must remain fully in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SignedReadoutProbeError("frozen MSHNet exposes a trainable parameter")
    captured: list[torch.Tensor] = []

    def capture(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured.append(output)

    handle = model.decoder_0.register_forward_hook(capture)
    try:
        with torch.no_grad():
            masks, final_z = model(images, True)
    finally:
        handle.remove()
    if len(captured) != 1 or len(masks) != 4:
        raise SignedReadoutProbeError("d0 hook/native side output accounting drifted")
    d0 = captured[0]
    output0 = masks[0]
    expected_shape = (images.shape[0], 1, images.shape[-2], images.shape[-1])
    if (
        d0.ndim != 4
        or d0.shape[1] != 16
        or tuple(output0.shape) != expected_shape
        or tuple(final_z.shape) != expected_shape
        or d0.requires_grad
        or output0.requires_grad
        or final_z.requires_grad
        or not bool(torch.isfinite(d0).all())
        or not bool(torch.isfinite(output0).all())
        or not bool(torch.isfinite(final_z).all())
    ):
        raise SignedReadoutProbeError("frozen d0/native logits are invalid")
    return d0, output0, final_z


def forward_probe_heads(
    heads: torch.nn.ModuleDict,
    d0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = {name: heads[name](d0) for name in TRAINABLE_VARIANTS}
    if any(
        tuple(value.shape) != (d0.shape[0], 1, d0.shape[2], d0.shape[3])
        or not bool(torch.isfinite(value).all())
        for value in result.values()
    ):
        raise SignedReadoutProbeError("a probe produced invalid logits")
    return result


def class_balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Mean per-image BCE with equal foreground/background class weight.

    Images without foreground use the background mean alone.  This makes the
    empty-image behavior explicit and avoids SoftIoU's all-off attraction for
    tiny targets on a 256x256 canvas.
    """

    if (
        not torch.is_tensor(logits)
        or not torch.is_tensor(targets)
        or logits.shape != targets.shape
        or logits.ndim != 4
        or logits.shape[1] != 1
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(targets).all())
    ):
        raise SignedReadoutProbeError("class-balanced BCE inputs are invalid")
    if bool(torch.any((targets < 0) | (targets > 1))):
        raise SignedReadoutProbeError("class-balanced BCE targets must lie in [0,1]")
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
    )
    image_losses = []
    foreground_images = 0
    empty_images = 0
    for image_index in range(int(logits.shape[0])):
        target = targets[image_index] > 0.5
        loss_map = losses[image_index]
        if bool(target.any()):
            if not bool((~target).any()):
                raise SignedReadoutProbeError("a fit image has no background pixels")
            image_loss = 0.5 * loss_map[target].mean() + 0.5 * loss_map[~target].mean()
            foreground_images += 1
        else:
            image_loss = loss_map.mean()
            empty_images += 1
        image_losses.append(image_loss)
    if not image_losses:
        raise SignedReadoutProbeError("class-balanced BCE received an empty batch")
    result = torch.stack(image_losses).mean()
    if not bool(torch.isfinite(result)):
        raise SignedReadoutProbeError("class-balanced BCE became non-finite")
    return result, {
        "foreground_images": foreground_images,
        "empty_images": empty_images,
    }


def train_probe_heads(
    model: MSHNet,
    heads: torch.nn.ModuleDict,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train four independent heads jointly on exactly shared frozen d0 batches."""

    model.eval()
    model.requires_grad_(False)
    heads.train()
    initial_hashes = {name: _module_state_sha256(head) for name, head in heads.items()}
    optimizer = torch.optim.Adagrad(heads.parameters(), lr=PROBE_LR)
    history: list[dict[str, Any]] = []
    total_steps = 0
    if epochs not in (PROBE_FORMAL_EPOCHS, PROBE_SMOKE_EPOCHS):
        raise SignedReadoutProbeError("probe epoch count is not pre-registered")
    for epoch in range(epochs):
        sums = {name: 0.0 for name in TRAINABLE_VARIANTS}
        images_seen = 0
        batches = 0
        foreground_images = 0
        empty_images = 0
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise SignedReadoutProbeError("fit loader must return image/mask pairs")
            images, masks = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            d0, _, _ = frozen_d0_and_native_logits(model, images)
            outputs = forward_probe_heads(heads, d0)
            losses: dict[str, torch.Tensor] = {}
            batch_class_counts: dict[str, int] | None = None
            for name, value in outputs.items():
                loss, class_counts = class_balanced_bce(value, masks)
                losses[name] = loss
                if batch_class_counts is None:
                    batch_class_counts = class_counts
                elif batch_class_counts != class_counts:
                    raise SignedReadoutProbeError("variant class accounting drifted")
            if any(not bool(torch.isfinite(value)) for value in losses.values()):
                raise SignedReadoutProbeError("probe training loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            # No averaging: every disjoint parameter block receives exactly
            # the gradient of its own class-balanced BCE objective.
            sum(losses.values()).backward()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise SignedReadoutProbeError("frozen MSHNet received a gradient")
            if any(
                parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                for parameter in heads.parameters()
            ):
                raise SignedReadoutProbeError("a probe gradient is absent/non-finite")
            optimizer.step()
            batch_size = int(images.shape[0])
            for name, value in losses.items():
                sums[name] += float(value.detach()) * batch_size
            images_seen += batch_size
            batches += 1
            total_steps += 1
            if batch_class_counts is None:
                raise AssertionError("unreachable empty variant set")
            foreground_images += batch_class_counts["foreground_images"]
            empty_images += batch_class_counts["empty_images"]
        if batches == 0 or images_seen == 0:
            raise SignedReadoutProbeError("fit loader produced no complete batch")
        for name in TRAINABLE_VARIANTS:
            history.append(
                {
                    "schema": TRAINING_SCHEMA,
                    "epoch": epoch,
                    "variant": name,
                    "mean_class_balanced_bce": sums[name] / images_seen,
                    "images": images_seen,
                    "foreground_images": foreground_images,
                    "empty_images": empty_images,
                    "batches": batches,
                    "shared_d0_batches": True,
                }
            )
    heads.eval()
    final_hashes = {name: _module_state_sha256(head) for name, head in heads.items()}
    if any(final_hashes[name] == initial_hashes[name] for name in TRAINABLE_VARIANTS):
        raise SignedReadoutProbeError("a trained probe state did not change")
    return history, {
        "optimizer": PROBE_OPTIMIZER,
        "learning_rate": PROBE_LR,
        "loss": PROBE_LOSS,
        "epochs": epochs,
        "batch_size": PROBE_BATCH_SIZE,
        "drop_last": True,
        "shuffle": True,
        "augmentation": "IRSTD_Dataset(mode=train), identical shared batch",
        "variants": list(TRAINABLE_VARIANTS),
        "variant_specific_hyperparameters": False,
        "trainable_parameters_per_variant": {
            name: sum(parameter.numel() for parameter in heads[name].parameters())
            for name in TRAINABLE_VARIANTS
        },
        "joint_gradient_semantics": (
            "sum of four losses without averaging; parameter blocks are disjoint"
        ),
        "weight_decay": 0.0,
        "annulus_outer_size": ANNULUS_OUTER_SIZE,
        "annulus_inner_size": ANNULUS_INNER_SIZE,
        "variance_floor_scale": VARIANCE_FLOOR_SCALE,
        "total_optimizer_steps": total_steps,
        "initial_state_sha256": initial_hashes,
        "final_state_sha256": final_hashes,
    }


def infer_development(
    model: MSHNet,
    heads: torch.nn.ModuleDict,
    loader: DataLoader,
    dataset: IRSTD_Dataset,
    authority: Mapping[str, StableTargetSet],
    *,
    dataset_name: str,
    device: torch.device,
) -> tuple[dict[str, tuple[np.ndarray, ...]], tuple[np.ndarray, ...]]:
    model.eval()
    heads.eval()
    logits: dict[str, list[np.ndarray]] = {name: [] for name in ALL_VARIANTS}
    targets: list[np.ndarray] = []
    cursor = 0
    with torch.no_grad():
        for images, masks in loader:
            images_device = images.to(device, non_blocking=True)
            d0, output0, final_z = frozen_d0_and_native_logits(model, images_device)
            outputs = forward_probe_heads(heads, d0)
            batch_outputs = {
                "original_final_z": final_z,
                "original_output0": output0,
                **outputs,
            }
            for name in ALL_VARIANTS:
                logits[name].extend(
                    batch_outputs[name][:, 0].detach().float().cpu().numpy()
                )
            target_arrays = (masks[:, 0] > 0.5).numpy().astype(bool, copy=False)
            for batch_index, target in enumerate(target_arrays):
                image_name = dataset.names[cursor + batch_index]
                observed = build_stable_target_set(
                    target,
                    dataset=dataset_name,
                    image_name=image_name,
                    connectivity=2,
                )
                try:
                    assert_same_target_set(authority[image_name], observed)
                except Exception as exc:
                    raise SignedReadoutProbeError(
                        f"development target authority drifted: {exc}"
                    ) from exc
                targets.append(target)
            cursor += int(images.shape[0])
    if cursor != len(dataset) or tuple(authority) != tuple(dataset.names):
        raise SignedReadoutProbeError("development inference coverage drifted")
    frozen = {name: tuple(values) for name, values in logits.items()}
    if any(len(values) != len(dataset) for values in frozen.values()):
        raise SignedReadoutProbeError("development logit sample count drifted")
    return frozen, tuple(targets)


def _match(
    scores: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    matcher: str,
):
    if matcher == "official_legacy":
        return match_connected_components(
            scores > threshold,
            target,
            max_centroid_distance=3.0,
            connectivity=2,
        )
    if matcher == "audit_hungarian":
        return match_components_hungarian(
            scores > threshold,
            target,
            centroid_radius=3.0,
            connectivity=2,
        )
    raise SignedReadoutProbeError(f"unknown matcher: {matcher}")


def _pooled_crossfit(
    image_rows: Sequence[Mapping[str, Any]], matcher: str, budget: int
) -> dict[str, Any]:
    selected = [
        row
        for row in image_rows
        if row["matcher"] == matcher
        and int(row["nominal_budget_fa_per_mpix"]) == budget
    ]
    if not selected:
        raise SignedReadoutProbeError("cross-fit aggregate group is empty")
    aggregate = selected[0]["dataset_seed_aggregate"]
    if any(row["dataset_seed_aggregate"] != aggregate for row in selected):
        raise SignedReadoutProbeError("cross-fit dataset aggregate drifted")
    folds: dict[str, Any] = {}
    for row in selected:
        key = str(int(row["evaluation_fold"]))
        value = row["held_out_fold_aggregate"]
        if key in folds and folds[key] != value:
            raise SignedReadoutProbeError("cross-fit fold aggregate drifted")
        folds[key] = value
    if set(folds) != {"0", "1"}:
        raise SignedReadoutProbeError("cross-fit aggregate lacks a fold")
    return {
        **aggregate,
        "held_out_folds": folds,
        "all_held_out_folds_feasible": all(
            bool(value["budget_feasible_zero_overshoot"])
            for value in folds.values()
        ),
    }


def _pooled_pixel_iou(
    scores: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    thresholds: Mapping[int, float],
) -> dict[str, Any]:
    """Return exact pooled pixel counts for one deterministic threshold route."""

    if set(thresholds) != set(range(len(scores))) or len(scores) != len(targets):
        raise SignedReadoutProbeError("pixel-IoU threshold route lacks an image")
    intersection = 0
    union = 0
    prediction_pixels = 0
    target_pixels = 0
    for index, (raw_scores, raw_target) in enumerate(zip(scores, targets)):
        score = np.asarray(raw_scores)
        target = np.asarray(raw_target, dtype=bool)
        threshold = float(thresholds[index])
        if (
            score.shape != target.shape
            or score.ndim != 2
            or not bool(np.isfinite(score).all())
            or not np.isfinite(threshold)
        ):
            raise SignedReadoutProbeError("invalid pixel-IoU sample or threshold")
        prediction = score > threshold
        intersection += int(np.logical_and(prediction, target).sum())
        union += int(np.logical_or(prediction, target).sum())
        prediction_pixels += int(prediction.sum())
        target_pixels += int(target.sum())
    iou = float(intersection / union) if union else 1.0
    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "prediction_pixels": prediction_pixels,
        "target_pixels": target_pixels,
        "iou": iou,
        "strict_prediction_rule": "logit > threshold",
    }


def _oracle_target_rows(
    scores: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    names: Sequence[str],
    registry: Mapping[str, StableTargetSet],
    nested: Any,
    *,
    dataset: str,
    seed: int,
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for matcher in MATCHERS:
        level = nested.matcher(matcher).level("Q2")
        for selection in level.selections:
            budget = int(selection.budget_fa_per_million_pixels)
            threshold = float(selection.threshold)
            for index, (score, target, image_name) in enumerate(
                zip(scores, targets, names)
            ):
                matched = _match(
                    score,
                    target,
                    threshold=threshold,
                    matcher=matcher,
                )
                matched_targets = {int(value[0]) for value in matched.matches}
                target_set = registry[image_name]
                source = {
                    identity.source_component_index: identity
                    for identity in target_set.targets
                }
                if set(source) != set(range(len(matched.target_regions))):
                    raise SignedReadoutProbeError("oracle target identity order drifted")
                for source_index in range(len(matched.target_regions)):
                    identity = source[source_index]
                    rows.append(
                        {
                            "schema": ORACLE_TARGET_SCHEMA,
                            "dataset": dataset,
                            "seed": seed,
                            "variant": variant,
                            "image_name": image_name,
                            "image_index": index,
                            "stable_target_id": identity.stable_key,
                            "component_mask_sha256": identity.component_mask_sha256,
                            "matcher": matcher,
                            "nominal_budget_fa_per_mpix": budget,
                            "oracle_threshold": threshold,
                            "oracle_matched": source_index in matched_targets,
                            "semantics": "same-development-set post-hoc Q2 oracle",
                        }
                    )
    return rows


def evaluate_variant(
    scores: tuple[np.ndarray, ...],
    targets: tuple[np.ndarray, ...],
    names: tuple[str, ...],
    registry: Mapping[str, StableTargetSet],
    *,
    dataset: str,
    seed: int,
    variant: str,
    checkpoint_record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict], list[dict], list[dict], list[dict]]:
    probability_grids = build_nested_quantile_probability_grids()
    q2_probabilities = probability_grids[-1].probabilities
    nested = evaluate_nested_component_grids(
        scores,
        targets,
        BUDGETS,
        fixed_thresholds=(0.0,),
        base_quantiles=probability_grids[0].probabilities,
    )
    if nested.probability_grids[-1].probabilities != q2_probabilities:
        raise SignedReadoutProbeError("nested oracle Q2 probabilities drifted")
    oracle = {
        matcher: {
            str(selection.budget_fa_per_million_pixels): {
                **asdict(selection.operating_point),
                "budget_fa_per_million_pixels": int(
                    selection.budget_fa_per_million_pixels
                ),
            }
            for selection in nested.matcher(matcher).level("Q2").selections
        }
        for matcher in MATCHERS
    }
    oracle_selection_audit: dict[str, Any] = {}
    for matcher in MATCHERS:
        q2_level = nested.matcher(matcher).level("Q2")
        all_off_candidates = [
            point for point in q2_level.curve if int(point.prediction_components) == 0
        ]
        if not all_off_candidates:
            raise SignedReadoutProbeError("oracle Q2 curve lacks an all-off candidate")
        oracle_selection_audit[matcher] = {
            "all_off_candidate_present": True,
            "all_off_candidate_count": len(all_off_candidates),
            "by_budget": {
                str(selection.budget_fa_per_million_pixels): {
                    "selected_all_off": int(
                        selection.operating_point.prediction_components
                    )
                    == 0,
                    "selected_matched_zero": int(
                        selection.operating_point.matched_components
                    )
                    == 0,
                }
                for selection in q2_level.selections
            },
        }
    target_rows, image_rows, calibration_rows = cross_fit_job(
        scores,
        targets,
        names,
        dataset=dataset,
        seed=seed,
        registry=registry,
        checkpoint={**dict(checkpoint_record), "variant": variant},
        tail_quantiles=q2_probabilities,
    )
    for rows in (target_rows, image_rows, calibration_rows):
        for row in rows:
            row["variant"] = variant
            row["grid_level"] = "Q2"
    oracle_rows = _oracle_target_rows(
        scores,
        targets,
        names,
        registry,
        nested,
        dataset=dataset,
        seed=seed,
        variant=variant,
    )
    crossfit = {
        matcher: {
            str(budget): _pooled_crossfit(image_rows, matcher, budget)
            for budget in BUDGETS
        }
        for matcher in MATCHERS
    }
    fixed_logit0_pixel = _pooled_pixel_iou(
        scores,
        targets,
        {index: 0.0 for index in range(len(scores))},
    )
    crossfit_pixel: dict[str, dict[str, Any]] = {}
    for matcher in MATCHERS:
        crossfit_pixel[matcher] = {}
        for budget in BUDGETS:
            selected_images = [
                row
                for row in image_rows
                if row["matcher"] == matcher
                and int(row["nominal_budget_fa_per_mpix"]) == budget
            ]
            threshold_by_image: dict[int, float] = {}
            for row in selected_images:
                image_index = int(row["image_index"])
                if image_index in threshold_by_image:
                    raise SignedReadoutProbeError(
                        "cross-fit pixel-IoU image threshold is duplicated"
                    )
                threshold_by_image[image_index] = float(
                    row["calibration_threshold"]
                )
            crossfit_pixel[matcher][str(budget)] = _pooled_pixel_iou(
                scores,
                targets,
                threshold_by_image,
            )
    expected_target_count = sum(len(value.targets) for value in registry.values())
    for matcher in MATCHERS:
        for budget in BUDGETS:
            oracle_group = [
                row
                for row in oracle_rows
                if row["matcher"] == matcher
                and int(row["nominal_budget_fa_per_mpix"]) == budget
            ]
            crossfit_group = [
                row
                for row in target_rows
                if row["matcher"] == matcher
                and int(row["nominal_budget_fa_per_mpix"]) == budget
            ]
            if len(oracle_group) != expected_target_count or len(
                crossfit_group
            ) != expected_target_count:
                raise SignedReadoutProbeError(
                    "target-level metric ledger does not cover the authority"
                )
            if sum(bool(row["oracle_matched"]) for row in oracle_group) != int(
                oracle[matcher][str(budget)]["matched_components"]
            ):
                raise SignedReadoutProbeError(
                    "oracle target ledger disagrees with its operating point"
                )
            if sum(bool(row["low_fa_matched"]) for row in crossfit_group) != int(
                crossfit[matcher][str(budget)]["matched_components"]
            ):
                raise SignedReadoutProbeError(
                    "cross-fit target ledger disagrees with its pooled aggregate"
                )
    crossfit_selection_audit: dict[str, Any] = {}
    for matcher in MATCHERS:
        crossfit_selection_audit[matcher] = {}
        for budget in BUDGETS:
            selected_calibrations = [
                row
                for row in calibration_rows
                if row["matcher"] == matcher
            ]
            if len(selected_calibrations) != 2:
                raise SignedReadoutProbeError(
                    "cross-fit calibration must contain exactly two folds"
                )
            selections = [row["selections"][str(budget)] for row in selected_calibrations]
            candidate_presence = [
                any(int(point["prediction_components"]) == 0 for point in row["curve"])
                for row in selected_calibrations
            ]
            if candidate_presence != [True, True]:
                raise SignedReadoutProbeError(
                    "a cross-fit calibration Q2 curve lacks an all-off candidate"
                )
            held_out = crossfit[matcher][str(budget)]["held_out_folds"]
            crossfit_selection_audit[matcher][str(budget)] = {
                "calibration_folds": 2,
                "calibration_all_off_candidate_present_folds": sum(
                    candidate_presence
                ),
                "calibration_selected_all_off_folds": sum(
                    int(point["prediction_components"]) == 0 for point in selections
                ),
                "calibration_selected_matched_zero_folds": sum(
                    int(point["matched_components"]) == 0 for point in selections
                ),
                "held_out_overshoot_folds": sum(
                    not bool(value["budget_feasible_zero_overshoot"])
                    for value in held_out.values()
                ),
                "held_out_all_off_folds": sum(
                    int(value["prediction_components"]) == 0
                    for value in held_out.values()
                ),
                "pooled_evaluation_all_off": int(
                    crossfit[matcher][str(budget)]["prediction_components"]
                )
                == 0,
                "all_held_out_folds_feasible": bool(
                    crossfit[matcher][str(budget)]["all_held_out_folds_feasible"]
                ),
            }
    return (
        {
            "oracle_q2": oracle,
            "oracle_selection_audit": oracle_selection_audit,
            "crossfit_q2": crossfit,
            "fixed_logit0_pixel": fixed_logit0_pixel,
            "crossfit_pixel": crossfit_pixel,
            "crossfit_selection_audit": crossfit_selection_audit,
            "q2_probability_count": len(q2_probabilities),
            "q2_probabilities_sha256": _sha256_json(list(q2_probabilities)),
            "development_logit_sha256": _array_sequence_sha256(scores),
        },
        oracle_rows,
        target_rows,
        image_rows,
        calibration_rows,
    )


def build_hard_core_matching(
    panel: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    crossfit_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    seed: int,
) -> list[dict[str, Any]]:
    selected_panel = [row for row in panel if str(row["dataset"]) == dataset]
    expected_counts = {"NUAA-SIRST": 1, "NUDT-SIRST": 4, "IRSTD-1K": 11}
    if len(selected_panel) != expected_counts.get(dataset, -1):
        raise SignedReadoutProbeError("dataset-specific hard-core panel count drifted")
    panel_by_id = {str(row["stable_target_id"]): row for row in selected_panel}
    if len(panel_by_id) != len(selected_panel):
        raise SignedReadoutProbeError("dataset hard-core panel contains duplicate IDs")
    oracle_index: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    crossfit_index: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for rows, index in ((oracle_rows, oracle_index), (crossfit_rows, crossfit_index)):
        for row in rows:
            stable_id = str(row["stable_target_id"])
            if stable_id not in panel_by_id:
                continue
            key = (
                str(row["variant"]),
                str(row["matcher"]),
                int(row["nominal_budget_fa_per_mpix"]),
                stable_id,
            )
            if key in index:
                raise SignedReadoutProbeError("hard-core evaluation key is duplicated")
            index[key] = row
    result: list[dict[str, Any]] = []
    expected_keys = {
        (variant, matcher, budget, stable_id)
        for variant in ALL_VARIANTS
        for matcher in MATCHERS
        for budget in BUDGETS
        for stable_id in panel_by_id
    }
    if set(oracle_index) != expected_keys or set(crossfit_index) != expected_keys:
        raise SignedReadoutProbeError("hard-core evaluation coverage is incomplete")
    for key in sorted(expected_keys):
        variant, matcher, budget, stable_id = key
        panel_row = panel_by_id[stable_id]
        oracle = oracle_index[key]
        crossfit = crossfit_index[key]
        if str(oracle["image_name"]) != str(panel_row["image_name"]) or str(
            crossfit["image_name"]
        ) != str(panel_row["image_name"]):
            raise SignedReadoutProbeError("hard-core image identity drifted")
        result.append(
            {
                "schema": HARD_CORE_SCHEMA,
                "dataset": dataset,
                "seed": seed,
                "variant": variant,
                "matcher": matcher,
                "nominal_budget_fa_per_mpix": budget,
                "stable_target_id": stable_id,
                "image_name": panel_row["image_name"],
                "target_area": int(panel_row["target_area"]),
                "oracle_matched": bool(oracle["oracle_matched"]),
                "oracle_threshold": float(oracle["oracle_threshold"]),
                "crossfit_matched": bool(crossfit["low_fa_matched"]),
                "crossfit_threshold": float(crossfit["calibration_threshold"]),
                "crossfit_evaluation_fold": int(crossfit["evaluation_fold"]),
                "variant_fixed_logit0_matched": bool(
                    crossfit["fixed_logit0_matched"]
                ),
            }
        )
    return result


def summarize_hard_core(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in ALL_VARIANTS:
        result[variant] = {}
        for matcher in MATCHERS:
            result[variant][matcher] = {}
            for budget in BUDGETS:
                selected = [
                    row
                    for row in rows
                    if row["variant"] == variant
                    and row["matcher"] == matcher
                    and int(row["nominal_budget_fa_per_mpix"]) == budget
                ]
                if not selected:
                    raise SignedReadoutProbeError("hard-core summary group is empty")
                result[variant][matcher][str(budget)] = {
                    "targets": len(selected),
                    "oracle_matched": sum(bool(row["oracle_matched"]) for row in selected),
                    "crossfit_matched": sum(
                        bool(row["crossfit_matched"]) for row in selected
                    ),
                }
    return result


def assert_native_final_hard_core_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_targets: int,
) -> None:
    """Fail if the frozen native final contradicts the formal Gate-G panel."""

    selected = [
        row
        for row in rows
        if row["variant"] == "original_final_z"
        and int(row["nominal_budget_fa_per_mpix"]) == 20
    ]
    if len(selected) != expected_targets * len(MATCHERS):
        raise SignedReadoutProbeError(
            "native-final hard-core replay coverage is incomplete"
        )
    if any(bool(row["oracle_matched"]) for row in selected) or any(
        bool(row["crossfit_matched"]) for row in selected
    ):
        raise SignedReadoutProbeError(
            "native final contradicts the formal Q2/FA20 hard-core source"
        )


def _jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _write_bundle(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    crossfit_target_rows: Sequence[Mapping[str, Any]],
    crossfit_image_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    hard_core_rows: Sequence[Mapping[str, Any]],
    logits: Mapping[str, Sequence[np.ndarray]],
    image_names: Sequence[str],
    head_payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        _jsonl_write(temporary / "training_history.jsonl", history)
        _jsonl_write(temporary / "oracle_targets.jsonl", oracle_rows)
        _jsonl_write(temporary / "crossfit_targets.jsonl", crossfit_target_rows)
        _jsonl_write(temporary / "crossfit_images.jsonl", crossfit_image_rows)
        _jsonl_write(temporary / "crossfit_calibration.jsonl", calibration_rows)
        _jsonl_write(temporary / "hard_core_matching.jsonl", hard_core_rows)
        arrays: dict[str, Any] = {
            "image_names": np.asarray(tuple(image_names), dtype=np.str_),
        }
        for name in ALL_VARIANTS:
            arrays[name] = np.stack(logits[name]).astype(np.float32, copy=False)
        np.savez_compressed(temporary / "dev_logits.npz", **arrays)
        torch.save(dict(head_payload), temporary / "probe_heads.pkl")
        artifact_hashes = {
            name: sha256_file(temporary / name) for name in BUNDLE_FILES[:-1]
        }
        complete_provenance = {**dict(provenance), "artifact_sha256": artifact_hashes}
        (temporary / "provenance.json").write_text(
            json.dumps(
                complete_provenance,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if {path.name for path in temporary.iterdir()} != set(BUNDLE_FILES):
            raise SignedReadoutProbeError("temporary Gate K bundle inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.num_workers < 0:
        raise SignedReadoutProbeError("num_workers must be non-negative")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as exc:
        raise SignedReadoutProbeError(f"invalid device: {args.device}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SignedReadoutProbeError("CUDA requested but unavailable")

    probe_epochs = protocol_epochs(args.protocol, args.dataset, int(args.seed))
    source_hashes_before = _source_hashes()

    seed_record = seed_everything(int(args.seed))
    front = validate_front_freeze_bundle(_resolve(args.front_freeze_dir))
    job, clean_batch = select_clean_job(args.batch_id, args.dataset, int(args.seed))
    gate_g_job_authority = validate_gate_g_job_authority(
        front,
        job,
        dataset=args.dataset,
        seed=int(args.seed),
    )
    stored_args = Namespace(**job["stored_args"])
    fit_dataset = IRSTD_Dataset(stored_args, mode="train")
    dev_dataset = IRSTD_Dataset(stored_args, mode="val")
    if (
        fit_dataset.split_sha256 != job["split_hashes"]["fit"]
        or dev_dataset.split_sha256 != job["split_hashes"]["validation"]
    ):
        raise SignedReadoutProbeError("fit/development manifest hash drifted")
    authority, authority_record = build_authoritative_target_registry(
        job,
        batch_size=PROBE_BATCH_SIZE,
        num_workers=args.num_workers,
    )
    if tuple(authority) != tuple(dev_dataset.names):
        raise SignedReadoutProbeError("development authority order drifted")
    dataset_hard_ids = {
        str(row["stable_target_id"])
        for row in front["records"]
        if str(row["dataset"]) == args.dataset
    }
    authority_by_id = {
        target.stable_key: target
        for target_set in authority.values()
        for target in target_set.targets
    }
    if not dataset_hard_ids.issubset(authority_by_id):
        raise SignedReadoutProbeError("dataset hard-core IDs are absent from authority")
    for row in front["records"]:
        if str(row["dataset"]) != args.dataset:
            continue
        identity = authority_by_id[str(row["stable_target_id"])]
        if (
            identity.image_name != str(row["image_name"])
            or identity.area != int(row["target_area"])
        ):
            raise SignedReadoutProbeError(
                "front-freeze hard-core metadata disagrees with mask authority"
            )

    checkpoint_path = Path(str(job["checkpoint"])).resolve()
    checkpoint = load_checkpoint_cpu(checkpoint_path)
    state = checkpoint.get("net")
    if not isinstance(state, Mapping) or not state:
        raise SignedReadoutProbeError("clean checkpoint has no network state")
    model = MSHNet(3)
    model.load_state_dict(_normalize_state_dict(state), strict=True)
    del checkpoint, state
    model.requires_grad_(False).to(device).eval()
    backbone_before = _module_state_sha256(model)
    bn_before, bn_count = _batchnorm_state_sha256(model)

    heads = build_probe_heads(int(args.seed), device)
    original_output0_state_sha256 = _module_state_sha256(model.output_0)
    original_final_state_sha256 = _state_sha256(
        {
            f"{module_name}.{key}": value
            for module_name in ("output_0", "output_1", "output_2", "output_3", "final")
            for key, value in getattr(model, module_name).state_dict().items()
        }
    )
    initial_parameter_counts = {
        "original_final_z": {
            "reported_head_parameters": int(
                sum(module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                    for module in (model.output_0, model.output_1, model.output_2, model.output_3, model.final))
            ),
            "trained_parameters": 0,
        },
        "original_output0": {
            "reported_head_parameters": sum(
                value.numel() for value in model.output_0.parameters()
            ),
            "trained_parameters": 0,
        },
        **{
            name: {
                "reported_head_parameters": sum(
                    value.numel() for value in head.parameters()
                ),
                "trained_parameters": sum(
                    value.numel() for value in head.parameters() if value.requires_grad
                ),
            }
            for name, head in heads.items()
        },
    }

    fit_loader = _loader(
        fit_dataset,
        training=True,
        num_workers=args.num_workers,
        device=device,
        seed=int(args.seed),
    )
    history, training_protocol = train_probe_heads(
        model,
        heads,
        fit_loader,
        device=device,
        epochs=probe_epochs,
    )
    del fit_loader
    backbone_after_training = _module_state_sha256(model)
    bn_after_training, _ = _batchnorm_state_sha256(model)
    if backbone_after_training != backbone_before or bn_after_training != bn_before:
        raise SignedReadoutProbeError("frozen MSHNet/BatchNorm changed during fit")

    dev_loader = _loader(
        dev_dataset,
        training=False,
        num_workers=args.num_workers,
        device=device,
        seed=int(args.seed) + 1,
    )
    logits, targets = infer_development(
        model,
        heads,
        dev_loader,
        dev_dataset,
        authority,
        dataset_name=args.dataset,
        device=device,
    )
    del dev_loader
    backbone_after_inference = _module_state_sha256(model)
    bn_after_inference, _ = _batchnorm_state_sha256(model)
    if backbone_after_inference != backbone_before or bn_after_inference != bn_before:
        raise SignedReadoutProbeError("frozen MSHNet/BatchNorm changed during dev inference")

    checkpoint_record = {
        "policy": "fixed_epoch",
        "path": str(checkpoint_path),
        "sha256": job["checkpoint_sha256"],
        "epoch": int(job["checkpoint_summary"]["epoch"]),
        "job_id": job["job_id"],
    }
    evaluations: dict[str, Any] = {}
    all_oracle_rows: list[dict] = []
    all_target_rows: list[dict] = []
    all_image_rows: list[dict] = []
    all_calibration_rows: list[dict] = []
    names = tuple(dev_dataset.names)
    for variant in ALL_VARIANTS:
        evaluation, oracle_rows, target_rows, image_rows, calibration_rows = (
            evaluate_variant(
                logits[variant],
                targets,
                names,
                authority,
                dataset=args.dataset,
                seed=int(args.seed),
                variant=variant,
                checkpoint_record=checkpoint_record,
            )
        )
        evaluations[variant] = evaluation
        all_oracle_rows.extend(oracle_rows)
        all_target_rows.extend(target_rows)
        all_image_rows.extend(image_rows)
        all_calibration_rows.extend(calibration_rows)
    hard_rows = build_hard_core_matching(
        front["records"],
        all_oracle_rows,
        all_target_rows,
        dataset=args.dataset,
        seed=int(args.seed),
    )
    assert_native_final_hard_core_replay(
        hard_rows,
        expected_targets=len(dataset_hard_ids),
    )

    trained_state_hashes = {
        name: _module_state_sha256(head) for name, head in heads.items()
    }
    if trained_state_hashes != training_protocol["final_state_sha256"]:
        raise SignedReadoutProbeError("trained probe state hash drifted before save")
    if sha256_file(checkpoint_path) != job["checkpoint_sha256"]:
        raise SignedReadoutProbeError("clean checkpoint changed during Gate K")
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": (
            "fit-only head training; deterministic internal development evaluation; "
            "official test not constructed or iterated"
        ),
        "dataset": args.dataset,
        "seed": int(args.seed),
        "protocol": args.protocol,
        "checkpoint": checkpoint_record,
        "fit_images": len(fit_dataset),
        "development_images": len(dev_dataset),
        "development_targets": sum(len(value.targets) for value in authority.values()),
        "formal_hard_core_targets_in_dataset": len(dataset_hard_ids),
        "variants": {
            name: {
                **initial_parameter_counts[name],
                "training": "none" if name in FROZEN_VARIANTS else "common_protocol",
                "state_sha256": (
                    original_final_state_sha256
                    if name == "original_final_z"
                    else (
                        original_output0_state_sha256
                        if name == "original_output0"
                        else trained_state_hashes.get(name)
                    )
                ),
                **evaluations[name],
            }
            for name in ALL_VARIANTS
        },
        "training_protocol": training_protocol,
        "hard_core_q2": summarize_hard_core(hard_rows),
        "scientific_boundary": {
            "diagnostic_only": True,
            "same_development_q2_oracle_is_not_deployable_performance": True,
            "crossfit_is_internal_development_only": True,
            "does_not_establish_a_paper_method": True,
        },
    }

    head_payload = {
        "schema": SCHEMA,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "checkpoint": checkpoint_record,
        "training_protocol": training_protocol,
        "state_dict": {
            name: {key: value.detach().cpu() for key, value in head.state_dict().items()}
            for name, head in heads.items()
        },
    }
    source_hashes_after = _source_hashes()
    if source_hashes_after != source_hashes_before:
        raise SignedReadoutProbeError("Gate K source changed during execution")
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "protocol": args.protocol,
        "device": str(device),
        "runtime": _runtime_versions(),
        "rng": seed_record,
        "clean_batch": clean_batch,
        "job": job,
        "manifest_hashes": {
            "fit": fit_dataset.split_sha256,
            "development": dev_dataset.split_sha256,
            "official_test_provenance_only": job["split_hashes"][
                "official_test_provenance_only"
            ],
        },
        "authority": authority_record,
        "front_freeze": front,
        "gate_g_job_authority": gate_g_job_authority,
        "source_sha256": source_hashes_before,
        "source_specific_hashes_unchanged": True,
        "freeze_audit": {
            "model_eval_for_all_fit_and_dev_forwards": True,
            "model_requires_grad_false": True,
            "d0_extracted_under_no_grad": True,
            "shared_d0_once_per_fit_batch": True,
            "backbone_state_sha256_before": backbone_before,
            "backbone_state_sha256_after_training": backbone_after_training,
            "backbone_state_sha256_after_inference": backbone_after_inference,
            "batchnorm_module_count": bn_count,
            "batchnorm_state_sha256_before": bn_before,
            "batchnorm_state_sha256_after_training": bn_after_training,
            "batchnorm_state_sha256_after_inference": bn_after_inference,
        },
        "data_access": {
            "fit_dataset_constructed_and_iterated": True,
            "development_dataset_constructed_and_iterated": True,
            "official_test_dataset_constructed": False,
            "official_test_sample_iterated": False,
        },
        "variant_order": list(ALL_VARIANTS),
        "development_logit_shapes": {
            name: list(np.stack(values).shape) for name, values in logits.items()
        },
    }
    _write_bundle(
        output_dir,
        summary=summary,
        history=history,
        oracle_rows=all_oracle_rows,
        crossfit_target_rows=all_target_rows,
        crossfit_image_rows=all_image_rows,
        calibration_rows=all_calibration_rows,
        hard_core_rows=hard_rows,
        logits=logits,
        image_names=names,
        head_payload=head_payload,
        provenance=provenance,
    )
    del heads, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary, output_dir = run(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "dataset": summary["dataset"],
                "seed": summary["seed"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
