#!/usr/bin/env python3
"""Fail-closed paired training and dev-only inference for TRACE-MSHNet.

Both ``trace`` and ``dense_bernoulli`` use the same frozen canonical MSHNet
front, fit/dev assignment, deterministic augmentation, optimizer family, and
checkpoint protocol.  Training cannot start unless authenticated T0-A,
T0-B-DP, and T0-B-INTEGRATION reports all agree with the current source,
geometry, baseline checkpoint, and data manifests.  This command has no test
loader, test-inference mode, or test-export option.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.trace_mshnet import (  # noqa: E402
    MatchedDenseMSHNet,
    TRACEMSHNet,
    render_trace_atoms,
)
from utils.trace_data import TraceDataBundle, build_trace_data  # noqa: E402
from utils.trace_gates import (  # noqa: E402
    DPGate,
    GeometryGate,
    IntegrationGate,
    require_dp_gate,
    require_geometry_gate,
    require_integration_gate,
)
from utils.trace_geometry import encode_trace_targets  # noqa: E402
from utils.trace_provenance import (  # noqa: E402
    canonical_json_sha256,
    git_state,
    provenance_path,
    runtime_environment,
    sha256_file,
)
from utils.trace_training import (  # noqa: E402
    TRACE_METHODS,
    TraceTrainingError,
    atomic_torch_save,
    build_training_checkpoint,
    load_training_checkpoint,
    make_paired_loaders,
    restore_training_checkpoint,
    seed_everything,
    source_inventory,
)


TRAINING_SOURCE_LOCATORS = (
    "model/mshnet_d0_backbone.py",
    "model/trace_front.py",
    "model/trace_mshnet.py",
    "model/trace_run_semiring.py",
    "tools/train_trace.py",
    "utils/trace_codec.py",
    "utils/trace_data.py",
    "utils/trace_gates.py",
    "utils/trace_geometry.py",
    "utils/trace_provenance.py",
    "utils/trace_training.py",
)
METRIC_SCHEMA_VERSION = "trace_dev_training_metrics_v1"
DEV_BUNDLE_SCHEMA_VERSION = "trace_dev_score_bundle_v1"


def _finite_probability(value: float | None, *, label: str) -> float:
    if value is None:
        raise TraceTrainingError(f"{label} is required for this method")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise TraceTrainingError(f"{label} must lie strictly in (0, 1)")
    return result


def _resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise TraceTrainingError(f"invalid device: {value!r}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise TraceTrainingError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise TraceTrainingError("requested CUDA device is not visible")
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    if device.type != "cpu":
        raise TraceTrainingError("TRACE training currently supports only CPU or CUDA")
    return device


def _gate_first_contract(args: argparse.Namespace) -> tuple[
    GeometryGate, DPGate, IntegrationGate
]:
    """Validate every release gate before constructing datasets or models."""

    dataset_name = Path(args.dataset_dir).name
    geometry_gate = require_geometry_gate(
        args.t0_a_report,
        expected_dataset=dataset_name,
    )
    dp_gate = require_dp_gate(args.t0_b_dp_report)
    baseline_sha256 = sha256_file(args.baseline_checkpoint)
    integration_gate = require_integration_gate(
        args.t0_b_integration_report,
        expected_dp_report_sha256=dp_gate.report_sha256,
        expected_geometry_sha256=geometry_gate.geometry_sha256,
        expected_baseline_checkpoint_sha256=baseline_sha256,
    )
    return geometry_gate, dp_gate, integration_gate


def _build_locked_data(
    args: argparse.Namespace, geometry_gate: GeometryGate
) -> TraceDataBundle:
    bundle = build_trace_data(
        args.dataset_dir,
        train_manifest=args.train_manifest or None,
        test_manifest=args.test_manifest or None,
        image_size=(
            geometry_gate.geometry.image_height,
            geometry_gate.geometry.image_width,
        ),
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        include_test=False,
        train_horizontal_flip=False,
    )
    if bundle.test is not None or bundle.provenance().get("test_assets_included") is not False:
        raise TraceTrainingError("training must not construct or audit test assets")
    current_mask_manifest_sha256 = canonical_json_sha256(
        {
            name: sha256_file(bundle.train._root / "masks" / f"{name}.png")
            for name in bundle.train.names
        }
    )
    if current_mask_manifest_sha256 != geometry_gate.mask_manifest_sha256:
        raise TraceTrainingError(
            "T0-A train-mask bytes differ from the paired training dataset"
        )
    # Re-run T0-A with the byte-exact canonical manifest identity now known.
    checked = require_geometry_gate(
        args.t0_a_report,
        expected_dataset=bundle.train._root.name,
        expected_train_split_sha256=bundle.train.split_sha256,
    )
    if checked.to_dict() != geometry_gate.to_dict():
        raise TraceTrainingError("T0-A changed between preflight and paired-data audit")
    return bundle


def _build_model(
    args: argparse.Namespace,
    bundle: TraceDataBundle,
    geometry_gate: GeometryGate,
    device: torch.device,
) -> TRACEMSHNet | MatchedDenseMSHNet:
    shared: dict[str, Any] = {
        "baseline_checkpoint": args.baseline_checkpoint,
        "geometry": geometry_gate.geometry,
        "expected_dataset": bundle.train._root.name,
        "expected_seed": args.seed,
        "expected_train_split_sha256": bundle.train.split_sha256,
        "expected_val_split_sha256": bundle.dev.split_sha256,
    }
    if args.method == "trace":
        model: TRACEMSHNet | MatchedDenseMSHNet = TRACEMSHNet(
            positive_cell_prior=_finite_probability(
                args.positive_cell_prior, label="positive_cell_prior"
            ),
            field_chunk_size=args.field_chunk_size,
            **shared,
        )
        if model.trainable_parameter_count != 306:
            raise TraceTrainingError("TRACE head parameter count changed")
    elif args.method == "dense_bernoulli":
        model = MatchedDenseMSHNet(
            foreground_pixel_prior=_finite_probability(
                args.foreground_pixel_prior, label="foreground_pixel_prior"
            ),
            **shared,
        )
        if model.trainable_parameter_count != 307:
            raise TraceTrainingError("dense-control head parameter count changed")
    else:  # pragma: no cover - argparse and caller validation
        raise TraceTrainingError("unsupported training method")
    model.to(device)
    model.assert_front_integrity()
    return model


def _run_config(
    args: argparse.Namespace,
    bundle: TraceDataBundle,
    geometry_gate: GeometryGate,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": "trace_paired_run_config_v1",
        "method": args.method,
        "dataset": bundle.train._root.name,
        "baseline_checkpoint": provenance_path(args.baseline_checkpoint),
        "baseline_checkpoint_sha256": sha256_file(args.baseline_checkpoint),
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "drop_last": False,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "positive_cell_prior": (
            args.positive_cell_prior if args.method == "trace" else None
        ),
        "foreground_pixel_prior": (
            args.foreground_pixel_prior
            if args.method == "dense_bernoulli"
            else None
        ),
        "field_chunk_size": (
            args.field_chunk_size if args.method == "trace" else None
        ),
        "geometry_sha256": geometry_gate.geometry_sha256,
        "split_assignment_sha256": bundle.provenance()[
            "split_assignment_sha256"
        ],
        "train_split_sha256": bundle.train.split_sha256,
        "dev_split_sha256": bundle.dev.split_sha256,
        "canonical_train_manifest_sha256": (
            bundle.canonical_train_manifest.normalized_sha256
        ),
        "device": str(device),
        "deterministic_algorithms": True,
        "amp": False,
        "selection_split": "dev",
        "selection_criterion": "mean_proper_nll",
        "selection_tie_break": "earliest_epoch",
        "test_assets_included": False,
    }


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _batch_loss(
    model: TRACEMSHNet | MatchedDenseMSHNet,
    method: str,
    images: torch.Tensor,
    masks: torch.Tensor,
    *,
    device: torch.device,
    training: bool,
) -> torch.Tensor:
    images = images.to(device=device, dtype=torch.float32, non_blocking=False)
    if method == "trace":
        if not isinstance(model, TRACEMSHNet):  # pragma: no cover - contract assertion
            raise TraceTrainingError("TRACE method/model mismatch")
        targets = [
            encode_trace_targets(masks[index, 0].numpy(), model.geometry)
            for index in range(masks.shape[0])
        ]
        output = model(
            images,
            return_map=False,
            return_marginals=False,
            create_graph=False,
        )
        loss = model.exact_nll(output, targets, reduction="mean").loss
    elif method == "dense_bernoulli":
        if not isinstance(model, MatchedDenseMSHNet):  # pragma: no cover
            raise TraceTrainingError("dense method/model mismatch")
        logits = model(images)
        loss = model.exact_bernoulli_nll(
            logits,
            masks.to(device=device, dtype=torch.float32, non_blocking=False),
        )
    else:  # pragma: no cover
        raise TraceTrainingError("unsupported method")
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        phase = "training" if training else "development"
        raise TraceTrainingError(f"non-finite scalar {phase} loss")
    return loss


def _train_epoch(
    model: TRACEMSHNet | MatchedDenseMSHNet,
    method: str,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train(True)
    total = 0.0
    count = 0
    for images, masks, _names in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(
            model, method, images, masks, device=device, training=True
        )
        loss.backward()
        parameters = tuple(model.potential_map.parameters())
        if any(
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise TraceTrainingError("trainable head has missing or non-finite gradients")
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        optimizer.step()
        batch = int(images.shape[0])
        total += float(loss.detach().cpu()) * batch
        count += batch
    if count != len(loader.dataset):
        raise TraceTrainingError("training loader omitted or duplicated samples")
    model.assert_front_integrity()
    return total / count


@torch.no_grad()
def _dev_epoch(
    model: TRACEMSHNet | MatchedDenseMSHNet,
    method: str,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> float:
    model.train(False)
    total = 0.0
    count = 0
    for images, masks, _names in loader:
        loss = _batch_loss(
            model, method, images, masks, device=device, training=False
        )
        batch = int(images.shape[0])
        total += float(loss.detach().cpu()) * batch
        count += batch
    if count != len(loader.dataset):
        raise TraceTrainingError("development loader omitted or duplicated samples")
    model.assert_front_integrity()
    return total / count


def _append_metric(path: Path, row: Mapping[str, Any]) -> None:
    rendered = json.dumps(
        dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")


def _validate_resume_metric_ledger(path: Path, *, checkpoint_epoch: int) -> None:
    if not path.is_file():
        raise TraceTrainingError("resume requires the existing development metric ledger")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TraceTrainingError("resume development metric ledger is invalid") from exc
    if len(rows) != checkpoint_epoch + 1:
        raise TraceTrainingError("resume metric ledger/checkpoint epoch mismatch")
    for expected_epoch, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != METRIC_SCHEMA_VERSION
            or row.get("epoch") != expected_epoch
            or row.get("selection_split") != "dev"
            or row.get("test_access") is not False
        ):
            raise TraceTrainingError("resume metric ledger violates the dev-only protocol")


def _checkpoint_common(
    *,
    args: argparse.Namespace,
    model: TRACEMSHNet | MatchedDenseMSHNet,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    dev_generator: torch.Generator,
    run_config: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    gates: Mapping[str, Any],
    geometry_gate: GeometryGate,
    sources: Mapping[str, str],
    git: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "method": args.method,
        "model": model,
        "head": model.potential_map,
        "optimizer": optimizer,
        "train_generator": train_generator,
        "dev_generator": dev_generator,
        "run_config": run_config,
        "data_provenance": data_provenance,
        "gates": gates,
        "geometry": geometry_gate.geometry.to_dict(),
        "geometry_sha256": geometry_gate.geometry_sha256,
        "logk_cache_sha256": (
            model.field.logk_cache_sha256
            if isinstance(model, TRACEMSHNet)
            else None
        ),
        "front_provenance": model.front.provenance.to_dict(),
        "front_state_sha256": model.assert_front_integrity(),
        "sources": sources,
        "git": git,
        "runtime": runtime,
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.method not in TRACE_METHODS:
        raise TraceTrainingError("unsupported method")
    if isinstance(args.seed, bool) or args.seed < 0:
        raise TraceTrainingError("seed must be non-negative")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise TraceTrainingError("epochs/batch_size/num_workers are invalid")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise TraceTrainingError("learning_rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise TraceTrainingError("weight_decay must be finite and non-negative")
    if not math.isfinite(args.gradient_clip_norm) or args.gradient_clip_norm < 0.0:
        raise TraceTrainingError("gradient_clip_norm must be finite and non-negative")

    # Release gates deliberately precede seed/device/data/model initialization.
    geometry_gate, dp_gate, integration_gate = _gate_first_contract(args)
    seed_everything(args.seed)
    device = _resolve_device(args.device)
    bundle = _build_locked_data(args, geometry_gate)
    model = _build_model(args, bundle, geometry_gate, device)
    train_loader, dev_loader, train_generator, dev_generator = make_paired_loaders(
        bundle.train,
        bundle.dev,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    optimizer = torch.optim.AdamW(
        model.potential_map.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    data_provenance = bundle.provenance()
    run_config = _run_config(args, bundle, geometry_gate, device)
    gates = {
        "t0_a": geometry_gate.to_dict(),
        "t0_b_dp": dp_gate.to_dict(),
        "t0_b_integration": integration_gate.to_dict(),
    }
    sources = source_inventory(PROJECT_ROOT, TRAINING_SOURCE_LOCATORS)
    return {
        "geometry_gate": geometry_gate,
        "dp_gate": dp_gate,
        "integration_gate": integration_gate,
        "device": device,
        "bundle": bundle,
        "model": model,
        "train_loader": train_loader,
        "dev_loader": dev_loader,
        "train_generator": train_generator,
        "dev_generator": dev_generator,
        "optimizer": optimizer,
        "data_provenance": data_provenance,
        "run_config": run_config,
        "gates": gates,
        "sources": sources,
        "git": git_state(),
        "runtime": runtime_environment(),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    state = _prepare(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    latest_path = output_dir / "checkpoint_latest.pt"
    best_path = output_dir / "checkpoint_best_dev.pt"
    metrics_path = output_dir / "dev_metrics.jsonl"
    if args.resume:
        if not output_dir.is_dir():
            raise TraceTrainingError("resume output directory does not exist")
        if Path(args.resume).expanduser().resolve() != latest_path:
            raise TraceTrainingError(
                "exact resume requires this run's checkpoint_latest.pt"
            )
    else:
        if any(path.exists() for path in (latest_path, best_path, metrics_path)):
            raise TraceTrainingError("output already contains TRACE training artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)

    model = state["model"]
    optimizer = state["optimizer"]
    common = _checkpoint_common(args=args, **{
        key: state[key]
        for key in (
            "model",
            "optimizer",
            "train_generator",
            "dev_generator",
            "run_config",
            "data_provenance",
            "gates",
            "geometry_gate",
            "sources",
            "git",
            "runtime",
        )
    })
    start_epoch = 0
    best_dev_loss = math.inf
    best_epoch = -1
    if args.resume:
        payload = load_training_checkpoint(args.resume)
        raw_epoch = payload.get("epoch")
        if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int) or raw_epoch < 0:
            raise TraceTrainingError("resume checkpoint epoch is malformed")
        _validate_resume_metric_ledger(metrics_path, checkpoint_epoch=raw_epoch)
        start_epoch, best_dev_loss, best_epoch = restore_training_checkpoint(
            payload,
            expected_method=args.method,
            expected_run_config_sha256=canonical_json_sha256(state["run_config"]),
            expected_data_provenance_sha256=canonical_json_sha256(
                state["data_provenance"]
            ),
            expected_gates=state["gates"],
            expected_geometry_sha256=state["geometry_gate"].geometry_sha256,
            expected_logk_cache_sha256=common["logk_cache_sha256"],
            expected_front_state_sha256=common["front_state_sha256"],
            expected_sources=state["sources"],
            model=model,
            head=model.potential_map,
            optimizer=optimizer,
            train_generator=state["train_generator"],
            dev_generator=state["dev_generator"],
        )
        _move_optimizer_state(optimizer, state["device"])
        model.assert_front_integrity()
        if start_epoch >= args.epochs:
            raise TraceTrainingError("resume checkpoint already reached configured epochs")

    for epoch in range(start_epoch, args.epochs):
        state["bundle"].set_epoch(epoch)
        train_loss = _train_epoch(
            model,
            args.method,
            state["train_loader"],
            optimizer,
            device=state["device"],
            gradient_clip_norm=args.gradient_clip_norm,
        )
        dev_loss = _dev_epoch(
            model,
            args.method,
            state["dev_loader"],
            device=state["device"],
        )
        if not math.isfinite(train_loss) or not math.isfinite(dev_loss):
            raise TraceTrainingError("epoch produced a non-finite aggregate loss")
        improved = dev_loss < best_dev_loss
        if improved:
            best_dev_loss = dev_loss
            best_epoch = epoch
        _append_metric(
            metrics_path,
            {
                "schema_version": METRIC_SCHEMA_VERSION,
                "epoch": epoch,
                "train_mean_proper_nll": train_loss,
                "dev_mean_proper_nll": dev_loss,
                "selected_as_best": improved,
                "selection_split": "dev",
                "test_access": False,
            },
        )
        if improved:
            best_payload = build_training_checkpoint(
                epoch=epoch,
                best_dev_loss=best_dev_loss,
                best_epoch=best_epoch,
                checkpoint_role="best_dev",
                **common,
            )
            atomic_torch_save(best_payload, best_path)
        latest_payload = build_training_checkpoint(
            epoch=epoch,
            best_dev_loss=best_dev_loss,
            best_epoch=best_epoch,
            checkpoint_role="latest",
            **common,
        )
        atomic_torch_save(latest_payload, latest_path)

    result = {
        "schema_version": "trace_training_result_v1",
        "status": "COMPLETE",
        "method": args.method,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "selection_split": "dev",
        "test_access": False,
        "latest_checkpoint": provenance_path(latest_path),
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "best_checkpoint": provenance_path(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "run_config_sha256": canonical_json_sha256(state["run_config"]),
        "data_provenance_sha256": canonical_json_sha256(
            state["data_provenance"]
        ),
    }
    (output_dir / "training_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def export_dev_bundle(args: argparse.Namespace) -> dict[str, Any]:
    if not args.checkpoint:
        raise TraceTrainingError("dev-infer requires --checkpoint")
    if not args.dev_bundle:
        raise TraceTrainingError("dev-infer requires --dev-bundle")
    destination = Path(args.dev_bundle).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise TraceTrainingError("development bundle path must end in .npz")
    if destination.exists() and not args.overwrite_dev_bundle:
        raise TraceTrainingError("development bundle already exists")

    state = _prepare(args)
    payload = load_training_checkpoint(args.checkpoint)
    if payload.get("checkpoint_role") != "best_dev" or payload.get("epoch") != payload.get(
        "best_epoch"
    ):
        raise TraceTrainingError("development inference requires a dev-selected best checkpoint")
    common = _checkpoint_common(args=args, **{
        key: state[key]
        for key in (
            "model",
            "optimizer",
            "train_generator",
            "dev_generator",
            "run_config",
            "data_provenance",
            "gates",
            "geometry_gate",
            "sources",
            "git",
            "runtime",
        )
    })
    restore_training_checkpoint(
        payload,
        expected_method=args.method,
        expected_run_config_sha256=canonical_json_sha256(state["run_config"]),
        expected_data_provenance_sha256=canonical_json_sha256(
            state["data_provenance"]
        ),
        expected_gates=state["gates"],
        expected_geometry_sha256=state["geometry_gate"].geometry_sha256,
        expected_logk_cache_sha256=common["logk_cache_sha256"],
        expected_front_state_sha256=common["front_state_sha256"],
        expected_sources=state["sources"],
        model=state["model"],
        head=state["model"].potential_map,
        optimizer=state["optimizer"],
        train_generator=state["train_generator"],
        dev_generator=state["dev_generator"],
    )
    state["model"].train(False)
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for images, masks, names in state["dev_loader"]:
            images = images.to(state["device"], dtype=torch.float32)
            if args.method == "trace":
                output = state["model"](
                    images, return_map=True, return_marginals=False
                )
                batch_scores = render_trace_atoms(
                    output, state["geometry_gate"].geometry
                ).scores
            else:
                batch_scores = state["model"](images)[:, 0]
            if not bool(torch.isfinite(batch_scores).all()):
                raise TraceTrainingError("development inference emitted non-finite scores")
            scores.extend(np.asarray(batch_scores.detach().cpu(), dtype=np.float64))
            targets.extend(np.asarray(masks[:, 0], dtype=np.uint8))
            sample_ids.extend(str(name) for name in names)
    if tuple(sample_ids) != state["bundle"].dev.names:
        raise TraceTrainingError("development inference order differs from locked manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        scores=np.stack(scores, axis=0),
        targets=np.stack(targets, axis=0),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
    )
    metadata = {
        "schema_version": DEV_BUNDLE_SCHEMA_VERSION,
        "method": args.method,
        "split": "dev",
        "selection_checkpoint_role": "best_dev",
        "checkpoint": provenance_path(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "bundle": provenance_path(destination),
        "bundle_sha256": sha256_file(destination),
        "sample_count": len(sample_ids),
        "sample_ids_sha256": canonical_json_sha256(sample_ids),
        "dev_split_sha256": state["bundle"].dev.split_sha256,
        "geometry_sha256": state["geometry_gate"].geometry_sha256,
        "data_provenance_sha256": canonical_json_sha256(
            state["data_provenance"]
        ),
        "score_semantics": (
            "log P(Y_cell = emitted_MAP_atom | image), max over whole atoms"
            if args.method == "trace"
            else "independent Bernoulli foreground logit"
        ),
        "test_access": False,
    }
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "dev-infer"), default="train")
    parser.add_argument("--method", choices=TRACE_METHODS, required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--test-manifest", default="")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--t0-a-report", required=True)
    parser.add_argument("--t0-b-dp-report", required=True)
    parser.add_argument("--t0-b-integration-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--positive-cell-prior", type=float)
    parser.add_argument("--foreground-pixel-prior", type=float)
    parser.add_argument("--field-chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dev-bundle", default="")
    parser.add_argument("--overwrite-dev-bundle", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "train":
            if args.checkpoint or args.dev_bundle:
                raise TraceTrainingError(
                    "train mode does not accept --checkpoint or --dev-bundle"
                )
            result = train(args)
        else:
            if args.resume:
                raise TraceTrainingError("dev-infer does not accept --resume")
            result = export_dev_bundle(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEV_BUNDLE_SCHEMA_VERSION",
    "METRIC_SCHEMA_VERSION",
    "TRAINING_SOURCE_LOCATORS",
    "build_parser",
    "export_dev_bundle",
    "main",
    "train",
]
