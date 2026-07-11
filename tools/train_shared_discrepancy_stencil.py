#!/usr/bin/env python3
"""Kernel-only training probe for the shared discrepancy stencil.

The complete MSHNet checkpoint is frozen.  Only the eight shared stencil
coefficients are optimized, and the best coefficients remain in memory; this
script never writes a checkpoint or result file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from model.dea_shared_discrepancy_stencil import (
    SharedDiscrepancyStencilMSHNet,
)
from model.loss import SLSIoULoss
from model.mshnet_checkpoint import strip_legacy_dea_lite_head
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="datasets/NUAA-SIRST")
    parser.add_argument("--train-split-file", default="")
    parser.add_argument("--val-split-file", default="")
    parser.add_argument("--test-split-file", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260710)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-l1", type=float, default=0.25)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument(
        "--max-pd-drop",
        type=float,
        default=0.0,
        help="Stop when validation PD falls this far below the frozen baseline.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(specification)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: str) -> tuple[dict[str, Tensor], dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    metadata: dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state_dict = checkpoint["net"]
        for key in ("epoch", "iou", "pd", "fa", "best_iou", "method_meta"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint does not contain a state dict")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return state_dict, metadata


def theta_snapshot(model: SharedDiscrepancyStencilMSHNet) -> dict[str, Any]:
    raw = model.stencil.theta.detach().float().cpu()
    effective = model.stencil.effective_weights().detach().float().cpu()
    return {
        "raw": [float(value) for value in raw],
        "effective": [float(value) for value in effective],
        "raw_l1": float(raw.abs().sum()),
        "effective_l1": float(effective.abs().sum()),
    }


def warm_segmentation_loss(
    model_output: dict[str, Any],
    labels: Tensor,
    criterion: SLSIoULoss,
    down: nn.Module,
    *,
    warm_epoch: int,
    epoch: int,
) -> Tensor:
    """Match main.py: final plus four native side SLSIoU losses, averaged."""

    loss = criterion(
        model_output["pred"], labels, warm_epoch, epoch
    )
    labels_for_scale = labels
    masks = model_output["masks"]
    if len(masks) != 4:
        raise RuntimeError("warm path must return four side masks")
    for index, mask in enumerate(masks):
        if index > 0:
            labels_for_scale = down(labels_for_scale)
        loss = loss + criterion(
            mask,
            labels_for_scale,
            warm_epoch,
            epoch,
        )
    return loss / 5.0


def evaluate(
    model: SharedDiscrepancyStencilMSHNet,
    loader: DataLoader,
    device: torch.device,
    *,
    crop_size: int,
) -> dict[str, float]:
    model.eval()
    intersection = 0
    union = 0
    metric = PD_FA(nclass=1, bins=10, size=crop_size)
    finite = True
    with torch.no_grad():
        for images, labels in loader:
            logits = model(
                images.to(device, non_blocking=True),
                True,
                return_dict=True,
            )["pred"]
            finite = finite and bool(torch.isfinite(logits).all())
            target = labels.to(device, non_blocking=True) > 0.5
            prediction = logits > 0.0
            intersection += int((prediction & target).sum())
            union += int((prediction | target).sum())
            metric.update(logits.cpu(), labels)
    false_alarm, detection_probability = metric.get()
    return {
        "iou": float(intersection / max(1, union)),
        "pd": float(detection_probability[0]),
        "fa_per_million": float(false_alarm[0] * 1e6),
        "finite": bool(finite),
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative")
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("invalid optimizer configuration")
    if args.max_l1 <= 0.0:
        raise ValueError("--max-l1 must be positive")
    if not 0.0 <= args.max_pd_drop <= 1.0:
        raise ValueError("--max-pd-drop must be in [0, 1]")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_dataset = IRSTD_Dataset(args, mode="train")
    val_dataset = IRSTD_Dataset(args, mode="val")
    if set(train_dataset.names).intersection(val_dataset.names):
        raise RuntimeError("train/validation split overlap")

    state_dict, checkpoint_metadata = load_checkpoint(args.checkpoint)
    method_meta = checkpoint_metadata.get("method_meta", {})
    if isinstance(method_meta, dict):
        expected_train_hash = method_meta.get("train_split_sha256")
        expected_val_hash = method_meta.get("val_split_sha256")
        if expected_train_hash and expected_train_hash != train_dataset.split_sha256:
            raise RuntimeError("training split hash differs from checkpoint")
        if expected_val_hash and expected_val_hash != val_dataset.split_sha256:
            raise RuntimeError("validation split hash differs from checkpoint")

    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    common_loader = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        **common_loader,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        generator=val_generator,
        **common_loader,
    )

    state_dict = strip_legacy_dea_lite_head(state_dict)
    model = SharedDiscrepancyStencilMSHNet(
        args.input_channels,
        max_l1=args.max_l1,
        freeze_bn_statistics=True,
    ).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"stencil.theta"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "incompatible checkpoint: missing=%s unexpected=%s"
            % (incompatible.missing_keys, incompatible.unexpected_keys)
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.stencil.theta.requires_grad_(True)
    trainable = [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if trainable != [("stencil.theta", 8)]:
        raise RuntimeError("kernel-only contract violated: %s" % (trainable,))

    optimizer = torch.optim.Adam(
        [model.stencil.theta],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = SLSIoULoss().to(device)
    down = nn.MaxPool2d(2, 2)
    checkpoint_epoch = int(checkpoint_metadata.get("epoch", -1))

    initial_metrics = evaluate(
        model, val_loader, device, crop_size=args.crop_size
    )
    initial_record = {
        "epoch": checkpoint_epoch,
        **initial_metrics,
        "theta": theta_snapshot(model),
    }
    best_metrics = dict(initial_metrics)
    best_theta = model.stencil.theta.detach().cpu().clone()
    best_epoch = checkpoint_epoch
    records: list[dict[str, Any]] = []
    stopped_early = False
    stop_reason = None
    pd_floor = initial_metrics["pd"] - args.max_pd_drop
    print(
        "initial epoch=%d iou=%.6f pd=%.6f fa=%.4f"
        % (
            checkpoint_epoch,
            initial_metrics["iou"],
            initial_metrics["pd"],
            initial_metrics["fa_per_million"],
        ),
        file=sys.stderr,
        flush=True,
    )

    for local_epoch in range(args.epochs):
        absolute_epoch = checkpoint_epoch + 1 + local_epoch
        model.train()
        running_loss = 0.0
        sample_count = 0
        train_finite = True
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images, True, return_dict=True)
            loss = warm_segmentation_loss(
                output,
                labels,
                criterion,
                down,
                warm_epoch=args.warm_epoch,
                epoch=absolute_epoch,
            )
            if not torch.isfinite(loss):
                train_finite = False
                stop_reason = "non-finite training loss at epoch %d" % absolute_epoch
                break
            loss.backward()
            gradient = model.stencil.theta.grad
            if gradient is None or not torch.isfinite(gradient).all():
                train_finite = False
                stop_reason = "non-finite stencil gradient at epoch %d" % absolute_epoch
                break
            optimizer.step()
            if not torch.isfinite(model.stencil.theta).all():
                train_finite = False
                stop_reason = "non-finite stencil weights at epoch %d" % absolute_epoch
                break
            batch_size = int(images.shape[0])
            running_loss += float(loss.detach()) * batch_size
            sample_count += batch_size

        if not train_finite:
            stopped_early = True
            break

        metrics = evaluate(model, val_loader, device, crop_size=args.crop_size)
        theta = theta_snapshot(model)
        record = {
            "epoch": absolute_epoch,
            "train_loss": float(running_loss / max(1, sample_count)),
            **metrics,
            "theta": theta,
        }
        records.append(record)
        print(
            "epoch=%d loss=%.6f iou=%.6f pd=%.6f fa=%.4f l1=%.6f"
            % (
                absolute_epoch,
                record["train_loss"],
                metrics["iou"],
                metrics["pd"],
                metrics["fa_per_million"],
                theta["effective_l1"],
            ),
            file=sys.stderr,
            flush=True,
        )

        if not metrics["finite"]:
            stopped_early = True
            stop_reason = "non-finite validation logits at epoch %d" % absolute_epoch
            break
        if metrics["pd"] < pd_floor:
            stopped_early = True
            stop_reason = (
                "validation PD %.6f below safety floor %.6f at epoch %d"
                % (metrics["pd"], pd_floor, absolute_epoch)
            )
            break
        if metrics["iou"] > best_metrics["iou"]:
            best_metrics = dict(metrics)
            best_theta = model.stencil.theta.detach().cpu().clone()
            best_epoch = absolute_epoch

    with torch.no_grad():
        model.stencil.theta.copy_(best_theta.to(device))
    restored_metrics = evaluate(model, val_loader, device, crop_size=args.crop_size)
    restored_theta = theta_snapshot(model)
    if not math.isclose(
        restored_metrics["iou"], best_metrics["iou"], rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("restored in-memory best does not reproduce best IoU")

    report = {
        "scope": "kernel-only training on fixed NUAA train/validation split",
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(device),
        "seed": args.seed,
        "deterministic": True,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "train_split_sha256": train_dataset.split_sha256,
        "val_split_sha256": val_dataset.split_sha256,
        "optimizer": {
            "name": "Adam",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "epochs_requested": args.epochs,
        "epochs_completed": len(records),
        "batch_size": args.batch_size,
        "loss": "mean(final + four side SLSIoU), checkpoint epochs continued",
        "trainable_parameters": trainable,
        "max_l1": args.max_l1,
        "pd_safety_floor": pd_floor,
        "initial": initial_record,
        "records": records,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "best_in_memory": {
            "epoch": best_epoch,
            **best_metrics,
            "theta": restored_theta,
        },
        "restored_best_evaluation": restored_metrics,
        "checkpoint_written": False,
    }
    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
