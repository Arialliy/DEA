#!/usr/bin/env python3
"""Frozen mechanics audit for persistent conditional-increment MSHNet.

This script performs no optimization.  It evaluates fixed external homotopy
values and reports segmentation metrics separately from decoder-state
mechanics.  Ground truth is never used by the model transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from model.dea_persistent_conditional_increment import (
    PersistentConditionalIncrementMSHNet,
)
from model.mshnet_checkpoint import strip_legacy_dea_lite_head
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="datasets/NUAA-SIRST")
    parser.add_argument("--mode", choices=("val", "test"), default="val")
    parser.add_argument("--train-split-file", default="")
    parser.add_argument("--val-split-file", default="")
    parser.add_argument("--test-split-file", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260710)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--anchor-mode", choices=("zero", "mean"), default="mean")
    parser.add_argument("--center-xi", action="store_true")
    parser.add_argument("--drift-eps", type=float, default=1e-6)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(specification)


def parse_alphas(specification: str) -> tuple[float, ...]:
    values = tuple(
        float(item.strip())
        for item in specification.split(",")
        if item.strip()
    )
    if not values or 0.0 not in values:
        raise ValueError("--alphas must contain 0 and at least one value")
    if len(values) != len(set(values)):
        raise ValueError("--alphas must not contain duplicates")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("--alphas values must be finite and in [0, 1]")
    return values


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
                value = checkpoint[key]
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metadata[key] = value
                elif key == "method_meta" and isinstance(value, dict):
                    metadata[key] = value
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint does not contain a state dict")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return state_dict, metadata


def new_stage_accumulator() -> dict[str, float | int]:
    return {
        "state_sum_squares": 0.0,
        "state_count": 0,
        "xi_sum_squares": 0.0,
        "xi_count": 0,
        "persistent_negative": 0,
        "persistent_count": 0,
        "inherited_mean_sum": 0.0,
        "persistent_mean_sum": 0.0,
        "inherited_std_sum": 0.0,
        "persistent_std_sum": 0.0,
        "mean_relative_shift_sum": 0.0,
        "mean_relative_shift_abs_sum": 0.0,
        "std_relative_shift_sum": 0.0,
        "std_relative_shift_abs_sum": 0.0,
        "drift_count": 0,
    }


def update_stage_accumulator(
    accumulator: dict[str, float | int],
    terms: dict[str, Any],
    *,
    eps: float,
) -> None:
    state = terms["state"]
    accumulator["state_sum_squares"] += float(state.double().square().sum())
    accumulator["state_count"] += int(state.numel())

    increment = terms.get("increment")
    if increment is not None:
        accumulator["xi_sum_squares"] += float(
            increment.double().square().sum()
        )
        accumulator["xi_count"] += int(increment.numel())

    persistent = terms.get("persistent_input")
    if persistent is None:
        return
    inherited = terms["inherited"]
    accumulator["persistent_negative"] += int((persistent < 0.0).sum())
    accumulator["persistent_count"] += int(persistent.numel())

    # Use one distribution per image over the complete inherited state.  A
    # per-channel denominator is unstable for channels that are nearly
    # spatially constant and can turn a small absolute shift into an
    # uninformative 1e2--1e5 relative value.
    inherited_flat = inherited.double().flatten(1)
    persistent_flat = persistent.double().flatten(1)
    inherited_mean = inherited_flat.mean(dim=1)
    persistent_mean = persistent_flat.mean(dim=1)
    inherited_std = inherited_flat.std(dim=1, unbiased=False)
    persistent_std = persistent_flat.std(dim=1, unbiased=False)
    relative_mean_shift = (
        persistent_mean - inherited_mean
    ) / (inherited_mean.abs() + eps)
    relative_std_shift = (
        persistent_std - inherited_std
    ) / (inherited_std + eps)
    drift_count = int(inherited_mean.numel())

    accumulator["inherited_mean_sum"] += float(inherited_mean.sum())
    accumulator["persistent_mean_sum"] += float(persistent_mean.sum())
    accumulator["inherited_std_sum"] += float(inherited_std.sum())
    accumulator["persistent_std_sum"] += float(persistent_std.sum())
    accumulator["mean_relative_shift_sum"] += float(
        relative_mean_shift.sum()
    )
    accumulator["mean_relative_shift_abs_sum"] += float(
        relative_mean_shift.abs().sum()
    )
    accumulator["std_relative_shift_sum"] += float(relative_std_shift.sum())
    accumulator["std_relative_shift_abs_sum"] += float(
        relative_std_shift.abs().sum()
    )
    accumulator["drift_count"] += drift_count


def finalize_stage(
    accumulator: dict[str, float | int]
) -> dict[str, float | int | None]:
    state_count = int(accumulator["state_count"])
    xi_count = int(accumulator["xi_count"])
    persistent_count = int(accumulator["persistent_count"])
    drift_count = int(accumulator["drift_count"])
    state_rms = float(
        np.sqrt(float(accumulator["state_sum_squares"]) / max(1, state_count))
    )
    xi_rms = (
        float(
            np.sqrt(float(accumulator["xi_sum_squares"]) / xi_count)
        )
        if xi_count
        else None
    )
    return {
        "state_rms": state_rms,
        "xi_rms": xi_rms,
        "xi_to_state_rms": (
            float(xi_rms / max(state_rms, 1e-12)) if xi_rms is not None else None
        ),
        "persistent_input_negative_fraction": (
            float(accumulator["persistent_negative"] / persistent_count)
            if persistent_count
            else None
        ),
        "factual_inherited_spatial_mean": (
            float(accumulator["inherited_mean_sum"] / drift_count)
            if drift_count
            else None
        ),
        "persistent_input_spatial_mean": (
            float(accumulator["persistent_mean_sum"] / drift_count)
            if drift_count
            else None
        ),
        "factual_inherited_spatial_std": (
            float(accumulator["inherited_std_sum"] / drift_count)
            if drift_count
            else None
        ),
        "persistent_input_spatial_std": (
            float(accumulator["persistent_std_sum"] / drift_count)
            if drift_count
            else None
        ),
        "relative_mean_shift_signed": (
            float(accumulator["mean_relative_shift_sum"] / drift_count)
            if drift_count
            else None
        ),
        "relative_mean_shift_abs": (
            float(accumulator["mean_relative_shift_abs_sum"] / drift_count)
            if drift_count
            else None
        ),
        "relative_std_shift_signed": (
            float(accumulator["std_relative_shift_sum"] / drift_count)
            if drift_count
            else None
        ),
        "relative_std_shift_abs": (
            float(accumulator["std_relative_shift_abs_sum"] / drift_count)
            if drift_count
            else None
        ),
        "state_elements": state_count,
        "xi_elements": xi_count,
    }


def main() -> None:
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    if args.drift_eps <= 0.0:
        raise ValueError("--drift-eps must be positive")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    dataset = IRSTD_Dataset(args, args.mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    state_dict, checkpoint_metadata = load_checkpoint(args.checkpoint)
    state_dict = strip_legacy_dea_lite_head(state_dict)
    model = PersistentConditionalIncrementMSHNet(
        args.input_channels,
        alpha=1.0,
        anchor_mode=args.anchor_mode,
        center_xi=args.center_xi,
        freeze_bn_statistics=True,
    ).to(device).eval()
    incompatible = model.load_state_dict(state_dict, strict=False)
    disallowed_missing = set(incompatible.missing_keys)
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "incompatible checkpoint: missing=%s unexpected=%s"
            % (sorted(disallowed_missing), sorted(incompatible.unexpected_keys))
        )

    method_meta = checkpoint_metadata.get("method_meta", {})
    expected_split_hash = (
        method_meta.get("val_split_sha256")
        if args.mode == "val" and isinstance(method_meta, dict)
        else None
    )
    if expected_split_hash and expected_split_hash != dataset.split_sha256:
        raise RuntimeError(
            "split hash differs from checkpoint metadata: %s != %s"
            % (dataset.split_sha256, expected_split_hash)
        )

    intersections = {alpha: 0 for alpha in alphas}
    unions = {alpha: 0 for alpha in alphas}
    pd_fa = {
        alpha: PD_FA(nclass=1, bins=10, size=args.crop_size)
        for alpha in alphas
    }
    threshold_changed = {alpha: 0 for alpha in alphas}
    absolute_logit_change = {alpha: 0.0 for alpha in alphas}
    mechanism = {
        alpha: {stage: new_stage_accumulator() for stage in (0, 1, 2, 3)}
        for alpha in alphas
    }
    total_pixels = 0

    with torch.no_grad():
        for images, labels in loader:
            images_device = images.to(device)
            labels_device = labels.to(device)
            outputs: dict[float, dict[str, Any]] = {}
            for alpha in alphas:
                output = model(
                    images_device,
                    True,
                    return_dict=True,
                    alpha=alpha,
                )
                outputs[alpha] = output
                for stage in (0, 1, 2, 3):
                    update_stage_accumulator(
                        mechanism[alpha][stage],
                        output["pci"]["stage_terms"][stage],
                        eps=args.drift_eps,
                    )

            baseline = outputs[0.0]["pred"]
            baseline_binary = baseline > 0.0
            target = labels_device > 0.5
            total_pixels += int(target.numel())
            for alpha, output in outputs.items():
                logits = output["pred"]
                prediction = logits > 0.0
                intersections[alpha] += int((prediction & target).sum())
                unions[alpha] += int((prediction | target).sum())
                threshold_changed[alpha] += int(
                    (prediction != baseline_binary).sum()
                )
                absolute_logit_change[alpha] += float(
                    (logits - baseline).abs().sum()
                )
                pd_fa[alpha].update(logits.cpu(), labels)

    rows = []
    for alpha in alphas:
        false_alarm, detection_probability = pd_fa[alpha].get()
        rows.append(
            {
                "alpha": alpha,
                "iou": float(intersections[alpha] / max(1, unions[alpha])),
                "pd": float(detection_probability[0]),
                "fa_per_million": float(false_alarm[0] * 1e6),
                "threshold_changed_pixels": int(threshold_changed[alpha]),
                "threshold_changed_fraction": float(
                    threshold_changed[alpha] / max(1, total_pixels)
                ),
                "mean_absolute_logit_change": float(
                    absolute_logit_change[alpha] / max(1, total_pixels)
                ),
                "stages": {
                    "d%d" % stage: finalize_stage(mechanism[alpha][stage])
                    for stage in (3, 2, 1, 0)
                },
            }
        )

    report = {
        "scope": (
            "frozen persistent conditional-increment mechanics on a "
            "design-used split; no optimization"
        ),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_metrics": {
            key: checkpoint_metadata.get(key) for key in ("iou", "pd", "fa")
        },
        "dataset_dir": args.dataset_dir,
        "mode": args.mode,
        "images": len(dataset),
        "split_sha256": dataset.split_sha256,
        "checkpoint_split_sha256": expected_split_hash,
        "device": str(device),
        "anchor_mode": args.anchor_mode,
        "center_xi": args.center_xi,
        "alphas": list(alphas),
        "alpha_is_trainable": False,
        "drift_definition": {
            "mean": "(persistent_mean - inherited_mean) / abs(inherited_mean)",
            "std": "(persistent_std - inherited_std) / inherited_std",
            "aggregation": "mean over per-image complete-state statistics",
            "eps": args.drift_eps,
        },
        "checkpoint_load_missing_allowed": sorted(allowed_missing),
        "rows": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
