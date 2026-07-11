#!/usr/bin/env python3
"""Audit label-free analytic fine-scale controls on a frozen MSHNet.

The controls are fixed before labels are read.  Ground truth is used only to
report IoU/PD/FA on the selected split.  This is a design-used mechanics audit,
not a trained model and not confirmatory evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.MSHNet import MSHNet
from model.mshnet_checkpoint import strip_legacy_dea_lite_head
from model.mshnet_evidence_view import forward_mshnet_evidence
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
    parser.add_argument("--gammas", default="0.5,1,2,4")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(specification)


def load_state_dict(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        return checkpoint["net"]
    return checkpoint


def parse_gammas(specification: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in specification.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("--gammas must contain positive comma-separated values")
    return values


def analytic_predictions(
    z_base: torch.Tensor,
    contributions: torch.Tensor,
    gammas: tuple[float, ...],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return fixed, label-free analytic interventions."""

    c0 = contributions[:, 0:1]
    coarse = contributions[:, 1:].sum(dim=1, keepdim=True)
    conflict = (c0 * coarse < 0).to(c0.dtype)
    predictions = {
        "baseline": z_base,
        "drop_fine": z_base - c0,
        "conflict_drop_fine": z_base - conflict * c0,
    }

    coarse_absolute_mass = contributions[:, 1:].abs().sum(
        dim=1, keepdim=True
    )
    for gamma in gammas:
        cap = float(gamma) * coarse_absolute_mass
        capped_c0 = c0.sign() * torch.minimum(c0.abs(), cap)
        predictions["cap_fine_gamma_%g" % gamma] = z_base - c0 + capped_c0
    return predictions, conflict


def main() -> None:
    args = parse_args()
    gammas = parse_gammas(args.gammas)
    device = resolve_device(args.device)
    dataset = IRSTD_Dataset(args, args.mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = MSHNet(args.input_channels).to(device).eval()
    model.load_state_dict(
        strip_legacy_dea_lite_head(load_state_dict(args.checkpoint))
    )

    names = [
        "baseline",
        "drop_fine",
        "conflict_drop_fine",
        *("cap_fine_gamma_%g" % gamma for gamma in gammas),
    ]
    intersections = {name: 0 for name in names}
    unions = {name: 0 for name in names}
    pd_fa = {
        name: PD_FA(nclass=1, bins=10, size=args.crop_size) for name in names
    }
    threshold_changed = {name: 0 for name in names}
    absolute_logit_change = {name: 0.0 for name in names}
    total_pixels = 0
    conflict_pixels = 0
    max_reconstruction_error = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_device = labels.to(device)
            evidence = forward_mshnet_evidence(model, images, detach=True)
            max_reconstruction_error = max(
                max_reconstruction_error,
                float(
                    (evidence["z_reconstructed"] - evidence["z_base"])
                    .abs()
                    .max()
                ),
            )
            predictions, conflict = analytic_predictions(
                evidence["z_base"], evidence["contributions"], gammas
            )
            target = labels_device > 0.5
            baseline_binary = predictions["baseline"] > 0
            total_pixels += int(conflict.numel())
            conflict_pixels += int(conflict.sum())

            for name, logits in predictions.items():
                prediction = logits > 0
                intersections[name] += int((prediction & target).sum())
                unions[name] += int((prediction | target).sum())
                threshold_changed[name] += int(
                    (prediction != baseline_binary).sum()
                )
                absolute_logit_change[name] += float(
                    (logits - predictions["baseline"]).abs().sum()
                )
                pd_fa[name].update(logits.cpu(), labels)

    rows = []
    for name in names:
        false_alarm, detection_probability = pd_fa[name].get()
        rows.append(
            {
                "name": name,
                "iou": float(intersections[name] / max(1, unions[name])),
                "pd": float(detection_probability[0]),
                "fa_per_million": float(false_alarm[0] * 1e6),
                "threshold_changed_pixels": int(threshold_changed[name]),
                "mean_absolute_logit_change": float(
                    absolute_logit_change[name] / max(1, total_pixels)
                ),
            }
        )

    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "mode": args.mode,
        "images": len(dataset),
        "split_sha256": dataset.split_sha256,
        "scope": "fixed analytic controls on a design-used split",
        "definition_uses_ground_truth": False,
        "max_grouped_reconstruction_abs_error": max_reconstruction_error,
        "conflict_pixel_fraction": float(conflict_pixels / max(1, total_pixels)),
        "rows": rows,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
