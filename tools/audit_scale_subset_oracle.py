#!/usr/bin/env python3
"""Audit all 16 global MSHNet scale subsets on a frozen data split.

This is a diagnostic control, not a component oracle and not a model.  The
all-scale entry always uses MSHNet's direct final-convolution output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.MSHNet import MSHNet
from model.mshnet_evidence_view import forward_mshnet_evidence
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA
from utils.scale_subset import kept_scale_indices, reconstruct_scale_subset


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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = IRSTD_Dataset(args, args.mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = MSHNet(args.input_channels).to(device).eval()
    model.load_state_dict(load_state_dict(args.checkpoint))

    subset_count = 16
    intersections = np.zeros(subset_count, dtype=np.int64)
    unions = np.zeros(subset_count, dtype=np.int64)
    component_metrics = [
        PD_FA(nclass=1, bins=10, size=args.crop_size)
        for _ in range(subset_count)
    ]
    max_reconstruction_error = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_device = labels.to(device)
            evidence = forward_mshnet_evidence(model, images, detach=True)
            max_reconstruction_error = max(
                max_reconstruction_error,
                float(
                    (
                        evidence["z_reconstructed"] - evidence["z_base"]
                    ).abs().max()
                ),
            )
            for subset in range(subset_count):
                logits = reconstruct_scale_subset(
                    evidence["contributions"],
                    evidence["fusion_bias"],
                    subset,
                    z_base=evidence["z_base"],
                )
                prediction = logits > 0
                target = labels_device > 0.5
                intersections[subset] += int((prediction & target).sum())
                unions[subset] += int((prediction | target).sum())
                component_metrics[subset].update(logits.cpu(), labels)

    rows = []
    for subset in range(subset_count):
        false_alarm, detection_probability = component_metrics[subset].get()
        rows.append(
            {
                "subset_id": subset,
                "bitmask_s3_to_s0": format(subset, "04b"),
                "kept_scales": list(kept_scale_indices(subset)),
                "iou": float(intersections[subset] / max(1, unions[subset])),
                "pd": float(detection_probability[0]),
                "fa_per_million": float(false_alarm[0] * 1e6),
            }
        )

    baseline = next(row for row in rows if row["subset_id"] == 15)
    strict_dominators = [
        row
        for row in rows
        if row["iou"] >= baseline["iou"]
        and row["pd"] >= baseline["pd"]
        and row["fa_per_million"] <= baseline["fa_per_million"]
        and (
            row["iou"] > baseline["iou"]
            or row["pd"] > baseline["pd"]
            or row["fa_per_million"] < baseline["fa_per_million"]
        )
    ]
    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "mode": args.mode,
        "images": len(dataset),
        "split_sha256": dataset.split_sha256,
        "scope": "global fixed subset; not a component oracle",
        "max_grouped_reconstruction_abs_error": max_reconstruction_error,
        "baseline_all_scales": baseline,
        "strict_dominators": strict_dominators,
        "rows_by_iou": sorted(rows, key=lambda row: row["iou"], reverse=True),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
