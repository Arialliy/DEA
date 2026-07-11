#!/usr/bin/env python3
"""Frozen decoder-0 mechanics audit for SIED.

One SIED forward per batch extracts the factual D0 state and its exact
two-input Mobius terms.  The requested alpha values are then evaluated
offline through the original output_0 and final layers.  Nothing is trained,
and ground truth is used only for reporting metrics and pixel strata.
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

from model.dea_scale_interaction_exchange import (
    ScaleInteractionExchangeMSHNet,
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
    parser.add_argument("--alphas", default="0,0.025,0.05,0.1,0.2")
    parser.add_argument("--anchor-mode", choices=("zero", "mean"), default="mean")
    parser.add_argument("--ratio-eps", type=float, default=1e-6)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(specification)


def parse_alphas(specification: str) -> tuple[float, ...]:
    alphas = tuple(
        float(item.strip())
        for item in specification.split(",")
        if item.strip()
    )
    if not alphas:
        raise ValueError("--alphas must contain at least one value")
    if any(not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("--alphas values must be finite and in [0, 1]")
    if len(alphas) != len(set(alphas)):
        raise ValueError("--alphas must not contain duplicates")
    if 0.0 not in alphas:
        raise ValueError("--alphas must contain 0 for the baseline reference")
    return alphas


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
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict, metadata


def offline_logits(
    model: ScaleInteractionExchangeMSHNet,
    output: dict[str, Any],
    alphas: tuple[float, ...],
) -> dict[float, Tensor]:
    """Apply all requested exchange strengths without another model forward."""

    terms = output["sied"]["stage_terms"][0]
    coarse_masks = {
        1: model.output_1(output["decoder_features"][1]),
        2: model.output_2(output["decoder_features"][2]),
        3: model.output_3(output["decoder_features"][3]),
    }
    logits: dict[float, Tensor] = {}
    for alpha in alphas:
        d0 = terms["q11"] + float(alpha) * terms["exchange"]
        mask0 = model.output_0(d0)
        scale_logits = torch.cat(
            [
                mask0,
                model.up(coarse_masks[1]),
                model.up_4(coarse_masks[2]),
                model.up_8(coarse_masks[3]),
            ],
            dim=1,
        )
        logits[alpha] = model.final(scale_logits)
    return logits


def distribution_statistics(chunks: list[np.ndarray]) -> dict[str, Any]:
    if not chunks:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    values = np.concatenate(chunks).astype(np.float64, copy=False)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("all interaction-ratio values are non-finite")
    quantiles = np.percentile(values, (25, 50, 75, 90, 95))
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    if args.ratio_eps <= 0.0:
        raise ValueError("--ratio-eps must be positive")

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
    model = ScaleInteractionExchangeMSHNet(
        args.input_channels,
        alpha=1.0,
        active_stages=(0,),
        anchor_mode=args.anchor_mode,
        freeze_bn_statistics=True,
    ).to(device).eval()
    incompatible = model.load_state_dict(state_dict, strict=False)
    disallowed_missing = set(incompatible.missing_keys)
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "incompatible checkpoint: missing=%s unexpected=%s"
            % (sorted(disallowed_missing), sorted(incompatible.unexpected_keys))
        )

    intersections = {alpha: 0 for alpha in alphas}
    unions = {alpha: 0 for alpha in alphas}
    pd_fa = {
        alpha: PD_FA(nclass=1, bins=10, size=args.crop_size)
        for alpha in alphas
    }
    threshold_changed = {alpha: 0 for alpha in alphas}
    absolute_logit_change = {alpha: 0.0 for alpha in alphas}
    ratio_chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in ("global", "TP", "FP", "FN", "TN")
    }
    total_pixels = 0
    max_decomposition_error = 0.0
    max_alpha_one_offline_error = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images_device = images.to(device)
            labels_device = labels.to(device)
            # Exactly one network forward.  It computes D0's four shared
            # coalition calls and leaves D1--D3 on the native path.
            output = model(
                images_device,
                True,
                return_dict=True,
                alpha=1.0,
            )
            terms = output["sied"]["stage_terms"][0]
            reconstructed = (
                terms["q00"]
                + terms["current_main"]
                + terms["inherited_main"]
                + terms["interaction"]
            )
            max_decomposition_error = max(
                max_decomposition_error,
                float((reconstructed - terms["q11"]).abs().max()),
            )

            predictions = offline_logits(model, output, alphas)
            if 1.0 in predictions:
                max_alpha_one_offline_error = max(
                    max_alpha_one_offline_error,
                    float((predictions[1.0] - output["pred"]).abs().max()),
                )
            baseline = predictions[0.0]
            baseline_binary = baseline > 0.0
            target = labels_device > 0.5
            total_pixels += int(target.numel())

            interaction_rms = terms["interaction"].square().mean(
                dim=1, keepdim=True
            ).sqrt()
            independent_rms = terms["current_main"].square().mean(
                dim=1, keepdim=True
            ).sqrt()
            ratio = interaction_rms / (independent_rms + args.ratio_eps)
            strata = {
                "global": torch.ones_like(target, dtype=torch.bool),
                "TP": baseline_binary & target,
                "FP": baseline_binary & ~target,
                "FN": ~baseline_binary & target,
                "TN": ~baseline_binary & ~target,
            }
            for name, mask in strata.items():
                selected = ratio[mask]
                if selected.numel():
                    ratio_chunks[name].append(
                        selected.float().cpu().numpy().reshape(-1)
                    )

            for alpha, logits in predictions.items():
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
            }
        )

    report = {
        "scope": (
            "frozen decoder_0 SIED mechanics on a design-used split; "
            "GT used only for metrics and pixel strata"
        ),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_metadata": checkpoint_metadata,
        "dataset_dir": args.dataset_dir,
        "mode": args.mode,
        "images": len(dataset),
        "split_sha256": dataset.split_sha256,
        "device": str(device),
        "anchor_mode": args.anchor_mode,
        "active_stages": [0],
        "alphas": list(alphas),
        "ratio_definition": (
            "pixelwise channel-RMS(interaction) / "
            "(channel-RMS(current_main) + eps)"
        ),
        "ratio_eps": args.ratio_eps,
        "strata_definition": "pixelwise baseline-logit>0 versus GT>0.5",
        "interaction_ratio_statistics": {
            name: distribution_statistics(chunks)
            for name, chunks in ratio_chunks.items()
        },
        "max_mobius_reconstruction_abs_error": max_decomposition_error,
        "max_alpha_one_offline_vs_forward_abs_error": (
            max_alpha_one_offline_error if 1.0 in alphas else None
        ),
        "checkpoint_load_missing_allowed": sorted(allowed_missing),
        "rows": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
