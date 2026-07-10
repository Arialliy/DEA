"""Audit a trained predictive-correction MSHNet on a fixed NUAA split.

This script reports segmentation metrics and mechanism diagnostics separately.
Local observation-energy descent is treated only as a numerical invariant; it
is never used to select a checkpoint or claim segmentation improvement.
"""

import argparse
import json
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.dea_mshnet import DEAMSHNet
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA, mIoU


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260710)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state_dict = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    return checkpoint, state_dict


def model_config(checkpoint):
    metadata = checkpoint.get("method_meta", {}) if isinstance(checkpoint, dict) else {}
    return {
        "state_channels": int(metadata.get("predictive_state_channels", 32)),
        "step_size": float(metadata.get("predictive_step_size", 1.0)),
        "delta_init": float(metadata.get("predictive_delta_init", 1.0)),
        "delta_min": float(metadata.get("predictive_delta_min", 0.05)),
        # Checkpoints created before this field was introduced used the legacy
        # algebraic form for the influence function.
        "legacy_influence_numerics": bool(
            metadata.get("predictive_legacy_numerics", True)
        ),
    }


def dataset_args(args):
    return Namespace(
        dataset_dir=args.dataset_dir,
        train_split_file="",
        val_split_file="",
        test_split_file="",
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        seed=args.split_seed,
        crop_size=args.base_size,
        base_size=args.base_size,
    )


def tensor_summary(values):
    values = torch.cat(values).float()
    return {
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def main():
    args = parse_args()
    checkpoint, state_dict = load_checkpoint(args.checkpoint)
    config = model_config(checkpoint)
    model = DEAMSHNet(3, **config)
    model.load_state_dict(state_dict, strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    dataset = IRSTD_Dataset(dataset_args(args), mode=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    final_iou = mIoU(1)
    final_pd_fa = PD_FA(1, 10, args.base_size)
    prefix_ious = [mIoU(1) for _ in range(5)]
    energy_before = [[] for _ in range(5)]
    energy_after = [[] for _ in range(5)]
    correction_to_prior = [[] for _ in range(5)]
    correction_prior_cosine = [[] for _ in range(5)]
    residual_norm_ratio = [[] for _ in range(5)]

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            output = model(
                images, True, return_dict=True, return_details=True
            )
            prediction = output["pred"]
            final_iou.update(prediction, masks)
            final_pd_fa.update(prediction, masks)

            for scale, logit in enumerate(output["state_logits"]):
                resized = F.interpolate(
                    logit, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
                prefix_ious[scale].update(resized, masks)

            for scale, (before, after) in enumerate(zip(
                output["local_observation_energies_before"],
                output["local_observation_energies_after"],
            )):
                energy_before[scale].append(before.detach().cpu())
                energy_after[scale].append(after.detach().cpu())

            for scale, (before, after) in enumerate(zip(
                output["residuals"], output["residuals_after"]
            )):
                before_norm = before.flatten(1).norm(dim=1)
                after_norm = after.flatten(1).norm(dim=1)
                valid = before_norm > 1e-8
                residual_norm_ratio[scale].append(
                    (after_norm[valid] / before_norm[valid]).detach().cpu()
                )

            for scale, (correction, state_bar) in enumerate(zip(
                output["corrections"], output["state_bars"]
            )):
                correction_flat = correction.flatten(1)
                prior_flat = state_bar.flatten(1)
                if scale == 0:
                    continue
                ratio = correction_flat.norm(dim=1) / (
                    prior_flat.norm(dim=1) + 1e-6
                )
                cosine = F.cosine_similarity(
                    correction_flat, prior_flat, dim=1, eps=1e-6
                )
                correction_to_prior[scale].append(ratio.detach().cpu())
                correction_prior_cosine[scale].append(cosine.detach().cpu())

    false_alarm, probability_detection = final_pd_fa.get()
    _, mean_iou = final_iou.get()
    bounded_depthwise, bounded_pointwise = (
        model.prediction_operator.bounded_weights()
    )
    pointwise_norm = torch.linalg.matrix_norm(
        bounded_pointwise.view(config["state_channels"], -1).float(), ord=2
    )
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "split": args.split,
        "split_count": len(dataset),
        "split_sha256": dataset.split_sha256,
        "config": config,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "segmentation": {
            "iou": float(mean_iou),
            "pd": float(probability_detection[0]),
            "fa_per_million": float(false_alarm[0] * 1_000_000),
            "prefix_full_resolution_iou_coarse_to_fine": [
                float(metric.get()[1]) for metric in prefix_ious
            ],
        },
        "operator": {
            "max_depthwise_l1": float(
                bounded_depthwise.abs().sum(dim=(1, 2, 3)).max().item()
            ),
            "pointwise_spectral_norm": float(pointwise_norm.item()),
            "delta": tensor_summary([model.delta.detach().cpu().flatten()]),
        },
        "local_observation_sanity": [],
        "state_update": [],
    }
    for scale in range(5):
        before = torch.cat(energy_before[scale]).float()
        after = torch.cat(energy_after[scale]).float()
        difference = after - before
        ratios = torch.cat(residual_norm_ratio[scale]).float()
        report["local_observation_sanity"].append({
            "scale_coarse_to_fine": scale,
            "energy_before_mean": float(before.mean().item()),
            "energy_after_mean": float(after.mean().item()),
            "max_energy_increase": float(difference.max().item()),
            "violation_fraction_gt_1e-6": float(
                (difference > 1e-6).float().mean().item()
            ),
            "residual_norm_ratio": tensor_summary([ratios]),
        })
        if scale == 0:
            report["state_update"].append({
                "scale_coarse_to_fine": scale,
                "note": "broadcast prior is zero-like; relative overwrite is undefined",
            })
        else:
            ratios = torch.cat(correction_to_prior[scale]).float()
            cosines = torch.cat(correction_prior_cosine[scale]).float()
            report["state_update"].append({
                "scale_coarse_to_fine": scale,
                "correction_to_prior": tensor_summary([ratios]),
                "correction_prior_cosine": tensor_summary([cosines]),
            })

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
