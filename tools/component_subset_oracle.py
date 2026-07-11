#!/usr/bin/env python3
"""Single-candidate 16-subset oracle for frozen MSHNet evidence.

Candidates are generated only from final/side predictions.  Ground truth is
used afterwards to label and score interventions.  This is a diagnostic upper
bound on local contribution reassembly, not a deployable adjudicator.
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
from utils.component_evidence import generate_prediction_candidates
from utils.data import IRSTD_Dataset
from utils.metric import match_connected_components
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
    parser.add_argument(
        "--candidate-thresholds",
        type=float,
        nargs="+",
        default=(0.5, 0.3, 0.2, 0.1),
    )
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


def binary_metrics(logit: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = logit > 0.0
    target_binary = target > 0.5
    intersection = int(np.logical_and(prediction, target_binary).sum())
    union = int(np.logical_or(prediction, target_binary).sum())
    match = match_connected_components(prediction, target_binary)
    false_alarm_area = int(
        sum(
            match.prediction_regions[index].area
            for index in match.unmatched_prediction_indices
        )
    )
    return {
        "intersection": intersection,
        "union": union,
        "iou": float(intersection / max(1, union)),
        "matched_targets": len(match.matches),
        "target_count": len(match.target_regions),
        "false_alarm_area": false_alarm_area,
        "prediction_count": len(match.prediction_regions),
        "match": match,
    }


def is_strict_dominator(candidate, baseline) -> bool:
    return (
        candidate["iou"] >= baseline["iou"]
        and candidate["matched_targets"] >= baseline["matched_targets"]
        and candidate["false_alarm_area"] <= baseline["false_alarm_area"]
        and (
            candidate["iou"] > baseline["iou"]
            or candidate["matched_targets"] > baseline["matched_targets"]
            or candidate["false_alarm_area"] < baseline["false_alarm_area"]
        )
    )


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
    model.load_state_dict(
        strip_legacy_dea_lite_head(load_state_dict(args.checkpoint))
    )

    summary = {
        "images": len(dataset),
        "targets": 0,
        "baseline_matched_targets": 0,
        "baseline_false_alarm_components": 0,
        "baseline_false_alarm_area": 0,
        "candidates": 0,
        "candidates_overlapping_baseline_fp": 0,
        "candidates_near_baseline_fn": 0,
        "candidates_with_strict_dominator": 0,
        "baseline_fp_candidates_repairable": 0,
        "baseline_fn_candidates_repairable": 0,
    }
    repairable_records = []
    image_cursor = 0

    with torch.no_grad():
        for images, labels in loader:
            evidence = forward_mshnet_evidence(
                model, images.to(device), detach=True
            )
            batch_size = images.shape[0]
            for batch_index in range(batch_size):
                image_id = dataset.names[image_cursor]
                image_cursor += 1
                z_base = evidence["z_base"][batch_index, 0].cpu().numpy()
                z_reconstructed = evidence["z_reconstructed"][
                    batch_index, 0
                ].cpu().numpy()
                scale_logits = evidence["scale_logits"][
                    batch_index
                ].cpu().numpy()
                contributions = evidence["contributions"][
                    batch_index : batch_index + 1
                ]
                fusion_bias = evidence["fusion_bias"]
                target = labels[batch_index, 0].numpy()
                baseline = binary_metrics(z_base, target)
                baseline_match = baseline["match"]

                summary["targets"] += baseline["target_count"]
                summary["baseline_matched_targets"] += baseline[
                    "matched_targets"
                ]
                summary["baseline_false_alarm_components"] += len(
                    baseline_match.unmatched_prediction_indices
                )
                summary["baseline_false_alarm_area"] += baseline[
                    "false_alarm_area"
                ]

                candidates = generate_prediction_candidates(
                    z_base,
                    scale_logits,
                    probability_thresholds=args.candidate_thresholds,
                )
                summary["candidates"] += len(candidates)

                fp_masks = [
                    baseline_match.prediction_label_map
                    == baseline_match.prediction_regions[index].label
                    for index in baseline_match.unmatched_prediction_indices
                ]
                fn_centroids = [
                    np.asarray(baseline_match.target_regions[index].centroid)
                    for index in baseline_match.unmatched_target_indices
                ]

                for candidate in candidates:
                    overlaps_fp = any(
                        np.logical_and(candidate.mask, mask).any()
                        for mask in fp_masks
                    )
                    near_fn = any(
                        np.linalg.norm(
                            np.asarray(candidate.centroid) - target_centroid
                        ) < 3.0
                        for target_centroid in fn_centroids
                    )
                    summary["candidates_overlapping_baseline_fp"] += int(
                        overlaps_fp
                    )
                    summary["candidates_near_baseline_fn"] += int(near_fn)

                    candidate_rows = []
                    candidate_mask = candidate.mask
                    for subset in range(16):
                        if subset == 15:
                            modified_logit = z_base
                        else:
                            subset_logit = reconstruct_scale_subset(
                                contributions,
                                fusion_bias,
                                subset,
                            )[0, 0].cpu().numpy()
                            modified_logit = z_base.copy()
                            modified_logit[candidate_mask] += (
                                subset_logit[candidate_mask]
                                - z_reconstructed[candidate_mask]
                            )
                        metrics = binary_metrics(modified_logit, target)
                        metrics["subset"] = subset
                        candidate_rows.append(metrics)

                    dominators = [
                        row
                        for row in candidate_rows
                        if row["subset"] != 15
                        and is_strict_dominator(row, baseline)
                    ]
                    if not dominators:
                        continue

                    summary["candidates_with_strict_dominator"] += 1
                    repairs_fp = overlaps_fp and any(
                        row["false_alarm_area"] < baseline["false_alarm_area"]
                        for row in dominators
                    )
                    repairs_fn = near_fn and any(
                        row["matched_targets"] > baseline["matched_targets"]
                        for row in dominators
                    )
                    summary["baseline_fp_candidates_repairable"] += int(
                        repairs_fp
                    )
                    summary["baseline_fn_candidates_repairable"] += int(
                        repairs_fn
                    )
                    best = max(dominators, key=lambda row: row["iou"])
                    repairable_records.append(
                        {
                            "image_id": image_id,
                            "candidate_id": candidate.candidate_id,
                            "source": candidate.source,
                            "threshold": candidate.probability_threshold,
                            "area": candidate.area,
                            "centroid": list(candidate.centroid),
                            "overlaps_baseline_fp": overlaps_fp,
                            "near_baseline_fn": near_fn,
                            "best_subset": best["subset"],
                            "best_kept_scales": list(
                                kept_scale_indices(best["subset"])
                            ),
                            "baseline_iou": baseline["iou"],
                            "modified_iou": best["iou"],
                            "baseline_matched_targets": baseline[
                                "matched_targets"
                            ],
                            "modified_matched_targets": best[
                                "matched_targets"
                            ],
                            "baseline_false_alarm_area": baseline[
                                "false_alarm_area"
                            ],
                            "modified_false_alarm_area": best[
                                "false_alarm_area"
                            ],
                        }
                    )

    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "mode": args.mode,
        "split_sha256": dataset.split_sha256,
        "scope": (
            "prediction-only candidates; one candidate changed at a time; "
            "GT used only after generation for oracle scoring"
        ),
        "candidate_thresholds": list(args.candidate_thresholds),
        "summary": summary,
        "repairable_records": repairable_records,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
