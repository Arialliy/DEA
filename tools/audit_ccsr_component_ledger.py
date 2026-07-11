#!/usr/bin/env python3
"""Offline CCSR Gate-C1 ledger audit on a sealed validation split.

The script is read-only with respect to checkpoints and datasets.  It refuses
to overwrite its output and never evaluates the official test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Keep direct ``python tools/audit_ccsr_component_ledger.py`` invocation valid.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.MSHNet import MSHNet
from model.mshnet_checkpoint import strip_legacy_dea_lite_head
from model.mshnet_evidence_view import forward_mshnet_evidence
from utils.component_ledger import build_component_ledger
from utils.data import IRSTD_Dataset
from utils.no_response_scale import analyze_no_response_scales


SUM_FIELDS = (
    "num_gt",
    "num_pred_components",
    "legacy_matches",
    "hungarian_matches",
    "unmatched_gt",
    "unmatched_pred_components",
    "unmatched_pred_area",
    "no_response_gt",
    "centroid_miss_gt",
    "merged_gt_count",
    "split_prediction_count",
    "bridge_candidate_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_best_iou.pkl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--threshold-probabilities",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-scale-audit", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "net" not in checkpoint:
        raise RuntimeError("audit requires a metadata checkpoint with a net field")
    return checkpoint


def parse_probabilities(specification: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(item) for item in specification.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError("threshold probabilities must be comma-separated numbers") from exc
    if not values or any(
        not math.isfinite(value) or not 0.0 < value < 1.0
        for value in values
    ):
        raise ValueError("threshold probabilities must be finite and in (0,1)")
    if len(values) != len(set(values)):
        raise ValueError("threshold probabilities must be unique")
    return values


def summarize_records(records: list[dict]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"images": len(records)}
    for field in SUM_FIELDS:
        summary[field] = int(sum(int(record[field]) for record in records))
    summary["images_with_bridge_candidate"] = int(
        sum(int(record["bridge_candidate_count"] > 0) for record in records)
    )
    summary["images_with_multiple_gt"] = int(
        sum(int(record["num_gt"] > 1) for record in records)
    )
    summary["mean_component_risk"] = float(
        np.mean([record["raw_component_edit_risk"] for record in records])
    )
    num_gt = int(summary["num_gt"])
    summary["hungarian_pd"] = (
        float(summary["hungarian_matches"]) / num_gt if num_gt else 0.0
    )
    summary["legacy_pd"] = (
        float(summary["legacy_matches"]) / num_gt if num_gt else 0.0
    )
    if records:
        height = int(records[0]["height"])
        width = int(records[0]["width"])
        summary["fa_per_million"] = (
            float(summary["unmatched_pred_area"])
            / (len(records) * height * width)
            * 1e6
        )
    else:
        summary["fa_per_million"] = 0.0
    scale_records = [
        item
        for record in records
        for item in record.get("no_response_scale_records", ())
    ]
    if scale_records:
        summary["no_response_with_any_side_support"] = int(
            sum(bool(item["side_support_scales"]) for item in scale_records)
        )
        summary["no_response_with_any_side_centroid"] = int(
            sum(
                bool(item["side_centroid_legal_scales"])
                for item in scale_records
            )
        )
        summary["no_response_matched_by_any_side"] = int(
            sum(bool(item["side_matched_scales"]) for item in scale_records)
        )
        summary["no_response_recoverable_by_global_subset"] = int(
            sum(bool(item["recovering_subsets"]) for item in scale_records)
        )
        summary["no_response_absent_from_all_sides"] = int(
            sum(not item["side_support_scales"] for item in scale_records)
        )
        summary["side_support_counts"] = {
            str(scale): int(
                sum(
                    scale in item["side_support_scales"]
                    for item in scale_records
                )
            )
            for scale in range(4)
        }
        summary["recovering_subset_counts"] = {
            str(subset): int(
                sum(
                    subset in item["recovering_subsets"]
                    for item in scale_records
                )
            )
            for subset in range(1, 15)
        }
    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output).resolve()
    checkpoint_path = run_dir / args.checkpoint_name
    config_path = run_dir / "run_config.json"
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("run-dir must contain checkpoint and run_config.json")
    if output_path.exists():
        raise FileExistsError("refusing to overwrite %s" % output_path)

    run_config = json.loads(config_path.read_text())
    stored_args = run_config.get("args", {})
    if stored_args.get("mode") != "train":
        raise RuntimeError("source run must be a training run")
    if stored_args.get("model_type") != "mshnet":
        raise RuntimeError("Gate-C1 audit currently supports plain MSHNet only")
    dataset_args = Namespace(**stored_args)
    dataset = IRSTD_Dataset(dataset_args, mode="val")
    expected_val_hash = stored_args.get("val_split_sha256", "")
    if expected_val_hash and dataset.split_sha256 != expected_val_hash:
        raise RuntimeError(
            "validation split hash mismatch: %s != %s"
            % (dataset.split_sha256, expected_val_hash)
        )

    probabilities = parse_probabilities(args.threshold_probabilities)
    logit_thresholds = tuple(
        math.log(probability / (1.0 - probability))
        for probability in probabilities
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = load_checkpoint(checkpoint_path)
    model = MSHNet(3)
    state = strip_legacy_dea_lite_head(checkpoint["net"])
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).to(device).eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    records_by_probability: dict[float, list[dict]] = defaultdict(list)
    image_cursor = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="CCSR ledger audit"):
            images_device = images.to(device, non_blocking=True)
            if args.include_scale_audit:
                evidence = forward_mshnet_evidence(
                    model,
                    images_device,
                    detach=True,
                )
                logits_tensor = evidence["pred"]
                scale_logits = evidence["scale_logits"].cpu().numpy()
                contributions = evidence["contributions"].cpu().numpy()
                fusion_bias = evidence["fusion_bias"].cpu().numpy()
            else:
                _, logits_tensor = model(images_device, True)
                scale_logits = contributions = fusion_bias = None
            logits = logits_tensor.detach().cpu().numpy()[:, 0]
            target_arrays = targets.numpy()[:, 0] > 0.5
            for batch_index in range(logits.shape[0]):
                sample_name = dataset.names[image_cursor + batch_index]
                for probability, threshold_logit in zip(
                    probabilities, logit_thresholds
                ):
                    ledger = build_component_ledger(
                        logits[batch_index],
                        target_arrays[batch_index],
                        threshold=threshold_logit,
                        input_semantics="logits",
                    )
                    record = ledger.as_dict()
                    record.update(
                        sample_name=sample_name,
                        probability_threshold=probability,
                        height=int(logits.shape[1]),
                        width=int(logits.shape[2]),
                    )
                    if args.include_scale_audit:
                        scale_result = analyze_no_response_scales(
                            logits[batch_index],
                            scale_logits[batch_index],
                            contributions[batch_index],
                            fusion_bias,
                            target_arrays[batch_index],
                            threshold_logit=threshold_logit,
                        )
                        if (
                            scale_result.final_no_response_target_indices
                            != ledger.no_response_target_indices
                        ):
                            raise RuntimeError(
                                "ledger and scale audit disagree on no-response targets"
                            )
                        record["no_response_scale_records"] = [
                            item.as_dict() for item in scale_result.records
                        ]
                    records_by_probability[probability].append(record)
            image_cursor += logits.shape[0]
    if image_cursor != len(dataset):
        raise RuntimeError("validation sample count drifted during audit")

    thresholds = []
    all_records = []
    for probability, threshold_logit in zip(probabilities, logit_thresholds):
        records = records_by_probability[probability]
        thresholds.append(
            {
                "probability": probability,
                "logit": threshold_logit,
                "summary": summarize_records(records),
            }
        )
        all_records.extend(records)

    output = {
        "schema": (
            "ccsr_component_scale_ledger_audit_v2"
            if args.include_scale_audit
            else "ccsr_component_ledger_audit_v1"
        ),
        "source_run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_dir": stored_args.get("dataset_dir"),
        "split": "internal_validation",
        "val_split_sha256": dataset.split_sha256,
        "num_images": len(dataset),
        "include_scale_audit": bool(args.include_scale_audit),
        "thresholds": thresholds,
        "records": all_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("wrote %s" % output_path)


if __name__ == "__main__":
    main()
