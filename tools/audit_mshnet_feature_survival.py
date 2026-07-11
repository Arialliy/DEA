#!/usr/bin/env python3
"""Read-only feature-survival audit for final no-response MSHNet targets."""

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
from skimage import measure
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.MSHNet import MSHNet
from model.mshnet_checkpoint import strip_legacy_dea_lite_head
from model.mshnet_stage_evidence_view import forward_mshnet_stage_evidence
from utils.component_ledger import build_component_ledger
from utils.data import IRSTD_Dataset
from utils.feature_survival import (
    build_translation_control_set,
    evaluate_feature_survival,
    project_geometry_controls,
)
from utils.metric import match_components_hungarian


MAIN_PATH = (
    "input",
    "stem",
    "e0",
    "p0",
    "e1",
    "p1",
    "e2",
    "p2",
    "e3",
    "p3",
    "m",
    "j3",
    "d3",
    "j2",
    "d2",
    "j1",
    "d1",
    "j0",
    "d0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-name", default="checkpoint_best_iou.pkl")
    parser.add_argument("--threshold-probability", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--required-controls", type=int, default=64)
    parser.add_argument("--max-candidate-controls", type=int, default=256)
    parser.add_argument("--matched-controls-per-miss", type=int, default=1)
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


def _candidate(
    *,
    sample_index: int,
    sample_name: str,
    target_index: int,
    outcome: str,
    region,
    image_shape: tuple[int, int],
) -> dict[str, object]:
    centroid = tuple(float(value) for value in region.centroid)
    height, width = image_shape
    border_distance = min(
        centroid[0],
        centroid[1],
        height - 1 - centroid[0],
        width - 1 - centroid[1],
    )
    return {
        "sample_index": int(sample_index),
        "sample_name": sample_name,
        "target_index": int(target_index),
        "outcome": outcome,
        "area": int(region.area),
        "centroid": centroid,
        "border_distance": float(border_distance),
    }


def identify_cohorts(
    model,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    threshold_logit: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    no_response = []
    matched = []
    cursor = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Feature audit cohort pass"):
            _, logits = model(images.to(device, non_blocking=True), True)
            logit_arrays = logits.detach().cpu().numpy()[:, 0]
            target_arrays = targets.numpy()[:, 0] > 0.5
            for batch_index in range(logit_arrays.shape[0]):
                sample_index = cursor + batch_index
                sample_name = dataset.names[sample_index]
                target = target_arrays[batch_index]
                ledger = build_component_ledger(
                    logit_arrays[batch_index],
                    target,
                    threshold=threshold_logit,
                    input_semantics="logits",
                )
                component_match = match_components_hungarian(
                    logit_arrays[batch_index] > threshold_logit,
                    target,
                )
                matched_indices = {item[0] for item in component_match.matches}
                if tuple(component_match.unmatched_target_indices) != tuple(
                    sorted(
                        set(ledger.no_response_target_indices)
                        | set(ledger.centroid_miss_target_indices)
                        | (
                            set(component_match.unmatched_target_indices)
                            - set(ledger.no_response_target_indices)
                            - set(ledger.centroid_miss_target_indices)
                        )
                    )
                ):
                    raise RuntimeError("component cohort taxonomy drifted")
                for target_index in ledger.no_response_target_indices:
                    no_response.append(
                        _candidate(
                            sample_index=sample_index,
                            sample_name=sample_name,
                            target_index=target_index,
                            outcome="no_response",
                            region=component_match.target_regions[target_index],
                            image_shape=target.shape,
                        )
                    )
                for target_index in sorted(matched_indices):
                    matched.append(
                        _candidate(
                            sample_index=sample_index,
                            sample_name=sample_name,
                            target_index=target_index,
                            outcome="matched_control",
                            region=component_match.target_regions[target_index],
                            image_shape=target.shape,
                        )
                    )
            cursor += logit_arrays.shape[0]
    if cursor != len(dataset):
        raise RuntimeError("validation sample count drifted during cohort pass")
    return no_response, matched


def select_matched_controls(
    no_response: list[dict[str, object]],
    matched: list[dict[str, object]],
    *,
    controls_per_miss: int,
) -> list[dict[str, object]]:
    if controls_per_miss < 0:
        raise ValueError("controls_per_miss must be non-negative")
    available = list(matched)
    selected = []
    for miss in sorted(
        no_response,
        key=lambda item: (
            item["sample_name"],
            item["target_index"],
        ),
    ):
        for _ in range(controls_per_miss):
            if not available:
                raise RuntimeError("not enough matched targets for controls")

            def distance(control):
                area_distance = abs(
                    math.log1p(float(control["area"]))
                    - math.log1p(float(miss["area"]))
                )
                border_distance = abs(
                    float(control["border_distance"])
                    - float(miss["border_distance"])
                ) / 256.0
                return (
                    area_distance + 0.1 * border_distance,
                    control["sample_name"],
                    control["target_index"],
                )

            chosen = min(available, key=distance)
            available.remove(chosen)
            selected.append(chosen)
    return selected


def _trajectory(stage_results: dict[str, dict[str, object]]) -> dict[str, object]:
    states = [stage_results[stage]["state"] for stage in MAIN_PATH]
    drop_edges = []
    recovery_edges = []
    for left, right, left_state, right_state in zip(
        MAIN_PATH[:-1], MAIN_PATH[1:], states[:-1], states[1:]
    ):
        if left_state == "distinct" and right_state != "distinct":
            drop_edges.append((left, right))
        if left_state != "distinct" and right_state == "distinct":
            recovery_edges.append((left, right))
    first_drop_interval = drop_edges[0] if drop_edges else None
    return {
        "input_state": states[0],
        "drop_edges": drop_edges,
        "recovery_edges": recovery_edges,
        "first_operational_drop": first_drop_interval,
    }


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        outcome = record["outcome"]
        for stage, result in record["stages"].items():
            grouped[outcome][stage].append(result)
    result = {}
    for outcome, stages in sorted(grouped.items()):
        result[outcome] = {}
        for stage, values in stages.items():
            available = [item for item in values if item["available"]]
            counts = {
                state: sum(item["state"] == state for item in values)
                for state in ("distinct", "uncertain", "background_like", "undefined")
            }
            result[outcome][stage] = {
                "targets": len(values),
                "available": len(available),
                "state_counts": counts,
                "distinct_rate_among_available": (
                    counts["distinct"] / len(available) if available else 0.0
                ),
                "median_rank": (
                    float(np.median([item["rank"] for item in available]))
                    if available
                    else None
                ),
                "median_directional_auc": (
                    float(
                        np.median(
                            [
                                item["directional_auc"]
                                for item in available
                                if item["directional_auc"] is not None
                            ]
                        )
                    )
                    if any(
                        item["directional_auc"] is not None
                        for item in available
                    )
                    else None
                ),
            }
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold_probability < 1.0:
        raise ValueError("threshold probability must lie strictly in (0,1)")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size/workers are invalid")
    if args.required_controls < 1 or args.max_candidate_controls < args.required_controls:
        raise ValueError("candidate controls must cover required controls")
    if args.matched_controls_per_miss < 0:
        raise ValueError("matched-controls-per-miss must be non-negative")

    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output).resolve()
    checkpoint_path = run_dir / args.checkpoint_name
    config_path = run_dir / "run_config.json"
    if output_path.exists():
        raise FileExistsError("refusing to overwrite %s" % output_path)
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("run-dir must contain checkpoint and run_config.json")

    run_config = json.loads(config_path.read_text())
    stored_args = run_config.get("args", {})
    if stored_args.get("mode") != "train":
        raise RuntimeError("source run must be a training run")
    if stored_args.get("model_type") != "mshnet":
        raise RuntimeError("feature audit supports plain MSHNet only")
    dataset_args = Namespace(**stored_args)
    dataset = IRSTD_Dataset(dataset_args, mode="val")
    expected_val_hash = stored_args.get("val_split_sha256", "")
    if expected_val_hash and dataset.split_sha256 != expected_val_hash:
        raise RuntimeError("validation split hash mismatch")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = load_checkpoint(checkpoint_path)
    model = MSHNet(3)
    model.load_state_dict(
        strip_legacy_dea_lite_head(checkpoint["net"]), strict=True
    )
    model.requires_grad_(False).to(device).eval()
    threshold_logit = math.log(
        args.threshold_probability / (1.0 - args.threshold_probability)
    )

    no_response, matched = identify_cohorts(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold_logit=threshold_logit,
    )
    matched_controls = select_matched_controls(
        no_response,
        matched,
        controls_per_miss=args.matched_controls_per_miss,
    )
    selected = no_response + matched_controls
    selected_by_sample = defaultdict(list)
    for item in selected:
        selected_by_sample[int(item["sample_index"])].append(item)

    records = []
    sample_indices = sorted(selected_by_sample)
    subset_loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0 and bool(sample_indices),
    )
    with torch.no_grad():
        for subset_index, (images, targets) in enumerate(
            tqdm(subset_loader, desc="Feature audit stage pass")
        ):
            sample_index = sample_indices[subset_index]
            sample_name = dataset.names[sample_index]
            evidence = forward_mshnet_stage_evidence(
                model,
                images.to(device, non_blocking=True),
                detach=True,
            )
            target = targets.numpy()[0, 0] > 0.5
            target_labels = measure.label(target, connectivity=2)
            target_regions = tuple(measure.regionprops(target_labels))
            stage_tensors = {}
            stage_tensors.update(evidence["path"])
            stage_tensors.update(evidence["native_sides"])
            stage_tensors.update(evidence["full_sides"])
            stage_tensors.update(evidence["contributions"])
            stage_tensors["z"] = evidence["pred"]

            for cohort in selected_by_sample[sample_index]:
                target_index = int(cohort["target_index"])
                if target_index >= len(target_regions):
                    raise RuntimeError("target component indexing drifted")
                component = target_labels == target_regions[target_index].label
                controls = build_translation_control_set(
                    component,
                    target,
                    sample_key="%s:%d" % (sample_name, target_index),
                    max_candidate_controls=args.max_candidate_controls,
                )
                geometry_by_shape = {}
                stage_results = {}
                for stage, tensor in stage_tensors.items():
                    feature = tensor[0].detach().cpu().numpy()
                    spatial_shape = tuple(int(value) for value in feature.shape[1:])
                    if spatial_shape not in geometry_by_shape:
                        geometry_by_shape[spatial_shape] = project_geometry_controls(
                            controls,
                            spatial_shape,
                            required_controls=args.required_controls,
                        )
                    scalar_threshold = (
                        0.0
                        if stage == "z"
                        or stage.startswith("mask")
                        or stage in {"s0", "s1", "s2", "s3"}
                        else None
                    )
                    stage_results[stage] = evaluate_feature_survival(
                        feature,
                        geometry_by_shape[spatial_shape],
                        scalar_threshold=scalar_threshold,
                    ).as_dict()
                record = dict(cohort)
                record["stages"] = stage_results
                record["main_path_trajectory"] = _trajectory(stage_results)
                records.append(record)

    output = {
        "schema": "mshnet_feature_survival_audit_v1",
        "exploratory_only": True,
        "leakage_note": (
            "best-IoU checkpoint selection and this audit use the same internal "
            "validation split; use only for direction triage"
        ),
        "source_run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_dir": stored_args.get("dataset_dir"),
        "split": "internal_validation",
        "val_split_sha256": dataset.split_sha256,
        "threshold_probability": args.threshold_probability,
        "threshold_logit": threshold_logit,
        "required_geometry_controls": args.required_controls,
        "max_candidate_controls": args.max_candidate_controls,
        "matched_controls_per_miss": args.matched_controls_per_miss,
        "num_validation_images": len(dataset),
        "num_no_response_targets": len(no_response),
        "num_matched_candidates": len(matched),
        "num_selected_matched_controls": len(matched_controls),
        "main_path": MAIN_PATH,
        "summary": _summarize(records),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("wrote %s" % output_path)


if __name__ == "__main__":
    main()
