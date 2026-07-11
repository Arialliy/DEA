#!/usr/bin/env python3
"""Read-only decision-conversion audit for pristine MSHNet checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.MSHNet import MSHNet
from model.mshnet_checkpoint import strip_legacy_dea_lite_head
from model.mshnet_stage_evidence_view import forward_mshnet_stage_evidence
from tools.audit_mshnet_feature_survival import (
    identify_cohorts,
    load_checkpoint,
    sha256_file,
)
from utils.data import IRSTD_Dataset
from utils.component_operating_point import (
    calibrate_component_operating_points,
    evaluate_component_operating_points,
    evaluate_target_operating_status,
)
from utils.feature_survival import (
    build_translation_control_set,
    evaluate_feature_survival,
    project_geometry_controls,
    select_context_matched_controls,
)
from utils.final_fusion_conversion import factorize_final_fusion_margin
from utils.head_conversion import (
    evaluate_linear_head_conversion,
    robust_channel_center_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-name", default="checkpoint_best_iou.pkl")
    parser.add_argument("--threshold-probability", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--required-controls", type=int, default=64)
    parser.add_argument("--max-candidate-controls", type=int, default=256)
    parser.add_argument("--context-control-count", type=int, default=64)
    parser.add_argument("--context-ring-width", type=float, default=8.0)
    parser.add_argument("--matched-controls-per-miss", type=int, default=1)
    parser.add_argument(
        "--fixed-fa-budgets",
        type=float,
        nargs="+",
        default=(1.0, 10.0, 20.0),
        metavar="FA_PER_MPIX",
    )
    return parser.parse_args()


def _normalized_rgb_to_context_luminance(images: torch.Tensor) -> np.ndarray:
    """Invert repository normalization and derive deterministic RGB luminance."""

    if images.shape[0] != 1 or images.shape[1] != 3:
        raise ValueError("context matching expects one normalized RGB image")
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    restored = images * std + mean
    if not bool(torch.isfinite(restored).all()):
        raise RuntimeError("inverse-normalized context image is non-finite")
    tolerance = 2e-6
    if bool(torch.any(restored < -tolerance)) or bool(
        torch.any(restored > 1.0 + tolerance)
    ):
        raise RuntimeError("inverse-normalized context image escaped [0,1]")
    luminance_weights = restored.new_tensor((0.299, 0.587, 0.114)).view(
        1, 3, 1, 1
    )
    luminance = torch.sum(restored * luminance_weights, dim=1)
    return luminance[0].detach().cpu().numpy().astype(np.float64)


def _select_paired_matched_controls(
    no_response: list[dict[str, object]],
    matched: list[dict[str, object]],
    *,
    controls_per_miss: int,
) -> list[dict[str, object]]:
    """Area/border match controls while retaining the exact pairing ledger."""

    if controls_per_miss < 0:
        raise ValueError("controls_per_miss must be non-negative")
    available = list(matched)
    selected = []
    for miss in sorted(
        no_response,
        key=lambda item: (item["sample_name"], item["target_index"]),
    ):
        miss_id = "%s:%d" % (miss["sample_name"], miss["target_index"])
        for pair_index in range(controls_per_miss):
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
            annotated = dict(chosen)
            annotated["paired_no_response_id"] = miss_id
            annotated["pair_index"] = pair_index
            selected.append(annotated)
    return selected


def _empty_conversion(reason: str) -> dict[str, object]:
    return {
        "available": False,
        "reason": reason,
        "channels": None,
        "mean_margin_availability": None,
        "normalized_mean_margin_availability": None,
        "head_sensitivity": None,
        "utilization_cosine": None,
        "mean_logit_margin": None,
        "reconstructed_margin": None,
        "reconstruction_error": None,
        "absolute_scale_floor_active_channels": None,
        "reparameterization_stable": None,
        "target_mean_logit": None,
        "background_mean_logit": None,
    }


def _empty_contribution_margins(reason: str) -> dict[str, object]:
    return {
        "available": False,
        "reason": reason,
        "per_scale": None,
        "sum": None,
        "final_direct": None,
        "positive_sum": None,
        "negative_sum": None,
        "has_sign_cancellation": None,
    }


def _context_selection_record(selection) -> dict[str, object]:
    return {
        "available": bool(selection.available),
        "reason": selection.reason,
        "descriptor_names": list(selection.descriptor_names),
        "target_descriptor": selection.target_descriptor,
        "descriptor_center": selection.descriptor_center,
        "descriptor_scale": selection.descriptor_scale,
        "active_descriptor_mask": selection.active_descriptor_mask,
        "covariance_condition_number": selection.covariance_condition_number,
        "context_distance_caliper": selection.context_distance_caliper,
        "target_ring_pixels": selection.target_ring_pixels,
        "target_stencil_pixels": selection.target_stencil_pixels,
        "target_ring_coverage": selection.target_ring_coverage,
        "eligible_candidate_count": selection.eligible_candidate_count,
        "selected_count": len(selection.selected),
        "selected_mask_digests": [item.mask_digest for item in selection.selected],
        "selected_distances": [
            item.mahalanobis_distance for item in selection.selected
        ],
        "rejected_candidate_counts": dict(selection.rejected_candidate_counts),
    }


def _operating_fold(sample_name: str) -> int:
    digest = hashlib.sha256(
        ("mshnet-decision-operating-fold-v1\0" + sample_name).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") % 2


def _budget_key(value: float) -> str:
    return "%.12g" % float(value)


def _collect_validation_predictions(
    model,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    logits = []
    targets = []
    with torch.no_grad():
        for images, batch_targets in tqdm(
            loader, desc="Operating-point prediction pass"
        ):
            _, batch_logits = model(images.to(device, non_blocking=True), True)
            logit_array = batch_logits.detach().cpu().numpy()[:, 0]
            target_array = batch_targets.numpy()[:, 0] > 0.5
            logits.extend(array.astype(np.float64) for array in logit_array)
            targets.extend(array.astype(bool) for array in target_array)
    if len(logits) != len(dataset) or len(targets) != len(dataset):
        raise RuntimeError("operating-point prediction count drifted")
    return tuple(logits), tuple(targets)


def _build_cross_fitted_operating_audit(
    logits: tuple[np.ndarray, ...],
    targets: tuple[np.ndarray, ...],
    names: list[str],
    *,
    fixed_threshold: float,
    budgets: tuple[float, ...],
) -> tuple[dict[str, object], dict[int, dict[str, dict[str, object]]]]:
    if not (len(logits) == len(targets) == len(names)):
        raise ValueError("operating-point samples and names must align")
    folds = tuple(_operating_fold(name) for name in names)
    if set(folds) != {0, 1}:
        raise RuntimeError("two-fold operating calibration requires both folds")
    fixed_probability_logits = tuple(
        math.log(probability / (1.0 - probability))
        for probability in (
            0.01,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
        )
    )
    threshold_lookup: dict[int, dict[str, dict[str, object]]] = {}
    fold_records = {}
    for evaluation_fold in (0, 1):
        evaluation_indices = [
            index for index, fold in enumerate(folds) if fold == evaluation_fold
        ]
        calibration_indices = [
            index for index, fold in enumerate(folds) if fold != evaluation_fold
        ]
        calibration = calibrate_component_operating_points(
            [logits[index] for index in calibration_indices],
            [targets[index] for index in calibration_indices],
            budgets,
            fixed_thresholds=(*fixed_probability_logits, fixed_threshold),
        )
        evaluation_curve = evaluate_component_operating_points(
            [logits[index] for index in evaluation_indices],
            [targets[index] for index in evaluation_indices],
            [selection.threshold for selection in calibration.selections],
        )
        evaluation_by_threshold = {
            point.threshold: point for point in evaluation_curve
        }
        selection_records = {}
        threshold_lookup[evaluation_fold] = {}
        for selection in calibration.selections:
            key = _budget_key(selection.fa_budget_per_million_pixels)
            evaluation_point = evaluation_by_threshold[selection.threshold]
            record = {
                "fa_budget_per_million_pixels": (
                    selection.fa_budget_per_million_pixels
                ),
                "threshold": selection.threshold,
                "calibration_operating_point": asdict(selection.operating_point),
                "evaluation_operating_point": asdict(evaluation_point),
            }
            selection_records[key] = record
            threshold_lookup[evaluation_fold][key] = record
        fold_records[str(evaluation_fold)] = {
            "evaluation_sample_count": len(evaluation_indices),
            "calibration_sample_count": len(calibration_indices),
            "evaluation_names": [names[index] for index in evaluation_indices],
            "calibration_names": [names[index] for index in calibration_indices],
            "threshold_grid": list(calibration.threshold_grid),
            "calibration_curve": [asdict(point) for point in calibration.curve],
            "fixed_fa_selections": selection_records,
        }

    fixed_point = evaluate_component_operating_points(
        logits,
        targets,
        [fixed_threshold],
    )[0]
    return (
        {
            "protocol": "deterministic_two_fold_image_disjoint_cross_fit_v1",
            "fold_assignment": {
                name: fold for name, fold in zip(names, folds)
            },
            "fa_budgets_per_million_pixels": list(budgets),
            "matching": "hungarian_max_cardinality_min_centroid_distance",
            "strict_threshold": True,
            "centroid_radius": 3.0,
            "local_peak_note": (
                "strict support-distance<3 peak is a diagnostic and is not "
                "component Pd"
            ),
            "fixed_threshold_full_validation": asdict(fixed_point),
            "folds": fold_records,
        },
        threshold_lookup,
    )


def _target_operating_record(
    logits: np.ndarray,
    target: np.ndarray,
    *,
    sample_name: str,
    target_index: int,
    fixed_threshold: float,
    threshold_lookup: dict[int, dict[str, dict[str, object]]],
) -> dict[str, object]:
    fold = _operating_fold(sample_name)
    fixed = evaluate_target_operating_status(
        logits,
        target,
        target_index=target_index,
        threshold=fixed_threshold,
    )
    fixed_fa = {}
    for key, calibration in threshold_lookup[fold].items():
        status = evaluate_target_operating_status(
            logits,
            target,
            target_index=target_index,
            threshold=float(calibration["threshold"]),
        )
        fixed_fa[key] = {
            "status": asdict(status),
            "fa_budget_per_million_pixels": calibration[
                "fa_budget_per_million_pixels"
            ],
            "calibration_threshold": calibration["threshold"],
            "calibration_achieved_fa_per_million_pixels": calibration[
                "calibration_operating_point"
            ]["fa_per_million_pixels"],
            "calibration_pd": calibration["calibration_operating_point"]["pd"],
            "evaluation_achieved_fa_per_million_pixels": calibration[
                "evaluation_operating_point"
            ]["fa_per_million_pixels"],
            "evaluation_pd": calibration["evaluation_operating_point"]["pd"],
        }
    return {
        "evaluation_fold": fold,
        "fixed_threshold": asdict(fixed),
        "cross_fitted_fixed_fa": fixed_fa,
    }


def _final_patch_scale(
    scale_logits: torch.Tensor,
    background_flat_indices: np.ndarray,
    *,
    weight: torch.Tensor,
    padding,
    stride,
    dilation,
) -> tuple[torch.Tensor, int]:
    if scale_logits.shape[0] != 1:
        raise ValueError("final patch scale requires batch size one")
    patches = F.unfold(
        scale_logits,
        kernel_size=tuple(weight.shape[-2:]),
        padding=padding,
        stride=stride,
        dilation=dilation,
    )[0]
    indices = torch.as_tensor(
        background_flat_indices,
        dtype=torch.long,
        device=patches.device,
    )
    background = patches[:, indices].detach().cpu().numpy()
    global_values = patches.detach().cpu().numpy()
    _, scale = robust_channel_center_scale(
        background,
        global_values=global_values,
    )
    floor_active = int(np.sum(scale <= 1e-8 * (1.0 + 1e-12)))
    return (
        torch.as_tensor(scale, dtype=patches.dtype, device=patches.device),
        floor_active,
    )


def _fusion_conversion(
    evidence,
    geometry,
) -> dict[str, object]:
    if evidence["pred"].shape[0] != 1:
        raise ValueError("target-level fusion conversion requires batch size one")
    scale_logits = torch.cat(
        tuple(evidence["full_sides"].values()), dim=1
    )
    final_head = evidence["final_head"]
    weight = final_head["weight"]
    bias = final_head["bias"]
    patch_scale, floor_active = _final_patch_scale(
        scale_logits,
        geometry.target.background_flat_indices,
        weight=weight,
        padding=final_head["padding"],
        stride=final_head["stride"],
        dilation=final_head["dilation"],
    )
    target_weights = torch.as_tensor(
        geometry.target.occupancy,
        dtype=scale_logits.dtype,
        device=scale_logits.device,
    )
    control_weights = torch.zeros_like(target_weights)
    control_weights.reshape(-1)[
        torch.as_tensor(
            geometry.target.background_flat_indices,
            dtype=torch.long,
            device=scale_logits.device,
        )
    ] = 1.0
    result = factorize_final_fusion_margin(
        scale_logits,
        target_weights,
        control_weights,
        fusion_weight=weight,
        fusion_bias=bias,
        patch_scale=patch_scale,
        stride=final_head["stride"],
        padding=final_head["padding"],
        dilation=final_head["dilation"],
    )
    if not torch.allclose(
        result.reconstructed_logits,
        evidence["pred"],
        atol=1e-6,
        rtol=1e-5,
    ):
        raise RuntimeError("final unfold did not reconstruct native prediction")
    reconstruction_error = float(
        torch.abs(
            result.signed_margin
            if result.reconstructed_margin is None
            else result.reconstructed_margin - result.signed_margin
        )
        .detach()
        .cpu()
    )
    utilization = (
        None
        if result.utilization_cosine is None
        else float(result.utilization_cosine.detach().cpu())
    )
    reconstructed_margin = (
        None
        if result.reconstructed_margin is None
        else float(result.reconstructed_margin.detach().cpu())
    )
    return {
        "available": True,
        "reason": (
            "zero_mean_margin_availability"
            if result.utilization_cosine is None
            and float(result.available_contrast.detach().cpu()) == 0.0
            else "zero_head_sensitivity"
            if result.utilization_cosine is None
            else None
        ),
        "channels": int(result.patch_difference.numel()),
        "mean_margin_availability": float(
            result.available_contrast.detach().cpu()
        ),
        "normalized_mean_margin_availability": float(
            result.available_contrast.detach().cpu()
            / math.sqrt(result.patch_difference.numel())
        ),
        "head_sensitivity": float(result.head_sensitivity.detach().cpu()),
        "utilization_cosine": utilization,
        "mean_logit_margin": float(result.signed_margin.detach().cpu()),
        "reconstructed_margin": reconstructed_margin,
        "reconstruction_error": reconstruction_error,
        "absolute_scale_floor_active_channels": floor_active,
        "reparameterization_stable": floor_active == 0,
        "target_mean_logit": float(result.target_logit_mean.detach().cpu()),
        "background_mean_logit": float(result.control_logit_mean.detach().cpu()),
    }


def _contribution_margins(evidence, geometry) -> dict[str, object]:
    if evidence["pred"].shape[0] != 1:
        raise ValueError("target-level contribution margins require batch size one")
    occupancy = geometry.target.occupancy.reshape(-1)
    target_indices = np.flatnonzero(occupancy > 0)
    target_weights = occupancy[target_indices]
    target_weights = target_weights / target_weights.sum()
    background_indices = geometry.target.background_flat_indices
    margins = {}
    for stage, tensor in evidence["contributions"].items():
        values = tensor[0, 0].detach().cpu().numpy().reshape(-1)
        target_mean = float(np.sum(values[target_indices] * target_weights))
        background_mean = float(np.mean(values[background_indices]))
        margins[stage] = target_mean - background_mean
    final_values = evidence["pred"][0, 0].detach().cpu().numpy().reshape(-1)
    final_margin = float(
        np.sum(final_values[target_indices] * target_weights)
        - np.mean(final_values[background_indices])
    )
    summed = float(sum(margins.values()))
    if not math.isclose(summed, final_margin, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError("per-scale contribution margins do not sum to final")
    return {
        "available": True,
        "reason": None,
        "per_scale": margins,
        "sum": summed,
        "final_direct": final_margin,
        "positive_sum": float(sum(max(0.0, value) for value in margins.values())),
        "negative_sum": float(sum(min(0.0, value) for value in margins.values())),
        "has_sign_cancellation": bool(
            any(value > 0 for value in margins.values())
            and any(value < 0 for value in margins.values())
        ),
    }


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    summary = {}
    for outcome in ("no_response", "matched_control"):
        selected = [record for record in records if record["outcome"] == outcome]
        outcome_summary = {"targets": len(selected)}
        for stage in ("d0", "d1", "d2", "d3", "final"):
            values = [record["conversion"][stage] for record in selected]
            defined = [
                value
                for value in values
                if value["available"]
                and value["utilization_cosine"] is not None
            ]
            utilization = [float(value["utilization_cosine"]) for value in defined]
            margins = [
                float(value["mean_logit_margin"])
                for value in values
                if value["available"] and value["mean_logit_margin"] is not None
            ]
            outcome_summary[stage] = {
                "available": sum(value["available"] for value in values),
                "mean_margin_defined": len(margins),
                "utilization_defined": len(defined),
                "utilization_positive": sum(value > 0 for value in utilization),
                "utilization_negative": sum(value < 0 for value in utilization),
                "median_utilization": (
                    float(np.median(utilization)) if utilization else None
                ),
                "positive_signed_margin": sum(value > 0 for value in margins),
                "median_signed_margin": (
                    float(np.median(margins)) if margins else None
                ),
            }
        footprint_peak_margins = [
            float(record["final_scalar"]["target_peak_margin"])
            for record in selected
            if record["final_scalar"]["available"]
            and record["final_scalar"]["target_peak_margin"] is not None
        ]
        outcome_summary["target_footprint_peak_margin_defined"] = len(
            footprint_peak_margins
        )
        outcome_summary["target_footprint_peak_above_fixed_threshold"] = sum(
            value > 0 for value in footprint_peak_margins
        )
        cancellations = [
            bool(record["contribution_margins"]["has_sign_cancellation"])
            for record in selected
            if record["contribution_margins"]["available"]
            and record["contribution_margins"]["has_sign_cancellation"] is not None
        ]
        outcome_summary["contribution_margin_defined"] = len(cancellations)
        outcome_summary["contribution_sign_cancellation"] = sum(cancellations)
        for policy in ("geometry", "context_matched"):
            policy_values = [record["controls"][policy] for record in selected]
            outcome_summary["%s_controls" % policy] = {
                "available": sum(value["available"] for value in policy_values),
                "d0_distinct": sum(
                    value["survival"]["d0"]["state"] == "distinct"
                    for value in policy_values
                    if value["available"]
                    and value["survival"]["d0"]["available"]
                ),
            }
        fixed_statuses = [
            record["operating_point"]["fixed_threshold"] for record in selected
        ]
        outcome_summary["fixed_threshold_component_detected"] = sum(
            bool(status["matched"]) for status in fixed_statuses
        )
        outcome_summary["fixed_threshold_local_peak_above"] = sum(
            bool(status["neighborhood_peak_above_threshold"])
            for status in fixed_statuses
        )
        budget_keys = sorted(
            {
                key
                for record in selected
                for key in record["operating_point"][
                    "cross_fitted_fixed_fa"
                ]
            },
            key=float,
        )
        outcome_summary["cross_fitted_fixed_fa"] = {}
        for key in budget_keys:
            statuses = [
                record["operating_point"]["cross_fitted_fixed_fa"][key][
                    "status"
                ]
                for record in selected
            ]
            outcome_summary["cross_fitted_fixed_fa"][key] = {
                "targets": len(statuses),
                "component_detected": sum(
                    bool(status["matched"]) for status in statuses
                ),
                "local_peak_above_threshold": sum(
                    bool(status["neighborhood_peak_above_threshold"])
                    for status in statuses
                ),
                "median_local_peak_margin": (
                    float(
                        np.median(
                            [status["neighborhood_margin"] for status in statuses]
                        )
                    )
                    if statuses
                    else None
                ),
            }
        summary[outcome] = outcome_summary
    return summary


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold_probability < 1.0:
        raise ValueError("threshold probability must lie strictly in (0,1)")
    if args.batch_size != 1 or args.num_workers < 0:
        raise ValueError(
            "decision-conversion audit requires batch size one for numerical "
            "identity across cohort, operating-point, and evidence passes"
        )
    if args.required_controls < 1 or args.max_candidate_controls < args.required_controls:
        raise ValueError("candidate controls must cover required controls")
    if args.context_control_count < args.required_controls:
        raise ValueError("context controls must cover required projected controls")
    if args.max_candidate_controls < args.context_control_count:
        raise ValueError("candidate controls must cover context controls")
    if not math.isfinite(args.context_ring_width) or args.context_ring_width <= 0:
        raise ValueError("context ring width must be finite and positive")
    if args.matched_controls_per_miss < 0:
        raise ValueError("matched-controls-per-miss must be non-negative")
    if not args.fixed_fa_budgets or any(
        not math.isfinite(value) or value < 0 for value in args.fixed_fa_budgets
    ):
        raise ValueError("fixed FA budgets must be finite and non-negative")
    fixed_fa_budgets = tuple(sorted(set(args.fixed_fa_budgets)))

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
    if stored_args.get("mode") != "train" or stored_args.get("model_type") != "mshnet":
        raise RuntimeError("source run must be a plain MSHNet training run")
    dataset = IRSTD_Dataset(Namespace(**stored_args), mode="val")
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
    operating_logits, operating_targets = _collect_validation_predictions(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    operating_audit, operating_threshold_lookup = (
        _build_cross_fitted_operating_audit(
            operating_logits,
            operating_targets,
            dataset.names,
            fixed_threshold=threshold_logit,
            budgets=fixed_fa_budgets,
        )
    )
    matched_controls = _select_paired_matched_controls(
        no_response,
        matched,
        controls_per_miss=args.matched_controls_per_miss,
    )
    selected = no_response + matched_controls
    selected_by_sample = defaultdict(list)
    for item in selected:
        selected_by_sample[int(item["sample_index"])].append(item)
    sample_indices = sorted(selected_by_sample)
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0 and bool(sample_indices),
    )

    records = []
    with torch.no_grad():
        for subset_index, (images, targets) in enumerate(
            tqdm(loader, desc="Decision-conversion audit")
        ):
            sample_index = sample_indices[subset_index]
            sample_name = dataset.names[sample_index]
            evidence = forward_mshnet_stage_evidence(
                model,
                images.to(device, non_blocking=True),
                detach=True,
            )
            target = targets.numpy()[0, 0] > 0.5
            if not np.array_equal(target, operating_targets[sample_index]):
                raise RuntimeError("selected target changed across audit passes")
            evidence_logits = evidence["pred"][0, 0].detach().cpu().numpy()
            if not np.allclose(
                evidence_logits,
                operating_logits[sample_index],
                atol=1e-6,
                rtol=1e-6,
            ):
                raise RuntimeError("read-only evidence view changed native logits")
            grayscale = _normalized_rgb_to_context_luminance(images)
            labels = measure.label(target, connectivity=2)
            regions = tuple(measure.regionprops(labels))

            for cohort in selected_by_sample[sample_index]:
                target_index = int(cohort["target_index"])
                if target_index >= len(regions):
                    raise RuntimeError("target component indexing drifted")
                component = labels == regions[target_index].label
                translation_controls = build_translation_control_set(
                    component,
                    target,
                    sample_key="%s:%d" % (sample_name, target_index),
                    max_candidate_controls=args.max_candidate_controls,
                )
                context_selection = select_context_matched_controls(
                    grayscale,
                    translation_controls,
                    protection_mask=translation_controls.guarded_target_mask,
                    context_ring_width=args.context_ring_width,
                    num_controls=args.context_control_count,
                )
                control_sets = {
                    "geometry": translation_controls,
                    "context_matched": context_selection.control_set,
                }
                geometry_by_policy_shape = {
                    "geometry": {},
                    "context_matched": {},
                }

                def geometry_for(tensor, policy="geometry"):
                    if policy not in control_sets:
                        raise ValueError("unknown control policy")
                    shape = tuple(int(value) for value in tensor.shape[-2:])
                    cache = geometry_by_policy_shape[policy]
                    if shape not in cache:
                        source_controls = control_sets[policy]
                        cache[shape] = (
                            None
                            if source_controls is None
                            else project_geometry_controls(
                                source_controls,
                                shape,
                                required_controls=args.required_controls,
                            )
                        )
                    return cache[shape]

                survival_by_policy = {}
                for policy in ("geometry", "context_matched"):
                    stage_survival = {}
                    for stage in ("d0", "d1", "d2", "d3"):
                        feature_tensor = evidence["path"][stage]
                        stage_survival[stage] = evaluate_feature_survival(
                            feature_tensor[0].detach().cpu().numpy(),
                            geometry_for(feature_tensor, policy),
                        ).as_dict()
                    stage_survival["final"] = evaluate_feature_survival(
                        evidence["pred"][0, 0].detach().cpu().numpy()[None],
                        geometry_for(evidence["pred"], policy),
                        scalar_threshold=threshold_logit,
                    ).as_dict()
                    policy_available = stage_survival["d0"]["available"]
                    if policy == "context_matched":
                        policy_available = bool(
                            context_selection.available and policy_available
                        )
                        reason = (
                            context_selection.reason
                            if not context_selection.available
                            else stage_survival["d0"]["reason"]
                        )
                    else:
                        reason = stage_survival["d0"]["reason"]
                    survival_by_policy[policy] = {
                        "available": policy_available,
                        "reason": reason,
                        "survival": stage_survival,
                    }
                survival_by_policy["context_matched"][
                    "selection"
                ] = _context_selection_record(context_selection)

                conversion = {}
                for index in range(4):
                    stage = "d%d" % index
                    mask_stage = "mask%d" % index
                    feature_tensor = evidence["path"][stage]
                    geometry = geometry_for(feature_tensor, "geometry")
                    head = evidence["side_heads"][mask_stage]
                    direct = F.conv2d(
                        feature_tensor,
                        head["weight"],
                        head["bias"],
                    )
                    if not torch.allclose(
                        direct,
                        evidence["native_sides"][mask_stage],
                        atol=1e-6,
                        rtol=1e-5,
                    ):
                        raise RuntimeError("native side head reconstruction failed")
                    conversion[stage] = evaluate_linear_head_conversion(
                        feature_tensor[0].detach().cpu().numpy(),
                        geometry,
                        head_weight=head["weight"],
                        head_bias=head["bias"],
                        scalar_threshold=threshold_logit,
                    ).as_dict()

                full_geometry = geometry_for(evidence["pred"], "geometry")
                if full_geometry is None:
                    conversion["final"] = _empty_conversion(
                        "insufficient_geometry_controls"
                    )
                    contribution_margins = _empty_contribution_margins(
                        "insufficient_geometry_controls"
                    )
                else:
                    conversion["final"] = _fusion_conversion(
                        evidence, full_geometry
                    )
                    contribution_margins = _contribution_margins(
                        evidence, full_geometry
                    )
                final_scalar = evaluate_feature_survival(
                    evidence["pred"][0, 0].detach().cpu().numpy()[None],
                    full_geometry,
                    scalar_threshold=threshold_logit,
                ).as_dict()
                operating_point = _target_operating_record(
                    operating_logits[sample_index],
                    operating_targets[sample_index],
                    sample_name=sample_name,
                    target_index=target_index,
                    fixed_threshold=threshold_logit,
                    threshold_lookup=operating_threshold_lookup,
                )
                fixed_detected = operating_point["fixed_threshold"]["matched"]
                if (cohort["outcome"] == "no_response" and fixed_detected) or (
                    cohort["outcome"] == "matched_control" and not fixed_detected
                ):
                    raise RuntimeError("fixed-threshold cohort membership drifted")
                record = dict(cohort)
                record.update(
                    conversion=conversion,
                    controls=survival_by_policy,
                    final_scalar=final_scalar,
                    contribution_margins=contribution_margins,
                    operating_point=operating_point,
                    failure_class="mixed_or_undefined",
                )
                records.append(record)

    output = {
        "schema": "mshnet_decision_conversion_audit_v2",
        "exploratory_only": True,
        "factorization_note": (
            "utilization is undefined when availability or sensitivity is zero; "
            "no epsilon is inserted into the cosine denominator"
        ),
        "source_run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_dir": stored_args.get("dataset_dir"),
        "split": "internal_validation",
        "val_split_sha256": dataset.split_sha256,
        "threshold_probability": args.threshold_probability,
        "threshold_logit": threshold_logit,
        "inference_batch_size": args.batch_size,
        "required_geometry_controls": args.required_controls,
        "context_control_count": args.context_control_count,
        "context_ring_width": args.context_ring_width,
        "num_validation_images": len(dataset),
        "num_no_response_targets": len(no_response),
        "num_selected_matched_controls": len(matched_controls),
        "operating_point_audit": operating_audit,
        "summary": _summarize(records),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("wrote %s" % output_path)


if __name__ == "__main__":
    main()
