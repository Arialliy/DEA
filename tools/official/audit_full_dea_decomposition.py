#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.utils.data as Data
from skimage import measure

from model.full_dea_loss import (
    build_component_hard_clutter_label,
    build_hard_clutter_label,
)
from model.full_dea_mshnet import FullDEAMSHNet
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA, mIoU


def load_torch_file(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        if "net" in obj:
            return obj["net"]
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise SystemExit("unsupported weight/checkpoint format")


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def finite_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: finite_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class OutputMetric:
    def __init__(self, base_size: int):
        self.miou = mIoU(1)
        self.pdfa = PD_FA(1, 10, base_size)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.miou.update(pred, target)
        self.pdfa.update(pred, target)

    def get(self, image_count: int) -> dict[str, float]:
        _, iou = self.miou.get()
        fa, pd = self.pdfa.get(image_count)
        return {
            "IoU": float(iou),
            "PD": float(pd[0]),
            "FA": float(fa[0] * 1_000_000.0),
        }


class MaskedAccumulator:
    def __init__(self):
        self.values: dict[str, float] = {}
        self.denoms: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    def add_masked(self, name: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        numerator = float((value.detach() * mask).sum().cpu())
        denominator = float(mask.sum().cpu())
        self.values[name] = self.values.get(name, 0.0) + numerator
        self.denoms[name] = self.denoms.get(name, 0.0) + denominator

    def add_mean(self, name: str, value: torch.Tensor) -> None:
        self.values[name] = self.values.get(name, 0.0) + float(value.detach().mean().cpu())
        self.counts[name] = self.counts.get(name, 0.0) + 1.0

    def get(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for name, value in self.values.items():
            if name in self.denoms:
                denom = self.denoms[name]
                out[name] = value / denom if denom > 0 else None
            else:
                count = self.counts.get(name, 0.0)
                out[name] = value / count if count > 0 else None
        return out


def count_components(mask: torch.Tensor) -> int:
    total = 0
    arr = mask.detach().cpu().numpy()
    for i in range(arr.shape[0]):
        total += int(measure.label(arr[i, 0] > 0.5, connectivity=2).max())
    return total


def build_loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=args.dataset_dir,
        crop_size=args.base_size,
        base_size=args.base_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--full-dea-version",
        choices=["v2", "v3", "v4", "v5"],
        default="v3",
    )
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tau-base", type=float, default=0.45)
    parser.add_argument("--tau-target", type=float, default=0.45)
    parser.add_argument("--tau-scale", type=float, default=0.45)
    parser.add_argument("--safe-kernel", type=int, default=15)
    parser.add_argument("--protect-kernel", type=int, default=9)
    parser.add_argument("--hard-min-area", type=int, default=1)
    parser.add_argument("--hard-max-area", type=int, default=256)
    parser.add_argument("--max-hard-bg-ratio", type=float, default=0.003)
    parser.add_argument("--baseline-iou", type=float, default=-1.0)
    parser.add_argument("--baseline-pd", type=float, default=-1.0)
    parser.add_argument("--baseline-fa", type=float, default=-1.0)
    parser.add_argument("--pd-tolerance", type=float, default=0.005)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = IRSTD_Dataset(build_loader_args(args), mode="val")
    loader = Data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = FullDEAMSHNet(3, full_dea_version=args.full_dea_version)
    state = strip_module_prefix(extract_state_dict(load_torch_file(args.weight)))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    output_keys = {
        "z_base": "z_base",
        "z_target": "z_target",
        "z_final": "z_final",
    }
    metrics = {name: OutputMetric(args.base_size) for name in output_keys}
    accum = MaskedAccumulator()
    image_count = 0
    hard_component_count = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float()
            out = model(data, warm_flag=True, return_dict=True)
            full = out["full_dea"]

            for name, key in output_keys.items():
                metrics[name].update(full[key], target)

            if args.full_dea_version in ("v3", "v4", "v5"):
                hard_clutter, regions = build_component_hard_clutter_label(
                    full_dea_out=full,
                    target=target,
                    tau_base=args.tau_base,
                    tau_target=args.tau_target,
                    tau_scale=args.tau_scale,
                    protect_kernel=args.protect_kernel,
                    safe_kernel=args.safe_kernel,
                    min_area=args.hard_min_area,
                    max_area=args.hard_max_area,
                    max_hard_bg_ratio=args.max_hard_bg_ratio,
                )
                target_core = regions["target_core"]
                target_protect = regions["target_protect"]
            else:
                hard_clutter, safe_bg = build_hard_clutter_label(
                    full_dea_out=full,
                    target=target,
                    tau_base=args.tau_base,
                    tau_target=args.tau_target,
                    tau_scale=args.tau_scale,
                    safe_kernel=args.safe_kernel,
                    topk_ratio=0.001,
                    topk_min_score=args.tau_base,
                    max_hard_bg_ratio=args.max_hard_bg_ratio,
                )
                target_core = target
                target_protect = 1.0 - safe_bg

            hard_component_count += count_components(hard_clutter)
            delta_target_base = full["z_target"] - full["z_base"]
            delta_final_base = full["z_final"] - full["z_base"]
            accum.add_masked("delta_target_base_on_gt", delta_target_base, target_core)
            accum.add_masked("delta_final_base_on_gt", delta_final_base, target_core)
            accum.add_masked(
                "delta_final_base_on_target_protect",
                delta_final_base,
                target_protect,
            )
            accum.add_masked(
                "delta_final_base_on_hard_clutter",
                delta_final_base,
                hard_clutter,
            )
            accum.add_mean("hard_clutter_pixel_ratio", hard_clutter.mean())

            if "protect_prob" in full:
                accum.add_masked("protect_on_gt", full["protect_prob"], target_core)
                accum.add_masked(
                    "protect_on_target_protect",
                    full["protect_prob"],
                    target_protect,
                )
                accum.add_masked(
                    "protect_on_hard_clutter",
                    full["protect_prob"],
                    hard_clutter,
                )
            if "target_boost" in full:
                accum.add_masked("target_boost_on_gt", full["target_boost"], target_core)
                accum.add_masked(
                    "target_boost_on_hard_clutter",
                    full["target_boost"],
                    hard_clutter,
                )
            if "suppression_gate" in full:
                accum.add_masked(
                    "suppression_on_gt",
                    full["suppression_gate"],
                    target_core,
                )
                accum.add_masked(
                    "suppression_on_target_protect",
                    full["suppression_gate"],
                    target_protect,
                )
                accum.add_masked(
                    "suppression_on_hard_clutter",
                    full["suppression_gate"],
                    hard_clutter,
                )
            if "topology_prior" in full:
                accum.add_masked(
                    "topology_prior_on_gt",
                    full["topology_prior"],
                    target_core,
                )
                accum.add_masked(
                    "topology_prior_off_gt",
                    full["topology_prior"],
                    1.0 - target_core,
                )
            if "bridge_gate" in full:
                accum.add_masked(
                    "bridge_gate_on_gt",
                    full["bridge_gate"],
                    target_core,
                )
                accum.add_masked(
                    "bridge_gate_off_gt",
                    full["bridge_gate"],
                    1.0 - target_core,
                )
            if "bridge_delta" in full:
                accum.add_masked(
                    "bridge_delta_on_gt",
                    full["bridge_delta"],
                    target_core,
                )
                accum.add_masked(
                    "bridge_delta_off_gt",
                    full["bridge_delta"],
                    1.0 - target_core,
                )
            if "endpoint_target_prior" in full:
                accum.add_masked(
                    "endpoint_target_prior_on_gt",
                    full["endpoint_target_prior"],
                    target_core,
                )
                accum.add_masked(
                    "endpoint_target_prior_off_gt",
                    full["endpoint_target_prior"],
                    1.0 - target_core,
                )
            if "relation_reconnect_map" in full:
                accum.add_masked(
                    "relation_reconnect_on_gt",
                    full["relation_reconnect_map"],
                    target_core,
                )
                accum.add_masked(
                    "relation_reconnect_off_gt",
                    full["relation_reconnect_map"],
                    1.0 - target_core,
                )
            if "relation_suppress_map" in full:
                accum.add_masked(
                    "relation_suppress_on_gt",
                    full["relation_suppress_map"],
                    target_core,
                )
                accum.add_masked(
                    "relation_suppress_on_hard_clutter",
                    full["relation_suppress_map"],
                    hard_clutter,
                )
                accum.add_masked(
                    "relation_suppress_off_gt",
                    full["relation_suppress_map"],
                    1.0 - target_core,
                )
            if "relation_probabilities" in full:
                accum.add_mean(
                    "relation_reconnect_probability",
                    full["relation_probabilities"][:, 0:1],
                )
                accum.add_mean(
                    "relation_suppress_probability",
                    full["relation_probabilities"][:, 1:3].sum(
                        dim=1,
                        keepdim=True,
                    ),
                )
                accum.add_mean(
                    "relation_identity_probability",
                    full["relation_probabilities"][:, 3:4],
                )

            image_count += int(data.shape[0])

    metric_result = {
        name: metric.get(image_count)
        for name, metric in metrics.items()
    }
    diagnostics = accum.get()
    diagnostics["hard_clutter_component_count"] = hard_component_count
    diagnostics["images"] = image_count

    conditions: dict[str, bool | None] = {
        "z_target_pd_ge_z_base_pd": (
            metric_result["z_target"]["PD"] >= metric_result["z_base"]["PD"]
        ),
        "z_final_fa_lt_z_base_fa": (
            metric_result["z_final"]["FA"] < metric_result["z_base"]["FA"]
        ),
        "z_final_pd_ge_z_base_pd_minus_tol": (
            metric_result["z_final"]["PD"]
            >= metric_result["z_base"]["PD"] - args.pd_tolerance
        ),
    }
    if args.baseline_iou >= 0:
        conditions["z_base_iou_close_reference"] = (
            abs(metric_result["z_base"]["IoU"] - args.baseline_iou) <= 0.001
        )
    if args.baseline_pd >= 0:
        conditions["z_base_pd_close_reference"] = (
            abs(metric_result["z_base"]["PD"] - args.baseline_pd) <= args.pd_tolerance
        )
        conditions["z_final_pd_ge_reference_minus_tol"] = (
            metric_result["z_final"]["PD"] >= args.baseline_pd - args.pd_tolerance
        )
    if args.baseline_fa >= 0:
        conditions["z_final_fa_lt_reference"] = (
            metric_result["z_final"]["FA"] < args.baseline_fa
        )
    if (
        diagnostics.get("protect_on_gt") is not None
        and diagnostics.get("protect_on_hard_clutter") is not None
    ):
        conditions["protect_gt_gt_hard_clutter"] = (
            diagnostics["protect_on_gt"] > diagnostics["protect_on_hard_clutter"]
        )
    if (
        diagnostics.get("suppression_on_hard_clutter") is not None
        and diagnostics.get("suppression_on_gt") is not None
    ):
        conditions["suppression_hard_clutter_gt_gt"] = (
            diagnostics["suppression_on_hard_clutter"]
            > diagnostics["suppression_on_gt"]
        )
    if diagnostics.get("delta_final_base_on_gt") is not None:
        conditions["mean_final_minus_base_on_gt_nonnegative"] = (
            diagnostics["delta_final_base_on_gt"] >= -1e-6
        )
    if diagnostics.get("delta_final_base_on_hard_clutter") is not None:
        conditions["mean_final_minus_base_on_hard_clutter_negative"] = (
            diagnostics["delta_final_base_on_hard_clutter"] < 0.0
        )

    method_names = {
        "v2": "FullDEA-v2",
        "v3": "FullDEA-v3-TPS",
        "v4": "FullDEA-v4-CRR",
        "v5": "FullDEA-v5-CRR-HT",
    }
    result = {
        "stage": "P1.5_FULL_DEA_DECOMPOSITION_AUDIT",
        "method": method_names[args.full_dea_version],
        "dataset": Path(args.dataset_dir).name,
        "weight": str(Path(args.weight).expanduser().resolve()),
        "full_dea_version": args.full_dea_version,
        "metrics": metric_result,
        "diagnostics": diagnostics,
        "mechanism_conditions": conditions,
        "mechanism_gate_pass": all(v is True for v in conditions.values()),
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(finite_json_value(result), allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(finite_json_value(result), allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
