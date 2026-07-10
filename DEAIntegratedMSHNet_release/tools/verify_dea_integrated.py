#!/usr/bin/env python3
"""Verify checkpoint equivalence, gradients, complexity, and model hygiene.

Example:
    python tools/verify_dea_integrated.py \
        --checkpoint weight/NUDT_MSHNet.tar \
        --device cuda --height 256 --width 256

The script exits non-zero if any hard structural gate fails.
"""

import argparse
import json
import os
import sys
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOSITORY_ROOT = os.path.dirname(RELEASE_ROOT)
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from model.MSHNet import MSHNet
from DEAIntegratedMSHNet_release.model.dea_integrated_mshnet import (
    DEAIntegratedMSHNet,
    count_trainable_parameters,
)


def load_torch_file(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(weight_object) -> Mapping[str, torch.Tensor]:
    return DEAIntegratedMSHNet.extract_state_dict(weight_object)


def strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_baseline(model, state_dict):
    clean = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(clean, strict=False)
    # Current DEA-lite MSHNet checkpoints may include the decidability head;
    # original MSHNet checkpoints may not.  Neither affects baseline output.
    bad_missing = [key for key in missing if not key.startswith("decidability_head.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            "baseline checkpoint mismatch: bad_missing=%s unexpected=%s"
            % (bad_missing, unexpected)
        )
    return missing, unexpected


def analytical_route_macs(height: int, width: int, route_channels: int) -> int:
    """Convolution MACs added by the four route cells (elementwise ops excluded)."""
    stage_specs = [
        # (downsample factor, encoder channels, decoder channels, output channels)
        (1, 16, 32, 16),
        (2, 32, 64, 32),
        (4, 64, 128, 64),
        (8, 128, 256, 128),
    ]
    total = 0
    r = int(route_channels)
    for factor, encoder_channels, decoder_channels, output_channels in stage_specs:
        h = height // factor
        w = width // factor
        per_location = r * (
            encoder_channels
            + decoder_channels
            + 6          # 2r -> 3 route logits
            + 18         # depthwise 3x3 over 2r channels
            + 2 * output_channels
        )
        total += h * w * per_location
    return total


def aggregate_cell_gradients(model) -> Dict[str, float]:
    result = {}
    for scale in range(4):
        cell = getattr(model, "dea_cell_%d" % scale)
        result[str(scale)] = float(sum(
            parameter.grad.detach().abs().sum().item()
            for parameter in cell.parameters()
            if parameter.grad is not None
        ))
    return result


def named_cell_gradients(model) -> Dict[str, float]:
    result = {}
    for scale in range(4):
        cell = getattr(model, "dea_cell_%d" % scale)
        for name, parameter in cell.named_parameters():
            key = "%d.%s" % (scale, name)
            if parameter.grad is None:
                result[key] = 0.0
            else:
                result[key] = float(parameter.grad.detach().abs().sum().item())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--route-channels", type=int, default=16)
    parser.add_argument("--decomposition-atol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    checkpoint = load_torch_file(args.checkpoint)
    state_dict = extract_state_dict(checkpoint)

    baseline = MSHNet(3)
    load_baseline(baseline, state_dict)
    integrated = DEAIntegratedMSHNet(3, route_channels=args.route_channels)
    missing, unexpected = integrated.load_mshnet_state_dict(state_dict)

    baseline.to(device).eval()
    integrated.to(device).eval()
    x = torch.randn(1, 3, args.height, args.width, device=device)

    with torch.no_grad():
        baseline_masks, baseline_pred = baseline(x, True)
        integrated_output = integrated(x, True, return_dict=True)

    mask_errors = [
        float((left - right).abs().max().item())
        for left, right in zip(baseline_masks, integrated_output["masks"])
    ]
    prediction_error = float(
        (baseline_pred - integrated_output["pred"]).abs().max().item()
    )

    all_uncertain = all(
        bool(torch.all(route["winner"] == 2).item())
        for route in integrated_output["routes"]
    )
    target_nonzero = sum(
        int(torch.count_nonzero(route["target_gate"]).item())
        for route in integrated_output["routes"]
    )
    clutter_nonzero = sum(
        int(torch.count_nonzero(route["clutter_gate"]).item())
        for route in integrated_output["routes"]
    )

    direct_final = F.conv2d(
        integrated_output["scale_logits"],
        integrated.final.weight,
        integrated.final.bias,
        stride=integrated.final.stride,
        padding=integrated.final.padding,
        dilation=integrated.final.dilation,
    )
    decomposed_final = integrated.final.baseline_from_contributions(
        integrated_output["scale_fusion"]["contributions"]
    )
    decomposition_error = float(
        (decomposed_final - direct_final).abs().max().item()
    )

    # Gradient gate: use a fresh train-mode pass.  The target/clutter forward
    # residual remains zero, but the straight-through route produces gradients.
    integrated.train()
    integrated.zero_grad(set_to_none=True)
    train_x = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device
    )
    train_output = integrated(train_x, True, return_dict=True)
    loss = train_output["pred"].square().mean()
    for mask in train_output["masks"]:
        loss = loss + 0.1 * mask.square().mean()
    loss.backward()
    cell_gradients = aggregate_cell_gradients(integrated)
    parameter_gradients = named_cell_gradients(integrated)

    forbidden_tokens = (
        "topology",
        "prototype",
        "component_graph",
        "relation_graph",
        "bridge",
    )
    forbidden_modules = [
        name for name, _ in integrated.named_modules()
        if any(token in name.lower() for token in forbidden_tokens)
    ]

    baseline_parameters = count_trainable_parameters(baseline)
    integrated_parameters = count_trainable_parameters(integrated)
    explicit_route_parameters = sum(
        parameter.numel()
        for name, parameter in integrated.named_parameters()
        if name.startswith("dea_cell_")
    )
    route_macs = analytical_route_macs(
        args.height, args.width, args.route_channels
    )
    scale_decomposition_macs = args.height * args.width * 4 * 3 * 3

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "input_shape": [1, 3, args.height, args.width],
        "checkpoint_missing_allowed": list(missing),
        "checkpoint_unexpected_allowed": list(unexpected),
        "mask_max_abs_errors": mask_errors,
        "prediction_max_abs_error": prediction_error,
        "final_decomposition_max_abs_error": decomposition_error,
        "all_routes_uncertain_at_initialization": all_uncertain,
        "initial_target_gate_nonzero_count": target_nonzero,
        "initial_clutter_gate_nonzero_count": clutter_nonzero,
        "routing_cell_gradient_l1": cell_gradients,
        "routing_parameter_gradient_l1": parameter_gradients,
        "baseline_trainable_parameters": baseline_parameters,
        "integrated_trainable_parameters": integrated_parameters,
        "net_parameter_difference_vs_current_repo_mshnet": (
            integrated_parameters - baseline_parameters
        ),
        "explicit_route_parameters": explicit_route_parameters,
        "route_conv_macs": route_macs,
        "route_conv_gmacs": route_macs / 1e9,
        "scale_decomposition_macs": scale_decomposition_macs,
        "scale_decomposition_gmacs": scale_decomposition_macs / 1e9,
        "total_added_conv_macs": route_macs + scale_decomposition_macs,
        "total_added_conv_gmacs": (
            route_macs + scale_decomposition_macs
        ) / 1e9,
        "forbidden_modules": forbidden_modules,
        "legacy_decidability_head_present": hasattr(integrated, "decidability_head"),
    }

    failures = []
    if any(error != 0.0 for error in mask_errors + [prediction_error]):
        failures.append("baseline embedding is not bitwise exact")
    if decomposition_error >= args.decomposition_atol:
        failures.append("algebraic decomposition error >= decomposition_atol")
    if not all_uncertain or target_nonzero != 0 or clutter_nonzero != 0:
        failures.append("initial route is not strict uncertain identity")
    if not all(value > 0.0 for value in cell_gradients.values()):
        failures.append("at least one routing cell has zero aggregate gradient")
    if not all(value > 0.0 for value in parameter_gradients.values()):
        failures.append("at least one routing parameter has zero gradient")
    if forbidden_modules:
        failures.append("forbidden discontinued modules are present")
    if hasattr(integrated, "decidability_head"):
        failures.append("legacy DEA-lite decidability_head is present")
    if explicit_route_parameters != 20_988 and args.route_channels == 16:
        failures.append("unexpected route parameter count")

    report["passed"] = not failures
    report["failures"] = failures
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
