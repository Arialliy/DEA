#!/usr/bin/env python3
"""Read-only paired low-component-FA evaluation for MSHNet checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.MSHNet import MSHNet  # noqa: E402
from utils.cross_fitted_low_fa import BUDGETS, MATCHERS, cross_fit_job  # noqa: E402
from utils.data import IRSTD_Dataset  # noqa: E402
from utils.nested_component_grid import (  # noqa: E402
    build_nested_quantile_probability_grids,
    evaluate_nested_component_grids,
)
from utils.target_identity import build_stable_target_set  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--train-split-file", required=True)
    parser.add_argument("--val-split-file", required=True)
    parser.add_argument("--test-split-file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="repeat as LABEL=PATH",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint must use LABEL=PATH")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if not label or label in result or not path.is_file():
            raise ValueError(f"invalid checkpoint declaration: {value}")
        result[label] = path
    return result


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    state = value.get("net", value)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint lacks a state dict: {path}")
    normalized = {
        (key[7:] if key.startswith("module.") else key): tensor
        for key, tensor in state.items()
    }
    metadata = {
        "epoch": int(value.get("epoch", -1)),
        "method_meta": value.get("method_meta", {}),
    }
    return normalized, metadata


def dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=str(Path(args.dataset_dir).resolve()),
        train_split_file=str(Path(args.train_split_file).resolve()),
        val_split_file=str(Path(args.val_split_file).resolve()),
        test_split_file=args.test_split_file,
        val_fraction=0.2,
        split_seed=args.seed,
        seed=args.seed,
        base_size=256,
        crop_size=256,
        return_instance_labels=False,
    )


def infer(
    checkpoint: Path,
    dataset: IRSTD_Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], dict]:
    state, metadata = checkpoint_state(checkpoint)
    model = MSHNet(3)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    scores = []
    targets = []
    with torch.inference_mode():
        for images, labels in loader:
            _, logits = model(images.to(device, non_blocking=True), True)
            scores.extend(logits[:, 0].float().cpu().numpy())
            targets.extend((labels[:, 0] > 0.5).numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(scores) != len(dataset) or len(targets) != len(dataset):
        raise RuntimeError("inference sample count drifted")
    return tuple(scores), tuple(targets), metadata


def pooled_crossfit(image_rows: list[dict], matcher: str, budget: int) -> dict:
    selected = [
        row
        for row in image_rows
        if row["matcher"] == matcher
        and int(row["nominal_budget_fa_per_mpix"]) == budget
    ]
    if not selected:
        raise RuntimeError("cross-fit group is empty")
    aggregate = selected[0]["dataset_seed_aggregate"]
    if any(row["dataset_seed_aggregate"] != aggregate for row in selected):
        raise RuntimeError("cross-fit aggregate drifted")
    folds = {}
    for row in selected:
        fold = str(int(row["evaluation_fold"]))
        held_out = row["held_out_fold_aggregate"]
        if fold in folds and folds[fold] != held_out:
            raise RuntimeError("held-out fold aggregate drifted")
        folds[fold] = held_out
    return {
        **aggregate,
        "all_held_out_folds_feasible": all(
            bool(value["budget_feasible_zero_overshoot"])
            for value in folds.values()
        ),
        "held_out_folds": folds,
    }


def evaluate_variant(
    label: str,
    checkpoint: Path,
    scores: tuple[np.ndarray, ...],
    targets: tuple[np.ndarray, ...],
    names: tuple[str, ...],
    registry: dict,
    *,
    dataset: str,
    seed: int,
    metadata: dict,
) -> dict:
    nested = evaluate_nested_component_grids(
        scores,
        targets,
        BUDGETS,
        fixed_thresholds=(0.0,),
    )
    oracle = {}
    for matcher in MATCHERS:
        oracle[matcher] = {
            str(selection.budget_fa_per_million_pixels): {
                **asdict(selection.operating_point),
                "budget_fa_per_million_pixels": selection.budget_fa_per_million_pixels,
            }
            for selection in nested.matcher(matcher).level("Q2").selections
        }
    quantiles = build_nested_quantile_probability_grids()[-1].probabilities
    target_rows, image_rows, calibration = cross_fit_job(
        scores,
        targets,
        names,
        dataset=dataset,
        seed=seed,
        registry=registry,
        checkpoint={
            "label": label,
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "epoch": metadata["epoch"],
        },
        tail_quantiles=quantiles,
    )
    crossfit = {
        matcher: {
            str(budget): pooled_crossfit(image_rows, matcher, budget)
            for budget in BUDGETS
        }
        for matcher in MATCHERS
    }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "epoch": metadata["epoch"],
        "method": metadata["method_meta"].get("method", ""),
        "oracle_q2": oracle,
        "crossfit_q2": crossfit,
        "crossfit_target_rows": len(target_rows),
        "crossfit_calibration_records": len(calibration),
    }


def main() -> int:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = IRSTD_Dataset(dataset_args(args), mode="val")
    names = tuple(dataset.names)
    registry = {}
    for index in range(len(dataset)):
        _, target = dataset[index]
        target_array = (target[0].numpy() > 0.5)
        registry[names[index]] = build_stable_target_set(
            target_array,
            dataset=args.dataset,
            image_name=names[index],
            connectivity=2,
        )
    device = torch.device(args.device)
    variants = {}
    for label, checkpoint in checkpoints.items():
        scores, targets, metadata = infer(
            checkpoint,
            dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        variants[label] = evaluate_variant(
            label,
            checkpoint,
            scores,
            targets,
            names,
            registry,
            dataset=args.dataset,
            seed=args.seed,
            metadata=metadata,
        )
    result = {
        "schema_version": "dea.crwd.paired_frontier.v1",
        "scope": "development-only; same-set Q2 oracle plus two-fold cross-fit",
        "dataset": args.dataset,
        "seed": args.seed,
        "image_count": len(dataset),
        "target_count": sum(len(value.targets) for value in registry.values()),
        "validation_split_sha256": dataset.split_sha256,
        "variants": variants,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote paired frontier: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
