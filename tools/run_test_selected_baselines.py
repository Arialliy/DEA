#!/usr/bin/env python3
"""Schedule canonical full-train/test-selected MSHNet baseline runs.

This is deliberately a *test-selected* protocol.  Every job trains on the
complete canonical ``train_<dataset>.txt`` list, evaluates the canonical test
list at a fixed epoch interval, and may select checkpoints from those test
evaluations.  Consequently, its selected-checkpoint numbers must never be
described as evaluation on an untouched test set.

The scheduler fail-closes on the exact split bytes and sample-name manifests
present in ``/home/ly/DEA/datasets`` when this protocol was frozen.  It does
not discover or read any validation/hcval list.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_DIR / "datasets"
TRAINING_ENTRY = PROJECT_DIR / "tools" / "train_test_selected_full_train.py"
PROTOCOL = "test_selected_full_train_interval_v1"
DATASET_NAMES = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TEST_INTERVAL = 10
EVAL_THRESHOLD = 0.5
PD_FA_MIN_PD = 0.93
PD_FA_MIN_IOU = 0.655
PAIRED_BASELINE_IOU = 0.0
SELECTION_TIE_BREAK = "earliest_epoch"
SELECTION_BEST_IOU_RULE = "strictly_greater_iou; ties_keep_earliest_epoch"
SELECTION_PD_FA_RULE = (
    "pd>=0.93 and iou>=0.655 then strictly_minimum_fa; "
    "ties_keep_earliest_epoch"
)


# Frozen against the repository's canonical img_idx files.  raw_sha256 checks
# byte identity (including line endings); ordered_names_sha256 checks the
# decoded, non-empty sample-id sequence independently.
CANONICAL_DATASETS: dict[str, dict[str, Any]] = {
    "NUAA-SIRST": {
        "train": {
            "count": 213,
            "raw_sha256": "324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4",
            "ordered_names_sha256": "815dcca749f087f27f5dad4b447015aee70bd7ae7779d6fdd7d6efa6d5c6943f",
        },
        "test": {
            "count": 214,
            "raw_sha256": "e49023203a323c247306b314f23c8b3b917093a26984067792355adff7a8386e",
            "ordered_names_sha256": "395eecd6bf0ed2a59f531de9145688597632c68f9d0933359aadcb93ec1a60b5",
        },
        "asset_count": 427,
        "asset_names_sha256": "d88e6455d627096488e6d2f2efc4df6a2e45892be41264e34038b82e0b2d2f50",
    },
    "NUDT-SIRST": {
        "train": {
            "count": 663,
            "raw_sha256": "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3",
            "ordered_names_sha256": "dc555df66b62dd1ea98d119ace8fe8ae86de94f3e4833d8d81e90c0e1f287922",
        },
        "test": {
            "count": 664,
            "raw_sha256": "a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e",
            "ordered_names_sha256": "cec44220c69d89a5b3fd245b8ee911404e959fef80bd96b32b6b74f28bb32af0",
        },
        "asset_count": 1327,
        "asset_names_sha256": "aaa99494be0219075407fff10c0e152eb8487b30097cee14e093e36c4a12ac2c",
    },
    "IRSTD-1K": {
        "train": {
            "count": 800,
            "raw_sha256": "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744",
            "ordered_names_sha256": "b698d2d9dbe9e26e1875978d23450e1e6ec45fd71d56d31415007f56c40bba88",
        },
        "test": {
            "count": 201,
            "raw_sha256": "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579",
            "ordered_names_sha256": "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579",
        },
        "asset_count": 1001,
        "asset_names_sha256": "86d1811451f321fa60b5aaf2fbb99982d2b473f10881e5c9db6df91ea2d0b740",
    },
}


def parse_csv(text: str, cast):
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            "expected at least one comma-separated value"
        )
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Schedule full-canonical-train MSHNet baselines whose checkpoints "
            "are selected using periodic canonical-test evaluations."
        )
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_NAMES),
        help="Comma-separated canonical dataset names.",
    )
    parser.add_argument(
        "--seeds",
        default="20260711,20260712,20260713",
        help="Comma-separated seeds; every dataset uses every seed.",
    )
    parser.add_argument(
        "--gpus", default="2,3", help="Physical GPU ids, one process per GPU."
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument(
        "--test-interval",
        type=int,
        default=TEST_INTERVAL,
        help=f"Protocol invariant; only {TEST_INTERVAL} is accepted.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=EVAL_THRESHOLD,
        help=f"Checkpoint metric threshold; locked to {EVAL_THRESHOLD}.",
    )
    parser.add_argument(
        "--pd-fa-min-pd",
        type=float,
        default=PD_FA_MIN_PD,
        help=f"Constrained-min-FA Pd floor; locked to {PD_FA_MIN_PD}.",
    )
    parser.add_argument(
        "--pd-fa-min-iou",
        type=float,
        default=PD_FA_MIN_IOU,
        help=f"Constrained-min-FA IoU floor; locked to {PD_FA_MIN_IOU}.",
    )
    parser.add_argument(
        "--paired-baseline-iou",
        type=float,
        default=PAIRED_BASELINE_IOU,
        help=f"Paired baseline IoU term; locked to {PAIRED_BASELINE_IOU}.",
    )
    parser.add_argument(
        "--deterministic", choices=("true", "false"), default="true"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Protocol invariant; only 0 is accepted (there is no validation set).",
    )
    parser.add_argument(
        "--batch-id",
        default="mshnet_test_selected_full_train_interval_v1",
        help="Stable id below weight/test_selected and repro_runs/test_selected.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scheduler-complete jobs and ask pending training jobs to resume.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit inputs and print commands without creating run artifacts.",
    )
    return parser.parse_args(argv)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names: Iterable[str]) -> str:
    return sha256_bytes(("\n".join(names) + "\n").encode("utf-8"))


def _assert_not_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")


def _assert_path_chain_not_symlink(path: Path, stop: Path) -> None:
    """Reject symlinks from ``stop`` through ``path`` (both inclusive)."""

    current = path
    stop = stop.absolute()
    while True:
        _assert_not_symlink(current, "canonical path component")
        if current.absolute() == stop:
            return
        if current.parent == current:
            raise RuntimeError(f"{path} is not below canonical root {stop}")
        current = current.parent


def _read_split(path: Path, expected: dict[str, Any], role: str) -> list[str]:
    _assert_not_symlink(path, f"{role} split")
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical {role} split: {path}")
    raw = path.read_bytes()
    raw_hash = sha256_bytes(raw)
    if raw_hash != expected["raw_sha256"]:
        raise RuntimeError(
            f"{role} raw split hash mismatch for {path}: "
            f"actual={raw_hash}, expected={expected['raw_sha256']}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{role} split is not UTF-8: {path}") from exc
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if len(names) != expected["count"]:
        raise RuntimeError(
            f"{role} count mismatch for {path}: actual={len(names)}, "
            f"expected={expected['count']}"
        )
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate sample id in canonical {role} split: {path}")
    for name in names:
        if (
            not name
            or Path(name).name != name
            or Path(name).suffix
            or "hcval" in name.lower()
        ):
            raise RuntimeError(f"invalid sample id in {role} split: {name!r}")
    decoded_hash = names_sha256(names)
    if decoded_hash != expected["ordered_names_sha256"]:
        raise RuntimeError(
            f"{role} ordered-name hash mismatch for {path}: "
            f"actual={decoded_hash}, expected={expected['ordered_names_sha256']}"
        )
    return names


def _collect_png_assets(directory: Path, label: str) -> tuple[set[str], str]:
    _assert_not_symlink(directory, label)
    if not directory.is_dir():
        raise FileNotFoundError(f"missing canonical {label} directory: {directory}")
    names: set[str] = set()
    for path in directory.iterdir():
        # Fail closed on every symlink in a canonical asset directory, even if
        # it is not a PNG.  Ordinary unrelated files (e.g. .DS_Store) do not
        # participate in the sample-id universe.
        _assert_not_symlink(path, f"entry in {label} directory")
        if path.suffix.lower() != ".png":
            continue
        if not path.is_file():
            raise RuntimeError(f"canonical {label} PNG is not a file: {path}")
        if path.stem in names:
            raise RuntimeError(f"duplicate canonical {label} id: {path.stem}")
        with path.open("rb") as handle:
            signature = handle.read(len(PNG_SIGNATURE))
        if signature != PNG_SIGNATURE:
            raise RuntimeError(f"invalid or truncated PNG in {label}: {path}")
        names.add(path.stem)
    ordered_hash = names_sha256(sorted(names))
    return names, ordered_hash


def validate_dataset(dataset_name: str) -> dict[str, Any]:
    if dataset_name not in CANONICAL_DATASETS:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; allowed={DATASET_NAMES}"
        )
    expected = CANONICAL_DATASETS[dataset_name]
    dataset_dir = DATASET_ROOT / dataset_name
    img_idx_dir = dataset_dir / "img_idx"
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    train_file = img_idx_dir / f"train_{dataset_name}.txt"
    test_file = img_idx_dir / f"test_{dataset_name}.txt"

    for path in (DATASET_ROOT, dataset_dir, img_idx_dir, images_dir, masks_dir):
        _assert_path_chain_not_symlink(path, DATASET_ROOT)
    if not dataset_dir.is_dir() or not img_idx_dir.is_dir():
        raise FileNotFoundError(f"incomplete canonical dataset layout: {dataset_dir}")
    # These exact basenames are the only split inputs.  In particular, an
    # hcval file may exist in img_idx but is neither discovered nor read.
    if "hcval" in train_file.name.lower() or "hcval" in test_file.name.lower():
        raise RuntimeError("hcval is prohibited by the full-train protocol")

    train_names = _read_split(train_file, expected["train"], "train")
    test_names = _read_split(test_file, expected["test"], "test")
    train_set = set(train_names)
    test_set = set(test_names)
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(
            f"canonical train/test overlap for {dataset_name}: {sorted(overlap)[:5]}"
        )

    image_names, image_hash = _collect_png_assets(images_dir, "images")
    mask_names, mask_hash = _collect_png_assets(masks_dir, "masks")
    split_union = train_set | test_set
    if image_names != mask_names:
        raise RuntimeError(
            f"image/mask id universe mismatch for {dataset_name}: "
            f"image_only={sorted(image_names - mask_names)[:5]}, "
            f"mask_only={sorted(mask_names - image_names)[:5]}"
        )
    if split_union != image_names:
        raise RuntimeError(
            f"train/test union does not equal asset universe for {dataset_name}: "
            f"split_only={sorted(split_union - image_names)[:5]}, "
            f"asset_only={sorted(image_names - split_union)[:5]}"
        )
    if len(image_names) != expected["asset_count"]:
        raise RuntimeError(
            f"canonical asset count mismatch for {dataset_name}: "
            f"actual={len(image_names)}, expected={expected['asset_count']}"
        )
    if image_hash != expected["asset_names_sha256"] or mask_hash != expected["asset_names_sha256"]:
        raise RuntimeError(
            f"canonical asset-name hash mismatch for {dataset_name}: "
            f"images={image_hash}, masks={mask_hash}, "
            f"expected={expected['asset_names_sha256']}"
        )

    return {
        "dataset": dataset_name,
        "dataset_dir": str(dataset_dir),
        "train_split_file": str(train_file),
        "test_split_file": str(test_file),
        "train_split_arg": str(train_file.relative_to(dataset_dir)),
        "test_split_arg": str(test_file.relative_to(dataset_dir)),
        "train_count": len(train_names),
        "test_count": len(test_names),
        "train_test_overlap_count": 0,
        "split_union_count": len(split_union),
        "image_count": len(image_names),
        "mask_count": len(mask_names),
        "train_raw_sha256": expected["train"]["raw_sha256"],
        "test_raw_sha256": expected["test"]["raw_sha256"],
        "train_ordered_names_sha256": expected["train"]["ordered_names_sha256"],
        "test_ordered_names_sha256": expected["test"]["ordered_names_sha256"],
        "asset_names_sha256": image_hash,
        "asset_universe_equality": "train_union_test=image_ids=mask_ids",
        "symlink_count": 0,
        "validation_split": None,
        "hcval_policy": "prohibited_not_read",
    }


def evaluation_epochs(epochs: int, test_interval: int) -> list[int]:
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    if test_interval < 1:
        raise ValueError("--test-interval must be positive")
    points = list(range(test_interval, epochs + 1, test_interval))
    if not points or points[-1] != epochs:
        points.append(epochs)
    return points


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def validate_resume_contract(
    manifest_path: Path, current_contract: dict[str, Any]
) -> dict[str, Any]:
    """Require exact agreement with the prior immutable batch contract."""

    try:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read prior manifest: {manifest_path}") from exc
    prior_contract = prior.get("immutable_contract")
    prior_digest = prior.get("immutable_contract_sha256")
    if not isinstance(prior_contract, dict) or not isinstance(prior_digest, str):
        raise RuntimeError(
            f"prior manifest lacks immutable resume contract: {manifest_path}"
        )
    actual_prior_digest = canonical_json_sha256(prior_contract)
    if actual_prior_digest != prior_digest:
        raise RuntimeError(
            f"prior immutable contract digest is invalid in {manifest_path}"
        )
    current_digest = canonical_json_sha256(current_contract)
    if prior_digest != current_digest or prior_contract != current_contract:
        differing = sorted(
            key
            for key in set(prior_contract) | set(current_contract)
            if prior_contract.get(key) != current_contract.get(key)
        )
        raise RuntimeError(
            "resume immutable contract mismatch; refusing to mix runs under "
            f"one batch id (differing sections={differing})"
        )
    return prior


def build_command(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(TRAINING_ENTRY),
        "--mode",
        "train",
        "--model-type",
        "mshnet",
        "--dataset-dir",
        job["dataset_dir"],
        "--train-split-file",
        job["train_split_arg"],
        "--test-split-file",
        job["test_split_arg"],
        "--val-fraction",
        "0",
        "--seed",
        str(job["seed"]),
        "--deterministic",
        args.deterministic,
        "--epochs",
        str(args.epochs),
        "--test-interval",
        str(args.test_interval),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--warm-epoch",
        str(args.warm_epoch),
        "--threshold",
        str(args.threshold),
        "--pd-fa-min-pd",
        str(args.pd_fa_min_pd),
        "--pd-fa-min-iou",
        str(args.pd_fa_min_iou),
        "--paired-baseline-iou",
        str(args.paired_baseline_iou),
        "--selection-tie-break",
        SELECTION_TIE_BREAK,
        "--run-dir",
        job["run_dir"],
        "--run-label",
        job["job_id"],
    ]
    checkpoint = Path(job["run_dir"]) / "checkpoint.pkl"
    if args.resume and checkpoint.is_file() and not checkpoint.is_symlink():
        command.extend(
            ["--if-checkpoint", "true", "--checkpoint-dir", job["run_dir"]]
        )
    return command


def _validate_args(args: argparse.Namespace) -> tuple[list[str], list[int], list[int]]:
    datasets = parse_csv(args.datasets, str)
    seeds = parse_csv(args.seeds, int)
    gpus = parse_csv(args.gpus, int)
    if len(datasets) != len(set(datasets)):
        raise ValueError("--datasets contains duplicates")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicates")
    if len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError("--gpus must contain unique non-negative ids")
    unknown = sorted(set(datasets).difference(DATASET_NAMES))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}; allowed={DATASET_NAMES}")
    if args.val_fraction != 0.0:
        raise ValueError("--val-fraction is fixed to 0; validation is prohibited")
    if args.test_interval != TEST_INTERVAL:
        raise ValueError(
            f"--test-interval is locked to {TEST_INTERVAL} by protocol {PROTOCOL}"
        )
    locked_selectors = {
        "--threshold": (args.threshold, EVAL_THRESHOLD),
        "--pd-fa-min-pd": (args.pd_fa_min_pd, PD_FA_MIN_PD),
        "--pd-fa-min-iou": (args.pd_fa_min_iou, PD_FA_MIN_IOU),
        "--paired-baseline-iou": (
            args.paired_baseline_iou,
            PAIRED_BASELINE_IOU,
        ),
    }
    for label, (actual, expected) in locked_selectors.items():
        if actual != expected:
            raise ValueError(
                f"{label} is locked to {expected} by protocol {PROTOCOL}"
            )
    evaluation_epochs(args.epochs, args.test_interval)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid batch-size or num-workers")
    if args.lr <= 0.0 or args.warm_epoch < 0:
        raise ValueError("invalid lr or warm-epoch")
    if not args.batch_id or Path(args.batch_id).name != args.batch_id:
        raise ValueError("--batch-id must be one safe path component")
    return datasets, seeds, gpus


def _load_torch_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before the weights_only argument.
            payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _checkpoint_matches_job(
    path: Path,
    job: dict[str, Any],
    eligible_zero_based: list[int],
    selection: dict[str, Any],
) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    checkpoint = _load_torch_checkpoint(path)
    if checkpoint is None:
        return False
    metadata = checkpoint.get("method_meta")
    if not isinstance(metadata, dict):
        return False
    expected_metadata = {
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL,
        "method": "MSHNet",
        "model_type": "mshnet",
        "dataset_name": job["dataset"],
        "dataset_dir": job["dataset_dir"],
        "train_split_file": str(
            Path(job["dataset_dir"]) / job["train_split_arg"]
        ),
        "test_split_file": str(
            Path(job["dataset_dir"]) / job["test_split_arg"]
        ),
        "val_split_file": "",
        "val_fraction": 0.0,
        "train_split_sha256": job["train_split_sha256"],
        "test_split_sha256": job["test_split_sha256"],
        "seed": job["seed"],
        "deterministic": job["deterministic"],
        "run_label": job["job_id"],
        "selection_split": "test",
        "evaluation_split": "test",
        "no_internal_holdout": True,
        "test_interval": job["test_interval"],
        "selection_threshold": EVAL_THRESHOLD,
        "selection_tie_break": SELECTION_TIE_BREAK,
        "selection_best_iou_rule": SELECTION_BEST_IOU_RULE,
        "selection_pd_fa_rule": SELECTION_PD_FA_RULE,
        "selection_pd_fa_min_pd": PD_FA_MIN_PD,
        "selection_pd_fa_min_iou": PD_FA_MIN_IOU,
        "selection_paired_baseline_iou": PAIRED_BASELINE_IOU,
        "train_loader_drop_last": False,
    }
    if "train_split_raw_sha256" in job:
        expected_metadata["train_split_raw_sha256"] = job[
            "train_split_raw_sha256"
        ]
    if "test_split_raw_sha256" in job:
        expected_metadata["test_split_raw_sha256"] = job[
            "test_split_raw_sha256"
        ]
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        return False
    try:
        checkpoint_epoch = int(checkpoint["epoch"])
        summary_epoch = int(selection["epoch_zero_based"])
    except (KeyError, TypeError, ValueError):
        return False
    if checkpoint_epoch != summary_epoch or checkpoint_epoch not in eligible_zero_based:
        return False
    if selection.get("sha256") != sha256_file(path):
        return False
    for key in ("iou", "pd", "fa"):
        try:
            checkpoint_value = float(checkpoint[key])
            summary_value = float(selection[key])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            not math.isfinite(checkpoint_value)
            or not math.isfinite(summary_value)
            or checkpoint_value != summary_value
        ):
            return False
        if key in {"iou", "pd"} and not 0.0 <= summary_value <= 1.0:
            return False
        if key == "fa" and summary_value < 0.0:
            return False
    return True


def _is_completed_result(job: dict[str, Any]) -> bool:
    path = Path(job["result_file"])
    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("returncode") != 0 or payload.get("protocol") != PROTOCOL:
        return False
    if payload.get("job_id") != job["job_id"]:
        return False
    if payload.get("dataset") != job["dataset"] or payload.get("seed") != job["seed"]:
        return False
    if payload.get("total_epochs") != job["total_epochs"]:
        return False
    if payload.get("test_interval") != job["test_interval"]:
        return False
    if payload.get("test_evaluation_epochs") != job["test_evaluation_epochs"]:
        return False
    run_dir_text = payload.get("run_dir")
    if run_dir_text != job["run_dir"]:
        return False
    summary = Path(run_dir_text) / "protocol_summary.json"
    if summary.is_symlink() or not summary.is_file():
        return False
    try:
        summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    eligible_zero_based = [
        completed_epoch - 1 for completed_epoch in job["test_evaluation_epochs"]
    ]
    if summary_payload.get("protocol") != PROTOCOL:
        return False
    if summary_payload.get("status") != "complete":
        return False
    if summary_payload.get("dataset") != job["dataset"]:
        return False
    if summary_payload.get("run_dir") != job["run_dir"]:
        return False
    if summary_payload.get("total_epochs") != job["total_epochs"]:
        return False
    if summary_payload.get("test_interval") != job["test_interval"]:
        return False
    if summary_payload.get("last_completed_epoch_zero_based") != job["total_epochs"] - 1:
        return False
    if summary_payload.get("executed_evaluation_epochs_zero_based") != eligible_zero_based:
        return False
    selection = summary_payload.get("checkpoint_selection")
    if not isinstance(selection, dict):
        return False
    best_iou = selection.get("best_iou")
    if not isinstance(best_iou, dict) or best_iou.get("status") != "found":
        return False
    if best_iou.get("file") != "checkpoint_best_iou.pkl":
        return False
    if not _checkpoint_matches_job(
        Path(run_dir_text) / "checkpoint_best_iou.pkl",
        job,
        eligible_zero_based,
        best_iou,
    ):
        return False
    constrained = selection.get("constrained_min_fa")
    if not isinstance(constrained, dict):
        return False
    if constrained.get("status") == "found":
        if constrained.get("file") != "checkpoint_pd_fa_best.pkl":
            return False
        try:
            constrained_pd = float(constrained["pd"])
            constrained_iou = float(constrained["iou"])
        except (KeyError, TypeError, ValueError):
            return False
        constrained_iou_floor = max(PD_FA_MIN_IOU, PAIRED_BASELINE_IOU)
        if (
            not math.isfinite(constrained_pd)
            or not math.isfinite(constrained_iou)
            or constrained_pd < PD_FA_MIN_PD
            or constrained_iou < constrained_iou_floor
        ):
            return False
        if not _checkpoint_matches_job(
            Path(run_dir_text) / "checkpoint_pd_fa_best.pkl",
            job,
            eligible_zero_based,
            constrained,
        ):
            return False
    elif constrained.get("status") == "not_found":
        if (
            constrained.get("file") is not None
            or constrained.get("reason") != "no_eligible_epoch"
        ):
            return False
        stale = Path(run_dir_text) / "checkpoint_pd_fa_best.pkl"
        if stale.exists() or stale.is_symlink():
            return False
    else:
        return False
    return True


def _directory_has_entries(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    datasets, seeds, gpus = _validate_args(args)
    dataset_meta = {name: validate_dataset(name) for name in datasets}
    eval_epochs = evaluation_epochs(args.epochs, args.test_interval)

    report_root = PROJECT_DIR / "repro_runs" / "test_selected" / args.batch_id
    weight_root = PROJECT_DIR / "weight" / "test_selected" / args.batch_id
    jobs: list[dict[str, Any]] = []
    for seed in seeds:
        for dataset_name in datasets:
            metadata = dataset_meta[dataset_name]
            job_id = f"mshnet__{dataset_name.lower()}__seed_{seed}"
            jobs.append(
                {
                    "job_id": job_id,
                    "protocol": PROTOCOL,
                    "dataset": dataset_name,
                    "seed": seed,
                    "dataset_dir": metadata["dataset_dir"],
                    "train_split_arg": metadata["train_split_arg"],
                    "test_split_arg": metadata["test_split_arg"],
                    "train_split_sha256": metadata.get(
                        "train_ordered_names_sha256",
                        CANONICAL_DATASETS[dataset_name]["train"][
                            "ordered_names_sha256"
                        ],
                    ),
                    "test_split_sha256": metadata.get(
                        "test_ordered_names_sha256",
                        CANONICAL_DATASETS[dataset_name]["test"][
                            "ordered_names_sha256"
                        ],
                    ),
                    "train_split_raw_sha256": metadata.get(
                        "train_raw_sha256",
                        CANONICAL_DATASETS[dataset_name]["train"][
                            "raw_sha256"
                        ],
                    ),
                    "test_split_raw_sha256": metadata.get(
                        "test_raw_sha256",
                        CANONICAL_DATASETS[dataset_name]["test"][
                            "raw_sha256"
                        ],
                    ),
                    "deterministic": args.deterministic == "true",
                    "run_dir": str(weight_root / dataset_name / f"seed_{seed}"),
                    "log_file": str(report_root / "logs" / f"{job_id}.log"),
                    "result_file": str(report_root / "jobs" / f"{job_id}.json"),
                    "total_epochs": args.epochs,
                    "test_interval": args.test_interval,
                    "test_evaluation_epochs": eval_epochs,
                }
            )

    scheduler_path = Path(__file__).resolve()
    scheduler_hash = sha256_file(scheduler_path)
    entry_hash = sha256_file(TRAINING_ENTRY) if TRAINING_ENTRY.is_file() else None
    immutable_args = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume", "dry_run"}
    }
    immutable_jobs = [
        {
            "job_id": job["job_id"],
            "dataset": job["dataset"],
            "seed": job["seed"],
            "dataset_dir": job["dataset_dir"],
            "train_split_arg": job["train_split_arg"],
            "test_split_arg": job["test_split_arg"],
            "train_split_sha256": job["train_split_sha256"],
            "test_split_sha256": job["test_split_sha256"],
            "train_split_raw_sha256": job["train_split_raw_sha256"],
            "test_split_raw_sha256": job["test_split_raw_sha256"],
            "deterministic": job["deterministic"],
            "run_dir": job["run_dir"],
            "total_epochs": job["total_epochs"],
            "test_interval": job["test_interval"],
            "test_evaluation_epochs": job["test_evaluation_epochs"],
        }
        for job in jobs
    ]
    immutable_contract = {
        "protocol": PROTOCOL,
        "model": "MSHNet baseline",
        "args_excluding_resume_and_dry_run": immutable_args,
        "datasets": dataset_meta,
        "jobs": immutable_jobs,
        "checkpoint_selector": {
            "threshold": EVAL_THRESHOLD,
            "pd_fa_min_pd": PD_FA_MIN_PD,
            "pd_fa_min_iou": PD_FA_MIN_IOU,
            "paired_baseline_iou": PAIRED_BASELINE_IOU,
            "tie_break": SELECTION_TIE_BREAK,
        },
        "source_sha256": {
            "scheduler": scheduler_hash,
            "training_entry": entry_hash,
        },
    }
    manifest = {
        "protocol": PROTOCOL,
        "batch_id": args.batch_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": "MSHNet baseline",
        "stage": "full_canonical_train_periodic_test_selection",
        "args": vars(args),
        "data_protocol": {
            "dataset_root": str(DATASET_ROOT),
            "fit_role": "complete canonical img_idx/train_<dataset>.txt",
            "validation_role": None,
            "validation_fraction": 0.0,
            "test_role": (
                "canonical img_idx/test_<dataset>.txt; evaluated periodically "
                "and used for checkpoint selection"
            ),
            "hcval_policy": "prohibited_not_read",
        },
        "selection_policy": {
            "test_selected": True,
            "test_is_untouched": False,
            "test_interval_completed_epochs": args.test_interval,
            "test_evaluation_epochs": eval_epochs,
            "final_epoch_added_when_not_interval_multiple": True,
            "eligible_checkpoint_epochs": eval_epochs,
            "metric_threshold": EVAL_THRESHOLD,
            "best_test_iou": "maximum test IoU over eligible epochs",
            "best_test_constrained_min_fa": (
                f"minimum test FA among eligible epochs with Pd >= {PD_FA_MIN_PD} "
                f"and IoU >= max({PD_FA_MIN_IOU}, "
                f"paired_baseline_iou={PAIRED_BASELINE_IOU})"
            ),
            "pd_fa_min_pd": PD_FA_MIN_PD,
            "pd_fa_min_iou": PD_FA_MIN_IOU,
            "paired_baseline_iou": PAIRED_BASELINE_IOU,
            "strict_tie_break": SELECTION_TIE_BREAK,
            "reporting_warning": (
                "Both selected checkpoints use test feedback and must be reported "
                "as test-selected, not as performance on an untouched test set."
            ),
        },
        "canonical_integrity": {
            "checks": [
                "frozen train/test counts",
                "frozen split raw-byte SHA-256",
                "frozen ordered-name SHA-256",
                "train/test overlap = 0",
                "train union test = image ids = mask ids",
                "all referenced assets have PNG signatures",
                "no symlinks in canonical paths/assets",
            ]
        },
        "sources": {
            "scheduler": str(scheduler_path),
            "scheduler_sha256": scheduler_hash,
            "training_entry": str(TRAINING_ENTRY),
            "training_entry_sha256": entry_hash,
            "training_entry_missing_is_allowed_only_for_dry_run": True,
        },
        "immutable_contract": immutable_contract,
        "immutable_contract_sha256": canonical_json_sha256(immutable_contract),
        "datasets": dataset_meta,
        "jobs": jobs,
    }

    manifest_path = report_root / "manifest.json"
    prior_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        if manifest_path.is_symlink():
            raise RuntimeError(f"manifest must not be a symlink: {manifest_path}")
        if args.resume:
            prior_manifest = validate_resume_contract(
                manifest_path, immutable_contract
            )
            manifest["created_at"] = prior_manifest.get(
                "created_at", manifest["created_at"]
            )
            manifest["resumed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            manifest["resume_count"] = int(prior_manifest.get("resume_count", 0)) + 1
        elif not args.dry_run:
            raise FileExistsError(
                f"batch already exists: {manifest_path}; pass --resume or use a new --batch-id"
            )
    elif args.resume and (
        _directory_has_entries(report_root) or _directory_has_entries(weight_root)
    ):
        raise RuntimeError(
            "cannot resume existing batch artifacts without the immutable manifest: "
            f"{manifest_path}"
        )
    elif not args.resume and not args.dry_run and (
        _directory_has_entries(report_root) or _directory_has_entries(weight_root)
    ):
        raise RuntimeError(
            "batch directories already contain artifacts but no manifest; choose a "
            "new --batch-id"
        )

    pending: list[dict[str, Any]] = []
    for job in jobs:
        if args.resume and _is_completed_result(job):
            print(f"skip completed {job['job_id']}", flush=True)
            continue
        job["command"] = build_command(args, job)
        pending.append(job)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        for index, job in enumerate(pending):
            gpu = gpus[index % len(gpus)]
            print(f"GPU {gpu}: {shlex.join(job['command'])}")
        return 0

    if not TRAINING_ENTRY.is_file() or TRAINING_ENTRY.is_symlink():
        raise FileNotFoundError(
            f"missing or symlinked training entry: {TRAINING_ENTRY}"
        )
    report_root.mkdir(parents=True, exist_ok=True)
    weight_root.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)

    active: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            log_path = Path(job["log_file"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            log_handle.write(
                json.dumps(
                    {
                        "event": "scheduler_start",
                        "protocol": PROTOCOL,
                        "gpu": gpu,
                        "command": job["command"],
                        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            started_at = dt.datetime.now(dt.timezone.utc).isoformat()
            process = subprocess.Popen(
                job["command"],
                cwd=PROJECT_DIR,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu] = {
                "job": job,
                "process": process,
                "log_handle": log_handle,
                "started_at": started_at,
                "started_monotonic": time.monotonic(),
            }
            print(
                f"start gpu={gpu} pid={process.pid} job={job['job_id']}",
                flush=True,
            )

        time.sleep(2.0)
        for gpu, state in list(active.items()):
            process = state["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            state["log_handle"].close()
            job = state["job"]
            payload = {
                "protocol": PROTOCOL,
                "job_id": job["job_id"],
                "dataset": job["dataset"],
                "seed": job["seed"],
                "gpu": gpu,
                "pid": process.pid,
                "started_at": state["started_at"],
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - state["started_monotonic"],
                "returncode": returncode,
                "command": job["command"],
                "total_epochs": job["total_epochs"],
                "test_interval": job["test_interval"],
                "test_evaluation_epochs": eval_epochs,
                "test_selected": True,
                "log_file": job["log_file"],
                "run_dir": job["run_dir"],
            }
            write_json(Path(job["result_file"]), payload)
            print(
                f"finish gpu={gpu} rc={returncode} job={job['job_id']} "
                f"elapsed={payload['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if returncode != 0:
                failures.append(job["job_id"])
            del active[gpu]

    if failures:
        print("failed jobs: " + ", ".join(failures), file=sys.stderr)
        return 1
    print(
        f"all {len(jobs)} {PROTOCOL} baseline jobs completed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
