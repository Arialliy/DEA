#!/usr/bin/env python3
"""Run clean, paired MSHNet development baselines on all three datasets.

The scheduler intentionally uses only official training lists for fitting and
an internal deterministic holdout for checkpoint selection. Official test
lists are loaded only by the training process's fail-closed split audit; they
are not iterated during this development stage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")


def parse_csv(text: str, cast):
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schedule clean MSHNet holdout baselines on fixed GPUs."
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_NAMES),
        help="Comma-separated dataset directory names.",
    )
    parser.add_argument(
        "--seeds",
        default="20260711,20260712,20260713",
        help="Comma-separated model seeds; every dataset uses every seed.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated physical GPU ids, one process per GPU.",
    )
    parser.add_argument("--split-seed", type=int, default=20260711)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument(
        "--deterministic", choices=("true", "false"), default="true"
    )
    parser.add_argument(
        "--batch-id",
        default="clean_baseline_holdout_v1",
        help="Stable id used below weight/clean and repro_runs/clean.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume jobs that contain checkpoint.pkl."
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f"empty split: {path}")
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate sample id in split: {path}")
    return names


def split_hash(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def validate_dataset(dataset_name: str, split_seed: int, val_fraction: float) -> dict:
    dataset_dir = PROJECT_DIR / "datasets" / dataset_name
    train_file = dataset_dir / "img_idx" / f"train_{dataset_name}.txt"
    test_file = dataset_dir / "img_idx" / f"test_{dataset_name}.txt"
    if not train_file.is_file() or not test_file.is_file():
        raise FileNotFoundError(f"missing official split files for {dataset_name}")

    train_names = read_names(train_file)
    test_names = read_names(test_file)
    overlap = sorted(set(train_names).intersection(test_names))
    if overlap:
        raise RuntimeError(
            f"official train/test overlap for {dataset_name}: {overlap[:5]}"
        )

    missing = []
    for name in train_names + test_names:
        for kind in ("images", "masks"):
            path = dataset_dir / kind / f"{name}.png"
            if not path.is_file():
                missing.append(str(path))
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break
    if missing:
        raise FileNotFoundError("missing image/mask files: " + ", ".join(missing))

    ranked = sorted(
        train_names,
        key=lambda name: hashlib.sha256(f"{split_seed}\0{name}".encode("utf-8")).digest(),
    )
    num_val = max(1, min(len(train_names) - 1, int(round(len(train_names) * val_fraction))))
    val_set = set(ranked[:num_val])
    fit_names = [name for name in train_names if name not in val_set]
    val_names = [name for name in train_names if name in val_set]
    return {
        "dataset": dataset_name,
        "dataset_dir": str(dataset_dir),
        # The dataset loader resolves relative split overrides from the
        # dataset directory, so keep these paths dataset-relative.
        "train_file": str(train_file.relative_to(dataset_dir)),
        "test_file": str(test_file.relative_to(dataset_dir)),
        "official_train_count": len(train_names),
        "fit_count": len(fit_names),
        "val_count": len(val_names),
        "official_test_count": len(test_names),
        "official_train_sha256": split_hash(train_names),
        "fit_sha256": split_hash(fit_names),
        "val_sha256": split_hash(val_names),
        "official_test_sha256": split_hash(test_names),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_command(args: argparse.Namespace, job: dict) -> list[str]:
    run_dir = Path(job["run_dir"])
    command = [
        sys.executable,
        str(PROJECT_DIR / "main.py"),
        "--mode", "train",
        "--model-type", "mshnet",
        "--dataset-dir", job["dataset_dir"],
        "--train-split-file", job["train_file"],
        "--test-split-file", job["test_file"],
        "--val-fraction", str(args.val_fraction),
        "--split-seed", str(args.split_seed),
        "--seed", str(job["seed"]),
        "--deterministic", args.deterministic,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--warm-epoch", str(args.warm_epoch),
        "--run-dir", str(run_dir),
        "--run-label", job["job_id"],
    ]
    checkpoint = run_dir / "checkpoint.pkl"
    if args.resume and checkpoint.is_file():
        command.extend(
            ["--if-checkpoint", "true", "--checkpoint-dir", str(run_dir)]
        )
    return command


def main() -> int:
    args = parse_args()
    datasets = parse_csv(args.datasets, str)
    seeds = parse_csv(args.seeds, int)
    gpus = parse_csv(args.gpus, int)
    unknown = sorted(set(datasets).difference(DATASET_NAMES))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}; allowed={DATASET_NAMES}")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")

    dataset_meta = {
        name: validate_dataset(name, args.split_seed, args.val_fraction)
        for name in datasets
    }
    report_root = PROJECT_DIR / "repro_runs" / "clean" / args.batch_id
    weight_root = PROJECT_DIR / "weight" / "clean" / args.batch_id
    report_root.mkdir(parents=True, exist_ok=True)
    weight_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in seeds:
        for dataset_name in datasets:
            job_id = f"mshnet__{dataset_name.lower()}__seed_{seed}"
            job = {
                "job_id": job_id,
                "dataset": dataset_name,
                "seed": seed,
                "dataset_dir": dataset_meta[dataset_name]["dataset_dir"],
                "train_file": dataset_meta[dataset_name]["train_file"],
                "test_file": dataset_meta[dataset_name]["test_file"],
                "run_dir": str(weight_root / dataset_name / f"seed_{seed}"),
                "log_file": str(report_root / "logs" / f"{job_id}.log"),
                "result_file": str(report_root / "jobs" / f"{job_id}.json"),
            }
            jobs.append(job)

    manifest = {
        "batch_id": args.batch_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": "development_holdout_baseline",
        "official_test_policy": "loaded only for disjoint/hash audit; not iterated",
        "args": vars(args),
        "datasets": dataset_meta,
        "jobs": jobs,
    }
    manifest_path = report_root / "manifest.json"
    if manifest_path.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"batch already exists: {manifest_path}; pass --resume or use a new --batch-id"
        )
    write_json(manifest_path, manifest)

    pending = []
    for job in jobs:
        result_path = Path(job["result_file"])
        if args.resume and result_path.is_file():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("returncode") == 0:
                print(f"skip completed {job['job_id']}", flush=True)
                continue
        job["command"] = build_command(args, job)
        pending.append(job)

    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = gpus[index % len(gpus)]
            print(f"GPU {gpu}: " + " ".join(job["command"]))
        return 0

    active: dict[int, dict] = {}
    failures = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            log_path = Path(job["log_file"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
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
            print(f"start gpu={gpu} pid={process.pid} job={job['job_id']}", flush=True)

        time.sleep(2.0)
        for gpu, state in list(active.items()):
            process = state["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            state["log_handle"].close()
            job = state["job"]
            payload = {
                "job_id": job["job_id"],
                "gpu": gpu,
                "pid": process.pid,
                "started_at": state["started_at"],
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - state["started_monotonic"],
                "returncode": returncode,
                "command": job["command"],
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
    print(f"all {len(jobs)} clean baseline jobs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
