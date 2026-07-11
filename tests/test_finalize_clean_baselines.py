from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.finalize_clean_baselines import (
    DATASET_NAMES,
    EXPECTED_EPOCHS,
    FinalizationError,
    OUTPUT_JSON,
    OUTPUT_MARKDOWN,
    finalize_batch,
    load_checkpoint_cpu,
)


SEEDS = (101, 102, 103)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def make_complete_batch(tmp_path: Path):
    batch_dir = tmp_path / "fixture_batch"
    dataset_meta = {}
    jobs = []
    checkpoints = {}
    for dataset_index, dataset in enumerate(DATASET_NAMES):
        dataset_meta[dataset] = {
            "dataset": dataset,
            "fit_sha256": digest(f"{dataset}:fit"),
            "val_sha256": digest(f"{dataset}:val"),
            "official_test_sha256": digest(f"{dataset}:test"),
        }
        for seed_index, seed in enumerate(SEEDS):
            job_id = f"mshnet__{dataset.lower()}__seed_{seed}"
            run_dir = tmp_path / "weights" / dataset / f"seed_{seed}"
            result_file = batch_dir / "jobs" / f"{job_id}.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_dir / "checkpoint_best_iou.pkl"
            checkpoint_path.touch()

            best_iou = 0.7000 + dataset_index * 0.0200 + seed_index * 0.0010
            best_pd = 0.9500 + seed_index * 0.0010
            best_fa = 10.0 - dataset_index - seed_index * 0.1
            lines = []
            for epoch in range(EXPECTED_EPOCHS):
                iou = best_iou - (EXPECTED_EPOCHS - 1 - epoch) * 0.0001
                pd = best_pd if epoch == EXPECTED_EPOCHS - 1 else best_pd - 0.01
                fa = best_fa if epoch == EXPECTED_EPOCHS - 1 else best_fa + 1.0
                lines.append(
                    f"2026-07-11-00-00-00 - {epoch:04d}\t - IoU {iou:.4f}"
                    f"\t - PD {pd:.4f}\t - FA {fa:.4f}"
                )
            (run_dir / "epoch_metric.log").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

            command = [
                "python",
                "main.py",
                "--mode",
                "train",
                "--model-type",
                "mshnet",
                "--epochs",
                str(EXPECTED_EPOCHS),
                "--seed",
                str(seed),
                "--run-label",
                job_id,
                "--run-dir",
                str(run_dir.resolve()),
            ]
            write_json(
                result_file,
                {
                    "job_id": job_id,
                    "returncode": 0,
                    "run_dir": str(run_dir.resolve()),
                    "command": command,
                },
            )
            job = {
                "job_id": job_id,
                "dataset": dataset,
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "result_file": str(result_file.resolve()),
            }
            jobs.append(job)
            checkpoints[checkpoint_path.resolve()] = {
                "epoch": EXPECTED_EPOCHS - 1,
                "iou": np.float64(best_iou),
                "pd": np.float64(best_pd),
                "fa": np.float64(best_fa),
                "best_iou": np.float64(best_iou),
                "method_meta": {
                    "method": "MSHNet",
                    "model_type": "mshnet",
                    "seed": seed,
                    "run_label": job_id,
                    "split_seed": 77,
                    "train_split_sha256": dataset_meta[dataset]["fit_sha256"],
                    "val_split_sha256": dataset_meta[dataset]["val_sha256"],
                    "test_split_sha256": dataset_meta[dataset]["official_test_sha256"],
                },
            }

    manifest = {
        "batch_id": batch_dir.name,
        "stage": "development_holdout_baseline",
        "official_test_policy": "loaded only for disjoint/hash audit; not iterated",
        "args": {
            "datasets": ",".join(DATASET_NAMES),
            "seeds": ",".join(str(seed) for seed in SEEDS),
            "epochs": EXPECTED_EPOCHS,
            "split_seed": 77,
        },
        "datasets": dataset_meta,
        "jobs": jobs,
    }
    write_json(batch_dir / "manifest.json", manifest)

    def checkpoint_loader(path: Path):
        return checkpoints[path.resolve()]

    return batch_dir, checkpoints, checkpoint_loader


def test_finalize_complete_grid_and_refuse_overwrite(tmp_path):
    batch_dir, _, checkpoint_loader = make_complete_batch(tmp_path)

    summary = finalize_batch(batch_dir, checkpoint_loader=checkpoint_loader)

    assert summary["status"] == "complete_and_validated"
    assert summary["official_test_status"].startswith("untouched")
    assert summary["datasets"]["NUAA-SIRST"]["mean"]["iou"] == pytest.approx(0.701)
    assert summary["datasets"]["NUAA-SIRST"]["std"]["iou"] == pytest.approx(0.001)
    assert (batch_dir / OUTPUT_JSON).is_file()
    markdown = (batch_dir / OUTPUT_MARKDOWN).read_text(encoding="utf-8")
    assert "official test sets remain untouched" in markdown
    assert "not official-test or paper-main-table results" in markdown

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_batch(batch_dir, checkpoint_loader=checkpoint_loader)


def test_finalize_fails_closed_on_checkpoint_identity_mismatch(tmp_path):
    batch_dir, checkpoints, checkpoint_loader = make_complete_batch(tmp_path)
    first_checkpoint = next(iter(checkpoints.values()))
    first_checkpoint["method_meta"]["run_label"] = "wrong-run"

    with pytest.raises(FinalizationError, match="run_label"):
        finalize_batch(batch_dir, checkpoint_loader=checkpoint_loader)

    assert not (batch_dir / OUTPUT_JSON).exists()
    assert not (batch_dir / OUTPUT_MARKDOWN).exists()


def test_safe_checkpoint_loader_maps_tensors_to_cpu(tmp_path):
    path = tmp_path / "checkpoint_best_iou.pkl"
    torch.save(
        {
            "iou": np.float64(0.75),
            "net": {"weight": torch.ones(1)},
            "method_meta": {"method": "MSHNet"},
        },
        path,
    )

    checkpoint = load_checkpoint_cpu(path)

    assert checkpoint["iou"] == pytest.approx(0.75)
    assert checkpoint["net"]["weight"].device.type == "cpu"
