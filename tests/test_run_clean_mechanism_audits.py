from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from tools.finalize_clean_baselines import DATASET_NAMES, EXPECTED_EPOCHS
from tools import run_clean_mechanism_audits as runner


SEEDS = (101, 102, 103)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def make_finalized_batch(tmp_path: Path):
    batch_dir = tmp_path / "fixture_batch"
    weights = tmp_path / "weights"
    datasets_meta: dict[str, dict] = {}
    jobs = []
    checkpoints: dict[Path, dict] = {}
    summary_datasets: dict[str, dict] = {}
    split_seed = 20260711

    for dataset_index, dataset in enumerate(DATASET_NAMES):
        dataset_dir = (tmp_path / "datasets" / dataset).resolve()
        train_file = f"img_idx/train_{dataset}.txt"
        test_file = f"img_idx/test_{dataset}.txt"
        datasets_meta[dataset] = {
            "dataset": dataset,
            "dataset_dir": str(dataset_dir),
            "fit_sha256": digest(f"{dataset}:fit"),
            "val_sha256": digest(f"{dataset}:val"),
            "official_test_sha256": digest(f"{dataset}:test"),
        }
        summary_runs = []
        for seed_index, seed in enumerate(SEEDS):
            job_id = f"mshnet__{dataset.lower()}__seed_{seed}"
            run_dir = (weights / dataset / f"seed_{seed}").resolve()
            result_file = (batch_dir / "jobs" / f"{job_id}.json").resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_dir / runner.CHECKPOINT_NAME
            checkpoint_path.write_bytes(f"{dataset}:{seed}:checkpoint".encode())

            best_iou = 0.70 + dataset_index * 0.02 + seed_index * 0.001
            best_pd = 0.95 + seed_index * 0.001
            best_fa = 10.0 - dataset_index - seed_index * 0.1
            lines = []
            for epoch in range(EXPECTED_EPOCHS):
                iou = best_iou - (EXPECTED_EPOCHS - 1 - epoch) * 0.0001
                pd = best_pd if epoch == EXPECTED_EPOCHS - 1 else best_pd - 0.01
                fa = best_fa if epoch == EXPECTED_EPOCHS - 1 else best_fa + 1.0
                lines.append(
                    f"2026-07-11 - {epoch:04d} - IoU {iou:.4f} "
                    f"- PD {pd:.4f} - FA {fa:.4f}"
                )
            (run_dir / "epoch_metric.log").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

            command = [
                "python", "main.py", "--mode", "train", "--model-type", "mshnet",
                "--epochs", str(EXPECTED_EPOCHS), "--seed", str(seed),
                "--run-label", job_id, "--run-dir", str(run_dir),
            ]
            write_json(
                result_file,
                {
                    "job_id": job_id,
                    "returncode": 0,
                    "run_dir": str(run_dir),
                    "command": command,
                },
            )
            job = {
                "job_id": job_id,
                "dataset": dataset,
                "seed": seed,
                "dataset_dir": str(dataset_dir),
                "train_file": train_file,
                "test_file": test_file,
                "run_dir": str(run_dir),
                "result_file": str(result_file),
            }
            jobs.append(job)
            checkpoint = {
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
                    "split_seed": split_seed,
                    "train_split_sha256": datasets_meta[dataset]["fit_sha256"],
                    "val_split_sha256": datasets_meta[dataset]["val_sha256"],
                    "test_split_sha256": datasets_meta[dataset]["official_test_sha256"],
                },
            }
            checkpoints[checkpoint_path.resolve()] = checkpoint
            write_json(
                run_dir / "run_config.json",
                {
                    "args": {
                        "mode": "train",
                        "model_type": "mshnet",
                        "deterministic": True,
                        "pin_memory": True,
                        "epochs": EXPECTED_EPOCHS,
                        "lr": 0.05,
                        "warm_epoch": 5,
                        "seed": seed,
                        "run_label": job_id,
                        "run_dir": str(run_dir),
                        "split_seed": split_seed,
                        "dataset_dir": str(dataset_dir),
                        "train_split_file": train_file,
                        "val_split_file": "",
                        "test_split_file": test_file,
                        "train_split_sha256": datasets_meta[dataset]["fit_sha256"],
                        "val_split_sha256": datasets_meta[dataset]["val_sha256"],
                        "test_split_sha256": datasets_meta[dataset]["official_test_sha256"],
                        "val_fraction": 0.2,
                        "base_size": 256,
                        "crop_size": 256,
                        "batch_size": 4,
                        "num_workers": 4,
                    }
                },
            )
            summary_runs.append(
                {
                    "seed": seed,
                    "best_epoch": EXPECTED_EPOCHS - 1,
                    "iou": best_iou,
                    "pd": best_pd,
                    "fa": best_fa,
                    "checkpoint": str(checkpoint_path),
                }
            )
        summary_datasets[dataset] = {
            "split_hashes": {
                "fit": datasets_meta[dataset]["fit_sha256"],
                "validation": datasets_meta[dataset]["val_sha256"],
                "official_test_audit_only": datasets_meta[dataset]["official_test_sha256"],
            },
            "runs": summary_runs,
        }

    manifest = {
        "batch_id": batch_dir.name,
        "stage": "development_holdout_baseline",
        "official_test_policy": "loaded only for disjoint/hash audit; not iterated",
        "args": {
            "datasets": ",".join(DATASET_NAMES),
            "seeds": ",".join(str(seed) for seed in SEEDS),
            "epochs": EXPECTED_EPOCHS,
            "batch_size": 4,
            "num_workers": 4,
            "lr": 0.05,
            "warm_epoch": 5,
            "val_fraction": 0.2,
            "split_seed": split_seed,
            "deterministic": "true",
        },
        "datasets": datasets_meta,
        "jobs": jobs,
    }
    write_json(batch_dir / "manifest.json", manifest)
    summary = {
        "schema_version": 1,
        "batch_id": batch_dir.name,
        "status": "complete_and_validated",
        "method": "mshnet",
        "model_type": "mshnet",
        "official_test_status": "untouched; not evaluated by this finalizer",
        "not_for_official_test_or_main_table_claims": True,
        "epochs_per_run": EXPECTED_EPOCHS,
        "seeds": list(SEEDS),
        "datasets": summary_datasets,
    }
    write_json(batch_dir / runner.BASELINE_SUMMARY_JSON, summary)

    def checkpoint_loader(path: Path):
        return checkpoints[path.resolve()]

    return batch_dir, checkpoint_loader


def command_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def make_completed_audit(job: dict) -> None:
    output_dir = Path(job["output_dir"])
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    array_path = arrays_dir / "sample.npz"
    array_path.write_bytes(b"fixture-array")
    array_hash = runner.sha256(array_path)
    array_size = array_path.stat().st_size
    image_row = {
        "image_id": "sample",
        "array_path": "arrays/sample.npz",
        "array_sha256": array_hash,
        "array_bytes": array_size,
    }
    images_path = output_dir / "images.jsonl"
    components_path = output_dir / "components.jsonl"
    images_path.write_text(json.dumps(image_row) + "\n", encoding="utf-8")
    components_path.write_text(json.dumps({"image_id": "sample"}) + "\n", encoding="utf-8")
    inventory = [
        {
            "image_id": "sample",
            "path": "arrays/sample.npz",
            "sha256": array_hash,
            "bytes": array_size,
        }
    ]
    inventory_hash = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metrics = job["baseline_metrics"]
    manifest = {
        "schema_version": runner.AUDIT_SCHEMA,
        "dataset": job["dataset"],
        "dataset_dir": job["config"]["dataset_dir"],
        "split_role": "val",
        "split_sha256": job["config"]["val_split_sha256"],
        "validation_split_sha256": job["config"]["val_split_sha256"],
        "seed": job["seed"],
        "method": "MSHNet",
        "model_type": "mshnet",
        "base_size": job["config"]["base_size"],
        "crop_size": job["config"]["crop_size"],
        "batch_size": job["config"]["batch_size"],
        "num_workers": job["config"]["num_workers"],
        "deterministic": True,
        "threshold_probability": 0.5,
        "threshold_logit": 0.0,
        "connectivity": 2,
        "max_centroid_distance": 3.0,
        "anchor_mode": "mean",
        "active_stage": 0,
        "eps": 1e-6,
        "candidate_probability_thresholds": [0.5, 0.3, 0.2, 0.1],
        "official_test_status": runner.OFFICIAL_TEST_STATUS,
        "source_sha256": job["source_sha256"],
        "checkpoint": {
            "role": "best_iou",
            "path": job["checkpoint"],
            "sha256": job["checkpoint_sha256"],
            "epoch": metrics["best_epoch"],
            "metrics": {key: metrics[key] for key in ("iou", "pd", "fa")},
        },
        "baseline_provenance": {
            "batch_id": job["batch_id"],
            "job_id": job["baseline_job_id"],
            "batch_manifest": job["baseline_manifest"],
            "baseline_summary": job["baseline_summary"],
            "completion": runner.GRID_COMPLETION,
        },
        "checkpoint_validation": {
            "model_seed_val_hash": "matched",
            "strict_state_dict": True,
            "frozen": True,
            "recomputed_metrics": {
                key: {"checkpoint": metrics[key], "recomputed": metrics[key]}
                for key in ("iou", "pd", "fa")
            },
        },
        "summary": {
            "images": 1,
            "pooled_iou": metrics["iou"],
            "pd": metrics["pd"],
            "fa_per_million": metrics["fa"],
        },
        "max_mobius_reconstruction_abs_error": 0.0,
        "artifacts": {
            "images_jsonl": "images.jsonl",
            "images_sha256": runner.sha256(images_path),
            "components_jsonl": "components.jsonl",
            "components_sha256": runner.sha256(components_path),
            "arrays_dir": "arrays",
            "array_count": 1,
            "array_inventory_sha256": inventory_hash,
            "array_total_bytes": array_size,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    log_path = Path(job["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("completed\n", encoding="utf-8")
    write_json(
        Path(job["result_file"]),
        {
            "schema_version": runner.RESULT_SCHEMA,
            "status": "completed_verified",
            "job_id": job["job_id"],
            "dataset": job["dataset"],
            "seed": job["seed"],
            "gpu": runner.FIXED_GPUS[0],
            "pid": 123,
            "started_at": "2026-07-11T00:00:00+00:00",
            "finished_at": "2026-07-11T00:00:01+00:00",
            "elapsed_seconds": 1.0,
            "returncode": 0,
            "command": job["command"],
            "output_dir": job["output_dir"],
            "log_file": job["log_file"],
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "source_sha256": job["source_sha256"],
        },
    )


def test_requires_completed_summary_and_builds_exact_3x3_commands(tmp_path: Path) -> None:
    batch_dir, loader = make_finalized_batch(tmp_path)
    summary_path = batch_dir / runner.BASELINE_SUMMARY_JSON
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_path.unlink()
    with pytest.raises(runner.AuditBatchError, match="missing completed clean baseline summary"):
        runner.load_validated_baseline_jobs(batch_dir, checkpoint_loader=loader)
    summary_path.write_text(summary_text, encoding="utf-8")

    baseline_jobs = runner.load_validated_baseline_jobs(
        batch_dir, checkpoint_loader=loader
    )
    audit_root = batch_dir / runner.AUDIT_DIR_NAME
    jobs = runner.build_audit_jobs(
        baseline_jobs,
        batch_id=batch_dir.name,
        audit_root=audit_root,
        python_executable=sys.executable,
    )

    assert len(jobs) == 9
    assert {(job["dataset"], job["seed"]) for job in jobs} == {
        (dataset, seed) for dataset in DATASET_NAMES for seed in SEEDS
    }
    assert len({job["output_dir"] for job in jobs}) == 9
    first = jobs[0]
    command = first["command"]
    assert command_value(command, "--mode") == "val"
    assert command_value(command, "--checkpoint-role") == "best_iou"
    assert command_value(command, "--batch-id") == batch_dir.name
    assert command_value(command, "--batch-size") == "4"
    assert command_value(command, "--train-split-file") == first["config"]["train_split_file"]
    assert command_value(command, "--val-split-file") == ""
    assert command_value(command, "--test-split-file") == first["config"]["test_split_file"]
    assert Path(first["output_dir"]) == (
        audit_root / "artifacts" / first["dataset"] / f"seed_{first['seed']}"
    )
    assert runner.FIXED_GPUS == (2, 3)


def test_safe_resume_verifies_result_and_every_array(tmp_path: Path) -> None:
    batch_dir, loader = make_finalized_batch(tmp_path)
    baseline_jobs = runner.load_validated_baseline_jobs(
        batch_dir, checkpoint_loader=loader
    )
    audit_root = batch_dir / runner.AUDIT_DIR_NAME
    jobs = runner.build_audit_jobs(
        baseline_jobs, batch_id=batch_dir.name, audit_root=audit_root
    )
    spec = runner.build_batch_spec(batch_dir, jobs)
    write_json(
        audit_root / runner.AUDIT_MANIFEST,
        {**spec, "created_at_utc": "2026-07-11T00:00:00+00:00"},
    )
    make_completed_audit(jobs[0])

    pending = runner.prepare_root(
        audit_root,
        spec=spec,
        jobs=jobs,
        resume=True,
        dry_run=True,
    )
    assert len(pending) == 8
    assert jobs[0] not in pending

    array_path = Path(jobs[0]["output_dir"]) / "arrays" / "sample.npz"
    array_path.write_bytes(b"tampered")
    with pytest.raises(runner.AuditBatchError, match="array integrity mismatch"):
        runner.classify_jobs(jobs, resume=True)


def test_refuses_partial_or_unrequested_nonempty_outputs(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    output.mkdir()
    (output / "partial.tmp").write_text("partial", encoding="utf-8")
    job = {
        "job_id": "fixture",
        "output_dir": str(output),
        "result_file": str(tmp_path / "fixture.json"),
        "log_file": str(tmp_path / "fixture.log"),
    }
    with pytest.raises(runner.AuditBatchError, match="refusing non-empty"):
        runner.classify_jobs([job], resume=False)
    with pytest.raises(runner.AuditBatchError, match="partial audit artifacts"):
        runner.classify_jobs([job], resume=True)


def test_scheduler_uses_one_process_per_fixed_gpu_and_captures_returncodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class FakeProcess:
        next_pid = 500

        def __init__(self, command, **kwargs):
            self.command = command
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.returncode = 0
            calls.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])

        def poll(self):
            return self.returncode

    jobs = []
    checkpoint = tmp_path / "checkpoint_best_iou.pkl"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = runner.sha256(checkpoint)
    for index in range(3):
        jobs.append(
            {
                "job_id": f"job_{index}",
                "dataset": "NUAA-SIRST",
                "seed": index,
                "command": ["fake", str(index)],
                "output_dir": str(tmp_path / "outputs" / str(index)),
                "log_file": str(tmp_path / "logs" / f"{index}.log"),
                "result_file": str(tmp_path / "jobs" / f"{index}.json"),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "source_sha256": runner.audit_source_sha256(),
            }
        )
    monkeypatch.setattr(runner, "verify_audit_output", lambda job: {})

    failures = runner.run_jobs(jobs, poll_seconds=0, popen=FakeProcess)

    assert failures == []
    assert calls == [
        str(runner.FIXED_GPUS[0]),
        str(runner.FIXED_GPUS[1]),
        str(runner.FIXED_GPUS[0]),
    ]
    for job in jobs:
        result = json.loads(Path(job["result_file"]).read_text(encoding="utf-8"))
        assert result["returncode"] == 0
        assert result["status"] == "completed_verified"
        assert result["gpu"] in runner.FIXED_GPUS
