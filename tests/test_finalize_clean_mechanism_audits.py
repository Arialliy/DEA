from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.finalize_clean_baselines import (
    DATASET_NAMES,
    EXPECTED_EPOCHS,
    finalize_batch as finalize_baselines,
)
from tools.finalize_clean_mechanism_audits import (
    OUTPUT_JSON,
    OUTPUT_MARKDOWN,
    FinalizationError,
    finalize_audits,
)


SEEDS = (101, 102, 103)
SOURCE_KEYS = (
    "exporter",
    "baseline_finalizer",
    "mshnet",
    "mean_anchor_probe",
    "component_candidates",
    "dataset",
    "metrics",
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_text(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_baseline(tmp_path: Path):
    batch_dir = tmp_path / "clean_baseline_holdout_v1"
    datasets = {}
    jobs = []
    checkpoints = {}
    for dataset in DATASET_NAMES:
        dataset_dir = tmp_path / "datasets" / dataset
        dataset_dir.mkdir(parents=True)
        datasets[dataset] = {
            "dataset": dataset,
            "dataset_dir": str(dataset_dir.resolve()),
            "fit_sha256": digest_text(f"{dataset}:fit"),
            "val_sha256": digest_text(f"{dataset}:val"),
            "official_test_sha256": digest_text(f"{dataset}:test"),
            "val_count": 1,
        }
        for seed in SEEDS:
            job_id = f"mshnet__{dataset.lower()}__seed_{seed}"
            run_dir = tmp_path / "weights" / dataset / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            checkpoint_path = run_dir / "checkpoint_best_iou.pkl"
            checkpoint_path.write_bytes(f"checkpoint:{dataset}:{seed}".encode())
            lines = []
            for epoch in range(EXPECTED_EPOCHS):
                iou = 1 / 3 if epoch == EXPECTED_EPOCHS - 1 else 0.25
                pd = 0.5
                fa = 250000.0
                lines.append(
                    f"2026-07-11-00-00-00 - {epoch:04d}\t - IoU {iou:.4f}"
                    f"\t - PD {pd:.4f}\t - FA {fa:.4f}"
                )
            (run_dir / "epoch_metric.log").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            result_file = batch_dir / "jobs" / f"{job_id}.json"
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
            jobs.append(
                {
                    "job_id": job_id,
                    "dataset": dataset,
                    "dataset_dir": str(dataset_dir.resolve()),
                    "seed": seed,
                    "run_dir": str(run_dir.resolve()),
                    "result_file": str(result_file.resolve()),
                }
            )
            checkpoints[checkpoint_path.resolve()] = {
                "epoch": EXPECTED_EPOCHS - 1,
                "iou": np.float64(1 / 3),
                "pd": np.float64(0.5),
                "fa": np.float64(250000.0),
                "best_iou": np.float64(1 / 3),
                "method_meta": {
                    "method": "MSHNet",
                    "model_type": "mshnet",
                    "seed": seed,
                    "run_label": job_id,
                    "split_seed": 77,
                    "train_split_sha256": datasets[dataset]["fit_sha256"],
                    "val_split_sha256": datasets[dataset]["val_sha256"],
                    "test_split_sha256": datasets[dataset]["official_test_sha256"],
                },
            }

    write_json(
        batch_dir / "manifest.json",
        {
            "batch_id": batch_dir.name,
            "stage": "development_holdout_baseline",
            "official_test_policy": "loaded only for disjoint/hash audit; not iterated",
            "args": {
                "datasets": ",".join(DATASET_NAMES),
                "seeds": ",".join(str(seed) for seed in SEEDS),
                "epochs": EXPECTED_EPOCHS,
                "split_seed": 77,
            },
            "datasets": datasets,
            "jobs": jobs,
        },
    )

    def checkpoint_loader(path: Path):
        return checkpoints[path.resolve()]

    finalize_baselines(batch_dir, checkpoint_loader=checkpoint_loader)
    return batch_dir, datasets, jobs, checkpoint_loader


def component_row(
    image_id: str,
    domain: str,
    role: str,
    component_id: int,
    *,
    recoverable: bool | None = None,
) -> dict:
    row = {
        "image_id": image_id,
        "domain": domain,
        "role": role,
        "component_index": component_id - 1,
        "component_id": component_id,
        "label": component_id,
        "area": 1,
        "p_z_mean": 0.25,
        "j_z_mean": -0.25,
        "interaction_ratio_mean": 0.5,
        "interaction_ratio_p95": 0.5,
        "mean_anchor_score_mean": 0.5,
        "conflict_pixels": 1,
        "conflict_fraction": 1.0,
        "prediction_logit_mean": 0.1,
    }
    if recoverable is not None:
        row["recoverable"] = recoverable
    return row


def make_audit_grid(
    batch_dir: Path,
    datasets: dict,
    baseline_jobs: list[dict],
) -> Path:
    audit_root = batch_dir / "mechanism_audits"
    (audit_root / "logs").mkdir(parents=True)
    (audit_root / "jobs").mkdir()
    sources = {key: digest_text(f"source:{key}") for key in SOURCE_KEYS}
    batch_jobs = []
    job_by_pair = {(job["dataset"], job["seed"]): job for job in baseline_jobs}
    summary = json.loads((batch_dir / "clean_baseline_holdout_summary.json").read_text())
    summary_run = {
        (dataset, run["seed"]): run
        for dataset in DATASET_NAMES
        for run in summary["datasets"][dataset]["runs"]
    }

    for dataset in DATASET_NAMES:
        for seed in SEEDS:
            baseline_job = job_by_pair[(dataset, seed)]
            checkpoint = Path(baseline_job["run_dir"]) / "checkpoint_best_iou.pkl"
            job_id = f"mean_anchor__{dataset.lower()}__seed_{seed}"
            output_dir = audit_root / "artifacts" / dataset / f"seed_{seed}"
            arrays_dir = output_dir / "arrays"
            arrays_dir.mkdir(parents=True)
            image_id = f"image_{seed}"
            array_path = arrays_dir / f"{image_id}.npz"
            arrays = {key: np.zeros((2, 2), dtype=np.float32) for key in (
                "z11", "z10", "z01", "z00", "p_z", "j_z", "p_feature_rms",
                "j_feature_rms", "ratio", "conflict_score", "conflict_mask",
                "pred_logit", "pred_probability", "prediction_mask", "target_mask",
                "target_component_labels", "prediction_component_labels",
                "candidate_component_labels", "recoverable_fn_mask",
            )}
            arrays["scale_logits"] = np.zeros((4, 2, 2), dtype=np.float32)
            np.savez_compressed(array_path, **arrays)
            array_hash = digest_file(array_path)
            array_bytes = array_path.stat().st_size
            image = {
                "image_index": 0,
                "image_id": image_id,
                "intersection_pixels": 1,
                "union_pixels": 3,
                "ground_truth_positive_pixels": 2,
                "predicted_positive_pixels": 2,
                "false_positive_pixels": 1,
                "false_negative_pixels": 1,
                "target_component_count": 2,
                "true_positive_component_count": 1,
                "false_negative_component_count": 1,
                "prediction_component_count": 2,
                "matched_prediction_component_count": 1,
                "false_positive_component_count": 1,
                "false_positive_component_area": 1,
                "recoverable_fn_component_count": 1,
                "recoverable_fn_target_component_area": 1,
                "candidate_component_count": 1,
                "conflict_pixels": 3,
                "conflict_on_true_positive_pixels": 1,
                "conflict_on_false_positive_pixels": 1,
                "conflict_on_false_negative_pixels": 1,
                "iou": 1 / 3,
                "mean_anchor_index": 0.5,
                "interaction_ratio_mean": 0.5,
                "interaction_ratio_p95": 0.5,
                "conflict_fraction": 0.75,
                "array_path": f"arrays/{image_id}.npz",
                "array_sha256": array_hash,
                "array_bytes": array_bytes,
            }
            components = [
                component_row(image_id, "target", "tp_target", 1, recoverable=False),
                component_row(image_id, "target", "fn_target", 2, recoverable=True),
                component_row(image_id, "prediction", "matched_pred", 1),
                component_row(image_id, "prediction", "fp_pred", 2),
                component_row(image_id, "candidate", "candidate", 1),
            ]
            images_path, components_path = output_dir / "images.jsonl", output_dir / "components.jsonl"
            write_jsonl(images_path, [image])
            write_jsonl(components_path, components)
            inventory = [{
                "image_id": image_id,
                "path": f"arrays/{image_id}.npz",
                "sha256": array_hash,
                "bytes": array_bytes,
            }]
            inventory_hash = digest_bytes(
                json.dumps(
                    inventory, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            )
            eps = 1e-6
            raw_summary = {
                **{key: image[key] for key in (
                    "intersection_pixels", "union_pixels", "ground_truth_positive_pixels",
                    "predicted_positive_pixels", "false_positive_pixels",
                    "false_negative_pixels", "target_component_count",
                    "true_positive_component_count", "false_negative_component_count",
                    "prediction_component_count", "matched_prediction_component_count",
                    "false_positive_component_count", "false_positive_component_area",
                    "recoverable_fn_component_count", "recoverable_fn_target_component_area",
                    "candidate_component_count", "conflict_pixels",
                    "conflict_on_true_positive_pixels", "conflict_on_false_positive_pixels",
                    "conflict_on_false_negative_pixels",
                )},
                "images": 1,
                "pixels": 4,
                "p_rms_sum": 8.0,
                "j_rms_sum": 4.0,
                "score_sum": 2.0,
                "mean_anchor_score_sum_true_positive": 0.5,
                "mean_anchor_score_sum_false_positive": 0.5,
                "mean_anchor_score_sum_false_negative": 0.5,
                "pooled_iou": 1 / 3,
                "pd": 0.5,
                "fa_per_million": 250000.0,
                "recoverable_fn_fraction": 1.0,
                "conflict_fraction": 0.75,
                "mean_anchor_index": 0.5,
                "global_r_ratio_of_sums": 4.0 / (8.0 + eps),
                "conflict_true_positive_coverage": 1.0,
                "conflict_false_positive_coverage": 1.0,
                "conflict_false_negative_coverage": 1.0,
            }
            baseline_metric = summary_run[(dataset, seed)]
            manifest = {
                "schema_version": "dea.clean_mechanism_audit.v1",
                "dataset": dataset,
                "dataset_dir": datasets[dataset]["dataset_dir"],
                "split_role": "val",
                "split_sha256": datasets[dataset]["val_sha256"],
                "validation_split_sha256": datasets[dataset]["val_sha256"],
                "seed": seed,
                "method": "MSHNet",
                "model_type": "mshnet",
                "checkpoint": {
                    "role": "best_iou",
                    "path": str(checkpoint.resolve()),
                    "sha256": digest_file(checkpoint),
                    "epoch": EXPECTED_EPOCHS - 1,
                    "metrics": {
                        "iou": baseline_metric["iou"],
                        "pd": baseline_metric["pd"],
                        "fa": baseline_metric["fa"],
                        "best_iou": baseline_metric["iou"],
                    },
                },
                "threshold_probability": 0.5,
                "threshold_logit": 0.0,
                "connectivity": 2,
                "max_centroid_distance": 3.0,
                "base_size": 2,
                "crop_size": 2,
                "batch_size": 1,
                "num_workers": 0,
                "deterministic": True,
                "anchor_mode": "mean",
                "active_stage": 0,
                "eps": eps,
                "candidate_probability_thresholds": [0.5, 0.3, 0.2, 0.1],
                "recoverable_fn_definition": "prediction-only recoverable FN definition",
                "conflict_definition": "fixed mean-anchor conflict definition",
                "global_ratio_of_sums_definition": "fixed interaction ratio-of-sums definition",
                "source_sha256": sources,
                "official_test_status": "sealed; this exporter accepts development validation only",
                "baseline_provenance": {
                    "batch_id": batch_dir.name,
                    "job_id": baseline_job["job_id"],
                    "batch_manifest": str((batch_dir / "manifest.json").resolve()),
                    "baseline_summary": str((batch_dir / "clean_baseline_holdout_summary.json").resolve()),
                    "completion": "all_3x3_jobs_400_epochs_returncode_0_and_finalizer_validated",
                },
                "checkpoint_validation": {
                    "model_seed_val_hash": "matched",
                    "strict_state_dict": True,
                    "frozen": True,
                    "recomputed_metrics": {
                        "iou": {"checkpoint": 1 / 3, "recomputed": 1 / 3},
                        "pd": {"checkpoint": 0.5, "recomputed": 0.5},
                        "fa": {"checkpoint": 250000.0, "recomputed": 250000.0},
                    },
                },
                "artifacts": {
                    "images_jsonl": "images.jsonl",
                    "images_sha256": digest_file(images_path),
                    "components_jsonl": "components.jsonl",
                    "components_sha256": digest_file(components_path),
                    "arrays_dir": "arrays",
                    "array_count": 1,
                    "array_inventory_sha256": inventory_hash,
                    "array_total_bytes": array_bytes,
                },
                "max_mobius_reconstruction_abs_error": 0.0,
                "summary": raw_summary,
            }
            write_json(output_dir / "manifest.json", manifest)

            log_path = audit_root / "logs" / f"{job_id}.log"
            result_path = audit_root / "jobs" / f"{job_id}.json"
            log_path.write_text("completed\n", encoding="utf-8")
            config = {
                "dataset_dir": datasets[dataset]["dataset_dir"],
                "val_split_sha256": datasets[dataset]["val_sha256"],
                "base_size": 2,
                "crop_size": 2,
                "batch_size": 1,
                "num_workers": 0,
            }
            audit_job = {
                "job_id": job_id,
                "dataset": dataset,
                "seed": seed,
                "output_dir": str(output_dir.resolve()),
                "log_file": str(log_path.resolve()),
                "result_file": str(result_path.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": digest_file(checkpoint),
                "baseline_metrics": {
                    "best_epoch": EXPECTED_EPOCHS - 1,
                    "iou": 1 / 3,
                    "pd": 0.5,
                    "fa": 250000.0,
                },
                "config": config,
            }
            batch_jobs.append(audit_job)
            write_json(
                result_path,
                {
                    "schema_version": "dea.clean_mechanism_audit_job.v1",
                    "status": "completed_verified",
                    "job_id": job_id,
                    "dataset": dataset,
                    "seed": seed,
                    "returncode": 0,
                    "output_dir": str(output_dir.resolve()),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": digest_file(checkpoint),
                    "source_sha256": sources,
                },
            )

    baseline_manifest = batch_dir / "manifest.json"
    baseline_summary = batch_dir / "clean_baseline_holdout_summary.json"
    write_json(
        audit_root / "batch_manifest.json",
        {
            "schema_version": "dea.clean_mechanism_audit_batch.v1",
            "batch_id": batch_dir.name,
            "stage": "development_holdout_mechanism_audit",
            "official_test_policy": (
                "validation mode only; official-test split path is propagated for frozen "
                "provenance checking and is never opened or iterated"
            ),
            "gpu_ids": [0, 1],
            "max_processes_per_gpu": 1,
            "baseline_manifest": str(baseline_manifest.resolve()),
            "baseline_manifest_sha256": digest_file(baseline_manifest),
            "baseline_summary": str(baseline_summary.resolve()),
            "baseline_summary_sha256": digest_file(baseline_summary),
            "source_sha256": sources,
            "jobs": batch_jobs,
        },
    )
    return audit_root


def make_complete_grid(tmp_path: Path):
    batch_dir, datasets, jobs, loader = make_baseline(tmp_path)
    audit_root = make_audit_grid(batch_dir, datasets, jobs)
    return batch_dir, audit_root, loader


def test_finalize_complete_grid_uses_ratio_of_sums_and_scope_guards(tmp_path: Path) -> None:
    batch_dir, audit_root, loader = make_complete_grid(tmp_path)

    summary = finalize_audits(
        batch_dir, audit_root=audit_root, checkpoint_loader=loader
    )

    assert summary["status"] == "complete_and_validated"
    assert summary["validated_grid"] == {
        "dataset_count": 3,
        "seed_count": 3,
        "run_count": 9,
    }
    assert summary["dea_evaluated"] is False
    assert summary["dea_gain_claimed"] is False
    assert summary["overall"]["counts"]["false_positive_component_count"] == 9
    assert summary["overall"]["counts"]["recoverable_fn_component_count"] == 9
    assert summary["overall"]["ratio_of_sums_metrics"]["pooled_iou"] == pytest.approx(1 / 3)
    assert summary["by_dataset"]["NUAA-SIRST"]["run_count"] == 3
    assert summary["by_seed"]["101"]["run_count"] == 3
    assert (audit_root / OUTPUT_JSON).is_file()
    markdown = (audit_root / OUTPUT_MARKDOWN).read_text(encoding="utf-8")
    assert "official test sets remain sealed" in markdown
    assert "No DEA model was evaluated" in markdown
    assert "no DEA gain" in markdown

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_audits(batch_dir, audit_root=audit_root, checkpoint_loader=loader)


def test_finalize_fails_closed_on_missing_array_without_partial_summary(tmp_path: Path) -> None:
    batch_dir, audit_root, loader = make_complete_grid(tmp_path)
    missing = next((audit_root / "artifacts").rglob("*.npz"))
    missing.unlink()

    with pytest.raises(FinalizationError, match="missing .*array_path|arrays directory"):
        finalize_audits(batch_dir, audit_root=audit_root, checkpoint_loader=loader)

    assert not (audit_root / OUTPUT_JSON).exists()
    assert not (audit_root / OUTPUT_MARKDOWN).exists()


def test_finalize_fails_closed_on_source_provenance_mismatch(tmp_path: Path) -> None:
    batch_dir, audit_root, loader = make_complete_grid(tmp_path)
    audit_manifest = next((audit_root / "artifacts").rglob("manifest.json"))
    payload = json.loads(audit_manifest.read_text(encoding="utf-8"))
    payload["source_sha256"]["metrics"] = digest_text("different source")
    write_json(audit_manifest, payload)

    with pytest.raises(FinalizationError, match="source hashes disagree"):
        finalize_audits(batch_dir, audit_root=audit_root, checkpoint_loader=loader)

    assert not (audit_root / OUTPUT_JSON).exists()
    assert not (audit_root / OUTPUT_MARKDOWN).exists()
