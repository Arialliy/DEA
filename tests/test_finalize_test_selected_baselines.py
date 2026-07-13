from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import finalize_test_selected_baselines as finalizer


SEEDS = (101, 102, 103)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _names_hash(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()


def _method_meta(
    job: dict[str, object], dataset_meta: dict[str, object], args: dict[str, object]
) -> dict[str, object]:
    dataset_dir = Path(str(job["dataset_dir"])).resolve()
    return {
        "protocol": finalizer.PROTOCOL,
        "protocol_version": finalizer.PROTOCOL,
        "method": "MSHNet",
        "model_type": "mshnet",
        "mshnet_objective": "sls",
        "mshnet_side_supervision": "canonical",
        "mshnet_train_graph": "canonical_warm",
        "location_loss": "legacy",
        "side_location_loss": "same",
        "lambda_location": 1.0,
        "crwd_lambda": 0.0,
        "crwd_enabled": False,
        "dea_lite_enabled": False,
        "dea_lambda_single": 0.0,
        "dea_lambda_dec": 0.0,
        "dea_lambda_empty": 0.0,
        "dataset_name": job["dataset"],
        "dataset_dir": str(dataset_dir),
        "canonical_datasets_root": str(dataset_dir.parent),
        "train_split_file": str(dataset_dir / str(job["train_split_arg"])),
        "test_split_file": str(dataset_dir / str(job["test_split_arg"])),
        "val_split_file": "",
        "val_split_sha256": "",
        "val_split_count": 0,
        "val_fraction": 0.0,
        "train_split_count": dataset_meta["train_count"],
        "test_split_count": dataset_meta["test_count"],
        "train_split_raw_sha256": job["train_split_raw_sha256"],
        "test_split_raw_sha256": job["test_split_raw_sha256"],
        "train_split_normalized_sha256": job["train_split_sha256"],
        "test_split_normalized_sha256": job["test_split_sha256"],
        "train_split_sha256": job["train_split_sha256"],
        "test_split_sha256": job["test_split_sha256"],
        "seed": job["seed"],
        "deterministic": True,
        "run_label": job["job_id"],
        "selection_split": "test",
        "evaluation_split": "test",
        "evaluation_alias": "val_loader_is_complete_canonical_test",
        "no_internal_holdout": True,
        "test_interval": finalizer.TEST_INTERVAL,
        "evaluation_epoch_rule": finalizer.EVALUATION_EPOCH_RULE,
        "selection_threshold": finalizer.SELECTION_THRESHOLD,
        "selection_prediction_rule": "sigmoid(logit) > 0.5",
        "selection_tie_break": finalizer.TIE_BREAK,
        "selection_best_iou_rule": finalizer.BEST_IOU_RULE,
        "selection_pd_fa_rule": finalizer.PD_FA_RULE,
        "selection_pd_fa_min_pd": finalizer.PD_FA_MIN_PD,
        "selection_pd_fa_min_iou": finalizer.PD_FA_MIN_IOU,
        "selection_paired_baseline_iou": finalizer.PAIRED_BASELINE_IOU,
        "train_loader_drop_last": False,
        "warm_epoch": args["warm_epoch"],
        "init_from_baseline": "",
        "init_checkpoint_sha256": "",
    }


def _command(job: dict[str, object]) -> list[str]:
    return [
        "python",
        "tools/train_test_selected_full_train.py",
        "--mode",
        "train",
        "--model-type",
        "mshnet",
        "--dataset-dir",
        str(job["dataset_dir"]),
        "--train-split-file",
        str(job["train_split_arg"]),
        "--test-split-file",
        str(job["test_split_arg"]),
        "--val-fraction",
        "0",
        "--seed",
        str(job["seed"]),
        "--deterministic",
        "true",
        "--epochs",
        "400",
        "--test-interval",
        "10",
        "--batch-size",
        "4",
        "--num-workers",
        "4",
        "--lr",
        "0.05",
        "--warm-epoch",
        "5",
        "--threshold",
        "0.5",
        "--pd-fa-min-pd",
        "0.93",
        "--pd-fa-min-iou",
        "0.655",
        "--paired-baseline-iou",
        "0.0",
        "--selection-tie-break",
        "earliest_epoch",
        "--run-dir",
        str(job["run_dir"]),
        "--run-label",
        str(job["job_id"]),
    ]


def make_batch(
    tmp_path: Path,
    *,
    missing_constrained: set[tuple[str, int]] | None = None,
):
    missing_constrained = missing_constrained or set()
    batch_dir = tmp_path / "fixture_batch"
    dataset_root = tmp_path / "datasets"
    source_root = tmp_path / "tools"
    scheduler = source_root / "run_test_selected_baselines.py"
    training_entry = source_root / "train_test_selected_full_train.py"
    source_root.mkdir(parents=True)
    scheduler.write_text("# frozen scheduler\n", encoding="utf-8")
    training_entry.write_text("# frozen trainer\n", encoding="utf-8")

    expected_datasets: dict[str, dict[str, dict[str, object]]] = {}
    manifest_datasets: dict[str, dict[str, object]] = {}
    split_payloads: dict[str, dict[str, bytes]] = {}
    for dataset_index, dataset in enumerate(finalizer.DATASET_NAMES):
        dataset_dir = dataset_root / dataset
        idx = dataset_dir / "img_idx"
        idx.mkdir(parents=True)
        train_names = [f"train_{dataset_index}_a", f"train_{dataset_index}_b"]
        test_names = [f"test_{dataset_index}_a", f"test_{dataset_index}_b"]
        roles = {"train": train_names, "test": test_names}
        expected_datasets[dataset] = {}
        split_payloads[dataset] = {}
        for role, names in roles.items():
            path = idx / f"{role}_{dataset}.txt"
            payload = ("\n".join(names) + "\n").encode()
            path.write_bytes(payload)
            split_payloads[dataset][role] = payload
            expected_datasets[dataset][role] = {
                "count": len(names),
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
                "ordered_names_sha256": _names_hash(names),
            }
        train = expected_datasets[dataset]["train"]
        test = expected_datasets[dataset]["test"]
        manifest_datasets[dataset] = {
            "dataset": dataset,
            "dataset_dir": str(dataset_dir.resolve()),
            "train_split_file": str((idx / f"train_{dataset}.txt").resolve()),
            "test_split_file": str((idx / f"test_{dataset}.txt").resolve()),
            "train_split_arg": f"img_idx/train_{dataset}.txt",
            "test_split_arg": f"img_idx/test_{dataset}.txt",
            "train_count": train["count"],
            "test_count": test["count"],
            "train_test_overlap_count": 0,
            "train_raw_sha256": train["raw_sha256"],
            "test_raw_sha256": test["raw_sha256"],
            "train_ordered_names_sha256": train["ordered_names_sha256"],
            "test_ordered_names_sha256": test["ordered_names_sha256"],
            "validation_split": None,
            "hcval_policy": "prohibited_not_read",
        }

    args: dict[str, object] = {
        "batch_id": batch_dir.name,
        "batch_size": 4,
        "datasets": ",".join(finalizer.DATASET_NAMES),
        "deterministic": "true",
        "dry_run": False,
        "epochs": 400,
        "gpus": "2,3",
        "lr": 0.05,
        "num_workers": 4,
        "paired_baseline_iou": 0.0,
        "pd_fa_min_iou": 0.655,
        "pd_fa_min_pd": 0.93,
        "resume": False,
        "seeds": ",".join(str(seed) for seed in SEEDS),
        "test_interval": 10,
        "threshold": 0.5,
        "val_fraction": 0.0,
        "warm_epoch": 5,
    }

    jobs: list[dict[str, object]] = []
    checkpoints: dict[Path, dict[str, object]] = {}
    for seed_index, seed in enumerate(SEEDS):
        for dataset_index, dataset in enumerate(finalizer.DATASET_NAMES):
            dataset_meta = manifest_datasets[dataset]
            job_id = f"mshnet__{dataset.lower()}__seed_{seed}"
            run_dir = tmp_path / "weights" / dataset / f"seed_{seed}"
            result_file = batch_dir / "jobs" / f"{job_id}.json"
            job: dict[str, object] = {
                "job_id": job_id,
                "protocol": finalizer.PROTOCOL,
                "dataset": dataset,
                "seed": seed,
                "dataset_dir": dataset_meta["dataset_dir"],
                "train_split_arg": dataset_meta["train_split_arg"],
                "test_split_arg": dataset_meta["test_split_arg"],
                "train_split_sha256": dataset_meta["train_ordered_names_sha256"],
                "test_split_sha256": dataset_meta["test_ordered_names_sha256"],
                "train_split_raw_sha256": dataset_meta["train_raw_sha256"],
                "test_split_raw_sha256": dataset_meta["test_raw_sha256"],
                "deterministic": True,
                "run_dir": str(run_dir.resolve()),
                "log_file": str((batch_dir / "logs" / f"{job_id}.log").resolve()),
                "result_file": str(result_file.resolve()),
                "total_epochs": 400,
                "test_interval": 10,
                "test_evaluation_epochs": list(finalizer.EVALUATION_COMPLETED_EPOCHS),
            }
            jobs.append(job)
            run_dir.mkdir(parents=True)
            (run_dir / "split_train.txt").write_bytes(split_payloads[dataset]["train"])
            (run_dir / "split_test.txt").write_bytes(split_payloads[dataset]["test"])

            method_meta = _method_meta(job, dataset_meta, args)
            config_args = {
                "mode": "train",
                "model_type": "mshnet",
                "mshnet_objective": "sls",
                "mshnet_side_supervision": "canonical",
                "mshnet_train_graph": "canonical_warm",
                "location_loss": "legacy",
                "side_location_loss": "same",
                "lambda_location": 1.0,
                "crwd_lambda": 0.0,
                "epochs": 400,
                "test_interval": 10,
                "threshold": 0.5,
                "pd_fa_min_pd": 0.93,
                "pd_fa_min_iou": 0.655,
                "paired_baseline_iou": 0.0,
                "seed": seed,
                "deterministic": True,
                "run_label": job_id,
                "run_dir": str(run_dir.resolve()),
                "dataset_dir": dataset_meta["dataset_dir"],
                "train_split_file": str(
                    Path(str(dataset_meta["dataset_dir"]))
                    / str(dataset_meta["train_split_arg"])
                ),
                "test_split_file": str(
                    Path(str(dataset_meta["dataset_dir"]))
                    / str(dataset_meta["test_split_arg"])
                ),
                "val_split_file": "",
                "val_fraction": 0.0,
                "warm_epoch": 5,
                "batch_size": 4,
                "num_workers": 4,
                "lr": 0.05,
                "init_from_baseline": "",
            }
            _write_json(
                run_dir / "run_config.json",
                {"args": config_args, "method_meta": method_meta},
            )

            best_iou = 0.7000 + dataset_index * 0.01 + seed_index * 0.001
            pdfa_fa = 10.0 - dataset_index - seed_index * 0.1
            rows: list[tuple[float, float, float]] = []
            for row_index in range(40):
                iou, pd, fa = 0.5000 + row_index * 0.0001, 0.80, 100.0 + row_index
                # Two equal persisted maxima test the earliest-epoch tie rule.
                if row_index in (12, 14):
                    iou, pd, fa = best_iou, 0.80, 50.0
                # Two eligible equal-FA rows test the earliest constrained tie.
                if row_index in (20, 21) and (dataset, seed) not in missing_constrained:
                    iou, pd, fa = 0.6600, 0.9400, pdfa_fa
                rows.append((iou, pd, fa))
            metric_lines = [
                f"2026-07-13-00-00-{index:02d} - {epoch:04d}\t - IoU {iou:.4f}"
                f"\t - PD {pd:.4f}\t - FA {fa:.4f}"
                for index, (epoch, (iou, pd, fa)) in enumerate(
                    zip(finalizer.EVALUATION_ZERO_BASED_EPOCHS, rows)
                )
            ]
            (run_dir / "epoch_metric.log").write_text(
                "\n".join(metric_lines) + "\n", encoding="utf-8"
            )

            best_epoch = finalizer.EVALUATION_ZERO_BASED_EPOCHS[12]
            best_path = run_dir / "checkpoint_best_iou.pkl"
            best_path.write_bytes(f"best:{job_id}".encode())
            best_checkpoint = {
                "epoch": best_epoch,
                "iou": best_iou,
                "pd": 0.80,
                "fa": 50.0,
                "best_iou": best_iou,
                "method_meta": method_meta,
            }
            checkpoints[best_path.resolve()] = best_checkpoint
            best_selection = {
                "status": "found",
                "file": "checkpoint_best_iou.pkl",
                "sha256": _sha(best_path),
                "epoch_zero_based": best_epoch,
                "iou": best_iou,
                "pd": 0.80,
                "fa": 50.0,
            }

            if (dataset, seed) in missing_constrained:
                constrained_selection: dict[str, object] = {
                    "status": "not_found",
                    "file": None,
                    "reason": "no_eligible_epoch",
                }
            else:
                pdfa_epoch = finalizer.EVALUATION_ZERO_BASED_EPOCHS[20]
                pdfa_path = run_dir / "checkpoint_pd_fa_best.pkl"
                pdfa_path.write_bytes(f"pdfa:{job_id}".encode())
                checkpoints[pdfa_path.resolve()] = {
                    "epoch": pdfa_epoch,
                    "iou": 0.6600,
                    "pd": 0.9400,
                    "fa": pdfa_fa,
                    "best_iou": best_iou,
                    "best_pd_fa": pdfa_fa,
                    "best_pd_fa_iou": 0.6600,
                    "best_pd_fa_pd": 0.9400,
                    "best_pd_fa_epoch": pdfa_epoch,
                    "method_meta": method_meta,
                }
                constrained_selection = {
                    "status": "found",
                    "file": "checkpoint_pd_fa_best.pkl",
                    "sha256": _sha(pdfa_path),
                    "epoch_zero_based": pdfa_epoch,
                    "iou": 0.6600,
                    "pd": 0.9400,
                    "fa": pdfa_fa,
                }
            _write_json(
                run_dir / "protocol_summary.json",
                {
                    "protocol": finalizer.PROTOCOL,
                    "status": "complete",
                    "dataset": dataset,
                    "run_dir": str(run_dir.resolve()),
                    "selection_split": "test",
                    "no_internal_holdout": True,
                    "test_interval": 10,
                    "evaluation_epoch_rule": finalizer.EVALUATION_EPOCH_RULE,
                    "start_epoch": 0,
                    "total_epochs": 400,
                    "planned_evaluation_epochs_zero_based": list(
                        finalizer.EVALUATION_ZERO_BASED_EPOCHS
                    ),
                    "current_process_evaluation_epochs_zero_based": list(
                        finalizer.EVALUATION_ZERO_BASED_EPOCHS
                    ),
                    "executed_evaluation_epochs_zero_based": list(
                        finalizer.EVALUATION_ZERO_BASED_EPOCHS
                    ),
                    "last_completed_epoch_zero_based": 399,
                    "checkpoint_selection": {
                        "best_iou": best_selection,
                        "constrained_min_fa": constrained_selection,
                    },
                    "method_meta": method_meta,
                },
            )
            _write_json(
                result_file,
                {
                    "protocol": finalizer.PROTOCOL,
                    "job_id": job_id,
                    "dataset": dataset,
                    "seed": seed,
                    "returncode": 0,
                    "command": _command(job),
                    "total_epochs": 400,
                    "test_interval": 10,
                    "test_evaluation_epochs": list(
                        finalizer.EVALUATION_COMPLETED_EPOCHS
                    ),
                    "test_selected": True,
                    "run_dir": str(run_dir.resolve()),
                },
            )

    source_hashes = {
        "scheduler": _sha(scheduler),
        "training_entry": _sha(training_entry),
    }
    immutable_jobs = [
        {
            key: job[key]
            for key in (
                "job_id",
                "dataset",
                "seed",
                "dataset_dir",
                "train_split_arg",
                "test_split_arg",
                "train_split_sha256",
                "test_split_sha256",
                "train_split_raw_sha256",
                "test_split_raw_sha256",
                "deterministic",
                "run_dir",
                "total_epochs",
                "test_interval",
                "test_evaluation_epochs",
            )
        }
        for job in jobs
    ]
    immutable = {
        "protocol": finalizer.PROTOCOL,
        "model": "MSHNet baseline",
        "args_excluding_resume_and_dry_run": {
            key: value for key, value in args.items() if key not in {"resume", "dry_run"}
        },
        "datasets": manifest_datasets,
        "jobs": immutable_jobs,
        "checkpoint_selector": {
            "threshold": 0.5,
            "pd_fa_min_pd": 0.93,
            "pd_fa_min_iou": 0.655,
            "paired_baseline_iou": 0.0,
            "tie_break": "earliest_epoch",
        },
        "source_sha256": source_hashes,
    }
    manifest = {
        "protocol": finalizer.PROTOCOL,
        "batch_id": batch_dir.name,
        "model": "MSHNet baseline",
        "stage": finalizer.EXPECTED_STAGE,
        "args": args,
        "data_protocol": {
            "dataset_root": str(dataset_root.resolve()),
            "fit_role": "complete canonical img_idx/train_<dataset>.txt",
            "validation_role": None,
            "validation_fraction": 0.0,
            "test_role": "periodic canonical test selection",
            "hcval_policy": "prohibited_not_read",
        },
        "selection_policy": {
            "test_selected": True,
            "test_is_untouched": False,
            "test_interval_completed_epochs": 10,
            "test_evaluation_epochs": list(finalizer.EVALUATION_COMPLETED_EPOCHS),
            "eligible_checkpoint_epochs": list(
                finalizer.EVALUATION_COMPLETED_EPOCHS
            ),
            "metric_threshold": 0.5,
            "pd_fa_min_pd": 0.93,
            "pd_fa_min_iou": 0.655,
            "paired_baseline_iou": 0.0,
            "strict_tie_break": "earliest_epoch",
        },
        "sources": {
            "scheduler": str(scheduler.resolve()),
            "scheduler_sha256": source_hashes["scheduler"],
            "training_entry": str(training_entry.resolve()),
            "training_entry_sha256": source_hashes["training_entry"],
        },
        "immutable_contract": immutable,
        "immutable_contract_sha256": finalizer.canonical_json_sha256(immutable),
        "datasets": manifest_datasets,
        "jobs": jobs,
    }
    _write_json(batch_dir / "manifest.json", manifest)

    def checkpoint_loader(path: Path):
        return checkpoints[path.resolve()]

    return {
        "batch_dir": batch_dir,
        "expected_datasets": expected_datasets,
        "checkpoints": checkpoints,
        "checkpoint_loader": checkpoint_loader,
        "scheduler": scheduler,
        "training_entry": training_entry,
        "dataset_root": dataset_root,
    }


def _finalize(fixture: dict[str, object]):
    return finalizer.finalize_batch(
        fixture["batch_dir"],  # type: ignore[arg-type]
        checkpoint_loader=fixture["checkpoint_loader"],  # type: ignore[arg-type]
        expected_datasets=fixture["expected_datasets"],  # type: ignore[arg-type]
        expected_dataset_root=fixture["dataset_root"],  # type: ignore[arg-type]
        expected_source_paths={
            "scheduler": fixture["scheduler"],  # type: ignore[dict-item]
            "training_entry": fixture["training_entry"],  # type: ignore[dict-item]
        },
    )


def test_complete_grid_recomputes_both_selectors_and_statistics(tmp_path: Path) -> None:
    fixture = make_batch(tmp_path)

    summary = _finalize(fixture)

    assert summary["status"] == "complete_and_validated"
    assert summary["selector_readiness"] == {
        "best_iou": "ready",
        "constrained_min_fa": "ready",
    }
    nuaa_best = summary["datasets"]["NUAA-SIRST"]["best_iou"]
    assert [run["completed_epoch"] for run in nuaa_best["per_seed"]] == [130, 130, 130]
    assert nuaa_best["aggregate"]["mean"]["iou"] == pytest.approx(0.701)
    assert nuaa_best["aggregate"]["sample_sd"]["iou"] == pytest.approx(0.001)
    assert nuaa_best["best_seed"]["seed"] == 103
    pdfa = summary["datasets"]["NUAA-SIRST"]["constrained_min_fa"]
    assert [run["completed_epoch"] for run in pdfa["per_seed"]] == [210, 210, 210]
    assert pdfa["best_seed"]["seed"] == 103
    markdown = (fixture["batch_dir"] / finalizer.OUTPUT_MARKDOWN).read_text()
    assert "test-selected results" in markdown
    assert "not performance on an untouched test set" in markdown
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _finalize(fixture)


def test_missing_constrained_checkpoint_is_na_and_selector_not_ready(tmp_path: Path) -> None:
    missing = {("NUDT-SIRST", 102)}
    fixture = make_batch(tmp_path, missing_constrained=missing)

    summary = _finalize(fixture)

    assert summary["status"] == "best_iou_complete__constrained_min_fa_not_ready"
    assert summary["selector_readiness"]["best_iou"] == "ready"
    assert summary["selector_readiness"]["constrained_min_fa"] == "not_ready"
    nudt = summary["datasets"]["NUDT-SIRST"]
    assert nudt["best_iou"]["aggregate"] is not None
    assert nudt["constrained_min_fa"]["aggregate"] is None
    assert nudt["constrained_min_fa"]["best_seed"] is None
    assert nudt["constrained_min_fa"]["per_seed"][1]["status"] == "not_found"
    markdown = (fixture["batch_dir"] / finalizer.OUTPUT_MARKDOWN).read_text()
    assert "| NUDT-SIRST | 102 | not_found | NA | NA | NA | NA |" in markdown


@pytest.mark.parametrize("tamper", ["source", "split"])
def test_source_and_canonical_split_hashes_fail_closed(
    tmp_path: Path, tamper: str
) -> None:
    fixture = make_batch(tmp_path)
    if tamper == "source":
        fixture["scheduler"].write_text("# changed after launch\n", encoding="utf-8")
        match = "current source hash"
    else:
        split = (
            fixture["dataset_root"]
            / "NUAA-SIRST"
            / "img_idx"
            / "train_NUAA-SIRST.txt"
        )
        split.write_text(split.read_text() + "tampered\n", encoding="utf-8")
        match = "count|raw hash"

    with pytest.raises(finalizer.FinalizationError, match=match):
        _finalize(fixture)
    assert not (fixture["batch_dir"] / finalizer.OUTPUT_JSON).exists()
    assert not (fixture["batch_dir"] / finalizer.OUTPUT_MARKDOWN).exists()


def test_late_equal_iou_checkpoint_is_rejected_by_earliest_tie_rule(tmp_path: Path) -> None:
    fixture = make_batch(tmp_path)
    job = "mshnet__nuaa-sirst__seed_101"
    run_dir = tmp_path / "weights" / "NUAA-SIRST" / "seed_101"
    summary_path = run_dir / "protocol_summary.json"
    protocol_summary = json.loads(summary_path.read_text())
    late_epoch = finalizer.EVALUATION_ZERO_BASED_EPOCHS[14]
    protocol_summary["checkpoint_selection"]["best_iou"]["epoch_zero_based"] = late_epoch
    checkpoint_path = run_dir / "checkpoint_best_iou.pkl"
    fixture["checkpoints"][checkpoint_path.resolve()]["epoch"] = late_epoch
    _write_json(summary_path, protocol_summary)

    with pytest.raises(finalizer.FinalizationError, match="checkpoint epoch"):
        _finalize(fixture)
    assert job in protocol_summary["method_meta"]["run_label"]


def test_checkpoint_metadata_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = make_batch(tmp_path)
    checkpoint_path = (
        tmp_path
        / "weights"
        / "NUAA-SIRST"
        / "seed_101"
        / "checkpoint_best_iou.pkl"
    ).resolve()
    fixture["checkpoints"][checkpoint_path]["method_meta"][
        "unexpected_changed_field"
    ] = True

    with pytest.raises(
        finalizer.FinalizationError, match="cross-artifact identity"
    ):
        _finalize(fixture)
    assert not (fixture["batch_dir"] / finalizer.OUTPUT_JSON).exists()
