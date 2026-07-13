from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools import run_test_selected_baselines as scheduler


EXPECTED_SPLITS = {
    "NUAA-SIRST": {
        "train": (
            213,
            "324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4",
            "815dcca749f087f27f5dad4b447015aee70bd7ae7779d6fdd7d6efa6d5c6943f",
        ),
        "test": (
            214,
            "e49023203a323c247306b314f23c8b3b917093a26984067792355adff7a8386e",
            "395eecd6bf0ed2a59f531de9145688597632c68f9d0933359aadcb93ec1a60b5",
        ),
    },
    "NUDT-SIRST": {
        "train": (
            663,
            "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3",
            "dc555df66b62dd1ea98d119ace8fe8ae86de94f3e4833d8d81e90c0e1f287922",
        ),
        "test": (
            664,
            "a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e",
            "cec44220c69d89a5b3fd245b8ee911404e959fef80bd96b32b6b74f28bb32af0",
        ),
    },
    "IRSTD-1K": {
        "train": (
            800,
            "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744",
            "b698d2d9dbe9e26e1875978d23450e1e6ec45fd71d56d31415007f56c40bba88",
        ),
        "test": (
            201,
            "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579",
            "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579",
        ),
    },
}


def _value_after(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def _fake_dataset_metadata(dataset: str) -> dict[str, object]:
    dataset_dir = scheduler.DATASET_ROOT / dataset
    return {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "train_split_arg": f"img_idx/train_{dataset}.txt",
        "test_split_arg": f"img_idx/test_{dataset}.txt",
        "train_count": EXPECTED_SPLITS[dataset]["train"][0],
        "test_count": EXPECTED_SPLITS[dataset]["test"][0],
        "train_ordered_names_sha256": EXPECTED_SPLITS[dataset]["train"][2],
        "test_ordered_names_sha256": EXPECTED_SPLITS[dataset]["test"][2],
        "validation_split": None,
        "hcval_policy": "prohibited_not_read",
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_job_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    run_dir = tmp_path / "run"
    result_file = tmp_path / "result.json"
    job: dict[str, object] = {
        "job_id": "mshnet__nuaa-sirst__seed_20260711",
        "dataset": "NUAA-SIRST",
        "seed": 20260711,
        "dataset_dir": "/home/ly/DEA/datasets/NUAA-SIRST",
        "train_split_arg": "img_idx/train_NUAA-SIRST.txt",
        "test_split_arg": "img_idx/test_NUAA-SIRST.txt",
        "train_split_sha256": EXPECTED_SPLITS["NUAA-SIRST"]["train"][2],
        "test_split_sha256": EXPECTED_SPLITS["NUAA-SIRST"]["test"][2],
        "train_split_raw_sha256": EXPECTED_SPLITS["NUAA-SIRST"]["train"][1],
        "test_split_raw_sha256": EXPECTED_SPLITS["NUAA-SIRST"]["test"][1],
        "deterministic": True,
        "run_dir": str(run_dir),
        "result_file": str(result_file),
        "total_epochs": 23,
        "test_interval": 10,
        "test_evaluation_epochs": [10, 20, 23],
    }
    _write_json(
        result_file,
        {
            "returncode": 0,
            "protocol": scheduler.PROTOCOL,
            "job_id": job["job_id"],
            "dataset": job["dataset"],
            "seed": job["seed"],
            "run_dir": job["run_dir"],
            "total_epochs": job["total_epochs"],
            "test_interval": job["test_interval"],
            "test_evaluation_epochs": job["test_evaluation_epochs"],
        },
    )
    selection = {
        "status": "found",
        "file": "checkpoint_best_iou.pkl",
        "epoch_zero_based": 19,
        "iou": 0.7,
        "pd": 0.95,
        "fa": 12.0,
    }
    method_meta = {
        "protocol": scheduler.PROTOCOL,
        "protocol_version": scheduler.PROTOCOL,
        "method": "MSHNet",
        "model_type": "mshnet",
        "dataset_name": "NUAA-SIRST",
        "dataset_dir": job["dataset_dir"],
        "train_split_file": str(
            Path(str(job["dataset_dir"])) / str(job["train_split_arg"])
        ),
        "test_split_file": str(
            Path(str(job["dataset_dir"])) / str(job["test_split_arg"])
        ),
        "val_split_file": "",
        "val_fraction": 0.0,
        "train_split_sha256": job["train_split_sha256"],
        "test_split_sha256": job["test_split_sha256"],
        "train_split_raw_sha256": job["train_split_raw_sha256"],
        "test_split_raw_sha256": job["test_split_raw_sha256"],
        "seed": job["seed"],
        "deterministic": True,
        "run_label": job["job_id"],
        "selection_split": "test",
        "evaluation_split": "test",
        "no_internal_holdout": True,
        "test_interval": 10,
        "selection_threshold": scheduler.EVAL_THRESHOLD,
        "selection_tie_break": scheduler.SELECTION_TIE_BREAK,
        "selection_best_iou_rule": scheduler.SELECTION_BEST_IOU_RULE,
        "selection_pd_fa_rule": scheduler.SELECTION_PD_FA_RULE,
        "selection_pd_fa_min_pd": scheduler.PD_FA_MIN_PD,
        "selection_pd_fa_min_iou": scheduler.PD_FA_MIN_IOU,
        "selection_paired_baseline_iou": scheduler.PAIRED_BASELINE_IOU,
        "train_loader_drop_last": False,
    }
    run_dir.mkdir(parents=True)
    torch.save(
        {
            "epoch": 19,
            "iou": 0.7,
            "pd": 0.95,
            "fa": 12.0,
            "method_meta": method_meta,
        },
        run_dir / "checkpoint_best_iou.pkl",
    )
    selection["sha256"] = scheduler.sha256_file(
        run_dir / "checkpoint_best_iou.pkl"
    )
    summary_path = run_dir / "protocol_summary.json"
    _write_json(
        summary_path,
        {
            "protocol": scheduler.PROTOCOL,
            "status": "complete",
            "dataset": "NUAA-SIRST",
            "run_dir": str(run_dir),
            "total_epochs": 23,
            "test_interval": 10,
            "last_completed_epoch_zero_based": 22,
            "executed_evaluation_epochs_zero_based": [9, 19, 22],
            "checkpoint_selection": {
                "best_iou": selection,
                "constrained_min_fa": {
                    "status": "not_found",
                    "file": None,
                    "reason": "no_eligible_epoch",
                },
            },
        },
    )
    return job, summary_path


def test_frozen_scheduler_contract_matches_canonical_img_idx() -> None:
    assert scheduler.PROTOCOL == "test_selected_full_train_interval_v1"
    assert scheduler.DATASET_ROOT == Path("/home/ly/DEA/datasets")
    assert set(scheduler.CANONICAL_DATASETS) == set(EXPECTED_SPLITS)

    for dataset, roles in EXPECTED_SPLITS.items():
        frozen = scheduler.CANONICAL_DATASETS[dataset]
        for role, (count, raw_hash, normalized_hash) in roles.items():
            assert frozen[role]["count"] == count
            assert frozen[role]["raw_sha256"] == raw_hash
            assert frozen[role]["ordered_names_sha256"] == normalized_hash

        audited = scheduler.validate_dataset(dataset)
        assert audited["dataset_dir"] == str(
            Path("/home/ly/DEA/datasets") / dataset
        )
        assert audited["train_count"] == roles["train"][0]
        assert audited["test_count"] == roles["test"][0]
        assert audited["train_raw_sha256"] == roles["train"][1]
        assert audited["test_raw_sha256"] == roles["test"][1]
        assert audited["train_ordered_names_sha256"] == roles["train"][2]
        assert audited["test_ordered_names_sha256"] == roles["test"][2]
        assert audited["train_test_overlap_count"] == 0
        assert audited["validation_split"] is None
        assert audited["hcval_policy"] == "prohibited_not_read"
        assert audited["asset_universe_equality"] == (
            "train_union_test=image_ids=mask_ids"
        )


def test_scheduler_command_has_only_canonical_train_test_roles_and_interval() -> None:
    args = scheduler.parse_args(
        [
            "--datasets",
            "NUDT-SIRST",
            "--seeds",
            "20260711",
            "--gpus",
            "3",
            "--epochs",
            "23",
            "--test-interval",
            "10",
        ]
    )
    scheduler._validate_args(args)
    job = {
        **_fake_dataset_metadata("NUDT-SIRST"),
        "seed": 20260711,
        "run_dir": "/tmp/run",
        "job_id": "mshnet__nudt-sirst__seed_20260711",
    }

    command = scheduler.build_command(args, job)

    assert Path(command[1]) == scheduler.TRAINING_ENTRY
    assert _value_after(command, "--dataset-dir") == str(
        Path("/home/ly/DEA/datasets/NUDT-SIRST")
    )
    assert _value_after(command, "--train-split-file") == (
        "img_idx/train_NUDT-SIRST.txt"
    )
    assert _value_after(command, "--test-split-file") == (
        "img_idx/test_NUDT-SIRST.txt"
    )
    assert _value_after(command, "--val-fraction") == "0"
    assert _value_after(command, "--test-interval") == "10"
    assert _value_after(command, "--threshold") == "0.5"
    assert _value_after(command, "--pd-fa-min-pd") == "0.93"
    assert _value_after(command, "--pd-fa-min-iou") == "0.655"
    assert _value_after(command, "--paired-baseline-iou") == "0.0"
    assert _value_after(command, "--selection-tie-break") == "earliest_epoch"
    assert "--val-split-file" not in command
    assert not any("hcval" in token.lower() for token in command)
    assert scheduler.evaluation_epochs(23, 10) == [10, 20, 23]


def test_scheduler_locks_interval_and_prohibits_validation() -> None:
    for argv, match in (
        (["--test-interval", "5"], "test-interval"),
        (["--val-fraction", "0.2"], "validation"),
    ):
        args = scheduler.parse_args(argv)
        with pytest.raises(ValueError, match=match):
            scheduler._validate_args(args)


def test_split_reader_fails_on_wrong_hash_duplicate_and_symlink(
    tmp_path: Path,
) -> None:
    split = tmp_path / "train.txt"
    split.write_bytes(b"a\nb\n")
    expected = {
        "count": 2,
        "raw_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "ordered_names_sha256": scheduler.names_sha256(["a", "b"]),
    }
    assert scheduler._read_split(split, expected, "train") == ["a", "b"]

    split.write_bytes(b"a\nc\n")
    with pytest.raises(RuntimeError, match="raw split hash mismatch"):
        scheduler._read_split(split, expected, "train")

    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_bytes(b"a\na\n")
    duplicate_expected = {
        "count": 2,
        "raw_sha256": hashlib.sha256(duplicate.read_bytes()).hexdigest(),
        "ordered_names_sha256": scheduler.names_sha256(["a", "a"]),
    }
    with pytest.raises(RuntimeError, match="duplicate sample"):
        scheduler._read_split(duplicate, duplicate_expected, "train")

    target = tmp_path / "target.txt"
    target.write_bytes(b"a\nb\n")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        scheduler._read_split(link, expected, "train")


def test_dry_run_manifest_declares_test_selection_and_no_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        scheduler,
        "validate_dataset",
        lambda dataset: _fake_dataset_metadata(dataset),
    )

    result = scheduler.main(
        [
            "--datasets",
            "NUAA-SIRST",
            "--seeds",
            "20260711",
            "--gpus",
            "3",
            "--epochs",
            "23",
            "--test-interval",
            "10",
            "--batch-id",
            "pytest_dry_run",
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    manifest, end = json.JSONDecoder().raw_decode(output)
    command_line = output[end:].strip()
    assert manifest["protocol"] == "test_selected_full_train_interval_v1"
    assert manifest["data_protocol"]["validation_role"] is None
    assert manifest["data_protocol"]["validation_fraction"] == 0.0
    assert manifest["data_protocol"]["hcval_policy"] == "prohibited_not_read"
    assert manifest["selection_policy"]["test_selected"] is True
    assert manifest["selection_policy"]["test_is_untouched"] is False
    assert manifest["selection_policy"]["test_interval_completed_epochs"] == 10
    assert manifest["selection_policy"]["test_evaluation_epochs"] == [10, 20, 23]
    assert manifest["selection_policy"]["eligible_checkpoint_epochs"] == [
        10,
        20,
        23,
    ]
    assert manifest["selection_policy"]["metric_threshold"] == 0.5
    assert manifest["selection_policy"]["pd_fa_min_pd"] == 0.93
    assert manifest["selection_policy"]["pd_fa_min_iou"] == 0.655
    assert manifest["selection_policy"]["paired_baseline_iou"] == 0.0
    assert manifest["selection_policy"]["strict_tie_break"] == "earliest_epoch"
    assert manifest["immutable_contract"]["checkpoint_selector"] == {
        "threshold": 0.5,
        "pd_fa_min_pd": 0.93,
        "pd_fa_min_iou": 0.655,
        "paired_baseline_iou": 0.0,
        "tie_break": "earliest_epoch",
    }
    assert "--train-split-file img_idx/train_NUAA-SIRST.txt" in command_line
    assert "--test-split-file img_idx/test_NUAA-SIRST.txt" in command_line
    assert "--test-interval 10" in command_line
    assert "--threshold 0.5" in command_line
    assert "--pd-fa-min-pd 0.93" in command_line
    assert "--pd-fa-min-iou 0.655" in command_line
    assert "--selection-tie-break earliest_epoch" in command_line
    assert "--val-fraction 0" in command_line
    assert "--val-split-file" not in command_line
    assert "hcval" not in command_line.lower()


def test_resume_skip_requires_complete_summary_and_valid_selected_checkpoint(
    tmp_path: Path,
) -> None:
    job, summary_path = _completed_job_fixture(tmp_path)
    assert scheduler._is_completed_result(job) is True

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "running"
    _write_json(summary_path, summary)
    assert scheduler._is_completed_result(job) is False

    summary["status"] = "complete"
    original_sha = summary["checkpoint_selection"]["best_iou"]["sha256"]
    summary["checkpoint_selection"]["best_iou"]["sha256"] = "0" * 64
    _write_json(summary_path, summary)
    assert scheduler._is_completed_result(job) is False

    summary["checkpoint_selection"]["best_iou"]["sha256"] = original_sha
    summary["checkpoint_selection"]["best_iou"]["iou"] = 0.69
    _write_json(summary_path, summary)
    assert scheduler._is_completed_result(job) is False

    summary["checkpoint_selection"]["best_iou"]["iou"] = 0.7
    summary["checkpoint_selection"]["best_iou"]["epoch_zero_based"] = 18
    _write_json(summary_path, summary)
    assert scheduler._is_completed_result(job) is False


def test_resume_immutable_contract_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    contract = {
        "protocol": scheduler.PROTOCOL,
        "checkpoint_selector": {
            "threshold": 0.5,
            "test_interval": 10,
        },
    }
    _write_json(
        manifest_path,
        {
            "immutable_contract": contract,
            "immutable_contract_sha256": scheduler.canonical_json_sha256(
                contract
            ),
        },
    )
    assert scheduler.validate_resume_contract(manifest_path, contract)[
        "immutable_contract"
    ] == contract

    changed = {
        **contract,
        "checkpoint_selector": {
            "threshold": 0.4,
            "test_interval": 10,
        },
    }
    with pytest.raises(RuntimeError, match="immutable contract mismatch"):
        scheduler.validate_resume_contract(manifest_path, changed)


def test_resume_rejects_ineligible_constrained_min_fa_checkpoint(
    tmp_path: Path,
) -> None:
    job, summary_path = _completed_job_fixture(tmp_path)
    run_dir = Path(str(job["run_dir"]))
    best = torch.load(
        run_dir / "checkpoint_best_iou.pkl",
        map_location="cpu",
        weights_only=False,
    )
    constrained_path = run_dir / "checkpoint_pd_fa_best.pkl"
    constrained_checkpoint = {
        **best,
        "epoch": 22,
        "iou": 0.7,
        "pd": 0.5,
        "fa": 1.0,
    }
    torch.save(constrained_checkpoint, constrained_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_selection"]["constrained_min_fa"] = {
        "status": "found",
        "file": "checkpoint_pd_fa_best.pkl",
        "sha256": scheduler.sha256_file(constrained_path),
        "epoch_zero_based": 22,
        "iou": 0.7,
        "pd": 0.5,
        "fa": 1.0,
    }
    _write_json(summary_path, summary)

    assert scheduler._is_completed_result(job) is False

    constrained_checkpoint["pd"] = 0.95
    torch.save(constrained_checkpoint, constrained_path)
    summary["checkpoint_selection"]["constrained_min_fa"]["pd"] = 0.95
    summary["checkpoint_selection"]["constrained_min_fa"][
        "sha256"
    ] = scheduler.sha256_file(constrained_path)
    _write_json(summary_path, summary)
    assert scheduler._is_completed_result(job) is True
