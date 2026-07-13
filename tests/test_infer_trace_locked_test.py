from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from tools import infer_trace_locked_test as locked
from utils.trace_provenance import canonical_json_sha256, sha256_file


def test_exact_unlock_phrase_is_required_before_any_test_access() -> None:
    with pytest.raises(locked.TraceLockedTestError, match="exact --unlock-test"):
        locked.require_explicit_unlock("yes")
    locked.require_explicit_unlock(locked.LOCK_PHRASE)


def test_dev_bundle_lock_binds_best_checkpoint_data_and_npz_bytes(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "dev.npz"
    np.savez_compressed(
        bundle,
        scores=np.zeros((1, 2, 2)),
        targets=np.zeros((1, 2, 2), dtype=np.uint8),
        sample_ids=np.asarray(["sample"], dtype=np.str_),
    )
    metadata = {
        "schema_version": "trace_dev_score_bundle_v1",
        "method": "trace",
        "split": "dev",
        "selection_checkpoint_role": "best_dev",
        "checkpoint_sha256": "a" * 64,
        "data_provenance_sha256": "b" * 64,
        "test_access": False,
        "bundle": "external:dev.npz",
        "bundle_sha256": sha256_file(bundle),
    }
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    observed = locked.validate_dev_bundle_lock(
        path,
        expected_checkpoint_sha256="a" * 64,
        expected_method="trace",
        expected_data_provenance_sha256="b" * 64,
    )
    assert observed == metadata

    with bundle.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(locked.TraceLockedTestError, match="bytes differ"):
        locked.validate_dev_bundle_lock(
            path,
            expected_checkpoint_sha256="a" * 64,
            expected_method="trace",
            expected_data_provenance_sha256="b" * 64,
        )


def test_reconstruct_training_args_uses_authenticated_checkpoint_config() -> None:
    config = {
        "schema_version": "trace_paired_run_config_v1",
        "method": "trace",
        "seed": 7,
        "dev_fraction": 0.2,
        "epochs": 4,
        "batch_size": 2,
        "num_workers": 0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "gradient_clip_norm": 0.0,
        "positive_cell_prior": 0.01,
        "foreground_pixel_prior": None,
        "field_chunk_size": 32,
        "device": "cpu",
    }
    cli = Namespace(
        dataset_dir="dataset",
        train_manifest="",
        test_manifest="",
        baseline_checkpoint="baseline.pt",
        t0_a_report="a.json",
        t0_b_dp_report="b.json",
        t0_b_integration_report="c.json",
        checkpoint="run/checkpoint_best_dev.pt",
        device="",
    )
    args = locked.reconstruct_training_args({"run_config": config}, cli)
    assert args.method == "trace"
    assert args.seed == 7
    assert args.field_chunk_size == 32
    assert args.device == "cpu"
    assert canonical_json_sha256(config) == canonical_json_sha256(
        {key: value for key, value in config.items()}
    )

    cli.device = "cuda:0"
    with pytest.raises(locked.TraceLockedTestError, match="device must match"):
        locked.reconstruct_training_args({"run_config": config}, cli)


def test_export_rejects_unlock_before_reading_missing_checkpoint(tmp_path: Path) -> None:
    cli = Namespace(
        unlock_test="wrong",
        checkpoint=str(tmp_path / "missing.pt"),
    )
    with pytest.raises(locked.TraceLockedTestError, match="unlock-test"):
        locked.export_locked_test_bundle(cli)
