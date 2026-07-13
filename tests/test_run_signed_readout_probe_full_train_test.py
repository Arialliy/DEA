from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from tools import run_signed_readout_probe_full_train_test as probe


def test_protocol_epochs_are_predeclared() -> None:
    assert probe.protocol_epochs("formal", "IRSTD-1K", 17) == 20
    assert probe.protocol_epochs("smoke", "NUAA-SIRST", 20260711) == 10
    with pytest.raises(probe.FullTrainTestProbeError, match="predeclared"):
        probe.protocol_epochs("smoke", "IRSTD-1K", 20260711)


def test_source_inventory_covers_previously_omitted_dependencies() -> None:
    paths = probe._source_paths()
    assert paths["component_operating_point"].name == "component_operating_point.py"
    assert paths["mshnet_checkpoint"].name == "mshnet_checkpoint.py"
    assert paths["baseline_finalizer"].name == "finalize_test_selected_baselines.py"
    assert paths["full_train_test_protocol"].name == "full_train_test_protocol.py"
    assert all(path.is_file() and not path.is_symlink() for path in paths.values())


def test_loader_never_drops_the_last_train_sample() -> None:
    dataset = TensorDataset(torch.arange(5))
    loader = probe._loader(
        dataset,  # type: ignore[arg-type]
        training=True,
        num_workers=0,
        device=torch.device("cpu"),
        seed=9,
    )
    assert loader.drop_last is False
    observed = sum(int(batch[0].shape[0]) for batch in loader)
    assert observed == len(dataset)

    test_loader = probe._loader(
        dataset,  # type: ignore[arg-type]
        training=False,
        num_workers=0,
        device=torch.device("cpu"),
        seed=10,
    )
    assert test_loader.batch_size == 1


def test_rng_roles_are_exactly_train_and_test() -> None:
    raw = {
        "fit_dataloader_generator_seed": 17,
        "dev_dataloader_generator_seed": 18,
        "worker_seed_rule": "sentinel",
    }
    observed = probe._canonicalize_rng_roles(raw, seed=17)
    assert observed["fit_dataloader_generator_seed"] == 17
    assert observed["test_dataloader_generator_seed"] == 18
    assert "dev_dataloader_generator_seed" not in observed
    with pytest.raises(probe.FullTrainTestProbeError, match="both dev and test"):
        probe._canonicalize_rng_roles(
            {
                **raw,
                "test_dataloader_generator_seed": 18,
            },
            seed=17,
        )
    with pytest.raises(probe.FullTrainTestProbeError, match="RNG seed drifted"):
        probe._canonicalize_rng_roles(
            {"dev_dataloader_generator_seed": 19}, seed=17
        )


def test_run_config_is_hashed_and_locks_both_spatial_sizes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_config = run_dir / "run_config.json"
    run_config.write_text(
        json.dumps({"args": {"base_size": 256, "crop_size": 256}}) + "\n",
        encoding="utf-8",
    )
    loaded, record = probe._read_locked_run_config(run_dir)
    assert loaded["args"] == {"base_size": 256, "crop_size": 256}
    assert record == {
        "path": str(run_config.resolve()),
        "sha256": probe.legacy_probe.sha256_file(run_config),
        "base_size": 256,
        "crop_size": 256,
    }

    run_config.write_text(
        json.dumps({"args": {"base_size": 256, "crop_size": 255}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(probe.FullTrainTestProbeError, match="crop_size must equal"):
        probe._read_locked_run_config(run_dir)


def _native_replay_fixture(
    tmp_path: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    checkpoint = tmp_path / "checkpoint_best_iou.pkl"
    checkpoint.write_bytes(b"selected checkpoint identity")

    scores = [
        np.zeros((256, 256), dtype=np.float32),
        np.zeros((256, 256), dtype=np.float32),
    ]
    targets = [
        np.zeros((256, 256), dtype=np.uint8),
        np.zeros((256, 256), dtype=np.uint8),
    ]
    scores[0][10, 10] = 1.0
    targets[0][10, 10] = 1
    # A zero logit is deliberately excluded by the strict logit > 0 rule.
    targets[1][20, 20] = 1
    scores[1][100, 100] = 1.0

    total_pixels = 2 * 256 * 256
    record = {
        "policy": "test_selected_best_iou",
        "selection_split": "canonical_test",
        "job_id": "mshnet__synthetic__seed_17",
        "path": str(checkpoint),
        "sha256": probe.legacy_probe.sha256_file(checkpoint),
        "epoch_zero_based": 19,
        "completed_epoch": 20,
        "selected_iou": 1.0 / 3.0,
        "selected_pd": 1.0 / 2.0,
        "selected_fa_per_mpix": 1_000_000.0 / total_pixels,
    }
    return scores, targets, record


def test_native_replay_exact_counts_official_metrics_and_checkpoint_binding(
    tmp_path: Path,
) -> None:
    scores, targets, record = _native_replay_fixture(tmp_path)
    replay = probe.build_native_selected_checkpoint_replay(scores, targets, record)
    assert replay["strict_prediction_rule"] == "original_final_z logit > 0"
    assert replay["integer_pixel_counts"] == {
        "intersection_pixels": 1,
        "union_pixels": 3,
        "prediction_pixels": 2,
        "target_pixels": 2,
    }
    assert replay["iou"] == 1.0 / 3.0
    assert replay["official_legacy"]["matched_components"] == 1
    assert replay["official_legacy"]["target_components"] == 2
    assert replay["official_legacy"]["unmatched_prediction_area"] == 1
    assert replay["official_legacy"]["pd"] == 0.5
    assert replay["official_legacy"]["fa_per_mpix"] == (
        1_000_000.0 / (2 * 256 * 256)
    )
    assert replay["selected_checkpoint"]["path"] == str(
        Path(record["path"]).resolve()
    )
    assert replay["checkpoint_metric_binding"] == {
        "status": "passed",
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 0.0,
        "metrics": ["iou", "pd", "fa_per_mpix"],
    }
    assert replay["replay_minus_checkpoint_reported"] == {
        "iou": 0.0,
        "pd": 0.0,
        "fa_per_mpix": 0.0,
    }


def test_native_replay_fails_closed_on_non_best_iou_or_metric_mismatch(
    tmp_path: Path,
) -> None:
    scores, targets, record = _native_replay_fixture(tmp_path)
    with pytest.raises(probe.FullTrainTestProbeError, match="policy drifted"):
        probe.build_native_selected_checkpoint_replay(
            scores,
            targets,
            {**record, "policy": "constrained_min_fa"},
        )
    with pytest.raises(
        probe.FullTrainTestProbeError,
        match="does not reproduce selected checkpoint iou",
    ):
        probe.build_native_selected_checkpoint_replay(
            scores,
            targets,
            {**record, "selected_iou": record["selected_iou"] + 2e-12},
        )


def _revalidation_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict]:
    checkpoint = tmp_path / "checkpoint_best_iou.pkl"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_record = {
        "path": str(checkpoint.resolve()),
        "sha256": probe.legacy_probe.sha256_file(checkpoint),
    }
    state = {
        "sources": {"tool": "source-hash", "metric": "metric-hash"},
        "front": {
            "summary_sha256": "front-summary",
            "provenance_sha256": "front-provenance",
            "hard_core_source_sha256": "front-source",
        },
        "job": {
            "job_id": "mshnet__synthetic__seed_17",
            "run_dir": str(tmp_path / "run"),
            "dataset_dir": str(tmp_path / "dataset"),
        },
        "validated": {
            "best_iou": {
                "status": "found",
                "checkpoint": str(checkpoint.resolve()),
            }
        },
        "manifest": {
            "path": str((tmp_path / "manifest.json").resolve()),
            "sha256": "manifest-hash",
        },
        "artifacts": {
            name: {
                "path": str((tmp_path / name).resolve()),
                "sha256": f"{name}-hash",
            }
            for name in (
                "manifest",
                "job_result",
                "run_config",
                "protocol_summary",
                "persisted_train_split",
                "persisted_test_split",
            )
        },
        "run_config": {
            "path": str((tmp_path / "run" / "run_config.json").resolve()),
            "sha256": "run-config-hash",
            "base_size": 256,
            "crop_size": 256,
        },
        "canonical": {
            "train_raw_sha256": "train-raw",
            "train_normalized_sha256": "train-normalized",
            "test_raw_sha256": "test-raw",
            "test_normalized_sha256": "test-normalized",
        },
        "checkpoint": checkpoint_record,
    }
    before = copy.deepcopy(state)

    monkeypatch.setattr(
        probe,
        "_source_hashes",
        lambda: copy.deepcopy(state["sources"]),
    )
    monkeypatch.setattr(
        probe.legacy_probe,
        "validate_front_freeze_bundle",
        lambda _path: copy.deepcopy(state["front"]),
    )
    monkeypatch.setattr(
        probe,
        "validate_selected_baseline",
        lambda *_args: (
            copy.deepcopy(state["job"]),
            copy.deepcopy(state["validated"]),
            copy.deepcopy(state["manifest"]),
        ),
    )
    monkeypatch.setattr(
        probe,
        "_baseline_artifact_records",
        lambda *_args: copy.deepcopy(state["artifacts"]),
    )
    monkeypatch.setattr(
        probe,
        "_read_locked_run_config",
        lambda _path: (
            {"args": {"base_size": 256, "crop_size": 256}},
            copy.deepcopy(state["run_config"]),
        ),
    )
    monkeypatch.setattr(probe, "audit_canonical_dataset", lambda _path: object())
    monkeypatch.setattr(
        probe,
        "_canonical_data_record",
        lambda _audit: copy.deepcopy(state["canonical"]),
    )
    monkeypatch.setattr(
        probe,
        "_plain_file_record",
        lambda _path, _label: copy.deepcopy(state["checkpoint"]),
    )

    kwargs = {
        "batch_dir": tmp_path,
        "dataset": "NUAA-SIRST",
        "seed": 17,
        "front_freeze_dir": tmp_path / "front",
        "sources_before": before["sources"],
        "front_before": before["front"],
        "job_before": before["job"],
        "validated_job_before": before["validated"],
        "manifest_before": before["manifest"],
        "baseline_artifacts_before": before["artifacts"],
        "run_config_before": before["run_config"],
        "canonical_data_before": before["canonical"],
        "checkpoint_before": before["checkpoint"],
    }
    return state, kwargs


def test_end_revalidation_covers_every_external_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _state, kwargs = _revalidation_fixture(tmp_path, monkeypatch)
    result = probe.revalidate_external_contract(**kwargs)
    assert result["status"] == "passed"
    assert set(result["checks"]) == {
        "manifest_path_and_hash",
        "job_result_path_and_hash",
        "run_config_path_hash_and_locked_256_sizes",
        "protocol_summary_path_and_hash",
        "front_freeze_validation_and_hashes",
        "canonical_and_persisted_train_test_split_hashes",
        "selected_checkpoint_path_hash_and_selection_binding",
        "all_source_hashes",
    }
    assert result["baseline_artifact_sha256"]["protocol_summary"] == (
        "protocol_summary-hash"
    )


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("source", "probe source hashes drifted"),
        ("front", "front-freeze authority drifted"),
        ("manifest", "baseline manifest drifted"),
        ("job", "baseline job drifted"),
        ("selection", "validated baseline selection drifted"),
        ("protocol_summary", "baseline artifact hashes drifted"),
        ("run_config", "baseline run_config drifted"),
        ("split", "canonical train/test splits drifted"),
        ("checkpoint", "selected checkpoint drifted"),
    ),
)
def test_end_revalidation_fails_closed_on_dependency_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    state, kwargs = _revalidation_fixture(tmp_path, monkeypatch)
    if drift == "source":
        state["sources"]["tool"] = "changed"
    elif drift == "front":
        state["front"]["summary_sha256"] = "changed"
    elif drift == "manifest":
        state["manifest"]["sha256"] = "changed"
    elif drift == "job":
        state["job"]["job_id"] = "changed"
    elif drift == "selection":
        state["validated"]["best_iou"]["completed_epoch"] = 999
    elif drift == "protocol_summary":
        state["artifacts"]["protocol_summary"]["sha256"] = "changed"
    elif drift == "run_config":
        state["run_config"]["sha256"] = "changed"
    elif drift == "split":
        state["canonical"]["test_raw_sha256"] = "changed"
    elif drift == "checkpoint":
        state["checkpoint"]["sha256"] = "changed"
    else:  # pragma: no cover - parametrization is the closed set above.
        raise AssertionError(drift)
    with pytest.raises(probe.FullTrainTestProbeError, match=message):
        probe.revalidate_external_contract(**kwargs)


def test_bundle_roundtrip_uses_explicit_variant_order(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    variants = {
        name: {"sentinel": index}
        for index, name in enumerate(probe.VARIANT_ORDER)
    }
    summary = {
        "schema": probe.SCHEMA,
        "status": "complete",
        "variant_order": list(probe.VARIANT_ORDER),
        "variants": variants,
    }
    logits = {
        name: (np.zeros((2, 2), dtype=np.float32),)
        for name in probe.VARIANT_ORDER
    }
    probe._write_bundle(
        output,
        summary=summary,
        history=(),
        oracle_rows=(),
        target_rows=(),
        image_rows=(),
        calibration_rows=(),
        logits=logits,
        image_names=("sample",),
        head_payload={"state_dict": {}},
        provenance={"schema": probe.PROVENANCE_SCHEMA},
    )

    loaded = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    # sort_keys=True intentionally changes nested mapping order on disk.  The
    # protocol therefore validates the explicit order field and the exact key
    # set independently, never mapping insertion order.
    assert tuple(loaded["variants"]) != probe.VARIANT_ORDER
    assert loaded["variant_order"] == list(probe.VARIANT_ORDER)
    assert set(loaded["variants"]) == set(probe.VARIANT_ORDER)

    provenance = json.loads(
        (output / "provenance.json").read_text(encoding="utf-8")
    )
    assert set(provenance["artifact_sha256"]) == set(probe.BUNDLE_FILES[:-1])
    assert {path.name for path in output.iterdir()} == set(probe.BUNDLE_FILES)


def test_validate_selected_baseline_wraps_finalizer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "manifest.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        probe.baseline_finalizer,
        "_validate_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            probe.baseline_finalizer.FinalizationError("incomplete grid")
        ),
    )
    with pytest.raises(probe.FullTrainTestProbeError, match="incomplete grid"):
        probe.validate_selected_baseline(batch, "NUAA-SIRST", 20260711)
