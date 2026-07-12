from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tools.audit_cross_seed_failure_persistence as persistence_audit
import tools.compare_gate_e_checkpoint_policies as policy_compare
from tools.compare_gate_e_checkpoint_policies import (
    build_transition_report,
    write_bundle as write_policy_transition_bundle,
)
from tools.audit_cross_seed_failure_persistence import (
    CANONICAL_SIZE,
    DATASET_NAMES,
    EXPECTED_EPOCHS,
    PersistenceAuditError,
    annotate_target_recurrence,
    build_image_envelope_row,
    build_image_target_rows,
    build_summary,
    compare_policy_target_recurrence,
    compute_image_operating_metrics,
    protocol_document_fingerprints,
    validate_protocol_documents_unchanged,
    validate_checkpoint_for_policy,
    validate_cross_run_target_registries,
    validate_grid_cardinality,
    write_output_bundle,
)
from utils.target_identity import build_stable_target_set


SEEDS = (11, 22, 33)


def test_gate_e_schema_versions_are_distinct_from_rejected_preflight_v1() -> None:
    assert (
        persistence_audit.SCHEMA
        == "dea.gate_e.cross_seed_failure_persistence.v2"
    )
    assert (
        persistence_audit.PROVENANCE_SCHEMA
        == "dea.gate_e.cross_seed_failure_persistence.provenance.v2"
    )
    assert (
        policy_compare.PROVENANCE_SCHEMA
        == "dea.gate_e.checkpoint_policy_transition.provenance.v2"
    )


def _audit_checkpoint_row(policy: str = "fixed_epoch") -> dict[str, object]:
    return {
        "policy": policy,
        "path": "/tmp/checkpoint.pkl",
        "sha256": "a" * 64,
        "epoch": 399,
        "run": {
            "job_id": "fixture",
            "run_config_sha256": "b" * 64,
            "split_hashes": {"validation": "c" * 64},
            "resume_evidence": {"resume_epoch": None},
        },
    }


def test_recorded_resume_flag_parser_is_fail_closed() -> None:
    command = ["python", "main.py", "--if-checkpoint", "true"]

    assert persistence_audit._command_option(command, "--if-checkpoint") == "true"
    assert persistence_audit._command_option(command, "--missing") is None
    with pytest.raises(PersistenceAuditError, match="repeats"):
        persistence_audit._command_option(
            command + ["--if-checkpoint", "false"], "--if-checkpoint"
        )
    with pytest.raises(PersistenceAuditError, match="list of strings"):
        persistence_audit._command_option("not-a-command", "--if-checkpoint")


def test_resume_evidence_seals_second_process_epoch(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "split train: first\nEpoch 0, loss 1.0\r"
        "Epoch 235, IoU 0.7\r"
        "split train: resumed\nset optimizer lr: 0.05\r"
        "Epoch 235, loss 0.6\r",
        encoding="utf-8",
    )

    evidence = persistence_audit.resume_evidence_from_training_log(
        log, resume_requested=True
    )
    assert evidence["resume_epoch"] == 235
    assert evidence["resume_from_completed_checkpoint_epoch"] == 234
    assert evidence["process_start_marker_count"] == 2
    assert evidence["training_log_sha256"] == persistence_audit.sha256_file(log)

    single = tmp_path / "single.log"
    single.write_text("split train: first\nEpoch 0, loss 1.0\n", encoding="utf-8")
    with pytest.raises(PersistenceAuditError, match="exactly two process starts"):
        persistence_audit.resume_evidence_from_training_log(
            single, resume_requested=True
        )
    not_resumed = persistence_audit.resume_evidence_from_training_log(
        single, resume_requested=False
    )
    assert not_resumed["resume_epoch"] is None


def _metric_rows(*, best_epoch: int = 123) -> list[dict[str, float | int]]:
    rows = []
    for epoch in range(EXPECTED_EPOCHS):
        iou = 0.2 + epoch / 10000.0
        if epoch == best_epoch:
            iou = 0.9
        rows.append({"epoch": epoch, "iou": iou, "pd": 0.75, "fa": 12.5})
    return rows


def _checkpoint(
    rows: list[dict[str, float | int]],
    *,
    epoch: int,
    job: dict[str, object],
    dataset_meta: dict[str, str],
    manifest_args: dict[str, int],
) -> dict[str, object]:
    row = rows[epoch]
    return {
        "net": {"weight": torch.zeros(1)},
        "optimizer": {
            "state": {
                0: {
                    "step": torch.tensor(float(epoch + 1)),
                    "sum": torch.zeros(1),
                }
            },
            "param_groups": [
                {
                    "lr": 0.05,
                    "lr_decay": 0,
                    "eps": 1e-10,
                    "weight_decay": 0,
                    "initial_accumulator_value": 0,
                    "foreach": None,
                    "maximize": False,
                    "differentiable": False,
                    "fused": None,
                    "params": [0],
                }
            ],
        },
        "epoch": epoch,
        "iou": row["iou"],
        "pd": row["pd"],
        "fa": row["fa"],
        "best_iou": max(float(item["iou"]) for item in rows),
        "method_meta": {
            "method": "MSHNet",
            "model_type": "mshnet",
            "seed": job["seed"],
            "run_label": job["job_id"],
            "split_seed": manifest_args["split_seed"],
            "train_split_sha256": dataset_meta["fit_sha256"],
            "val_split_sha256": dataset_meta["val_sha256"],
            "test_split_sha256": dataset_meta["official_test_sha256"],
            "deterministic": True,
        },
    }


def _checkpoint_fixture():
    job = {"job_id": "mshnet__d__seed_11", "seed": 11}
    dataset_meta = {
        "fit_sha256": "a" * 64,
        "val_sha256": "b" * 64,
        "official_test_sha256": "c" * 64,
    }
    manifest_args = {"split_seed": 7}
    return job, dataset_meta, manifest_args


def test_fixed_epoch_policy_requires_epoch_399_and_all_400_rows() -> None:
    job, dataset_meta, manifest_args = _checkpoint_fixture()
    rows = _metric_rows(best_epoch=123)
    checkpoint = _checkpoint(
        rows,
        epoch=EXPECTED_EPOCHS - 1,
        job=job,
        dataset_meta=dataset_meta,
        manifest_args=manifest_args,
    )

    result = validate_checkpoint_for_policy(
        checkpoint,
        policy="fixed_epoch",
        job=job,
        dataset_meta=dataset_meta,
        manifest_args=manifest_args,
        metric_rows=rows,
    )

    assert result["epoch"] == 399
    assert result["selection_scope"] == "fixed_training_duration_primary"
    assert result["optimizer"]["optimizer"] == "Adagrad"
    assert result["optimizer"]["parameter_state_count"] == 1

    wrong_epoch = copy.deepcopy(checkpoint)
    wrong_epoch["epoch"] = 398
    wrong_epoch.update({key: rows[398][key] for key in ("iou", "pd", "fa")})
    with pytest.raises(PersistenceAuditError, match="requires epoch 399"):
        validate_checkpoint_for_policy(
            wrong_epoch,
            policy="fixed_epoch",
            job=job,
            dataset_meta=dataset_meta,
            manifest_args=manifest_args,
            metric_rows=rows,
        )

    with pytest.raises(PersistenceAuditError, match="exactly 400 rows"):
        validate_checkpoint_for_policy(
            checkpoint,
            policy="fixed_epoch",
            job=job,
            dataset_meta=dataset_meta,
            manifest_args=manifest_args,
            metric_rows=rows[:-1],
        )

    missing_optimizer = copy.deepcopy(checkpoint)
    missing_optimizer["optimizer"] = {}
    with pytest.raises(PersistenceAuditError, match="state mapping"):
        validate_checkpoint_for_policy(
            missing_optimizer,
            policy="fixed_epoch",
            job=job,
            dataset_meta=dataset_meta,
            manifest_args=manifest_args,
            metric_rows=rows,
        )


def test_best_iou_policy_is_separate_and_rejects_latest_nonbest() -> None:
    job, dataset_meta, manifest_args = _checkpoint_fixture()
    rows = _metric_rows(best_epoch=123)
    selected = _checkpoint(
        rows,
        epoch=123,
        job=job,
        dataset_meta=dataset_meta,
        manifest_args=manifest_args,
    )
    result = validate_checkpoint_for_policy(
        selected,
        policy="best_iou",
        job=job,
        dataset_meta=dataset_meta,
        manifest_args=manifest_args,
        metric_rows=rows,
    )
    assert result["filename"] == "checkpoint_best_iou.pkl"
    assert result["selection_scope"].startswith("retrospective")

    latest = _checkpoint(
        rows,
        epoch=399,
        job=job,
        dataset_meta=dataset_meta,
        manifest_args=manifest_args,
    )
    with pytest.raises(PersistenceAuditError, match="not at the logged best"):
        validate_checkpoint_for_policy(
            latest,
            policy="best_iou",
            job=job,
            dataset_meta=dataset_meta,
            manifest_args=manifest_args,
            metric_rows=rows,
        )


def test_checkpoint_metadata_and_logged_metrics_fail_closed() -> None:
    job, dataset_meta, manifest_args = _checkpoint_fixture()
    rows = _metric_rows(best_epoch=399)
    checkpoint = _checkpoint(
        rows,
        epoch=399,
        job=job,
        dataset_meta=dataset_meta,
        manifest_args=manifest_args,
    )

    checkpoint["method_meta"]["val_split_sha256"] = "d" * 64
    checkpoint["pd"] = 0.5
    with pytest.raises(PersistenceAuditError) as error:
        validate_checkpoint_for_policy(
            checkpoint,
            policy="fixed_epoch",
            job=job,
            dataset_meta=dataset_meta,
            manifest_args=manifest_args,
            metric_rows=rows,
        )
    assert "val_split_sha256" in str(error.value)
    assert "pd disagrees" in str(error.value)


def _grid_jobs() -> list[dict[str, object]]:
    return [
        {"dataset": dataset, "seed": seed, "job_id": f"{dataset}-{seed}"}
        for dataset in DATASET_NAMES
        for seed in SEEDS
    ]


def test_grid_must_be_exactly_three_datasets_by_three_seeds() -> None:
    jobs = _grid_jobs()
    assert validate_grid_cardinality(jobs) == SEEDS

    with pytest.raises(PersistenceAuditError, match="complete"):
        validate_grid_cardinality(jobs[:-1])
    duplicated = jobs + [dict(jobs[0])]
    with pytest.raises(PersistenceAuditError, match="duplicate"):
        validate_grid_cardinality(duplicated)


def test_complete_image_ledger_uses_strict_logit_and_exhaustive_taxonomy() -> None:
    target = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=bool)
    target[20, 20] = True  # direct match
    target[50, 50] = True  # no response
    target[80, 80] = True  # nearby support, illegal centroid
    target[120, 118] = True  # one of this pair is matched
    target[120, 122] = True  # the other is assignment residual

    logits = np.full(target.shape, -1.0, dtype=np.float32)
    logits[20, 20] = 1.0
    logits[80, 82:91] = 1.0
    logits[120, 120] = 1.0
    logits[50, 50] = 0.0  # strict > 0 means this is inactive
    checkpoint = _audit_checkpoint_row()

    rows, target_set = build_image_target_rows(
        logits,
        target,
        dataset="D",
        image_name="image",
        seed=11,
        image_index=0,
        checkpoint=checkpoint,
    )

    assert len(rows) == len(target_set.targets) == 5
    assert {row["stable_target_id"] for row in rows} == {
        target_id.stable_key for target_id in target_set.targets
    }
    assert {row["label_mask_sha256"] for row in rows} == {
        target_set.label_mask_sha256
    }
    assert all(
        row["target_identity"]["component_mask_sha256"]
        for row in rows
    )
    assert all(row["source_component_index"] >= 0 for row in rows)
    assert all(row["source_label"] > 0 for row in rows)
    assert all(row["run"]["job_id"] == "fixture" for row in rows)
    subtypes = [row["outcome_subtype"] for row in rows]
    assert subtypes.count("matched") == 2
    assert subtypes.count("no_response") == 1
    assert subtypes.count("centroid_miss") == 1
    assert subtypes.count("assignment_residual") == 1
    assert all(row["matched"] is (row["outcome_subtype"] == "matched") for row in rows)
    assert all(row["decision"]["operator"] == ">" for row in rows)


def test_no_response_priority_matches_historical_ledger_for_hollow_component() -> None:
    target = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=bool)
    target[128, 126] = True
    target[128, 130] = True
    logits = np.full(target.shape, -1.0, dtype=np.float32)
    logits[123, 123:134] = 1.0
    logits[133, 123:134] = 1.0
    logits[123:134, 123] = 1.0
    logits[123:134, 133] = 1.0
    checkpoint = _audit_checkpoint_row()

    rows, _ = build_image_target_rows(
        logits,
        target,
        dataset="D",
        image_name="hollow",
        seed=11,
        image_index=0,
        checkpoint=checkpoint,
    )

    assert len(rows) == 2
    assert sorted(row["outcome_subtype"] for row in rows) == [
        "matched",
        "no_response",
    ]


def test_noncanonical_or_nonfinite_inference_arrays_fail_closed() -> None:
    checkpoint = _audit_checkpoint_row()
    with pytest.raises(PersistenceAuditError, match="256x256"):
        build_image_target_rows(
            np.zeros((16, 16)),
            np.zeros((16, 16), dtype=bool),
            dataset="D",
            image_name="I",
            seed=11,
            image_index=0,
            checkpoint=checkpoint,
        )
    logits = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE))
    logits[0, 0] = np.nan
    with pytest.raises(PersistenceAuditError, match="non-finite"):
        build_image_target_rows(
            logits,
            np.zeros_like(logits, dtype=bool),
            dataset="D",
            image_name="I",
            seed=11,
            image_index=0,
            checkpoint=checkpoint,
        )


def test_image_operating_metrics_report_achieved_fa_for_both_matchers() -> None:
    logits = np.full((CANONICAL_SIZE, CANONICAL_SIZE), -1.0, dtype=np.float32)
    target = np.zeros_like(logits, dtype=bool)
    target[20, 20] = True
    logits[0, 0] = 1.0
    logits[1, 1] = 0.0  # strict > 0: inactive

    result = compute_image_operating_metrics(logits, target)

    expected = 1_000_000.0 / (CANONICAL_SIZE * CANONICAL_SIZE)
    assert result["threshold_operator"] == ">"
    for matcher in ("hungarian", "legacy"):
        assert result[matcher]["matched_target_components"] == 0
        assert result[matcher]["unmatched_target_components"] == 1
        assert result[matcher]["unmatched_prediction_area"] == 1
        assert result[matcher]["fa_per_million_pixels"] == pytest.approx(expected)


def test_recurrence_annotation_supports_exact_cross_policy_transitions() -> None:
    rows = []
    for seed, subtype in zip(SEEDS, ("no_response", "centroid_miss", "matched")):
        matched = subtype == "matched"
        rows.append(
            {
                "row_kind": "target",
                "dataset": "D",
                "image_name": "I",
                "stable_target_id": "stable",
                "seed": seed,
                "matched": matched,
                "unmatched": not matched,
                "outcome_subtype": subtype,
            }
        )

    annotated = annotate_target_recurrence(rows)

    assert {row["miss_count"] for row in annotated} == {2}
    assert {tuple(row["miss_seed_ids"]) for row in annotated} == {(11, 22)}
    assert {row["no_response_count"] for row in annotated} == {1}
    assert {tuple(row["no_response_seed_ids"]) for row in annotated} == {(11,)}
    assert all(row["observed_two_or_more_miss"] for row in annotated)
    assert not any(row["observed_three_of_three_miss"] for row in annotated)

    with pytest.raises(PersistenceAuditError, match="one row per seed"):
        annotate_target_recurrence(rows + [dict(rows[0])])


def _policy_rows(policy: str, counts: list[int]) -> list[dict[str, object]]:
    rows = []
    for target_index, miss_count in enumerate(counts):
        for seed_index, seed in enumerate(SEEDS):
            unmatched = seed_index < miss_count
            rows.append(
                {
                    "row_kind": "target",
                    "dataset": "D",
                    "image_name": f"I{target_index}",
                    "stable_target_id": f"T{target_index}",
                    "height": CANONICAL_SIZE,
                    "width": CANONICAL_SIZE,
                    "pixel_connectivity": 8,
                    "skimage_connectivity": 2,
                    "label_mask_sha256": f"{target_index + 1:064x}",
                    "component_index": 0,
                    "source_component_index": 0,
                    "source_label": 1,
                    "bbox": [10, 20, 11, 21],
                    "area": 1,
                    "centroid_y": 10.0,
                    "centroid_x": 20.0,
                    "component_mask_sha256": f"{target_index + 11:064x}",
                    "seed": seed,
                    "matched": not unmatched,
                    "unmatched": unmatched,
                    "miss_count": miss_count,
                    "checkpoint": {"policy": policy},
                }
            )
    return rows


def test_cross_policy_comparison_reports_exact_transition_and_retention() -> None:
    fixed = _policy_rows("fixed_epoch", [0, 1, 2, 3])
    best = _policy_rows("best_iou", [3, 2, 1, 0])

    result = compare_policy_target_recurrence(fixed, best)

    matrix = result["overall"]["transition_matrix"]["counts"]
    assert matrix[0][3] == matrix[1][2] == matrix[2][1] == matrix[3][0] == 1
    assert sum(sum(row) for row in matrix) == 4
    jaccard = result["overall"]["missed_set_jaccard"]
    assert jaccard["intersection_target_count"] == 2
    assert jaccard["union_target_count"] == 4
    assert jaccard["value"] == pytest.approx(0.5)
    retention = result["overall"]["fixed_c3_to_best_c_ge2_retention"]
    assert retention["denominator_fixed_c3"] == 1
    assert retention["numerator_best_c_ge2"] == 0
    assert retention["value"] == 0.0
    assert result["overall"]["fixed_epoch_recurrence"]["N3_over_N"] == 0.25
    assert result["overall"]["fixed_epoch_recurrence"][
        "persistent_event_share"
    ] == pytest.approx(0.5)
    assert result["overall"]["best_iou_recurrence"]["N3_over_N"] == 0.25

    with pytest.raises(PersistenceAuditError, match="target universes differ"):
        compare_policy_target_recurrence(fixed, best[:-3])
    tampered = copy.deepcopy(best)
    tampered[0]["area"] = 2
    with pytest.raises(PersistenceAuditError, match="assertion metadata differs"):
        compare_policy_target_recurrence(fixed, tampered)


def test_cross_policy_undefined_denominators_are_explicit() -> None:
    result = compare_policy_target_recurrence(
        _policy_rows("fixed_epoch", [0]),
        _policy_rows("best_iou", [0]),
    )["overall"]

    assert result["missed_set_jaccard"] == {
        "definition": "J({targets:c_fixed>=1},{targets:c_best>=1})",
        "fixed_missed_target_count": 0,
        "best_missed_target_count": 0,
        "intersection_target_count": 0,
        "union_target_count": 0,
        "defined": False,
        "value": None,
        "undefined_reason": "both_missed_sets_empty",
    }
    assert result["fixed_c3_to_best_c_ge2_retention"]["defined"] is False


def test_policy_transition_report_applies_frozen_gate_and_writes_atomically(
    tmp_path: Path,
) -> None:
    report = build_transition_report(
        _policy_rows("fixed_epoch", [3, 1]),
        _policy_rows("best_iou", [2, 0]),
    )

    assert report["overall_routing_gate"] == {
        "thresholds": {
            "missed_set_jaccard_minimum": 0.5,
            "fixed_c3_to_best_c_ge2_retention_minimum": 0.5,
        },
        "missed_set_jaccard_pass": True,
        "fixed_c3_to_best_c_ge2_retention_pass": True,
        "policy_transition_gate_pass": True,
        "undefined_is_pass": False,
    }
    output = tmp_path / "transition"
    write_policy_transition_bundle(
        output,
        report=report,
        provenance={"schema_version": "test"},
    )
    assert {path.name for path in output.iterdir()} == {
        "policy_transition.json",
        "policy_transition.md",
        "provenance.json",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_policy_transition_bundle(
            output,
            report=report,
            provenance={"schema_version": "test"},
        )


def test_policy_comparison_rejects_ledger_hash_drift(tmp_path: Path) -> None:
    bundle = tmp_path / "fixed"
    bundle.mkdir()
    ledger = bundle / "target_persistence.jsonl"
    ledger.write_text('{"row_kind":"target"}\n', encoding="utf-8")
    provenance = bundle / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "artifact_sha256": {
                    ledger.name: persistence_audit.sha256_file(ledger),
                }
            }
        ),
        encoding="utf-8",
    )

    validated = policy_compare._validate_bundle_ledger_hash(ledger)
    assert validated["ledger_sha256"] == persistence_audit.sha256_file(ledger)
    ledger.write_text('{"row_kind":"image"}\n', encoding="utf-8")
    with pytest.raises(PersistenceAuditError, match="hash disagrees"):
        policy_compare._validate_bundle_ledger_hash(ledger)


def test_protocol_fingerprint_freezes_hash_and_mtime_before_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "gate.md"
    second = tmp_path / "north.md"
    first.write_text("frozen gate\n", encoding="utf-8")
    second.write_text("frozen north\n", encoding="utf-8")
    monkeypatch.setattr(
        persistence_audit,
        "PROTOCOL_DOCUMENTS",
        {"gate": first, "north": second},
    )

    frozen = protocol_document_fingerprints()
    validate_protocol_documents_unchanged(frozen)
    assert set(frozen["gate"]) == {
        "path",
        "sha256",
        "mtime_ns",
        "mtime_utc",
    }

    second.write_text("changed after freeze\n", encoding="utf-8")
    with pytest.raises(PersistenceAuditError, match="changed during audit"):
        validate_protocol_documents_unchanged(frozen)


def test_source_freeze_detects_midrun_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter(({"audit": "a"}, {"audit": "b"}))
    monkeypatch.setattr(persistence_audit, "_source_hashes", lambda: next(states))

    persistence_audit.validate_source_files_unchanged({"audit": "a"})
    with pytest.raises(PersistenceAuditError, match="source changed"):
        persistence_audit.validate_source_files_unchanged({"audit": "a"})


def _registries(mask_by_dataset):
    jobs = _grid_jobs()
    authority = {}
    by_job = {}
    for dataset in DATASET_NAMES:
        mask = mask_by_dataset[dataset]
        authority[dataset] = {
            "target": build_stable_target_set(
                mask, dataset=dataset, image_name="target"
            ),
            "empty": build_stable_target_set(
                np.zeros_like(mask), dataset=dataset, image_name="empty"
            ),
        }
    for job in jobs:
        by_job[job["job_id"]] = copy.deepcopy(authority[job["dataset"]])
    return jobs, authority, by_job


def test_every_seed_must_equal_authoritative_masks_including_empty_images() -> None:
    masks = {}
    for index, dataset in enumerate(DATASET_NAMES):
        mask = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=bool)
        mask[10 + index, 20] = True
        masks[dataset] = mask
    jobs, authority, by_job = _registries(masks)

    summary = validate_cross_run_target_registries(by_job, jobs, authority)
    assert all(
        record["all_seeds_exactly_equal_to_authority"]
        for record in summary.values()
    )
    assert all(record["target_free_image_count"] == 1 for record in summary.values())

    changed = masks[DATASET_NAMES[0]].copy()
    changed[10, 21] = True
    bad_job = next(job for job in jobs if job["dataset"] == DATASET_NAMES[0])
    by_job[bad_job["job_id"]]["target"] = build_stable_target_set(
        changed, dataset=DATASET_NAMES[0], image_name="target"
    )
    with pytest.raises(PersistenceAuditError, match="canonical-mask authority"):
        validate_cross_run_target_registries(by_job, jobs, authority)


def test_summary_reports_unique_units_and_no_response_recurrence() -> None:
    masks = {}
    for index, dataset in enumerate(DATASET_NAMES):
        mask = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=bool)
        mask[10 + index, 20] = True
        masks[dataset] = mask
    jobs, authority, by_job = _registries(masks)
    registry_summary = validate_cross_run_target_registries(
        by_job, jobs, authority
    )
    checkpoint = _audit_checkpoint_row()
    rows = []
    for job in jobs:
        dataset = job["dataset"]
        seed = job["seed"]
        for image_index, target_set in enumerate(authority[dataset].values()):
            rows.append(
                build_image_envelope_row(
                    target_set,
                    seed=seed,
                    image_index=image_index,
                    checkpoint=checkpoint,
                    operating_metrics=compute_image_operating_metrics(
                        np.full(
                            (CANONICAL_SIZE, CANONICAL_SIZE),
                            -1.0,
                            dtype=np.float32,
                        ),
                        masks[dataset]
                        if target_set.image_name == "target"
                        else np.zeros_like(masks[dataset]),
                    ),
                )
            )
            for target_id in target_set.targets:
                no_response = seed == SEEDS[0]
                rows.append(
                    {
                        "schema_version": "test",
                        "row_kind": "target",
                        "dataset": dataset,
                        "image_name": target_set.image_name,
                        "image_index": image_index,
                        "seed": seed,
                        "height": target_set.height,
                        "width": target_set.width,
                        "pixel_connectivity": target_set.pixel_connectivity,
                        "skimage_connectivity": target_set.skimage_connectivity,
                        "label_mask_sha256": target_set.label_mask_sha256,
                        "stable_target_id": target_id.stable_key,
                        "component_index": target_id.component_index,
                        "source_component_index": target_id.source_component_index,
                        "source_label": target_id.source_label,
                        "bbox": target_id.bbox,
                        "area": target_id.area,
                        "centroid_y": target_id.centroid_y,
                        "centroid_x": target_id.centroid_x,
                        "component_mask_sha256": target_id.component_mask_sha256,
                        "matched": False,
                        "unmatched": True,
                        "outcome_subtype": (
                            "no_response" if no_response else "centroid_miss"
                        ),
                    }
                )
    rows = annotate_target_recurrence(rows)
    expected_registry = tuple(
        target_set
        for dataset in DATASET_NAMES
        for target_set in authority[dataset].values()
    )

    summary = build_summary(
        rows,
        policy="fixed_epoch",
        registry_summary=registry_summary,
        expected_registry=expected_registry,
        bootstrap_replicates=10,
        bootstrap_seed=7,
    )

    assert summary["ledger_row_count"] == 9
    assert summary["image_envelope_row_count"] == 18
    assert summary["recurrence"]["overall"]["target_micro"]["N3"] == 3
    assert summary["no_response_recurrence"]["overall"]["target_micro"]["N1"] == 3
    units = summary["failure_units"]["overall"]
    assert units["unique_missed_targets"] == 3
    assert units["unique_images_with_at_least_one_miss"] == 3
    assert units["observed_three_of_three_image_count"] == 3
    achieved = summary["achieved_operating_point"]
    assert len(achieved["by_run"]) == 9
    assert all(
        record["hungarian"]["achieved_fa_per_million_pixels"] == 0.0
        and record["legacy"]["achieved_fa_per_million_pixels"] == 0.0
        and record["hungarian"]["pd"] == 0.0
        and record["legacy"]["pd"] == 0.0
        and record["matcher_delta_hungarian_minus_legacy"]["pd"] == 0.0
        for record in achieved["by_run"]
    )
    assert achieved["not_a_budget_matched_operating_point"] is True
    event_bootstrap = summary["recurrence"]["overall"]["bootstrap"]["metrics"][
        "target_micro"
    ]["persistent_event_share"]
    assert "conditional_ci95" in event_bootstrap
    assert "replicates_undefined" in event_bootstrap


def _minimal_summary() -> dict[str, object]:
    count = {
        "target_count": 1,
        "N0": 1,
        "N1": 0,
        "N2": 0,
        "N3": 0,
        "N3_over_N": 0.0,
        "persistent_event_share": None,
    }
    taxonomy = {
        dataset: {
            "matched": 3,
            "no_response": 0,
            "centroid_miss": 0,
            "assignment_residual": 0,
        }
        for dataset in DATASET_NAMES
    }
    return {
        "checkpoint_policy": "fixed_epoch",
        "selection_scope": "fixed_epoch_399_primary",
        "ledger_row_count": 3,
        "recurrence": {
            "overall": {"target_micro": dict(count)},
            "by_dataset": {
                dataset: {"target_micro": dict(count)} for dataset in DATASET_NAMES
            },
        },
        "no_response_recurrence": {
            "overall": {"target_micro": dict(count)},
            "by_dataset": {
                dataset: {"target_micro": dict(count)} for dataset in DATASET_NAMES
            },
        },
        "miss_taxonomy": {"by_dataset": taxonomy},
        "achieved_operating_point": {"by_run": []},
        "failure_units": {
            "overall": {
                "unique_missed_targets": 0,
                "unique_images_with_at_least_one_miss": 0,
                "observed_three_of_three_target_count": 0,
                "observed_three_of_three_image_count": 0,
            },
            "by_dataset": {
                dataset: {
                    "unique_missed_targets": 0,
                    "unique_images_with_at_least_one_miss": 0,
                    "observed_three_of_three_target_count": 0,
                    "observed_three_of_three_image_count": 0,
                }
                for dataset in DATASET_NAMES
            },
        },
        "interpretation_guard": "observed only",
    }


def test_output_bundle_is_atomic_complete_and_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "fixed_epoch"
    rows = [{"dataset": "D", "matched": True}]
    write_output_bundle(
        output,
        rows=rows,
        summary=_minimal_summary(),
        provenance={"schema_version": "p"},
    )

    assert {path.name for path in output.iterdir()} == {
        "target_persistence.jsonl",
        "target_persistence_summary.json",
        "target_persistence_summary.md",
        "provenance.json",
    }
    assert json.loads((output / "target_persistence.jsonl").read_text()) == rows[0]
    provenance = json.loads((output / "provenance.json").read_text())
    assert set(provenance["artifact_sha256"]) == {
        "target_persistence.jsonl",
        "target_persistence_summary.json",
        "target_persistence_summary.md",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_output_bundle(
            output,
            rows=rows,
            summary=_minimal_summary(),
            provenance={"schema_version": "p"},
        )
