import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.MSHNet import MSHNet
from tools.run_signed_readout_probe import (
    ALL_VARIANTS,
    BUDGETS,
    BUNDLE_FILES,
    MATCHERS,
    PROBE_FORMAL_EPOCHS,
    PROBE_SMOKE_EPOCHS,
    TRAINABLE_VARIANTS,
    SignedReadoutProbeError,
    _module_state_sha256,
    _write_bundle,
    assert_native_final_hard_core_replay,
    build_hard_core_matching,
    build_probe_heads,
    class_balanced_bce,
    evaluate_variant,
    frozen_d0_and_native_logits,
    parse_args,
    protocol_epochs,
    summarize_hard_core,
    train_probe_heads,
    validate_front_freeze_bundle,
    validate_gate_g_job_authority,
)
from utils.cross_fitted_low_fa import image_fold
from utils.nested_component_grid import build_nested_quantile_probability_grids
from utils.target_identity import build_stable_target_set


ROOT = Path(__file__).resolve().parents[1]


def test_variant_inventory_and_parameter_matching_are_frozen() -> None:
    assert ALL_VARIANTS == (
        "original_final_z",
        "original_output0",
        "refit_raw",
        "refit_annulus_centered",
        "refit_signed_standardized",
        "refit_unsigned_standardized_projection",
    )
    heads = build_probe_heads(20260711, torch.device("cpu"))
    assert tuple(heads) == TRAINABLE_VARIANTS
    assert {
        name: sum(parameter.numel() for parameter in head.parameters())
        for name, head in heads.items()
    } == {name: 17 for name in TRAINABLE_VARIANTS}
    initial_weights = [heads[name].readout.weight.detach() for name in heads]
    assert all(torch.equal(initial_weights[0], value) for value in initial_weights[1:])


def test_protocol_epochs_disallows_post_hoc_smoke_scope() -> None:
    assert protocol_epochs("formal", "IRSTD-1K", 1) == PROBE_FORMAL_EPOCHS
    assert (
        protocol_epochs("smoke", "NUAA-SIRST", 20260711)
        == PROBE_SMOKE_EPOCHS
    )
    with pytest.raises(SignedReadoutProbeError, match="predeclared"):
        protocol_epochs("smoke", "NUDT-SIRST", 20260711)
    with pytest.raises(SignedReadoutProbeError, match="unknown"):
        protocol_epochs("tuned", "NUAA-SIRST", 20260711)


def test_cli_is_one_job_and_has_no_variant_hyperparameters() -> None:
    args = parse_args(
        (
            "--dataset",
            "NUAA-SIRST",
            "--seed",
            "20260711",
            "--protocol",
            "smoke",
            "--device",
            "cpu",
            "--output-dir",
            "/tmp/gate-k-test",
        )
    )
    assert (args.dataset, args.seed, args.protocol) == (
        "NUAA-SIRST",
        20260711,
        "smoke",
    )
    assert not hasattr(args, "epochs")
    assert not hasattr(args, "lr")
    assert not any(name.startswith("raw_") for name in vars(args))


def test_class_balanced_bce_is_per_image_and_defines_empty_images() -> None:
    logits = torch.tensor(
        [
            [[[2.0, -1.0], [0.5, -2.0]]],
            [[[1.0, -1.0], [2.0, -2.0]]],
        ],
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
        ]
    )
    loss, counts = class_balanced_bce(logits, targets)
    pixel = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected_positive = 0.5 * pixel[0, 0, 0, 0] + 0.5 * torch.stack(
        (pixel[0, 0, 0, 1], pixel[0, 0, 1, 0], pixel[0, 0, 1, 1])
    ).mean()
    expected_empty = pixel[1].mean()
    assert torch.allclose(loss, 0.5 * (expected_positive + expected_empty))
    assert counts == {"foreground_images": 1, "empty_images": 1}
    loss.backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())


@pytest.mark.parametrize(
    "logits,targets",
    (
        (torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2)),
        (torch.full((1, 1, 2, 2), float("nan")), torch.zeros(1, 1, 2, 2)),
        (torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), 2.0)),
        (torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2)),
    ),
)
def test_class_balanced_bce_fails_closed(logits: torch.Tensor, targets: torch.Tensor) -> None:
    with pytest.raises(SignedReadoutProbeError):
        class_balanced_bce(logits, targets)


def test_frozen_d0_extraction_matches_native_graph_and_preserves_bn() -> None:
    torch.manual_seed(3)
    model = MSHNet(3).requires_grad_(False).eval()
    images = torch.randn(1, 3, 32, 32)
    state_before = _module_state_sha256(model)
    hook_count = len(model.decoder_0._forward_hooks)
    d0, output0, final_z = frozen_d0_and_native_logits(model, images)
    with torch.no_grad():
        masks, direct_final = model(images, True)
    assert tuple(d0.shape) == (1, 16, 32, 32)
    assert torch.equal(output0, masks[0])
    assert torch.equal(final_z, direct_final)
    assert not d0.requires_grad
    assert _module_state_sha256(model) == state_before
    assert len(model.decoder_0._forward_hooks) == hook_count
    assert all(not module.training for module in model.modules())


def test_frozen_d0_extraction_rejects_training_or_trainable_backbone() -> None:
    model = MSHNet(3)
    images = torch.randn(1, 3, 32, 32)
    with pytest.raises(SignedReadoutProbeError, match="eval"):
        frozen_d0_and_native_logits(model, images)
    model.eval()
    with pytest.raises(SignedReadoutProbeError, match="trainable"):
        frozen_d0_and_native_logits(model, images)


def test_common_training_uses_one_shared_d0_and_never_updates_backbone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_signed_readout_probe as tool

    class TinyFrozen(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, 1)
            self.bn = torch.nn.BatchNorm2d(1)

    model = TinyFrozen().requires_grad_(False).eval()
    model_before = _module_state_sha256(model)
    calls = []

    def fake_extract(frozen_model, images):
        assert frozen_model is model
        assert all(not module.training for module in frozen_model.modules())
        assert all(not parameter.requires_grad for parameter in frozen_model.parameters())
        calls.append(id(images))
        d0 = images[:, :1].detach().repeat(1, 16, 1, 1)
        native = torch.zeros_like(images[:, :1])
        return d0, native, native

    monkeypatch.setattr(tool, "frozen_d0_and_native_logits", fake_extract)
    heads = build_probe_heads(17, torch.device("cpu"))
    initial = {name: _module_state_sha256(head) for name, head in heads.items()}
    generator = torch.Generator().manual_seed(5)
    images = torch.randn(4, 3, 12, 12, generator=generator)
    masks = torch.zeros(4, 1, 12, 12)
    masks[0, 0, 5, 5] = 1.0
    masks[1, 0, 3:5, 3:5] = 1.0
    history, protocol = train_probe_heads(
        model,
        heads,
        [(images, masks)],
        device=torch.device("cpu"),
        epochs=PROBE_SMOKE_EPOCHS,
    )
    assert len(calls) == PROBE_SMOKE_EPOCHS
    assert len(history) == PROBE_SMOKE_EPOCHS * len(TRAINABLE_VARIANTS)
    assert protocol["epochs"] == PROBE_SMOKE_EPOCHS
    assert protocol["loss"].startswith("per-image class-balanced")
    assert protocol["total_optimizer_steps"] == PROBE_SMOKE_EPOCHS
    assert {row["foreground_images"] for row in history} == {2}
    assert {row["empty_images"] for row in history} == {2}
    assert _module_state_sha256(model) == model_before
    assert all(
        _module_state_sha256(heads[name]) != initial[name]
        for name in TRAINABLE_VARIANTS
    )


def _two_fold_names(count: int = 4) -> tuple[str, ...]:
    by_fold = {0: [], 1: []}
    index = 0
    while min(len(values) for values in by_fold.values()) < count // 2:
        name = f"image_{index}"
        fold = image_fold(name)
        if len(by_fold[fold]) < count // 2:
            by_fold[fold].append(name)
        index += 1
    # Interleave folds while retaining an explicit deterministic order.
    return tuple(value for pair in zip(by_fold[0], by_fold[1]) for value in pair)


def test_variant_evaluation_uses_explicit_q2_and_reports_overshoot_all_off() -> None:
    names = _two_fold_names()
    scores = []
    targets = []
    registry = {}
    for image_index, name in enumerate(names):
        target = np.zeros((8, 8), dtype=bool)
        target[3, 3] = True
        score = np.full((8, 8), -4.0, dtype=np.float32)
        score[3, 3] = 4.0 + image_index * 0.01
        scores.append(score)
        targets.append(target)
        registry[name] = build_stable_target_set(
            target,
            dataset="SYNTH",
            image_name=name,
            connectivity=2,
        )
    evaluation, oracle, cross_targets, cross_images, calibration = evaluate_variant(
        tuple(scores),
        tuple(targets),
        names,
        registry,
        dataset="SYNTH",
        seed=7,
        variant="refit_raw",
        checkpoint_record={"path": "/frozen/checkpoint.pkl", "sha256": "0" * 64},
    )
    q2 = build_nested_quantile_probability_grids()[-1].probabilities
    assert evaluation["q2_probability_count"] == len(q2)
    assert len(calibration) == len(MATCHERS) * 2
    assert all(tuple(row["tail_quantiles"]) == q2 for row in calibration)
    assert len(oracle) == len(names) * len(MATCHERS) * len(BUDGETS)
    assert len(cross_targets) == len(oracle)
    assert len(cross_images) == len(names) * len(MATCHERS) * len(BUDGETS)
    assert evaluation["fixed_logit0_pixel"] == {
        "intersection_pixels": len(names),
        "union_pixels": len(names),
        "prediction_pixels": len(names),
        "target_pixels": len(names),
        "iou": 1.0,
        "strict_prediction_rule": "logit > threshold",
    }
    json.dumps(evaluation, allow_nan=False)
    for matcher in MATCHERS:
        assert evaluation["oracle_selection_audit"][matcher][
            "all_off_candidate_present"
        ]
        for budget in BUDGETS:
            audit = evaluation["crossfit_selection_audit"][matcher][str(budget)]
            assert audit["calibration_folds"] == 2
            assert audit["calibration_all_off_candidate_present_folds"] == 2
            assert 0 <= audit["calibration_selected_all_off_folds"] <= 2
            assert 0 <= audit["held_out_overshoot_folds"] <= 2
            assert 0 <= audit["held_out_all_off_folds"] <= 2
            pixel = evaluation["crossfit_pixel"][matcher][str(budget)]
            assert 0.0 <= pixel["iou"] <= 1.0
            assert pixel["target_pixels"] == len(names)
            assert pixel["strict_prediction_rule"] == "logit > threshold"


def test_hard_core_report_is_dataset_local_and_complete() -> None:
    stable_id = "formal-nuaa-id"
    panel = [
        {
            "dataset": "NUAA-SIRST",
            "stable_target_id": stable_id,
            "image_name": "Misc_388",
            "target_area": 29,
        },
        {
            "dataset": "IRSTD-1K",
            "stable_target_id": "foreign-id",
            "image_name": "XDU1",
            "target_area": 1,
        },
    ]
    oracle = []
    crossfit = []
    for variant in ALL_VARIANTS:
        for matcher in MATCHERS:
            for budget in BUDGETS:
                base = {
                    "variant": variant,
                    "matcher": matcher,
                    "nominal_budget_fa_per_mpix": budget,
                    "stable_target_id": stable_id,
                    "image_name": "Misc_388",
                }
                oracle.append(
                    {**base, "oracle_matched": True, "oracle_threshold": 0.25}
                )
                crossfit.append(
                    {
                        **base,
                        "low_fa_matched": budget >= 10,
                        "calibration_threshold": 0.5,
                        "evaluation_fold": 1,
                        "fixed_logit0_matched": False,
                    }
                )
    rows = build_hard_core_matching(
        panel,
        oracle,
        crossfit,
        dataset="NUAA-SIRST",
        seed=20260711,
    )
    assert len(rows) == len(ALL_VARIANTS) * len(MATCHERS) * len(BUDGETS)
    assert {row["stable_target_id"] for row in rows} == {stable_id}
    summary = summarize_hard_core(rows)
    assert summary["original_final_z"]["official_legacy"]["5"] == {
        "targets": 1,
        "oracle_matched": 1,
        "crossfit_matched": 0,
    }
    with pytest.raises(SignedReadoutProbeError, match="contradicts"):
        assert_native_final_hard_core_replay(rows, expected_targets=1)
    for row in rows:
        if row["variant"] == "original_final_z" and int(
            row["nominal_budget_fa_per_mpix"]
        ) == 20:
            row["oracle_matched"] = False
            row["crossfit_matched"] = False
    assert_native_final_hard_core_replay(rows, expected_targets=1)


def test_front_freeze_authority_is_current_and_hash_validated() -> None:
    record = validate_front_freeze_bundle(
        ROOT / "repro_runs" / "gate_i" / "front_freeze_confirmatory_v1"
    )
    assert len(record["records"]) == 16
    assert record["routing"]["recommended_first_mutable_boundary"] == (
        "after_d0_prediction_conversion"
    )


def test_gate_g_job_authority_binds_checkpoint_split_and_formal_status(
    tmp_path: Path,
) -> None:
    source = tmp_path / "targets.jsonl"
    checkpoint_sha = "a" * 64
    split_sha = "b" * 64
    row = {
        "grid_level": "Q2",
        "nominal_budget_fa_per_mpix": 20,
        "seed": 7,
        "dataset": "D",
        "stable_target_id": "id",
        "category_core": "no_feasible_local_peak_activation",
        "joint_global_oracle_matched": False,
        "selected_legacy_matched": False,
        "selected_hungarian_matched": False,
        "checkpoint": {
            "sha256": checkpoint_sha,
            "job_id": "job",
            "validation_split_sha256": split_sha,
            "policy": "fixed_epoch",
            "epoch": 399,
        },
    }
    source.write_text(json.dumps(row) + "\n")
    front = {
        "records": [{"dataset": "D", "stable_target_id": "id"}],
        "hard_core_source": str(source),
        "hard_core_source_sha256": "c" * 64,
    }
    job = {
        "checkpoint_sha256": checkpoint_sha,
        "job_id": "job",
        "split_hashes": {"validation": split_sha},
        "checkpoint_summary": {"epoch": 399},
    }
    record = validate_gate_g_job_authority(front, job, dataset="D", seed=7)
    assert record["row_count"] == 1
    assert record["checkpoint_sha256"] == checkpoint_sha
    row["selected_legacy_matched"] = True
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises(SignedReadoutProbeError, match="phenotype"):
        validate_gate_g_job_authority(front, job, dataset="D", seed=7)


def test_atomic_bundle_contains_saved_six_way_logits_and_hashes(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    logits = {
        name: (
            np.full((4, 4), index, dtype=np.float32),
            np.full((4, 4), index + 0.5, dtype=np.float32),
        )
        for index, name in enumerate(ALL_VARIANTS)
    }
    _write_bundle(
        output,
        summary={"schema": "summary", "status": "complete"},
        history=[{"epoch": 0}],
        oracle_rows=[],
        crossfit_target_rows=[],
        crossfit_image_rows=[],
        calibration_rows=[],
        hard_core_rows=[],
        logits=logits,
        image_names=("a", "b"),
        head_payload={"state_dict": {"x": torch.ones(1)}},
        provenance={"schema": "provenance"},
    )
    assert {path.name for path in output.iterdir()} == set(BUNDLE_FILES)
    with np.load(output / "dev_logits.npz") as arrays:
        assert set(arrays.files) == {"image_names", *ALL_VARIANTS}
        assert arrays["refit_signed_standardized"].shape == (2, 4, 4)
    provenance = json.loads((output / "provenance.json").read_text())
    assert set(provenance["artifact_sha256"]) == set(BUNDLE_FILES[:-1])
    with pytest.raises(FileExistsError):
        _write_bundle(
            output,
            summary={},
            history=[],
            oracle_rows=[],
            crossfit_target_rows=[],
            crossfit_image_rows=[],
            calibration_rows=[],
            hard_core_rows=[],
            logits=logits,
            image_names=("a", "b"),
            head_payload={},
            provenance={},
        )
