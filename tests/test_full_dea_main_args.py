from __future__ import annotations

import os
import sys
from argparse import Namespace

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import Trainer, get_method_metadata, get_method_name, get_run_folder_name, validate_args
from model.MSHNet import MSHNet
from model.dea_mshnet import DEAMSHNet
from model.full_dea_mshnet import FullDEAMSHNet


def make_args(**kwargs):
    args = Namespace(
        model_type="mshnet",
        init_from_baseline="",
        if_checkpoint=False,
        dea_lambda_single=0.0,
        dea_lambda_dec=0.0,
        dea_lambda_empty=0.0,
        full_dea_lambda=1.0,
        full_dea_version="v3",
        full_dea_ramp_epochs=30,
        full_dea_start_epoch=0,
        full_dea_freeze_backbone_epochs=0,
        full_dea_tau_base=0.45,
        full_dea_tau_target=0.45,
        full_dea_tau_scale=0.45,
        full_dea_topk_ratio=0.001,
        full_dea_topk_min_score=0.45,
        full_dea_max_hard_bg_ratio=0.003,
        full_dea_safe_kernel=15,
        full_dea_protect_kernel=9,
        full_dea_hard_min_area=1,
        full_dea_hard_max_area=256,
        integrated_route_channels=16,
        integrated_route_temperature=1.0,
        integrated_routing_mode="dea",
        integrated_decoder_routing=True,
        integrated_scale_routing=True,
        integrated_route_upsample_mode="nearest-exact",
        integrated_update_limit=0.25,
        integrated_uncertain_margin=1.0,
        integrated_route_loss_weight=0.05,
        integrated_route_ramp_epochs=3,
        integrated_isolate_route_gradients=True,
        predictive_state_channels=32,
        predictive_step_size=1.0,
        predictive_delta_init=1.0,
        predictive_delta_min=0.05,
        predictive_legacy_numerics=False,
        predictive_log_interval=50,
        dataset_dir="datasets/NUAA-SIRST",
        seed=20260706,
        deterministic=True,
    )
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def test_full_dea_rejects_dea_lite_lambdas() -> None:
    args = make_args(model_type="full_dea", dea_lambda_single=0.01)
    with pytest.raises(ValueError):
        validate_args(args)


def test_full_dea_rejects_invalid_safe_kernel() -> None:
    args = make_args(model_type="full_dea", full_dea_safe_kernel=14)
    with pytest.raises(ValueError):
        validate_args(args)


def test_method_metadata_names_full_dea_v3() -> None:
    args = validate_args(make_args(model_type="full_dea"))
    assert get_method_name(args) == "FullDEA-v3-TPS"
    meta = get_method_metadata(args)
    assert meta["model_type"] == "full_dea"
    assert meta["method"] == "FullDEA-v3-TPS"
    assert meta["full_dea_version"] == "v3"


def test_method_metadata_can_name_full_dea_v2_for_audit() -> None:
    args = validate_args(make_args(model_type="full_dea", full_dea_version="v2"))
    assert get_method_name(args) == "FullDEA-v2"


def test_method_metadata_names_full_dea_v4_relation_reasoner() -> None:
    args = validate_args(make_args(model_type="full_dea", full_dea_version="v4"))
    assert get_method_name(args) == "FullDEA-v4-CRR"
    meta = get_method_metadata(args)
    assert meta["full_dea_version"] == "v4"


def test_method_metadata_names_full_dea_v5_hard_transport() -> None:
    args = validate_args(make_args(model_type="full_dea", full_dea_version="v5"))
    assert get_method_name(args) == "FullDEA-v5-CRR-HT"


def test_run_folder_name_uses_method_name() -> None:
    args = validate_args(make_args(model_type="full_dea"))
    assert get_run_folder_name(args, "2026-07-09-22-00-00") == (
        "FullDEA-v3-TPS-2026-07-09-22-00-00"
    )


def test_method_metadata_persists_run_label() -> None:
    args = validate_args(make_args(run_label="nuaa_seed_11"))
    assert get_method_metadata(args)["run_label"] == "nuaa_seed_11"


def test_integrated_method_name_exposes_residual_alignment() -> None:
    args = validate_args(make_args(model_type="dea_integrated"))
    assert get_method_name(args) == "DEAIntegrated-ResidualAligned"


def test_integrated_rejects_route_loss_for_attention_control() -> None:
    args = make_args(
        model_type="dea_integrated",
        integrated_routing_mode="attention",
        integrated_route_loss_weight=0.05,
    )
    with pytest.raises(ValueError, match="not defined for the attention"):
        validate_args(args)


def test_integrated_rejects_nonexclusive_hard_gate_interpolation() -> None:
    args = make_args(
        model_type="dea_integrated",
        integrated_route_upsample_mode="bilinear",
    )
    with pytest.raises(ValueError, match="Hard scale routing"):
        validate_args(args)


def test_dea_main_method_name_exposes_state_width() -> None:
    args = validate_args(make_args(model_type="dea"))
    assert get_method_name(args) == "DEA-v0-C32-Eta1"
    metadata = get_method_metadata(args)
    assert metadata["dea_state_channels"] == 32
    assert metadata["dea_step_size"] == 1.0
    assert metadata["dea_legacy_numerics"] is False

    half_step = validate_args(make_args(
        model_type="dea",
        predictive_step_size=0.5,
        predictive_legacy_numerics=True,
    ))
    assert get_method_name(half_step) == (
        "DEA-v0-C32-Eta0p5-LegacyNum"
    )
    compatibility_alias = validate_args(make_args(
        model_type="predictive_correction"
    ))
    assert get_method_name(compatibility_alias) == (
        "PredictiveCorrection-C32-Eta1"
    )


def test_dea_checkpoint_metadata_rejects_numerics_mismatch() -> None:
    args = validate_args(make_args(model_type="dea"))
    trainer = Trainer.__new__(Trainer)
    trainer.args = args
    metadata = get_method_metadata(args)

    Trainer.validate_integrated_checkpoint_metadata(
        trainer, {"method_meta": metadata}, check_split_hashes=False
    )
    incompatible = dict(metadata)
    incompatible["dea_legacy_numerics"] = True
    with pytest.raises(RuntimeError, match="legacy_numerics"):
        Trainer.validate_integrated_checkpoint_metadata(
            trainer, {"method_meta": incompatible}, check_split_hashes=False
        )
    missing_field = dict(metadata)
    missing_field.pop("dea_legacy_numerics")
    with pytest.raises(RuntimeError, match="<missing>"):
        Trainer.validate_integrated_checkpoint_metadata(
            trainer, {"method_meta": missing_field}, check_split_hashes=False
        )


def test_dea_partial_load_accepts_only_replaced_decoder_keys() -> None:
    torch.manual_seed(11)
    baseline = MSHNet(3)
    predictive = DEAMSHNet(3, state_channels=32)
    trainer = Trainer.__new__(Trainer)
    trainer.model = predictive

    Trainer.load_model_state_partial(
        trainer,
        baseline.state_dict(),
        allowed_missing_prefixes=DEAMSHNet.BASELINE_MISSING_PREFIXES,
        allowed_unexpected_prefixes=DEAMSHNet.BASELINE_UNEXPECTED_PREFIXES,
    )

    assert torch.equal(
        predictive.conv_init.weight, baseline.conv_init.weight
    )
    assert torch.equal(
        predictive.encoder_3[1].conv2.weight,
        baseline.encoder_3[1].conv2.weight,
    )


def test_dea_main_rejects_invalid_dynamics() -> None:
    with pytest.raises(ValueError, match="state-channels"):
        validate_args(make_args(
            model_type="dea",
            predictive_state_channels=1,
        ))
    with pytest.raises(ValueError, match="step-size"):
        validate_args(make_args(
            model_type="dea",
            predictive_step_size=1.5,
        ))
    with pytest.raises(ValueError, match="delta-init"):
        validate_args(make_args(
            model_type="dea",
            predictive_delta_init=0.01,
        ))


def test_frozen_backbone_keeps_batchnorm_statistics_fixed() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.args = Namespace(
        model_type="full_dea",
        full_dea_freeze_backbone_epochs=2,
    )
    trainer.model = FullDEAMSHNet(input_channels=3, full_dea_version="v3")
    trainer.model.train()

    Trainer.configure_full_dea_trainable(trainer, epoch=0)

    for name, parameter in trainer.model.named_parameters():
        assert parameter.requires_grad == name.startswith("full_dea_head")
    for name, module in trainer.model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert not module.training, name
    assert trainer.model.full_dea_head.training
