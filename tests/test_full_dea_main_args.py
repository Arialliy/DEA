from __future__ import annotations

import os
import sys
from argparse import Namespace

import pytest
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import Trainer, get_method_metadata, get_method_name, get_run_folder_name, validate_args
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
