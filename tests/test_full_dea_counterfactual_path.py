from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.full_dea_head import FullDEAHeadV3
from model.full_dea_loss import full_dea_aux_loss_v3


def make_out_and_target():
    torch.manual_seed(13)
    batch, size = 2, 32
    head = FullDEAHeadV3(hidden_channels=16)
    x_d0 = torch.randn(batch, 16, size, size)
    x_d1 = torch.randn(batch, 32, size // 2, size // 2)
    x_d2 = torch.randn(batch, 64, size // 4, size // 4)
    x_d3 = torch.randn(batch, 128, size // 8, size // 8)
    scale_logits_full = torch.randn(batch, 4, size, size)
    fusion_weight = torch.randn(1, 4, 3, 3)
    fusion_bias = torch.randn(1)
    z_base = F.conv2d(scale_logits_full, fusion_weight, fusion_bias, padding=1)
    out = head(
        x_d0,
        x_d1,
        x_d2,
        x_d3,
        scale_logits_full,
        z_base,
        fusion_weight,
        fusion_bias,
    )
    target = (torch.rand(batch, 1, size, size) > 0.97).float()
    return head, out, target


def test_clutter_magnitude_changes_loss() -> None:
    _, out, target = make_out_and_target()
    loss_a, _ = full_dea_aux_loss_v3(
        out,
        target,
        epoch=1,
        warm_epoch=0,
        tau_base=0.0,
        tau_target=0.0,
        tau_scale=0.0,
    )

    changed = dict(out)
    changed["clutter_amount"] = out["clutter_amount"] + 1.0
    changed["z_final"] = (
        out["z_target"]
        - out["alpha"] * out["clutter_prob"].detach() * changed["clutter_amount"]
    )
    loss_b, _ = full_dea_aux_loss_v3(
        changed,
        target,
        epoch=1,
        warm_epoch=0,
        tau_base=0.0,
        tau_target=0.0,
        tau_scale=0.0,
    )

    assert torch.abs(loss_a - loss_b) > 1e-5


def test_decision_and_magnitude_heads_receive_gradient() -> None:
    head, out, target = make_out_and_target()

    loss, _ = full_dea_aux_loss_v3(
        out,
        target,
        epoch=1,
        warm_epoch=0,
        tau_base=0.0,
        tau_target=0.0,
        tau_scale=0.0,
    )
    loss.backward()

    assert head.decision_head.weight.grad is not None
    assert torch.isfinite(head.decision_head.weight.grad).all()
    assert head.decision_head.weight.grad.abs().sum() > 0
    assert head.magnitude_head.weight.grad is not None
    assert torch.isfinite(head.magnitude_head.weight.grad).all()
    assert head.magnitude_head.weight.grad.abs().sum() > 0


if __name__ == "__main__":
    test_clutter_magnitude_changes_loss()
    test_decision_and_magnitude_heads_receive_gradient()
