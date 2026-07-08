from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.full_dea_head import FullDEAHead
from model.full_dea_loss import full_dea_loss


def test_counterfactual_prediction_changes_loss() -> None:
    torch.manual_seed(13)
    feature = torch.randn(2, 16, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.96).float()

    out = FullDEAHead(in_channels=16)(feature)
    loss_a, _ = full_dea_loss(out, target, lambda_cf=0.5)

    changed = dict(out)
    changed["y_cf"] = out["y_cf"] + 1.0
    loss_b, _ = full_dea_loss(changed, target, lambda_cf=0.5)

    assert torch.abs(loss_a - loss_b) > 1e-5


def test_counterfactual_branch_receives_gradient() -> None:
    torch.manual_seed(17)
    feature = torch.randn(2, 16, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.97).float()

    head = FullDEAHead(in_channels=16)
    out = head(feature)
    out["y_cf"].retain_grad()

    loss, _ = full_dea_loss(out, target, lambda_cf=0.5)
    loss.backward()

    assert out["y_cf"].grad is not None
    assert torch.isfinite(out["y_cf"].grad).all()
    assert out["y_cf"].grad.abs().sum() > 0


if __name__ == "__main__":
    test_counterfactual_prediction_changes_loss()
    test_counterfactual_branch_receives_gradient()
