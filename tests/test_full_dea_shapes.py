from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.full_dea_head import FullDEAHead
from model.full_dea_loss import full_dea_loss
from model.full_dea_mshnet import FullDEAMSHNet


def test_full_dea_head_shapes_and_finite_loss() -> None:
    torch.manual_seed(7)
    feature = torch.randn(2, 16, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.95).float()

    head = FullDEAHead(in_channels=16)
    out = head(feature)

    expected_image_shape = (2, 1, 32, 32)
    for key in [
        "target_evidence",
        "clutter_evidence",
        "counterfactual_gate",
        "y_real",
        "y_cf",
        "evidence_gate",
        "y_final",
    ]:
        assert out[key].shape == expected_image_shape, (key, out[key].shape)

    assert out["counterfactual_feature"].shape == feature.shape

    loss, logs = full_dea_loss(out, target)
    assert torch.isfinite(loss)
    assert logs["loss_cf"].ndim == 0


def test_full_dea_mshnet_wrapper_shapes() -> None:
    torch.manual_seed(11)
    model = FullDEAMSHNet(input_channels=3)
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        scale_logits, y_final, out = model(x, warm_flag=True, return_full_dea=True)

    assert len(scale_logits) == 4
    assert y_final.shape == (2, 1, 64, 64)
    assert out["target_evidence"].shape == (2, 1, 64, 64)
    assert out["clutter_evidence"].shape == (2, 1, 64, 64)
    assert out["counterfactual_feature"].shape == (2, 16, 64, 64)


if __name__ == "__main__":
    test_full_dea_head_shapes_and_finite_loss()
    test_full_dea_mshnet_wrapper_shapes()
