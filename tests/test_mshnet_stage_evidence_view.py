import pytest
import torch

from model.MSHNet import MSHNet
from model.mshnet_stage_evidence_view import forward_mshnet_stage_evidence


def _hook_counts(model):
    return {
        name: (len(module._forward_hooks), len(module._forward_pre_hooks))
        for name, module in model.named_modules()
    }


def test_stage_evidence_captures_exact_native_dag_and_prediction():
    torch.manual_seed(31)
    model = MSHNet(3).eval()
    image = torch.randn(2, 3, 32, 48)
    before_hooks = _hook_counts(model)

    with torch.no_grad():
        expected_masks, expected_pred = model(image, True)
        evidence = forward_mshnet_stage_evidence(model, image)

    assert _hook_counts(model) == before_hooks
    assert torch.equal(evidence["pred"], expected_pred)
    for actual, expected in zip(evidence["native_sides"].values(), expected_masks):
        assert torch.equal(actual, expected)

    expected_shapes = {
        "input": (2, 3, 32, 48),
        "stem": (2, 16, 32, 48),
        "e0": (2, 16, 32, 48),
        "p0": (2, 16, 16, 24),
        "e1": (2, 32, 16, 24),
        "p1": (2, 32, 8, 12),
        "e2": (2, 64, 8, 12),
        "p2": (2, 64, 4, 6),
        "e3": (2, 128, 4, 6),
        "p3": (2, 128, 2, 3),
        "m": (2, 256, 2, 3),
        "j3": (2, 384, 4, 6),
        "d3": (2, 128, 4, 6),
        "j2": (2, 192, 8, 12),
        "d2": (2, 64, 8, 12),
        "j1": (2, 96, 16, 24),
        "d1": (2, 32, 16, 24),
        "j0": (2, 48, 32, 48),
        "d0": (2, 16, 32, 48),
    }
    assert {
        key: tuple(value.shape) for key, value in evidence["path"].items()
    } == expected_shapes
    assert [tuple(value.shape) for value in evidence["native_sides"].values()] == [
        (2, 1, 32, 48),
        (2, 1, 16, 24),
        (2, 1, 8, 12),
        (2, 1, 4, 6),
    ]
    assert all(
        tuple(value.shape) == (2, 1, 32, 48)
        for value in evidence["full_sides"].values()
    )
    assert all(
        tuple(value.shape) == (2, 1, 32, 48)
        for value in evidence["contributions"].values()
    )
    assert torch.allclose(
        evidence["z_reconstructed"], evidence["pred"], atol=1e-6, rtol=1e-5
    )


def test_stage_evidence_detaches_every_exposed_tensor():
    model = MSHNet(3).eval()
    evidence = forward_mshnet_stage_evidence(
        model,
        torch.randn(1, 3, 32, 32, requires_grad=True),
        detach=True,
    )

    tensors = [evidence["pred"], evidence["fusion_bias"], evidence["z_reconstructed"]]
    for group in ("path", "native_sides", "full_sides", "contributions"):
        tensors.extend(evidence[group].values())
    assert all(not tensor.requires_grad for tensor in tensors)


def test_stage_evidence_rejects_non_mshnet_or_non_image_input():
    with pytest.raises(ValueError, match="4-D"):
        forward_mshnet_stage_evidence(MSHNet(3), torch.randn(3, 32, 32))

    with pytest.raises(TypeError, match="required MSHNet modules"):
        forward_mshnet_stage_evidence(
            torch.nn.Conv2d(3, 1, 1), torch.randn(1, 3, 32, 32)
        )
