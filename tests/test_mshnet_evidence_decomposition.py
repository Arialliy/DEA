from __future__ import annotations

import torch

from model.MSHNet import MSHNet
from model.mshnet_evidence_view import forward_mshnet_evidence


def _hook_counts(model: MSHNet) -> tuple[int, ...]:
    return tuple(
        len(getattr(model, name)._forward_hooks)
        for name in ("decoder_0", "decoder_1", "decoder_2", "decoder_3")
    )


def test_evidence_view_preserves_direct_output_and_parameters() -> None:
    torch.manual_seed(31)
    model = MSHNet(3).eval()
    x = torch.randn(2, 3, 64, 80)
    state_before = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    hooks_before = _hook_counts(model)

    with torch.no_grad():
        direct_masks, direct_pred = model(x, True)
        evidence = forward_mshnet_evidence(model, x)

    assert torch.equal(evidence["pred"], direct_pred)
    for observed, direct in zip(evidence["masks"], direct_masks):
        assert torch.equal(observed, direct)
    assert _hook_counts(model) == hooks_before
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key]), key


def test_exact_contribution_reconstruction_and_decoder_shapes() -> None:
    torch.manual_seed(37)
    model = MSHNet(3).eval()
    x = torch.randn(2, 3, 64, 80)

    with torch.no_grad():
        evidence = forward_mshnet_evidence(model, x, detach=True)

    assert evidence["scale_logits"].shape == (2, 4, 64, 80)
    assert evidence["contributions"].shape == (2, 4, 64, 80)
    assert [tuple(feature.shape) for feature in evidence["decoder_features"]] == [
        (2, 16, 64, 80),
        (2, 32, 32, 40),
        (2, 64, 16, 20),
        (2, 128, 8, 10),
    ]
    assert torch.allclose(
        evidence["z_reconstructed"],
        evidence["z_base"],
        atol=1e-4,
        rtol=1e-5,
    )
    assert torch.allclose(
        evidence["z_without_scale"],
        evidence["z_base"] - evidence["contributions"],
        atol=0.0,
        rtol=0.0,
    )
    assert all(not tensor.requires_grad for tensor in evidence["decoder_features"])


def test_evidence_view_keeps_autograd_path() -> None:
    torch.manual_seed(41)
    model = MSHNet(3).eval()
    x = torch.randn(1, 3, 32, 32, requires_grad=True)

    evidence = forward_mshnet_evidence(model, x)
    loss = evidence["pred"].mean() + evidence["contributions"].square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0


if __name__ == "__main__":
    test_evidence_view_preserves_direct_output_and_parameters()
    test_exact_contribution_reconstruction_and_decoder_shapes()
    test_evidence_view_keeps_autograd_path()
