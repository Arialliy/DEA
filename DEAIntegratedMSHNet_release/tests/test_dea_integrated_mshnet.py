"""Structural tests for DEAIntegratedMSHNet.

Run from the DEA repository root after copying
``model/dea_integrated_mshnet.py`` into ``model/``:

    PYTHONPATH=. pytest -q tests/test_dea_integrated_mshnet.py
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from model.MSHNet import MSHNet
from model.dea_integrated_mshnet import (
    DEAIntegratedMSHNet,
    DecidableEvidenceRoutingCell,
    IntegratedScaleEvidenceFusion,
    count_trainable_parameters,
)


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(20260710)


def _make_pair(route_channels=16):
    baseline = MSHNet(3)
    integrated = DEAIntegratedMSHNet(3, route_channels=route_channels)
    missing, unexpected = integrated.load_mshnet_state_dict(
        copy.deepcopy(baseline.state_dict())
    )
    assert all(key.startswith("dea_cell_") for key in missing)
    assert all(key.startswith("decidability_head.") for key in unexpected)
    return baseline, integrated


def test_checkpoint_embedding_preserves_all_initial_outputs():
    baseline, integrated = _make_pair(route_channels=8)
    baseline.eval()
    integrated.eval()
    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        baseline_masks, baseline_pred = baseline(x, True)
        output = integrated(x, True, return_dict=True)

    for baseline_mask, integrated_mask in zip(baseline_masks, output["masks"]):
        assert torch.max(torch.abs(baseline_mask - integrated_mask)).item() < 1e-5
    assert torch.equal(baseline_pred, output["pred"])
    assert all(torch.all(route["winner"] == 2) for route in output["routes"])


def test_uncertain_winner_is_exact_local_identity():
    cell = DecidableEvidenceRoutingCell(
        encoder_channels=8,
        decoder_channels=8,
        output_channels=8,
        route_channels=4,
        routing_mode="dea",
    )
    baseline_decoder = torch.nn.Conv2d(16, 8, 3, padding=1)
    encoder = torch.randn(2, 8, 16, 16)
    decoder = torch.randn(2, 8, 16, 16)

    with torch.no_grad():
        expected = baseline_decoder(torch.cat([encoder, decoder], dim=1))
        actual, route = cell(encoder, decoder, baseline_decoder)

    assert torch.all(route["winner"] == 2)
    assert torch.count_nonzero(route["target_gate"]).item() == 0
    assert torch.count_nonzero(route["clutter_gate"]).item() == 0
    assert torch.equal(actual, expected)


def test_final_fusion_is_exact_four_scale_decomposition():
    conv = torch.nn.Conv2d(4, 1, 3, padding=1)
    fusion = IntegratedScaleEvidenceFusion.from_conv(conv)
    scale_logits = torch.randn(2, 4, 32, 32)
    contributions = fusion.decompose(scale_logits)
    decomposed = fusion.baseline_from_contributions(contributions)
    baseline_direct = fusion.direct_baseline(scale_logits)
    direct = conv(scale_logits)

    assert contributions.shape == (2, 4, 32, 32)
    assert torch.max(torch.abs(decomposed - direct)).item() < 1e-5
    assert torch.equal(baseline_direct, direct)
    assert tuple(fusion.state_dict()["weight"].shape) == (1, 4, 3, 3)
    assert tuple(fusion.state_dict()["bias"].shape) == (1,)


def test_all_four_routing_cells_receive_nonzero_gradient():
    _, integrated = _make_pair(route_channels=8)
    integrated.train()
    x = torch.randn(2, 3, 32, 32)
    output = integrated(x, True, return_dict=True)
    loss = output["pred"].square().mean()
    for mask in output["masks"]:
        loss = loss + 0.1 * mask.square().mean()
    loss.backward()

    for scale in range(4):
        cell = getattr(integrated, "dea_cell_%d" % scale)
        grad_l1 = sum(
            parameter.grad.detach().abs().sum().item()
            for parameter in cell.parameters()
            if parameter.grad is not None
        )
        assert grad_l1 > 0.0, "dea_cell_%d has zero aggregate gradient" % scale
        for name, parameter in cell.named_parameters():
            assert parameter.grad is not None, "dea_cell_%d.%s has no gradient" % (
                scale,
                name,
            )
            assert parameter.grad.detach().abs().sum().item() > 0.0, (
                "dea_cell_%d.%s has zero gradient" % (scale, name)
            )


def test_all_keep_route_does_not_change_paired_baseline_parameter_update():
    baseline, integrated = _make_pair(route_channels=4)
    baseline.train()
    integrated.train()
    baseline_optimizer = torch.optim.Adagrad(baseline.parameters(), lr=0.005)
    integrated_optimizer = torch.optim.Adagrad(integrated.parameters(), lr=0.005)
    x = torch.randn(2, 3, 32, 32)

    baseline_masks, baseline_pred = baseline(x, True)
    integrated_output = integrated(x, True, return_dict=True)
    assert torch.equal(baseline_pred, integrated_output["pred"])
    assert all(
        torch.all(route["winner"] == 2)
        for route in integrated_output["routes"]
    )

    baseline_loss = baseline_pred.square().mean()
    integrated_loss = integrated_output["pred"].square().mean()
    for baseline_mask, integrated_mask in zip(
        baseline_masks, integrated_output["masks"]
    ):
        baseline_loss = baseline_loss + 0.1 * baseline_mask.square().mean()
        integrated_loss = integrated_loss + 0.1 * integrated_mask.square().mean()
    baseline_loss.backward()
    integrated_loss.backward()

    integrated_parameters = dict(integrated.named_parameters())
    for name, parameter in baseline.named_parameters():
        if name.startswith("decidability_head."):
            continue
        integrated_parameter = integrated_parameters[name]
        assert parameter.grad is not None, name
        assert integrated_parameter.grad is not None, name
        assert torch.equal(parameter.grad, integrated_parameter.grad), name

    baseline_optimizer.step()
    integrated_optimizer.step()
    for name, parameter in baseline.named_parameters():
        if name.startswith("decidability_head."):
            continue
        assert torch.equal(parameter, integrated_parameters[name]), name


def test_uncertain_margin_must_cover_the_bounded_initial_response():
    with pytest.raises(ValueError, match="all-uncertain"):
        DecidableEvidenceRoutingCell(
            encoder_channels=8,
            decoder_channels=8,
            output_channels=8,
            route_channels=4,
            uncertain_margin=0.1,
        )
    DecidableEvidenceRoutingCell(
        encoder_channels=8,
        decoder_channels=8,
        output_channels=8,
        route_channels=4,
        uncertain_margin=0.100001,
    )


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_hard_scale_routing_rejects_continuous_gate_interpolation(mode):
    with pytest.raises(ValueError, match="hard DEA scale routing"):
        DEAIntegratedMSHNet(
            3,
            routing_mode="dea",
            scale_routing=True,
            route_upsample_mode=mode,
        )

    # Continuous interpolation remains available to the explicit soft ablation.
    DEAIntegratedMSHNet(
        3,
        routing_mode="soft_tri",
        scale_routing=True,
        route_upsample_mode=mode,
    )


def test_default_terminal_upsampling_preserves_action_exclusivity():
    conv = torch.nn.Conv2d(4, 1, 3, padding=1)
    fusion = IntegratedScaleEvidenceFusion.from_conv(conv)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    clutter = 1.0 - target
    route = {"target_gate": target, "clutter_gate": clutter}
    details = fusion(
        torch.randn(1, 4, 8, 8),
        [route, route, route, route],
        return_details=True,
    )
    overlap = (details["target_gates"] > 0) & (details["clutter_gates"] > 0)
    assert torch.count_nonzero(overlap).item() == 0


def test_formal_model_contains_no_discontinued_mechanisms():
    model = DEAIntegratedMSHNet(3)
    forbidden_tokens = (
        "topology",
        "prototype",
        "component_graph",
        "relation_graph",
        "bridge",
    )
    bad = [
        name for name, _ in model.named_modules()
        if any(token in name.lower() for token in forbidden_tokens)
    ]
    assert bad == []
    assert not hasattr(model, "decidability_head")


def test_parameter_increment_is_small_and_explicit():
    baseline = MSHNet(3)
    integrated = DEAIntegratedMSHNet(3, route_channels=16)
    route_parameters = sum(
        parameter.numel()
        for name, parameter in integrated.named_parameters()
        if name.startswith("dea_cell_")
    )

    # Exact count for the four route cells at route_channels=16.  The final
    # fusion reuses the original final.weight/final.bias and adds no parameter.
    assert route_parameters == 20_988
    assert count_trainable_parameters(integrated.final) == count_trainable_parameters(baseline.final)
    assert route_parameters < 25_000


def test_required_ablation_switches_share_one_implementation():
    cases = [
        # scale fusion only
        dict(decoder_routing=False, scale_routing=True, routing_mode="dea"),
        # decoder fusion only
        dict(decoder_routing=True, scale_routing=False, routing_mode="dea"),
        # complete integrated DEA
        dict(decoder_routing=True, scale_routing=True, routing_mode="dea"),
        # no uncertain identity
        dict(decoder_routing=True, scale_routing=True, routing_mode="soft_tri"),
        # ordinary continuous attention
        dict(decoder_routing=True, scale_routing=True, routing_mode="attention"),
    ]
    x = torch.randn(1, 3, 32, 32)
    for kwargs in cases:
        model = DEAIntegratedMSHNet(3, route_channels=4, **kwargs).eval()
        with torch.no_grad():
            masks, pred = model(x, True)
        assert len(masks) == 4
        assert pred.shape == (1, 1, 32, 32)
