from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.full_dea_head import FullDEAHeadV3, FullDEAHeadV4, FullDEAHeadV5
from model.dea_evidence import (
    AttributionGuidedRelationSelector,
    AttributionTopologyBridge,
    gather_chosen_relation_endpoints,
)
from model.full_dea_loss import (
    _limit_component_mask_by_score,
    build_component_hard_clutter_label,
    full_dea_aux_loss_v3,
    full_dea_aux_loss_v4,
)
from model.full_dea_mshnet import FullDEAMSHNet


def make_head_inputs(batch: int = 2, size: int = 32):
    torch.manual_seed(7)
    x_d0 = torch.randn(batch, 16, size, size)
    x_d1 = torch.randn(batch, 32, size // 2, size // 2)
    x_d2 = torch.randn(batch, 64, size // 4, size // 4)
    x_d3 = torch.randn(batch, 128, size // 8, size // 8)
    scale_logits_full = torch.randn(batch, 4, size, size)
    fusion_weight = torch.randn(1, 4, 3, 3)
    fusion_bias = torch.randn(1)
    z_base = F.conv2d(scale_logits_full, fusion_weight, fusion_bias, padding=1)
    return (
        x_d0,
        x_d1,
        x_d2,
        x_d3,
        scale_logits_full,
        z_base,
        fusion_weight,
        fusion_bias,
    )


def test_full_dea_v3_head_shapes_and_baseline_init() -> None:
    head = FullDEAHeadV3(hidden_channels=16)
    head.eval()
    inputs = make_head_inputs(size=32)

    with torch.no_grad():
        out = head(*inputs)

    expected = (2, 1, 32, 32)
    for key in [
        "target_evidence",
        "clutter_evidence",
        "uncertain_evidence",
        "target_gate",
        "clutter_gate",
        "protect_prob",
        "target_boost",
        "target_delta",
        "z_target",
        "z_clutter",
        "raw_suppression_gate",
        "protected_suppression_gate",
        "suppression_gate",
        "z_final",
        "y_final",
    ]:
        assert out[key].shape == expected, (key, out[key].shape)

    z_base = inputs[5]
    assert torch.mean(torch.abs(out["z_final"] - z_base)) < 1e-3
    assert torch.allclose(out["z_reconstructed"], z_base, atol=1e-5, rtol=1e-5)
    assert out["bridge_feature"].shape == (2, 16, 32, 32)
    assert out["topology_prior"].shape == expected
    assert out["bridge_delta"].shape == expected
    assert torch.isfinite(out["bridge_feature"]).all()
    assert torch.allclose(
        out["decision_probs"].sum(dim=1, keepdim=True),
        torch.ones_like(z_base),
        atol=1e-6,
    )
    assert float(out["uncertain_prob"].min()) > 0.9
    assert float(out["target_delta"].min()) >= 0.0
    assert float(out["clutter_delta"].min()) >= 0.0
    assert float(out["bridge_delta"].min()) >= 0.0


def test_topology_bridge_proposes_gap_between_fragments() -> None:
    bridge = AttributionTopologyBridge(scale_channels=4, out_channels=8)
    z_base = torch.full((1, 1, 21, 21), -10.0)
    z_base[:, :, 10, 9] = 10.0
    z_base[:, :, 10, 11] = 10.0
    contributions = torch.zeros(1, 4, 21, 21)
    contributions[:, 0, 10, 9] = 1.0
    contributions[:, 0, 10, 11] = 1.0

    out = bridge(z_base, contributions)

    assert out["bridge_feature"].shape == (1, 8, 21, 21)
    assert out["topology_prior"].shape == z_base.shape
    assert float(out["topology_prior"][0, 0, 10, 10]) > 0.9


def test_topology_bridge_rejects_one_sided_boundary_and_attribution_mismatch() -> None:
    bridge = AttributionTopologyBridge(scale_channels=4, out_channels=8)
    z_base = torch.full((1, 1, 21, 21), -10.0)
    z_base[:, :, 10, 9] = 10.0
    contributions = torch.zeros(1, 4, 21, 21)
    contributions[:, 0, 10, 9] = 1.0

    one_sided = bridge(z_base, contributions)
    assert float(one_sided["topology_prior"][0, 0, 10, 10]) < 1e-4

    z_base[:, :, 10, 11] = 10.0
    contributions[:, 1, 10, 11] = 1.0
    mismatched = bridge(z_base, contributions)
    assert float(mismatched["topology_prior"][0, 0, 10, 10]) < 1e-4


def test_topology_bridge_requires_target_ownership_at_both_endpoints() -> None:
    bridge = AttributionTopologyBridge(scale_channels=4, out_channels=8)
    z_base = torch.full((1, 1, 21, 21), -10.0)
    z_base[:, :, 10, 9] = 10.0
    z_base[:, :, 10, 11] = 10.0
    contributions = torch.zeros(1, 4, 21, 21)
    contributions[:, 0, 10, 9] = 1.0
    contributions[:, 0, 10, 11] = 1.0
    ownership = torch.zeros_like(z_base)
    ownership[:, :, 10, 9] = 1.0

    rejected = bridge.endpoint_owned_prior(z_base, contributions, ownership)
    assert float(rejected[0, 0, 10, 10]) < 1e-4

    ownership[:, :, 10, 11] = 1.0
    accepted = bridge.endpoint_owned_prior(z_base, contributions, ownership)
    assert float(accepted[0, 0, 10, 10]) > 0.9


def test_relation_selector_identity_init_and_oriented_operations() -> None:
    selector = AttributionGuidedRelationSelector(
        scale_channels=4,
        hidden_channels=8,
        max_offset=2,
    )
    z_base = torch.full((1, 1, 21, 21), -10.0)
    z_base[:, :, 10, 9] = 10.0
    z_base[:, :, 10, 11] = 10.0
    contributions = torch.zeros(1, 4, 21, 21)
    contributions[:, 0, 10, 9] = 1.0
    contributions[:, 0, 10, 11] = 1.0
    decisions = torch.zeros(1, 3, 21, 21)
    decisions[:, 2] = 1.0

    identity = selector(z_base, contributions, decisions)
    assert float(identity["relation_reconnect_map"].max()) == 0.0
    assert float(identity["relation_suppress_map"].max()) == 0.0

    with torch.no_grad():
        selector.relation_head[-1].bias.copy_(
            torch.tensor([4.0, -4.0, -4.0, -4.0])
        )
    decisions[:, :, 10, 9] = 0.0
    decisions[:, :, 10, 11] = 0.0
    decisions[:, 0, 10, 9] = 1.0
    decisions[:, 0, 10, 11] = 1.0
    reconnect = selector(z_base, contributions, decisions)
    assert float(reconnect["relation_reconnect_map"][0, 0, 10, 10]) > 0.9
    assert float(
        reconnect["relation_hard_reconnect_map"][0, 0, 10, 10]
    ) == 1.0

    with torch.no_grad():
        selector.relation_head[-1].bias.copy_(
            torch.tensor([-4.0, 4.0, -4.0, -4.0])
        )
    decisions[:, :, 10, 11] = 0.0
    decisions[:, 1, 10, 11] = 1.0
    suppress_positive = selector(z_base, contributions, decisions)
    assert float(
        suppress_positive["relation_suppress_positive_seed"][0, 0, 10, 11]
    ) > 0.9
    assert float(
        suppress_positive["relation_suppress_negative_seed"][0, 0, 10, 9]
    ) == 0.0
    assert float(
        suppress_positive[
            "relation_hard_suppress_positive_seed"
        ][0, 0, 10, 11]
    ) == 1.0


def test_relation_endpoint_targets_follow_selected_orientation() -> None:
    value = torch.zeros(1, 1, 11, 11)
    value[:, :, 5, 6] = 1.0
    choice = torch.full((1, 1, 11, 11), -1, dtype=torch.long)
    # First candidate is horizontal with positive and negative distances 1.
    choice[:, :, 5, 5] = 0

    positive, negative = gather_chosen_relation_endpoints(
        value,
        choice,
        max_offset=2,
    )

    assert float(positive[0, 0, 5, 5]) == 1.0
    assert float(negative[0, 0, 5, 5]) == 0.0


def test_full_dea_v4_relation_loss_and_baseline_init() -> None:
    head = FullDEAHeadV4(hidden_channels=16)
    out = head(*make_head_inputs(size=32))
    target = (torch.rand(2, 1, 32, 32) > 0.97).float()

    assert torch.mean(torch.abs(out["z_final"] - out["z_base"])) < 1e-3
    assert out["relation_logits"].shape == (2, 4, 32, 32)
    assert out["relation_pair_choice"].shape == (2, 1, 32, 32)
    assert out["relation_reconnect_map"].shape == target.shape
    assert out["relation_suppress_map"].shape == target.shape

    loss, logs = full_dea_aux_loss_v4(
        out,
        target,
        epoch=1,
        warm_epoch=0,
        tau_base=0.0,
        tau_target=0.0,
        tau_scale=0.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "full_dea_loss_relation" in logs
    relation_grad = head.relation_selector.relation_head[-1].weight.grad
    assert relation_grad is not None
    assert torch.isfinite(relation_grad).all()
    assert relation_grad.abs().sum() > 0


def test_full_dea_v5_hard_transport_preserves_baseline_at_identity_init() -> None:
    head = FullDEAHeadV5(hidden_channels=16)
    out = head(*make_head_inputs(size=32))

    assert torch.mean(torch.abs(out["z_final"] - out["z_base"])) < 1e-3
    assert out["relation_hard_reconnect_map"].shape == out["z_base"].shape
    assert out["relation_hard_suppress_map"].shape == out["z_base"].shape
    assert torch.equal(
        out["relation_reconnect_map"],
        out["relation_hard_reconnect_map"],
    )
    assert torch.equal(
        out["relation_suppress_map"],
        out["relation_hard_suppress_map"],
    )
    assert float(out["relation_reconnect_map"].sum()) == 0.0
    assert float(out["relation_suppress_map"].sum()) == 0.0


def test_full_dea_v3_loss_finite_and_component_hard_bg_bounded() -> None:
    head = FullDEAHeadV3(hidden_channels=16)
    out = head(*make_head_inputs(size=32))
    target = (torch.rand(2, 1, 32, 32) > 0.97).float()

    hard_bg, regions = build_component_hard_clutter_label(
        out,
        target,
        tau_base=0.0,
        tau_target=0.0,
        tau_scale=0.0,
        max_hard_bg_ratio=0.005,
    )
    assert hard_bg.shape == target.shape
    assert regions["safe_bg"].shape == target.shape
    assert float(hard_bg.mean()) <= 0.006
    assert float((hard_bg * regions["target_protect"]).sum()) == 0.0

    loss, logs = full_dea_aux_loss_v3(
        out,
        target,
        epoch=1,
        warm_epoch=0,
        tau_base=0.0,
        tau_target=0.0,
        tau_scale=0.0,
        max_hard_bg_ratio=0.005,
    )
    assert torch.isfinite(loss)
    assert logs["hard_clutter_ratio"].ndim == 0


def test_component_budget_never_fragments_selected_regions() -> None:
    mask = torch.zeros(1, 1, 20, 20)
    score = torch.zeros_like(mask)
    mask[:, :, 2:4, 2:4] = 1.0
    score[:, :, 2:4, 2:4] = 0.9
    mask[:, :, 10:13, 10:13] = 1.0
    score[:, :, 10:13, 10:13] = 0.8

    limited = _limit_component_mask_by_score(mask, score, max_ratio=0.02)

    assert int(limited.sum()) == 4
    assert torch.equal(limited[:, :, 2:4, 2:4], torch.ones(1, 1, 2, 2))
    assert int(limited[:, :, 10:13, 10:13].sum()) == 0


def test_counterfactual_scale_fragility_can_mine_clutter() -> None:
    target = torch.zeros(1, 1, 20, 20)
    z_base = torch.full_like(target, -10.0)
    scale_logits = torch.full((1, 4, 20, 20), -10.0)
    z_without_scale = torch.full((1, 4, 20, 20), -10.0)
    z_without_scale[:, 2, 8:10, 8:10] = 10.0
    out = {
        "z_base": z_base,
        "scale_logits_full": scale_logits,
        "z_without_scale": z_without_scale,
    }

    hard_clutter, _ = build_component_hard_clutter_label(
        out,
        target,
        tau_base=0.45,
        tau_target=0.45,
        tau_scale=0.45,
        max_hard_bg_ratio=0.02,
    )

    assert int(hard_clutter.sum()) == 4
    assert torch.equal(
        hard_clutter[:, :, 8:10, 8:10],
        torch.ones(1, 1, 2, 2),
    )


def test_full_dea_mshnet_wrapper_contract() -> None:
    torch.manual_seed(11)
    model = FullDEAMSHNet(input_channels=3)
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x, warm_flag=True, return_dict=True)

    masks = out["masks"]
    assert [tuple(m.shape[-2:]) for m in masks] == [
        (64, 64),
        (32, 32),
        (16, 16),
        (8, 8),
    ]
    assert out["pred"].shape == (2, 1, 64, 64)
    assert out["z_base"].shape == (2, 1, 64, 64)
    assert out["scale_logits_full"].shape == (2, 4, 64, 64)
    assert out["full_dea"]["z_final"].shape == (2, 1, 64, 64)


if __name__ == "__main__":
    test_full_dea_v3_head_shapes_and_baseline_init()
    test_full_dea_v3_loss_finite_and_component_hard_bg_bounded()
    test_topology_bridge_proposes_gap_between_fragments()
    test_topology_bridge_rejects_one_sided_boundary_and_attribution_mismatch()
    test_topology_bridge_requires_target_ownership_at_both_endpoints()
    test_component_budget_never_fragments_selected_regions()
    test_counterfactual_scale_fragility_can_mine_clutter()
    test_full_dea_mshnet_wrapper_contract()
