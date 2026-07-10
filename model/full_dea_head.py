from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dea_evidence import (
    AttributionGuidedComponentRelationGraph,
    AttributionGuidedRelationSelector,
    AttributionTopologyBridge,
    DepthwiseSeparableGNAct,
    ExactScaleContributionDecomposer,
    MultiRadiusContrastEncoder,
    gather_chosen_relation_endpoints,
)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2
        group_count = min(groups, out_channels)
        while out_channels % group_count != 0:
            group_count -= 1

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(group_count, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FullDEAHeadV2(nn.Module):
    """Baseline-preserving Full DEA v2 head.

    The head operates at the MSHNet multi-scale fusion point. It starts close to
    the original fused logit and learns target residuals plus subtractive clutter
    suppression.
    """

    def __init__(self, hidden_channels: int = 32, scale_channels: int = 4):
        super().__init__()
        h = hidden_channels

        self.proj0 = ConvBNAct(16, h, kernel_size=1)
        self.proj1 = ConvBNAct(32, h, kernel_size=1)
        self.proj2 = ConvBNAct(64, h, kernel_size=1)
        self.proj3 = ConvBNAct(128, h, kernel_size=1)

        self.feature_fuse = nn.Sequential(
            ConvBNAct(h * 4, h * 2, kernel_size=3),
            ConvBNAct(h * 2, h, kernel_size=3),
        )

        scale_stat_channels = scale_channels + 4
        self.scale_fuse = nn.Sequential(
            ConvBNAct(scale_stat_channels, h // 2, kernel_size=3),
            ConvBNAct(h // 2, h // 2, kernel_size=3),
        )

        evidence_in_channels = h + h // 2
        self.evidence_head = nn.Sequential(
            ConvBNAct(evidence_in_channels, h, kernel_size=3),
            nn.Conv2d(h, 2, kernel_size=1),
        )

        target_in_channels = h + scale_channels + 3
        self.target_delta_head = nn.Sequential(
            ConvBNAct(target_in_channels, h, kernel_size=3),
            ConvBNAct(h, h, kernel_size=3),
            nn.Conv2d(h, 1, kernel_size=1),
        )

        clutter_in_channels = h + scale_channels + 3
        self.clutter_head = nn.Sequential(
            ConvBNAct(clutter_in_channels, h, kernel_size=3),
            ConvBNAct(h, h, kernel_size=3),
            nn.Conv2d(h, 1, kernel_size=1),
        )

        gate_in_channels = h + 10
        self.suppression_head = nn.Sequential(
            ConvBNAct(gate_in_channels, h, kernel_size=3),
            ConvBNAct(h, h, kernel_size=3),
            nn.Conv2d(h, 1, kernel_size=1),
        )

        self.log_alpha = nn.Parameter(torch.tensor(-1.0))
        self._init_close_to_baseline()

    def _init_close_to_baseline(self) -> None:
        nn.init.zeros_(self.target_delta_head[-1].weight)
        nn.init.zeros_(self.target_delta_head[-1].bias)

        nn.init.zeros_(self.clutter_head[-1].weight)
        nn.init.constant_(self.clutter_head[-1].bias, -4.0)

        nn.init.zeros_(self.suppression_head[-1].weight)
        nn.init.constant_(self.suppression_head[-1].bias, -4.0)

    @staticmethod
    def _scale_stats(scale_logits_full: torch.Tensor) -> torch.Tensor:
        scale_mean = scale_logits_full.mean(dim=1, keepdim=True)
        scale_max = scale_logits_full.max(dim=1, keepdim=True)[0]
        scale_min = scale_logits_full.min(dim=1, keepdim=True)[0]
        scale_var = scale_logits_full.var(dim=1, keepdim=True, unbiased=False)
        return torch.cat(
            [scale_logits_full, scale_mean, scale_max, scale_min, scale_var],
            dim=1,
        )

    def forward(
        self,
        x_d0: torch.Tensor,
        x_d1: torch.Tensor,
        x_d2: torch.Tensor,
        x_d3: torch.Tensor,
        scale_logits_full: torch.Tensor,
        z_base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        size = x_d0.shape[-2:]

        f0 = self.proj0(x_d0)
        f1 = F.interpolate(
            self.proj1(x_d1),
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        f2 = F.interpolate(
            self.proj2(x_d2),
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        f3 = F.interpolate(
            self.proj3(x_d3),
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        fused_feature = self.feature_fuse(torch.cat([f0, f1, f2, f3], dim=1))

        scale_stats = self._scale_stats(scale_logits_full)
        scale_feature = self.scale_fuse(scale_stats)

        evidence_logits = self.evidence_head(
            torch.cat([fused_feature, scale_feature], dim=1)
        )
        target_evidence_logit, clutter_evidence_logit = torch.chunk(
            evidence_logits,
            chunks=2,
            dim=1,
        )
        target_evidence = torch.sigmoid(target_evidence_logit)
        clutter_evidence = torch.sigmoid(clutter_evidence_logit)

        target_gate = torch.sigmoid(target_evidence_logit - clutter_evidence_logit)
        clutter_gate = torch.sigmoid(clutter_evidence_logit - target_evidence_logit)

        target_input = torch.cat(
            [
                fused_feature * (1.0 + target_gate),
                scale_logits_full,
                z_base,
                target_evidence_logit,
                clutter_evidence_logit,
            ],
            dim=1,
        )
        target_delta = self.target_delta_head(target_input)
        z_target = z_base + target_delta

        clutter_input = torch.cat(
            [
                fused_feature * (1.0 + clutter_gate),
                scale_logits_full,
                z_base,
                target_evidence_logit,
                clutter_evidence_logit,
            ],
            dim=1,
        )
        z_clutter = self.clutter_head(clutter_input)

        scale_aux_stats = scale_stats[:, 4:, :, :]
        gate_input = torch.cat(
            [
                fused_feature,
                target_evidence_logit,
                clutter_evidence_logit,
                z_target,
                z_clutter,
                target_gate,
                clutter_gate,
                scale_aux_stats,
            ],
            dim=1,
        )
        suppression_logit = self.suppression_head(gate_input)
        suppression_gate = torch.sigmoid(suppression_logit)

        alpha = F.softplus(self.log_alpha) + 1e-6
        z_final = z_target - alpha * suppression_gate * F.softplus(z_clutter)

        return {
            "z_base": z_base,
            "scale_logits_full": scale_logits_full,
            "fused_feature": fused_feature,
            "target_evidence_logit": target_evidence_logit,
            "clutter_evidence_logit": clutter_evidence_logit,
            "target_evidence": target_evidence,
            "clutter_evidence": clutter_evidence,
            "target_gate": target_gate,
            "clutter_gate": clutter_gate,
            "target_delta": target_delta,
            "z_target": z_target,
            "z_clutter": z_clutter,
            "suppression_logit": suppression_logit,
            "suppression_gate": suppression_gate,
            "alpha": alpha.detach(),
            "y_real": z_target,
            "y_cf": z_clutter,
            "y_final": z_final,
            "z_final": z_final,
        }


class FullDEAHeadV3(nn.Module):
    """Decidable Evidence Aggregation over exact MSHNet interventions.

    The head does not relearn four free target/clutter/protection branches.
    Instead it decomposes MSHNet's final fusion exactly, combines those
    counterfactual scale effects with independent center-surround evidence,
    predicts target/clutter/uncertain states, and calibrates the baseline only
    when the evidence is decidable.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        scale_channels: int = 4,
        contrast_channels: int = 16,
    ):
        super().__init__()
        h = hidden_channels
        self.scale_channels = int(scale_channels)
        self.decomposer = ExactScaleContributionDecomposer(scale_channels)
        self.contrast_encoder = MultiRadiusContrastEncoder(
            in_channels=16,
            projected_channels=8,
            out_channels=contrast_channels,
        )

        # scale logits + exact contributions + leave-one-out outputs + z_base
        # + six contribution statistics.
        scale_evidence_channels = scale_channels * 3 + 1 + 6
        bridge_channels = 16
        self.topology_bridge = AttributionTopologyBridge(
            scale_channels=scale_channels,
            out_channels=bridge_channels,
        )
        decision_in_channels = (
            scale_evidence_channels + contrast_channels + bridge_channels
        )
        self.evidence_fuse = nn.Sequential(
            DepthwiseSeparableGNAct(decision_in_channels, h),
            DepthwiseSeparableGNAct(h, h),
        )
        self.decision_head = nn.Conv2d(h, 3, kernel_size=1)
        self.magnitude_head = nn.Conv2d(h, 2, kernel_size=1)
        self.bridge_head = nn.Conv2d(h, 2, kernel_size=1)

        self.log_alpha = nn.Parameter(torch.tensor(-1.0))
        self.log_beta = nn.Parameter(torch.tensor(-1.0))
        self.log_gamma = nn.Parameter(torch.tensor(-1.0))
        self._init_close_to_baseline()

    def _init_close_to_baseline(self) -> None:
        nn.init.zeros_(self.decision_head.weight)
        with torch.no_grad():
            self.decision_head.bias.copy_(torch.tensor([-2.0, -2.0, 2.0]))

        nn.init.zeros_(self.magnitude_head.weight)
        # Symmetric target/clutter magnitudes cancel exactly at initialization,
        # so z_final still equals z_base. A -2 bias keeps both branches in a
        # learnable softplus regime; the former -8 bias made the component
        # suppression gradient effectively vanish.
        nn.init.constant_(self.magnitude_head.bias, -2.0)

        nn.init.zeros_(self.bridge_head.weight)
        nn.init.constant_(self.bridge_head.bias, -2.0)

    def forward(
        self,
        x_d0: torch.Tensor,
        x_d1: torch.Tensor,
        x_d2: torch.Tensor,
        x_d3: torch.Tensor,
        scale_logits_full: torch.Tensor,
        z_base: torch.Tensor,
        fusion_weight: torch.Tensor | None = None,
        fusion_bias: torch.Tensor | None = None,
        fusion_stride=1,
        fusion_padding=1,
        fusion_dilation=1,
    ) -> dict[str, torch.Tensor]:
        del x_d1, x_d2, x_d3
        if fusion_weight is None:
            raise ValueError("FullDEAHeadV3 requires MSHNet final-fusion weights.")

        scale_evidence = self.decomposer(
            scale_logits=scale_logits_full,
            z_base=z_base,
            fusion_weight=fusion_weight,
            fusion_bias=fusion_bias,
            stride=fusion_stride,
            padding=fusion_padding,
            dilation=fusion_dilation,
        )
        contrast_feature = self.contrast_encoder(x_d0)
        topology_out = self.topology_bridge(
            z_base,
            scale_evidence["scale_contributions"],
        )
        fused_feature = self.evidence_fuse(
            torch.cat(
                [
                    scale_evidence["evidence_tensor"],
                    contrast_feature,
                    topology_out["bridge_feature"],
                ],
                dim=1,
            )
        )

        decision_logits = self.decision_head(fused_feature)
        decision_probs = torch.softmax(decision_logits, dim=1)
        target_prob, clutter_prob, uncertain_prob = torch.chunk(
            decision_probs,
            chunks=3,
            dim=1,
        )

        magnitude_logits = self.magnitude_head(fused_feature)
        target_amount_logit, clutter_amount_logit = torch.chunk(
            magnitude_logits,
            chunks=2,
            dim=1,
        )
        target_amount = F.softplus(target_amount_logit)
        clutter_amount = F.softplus(clutter_amount_logit)
        bridge_logits = self.bridge_head(fused_feature)
        bridge_evidence_logit, bridge_amount_logit = torch.chunk(
            bridge_logits,
            chunks=2,
            dim=1,
        )
        bridge_amount = F.softplus(bridge_amount_logit)

        beta = F.softplus(self.log_beta) + 1e-6
        alpha = F.softplus(self.log_alpha) + 1e-6
        gamma = F.softplus(self.log_gamma) + 1e-6

        # Calibrate only when target or clutter evidence strictly wins the
        # three-state decision. If uncertainty wins, both residuals are zero
        # and the frozen MSHNet logit is preserved exactly. The two decisive
        # gates are mutually exclusive, which structurally prevents residual
        # clutter probability from suppressing a target-dominant location.
        target_competitor = torch.maximum(clutter_prob, uncertain_prob)
        clutter_competitor = torch.maximum(target_prob, uncertain_prob)
        target_gate = F.relu(target_prob - target_competitor).detach()
        clutter_gate = F.relu(clutter_prob - clutter_competitor).detach()
        target_boost = target_gate * target_amount
        protected_suppression = clutter_gate * clutter_amount
        target_delta = beta * target_boost
        clutter_delta = alpha * protected_suppression
        endpoint_target_prior = self.topology_bridge.endpoint_owned_prior(
            z_base,
            scale_evidence["scale_contributions"],
            target_gate,
        )
        bridge_gate = (
            target_gate
            * torch.sigmoid(bridge_evidence_logit).detach()
            * endpoint_target_prior
        )
        bridge_delta = gamma * bridge_gate * bridge_amount
        z_target = z_base + target_delta + bridge_delta
        z_final = z_target - clutter_delta

        target_evidence_logit = decision_logits[:, 0:1]
        clutter_evidence_logit = decision_logits[:, 1:2]
        uncertain_evidence_logit = decision_logits[:, 2:3]

        return {
            "z_base": z_base,
            "scale_logits_full": scale_logits_full,
            **scale_evidence,
            **topology_out,
            "contrast_feature": contrast_feature,
            "fused_feature": fused_feature,
            "decision_logits": decision_logits,
            "decision_probs": decision_probs,
            "target_evidence_logit": target_evidence_logit,
            "clutter_evidence_logit": clutter_evidence_logit,
            "uncertain_evidence_logit": uncertain_evidence_logit,
            "target_evidence": target_prob,
            "clutter_evidence": clutter_prob,
            "uncertain_evidence": uncertain_prob,
            "target_prob": target_prob,
            "clutter_prob": clutter_prob,
            "uncertain_prob": uncertain_prob,
            "target_gate": target_gate,
            "clutter_gate": clutter_gate,
            "protect_logit": target_evidence_logit,
            "protect_prob": target_prob,
            "target_boost_logit": target_amount_logit,
            "target_boost": target_boost,
            "target_amount": target_amount,
            "target_delta": target_delta,
            "bridge_evidence_logit": bridge_evidence_logit,
            "bridge_amount_logit": bridge_amount_logit,
            "bridge_amount": bridge_amount,
            "endpoint_target_prior": endpoint_target_prior,
            "bridge_gate": bridge_gate,
            "bridge_delta": bridge_delta,
            "z_target": z_target,
            "z_clutter": clutter_amount_logit,
            "clutter_amount": clutter_amount,
            "clutter_delta": clutter_delta,
            "suppression_logit": clutter_evidence_logit,
            "raw_suppression_gate": clutter_prob,
            "evidence_suppression_gate": clutter_prob,
            "protected_suppression_gate": clutter_gate,
            "suppression_gate": clutter_gate,
            "alpha": alpha.detach(),
            "beta": beta.detach(),
            "gamma": gamma.detach(),
            "y_real": z_target,
            "y_cf": z_base - clutter_delta,
            "y_final": z_final,
            "z_final": z_final,
        }

class FullDEAHeadV4(FullDEAHeadV3):
    """DEA with attribution-guided soft-component relation operations.

    V3 makes a target-owned pixel bridge decision. V4 keeps its exact scale
    decomposition and tri-state pixel calibration, but replaces executable
    topology editing with an explicit pair-level operation selector. The
    selector can reconnect same-target fragments, suppress either endpoint as
    satellite clutter, or preserve the frozen MSHNet output through identity.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        scale_channels: int = 4,
        contrast_channels: int = 16,
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            scale_channels=scale_channels,
            contrast_channels=contrast_channels,
        )
        self.relation_selector = AttributionGuidedRelationSelector(
            scale_channels=scale_channels,
            hidden_channels=max(8, hidden_channels // 2),
            max_offset=4,
            component_kernel=3,
        )
        self.log_relation_suppress = nn.Parameter(torch.tensor(-1.0))

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        out = super().forward(*args, **kwargs)
        relation_out = self.relation_selector(
            z_base=out["z_base"],
            contributions=out["scale_contributions"],
            decision_probabilities=out["decision_probs"],
        )

        gamma = F.softplus(self.log_gamma) + 1e-6
        relation_suppress_scale = F.softplus(self.log_relation_suppress) + 1e-6
        relation_bridge_delta = (
            gamma
            * relation_out["relation_reconnect_map"]
            * out["bridge_amount"]
        )
        relation_suppress_delta = (
            relation_suppress_scale
            * relation_out["relation_suppress_map"]
            * out["clutter_amount"]
        )

        # The old endpoint bridge is retained only as a descriptor inside the
        # evidence encoder. All executable topology changes now come from the
        # relation selector, so the operation semantics can be supervised and
        # audited independently.
        z_target = out["z_base"] + out["target_delta"] + relation_bridge_delta
        pixel_clutter_delta = out["clutter_delta"]
        total_clutter_delta = pixel_clutter_delta + relation_suppress_delta
        z_final = z_target - total_clutter_delta

        out.update(relation_out)
        out.update(
            {
                "topology_prior": relation_out["relation_pair_prior"],
                "fragmentation_prior": relation_out["relation_pair_prior"],
                "endpoint_target_prior": relation_out["relation_pair_prior"],
                "bridge_evidence_logit": relation_out["relation_logits"][:, 0:1],
                "bridge_gate": relation_out["relation_reconnect_map"],
                "bridge_delta": relation_bridge_delta,
                "relation_bridge_delta": relation_bridge_delta,
                "relation_suppress_delta": relation_suppress_delta,
                "pixel_clutter_delta": pixel_clutter_delta,
                "clutter_delta": total_clutter_delta,
                "z_target": z_target,
                "z_final": z_final,
                "y_real": z_target,
                "y_cf": out["z_base"] - total_clutter_delta,
                "y_final": z_final,
                "relation_suppress_scale": relation_suppress_scale.detach(),
            }
        )
        return out


class FullDEAHeadV5(FullDEAHeadV4):
    """Component-relation routing with parameter-free evidence transport.

    V4 verifies that identity routing prevents cross-dataset degradation, but
    its product of soft relation margin, pair prior, and learned magnitude can
    collapse to an imperceptible residual. V5 separates *which operation* from
    *how the evidence is transported*: the learned relation head makes a hard
    operation choice, while exact endpoint logits determine the reconnect
    amount and the baseline decision boundary determines satellite removal.
    """

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        out = super().forward(*args, **kwargs)
        positive_logit, negative_logit = gather_chosen_relation_endpoints(
            out["z_base"].detach(),
            out["relation_pair_choice"],
            max_offset=self.relation_selector.max_offset,
        )
        weaker_endpoint_logit = torch.minimum(positive_logit, negative_logit)
        reconnect_transport = F.relu(
            weaker_endpoint_logit - out["z_base"].detach()
        )
        relation_bridge_delta = (
            out["relation_hard_reconnect_map"] * reconnect_transport
        )

        # softplus(z) is always larger than z. Subtracting it at a selected
        # satellite therefore moves a positive baseline logit strictly below
        # the 0.5 probability boundary without a free suppression magnitude.
        relation_suppress_transport = F.softplus(out["z_base"].detach())
        relation_suppress_delta = (
            out["relation_hard_suppress_map"]
            * relation_suppress_transport
        )
        z_target = out["z_base"] + out["target_delta"] + relation_bridge_delta
        pixel_clutter_delta = out["pixel_clutter_delta"]
        total_clutter_delta = pixel_clutter_delta + relation_suppress_delta
        z_final = z_target - total_clutter_delta

        out.update(
            {
                "relation_reconnect_map": out["relation_hard_reconnect_map"],
                "relation_suppress_map": out["relation_hard_suppress_map"],
                "bridge_gate": out["relation_hard_reconnect_map"],
                "bridge_delta": relation_bridge_delta,
                "relation_bridge_delta": relation_bridge_delta,
                "relation_positive_endpoint_logit": positive_logit,
                "relation_negative_endpoint_logit": negative_logit,
                "relation_weaker_endpoint_logit": weaker_endpoint_logit,
                "relation_reconnect_transport": reconnect_transport,
                "relation_suppress_transport": relation_suppress_transport,
                "relation_suppress_delta": relation_suppress_delta,
                "clutter_delta": total_clutter_delta,
                "z_target": z_target,
                "z_final": z_final,
                "y_real": z_target,
                "y_cf": out["z_base"] - total_clutter_delta,
                "y_final": z_final,
                "relation_suppress_scale": out["z_base"].new_tensor(1.0),
            }
        )
        return out


class FullDEAHeadV6(FullDEAHeadV3):
    """Exact connected-component relation reasoning for DEA.

    The graph treats thresholded MSHNet responses as proposal nodes. It edits
    only the shortest background corridor between two target-owned components
    or the complete non-target component paired with a target-owned component.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        scale_channels: int = 4,
        contrast_channels: int = 16,
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            scale_channels=scale_channels,
            contrast_channels=contrast_channels,
        )
        self.component_relation_graph = AttributionGuidedComponentRelationGraph(
            scale_channels=scale_channels,
            hidden_channels=hidden_channels,
            max_pair_distance=8.0,
        )

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        out = super().forward(*args, **kwargs)
        relation_out = self.component_relation_graph(
            z_base=out["z_base"],
            contributions=out["scale_contributions"],
            decision_probabilities=out["decision_probs"],
        )
        epsilon = torch.finfo(out["z_base"].dtype).eps
        relation_bridge_delta = (
            relation_out["component_relation_bridge_mask"]
            * F.relu(epsilon - out["z_base"].detach())
        )
        relation_suppress_delta = (
            relation_out["component_relation_suppress_mask"]
            * (F.softplus(out["z_base"].detach()) + epsilon)
        )
        z_target = out["z_base"] + out["target_delta"] + relation_bridge_delta
        pixel_clutter_delta = out["clutter_delta"]
        total_clutter_delta = pixel_clutter_delta + relation_suppress_delta
        z_final = z_target - total_clutter_delta

        out.update(relation_out)
        out.update(
            {
                "relation_logits": relation_out["component_relation_logits"],
                "relation_probabilities": relation_out[
                    "component_relation_probabilities"
                ],
                "relation_reconnect_map": relation_out[
                    "component_relation_bridge_mask"
                ],
                "relation_suppress_map": relation_out[
                    "component_relation_suppress_mask"
                ],
                "topology_prior": relation_out[
                    "component_relation_candidate_bridge_mask"
                ],
                "fragmentation_prior": relation_out[
                    "component_relation_candidate_bridge_mask"
                ],
                "endpoint_target_prior": relation_out[
                    "component_relation_candidate_bridge_mask"
                ],
                "bridge_gate": relation_out["component_relation_bridge_mask"],
                "bridge_delta": relation_bridge_delta,
                "relation_bridge_delta": relation_bridge_delta,
                "relation_suppress_delta": relation_suppress_delta,
                "pixel_clutter_delta": pixel_clutter_delta,
                "clutter_delta": total_clutter_delta,
                "z_target": z_target,
                "z_final": z_final,
                "y_real": z_target,
                "y_cf": out["z_base"] - total_clutter_delta,
                "y_final": z_final,
                "relation_suppress_scale": out["z_base"].new_tensor(1.0),
            }
        )
        return out


FullDEAHead = FullDEAHeadV3
