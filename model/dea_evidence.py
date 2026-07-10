from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import draw, measure


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class DepthwiseSeparableGNAct(nn.Module):
    """Cheap full-resolution spatial mixing for tiny-target evidence maps."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ExactScaleContributionDecomposer(nn.Module):
    """Decompose MSHNet's linear four-scale fusion into exact contributions.

    For MSHNet's final convolution, z_base = bias + sum_s contribution_s.
    The decomposition therefore gives exact branch-only and leave-one-scale-out
    interventions without introducing a learned surrogate evidence branch.
    """

    def __init__(self, scale_channels: int = 4, support_temperature: float = 1.0):
        super().__init__()
        self.scale_channels = int(scale_channels)
        self.support_temperature = float(support_temperature)

    def forward(
        self,
        scale_logits: torch.Tensor,
        z_base: torch.Tensor,
        fusion_weight: torch.Tensor,
        fusion_bias: torch.Tensor | None,
        stride=1,
        padding=1,
        dilation=1,
    ) -> dict[str, torch.Tensor]:
        if scale_logits.shape[1] != self.scale_channels:
            raise ValueError(
                "Expected %d scale channels, got %d"
                % (self.scale_channels, scale_logits.shape[1])
            )
        if fusion_weight.shape[:2] != (1, self.scale_channels):
            raise ValueError(
                "Expected fusion weight [1, %d, k, k], got %s"
                % (self.scale_channels, tuple(fusion_weight.shape))
            )

        grouped_weight = fusion_weight[0].unsqueeze(1)
        contributions = F.conv2d(
            scale_logits,
            grouped_weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=self.scale_channels,
        )

        if fusion_bias is None:
            bias = torch.zeros_like(z_base)
        else:
            bias = fusion_bias.view(1, 1, 1, 1).expand_as(z_base)

        z_reconstructed = contributions.sum(dim=1, keepdim=True) + bias
        z_without_scale = z_base - contributions
        z_only_scale = contributions + bias

        contribution_mean = contributions.mean(dim=1, keepdim=True)
        contribution_max = contributions.max(dim=1, keepdim=True)[0]
        contribution_min = contributions.min(dim=1, keepdim=True)[0]
        contribution_var = contributions.var(dim=1, keepdim=True, unbiased=False)
        contribution_abs = contributions.abs()
        contribution_abs_sum = contribution_abs.sum(dim=1, keepdim=True)
        contribution_dominance = contribution_abs.max(dim=1, keepdim=True)[0] / (
            contribution_abs_sum + 1e-6
        )
        positive_support = torch.sigmoid(
            contributions / max(self.support_temperature, 1e-6)
        ).mean(dim=1, keepdim=True)

        statistics = torch.cat(
            [
                contribution_mean,
                contribution_max,
                contribution_min,
                contribution_var,
                contribution_dominance,
                positive_support,
            ],
            dim=1,
        )
        evidence_tensor = torch.cat(
            [
                scale_logits,
                contributions,
                z_without_scale,
                z_base,
                statistics,
            ],
            dim=1,
        )

        return {
            "scale_contributions": contributions,
            "z_reconstructed": z_reconstructed,
            "z_without_scale": z_without_scale,
            "z_only_scale": z_only_scale,
            "scale_statistics": statistics,
            "evidence_tensor": evidence_tensor,
        }


class MultiRadiusContrastEncoder(nn.Module):
    """Independent center-surround evidence from MSHNet's high-res decoder."""

    def __init__(
        self,
        in_channels: int = 16,
        projected_channels: int = 8,
        out_channels: int = 16,
        radii: tuple[int, ...] = (3, 7, 15),
    ):
        super().__init__()
        self.radii = tuple(int(radius) for radius in radii)
        if any(radius <= 0 or radius % 2 == 0 for radius in self.radii):
            raise ValueError("Contrast radii must be positive odd integers.")

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, projected_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(projected_channels), projected_channels),
            nn.ReLU(inplace=True),
        )
        contrast_channels = projected_channels * (len(self.radii) + 1)
        self.fuse = DepthwiseSeparableGNAct(contrast_channels, out_channels)

    def forward(self, x_d0: torch.Tensor) -> torch.Tensor:
        center = self.project(x_d0)
        evidence = [center]
        for radius in self.radii:
            context = F.avg_pool2d(
                center,
                kernel_size=radius,
                stride=1,
                padding=radius // 2,
            )
            evidence.append(center - context)
        return self.fuse(torch.cat(evidence, dim=1))


class AttributionTopologyBridge(nn.Module):
    """Propose target-fragment bridges with attribution-consistency cues.

    Most object-level false alarms in the paired MSHNet audit are disconnected
    pieces near a labelled target, not remote clutter. Morphological closing is
    useful as a learned descriptor, but using it directly as a bridge prior also
    fires on ordinary one-sided object boundaries and expands dense masks. The
    executable prior below is therefore stricter: a gap needs response support
    on two opposite sides and the two endpoints must have compatible exact
    scale-attribution signatures. A one-sided boundary cannot pass this test.
    """

    def __init__(
        self,
        scale_channels: int = 4,
        out_channels: int = 16,
        radii: tuple[int, ...] = (3, 5, 9),
    ):
        super().__init__()
        self.radii = tuple(int(radius) for radius in radii)
        if any(radius <= 0 or radius % 2 == 0 for radius in self.radii):
            raise ValueError("Bridge radii must be positive odd integers.")
        # closing residual, bidirectional fragmentation prior, support mass,
        # attribution variance, and local logit range for every radius.
        in_channels = len(self.radii) * 5
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableGNAct(out_channels, out_channels),
        )

    @staticmethod
    def _sample_padded(
        padded: torch.Tensor,
        height: int,
        width: int,
        dy: int,
        dx: int,
        padding: int,
    ) -> torch.Tensor:
        """Return value(y + dy, x + dx) from a pre-padded tensor."""

        y0 = padding + int(dy)
        x0 = padding + int(dx)
        return padded[..., y0 : y0 + height, x0 : x0 + width]

    def _bidirectional_prior(
        self,
        endpoint_support: torch.Tensor,
        attribution: torch.Tensor,
        max_offset: int,
        center_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Find gaps bracketed by attribution-consistent response endpoints.

        Opposite endpoints may be at different distances from the candidate
        pixel. This matters for even-width gaps, whose pixels do not have a
        perfectly symmetric endpoint pair.
        """

        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        best = torch.zeros_like(endpoint_support)
        height, width = endpoint_support.shape[-2:]
        padded_support = F.pad(
            endpoint_support,
            (max_offset, max_offset, max_offset, max_offset),
        )
        padded_attribution = F.pad(
            attribution,
            (max_offset, max_offset, max_offset, max_offset),
        )
        center_gap = (1.0 - center_probability).unsqueeze(2).unsqueeze(2)
        for dy, dx in directions:
            positive_support = torch.stack(
                [
                    self._sample_padded(
                        padded_support,
                        height,
                        width,
                        dy * step,
                        dx * step,
                        max_offset,
                    )
                    for step in range(1, max_offset + 1)
                ],
                dim=2,
            )
            negative_support = torch.stack(
                [
                    self._sample_padded(
                        padded_support,
                        height,
                        width,
                        -dy * step,
                        -dx * step,
                        max_offset,
                    )
                    for step in range(1, max_offset + 1)
                ],
                dim=2,
            )
            positive_attribution = torch.stack(
                [
                    self._sample_padded(
                        padded_attribution,
                        height,
                        width,
                        dy * step,
                        dx * step,
                        max_offset,
                    )
                    for step in range(1, max_offset + 1)
                ],
                dim=2,
            )
            negative_attribution = torch.stack(
                [
                    self._sample_padded(
                        padded_attribution,
                        height,
                        width,
                        -dy * step,
                        -dx * step,
                        max_offset,
                    )
                    for step in range(1, max_offset + 1)
                ],
                dim=2,
            )

            paired_support = torch.minimum(
                positive_support.unsqueeze(3),
                negative_support.unsqueeze(2),
            )
            attribution_distance = (
                positive_attribution.unsqueeze(3)
                - negative_attribution.unsqueeze(2)
            ).abs().sum(dim=1, keepdim=True)
            attribution_consistency = (
                1.0 - 0.5 * attribution_distance
            ).clamp(min=0.0, max=1.0)
            candidates = paired_support * center_gap * attribution_consistency
            direction_best = candidates.flatten(2, 3).max(dim=2)[0]
            best = torch.maximum(best, direction_best)
        return best

    def endpoint_owned_prior(
        self,
        z_base: torch.Tensor,
        contributions: torch.Tensor,
        endpoint_ownership: torch.Tensor,
    ) -> torch.Tensor:
        """Require both response endpoints to belong to the target operation.

        Attribution consistency alone cannot distinguish a missing target piece
        from a target-adjacent clutter satellite. Multiplying endpoint response
        support by the decisive target gate turns bridging into a pairwise
        operation: target--target gaps are eligible, while target--clutter and
        target--uncertain gaps are structurally blocked.
        """

        probability = torch.sigmoid(z_base.detach())
        attribution = (
            contributions
            / (contributions.abs().sum(dim=1, keepdim=True) + 1e-6)
        ).detach()
        owned_support = probability * endpoint_ownership.detach()
        priors = []
        for radius in self.radii:
            priors.append(
                self._bidirectional_prior(
                    endpoint_support=owned_support,
                    attribution=attribution,
                    max_offset=radius // 2,
                    center_probability=probability,
                )
            )
        return torch.stack(priors, dim=0).max(dim=0)[0]

    def forward(
        self,
        z_base: torch.Tensor,
        contributions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        probability = torch.sigmoid(z_base.detach())
        attribution = (
            contributions
            / (contributions.abs().sum(dim=1, keepdim=True) + 1e-6)
        ).detach()
        descriptors = []
        closing_residuals = []
        bidirectional_priors = []
        for radius in self.radii:
            padding = radius // 2
            dilated = F.max_pool2d(
                probability,
                kernel_size=radius,
                stride=1,
                padding=padding,
            )
            closed = -F.max_pool2d(
                -dilated,
                kernel_size=radius,
                stride=1,
                padding=padding,
            )
            closing_residual = F.relu(closed - probability)
            bidirectional_prior = self._bidirectional_prior(
                endpoint_support=probability,
                attribution=attribution,
                max_offset=padding,
                center_probability=probability,
            )
            support_mass = F.avg_pool2d(
                probability,
                kernel_size=radius,
                stride=1,
                padding=padding,
            )
            attribution_mean = F.avg_pool2d(
                attribution,
                kernel_size=radius,
                stride=1,
                padding=padding,
            )
            attribution_second = F.avg_pool2d(
                attribution.square(),
                kernel_size=radius,
                stride=1,
                padding=padding,
            )
            attribution_variance = (
                attribution_second - attribution_mean.square()
            ).clamp_min(0.0).mean(dim=1, keepdim=True)
            local_max = F.max_pool2d(
                z_base.detach(),
                kernel_size=radius,
                stride=1,
                padding=padding,
            )
            local_min = -F.max_pool2d(
                -z_base.detach(),
                kernel_size=radius,
                stride=1,
                padding=padding,
            )
            descriptors.extend(
                [
                    closing_residual,
                    bidirectional_prior,
                    support_mass,
                    attribution_variance,
                    local_max - local_min,
                ]
            )
            closing_residuals.append(closing_residual)
            bidirectional_priors.append(bidirectional_prior)

        closing_prior = torch.stack(closing_residuals, dim=0).max(dim=0)[0]
        topology_prior = torch.stack(bidirectional_priors, dim=0).max(dim=0)[0]
        return {
            "bridge_feature": self.fuse(torch.cat(descriptors, dim=1)),
            "topology_prior": topology_prior,
            "fragmentation_prior": topology_prior,
            "closing_prior": closing_prior,
        }


def attribution_relation_offsets(
    max_offset: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Enumerate oriented endpoint pairs around a candidate relation center."""

    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    offsets = []
    for dy, dx in directions:
        for positive_distance in range(1, int(max_offset) + 1):
            for negative_distance in range(1, int(max_offset) + 1):
                offsets.append((dy, dx, positive_distance, negative_distance))
    return tuple(offsets)


def _sample_pre_padded(
    padded: torch.Tensor,
    height: int,
    width: int,
    dy: int,
    dx: int,
    padding: int,
) -> torch.Tensor:
    y0 = padding + int(dy)
    x0 = padding + int(dx)
    return padded[..., y0 : y0 + height, x0 : x0 + width]


def gather_chosen_relation_endpoints(
    value: torch.Tensor,
    choice: torch.Tensor,
    max_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather endpoint values for the discrete pair selected at every pixel.

    Pair selection is intentionally detached from gradients. The gathered
    values are used to construct relation supervision from the ground-truth
    mask without passing annotations into the model forward path.
    """

    if choice.shape[1] != 1:
        raise ValueError("Relation choice must have one channel.")
    height, width = value.shape[-2:]
    padded = F.pad(
        value,
        (max_offset, max_offset, max_offset, max_offset),
    )
    positive = torch.zeros_like(value)
    negative = torch.zeros_like(value)
    for index, (dy, dx, positive_distance, negative_distance) in enumerate(
        attribution_relation_offsets(max_offset)
    ):
        selected = choice == index
        positive_value = _sample_pre_padded(
            padded,
            height,
            width,
            dy * positive_distance,
            dx * positive_distance,
            max_offset,
        )
        negative_value = _sample_pre_padded(
            padded,
            height,
            width,
            -dy * negative_distance,
            -dx * negative_distance,
            max_offset,
        )
        positive = torch.where(selected, positive_value, positive)
        negative = torch.where(selected, negative_value, negative)
    return positive, negative


class AttributionGuidedRelationSelector(nn.Module):
    """Choose an operation for an attribution-consistent soft-component pair.

    The selector upgrades a pixel bridge into an explicit relation decision.
    Around every potential gap, it first chooses the strongest pair of response
    endpoints under exact scale-attribution consistency. A shared relation MLP
    then selects one of four executable operations:

    0. reconnect the two endpoints;
    1. suppress the positive-side endpoint as satellite clutter;
    2. suppress the negative-side endpoint as satellite clutter;
    3. preserve the baseline (identity).

    The two suppression orientations implement the single semantic category
    "target + satellite clutter" while retaining which component must be
    modified. Identity is initialized to win decisively, so adding the module
    cannot perturb the baseline before relation evidence is learned.
    """

    RECONNECT = 0
    SUPPRESS_POSITIVE = 1
    SUPPRESS_NEGATIVE = 2
    IDENTITY = 3

    def __init__(
        self,
        scale_channels: int = 4,
        hidden_channels: int = 16,
        max_offset: int = 4,
        component_kernel: int = 3,
    ):
        super().__init__()
        self.scale_channels = int(scale_channels)
        self.max_offset = int(max_offset)
        self.component_kernel = int(component_kernel)
        if self.max_offset < 1:
            raise ValueError("max_offset must be positive.")
        if self.component_kernel < 1 or self.component_kernel % 2 == 0:
            raise ValueError("component_kernel must be a positive odd integer.")

        # Six response values, one attribution-consistency value, three exact
        # attribution vectors, two contribution strengths, two three-state
        # endpoint ownership vectors, and four geometric values.
        pair_channels = 6 + 1 + scale_channels * 3 + 2 + 6 + 4
        self.relation_head = nn.Sequential(
            nn.Conv2d(pair_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=1),
        )
        nn.init.zeros_(self.relation_head[-1].weight)
        with torch.no_grad():
            self.relation_head[-1].bias.copy_(
                torch.tensor([-2.0, -2.0, -2.0, 2.0])
            )

    @staticmethod
    def _decisive_gate(
        probabilities: torch.Tensor,
        operation: int,
    ) -> torch.Tensor:
        alternatives = torch.cat(
            [
                probabilities[:, :operation],
                probabilities[:, operation + 1 :],
            ],
            dim=1,
        )
        competitor = alternatives.max(dim=1, keepdim=True)[0]
        return F.relu(probabilities[:, operation : operation + 1] - competitor)

    def _shift_center_scores_to_endpoint(
        self,
        center_scores: torch.Tensor,
        dy: int,
        dx: int,
    ) -> torch.Tensor:
        """Move a center-indexed relation score to its selected endpoint."""

        height, width = center_scores.shape[-2:]
        padded = F.pad(
            center_scores,
            (
                self.max_offset,
                self.max_offset,
                self.max_offset,
                self.max_offset,
            ),
        )
        return _sample_pre_padded(
            padded,
            height,
            width,
            -dy,
            -dx,
            self.max_offset,
        )

    def forward(
        self,
        z_base: torch.Tensor,
        contributions: torch.Tensor,
        decision_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if contributions.shape[1] != self.scale_channels:
            raise ValueError(
                "Expected %d scale contributions, got %d"
                % (self.scale_channels, contributions.shape[1])
            )
        if decision_probabilities.shape[1] != 3:
            raise ValueError("Relation selector requires three-state decisions.")

        probability = torch.sigmoid(z_base.detach())
        contribution_abs_sum = contributions.detach().abs().sum(
            dim=1,
            keepdim=True,
        )
        attribution = (
            contributions.detach() / (contribution_abs_sum + 1e-6)
        )
        ownership = decision_probabilities.detach()
        height, width = z_base.shape[-2:]
        padding = self.max_offset
        padded_probability = F.pad(
            probability,
            (padding, padding, padding, padding),
        )
        padded_attribution = F.pad(
            attribution,
            (padding, padding, padding, padding),
        )
        padded_strength = F.pad(
            torch.log1p(contribution_abs_sum),
            (padding, padding, padding, padding),
        )
        padded_ownership = F.pad(
            ownership,
            (padding, padding, padding, padding),
        )

        pair_channels = self.relation_head[0].in_channels
        best_prior = torch.zeros_like(probability)
        best_feature = probability.new_zeros(
            probability.shape[0],
            pair_channels,
            height,
            width,
        )
        best_choice = torch.full(
            probability.shape,
            fill_value=-1,
            device=probability.device,
            dtype=torch.long,
        )
        best_positive_ownership = probability.new_zeros(
            probability.shape[0],
            3,
            height,
            width,
        )
        best_negative_ownership = torch.zeros_like(best_positive_ownership)

        offsets = attribution_relation_offsets(self.max_offset)
        center_gap = 1.0 - probability
        for index, (dy, dx, positive_distance, negative_distance) in enumerate(
            offsets
        ):
            positive_dy = dy * positive_distance
            positive_dx = dx * positive_distance
            negative_dy = -dy * negative_distance
            negative_dx = -dx * negative_distance
            positive_probability = _sample_pre_padded(
                padded_probability,
                height,
                width,
                positive_dy,
                positive_dx,
                padding,
            )
            negative_probability = _sample_pre_padded(
                padded_probability,
                height,
                width,
                negative_dy,
                negative_dx,
                padding,
            )
            positive_attribution = _sample_pre_padded(
                padded_attribution,
                height,
                width,
                positive_dy,
                positive_dx,
                padding,
            )
            negative_attribution = _sample_pre_padded(
                padded_attribution,
                height,
                width,
                negative_dy,
                negative_dx,
                padding,
            )
            positive_strength = _sample_pre_padded(
                padded_strength,
                height,
                width,
                positive_dy,
                positive_dx,
                padding,
            )
            negative_strength = _sample_pre_padded(
                padded_strength,
                height,
                width,
                negative_dy,
                negative_dx,
                padding,
            )
            positive_ownership = _sample_pre_padded(
                padded_ownership,
                height,
                width,
                positive_dy,
                positive_dx,
                padding,
            )
            negative_ownership = _sample_pre_padded(
                padded_ownership,
                height,
                width,
                negative_dy,
                negative_dx,
                padding,
            )

            paired_support = torch.minimum(
                positive_probability,
                negative_probability,
            )
            attribution_difference = (
                positive_attribution - negative_attribution
            ).abs()
            attribution_consistency = (
                1.0 - 0.5 * attribution_difference.sum(dim=1, keepdim=True)
            ).clamp(min=0.0, max=1.0)
            pair_prior = paired_support * center_gap * attribution_consistency

            geometry = torch.cat(
                [
                    probability.new_full(
                        probability.shape,
                        float(dy),
                    ),
                    probability.new_full(
                        probability.shape,
                        float(dx),
                    ),
                    probability.new_full(
                        probability.shape,
                        float(positive_distance) / self.max_offset,
                    ),
                    probability.new_full(
                        probability.shape,
                        float(negative_distance) / self.max_offset,
                    ),
                ],
                dim=1,
            )
            pair_feature = torch.cat(
                [
                    positive_probability,
                    negative_probability,
                    paired_support,
                    (positive_probability - negative_probability).abs(),
                    probability,
                    center_gap,
                    attribution_consistency,
                    positive_attribution,
                    negative_attribution,
                    attribution_difference,
                    positive_strength,
                    negative_strength,
                    positive_ownership,
                    negative_ownership,
                    geometry,
                ],
                dim=1,
            )
            better = pair_prior > best_prior
            best_prior = torch.where(better, pair_prior, best_prior)
            best_feature = torch.where(better, pair_feature, best_feature)
            best_choice = torch.where(
                better,
                torch.full_like(best_choice, index),
                best_choice,
            )
            best_positive_ownership = torch.where(
                better,
                positive_ownership,
                best_positive_ownership,
            )
            best_negative_ownership = torch.where(
                better,
                negative_ownership,
                best_negative_ownership,
            )

        relation_logits = self.relation_head(best_feature)
        relation_probabilities = torch.softmax(relation_logits, dim=1)
        reconnect_gate = self._decisive_gate(
            relation_probabilities,
            self.RECONNECT,
        ).detach()
        suppress_positive_gate = self._decisive_gate(
            relation_probabilities,
            self.SUPPRESS_POSITIVE,
        ).detach()
        suppress_negative_gate = self._decisive_gate(
            relation_probabilities,
            self.SUPPRESS_NEGATIVE,
        ).detach()
        relation_winner = relation_probabilities.argmax(
            dim=1,
            keepdim=True,
        ).detach()
        # A pair is structurally decidable only if the joint endpoint support,
        # center gap, and attribution consistency exceed the same 0.5 decision
        # boundary used by the baseline segmentation. This is not a learned
        # amplitude: it separates an executable component relation from weak
        # evidence that must fall back to identity.
        hard_pair_decidable = (best_prior >= 0.5).to(probability.dtype)
        positive_state = best_positive_ownership.argmax(
            dim=1,
            keepdim=True,
        )
        negative_state = best_negative_ownership.argmax(
            dim=1,
            keepdim=True,
        )
        positive_is_target = (positive_state == 0).to(probability.dtype)
        negative_is_target = (negative_state == 0).to(probability.dtype)
        target_target_relation = positive_is_target * negative_is_target
        positive_satellite_relation = (
            (1.0 - positive_is_target) * negative_is_target
        )
        negative_satellite_relation = (
            positive_is_target * (1.0 - negative_is_target)
        )
        hard_reconnect_map = (
            (relation_winner == self.RECONNECT).to(probability.dtype)
            * hard_pair_decidable
            * target_target_relation
        )
        hard_suppress_positive_center = (
            (relation_winner == self.SUPPRESS_POSITIVE).to(probability.dtype)
            * hard_pair_decidable
            * positive_satellite_relation
        )
        hard_suppress_negative_center = (
            (relation_winner == self.SUPPRESS_NEGATIVE).to(probability.dtype)
            * hard_pair_decidable
            * negative_satellite_relation
        )

        reconnect_map = best_prior * reconnect_gate
        positive_center_score = best_prior * suppress_positive_gate
        negative_center_score = best_prior * suppress_negative_gate
        positive_seed = torch.zeros_like(probability)
        negative_seed = torch.zeros_like(probability)
        hard_positive_seed = torch.zeros_like(probability)
        hard_negative_seed = torch.zeros_like(probability)
        for index, (dy, dx, positive_distance, negative_distance) in enumerate(
            offsets
        ):
            selected = (best_choice == index).to(probability.dtype)
            positive_seed = torch.maximum(
                positive_seed,
                self._shift_center_scores_to_endpoint(
                    positive_center_score * selected,
                    dy * positive_distance,
                    dx * positive_distance,
                ),
            )
            negative_seed = torch.maximum(
                negative_seed,
                self._shift_center_scores_to_endpoint(
                    negative_center_score * selected,
                    -dy * negative_distance,
                    -dx * negative_distance,
                ),
            )
            hard_positive_seed = torch.maximum(
                hard_positive_seed,
                self._shift_center_scores_to_endpoint(
                    hard_suppress_positive_center * selected,
                    dy * positive_distance,
                    dx * positive_distance,
                ),
            )
            hard_negative_seed = torch.maximum(
                hard_negative_seed,
                self._shift_center_scores_to_endpoint(
                    hard_suppress_negative_center * selected,
                    -dy * negative_distance,
                    -dx * negative_distance,
                ),
            )
        suppress_seed = torch.maximum(positive_seed, negative_seed)
        component_support = (
            F.max_pool2d(
                suppress_seed,
                kernel_size=self.component_kernel,
                stride=1,
                padding=self.component_kernel // 2,
            )
            * probability
        )
        suppress_map = torch.maximum(suppress_seed, component_support)
        hard_suppress_seed = torch.maximum(
            hard_positive_seed,
            hard_negative_seed,
        )
        hard_component_support = (
            F.max_pool2d(
                hard_suppress_seed,
                kernel_size=self.component_kernel,
                stride=1,
                padding=self.component_kernel // 2,
            )
            * (probability >= 0.5).to(probability.dtype)
        )
        hard_suppress_map = torch.maximum(
            hard_suppress_seed,
            hard_component_support,
        )

        return {
            "relation_logits": relation_logits,
            "relation_probabilities": relation_probabilities,
            "relation_pair_prior": best_prior,
            "relation_pair_choice": best_choice,
            "relation_positive_endpoint_ownership": best_positive_ownership,
            "relation_negative_endpoint_ownership": best_negative_ownership,
            "relation_positive_endpoint_state": positive_state,
            "relation_negative_endpoint_state": negative_state,
            "relation_target_target_eligibility": target_target_relation,
            "relation_positive_satellite_eligibility": (
                positive_satellite_relation
            ),
            "relation_negative_satellite_eligibility": (
                negative_satellite_relation
            ),
            "relation_winner": relation_winner,
            "relation_hard_pair_decidable": hard_pair_decidable,
            "relation_reconnect_gate": reconnect_gate,
            "relation_suppress_positive_gate": suppress_positive_gate,
            "relation_suppress_negative_gate": suppress_negative_gate,
            "relation_identity_probability": relation_probabilities[
                :, self.IDENTITY : self.IDENTITY + 1
            ],
            "relation_reconnect_map": reconnect_map,
            "relation_suppress_positive_seed": positive_seed,
            "relation_suppress_negative_seed": negative_seed,
            "relation_suppress_seed": suppress_seed,
            "relation_component_support": component_support,
            "relation_suppress_map": suppress_map,
            "relation_hard_reconnect_map": hard_reconnect_map,
            "relation_hard_suppress_positive_seed": hard_positive_seed,
            "relation_hard_suppress_negative_seed": hard_negative_seed,
            "relation_hard_suppress_seed": hard_suppress_seed,
            "relation_hard_component_support": hard_component_support,
            "relation_hard_suppress_map": hard_suppress_map,
        }


class AttributionGuidedComponentRelationGraph(nn.Module):
    """Reason over actual MSHNet response components instead of pixel proxies.

    Connected components are discrete proposal nodes extracted from the frozen
    baseline decision. Their pooled exact attribution, confidence, ownership,
    shape, and pair geometry are classified into reconnect, suppress-first,
    suppress-second, or identity. The grouping itself is non-differentiable,
    while the relation classifier is trained from component-pair supervision.
    """

    RECONNECT = 0
    SUPPRESS_FIRST = 1
    SUPPRESS_SECOND = 2
    IDENTITY = 3

    def __init__(
        self,
        scale_channels: int = 4,
        hidden_channels: int = 32,
        max_pair_distance: float = 8.0,
    ):
        super().__init__()
        self.scale_channels = int(scale_channels)
        self.max_pair_distance = float(max_pair_distance)
        if self.max_pair_distance <= 0:
            raise ValueError("max_pair_distance must be positive.")

        # Two 14-D component descriptors plus attribution difference (4),
        # consistency, distance, direction (2), area ratio, three corridor
        # confidence statistics, and two corridor-to-endpoint attribution gaps.
        pair_channels = 28 + scale_channels + 1 + 1 + 2 + 1 + 3 + 2
        self.pair_channels = pair_channels
        self.relation_head = nn.Sequential(
            nn.Linear(pair_channels, hidden_channels, bias=False),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 4),
        )
        nn.init.zeros_(self.relation_head[-1].weight)
        with torch.no_grad():
            self.relation_head[-1].bias.copy_(
                torch.tensor([-2.0, -2.0, -2.0, 2.0])
            )

    @staticmethod
    def _nearest_points(
        first_coordinates: np.ndarray,
        second_coordinates: np.ndarray,
    ) -> tuple[float, tuple[int, int], tuple[int, int]]:
        difference = (
            first_coordinates[:, None, :].astype(np.float32)
            - second_coordinates[None, :, :].astype(np.float32)
        )
        squared_distance = np.square(difference).sum(axis=2)
        flat_index = int(np.argmin(squared_distance))
        first_index, second_index = np.unravel_index(
            flat_index,
            squared_distance.shape,
        )
        return (
            float(np.sqrt(squared_distance[first_index, second_index])),
            tuple(int(v) for v in first_coordinates[first_index]),
            tuple(int(v) for v in second_coordinates[second_index]),
        )

    def _component_descriptor(
        self,
        region,
        probability: torch.Tensor,
        logit: torch.Tensor,
        attribution: torch.Tensor,
        ownership: torch.Tensor,
        height: int,
        width: int,
    ) -> dict[str, object]:
        coordinates_np = region.coords
        coordinates = torch.as_tensor(
            coordinates_np,
            device=probability.device,
            dtype=torch.long,
        )
        rows = coordinates[:, 0]
        columns = coordinates[:, 1]
        component_probability = probability[0, rows, columns]
        component_logit = logit[0, rows, columns]
        component_attribution = attribution[:, rows, columns]
        component_ownership = ownership[:, rows, columns]
        min_row, min_col, max_row, max_col = region.bbox
        area_fraction = float(region.area) / float(height * width)
        descriptor = torch.cat(
            [
                component_probability.mean().view(1),
                component_probability.max().view(1),
                component_logit.mean().view(1),
                component_logit.max().view(1),
                probability.new_tensor([area_fraction]),
                probability.new_tensor([(max_row - min_row) / float(height)]),
                probability.new_tensor([(max_col - min_col) / float(width)]),
                component_attribution.mean(dim=1),
                component_ownership.mean(dim=1),
            ],
            dim=0,
        )
        return {
            "id": int(region.label),
            "area": int(region.area),
            "centroid": tuple(float(v) for v in region.centroid),
            "coordinates": coordinates_np,
            "descriptor": descriptor,
            "attribution": component_attribution.mean(dim=1),
            "ownership_state": int(
                component_ownership.mean(dim=1).argmax().item()
            ),
        }

    def forward(
        self,
        z_base: torch.Tensor,
        contributions: torch.Tensor,
        decision_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if contributions.shape[1] != self.scale_channels:
            raise ValueError("Unexpected scale-contribution channels.")
        probability = torch.sigmoid(z_base.detach())
        attribution = (
            contributions.detach()
            / (contributions.detach().abs().sum(dim=1, keepdim=True) + 1e-6)
        )
        ownership = decision_probabilities.detach()
        batch_size, _, height, width = z_base.shape

        label_maps = []
        pair_features = []
        pair_records: list[dict[str, object]] = []
        candidate_bridge_mask = torch.zeros_like(z_base)

        for batch_index in range(batch_size):
            binary = (
                probability[batch_index, 0].detach().cpu().numpy() > 0.5
            )
            labels_np = measure.label(binary, connectivity=2)
            label_map = torch.as_tensor(
                labels_np,
                device=z_base.device,
                dtype=torch.long,
            )
            label_maps.append(label_map.unsqueeze(0))
            components = [
                self._component_descriptor(
                    region=region,
                    probability=probability[batch_index],
                    logit=z_base[batch_index].detach(),
                    attribution=attribution[batch_index],
                    ownership=ownership[batch_index],
                    height=height,
                    width=width,
                )
                for region in measure.regionprops(labels_np)
            ]

            for first_index in range(len(components)):
                for second_index in range(first_index + 1, len(components)):
                    first = components[first_index]
                    second = components[second_index]
                    distance, first_point, second_point = self._nearest_points(
                        first["coordinates"],
                        second["coordinates"],
                    )
                    if distance > self.max_pair_distance:
                        continue
                    line_rows, line_columns = draw.line(
                        first_point[0],
                        first_point[1],
                        second_point[0],
                        second_point[1],
                    )
                    corridor = torch.zeros(
                        (height, width),
                        device=z_base.device,
                        dtype=z_base.dtype,
                    )
                    corridor[line_rows, line_columns] = 1.0
                    corridor = corridor * (
                        (label_map != first["id"])
                        & (label_map != second["id"])
                    ).to(corridor.dtype)
                    corridor_coordinates = torch.nonzero(
                        corridor > 0.5,
                        as_tuple=False,
                    )
                    if corridor_coordinates.numel() == 0:
                        continue
                    corridor_rows = corridor_coordinates[:, 0]
                    corridor_columns = corridor_coordinates[:, 1]
                    corridor_probability = probability[
                        batch_index,
                        0,
                        corridor_rows,
                        corridor_columns,
                    ]
                    corridor_attribution = attribution[
                        batch_index,
                        :,
                        corridor_rows,
                        corridor_columns,
                    ].mean(dim=1)
                    first_attribution = first["attribution"]
                    second_attribution = second["attribution"]
                    attribution_difference = (
                        first_attribution - second_attribution
                    ).abs()
                    attribution_consistency = (
                        1.0 - 0.5 * attribution_difference.sum()
                    ).clamp(min=0.0, max=1.0)
                    first_centroid = first["centroid"]
                    second_centroid = second["centroid"]
                    area_ratio = min(first["area"], second["area"]) / float(
                        max(first["area"], second["area"])
                    )
                    relation_descriptor = torch.cat(
                        [
                            first["descriptor"],
                            second["descriptor"],
                            attribution_difference,
                            attribution_consistency.view(1),
                            probability.new_tensor(
                                [distance / self.max_pair_distance]
                            ),
                            probability.new_tensor(
                                [
                                    (second_centroid[0] - first_centroid[0])
                                    / float(height),
                                    (second_centroid[1] - first_centroid[1])
                                    / float(width),
                                ]
                            ),
                            probability.new_tensor([area_ratio]),
                            torch.stack(
                                [
                                    corridor_probability.mean(),
                                    corridor_probability.max(),
                                    corridor_probability.min(),
                                ]
                            ),
                            torch.stack(
                                [
                                    (corridor_attribution - first_attribution)
                                    .abs()
                                    .mean(),
                                    (corridor_attribution - second_attribution)
                                    .abs()
                                    .mean(),
                                ]
                            ),
                        ],
                        dim=0,
                    )
                    pair_features.append(relation_descriptor)
                    pair_records.append(
                        {
                            "batch": batch_index,
                            "first_id": first["id"],
                            "second_id": second["id"],
                            "first_state": first["ownership_state"],
                            "second_state": second["ownership_state"],
                            "corridor": corridor,
                        }
                    )
                    candidate_bridge_mask[batch_index, 0] = torch.maximum(
                        candidate_bridge_mask[batch_index, 0],
                        corridor,
                    )

        component_label_map = torch.stack(label_maps, dim=0)
        bridge_mask = torch.zeros_like(z_base)
        suppress_mask = torch.zeros_like(z_base)
        if pair_features:
            pair_feature_tensor = torch.stack(pair_features, dim=0)
            relation_logits = self.relation_head(pair_feature_tensor)
            relation_probabilities = torch.softmax(relation_logits, dim=1)
            winners = relation_probabilities.detach().argmax(dim=1).cpu().tolist()
            pair_batch = []
            pair_first_id = []
            pair_second_id = []
            for pair_index, (record, winner) in enumerate(
                zip(pair_records, winners)
            ):
                del pair_index
                batch_index = int(record["batch"])
                first_id = int(record["first_id"])
                second_id = int(record["second_id"])
                first_state = int(record["first_state"])
                second_state = int(record["second_state"])
                pair_batch.append(batch_index)
                pair_first_id.append(first_id)
                pair_second_id.append(second_id)
                if (
                    winner == self.RECONNECT
                    and first_state == 0
                    and second_state == 0
                ):
                    bridge_mask[batch_index, 0] = torch.maximum(
                        bridge_mask[batch_index, 0],
                        record["corridor"],
                    )
                elif (
                    winner == self.SUPPRESS_FIRST
                    and first_state != 0
                    and second_state == 0
                ):
                    suppress_mask[batch_index, 0] = torch.maximum(
                        suppress_mask[batch_index, 0],
                        (label_maps[batch_index][0] == first_id).to(z_base.dtype),
                    )
                elif (
                    winner == self.SUPPRESS_SECOND
                    and first_state == 0
                    and second_state != 0
                ):
                    suppress_mask[batch_index, 0] = torch.maximum(
                        suppress_mask[batch_index, 0],
                        (label_maps[batch_index][0] == second_id).to(z_base.dtype),
                    )
            pair_batch_tensor = torch.tensor(
                pair_batch,
                device=z_base.device,
                dtype=torch.long,
            )
            pair_first_id_tensor = torch.tensor(
                pair_first_id,
                device=z_base.device,
                dtype=torch.long,
            )
            pair_second_id_tensor = torch.tensor(
                pair_second_id,
                device=z_base.device,
                dtype=torch.long,
            )
        else:
            relation_logits = z_base.new_empty((0, 4))
            relation_probabilities = z_base.new_empty((0, 4))
            pair_batch_tensor = torch.empty(
                (0,),
                device=z_base.device,
                dtype=torch.long,
            )
            pair_first_id_tensor = torch.empty_like(pair_batch_tensor)
            pair_second_id_tensor = torch.empty_like(pair_batch_tensor)

        return {
            "component_relation_logits": relation_logits,
            "component_relation_probabilities": relation_probabilities,
            "component_relation_label_map": component_label_map,
            "component_relation_pair_batch": pair_batch_tensor,
            "component_relation_pair_first_id": pair_first_id_tensor,
            "component_relation_pair_second_id": pair_second_id_tensor,
            "component_relation_candidate_bridge_mask": candidate_bridge_mask,
            "component_relation_bridge_mask": bridge_mask,
            "component_relation_suppress_mask": suppress_mask,
            "component_relation_pair_count": z_base.new_tensor(
                float(len(pair_records))
            ),
        }
