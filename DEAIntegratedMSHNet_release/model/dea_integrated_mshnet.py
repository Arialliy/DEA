"""Endogenous Decidable Evidence Routing for MSHNet.

This file deliberately keeps the original MSHNet encoder, decoder blocks, output
heads, and final-fusion parameter names.  It changes only the *fusion algebra*:

1. Every decoder fusion is evaluated through a tri-state routing cell.
2. The original 4->1 final convolution is represented as four exact per-scale
   contributions and is recursively corrected by the same decoder routes.

At initialization, every hard routing decision is ``uncertain``.  Consequently,
the decoder residual and the scale-evidence correction are exactly zero in the
forward pass, so a loaded MSHNet checkpoint is functionally preserved.

The hard decision uses a straight-through estimator (hard forward, soft
backward).  A literal ``argmax`` branch would provide exact identity but would
also block gradients to the routing predictor; the straight-through form gives
both properties without adding a routing loss.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MSHNet import MSHNet, ResNet


Tensor = torch.Tensor
RouteDict = Dict[str, Tensor]

_ROUTE_TARGET = 0
_ROUTE_CLUTTER = 1
_ROUTE_UNCERTAIN = 2
_VALID_ROUTING_MODES = ("dea", "soft_tri", "attention")
_MIN_STRICT_UNCERTAIN_MARGIN = 0.1


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (int(value), int(value))


def _strip_module_prefix(state_dict: Mapping[str, Tensor]) -> "OrderedDict[str, Tensor]":
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return OrderedDict((key[len("module."):], value) for key, value in state_dict.items())
    return OrderedDict(state_dict.items())


def _resize_gate(gate: Tensor, size: Tuple[int, int], mode: str) -> Tensor:
    if gate.shape[-2:] == size:
        return gate
    if mode in ("nearest", "nearest-exact", "area"):
        return F.interpolate(gate, size=size, mode=mode)
    return F.interpolate(gate, size=size, mode=mode, align_corners=True)


class DecidableEvidenceRoutingCell(nn.Module):
    """A baseline-preserving two-input decoder fusion operator.

    Given encoder feature ``e`` and upsampled decoder feature ``d``:

    ``agreement = tanh(Pe(e)) * tanh(Pd(d))``
    ``disagreement = abs(tanh(Pe(e)) - tanh(Pd(d)))``

    A three-way predictor chooses target, clutter, or uncertain.  In ``dea``
    mode, a straight-through one-hot winner masks soft confidence values.  Thus
    an uncertain winner gives exactly zero target/clutter gates in the forward
    pass while gradients still reach the routing predictor.

    The original decoder block remains the baseline operator inside the cell:

    ``y = decoder(cat(e, d)) + (g_target - g_clutter) * update(e, d)``

    This is not an output head: the routed feature is consumed recursively by
    every finer decoder stage.
    """

    def __init__(
        self,
        encoder_channels: int,
        decoder_channels: int,
        output_channels: int,
        route_channels: int = 16,
        temperature: float = 1.0,
        routing_mode: str = "dea",
        update_limit: float = 0.25,
        uncertain_margin: float = 1.0,
        isolate_route_gradients: bool = True,
    ) -> None:
        super().__init__()
        if route_channels < 1:
            raise ValueError("route_channels must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if routing_mode not in _VALID_ROUTING_MODES:
            raise ValueError(
                "routing_mode must be one of %s, got %r"
                % (_VALID_ROUTING_MODES, routing_mode)
            )
        if update_limit <= 0:
            raise ValueError("update_limit must be > 0")
        if (
            routing_mode in ("dea", "soft_tri")
            and uncertain_margin <= _MIN_STRICT_UNCERTAIN_MARGIN
        ):
            raise ValueError(
                "uncertain_margin must be > %.3f to guarantee an all-uncertain "
                "initial route" % _MIN_STRICT_UNCERTAIN_MARGIN
            )

        self.encoder_channels = int(encoder_channels)
        self.decoder_channels = int(decoder_channels)
        self.output_channels = int(output_channels)
        self.route_channels = int(route_channels)
        self.temperature = float(temperature)
        self.routing_mode = routing_mode
        self.update_limit = float(update_limit)
        self.uncertain_margin = float(uncertain_margin)
        self.isolate_route_gradients = bool(isolate_route_gradients)

        # Tanh after projection bounds each evidence channel.  This allows the
        # initialization below to *guarantee* that uncertain is the winner,
        # rather than merely making it likely on a calibration batch.
        self.encoder_projection = nn.Conv2d(
            self.encoder_channels, self.route_channels, kernel_size=1, bias=False
        )
        self.decoder_projection = nn.Conv2d(
            self.decoder_channels, self.route_channels, kernel_size=1, bias=False
        )

        evidence_channels = 2 * self.route_channels
        self.route_predictor = nn.Conv2d(evidence_channels, 3, kernel_size=1, bias=True)

        # A lightweight depthwise-separable evidence update.  No BatchNorm is
        # used, so an inactive uncertain branch cannot silently alter running
        # statistics while the forward mapping is supposed to be identical.
        self.update_depthwise = nn.Conv2d(
            evidence_channels,
            evidence_channels,
            kernel_size=3,
            padding=1,
            groups=evidence_channels,
            bias=False,
        )
        self.update_pointwise = nn.Conv2d(
            evidence_channels, self.output_channels, kernel_size=1, bias=True
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.encoder_projection.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.decoder_projection.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.update_depthwise.weight, mode="fan_in", nonlinearity="relu")
        # A tiny, non-zero output projection keeps decoder-only routing
        # trainable while preventing an untrained update from causing a large
        # feature jump at the first target/clutter decision.
        nn.init.normal_(self.update_pointwise.weight, mean=0.0, std=1e-3)
        if self.update_pointwise.bias is not None:
            nn.init.zeros_(self.update_pointwise.bias)

        # For bounded evidence, the target/clutter response magnitude is at
        # most approximately 3 * route_channels * init_scale = 0.1.  The fixed
        # uncertain bias therefore wins everywhere at initialization.
        init_scale = 0.1 / float(3 * self.route_channels)
        with torch.no_grad():
            self.route_predictor.weight.zero_()
            self.route_predictor.bias.zero_()

            if self.routing_mode == "dea":
                nn.init.uniform_(
                    self.route_predictor.weight[_ROUTE_TARGET:_ROUTE_UNCERTAIN],
                    -init_scale,
                    init_scale,
                )
                self.route_predictor.bias[_ROUTE_UNCERTAIN] = self.uncertain_margin
            elif self.routing_mode == "soft_tri":
                # No structural abstention: target and clutter begin equal, so
                # the initial residual cancels only by parameter symmetry.
                shared = torch.empty_like(self.route_predictor.weight[_ROUTE_TARGET])
                nn.init.uniform_(shared, -init_scale, init_scale)
                self.route_predictor.weight[_ROUTE_TARGET].copy_(shared)
                self.route_predictor.weight[_ROUTE_CLUTTER].copy_(shared)
                self.route_predictor.bias[_ROUTE_UNCERTAIN] = self.uncertain_margin
            else:  # parameter-matched continuous signed attention comparator
                shared = torch.empty_like(self.route_predictor.weight[_ROUTE_TARGET])
                nn.init.uniform_(shared, -init_scale, init_scale)
                self.route_predictor.weight[_ROUTE_TARGET].copy_(shared)
                self.route_predictor.weight[_ROUTE_CLUTTER].copy_(shared)
                nn.init.uniform_(
                    self.route_predictor.weight[_ROUTE_UNCERTAIN],
                    -init_scale,
                    init_scale,
                )

    def _build_evidence(self, encoder_feature: Tensor, decoder_feature: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if encoder_feature.shape[-2:] != decoder_feature.shape[-2:]:
            raise ValueError(
                "encoder and decoder features must have the same spatial size, got %s and %s"
                % (tuple(encoder_feature.shape), tuple(decoder_feature.shape))
            )
        # Route parameters may learn while the hard action is keep/abstain, but
        # their straight-through surrogate must not silently regularize the
        # transported MSHNet features.  Otherwise an all-keep model can beat a
        # paired continued baseline without executing a single route action.
        # Detaching only the route observations leaves the original decoder
        # baseline path and all forward values unchanged.
        route_encoder = (
            encoder_feature.detach()
            if self.isolate_route_gradients
            else encoder_feature
        )
        route_decoder = (
            decoder_feature.detach()
            if self.isolate_route_gradients
            else decoder_feature
        )
        encoder_projected = torch.tanh(self.encoder_projection(route_encoder))
        decoder_projected = torch.tanh(self.decoder_projection(route_decoder))
        agreement = encoder_projected * decoder_projected
        disagreement = torch.abs(encoder_projected - decoder_projected)
        evidence = torch.cat([agreement, disagreement], dim=1)
        return evidence, agreement, disagreement

    def _compute_route(self, logits: Tensor) -> RouteDict:
        probabilities = torch.softmax(logits / self.temperature, dim=1)
        winner = torch.argmax(logits, dim=1)
        hard = F.one_hot(winner, num_classes=3).permute(0, 3, 1, 2).to(logits.dtype)

        if self.routing_mode == "dea":
            # Exact hard winner in the forward pass; softmax Jacobian in the
            # backward pass.  Parentheses preserve an exact forward zero for
            # the surrogate term.
            hard_st = hard + (probabilities - probabilities.detach())
            target_gate = hard_st[:, _ROUTE_TARGET:_ROUTE_TARGET + 1] * probabilities[:, _ROUTE_TARGET:_ROUTE_TARGET + 1]
            clutter_gate = hard_st[:, _ROUTE_CLUTTER:_ROUTE_CLUTTER + 1] * probabilities[:, _ROUTE_CLUTTER:_ROUTE_CLUTTER + 1]
            uncertain_gate = hard_st[:, _ROUTE_UNCERTAIN:_ROUTE_UNCERTAIN + 1]
        elif self.routing_mode == "soft_tri":
            # This ablation always mixes target/clutter evidence.  The
            # uncertain probability is diagnostic only and cannot enforce an
            # identity action.
            target_gate = probabilities[:, _ROUTE_TARGET:_ROUTE_TARGET + 1]
            clutter_gate = probabilities[:, _ROUTE_CLUTTER:_ROUTE_CLUTTER + 1]
            uncertain_gate = probabilities[:, _ROUTE_UNCERTAIN:_ROUTE_UNCERTAIN + 1]
        else:
            # Parameter-matched continuous attention: no three-way winner and
            # no abstention state.  Equal target/clutter initialization gives
            # a zero residual, but this is not a structural identity region.
            direction = torch.tanh(
                logits[:, _ROUTE_TARGET:_ROUTE_TARGET + 1]
                - logits[:, _ROUTE_CLUTTER:_ROUTE_CLUTTER + 1]
            )
            amplitude = torch.sigmoid(
                logits[:, _ROUTE_UNCERTAIN:_ROUTE_UNCERTAIN + 1]
            )
            signed_attention = direction * amplitude
            target_gate = F.relu(signed_attention)
            clutter_gate = F.relu(-signed_attention)
            uncertain_gate = 1.0 - torch.abs(signed_attention)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "winner": winner,
            "hard": hard,
            "target_gate": target_gate,
            "clutter_gate": clutter_gate,
            "uncertain_gate": uncertain_gate,
            "signed_gate": target_gate - clutter_gate,
            "soft_signed_gate": (
                probabilities[:, _ROUTE_TARGET:_ROUTE_TARGET + 1]
                - probabilities[:, _ROUTE_CLUTTER:_ROUTE_CLUTTER + 1]
            ),
        }

    def forward(
        self,
        encoder_feature: Tensor,
        decoder_feature: Tensor,
        baseline_decoder: nn.Module,
        apply_update: bool = True,
        return_evidence: bool = False,
    ) -> Tuple[Tensor, RouteDict]:
        evidence, agreement, disagreement = self._build_evidence(
            encoder_feature, decoder_feature
        )
        route = self._compute_route(self.route_predictor(evidence))

        baseline = baseline_decoder(torch.cat([encoder_feature, decoder_feature], dim=1))
        update = self.update_pointwise(F.silu(self.update_depthwise(evidence)))
        update = self.update_limit * torch.tanh(update)

        if apply_update:
            residual = route["signed_gate"] * update
            if self.routing_mode == "dea":
                # Keep the exact uncertain identity in the forward pass while
                # allowing the update branch to learn before the first hard
                # target/clutter route appears.  Restrict this surrogate to
                # hard-uncertain pixels so active hard actions are not counted
                # twice in the backward pass.
                uncertain_mask = route["hard"][
                    :, _ROUTE_UNCERTAIN:_ROUTE_UNCERTAIN + 1
                ].detach()
                surrogate = (
                    uncertain_mask
                    * route["soft_signed_gate"].detach()
                    * update
                )
                residual = residual + surrogate - surrogate.detach()
            output = baseline + residual
        else:
            output = baseline

        route["update"] = update
        route["baseline"] = baseline
        if return_evidence:
            route["agreement"] = agreement
            route["disagreement"] = disagreement
        return output, route


class IntegratedScaleEvidenceFusion(nn.Module):
    """Route-coupled closure of MSHNet's final 4->1 convolution.

    The module owns parameters named ``weight`` and ``bias`` with exactly the
    same shapes as the original ``nn.Conv2d(4, 1, 3, 1, 1)``.  Existing
    checkpoint keys ``final.weight`` and ``final.bias`` therefore load without
    remapping.

    For scale logits ``s`` and original kernel slices ``W_i``:

    ``c_i = conv2d(s_i, W_i)``
    ``z_base = conv2d(cat(s_i), W, bias)``
    ``z_dea  = z_base + sum_i (g_t_i - g_c_i) * abs(c_i)``

    ``z_base`` deliberately uses the original four-channel convolution instead
    of summing grouped per-scale convolutions.  The two forms are algebraically
    equivalent, but only the direct convolution preserves a transported MSHNet
    checkpoint bit-for-bit under finite-precision arithmetic.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Optional[Tensor],
        stride=1,
        padding=1,
        dilation=1,
        route_upsample_mode: str = "nearest-exact",
    ) -> None:
        super().__init__()
        if weight.ndim != 4 or tuple(weight.shape[:2]) != (1, 4):
            raise ValueError(
                "expected final weight with shape [1, 4, k, k], got %s"
                % (tuple(weight.shape),)
            )
        if route_upsample_mode not in ("nearest", "nearest-exact", "bilinear", "bicubic"):
            raise ValueError("unsupported route_upsample_mode: %s" % route_upsample_mode)

        self.weight = nn.Parameter(weight.detach().clone())
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone())

        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.route_upsample_mode = route_upsample_mode

    @classmethod
    def from_conv(
        cls,
        conv: nn.Conv2d,
        route_upsample_mode: str = "nearest-exact",
    ) -> "IntegratedScaleEvidenceFusion":
        if conv.in_channels != 4 or conv.out_channels != 1 or conv.groups != 1:
            raise ValueError(
                "MSHNet final layer must be Conv2d(4, 1, ...), got in=%d out=%d groups=%d"
                % (conv.in_channels, conv.out_channels, conv.groups)
            )
        if conv.padding_mode != "zeros":
            raise ValueError("only zero-padding final convolutions are supported")
        return cls(
            weight=conv.weight,
            bias=conv.bias,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            route_upsample_mode=route_upsample_mode,
        )

    def decompose(self, scale_logits: Tensor) -> Tensor:
        if scale_logits.ndim != 4 or scale_logits.shape[1] != 4:
            raise ValueError(
                "scale_logits must have shape [B, 4, H, W], got %s"
                % (tuple(scale_logits.shape),)
            )
        # [1, 4, k, k] -> [4, 1, k, k], one independent convolution per scale.
        per_scale_weight = self.weight.permute(1, 0, 2, 3).contiguous()
        return F.conv2d(
            scale_logits,
            per_scale_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=4,
        )

    def direct_baseline(self, scale_logits: Tensor) -> Tensor:
        """Execute the original final convolution without changing reduction order."""
        return F.conv2d(
            scale_logits,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def baseline_from_contributions(self, contributions: Tensor) -> Tensor:
        z_base = contributions.sum(dim=1, keepdim=True)
        if self.bias is not None:
            z_base = z_base + self.bias.view(1, 1, 1, 1)
        return z_base

    def forward(
        self,
        scale_logits: Tensor,
        routes: Sequence[RouteDict],
        apply_routing: bool = True,
        return_details: bool = False,
    ):
        if len(routes) != 4:
            raise ValueError("exactly four routes are required, got %d" % len(routes))

        z_base = self.direct_baseline(scale_logits)
        if not apply_routing and not return_details:
            return z_base
        contributions = self.decompose(scale_logits)

        target_gates: List[Tensor] = []
        clutter_gates: List[Tensor] = []
        deltas: List[Tensor] = []
        spatial_size = contributions.shape[-2:]

        # Route order is fine-to-coarse: route_0, route_1, route_2, route_3.
        for scale_index, route in enumerate(routes):
            target_gate = _resize_gate(
                route["target_gate"], spatial_size, self.route_upsample_mode
            )
            clutter_gate = _resize_gate(
                route["clutter_gate"], spatial_size, self.route_upsample_mode
            )
            contribution = contributions[:, scale_index:scale_index + 1]
            delta = target_gate * contribution.abs() - clutter_gate * contribution.abs()
            target_gates.append(target_gate)
            clutter_gates.append(clutter_gate)
            deltas.append(delta)

        # Coarse-to-fine route corrections are retained for diagnostics.  The
        # executable state starts at the original direct convolution so the
        # strict identity route is numerically exact, not merely algebraically
        # exact after a reordered reduction.
        recursive_state = z_base
        recursive_states: List[Tensor] = [recursive_state]
        for scale_index in (3, 2, 1, 0):
            if apply_routing:
                recursive_state = recursive_state + deltas[scale_index]
            recursive_states.append(recursive_state)
        z_dea = recursive_state

        if not return_details:
            return z_dea
        return {
            "z_base": z_base,
            "z_dea": z_dea,
            "contributions": contributions,
            "target_gates": torch.cat(target_gates, dim=1),
            "clutter_gates": torch.cat(clutter_gates, dim=1),
            "deltas": torch.cat(deltas, dim=1),
            "recursive_states": recursive_states,
        }


class DEAIntegratedMSHNet(MSHNet):
    """MSHNet with endogenous, recursively reused decidable evidence routing."""

    BASELINE_MISSING_PREFIXES = (
        "dea_cell_0.",
        "dea_cell_1.",
        "dea_cell_2.",
        "dea_cell_3.",
    )
    BASELINE_UNEXPECTED_PREFIXES = ("decidability_head.",)

    def __init__(
        self,
        input_channels: int,
        block=ResNet,
        route_channels: int = 16,
        route_temperature: float = 1.0,
        routing_mode: str = "dea",
        decoder_routing: bool = True,
        scale_routing: bool = True,
        route_upsample_mode: str = "nearest-exact",
        update_limit: float = 0.25,
        uncertain_margin: float = 1.0,
        isolate_route_gradients: bool = True,
    ) -> None:
        if (
            routing_mode == "dea"
            and scale_routing
            and route_upsample_mode not in ("nearest", "nearest-exact")
        ):
            raise ValueError(
                "hard DEA scale routing requires nearest or nearest-exact "
                "route upsampling"
            )
        super().__init__(input_channels, block=block)

        self.routing_mode = routing_mode
        self.decoder_routing = bool(decoder_routing)
        self.scale_routing = bool(scale_routing)

        # DEA-lite is deliberately not part of the formal integrated model.
        if hasattr(self, "decidability_head"):
            delattr(self, "decidability_head")

        original_final = self.final
        self.final = IntegratedScaleEvidenceFusion.from_conv(
            original_final, route_upsample_mode=route_upsample_mode
        )

        # Channel pairs follow the unmodified MSHNet decoder inputs.
        self.dea_cell_3 = DecidableEvidenceRoutingCell(
            128, 256, 128, route_channels, route_temperature, routing_mode,
            update_limit, uncertain_margin, isolate_route_gradients,
        )
        self.dea_cell_2 = DecidableEvidenceRoutingCell(
            64, 128, 64, route_channels, route_temperature, routing_mode,
            update_limit, uncertain_margin, isolate_route_gradients,
        )
        self.dea_cell_1 = DecidableEvidenceRoutingCell(
            32, 64, 32, route_channels, route_temperature, routing_mode,
            update_limit, uncertain_margin, isolate_route_gradients,
        )
        self.dea_cell_0 = DecidableEvidenceRoutingCell(
            16, 32, 16, route_channels, route_temperature, routing_mode,
            update_limit, uncertain_margin, isolate_route_gradients,
        )

    @staticmethod
    def extract_state_dict(weight_object) -> Mapping[str, Tensor]:
        if isinstance(weight_object, Mapping):
            if "state_dict" in weight_object:
                return weight_object["state_dict"]
            if "net" in weight_object:
                return weight_object["net"]
            if all(torch.is_tensor(value) for value in weight_object.values()):
                return weight_object
        raise RuntimeError(
            "unsupported checkpoint format; expected a raw state_dict or a dict containing 'state_dict'/'net'"
        )

    def load_mshnet_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict_baseline: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Load an MSHNet checkpoint while allowing only known DEA differences."""
        clean_state = _strip_module_prefix(state_dict)
        missing, unexpected = self.load_state_dict(clean_state, strict=False)
        bad_missing = [
            key for key in missing
            if not any(key.startswith(prefix) for prefix in self.BASELINE_MISSING_PREFIXES)
        ]
        bad_unexpected = [
            key for key in unexpected
            if not any(key.startswith(prefix) for prefix in self.BASELINE_UNEXPECTED_PREFIXES)
        ]
        if strict_baseline and (bad_missing or bad_unexpected):
            raise RuntimeError(
                "baseline load failed: bad_missing=%s bad_unexpected=%s"
                % (bad_missing, bad_unexpected)
            )
        return list(missing), list(unexpected)

    def _decode(self, x: Tensor, return_evidence: bool = False):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))

        x_d3, route_3 = self.dea_cell_3(
            x_e3,
            self.up(x_m),
            self.decoder_3,
            apply_update=self.decoder_routing,
            return_evidence=return_evidence,
        )
        x_d2, route_2 = self.dea_cell_2(
            x_e2,
            self.up(x_d3),
            self.decoder_2,
            apply_update=self.decoder_routing,
            return_evidence=return_evidence,
        )
        x_d1, route_1 = self.dea_cell_1(
            x_e1,
            self.up(x_d2),
            self.decoder_1,
            apply_update=self.decoder_routing,
            return_evidence=return_evidence,
        )
        x_d0, route_0 = self.dea_cell_0(
            x_e0,
            self.up(x_d1),
            self.decoder_0,
            apply_update=self.decoder_routing,
            return_evidence=return_evidence,
        )

        decoder_features = (x_d0, x_d1, x_d2, x_d3)
        routes = (route_0, route_1, route_2, route_3)
        return decoder_features, routes

    def forward(
        self,
        x: Tensor,
        warm_flag: bool = True,
        return_dict: bool = False,
        return_evidence: bool = False,
    ):
        decoder_features, routes = self._decode(x, return_evidence=return_evidence)
        x_d0, x_d1, x_d2, x_d3 = decoder_features

        if not warm_flag:
            output = self.output_0(x_d0)
            if return_dict:
                return {"masks": [], "pred": output, "routes": routes}
            return [], output

        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)
        masks = [mask0, mask1, mask2, mask3]

        scale_logits = torch.cat(
            [mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1
        )
        fusion = self.final(
            scale_logits,
            routes,
            apply_routing=self.scale_routing,
            return_details=return_dict,
        )

        if return_dict:
            return {
                "masks": masks,
                "pred": fusion["z_dea"],
                "scale_logits": scale_logits,
                "routes": routes,
                "scale_fusion": fusion,
            }
        return masks, fusion

    def route_statistics(self, routes: Sequence[RouteDict]) -> Dict[str, float]:
        """Return non-differentiable routing occupancy for experiment logs."""
        stats: Dict[str, float] = {}
        with torch.no_grad():
            for scale_index, route in enumerate(routes):
                winner = route["winner"]
                stats["route_%d_target" % scale_index] = float(
                    (winner == _ROUTE_TARGET).float().mean().item()
                )
                stats["route_%d_clutter" % scale_index] = float(
                    (winner == _ROUTE_CLUTTER).float().mean().item()
                )
                stats["route_%d_uncertain" % scale_index] = float(
                    (winner == _ROUTE_UNCERTAIN).float().mean().item()
                )
        return stats


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


__all__ = [
    "DecidableEvidenceRoutingCell",
    "IntegratedScaleEvidenceFusion",
    "DEAIntegratedMSHNet",
    "count_trainable_parameters",
]
