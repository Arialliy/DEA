"""MSHNet encoder with one shared coarse-to-fine predictive-correction decoder.

The decoder carries a multi-channel state.  At every encoder scale the same
linear operator predicts the aligned encoder observation from that state; the
robust, unexplained residual is then mapped back with the exact adjoint of the
same operator.  There are no per-scale decoder blocks, gates, side heads, or
terminal scale-fusion layer.
"""

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.MSHNet import ResNet


def _inverse_softplus(value: float) -> float:
    value_tensor = torch.tensor(float(value), dtype=torch.float64)
    return float(torch.log(torch.expm1(value_tensor)).item())


class TiedPredictionOperator(nn.Module):
    """Shared spatial/channel predictor and its parameter-tied adjoint."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.depthwise_weight = nn.Parameter(
            torch.zeros(self.channels, 1, 3, 3)
        )
        self.pointwise_weight = nn.Parameter(
            torch.empty(self.channels, self.channels, 1, 1)
        )
        nn.init.orthogonal_(
            self.pointwise_weight.view(self.channels, self.channels)
        )
        with torch.no_grad():
            self.depthwise_weight[:, 0, 1, 1] = 1.0

    def bounded_weights(self) -> Tuple[Tensor, Tensor]:
        # The per-channel l1 cap keeps the depthwise convolution from acquiring
        # unbounded gain while preserving an exact identity initialization.
        weight = self.depthwise_weight
        scale = weight.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        depthwise_weight = weight / scale
        pointwise_matrix = self.pointwise_weight.view(
            self.channels, self.channels
        )
        spectral_scale = torch.linalg.matrix_norm(
            pointwise_matrix.float(), ord=2
        ).to(dtype=pointwise_matrix.dtype).clamp_min(1.0)
        pointwise_weight = self.pointwise_weight / spectral_scale
        return depthwise_weight, pointwise_weight

    def forward(
        self,
        state: Tensor,
        weights: Tuple[Tensor, Tensor] = None,
    ) -> Tensor:
        depthwise_weight, pointwise_weight = (
            self.bounded_weights() if weights is None else weights
        )
        depthwise = F.conv2d(
            state,
            depthwise_weight,
            bias=None,
            stride=1,
            padding=1,
            groups=self.channels,
        )
        return F.conv2d(depthwise, pointwise_weight, bias=None)

    def adjoint(
        self,
        residual: Tensor,
        weights: Tuple[Tensor, Tensor] = None,
    ) -> Tensor:
        # K = M D, hence K* = D* M*.  conv_transpose2d with the exact forward
        # weights implements the corresponding Euclidean adjoints.
        depthwise_weight, pointwise_weight = (
            self.bounded_weights() if weights is None else weights
        )
        channel_backprojection = F.conv_transpose2d(
            residual, pointwise_weight, bias=None
        )
        return F.conv_transpose2d(
            channel_backprojection,
            depthwise_weight,
            bias=None,
            stride=1,
            padding=1,
            groups=self.channels,
        )


class PredictiveCorrectionMSHNet(nn.Module):
    """Replace MSHNet's complete decoder with one shared state transition."""

    BASELINE_MISSING_PREFIXES = (
        "scale_adapters.",
        "observation_norm.",
        "prediction_operator.",
        "state_prior",
        "raw_delta",
        "readout.",
    )
    BASELINE_UNEXPECTED_PREFIXES = (
        "decoder_0.",
        "decoder_1.",
        "decoder_2.",
        "decoder_3.",
        "output_0.",
        "output_1.",
        "output_2.",
        "output_3.",
        "final.",
        "decidability_head.",
    )

    def __init__(
        self,
        input_channels: int,
        state_channels: int = 32,
        step_size: float = 1.0,
        delta_init: float = 1.0,
        delta_min: float = 0.05,
        legacy_influence_numerics: bool = False,
        block=ResNet,
    ):
        super().__init__()
        if state_channels < 4:
            raise ValueError("state_channels must be >= 4")
        if not 0.0 < step_size <= 1.0:
            raise ValueError("step_size must be in (0, 1]")
        if delta_min <= 0.0 or delta_init <= delta_min:
            raise ValueError("delta_init must be greater than positive delta_min")

        encoder_channels = (16, 32, 64, 128, 256)
        encoder_blocks = (2, 2, 2, 2)
        self.state_channels = int(state_channels)
        self.step_size = float(step_size)
        self.delta_min = float(delta_min)
        self.legacy_influence_numerics = bool(legacy_influence_numerics)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv_init = nn.Conv2d(input_channels, encoder_channels[0], 1, 1)
        self.encoder_0 = self._make_layer(
            encoder_channels[0], encoder_channels[0], block
        )
        self.encoder_1 = self._make_layer(
            encoder_channels[0], encoder_channels[1], block, encoder_blocks[0]
        )
        self.encoder_2 = self._make_layer(
            encoder_channels[1], encoder_channels[2], block, encoder_blocks[1]
        )
        self.encoder_3 = self._make_layer(
            encoder_channels[2], encoder_channels[3], block, encoder_blocks[2]
        )
        self.middle_layer = self._make_layer(
            encoder_channels[3], encoder_channels[4], block, encoder_blocks[3]
        )

        # Feature order below is coarse -> fine: 256, 128, 64, 32, 16.
        self.scale_adapters = nn.ModuleList(
            nn.Conv2d(channels, self.state_channels, kernel_size=1, bias=False)
            for channels in reversed(encoder_channels)
        )
        group_count = math.gcd(self.state_channels, 8)
        self.observation_norm = nn.GroupNorm(group_count, self.state_channels)
        self.prediction_operator = TiedPredictionOperator(self.state_channels)
        self.state_prior = nn.Parameter(torch.zeros(1, self.state_channels, 1, 1))

        raw_delta = _inverse_softplus(delta_init - delta_min)
        self.raw_delta = nn.Parameter(
            torch.full((1, self.state_channels, 1, 1), raw_delta)
        )
        self.readout = nn.Conv2d(self.state_channels, 1, kernel_size=1, bias=True)

    @staticmethod
    def _make_layer(in_channels, out_channels, block, block_num=1):
        layers = [block(in_channels, out_channels)]
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    @property
    def delta(self) -> Tensor:
        return self.delta_min + F.softplus(self.raw_delta)

    def _encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))
        return x_m, x_e3, x_e2, x_e1, x_e0

    def _influence(self, residual: Tensor) -> Tensor:
        delta = self.delta.to(dtype=residual.dtype)
        if self.legacy_influence_numerics:
            # Compatibility path for the 2026-07-10 NUAA exploratory runs.
            # It is mathematically equivalent but not bitwise identical to
            # the stable form below.
            return residual / torch.sqrt(
                1.0 + (residual / delta).square()
            )
        # Stable pseudo-Huber derivative.  This avoids squaring residual/delta,
        # which can overflow before the bounded result is formed.
        return delta * (residual / torch.hypot(residual, delta))

    def _robust_energy(self, residual: Tensor) -> Tensor:
        # Diagnostics are accumulated in float32 even if inference later uses
        # lower precision.  The rationalized form avoids cancellation near
        # zero: delta * (hypot(r, delta) - delta)
        #       = delta * |r| * |r| / (hypot(r, delta) + delta).
        residual_float = residual.float()
        delta = self.delta.float()
        magnitude = residual_float.abs()
        denominator = torch.hypot(magnitude, delta) + delta
        return delta * magnitude * (magnitude / denominator)

    def forward(
        self,
        x: Tensor,
        warm_flag: bool = True,
        return_dict: bool = False,
        return_details: bool = False,
    ):
        features: Sequence[Tensor] = self._encode(x)
        state = None
        logits: List[Tensor] = []
        observations: List[Tensor] = []
        residuals: List[Tensor] = []
        residuals_after: List[Tensor] = []
        corrections: List[Tensor] = []
        state_bars: List[Tensor] = []
        states: List[Tensor] = []
        local_energies_before: List[Tensor] = []
        local_energies_after: List[Tensor] = []
        operator_weights = self.prediction_operator.bounded_weights()

        for feature, adapter in zip(features, self.scale_adapters):
            observation = self.observation_norm(adapter(feature))
            if state is None:
                prior = self.state_prior.to(dtype=observation.dtype)
                state_bar = prior.expand(
                    observation.shape[0], -1,
                    observation.shape[-2], observation.shape[-1]
                )
            else:
                state_bar = F.interpolate(
                    state,
                    size=observation.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            predicted_observation = self.prediction_operator(
                state_bar, weights=operator_weights
            )
            residual = observation - predicted_observation
            correction = self.step_size * self.prediction_operator.adjoint(
                self._influence(residual), weights=operator_weights
            )
            state = state_bar + correction
            logit = self.readout(state)

            logits.append(logit)
            if return_details:
                observations.append(observation)
                residuals.append(residual)
                corrections.append(correction)
                state_bars.append(state_bar)
                states.append(state)
                residual_after = observation - self.prediction_operator(
                    state, weights=operator_weights
                )
                residuals_after.append(residual_after)
                local_energies_before.append(
                    self._robust_energy(residual).flatten(1).mean(dim=1)
                )
                local_energies_after.append(
                    self._robust_energy(residual_after).flatten(1).mean(dim=1)
                )

        pred = logits[-1]
        if return_dict:
            output: Dict[str, object] = {
                "pred": pred,
                "state_logits": logits,
                "aux_enabled": bool(warm_flag),
            }
            if return_details:
                output.update({
                    "observations": observations,
                    "residuals": residuals,
                    "residuals_after": residuals_after,
                    "corrections": corrections,
                    "state_bars": state_bars,
                    "states": states,
                    "local_observation_energies_before": local_energies_before,
                    "local_observation_energies_after": local_energies_after,
                    "delta": self.delta,
                })
            return output

        # Keep the historical two-value interface for generic inference code.
        return [], pred

    @staticmethod
    def state_statistics(output: Dict[str, object]) -> Dict[str, float]:
        residuals = output.get("residuals", [])
        residuals_after = output.get("residuals_after", [])
        corrections = output.get("corrections", [])
        state_bars = output.get("state_bars", [])
        energies_before = output.get(
            "local_observation_energies_before", []
        )
        energies_after = output.get(
            "local_observation_energies_after", []
        )
        stats: Dict[str, float] = {}
        for step, (residual, correction, state_bar) in enumerate(
            zip(residuals, corrections, state_bars)
        ):
            stats["pc_step_%d_residual_abs" % step] = float(
                residual.detach().abs().mean().item()
            )
            stats["pc_step_%d_correction_abs" % step] = float(
                correction.detach().abs().mean().item()
            )
            if step > 0:
                correction_flat = correction.detach().flatten(1)
                prior_flat = state_bar.detach().flatten(1)
                ratio = correction_flat.norm() / (prior_flat.norm() + 1e-6)
                cosine = F.cosine_similarity(
                    correction_flat, prior_flat, dim=1, eps=1e-6
                ).mean()
                stats["pc_step_%d_correction_to_prior" % step] = float(
                    ratio.item()
                )
                stats["pc_step_%d_correction_prior_cosine" % step] = float(
                    cosine.item()
                )
        for step, (residual_before, residual_after) in enumerate(zip(
            residuals, residuals_after
        )):
            before_norm = residual_before.detach().flatten(1).norm(dim=1)
            after_norm = residual_after.detach().flatten(1).norm(dim=1)
            valid = before_norm > 1e-8
            if valid.any():
                ratio = after_norm[valid] / before_norm[valid]
                stats["pc_step_%d_residual_norm_ratio_mean" % step] = float(
                    ratio.mean().item()
                )
                stats["pc_step_%d_residual_norm_ratio_max" % step] = float(
                    ratio.max().item()
                )
        for step, (energy_before, energy_after) in enumerate(
            zip(energies_before, energies_after)
        ):
            before = energy_before.detach().float()
            after = energy_after.detach().float()
            difference = after - before
            stats["pc_step_%d_local_energy_before" % step] = float(
                before.mean().item()
            )
            stats["pc_step_%d_local_energy_after" % step] = float(
                after.mean().item()
            )
            stats["pc_step_%d_local_energy_max_increase" % step] = float(
                difference.max().item()
            )
            stats["pc_step_%d_local_energy_violation_fraction" % step] = float(
                (difference > 1e-6).float().mean().item()
            )
        delta = output.get("delta")
        if torch.is_tensor(delta):
            stats["pc_delta_mean"] = float(delta.detach().mean().item())
            stats["pc_delta_min"] = float(delta.detach().min().item())
            stats["pc_delta_max"] = float(delta.detach().max().item())
        return stats


__all__ = ["PredictiveCorrectionMSHNet", "TiedPredictionOperator"]
