"""Minimal signed local-reference probes for frozen MSHNet ``d0`` features.

These heads are diagnostics, not the proposed paper method.  They test whether
the target direction remains linearly readable after removing a local annular
background location and scale.  Only the final direction vector and bias are
learned; the annulus and standardization are fixed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnnularReference(nn.Module):
    """Compute valid-pixel annular mean, variance, and signed residuals."""

    def __init__(
        self,
        channels: int,
        *,
        outer_size: int = 9,
        inner_size: int = 3,
        variance_floor_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        for value, name in ((outer_size, "outer_size"), (inner_size, "inner_size")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value % 2 != 1:
                raise ValueError(f"{name} must be a positive odd integer")
        if inner_size >= outer_size:
            raise ValueError("inner_size must be smaller than outer_size")
        if (
            not math.isfinite(float(variance_floor_scale))
            or float(variance_floor_scale) <= 0.0
        ):
            raise ValueError("variance_floor_scale must be finite and strictly positive")

        kernel = torch.ones(1, 1, outer_size, outer_size, dtype=torch.float32)
        start = (outer_size - inner_size) // 2
        kernel[:, :, start : start + inner_size, start : start + inner_size] = 0.0
        self.channels = channels
        self.outer_size = outer_size
        self.inner_size = inner_size
        self.variance_floor_scale = float(variance_floor_scale)
        self.register_buffer("annulus_kernel", kernel, persistent=True)

    @property
    def annulus_pixels(self) -> int:
        return self.outer_size**2 - self.inner_size**2

    def _validate(self, features: torch.Tensor) -> None:
        if not torch.is_tensor(features) or features.ndim != 4:
            raise ValueError("features must be a four-dimensional tensor")
        if features.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {features.shape[1]}"
            )
        if not features.is_floating_point():
            raise ValueError("features must use a floating-point dtype")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must contain only finite values")

    def _annular_mean(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding = self.outer_size // 2
        kernel = self.annulus_kernel.to(
            device=features.device,
            dtype=features.dtype,
        )
        channel_kernel = kernel.expand(self.channels, 1, -1, -1)
        numerator = F.conv2d(
            features,
            channel_kernel,
            padding=padding,
            groups=self.channels,
        )
        valid = torch.ones_like(features[:, :1])
        count = F.conv2d(valid, kernel, padding=padding)
        if not bool(torch.all(count > 0)):
            raise RuntimeError("annulus has no valid reference pixels at some location")
        mean = numerator / count
        if not bool(torch.isfinite(mean).all()):
            raise RuntimeError("annular mean produced non-finite values")
        return mean, count

    def statistics(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return annular mean, population variance, and valid-pixel count."""

        self._validate(features)
        # Removing one detached per-image/per-channel anchor before the
        # second-moment identity avoids catastrophic cancellation under large
        # additive feature offsets while preserving the exact local residual.
        anchor = features.detach().mean(dim=(-2, -1), keepdim=True)
        shifted = features - anchor
        mean_shifted, count = self._annular_mean(shifted)
        second_shifted, second_count = self._annular_mean(shifted.square())
        if not torch.equal(count, second_count):
            raise RuntimeError("annular valid-pixel counts drifted")
        mean = mean_shifted + anchor
        variance = (second_shifted - mean_shifted.square()).clamp_min(0.0)
        if not bool(torch.isfinite(variance).all()):
            raise RuntimeError("annular variance produced non-finite values")
        return mean, variance, count

    def centered(self, features: torch.Tensor) -> torch.Tensor:
        mean, _, _ = self.statistics(features)
        return features - mean

    def standardized(self, features: torch.Tensor) -> torch.Tensor:
        mean, variance, _ = self.statistics(features)
        residual = features - mean
        anchor = features.detach().mean(dim=(-2, -1), keepdim=True)
        global_variance = (features.detach() - anchor).square().mean(
            dim=(-2, -1),
            keepdim=True,
        )
        floor = self.variance_floor_scale * global_variance
        denominator_squared = variance + floor
        tiny = torch.finfo(features.dtype).tiny
        normalized = residual * torch.rsqrt(denominator_squared.clamp_min(tiny))
        output = torch.where(
            denominator_squared > 0,
            normalized,
            torch.zeros_like(normalized),
        )
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("annular standardization produced non-finite values")
        return output

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.standardized(features)


class LinearProbeReadout(nn.Module):
    """A parameter-matched ordinary 1x1 affine readout."""

    def __init__(self, channels: int, *, initialization_seed: int = 0) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(initialization_seed))
        weight = torch.randn(1, channels, 1, 1, generator=generator)
        weight = F.normalize(weight.flatten(1), p=2.0, dim=1).view_as(weight)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(features) or features.ndim != 4:
            raise ValueError("features must be a four-dimensional tensor")
        if features.shape[1] != self.weight.shape[1]:
            raise ValueError("feature channel count disagrees with the readout")
        output = F.conv2d(
            features,
            self.weight,
            bias=self.bias,
        )
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("linear probe produced non-finite values")
        return output


class RawUnitLinearProbe(nn.Module):
    """Parameter-matched raw-feature linear control."""

    def __init__(self, channels: int, *, initialization_seed: int = 0) -> None:
        super().__init__()
        self.readout = LinearProbeReadout(
            channels,
            initialization_seed=initialization_seed,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.readout(features)


class CenteredLocalReferenceProbe(nn.Module):
    """Signed annulus-centered control without variance standardization."""

    def __init__(
        self,
        channels: int,
        *,
        outer_size: int = 9,
        inner_size: int = 3,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        self.reference = AnnularReference(
            channels,
            outer_size=outer_size,
            inner_size=inner_size,
        )
        self.readout = LinearProbeReadout(
            channels,
            initialization_seed=initialization_seed,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.readout(self.reference.centered(features))


class SignedStandardizedLocalReferenceProbe(nn.Module):
    """The signed quotient-space direction probe used by the readout gate."""

    def __init__(
        self,
        channels: int,
        *,
        outer_size: int = 9,
        inner_size: int = 3,
        variance_floor_scale: float = 1e-4,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        self.reference = AnnularReference(
            channels,
            outer_size=outer_size,
            inner_size=inner_size,
            variance_floor_scale=variance_floor_scale,
        )
        self.readout = LinearProbeReadout(
            channels,
            initialization_seed=initialization_seed,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.readout(self.reference.standardized(features))


class UnsignedStandardizedNormControl(nn.Module):
    """Parameter-light even-symmetry control; deliberately not a candidate."""

    def __init__(
        self,
        channels: int,
        *,
        outer_size: int = 9,
        inner_size: int = 3,
        variance_floor_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        self.reference = AnnularReference(
            channels,
            outer_size=outer_size,
            inner_size=inner_size,
            variance_floor_scale=variance_floor_scale,
        )
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        standardized = self.reference.standardized(features)
        energy = standardized.square().mean(dim=1, keepdim=True)
        scale = self.log_scale.exp().view(1, 1, 1, 1)
        return scale * energy + self.bias.view(1, 1, 1, 1)


class UnsignedStandardizedProjectionControl(nn.Module):
    """Parameter-matched even control: ``abs(w^T u) + b``."""

    def __init__(
        self,
        channels: int,
        *,
        outer_size: int = 9,
        inner_size: int = 3,
        variance_floor_scale: float = 1e-4,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        self.reference = AnnularReference(
            channels,
            outer_size=outer_size,
            inner_size=inner_size,
            variance_floor_scale=variance_floor_scale,
        )
        self.readout = LinearProbeReadout(
            channels,
            initialization_seed=initialization_seed,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        standardized = self.reference.standardized(features)
        projection = F.conv2d(standardized, self.readout.weight, bias=None)
        output = projection.abs() + self.readout.bias.view(1, 1, 1, 1)
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("unsigned projection control produced non-finite values")
        return output


__all__ = [
    "AnnularReference",
    "CenteredLocalReferenceProbe",
    "LinearProbeReadout",
    "RawUnitLinearProbe",
    "SignedStandardizedLocalReferenceProbe",
    "UnsignedStandardizedNormControl",
    "UnsignedStandardizedProjectionControl",
]
