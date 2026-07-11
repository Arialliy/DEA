"""Read-only factorization of feature contrast through existing linear heads."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from utils.feature_survival import ProjectedGeometry


@dataclass(frozen=True)
class HeadConversionResult:
    available: bool
    reason: str | None
    channels: int
    target_effective_cells: float
    background_cells: int
    mean_margin_availability: float | None
    normalized_mean_margin_availability: float | None
    head_sensitivity: float | None
    utilization_cosine: float | None
    mean_logit_margin: float | None
    reconstructed_margin: float | None
    reconstruction_error: float | None
    absolute_scale_floor_active_channels: int | None
    reparameterization_stable: bool | None
    target_mean_logit: float | None
    background_mean_logit: float | None
    target_peak_logit: float | None
    absolute_peak_margin: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite_feature(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or any(size < 1 for size in array.shape):
        raise ValueError("feature must have shape [C,H,W]")
    if not np.isfinite(array).all():
        raise ValueError("feature must contain only finite values")
    return array


def _head_vector(value, *, channels: int) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (1, channels, 1, 1):
        array = array.reshape(channels)
    elif array.shape == (channels,):
        pass
    else:
        raise ValueError(
            "head_weight must have shape [C] or [1,C,1,1]"
        )
    if not np.isfinite(array).all():
        raise ValueError("head_weight must contain only finite values")
    return array


def _scalar_bias(value) -> float:
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError("head_bias must contain one finite scalar")
    return float(array.reshape(-1)[0])


def _robust_channel_center_scale_details(
    background_values,
    *,
    global_values=None,
    absolute_floor: float = 1e-8,
    relative_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return robust center, scale, and absolute-floor activity mask."""

    background = np.asarray(background_values, dtype=np.float64)
    if background.ndim != 2 or background.shape[1] < 2:
        raise ValueError("background_values must have shape [C,N] with N>=2")
    if not np.isfinite(background).all():
        raise ValueError("background_values must be finite")
    if (
        not math.isfinite(float(absolute_floor))
        or absolute_floor <= 0
        or not math.isfinite(float(relative_floor))
        or relative_floor < 0
    ):
        raise ValueError("scale floors must be finite with absolute_floor > 0")
    if global_values is None:
        global_array = background
    else:
        global_array = np.asarray(global_values, dtype=np.float64)
        if (
            global_array.ndim != 2
            or global_array.shape[0] != background.shape[0]
            or global_array.shape[1] < 1
            or not np.isfinite(global_array).all()
        ):
            raise ValueError("global_values must be finite with shape [C,M]")

    center = np.median(background, axis=1)
    deviation = background - center[:, None]
    mad = 1.4826 * np.median(np.abs(deviation), axis=1)
    rms = np.sqrt(np.mean(deviation**2, axis=1))
    global_center = np.median(global_array, axis=1)
    global_deviation = global_array - global_center[:, None]
    global_mad = 1.4826 * np.median(np.abs(global_deviation), axis=1)
    global_rms = np.sqrt(np.mean(global_deviation**2, axis=1))
    relative = float(relative_floor) * np.maximum(global_mad, global_rms)
    data_scale = np.maximum.reduce((mad, 0.1 * rms, relative))
    absolute_floor_active = data_scale <= float(absolute_floor)
    scale = np.maximum(data_scale, float(absolute_floor))
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError("robust channel scale must be finite and positive")
    return center, scale, absolute_floor_active


def robust_channel_center_scale(
    background_values,
    *,
    global_values=None,
    absolute_floor: float = 1e-8,
    relative_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return component-wise median and strictly positive robust scale."""

    center, scale, _ = _robust_channel_center_scale_details(
        background_values,
        global_values=global_values,
        absolute_floor=absolute_floor,
        relative_floor=relative_floor,
    )
    return center, scale


def evaluate_linear_head_conversion(
    feature,
    geometry: ProjectedGeometry | None,
    *,
    head_weight,
    head_bias=None,
    scalar_threshold: float | None = None,
) -> HeadConversionResult:
    """Factor target/background contrast through an existing linear readout.

    ``delta`` is the target occupancy-weighted mean minus the arithmetic mean
    of the selected local-background cells.  The robust median/MAD statistics
    define only the positive diagonal scale ``D``.  Thus the signed margin is
    the actual target-mean minus background-mean head logit (bias cancels).

    With ``a = D^-1 delta`` and ``h = D w``, the utilization is defined as
    ``a^T h / (||a|| ||h||)`` only when both norms are non-zero.  No epsilon is
    added to that denominator: doing so would invalidate the exact identity
    ``w^T delta = availability * sensitivity * utilization``.  Degenerate
    cases return ``utilization_cosine=None`` and
    ``reconstructed_margin=None`` so they cannot be confused with genuine
    orthogonality.
    """

    array = _finite_feature(feature)
    weight = _head_vector(head_weight, channels=array.shape[0])
    bias = _scalar_bias(head_bias)
    if scalar_threshold is not None and not math.isfinite(float(scalar_threshold)):
        raise ValueError("scalar_threshold must be finite when provided")
    if geometry is None:
        return HeadConversionResult(
            available=False,
            reason="insufficient_geometry_controls",
            channels=array.shape[0],
            target_effective_cells=0.0,
            background_cells=0,
            mean_margin_availability=None,
            normalized_mean_margin_availability=None,
            head_sensitivity=None,
            utilization_cosine=None,
            mean_logit_margin=None,
            reconstructed_margin=None,
            reconstruction_error=None,
            absolute_scale_floor_active_channels=None,
            reparameterization_stable=None,
            target_mean_logit=None,
            background_mean_logit=None,
            target_peak_logit=None,
            absolute_peak_margin=None,
        )
    if tuple(array.shape[1:]) != geometry.output_shape:
        raise ValueError("feature and projected geometry spatial shapes differ")

    flat = array.reshape(array.shape[0], -1)
    occupancy = geometry.target.occupancy.reshape(-1)
    target_indices = np.flatnonzero(occupancy > 0)
    target_weights = occupancy[target_indices]
    target_weights = target_weights / target_weights.sum()
    target_values = flat[:, target_indices]
    background_values = flat[:, geometry.target.background_flat_indices]
    _, scale, absolute_floor_active = _robust_channel_center_scale_details(
        background_values,
        global_values=flat,
    )
    target_mean = np.sum(target_values * target_weights[None, :], axis=1)
    background_mean = np.mean(background_values, axis=1)
    delta = target_mean - background_mean
    availability_vector = delta / scale
    head_vector = weight * scale
    availability = float(np.linalg.norm(availability_vector))
    sensitivity = float(np.linalg.norm(head_vector))
    signed_margin = float(np.dot(weight, delta))

    finite_scalars = np.asarray(
        (availability, sensitivity, signed_margin), dtype=np.float64
    )
    if not (
        np.isfinite(availability_vector).all()
        and np.isfinite(head_vector).all()
        and np.isfinite(finite_scalars).all()
    ):
        raise RuntimeError("linear margin factorization produced non-finite values")
    availability_zero = not bool(np.any(availability_vector != 0.0))
    sensitivity_zero = not bool(np.any(head_vector != 0.0))
    if availability == 0.0 and not availability_zero:
        raise RuntimeError("availability norm underflowed for a non-zero vector")
    if sensitivity == 0.0 and not sensitivity_zero:
        raise RuntimeError("head-sensitivity norm underflowed for a non-zero vector")

    if availability_zero:
        reason = "zero_availability"
        utilization = None
        reconstructed = None
    elif sensitivity_zero:
        reason = "zero_head_sensitivity"
        utilization = None
        reconstructed = None
    else:
        reason = None
        utilization = float(
            np.dot(availability_vector, head_vector)
            / (availability * sensitivity)
        )
        if utilization < -1.0 - 1e-12 or utilization > 1.0 + 1e-12:
            raise RuntimeError("utilization cosine escaped its valid range")
        utilization = float(np.clip(utilization, -1.0, 1.0))
        reconstructed = float(availability * sensitivity * utilization)
        if not np.isfinite((utilization, reconstructed)).all():
            raise RuntimeError("linear reconstruction produced non-finite values")
    reconstruction_error = (
        float(abs(signed_margin - reconstructed))
        if reconstructed is not None
        else float(abs(signed_margin))
    )
    tolerance = 1e-9 + 1e-8 * abs(signed_margin)
    if reconstruction_error > tolerance:
        raise RuntimeError("linear margin factorization lost exactness")

    target_logits = weight @ target_values + bias
    target_mean_logit = float(np.sum(target_logits * target_weights))
    background_mean_logit = float(np.dot(weight, background_mean) + bias)
    target_peak = float(np.max(target_logits))
    absolute_margin = (
        target_peak - float(scalar_threshold)
        if scalar_threshold is not None
        else None
    )
    return HeadConversionResult(
        available=True,
        reason=reason,
        channels=array.shape[0],
        target_effective_cells=geometry.target_effective_cells,
        background_cells=int(geometry.target.background_flat_indices.size),
        mean_margin_availability=availability,
        normalized_mean_margin_availability=(
            availability / math.sqrt(array.shape[0])
        ),
        head_sensitivity=sensitivity,
        utilization_cosine=utilization,
        mean_logit_margin=signed_margin,
        reconstructed_margin=reconstructed,
        reconstruction_error=reconstruction_error,
        absolute_scale_floor_active_channels=int(absolute_floor_active.sum()),
        reparameterization_stable=not bool(absolute_floor_active.any()),
        target_mean_logit=target_mean_logit,
        background_mean_logit=background_mean_logit,
        target_peak_logit=target_peak,
        absolute_peak_margin=absolute_margin,
    )


__all__ = [
    "HeadConversionResult",
    "evaluate_linear_head_conversion",
    "robust_channel_center_scale",
]
