"""Exact, read-only conversion diagnostics for MSHNet's final fusion.

MSHNet fuses four full-resolution side logits with a single-output ``3x3``
convolution.  At every output location ``x``, zero-padded ``unfold`` gives a
patch vector ``q_x`` and the native prediction is exactly

``z_x = <w, q_x> + b``.

This module factors the difference between *normalized weighted means* over
target and control output locations.  Because both means have unit mass, the
same scalar convolution bias is added once to each mean and cancels from the
margin.  Zero-padding remains part of each border patch; padded entries are
not dropped or renormalized.

The functions are numerical views only: they neither modify their inputs nor
register parameters, buffers, hooks, or training state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class FinalFusionMarginFactorization:
    """Exact target-control margin decomposition for one fusion head.

    ``utilization_cosine`` and ``reconstructed_margin`` are ``None`` when the
    available contrast or head sensitivity is exactly zero.  Returning an
    undefined value is intentional: inserting an epsilon into the cosine
    denominator would invalidate the claimed exact factorization.
    """

    reconstructed_logits: Tensor
    target_patch_mean: Tensor
    control_patch_mean: Tensor
    patch_difference: Tensor
    patch_scale: Tensor
    whitened_difference: Tensor
    scaled_head: Tensor
    available_contrast: Tensor
    head_sensitivity: Tensor
    utilization_cosine: Tensor | None
    signed_margin: Tensor
    reconstructed_margin: Tensor | None
    target_logit_mean: Tensor
    control_logit_mean: Tensor
    fusion_bias: Tensor
    output_shape: tuple[int, int]


def _pair(value, *, name: str, allow_zero: bool) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        result = (value[0], value[1])
    else:
        raise TypeError(f"{name} must be an int or a length-two sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise TypeError(f"{name} entries must be integers")
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in result):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} entries must be {qualifier}")
    return result


def _validate_fusion_inputs(
    scale_logits: Tensor,
    fusion_weight: Tensor,
    fusion_bias: Tensor | None,
) -> tuple[int, int, int, Tensor]:
    if not torch.is_tensor(scale_logits) or scale_logits.ndim != 4:
        raise ValueError("scale_logits must have shape [N,C,H,W]")
    if not scale_logits.is_floating_point():
        raise TypeError("scale_logits must be floating point")
    if not torch.isfinite(scale_logits).all():
        raise ValueError("scale_logits must contain only finite values")
    if not torch.is_tensor(fusion_weight) or fusion_weight.ndim != 4:
        raise ValueError("fusion_weight must have shape [1,C,3,3]")
    if tuple(fusion_weight.shape[2:]) != (3, 3):
        raise ValueError("fusion_weight must use a 3x3 kernel")
    if fusion_weight.shape[0] != 1:
        raise ValueError("fusion_weight must have exactly one output channel")
    if fusion_weight.shape[1] != scale_logits.shape[1]:
        raise ValueError("fusion input channel counts do not agree")
    if fusion_weight.device != scale_logits.device:
        raise ValueError("scale_logits and fusion_weight must share a device")
    if fusion_weight.dtype != scale_logits.dtype:
        raise ValueError("scale_logits and fusion_weight must share a dtype")
    if not torch.isfinite(fusion_weight).all():
        raise ValueError("fusion_weight must contain only finite values")

    if fusion_bias is None:
        bias = scale_logits.new_zeros(())
    else:
        if not torch.is_tensor(fusion_bias) or fusion_bias.numel() != 1:
            raise ValueError("fusion_bias must be None or contain one scalar")
        if fusion_bias.device != scale_logits.device:
            raise ValueError("scale_logits and fusion_bias must share a device")
        if fusion_bias.dtype != scale_logits.dtype:
            raise ValueError("scale_logits and fusion_bias must share a dtype")
        if not torch.isfinite(fusion_bias).all():
            raise ValueError("fusion_bias must be finite")
        bias = fusion_bias.reshape(())

    batch, channels, height, width = scale_logits.shape
    if batch < 1 or channels < 1 or height < 1 or width < 1:
        raise ValueError("scale_logits dimensions must be non-empty")
    if batch != 1:
        raise ValueError(
            "target-level final-fusion factorization requires batch size one"
        )
    return batch, height, width, bias


def _output_shape(
    input_shape: tuple[int, int],
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    result = tuple(
        math.floor(
            (size + 2 * pad - dil * (kernel - 1) - 1) / step + 1
        )
        for size, kernel, step, pad, dil in zip(
            input_shape, kernel_size, stride, padding, dilation
        )
    )
    if any(size < 1 for size in result):
        raise ValueError("fusion configuration produces an empty output")
    return result


def _normalized_spatial_weights(
    value: Tensor,
    *,
    name: str,
    batch: int,
    output_shape: tuple[int, int],
    reference: Tensor,
) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim == 2:
        if batch != 1:
            raise ValueError(f"2-D {name} is valid only for batch size one")
        value = value.unsqueeze(0)
    elif value.ndim == 4:
        if value.shape[1] != 1:
            raise ValueError(f"4-D {name} must have a singleton channel")
        value = value[:, 0]
    elif value.ndim != 3:
        raise ValueError(f"{name} must have shape [H,W], [N,H,W], or [N,1,H,W]")
    if tuple(value.shape) != (batch, *output_shape):
        raise ValueError(f"{name} must align with the fusion output")
    if value.device != reference.device:
        raise ValueError(f"{name} must share the input device")
    if value.dtype == torch.bool:
        value = value.to(dtype=reference.dtype)
    elif not value.is_floating_point():
        raise TypeError(f"{name} must be boolean or floating point")
    elif value.dtype != reference.dtype:
        value = value.to(dtype=reference.dtype)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if torch.any(value < 0):
        raise ValueError(f"{name} must be non-negative")
    flat = value.reshape(batch, -1)
    mass = flat.sum()
    if not bool(mass > 0):
        raise ValueError(f"{name} must have positive total mass")
    return flat / mass


def _canonical_patch_scale(
    patch_scale: Tensor,
    *,
    channels: int,
    reference: Tensor,
) -> Tensor:
    if not torch.is_tensor(patch_scale):
        raise TypeError("patch_scale must be a tensor")
    if patch_scale.device != reference.device:
        raise ValueError("patch_scale must share the input device")
    if not patch_scale.is_floating_point():
        raise TypeError("patch_scale must be floating point")
    if patch_scale.dtype != reference.dtype:
        patch_scale = patch_scale.to(dtype=reference.dtype)

    patch_features = channels * 9
    if tuple(patch_scale.shape) == (channels,):
        flat = patch_scale.repeat_interleave(9)
    elif tuple(patch_scale.shape) in ((channels, 3, 3), (1, channels, 3, 3)):
        flat = patch_scale.reshape(-1)
    elif tuple(patch_scale.shape) == (patch_features,):
        flat = patch_scale
    else:
        raise ValueError(
            "patch_scale must have shape [C], [C,3,3], [1,C,3,3], or [9C]"
        )
    if not torch.isfinite(flat).all() or torch.any(flat <= 0):
        raise ValueError("patch_scale entries must be finite and strictly positive")
    return flat


def factorize_final_fusion_margin(
    scale_logits: Tensor,
    target_weights: Tensor,
    control_weights: Tensor,
    *,
    fusion_weight: Tensor,
    fusion_bias: Tensor | None,
    patch_scale: Tensor,
    stride=1,
    padding=1,
    dilation=1,
    padding_mode: str = "zeros",
) -> FinalFusionMarginFactorization:
    """Factor a weighted target-control margin through a final ``3x3`` head.

    ``target_weights`` and ``control_weights`` are independently normalized
    over batch and output positions.  Consequently the returned
    ``signed_margin`` equals ``target_logit_mean - control_logit_mean`` and is
    independent of ``fusion_bias``.

    ``patch_scale`` is the strictly positive diagonal scale ``D`` used by the
    availability/utilization diagnostic.  It may contain one value per input
    channel (repeated over the nine taps) or one value per unfolded feature.
    For a channel reparameterization ``x_c' = g_c x_c``, exact invariance uses
    ``w_c' = w_c/g_c`` and ``D_c' = |g_c|D_c``.

    Only zero padding is accepted because it is the native ``nn.Conv2d``
    padding mode of MSHNet's final head and the mode implemented by
    :func:`torch.nn.functional.unfold`.
    """

    batch, input_height, input_width, bias = _validate_fusion_inputs(
        scale_logits, fusion_weight, fusion_bias
    )
    if padding_mode != "zeros":
        raise ValueError("only zero padding is supported")
    stride_pair = _pair(stride, name="stride", allow_zero=False)
    padding_pair = _pair(padding, name="padding", allow_zero=True)
    dilation_pair = _pair(dilation, name="dilation", allow_zero=False)
    output_shape = _output_shape(
        (input_height, input_width),
        kernel_size=(3, 3),
        stride=stride_pair,
        padding=padding_pair,
        dilation=dilation_pair,
    )

    patches = F.unfold(
        scale_logits,
        kernel_size=(3, 3),
        dilation=dilation_pair,
        padding=padding_pair,
        stride=stride_pair,
    )
    if patches.shape[-1] != output_shape[0] * output_shape[1]:
        raise RuntimeError("unfold output size disagrees with convolution geometry")
    flat_head = fusion_weight.reshape(-1)
    flat_scale = _canonical_patch_scale(
        patch_scale,
        channels=scale_logits.shape[1],
        reference=scale_logits,
    )

    flat_logits = torch.einsum("d,ndl->nl", flat_head, patches) + bias
    reconstructed_logits = flat_logits.reshape(batch, 1, *output_shape)
    target = _normalized_spatial_weights(
        target_weights,
        name="target_weights",
        batch=batch,
        output_shape=output_shape,
        reference=scale_logits,
    )
    control = _normalized_spatial_weights(
        control_weights,
        name="control_weights",
        batch=batch,
        output_shape=output_shape,
        reference=scale_logits,
    )

    target_patch_mean = torch.einsum("ndl,nl->d", patches, target)
    control_patch_mean = torch.einsum("ndl,nl->d", patches, control)
    difference = target_patch_mean - control_patch_mean
    whitened_difference = difference / flat_scale
    scaled_head = flat_head * flat_scale
    available = torch.linalg.vector_norm(whitened_difference)
    sensitivity = torch.linalg.vector_norm(scaled_head)
    signed_margin = torch.dot(flat_head, difference)
    target_logit_mean = torch.sum(flat_logits * target)
    control_logit_mean = torch.sum(flat_logits * control)

    derived_tensors = (
        reconstructed_logits,
        target_patch_mean,
        control_patch_mean,
        difference,
        whitened_difference,
        scaled_head,
        available,
        sensitivity,
        signed_margin,
        target_logit_mean,
        control_logit_mean,
    )
    if not all(bool(torch.isfinite(value).all()) for value in derived_tensors):
        raise RuntimeError("final-fusion factorization produced non-finite values")
    difference_zero = not bool(torch.any(whitened_difference != 0).detach())
    head_zero = not bool(torch.any(scaled_head != 0).detach())
    if bool((available == 0).detach()) != difference_zero:
        raise RuntimeError("available-contrast norm underflowed")
    if bool((sensitivity == 0).detach()) != head_zero:
        raise RuntimeError("head-sensitivity norm underflowed")

    direct_margin = target_logit_mean - control_logit_mean
    numeric_scale = max(
        1.0,
        float(torch.abs(target_logit_mean).detach().cpu()),
        float(torch.abs(control_logit_mean).detach().cpu()),
        float(torch.sum(torch.abs(flat_head * difference)).detach().cpu()),
    )
    # The two algebraically identical paths use reductions of different
    # lengths (spatial weighted means versus a 36-D dot product).  Bound their
    # expected floating-point disagreement using the active dtype rather than
    # an epsilon in the mathematical factorization itself.
    tolerance = max(
        1e-12,
        512.0 * torch.finfo(scale_logits.dtype).eps * numeric_scale,
    )
    if not torch.allclose(
        signed_margin,
        direct_margin,
        atol=tolerance,
        rtol=0.0,
    ):
        raise RuntimeError("fusion margin disagrees with reconstructed logits")

    if difference_zero or head_zero:
        utilization = None
        reconstructed_margin = None
        if not torch.allclose(
            signed_margin,
            torch.zeros_like(signed_margin),
            atol=tolerance,
            rtol=0.0,
        ):
            raise RuntimeError("undefined fusion factor has a non-zero margin")
    else:
        utilization = torch.dot(
            whitened_difference, scaled_head
        ) / (available * sensitivity)
        if not bool(torch.isfinite(utilization)):
            raise RuntimeError("fusion utilization is non-finite")
        utilization_value = float(utilization.detach().cpu())
        if utilization_value < -1.0 - 1e-6 or utilization_value > 1.0 + 1e-6:
            raise RuntimeError("fusion utilization escaped its valid range")
        utilization = torch.clamp(utilization, -1.0, 1.0)
        reconstructed_margin = available * sensitivity * utilization
        if not torch.allclose(
            reconstructed_margin,
            signed_margin,
            atol=tolerance,
            rtol=0.0,
        ):
            raise RuntimeError("fusion margin factorization lost exactness")

    return FinalFusionMarginFactorization(
        reconstructed_logits=reconstructed_logits,
        target_patch_mean=target_patch_mean,
        control_patch_mean=control_patch_mean,
        patch_difference=difference,
        patch_scale=flat_scale,
        whitened_difference=whitened_difference,
        scaled_head=scaled_head,
        available_contrast=available,
        head_sensitivity=sensitivity,
        utilization_cosine=utilization,
        signed_margin=signed_margin,
        reconstructed_margin=reconstructed_margin,
        target_logit_mean=target_logit_mean,
        control_logit_mean=control_logit_mean,
        fusion_bias=bias,
        output_shape=output_shape,
    )


__all__ = [
    "FinalFusionMarginFactorization",
    "factorize_final_fusion_margin",
]
