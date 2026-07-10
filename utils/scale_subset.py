"""Exact scale-subset interventions for MSHNet fusion contributions."""

from __future__ import annotations

import torch


def kept_scale_indices(subset: int, scale_count: int = 4) -> tuple[int, ...]:
    if scale_count <= 0:
        raise ValueError("scale_count must be positive")
    if subset < 0 or subset >= (1 << scale_count):
        raise ValueError("subset is outside the scale bitmask range")
    return tuple(index for index in range(scale_count) if subset & (1 << index))


def reconstruct_scale_subset(
    contributions: torch.Tensor,
    fusion_bias: torch.Tensor,
    subset: int,
    *,
    z_base: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the logit after retaining exactly the selected scale branches.

    ``contributions`` must be bias-free.  When all scales are retained and a
    direct ``z_base`` is supplied, that tensor is returned verbatim so the
    baseline operating point is not changed by floating-point reduction order.
    """

    if contributions.ndim != 4:
        raise ValueError("contributions must have shape [B,S,H,W]")
    scale_count = int(contributions.shape[1])
    selected = kept_scale_indices(subset, scale_count)
    full_subset = (1 << scale_count) - 1
    if subset == full_subset and z_base is not None:
        if z_base.shape != contributions[:, :1].shape:
            raise ValueError("z_base shape must match a single contribution")
        return z_base

    if not torch.is_tensor(fusion_bias):
        raise TypeError("fusion_bias must be a tensor")
    if selected:
        retained = contributions[:, selected].sum(dim=1, keepdim=True)
    else:
        retained = torch.zeros_like(contributions[:, :1])
    return retained + fusion_bias


__all__ = ["kept_scale_indices", "reconstruct_scale_subset"]
