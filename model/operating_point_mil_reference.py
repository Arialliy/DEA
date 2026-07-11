"""Exact output-only operating-point MIL reference.

This is a negative-control objective, not a proposed training method.  It is
the Chebyshev aggregation of per-instance max-pooling MIL existence and an
exact background order-statistic budget.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _validate_inputs(
    logits: torch.Tensor,
    instance_labels: torch.Tensor,
    safe_background_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must have shape [B,1,H,W]")
    if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    if not torch.is_tensor(instance_labels) or instance_labels.shape != logits.shape:
        raise ValueError("instance_labels must match logits shape")
    if instance_labels.dtype == torch.bool or torch.is_floating_point(instance_labels):
        raise ValueError("instance_labels must use a non-boolean integer dtype")
    if torch.any(instance_labels < 0):
        raise ValueError("instance_labels must be non-negative")
    if instance_labels.device != logits.device:
        raise ValueError("logits and instance_labels must share a device")

    target = instance_labels > 0
    if safe_background_mask is None:
        safe_background = ~target
    else:
        if (
            not torch.is_tensor(safe_background_mask)
            or safe_background_mask.shape != logits.shape
        ):
            raise ValueError("safe_background_mask must match logits shape")
        if safe_background_mask.device != logits.device:
            raise ValueError("safe background and logits must share a device")
        if safe_background_mask.dtype != torch.bool:
            if torch.is_floating_point(safe_background_mask):
                if not torch.isfinite(safe_background_mask).all():
                    raise ValueError("safe_background_mask must be finite")
            if not torch.logical_or(
                safe_background_mask == 0,
                safe_background_mask == 1,
            ).all():
                raise ValueError("safe_background_mask must be binary")
        safe_background = safe_background_mask.bool()
        if torch.any(safe_background & target):
            raise ValueError("safe background cannot include target pixels")
    return logits[:, 0], instance_labels[:, 0], safe_background[:, 0]


def exact_operating_point_mil_reference(
    logits: torch.Tensor,
    instance_labels: torch.Tensor,
    *,
    threshold: float = 0.0,
    positive_margin: float = 1.0,
    negative_margin: float = 1.0,
    allowed_background_exceedances: int = 0,
    safe_background_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    """Evaluate exact instance-existence and background-budget violations.

    For each image, the positive term is the largest hinge violation among
    instance bags.  The negative term is the hinge on the ``B+1``-th largest
    safe-background logit, where ``B`` background exceedances are exempt.
    Their outer maximum is the exact ``L_inf`` aggregation.
    """

    logit_maps, labels, safe_background = _validate_inputs(
        logits,
        instance_labels,
        safe_background_mask,
    )
    numeric = (threshold, positive_margin, negative_margin)
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError("threshold and margins must be finite")
    if positive_margin <= 0 or negative_margin < 0:
        raise ValueError(
            "positive_margin must be positive and negative_margin non-negative"
        )
    if (
        not isinstance(allowed_background_exceedances, int)
        or isinstance(allowed_background_exceedances, bool)
        or allowed_background_exceedances < 0
    ):
        raise ValueError("allowed background exceedances must be a non-negative integer")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    positive_violations = []
    background_violations = []
    target_floor = float(threshold) + float(positive_margin)
    background_ceiling = float(threshold) - float(negative_margin)
    for batch_index in range(logit_maps.shape[0]):
        image_logits = logit_maps[batch_index]
        image_labels = labels[batch_index]
        identifiers = torch.unique(image_labels)
        identifiers = identifiers[identifiers > 0]
        if identifiers.numel() == 0:
            positive_violation = image_logits.new_zeros(())
        else:
            instance_peaks = torch.stack(
                [
                    torch.max(image_logits[image_labels == identifier])
                    for identifier in identifiers
                ]
            )
            positive_violation = torch.max(
                F.relu(target_floor - instance_peaks)
            )
        positive_violations.append(positive_violation)

        background_logits = image_logits[safe_background[batch_index]]
        if (
            background_logits.numel() == 0
            or allowed_background_exceedances >= background_logits.numel()
        ):
            background_violation = image_logits.new_zeros(())
        else:
            descending = torch.sort(background_logits, descending=True).values
            order_statistic = descending[allowed_background_exceedances]
            background_violation = F.relu(
                order_statistic - background_ceiling
            )
        background_violations.append(background_violation)

    positive_tensor = torch.stack(positive_violations)
    background_tensor = torch.stack(background_violations)
    per_image = torch.maximum(positive_tensor, background_tensor)
    if reduction == "mean":
        loss = per_image.mean()
    elif reduction == "sum":
        loss = per_image.sum()
    else:
        loss = per_image
    return {
        "loss": loss,
        "per_image": per_image,
        "positive_violation": positive_tensor,
        "background_violation": background_tensor,
    }


__all__ = ["exact_operating_point_mil_reference"]
