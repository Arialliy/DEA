"""Auditable building blocks for the revised OMM-Flow proposal.

This module deliberately contains only the parts whose optimization semantics
are closed and testable without a differentiable min-cost-flow solver:

* ``omm2d_identity_risk`` is the zero-displacement, single-scale restriction
  of the proposed instance-balanced partial-attribution risk.
* ``experimental_categorical_odds_fusion`` is an experimental replacement for MSHNet's
  signed additive logit fusion.  It gives every scale a non-negative share of
  the final foreground odds, so its scale mass is an actual decomposition of
  the prediction rather than ``p * softmax(signed_contribution)`` applied
  after the prediction has already been formed.

Neither function implements the full spatial transport solver.  In
particular, an optimal transport plan must not be detached and treated as a
constant during training: the supply constraints depend on the prediction,
so doing that omits the LP dual contribution to the gradient.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from skimage import measure
from torch import Tensor
from torch.nn import functional as F


def _validate_single_channel_pair(
    prediction: Tensor,
    target: Tensor,
) -> None:
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(
            "prediction must have shape [B, 1, H, W], got %s"
            % (tuple(prediction.shape),)
        )
    if target.shape != prediction.shape:
        raise ValueError(
            "target must have the same shape as prediction, got %s and %s"
            % (tuple(target.shape), tuple(prediction.shape))
        )
    if not prediction.is_floating_point():
        raise TypeError("prediction must be floating point")


@torch.no_grad()
def label_target_components(
    target: Tensor,
    *,
    threshold: float = 0.5,
    connectivity: int = 2,
) -> Tensor:
    """Label GT connected components after all geometric transforms.

    Component IDs are local to each image and start at one; zero denotes
    background.  This operation is intentionally non-differentiable because
    it acts only on the annotation.  ``connectivity=2`` matches the repository
    PD/FA implementation.
    """

    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(
            "target must have shape [B, 1, H, W], got %s"
            % (tuple(target.shape),)
        )
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2")

    target_cpu = target.detach().cpu().numpy()
    label_maps = []
    for batch_index in range(target.shape[0]):
        labels = measure.label(
            target_cpu[batch_index, 0] > float(threshold),
            connectivity=connectivity,
        )
        label_maps.append(torch.from_numpy(labels).unsqueeze(0))
    return torch.stack(label_maps, dim=0).to(
        device=target.device,
        dtype=torch.long,
    )


def _validate_instance_labels(
    instance_labels: Tensor,
    target_mask: Tensor,
) -> None:
    if instance_labels.shape != target_mask.shape:
        raise ValueError(
            "instance_labels must have shape %s, got %s"
            % (tuple(target_mask.shape), tuple(instance_labels.shape))
        )
    if instance_labels.dtype == torch.bool or instance_labels.is_floating_point():
        raise TypeError("instance_labels must use an integer dtype")
    if bool((instance_labels < 0).any()):
        raise ValueError("instance_labels must be non-negative")
    labelled_foreground = instance_labels > 0
    if not torch.equal(labelled_foreground, target_mask):
        raise ValueError(
            "positive instance labels must coincide exactly with target foreground"
        )


def _prepare_instance_labels(
    target: Tensor,
    target_mask: Tensor,
    instance_labels: Tensor | None,
    *,
    target_threshold: float,
    validate_instance_labels: bool,
) -> Tensor:
    if instance_labels is None:
        instance_labels = label_target_components(
            target,
            threshold=target_threshold,
            connectivity=2,
        )
    else:
        instance_labels = instance_labels.to(device=target.device)
    if validate_instance_labels:
        _validate_instance_labels(instance_labels, target_mask)
    return instance_labels


def _reduce_foreground_by_instance(
    foreground_penalty: Tensor,
    target_mask: Tensor,
    instance_labels: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Average differentiable foreground penalties per GT component.

    Instance IDs are local to each image and need not be contiguous.  Pairing
    each ID with its batch index prevents equal local IDs in different images
    from being merged.  ``scatter_add_`` keeps the penalty path differentiable;
    weighted ``torch.bincount`` does not provide the required backward in the
    repository training environment.
    """

    foreground = target_mask[:, 0]
    batch_grid = torch.arange(
        target_mask.shape[0],
        device=target_mask.device,
        dtype=torch.long,
    ).view(-1, 1, 1).expand_as(foreground)
    pair_keys = torch.stack(
        [batch_grid[foreground], instance_labels[:, 0][foreground]],
        dim=1,
    )
    unique_pairs, inverse = torch.unique(
        pair_keys,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    pixel_penalty = foreground_penalty[:, 0][foreground]
    num_instances = unique_pairs.shape[0]
    penalty_sums = pixel_penalty.new_zeros((num_instances,))
    penalty_sums.scatter_add_(0, inverse, pixel_penalty)
    areas = pixel_penalty.new_zeros((num_instances,))
    areas.scatter_add_(0, inverse, torch.ones_like(pixel_penalty))
    per_instance = penalty_sums / areas.clamp_min(1.0)
    return (
        per_instance,
        areas.to(dtype=torch.long),
        unique_pairs[:, 0],
        unique_pairs[:, 1],
    )


def _instance_balanced_risk(
    logits: Tensor,
    target: Tensor,
    *,
    background_penalty: Tensor,
    foreground_penalty: Tensor,
    probability: Tensor,
    instance_labels: Tensor | None,
    target_threshold: float,
    validate_instance_labels: bool,
) -> dict[str, Any]:
    _validate_single_channel_pair(logits, target)
    target_mask = target > float(target_threshold)
    instance_labels = _prepare_instance_labels(
        target,
        target_mask,
        instance_labels,
        target_threshold=target_threshold,
        validate_instance_labels=validate_instance_labels,
    )

    background = (~target_mask).to(dtype=logits.dtype)
    false_alarm_numerator = (background_penalty * background).sum()
    num_pixels = logits.numel()
    false_alarm_risk = false_alarm_numerator / float(num_pixels)

    (
        per_instance_miss,
        instance_areas,
        instance_batch_indices,
        instance_ids,
    ) = _reduce_foreground_by_instance(
        foreground_penalty,
        target_mask,
        instance_labels,
    )
    if per_instance_miss.numel():
        miss_numerator = per_instance_miss.sum()
        miss_risk = miss_numerator / float(per_instance_miss.numel())
    else:
        miss_numerator = logits.sum() * 0.0
        miss_risk = miss_numerator

    loss = false_alarm_risk + miss_risk
    return {
        "loss": loss,
        "false_alarm_risk": false_alarm_risk,
        "miss_risk": miss_risk,
        "false_alarm_numerator": false_alarm_numerator,
        "miss_numerator": miss_numerator,
        "num_pixels": num_pixels,
        "num_instances": int(per_instance_miss.numel()),
        "num_images": logits.shape[0],
        "num_empty_images": int(
            (target_mask.flatten(1).sum(dim=1) == 0).sum().item()
        ),
        "probability": probability,
        "instance_labels": instance_labels,
        "per_instance_miss": per_instance_miss,
        "instance_areas": instance_areas,
        "instance_batch_indices": instance_batch_indices,
        "instance_ids": instance_ids,
    }


def omm2d_identity_risk(
    logits: Tensor,
    target: Tensor,
    *,
    instance_labels: Tensor | None = None,
    target_threshold: float = 0.5,
    validate_instance_labels: bool = True,
) -> dict[str, Any]:
    """Compute the closed-form single-scale, zero-displacement OMM control.

    The risk is

    ``R_FA + R_miss``, where ``R_FA`` is foreground probability on GT
    background divided by all batch pixels and ``R_miss`` is the mean, over
    all GT instances in the batch, of that instance's mean missing
    probability.  Thus every image contributes equal area to ``R_FA`` and
    every target contributes equal mass to ``R_miss``.  This is an explicit
    equal-unit scalarization, not an exact differentiable form of the
    repository's connected-component PD/FA metric.

    When a batch contains no target, ``R_miss`` is an autograd-connected zero
    and the loss reduces exactly to the mean foreground probability.
    """

    probability = torch.sigmoid(logits)
    return _instance_balanced_risk(
        logits,
        target,
        background_penalty=probability,
        foreground_penalty=1.0 - probability,
        probability=probability,
        instance_labels=instance_labels,
        target_threshold=target_threshold,
        validate_instance_labels=validate_instance_labels,
    )


def instance_balanced_logistic_risk(
    logits: Tensor,
    target: Tensor,
    *,
    instance_labels: Tensor | None = None,
    target_threshold: float = 0.5,
    validate_instance_labels: bool = True,
) -> dict[str, Any]:
    """Strong proper-composite control with the same instance reductions.

    Background pixels use ``softplus(z)`` and each GT instance contributes the
    mean ``softplus(-z)`` over its pixels.  The reductions therefore match
    ``omm2d_identity_risk`` exactly, while high-confidence wrong logits retain
    order-one gradients.  This is a required reweighting/properness control,
    not an OMM contribution.
    """

    probability = torch.sigmoid(logits)
    return _instance_balanced_risk(
        logits,
        target,
        background_penalty=F.softplus(logits),
        foreground_penalty=F.softplus(-logits),
        probability=probability,
        instance_labels=instance_labels,
        target_threshold=target_threshold,
        validate_instance_labels=validate_instance_labels,
    )


def _broadcast_fusion_bias(contributions: Tensor, bias: Tensor | None) -> Tensor:
    if bias is None:
        return contributions.new_zeros((1, 1, 1, 1))
    if not torch.is_tensor(bias):
        raise TypeError("bias must be a Tensor or None")
    if bias.numel() == 1:
        return bias.reshape(1, 1, 1, 1)
    expected = (contributions.shape[0], 1, *contributions.shape[-2:])
    if tuple(bias.shape) != expected:
        raise ValueError(
            "non-scalar bias must have shape %s, got %s"
            % (expected, tuple(bias.shape))
        )
    return bias


def experimental_categorical_odds_fusion(
    contributions: Tensor,
    *,
    bias: Tensor | None = None,
) -> dict[str, Tensor]:
    """Fuse scale evidence as additive non-negative foreground odds.

    For ``S`` exact per-scale convolutional contributions ``e_s``, define

    ``h_s = S * e_s + b - log(S)`` and ``o_s = exp(h_s)``.

    The final logit is ``log(sum_s o_s)`` and each scale owns the odds fraction
    ``softmax(h)_s``.  Multiplying that fraction by the final probability gives
    a non-negative source measure that sums exactly to the prediction.

    The factor ``S`` is fixed, not tuned: when every contribution is equal,
    this fusion equals MSHNet's linear logit fusion exactly, and its derivative
    with respect to every contribution is one at that symmetric point.
    Unlike the rejected post-hoc lift, a zero-sum change of unequal signed
    contributions changes the fused prediction and therefore cannot hide an
    arbitrary scale code behind an unchanged final logit.

    This remains an experimental *replacement* of MSHNet's final fusion, not a
    read-only attribution of its canonical output.
    """

    if contributions.ndim != 4 or contributions.shape[1] < 1:
        raise ValueError(
            "contributions must have shape [B, S, H, W], got %s"
            % (tuple(contributions.shape),)
        )
    if not contributions.is_floating_point():
        raise TypeError("contributions must be floating point")

    num_scales = contributions.shape[1]
    fusion_bias = _broadcast_fusion_bias(contributions, bias)
    branch_log_odds = (
        float(num_scales) * contributions
        + fusion_bias
        - math.log(float(num_scales))
    )
    fused_logit = torch.logsumexp(branch_log_odds, dim=1, keepdim=True)
    # This is exactly an (S + 1)-class model: class zero is background and
    # classes 1..S are foreground at the corresponding scale.  Computing the
    # joint probabilities in one softmax makes conservation numerical, not
    # merely algebraic.
    background_logit = torch.zeros_like(branch_log_odds[:, :1])
    joint_probability = torch.softmax(
        torch.cat([background_logit, branch_log_odds], dim=1),
        dim=1,
    )
    background_probability = joint_probability[:, :1]
    source_mass = joint_probability[:, 1:]
    probability = source_mass.sum(dim=1, keepdim=True)
    scale_responsibility = torch.softmax(branch_log_odds, dim=1)
    linear_logit = contributions.sum(dim=1, keepdim=True) + fusion_bias

    return {
        "fused_logit": fused_logit,
        "linear_logit": linear_logit,
        "probability": probability,
        "background_probability": background_probability,
        "joint_probability": joint_probability,
        "branch_log_odds": branch_log_odds,
        "scale_responsibility": scale_responsibility,
        "source_mass": source_mass,
    }


__all__ = [
    "experimental_categorical_odds_fusion",
    "instance_balanced_logistic_risk",
    "label_target_components",
    "omm2d_identity_risk",
]
