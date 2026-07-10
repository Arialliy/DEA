"""Training-only objective for semantically identifiable DEA actions.

The model's three actions are interventions on a logit: increase, decrease, or
keep it unchanged.  A detached pre-closure prediction induces exactly the same
three-way correction distribution through the Bernoulli residual.  This file
contains no inference module and does not change the baseline-preserving
forward graph.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def residual_action_distribution(
    preclosure_logits: Tensor,
    target: Tensor,
) -> Tensor:
    """Return increase/decrease/keep action mass induced by BCE geometry.

    For ``p = sigmoid(stopgrad(z))`` and binary ``y``:

    ``q_increase = y * (1 - p)``
    ``q_decrease = (1 - y) * p``
    ``q_keep = y * p + (1 - y) * (1 - p)``

    Hence ``q_increase - q_decrease = y - p``, the negative BCE derivative
    with respect to ``z``.  The returned teacher is detached by construction.
    """
    if preclosure_logits.ndim != 4 or preclosure_logits.shape[1] != 1:
        raise ValueError("preclosure_logits must have shape [B, 1, H, W]")
    if target.shape != preclosure_logits.shape:
        raise ValueError(
            "target and preclosure_logits must have identical shapes, got %s and %s"
            % (tuple(target.shape), tuple(preclosure_logits.shape))
        )

    with torch.no_grad():
        y = (target > 0.5).to(preclosure_logits.dtype)
        probability = torch.sigmoid(preclosure_logits.detach())
        increase = y * (1.0 - probability)
        decrease = (1.0 - y) * probability
        keep = y * probability + (1.0 - y) * (1.0 - probability)
        teacher = torch.cat([increase, decrease, keep], dim=1)
    return teacher


def _region_mean(value: Tensor, region: Tensor, eps: float) -> Tuple[Tensor, bool]:
    mass = region.sum()
    if float(mass.detach().item()) <= eps:
        return value.sum() * 0.0, False
    return (value * region).sum() / mass.clamp_min(eps), True


def residual_aligned_route_loss(
    integrated_output: Dict[str, object],
    target: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Supervise all reused routes with one pre-closure correction field.

    Foreground and background regions are averaged separately.  This prevents
    SIRST's background pixel count from erasing the increase action while
    preserving the teacher's per-pixel correction magnitude.  In particular,
    action classes are *not* independently mass-normalized: a vanishing error
    must be allowed to converge to the keep/abstain action.
    """
    if "routes" not in integrated_output or "scale_fusion" not in integrated_output:
        raise KeyError("Integrated output must contain routes and scale_fusion")
    scale_fusion = integrated_output["scale_fusion"]
    if not isinstance(scale_fusion, dict) or "z_base" not in scale_fusion:
        raise KeyError("scale_fusion must contain the pre-closure z_base logit")

    preclosure_logits = scale_fusion["z_base"]
    teacher = residual_action_distribution(preclosure_logits, target)
    binary_target = (target > 0.5).to(preclosure_logits.dtype)
    background = 1.0 - binary_target

    scale_losses = []
    log: Dict[str, Tensor] = {}
    for scale, route in enumerate(integrated_output["routes"]):
        probabilities = route["probabilities"]
        probabilities = F.interpolate(
            probabilities,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp_min(eps)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        cross_entropy = -(teacher * probabilities.log()).sum(dim=1, keepdim=True)

        foreground_loss, has_foreground = _region_mean(
            cross_entropy, binary_target, eps
        )
        background_loss, has_background = _region_mean(
            cross_entropy, background, eps
        )
        active_losses = []
        if has_foreground:
            active_losses.append(foreground_loss)
        if has_background:
            active_losses.append(background_loss)
        if not active_losses:
            scale_loss = cross_entropy.sum() * 0.0
        else:
            scale_loss = torch.stack(active_losses).mean()
        scale_losses.append(scale_loss)
        log["route_loss_scale_%d" % scale] = scale_loss.detach()

        with torch.no_grad():
            base_probability = torch.sigmoid(preclosure_logits.detach())
            false_negative = (binary_target > 0.5) & (base_probability < 0.5)
            false_positive = (background > 0.5) & (base_probability >= 0.5)
            correct = ~(false_negative | false_positive)
            conditions = (
                ("target_prob_on_fn_%d" % scale, probabilities[:, 0:1], false_negative),
                ("clutter_prob_on_fp_%d" % scale, probabilities[:, 1:2], false_positive),
                ("keep_prob_on_correct_%d" % scale, probabilities[:, 2:3], correct),
            )
            for name, probability_map, condition in conditions:
                count = condition.sum()
                if int(count.item()) == 0:
                    log[name] = probability_map.new_tensor(float("nan"))
                else:
                    log[name] = probability_map[condition].mean()

    total_loss = torch.stack(scale_losses).mean()
    with torch.no_grad():
        teacher_mass = teacher.mean(dim=(0, 2, 3))
        log["teacher_target_mass"] = teacher_mass[0]
        log["teacher_clutter_mass"] = teacher_mass[1]
        log["teacher_keep_mass"] = teacher_mass[2]
        log["route_loss_raw"] = total_loss.detach()
    return total_loss, log


__all__ = [
    "residual_action_distribution",
    "residual_aligned_route_loss",
]
