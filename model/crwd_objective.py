"""Counterfactual Residue Witness Distillation (CRWD).

CRWD is a training-only objective for an unchanged dense predictor.  A
stride-preserving control view and its one-pixel residue-changing neighbour
are aligned to canonical coordinates and used as detached counterfactual
teachers.  A teacher pair receives credit only when the residue view improves
the target component itself, improves its target-vs-tail margin, does not
inflate the background tail, and already survives that tail.  The canonical
student is then optimized against the single worst violation of the witness
feasible set; no pixel-wise consistency is imposed.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_COMPONENT_BUDGETS: tuple[int, ...] = (1, 5, 10, 20)


class CRWDError(ValueError):
    """Raised when the CRWD tensor or hyperparameter contract is invalid."""


def _finite(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CRWDError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise CRWDError(f"{name} must be finite")
    return result


def _positive_odd(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2 == 0:
        raise CRWDError(f"{name} must be a positive odd integer")
    return int(value)


def _budgets(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or tuple(sorted(result)) != result
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in result)
    ):
        raise CRWDError("budgets must be unique positive integers in increasing order")
    return result


def _smooth_max(values: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise CRWDError("smooth maximum requires a non-empty vector")
    maximum = values.max()
    return maximum + temperature * (
        torch.logsumexp((values - maximum) / temperature, dim=0)
        - math.log(values.numel())
    )


def _smooth_min(values: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    return -_smooth_max(-values, temperature)


def _robust_scale(values: torch.Tensor) -> torch.Tensor:
    detached = values.detach().float()
    median = detached.median()
    mad = (detached - median).abs().median() * 1.4826
    return mad.clamp_min(1.0)


def _normalize_validity(
    value: torch.Tensor,
    *,
    batch: int,
    height: int,
    width: int,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(value) or value.dtype != torch.bool:
        raise CRWDError(f"{name} must be a boolean tensor")
    if tuple(value.shape) == (1, 1, height, width):
        return value.expand(batch, -1, -1, -1)
    if tuple(value.shape) == (batch, 1, height, width):
        return value
    raise CRWDError(f"{name} must be [1,1,H,W] or [B,1,H,W]")


def counterfactual_residue_witness_loss(
    canonical_logits: torch.Tensor,
    control_logits: torch.Tensor,
    residue_logits: torch.Tensor,
    control_validity: torch.Tensor,
    residue_validity: torch.Tensor,
    target: torch.Tensor,
    instance_labels: torch.Tensor,
    *,
    budgets: Sequence[int] = DEFAULT_COMPONENT_BUDGETS,
    protect_kernel: int = 7,
    target_temperature: float = 0.25,
    tail_temperature: float = 0.25,
    delta_target: float = 0.05,
    delta_margin: float = 0.05,
    tail_tolerance: float = 0.05,
    margin_floor: float = 0.0,
    max_margin_credit: float = 1.0,
    confidence_width: float = 0.25,
    huber_delta: float = 0.25,
    validate_instance_labels: bool = True,
) -> dict[str, Any]:
    """Project canonical component events into detached witness feasible sets.

    All logits use ``[B,1,H,W]``.  ``control_logits`` and ``residue_logits``
    are detached internally even if a caller accidentally supplies tensors
    requiring gradients.  Candidate background coordinates are selected once
    from ``max(control, residue)`` and shared by all three views.
    """

    budget_values = _budgets(budgets)
    protect_kernel_value = _positive_odd(protect_kernel, name="protect_kernel")
    target_temperature_value = _finite(target_temperature, name="target_temperature")
    tail_temperature_value = _finite(tail_temperature, name="tail_temperature")
    delta_target_value = _finite(delta_target, name="delta_target")
    delta_margin_value = _finite(delta_margin, name="delta_margin")
    tail_tolerance_value = _finite(tail_tolerance, name="tail_tolerance")
    margin_floor_value = _finite(margin_floor, name="margin_floor")
    max_margin_credit_value = _finite(max_margin_credit, name="max_margin_credit")
    confidence_width_value = _finite(confidence_width, name="confidence_width")
    huber_delta_value = _finite(huber_delta, name="huber_delta")
    if target_temperature_value <= 0.0 or tail_temperature_value <= 0.0:
        raise CRWDError("temperatures must be positive")
    if any(
        value < 0.0
        for value in (
            delta_target_value,
            delta_margin_value,
            tail_tolerance_value,
        )
    ):
        raise CRWDError("witness thresholds must be non-negative")
    if max_margin_credit_value <= 0.0:
        raise CRWDError("max_margin_credit must be positive")
    if confidence_width_value <= 0.0 or huber_delta_value <= 0.0:
        raise CRWDError("confidence_width and huber_delta must be positive")
    if not isinstance(validate_instance_labels, bool):
        raise CRWDError("validate_instance_labels must be a bool")

    tensors = {
        "canonical_logits": canonical_logits,
        "control_logits": control_logits,
        "residue_logits": residue_logits,
    }
    if not torch.is_tensor(canonical_logits) or canonical_logits.ndim != 4:
        raise CRWDError("canonical_logits must be a [B,1,H,W] tensor")
    batch, channels, height, width = canonical_logits.shape
    expected = (batch, 1, height, width)
    if batch < 1 or channels != 1 or height < 2 or width < 2:
        raise CRWDError("canonical_logits has an invalid shape")
    for name, value in tensors.items():
        if not torch.is_tensor(value) or tuple(value.shape) != expected:
            raise CRWDError(f"{name} shape differs from canonical_logits")
        if value.device != canonical_logits.device or not bool(torch.isfinite(value).all()):
            raise CRWDError(f"{name} must be finite and share the canonical device")
    if not torch.is_tensor(target) or tuple(target.shape) != expected:
        raise CRWDError("target shape differs from canonical_logits")
    if target.device != canonical_logits.device or not bool(torch.isfinite(target).all()):
        raise CRWDError("target must be finite and share the canonical device")
    if not torch.is_tensor(instance_labels) or tuple(instance_labels.shape) != expected:
        raise CRWDError("instance_labels shape differs from canonical_logits")
    if instance_labels.device != canonical_logits.device:
        raise CRWDError("instance_labels must share the canonical device")
    if instance_labels.dtype == torch.bool or instance_labels.is_floating_point():
        raise CRWDError("instance_labels must use an integer dtype")

    control_valid = _normalize_validity(
        control_validity,
        batch=batch,
        height=height,
        width=width,
        name="control_validity",
    )
    residue_valid = _normalize_validity(
        residue_validity,
        batch=batch,
        height=height,
        width=width,
        name="residue_validity",
    )
    if control_valid.device != canonical_logits.device or residue_valid.device != canonical_logits.device:
        raise CRWDError("validity masks must share the logits device")
    common_valid = torch.logical_and(control_valid, residue_valid)

    truth = target > 0.5
    identities = instance_labels.to(dtype=torch.long)
    if bool((identities < 0).any()):
        raise CRWDError("instance_labels cannot be negative")
    if validate_instance_labels and not torch.equal(identities > 0, truth):
        raise CRWDError("instance_labels foreground differs from target")

    protected = F.max_pool2d(
        truth.float(),
        kernel_size=protect_kernel_value,
        stride=1,
        padding=protect_kernel_value // 2,
    ) > 0.0
    safe_background = torch.logical_and(common_valid, ~protected)
    control_teacher = control_logits.detach().float()
    residue_teacher = residue_logits.detach().float()
    worst_teacher = torch.maximum(control_teacher, residue_teacher)
    candidate_mask = safe_background
    worst_candidates = worst_teacher[candidate_mask]
    zero = canonical_logits.sum() * 0.0
    if worst_candidates.numel() == 0:
        return {
            "loss": zero,
            "witness_components": 0,
            "witness_events": 0,
            "eligible_components": 0,
            "skipped_invalid_components": 0,
            "candidate_count": 0,
            "mean_violation": zero.detach(),
            "mean_weight": zero.detach(),
            "budgets": budget_values,
        }

    valid_pixels = int(common_valid.sum().item())
    candidate_sizes = {
        budget: int(math.floor(budget * valid_pixels / 1_000_000.0)) + 1
        for budget in budget_values
    }
    maximum_candidates = max(candidate_sizes.values())
    if maximum_candidates > int(worst_candidates.numel()):
        raise CRWDError("safe background is too small for the requested budgets")
    shared_indices = torch.topk(
        worst_candidates,
        k=maximum_candidates,
        largest=True,
        sorted=True,
    ).indices.detach()
    student_candidates = canonical_logits.float()[candidate_mask]
    control_candidates = control_teacher[candidate_mask]
    residue_candidates = residue_teacher[candidate_mask]
    scale = _robust_scale(worst_candidates)
    tail_temperature_tensor = scale * tail_temperature_value

    tails: dict[int, dict[str, torch.Tensor]] = {}
    for budget in budget_values:
        count = candidate_sizes[budget]
        indices = shared_indices[:count]
        tails[budget] = {
            "student": _smooth_min(student_candidates[indices], tail_temperature_tensor),
            "control": _smooth_min(control_candidates[indices], tail_temperature_tensor),
            "residue": _smooth_min(residue_candidates[indices], tail_temperature_tensor),
        }

    component_losses: list[torch.Tensor] = []
    component_weights: list[torch.Tensor] = []
    violations: list[torch.Tensor] = []
    eligible_components = 0
    skipped_invalid_components = 0
    witness_events = 0
    target_temperature_tensor = scale * target_temperature_value

    for batch_index in range(batch):
        component_ids = [
            int(value)
            for value in torch.unique(identities[batch_index, 0]).tolist()
            if int(value) > 0
        ]
        for component_id in component_ids:
            component = identities[batch_index, 0] == component_id
            if not bool(common_valid[batch_index, 0][component].all()):
                skipped_invalid_components += 1
                continue
            eligible_components += 1
            student_target = _smooth_max(
                canonical_logits[batch_index, 0][component].float(),
                target_temperature_tensor,
            )
            control_target = _smooth_max(
                control_teacher[batch_index, 0][component],
                target_temperature_tensor,
            )
            residue_target = _smooth_max(
                residue_teacher[batch_index, 0][component],
                target_temperature_tensor,
            )
            event_losses: list[torch.Tensor] = []
            event_weights: list[torch.Tensor] = []
            for budget in budget_values:
                student_tail = tails[budget]["student"]
                control_tail = tails[budget]["control"]
                residue_tail = tails[budget]["residue"]
                student_margin = (student_target - student_tail) / scale
                control_margin = (control_target - control_tail) / scale
                residue_margin = (residue_target - residue_tail) / scale
                target_gain = ((residue_target - control_target) / scale).detach()
                margin_gain = (residue_margin - control_margin).detach()
                tail_inflation = ((residue_tail - control_tail) / scale).detach()
                active = (
                    float(target_gain) > delta_target_value
                    and float(margin_gain) > delta_margin_value
                    and float(tail_inflation) <= tail_tolerance_value
                    and float(residue_margin.detach()) >= margin_floor_value
                )
                if not active:
                    continue
                weight = torch.clamp(
                    (margin_gain - delta_margin_value) / confidence_width_value,
                    min=0.0,
                    max=1.0,
                ).detach()
                margin_star = torch.clamp(
                    residue_margin.detach(),
                    min=margin_floor_value,
                    max=margin_floor_value + max_margin_credit_value,
                )
                common_center = control_tail.detach()
                student_tail_normalized = (student_tail - common_center) / scale
                tail_star = (
                    torch.minimum(residue_tail, control_tail).detach() - common_center
                ) / scale
                margin_violation = margin_star - student_margin
                tail_violation = student_tail_normalized - tail_star
                violation = torch.relu(torch.maximum(margin_violation, tail_violation))
                event_loss = F.smooth_l1_loss(
                    violation,
                    torch.zeros_like(violation),
                    beta=huber_delta_value,
                    reduction="none",
                )
                event_losses.append(event_loss)
                event_weights.append(weight)
                violations.append(violation.detach())
                witness_events += 1
            if event_losses:
                weights = torch.stack(event_weights)
                losses = torch.stack(event_losses)
                component_losses.append((weights * losses).sum() / weights.sum().clamp_min(1e-8))
                component_weights.append(weights.max())

    if component_losses:
        loss = torch.stack(component_losses).mean()
        mean_weight = torch.stack(component_weights).mean().detach()
    else:
        loss = zero
        mean_weight = zero.detach()
    return {
        "loss": loss,
        "witness_components": len(component_losses),
        "witness_events": witness_events,
        "eligible_components": eligible_components,
        "skipped_invalid_components": skipped_invalid_components,
        "candidate_count": int(maximum_candidates),
        "mean_violation": (
            torch.stack(violations).mean() if violations else zero.detach()
        ),
        "mean_weight": mean_weight,
        "budgets": budget_values,
    }


__all__ = [
    "CRWDError",
    "DEFAULT_COMPONENT_BUDGETS",
    "counterfactual_residue_witness_loss",
]
