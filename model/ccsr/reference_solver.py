"""Exhaustive corrected CCSR hinge on tiny pixel-edit state spaces.

This module is a semantic gold standard, not a production solver.  Matching
is minimized inside each repaired mask.  It is never exposed as an adversarial
outer state, which would make perfect predictions admit positive zero-gradient
losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .pixel_edit_reference import (
    PixelEditConfig,
    PixelEditState,
    enumerate_pixel_edit_states,
    reconstruct_edited_logits,
)
from .task_risk import (
    ComponentEditRiskResult,
    build_components_from_binary,
    build_gt_components,
    exact_component_edit_risk,
)


@dataclass(frozen=True)
class StructuredCandidate:
    pixel_edit: PixelEditState
    inner_risk: ComponentEditRiskResult
    score: float


@dataclass(frozen=True)
class StructuredHingeResult:
    decoded: StructuredCandidate
    oracle: StructuredCandidate
    loss_augmented: StructuredCandidate
    oracle_risk: float
    hinge: float
    excess_risk: float
    upper_bound_slack: float
    num_states: int


def _binary_2d(value, *, name: str, shape=None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("%s must be a non-empty 2-D array" % name)
    if shape is not None and array.shape != shape:
        raise ValueError("%s must match the logits shape" % name)
    if not np.isfinite(array).all():
        raise ValueError("%s must contain only finite values" % name)
    if not np.logical_or(array == 0, array == 1).all():
        raise ValueError("%s must be binary" % name)
    return array.astype(bool, copy=False)


def _max_by_value_then_mask(candidates, value_fn):
    # Smaller mask_bits wins exact ties.  No target-risk field is allowed in
    # the decoded tie-break, so decoding remains target-independent.
    return max(
        candidates,
        key=lambda candidate: (
            float(value_fn(candidate)),
            -candidate.pixel_edit.mask_bits,
        ),
    )


def solve_exhaustive_structured_hinge(
    logits_2d,
    target_2d,
    instance_labels_2d,
    *,
    pixel_config: PixelEditConfig,
    centroid_radius: float = 3.0,
    max_structured_states: int = 4096,
    numeric_tolerance: float = 1e-12,
) -> StructuredHingeResult:
    """Solve the corrected outer-repair/inner-matching hinge exactly."""
    pixel_config.validate()
    if pixel_config.activation_semantics != "margin":
        raise ValueError(
            "structured reference requires realizable margin activation semantics"
        )
    if (
        not isinstance(max_structured_states, int)
        or isinstance(max_structured_states, bool)
        or max_structured_states <= 0
    ):
        raise ValueError("max_structured_states must be a positive integer")
    if not np.isfinite(numeric_tolerance) or numeric_tolerance < 0:
        raise ValueError("numeric_tolerance must be finite and non-negative")

    if torch.is_tensor(logits_2d):
        logits_array = logits_2d.detach().cpu().numpy()
    else:
        logits_array = np.asarray(logits_2d)
    if logits_array.ndim != 2 or 0 in logits_array.shape:
        raise ValueError("logits_2d must be a non-empty 2-D array")
    if not np.isfinite(logits_array).all():
        raise ValueError("logits_2d must contain only finite values")
    logits_array = logits_array.astype(np.float64, copy=False)
    target = _binary_2d(target_2d, name="target_2d", shape=logits_array.shape)
    labels = np.asarray(
        instance_labels_2d.detach().cpu().numpy()
        if torch.is_tensor(instance_labels_2d)
        else instance_labels_2d
    )
    if labels.shape != logits_array.shape:
        raise ValueError("instance_labels_2d must match the logits shape")
    gt_components = build_gt_components(
        target,
        labels,
        connectivity=pixel_config.connectivity,
    )

    num_states = 1 << logits_array.size
    if num_states > max_structured_states:
        raise ValueError(
            "structured reference would enumerate %d states, limit is %d"
            % (num_states, max_structured_states)
        )

    candidates = []
    for pixel_state in enumerate_pixel_edit_states(
        logits_array,
        config=pixel_config,
    ):
        reconstructed = reconstruct_edited_logits(
            logits_array,
            pixel_state,
        )
        reconstructed_mask = reconstructed > pixel_config.threshold_logit
        if not np.array_equal(reconstructed_mask, pixel_state.mask()):
            raise RuntimeError("pixel state reconstruction changed its mask")
        pred_components = build_components_from_binary(
            reconstructed_mask,
            connectivity=pixel_config.connectivity,
        )
        inner_risk = exact_component_edit_risk(
            pred_components,
            gt_components,
            centroid_radius=centroid_radius,
            image_shape=logits_array.shape,
            connectivity=pixel_config.connectivity,
        )
        candidates.append(
            StructuredCandidate(
                pixel_edit=pixel_state,
                inner_risk=inner_risk,
                score=-pixel_state.edit_energy,
            )
        )
    candidates = tuple(candidates)

    decoded = _max_by_value_then_mask(candidates, lambda item: item.score)
    raw_mask = logits_array > pixel_config.threshold_logit
    if not np.array_equal(decoded.pixel_edit.mask(), raw_mask):
        raise RuntimeError("score decoder is not the raw strict-threshold mask")

    oracle_risk = min(candidate.inner_risk.risk for candidate in candidates)
    oracle_candidates = tuple(
        candidate
        for candidate in candidates
        if abs(candidate.inner_risk.risk - oracle_risk) <= numeric_tolerance
    )
    oracle = _max_by_value_then_mask(
        oracle_candidates,
        lambda item: item.score,
    )
    loss_augmented = _max_by_value_then_mask(
        candidates,
        lambda item: (
            item.score + item.inner_risk.risk - oracle_risk
        ),
    )

    hinge = (
        loss_augmented.score
        + loss_augmented.inner_risk.risk
        - oracle_risk
        - oracle.score
    )
    excess_risk = decoded.inner_risk.risk - oracle_risk
    slack = hinge - excess_risk
    if hinge < -numeric_tolerance:
        raise RuntimeError("corrected structured hinge became negative")
    if slack < -numeric_tolerance:
        raise RuntimeError("structured excess-risk upper bound was violated")

    return StructuredHingeResult(
        decoded=decoded,
        oracle=oracle,
        loss_augmented=loss_augmented,
        oracle_risk=float(oracle_risk),
        hinge=float(hinge),
        excess_risk=float(excess_risk),
        upper_bound_slack=float(slack),
        num_states=len(candidates),
    )


__all__ = [
    "StructuredCandidate",
    "StructuredHingeResult",
    "solve_exhaustive_structured_hinge",
]
