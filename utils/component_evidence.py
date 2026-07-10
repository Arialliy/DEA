"""Prediction-only candidate components for evidence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from skimage import measure


@dataclass(frozen=True)
class CandidateComponent:
    candidate_id: int
    mask: np.ndarray
    source: str
    probability_threshold: float
    area: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]


def _as_numpy_logits(value, expected_ndim: int, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != expected_ndim:
        raise ValueError(
            "%s must have %d dimensions, got %s"
            % (name, expected_ndim, tuple(array.shape))
        )
    return array.astype(np.float64, copy=False)


def generate_prediction_candidates(
    z_base,
    scale_logits,
    *,
    probability_thresholds=(0.5, 0.3, 0.2, 0.1),
    connectivity: int = 2,
) -> tuple[CandidateComponent, ...]:
    """Generate non-overlapping candidates without reading ground truth.

    Raw connected components are considered from high to low probability and
    from the final prediction to side0..sideN.  A later component that overlaps
    an accepted component is treated as another view of the same candidate and
    is skipped.  Disjoint side/low-threshold components remain available for
    recoverable-FN diagnostics.
    """

    final_logit = _as_numpy_logits(z_base, 2, "z_base")
    side_logits = _as_numpy_logits(scale_logits, 3, "scale_logits")
    if side_logits.shape[1:] != final_logit.shape:
        raise ValueError("scale_logits and z_base spatial shapes must match")
    thresholds = tuple(float(value) for value in probability_thresholds)
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("probability thresholds must lie strictly in (0, 1)")
    thresholds = tuple(sorted(set(thresholds), reverse=True))

    sources = [("final", final_logit)] + [
        ("scale%d" % index, side_logits[index])
        for index in range(side_logits.shape[0])
    ]
    accepted: list[CandidateComponent] = []
    accepted_union = np.zeros(final_logit.shape, dtype=bool)

    for threshold in thresholds:
        logit_threshold = float(np.log(threshold / (1.0 - threshold)))
        for source_name, source_logit in sources:
            label_map = measure.label(
                source_logit > logit_threshold,
                connectivity=connectivity,
            )
            for region in measure.regionprops(label_map):
                mask = label_map == region.label
                if np.logical_and(mask, accepted_union).any():
                    continue
                candidate = CandidateComponent(
                    candidate_id=len(accepted),
                    mask=mask,
                    source=source_name,
                    probability_threshold=threshold,
                    area=int(region.area),
                    centroid=(float(region.centroid[0]), float(region.centroid[1])),
                    bbox=tuple(int(value) for value in region.bbox),
                )
                accepted.append(candidate)
                accepted_union |= mask

    return tuple(accepted)


def candidate_label_map(
    candidates: tuple[CandidateComponent, ...],
    shape: tuple[int, int],
) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    for candidate in candidates:
        if candidate.mask.shape != shape:
            raise ValueError("candidate mask shape mismatch")
        if np.logical_and(labels != 0, candidate.mask).any():
            raise ValueError("candidate masks must not overlap")
        labels[candidate.mask] = candidate.candidate_id + 1
    return labels


__all__ = [
    "CandidateComponent",
    "candidate_label_map",
    "generate_prediction_candidates",
]
