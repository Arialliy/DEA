"""Read-only scale recoverability diagnostics for MSHNet no-response GTs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from skimage import measure

from utils.metric import match_components_hungarian


@dataclass(frozen=True)
class NoResponseTargetScaleRecord:
    target_index: int
    side_support_scales: tuple[int, ...]
    side_centroid_legal_scales: tuple[int, ...]
    side_matched_scales: tuple[int, ...]
    recovering_subsets: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NoResponseScaleResult:
    num_gt: int
    final_matches: int
    final_no_response_target_indices: tuple[int, ...]
    records: tuple[NoResponseTargetScaleRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite_array(value, *, ndim: int, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    if not np.isfinite(array).all():
        raise ValueError("%s must contain only finite values" % name)
    return array.astype(np.float64, copy=False)


def _binary_target(value, *, shape: tuple[int, int]) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != shape or array.ndim != 2:
        raise ValueError("target must be 2-D and match logits")
    if not np.logical_or(array == 0, array == 1).all():
        raise ValueError("target must be binary")
    return array.astype(bool, copy=False)


def _target_has_support_near(
    component_match,
    target_index: int,
    *,
    radius: float,
) -> bool:
    target_region = component_match.target_regions[target_index]
    target_coordinates = np.asarray(target_region.coords, dtype=np.float64)
    for prediction_region in component_match.prediction_regions:
        prediction_coordinates = np.asarray(
            prediction_region.coords,
            dtype=np.float64,
        )
        differences = (
            target_coordinates[:, None, :]
            - prediction_coordinates[None, :, :]
        )
        if float(np.sqrt(np.sum(differences**2, axis=2)).min()) < radius:
            return True
    return False


def _target_has_centroid_legal_component(
    component_match,
    target_index: int,
    *,
    radius: float,
) -> bool:
    target_centroid = np.asarray(
        component_match.target_regions[target_index].centroid,
        dtype=np.float64,
    )
    return any(
        float(
            np.linalg.norm(
                np.asarray(region.centroid, dtype=np.float64)
                - target_centroid
            )
        )
        < radius
        for region in component_match.prediction_regions
    )


def analyze_no_response_scales(
    final_logit,
    scale_logits,
    scale_contributions,
    fusion_bias,
    target,
    *,
    threshold_logit: float = 0.0,
    centroid_radius: float = 3.0,
    connectivity: int = 2,
    reconstruction_atol: float = 1e-4,
) -> NoResponseScaleResult:
    """Classify final no-response targets by side/subset recoverability.

    ``recovering_subsets`` is a GT-conditioned upper-bound diagnostic over the
    14 non-empty, non-full global scale subsets.  It is not a deployable
    selector and must never be reported as model performance.
    """
    final = _finite_array(final_logit, ndim=2, name="final_logit")
    sides = _finite_array(scale_logits, ndim=3, name="scale_logits")
    contributions = _finite_array(
        scale_contributions,
        ndim=3,
        name="scale_contributions",
    )
    if sides.shape != contributions.shape or sides.shape[0] != 4:
        raise ValueError("scale logits/contributions must both have shape [4,H,W]")
    if sides.shape[1:] != final.shape:
        raise ValueError("all logits must share the final spatial shape")
    target_array = _binary_target(target, shape=final.shape)
    try:
        bias = float(np.asarray(fusion_bias).reshape(-1)[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("fusion_bias must contain one finite scalar") from exc
    if np.asarray(fusion_bias).size != 1 or not np.isfinite(bias):
        raise ValueError("fusion_bias must contain one finite scalar")
    if not np.isfinite(threshold_logit):
        raise ValueError("threshold_logit must be finite")
    if not np.isfinite(centroid_radius) or centroid_radius <= 0:
        raise ValueError("centroid_radius must be finite and positive")
    if connectivity != 2:
        raise ValueError("connectivity must be 2")
    if not np.isfinite(reconstruction_atol) or reconstruction_atol < 0:
        raise ValueError("reconstruction_atol must be finite and non-negative")

    reconstructed = contributions.sum(axis=0) + bias
    if not np.allclose(
        reconstructed,
        final,
        atol=reconstruction_atol,
        rtol=1e-5,
    ):
        raise RuntimeError("scale contributions do not reconstruct final logits")

    final_match = match_components_hungarian(
        final > threshold_logit,
        target_array,
        centroid_radius=centroid_radius,
        connectivity=connectivity,
    )
    matched_final_targets = {item[0] for item in final_match.matches}
    no_response_targets = tuple(
        target_index
        for target_index in final_match.unmatched_target_indices
        if not _target_has_support_near(
            final_match,
            target_index,
            radius=centroid_radius,
        )
    )

    side_matches = []
    for scale_index in range(4):
        side_matches.append(
            match_components_hungarian(
                sides[scale_index] > threshold_logit,
                target_array,
                centroid_radius=centroid_radius,
                connectivity=connectivity,
            )
        )

    subset_matches = {}
    for subset in range(1, 15):
        selected = tuple(
            index for index in range(4) if subset & (1 << index)
        )
        subset_logit = contributions[list(selected)].sum(axis=0) + bias
        subset_matches[subset] = match_components_hungarian(
            subset_logit > threshold_logit,
            target_array,
            centroid_radius=centroid_radius,
            connectivity=connectivity,
        )

    records = []
    for target_index in no_response_targets:
        support_scales = tuple(
            scale_index
            for scale_index, component_match in enumerate(side_matches)
            if _target_has_support_near(
                component_match,
                target_index,
                radius=centroid_radius,
            )
        )
        centroid_scales = tuple(
            scale_index
            for scale_index, component_match in enumerate(side_matches)
            if _target_has_centroid_legal_component(
                component_match,
                target_index,
                radius=centroid_radius,
            )
        )
        matched_scales = tuple(
            scale_index
            for scale_index, component_match in enumerate(side_matches)
            if target_index in {item[0] for item in component_match.matches}
        )
        recovering_subsets = tuple(
            subset
            for subset, component_match in subset_matches.items()
            if target_index in {item[0] for item in component_match.matches}
        )
        records.append(
            NoResponseTargetScaleRecord(
                target_index=target_index,
                side_support_scales=support_scales,
                side_centroid_legal_scales=centroid_scales,
                side_matched_scales=matched_scales,
                recovering_subsets=recovering_subsets,
            )
        )

    if matched_final_targets.intersection(no_response_targets):
        raise RuntimeError("a final matched target was classified as no-response")
    return NoResponseScaleResult(
        num_gt=len(final_match.target_regions),
        final_matches=len(final_match.matches),
        final_no_response_target_indices=no_response_targets,
        records=tuple(records),
    )


__all__ = [
    "NoResponseScaleResult",
    "NoResponseTargetScaleRecord",
    "analyze_no_response_scales",
]
