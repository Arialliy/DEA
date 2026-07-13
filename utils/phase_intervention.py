"""Deterministic input-lattice interventions for frozen dense predictors.

The utilities in this module do not define a training method.  They create
integer translations after canonical preprocessing, align the resulting score
maps back to the original coordinates, and combine only valid aligned values.
Keeping the intervention separate from model and metric code makes it possible
to test whether a stride lattice changes target/component survival without
silently changing the evaluator.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral

import torch


Offset = tuple[int, int]
UNIT_PHASE_OFFSETS: tuple[Offset, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


class PhaseInterventionError(ValueError):
    """Raised when an intervention or alignment contract is invalid."""


def _integer_offset(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PhaseInterventionError(f"{name} must be an integer")
    return int(value)


def validate_offset(offset: object, *, height: int, width: int) -> Offset:
    if (
        not isinstance(offset, Sequence)
        or isinstance(offset, (str, bytes))
        or len(offset) != 2
    ):
        raise PhaseInterventionError("offset must be a (dy, dx) pair")
    dy = _integer_offset(offset[0], name="dy")
    dx = _integer_offset(offset[1], name="dx")
    if height < 2 or width < 2:
        raise PhaseInterventionError("reflection translation requires H,W >= 2")
    if abs(dy) >= height or abs(dx) >= width:
        raise PhaseInterventionError("offset magnitude must be smaller than the image")
    return dy, dx


def phase_preserving_offsets(stride: int) -> tuple[Offset, ...]:
    stride_value = _integer_offset(stride, name="stride")
    if stride_value < 1:
        raise PhaseInterventionError("stride must be positive")
    return (
        (0, 0),
        (0, stride_value),
        (stride_value, 0),
        (stride_value, stride_value),
    )


def residue_shifted_offsets(stride: int, residue: int = 1) -> tuple[Offset, ...]:
    """Return a near-magnitude control that changes the stride residue.

    Comparing a one-pixel translation directly with a ``stride``-pixel
    translation confounds lattice residue with displacement magnitude and the
    size of the invalid alignment border.  ``stride + residue`` is only one
    pixel farther than the phase-preserving control when ``residue=1`` while
    carrying the same non-zero residue as a unit shift at every divisor of the
    deepest dyadic stride.
    """

    stride_value = _integer_offset(stride, name="stride")
    residue_value = _integer_offset(residue, name="residue")
    if stride_value < 1:
        raise PhaseInterventionError("stride must be positive")
    if residue_value == 0 or abs(residue_value) >= stride_value:
        raise PhaseInterventionError("residue must be non-zero and smaller than stride")
    displacement = stride_value + residue_value
    if displacement < 1:
        raise PhaseInterventionError("stride + residue must be positive")
    return (
        (0, 0),
        (0, displacement),
        (displacement, 0),
        (displacement, displacement),
    )


def _reflection_indices(length: int, displacement: int, device) -> torch.Tensor:
    source = torch.arange(length, device=device, dtype=torch.long) - displacement
    period = 2 * (length - 1)
    folded = torch.remainder(source, period)
    return torch.where(folded < length, folded, period - folded)


def translate_reflect(value: torch.Tensor, offset: Offset) -> torch.Tensor:
    """Translate a BCHW tensor exactly on the integer lattice.

    Positive ``dy``/``dx`` move the original content down/right.  Values outside
    the source field use reflection.  Index selection is used instead of
    interpolation so integer translations introduce no numerical resampling.
    """

    if not torch.is_tensor(value) or value.ndim != 4:
        raise PhaseInterventionError("value must be a BCHW tensor")
    height, width = (int(value.shape[-2]), int(value.shape[-1]))
    dy, dx = validate_offset(offset, height=height, width=width)
    rows = _reflection_indices(height, dy, value.device)
    columns = _reflection_indices(width, dx, value.device)
    return value.index_select(-2, rows).index_select(-1, columns)


def align_translated_scores(
    scores: torch.Tensor,
    offset: Offset,
    *,
    fill_value: float = float("-inf"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse an input translation without inventing out-of-frame scores.

    Returns an aligned BCHW score tensor and a broadcastable ``[1,1,H,W]``
    validity mask.  Invalid locations are set to ``fill_value`` and therefore
    cannot win a max reduction when the zero-offset view is present.
    """

    if not torch.is_tensor(scores) or scores.ndim != 4:
        raise PhaseInterventionError("scores must be a BCHW tensor")
    height, width = (int(scores.shape[-2]), int(scores.shape[-1]))
    dy, dx = validate_offset(offset, height=height, width=width)
    aligned = torch.full_like(scores, fill_value)
    valid = torch.zeros((1, 1, height, width), dtype=torch.bool, device=scores.device)

    if dy >= 0:
        source_y = slice(dy, height)
        target_y = slice(0, height - dy)
    else:
        source_y = slice(0, height + dy)
        target_y = slice(-dy, height)
    if dx >= 0:
        source_x = slice(dx, width)
        target_x = slice(0, width - dx)
    else:
        source_x = slice(0, width + dx)
        target_x = slice(-dx, width)

    aligned[..., target_y, target_x] = scores[..., source_y, source_x]
    valid[..., target_y, target_x] = True
    return aligned, valid


def aggregate_aligned_scores(
    score_views: Iterable[torch.Tensor],
    offsets: Sequence[Offset],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align and max-reduce score views using a fixed, auditable rule.

    The zero-offset view is mandatory and guarantees that the aggregate is
    finite everywhere.  The return values are ``aggregate``, aligned view
    stack ``[V,B,C,H,W]``, and validity stack ``[V,1,1,H,W]``.
    """

    views = tuple(score_views)
    normalized_offsets = tuple(offsets)
    if not views or len(views) != len(normalized_offsets):
        raise PhaseInterventionError("score views and offsets must be non-empty and aligned")
    if normalized_offsets.count((0, 0)) != 1:
        raise PhaseInterventionError("offsets must contain exactly one zero-offset view")
    reference_shape = tuple(views[0].shape)
    if any(not torch.is_tensor(view) or tuple(view.shape) != reference_shape for view in views):
        raise PhaseInterventionError("all score views must be tensors with one common shape")

    aligned_views = []
    valid_masks = []
    for view, offset in zip(views, normalized_offsets):
        aligned, valid = align_translated_scores(view, offset)
        aligned_views.append(aligned)
        valid_masks.append(valid)
    stack = torch.stack(aligned_views, dim=0)
    validity = torch.stack(valid_masks, dim=0)
    aggregate = stack.max(dim=0).values
    if not bool(torch.isfinite(aggregate).all()):
        raise PhaseInterventionError("aligned aggregate is not finite")
    return aggregate, stack, validity


__all__ = [
    "Offset",
    "UNIT_PHASE_OFFSETS",
    "PhaseInterventionError",
    "aggregate_aligned_scores",
    "align_translated_scores",
    "phase_preserving_offsets",
    "residue_shifted_offsets",
    "translate_reflect",
    "validate_offset",
]
