"""Exact run-semiring inference for the TRACE atomic component family.

The random variable represented here is *one* empty-or-component variable in
each local field.  A non-empty component is a consecutive chain of horizontal
intervals, one interval per occupied row.  Consecutive intervals must be
8-connected, and the canonical root is the left endpoint of the first
interval.  The implementation deliberately contains no proposal, refinement,
or auxiliary prediction path.

``root_energy`` and ``support_energy`` are natural parameters, not independent
probabilities.  Probabilities only arise after adding the empty state and
normalizing the complete local state space.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


class TraceSemiringError(ValueError):
    """Raised when a local TRACE field violates the exact-solver contract."""


@dataclass(frozen=True)
class TracePartitionResult:
    """Empty-inclusive normalized partition quantities for a batch of fields."""

    logZ_positive: torch.Tensor
    logZ_total: torch.Tensor
    p_nonempty: torch.Tensor
    log_cardinality: torch.Tensor
    cardinality: torch.Tensor | None
    correction: torch.Tensor

    @property
    def log_positive_partition(self) -> torch.Tensor:
        return self.logZ_positive

    @property
    def log_partition(self) -> torch.Tensor:
        return self.logZ_total

    @property
    def nonempty_probability(self) -> torch.Tensor:
        return self.p_nonempty


@dataclass(frozen=True)
class TraceMAPResult:
    """Exact maximum-semiring non-empty atom and its local backtrace."""

    map_energy: torch.Tensor
    map_root: torch.Tensor
    map_intervals: torch.Tensor
    map_support: torch.Tensor


@dataclass(frozen=True)
class TraceMarginalResult:
    """Unconditional marginals, including the probability of the empty state."""

    root: torch.Tensor
    support: torch.Tensor
    logZ_total: torch.Tensor

    @property
    def root_marginal(self) -> torch.Tensor:
        return self.root

    @property
    def support_marginal(self) -> torch.Tensor:
        return self.support


@dataclass(frozen=True)
class TraceSemiringOutput:
    """Joint output of the exact sum and (optionally) max semirings."""

    logZ_positive: torch.Tensor
    logZ_total: torch.Tensor
    p_nonempty: torch.Tensor
    log_cardinality: torch.Tensor
    cardinality: torch.Tensor | None
    correction: torch.Tensor
    map_energy: torch.Tensor | None
    map_log_joint_posterior: torch.Tensor | None
    map_root: torch.Tensor | None
    map_intervals: torch.Tensor | None
    map_support: torch.Tensor | None
    root_marginal: torch.Tensor | None = None
    support_marginal: torch.Tensor | None = None
    state_count: torch.Tensor | None = None

    @property
    def log_positive_partition(self) -> torch.Tensor:
        return self.logZ_positive

    @property
    def log_partition(self) -> torch.Tensor:
        return self.logZ_total

    @property
    def nonempty_probability(self) -> torch.Tensor:
        return self.p_nonempty

    @property
    def log_joint_posterior(self) -> torch.Tensor | None:
        """Log posterior of the emitted non-empty MAP atom."""

        return self.map_log_joint_posterior


@dataclass(frozen=True)
class EnumeratedRunChain:
    """One tiny-window reference state, represented by inclusive intervals."""

    intervals: tuple[tuple[int, int, int], ...]

    @property
    def root(self) -> tuple[int, int]:
        y, left, _ = self.intervals[0]
        return y, left


def _computation_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.float64
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float32
    raise TraceSemiringError("energies must use a floating-point dtype")


def _validate_energies(
    root_energy: torch.Tensor,
    support_energy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(root_energy) or not torch.is_tensor(support_energy):
        raise TraceSemiringError("root_energy and support_energy must be tensors")
    if root_energy.ndim != 3 or tuple(support_energy.shape) != tuple(root_energy.shape):
        raise TraceSemiringError("energies must have the same [N,H,W] shape")
    if min(root_energy.shape) < 1:
        raise TraceSemiringError("local fields must have positive N, H, and W")
    if root_energy.device != support_energy.device:
        raise TraceSemiringError("root and support energies must share a device")
    if root_energy.dtype != support_energy.dtype:
        raise TraceSemiringError("root and support energies must share a dtype")
    if not root_energy.is_floating_point():
        raise TraceSemiringError("energies must be floating point")
    if not bool(torch.isfinite(root_energy).all()) or not bool(
        torch.isfinite(support_energy).all()
    ):
        raise TraceSemiringError("energies must be finite")
    dtype = _computation_dtype(root_energy.dtype)
    return root_energy.to(dtype=dtype), support_energy.to(dtype=dtype)


def _normalize_mask(
    mask: torch.Tensor | None,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    name: str,
    default: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is None:
        if default is not None:
            return default
        return torch.ones((batch, height, width), dtype=torch.bool, device=device)
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise TraceSemiringError(f"{name} must be a boolean tensor")
    if tuple(mask.shape) == (height, width):
        result = mask.to(device=device).unsqueeze(0).expand(batch, -1, -1)
    elif tuple(mask.shape) == (1, height, width):
        result = mask.to(device=device).expand(batch, -1, -1)
    elif tuple(mask.shape) == (batch, height, width):
        result = mask.to(device=device)
    else:
        raise TraceSemiringError(
            f"{name} must be [H,W], [1,H,W], or [N,H,W]"
        )
    return result


def _normalize_masks(
    valid_support_mask: torch.Tensor | None,
    valid_root_mask: torch.Tensor | None,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    support = _normalize_mask(
        valid_support_mask,
        batch=batch,
        height=height,
        width=width,
        device=device,
        name="valid_support_mask",
    )
    root = _normalize_mask(
        valid_root_mask,
        batch=batch,
        height=height,
        width=width,
        device=device,
        name="valid_root_mask",
        default=support,
    )
    # A canonical root is itself a support pixel.  Taking the intersection is
    # the literal conjunction of the two validity predicates, not padding or
    # a repair of an invalid state.
    root = root & support
    if not bool(root.reshape(batch, -1).any(dim=1).all()):
        raise TraceSemiringError("every local field must contain at least one legal atom")
    return support, root


def _interval_fields(
    support_energy: torch.Tensor,
    valid_support_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return interval sums and legality as [N,H,left,right]."""

    _, _, width = support_energy.shape
    zero = support_energy.new_zeros((*support_energy.shape[:-1], 1))
    # Mask before the prefix sum so padded/invalid natural parameters have an
    # identically zero derivative, including for valid intervals to their
    # right whose two prefix terms would otherwise cancel only numerically.
    scored_support = torch.where(
        valid_support_mask, support_energy, torch.zeros_like(support_energy)
    )
    prefix = torch.cat((zero, torch.cumsum(scored_support, dim=-1)), dim=-1)
    interval_score = prefix[..., 1:].unsqueeze(-2) - prefix[..., :-1].unsqueeze(-1)

    invalid = (~valid_support_mask).to(dtype=torch.int64)
    invalid_zero = torch.zeros((*invalid.shape[:-1], 1), dtype=torch.int64, device=invalid.device)
    invalid_prefix = torch.cat((invalid_zero, torch.cumsum(invalid, dim=-1)), dim=-1)
    invalid_count = (
        invalid_prefix[..., 1:].unsqueeze(-2)
        - invalid_prefix[..., :-1].unsqueeze(-1)
    )
    coordinates = torch.arange(width, device=support_energy.device)
    triangle = coordinates[:, None] <= coordinates[None, :]
    legal = triangle[None, None] & invalid_count.eq(0)
    return interval_score, legal


def _safe_logcumsumexp(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    """``logcumsumexp`` with zero gradients for all-``-inf`` prefixes.

    PyTorch's mathematical result for an all-``-inf`` reduction is correct,
    but its generic backward can encounter ``-inf - -inf``.  Replacing only
    masked entries by the smallest finite value makes their exponential mass
    exactly underflow to zero in FP32/FP64, after which all-empty prefixes are
    restored to ``-inf``.
    """

    finite = torch.isfinite(values)
    sentinel = torch.full_like(values, torch.finfo(values.dtype).min)
    safe_values = torch.where(finite, values, sentinel)
    cumulative = torch.logcumsumexp(safe_values, dim=dim)
    has_value = torch.cumsum(finite.to(dtype=torch.int64), dim=dim).gt(0)
    return torch.where(has_value, cumulative, torch.full_like(cumulative, -torch.inf))


def _safe_logsumexp(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    """``logsumexp`` whose all-empty slices have finite, zero backward."""

    finite = torch.isfinite(values)
    safe_values = torch.where(
        finite,
        values,
        torch.full_like(values, torch.finfo(values.dtype).min),
    )
    reduced = torch.logsumexp(safe_values, dim=dim)
    has_value = finite.any(dim=dim)
    return torch.where(has_value, reduced, torch.full_like(reduced, -torch.inf))


def _safe_logaddexp(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    both_empty = torch.isneginf(left) & torch.isneginf(right)
    safe_left = torch.where(both_empty, torch.zeros_like(left), left)
    safe_right = torch.where(both_empty, torch.zeros_like(right), right)
    result = torch.logaddexp(safe_left, safe_right)
    return torch.where(both_empty, torch.full_like(result, -torch.inf), result)


def _sum_transition(previous: torch.Tensor) -> torch.Tensor:
    """Exact O(W^2) orthant LSE for every current interval [left,right]."""

    _, width, _ = previous.shape
    suffix_right = torch.flip(
        _safe_logcumsumexp(torch.flip(previous, dims=(-1,)), dim=-1),
        dims=(-1,),
    )
    prefix_left = _safe_logcumsumexp(suffix_right, dim=-2)

    coordinate = torch.arange(width, device=previous.device)
    max_previous_left = (coordinate + 1).clamp_max(width - 1)
    min_previous_right = (coordinate - 1).clamp_min(0)
    p = max_previous_left[None, :].expand(width, width)
    q = min_previous_right[:, None].expand(width, width)
    return prefix_left[:, p, q]


def _positive_log_partition(
    root_energy: torch.Tensor,
    support_energy: torch.Tensor,
    valid_support_mask: torch.Tensor,
    valid_root_mask: torch.Tensor,
    correction: torch.Tensor,
) -> torch.Tensor:
    """Run the exact sum semiring; all tensors are already normalized."""

    batch, height, width = root_energy.shape
    interval_score, legal_interval = _interval_fields(
        support_energy, valid_support_mask
    )
    neg_inf = root_energy.new_full((batch, width, width), -torch.inf)
    previous = neg_inf
    logZ_positive = root_energy.new_full((batch,), -torch.inf)

    for y in range(height):
        start = root_energy[:, y, :, None].expand(-1, -1, width)
        start = start - correction[:, None, None]
        start_valid = valid_root_mask[:, y, :, None] & legal_interval[:, y]
        start = torch.where(start_valid, start, neg_inf)

        transition = neg_inf if y == 0 else _sum_transition(previous)
        state = interval_score[:, y] + _safe_logaddexp(start, transition)
        state = torch.where(legal_interval[:, y], state, neg_inf)
        row_mass = _safe_logsumexp(state.reshape(batch, -1), dim=-1)
        logZ_positive = _safe_logaddexp(logZ_positive, row_mass)
        previous = state
    return logZ_positive


def _normalize_log_cardinality(
    value: torch.Tensor | float,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=dtype, device=device)
    if result.ndim == 0:
        result = result.expand(batch)
    elif tuple(result.shape) == (1,):
        result = result.expand(batch)
    elif tuple(result.shape) != (batch,):
        raise TraceSemiringError("log_cardinality must be scalar, [1], or [N]")
    if not bool(torch.isfinite(result).all()):
        raise TraceSemiringError("every local field must contain at least one legal atom")
    if bool((result < -32.0 * torch.finfo(dtype).eps).any()):
        raise TraceSemiringError("log_cardinality cannot be negative")
    return result


def _zero_score_log_cardinality_normalized(
    valid_support_mask: torch.Tensor,
    valid_root_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch, height, width = valid_support_mask.shape
    zeros = torch.zeros(
        (batch, height, width), dtype=dtype, device=valid_support_mask.device
    )
    correction = torch.zeros((batch,), dtype=dtype, device=zeros.device)
    with torch.no_grad():
        result = _positive_log_partition(
            zeros,
            zeros,
            valid_support_mask,
            valid_root_mask,
            correction,
        )
    if not bool(torch.isfinite(result).all()):
        raise TraceSemiringError("every local field must contain at least one legal atom")
    return result


def zero_score_log_cardinality(
    valid_support_mask: torch.Tensor,
    valid_root_mask: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Compute ``log K`` by running the same sum semiring at zero energy."""

    if not torch.is_tensor(valid_support_mask) or valid_support_mask.dtype != torch.bool:
        raise TraceSemiringError("valid_support_mask must be a boolean tensor")
    if valid_support_mask.ndim == 2:
        batch, height, width = 1, *valid_support_mask.shape
    elif valid_support_mask.ndim == 3:
        batch, height, width = valid_support_mask.shape
    else:
        raise TraceSemiringError("valid_support_mask must be [H,W] or [N,H,W]")
    if dtype not in (torch.float32, torch.float64):
        raise TraceSemiringError("cardinality dtype must be float32 or float64")
    support, root = _normalize_masks(
        valid_support_mask,
        valid_root_mask,
        batch=batch,
        height=height,
        width=width,
        device=valid_support_mask.device,
    )
    return _zero_score_log_cardinality_normalized(support, root, dtype=dtype)


def zero_score_cardinality(
    valid_support_mask: torch.Tensor,
    valid_root_mask: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return ``K`` as a floating tensor (``log K`` is safer for large fields)."""

    logK = zero_score_log_cardinality(
        valid_support_mask, valid_root_mask, dtype=dtype
    )
    cardinality = _materialize_cardinality(logK)
    if cardinality is None:
        raise TraceSemiringError(
            "cardinality exceeds FP64 range; use zero_score_log_cardinality"
        )
    return cardinality


def _materialize_cardinality(logK: torch.Tensor) -> torch.Tensor | None:
    """Materialize K in FP64 when representable; never return NaN or Inf."""

    logK64 = logK.to(dtype=torch.float64)
    maximum_log = math.log(torch.finfo(torch.float64).max)
    if bool((logK64 > maximum_log).any()):
        return None
    return torch.exp(logK64)


def _resolve_log_cardinality(
    supplied: torch.Tensor | float | None,
    valid_support_mask: torch.Tensor,
    valid_root_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch = valid_support_mask.shape[0]
    if supplied is None:
        return _zero_score_log_cardinality_normalized(
            valid_support_mask, valid_root_mask, dtype=dtype
        )
    return _normalize_log_cardinality(
        supplied,
        batch=batch,
        dtype=dtype,
        device=valid_support_mask.device,
    )


def exact_sum_product(
    root_energy: torch.Tensor,
    support_energy: torch.Tensor,
    valid_support_mask: torch.Tensor | None = None,
    valid_root_mask: torch.Tensor | None = None,
    *,
    log_cardinality: torch.Tensor | float | None = None,
    cardinality_correction: bool = True,
) -> TracePartitionResult:
    """Compute exact positive and empty-inclusive log partitions.

    ``logZ_positive`` is optionally corrected by ``-log(K)`` once per
    non-empty atom.  ``logZ_total = log(1 + exp(logZ_positive))`` includes the
    empty state whose energy is fixed to zero.
    """

    if not isinstance(cardinality_correction, bool):
        raise TraceSemiringError("cardinality_correction must be a bool")
    root, support_energy = _validate_energies(root_energy, support_energy)
    batch, height, width = root.shape
    support_mask, root_mask = _normalize_masks(
        valid_support_mask,
        valid_root_mask,
        batch=batch,
        height=height,
        width=width,
        device=root.device,
    )
    logK = _resolve_log_cardinality(
        log_cardinality,
        support_mask,
        root_mask,
        dtype=root.dtype,
    )
    correction = logK if cardinality_correction else torch.zeros_like(logK)
    logZ_positive = _positive_log_partition(
        root, support_energy, support_mask, root_mask, correction
    )
    logZ_total = torch.logaddexp(torch.zeros_like(logZ_positive), logZ_positive)
    return TracePartitionResult(
        logZ_positive=logZ_positive,
        logZ_total=logZ_total,
        p_nonempty=torch.sigmoid(logZ_positive),
        log_cardinality=logK,
        cardinality=_materialize_cardinality(logK),
        correction=correction,
    )


def _max_transition(
    previous: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """O(W^2) max orthant transform and previous-interval argmax."""

    batch, width, _ = previous.shape
    reversed_previous = torch.flip(previous, dims=(-1,))
    reversed_value, reversed_index = torch.cummax(reversed_previous, dim=-1)
    suffix_value = torch.flip(reversed_value, dims=(-1,))
    suffix_right = width - 1 - torch.flip(reversed_index, dims=(-1,))

    prefix_value, prefix_left = torch.cummax(suffix_value, dim=-2)
    prefix_right = torch.gather(suffix_right, dim=-2, index=prefix_left)

    coordinate = torch.arange(width, device=previous.device)
    max_previous_left = (coordinate + 1).clamp_max(width - 1)
    min_previous_right = (coordinate - 1).clamp_min(0)
    p = max_previous_left[None, :].expand(width, width)
    q = min_previous_right[:, None].expand(width, width)
    batch_index = torch.arange(batch, device=previous.device)[:, None, None]
    return (
        prefix_value[batch_index, p, q],
        prefix_left[batch_index, p, q],
        prefix_right[batch_index, p, q],
    )


def exact_max_semiring(
    root_energy: torch.Tensor,
    support_energy: torch.Tensor,
    valid_support_mask: torch.Tensor | None = None,
    valid_root_mask: torch.Tensor | None = None,
    *,
    log_cardinality: torch.Tensor | float | None = None,
    cardinality_correction: bool = True,
) -> TraceMAPResult:
    """Return the exact highest-energy non-empty atom and its backtrace."""

    if not isinstance(cardinality_correction, bool):
        raise TraceSemiringError("cardinality_correction must be a bool")
    root, support_energy = _validate_energies(root_energy, support_energy)
    batch, height, width = root.shape
    support_mask, root_mask = _normalize_masks(
        valid_support_mask,
        valid_root_mask,
        batch=batch,
        height=height,
        width=width,
        device=root.device,
    )
    logK = _resolve_log_cardinality(
        log_cardinality,
        support_mask,
        root_mask,
        dtype=root.dtype,
    )
    correction = logK if cardinality_correction else torch.zeros_like(logK)
    interval_score, legal_interval = _interval_fields(support_energy, support_mask)
    neg_inf = root.new_full((batch, width, width), -torch.inf)
    previous = neg_inf

    start_flags: list[torch.Tensor] = []
    previous_left: list[torch.Tensor] = []
    previous_right: list[torch.Tensor] = []
    best_energy = root.new_full((batch,), -torch.inf)
    best_y = torch.full((batch,), -1, dtype=torch.long, device=root.device)
    best_left = torch.full_like(best_y, -1)
    best_right = torch.full_like(best_y, -1)

    for y in range(height):
        start = root[:, y, :, None].expand(-1, -1, width)
        start = start - correction[:, None, None]
        start_valid = root_mask[:, y, :, None] & legal_interval[:, y]
        start = torch.where(start_valid, start, neg_inf)

        if y == 0:
            transition = neg_inf
            arg_left = torch.zeros(
                (batch, width, width), dtype=torch.long, device=root.device
            )
            arg_right = torch.zeros_like(arg_left)
        else:
            transition, arg_left, arg_right = _max_transition(previous)
        choose_start = start >= transition
        predecessor_score = torch.where(choose_start, start, transition)
        state = interval_score[:, y] + predecessor_score
        state = torch.where(legal_interval[:, y], state, neg_inf)

        start_flags.append(choose_start)
        previous_left.append(arg_left)
        previous_right.append(arg_right)
        row_energy, flat_index = torch.max(state.reshape(batch, -1), dim=-1)
        improve = row_energy > best_energy
        best_energy = torch.where(improve, row_energy, best_energy)
        best_y = torch.where(improve, torch.full_like(best_y, y), best_y)
        best_left = torch.where(improve, flat_index // width, best_left)
        best_right = torch.where(improve, flat_index % width, best_right)
        previous = state

    if not bool(torch.isfinite(best_energy).all()):
        raise TraceSemiringError("every local field must contain at least one legal atom")

    intervals = torch.full(
        (batch, height, 2), -1, dtype=torch.long, device=root.device
    )
    local_support = torch.zeros(
        (batch, height, width), dtype=torch.bool, device=root.device
    )
    map_root = torch.full((batch, 2), -1, dtype=torch.long, device=root.device)

    # Fixed-H batched tensor backtrace: there is no Python loop or device
    # synchronization per local field.  Inactive fields use safe dummy
    # coordinates, and masks ensure that those coordinates cannot alter the
    # reconstructed atom.
    flag_table = torch.stack(start_flags, dim=1)
    left_table = torch.stack(previous_left, dim=1)
    right_table = torch.stack(previous_right, dim=1)
    batch_index = torch.arange(batch, device=root.device)
    row_coordinates = torch.arange(height, device=root.device)[None, :]
    column_coordinates = torch.arange(width, device=root.device)[None, None, :]
    current_y = best_y
    current_left = best_left
    current_right = best_right
    active = torch.ones((batch,), dtype=torch.bool, device=root.device)
    for _ in range(height):
        safe_y = current_y.clamp(0, height - 1)
        safe_left = current_left.clamp(0, width - 1)
        safe_right = current_right.clamp(0, width - 1)
        row_update = active[:, None] & row_coordinates.eq(safe_y[:, None])
        intervals[..., 0] = torch.where(
            row_update, safe_left[:, None], intervals[..., 0]
        )
        intervals[..., 1] = torch.where(
            row_update, safe_right[:, None], intervals[..., 1]
        )
        support_update = (
            row_update[:, :, None]
            & column_coordinates.ge(safe_left[:, None, None])
            & column_coordinates.le(safe_right[:, None, None])
        )
        local_support = local_support | support_update

        starts_here = active & flag_table[
            batch_index, safe_y, safe_left, safe_right
        ]
        map_root[:, 0] = torch.where(starts_here, safe_y, map_root[:, 0])
        map_root[:, 1] = torch.where(starts_here, safe_left, map_root[:, 1])
        next_left = left_table[batch_index, safe_y, safe_left, safe_right]
        next_right = right_table[batch_index, safe_y, safe_left, safe_right]
        active = active & ~starts_here
        current_y = torch.where(active, safe_y - 1, torch.zeros_like(safe_y))
        current_left = torch.where(active, next_left, torch.zeros_like(next_left))
        current_right = torch.where(active, next_right, torch.zeros_like(next_right))

    return TraceMAPResult(
        map_energy=best_energy,
        map_root=map_root,
        map_intervals=intervals,
        map_support=local_support,
    )


def exact_marginals(
    root_energy: torch.Tensor,
    support_energy: torch.Tensor,
    valid_support_mask: torch.Tensor | None = None,
    valid_root_mask: torch.Tensor | None = None,
    *,
    log_cardinality: torch.Tensor | float | None = None,
    cardinality_correction: bool = True,
    create_graph: bool = False,
) -> TraceMarginalResult:
    """Differentiate the *empty-inclusive* ``logZ_total`` for marginals.

    Consequently, these are unconditional joint marginals:
    ``d logZ_total / d root_energy`` and
    ``d logZ_total / d support_energy``.  They sum/occupy less than one when
    the empty state has non-zero posterior mass.  Differentiating
    ``logZ_positive`` instead would incorrectly return conditional-on-present
    marginals.
    """

    _, result = _partition_and_marginals(
        root_energy,
        support_energy,
        valid_support_mask,
        valid_root_mask,
        log_cardinality=log_cardinality,
        cardinality_correction=cardinality_correction,
        create_graph=create_graph,
    )
    return result


def _partition_and_marginals(
    root_energy: torch.Tensor,
    support_energy: torch.Tensor,
    valid_support_mask: torch.Tensor | None,
    valid_root_mask: torch.Tensor | None,
    *,
    log_cardinality: torch.Tensor | float | None,
    cardinality_correction: bool,
    create_graph: bool,
) -> tuple[TracePartitionResult, TraceMarginalResult]:
    """Return a partition and its marginals from one shared autograd graph."""

    if not isinstance(create_graph, bool):
        raise TraceSemiringError("create_graph must be a bool")
    with torch.enable_grad():
        root_work = root_energy
        support_work = support_energy
        if not root_work.requires_grad:
            root_work = root_work.detach().requires_grad_(True)
        if not support_work.requires_grad:
            support_work = support_work.detach().requires_grad_(True)
        partition = exact_sum_product(
            root_work,
            support_work,
            valid_support_mask,
            valid_root_mask,
            log_cardinality=log_cardinality,
            cardinality_correction=cardinality_correction,
        )
        root_marginal, support_marginal = torch.autograd.grad(
            partition.logZ_total.sum(),
            (root_work, support_work),
            create_graph=create_graph,
            retain_graph=True,
        )
    result = TraceMarginalResult(
        root=root_marginal,
        support=support_marginal,
        logZ_total=partition.logZ_total,
    )
    return partition, result


def enumerate_run_chains(
    valid_support_mask: torch.Tensor,
    valid_root_mask: torch.Tensor | None = None,
    *,
    max_states: int = 1_000_000,
) -> tuple[tuple[EnumeratedRunChain, ...], ...]:
    """Enumerate tiny reference spaces without using the dynamic program."""

    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise TraceSemiringError("max_states must be a positive integer")
    if not torch.is_tensor(valid_support_mask) or valid_support_mask.dtype != torch.bool:
        raise TraceSemiringError("valid_support_mask must be a boolean tensor")
    if valid_support_mask.ndim == 2:
        batch, height, width = 1, *valid_support_mask.shape
    elif valid_support_mask.ndim == 3:
        batch, height, width = valid_support_mask.shape
    else:
        raise TraceSemiringError("valid_support_mask must be [H,W] or [N,H,W]")
    support, root = _normalize_masks(
        valid_support_mask,
        valid_root_mask,
        batch=batch,
        height=height,
        width=width,
        device=valid_support_mask.device,
    )
    support_cpu = support.detach().cpu()
    root_cpu = root.detach().cpu()
    all_batches: list[tuple[EnumeratedRunChain, ...]] = []

    for n in range(batch):
        legal_by_row: list[list[tuple[int, int]]] = []
        for y in range(height):
            intervals: list[tuple[int, int]] = []
            for left in range(width):
                for right in range(left, width):
                    if bool(support_cpu[n, y, left : right + 1].all()):
                        intervals.append((left, right))
            legal_by_row.append(intervals)

        states: list[EnumeratedRunChain] = []

        def append_state(intervals: Sequence[tuple[int, int, int]]) -> None:
            states.append(EnumeratedRunChain(tuple(intervals)))
            if len(states) > max_states:
                raise TraceSemiringError(
                    f"brute-force reference exceeds max_states={max_states}"
                )

        def extend(chain: list[tuple[int, int, int]]) -> None:
            append_state(chain)
            y, previous_left, previous_right = chain[-1]
            next_y = y + 1
            if next_y >= height:
                return
            for left, right in legal_by_row[next_y]:
                if left <= previous_right + 1 and right >= previous_left - 1:
                    extend([*chain, (next_y, left, right)])

        for y in range(height):
            for left, right in legal_by_row[y]:
                if bool(root_cpu[n, y, left]):
                    extend([(y, left, right)])
        if not states:
            raise TraceSemiringError("every local field must contain at least one legal atom")
        all_batches.append(tuple(states))
    return tuple(all_batches)


def brute_force_reference(
    root_energy: torch.Tensor,
    support_energy: torch.Tensor,
    valid_support_mask: torch.Tensor | None = None,
    valid_root_mask: torch.Tensor | None = None,
    *,
    log_cardinality: torch.Tensor | float | None = None,
    cardinality_correction: bool = True,
    max_states: int = 1_000_000,
    return_marginals: bool = False,
    create_graph: bool = False,
) -> TraceSemiringOutput:
    """Independent exhaustive reference for 3x3/4x4 correctness tests."""

    if not isinstance(cardinality_correction, bool):
        raise TraceSemiringError("cardinality_correction must be a bool")
    if not isinstance(return_marginals, bool) or not isinstance(create_graph, bool):
        raise TraceSemiringError("marginal flags must be bools")
    root, support_energy = _validate_energies(root_energy, support_energy)
    batch, height, width = root.shape
    support_mask, root_mask = _normalize_masks(
        valid_support_mask,
        valid_root_mask,
        batch=batch,
        height=height,
        width=width,
        device=root.device,
    )
    chains_by_batch = enumerate_run_chains(
        support_mask, root_mask, max_states=max_states
    )
    counts = torch.tensor(
        [len(chains) for chains in chains_by_batch],
        dtype=torch.long,
        device=root.device,
    )
    reference_logK = torch.log(counts.to(dtype=root.dtype))
    if log_cardinality is None:
        logK = reference_logK
    else:
        logK = _normalize_log_cardinality(
            log_cardinality,
            batch=batch,
            dtype=root.dtype,
            device=root.device,
        )
    correction = logK if cardinality_correction else torch.zeros_like(logK)

    logZ_positive_values: list[torch.Tensor] = []
    map_energy_values: list[torch.Tensor] = []
    map_roots: list[tuple[int, int]] = []
    map_interval_values: list[torch.Tensor] = []
    map_support_values: list[torch.Tensor] = []
    for n, chains in enumerate(chains_by_batch):
        energies: list[torch.Tensor] = []
        for chain in chains:
            root_y, root_x = chain.root
            energy = root[n, root_y, root_x] - correction[n]
            for y, left, right in chain.intervals:
                energy = energy + support_energy[n, y, left : right + 1].sum()
            energies.append(energy)
        stacked = torch.stack(energies)
        logZ_positive_values.append(torch.logsumexp(stacked, dim=0))
        map_energy, map_index = torch.max(stacked, dim=0)
        chain = chains[int(map_index.item())]
        interval_tensor = torch.full(
            (height, 2), -1, dtype=torch.long, device=root.device
        )
        support_tensor = torch.zeros(
            (height, width), dtype=torch.bool, device=root.device
        )
        for y, left, right in chain.intervals:
            interval_tensor[y] = torch.tensor(
                (left, right), dtype=torch.long, device=root.device
            )
            support_tensor[y, left : right + 1] = True
        map_energy_values.append(map_energy)
        map_roots.append(chain.root)
        map_interval_values.append(interval_tensor)
        map_support_values.append(support_tensor)

    logZ_positive = torch.stack(logZ_positive_values)
    logZ_total = torch.logaddexp(torch.zeros_like(logZ_positive), logZ_positive)
    map_energy = torch.stack(map_energy_values)
    root_marginal: torch.Tensor | None = None
    support_marginal: torch.Tensor | None = None
    if return_marginals:
        if not root.requires_grad or not support_energy.requires_grad:
            raise TraceSemiringError(
                "brute-force marginals require both energy tensors to require gradients"
            )
        root_marginal, support_marginal = torch.autograd.grad(
            logZ_total.sum(),
            (root, support_energy),
            create_graph=create_graph,
            retain_graph=True,
        )
    return TraceSemiringOutput(
        logZ_positive=logZ_positive,
        logZ_total=logZ_total,
        p_nonempty=torch.sigmoid(logZ_positive),
        log_cardinality=reference_logK,
        cardinality=counts.to(dtype=torch.float64),
        correction=correction,
        map_energy=map_energy,
        map_log_joint_posterior=map_energy - logZ_total,
        map_root=torch.tensor(map_roots, dtype=torch.long, device=root.device),
        map_intervals=torch.stack(map_interval_values),
        map_support=torch.stack(map_support_values),
        root_marginal=root_marginal,
        support_marginal=support_marginal,
        state_count=counts,
    )


class RootCellRunSemiring(nn.Module):
    """Stateless exact solver with an optional cached cardinality correction."""

    def __init__(
        self,
        *,
        cardinality_correction: bool = True,
        log_cardinality: torch.Tensor | float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(cardinality_correction, bool):
            raise TraceSemiringError("cardinality_correction must be a bool")
        self.cardinality_correction = cardinality_correction
        if log_cardinality is None:
            self.register_buffer("_cached_log_cardinality", None, persistent=True)
        else:
            cached = torch.as_tensor(log_cardinality)
            if not cached.is_floating_point():
                cached = cached.to(dtype=torch.float64)
            if cached.ndim > 1 or not bool(torch.isfinite(cached).all()):
                raise TraceSemiringError(
                    "cached log_cardinality must be a finite scalar or vector"
                )
            self.register_buffer("_cached_log_cardinality", cached, persistent=True)

    def _cardinality_argument(
        self, value: torch.Tensor | float | None
    ) -> torch.Tensor | float | None:
        return self._cached_log_cardinality if value is None else value

    def sum_product(
        self,
        root_energy: torch.Tensor,
        support_energy: torch.Tensor,
        valid_support_mask: torch.Tensor | None = None,
        valid_root_mask: torch.Tensor | None = None,
        *,
        log_cardinality: torch.Tensor | float | None = None,
        cardinality_correction: bool | None = None,
    ) -> TracePartitionResult:
        correction_enabled = (
            self.cardinality_correction
            if cardinality_correction is None
            else cardinality_correction
        )
        return exact_sum_product(
            root_energy,
            support_energy,
            valid_support_mask,
            valid_root_mask,
            log_cardinality=self._cardinality_argument(log_cardinality),
            cardinality_correction=correction_enabled,
        )

    def map(
        self,
        root_energy: torch.Tensor,
        support_energy: torch.Tensor,
        valid_support_mask: torch.Tensor | None = None,
        valid_root_mask: torch.Tensor | None = None,
        *,
        log_cardinality: torch.Tensor | float | None = None,
        cardinality_correction: bool | None = None,
    ) -> TraceMAPResult:
        correction_enabled = (
            self.cardinality_correction
            if cardinality_correction is None
            else cardinality_correction
        )
        return exact_max_semiring(
            root_energy,
            support_energy,
            valid_support_mask,
            valid_root_mask,
            log_cardinality=self._cardinality_argument(log_cardinality),
            cardinality_correction=correction_enabled,
        )

    def marginals(
        self,
        root_energy: torch.Tensor,
        support_energy: torch.Tensor,
        valid_support_mask: torch.Tensor | None = None,
        valid_root_mask: torch.Tensor | None = None,
        *,
        log_cardinality: torch.Tensor | float | None = None,
        cardinality_correction: bool | None = None,
        create_graph: bool = False,
    ) -> TraceMarginalResult:
        correction_enabled = (
            self.cardinality_correction
            if cardinality_correction is None
            else cardinality_correction
        )
        return exact_marginals(
            root_energy,
            support_energy,
            valid_support_mask,
            valid_root_mask,
            log_cardinality=self._cardinality_argument(log_cardinality),
            cardinality_correction=correction_enabled,
            create_graph=create_graph,
        )

    def log_cardinality(
        self,
        valid_support_mask: torch.Tensor,
        valid_root_mask: torch.Tensor | None = None,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        return zero_score_log_cardinality(
            valid_support_mask, valid_root_mask, dtype=dtype
        )

    def cardinality(
        self,
        valid_support_mask: torch.Tensor,
        valid_root_mask: torch.Tensor | None = None,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Materialize tiny/diagnostic K; production code should retain log K."""

        return zero_score_cardinality(
            valid_support_mask, valid_root_mask, dtype=dtype
        )

    def forward(
        self,
        root_energy: torch.Tensor,
        support_energy: torch.Tensor,
        valid_support_mask: torch.Tensor | None = None,
        valid_root_mask: torch.Tensor | None = None,
        *,
        log_cardinality: torch.Tensor | float | None = None,
        cardinality_correction: bool | None = None,
        return_map: bool = True,
        return_marginals: bool = False,
        create_graph: bool = False,
    ) -> TraceSemiringOutput:
        """Run exact inference for a batch of local fields.

        The emitted MAP atom is always the best *non-empty* atom.  Its
        ``map_log_joint_posterior = map_energy - logZ_total`` is the calibrated
        score for accepting or rejecting that whole atom; thresholding never
        changes its internal support.
        """

        if not isinstance(return_map, bool) or not isinstance(return_marginals, bool):
            raise TraceSemiringError("return flags must be bools")
        if return_marginals:
            # Marginals and the reported partition are intentionally obtained
            # from one shared empty-inclusive logZ autograd graph.
            correction_enabled = (
                self.cardinality_correction
                if cardinality_correction is None
                else cardinality_correction
            )
            partition, marginal = _partition_and_marginals(
                root_energy,
                support_energy,
                valid_support_mask,
                valid_root_mask,
                log_cardinality=self._cardinality_argument(log_cardinality),
                cardinality_correction=correction_enabled,
                create_graph=create_graph,
            )
        else:
            marginal = None
            partition = self.sum_product(
                root_energy,
                support_energy,
                valid_support_mask,
                valid_root_mask,
                log_cardinality=log_cardinality,
                cardinality_correction=cardinality_correction,
            )
        maximum = (
            self.map(
                root_energy,
                support_energy,
                valid_support_mask,
                valid_root_mask,
                # Reuse the exact K already resolved by the sum semiring;
                # without a cache this avoids a second zero-score DP.
                log_cardinality=partition.log_cardinality,
                cardinality_correction=cardinality_correction,
            )
            if return_map
            else None
        )
        map_energy = None if maximum is None else maximum.map_energy
        return TraceSemiringOutput(
            logZ_positive=partition.logZ_positive,
            logZ_total=partition.logZ_total,
            p_nonempty=partition.p_nonempty,
            log_cardinality=partition.log_cardinality,
            cardinality=partition.cardinality,
            correction=partition.correction,
            map_energy=map_energy,
            map_log_joint_posterior=(
                None if map_energy is None else map_energy - partition.logZ_total
            ),
            map_root=None if maximum is None else maximum.map_root,
            map_intervals=None if maximum is None else maximum.map_intervals,
            map_support=None if maximum is None else maximum.map_support,
            root_marginal=None if marginal is None else marginal.root,
            support_marginal=None if marginal is None else marginal.support,
        )


__all__ = [
    "EnumeratedRunChain",
    "RootCellRunSemiring",
    "TraceMAPResult",
    "TraceMarginalResult",
    "TracePartitionResult",
    "TraceSemiringError",
    "TraceSemiringOutput",
    "brute_force_reference",
    "enumerate_run_chains",
    "exact_marginals",
    "exact_max_semiring",
    "exact_sum_product",
    "zero_score_cardinality",
    "zero_score_log_cardinality",
]
