"""Fail-closed ground-truth codec for TRACE atomic components.

TRACE v1 represents one connected component as a consecutive chain of one
horizontal run per row.  This module deliberately keeps the representation
small and explicit: it never fills holes, takes a row hull, drops pixels, or
merges components to make an annotation fit the model family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from skimage import measure


class TraceCodecError(ValueError):
    """A mask cannot be represented by the frozen TRACE-v1 atom family."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, order=True)
class HorizontalRun:
    """Inclusive horizontal interval on one image row."""

    y: int
    left: int
    right: int

    def __post_init__(self) -> None:
        if min(self.y, self.left, self.right) < 0:
            raise ValueError("run coordinates must be non-negative")
        if self.left > self.right:
            raise ValueError("run left endpoint exceeds right endpoint")

    @property
    def length(self) -> int:
        return self.right - self.left + 1


@dataclass(frozen=True)
class RunChain:
    """Canonical TRACE-v1 encoding of one non-empty component."""

    runs: tuple[HorizontalRun, ...]

    def __post_init__(self) -> None:
        if not self.runs:
            raise ValueError("a run chain must be non-empty")
        for previous, current in zip(self.runs, self.runs[1:]):
            if current.y != previous.y + 1:
                raise ValueError("run-chain rows must be consecutive")
            if current.left > previous.right + 1 or current.right < previous.left - 1:
                raise ValueError("adjacent runs are not 8-connected")

    @property
    def root(self) -> tuple[int, int]:
        first = self.runs[0]
        return first.y, first.left

    @property
    def area(self) -> int:
        return sum(run.length for run in self.runs)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return ``(top, left, bottom_exclusive, right_exclusive)``."""

        return (
            self.runs[0].y,
            min(run.left for run in self.runs),
            self.runs[-1].y + 1,
            max(run.right for run in self.runs) + 1,
        )

    @property
    def relative_extents(self) -> tuple[int, int, int, int]:
        """Return root-relative ``(up, down, left, right)`` extents."""

        root_y, root_x = self.root
        return (
            root_y - self.runs[0].y,
            self.runs[-1].y - root_y,
            root_x - min(run.left for run in self.runs),
            max(run.right for run in self.runs) - root_x,
        )


@dataclass(frozen=True)
class ComponentRecord:
    """One 8-connected component and its exact TRACE codec status."""

    label: int
    area: int
    bbox: tuple[int, int, int, int]
    root: tuple[int, int]
    chain: RunChain | None
    error_code: str | None
    error_message: str | None
    max_runs_per_row: int

    @property
    def exact(self) -> bool:
        return self.chain is not None


def as_binary_numpy(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    """Normalize a mask to a contiguous two-dimensional boolean array."""

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("mask must be two-dimensional after singleton removal")
    if not np.isfinite(array).all():
        raise ValueError("mask contains NaN or Inf")
    return np.ascontiguousarray(array > 0)


def _runs_in_row(xs: np.ndarray) -> tuple[tuple[int, int], ...]:
    if xs.size == 0:
        return ()
    ordered = np.sort(xs.astype(np.int64, copy=False))
    breaks = np.flatnonzero(np.diff(ordered) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, ordered.size - 1]
    return tuple((int(ordered[start]), int(ordered[end])) for start, end in zip(starts, ends))


def coordinates_to_run_chain(coordinates: np.ndarray) -> RunChain:
    """Encode component coordinates without altering a single support pixel."""

    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        raise ValueError("coordinates must have shape [nonzero, 2]")
    if not np.issubdtype(coords.dtype, np.integer):
        if not np.equal(coords, np.floor(coords)).all():
            raise ValueError("component coordinates must be integral")
        coords = coords.astype(np.int64)
    if (coords < 0).any():
        raise ValueError("component coordinates must be non-negative")

    rows = np.unique(coords[:, 0])
    if int(rows[-1]) - int(rows[0]) + 1 != rows.size:
        raise TraceCodecError(
            "nonconsecutive_rows",
            "component has an empty row between non-empty rows",
        )

    runs: list[HorizontalRun] = []
    for y in rows:
        row_runs = _runs_in_row(coords[coords[:, 0] == y, 1])
        if len(row_runs) != 1:
            raise TraceCodecError(
                "multiple_runs_per_row",
                f"row {int(y)} contains {len(row_runs)} disjoint foreground runs",
            )
        left, right = row_runs[0]
        runs.append(HorizontalRun(int(y), left, right))

    try:
        return RunChain(tuple(runs))
    except ValueError as exc:
        raise TraceCodecError("disconnected_adjacent_runs", str(exc)) from exc


def run_chain_to_mask(
    chain: RunChain,
    shape: Sequence[int],
    *,
    dtype: np.dtype | type = np.bool_,
) -> np.ndarray:
    """Render a run chain; raises instead of clipping out-of-bounds support."""

    if len(shape) != 2:
        raise ValueError("shape must contain height and width")
    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1:
        raise ValueError("shape dimensions must be positive")
    output = np.zeros((height, width), dtype=dtype)
    for run in chain.runs:
        if run.y >= height or run.right >= width:
            raise TraceCodecError("window_or_image_clip", "run lies outside the requested canvas")
        output[run.y, run.left : run.right + 1] = 1
    return output


def component_records(mask: np.ndarray | torch.Tensor) -> tuple[ComponentRecord, ...]:
    """Extract 8-connected components and attempt exact TRACE-v1 encoding."""

    binary = as_binary_numpy(mask)
    labels = measure.label(binary, connectivity=2)
    records: list[ComponentRecord] = []
    for region in measure.regionprops(labels):
        coords = region.coords.astype(np.int64, copy=False)
        top_y = int(coords[:, 0].min())
        top_x = int(coords[coords[:, 0] == top_y, 1].min())
        max_runs = max(
            len(_runs_in_row(coords[coords[:, 0] == y, 1]))
            for y in np.unique(coords[:, 0])
        )
        chain: RunChain | None = None
        error_code: str | None = None
        error_message: str | None = None
        try:
            chain = coordinates_to_run_chain(coords)
            decoded = run_chain_to_mask(chain, binary.shape)
            expected = labels == region.label
            if not np.array_equal(decoded, expected):
                raise TraceCodecError(
                    "encode_decode_mismatch",
                    "run-chain decode differs from the source component",
                )
        except TraceCodecError as exc:
            error_code = exc.code
            error_message = str(exc)
        records.append(
            ComponentRecord(
                label=int(region.label),
                area=int(region.area),
                bbox=tuple(int(value) for value in region.bbox),
                root=(top_y, top_x),
                chain=chain,
                error_code=error_code,
                error_message=error_message,
                max_runs_per_row=max_runs,
            )
        )
    return tuple(records)


def assign_root_cell(root: tuple[int, int], cell_size: int) -> tuple[int, int]:
    if isinstance(cell_size, bool) or not isinstance(cell_size, int) or cell_size < 1:
        raise ValueError("cell_size must be a positive integer")
    y, x = root
    if y < 0 or x < 0:
        raise ValueError("root coordinates must be non-negative")
    return y // cell_size, x // cell_size


def root_cell_collisions(
    roots: Iterable[tuple[int, int]],
    cell_size: int,
) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for root in roots:
        grouped.setdefault(assign_root_cell(root, cell_size), []).append(root)
    return {
        cell: tuple(items)
        for cell, items in grouped.items()
        if len(items) > 1
    }


__all__ = [
    "ComponentRecord",
    "HorizontalRun",
    "RunChain",
    "TraceCodecError",
    "as_binary_numpy",
    "assign_root_cell",
    "component_records",
    "coordinates_to_run_chain",
    "root_cell_collisions",
    "run_chain_to_mask",
]
