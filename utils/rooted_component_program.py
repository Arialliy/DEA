"""Canonical rooted programs for finite 8-connected component masks.

The representation is deliberately small and structural.  A component is a
root pixel followed by a sequence of ``(parent, neighbour-offset)`` commands.
Every parent must precede its child and every offset is one of the eight image
neighbours.  Consequently every valid prefix renders an 8-connected support.

This file contains no learned model and no metric surrogate.  It is the codec
used to decide whether a rooted component representation is mechanically
sound and complete enough before a neural decoder is designed around it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Iterable

import numpy as np


# Clockwise order starting at north.  The order is part of the canonical codec
# and must not be changed after the representation gate is frozen.
NEIGHBOUR_OFFSETS_8: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


class RootedComponentProgramError(ValueError):
    """Raised when a mask or rooted component program is invalid."""


@dataclass(frozen=True)
class RootedComponentProgram:
    """One root followed by ancestor-closed 8-neighbour growth commands."""

    root_y: int
    root_x: int
    parent_indices: tuple[int, ...]
    offset_codes: tuple[int, ...]

    @property
    def node_count(self) -> int:
        return 1 + len(self.parent_indices)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["node_count"] = self.node_count
        return payload


def _exact_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise RootedComponentProgramError(f"{name} must be an integer")
    return int(value)


def _binary_component_mask(mask: object) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise RootedComponentProgramError(
            "component mask must be a non-empty two-dimensional array"
        )
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise RootedComponentProgramError("component mask must be numeric")
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise RootedComponentProgramError("component mask must be numeric") from exc
    if not bool(np.all(finite)) or not bool(np.all((array == 0) | (array == 1))):
        raise RootedComponentProgramError("component mask must be exactly binary")
    result = np.asarray(array, dtype=np.bool_)
    if not bool(result.any()):
        raise RootedComponentProgramError("component mask cannot be empty")
    return result


def canonical_component_root(mask: object) -> tuple[int, int]:
    """Choose the pixel nearest the component centroid, then lexicographically."""

    component = _binary_component_mask(mask)
    coordinates = np.argwhere(component)
    centroid = coordinates.astype(np.float64).mean(axis=0)
    squared_distance = np.sum((coordinates - centroid[None, :]) ** 2, axis=1)
    order = np.lexsort(
        (
            coordinates[:, 1],
            coordinates[:, 0],
            squared_distance,
        )
    )
    root = coordinates[int(order[0])]
    return int(root[0]), int(root[1])


def encode_rooted_component(mask: object) -> RootedComponentProgram:
    """Encode one 8-connected binary component with deterministic BFS.

    The returned program contains exactly one node per foreground pixel.  A
    disconnected input fails closed rather than silently encoding one island.
    """

    component = _binary_component_mask(mask)
    height, width = component.shape
    root = canonical_component_root(component)
    positions: list[tuple[int, int]] = [root]
    index_by_position = {root: 0}
    parent_indices: list[int] = []
    offset_codes: list[int] = []
    cursor = 0
    while cursor < len(positions):
        y, x = positions[cursor]
        for offset_code, (dy, dx) in enumerate(NEIGHBOUR_OFFSETS_8):
            child = (y + dy, x + dx)
            child_y, child_x = child
            if (
                child_y < 0
                or child_y >= height
                or child_x < 0
                or child_x >= width
                or not component[child_y, child_x]
                or child in index_by_position
            ):
                continue
            index_by_position[child] = len(positions)
            positions.append(child)
            parent_indices.append(cursor)
            offset_codes.append(offset_code)
        cursor += 1

    foreground_count = int(component.sum())
    if len(positions) != foreground_count:
        raise RootedComponentProgramError(
            "component mask must contain exactly one 8-connected component"
        )
    return RootedComponentProgram(
        root_y=root[0],
        root_x=root[1],
        parent_indices=tuple(parent_indices),
        offset_codes=tuple(offset_codes),
    )


def program_positions(
    program: RootedComponentProgram,
    *,
    shape: tuple[int, int] | None = None,
    allow_duplicate_positions: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Validate and expand a rooted program into its ordered pixel positions."""

    if not isinstance(program, RootedComponentProgram):
        raise TypeError("program must be a RootedComponentProgram")
    root_y = _exact_int(program.root_y, name="root_y")
    root_x = _exact_int(program.root_x, name="root_x")
    parents = tuple(
        _exact_int(value, name=f"parent_indices[{index}]")
        for index, value in enumerate(program.parent_indices)
    )
    offsets = tuple(
        _exact_int(value, name=f"offset_codes[{index}]")
        for index, value in enumerate(program.offset_codes)
    )
    if len(parents) != len(offsets):
        raise RootedComponentProgramError(
            "parent_indices and offset_codes must have equal length"
        )
    if shape is not None:
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, Integral) for value in shape)
            or int(shape[0]) <= 0
            or int(shape[1]) <= 0
        ):
            raise RootedComponentProgramError("shape must contain two positive integers")
        height, width = int(shape[0]), int(shape[1])
    else:
        height = width = None

    positions: list[tuple[int, int]] = [(root_y, root_x)]
    occupied = {positions[0]}
    if height is not None and not (0 <= root_y < height and 0 <= root_x < width):
        raise RootedComponentProgramError("program root lies outside the output shape")
    for command_index, (parent, offset_code) in enumerate(zip(parents, offsets)):
        child_index = command_index + 1
        if parent < 0 or parent >= child_index:
            raise RootedComponentProgramError(
                f"parent_indices[{command_index}] must lie in [0, {child_index})"
            )
        if offset_code < 0 or offset_code >= len(NEIGHBOUR_OFFSETS_8):
            raise RootedComponentProgramError(
                f"offset_codes[{command_index}] is not an 8-neighbour code"
            )
        parent_y, parent_x = positions[parent]
        dy, dx = NEIGHBOUR_OFFSETS_8[offset_code]
        child = (parent_y + dy, parent_x + dx)
        if height is not None and not (
            0 <= child[0] < height and 0 <= child[1] < width
        ):
            raise RootedComponentProgramError(
                f"program node {child_index} lies outside the output shape"
            )
        if child in occupied and not allow_duplicate_positions:
            raise RootedComponentProgramError(
                f"program node {child_index} duplicates an earlier position"
            )
        positions.append(child)
        occupied.add(child)
    return tuple(positions)


def render_rooted_component(
    program: RootedComponentProgram,
    shape: tuple[int, int],
    *,
    allow_duplicate_positions: bool = False,
) -> np.ndarray:
    """Render a validated program as a binary support mask."""

    positions = program_positions(
        program,
        shape=shape,
        allow_duplicate_positions=allow_duplicate_positions,
    )
    result = np.zeros(shape, dtype=np.bool_)
    rows, columns = zip(*positions)
    result[np.asarray(rows), np.asarray(columns)] = True
    return result


def truncate_rooted_component(
    program: RootedComponentProgram,
    max_nodes: int,
) -> RootedComponentProgram:
    """Return an ancestor-closed prefix with at most ``max_nodes`` pixels."""

    max_nodes = _exact_int(max_nodes, name="max_nodes")
    if max_nodes < 1:
        raise RootedComponentProgramError("max_nodes must be positive")
    keep_commands = min(len(program.parent_indices), max_nodes - 1)
    truncated = RootedComponentProgram(
        root_y=program.root_y,
        root_x=program.root_x,
        parent_indices=program.parent_indices[:keep_commands],
        offset_codes=program.offset_codes[:keep_commands],
    )
    # Validate the prefix without imposing an image shape.  Every retained
    # parent must still precede its child by construction.
    program_positions(truncated)
    return truncated


def programs_node_counts(
    components: Iterable[RootedComponentProgram],
) -> tuple[int, ...]:
    """Validate an iterable of programs and return their node counts."""

    result = []
    for program in components:
        program_positions(program)
        result.append(program.node_count)
    return tuple(result)


__all__ = [
    "NEIGHBOUR_OFFSETS_8",
    "RootedComponentProgram",
    "RootedComponentProgramError",
    "canonical_component_root",
    "encode_rooted_component",
    "program_positions",
    "programs_node_counts",
    "render_rooted_component",
    "truncate_rooted_component",
]
