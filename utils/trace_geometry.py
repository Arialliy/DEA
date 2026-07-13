"""Spatial ownership and local-window contracts for TRACE-MSHNet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
import torch

from utils.trace_codec import RunChain, TraceCodecError, assign_root_cell, component_records


TRACE_GEOMETRY_SPEC_VERSION = "trace_geometry_v1"


@dataclass(frozen=True)
class TraceGeometrySpec:
    """Train-only geometry frozen before TRACE optimization or evaluation."""

    image_height: int
    image_width: int
    cell_size: int
    max_down: int
    max_left: int
    max_right: int
    margin: int = 1
    version: str = TRACE_GEOMETRY_SPEC_VERSION

    def __post_init__(self) -> None:
        integer_fields = (
            "image_height",
            "image_width",
            "cell_size",
            "max_down",
            "max_left",
            "max_right",
            "margin",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
        if self.image_height < 1 or self.image_width < 1 or self.cell_size < 1:
            raise ValueError("image dimensions and cell_size must be positive")
        if min(self.max_down, self.max_left, self.max_right, self.margin) < 0:
            raise ValueError("extents and margin must be non-negative")
        if self.image_height % self.cell_size or self.image_width % self.cell_size:
            raise ValueError("image dimensions must be divisible by cell_size")
        if self.version != TRACE_GEOMETRY_SPEC_VERSION:
            raise ValueError(f"unsupported TRACE geometry version: {self.version}")

    @property
    def grid_height(self) -> int:
        return self.image_height // self.cell_size

    @property
    def grid_width(self) -> int:
        return self.image_width // self.cell_size

    @property
    def number_of_cells(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def left_radius(self) -> int:
        return self.max_left + self.margin

    @property
    def right_radius(self) -> int:
        return self.max_right + self.margin

    @property
    def down_radius(self) -> int:
        return self.max_down + self.margin

    @property
    def local_height(self) -> int:
        return self.cell_size + self.down_radius

    @property
    def local_width(self) -> int:
        return self.left_radius + self.cell_size + self.right_radius

    @property
    def core_local_bounds(self) -> tuple[int, int, int, int]:
        return (0, self.left_radius, self.cell_size, self.left_radius + self.cell_size)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "grid_height": self.grid_height,
                "grid_width": self.grid_width,
                "number_of_cells": self.number_of_cells,
                "local_height": self.local_height,
                "local_width": self.local_width,
                "core_local_bounds": list(self.core_local_bounds),
            }
        )
        return payload

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceGeometrySpec":
        keys = {
            "image_height",
            "image_width",
            "cell_size",
            "max_down",
            "max_left",
            "max_right",
            "margin",
            "version",
        }
        return cls(**{key: payload[key] for key in keys if key in payload})

    def cell_coordinates(self, cell_index: int) -> tuple[int, int]:
        if not 0 <= cell_index < self.number_of_cells:
            raise IndexError("cell index is outside the geometry grid")
        return divmod(cell_index, self.grid_width)

    def cell_index(self, cell: tuple[int, int]) -> int:
        row, column = cell
        if not (0 <= row < self.grid_height and 0 <= column < self.grid_width):
            raise IndexError("cell coordinates are outside the geometry grid")
        return row * self.grid_width + column

    def window_origin(self, cell: tuple[int, int]) -> tuple[int, int]:
        row, column = cell
        self.cell_index(cell)
        return row * self.cell_size, column * self.cell_size - self.left_radius

    def root_to_cell(self, root: tuple[int, int]) -> tuple[int, int]:
        row, column = assign_root_cell(root, self.cell_size)
        self.cell_index((row, column))
        return row, column

    def root_to_local(self, root: tuple[int, int]) -> tuple[int, int, int]:
        cell = self.root_to_cell(root)
        origin_y, origin_x = self.window_origin(cell)
        local_y, local_x = root[0] - origin_y, root[1] - origin_x
        top, left, bottom, right = self.core_local_bounds
        if not (top <= local_y < bottom and left <= local_x < right):
            raise TraceCodecError("root_outside_core", "canonical root is outside its owner core")
        return self.cell_index(cell), local_y, local_x

    def chain_to_local_mask(self, chain: RunChain) -> tuple[int, int, int, np.ndarray]:
        cell_index, root_y, root_x = self.root_to_local(chain.root)
        cell = self.cell_coordinates(cell_index)
        origin_y, origin_x = self.window_origin(cell)
        support = np.zeros((self.local_height, self.local_width), dtype=np.bool_)
        for run in chain.runs:
            local_y = run.y - origin_y
            local_left = run.left - origin_x
            local_right = run.right - origin_x
            if not (
                0 <= local_y < self.local_height
                and 0 <= local_left <= local_right < self.local_width
            ):
                raise TraceCodecError(
                    "window_coverage",
                    "component support is outside the frozen owner-cell window",
                )
            support[local_y, local_left : local_right + 1] = True
        if int(support.sum()) != chain.area:
            raise TraceCodecError("local_encode_mismatch", "local support lost or duplicated pixels")
        return cell_index, root_y, root_x, support

    def global_index_grid(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        """Return local-window to flattened-image indices, with ``-1`` padding."""

        cell_rows = torch.arange(self.grid_height, device=device).repeat_interleave(
            self.grid_width
        )
        cell_columns = torch.arange(self.grid_width, device=device).repeat(
            self.grid_height
        )
        local_y = torch.arange(self.local_height, device=device)
        local_x = torch.arange(self.local_width, device=device)
        global_y = cell_rows[:, None, None] * self.cell_size + local_y[None, :, None]
        global_x = (
            cell_columns[:, None, None] * self.cell_size
            - self.left_radius
            + local_x[None, None, :]
        )
        valid = (
            (global_y >= 0)
            & (global_y < self.image_height)
            & (global_x >= 0)
            & (global_x < self.image_width)
        )
        flat = global_y * self.image_width + global_x
        return torch.where(valid, flat, torch.full_like(flat, -1)).long()

    def valid_support_mask(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return self.global_index_grid(device=device).ge(0)

    def valid_root_mask(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        mask = torch.zeros(
            (self.number_of_cells, self.local_height, self.local_width),
            dtype=torch.bool,
            device=device,
        )
        top, left, bottom, right = self.core_local_bounds
        mask[:, top:bottom, left:right] = True
        return mask & self.valid_support_mask(device=device)


@dataclass(frozen=True)
class EncodedTraceTargets:
    """Sparse positive-cell targets; every omitted cell is exactly empty."""

    number_of_cells: int
    positive_cell_indices: torch.Tensor
    root_local_y: torch.Tensor
    root_local_x: torch.Tensor
    support_local: torch.Tensor

    def __post_init__(self) -> None:
        count = int(self.positive_cell_indices.numel())
        if self.positive_cell_indices.dtype != torch.long:
            raise TypeError("positive_cell_indices must be torch.long")
        if self.root_local_y.shape != (count,) or self.root_local_x.shape != (count,):
            raise ValueError("root tensors do not match positive-cell count")
        if self.support_local.ndim != 3 or self.support_local.shape[0] != count:
            raise ValueError("support_local must have shape [positive, local_h, local_w]")
        if self.support_local.dtype != torch.bool:
            raise TypeError("support_local must be boolean")
        if count and int(torch.unique(self.positive_cell_indices).numel()) != count:
            raise ValueError("multiple components share one root cell")

    @property
    def positive_count(self) -> int:
        return int(self.positive_cell_indices.numel())

    def to(self, device: torch.device | str) -> "EncodedTraceTargets":
        return EncodedTraceTargets(
            number_of_cells=self.number_of_cells,
            positive_cell_indices=self.positive_cell_indices.to(device),
            root_local_y=self.root_local_y.to(device),
            root_local_x=self.root_local_x.to(device),
            support_local=self.support_local.to(device),
        )


def encode_trace_targets(
    mask: np.ndarray | torch.Tensor,
    spec: TraceGeometrySpec,
    *,
    device: torch.device | str | None = None,
) -> EncodedTraceTargets:
    """Fail closed when any component violates the frozen geometry contract."""

    records = component_records(mask)
    cell_indices: list[int] = []
    root_y: list[int] = []
    root_x: list[int] = []
    supports: list[np.ndarray] = []
    for record in records:
        if record.chain is None:
            raise TraceCodecError(
                record.error_code or "unrepresentable_component",
                record.error_message or "component is outside the TRACE-v1 family",
            )
        cell_index, local_y, local_x, support = spec.chain_to_local_mask(record.chain)
        cell_indices.append(cell_index)
        root_y.append(local_y)
        root_x.append(local_x)
        supports.append(support)
    if len(cell_indices) != len(set(cell_indices)):
        raise TraceCodecError(
            "root_cell_collision",
            "two ground-truth components are owned by the same root cell",
        )
    count = len(cell_indices)
    support_array = (
        np.stack(supports, axis=0)
        if supports
        else np.zeros((0, spec.local_height, spec.local_width), dtype=np.bool_)
    )
    return EncodedTraceTargets(
        number_of_cells=spec.number_of_cells,
        positive_cell_indices=torch.tensor(cell_indices, dtype=torch.long, device=device),
        root_local_y=torch.tensor(root_y, dtype=torch.long, device=device),
        root_local_x=torch.tensor(root_x, dtype=torch.long, device=device),
        support_local=torch.from_numpy(support_array).to(device=device),
    )


__all__ = [
    "EncodedTraceTargets",
    "TRACE_GEOMETRY_SPEC_VERSION",
    "TraceGeometrySpec",
    "encode_trace_targets",
]
