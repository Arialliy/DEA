"""TRACE-MSHNet: a frozen MSHNet d0 front and one exact atomic field.

This module is deliberately narrow.  The canonical MSHNet prediction tail
(``output_0`` ... ``output_3`` and ``final``) is not instantiated.  A single
306-parameter pointwise map instead supplies the two natural coordinates of
one normalized empty-or-run-chain variable per root cell.

The implementation keeps the distinction between

* ``p_nonempty = P(Y != empty | D)``; and
* ``exp(map_log_joint_posterior) = P(Y = C_map | D)``.

Only the latter is the posterior probability of the emitted atom and is used
by the atomic renderer.  Substituting the existence probability would
overstate confidence whenever the positive posterior is spread across many
shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model.trace_front import FrozenMSHNetD0
from model.trace_run_semiring import RootCellRunSemiring
from utils.trace_codec import TraceCodecError, coordinates_to_run_chain
from utils.trace_geometry import EncodedTraceTargets, TraceGeometrySpec


TRACE_FIELD_VERSION = "trace_atomic_field_v1"
TRACE_RENDERER_VERSION = "trace_atomic_renderer_joint_map_posterior_v1"


class TraceModelError(ValueError):
    """Raised when model inputs violate the frozen TRACE contract."""


@dataclass(frozen=True)
class TraceFieldOutput:
    """Exact empty-inclusive inference for every owner cell in every image."""

    root_energy: torch.Tensor
    support_energy: torch.Tensor
    logZ_positive: torch.Tensor
    logZ_total: torch.Tensor
    p_nonempty: torch.Tensor
    log_cardinality: torch.Tensor
    map_energy: torch.Tensor | None
    map_log_joint_posterior: torch.Tensor | None
    map_root: torch.Tensor | None
    map_intervals: torch.Tensor | None
    map_support: torch.Tensor | None
    root_marginal: torch.Tensor | None
    support_marginal: torch.Tensor | None
    geometry_sha256: str
    logk_cache_sha256: str

    @property
    def batch_size(self) -> int:
        return int(self.logZ_total.shape[0])

    @property
    def number_of_cells(self) -> int:
        return int(self.logZ_total.shape[1])


@dataclass(frozen=True)
class TraceNLLResult:
    """Exact likelihood loss plus auditable unscaled terms."""

    loss: torch.Tensor
    per_image_sum: torch.Tensor
    positive_energy_sum: torch.Tensor
    positive_count: int
    cell_count: int
    reduction: str


@dataclass(frozen=True)
class TraceRenderedAtoms:
    """Dense compatibility scores made from whole, threshold-invariant atoms."""

    scores: torch.Tensor
    background_score: float
    score_semantics: str = "log P(Y_cell = emitted_MAP_atom | image)"
    threshold_operator: str = "score > threshold"
    threshold_domain: str = "[background_score, +inf)"
    version: str = TRACE_RENDERER_VERSION

    def binary(self, threshold: float | torch.Tensor) -> torch.Tensor:
        """Return the union selected with the repository's strict convention."""

        value = torch.as_tensor(
            threshold, dtype=self.scores.dtype, device=self.scores.device
        )
        if value.numel() != 1 or not bool(torch.isfinite(value)):
            raise TraceModelError("threshold must be one finite scalar")
        background = torch.as_tensor(
            self.background_score,
            dtype=self.scores.dtype,
            device=self.scores.device,
        )
        if not bool(torch.isfinite(background)):
            raise TraceModelError("background_score must be finite")
        if bool(value < background):
            raise TraceModelError(
                "threshold must be greater than or equal to background_score"
            )
        return self.scores > value


def _tensor_bytes(value: torch.Tensor) -> bytes:
    array = value.detach().cpu().contiguous().numpy()
    return array.tobytes(order="C")


def _source_sha256(obj: object) -> str:
    source = inspect.getsourcefile(obj)
    if source is None:
        raise TraceModelError("cannot bind logK cache to semiring source")
    try:
        payload = Path(source).read_bytes()
    except OSError as exc:  # pragma: no cover - packaging/environment failure
        raise TraceModelError("cannot read semiring source for cache binding") from exc
    return hashlib.sha256(payload).hexdigest()


def _boundary_patterns(
    spec: TraceGeometrySpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the few distinct boundary masks without a full int64 grid."""

    patterns: list[tuple[int, int, int]] = []
    pattern_index: dict[tuple[int, int, int], int] = {}
    cell_pattern: list[int] = []
    for cell in range(spec.number_of_cells):
        row, column = spec.cell_coordinates(cell)
        valid_height = min(
            spec.local_height, spec.image_height - row * spec.cell_size
        )
        valid_left = max(0, spec.left_radius - column * spec.cell_size)
        valid_right = min(
            spec.local_width,
            spec.left_radius + spec.image_width - column * spec.cell_size,
        )
        key = (valid_height, valid_left, valid_right)
        if key not in pattern_index:
            pattern_index[key] = len(patterns)
            patterns.append(key)
        cell_pattern.append(pattern_index[key])

    support = torch.zeros(
        (len(patterns), spec.local_height, spec.local_width), dtype=torch.bool
    )
    for index, (height, left, right) in enumerate(patterns):
        support[index, :height, left:right] = True
    root = torch.zeros_like(support)
    top, left, bottom, right = spec.core_local_bounds
    root[:, top:bottom, left:right] = True
    root &= support
    mapping = torch.tensor(cell_pattern, dtype=torch.long)

    # This is a contract assertion against the public geometry implementation,
    # not a repair path.  It catches either side changing boundary semantics.
    expected_support = spec.valid_support_mask(device="cpu")
    expected_root = spec.valid_root_mask(device="cpu")
    if not torch.equal(support[mapping], expected_support):
        raise TraceModelError("boundary support patterns disagree with geometry spec")
    if not torch.equal(root[mapping], expected_root):
        raise TraceModelError("boundary root patterns disagree with geometry spec")
    return support, root, mapping


class TracePotentialMap(nn.Sequential):
    """The only trainable TRACE mapping (16 -> 16 -> two natural fields)."""

    parameter_count = 306

    def __init__(self, positive_cell_prior: float) -> None:
        if isinstance(positive_cell_prior, bool):
            raise TraceModelError("positive_cell_prior must lie strictly in (0, 1)")
        prior = float(positive_cell_prior)
        if not math.isfinite(prior) or not 0.0 < prior < 1.0:
            raise TraceModelError("positive_cell_prior must lie strictly in (0, 1)")
        first = nn.Conv2d(16, 16, kernel_size=1, bias=True)
        second = nn.Conv2d(16, 2, kernel_size=1, bias=True)
        super().__init__(first, nn.GELU(), second)
        nn.init.xavier_normal_(first.weight, gain=1.0e-2)
        nn.init.zeros_(first.bias)
        nn.init.xavier_normal_(second.weight, gain=1.0e-2)
        with torch.no_grad():
            second.bias[0] = math.log(prior / (1.0 - prior))
            second.bias[1] = 0.0
        if sum(parameter.numel() for parameter in self.parameters()) != self.parameter_count:
            raise TraceModelError("TRACE potential-map parameter contract changed")


class MatchedDensePotentialMap(nn.Sequential):
    """307-parameter dense Bernoulli control, matched to TRACE capacity."""

    parameter_count = 307

    def __init__(self, foreground_pixel_prior: float = 0.01) -> None:
        if isinstance(foreground_pixel_prior, bool):
            raise TraceModelError("foreground_pixel_prior must lie strictly in (0, 1)")
        prior = float(foreground_pixel_prior)
        if not math.isfinite(prior) or not 0.0 < prior < 1.0:
            raise TraceModelError("foreground_pixel_prior must lie strictly in (0, 1)")
        first = nn.Conv2d(16, 17, kernel_size=1, bias=True)
        second = nn.Conv2d(17, 1, kernel_size=1, bias=True)
        super().__init__(first, nn.GELU(), second)
        nn.init.xavier_normal_(first.weight, gain=1.0e-2)
        nn.init.zeros_(first.bias)
        nn.init.xavier_normal_(second.weight, gain=1.0e-2)
        nn.init.constant_(second.bias, math.log(prior / (1.0 - prior)))
        if sum(parameter.numel() for parameter in self.parameters()) != self.parameter_count:
            raise TraceModelError("dense-control potential-map parameter contract changed")


class TraceAtomicField(nn.Module):
    """Chunked exact inference over all root-cell-owned local variables."""

    def __init__(
        self,
        geometry: TraceGeometrySpec,
        *,
        field_chunk_size: int = 256,
        cardinality_correction: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(geometry, TraceGeometrySpec):
            raise TraceModelError("geometry must be a TraceGeometrySpec")
        if isinstance(field_chunk_size, bool) or not isinstance(field_chunk_size, int):
            raise TraceModelError("field_chunk_size must be a positive integer")
        if field_chunk_size < 1:
            raise TraceModelError("field_chunk_size must be a positive integer")
        if cardinality_correction is not True:
            raise TraceModelError(
                "production TRACE requires cardinality correction; use the semiring API for ablation"
            )
        self.geometry = geometry
        self.field_chunk_size = field_chunk_size
        self.solver = RootCellRunSemiring(cardinality_correction=True)

        support, root, mapping = _boundary_patterns(geometry)
        with torch.no_grad():
            logk = self.solver.log_cardinality(support, root, dtype=torch.float64)
        if tuple(logk.shape) != (support.shape[0],) or not bool(torch.isfinite(logk).all()):
            raise TraceModelError("failed to construct finite boundary-pattern logK cache")
        self.register_buffer("pattern_support_mask", support, persistent=True)
        self.register_buffer("pattern_root_mask", root, persistent=True)
        self.register_buffer("cell_pattern_index", mapping, persistent=True)
        self.register_buffer("pattern_log_cardinality", logk, persistent=True)

        semiring_sha = _source_sha256(RootCellRunSemiring)
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "field_version": TRACE_FIELD_VERSION,
                    "geometry_sha256": geometry.sha256,
                    "semiring_source_sha256": semiring_sha,
                    "cardinality_correction": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for value in (support, root, mapping, logk):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(_tensor_bytes(value))
        self.logk_cache_sha256 = digest.hexdigest()
        self.semiring_source_sha256 = semiring_sha

    def _validate_natural_fields(
        self, root_energy: torch.Tensor, support_energy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            root_energy.shape[0] if root_energy.ndim == 3 else -1,
            self.geometry.image_height,
            self.geometry.image_width,
        )
        if root_energy.ndim != 3 or tuple(root_energy.shape) != expected:
            raise TraceModelError(
                "root energy must have shape [B, image_height, image_width]"
            )
        if tuple(support_energy.shape) != tuple(root_energy.shape):
            raise TraceModelError("root and support natural fields must have equal shape")
        if root_energy.device != support_energy.device:
            raise TraceModelError("root and support natural fields must share a device")
        if not root_energy.is_floating_point() or root_energy.dtype != support_energy.dtype:
            raise TraceModelError("natural fields must share a floating-point dtype")
        if not bool(torch.isfinite(root_energy).all()) or not bool(
            torch.isfinite(support_energy).all()
        ):
            raise TraceModelError("natural fields must be finite")
        # DP computation is explicitly at least FP32, independent of AMP.
        return root_energy.float(), support_energy.float()

    def _local_indices(
        self, cells: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.geometry
        rows = torch.div(cells, spec.grid_width, rounding_mode="floor")
        columns = torch.remainder(cells, spec.grid_width)
        local_y = torch.arange(spec.local_height, device=cells.device)
        local_x = torch.arange(spec.local_width, device=cells.device)
        global_y = rows[:, None, None] * spec.cell_size + local_y[None, :, None]
        global_x = (
            columns[:, None, None] * spec.cell_size
            - spec.left_radius
            + local_x[None, None, :]
        )
        valid = (
            (global_y >= 0)
            & (global_y < spec.image_height)
            & (global_x >= 0)
            & (global_x < spec.image_width)
        )
        flat = global_y * spec.image_width + global_x
        return flat.clamp(0, spec.image_height * spec.image_width - 1).long(), valid

    def _gather_local(
        self, field: torch.Tensor, cells: torch.Tensor
    ) -> torch.Tensor:
        indices, valid = self._local_indices(cells)
        batch = field.shape[0]
        flat_indices = indices.reshape(1, -1).expand(batch, -1)
        gathered = torch.gather(field.reshape(batch, -1), 1, flat_indices)
        result = gathered.reshape(
            batch, cells.numel(), self.geometry.local_height, self.geometry.local_width
        )
        return torch.where(valid.unsqueeze(0), result, torch.zeros_like(result))

    def forward(
        self,
        root_energy: torch.Tensor,
        support_energy: torch.Tensor,
        *,
        return_map: bool = True,
        return_marginals: bool = False,
        create_graph: bool = False,
    ) -> TraceFieldOutput:
        root_energy, support_energy = self._validate_natural_fields(
            root_energy, support_energy
        )
        if not isinstance(return_map, bool) or not isinstance(return_marginals, bool):
            raise TraceModelError("return_map and return_marginals must be bools")
        batch = root_energy.shape[0]
        cells_total = self.geometry.number_of_cells
        device = root_energy.device

        accumulated: dict[str, list[torch.Tensor]] = {
            "logZ_positive": [],
            "logZ_total": [],
            "p_nonempty": [],
            "log_cardinality": [],
            "map_energy": [],
            "map_log_joint_posterior": [],
            "map_root": [],
            "map_intervals": [],
            "map_support": [],
            "root_marginal": [],
            "support_marginal": [],
        }
        for start in range(0, cells_total, self.field_chunk_size):
            stop = min(cells_total, start + self.field_chunk_size)
            cells = torch.arange(start, stop, dtype=torch.long, device=device)
            count = stop - start
            local_root = self._gather_local(root_energy, cells)
            local_support = self._gather_local(support_energy, cells)
            pattern_ids = self.cell_pattern_index[cells]
            support_mask = self.pattern_support_mask[pattern_ids]
            root_mask = self.pattern_root_mask[pattern_ids]
            logk = self.pattern_log_cardinality[pattern_ids]
            support_mask = support_mask.unsqueeze(0).expand(batch, -1, -1, -1)
            root_mask = root_mask.unsqueeze(0).expand(batch, -1, -1, -1)
            logk = logk.unsqueeze(0).expand(batch, -1)

            device_type = root_energy.device.type
            with torch.autocast(device_type=device_type, enabled=False):
                result = self.solver(
                    local_root.reshape(batch * count, *local_root.shape[-2:]),
                    local_support.reshape(batch * count, *local_support.shape[-2:]),
                    support_mask.reshape(batch * count, *support_mask.shape[-2:]),
                    root_mask.reshape(batch * count, *root_mask.shape[-2:]),
                    log_cardinality=logk.reshape(-1),
                    return_map=return_map,
                    return_marginals=return_marginals,
                    create_graph=create_graph,
                )

            for name in ("logZ_positive", "logZ_total", "p_nonempty", "log_cardinality"):
                accumulated[name].append(getattr(result, name).reshape(batch, count))
            if return_map:
                accumulated["map_energy"].append(result.map_energy.reshape(batch, count))
                accumulated["map_log_joint_posterior"].append(
                    result.map_log_joint_posterior.reshape(batch, count)
                )
                accumulated["map_root"].append(
                    result.map_root.reshape(batch, count, 2)
                )
                accumulated["map_intervals"].append(
                    result.map_intervals.reshape(
                        batch, count, self.geometry.local_height, 2
                    )
                )
                accumulated["map_support"].append(
                    result.map_support.reshape(
                        batch,
                        count,
                        self.geometry.local_height,
                        self.geometry.local_width,
                    )
                )
            if return_marginals:
                for name in ("root_marginal", "support_marginal"):
                    accumulated[name].append(
                        getattr(result, name).reshape(
                            batch,
                            count,
                            self.geometry.local_height,
                            self.geometry.local_width,
                        )
                    )

        def cat(name: str, *, present: bool) -> torch.Tensor | None:
            return torch.cat(accumulated[name], dim=1) if present else None

        return TraceFieldOutput(
            root_energy=root_energy,
            support_energy=support_energy,
            logZ_positive=cat("logZ_positive", present=True),
            logZ_total=cat("logZ_total", present=True),
            p_nonempty=cat("p_nonempty", present=True),
            log_cardinality=cat("log_cardinality", present=True),
            map_energy=cat("map_energy", present=return_map),
            map_log_joint_posterior=cat(
                "map_log_joint_posterior", present=return_map
            ),
            map_root=cat("map_root", present=return_map),
            map_intervals=cat("map_intervals", present=return_map),
            map_support=cat("map_support", present=return_map),
            root_marginal=cat("root_marginal", present=return_marginals),
            support_marginal=cat("support_marginal", present=return_marginals),
            geometry_sha256=self.geometry.sha256,
            logk_cache_sha256=self.logk_cache_sha256,
        )

    def _validate_target(self, target: EncodedTraceTargets) -> None:
        spec = self.geometry
        if not isinstance(target, EncodedTraceTargets):
            raise TraceModelError("targets must be EncodedTraceTargets instances")
        if isinstance(target.number_of_cells, bool) or not isinstance(
            target.number_of_cells, int
        ):
            raise TraceModelError("target cell count must be an integer")
        if target.number_of_cells != spec.number_of_cells:
            raise TraceModelError("target cell count disagrees with geometry")
        if target.positive_cell_indices.ndim != 1:
            raise TraceModelError("positive target cell indices must be one-dimensional")
        if (
            target.root_local_y.dtype != torch.long
            or target.root_local_x.dtype != torch.long
        ):
            raise TraceModelError("positive target root coordinates must be torch.long")
        if tuple(target.support_local.shape[1:]) != (
            spec.local_height,
            spec.local_width,
        ):
            raise TraceModelError("target local support shape disagrees with geometry")
        for index in range(target.positive_count):
            cell = int(target.positive_cell_indices[index].detach().cpu())
            if not 0 <= cell < spec.number_of_cells:
                raise TraceModelError("positive target cell is out of range")
            support = target.support_local[index].detach().cpu().bool()
            y = int(target.root_local_y[index].detach().cpu())
            x = int(target.root_local_x[index].detach().cpu())
            if not (0 <= y < spec.local_height and 0 <= x < spec.local_width):
                raise TraceModelError("positive target root is outside its local field")
            pattern = int(self.cell_pattern_index[cell].detach().cpu())
            valid_support = self.pattern_support_mask[pattern].detach().cpu()
            valid_root = self.pattern_root_mask[pattern].detach().cpu()
            if not bool(support.any()) or bool((support & ~valid_support).any()):
                raise TraceModelError("positive target support is empty or outside its field")
            if not bool(valid_root[y, x]) or not bool(support[y, x]):
                raise TraceModelError("positive target root is illegal or outside its support")
            coordinates = torch.nonzero(support, as_tuple=False).numpy()
            try:
                chain = coordinates_to_run_chain(coordinates)
            except (TraceCodecError, ValueError) as exc:
                raise TraceModelError("positive target is not a legal run chain") from exc
            if chain.root != (y, x):
                raise TraceModelError("positive target root is not the canonical chain root")

    def exact_nll(
        self,
        output: TraceFieldOutput,
        targets: Sequence[EncodedTraceTargets],
        *,
        reduction: str = "mean",
    ) -> TraceNLLResult:
        """Compute the sole TRACE objective over empty and exact positive cells."""

        if not isinstance(output, TraceFieldOutput):
            raise TraceModelError("output must be a TraceFieldOutput")
        if output.geometry_sha256 != self.geometry.sha256:
            raise TraceModelError("output geometry does not belong to this field")
        if output.logk_cache_sha256 != self.logk_cache_sha256:
            raise TraceModelError("output logK cache does not belong to this field")
        if len(targets) != output.batch_size:
            raise TraceModelError("one encoded target is required per image")
        if reduction not in {"mean", "sum"}:
            raise TraceModelError("reduction must be 'mean' or 'sum'")

        partition_per_image = output.logZ_total.sum(dim=1)
        positive_sums: list[torch.Tensor] = []
        positive_count = 0
        for batch_index, target in enumerate(targets):
            self._validate_target(target)
            target = target.to(output.root_energy.device)
            count = target.positive_count
            positive_count += count
            if count == 0:
                positive_sums.append(output.root_energy.new_zeros(()))
                continue
            cells = target.positive_cell_indices
            local_root = self._gather_local(
                output.root_energy[batch_index : batch_index + 1], cells
            )[0]
            local_support = self._gather_local(
                output.support_energy[batch_index : batch_index + 1], cells
            )[0]
            row = torch.arange(count, device=cells.device)
            root_term = local_root[row, target.root_local_y, target.root_local_x]
            support_term = (
                local_support * target.support_local.to(dtype=local_support.dtype)
            ).sum(dim=(-2, -1))
            correction = output.log_cardinality[batch_index, cells]
            positive_energy = root_term + support_term - correction
            positive_sum = positive_energy.sum()
            positive_sums.append(positive_sum)
        per_image = partition_per_image - torch.stack(positive_sums)

        total = per_image.sum()
        cell_count = output.batch_size * self.geometry.number_of_cells
        loss = total / cell_count if reduction == "mean" else total
        return TraceNLLResult(
            loss=loss,
            per_image_sum=per_image,
            positive_energy_sum=torch.stack(positive_sums),
            positive_count=positive_count,
            cell_count=cell_count,
            reduction=reduction,
        )


class TRACEMSHNet(nn.Module):
    """Authenticated frozen ``input -> d0`` plus one exact atomic variable field."""

    def __init__(
        self,
        *,
        baseline_checkpoint: str | Path,
        geometry: TraceGeometrySpec,
        positive_cell_prior: float,
        input_channels: int = 3,
        field_chunk_size: int = 256,
        expected_dataset: str | None = None,
        expected_seed: int | None = None,
        expected_train_split_sha256: str | None = None,
        expected_val_split_sha256: str | None = None,
    ) -> None:
        super().__init__()
        self.front = FrozenMSHNetD0(
            baseline_checkpoint,
            input_channels=input_channels,
            expected_dataset=expected_dataset,
            expected_seed=expected_seed,
            expected_train_split_sha256=expected_train_split_sha256,
            expected_val_split_sha256=expected_val_split_sha256,
        )
        self.potential_map = TracePotentialMap(positive_cell_prior)
        self.field = TraceAtomicField(
            geometry, field_chunk_size=field_chunk_size, cardinality_correction=True
        )
        self.train(self.training)

    @property
    def geometry(self) -> TraceGeometrySpec:
        return self.field.geometry

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def train(self, mode: bool = True) -> "TRACEMSHNet":
        super().train(mode)
        self.front.train(False)
        return self

    def assert_front_integrity(self) -> str:
        return self.front.assert_integrity()

    def forward(
        self,
        image: torch.Tensor,
        *,
        return_map: bool | None = None,
        return_marginals: bool = False,
        create_graph: bool = False,
    ) -> TraceFieldOutput:
        if image.ndim != 4 or tuple(image.shape[-2:]) != (
            self.geometry.image_height,
            self.geometry.image_width,
        ):
            raise TraceModelError("image shape disagrees with frozen TRACE geometry")
        d0 = self.front(image)
        if tuple(d0.shape[1:]) != (
            16,
            self.geometry.image_height,
            self.geometry.image_width,
        ):
            raise TraceModelError("MSHNet d0 shape disagrees with TRACE geometry")
        natural = self.potential_map(d0.float())
        emit_map = (not self.training) if return_map is None else return_map
        return self.field(
            natural[:, 0],
            natural[:, 1],
            return_map=emit_map,
            return_marginals=return_marginals,
            create_graph=create_graph,
        )

    def exact_nll(
        self,
        output: TraceFieldOutput,
        targets: Sequence[EncodedTraceTargets],
        *,
        reduction: str = "mean",
    ) -> TraceNLLResult:
        return self.field.exact_nll(output, targets, reduction=reduction)


class MatchedDenseMSHNet(nn.Module):
    """Same frozen d0 and protocol, with an independent Bernoulli dense control."""

    def __init__(
        self,
        *,
        baseline_checkpoint: str | Path,
        geometry: TraceGeometrySpec,
        foreground_pixel_prior: float = 0.01,
        input_channels: int = 3,
        expected_dataset: str | None = None,
        expected_seed: int | None = None,
        expected_train_split_sha256: str | None = None,
        expected_val_split_sha256: str | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.front = FrozenMSHNetD0(
            baseline_checkpoint,
            input_channels=input_channels,
            expected_dataset=expected_dataset,
            expected_seed=expected_seed,
            expected_train_split_sha256=expected_train_split_sha256,
            expected_val_split_sha256=expected_val_split_sha256,
        )
        self.potential_map = MatchedDensePotentialMap(foreground_pixel_prior)
        self.train(self.training)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def train(self, mode: bool = True) -> "MatchedDenseMSHNet":
        super().train(mode)
        self.front.train(False)
        return self

    def assert_front_integrity(self) -> str:
        return self.front.assert_integrity()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or tuple(image.shape[-2:]) != (
            self.geometry.image_height,
            self.geometry.image_width,
        ):
            raise TraceModelError("image shape disagrees with dense-control geometry")
        d0 = self.front(image)
        logits = self.potential_map(d0.float())
        if tuple(logits.shape[1:]) != (
            1,
            self.geometry.image_height,
            self.geometry.image_width,
        ):
            raise TraceModelError("dense-control logits disagree with geometry")
        return logits

    @staticmethod
    def exact_bernoulli_nll(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if tuple(logits.shape) != tuple(target.shape):
            raise TraceModelError("dense logits and masks must have equal shape")
        if not bool(((target == 0) | (target == 1)).all()):
            raise TraceModelError("dense Bernoulli targets must be exactly binary")
        return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))


def render_trace_atoms(
    output: TraceFieldOutput,
    geometry: TraceGeometrySpec,
    *,
    cell_chunk_size: int = 512,
) -> TraceRenderedAtoms:
    """Render each emitted MAP atom with its *joint* posterior log-score.

    No pixel of an atom is independently scored or thresholded.  Overlapping
    atoms use ``max`` so a dense strict-threshold operation is exactly the
    union of whole atoms whose joint posterior log-score exceeds the same
    threshold.
    """

    if not isinstance(output, TraceFieldOutput) or not isinstance(
        geometry, TraceGeometrySpec
    ):
        raise TraceModelError("renderer requires a TRACE field output and geometry")
    if output.geometry_sha256 != geometry.sha256:
        raise TraceModelError("renderer geometry disagrees with model output")
    if output.map_support is None or output.map_log_joint_posterior is None:
        raise TraceModelError("renderer requires MAP inference output")
    if isinstance(cell_chunk_size, bool) or not isinstance(cell_chunk_size, int) or cell_chunk_size < 1:
        raise TraceModelError("cell_chunk_size must be a positive integer")
    scores = output.map_log_joint_posterior
    if not bool(torch.isfinite(scores).all()):
        raise TraceModelError("MAP joint posterior log-scores must be finite")
    batch, cells = scores.shape
    if cells != geometry.number_of_cells:
        raise TraceModelError("MAP cell dimension disagrees with geometry")
    background = float(torch.finfo(scores.dtype).min / 4.0)
    dense = scores.new_full(
        (batch, geometry.image_height * geometry.image_width), background
    )

    for start in range(0, cells, cell_chunk_size):
        stop = min(cells, start + cell_chunk_size)
        cell_ids = torch.arange(start, stop, device=scores.device)
        rows = torch.div(cell_ids, geometry.grid_width, rounding_mode="floor")
        columns = torch.remainder(cell_ids, geometry.grid_width)
        local_y = torch.arange(geometry.local_height, device=scores.device)
        local_x = torch.arange(geometry.local_width, device=scores.device)
        global_y = rows[:, None, None] * geometry.cell_size + local_y[None, :, None]
        global_x = (
            columns[:, None, None] * geometry.cell_size
            - geometry.left_radius
            + local_x[None, None, :]
        )
        valid = (
            (global_y >= 0)
            & (global_y < geometry.image_height)
            & (global_x >= 0)
            & (global_x < geometry.image_width)
        )
        support = output.map_support[:, start:stop] & valid.unsqueeze(0)
        flat_index = (global_y * geometry.image_width + global_x).clamp(
            0, geometry.image_height * geometry.image_width - 1
        )
        flat_index = flat_index.reshape(1, -1).expand(batch, -1)
        values = scores[:, start:stop, None, None].expand_as(support).reshape(batch, -1)
        values = torch.where(
            support.reshape(batch, -1), values, values.new_full((), background)
        )
        dense.scatter_reduce_(1, flat_index, values, reduce="amax", include_self=True)

    dense = dense.reshape(batch, geometry.image_height, geometry.image_width)
    return TraceRenderedAtoms(scores=dense, background_score=background)


__all__ = [
    "MatchedDenseMSHNet",
    "MatchedDensePotentialMap",
    "TRACE_FIELD_VERSION",
    "TRACE_RENDERER_VERSION",
    "TRACEMSHNet",
    "TraceAtomicField",
    "TraceFieldOutput",
    "TraceModelError",
    "TraceNLLResult",
    "TracePotentialMap",
    "TraceRenderedAtoms",
    "render_trace_atoms",
]
