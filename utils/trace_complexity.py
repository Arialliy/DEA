"""Auditable static complexity and diagnostic profiling for TRACE-MSHNet.

The functions in this module deliberately keep three facts separate:

* the status of the train-only T0-A geometry gate;
* exact, geometry-derived operation/storage counts; and
* an optional machine-local runtime budget check.

A NO-GO T0-A report may be inspected for feasibility, but neither a fast
profile nor a direct (unauthenticated) :class:`TraceGeometrySpec` can turn it
into a training authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from pathlib import PureWindowsPath
import platform
import resource
import statistics
import sys
import time
from typing import Any, Callable, Mapping

import torch

from model.MSHNet import MSHNet
from model.trace_mshnet import (
    MatchedDensePotentialMap,
    TraceAtomicField,
    TracePotentialMap,
)
from utils.trace_gates import load_json_report, verify_embedded_report_hash
from utils.trace_geometry import TraceGeometrySpec
from utils.trace_provenance import canonical_json_sha256, sha256_file


TRACE_COMPLEXITY_SCHEMA_VERSION = "trace_complexity_profile_v1"
TRACE_PARAMETER_COUNT = 306
MSHNET_REPLACED_TAIL_PARAMETER_COUNT = 281
MATCHED_DENSE_PARAMETER_COUNT = 307


class TraceComplexityError(RuntimeError):
    """A complexity input or requested measurement is unsafe or malformed."""


@dataclass(frozen=True)
class TraceGeometrySource:
    """Path-free provenance for one geometry used by the profiler."""

    geometry: TraceGeometrySpec
    source_type: str
    t0_a_status: str
    authenticated: bool
    dataset: str | None = None
    report_name: str | None = None
    report_file_sha256: str | None = None
    report_sha256: str | None = None
    train_split_sha256: str | None = None
    criteria: dict[str, bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "authenticated": self.authenticated,
            "t0_a_status": self.t0_a_status,
            "dataset": self.dataset,
            "report_name": self.report_name,
            "report_file_sha256": self.report_file_sha256,
            "report_sha256": self.report_sha256,
            "train_split_sha256": self.train_split_sha256,
            "criteria": self.criteria,
            "geometry_sha256": self.geometry.sha256,
        }


def direct_geometry_source(spec: TraceGeometrySpec) -> TraceGeometrySource:
    """Wrap a direct spec as explicitly unauthenticated diagnostic input."""

    if not isinstance(spec, TraceGeometrySpec):
        raise TraceComplexityError("spec must be a TraceGeometrySpec")
    return TraceGeometrySource(
        geometry=spec,
        source_type="direct_geometry_spec",
        t0_a_status="UNAUTHENTICATED",
        authenticated=False,
    )


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def load_t0_a_geometry_for_profile(path: str | Path) -> TraceGeometrySource:
    """Load an authenticated PASS *or* NO-GO T0-A report for diagnosis.

    This loader is intentionally distinct from ``require_geometry_gate``:
    training must reject NO-GO, whereas feasibility profiling needs to explain
    why a proposed state space is too expensive even when representability
    failed.  Authentication and semantic validation remain fail closed.
    """

    report_path, payload = load_json_report(path)
    try:
        embedded_hash = verify_embedded_report_hash(payload)
    except Exception as exc:
        raise TraceComplexityError("T0-A report authentication failed") from exc
    if payload.get("schema_version") != "trace_t0_a_geometry_report_v1":
        raise TraceComplexityError("unsupported T0-A report schema")
    if payload.get("gate") != "T0-A":
        raise TraceComplexityError("profile input is not a T0-A report")
    status = payload.get("status")
    if status not in {"PASS", "NO-GO"}:
        raise TraceComplexityError("T0-A status must be PASS or NO-GO")
    if payload.get("train_only") is not True:
        raise TraceComplexityError("T0-A report is not declared train-only")

    dataset = payload.get("dataset")
    split = payload.get("train_split")
    criteria = payload.get("criteria")
    if not isinstance(dataset, str) or not dataset.strip():
        raise TraceComplexityError("T0-A report lacks a dataset name")
    if not isinstance(split, dict) or not _is_sha256(split.get("ordered_names_sha256")):
        raise TraceComplexityError("T0-A report lacks an authenticated train split")
    overlap = split.get("canonical_test_overlap")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap != 0:
        raise TraceComplexityError("T0-A report overlaps the canonical test split")
    if (
        not isinstance(criteria, dict)
        or not criteria
        or not all(isinstance(key, str) and isinstance(value, bool) for key, value in criteria.items())
    ):
        raise TraceComplexityError("T0-A criteria must be a non-empty boolean mapping")
    criteria_pass = all(criteria.values())
    if (status == "PASS") != criteria_pass:
        raise TraceComplexityError("T0-A status contradicts its criteria")

    geometry_payload = payload.get("candidate_geometry_spec")
    geometry_sha256 = payload.get("candidate_geometry_sha256")
    if not isinstance(geometry_payload, dict) or not _is_sha256(geometry_sha256):
        raise TraceComplexityError("T0-A report has no profileable candidate geometry")
    try:
        geometry = TraceGeometrySpec.from_dict(geometry_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise TraceComplexityError("T0-A candidate geometry is malformed") from exc
    # Derived fields are part of the report contract too.  Requiring exact
    # equality prevents a signed report with contradictory display metadata.
    if geometry_payload != geometry.to_dict():
        raise TraceComplexityError("T0-A candidate geometry has inconsistent derived fields")
    if geometry.sha256 != geometry_sha256:
        raise TraceComplexityError("T0-A candidate geometry hash is inconsistent")
    if payload.get("selected_cell_size") != geometry.cell_size:
        raise TraceComplexityError("T0-A selected cell size contradicts its geometry")
    resize = payload.get("resize")
    if not isinstance(resize, dict) or (
        resize.get("height") != geometry.image_height
        or resize.get("width") != geometry.image_width
    ):
        raise TraceComplexityError("T0-A resize contract contradicts its geometry")

    return TraceGeometrySource(
        geometry=geometry,
        source_type="authenticated_t0_a_report",
        t0_a_status=status,
        authenticated=True,
        dataset=dataset,
        report_name=report_path.name,
        report_file_sha256=sha256_file(report_path),
        report_sha256=embedded_hash,
        train_split_sha256=split["ordered_names_sha256"],
        criteria=dict(sorted(criteria.items())),
    )


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TraceComplexityError(f"{name} must be a positive integer")
    return value


def _count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def trace_parameter_counts() -> dict[str, Any]:
    """Verify all three claimed trainable-tail capacities from live modules."""

    trace = _count_parameters(TracePotentialMap(positive_cell_prior=0.01))
    dense = _count_parameters(MatchedDensePotentialMap(foreground_pixel_prior=0.01))
    baseline = MSHNet(input_channels=3)
    tail_names = ("output_0", "output_1", "output_2", "output_3", "final")
    per_module = {
        name: _count_parameters(getattr(baseline, name)) for name in tail_names
    }
    old_tail = sum(per_module.values())
    expected = (
        (trace, TRACE_PARAMETER_COUNT, "TRACE"),
        (old_tail, MSHNET_REPLACED_TAIL_PARAMETER_COUNT, "MSHNet tail"),
        (dense, MATCHED_DENSE_PARAMETER_COUNT, "dense control"),
    )
    for actual, declared, label in expected:
        if actual != declared:
            raise TraceComplexityError(
                f"{label} parameter contract changed: expected {declared}, got {actual}"
            )
    return {
        "trace_atomic_potential_map": trace,
        "canonical_mshnet_replaced_tail": old_tail,
        "matched_dense_bernoulli_control": dense,
        "canonical_mshnet_tail_breakdown": per_module,
        "trace_minus_canonical_tail": trace - old_tail,
        "dense_minus_trace": dense - trace,
    }


def _legal_intervals_in_row(row: torch.Tensor) -> int:
    """Count all contiguous all-valid intervals in one arbitrary boolean row."""

    if row.ndim != 1 or row.dtype != torch.bool:
        raise TraceComplexityError("boundary pattern rows must be one-dimensional booleans")
    total = 0
    run = 0
    for value in row.tolist():
        if value:
            run += 1
        elif run:
            total += run * (run + 1) // 2
            run = 0
    if run:
        total += run * (run + 1) // 2
    return total


def _root_start_states(support: torch.Tensor, root: torch.Tensor) -> int:
    """Count legal start intervals ``[left,right]`` whose left is a root."""

    if support.shape != root.shape or support.ndim != 2:
        raise TraceComplexityError("boundary root/support patterns disagree")
    total = 0
    height, width = support.shape
    for y in range(height):
        for left in range(width):
            if not bool(root[y, left]) or not bool(support[y, left]):
                continue
            right = left
            while right < width and bool(support[y, right]):
                total += 1
                right += 1
    return total


def _bytes_payload(
    spec: TraceGeometrySpec,
    *,
    batch_size: int,
    field_chunk_size: int,
    natural_dtype: torch.dtype,
    pattern_count: int,
) -> dict[str, Any]:
    natural_bytes = torch.empty((), dtype=natural_dtype).element_size()
    # TraceAtomicField converts every supported natural field to FP32 before
    # the DP.  This is an implementation fact, not a generic DP assumption.
    semiring_bytes = torch.empty((), dtype=torch.float32).element_size()
    bool_bytes = torch.empty((), dtype=torch.bool).element_size()
    int64_bytes = torch.empty((), dtype=torch.int64).element_size()
    batch = batch_size
    cells = spec.number_of_cells
    chunk = min(field_chunk_size, cells)
    height = spec.local_height
    width = spec.local_width
    image_pixels = spec.image_height * spec.image_width
    local_pixels = height * width
    interval_slots = height * width * width

    global_values = {
        "input_natural_fields_root_plus_support": batch
        * 2
        * image_pixels
        * natural_bytes,
        "partition_scalar_outputs_four_fields": batch * cells * 4 * semiring_bytes,
        "map_support_full_result": batch * cells * local_pixels * bool_bytes,
        "map_intervals_full_result": batch * cells * height * 2 * int64_bytes,
        "map_roots_full_result": batch * cells * 2 * int64_bytes,
        "map_scalar_outputs_energy_plus_log_posterior": batch
        * cells
        * 2
        * semiring_bytes,
        "marginals_full_result_root_plus_support": batch
        * cells
        * local_pixels
        * 2
        * semiring_bytes,
    }
    chunk_values = {
        "local_natural_fields_root_plus_support": batch
        * chunk
        * local_pixels
        * 2
        * semiring_bytes,
        "expanded_support_plus_root_masks": batch
        * chunk
        * local_pixels
        * 2
        * bool_bytes,
        "interval_score": batch * chunk * interval_slots * semiring_bytes,
        "interval_legality": batch * chunk * interval_slots * bool_bytes,
        "one_dp_state_or_transition": batch
        * chunk
        * width
        * width
        * semiring_bytes,
        "map_backtrace_start_flags_stacked": batch
        * chunk
        * interval_slots
        * bool_bytes,
        "map_backtrace_predecessor_indices_stacked": batch
        * chunk
        * interval_slots
        * 2
        * int64_bytes,
        "map_support": batch * chunk * local_pixels * bool_bytes,
        "map_intervals": batch * chunk * height * 2 * int64_bytes,
        "map_roots": batch * chunk * 2 * int64_bytes,
    }
    cache_values = {
        "pattern_support_plus_root_masks": pattern_count
        * local_pixels
        * 2
        * bool_bytes,
        "cell_to_pattern_index": cells * int64_bytes,
        "pattern_log_cardinality_fp64": pattern_count
        * torch.empty((), dtype=torch.float64).element_size(),
    }
    return {
        "assumptions": {
            "kind": "named_dense_tensor_storage_estimate_not_allocator_peak",
            "excludes": [
                "autograd_saved_tensors",
                "temporary_operator_workspaces",
                "framework_allocator_fragmentation",
                "frozen_MSHNet_d0_front",
            ],
            "natural_field_dtype": str(natural_dtype).removeprefix("torch."),
            "natural_field_bytes_per_value": natural_bytes,
            "production_semiring_dtype": "float32",
            "semiring_bytes_per_value": semiring_bytes,
            "bool_bytes_per_value": bool_bytes,
            "int64_bytes_per_value": int64_bytes,
            "batch_size": batch,
            "configured_field_chunk_size": field_chunk_size,
            "peak_chunk_cells": chunk,
        },
        "global_or_full_result_tensors": global_values,
        "per_peak_chunk_tensors": chunk_values,
        "persistent_logk_cache": cache_values,
        "named_bytes_sums": {
            "global_or_full_result_all_optional_outputs": sum(global_values.values()),
            "per_peak_chunk_all_partition_and_map_named_tensors": sum(
                chunk_values.values()
            ),
            "persistent_logk_cache": sum(cache_values.values()),
        },
    }


_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def resolve_dtype(value: str | torch.dtype) -> tuple[str, torch.dtype]:
    if isinstance(value, torch.dtype):
        for name, dtype in _DTYPES.items():
            if value == dtype:
                return name, dtype
        raise TraceComplexityError(f"unsupported floating dtype: {value}")
    if not isinstance(value, str) or value not in _DTYPES:
        raise TraceComplexityError(
            "dtype must be one of float16, bfloat16, float32, or float64"
        )
    return value, _DTYPES[value]


def static_trace_complexity(
    spec: TraceGeometrySpec,
    *,
    batch_size: int = 1,
    field_chunk_size: int = 256,
    natural_dtype: str | torch.dtype = "float32",
) -> dict[str, Any]:
    """Return exact geometry counts and auditable named-tensor byte estimates."""

    if not isinstance(spec, TraceGeometrySpec):
        raise TraceComplexityError("spec must be a TraceGeometrySpec")
    batch = _require_positive_integer(batch_size, "batch_size")
    chunk = _require_positive_integer(field_chunk_size, "field_chunk_size")
    dtype_name, dtype = resolve_dtype(natural_dtype)

    # Constructing the production field makes this audit consume the exact
    # boundary-pattern and logK cache implementation used at train/inference.
    field = TraceAtomicField(spec, field_chunk_size=chunk)
    support_patterns = field.pattern_support_mask.detach().cpu()
    root_patterns = field.pattern_root_mask.detach().cpu()
    mapping = field.cell_pattern_index.detach().cpu()
    logk = field.pattern_log_cardinality.detach().cpu()
    pattern_count = int(support_patterns.shape[0])
    histogram = torch.bincount(mapping, minlength=pattern_count)

    valid_interval_per_pattern = []
    root_start_per_pattern = []
    support_pixels_per_pattern = []
    for index in range(pattern_count):
        support = support_patterns[index]
        root = root_patterns[index]
        valid_interval_per_pattern.append(
            sum(_legal_intervals_in_row(row) for row in support)
        )
        root_start_per_pattern.append(_root_start_states(support, root))
        support_pixels_per_pattern.append(int(support.sum()))
    valid_interval_states = sum(
        int(histogram[index]) * valid_interval_per_pattern[index]
        for index in range(pattern_count)
    )
    valid_root_start_states = sum(
        int(histogram[index]) * root_start_per_pattern[index]
        for index in range(pattern_count)
    )

    cells = spec.number_of_cells
    height = spec.local_height
    width = spec.local_width
    rectangular_interval_states = cells * height * width * (width + 1) // 2
    transition_work_proxy = cells * height * width * width
    if valid_interval_states > rectangular_interval_states:
        raise TraceComplexityError("boundary-valid state count exceeds its upper bound")

    patterns: list[dict[str, Any]] = []
    for index in range(pattern_count):
        support_hash = hashlib.sha256(
            support_patterns[index].contiguous().numpy().tobytes()
        ).hexdigest()
        root_hash = hashlib.sha256(
            root_patterns[index].contiguous().numpy().tobytes()
        ).hexdigest()
        patterns.append(
            {
                "pattern_index": index,
                "cell_count": int(histogram[index]),
                "valid_support_pixels": support_pixels_per_pattern[index],
                "valid_interval_states_per_cell": valid_interval_per_pattern[index],
                "valid_root_start_states_per_cell": root_start_per_pattern[index],
                "log_cardinality": float(logk[index]),
                "support_mask_sha256": support_hash,
                "root_mask_sha256": root_hash,
            }
        )

    return {
        "geometry": spec.to_dict(),
        "geometry_sha256": spec.sha256,
        "counts": {
            "number_of_cells": cells,
            "local_height": height,
            "local_width": width,
            "rectangular_interval_state_upper_bound": rectangular_interval_states,
            "boundary_valid_interval_states_exact": valid_interval_states,
            "boundary_invalid_interval_slots_pruned": rectangular_interval_states
            - valid_interval_states,
            "boundary_valid_root_start_states_exact": valid_root_start_states,
            "transition_work_proxy_N_times_H_times_W_squared": transition_work_proxy,
        },
        "formulae": {
            "rectangular_interval_state_upper_bound": "N*H*W*(W+1)/2",
            "boundary_valid_interval_states_exact": (
                "sum_cells,sum_rows,sum_contiguous_valid_runs r*(r+1)/2"
            ),
            "transition_work_proxy": "N*H*W^2",
            "scope": "one image; multiply operation counts by batch size",
        },
        "log_cardinality_cache": {
            "pattern_count": pattern_count,
            "cell_pattern_histogram": [int(value) for value in histogram],
            "patterns": patterns,
            "cache_sha256": field.logk_cache_sha256,
            "semiring_source_sha256": field.semiring_source_sha256,
            "cardinality_materialization": "logK is primary; K may exceed FP64",
        },
        "trainable_parameter_counts": trace_parameter_counts(),
        "tensor_byte_footprints": _bytes_payload(
            spec,
            batch_size=batch,
            field_chunk_size=chunk,
            natural_dtype=dtype,
            pattern_count=pattern_count,
        ),
        "configuration": {
            "batch_size": batch,
            "field_chunk_size": chunk,
            "natural_dtype": dtype_name,
        },
    }


def _prepare_solver_chunk(
    field: TraceAtomicField,
    root: torch.Tensor,
    support: torch.Tensor,
    cells: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = root.shape[0]
    count = int(cells.numel())
    local_root = field._gather_local(root, cells)
    local_support = field._gather_local(support, cells)
    pattern_ids = field.cell_pattern_index[cells]
    support_mask = field.pattern_support_mask[pattern_ids].unsqueeze(0).expand(
        batch, -1, -1, -1
    )
    root_mask = field.pattern_root_mask[pattern_ids].unsqueeze(0).expand(
        batch, -1, -1, -1
    )
    logk = field.pattern_log_cardinality[pattern_ids].unsqueeze(0).expand(batch, -1)
    local_shape = (batch * count, field.geometry.local_height, field.geometry.local_width)
    return (
        local_root.reshape(local_shape),
        local_support.reshape(local_shape),
        support_mask.reshape(local_shape),
        root_mask.reshape(local_shape),
        logk.reshape(-1),
    )


def _timed_iterations(
    function: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    def process_max_rss_bytes() -> int:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; Darwin reports bytes.  These are the two targets
        # on which CPython exposes this interface.
        return value if sys.platform == "darwin" else value * 1024

    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    rss_before = process_max_rss_bytes()
    for _ in range(warmup):
        function()
    synchronize(device) if device.type == "cuda" else synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for _ in range(iterations):
        synchronize(device) if device.type == "cuda" else synchronize()
        start = time.perf_counter_ns()
        function()
        synchronize(device) if device.type == "cuda" else synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    if not samples or not all(math.isfinite(value) and value >= 0.0 for value in samples):
        raise TraceComplexityError("runtime benchmark produced invalid timings")
    timing = {
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "samples_ms": samples,
    }
    if device.type == "cuda":
        memory: dict[str, int | None] = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "process_lifetime_max_rss_before_bytes": rss_before,
            "process_lifetime_max_rss_after_bytes": process_max_rss_bytes(),
        }
    else:
        memory = {
            "max_memory_allocated_bytes": None,
            "max_memory_reserved_bytes": None,
            "process_lifetime_max_rss_before_bytes": rss_before,
            "process_lifetime_max_rss_after_bytes": process_max_rss_bytes(),
        }
    return timing, memory


def benchmark_trace_atomic_field(
    spec: TraceGeometrySpec,
    *,
    device: str | torch.device,
    batch_size: int = 1,
    field_chunk_size: int = 256,
    benchmark_cells: int | None = None,
    warmup: int = 2,
    iterations: int = 5,
    seed: int = 20260713,
) -> dict[str, Any]:
    """Benchmark partition and max semirings separately on synthetic fields.

    The measured workload may cover a declared prefix of cells to make a
    feasibility probe possible before attempting the full image.  Timings are
    never extrapolated and the exact measured cell count is recorded.
    """

    if not isinstance(spec, TraceGeometrySpec):
        raise TraceComplexityError("spec must be a TraceGeometrySpec")
    batch = _require_positive_integer(batch_size, "batch_size")
    chunk = _require_positive_integer(field_chunk_size, "field_chunk_size")
    count_iterations = _require_positive_integer(iterations, "iterations")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise TraceComplexityError("warmup must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TraceComplexityError("seed must be an integer")
    total_cells = spec.number_of_cells
    if benchmark_cells is None:
        measured_cells = total_cells
    else:
        measured_cells = _require_positive_integer(benchmark_cells, "benchmark_cells")
        if measured_cells > total_cells:
            raise TraceComplexityError("benchmark_cells exceeds geometry cell count")

    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise TraceComplexityError("benchmark device must be cpu or cuda")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise TraceComplexityError("CUDA benchmark requested but CUDA is unavailable")
        if target.index is not None and not 0 <= target.index < torch.cuda.device_count():
            raise TraceComplexityError("CUDA device index is out of range")

    field = TraceAtomicField(spec, field_chunk_size=chunk).to(target).eval()
    generator_device = target.type
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    root = torch.randn(
        (batch, spec.image_height, spec.image_width),
        generator=generator,
        device=target,
        dtype=torch.float32,
    ) * 0.1
    support = torch.randn(
        root.shape,
        generator=generator,
        device=target,
        dtype=torch.float32,
    ) * 0.1

    def chunks() -> list[torch.Tensor]:
        return [
            torch.arange(start, min(measured_cells, start + chunk), device=target)
            for start in range(0, measured_cells, chunk)
        ]

    cell_chunks = chunks()

    @torch.inference_mode()
    def partition() -> None:
        accumulated: list[list[torch.Tensor]] = [[], [], [], []]
        for cells in cell_chunks:
            local_root, local_support, support_mask, root_mask, logk = _prepare_solver_chunk(
                field, root, support, cells
            )
            result = field.solver.sum_product(
                local_root,
                local_support,
                support_mask,
                root_mask,
                log_cardinality=logk,
            )
            for target_list, value in zip(
                accumulated,
                (
                    result.logZ_positive,
                    result.logZ_total,
                    result.p_nonempty,
                    result.log_cardinality,
                ),
                strict=True,
            ):
                target_list.append(value.reshape(batch, -1))
        # Retain and concatenate the complete measured prefix so the peak is
        # not an artificially low "discard every chunk" measurement.
        complete_partition_outputs = [
            torch.cat(values, dim=1) for values in accumulated
        ]
        if len(complete_partition_outputs) != 4:  # pragma: no cover - invariant
            raise TraceComplexityError("partition output accumulation failed")

    @torch.inference_mode()
    def maximum() -> None:
        energies: list[torch.Tensor] = []
        roots: list[torch.Tensor] = []
        intervals: list[torch.Tensor] = []
        supports: list[torch.Tensor] = []
        for cells in cell_chunks:
            local_root, local_support, support_mask, root_mask, logk = _prepare_solver_chunk(
                field, root, support, cells
            )
            result = field.solver.map(
                local_root,
                local_support,
                support_mask,
                root_mask,
                log_cardinality=logk,
            )
            count = int(cells.numel())
            energies.append(result.map_energy.reshape(batch, count))
            roots.append(result.map_root.reshape(batch, count, 2))
            intervals.append(
                result.map_intervals.reshape(batch, count, spec.local_height, 2)
            )
            supports.append(
                result.map_support.reshape(
                    batch, count, spec.local_height, spec.local_width
                )
            )
        complete_map_outputs = [
            torch.cat(energies, dim=1),
            torch.cat(roots, dim=1),
            torch.cat(intervals, dim=1),
            torch.cat(supports, dim=1),
        ]
        if len(complete_map_outputs) != 4:  # pragma: no cover - invariant
            raise TraceComplexityError("MAP output accumulation failed")

    partition_timing, partition_memory = _timed_iterations(
        partition,
        device=target,
        warmup=warmup,
        iterations=count_iterations,
    )
    map_timing, map_memory = _timed_iterations(
        maximum,
        device=target,
        warmup=warmup,
        iterations=count_iterations,
    )
    if target.type == "cuda":
        device_index = target.index if target.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
    else:
        device_index = None
        device_name = platform.processor() or "cpu"
    return {
        "status": "MEASURED_NO_BUDGET",
        "workload": {
            "synthetic_natural_fields": True,
            "field_distribution": "independent Normal(0, 0.1^2)",
            "seed": seed,
            "batch_size": batch,
            "geometry_cells": total_cells,
            "measured_cell_prefix": measured_cells,
            "full_geometry_cells_measured": measured_cells == total_cells,
            "field_chunk_size": chunk,
            "warmup_iterations": warmup,
            "timed_iterations": count_iterations,
            "measurement_scope": (
                "natural-field local gather plus exact semiring; full measured-prefix "
                "outputs are retained and concatenated per iteration"
            ),
            "excluded_from_timing": [
                "frozen_MSHNet_d0_front",
                "306_parameter_potential_map",
                "dense_atomic_renderer",
                "loss_backward",
            ],
        },
        "partition_only": {
            "timing": partition_timing,
            "peak_memory": partition_memory,
        },
        "map_only": {"timing": map_timing, "peak_memory": map_memory},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_type": target.type,
            "device_index": device_index,
            "device_name": device_name,
            "semiring_dtype": "float32",
            "grad_enabled": False,
        },
    }


def evaluate_runtime_budgets(
    benchmark: Mapping[str, Any] | None,
    *,
    max_partition_latency_ms: float | None = None,
    max_map_latency_ms: float | None = None,
    max_peak_memory_mib: float | None = None,
) -> dict[str, Any]:
    """Evaluate all-or-nothing budgets; incomplete declarations cannot PASS."""

    declared_values = {
        "max_partition_latency_ms": max_partition_latency_ms,
        "max_map_latency_ms": max_map_latency_ms,
        "max_peak_memory_mib": max_peak_memory_mib,
    }
    supplied = {key: value is not None for key, value in declared_values.items()}
    if benchmark is None:
        return {
            "status": "NOT_EVALUATED",
            "budgets": declared_values,
            "reason": "benchmark_not_run",
        }
    if not all(supplied.values()):
        return {
            "status": "NOT_EVALUATED",
            "budgets": declared_values,
            "reason": "all_partition_map_and_memory_budgets_are_required_for_PASS",
        }
    workload = benchmark.get("workload")
    if not isinstance(workload, Mapping):
        raise TraceComplexityError("benchmark payload lacks its workload contract")
    if workload.get("full_geometry_cells_measured") is not True:
        return {
            "status": "NOT_EVALUATED",
            "budgets": declared_values,
            "reason": "full_geometry_cells_were_not_measured",
        }
    budgets: dict[str, float] = {}
    for name, raw in declared_values.items():
        if isinstance(raw, bool):
            raise TraceComplexityError(f"{name} must be a positive finite number")
        value = float(raw)  # type: ignore[arg-type]
        if not math.isfinite(value) or value <= 0.0:
            raise TraceComplexityError(f"{name} must be a positive finite number")
        budgets[name] = value
    try:
        partition_ms = float(benchmark["partition_only"]["timing"]["median_ms"])
        map_ms = float(benchmark["map_only"]["timing"]["median_ms"])
        partition_peak = benchmark["partition_only"]["peak_memory"][
            "max_memory_allocated_bytes"
        ]
        map_peak = benchmark["map_only"]["peak_memory"]["max_memory_allocated_bytes"]
        partition_rss = benchmark["partition_only"]["peak_memory"][
            "process_lifetime_max_rss_after_bytes"
        ]
        map_rss = benchmark["map_only"]["peak_memory"][
            "process_lifetime_max_rss_after_bytes"
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise TraceComplexityError("benchmark payload is malformed") from exc
    for name, value in (
        ("partition_median_ms", partition_ms),
        ("map_median_ms", map_ms),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise TraceComplexityError(f"benchmark {name} is invalid")

    def valid_byte_count(value: Any) -> bool:
        return (
            value is None
            or (not isinstance(value, bool) and isinstance(value, int) and value >= 0)
        )

    if not all(
        valid_byte_count(value)
        for value in (partition_peak, map_peak, partition_rss, map_rss)
    ):
        raise TraceComplexityError("benchmark peak-memory values are invalid")
    if partition_peak is not None and map_peak is not None:
        peak_bytes = max(int(partition_peak), int(map_peak))
        memory_metric = "torch_cuda_max_memory_allocated"
    elif partition_rss is not None and map_rss is not None:
        # CPU native tensor allocations are not visible to tracemalloc.
        # Process-lifetime max RSS is conservative and not operation-isolated,
        # so the report names it explicitly instead of pretending it is an
        # allocator-exact PyTorch peak.
        peak_bytes = max(int(partition_rss), int(map_rss))
        memory_metric = "process_lifetime_max_rss_conservative"
    else:
        return {
            "status": "NOT_EVALUATED",
            "budgets": budgets,
            "reason": "peak_memory_unavailable_on_this_device",
        }
    peak_mib = peak_bytes / (1024.0 * 1024.0)
    observed = {
        "partition_median_ms": partition_ms,
        "map_median_ms": map_ms,
        "peak_memory_mib": peak_mib,
        "peak_memory_metric": memory_metric,
    }
    checks = {
        "partition_latency": partition_ms <= budgets["max_partition_latency_ms"],
        "map_latency": map_ms <= budgets["max_map_latency_ms"],
        "peak_memory": peak_mib <= budgets["max_peak_memory_mib"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "budgets": budgets,
        "observed": observed,
        "checks": checks,
    }


def profile_status(source: TraceGeometrySource, budget_gate: Mapping[str, Any]) -> str:
    """Combine statuses without allowing runtime evidence to erase T0-A."""

    runtime_status = budget_gate.get("status")
    if source.t0_a_status == "NO-GO":
        return "DIAGNOSTIC_T0_A_NO_GO"
    if not source.authenticated:
        return "DIAGNOSTIC_UNAUTHENTICATED_GEOMETRY"
    if source.t0_a_status != "PASS":
        return "DIAGNOSTIC_ONLY"
    if runtime_status == "PASS":
        return "PASS"
    if runtime_status == "FAIL":
        return "FAIL"
    return "DIAGNOSTIC_ONLY"


def authenticate_complexity_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add one embedded hash after validating finite, path-free JSON."""

    if not isinstance(payload, Mapping):
        raise TraceComplexityError("complexity report must be a mapping")
    report = dict(payload)
    if "report_sha256" in report:
        raise TraceComplexityError("report_sha256 must be added exactly once")
    try:
        rendered = json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TraceComplexityError("complexity report is not finite JSON") from exc

    def check_path(value: Any) -> None:
        if isinstance(value, str) and (
            Path(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or value.lower().startswith("file://")
        ):
            raise TraceComplexityError("complexity reports must not contain absolute paths")
        if isinstance(value, Mapping):
            for child in value.values():
                check_path(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_path(child)

    check_path(report)
    # Bind the exact canonical payload used by gate consumers.
    digest = canonical_json_sha256(json.loads(rendered))
    report["report_sha256"] = digest
    return report


__all__ = [
    "MATCHED_DENSE_PARAMETER_COUNT",
    "MSHNET_REPLACED_TAIL_PARAMETER_COUNT",
    "TRACE_COMPLEXITY_SCHEMA_VERSION",
    "TRACE_PARAMETER_COUNT",
    "TraceComplexityError",
    "TraceGeometrySource",
    "authenticate_complexity_report",
    "benchmark_trace_atomic_field",
    "direct_geometry_source",
    "evaluate_runtime_budgets",
    "load_t0_a_geometry_for_profile",
    "profile_status",
    "resolve_dtype",
    "static_trace_complexity",
    "trace_parameter_counts",
]
