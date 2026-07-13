#!/usr/bin/env python3
"""Audit TRACE state-space complexity and optionally benchmark exact inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.trace_complexity import (
    TRACE_COMPLEXITY_SCHEMA_VERSION,
    TraceComplexityError,
    authenticate_complexity_report,
    benchmark_trace_atomic_field,
    direct_geometry_source,
    evaluate_runtime_budgets,
    load_t0_a_geometry_for_profile,
    profile_status,
    static_trace_complexity,
)
from utils.trace_geometry import TraceGeometrySpec
from utils.trace_provenance import sha256_file


def _source_inventory() -> dict[str, str]:
    relative = (
        "utils/trace_complexity.py",
        "utils/trace_geometry.py",
        "model/trace_mshnet.py",
        "model/trace_run_semiring.py",
        "tools/profile_trace.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in relative}


def _geometry_from_args(args: argparse.Namespace):
    explicit_names = (
        "image_height",
        "image_width",
        "cell_size",
        "max_down",
        "max_left",
        "max_right",
    )
    explicit = {name: getattr(args, name) for name in explicit_names}
    has_explicit = any(value is not None for value in explicit.values())
    if args.geometry_report:
        if has_explicit:
            raise TraceComplexityError(
                "--geometry-report cannot be combined with explicit geometry fields"
            )
        return load_t0_a_geometry_for_profile(args.geometry_report)
    missing = [name for name, value in explicit.items() if value is None]
    if missing:
        raise TraceComplexityError(
            "provide --geometry-report or every explicit geometry field; missing "
            + ", ".join(missing)
        )
    try:
        spec = TraceGeometrySpec(
            image_height=args.image_height,
            image_width=args.image_width,
            cell_size=args.cell_size,
            max_down=args.max_down,
            max_left=args.max_left,
            max_right=args.max_right,
            margin=args.margin,
        )
    except (TypeError, ValueError) as exc:
        raise TraceComplexityError("explicit geometry is invalid") from exc
    return direct_geometry_source(spec)


def build_report(args: argparse.Namespace) -> dict[str, object]:
    source = _geometry_from_args(args)
    static = static_trace_complexity(
        source.geometry,
        batch_size=args.batch_size,
        field_chunk_size=args.field_chunk_size,
        natural_dtype=args.dtype,
    )
    runtime = None
    if args.benchmark:
        runtime = benchmark_trace_atomic_field(
            source.geometry,
            device=args.device,
            batch_size=args.batch_size,
            field_chunk_size=args.field_chunk_size,
            benchmark_cells=(
                None if args.benchmark_cells == 0 else args.benchmark_cells
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
        )
    budget_gate = evaluate_runtime_budgets(
        runtime,
        max_partition_latency_ms=args.max_partition_latency_ms,
        max_map_latency_ms=args.max_map_latency_ms,
        max_peak_memory_mib=args.max_peak_memory_mib,
    )
    status = profile_status(source, budget_gate)
    report = {
        "schema_version": TRACE_COMPLEXITY_SCHEMA_VERSION,
        "status": status,
        "training_authorized": False,
        "training_authorization_note": (
            "This profile is diagnostic and cannot replace T0-A, T0-B, or the training gate."
        ),
        "geometry_source": source.to_dict(),
        "static_complexity": static,
        "runtime_benchmark": runtime,
        "runtime_budget_gate": budget_gate,
        "source_sha256": _source_inventory(),
    }
    return authenticate_complexity_report(report)


def dump_report(
    report: dict[str, object], path: str | Path, *, overwrite: bool = False
) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise TraceComplexityError(
            f"output already exists (pass --overwrite explicitly): {output.name}"
        )
    try:
        rendered = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise TraceComplexityError("complexity report is not finite JSON") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-report",
        default="",
        help="authenticated T0-A JSON; NO-GO is accepted for diagnosis only",
    )
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--cell-size", type=int)
    parser.add_argument("--max-down", type=int)
    parser.add_argument("--max-left", type=int)
    parser.add_argument("--max-right", type=int)
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--field-chunk-size", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float32",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--benchmark-cells",
        type=int,
        default=0,
        help="measured cell prefix; 0 means the full geometry",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-partition-latency-ms", type=float)
    parser.add_argument("--max-map-latency-ms", type=float)
    parser.add_argument("--max-peak-memory-mib", type=float)
    parser.add_argument("--output", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.benchmark_cells < 0:
            raise TraceComplexityError("--benchmark-cells must be zero or positive")
        if not args.benchmark and args.benchmark_cells != 0:
            raise TraceComplexityError("--benchmark-cells requires --benchmark")
        report = build_report(args)
        if args.output:
            dump_report(report, args.output, overwrite=args.overwrite)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except (TraceComplexityError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover - argparse.error raises SystemExit


if __name__ == "__main__":
    raise SystemExit(main())
