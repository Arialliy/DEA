#!/usr/bin/env python3
"""Run the complete TRACE T0-B integration and engineering gate.

The exact DP/brute-force report is a prerequisite, not a substitute for this
gate.  This command additionally checks the physically headless model, one
real optimization step with an authenticated clean MSHNet front, the atomic
renderer, and the predeclared 256x256 latency/memory budgets.

A T0-A report with ``NO-GO`` may be supplied for diagnostic profiling: its
candidate geometry is still authenticated and useful.  That does not unlock
training; the separate T0-A gate remains mandatory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path, PureWindowsPath
import statistics
import sys
import textwrap
import time
from typing import Any, Iterable, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.MSHNet import MSHNet  # noqa: E402
from model.trace_mshnet import (  # noqa: E402
    MatchedDensePotentialMap,
    TRACEMSHNet,
    TraceAtomicField,
    TracePotentialMap,
    render_trace_atoms,
)
from tools import verify_trace_dp  # noqa: E402
from utils.trace_gates import load_json_report, verify_embedded_report_hash  # noqa: E402
from utils.trace_geometry import EncodedTraceTargets, TraceGeometrySpec  # noqa: E402
from utils.trace_provenance import (  # noqa: E402
    canonical_json_sha256,
    normalize_state_dict_keys,
    provenance_path,
    sha256_file,
)


SCHEMA_VERSION = "trace_t0_b_integration_verification_v1"
GATE = "T0-B-INTEGRATION"
LATENCY_MULTIPLIER_BUDGET = 2.0
ADDED_PEAK_MEMORY_BYTES_BUDGET = 2 * 1024**3
DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 7


class TraceIntegrationVerificationError(RuntimeError):
    """The integrated T0-B contract cannot be authenticated or satisfied."""


def _walk_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_strings(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _assert_no_absolute_paths(payload: object) -> None:
    for value in _walk_strings(payload):
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise TraceIntegrationVerificationError(
                "absolute paths are forbidden in integration reports"
            )


def _source_inventory() -> dict[str, str]:
    relative_paths = (
        "model/trace_front.py",
        "model/trace_mshnet.py",
        "model/trace_run_semiring.py",
        "tools/verify_trace_integration.py",
        "utils/trace_geometry.py",
    )
    return {
        relative: sha256_file(PROJECT_ROOT / relative) for relative in relative_paths
    }


def _require_fresh_source_inventory(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise TraceIntegrationVerificationError(f"{label} source inventory is missing")
    for locator, declared in value.items():
        if not isinstance(locator, str) or not isinstance(declared, str):
            raise TraceIntegrationVerificationError(f"{label} source inventory is malformed")
        path = Path(locator)
        if path.is_absolute() or ".." in path.parts or locator.startswith("repo:"):
            raise TraceIntegrationVerificationError(f"{label} source locator is unsafe")
        resolved = PROJECT_ROOT / path
        if not resolved.is_file() or sha256_file(resolved) != declared:
            raise TraceIntegrationVerificationError(f"{label} source inventory is stale")


def _load_dp_gate(path: str | Path) -> tuple[dict[str, Any], str, str]:
    report_path, payload = load_json_report(path)
    try:
        digest = verify_trace_dp.authenticate_report(payload)
    except verify_trace_dp.TraceDPVerificationError as exc:
        raise TraceIntegrationVerificationError("T0-B-DP report is invalid") from exc
    if payload.get("status") != "PASS" or not all(payload.get("criteria", {}).values()):
        raise TraceIntegrationVerificationError("T0-B-DP did not pass")
    _require_fresh_source_inventory(payload.get("source_sha256"), label="T0-B-DP")
    return payload, digest, sha256_file(report_path)


def _load_geometry_candidate(
    path: str | Path,
) -> tuple[dict[str, Any], TraceGeometrySpec, str, str]:
    report_path, payload = load_json_report(path)
    digest = verify_embedded_report_hash(payload)
    if payload.get("schema_version") != "trace_t0_a_geometry_report_v1":
        raise TraceIntegrationVerificationError("unsupported T0-A report schema")
    if payload.get("gate") != "T0-A" or payload.get("train_only") is not True:
        raise TraceIntegrationVerificationError("geometry candidate is not a train-only T0-A report")
    _require_fresh_source_inventory(payload.get("source_sha256"), label="T0-A")
    geometry_payload = payload.get("candidate_geometry_spec")
    if not isinstance(geometry_payload, dict):
        raise TraceIntegrationVerificationError("T0-A lacks a candidate geometry")
    geometry = TraceGeometrySpec.from_dict(geometry_payload)
    if geometry.sha256 != payload.get("candidate_geometry_sha256"):
        raise TraceIntegrationVerificationError("T0-A candidate geometry hash is invalid")
    return payload, geometry, digest, sha256_file(report_path)


def _empty_target(spec: TraceGeometrySpec) -> EncodedTraceTargets:
    return EncodedTraceTargets(
        number_of_cells=spec.number_of_cells,
        positive_cell_indices=torch.empty(0, dtype=torch.long),
        root_local_y=torch.empty(0, dtype=torch.long),
        root_local_x=torch.empty(0, dtype=torch.long),
        support_local=torch.empty(
            (0, spec.local_height, spec.local_width), dtype=torch.bool
        ),
    )


def _manual_atom_union(
    output: Any,
    geometry: TraceGeometrySpec,
    threshold: float,
) -> torch.Tensor:
    support = output.map_support
    atom_score = output.map_log_joint_posterior
    if support is None or atom_score is None:
        raise TraceIntegrationVerificationError("manual renderer requires MAP output")
    index = geometry.global_index_grid(device=support.device)
    valid = index.ge(0)
    batch = support.shape[0]
    selected = support & valid.unsqueeze(0) & (atom_score > threshold)[..., None, None]
    flat = torch.zeros(
        (batch, geometry.image_height * geometry.image_width),
        dtype=torch.int32,
        device=support.device,
    )
    safe_index = index.clamp_min(0).reshape(1, -1).expand(batch, -1)
    values = selected.reshape(batch, -1).to(torch.int32)
    flat.scatter_add_(1, safe_index, values)
    return flat.reshape(batch, geometry.image_height, geometry.image_width).gt(0)


def _has_no_python_per_cell_loop() -> bool:
    """Allow a fixed-height DP loop and a cell-*chunk* loop, never cell iteration."""

    source = inspect.getsource(TraceAtomicField.forward)
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
            continue
        function = node.iter.func
        if not isinstance(function, ast.Name) or function.id != "range":
            continue
        # ``range(cells_total)`` is forbidden.  Production uses
        # ``range(0, cells_total, field_chunk_size)``.
        if len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Name):
            if node.iter.args[0].id in {"cells_total", "number_of_cells"}:
                return False
    return ".item(" not in source


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latencies_ms(
    function: Any,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[list[float], Any]:
    result = None
    with torch.no_grad():
        for _ in range(warmup):
            if result is not None:
                del result
            result = function()
        _synchronize(device)
        values = []
        for _ in range(repeats):
            if result is not None:
                del result
            start = time.perf_counter()
            result = function()
            _synchronize(device)
            values.append((time.perf_counter() - start) * 1000.0)
    return values, result


def _load_canonical_model(checkpoint: Path, device: torch.device) -> MSHNet:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or "net" not in payload:
        raise TraceIntegrationVerificationError("canonical checkpoint lacks net")
    state = normalize_state_dict_keys(payload["net"])
    enable_dea_lite = any(key.startswith("decidability_head.") for key in state)
    model = MSHNet(3, enable_dea_lite=enable_dea_lite)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TraceIntegrationVerificationError(f"{name} must be a positive integer")
    return value


def run_verification(
    *,
    dp_report: str | Path,
    geometry_report: str | Path,
    baseline_checkpoint: str | Path,
    device: str = "cuda:0",
    field_chunk_size: int = 1024,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Return one authenticated PASS/NO-GO full integration report."""

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "status": "NO-GO",
        "budgets": {
            "new_prediction_latency_le_canonical_total_multiplier": LATENCY_MULTIPLIER_BUDGET,
            "added_peak_memory_bytes": ADDED_PEAK_MEMORY_BYTES_BUDGET,
            "batch_size": 1,
            "input_shape": [1, 3, 256, 256],
            "no_python_per_cell_loop": True,
        },
        "source_sha256": _source_inventory(),
    }
    try:
        chunk = _validate_positive_integer(field_chunk_size, name="field_chunk_size")
        warmup = _validate_positive_integer(warmup, name="warmup")
        repeats = _validate_positive_integer(repeats, name="repeats")
        dp, dp_digest, dp_file_digest = _load_dp_gate(dp_report)
        geometry_gate, geometry, geometry_digest, geometry_file_digest = (
            _load_geometry_candidate(geometry_report)
        )
        if (geometry.image_height, geometry.image_width) != (256, 256):
            raise TraceIntegrationVerificationError("engineering gate requires 256x256 geometry")
        checkpoint = Path(baseline_checkpoint).resolve()
        if not checkpoint.is_file():
            raise TraceIntegrationVerificationError("baseline checkpoint is missing")
        runtime_device = torch.device(device)
        if runtime_device.type != "cuda" or not torch.cuda.is_available():
            raise TraceIntegrationVerificationError("full engineering gate requires CUDA")
        torch.cuda.set_device(runtime_device)
        torch.manual_seed(20260713)
        torch.cuda.manual_seed_all(20260713)

        train_split = geometry_gate.get("train_split")
        expected_train_hash = (
            train_split.get("ordered_names_sha256")
            if isinstance(train_split, Mapping)
            else None
        )
        dataset = geometry_gate.get("dataset")
        model = TRACEMSHNet(
            baseline_checkpoint=checkpoint,
            geometry=geometry,
            positive_cell_prior=0.001,
            field_chunk_size=chunk,
            expected_dataset=dataset if isinstance(dataset, str) else None,
            expected_train_split_sha256=expected_train_hash,
        ).to(runtime_device)
        forbidden = {"output_0", "output_1", "output_2", "output_3", "final"}
        physical_modules = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        no_old_heads = forbidden.isdisjoint(physical_modules)
        trainable_count = model.trainable_parameter_count
        dense_count = MatchedDensePotentialMap.parameter_count

        image = torch.zeros((1, 3, 256, 256), device=runtime_device)
        before_front = model.assert_front_integrity()
        model.train()
        optimizer = torch.optim.AdamW(model.potential_map.parameters(), lr=1.0e-4)
        optimizer.zero_grad(set_to_none=True)
        training_output = model(image, return_map=False)
        nll = model.exact_nll(training_output, [_empty_target(geometry)])
        nll.loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.potential_map.parameters()
            if parameter.grad is not None
        ]
        finite_nonzero_gradient = bool(gradients) and all(
            bool(torch.isfinite(value).all()) for value in gradients
        ) and any(bool(value.ne(0).any()) for value in gradients)
        optimizer.step()
        after_front = model.assert_front_integrity()
        nll_finite = bool(torch.isfinite(nll.loss))

        model.eval()
        with torch.no_grad():
            inference_output = model(image, return_map=True)
            rendered = render_trace_atoms(inference_output, geometry)
        thresholds = [
            rendered.background_score,
            float(torch.median(inference_output.map_log_joint_posterior).item()),
            float(torch.max(inference_output.map_log_joint_posterior).item()),
        ]
        renderer_equal = all(
            bool(
                torch.equal(
                    rendered.binary(threshold),
                    _manual_atom_union(inference_output, geometry, threshold),
                )
            )
            for threshold in thresholds
        )
        support_snapshot = inference_output.map_support.detach().clone()
        for threshold in thresholds:
            rendered.binary(threshold)
        support_invariant = bool(torch.equal(support_snapshot, inference_output.map_support))
        joint_differs_from_existence = bool(
            (
                inference_output.map_log_joint_posterior
                - torch.log(inference_output.p_nonempty)
            ).abs().gt(1.0e-6).any()
        )

        empty_nll_value = float(nll.loss.detach().item())
        del training_output, nll, inference_output, rendered, support_snapshot
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        canonical = _load_canonical_model(checkpoint, runtime_device)
        canonical_times, _ = _latencies_ms(
            lambda: canonical(image, True),
            device=runtime_device,
            warmup=warmup,
            repeats=repeats,
        )
        with torch.no_grad():
            d0 = model.front(image)

        def prediction_end() -> Any:
            natural = model.potential_map(d0.float())
            return model.field(natural[:, 0], natural[:, 1], return_map=True)

        # Warm once before resetting peak memory.  Allocator cache is not
        # counted by max_memory_allocated; live d0/model storage is subtracted.
        with torch.no_grad():
            warm_output = prediction_end()
        _synchronize(runtime_device)
        del warm_output
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime_device)
        base_allocated = torch.cuda.memory_allocated(runtime_device)
        prediction_times, _ = _latencies_ms(
            prediction_end,
            device=runtime_device,
            warmup=warmup,
            repeats=repeats,
        )
        peak_allocated = torch.cuda.max_memory_allocated(runtime_device)
        added_peak = max(0, int(peak_allocated - base_allocated))
        canonical_median = float(statistics.median(canonical_times))
        prediction_median = float(statistics.median(prediction_times))
        latency_limit = LATENCY_MULTIPLIER_BUDGET * canonical_median

        metrics = {
            "canonical_total_latency_ms": {
                "median": canonical_median,
                "samples": canonical_times,
            },
            "new_prediction_end_latency_ms": {
                "median": prediction_median,
                "samples": prediction_times,
            },
            "new_prediction_latency_limit_ms": latency_limit,
            "new_prediction_to_canonical_total_ratio": (
                prediction_median / canonical_median
            ),
            "new_prediction_added_peak_memory_bytes": added_peak,
            "trace_trainable_parameters": trainable_count,
            "dense_control_trainable_parameters": dense_count,
            "old_mshnet_prediction_tail_parameters": 281,
            "field_chunk_size": chunk,
            "boundary_pattern_count": int(model.field.pattern_log_cardinality.numel()),
            "logk_cache_sha256": model.field.logk_cache_sha256,
            "empty_training_nll": empty_nll_value,
        }
        criteria = {
            "dp_report_pass_and_source_fresh": dp.get("status") == "PASS",
            "geometry_report_authenticated_and_source_fresh": True,
            "canonical_clean_front_loaded": True,
            "old_side_heads_and_final_physically_absent": no_old_heads,
            "trace_trainable_parameter_count_is_306": trainable_count
            == TracePotentialMap.parameter_count,
            "dense_control_parameter_count_is_307": dense_count == 307,
            "front_parameter_and_bn_hash_unchanged_after_train_step": before_front
            == after_front,
            "exact_nll_and_head_gradients_finite_nonzero": bool(
                nll_finite
            )
            and finite_nonzero_gradient,
            "atomic_threshold_support_invariant": support_invariant,
            "dense_renderer_equals_atom_union_bit_exact": renderer_equal,
            "renderer_uses_map_joint_not_existence_probability": joint_differs_from_existence,
            "no_python_per_cell_loop": _has_no_python_per_cell_loop(),
            "new_prediction_latency_within_budget": prediction_median <= latency_limit,
            "new_prediction_added_peak_memory_within_budget": added_peak
            <= ADDED_PEAK_MEMORY_BYTES_BUDGET,
        }
        payload.update(
            {
                "dp_report_sha256": dp_digest,
                "dp_report_file_sha256": dp_file_digest,
                "geometry_report_sha256": geometry_digest,
                "geometry_report_file_sha256": geometry_file_digest,
                "geometry_sha256": geometry.sha256,
                "t0_a_status": geometry_gate.get("status"),
                "dataset": dataset,
                "baseline_checkpoint": provenance_path(checkpoint),
                "baseline_checkpoint_sha256": sha256_file(checkpoint),
                "device": {
                    "type": runtime_device.type,
                    "name": torch.cuda.get_device_name(runtime_device),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                "metrics": metrics,
                "criteria": criteria,
                "status": "PASS" if all(criteria.values()) else "NO-GO",
                "failure": None,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "NO-GO",
                "criteria": {"verification_completed_without_exception": False},
                "failure": {
                    "type": type(exc).__name__,
                    "message_code": "integration_verification_exception",
                },
            }
        )

    _assert_no_absolute_paths(payload)
    # JSON roundtrip is also the finite-number guard.
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TraceIntegrationVerificationError("report contains non-JSON values") from exc
    payload["report_sha256"] = canonical_json_sha256(payload)
    authenticate_report(payload)
    return payload


def authenticate_report(report: Mapping[str, Any]) -> str:
    if not isinstance(report, Mapping):
        raise TraceIntegrationVerificationError("report root must be an object")
    declared = report.get("report_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise TraceIntegrationVerificationError("integration report lacks report_sha256")
    authenticated = dict(report)
    authenticated.pop("report_sha256", None)
    _assert_no_absolute_paths(authenticated)
    if canonical_json_sha256(authenticated) != declared:
        raise TraceIntegrationVerificationError("integration report hash mismatch")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("gate") != GATE:
        raise TraceIntegrationVerificationError("unsupported integration report schema")
    if report.get("status") not in {"PASS", "NO-GO"}:
        raise TraceIntegrationVerificationError("integration status must be PASS or NO-GO")
    criteria = report.get("criteria")
    if not isinstance(criteria, Mapping) or not criteria or not all(
        isinstance(value, bool) for value in criteria.values()
    ):
        raise TraceIntegrationVerificationError("integration criteria are malformed")
    if report.get("status") == "PASS" and (
        not all(criteria.values()) or report.get("failure") is not None
    ):
        raise TraceIntegrationVerificationError("passing integration report contradicts checks")
    if report.get("status") == "NO-GO" and all(criteria.values()):
        raise TraceIntegrationVerificationError("NO-GO integration report lacks a failed check")
    return declared


def render_report(report: Mapping[str, Any]) -> str:
    authenticate_report(report)
    return json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-report", required=True, type=Path)
    parser.add_argument("--geometry-report", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--field-chunk-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="return zero for an authenticated NO-GO (default exit is 2)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_verification(
        dp_report=args.dp_report,
        geometry_report=args.geometry_report,
        baseline_checkpoint=args.baseline_checkpoint,
        device=args.device,
        field_chunk_size=args.field_chunk_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    rendered = render_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["status"] != "PASS" and not args.report_only:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GATE",
    "SCHEMA_VERSION",
    "TraceIntegrationVerificationError",
    "authenticate_report",
    "main",
    "render_report",
    "run_verification",
]
