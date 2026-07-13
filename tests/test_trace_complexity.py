import json
from pathlib import Path

import pytest
import torch

from tools import profile_trace
from utils.trace_complexity import (
    TraceComplexityError,
    authenticate_complexity_report,
    benchmark_trace_atomic_field,
    direct_geometry_source,
    evaluate_runtime_budgets,
    load_t0_a_geometry_for_profile,
    profile_status,
    static_trace_complexity,
    trace_parameter_counts,
)
from utils.trace_geometry import TraceGeometrySpec
from utils.trace_provenance import canonical_json_sha256


def tiny_spec() -> TraceGeometrySpec:
    return TraceGeometrySpec(
        image_height=4,
        image_width=4,
        cell_size=2,
        max_down=1,
        max_left=0,
        max_right=1,
        margin=0,
    )


def t0_a_payload(spec: TraceGeometrySpec, *, status: str = "PASS") -> dict:
    criteria = {
        "all_components_exact_row_run_chains": status == "PASS",
        "zero_root_cell_collisions": True,
        "all_exact_components_inside_frozen_window": True,
        "train_only_no_test_overlap": True,
    }
    payload = {
        "schema_version": "trace_t0_a_geometry_report_v1",
        "gate": "T0-A",
        "status": status,
        "dataset": "fixture",
        "train_only": True,
        "train_split": {
            "ordered_names_sha256": "1" * 64,
            "canonical_test_overlap": 0,
        },
        "resize": {"height": spec.image_height, "width": spec.image_width},
        "selected_cell_size": spec.cell_size,
        "candidate_geometry_spec": spec.to_dict(),
        "candidate_geometry_sha256": spec.sha256,
        "criteria": criteria,
    }
    payload["report_sha256"] = canonical_json_sha256(payload)
    return payload


def write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "t0_a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exact_static_counts_include_boundary_pruning_and_formulas():
    report = static_trace_complexity(
        tiny_spec(), batch_size=1, field_chunk_size=2, natural_dtype="float32"
    )
    counts = report["counts"]
    assert counts == {
        "number_of_cells": 4,
        "local_height": 3,
        "local_width": 3,
        "rectangular_interval_state_upper_bound": 72,
        "boundary_valid_interval_states_exact": 45,
        "boundary_invalid_interval_slots_pruned": 27,
        "boundary_valid_root_start_states_exact": 32,
        "transition_work_proxy_N_times_H_times_W_squared": 108,
    }
    cache = report["log_cardinality_cache"]
    assert cache["pattern_count"] == 4
    assert sum(cache["cell_pattern_histogram"]) == 4
    assert all(item["cell_count"] == 1 for item in cache["patterns"])
    assert len(cache["cache_sha256"]) == 64


def test_parameter_counts_are_verified_from_live_modules():
    counts = trace_parameter_counts()
    assert counts["trace_atomic_potential_map"] == 306
    assert counts["canonical_mshnet_replaced_tail"] == 281
    assert counts["matched_dense_bernoulli_control"] == 307
    assert counts["canonical_mshnet_tail_breakdown"] == {
        "output_0": 17,
        "output_1": 33,
        "output_2": 65,
        "output_3": 129,
        "final": 37,
    }


def test_named_tensor_byte_footprints_bind_dtype_batch_and_chunk():
    report = static_trace_complexity(
        tiny_spec(), batch_size=2, field_chunk_size=3, natural_dtype="float16"
    )
    memory = report["tensor_byte_footprints"]
    assumptions = memory["assumptions"]
    assert assumptions["natural_field_dtype"] == "float16"
    assert assumptions["natural_field_bytes_per_value"] == 2
    assert assumptions["production_semiring_dtype"] == "float32"
    assert assumptions["peak_chunk_cells"] == 3
    # 2 fields * B=2 * 4*4 pixels * FP16=2 bytes.
    assert (
        memory["global_or_full_result_tensors"][
            "input_natural_fields_root_plus_support"
        ]
        == 128
    )
    # 2 fields * B=2 * chunk=3 * local 3*3 * FP32=4 bytes.
    assert (
        memory["per_peak_chunk_tensors"][
            "local_natural_fields_root_plus_support"
        ]
        == 432
    )


@pytest.mark.parametrize("status", ["PASS", "NO-GO"])
def test_authenticated_t0_a_loader_accepts_diagnostic_no_go(tmp_path, status):
    spec = tiny_spec()
    source = load_t0_a_geometry_for_profile(
        write_payload(tmp_path, t0_a_payload(spec, status=status))
    )
    assert source.authenticated
    assert source.t0_a_status == status
    assert source.geometry == spec
    assert source.report_name == "t0_a.json"
    assert source.train_split_sha256 == "1" * 64


def test_authenticated_t0_a_loader_rejects_tampering_and_semantic_contradiction(
    tmp_path,
):
    payload = t0_a_payload(tiny_spec())
    payload["selected_cell_size"] = 1
    with pytest.raises(TraceComplexityError, match="authentication"):
        load_t0_a_geometry_for_profile(write_payload(tmp_path, payload))

    payload = t0_a_payload(tiny_spec(), status="NO-GO")
    payload["criteria"] = {key: True for key in payload["criteria"]}
    payload.pop("report_sha256")
    payload["report_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(TraceComplexityError, match="contradicts"):
        load_t0_a_geometry_for_profile(write_payload(tmp_path, payload))


def fake_benchmark(*, partition_ms=2.0, map_ms=3.0, peak_bytes=4 * 1024**2):
    return {
        "workload": {"full_geometry_cells_measured": True},
        "partition_only": {
            "timing": {"median_ms": partition_ms},
            "peak_memory": {
                "max_memory_allocated_bytes": peak_bytes,
                "process_lifetime_max_rss_after_bytes": peak_bytes,
            },
        },
        "map_only": {
            "timing": {"median_ms": map_ms},
            "peak_memory": {
                "max_memory_allocated_bytes": peak_bytes,
                "process_lifetime_max_rss_after_bytes": peak_bytes,
            },
        },
    }


def test_runtime_budget_gate_needs_all_budgets_and_all_checks():
    benchmark = fake_benchmark()
    incomplete = evaluate_runtime_budgets(
        benchmark, max_partition_latency_ms=10, max_map_latency_ms=10
    )
    assert incomplete["status"] == "NOT_EVALUATED"

    passed = evaluate_runtime_budgets(
        benchmark,
        max_partition_latency_ms=10,
        max_map_latency_ms=10,
        max_peak_memory_mib=10,
    )
    assert passed["status"] == "PASS"
    failed = evaluate_runtime_budgets(
        benchmark,
        max_partition_latency_ms=1,
        max_map_latency_ms=10,
        max_peak_memory_mib=10,
    )
    assert failed["status"] == "FAIL"
    assert failed["checks"]["partition_latency"] is False


def test_profile_status_never_erases_no_go_or_unauthenticated_geometry(tmp_path):
    pass_budget = {"status": "PASS"}
    direct = direct_geometry_source(tiny_spec())
    assert profile_status(direct, pass_budget) == "DIAGNOSTIC_UNAUTHENTICATED_GEOMETRY"
    no_go = load_t0_a_geometry_for_profile(
        write_payload(tmp_path, t0_a_payload(tiny_spec(), status="NO-GO"))
    )
    assert profile_status(no_go, pass_budget) == "DIAGNOSTIC_T0_A_NO_GO"


def test_cpu_prefix_benchmark_is_separate_and_cannot_pass_a_full_budget_gate():
    benchmark = benchmark_trace_atomic_field(
        tiny_spec(),
        device="cpu",
        batch_size=1,
        field_chunk_size=2,
        benchmark_cells=1,
        warmup=0,
        iterations=1,
    )
    assert benchmark["workload"]["measured_cell_prefix"] == 1
    assert benchmark["workload"]["full_geometry_cells_measured"] is False
    assert benchmark["partition_only"]["timing"]["median_ms"] >= 0
    assert benchmark["map_only"]["timing"]["median_ms"] >= 0
    gate = evaluate_runtime_budgets(
        benchmark,
        max_partition_latency_ms=1e6,
        max_map_latency_ms=1e6,
        max_peak_memory_mib=1e6,
    )
    assert gate["status"] == "NOT_EVALUATED"
    assert gate["reason"] == "full_geometry_cells_were_not_measured"


def test_report_authentication_rejects_paths_and_cli_build_is_diagnostic(tmp_path):
    with pytest.raises(TraceComplexityError, match="absolute paths"):
        authenticate_complexity_report({"leak": "/home/user/private.json"})

    args = profile_trace.build_parser().parse_args(
        [
            "--image-height",
            "4",
            "--image-width",
            "4",
            "--cell-size",
            "2",
            "--max-down",
            "1",
            "--max-left",
            "0",
            "--max-right",
            "1",
            "--margin",
            "0",
            "--field-chunk-size",
            "2",
        ]
    )
    report = profile_trace.build_report(args)
    assert report["schema_version"] == "trace_complexity_profile_v1"
    assert report["status"] == "DIAGNOSTIC_UNAUTHENTICATED_GEOMETRY"
    assert report["training_authorized"] is False
    declared = report.pop("report_sha256")
    assert declared == canonical_json_sha256(report)
    report["report_sha256"] = declared
    output = profile_trace.dump_report(report, tmp_path / "profile.json")
    loaded = json.loads(output.read_text())
    assert "/home/" not in json.dumps(loaded)


def test_invalid_dtype_and_sizes_fail_closed():
    with pytest.raises(TraceComplexityError, match="dtype"):
        static_trace_complexity(tiny_spec(), natural_dtype="int8")
    with pytest.raises(TraceComplexityError, match="field_chunk_size"):
        static_trace_complexity(tiny_spec(), field_chunk_size=0)
    with pytest.raises(TraceComplexityError, match="benchmark_cells"):
        benchmark_trace_atomic_field(
            tiny_spec(), device="cpu", benchmark_cells=5, warmup=0, iterations=1
        )
