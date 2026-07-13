import json
from pathlib import Path

import pytest

from utils.trace_gates import (
    TraceGateError,
    require_dp_gate,
    require_geometry_gate,
    require_integration_gate,
)
from utils.trace_geometry import TraceGeometrySpec
from utils.trace_provenance import PROJECT_ROOT, canonical_json_sha256, sha256_file


def _sources(*locators: str) -> dict[str, str]:
    return {locator: sha256_file(PROJECT_ROOT / locator) for locator in locators}


def _report(tmp_path, *, status="PASS"):
    spec = TraceGeometrySpec(
        image_height=8,
        image_width=8,
        cell_size=2,
        max_down=2,
        max_left=1,
        max_right=2,
        margin=1,
    )
    payload = {
        "schema_version": "trace_t0_a_geometry_report_v1",
        "gate": "T0-A",
        "status": status,
        "dataset": "toy",
        "train_only": True,
        "train_split": {
            "ordered_names_sha256": "a" * 64,
            "canonical_test_overlap": 0,
        },
        "criteria": {
            "all_components_exact_row_run_chains": status == "PASS",
            "zero_root_cell_collisions": True,
            "all_exact_components_inside_frozen_window": True,
            "train_only_no_test_overlap": True,
        },
        "candidate_geometry_spec": spec.to_dict(),
        "candidate_geometry_sha256": spec.sha256,
        "resize": {
            "height": spec.image_height,
            "width": spec.image_width,
            "interpolation": "PIL.Image.Resampling.NEAREST",
        },
        "mask_manifest_sha256": "c" * 64,
        "source_sha256": _sources(
            "tools/audit_trace_geometry.py",
            "utils/trace_codec.py",
            "utils/trace_geometry.py",
        ),
    }
    payload["report_sha256"] = canonical_json_sha256(payload)
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload, spec


def test_require_geometry_gate_accepts_authenticated_pass(tmp_path):
    path, payload, spec = _report(tmp_path)
    gate = require_geometry_gate(
        path,
        expected_dataset="toy",
        expected_train_split_sha256="a" * 64,
    )
    assert gate.geometry == spec
    assert gate.geometry_sha256 == spec.sha256
    assert gate.report_sha256 == payload["report_sha256"]


def test_require_geometry_gate_rejects_no_go(tmp_path):
    path, _, _ = _report(tmp_path, status="NO-GO")
    with pytest.raises(TraceGateError, match="locked"):
        require_geometry_gate(path)


def test_require_geometry_gate_rejects_tamper(tmp_path):
    path, payload, _ = _report(tmp_path)
    payload["dataset"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TraceGateError, match="report_sha256"):
        require_geometry_gate(path)


@pytest.mark.parametrize(
    ("expected_dataset", "expected_hash", "message"),
    [
        ("other", None, "dataset mismatch"),
        (None, "b" * 64, "train split differs"),
    ],
)
def test_require_geometry_gate_rejects_pairing_mismatch(
    tmp_path, expected_dataset, expected_hash, message
):
    path, _, _ = _report(tmp_path)
    with pytest.raises(TraceGateError, match=message):
        require_geometry_gate(
            path,
            expected_dataset=expected_dataset,
            expected_train_split_sha256=expected_hash,
        )


def test_require_geometry_gate_rejects_geometry_hash_mismatch(tmp_path):
    path, payload, _ = _report(tmp_path)
    payload["candidate_geometry_sha256"] = "0" * 64
    payload["report_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TraceGateError, match="frozen geometry"):
        require_geometry_gate(path)


def test_require_geometry_gate_rejects_stale_source(tmp_path):
    path, payload, _ = _report(tmp_path)
    payload["source_sha256"]["utils/trace_codec.py"] = "0" * 64
    payload["report_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TraceGateError, match="stale"):
        require_geometry_gate(path)


def _dp_report(tmp_path: Path, *, status: str = "PASS") -> tuple[Path, dict]:
    payload = {
        "schema_version": "trace_t0_b_dp_verification_v1",
        "gate": "T0-B-DP",
        "scope": "exact_run_semiring_core_only",
        "status": status,
        "criteria": {"dp_matches_brute": status == "PASS"},
        "failure": None if status == "PASS" else {"type": "Synthetic"},
        "pending_integration_checks": [
            {"id": "integrated_model", "status": "PENDING"}
        ],
        "full_t0_b_release_status": "PENDING",
        "source_sha256": _sources(
            "model/trace_run_semiring.py", "tools/verify_trace_dp.py"
        ),
    }
    payload["report_sha256"] = canonical_json_sha256(payload)
    path = tmp_path / "dp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _integration_report(
    tmp_path: Path,
    *,
    dp_sha256: str,
    geometry_sha256: str,
    baseline_sha256: str,
    status: str = "PASS",
) -> tuple[Path, dict]:
    payload = {
        "schema_version": "trace_t0_b_integration_verification_v1",
        "gate": "T0-B-INTEGRATION",
        "status": status,
        "criteria": {"renderer_and_front_integrated": status == "PASS"},
        "failure": None if status == "PASS" else {"type": "Synthetic"},
        "dp_report_sha256": dp_sha256,
        "geometry_sha256": geometry_sha256,
        "baseline_checkpoint_sha256": baseline_sha256,
        "source_sha256": _sources(
            "model/trace_front.py",
            "model/trace_mshnet.py",
            "model/trace_run_semiring.py",
            "tools/verify_trace_integration.py",
            "utils/trace_geometry.py",
        ),
    }
    payload["report_sha256"] = canonical_json_sha256(payload)
    path = tmp_path / "integration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_t0_b_requires_separate_fresh_dp_and_integration_passes(tmp_path):
    dp_path, dp_payload = _dp_report(tmp_path)
    dp = require_dp_gate(dp_path)
    assert dp.report_sha256 == dp_payload["report_sha256"]

    integration_path, integration_payload = _integration_report(
        tmp_path,
        dp_sha256=dp.report_sha256,
        geometry_sha256="3" * 64,
        baseline_sha256="4" * 64,
    )
    integration = require_integration_gate(
        integration_path,
        expected_dp_report_sha256=dp.report_sha256,
        expected_geometry_sha256="3" * 64,
        expected_baseline_checkpoint_sha256="4" * 64,
    )
    assert integration.report_sha256 == integration_payload["report_sha256"]


def test_t0_b_no_go_pending_or_binding_mismatch_never_unlocks(tmp_path):
    no_go_path, _ = _dp_report(tmp_path, status="NO-GO")
    with pytest.raises(TraceGateError, match="locked"):
        require_dp_gate(no_go_path)

    dp_path, dp_payload = _dp_report(tmp_path)
    dp = require_dp_gate(dp_path)
    integration_path, _ = _integration_report(
        tmp_path,
        dp_sha256=dp_payload["report_sha256"],
        geometry_sha256="3" * 64,
        baseline_sha256="4" * 64,
    )
    with pytest.raises(TraceGateError, match="geometry hash mismatch"):
        require_integration_gate(
            integration_path,
            expected_dp_report_sha256=dp.report_sha256,
            expected_geometry_sha256="5" * 64,
            expected_baseline_checkpoint_sha256="4" * 64,
        )


def test_t0_b_rejects_authenticated_but_stale_source(tmp_path):
    path, payload = _dp_report(tmp_path)
    payload["source_sha256"]["model/trace_run_semiring.py"] = "f" * 64
    payload["report_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TraceGateError, match="stale"):
        require_dp_gate(path)
