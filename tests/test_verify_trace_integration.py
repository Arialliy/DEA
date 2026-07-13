from __future__ import annotations

import copy

import pytest
import torch

from model.trace_mshnet import TraceAtomicField, render_trace_atoms
from tools import verify_trace_integration as verifier
from utils.trace_geometry import TraceGeometrySpec
from utils.trace_provenance import canonical_json_sha256


def test_missing_prerequisites_produce_authenticated_path_free_no_go() -> None:
    report = verifier.run_verification(
        dp_report="missing_dp.json",
        geometry_report="missing_geometry.json",
        baseline_checkpoint="missing_checkpoint.pkl",
        device="cpu",
    )
    assert report["status"] == "NO-GO"
    assert report["criteria"] == {"verification_completed_without_exception": False}
    assert report["failure"]["message_code"] == "integration_verification_exception"
    assert verifier.authenticate_report(report) == report["report_sha256"]
    rendered = verifier.render_report(report)
    assert "/home/" not in rendered


def test_integration_authentication_rejects_tampering_and_contradiction() -> None:
    report = verifier.run_verification(
        dp_report="missing_dp.json",
        geometry_report="missing_geometry.json",
        baseline_checkpoint="missing_checkpoint.pkl",
        device="cpu",
    )
    tampered = copy.deepcopy(report)
    tampered["status"] = "PASS"
    with pytest.raises(verifier.TraceIntegrationVerificationError, match="hash"):
        verifier.authenticate_report(tampered)

    contradictory = copy.deepcopy(report)
    contradictory.pop("report_sha256")
    contradictory["status"] = "PASS"
    contradictory["report_sha256"] = canonical_json_sha256(contradictory)
    with pytest.raises(verifier.TraceIntegrationVerificationError, match="contradicts"):
        verifier.authenticate_report(contradictory)


def test_static_contract_has_chunk_loop_not_python_per_cell_loop() -> None:
    assert verifier._has_no_python_per_cell_loop() is True


def test_manual_union_is_independent_and_bit_exact_with_renderer() -> None:
    spec = TraceGeometrySpec(
        image_height=4,
        image_width=6,
        cell_size=2,
        max_down=1,
        max_left=1,
        max_right=1,
        margin=0,
    )
    field = TraceAtomicField(spec, field_chunk_size=4)
    root = torch.linspace(-1.0, 1.0, 24, dtype=torch.float64).reshape(1, 4, 6)
    support = torch.linspace(0.5, -0.5, 24, dtype=torch.float64).reshape(1, 4, 6)
    output = field(root, support, return_map=True)
    rendered = render_trace_atoms(output, spec, cell_chunk_size=3)
    thresholds = (
        rendered.background_score,
        float(torch.median(output.map_log_joint_posterior)),
        float(torch.max(output.map_log_joint_posterior)),
    )
    for threshold in thresholds:
        assert torch.equal(
            rendered.binary(threshold),
            verifier._manual_atom_union(output, spec, threshold),
        )
