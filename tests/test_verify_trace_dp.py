from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import verify_trace_dp as verifier


@pytest.fixture(scope="module")
def passing_report() -> dict[str, object]:
    return verifier.run_verification()


def test_fp64_core_report_covers_all_predeclared_mathematical_checks(
    passing_report: dict[str, object],
) -> None:
    report = passing_report
    assert report["schema_version"] == verifier.SCHEMA_VERSION
    assert report["gate"] == "T0-B-DP"
    assert report["scope"] == "exact_run_semiring_core_only"
    assert report["status"] == "PASS"
    assert verifier.authenticate_report(report) == report["report_sha256"]
    assert report["precision"] == "FP64"
    assert report["device"] == "CPU"
    assert all(report["criteria"].values())

    cases = report["energy_cases"]
    assert {(case["window"]["height"], case["window"]["width"]) for case in cases} == {
        (3, 3),
        (4, 4),
        (5, 5),
    }
    assert {case["energy_regime"] for case in cases} == {"nominal", "extreme"}
    assert len(cases) == 6
    for case in cases:
        assert case["status"] == "PASS"
        assert case["dtype"] == "torch.float64"
        assert all(case["checks"].values())
        assert case["checks"]["map_support_bit_exact"] is True
        assert case["checks"]["unique_map_no_tie"] is True
        assert case["metrics"]["logZ_positive_max_abs"] < 1.0e-6
        assert case["metrics"]["logZ_total_max_abs"] < 1.0e-6
        assert case["metrics"]["map_energy_max_abs"] < 1.0e-6
        assert case["metrics"]["root_marginal_max_abs"] < 1.0e-6
        assert case["metrics"]["support_marginal_max_abs"] < 1.0e-6
        assert case["metrics"]["root_finite_difference_abs"] < 1.0e-4
        assert case["metrics"]["support_finite_difference_abs"] < 1.0e-4

    zero_score = report["zero_score"]
    assert zero_score["status"] == "PASS"
    assert zero_score["reference"] == "independent_run_chain_enumeration"
    assert len(zero_score["cases"]) == 3
    assert all(all(case["checks"].values()) for case in zero_score["cases"])

    prior = report["prior_calibration"]
    assert prior["status"] == "PASS"
    assert prior["logits"][0] == -30.0
    assert prior["logits"][-1] == 30.0
    assert all(prior["checks"].values())
    assert prior["max_p_nonempty_abs"] < 1.0e-6
    assert prior["max_logZ_positive_abs"] < 1.0e-6


def test_report_is_deterministic_authenticated_and_contains_no_absolute_paths(
    passing_report: dict[str, object],
) -> None:
    repeated = verifier.run_verification()
    assert repeated == passing_report
    rendered = verifier.render_report(passing_report)
    assert json.loads(rendered) == passing_report
    assert str(Path.home()) not in rendered
    assert str(Path(__file__).resolve().parents[1]) not in rendered
    assert set(passing_report["source_sha256"]) == {
        "model/trace_run_semiring.py",
        "tools/verify_trace_dp.py",
    }

    tampered = copy.deepcopy(passing_report)
    tampered["energy_cases"][0]["status"] = "NO-GO"
    with pytest.raises(
        verifier.TraceDPVerificationError,
        match="report_sha256",
    ):
        verifier.authenticate_report(tampered)

    semantically_invalid = copy.deepcopy(passing_report)
    semantically_invalid.pop("report_sha256")
    semantically_invalid["status"] = "MAYBE"
    semantically_invalid["report_sha256"] = verifier.canonical_json_sha256(
        semantically_invalid
    )
    with pytest.raises(verifier.TraceDPVerificationError, match="PASS or NO-GO"):
        verifier.authenticate_report(semantically_invalid)


def test_not_yet_integrated_checks_are_explicitly_pending_not_invented(
    passing_report: dict[str, object],
) -> None:
    assert passing_report["full_t0_b_release_status"] == "PENDING"
    pending = passing_report["pending_integration_checks"]
    assert {item["id"] for item in pending} == {
        "atomic_threshold_support_invariance",
        "dense_renderer_atom_union_bit_exact",
        "frozen_front_parameter_and_bn_hash",
        "integrated_latency_memory_and_no_python_cell_loop",
    }
    assert all(item["status"] == "PENDING" for item in pending)


def test_unexpected_verification_error_emits_authenticated_path_free_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_seed: int) -> dict[str, object]:
        raise RuntimeError("private failure at /home/example/secret/model.pt")

    monkeypatch.setattr(verifier, "_run_core_checks", fail)
    report = verifier.run_verification()
    assert report["status"] == "NO-GO"
    assert report["criteria"] == {
        "verification_completed_without_exception": False,
    }
    assert report["failure"] == {
        "type": "RuntimeError",
        "message_code": "verification_exception",
    }
    verifier.authenticate_report(report)
    assert "/home/example" not in verifier.render_report(report)


def test_cli_writes_the_authenticated_report_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    passing_report: dict[str, object],
) -> None:
    output = tmp_path / "t0_b.json"
    monkeypatch.setattr(verifier, "run_verification", lambda seed: passing_report)
    assert verifier.main(["--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == passing_report
    verifier.authenticate_report(written)
    assert str(output.resolve()) not in output.read_text(encoding="utf-8")
    capsys.readouterr()

    no_go = copy.deepcopy(passing_report)
    no_go.pop("report_sha256")
    no_go["status"] = "NO-GO"
    no_go["criteria"] = {"synthetic_test_failure": False}
    no_go = verifier._finalize_report(no_go)
    monkeypatch.setattr(verifier, "run_verification", lambda seed: no_go)
    assert verifier.main([]) == 2
    assert verifier.main(["--report-only"]) == 0


def test_invalid_seed_is_an_authenticated_no_go() -> None:
    report = verifier.run_verification(seed=-1)
    assert report["status"] == "NO-GO"
    assert report["seed"] == -1
    assert report["failure"]["type"] == "TraceDPVerificationError"
    verifier.authenticate_report(report)
