"""Machine-readable fail-closed gates for the TRACE implementation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from utils.trace_geometry import TraceGeometrySpec
from utils.trace_provenance import (
    PROJECT_ROOT,
    TraceProvenanceError,
    canonical_json_sha256,
    sha256_file,
)


class TraceGateError(TraceProvenanceError):
    """A prerequisite gate is missing, stale, malformed, or did not pass."""


@dataclass(frozen=True)
class GeometryGate:
    report_path: str
    report_file_sha256: str
    report_sha256: str
    dataset: str
    train_split_sha256: str
    geometry: TraceGeometrySpec
    geometry_sha256: str
    source_sha256: dict[str, str]
    mask_manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_path": self.report_path,
            "report_file_sha256": self.report_file_sha256,
            "report_sha256": self.report_sha256,
            "dataset": self.dataset,
            "train_split_sha256": self.train_split_sha256,
            "geometry": self.geometry.to_dict(),
            "geometry_sha256": self.geometry_sha256,
            "source_sha256": dict(self.source_sha256),
            "mask_manifest_sha256": self.mask_manifest_sha256,
        }


@dataclass(frozen=True)
class DPGate:
    """Authenticated exact-semiring prerequisite for the full T0-B gate."""

    report_path: str
    report_file_sha256: str
    report_sha256: str
    source_sha256: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_path": self.report_path,
            "report_file_sha256": self.report_file_sha256,
            "report_sha256": self.report_sha256,
            "source_sha256": dict(self.source_sha256),
        }


@dataclass(frozen=True)
class IntegrationGate:
    """Authenticated full-model half of the T0-B release gate."""

    report_path: str
    report_file_sha256: str
    report_sha256: str
    dp_report_sha256: str
    geometry_sha256: str
    baseline_checkpoint_sha256: str
    source_sha256: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_path": self.report_path,
            "report_file_sha256": self.report_file_sha256,
            "report_sha256": self.report_sha256,
            "dp_report_sha256": self.dp_report_sha256,
            "geometry_sha256": self.geometry_sha256,
            "baseline_checkpoint_sha256": self.baseline_checkpoint_sha256,
            "source_sha256": dict(self.source_sha256),
        }


def load_json_report(path: str | Path) -> tuple[Path, dict[str, Any]]:
    report_path = Path(path).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON constant: {token}")
    try:
        payload = json.loads(
            report_path.read_text(encoding="utf-8"),
            parse_constant=reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise TraceGateError(f"invalid JSON gate report: {report_path}") from exc
    if not isinstance(payload, dict):
        raise TraceGateError("gate report root must be a JSON object")
    return report_path, payload


def verify_embedded_report_hash(payload: dict[str, Any]) -> str:
    declared = payload.get("report_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise TraceGateError("gate report lacks a valid report_sha256")
    authenticated = dict(payload)
    authenticated.pop("report_sha256", None)
    actual = canonical_json_sha256(authenticated)
    if actual != declared:
        raise TraceGateError("gate report content does not match report_sha256")
    return declared


def _require_current_source_inventory(value: Any, *, gate: str) -> dict[str, str]:
    """Verify every declared repository-relative source against current bytes."""

    if not isinstance(value, dict) or not value:
        raise TraceGateError(f"{gate} lacks a source_sha256 inventory")
    verified: dict[str, str] = {}
    for locator, declared in value.items():
        if (
            not isinstance(locator, str)
            or not locator
            or Path(locator).is_absolute()
            or ".." in Path(locator).parts
        ):
            raise TraceGateError(f"{gate} contains an unsafe source locator")
        if not isinstance(declared, str) or len(declared) != 64:
            raise TraceGateError(f"{gate} contains a malformed source hash")
        source = (PROJECT_ROOT / locator).resolve()
        try:
            source.relative_to(PROJECT_ROOT)
        except ValueError as exc:  # pragma: no cover - guarded above
            raise TraceGateError(f"{gate} source escapes the repository") from exc
        if not source.is_file() or sha256_file(source) != declared:
            raise TraceGateError(
                f"{gate} report is stale: current source differs for {locator}"
            )
        verified[locator] = declared
    return verified


def require_dp_gate(path: str | Path) -> DPGate:
    """Require the authenticated FP64 DP/brute-force half of T0-B.

    A passing core report intentionally does *not* unlock training by itself:
    its integration checks and full release status must remain explicit
    ``PENDING``.  The training entry point additionally requires a distinct,
    authenticated integration PASS report.
    """

    report_path, payload = load_json_report(path)
    declared_hash = verify_embedded_report_hash(payload)
    if payload.get("schema_version") != "trace_t0_b_dp_verification_v1":
        raise TraceGateError("unsupported T0-B DP report schema")
    if payload.get("gate") != "T0-B-DP" or payload.get("status") != "PASS":
        raise TraceGateError(
            f"TRACE training is locked because T0-B-DP status is {payload.get('status')!r}"
        )
    if payload.get("scope") != "exact_run_semiring_core_only":
        raise TraceGateError("T0-B-DP scope is not the frozen semiring core")
    criteria = payload.get("criteria")
    if not isinstance(criteria, dict) or not criteria or not all(
        value is True for value in criteria.values()
    ):
        raise TraceGateError("T0-B-DP criteria are incomplete or contain a failed item")
    if payload.get("failure") is not None:
        raise TraceGateError("passing T0-B-DP report contains a failure record")
    pending = payload.get("pending_integration_checks")
    if (
        not isinstance(pending, list)
        or not pending
        or any(
            not isinstance(item, dict) or item.get("status") != "PENDING"
            for item in pending
        )
        or payload.get("full_t0_b_release_status") != "PENDING"
    ):
        raise TraceGateError(
            "T0-B-DP must leave integration checks and the full release pending"
        )
    sources = _require_current_source_inventory(
        payload.get("source_sha256"), gate="T0-B-DP"
    )
    required_sources = {"model/trace_run_semiring.py", "tools/verify_trace_dp.py"}
    if set(sources) != required_sources:
        raise TraceGateError("T0-B-DP source inventory is not exact")
    return DPGate(
        report_path=report_path.name,
        report_file_sha256=sha256_file(report_path),
        report_sha256=declared_hash,
        source_sha256=sources,
    )


def require_integration_gate(
    path: str | Path,
    *,
    expected_dp_report_sha256: str,
    expected_geometry_sha256: str,
    expected_baseline_checkpoint_sha256: str,
) -> IntegrationGate:
    """Require the authenticated model/renderer/front integration half of T0-B."""

    report_path, payload = load_json_report(path)
    declared_hash = verify_embedded_report_hash(payload)
    if payload.get("schema_version") != "trace_t0_b_integration_verification_v1":
        raise TraceGateError("unsupported T0-B integration report schema")
    if payload.get("gate") != "T0-B-INTEGRATION" or payload.get("status") != "PASS":
        raise TraceGateError(
            "TRACE training is locked because T0-B-INTEGRATION status is %r"
            % payload.get("status")
        )
    criteria = payload.get("criteria")
    if not isinstance(criteria, dict) or not criteria or not all(
        value is True for value in criteria.values()
    ):
        raise TraceGateError(
            "T0-B-INTEGRATION criteria are incomplete or contain a failed item"
        )
    if payload.get("failure") is not None:
        raise TraceGateError(
            "passing T0-B-INTEGRATION report contains a failure record"
        )
    bindings = {
        "DP report": (
            payload.get("dp_report_sha256"),
            expected_dp_report_sha256,
        ),
        "geometry": (
            payload.get("geometry_sha256"),
            expected_geometry_sha256,
        ),
        "baseline checkpoint": (
            payload.get("baseline_checkpoint_sha256"),
            expected_baseline_checkpoint_sha256,
        ),
    }
    for label, (actual, expected) in bindings.items():
        if not isinstance(actual, str) or len(actual) != 64 or actual != expected:
            raise TraceGateError(f"T0-B-INTEGRATION {label} hash mismatch")
    sources = _require_current_source_inventory(
        payload.get("source_sha256"), gate="T0-B-INTEGRATION"
    )
    required_sources = {
        "model/trace_front.py",
        "model/trace_mshnet.py",
        "model/trace_run_semiring.py",
        "tools/verify_trace_integration.py",
        "utils/trace_geometry.py",
    }
    if set(sources) != required_sources:
        raise TraceGateError("T0-B-INTEGRATION source inventory is not exact")
    return IntegrationGate(
        report_path=report_path.name,
        report_file_sha256=sha256_file(report_path),
        report_sha256=declared_hash,
        dp_report_sha256=expected_dp_report_sha256,
        geometry_sha256=expected_geometry_sha256,
        baseline_checkpoint_sha256=expected_baseline_checkpoint_sha256,
        source_sha256=sources,
    )


def require_geometry_gate(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_train_split_sha256: str | None = None,
) -> GeometryGate:
    report_path, payload = load_json_report(path)
    declared_hash = verify_embedded_report_hash(payload)
    if payload.get("schema_version") != "trace_t0_a_geometry_report_v1":
        raise TraceGateError("unsupported T0-A report schema")
    if payload.get("gate") != "T0-A" or payload.get("status") != "PASS":
        raise TraceGateError(
            f"TRACE training is locked because T0-A status is {payload.get('status')!r}"
        )
    if payload.get("train_only") is not True:
        raise TraceGateError("T0-A was not declared train-only")
    sources = _require_current_source_inventory(
        payload.get("source_sha256"), gate="T0-A"
    )
    required_sources = {
        "tools/audit_trace_geometry.py",
        "utils/trace_codec.py",
        "utils/trace_geometry.py",
    }
    if set(sources) != required_sources:
        raise TraceGateError("T0-A source inventory is not exact")
    criteria = payload.get("criteria")
    if not isinstance(criteria, dict) or not criteria or not all(
        value is True for value in criteria.values()
    ):
        raise TraceGateError("T0-A criteria are incomplete or contain a failed item")
    dataset = payload.get("dataset")
    split = payload.get("train_split")
    if not isinstance(dataset, str) or not isinstance(split, dict):
        raise TraceGateError("T0-A lacks dataset/split provenance")
    train_hash = split.get("ordered_names_sha256")
    if not isinstance(train_hash, str) or len(train_hash) != 64:
        raise TraceGateError("T0-A lacks a normalized train split hash")
    if int(split.get("canonical_test_overlap", -1)) != 0:
        raise TraceGateError("T0-A train split overlaps canonical test")
    if expected_dataset is not None and dataset != expected_dataset:
        raise TraceGateError(
            f"T0-A dataset mismatch: expected {expected_dataset}, got {dataset}"
        )
    if expected_train_split_sha256 and train_hash != expected_train_split_sha256:
        raise TraceGateError("T0-A train split differs from the paired training protocol")

    geometry_payload = payload.get("candidate_geometry_spec")
    geometry_hash = payload.get("candidate_geometry_sha256")
    if not isinstance(geometry_payload, dict) or not isinstance(geometry_hash, str):
        raise TraceGateError("passing T0-A report lacks its frozen geometry")
    geometry = TraceGeometrySpec.from_dict(geometry_payload)
    if geometry.sha256 != geometry_hash:
        raise TraceGateError("frozen geometry does not match its declared hash")
    resize = payload.get("resize")
    if (
        not isinstance(resize, dict)
        or resize.get("height") != geometry.image_height
        or resize.get("width") != geometry.image_width
        or resize.get("interpolation") != "PIL.Image.Resampling.NEAREST"
    ):
        raise TraceGateError("T0-A resize contract disagrees with frozen geometry")
    mask_manifest_hash = payload.get("mask_manifest_sha256")
    if not isinstance(mask_manifest_hash, str) or len(mask_manifest_hash) != 64:
        raise TraceGateError("T0-A lacks an authenticated train-mask manifest hash")
    return GeometryGate(
        report_path=report_path.name,
        report_file_sha256=sha256_file(report_path),
        report_sha256=declared_hash,
        dataset=dataset,
        train_split_sha256=train_hash,
        geometry=geometry,
        geometry_sha256=geometry_hash,
        source_sha256=sources,
        mask_manifest_sha256=mask_manifest_hash,
    )


__all__ = [
    "DPGate",
    "GeometryGate",
    "IntegrationGate",
    "TraceGateError",
    "load_json_report",
    "require_dp_gate",
    "require_geometry_gate",
    "require_integration_gate",
    "verify_embedded_report_hash",
]
