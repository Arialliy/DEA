#!/usr/bin/env python3
"""Verify the FP64 mathematical core of TRACE T0-B, fail closed.

This command authenticates only the exact run-semiring core.  Checks that
require the not-yet-integrated MSHNet front or atomic renderer are emitted as
explicit ``PENDING`` records and never silently promoted to PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import sys
from typing import Iterable, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.trace_run_semiring import (  # noqa: E402
    RootCellRunSemiring,
    brute_force_reference,
    enumerate_run_chains,
    zero_score_cardinality,
    zero_score_log_cardinality,
)


SCHEMA_VERSION = "trace_t0_b_dp_verification_v1"
GATE = "T0-B-DP"
DEFAULT_SEED = 20260713
WINDOW_SHAPES = ((3, 3), (4, 4), (5, 5))
ENERGY_REGIMES = (("nominal", 2.0), ("extreme", 30.0))
PRIOR_LOGITS = (-30.0, -20.0, -10.0, -2.0, 0.0, 2.0, 10.0, 20.0, 30.0)
MAX_STATES = 1_000_000

# These are the frozen T0-B release bounds.  Reported errors are normally
# much smaller; using the declared gate limits here avoids post-hoc tolerance
# selection.
TOLERANCES = {
    "log_partition_abs": 1.0e-6,
    "map_energy_abs": 1.0e-6,
    "marginal_abs": 1.0e-6,
    "finite_difference_abs": 1.0e-4,
    "zero_score_logK_abs": 1.0e-6,
    "zero_score_cardinality_abs": 1.0e-6,
    "prior_probability_abs": 1.0e-6,
    "prior_log_partition_abs": 1.0e-6,
    "minimum_unique_map_gap": 0.0,
}

PENDING_INTEGRATION_CHECKS = (
    {
        "id": "atomic_threshold_support_invariance",
        "status": "PENDING",
        "requires": "integrated atomic renderer",
        "claim": "threshold accepts/rejects one whole MAP atom without support edits",
    },
    {
        "id": "dense_renderer_atom_union_bit_exact",
        "status": "PENDING",
        "requires": "integrated atomic renderer",
        "claim": "dense render equals the bit-exact union of accepted atoms",
    },
    {
        "id": "frozen_front_parameter_and_bn_hash",
        "status": "PENDING",
        "requires": "integrated TRACE model and one train step",
        "claim": "MSHNet front parameters and BN buffers remain unchanged",
    },
    {
        "id": "integrated_latency_memory_and_no_python_cell_loop",
        "status": "PENDING",
        "requires": "integrated TRACE model and production geometry",
        "claim": "the complete prediction path satisfies the engineering budget",
    },
)


class TraceDPVerificationError(RuntimeError):
    """An authenticated T0-B report is malformed, stale, or tampered."""


def _strict_canonical_json(payload: object) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TraceDPVerificationError(
            "verification report is not strict JSON"
        ) from exc
    return rendered.encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(_strict_canonical_json(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_inventory() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "model" / "trace_run_semiring.py",
        Path(__file__).resolve(),
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
        for path in paths
    }


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
            raise TraceDPVerificationError(
                "absolute paths are forbidden in T0-B reports"
            )


def _validate_report_contract(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("gate") != GATE:
        raise TraceDPVerificationError("unsupported T0-B DP report schema")
    if report.get("scope") != "exact_run_semiring_core_only":
        raise TraceDPVerificationError("T0-B DP report scope is not locked")
    status = report.get("status")
    if status not in ("PASS", "NO-GO"):
        raise TraceDPVerificationError("T0-B DP status must be PASS or NO-GO")
    criteria = report.get("criteria")
    if (
        not isinstance(criteria, Mapping)
        or not criteria
        or not all(isinstance(value, bool) for value in criteria.values())
    ):
        raise TraceDPVerificationError("T0-B DP criteria must be non-empty booleans")
    if status == "PASS":
        if not all(criteria.values()) or report.get("failure") is not None:
            raise TraceDPVerificationError("passing T0-B DP report contradicts its checks")
        for section in ("energy_cases", "zero_score", "prior_calibration"):
            if section not in report:
                raise TraceDPVerificationError(
                    "passing T0-B DP report lacks %s" % section
                )
    elif all(criteria.values()):
        raise TraceDPVerificationError("NO-GO report must contain a failed criterion")

    pending = report.get("pending_integration_checks")
    expected_pending = {item["id"] for item in PENDING_INTEGRATION_CHECKS}
    if (
        not isinstance(pending, list)
        or {item.get("id") for item in pending if isinstance(item, Mapping)}
        != expected_pending
        or any(
            not isinstance(item, Mapping) or item.get("status") != "PENDING"
            for item in pending
        )
        or report.get("full_t0_b_release_status") != "PENDING"
    ):
        raise TraceDPVerificationError(
            "integration-dependent T0-B checks must remain explicitly PENDING"
        )

    source = report.get("source_sha256")
    if not isinstance(source, Mapping) or set(source) != {
        "model/trace_run_semiring.py",
        "tools/verify_trace_dp.py",
    }:
        raise TraceDPVerificationError("T0-B DP source inventory is incomplete")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in source.values()
    ):
        raise TraceDPVerificationError("T0-B DP source hash is malformed")


def authenticate_report(report: Mapping[str, object]) -> str:
    """Validate and return the embedded canonical report digest."""

    if not isinstance(report, Mapping):
        raise TraceDPVerificationError("report root must be an object")
    declared = report.get("report_sha256")
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
    ):
        raise TraceDPVerificationError("report lacks a valid report_sha256")
    authenticated = dict(report)
    authenticated.pop("report_sha256", None)
    _assert_no_absolute_paths(authenticated)
    actual = canonical_json_sha256(authenticated)
    if actual != declared:
        raise TraceDPVerificationError("report content does not match report_sha256")
    _validate_report_contract(report)
    return declared


def _finalize_report(payload: dict[str, object]) -> dict[str, object]:
    if "report_sha256" in payload:
        raise TraceDPVerificationError("report_sha256 must be added only once")
    _assert_no_absolute_paths(payload)
    # Strict serialization rejects NaN/Inf before the report is authenticated.
    digest = canonical_json_sha256(payload)
    finalized = dict(payload)
    finalized["report_sha256"] = digest
    authenticate_report(finalized)
    return finalized


def _admissible_masks(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two deterministic legal boundary geometries for one tiny shape."""

    support = torch.ones((2, height, width), dtype=torch.bool)
    roots = torch.zeros_like(support)
    roots[0, : min(2, height), : min(2, width)] = True
    roots[1, 0, : min(3, width)] = True
    roots[1, min(1, height - 1), 1 : min(3, width)] = True
    support[0, height - 1, width - 1] = False
    support[1, 0, 0] = False
    if (height, width) == (4, 4):
        support[1, 3, 0] = False
        support[1, 2, 3] = False
    if (height, width) == (5, 5):
        # Retain a literal 5x5 DP/backward field required by the frozen design,
        # while using an authenticated boundary geometry whose bottom row is
        # outside the image.  This keeps the independent exhaustive reference
        # below the declared one-million-state safety cap.
        support[:, 4, :] = False
        support[1, 3, 0] = False
        support[1, 2, 4] = False
    return support, roots


def _hashed_unit_interval(key: str) -> float:
    integer = int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    # Midpoint mapping avoids exact endpoints while remaining platform-free.
    return (integer + 0.5) / float(1 << 64)


def _energy_fixture(
    *,
    seed: int,
    case_id: str,
    batch: int,
    height: int,
    width: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    def values(kind: str) -> list[float]:
        result = []
        for index in range(batch * height * width):
            unit = _hashed_unit_interval(
                "%s\0%d\0%s\0%d" % (case_id, seed, kind, index)
            )
            result.append(scale * (2.0 * unit - 1.0))
        return result

    root_values = values("root")
    support_values = values("support")
    root = torch.tensor(root_values, dtype=torch.float64).reshape(
        batch, height, width
    )
    support = torch.tensor(support_values, dtype=torch.float64).reshape(
        batch, height, width
    )
    fixture_hash = canonical_json_sha256(
        {
            "case_id": case_id,
            "seed": seed,
            "shape": [batch, height, width],
            "root": root_values,
            "support": support_values,
        }
    )
    return root, support, fixture_hash


def _finite_max_abs(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, float | None]:
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite or tuple(actual.shape) != tuple(expected.shape):
        return False, None
    return True, float(torch.max(torch.abs(actual - expected)).item())


def _unique_map_gap(
    root: torch.Tensor,
    support: torch.Tensor,
    valid_support: torch.Tensor,
    valid_root: torch.Tensor,
) -> float:
    chains_by_batch = enumerate_run_chains(
        valid_support, valid_root, max_states=MAX_STATES
    )
    minimum_gap = math.inf
    for batch_index, chains in enumerate(chains_by_batch):
        energies = []
        for chain in chains:
            root_y, root_x = chain.root
            energy = float(root[batch_index, root_y, root_x].item())
            for y, left, right in chain.intervals:
                energy += float(support[batch_index, y, left : right + 1].sum().item())
            energies.append(energy)
        energies.sort(reverse=True)
        if len(energies) < 2:
            raise TraceDPVerificationError(
                "tiny verification field must contain at least two atoms"
            )
        minimum_gap = min(minimum_gap, energies[0] - energies[1])
    if not math.isfinite(minimum_gap):
        raise TraceDPVerificationError("MAP gap is not finite")
    return float(minimum_gap)


def _verify_energy_case(
    *,
    seed: int,
    height: int,
    width: int,
    regime: str,
    scale: float,
) -> dict[str, object]:
    case_id = "%dx%d_%s" % (height, width, regime)
    valid_support, valid_root = _admissible_masks(height, width)
    root, support, fixture_hash = _energy_fixture(
        seed=seed,
        case_id=case_id,
        batch=valid_support.shape[0],
        height=height,
        width=width,
        scale=scale,
    )
    root.requires_grad_(True)
    support.requires_grad_(True)
    exact = RootCellRunSemiring(cardinality_correction=True)(
        root,
        support,
        valid_support,
        valid_root,
        return_map=True,
        return_marginals=True,
    )
    brute = brute_force_reference(
        root,
        support,
        valid_support,
        valid_root,
        cardinality_correction=True,
        max_states=MAX_STATES,
        return_marginals=True,
    )

    metrics: dict[str, float | None] = {}
    finite_flags = []
    for key, actual, expected in (
        ("logZ_positive_max_abs", exact.logZ_positive, brute.logZ_positive),
        ("logZ_total_max_abs", exact.logZ_total, brute.logZ_total),
        ("map_energy_max_abs", exact.map_energy, brute.map_energy),
        ("root_marginal_max_abs", exact.root_marginal, brute.root_marginal),
        (
            "support_marginal_max_abs",
            exact.support_marginal,
            brute.support_marginal,
        ),
    ):
        finite, error = _finite_max_abs(actual, expected)
        finite_flags.append(finite)
        metrics[key] = error

    map_support_equal = bool(torch.equal(exact.map_support, brute.map_support))
    map_root_equal = bool(torch.equal(exact.map_root, brute.map_root))
    map_intervals_equal = bool(torch.equal(exact.map_intervals, brute.map_intervals))
    minimum_map_gap = _unique_map_gap(
        root.detach(), support.detach(), valid_support, valid_root
    )
    epsilon = 1.0e-5

    def partition_sum(root_value: torch.Tensor, support_value: torch.Tensor) -> float:
        value = RootCellRunSemiring(cardinality_correction=True)(
            root_value,
            support_value,
            valid_support,
            valid_root,
            log_cardinality=exact.log_cardinality.detach(),
            return_map=False,
        ).logZ_total.sum()
        return float(value.item())

    root_coordinate = tuple(
        int(value) for value in torch.nonzero(valid_root[0], as_tuple=False)[0]
    )
    support_coordinate = tuple(
        int(value) for value in torch.nonzero(valid_support[0], as_tuple=False)[-1]
    )
    root_plus = root.detach().clone()
    root_minus = root.detach().clone()
    root_plus[(0, *root_coordinate)] += epsilon
    root_minus[(0, *root_coordinate)] -= epsilon
    root_finite_difference = (
        partition_sum(root_plus, support.detach())
        - partition_sum(root_minus, support.detach())
    ) / (2.0 * epsilon)
    support_plus = support.detach().clone()
    support_minus = support.detach().clone()
    support_plus[(0, *support_coordinate)] += epsilon
    support_minus[(0, *support_coordinate)] -= epsilon
    support_finite_difference = (
        partition_sum(root.detach(), support_plus)
        - partition_sum(root.detach(), support_minus)
    ) / (2.0 * epsilon)
    root_fd_error = abs(
        root_finite_difference
        - float(exact.root_marginal[(0, *root_coordinate)].item())
    )
    support_fd_error = abs(
        support_finite_difference
        - float(exact.support_marginal[(0, *support_coordinate)].item())
    )
    metrics["root_finite_difference_abs"] = root_fd_error
    metrics["support_finite_difference_abs"] = support_fd_error
    state_counts = [int(value) for value in brute.state_count.tolist()]
    checks = {
        "all_compared_tensors_finite": all(finite_flags),
        "logZ_positive_within_tolerance": metrics["logZ_positive_max_abs"]
        is not None
        and metrics["logZ_positive_max_abs"] < TOLERANCES["log_partition_abs"],
        "logZ_total_within_tolerance": metrics["logZ_total_max_abs"] is not None
        and metrics["logZ_total_max_abs"] < TOLERANCES["log_partition_abs"],
        "map_energy_within_tolerance": metrics["map_energy_max_abs"] is not None
        and metrics["map_energy_max_abs"] < TOLERANCES["map_energy_abs"],
        "map_support_bit_exact": map_support_equal,
        "map_root_exact": map_root_equal,
        "map_backpointer_exact": map_intervals_equal,
        "unique_map_no_tie": minimum_map_gap
        > TOLERANCES["minimum_unique_map_gap"],
        "root_marginals_within_tolerance": metrics["root_marginal_max_abs"]
        is not None
        and metrics["root_marginal_max_abs"] < TOLERANCES["marginal_abs"],
        "support_marginals_within_tolerance": metrics[
            "support_marginal_max_abs"
        ]
        is not None
        and metrics["support_marginal_max_abs"] < TOLERANCES["marginal_abs"],
        "root_autograd_matches_finite_difference": root_fd_error
        < TOLERANCES["finite_difference_abs"],
        "support_autograd_matches_finite_difference": support_fd_error
        < TOLERANCES["finite_difference_abs"],
    }
    return {
        "case_id": case_id,
        "window": {"height": height, "width": width},
        "batch_fields": int(valid_support.shape[0]),
        "energy_regime": regime,
        "score_bound": scale,
        "dtype": "torch.float64",
        "device": "cpu",
        "fixture_sha256": fixture_hash,
        "state_counts": state_counts,
        "minimum_unique_map_gap": minimum_map_gap,
        "metrics": metrics,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "NO-GO",
    }


def _verify_zero_score() -> dict[str, object]:
    cases = []
    for height, width in WINDOW_SHAPES:
        valid_support, valid_root = _admissible_masks(height, width)
        chains = enumerate_run_chains(
            valid_support, valid_root, max_states=MAX_STATES
        )
        reference_count = torch.tensor(
            [len(field) for field in chains], dtype=torch.long
        )
        expected_logK = torch.log(reference_count.to(dtype=torch.float64))
        observed_logK = zero_score_log_cardinality(
            valid_support, valid_root, dtype=torch.float64
        )
        observed_cardinality = zero_score_cardinality(
            valid_support, valid_root, dtype=torch.float64
        )
        zeros = torch.zeros(
            (valid_support.shape[0], height, width), dtype=torch.float64
        )
        corrected = RootCellRunSemiring(cardinality_correction=True)(
            zeros,
            zeros,
            valid_support,
            valid_root,
            return_map=False,
        )
        logK_finite, logK_error = _finite_max_abs(observed_logK, expected_logK)
        count_finite, cardinality_error = _finite_max_abs(
            observed_cardinality, reference_count.to(dtype=torch.float64)
        )
        zero_finite, zero_partition_error = _finite_max_abs(
            corrected.logZ_positive, torch.zeros_like(corrected.logZ_positive)
        )
        checks = {
            "finite": logK_finite and count_finite and zero_finite,
            "logK_matches_enumeration": logK_error is not None
            and logK_error < TOLERANCES["zero_score_logK_abs"],
            "cardinality_matches_enumeration": cardinality_error is not None
            and cardinality_error < TOLERANCES["zero_score_cardinality_abs"],
            "cardinality_roundtrip_exact": bool(
                torch.equal(observed_cardinality.round().to(torch.long), reference_count)
            ),
            "corrected_zero_score_logZ_positive_is_zero": zero_partition_error
            is not None
            and zero_partition_error < TOLERANCES["log_partition_abs"],
        }
        cases.append(
            {
                "window": {"height": height, "width": width},
                "state_counts": [int(value) for value in reference_count.tolist()],
                "logK_max_abs": logK_error,
                "cardinality_max_abs": cardinality_error,
                "corrected_logZ_positive_max_abs": zero_partition_error,
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "NO-GO",
            }
        )
    return {
        "reference": "independent_run_chain_enumeration",
        "cases": cases,
        "status": "PASS"
        if cases and all(case["status"] == "PASS" for case in cases)
        else "NO-GO",
    }


def _verify_prior_calibration() -> dict[str, object]:
    rows = []
    global_probability_error = 0.0
    global_log_partition_error = 0.0
    all_finite = True
    for height, width in WINDOW_SHAPES:
        support_mask, root_mask = _admissible_masks(height, width)
        batch = support_mask.shape[0]
        per_logit = []
        for bias in PRIOR_LOGITS:
            root = torch.full(
                (batch, height, width), bias, dtype=torch.float64
            )
            support = torch.zeros_like(root)
            result = RootCellRunSemiring(cardinality_correction=True)(
                root,
                support,
                support_mask,
                root_mask,
                return_map=False,
            )
            expected_logZ_positive = torch.full((batch,), bias, dtype=torch.float64)
            expected_probability = torch.sigmoid(expected_logZ_positive)
            log_finite, log_error = _finite_max_abs(
                result.logZ_positive, expected_logZ_positive
            )
            probability_finite, probability_error = _finite_max_abs(
                result.p_nonempty, expected_probability
            )
            row_finite = log_finite and probability_finite
            all_finite = all_finite and row_finite
            if log_error is not None:
                global_log_partition_error = max(global_log_partition_error, log_error)
            if probability_error is not None:
                global_probability_error = max(
                    global_probability_error, probability_error
                )
            per_logit.append(
                {
                    "logit": bias,
                    "expected_p_nonempty": float(expected_probability[0].item()),
                    "max_p_nonempty_abs": probability_error,
                    "max_logZ_positive_abs": log_error,
                    "finite": row_finite,
                }
            )
        rows.append(
            {
                "window": {"height": height, "width": width},
                "fields_per_logit": int(batch),
                "results": per_logit,
            }
        )
    checks = {
        "includes_minus_30": -30.0 in PRIOR_LOGITS,
        "includes_plus_30": 30.0 in PRIOR_LOGITS,
        "all_outputs_finite": all_finite,
        "logZ_positive_equals_bias": global_log_partition_error
        < TOLERANCES["prior_log_partition_abs"],
        "p_nonempty_equals_sigmoid_bias": global_probability_error
        < TOLERANCES["prior_probability_abs"],
    }
    return {
        "root_energy": "constant_bias_on_every_pixel",
        "support_energy": "zero",
        "logits": list(PRIOR_LOGITS),
        "windows": rows,
        "max_logZ_positive_abs": global_log_partition_error,
        "max_p_nonempty_abs": global_probability_error,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "NO-GO",
    }


def _base_payload(seed: int | None) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "scope": "exact_run_semiring_core_only",
        "status": "NO-GO",
        "seed": seed,
        "deterministic": True,
        "precision": "FP64",
        "device": "CPU",
        "state_family": (
            "empty_or_nonempty_consecutive_horizontal_run_chain_8_connected_"
            "canonical_root_is_first_run_left_endpoint"
        ),
        "reference": "independent exhaustive enumeration in brute_force_reference",
        "windows": [[height, width] for height, width in WINDOW_SHAPES],
        "energy_regimes": [
            {"name": name, "score_bound": scale}
            for name, scale in ENERGY_REGIMES
        ],
        "tolerances": dict(TOLERANCES),
        "comparison_rule": "strict_absolute_error_less_than_tolerance",
        "source_sha256": _source_inventory(),
        "pending_integration_checks": [
            dict(item) for item in PENDING_INTEGRATION_CHECKS
        ],
        "full_t0_b_release_status": "PENDING",
    }


def _run_core_checks(seed: int) -> dict[str, object]:
    energy_cases = [
        _verify_energy_case(
            seed=seed,
            height=height,
            width=width,
            regime=regime,
            scale=scale,
        )
        for height, width in WINDOW_SHAPES
        for regime, scale in ENERGY_REGIMES
    ]
    zero_score = _verify_zero_score()
    prior = _verify_prior_calibration()
    criteria = {
        "fp64_cpu_only": all(
            case["dtype"] == "torch.float64" and case["device"] == "cpu"
            for case in energy_cases
        ),
        "three_by_three_covered": any(
            case["window"] == {"height": 3, "width": 3}
            for case in energy_cases
        ),
        "four_by_four_covered": any(
            case["window"] == {"height": 4, "width": 4}
            for case in energy_cases
        ),
        "five_by_five_covered": any(
            case["window"] == {"height": 5, "width": 5}
            for case in energy_cases
        ),
        "nominal_and_extreme_scores_covered": {
            case["energy_regime"] for case in energy_cases
        }
        == {"nominal", "extreme"},
        "all_dp_brute_cases_pass": bool(energy_cases)
        and all(case["status"] == "PASS" for case in energy_cases),
        "zero_score_count_and_logK_pass": zero_score["status"] == "PASS",
        "prior_calibration_pass": prior["status"] == "PASS",
    }
    return {
        "energy_cases": energy_cases,
        "zero_score": zero_score,
        "prior_calibration": prior,
        "criteria": criteria,
    }


def run_verification(seed: int = DEFAULT_SEED) -> dict[str, object]:
    """Return one deterministic authenticated PASS/NO-GO core report."""

    seed_for_report = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
    payload = _base_payload(seed_for_report)
    try:
        if seed_for_report is None or seed_for_report < 0:
            raise TraceDPVerificationError("seed must be a non-negative integer")
        results = _run_core_checks(seed_for_report)
        criteria = results["criteria"]
        if not isinstance(criteria, dict) or not criteria:
            raise TraceDPVerificationError("verification criteria are missing")
        payload.update(results)
        payload["status"] = "PASS" if all(criteria.values()) else "NO-GO"
        payload["failure"] = None
    except Exception as exc:  # authenticated fail-closed artifact
        payload["status"] = "NO-GO"
        payload["criteria"] = {
            "verification_completed_without_exception": False,
        }
        # Deliberately omit the raw message: dependency errors can contain
        # absolute workspace paths, which are forbidden in authenticated data.
        payload["failure"] = {
            "type": type(exc).__name__,
            "message_code": "verification_exception",
        }
    return _finalize_report(payload)


def render_report(report: Mapping[str, object]) -> str:
    authenticate_report(report)
    return json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="return zero for an authenticated NO-GO report (default exit is 2)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_verification(seed=args.seed)
    rendered = render_report(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["status"] != "PASS" and not args.report_only:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEED",
    "GATE",
    "PRIOR_LOGITS",
    "SCHEMA_VERSION",
    "TOLERANCES",
    "TraceDPVerificationError",
    "authenticate_report",
    "canonical_json_sha256",
    "main",
    "render_report",
    "run_verification",
]
