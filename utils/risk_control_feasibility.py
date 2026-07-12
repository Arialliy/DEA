"""Best-case finite-sample feasibility checks for Gate F0 risk control.

The North-Star false-alarm budgets are area fractions of only 1e-6 to 2e-5.
This module asks whether standard distribution-free bounded-loss baselines can
possibly certify such risks at the available image sample sizes, even under
the optimistic assumption of zero empirical loss.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


PIXELS_PER_MPIX = 1_000_000
DEFAULT_CONFIDENCE_DELTAS = (0.10, 0.05, 0.01)


class RiskControlFeasibilityError(RuntimeError):
    """Raised when a Gate F0 feasibility input is malformed."""


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RiskControlFeasibilityError(f"{label} must be a positive integer")
    return value


def _unit_interval(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RiskControlFeasibilityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise RiskControlFeasibilityError(f"{label} must lie strictly in (0, 1)")
    return result


def hb_ltt_zero_loss_pvalue(sample_count: int, target_risk: float) -> float:
    """Minimum HB-LTT p-value at zero empirical bounded loss.

    For H0: R > alpha and losses in [0, 1], the hybrid
    Hoeffding--Bentkus p-value at empirical risk zero is
    ``(1 - alpha) ** n``.  Any positive empirical loss is no more favorable.
    """

    n = _positive_int(sample_count, label="sample_count")
    alpha = _unit_interval(target_risk, label="target_risk")
    return math.exp(n * math.log1p(-alpha))


def minimum_zero_loss_sample_count(
    target_risk: float,
    rejection_level: float,
) -> int:
    """Smallest n with (1-alpha)^n <= rejection_level."""

    alpha = _unit_interval(target_risk, label="target_risk")
    level = _unit_interval(rejection_level, label="rejection_level")
    estimate = max(1, math.ceil(math.log(level) / math.log1p(-alpha)))
    while hb_ltt_zero_loss_pvalue(estimate, alpha) > level:
        estimate += 1
    while estimate > 1 and hb_ltt_zero_loss_pvalue(estimate - 1, alpha) <= level:
        estimate -= 1
    return estimate


def standard_crc_unit_bound_floor(sample_count: int) -> float:
    """The B/(n+1) correction term for standard CRC with B=1."""

    n = _positive_int(sample_count, label="sample_count")
    return 1.0 / float(n + 1)


def minimum_crc_sample_count_for_budget(budget_fa_per_mpix: int) -> int:
    """Smallest n such that 1/(n+1) <= budget/1e6."""

    budget = _positive_int(budget_fa_per_mpix, label="budget_fa_per_mpix")
    if budget >= PIXELS_PER_MPIX:
        raise RiskControlFeasibilityError("budget must be below one Mpix")
    quotient, remainder = divmod(PIXELS_PER_MPIX, budget)
    ceiling = quotient + int(remainder > 0)
    return max(1, ceiling - 1)


def _sample_scope_rows(
    fold_sizes: Mapping[str, Mapping[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(fold_sizes):
        sizes = fold_sizes[dataset]
        if not isinstance(sizes, Mapping) or len(sizes) < 2:
            raise RiskControlFeasibilityError(
                f"{dataset} must contain at least two image folds"
            )
        normalized: dict[int, int] = {}
        for fold, count in sizes.items():
            if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
                raise RiskControlFeasibilityError("fold ids must be non-negative ints")
            normalized[fold] = _positive_int(count, label="fold image count")
        total = sum(normalized.values())
        for evaluation_fold in sorted(normalized):
            rows.append(
                {
                    "dataset": dataset,
                    "sample_scope": "crossfit_calibration_images",
                    "evaluation_fold": evaluation_fold,
                    "sample_count": total - normalized[evaluation_fold],
                    "development_image_count": total,
                    "deployable_split": True,
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "sample_scope": "all_development_images_optimistic_ceiling",
                "evaluation_fold": None,
                "sample_count": total,
                "development_image_count": total,
                "deployable_split": False,
            }
        )
    return rows


def analyze_risk_control_feasibility(
    fold_sizes: Mapping[str, Mapping[int, int]],
    *,
    budgets_fa_per_mpix: Sequence[int],
    candidate_grid_size: int,
    confidence_deltas: Sequence[float] = DEFAULT_CONFIDENCE_DELTAS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the optimistic HB-LTT/standard-CRC feasibility ledger."""

    budgets = tuple(
        _positive_int(value, label="budget_fa_per_mpix")
        for value in budgets_fa_per_mpix
    )
    if not budgets or len(set(budgets)) != len(budgets):
        raise RiskControlFeasibilityError("budgets must be unique and non-empty")
    if any(value >= PIXELS_PER_MPIX for value in budgets):
        raise RiskControlFeasibilityError("budgets must be below one Mpix")
    grid_size = _positive_int(candidate_grid_size, label="candidate_grid_size")
    deltas = tuple(
        _unit_interval(value, label="confidence_delta")
        for value in confidence_deltas
    )
    if not deltas or len(set(deltas)) != len(deltas):
        raise RiskControlFeasibilityError(
            "confidence deltas must be unique and non-empty"
        )

    rows: list[dict[str, Any]] = []
    for scope in _sample_scope_rows(fold_sizes):
        n = int(scope["sample_count"])
        for budget in budgets:
            alpha = float(budget) / PIXELS_PER_MPIX
            p_zero = hb_ltt_zero_loss_pvalue(n, alpha)
            crc_floor = standard_crc_unit_bound_floor(n)
            confidence: dict[str, Any] = {}
            for delta in deltas:
                fixed_required = minimum_zero_loss_sample_count(alpha, delta)
                bonferroni_level = delta / grid_size
                bonferroni_required = minimum_zero_loss_sample_count(
                    alpha, bonferroni_level
                )
                confidence[str(delta)] = {
                    "delta": delta,
                    "hb_ltt_fixed_sequence_rejection_level": delta,
                    "hb_ltt_bonferroni_rejection_level": bonferroni_level,
                    "zero_loss_fixed_sequence_certifiable": p_zero <= delta,
                    "zero_loss_bonferroni_certifiable": p_zero
                    <= bonferroni_level,
                    "minimum_sample_count_fixed_sequence_zero_loss": fixed_required,
                    "minimum_sample_count_bonferroni_zero_loss": bonferroni_required,
                    "fixed_sequence_sample_shortfall": max(0, fixed_required - n),
                    "bonferroni_sample_shortfall": max(
                        0, bonferroni_required - n
                    ),
                }
            rows.append(
                {
                    "schema_version": "dea.gate_f0.risk_feasibility_row.v1",
                    **scope,
                    "budget_fa_per_mpix": budget,
                    "target_area_fraction_risk": alpha,
                    "bounded_image_loss": "unmatched prediction area / image pixels",
                    "loss_bound": 1.0,
                    "hb_ltt_best_case": {
                        "assumed_empirical_loss": 0.0,
                        "zero_loss_pvalue": p_zero,
                        "confidence": confidence,
                    },
                    "standard_crc": {
                        "unit_bound_correction_floor": crc_floor,
                        "correction_floor_within_target": crc_floor <= alpha,
                        "minimum_sample_count_for_correction_floor": (
                            minimum_crc_sample_count_for_budget(budget)
                        ),
                    },
                }
            )

    crossfit_rows = [
        row for row in rows if row["sample_scope"] == "crossfit_calibration_images"
    ]
    optimistic_rows = [
        row
        for row in rows
        if row["sample_scope"]
        == "all_development_images_optimistic_ceiling"
    ]
    by_budget: dict[str, Any] = {}
    for budget in budgets:
        budget_crossfit = [
            row for row in crossfit_rows if row["budget_fa_per_mpix"] == budget
        ]
        budget_optimistic = [
            row for row in optimistic_rows if row["budget_fa_per_mpix"] == budget
        ]
        by_budget[str(budget)] = {
            "maximum_crossfit_calibration_image_count": max(
                int(row["sample_count"]) for row in budget_crossfit
            ),
            "maximum_full_development_image_count": max(
                int(row["sample_count"]) for row in budget_optimistic
            ),
            "best_zero_loss_pvalue_across_crossfit_folds": min(
                float(row["hb_ltt_best_case"]["zero_loss_pvalue"])
                for row in budget_crossfit
            ),
            "best_zero_loss_pvalue_across_full_development_sets": min(
                float(row["hb_ltt_best_case"]["zero_loss_pvalue"])
                for row in budget_optimistic
            ),
            "any_crossfit_fixed_sequence_certifiable": any(
                any(
                    value["zero_loss_fixed_sequence_certifiable"]
                    for value in row["hb_ltt_best_case"]["confidence"].values()
                )
                for row in budget_crossfit
            ),
            "any_full_development_fixed_sequence_certifiable": any(
                any(
                    value["zero_loss_fixed_sequence_certifiable"]
                    for value in row["hb_ltt_best_case"]["confidence"].values()
                )
                for row in budget_optimistic
            ),
            "any_crossfit_standard_crc_floor_within_target": any(
                row["standard_crc"]["correction_floor_within_target"]
                for row in budget_crossfit
            ),
            "minimum_hb_ltt_images_at_delta_0.1_single_candidate": (
                minimum_zero_loss_sample_count(budget / PIXELS_PER_MPIX, 0.1)
            ),
            "minimum_hb_ltt_images_at_delta_0.1_bonferroni": (
                minimum_zero_loss_sample_count(
                    budget / PIXELS_PER_MPIX, 0.1 / grid_size
                )
            ),
            "minimum_standard_crc_images_for_unit_bound_floor": (
                minimum_crc_sample_count_for_budget(budget)
            ),
        }

    summary = {
        "schema_version": "dea.gate_f0.risk_feasibility_summary.v1",
        "analysis_scope": (
            "analytic best-case sample-size precheck; no model inference and no "
            "threshold outcomes are required"
        ),
        "risk_mapping": (
            "for equal-size canonical images, mean(unmatched_area/image_pixels) "
            "equals pooled FA/Mpix divided by 1e6"
        ),
        "optimistic_assumptions": [
            "i.i.d./exchangeable calibration images",
            "bounded per-image loss in [0,1]",
            "zero empirical false-alarm loss",
            "candidate family fixed independently of labeled calibration data",
            "fixed-sequence single-candidate testing receives the full delta",
            "all development images are additionally shown as a non-deployable sample ceiling",
        ],
        "counts": {
            "dataset_count": len(fold_sizes),
            "row_count": len(rows),
            "candidate_grid_size": grid_size,
        },
        "protocol": {
            "budgets_fa_per_mpix": list(budgets),
            "confidence_deltas": list(deltas),
            "hb_ltt_zero_loss_pvalue": "(1-alpha)^n",
            "bonferroni_level": "delta / candidate_grid_size",
            "standard_crc_unit_bound_floor": "1/(n+1)",
        },
        "by_budget": by_budget,
        "pre_gate": {
            "any_best_case_crossfit_hb_ltt_certifiable": any(
                value["any_crossfit_fixed_sequence_certifiable"]
                for value in by_budget.values()
            ),
            "any_optimistic_full_development_hb_ltt_certifiable": any(
                value["any_full_development_fixed_sequence_certifiable"]
                for value in by_budget.values()
            ),
            "any_crossfit_standard_crc_floor_within_target": any(
                value["any_crossfit_standard_crc_floor_within_target"]
                for value in by_budget.values()
            ),
            "full_threshold_inference_needed_for_this_precheck": False,
            "deterministic_all_off_note": (
                "an a-priori no-prediction rule has zero FA but zero Pd and is "
                "excluded by the North-Star non-all-off requirement"
            ),
            "decision": (
                "standard distribution-free HB-LTT and unit-bound CRC are "
                "sample-size-vacuous at the frozen FA budgets"
            ),
        },
        "interpretation": {
            "raw_crc_applicability": (
                "the component false-area loss is non-monotone, so standard CRC "
                "is not directly applicable; its 1/(n+1) term is used only as an "
                "optimistic necessary floor for a monotone or majorant variant"
            ),
            "zero_loss_information_limit": (
                "under a Bernoulli risk just above alpha, an all-zero calibration "
                "sample still occurs with probability approximately (1-alpha)^n"
            ),
            "cache_routing": (
                "a full outcome cache cannot change this best-case sample-size "
                "precheck, but may still be needed for oracle/frontier diagnostics"
            ),
        },
        "scope_limit": (
            "this precheck does not prove impossibility under additional "
            "distributional/spatial assumptions, does not evaluate oracle Pd, "
            "and does not alter Gate E-1c"
        ),
    }
    return rows, summary
