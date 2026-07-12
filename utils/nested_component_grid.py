"""Nested quantile-grid audit for frozen component operating points.

The utility keeps the existing Gate E quantile grid as ``Q0`` and refines
only the quantile probabilities:

* ``Q0`` is :data:`DEFAULT_TAIL_QUANTILES`;
* ``Q1`` inserts one arithmetic midpoint between every adjacent Q0 pair;
* ``Q2`` repeats the same refinement on Q1.

Only the Q2 threshold curve is evaluated for each matcher.  Q0 and Q1 are
exact threshold-membership projections of that curve, so comparing levels
cannot accidentally change component evaluation semantics.  FA feasibility
uses integer cross multiplication and never relies on the stored floating
``fa_per_million_pixels`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterable, Sequence

import numpy as np

from utils.component_operating_point import (
    DEFAULT_TAIL_QUANTILES,
    ComponentOperatingPoint,
    build_logit_threshold_grid,
    evaluate_component_operating_points,
)


GRID_LEVELS = ("Q0", "Q1", "Q2")
MATCHER_IMPLEMENTATIONS = (
    ("official_legacy", "legacy"),
    ("audit_hungarian", "hungarian"),
)
PIXELS_PER_MILLION = 1_000_000


class NestedComponentGridError(ValueError):
    """Raised when a nested-grid contract is invalid or infeasible."""


@dataclass(frozen=True)
class QuantileProbabilityGrid:
    """One named, strictly increasing quantile-probability grid."""

    level: str
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class ExactBudgetSelection:
    """Lexicographic operating-point choice under an exact integer budget."""

    budget_fa_per_million_pixels: int
    operating_point: ComponentOperatingPoint

    @property
    def threshold(self) -> float:
        return self.operating_point.threshold

    @property
    def integer_margin(self) -> int:
        """Non-negative budget slack in the cross-multiplied integer scale."""

        point = self.operating_point
        return (
            self.budget_fa_per_million_pixels * point.total_pixels
            - point.unmatched_prediction_area * PIXELS_PER_MILLION
        )


@dataclass(frozen=True)
class NestedGridLevelResult:
    """Projected curve and exact-budget selections for one grid level."""

    level: str
    probabilities: tuple[float, ...]
    threshold_grid: tuple[float, ...]
    curve: tuple[ComponentOperatingPoint, ...]
    selections: tuple[ExactBudgetSelection, ...]


@dataclass(frozen=True)
class NestedMatcherGridResult:
    """All nested-grid results for one component matcher."""

    matcher: str
    matching_implementation: str
    levels: tuple[NestedGridLevelResult, ...]

    def level(self, name: str) -> NestedGridLevelResult:
        for result in self.levels:
            if result.level == name:
                return result
        raise KeyError(name)


@dataclass(frozen=True)
class NestedComponentGridResult:
    """Complete two-matcher nested-grid audit result."""

    fixed_thresholds: tuple[float, ...]
    probability_grids: tuple[QuantileProbabilityGrid, ...]
    matchers: tuple[NestedMatcherGridResult, ...]

    def matcher(self, name: str) -> NestedMatcherGridResult:
        for result in self.matchers:
            if result.matcher == name:
                return result
        raise KeyError(name)


def _probability(value: object, *, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise NestedComponentGridError(
            f"base_quantiles[{index}] must be a finite real number"
        )
    result = float(value)
    if not np.isfinite(result):
        raise NestedComponentGridError(
            f"base_quantiles[{index}] must be a finite real number"
        )
    if result < 0.5 or result > 1.0:
        raise NestedComponentGridError("base quantiles must lie in [0.5, 1]")
    return result


def _refine_probability_grid(
    probabilities: tuple[float, ...],
) -> tuple[float, ...]:
    refined = [probabilities[0]]
    for left, right in zip(probabilities, probabilities[1:]):
        midpoint = left + (right - left) / 2.0
        if not left < midpoint < right:
            raise NestedComponentGridError(
                "adjacent base quantiles are too close for midpoint refinement"
            )
        refined.extend((midpoint, right))
    return tuple(refined)


def build_nested_quantile_probability_grids(
    base_quantiles: Sequence[float] = DEFAULT_TAIL_QUANTILES,
) -> tuple[QuantileProbabilityGrid, ...]:
    """Return deterministic nested Q0/Q1/Q2 probability grids.

    The base grid is not sorted or deduplicated silently.  Requiring a
    strictly increasing input makes Q0 identity auditable and prevents an
    accidental protocol change.
    """

    q0 = tuple(
        _probability(value, index=index)
        for index, value in enumerate(base_quantiles)
    )
    if len(q0) < 2:
        raise NestedComponentGridError(
            "base_quantiles must contain at least two probabilities"
        )
    if any(left >= right for left, right in zip(q0, q0[1:])):
        raise NestedComponentGridError(
            "base_quantiles must be strictly increasing and unique"
        )

    q1 = _refine_probability_grid(q0)
    q2 = _refine_probability_grid(q1)
    grids = (
        QuantileProbabilityGrid("Q0", q0),
        QuantileProbabilityGrid("Q1", q1),
        QuantileProbabilityGrid("Q2", q2),
    )
    if not set(q0).issubset(q1) or not set(q1).issubset(q2):
        raise AssertionError("nested quantile construction lost a parent point")
    return grids


def _non_negative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise NestedComponentGridError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise NestedComponentGridError(f"{name} must be a non-negative integer")
    return result


def exact_budget_feasible(
    unmatched_prediction_area: int,
    total_pixels: int,
    budget_fa_per_million_pixels: int,
) -> bool:
    """Test FA/Mpix feasibility without a floating-point division."""

    area = _non_negative_integer(
        unmatched_prediction_area,
        name="unmatched_prediction_area",
    )
    pixels = _non_negative_integer(total_pixels, name="total_pixels")
    if pixels == 0:
        raise NestedComponentGridError("total_pixels must be a positive integer")
    budget = _non_negative_integer(
        budget_fa_per_million_pixels,
        name="budget_fa_per_million_pixels",
    )
    return area * PIXELS_PER_MILLION <= budget * pixels


def _validated_budgets(
    budgets_fa_per_million_pixels: Iterable[int],
) -> tuple[int, ...]:
    budgets = tuple(
        _non_negative_integer(value, name="budgets_fa_per_million_pixels")
        for value in budgets_fa_per_million_pixels
    )
    if not budgets:
        raise NestedComponentGridError(
            "budgets_fa_per_million_pixels must be non-empty"
        )
    if len(set(budgets)) != len(budgets):
        raise NestedComponentGridError(
            "budgets_fa_per_million_pixels must be unique"
        )
    return budgets


def select_exact_budget_point(
    curve: Sequence[ComponentOperatingPoint],
    budget_fa_per_million_pixels: int,
) -> ExactBudgetSelection:
    """Select one point by matches, false area, then conservative threshold.

    Feasibility is decided from integer counts.  The floating FA value stored
    on a point is deliberately ignored.
    """

    budget = _non_negative_integer(
        budget_fa_per_million_pixels,
        name="budget_fa_per_million_pixels",
    )
    points = tuple(curve)
    if not points:
        raise NestedComponentGridError("curve must be non-empty")
    if not all(isinstance(point, ComponentOperatingPoint) for point in points):
        raise NestedComponentGridError(
            "curve must contain only ComponentOperatingPoint values"
        )
    population = {
        (point.sample_count, point.total_pixels, point.target_components)
        for point in points
    }
    if len(population) != 1:
        raise NestedComponentGridError(
            "all curve points must describe the same population"
        )
    sample_count, total_pixels, target_components = next(iter(population))
    if sample_count <= 0 or total_pixels <= 0:
        raise NestedComponentGridError("curve population size must be positive")
    if target_components <= 0:
        raise NestedComponentGridError(
            "exact-budget Pd selection requires at least one target component"
        )
    if len({point.threshold for point in points}) != len(points):
        raise NestedComponentGridError("curve thresholds must be unique")
    if any(not np.isfinite(point.threshold) for point in points):
        raise NestedComponentGridError("curve thresholds must be finite")

    feasible = tuple(
        point
        for point in points
        if exact_budget_feasible(
            point.unmatched_prediction_area,
            point.total_pixels,
            budget,
        )
    )
    if not feasible:
        minimum_area = min(point.unmatched_prediction_area for point in points)
        raise NestedComponentGridError(
            f"no evaluated threshold is feasible for budget {budget}; "
            f"minimum unmatched prediction area is {minimum_area}"
        )
    chosen = max(
        feasible,
        key=lambda point: (
            point.matched_components,
            -point.unmatched_prediction_area,
            point.threshold,
        ),
    )
    selection = ExactBudgetSelection(budget, chosen)
    if selection.integer_margin < 0:
        raise AssertionError("exact selector returned an infeasible point")
    return selection


def _threshold_grids(
    logits: tuple[object, ...],
    probability_grids: tuple[QuantileProbabilityGrid, ...],
    fixed_thresholds: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    grids = tuple(
        build_logit_threshold_grid(
            logits,
            fixed_thresholds=fixed_thresholds,
            tail_quantiles=grid.probabilities,
        )
        for grid in probability_grids
    )
    for parent, child in zip(grids, grids[1:]):
        if not set(parent).issubset(child):
            raise NestedComponentGridError(
                "quantile refinement did not produce nested threshold grids"
            )
    return grids


def evaluate_nested_component_grids(
    logit_samples: Iterable[object],
    target_samples: Iterable[object],
    budgets_fa_per_million_pixels: Iterable[int],
    *,
    fixed_thresholds: Sequence[float] = (0.0,),
    base_quantiles: Sequence[float] = DEFAULT_TAIL_QUANTILES,
    centroid_radius: float = 3.0,
    connectivity: int = 2,
) -> NestedComponentGridResult:
    """Evaluate Q2 once per matcher and project exact Q0/Q1 subcurves."""

    logits = tuple(logit_samples)
    targets = tuple(target_samples)
    if not logits:
        raise NestedComponentGridError(
            "at least one logit/target sample is required"
        )
    if len(logits) != len(targets):
        raise NestedComponentGridError(
            "logit_samples and target_samples must have equal length"
        )
    fixed = tuple(fixed_thresholds)
    budgets = _validated_budgets(budgets_fa_per_million_pixels)
    probability_grids = build_nested_quantile_probability_grids(base_quantiles)
    threshold_grids = _threshold_grids(
        logits,
        probability_grids,
        fixed,
    )
    q2_thresholds = threshold_grids[-1]

    matcher_results = []
    for matcher, matching_implementation in MATCHER_IMPLEMENTATIONS:
        q2_curve = evaluate_component_operating_points(
            logits,
            targets,
            q2_thresholds,
            matching=matching_implementation,
            centroid_radius=centroid_radius,
            connectivity=connectivity,
        )
        q2_by_threshold = {point.threshold: point for point in q2_curve}
        if len(q2_by_threshold) != len(q2_curve):
            raise AssertionError("Q2 component curve contains duplicate thresholds")

        levels = []
        for probability_grid, threshold_grid in zip(
            probability_grids,
            threshold_grids,
        ):
            try:
                projected_curve = tuple(
                    q2_by_threshold[threshold] for threshold in threshold_grid
                )
            except KeyError as exc:
                raise NestedComponentGridError(
                    "a coarse threshold is absent from the evaluated Q2 curve"
                ) from exc
            selections = tuple(
                select_exact_budget_point(projected_curve, budget)
                for budget in budgets
            )
            levels.append(
                NestedGridLevelResult(
                    level=probability_grid.level,
                    probabilities=probability_grid.probabilities,
                    threshold_grid=threshold_grid,
                    curve=projected_curve,
                    selections=selections,
                )
            )
        matcher_results.append(
            NestedMatcherGridResult(
                matcher=matcher,
                matching_implementation=matching_implementation,
                levels=tuple(levels),
            )
        )

    # The three canonical builder calls above validate every fixed value.
    # Retain their normalized protocol values even when one also happens to
    # equal a quantile-derived threshold.
    canonical_fixed = tuple(sorted(set(float(value) for value in fixed)))
    return NestedComponentGridResult(
        fixed_thresholds=canonical_fixed,
        probability_grids=probability_grids,
        matchers=tuple(matcher_results),
    )


def _selection_by_budget(
    level: NestedGridLevelResult,
) -> dict[int, ExactBudgetSelection]:
    return {
        selection.budget_fa_per_million_pixels: selection
        for selection in level.selections
    }


def summarize_nested_component_grids(
    result: NestedComponentGridResult,
) -> dict[str, Any]:
    """Return a deterministic JSON-serializable nested-grid comparison."""

    if not isinstance(result, NestedComponentGridResult):
        raise NestedComponentGridError(
            "result must be a NestedComponentGridResult"
        )
    level_sizes = {}
    for grid in result.probability_grids:
        threshold_counts = {
            matcher.matcher: len(matcher.level(grid.level).threshold_grid)
            for matcher in result.matchers
        }
        if len(set(threshold_counts.values())) != 1:
            raise NestedComponentGridError(
                "threshold grid size differs between matchers"
            )
        level_sizes[grid.level] = {
            "quantile_probability_count": len(grid.probabilities),
            "unique_threshold_count": next(iter(threshold_counts.values())),
        }

    matcher_summaries: dict[str, Any] = {}
    for matcher in result.matchers:
        q2 = matcher.level("Q2")
        q2_selections = _selection_by_budget(q2)
        levels: dict[str, Any] = {}
        change_count = 0
        for level in matcher.levels:
            selections = {}
            for selection in level.selections:
                budget = selection.budget_fa_per_million_pixels
                point = selection.operating_point
                reference = q2_selections[budget].operating_point
                threshold_changed = point.threshold != reference.threshold
                change_count += int(level.level != "Q2" and threshold_changed)
                selections[str(budget)] = {
                    "threshold": point.threshold,
                    "matched_components": point.matched_components,
                    "target_components": point.target_components,
                    "unmatched_prediction_area": point.unmatched_prediction_area,
                    "total_pixels": point.total_pixels,
                    "fa_per_million_pixels": point.fa_per_million_pixels,
                    "budget_feasible_exact": exact_budget_feasible(
                        point.unmatched_prediction_area,
                        point.total_pixels,
                        budget,
                    ),
                    "budget_integer_margin": selection.integer_margin,
                    "threshold_changed_vs_q2": threshold_changed,
                    "matched_component_regret_vs_q2": (
                        reference.matched_components - point.matched_components
                    ),
                    "unmatched_area_difference_vs_q2": (
                        point.unmatched_prediction_area
                        - reference.unmatched_prediction_area
                    ),
                }
            levels[level.level] = {
                "quantile_probability_count": len(level.probabilities),
                "unique_threshold_count": len(level.threshold_grid),
                "budget_selections": selections,
            }
        matcher_summaries[matcher.matcher] = {
            "matching_implementation": matcher.matching_implementation,
            "q2_curve_point_count": len(q2.curve),
            "coarse_selection_threshold_change_count_vs_q2": change_count,
            "levels": levels,
        }

    probability_sets = [
        set(grid.probabilities) for grid in result.probability_grids
    ]
    threshold_sets = [
        set(result.matchers[0].level(level).threshold_grid)
        for level in GRID_LEVELS
    ]
    return {
        "schema_version": "dea.nested_component_grid.v1",
        "grid_levels": list(GRID_LEVELS),
        "fixed_thresholds": list(result.fixed_thresholds),
        "probability_grids_are_nested": all(
            parent.issubset(child)
            for parent, child in zip(probability_sets, probability_sets[1:])
        ),
        "threshold_grids_are_nested": all(
            parent.issubset(child)
            for parent, child in zip(threshold_sets, threshold_sets[1:])
        ),
        "level_sizes": level_sizes,
        "matchers": matcher_summaries,
    }
