"""Target-level decomposition on finite component operating-point families.

The routines in this module are diagnostic only.  A target-wise oracle may
choose a different threshold pair for every target, so its recoveries cannot
be summed into a deployable Pd.  A separate global pair oracle is provided to
make that distinction explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence

import numpy as np


PIXELS_PER_MILLION = 1_000_000
CATEGORY_SELECTED_HIT = "selected_hit"
CATEGORY_SELECTION_SENSITIVE = "targetwise_grid_recoverable"
CATEGORY_COMPONENT_CONVERSION = "active_local_peak_but_unmatched"
CATEGORY_PEAK_ORDER = "no_feasible_local_peak_activation"
CATEGORY_NO_FEASIBLE_PAIR = "no_feasible_finite_pair"
CATEGORY_MATCHER_SENSITIVE = "matcher_sensitive"
CATEGORIES = (
    CATEGORY_SELECTED_HIT,
    CATEGORY_SELECTION_SENSITIVE,
    CATEGORY_COMPONENT_CONVERSION,
    CATEGORY_PEAK_ORDER,
)

JOINT_DIRECTION_CATEGORIES = (
    CATEGORY_SELECTED_HIT,
    CATEGORY_MATCHER_SENSITIVE,
    CATEGORY_NO_FEASIBLE_PAIR,
    CATEGORY_SELECTION_SENSITIVE,
    CATEGORY_COMPONENT_CONVERSION,
    CATEGORY_PEAK_ORDER,
)


class FrontierDecompositionError(ValueError):
    """Raised when a finite frontier decomposition contract is malformed."""


@dataclass(frozen=True)
class FoldCandidateState:
    """One held-out fold state at a finite threshold or all-off sentinel."""

    threshold: float | None
    all_off_sentinel: bool
    total_pixels: int
    target_components: int
    matched_components: int
    prediction_components: int
    unmatched_prediction_components: int
    unmatched_prediction_area: int
    matched_target_ids: frozenset[str]
    support_active_target_ids: frozenset[str]
    core_active_target_ids: frozenset[str]


@dataclass(frozen=True)
class TargetFrontierStatus:
    """Mutually exclusive target status under an optimistic pooled oracle."""

    stable_target_id: str
    selected_matched: bool
    globally_oracle_matched: bool
    targetwise_exact_match_exists: bool
    targetwise_support_active_exists: bool
    targetwise_core_active_exists: bool
    category_support: str
    category_core: str
    feasible_finite_candidate_count: int
    minimum_exact_match_unmatched_area: int | None
    minimum_support_active_unmatched_area: int | None
    minimum_core_active_unmatched_area: int | None


@dataclass(frozen=True)
class GlobalOraclePair:
    """One simultaneously realizable threshold pair selected post hoc."""

    fold0_index: int
    fold1_index: int
    matched_components: int
    target_components: int
    unmatched_prediction_area: int
    total_pixels: int


@dataclass(frozen=True)
class JointTargetFrontierStatus:
    """Target status requiring matcher-joint feasibility and exact matching."""

    stable_target_id: str
    category_support: str
    category_core: str
    selected_legacy_matched: bool
    selected_hungarian_matched: bool
    joint_global_oracle_matched: bool
    targetwise_joint_exact_match_exists: bool
    targetwise_joint_support_active_exists: bool
    targetwise_joint_core_active_exists: bool
    joint_feasible_candidate_count: int


@dataclass(frozen=True)
class JointGlobalOraclePair:
    """One finite pair feasible and matched under every audited matcher."""

    fold0_index: int
    fold1_index: int
    joint_matched_target_ids: frozenset[str]
    matcher_unmatched_prediction_area: tuple[tuple[str, int], ...]
    total_pixels: int


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise FrontierDecompositionError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise FrontierDecompositionError(f"{name} must be a non-negative integer")
    return result


def _finite_threshold(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FrontierDecompositionError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise FrontierDecompositionError(f"{name} must be finite")
    return result


def validate_candidate_state(state: FoldCandidateState) -> FoldCandidateState:
    if not isinstance(state, FoldCandidateState):
        raise FrontierDecompositionError("state must be a FoldCandidateState")
    counts = {
        name: _nonnegative_integer(getattr(state, name), name=name)
        for name in (
            "total_pixels",
            "target_components",
            "matched_components",
            "prediction_components",
            "unmatched_prediction_components",
            "unmatched_prediction_area",
        )
    }
    if counts["total_pixels"] <= 0 or counts["target_components"] <= 0:
        raise FrontierDecompositionError("state population must be positive")
    if counts["matched_components"] > counts["target_components"]:
        raise FrontierDecompositionError("matches exceed target components")
    if counts["matched_components"] > counts["prediction_components"]:
        raise FrontierDecompositionError("matches exceed prediction components")
    if counts["unmatched_prediction_components"] > counts[
        "prediction_components"
    ]:
        raise FrontierDecompositionError("invalid unmatched prediction count")
    if len(state.matched_target_ids) != counts["matched_components"]:
        raise FrontierDecompositionError("matched target identity count drifted")
    for values, label in (
        (state.matched_target_ids, "matched_target_ids"),
        (state.support_active_target_ids, "support_active_target_ids"),
        (state.core_active_target_ids, "core_active_target_ids"),
    ):
        if not isinstance(values, frozenset) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise FrontierDecompositionError(f"{label} must be a string frozenset")
    if not state.core_active_target_ids.issubset(
        state.support_active_target_ids
    ):
        raise FrontierDecompositionError("core-active targets must be support-active")
    if state.all_off_sentinel:
        if state.threshold is not None:
            raise FrontierDecompositionError("all-off sentinel threshold must be None")
        if any(
            (
                counts["matched_components"],
                counts["prediction_components"],
                counts["unmatched_prediction_components"],
                counts["unmatched_prediction_area"],
                len(state.matched_target_ids),
                len(state.support_active_target_ids),
                len(state.core_active_target_ids),
            )
        ):
            raise FrontierDecompositionError("all-off sentinel must be empty")
    else:
        _finite_threshold(state.threshold, name="threshold")
    return state


def make_all_off_sentinel(
    *,
    total_pixels: int,
    target_components: int,
) -> FoldCandidateState:
    pixels = _nonnegative_integer(total_pixels, name="total_pixels")
    targets = _nonnegative_integer(target_components, name="target_components")
    if pixels <= 0 or targets <= 0:
        raise FrontierDecompositionError("all-off population must be positive")
    return FoldCandidateState(
        threshold=None,
        all_off_sentinel=True,
        total_pixels=pixels,
        target_components=targets,
        matched_components=0,
        prediction_components=0,
        unmatched_prediction_components=0,
        unmatched_prediction_area=0,
        matched_target_ids=frozenset(),
        support_active_target_ids=frozenset(),
        core_active_target_ids=frozenset(),
    )


def _validated_states(
    values: Iterable[FoldCandidateState],
    *,
    name: str,
) -> tuple[FoldCandidateState, ...]:
    states = tuple(validate_candidate_state(value) for value in values)
    if not states:
        raise FrontierDecompositionError(f"{name} must be non-empty")
    populations = {
        (state.total_pixels, state.target_components) for state in states
    }
    if len(populations) != 1:
        raise FrontierDecompositionError(f"{name} population changes across states")
    finite_thresholds = [
        state.threshold for state in states if not state.all_off_sentinel
    ]
    if len(set(finite_thresholds)) != len(finite_thresholds):
        raise FrontierDecompositionError(f"{name} finite thresholds must be unique")
    if sum(state.all_off_sentinel for state in states) > 1:
        raise FrontierDecompositionError(f"{name} has duplicate all-off sentinels")
    return states


def pooled_budget_feasible(
    unmatched_prediction_area: int,
    total_pixels: int,
    budget_fa_per_million_pixels: int,
) -> bool:
    area = _nonnegative_integer(
        unmatched_prediction_area, name="unmatched_prediction_area"
    )
    pixels = _nonnegative_integer(total_pixels, name="total_pixels")
    budget = _nonnegative_integer(
        budget_fa_per_million_pixels,
        name="budget_fa_per_million_pixels",
    )
    if pixels <= 0:
        raise FrontierDecompositionError("total_pixels must be positive")
    return area * PIXELS_PER_MILLION <= budget * pixels


def _threshold_preference(state: FoldCandidateState) -> tuple[int, float]:
    if state.all_off_sentinel:
        return (1, 0.0)
    return (0, float(state.threshold))


def select_global_oracle_pair(
    fold0_states: Iterable[FoldCandidateState],
    fold1_states: Iterable[FoldCandidateState],
    *,
    budget_fa_per_million_pixels: int,
) -> GlobalOraclePair:
    """Select one simultaneous pair by matches, area, then high thresholds."""

    states0 = _validated_states(fold0_states, name="fold0_states")
    states1 = _validated_states(fold1_states, name="fold1_states")
    total_pixels = states0[0].total_pixels + states1[0].total_pixels
    target_components = (
        states0[0].target_components + states1[0].target_components
    )
    feasible = []
    for index0, state0 in enumerate(states0):
        for index1, state1 in enumerate(states1):
            area = (
                state0.unmatched_prediction_area
                + state1.unmatched_prediction_area
            )
            if pooled_budget_feasible(
                area, total_pixels, budget_fa_per_million_pixels
            ):
                feasible.append((index0, index1, state0, state1, area))
    if not feasible:
        raise FrontierDecompositionError("candidate pair family has no feasible state")
    index0, index1, state0, state1, area = max(
        feasible,
        key=lambda value: (
            value[2].matched_components + value[3].matched_components,
            -value[4],
            _threshold_preference(value[2]),
            _threshold_preference(value[3]),
        ),
    )
    return GlobalOraclePair(
        fold0_index=index0,
        fold1_index=index1,
        matched_components=(
            state0.matched_components + state1.matched_components
        ),
        target_components=target_components,
        unmatched_prediction_area=area,
        total_pixels=total_pixels,
    )


def global_oracle_matched_target_ids(
    fold0_states: Sequence[FoldCandidateState],
    fold1_states: Sequence[FoldCandidateState],
    oracle: GlobalOraclePair,
) -> frozenset[str]:
    states0 = _validated_states(fold0_states, name="fold0_states")
    states1 = _validated_states(fold1_states, name="fold1_states")
    try:
        return frozenset(
            states0[oracle.fold0_index].matched_target_ids
            | states1[oracle.fold1_index].matched_target_ids
        )
    except IndexError as exc:
        raise FrontierDecompositionError("global oracle indices are invalid") from exc


def _minimum_area(
    states: Sequence[FoldCandidateState],
    stable_target_id: str,
    attribute: str,
) -> int | None:
    values = [
        state.unmatched_prediction_area
        for state in states
        if stable_target_id in getattr(state, attribute)
    ]
    return min(values) if values else None


def classify_fold_targets(
    finite_fold_states: Iterable[FoldCandidateState],
    *,
    other_fold_all_off: FoldCandidateState,
    budget_fa_per_million_pixels: int,
    selected_matched: Mapping[str, bool],
    globally_oracle_matched_target_ids: Iterable[str],
) -> tuple[TargetFrontierStatus, ...]:
    """Classify targets using an optimistic target-specific pooled oracle.

    The other fold is set to a predetermined ``+infinity`` all-off sentinel.
    Consequently, each target may use the entire pooled FA allowance in its
    own fold.  This is deliberately optimistic and must not be reported as a
    simultaneously realizable Pd.
    """

    states = _validated_states(finite_fold_states, name="finite_fold_states")
    if any(state.all_off_sentinel for state in states):
        raise FrontierDecompositionError(
            "finite_fold_states cannot contain an all-off sentinel"
        )
    other = validate_candidate_state(other_fold_all_off)
    if not other.all_off_sentinel:
        raise FrontierDecompositionError("other fold state must be all-off")
    if not isinstance(selected_matched, Mapping) or not selected_matched:
        raise FrontierDecompositionError("selected_matched must be non-empty")
    universe = set(selected_matched)
    if any(not isinstance(key, str) or not key for key in universe):
        raise FrontierDecompositionError("target ids must be non-empty strings")
    if any(not isinstance(value, bool) for value in selected_matched.values()):
        raise FrontierDecompositionError("selected match statuses must be boolean")
    if len(universe) != states[0].target_components:
        raise FrontierDecompositionError(
            "selected target universe disagrees with target component count"
        )
    observed = set().union(
        *(state.matched_target_ids for state in states),
        *(state.support_active_target_ids for state in states),
        *(state.core_active_target_ids for state in states),
    )
    if not observed.issubset(universe):
        raise FrontierDecompositionError("candidate target ids leave the universe")
    global_ids = frozenset(globally_oracle_matched_target_ids)
    if not global_ids.issubset(universe):
        raise FrontierDecompositionError("global oracle ids leave the universe")
    total_pixels = states[0].total_pixels + other.total_pixels
    feasible = tuple(
        state
        for state in states
        if pooled_budget_feasible(
            state.unmatched_prediction_area,
            total_pixels,
            budget_fa_per_million_pixels,
        )
    )
    results = []
    for stable_target_id in sorted(universe):
        selected = selected_matched[stable_target_id]
        exact = any(
            stable_target_id in state.matched_target_ids for state in feasible
        )
        support = any(
            stable_target_id in state.support_active_target_ids
            for state in feasible
        )
        core = any(
            stable_target_id in state.core_active_target_ids for state in feasible
        )
        if selected:
            category_support = CATEGORY_SELECTED_HIT
            category_core = CATEGORY_SELECTED_HIT
        elif exact:
            category_support = CATEGORY_SELECTION_SENSITIVE
            category_core = CATEGORY_SELECTION_SENSITIVE
        else:
            category_support = (
                CATEGORY_COMPONENT_CONVERSION if support else CATEGORY_PEAK_ORDER
            )
            category_core = (
                CATEGORY_COMPONENT_CONVERSION if core else CATEGORY_PEAK_ORDER
            )
        results.append(
            TargetFrontierStatus(
                stable_target_id=stable_target_id,
                selected_matched=selected,
                globally_oracle_matched=stable_target_id in global_ids,
                targetwise_exact_match_exists=exact,
                targetwise_support_active_exists=support,
                targetwise_core_active_exists=core,
                category_support=category_support,
                category_core=category_core,
                feasible_finite_candidate_count=len(feasible),
                minimum_exact_match_unmatched_area=_minimum_area(
                    feasible, stable_target_id, "matched_target_ids"
                ),
                minimum_support_active_unmatched_area=_minimum_area(
                    feasible, stable_target_id, "support_active_target_ids"
                ),
                minimum_core_active_unmatched_area=_minimum_area(
                    feasible, stable_target_id, "core_active_target_ids"
                ),
            )
        )
    return tuple(results)


def _validated_matcher_state_families(
    families: Mapping[str, Sequence[FoldCandidateState]],
    *,
    name: str,
) -> dict[str, tuple[FoldCandidateState, ...]]:
    if not isinstance(families, Mapping) or len(families) < 2:
        raise FrontierDecompositionError(
            f"{name} must contain at least two matcher families"
        )
    result = {
        str(matcher): _validated_states(states, name=f"{name}.{matcher}")
        for matcher, states in families.items()
    }
    lengths = {len(states) for states in result.values()}
    if len(lengths) != 1:
        raise FrontierDecompositionError(f"{name} matcher lengths differ")
    reference = next(iter(result.values()))
    for matcher, states in result.items():
        for index, (left, right) in enumerate(zip(reference, states)):
            if (
                left.threshold != right.threshold
                or left.all_off_sentinel != right.all_off_sentinel
                or left.total_pixels != right.total_pixels
                or left.target_components != right.target_components
                or left.support_active_target_ids
                != right.support_active_target_ids
                or left.core_active_target_ids != right.core_active_target_ids
            ):
                raise FrontierDecompositionError(
                    f"{name}.{matcher}[{index}] matcher-independent state drifted"
                )
    return result


def joint_feasible_pairs(
    fold0_families: Mapping[str, Sequence[FoldCandidateState]],
    fold1_families: Mapping[str, Sequence[FoldCandidateState]],
    *,
    budget_fa_per_million_pixels: int,
) -> tuple[tuple[int, int], ...]:
    """Return finite threshold pairs feasible under every matcher."""

    families0 = _validated_matcher_state_families(
        fold0_families, name="fold0_families"
    )
    families1 = _validated_matcher_state_families(
        fold1_families, name="fold1_families"
    )
    if set(families0) != set(families1):
        raise FrontierDecompositionError("fold matcher families differ")
    reference0 = next(iter(families0.values()))
    reference1 = next(iter(families1.values()))
    total_pixels = reference0[0].total_pixels + reference1[0].total_pixels
    pairs = []
    for index0 in range(len(reference0)):
        for index1 in range(len(reference1)):
            if all(
                pooled_budget_feasible(
                    families0[matcher][index0].unmatched_prediction_area
                    + families1[matcher][index1].unmatched_prediction_area,
                    total_pixels,
                    budget_fa_per_million_pixels,
                )
                for matcher in families0
            ):
                pairs.append((index0, index1))
    return tuple(pairs)


def select_joint_global_oracle_pair(
    fold0_families: Mapping[str, Sequence[FoldCandidateState]],
    fold1_families: Mapping[str, Sequence[FoldCandidateState]],
    *,
    budget_fa_per_million_pixels: int,
) -> JointGlobalOraclePair | None:
    """Select a pair maximizing targets matched under all audited matchers."""

    families0 = _validated_matcher_state_families(
        fold0_families, name="fold0_families"
    )
    families1 = _validated_matcher_state_families(
        fold1_families, name="fold1_families"
    )
    if set(families0) != set(families1):
        raise FrontierDecompositionError("fold matcher families differ")
    pairs = joint_feasible_pairs(
        families0,
        families1,
        budget_fa_per_million_pixels=budget_fa_per_million_pixels,
    )
    if not pairs:
        return None
    reference0 = next(iter(families0.values()))
    reference1 = next(iter(families1.values()))

    def payload(pair: tuple[int, int]):
        index0, index1 = pair
        per_matcher_ids = []
        per_matcher_areas = []
        for matcher in sorted(families0):
            per_matcher_ids.append(
                families0[matcher][index0].matched_target_ids
                | families1[matcher][index1].matched_target_ids
            )
            per_matcher_areas.append(
                families0[matcher][index0].unmatched_prediction_area
                + families1[matcher][index1].unmatched_prediction_area
            )
        joint_ids = frozenset.intersection(*per_matcher_ids)
        return joint_ids, tuple(per_matcher_areas)

    chosen = max(
        pairs,
        key=lambda pair: (
            len(payload(pair)[0]),
            -max(payload(pair)[1]),
            -sum(payload(pair)[1]),
            _threshold_preference(reference0[pair[0]]),
            _threshold_preference(reference1[pair[1]]),
        ),
    )
    joint_ids, areas = payload(chosen)
    return JointGlobalOraclePair(
        fold0_index=chosen[0],
        fold1_index=chosen[1],
        joint_matched_target_ids=joint_ids,
        matcher_unmatched_prediction_area=tuple(
            zip(sorted(families0), areas)
        ),
        total_pixels=reference0[0].total_pixels + reference1[0].total_pixels,
    )


def classify_joint_matcher_targets(
    fold_index: int,
    fold0_families: Mapping[str, Sequence[FoldCandidateState]],
    fold1_families: Mapping[str, Sequence[FoldCandidateState]],
    *,
    budget_fa_per_million_pixels: int,
    selected_by_matcher: Mapping[str, Mapping[str, bool]],
    joint_global_oracle: JointGlobalOraclePair | None,
) -> tuple[JointTargetFrontierStatus, ...]:
    """Classify a fixed Q-grid selected cohort under joint matcher evidence."""

    if fold_index not in (0, 1):
        raise FrontierDecompositionError("fold_index must be 0 or 1")
    families0 = _validated_matcher_state_families(
        fold0_families, name="fold0_families"
    )
    families1 = _validated_matcher_state_families(
        fold1_families, name="fold1_families"
    )
    if set(families0) != set(families1) or set(selected_by_matcher) != set(
        families0
    ):
        raise FrontierDecompositionError("matcher universes differ")
    target_universes = [set(values) for values in selected_by_matcher.values()]
    if any(universe != target_universes[0] for universe in target_universes[1:]):
        raise FrontierDecompositionError("selected target universes differ")
    if any(
        any(not isinstance(status, bool) for status in values.values())
        for values in selected_by_matcher.values()
    ):
        raise FrontierDecompositionError("selected statuses must be boolean")
    states_by_matcher = families0 if fold_index == 0 else families1
    reference = next(iter(states_by_matcher.values()))
    if len(target_universes[0]) != reference[0].target_components:
        raise FrontierDecompositionError("selected target count drifted")
    pairs = joint_feasible_pairs(
        families0,
        families1,
        budget_fa_per_million_pixels=budget_fa_per_million_pixels,
    )
    candidate_indices = {
        pair[fold_index] for pair in pairs
    }
    observed_target_ids = set().union(
        *(
            state.matched_target_ids
            | state.support_active_target_ids
            | state.core_active_target_ids
            for states in states_by_matcher.values()
            for state in states
        )
    )
    if not observed_target_ids.issubset(target_universes[0]):
        raise FrontierDecompositionError(
            "candidate target ids leave the selected fold universe"
        )
    globally_matched = (
        joint_global_oracle.joint_matched_target_ids
        if joint_global_oracle is not None
        else frozenset()
    )
    results = []
    sorted_matchers = sorted(states_by_matcher)
    legacy_name = next(
        (name for name in sorted_matchers if "legacy" in name),
        sorted_matchers[0],
    )
    hungarian_name = next(
        (name for name in sorted_matchers if "hungarian" in name),
        sorted_matchers[-1],
    )
    for stable_target_id in sorted(target_universes[0]):
        selected_statuses = {
            matcher: selected_by_matcher[matcher][stable_target_id]
            for matcher in sorted_matchers
        }
        selected_values = set(selected_statuses.values())
        exact = any(
            all(
                stable_target_id
                in states_by_matcher[matcher][candidate_index].matched_target_ids
                for matcher in sorted_matchers
            )
            for candidate_index in candidate_indices
        )
        support = any(
            stable_target_id
            in reference[candidate_index].support_active_target_ids
            for candidate_index in candidate_indices
        )
        core = any(
            stable_target_id in reference[candidate_index].core_active_target_ids
            for candidate_index in candidate_indices
        )
        if len(selected_values) != 1:
            category_support = CATEGORY_MATCHER_SENSITIVE
            category_core = CATEGORY_MATCHER_SENSITIVE
        elif next(iter(selected_values)):
            category_support = CATEGORY_SELECTED_HIT
            category_core = CATEGORY_SELECTED_HIT
        elif joint_global_oracle is None:
            category_support = CATEGORY_NO_FEASIBLE_PAIR
            category_core = CATEGORY_NO_FEASIBLE_PAIR
        elif exact:
            category_support = CATEGORY_SELECTION_SENSITIVE
            category_core = CATEGORY_SELECTION_SENSITIVE
        else:
            category_support = (
                CATEGORY_COMPONENT_CONVERSION if support else CATEGORY_PEAK_ORDER
            )
            category_core = (
                CATEGORY_COMPONENT_CONVERSION if core else CATEGORY_PEAK_ORDER
            )
        results.append(
            JointTargetFrontierStatus(
                stable_target_id=stable_target_id,
                category_support=category_support,
                category_core=category_core,
                selected_legacy_matched=selected_statuses[legacy_name],
                selected_hungarian_matched=selected_statuses[hungarian_name],
                joint_global_oracle_matched=stable_target_id in globally_matched,
                targetwise_joint_exact_match_exists=exact,
                targetwise_joint_support_active_exists=support,
                targetwise_joint_core_active_exists=core,
                joint_feasible_candidate_count=len(candidate_indices),
            )
        )
    return tuple(results)
