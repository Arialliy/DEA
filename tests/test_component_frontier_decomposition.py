from __future__ import annotations

import pytest

from utils.component_frontier_decomposition import (
    CATEGORY_COMPONENT_CONVERSION,
    CATEGORY_NO_FEASIBLE_PAIR,
    CATEGORY_PEAK_ORDER,
    CATEGORY_SELECTED_HIT,
    CATEGORY_SELECTION_SENSITIVE,
    FoldCandidateState,
    FrontierDecompositionError,
    classify_fold_targets,
    classify_joint_matcher_targets,
    global_oracle_matched_target_ids,
    joint_feasible_pairs,
    make_all_off_sentinel,
    pooled_budget_feasible,
    select_global_oracle_pair,
    select_joint_global_oracle_pair,
)


def _state(
    threshold: float,
    *,
    area: int,
    matched: tuple[str, ...] = (),
    support: tuple[str, ...] = (),
    core: tuple[str, ...] = (),
    pixels: int = 100_000,
    targets: int = 4,
) -> FoldCandidateState:
    return FoldCandidateState(
        threshold=threshold,
        all_off_sentinel=False,
        total_pixels=pixels,
        target_components=targets,
        matched_components=len(matched),
        prediction_components=len(matched) + int(area > 0),
        unmatched_prediction_components=int(area > 0),
        unmatched_prediction_area=area,
        matched_target_ids=frozenset(matched),
        support_active_target_ids=frozenset(support),
        core_active_target_ids=frozenset(core),
    )


def test_target_categories_are_mutually_exclusive_and_optimistic() -> None:
    states = (
        _state(
            3.0,
            area=0,
            matched=("selected",),
            support=("selected", "convert", "near_only"),
            core=("selected", "convert"),
            targets=5,
        ),
        _state(
            2.0,
            area=2,
            matched=("recoverable",),
            support=("selected", "recoverable", "convert", "near_only"),
            core=("selected", "recoverable", "convert"),
            targets=5,
        ),
    )
    other = make_all_off_sentinel(total_pixels=100_000, target_components=1)
    rows = classify_fold_targets(
        states,
        other_fold_all_off=other,
        budget_fa_per_million_pixels=10,
        selected_matched={
            "selected": True,
            "recoverable": False,
            "convert": False,
            "near_only": False,
            "order": False,
        },
        globally_oracle_matched_target_ids=("selected", "recoverable"),
    )
    by_id = {row.stable_target_id: row for row in rows}

    assert by_id["selected"].category_support == CATEGORY_SELECTED_HIT
    assert by_id["recoverable"].category_support == CATEGORY_SELECTION_SENSITIVE
    assert by_id["convert"].category_support == CATEGORY_COMPONENT_CONVERSION
    assert by_id["convert"].category_core == CATEGORY_COMPONENT_CONVERSION
    assert by_id["near_only"].category_support == CATEGORY_COMPONENT_CONVERSION
    assert by_id["near_only"].category_core == CATEGORY_PEAK_ORDER
    assert by_id["order"].category_support == CATEGORY_PEAK_ORDER
    assert by_id["recoverable"].minimum_exact_match_unmatched_area == 2


def test_targetwise_recoveries_are_not_the_global_pair_pd() -> None:
    fold0 = (
        _state(
            3.0,
            area=0,
            matched=("a",),
            support=("a",),
            core=("a",),
            targets=2,
        ),
        _state(
            2.0,
            area=2,
            matched=("b",),
            support=("b",),
            core=("b",),
            targets=2,
        ),
        make_all_off_sentinel(total_pixels=100_000, target_components=2),
    )
    fold1 = (
        _state(
            3.0,
            area=0,
            matched=("c",),
            support=("c",),
            core=("c",),
            targets=1,
        ),
        make_all_off_sentinel(total_pixels=100_000, target_components=1),
    )
    oracle = select_global_oracle_pair(
        fold0,
        fold1,
        budget_fa_per_million_pixels=10,
    )
    matched = global_oracle_matched_target_ids(fold0, fold1, oracle)

    assert oracle.matched_components == 2
    assert matched in (frozenset(("a", "c")), frozenset(("b", "c")))
    assert {"a", "b"} != set(matched) & {"a", "b"}


def test_global_pair_uses_exact_pooled_integer_budget() -> None:
    fold0 = (
        _state(
            1.0,
            area=2,
            matched=("a",),
            support=("a",),
            core=("a",),
            targets=1,
        ),
        make_all_off_sentinel(total_pixels=100_000, target_components=1),
    )
    fold1 = (
        _state(
            1.0,
            area=1,
            matched=("b",),
            support=("b",),
            core=("b",),
            targets=1,
        ),
        make_all_off_sentinel(total_pixels=100_000, target_components=1),
    )
    oracle = select_global_oracle_pair(
        fold0,
        fold1,
        budget_fa_per_million_pixels=10,
    )
    assert oracle.unmatched_prediction_area <= 2
    assert oracle.matched_components == 1
    assert pooled_budget_feasible(2, 200_000, 10)
    assert not pooled_budget_feasible(3, 200_000, 10)


def test_invalid_state_and_target_universe_fail_closed() -> None:
    invalid = _state(
        1.0,
        area=0,
        support=(),
        core=("core_without_support",),
        targets=1,
    )
    other = make_all_off_sentinel(total_pixels=100_000, target_components=1)
    with pytest.raises(FrontierDecompositionError, match="core-active"):
        classify_fold_targets(
            (invalid,),
            other_fold_all_off=other,
            budget_fa_per_million_pixels=10,
            selected_matched={"core_without_support": False},
            globally_oracle_matched_target_ids=(),
        )

    valid = _state(1.0, area=0, matched=("outside",), targets=1)
    with pytest.raises(FrontierDecompositionError, match="leave the universe"):
        classify_fold_targets(
            (valid,),
            other_fold_all_off=other,
            budget_fa_per_million_pixels=10,
            selected_matched={"inside": False},
            globally_oracle_matched_target_ids=(),
        )


def test_joint_matcher_classification_requires_joint_feasibility_and_match() -> None:
    fold0_legacy = (
        _state(
            3.0,
            area=0,
            matched=("hit",),
            support=("hit", "convert"),
            core=("hit", "convert"),
            targets=4,
        ),
        _state(
            2.0,
            area=2,
            matched=("recover",),
            support=("hit", "recover", "convert"),
            core=("hit", "recover", "convert"),
            targets=4,
        ),
    )
    fold0_hungarian = (
        fold0_legacy[0],
        _state(
            2.0,
            area=1,
            matched=("recover",),
            support=("hit", "recover", "convert"),
            core=("hit", "recover", "convert"),
            targets=4,
        ),
    )
    fold1 = (
        _state(
            4.0,
            area=0,
            matched=("other",),
            support=("other",),
            core=("other",),
            targets=1,
        ),
    )
    families0 = {
        "official_legacy": fold0_legacy,
        "audit_hungarian": fold0_hungarian,
    }
    families1 = {
        "official_legacy": fold1,
        "audit_hungarian": fold1,
    }
    pairs = joint_feasible_pairs(
        families0,
        families1,
        budget_fa_per_million_pixels=10,
    )
    oracle = select_joint_global_oracle_pair(
        families0,
        families1,
        budget_fa_per_million_pixels=10,
    )
    assert pairs == ((0, 0), (1, 0))
    assert oracle is not None
    rows = classify_joint_matcher_targets(
        0,
        families0,
        families1,
        budget_fa_per_million_pixels=10,
        selected_by_matcher={
            "official_legacy": {
                "hit": True,
                "recover": False,
                "convert": False,
                "order": False,
            },
            "audit_hungarian": {
                "hit": True,
                "recover": False,
                "convert": False,
                "order": False,
            },
        },
        joint_global_oracle=oracle,
    )
    by_id = {row.stable_target_id: row for row in rows}
    assert by_id["hit"].category_support == CATEGORY_SELECTED_HIT
    assert by_id["recover"].category_support == CATEGORY_SELECTION_SENSITIVE
    assert by_id["convert"].category_support == CATEGORY_COMPONENT_CONVERSION
    assert by_id["order"].category_support == CATEGORY_PEAK_ORDER


def test_joint_no_feasible_pair_is_not_mislabeled_as_peak_order() -> None:
    state = _state(
        1.0,
        area=10,
        support=("target",),
        core=("target",),
        targets=1,
    )
    families = {
        "official_legacy": (state,),
        "audit_hungarian": (state,),
    }
    oracle = select_joint_global_oracle_pair(
        families,
        families,
        budget_fa_per_million_pixels=1,
    )
    assert oracle is None
    rows = classify_joint_matcher_targets(
        0,
        families,
        families,
        budget_fa_per_million_pixels=1,
        selected_by_matcher={
            "official_legacy": {"target": False},
            "audit_hungarian": {"target": False},
        },
        joint_global_oracle=oracle,
    )
    assert rows[0].category_support == CATEGORY_NO_FEASIBLE_PAIR


def test_feasible_zero_joint_hit_is_classified_by_activation() -> None:
    active = _state(
        2.0,
        area=0,
        support=("active",),
        core=("active",),
        targets=2,
    )
    inactive = _state(3.0, area=0, targets=1)
    fold0 = {
        "official_legacy": (active,),
        "audit_hungarian": (active,),
    }
    fold1 = {
        "official_legacy": (inactive,),
        "audit_hungarian": (inactive,),
    }
    oracle = select_joint_global_oracle_pair(
        fold0,
        fold1,
        budget_fa_per_million_pixels=1,
    )
    assert oracle is not None
    assert not oracle.joint_matched_target_ids
    rows = classify_joint_matcher_targets(
        0,
        fold0,
        fold1,
        budget_fa_per_million_pixels=1,
        selected_by_matcher={
            "official_legacy": {"active": False, "order": False},
            "audit_hungarian": {"active": False, "order": False},
        },
        joint_global_oracle=oracle,
    )
    by_id = {row.stable_target_id: row for row in rows}
    assert by_id["active"].category_core == CATEGORY_COMPONENT_CONVERSION
    assert by_id["order"].category_core == CATEGORY_PEAK_ORDER
