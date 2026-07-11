import pytest

from tools.summarize_ccsr_component_ledgers import finalize_scale


def _scale_counts():
    return {
        "no_response_with_any_side_support": 6,
        "no_response_with_any_side_centroid": 3,
        "no_response_matched_by_any_side": 3,
        "no_response_recoverable_by_global_subset": 15,
        "no_response_absent_from_all_sides": 65,
    }


def _intersections():
    return {
        "no_response_side_and_subset": 3,
        "no_response_side_only": 3,
        "no_response_subset_only": 12,
        "no_response_neither_side_nor_subset": 53,
    }


def test_finalize_scale_preserves_overlapping_diagnostic_taxonomies():
    result = finalize_scale(
        _scale_counts(),
        {"2": 4, "3": 5},
        {"1": 2, "4": 5, "13": 3},
        _intersections(),
        no_response_gt=71,
    )

    assert result["side_support_counts"] == {
        "0": 0,
        "1": 0,
        "2": 4,
        "3": 5,
    }
    assert result["recovering_subset_counts"]["4"] == 5
    assert result["recovering_subset_counts"]["14"] == 0
    assert result["absent_from_all_sides_rate"] == pytest.approx(65 / 71)
    assert result["neither_side_nor_subset_rate"] == pytest.approx(53 / 71)


def test_finalize_scale_rejects_nonpartitioning_intersections():
    intersections = _intersections()
    intersections["no_response_neither_side_nor_subset"] = 52

    with pytest.raises(RuntimeError, match="do not partition"):
        finalize_scale(
            _scale_counts(),
            {},
            {},
            intersections,
            no_response_gt=71,
        )


def test_finalize_scale_rejects_incomplete_side_support_partition():
    counts = _scale_counts()
    counts["no_response_absent_from_all_sides"] = 64

    with pytest.raises(RuntimeError, match="does not partition"):
        finalize_scale(
            counts,
            {},
            {},
            _intersections(),
            no_response_gt=71,
        )
