from __future__ import annotations

import numpy as np
import pytest

from utils.no_response_scale import analyze_no_response_scales


def _synthetic_evidence():
    target = np.zeros((5, 5), dtype=np.uint8)
    target[2, 2] = 1
    final = np.full((5, 5), -1.0)
    sides = np.full((4, 5, 5), -1.0)
    sides[0, 2, 2] = 1.0
    contributions = np.zeros((4, 5, 5), dtype=np.float64)
    contributions[0, 2, 2] = 2.0
    contributions[1, 2, 2] = -2.0
    bias = -1.0
    return final, sides, contributions, bias, target


def test_final_no_response_can_be_side_and_subset_recoverable() -> None:
    final, sides, contributions, bias, target = _synthetic_evidence()
    result = analyze_no_response_scales(
        final,
        sides,
        contributions,
        bias,
        target,
    )

    assert result.num_gt == 1
    assert result.final_matches == 0
    assert result.final_no_response_target_indices == (0,)
    record = result.records[0]
    assert record.side_support_scales == (0,)
    assert record.side_centroid_legal_scales == (0,)
    assert record.side_matched_scales == (0,)
    assert 1 in record.recovering_subsets
    assert all(subset < 15 for subset in record.recovering_subsets)


def test_all_scales_absent_is_explicit() -> None:
    target = np.zeros((5, 5), dtype=np.uint8)
    target[2, 2] = 1
    final = np.full((5, 5), -1.0)
    sides = np.full((4, 5, 5), -1.0)
    contributions = np.zeros((4, 5, 5))
    result = analyze_no_response_scales(
        final,
        sides,
        contributions,
        -1.0,
        target,
    )

    record = result.records[0]
    assert record.side_support_scales == ()
    assert record.side_centroid_legal_scales == ()
    assert record.side_matched_scales == ()
    assert record.recovering_subsets == ()


def test_final_match_is_not_a_no_response_record() -> None:
    target = np.zeros((5, 5), dtype=np.uint8)
    target[2, 2] = 1
    final = np.full((5, 5), -1.0)
    final[2, 2] = 1.0
    sides = np.repeat(final[None], 4, axis=0)
    contributions = np.zeros((4, 5, 5))
    contributions[0] = final + 1.0
    result = analyze_no_response_scales(
        final,
        sides,
        contributions,
        -1.0,
        target,
    )
    assert result.final_matches == 1
    assert result.records == ()


def test_no_response_scale_audit_rejects_inconsistent_evidence() -> None:
    final, sides, contributions, bias, target = _synthetic_evidence()
    contributions[3, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="do not reconstruct"):
        analyze_no_response_scales(
            final,
            sides,
            contributions,
            bias,
            target,
        )
    with pytest.raises(ValueError, match="shape"):
        analyze_no_response_scales(
            final,
            sides[:3],
            contributions[:3],
            bias,
            target,
        )
