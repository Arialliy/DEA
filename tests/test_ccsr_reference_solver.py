from __future__ import annotations

import numpy as np
import pytest
from skimage import measure

from model.ccsr.pixel_edit_reference import PixelEditConfig
from model.ccsr.reference_solver import solve_exhaustive_structured_hinge


def _instance_labels(target: np.ndarray) -> np.ndarray:
    return measure.label(target.astype(bool), connectivity=2).astype(np.int64)


def _config(**kwargs) -> PixelEditConfig:
    values = {
        "activation_semantics": "margin",
        "activation_margin": 0.125,
        "max_pixels": 9,
    }
    values.update(kwargs)
    return PixelEditConfig(**values)


def test_corrected_hinge_upper_bounds_excess_risk_on_random_small_grids() -> None:
    generator = np.random.default_rng(20260712)
    minimum_slack = float("inf")
    for case_index in range(200):
        logits = generator.normal(size=(2, 2))
        if case_index % 7 == 0:
            logits.reshape(-1)[case_index % 4] = 0.0
        target = generator.integers(0, 2, size=(2, 2), dtype=np.uint8)
        result = solve_exhaustive_structured_hinge(
            logits,
            target,
            _instance_labels(target),
            pixel_config=_config(),
        )

        assert np.array_equal(result.decoded.pixel_edit.mask(), logits > 0.0)
        assert result.hinge >= -1e-12
        assert result.hinge + 1e-12 >= result.excess_risk
        assert result.upper_bound_slack >= -1e-12
        assert result.num_states == 16
        minimum_slack = min(minimum_slack, result.upper_bound_slack)
    assert minimum_slack == pytest.approx(0.0, abs=1e-12)


def test_perfect_high_margin_prediction_has_zero_corrected_hinge() -> None:
    logits = np.array([[10.0]])
    target = np.array([[1]], dtype=np.uint8)
    result = solve_exhaustive_structured_hinge(
        logits,
        target,
        _instance_labels(target),
        pixel_config=_config(max_pixels=1),
    )

    assert result.decoded.inner_risk.risk == 0.0
    assert result.oracle_risk == 0.0
    assert result.hinge == 0.0
    assert result.excess_risk == 0.0

    # The invalid old max_{F,M} formulation admits an empty matching for this
    # exact same frontier: one miss plus one full-image clutter component.
    old_adversarial_matching_risk = 1.0 + 1.0
    assert old_adversarial_matching_risk > result.hinge
    # Frontier/action scores are identical, so that old positive term has no
    # score-gradient signal. The corrected solver never enumerates M outside
    # the inner minimum certificate.
    assert result.decoded.score == result.oracle.score == 0.0


def test_decoding_is_target_independent() -> None:
    logits = np.array([[0.4, -0.3], [-0.2, 0.7]])
    empty = np.zeros((2, 2), dtype=np.uint8)
    full = np.ones((2, 2), dtype=np.uint8)
    empty_result = solve_exhaustive_structured_hinge(
        logits,
        empty,
        _instance_labels(empty),
        pixel_config=_config(),
    )
    full_result = solve_exhaustive_structured_hinge(
        logits,
        full,
        _instance_labels(full),
        pixel_config=_config(),
    )

    assert empty_result.decoded.pixel_edit.mask_bits == (
        full_result.decoded.pixel_edit.mask_bits
    )
    assert np.array_equal(empty_result.decoded.pixel_edit.mask(), logits > 0)


def test_low_margin_perfect_mask_may_have_positive_hinge() -> None:
    logits = np.array([[0.01]])
    target = np.array([[1]], dtype=np.uint8)
    result = solve_exhaustive_structured_hinge(
        logits,
        target,
        _instance_labels(target),
        pixel_config=_config(max_pixels=1),
    )

    assert result.decoded.inner_risk.risk == 0.0
    assert result.hinge > 0.0
    assert result.excess_risk == 0.0


def test_structured_reference_handles_empty_target() -> None:
    logits = np.array([[1.0, -1.0], [-1.0, -1.0]])
    target = np.zeros((2, 2), dtype=np.uint8)
    result = solve_exhaustive_structured_hinge(
        logits,
        target,
        _instance_labels(target),
        pixel_config=_config(),
    )

    assert result.decoded.inner_risk.clutter_risk == 0.25
    assert result.oracle_risk == 0.0
    assert result.hinge >= result.excess_risk


def test_structured_reference_fails_closed_on_unrealizable_or_large_space() -> None:
    target = np.zeros((2, 2), dtype=np.uint8)
    labels = _instance_labels(target)
    with pytest.raises(ValueError, match="realizable margin"):
        solve_exhaustive_structured_hinge(
            np.zeros((2, 2)),
            target,
            labels,
            pixel_config=PixelEditConfig(max_pixels=4),
        )
    with pytest.raises(ValueError, match="would enumerate"):
        solve_exhaustive_structured_hinge(
            np.zeros((3, 3)),
            np.zeros((3, 3), dtype=np.uint8),
            np.zeros((3, 3), dtype=np.int64),
            pixel_config=_config(max_pixels=9),
            max_structured_states=128,
        )
