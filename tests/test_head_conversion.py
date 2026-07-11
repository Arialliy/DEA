import numpy as np
import pytest

from utils.feature_survival import (
    build_translation_control_set,
    evaluate_feature_survival,
    project_geometry_controls,
)
from utils.head_conversion import evaluate_linear_head_conversion


def _geometry():
    target = np.zeros((48, 48), dtype=bool)
    target[23, 25] = True
    controls = build_translation_control_set(
        target,
        target,
        sample_key="head-conversion",
        min_translation_radius=7,
        max_translation_radius=20,
        max_candidate_controls=160,
    )
    geometry = project_geometry_controls(
        controls,
        (48, 48),
        required_controls=32,
        background_radii=(5.0, 7.0, 10.0, 14.0, 20.0),
        minimum_background_cells=8,
        maximum_background_cells=48,
    )
    assert geometry is not None
    return target, geometry


def test_margin_factorization_is_exact_without_epsilon_denominator():
    target, geometry = _geometry()
    rng = np.random.default_rng(43)
    feature = rng.normal(size=(4, 48, 48))
    feature[:, target] += np.asarray([3.0, -2.0, 1.0, 4.0])[:, None]
    weight = np.asarray([0.4, -0.7, 1.2, 0.1])

    result = evaluate_linear_head_conversion(
        feature,
        geometry,
        head_weight=weight,
        head_bias=0.3,
        scalar_threshold=0.0,
    )

    assert result.available
    assert result.reason is None
    assert result.reconstructed_margin == pytest.approx(
        result.mean_logit_margin, abs=1e-10
    )
    assert result.reconstruction_error < 1e-10
    assert -1.0 <= result.utilization_cosine <= 1.0
    flat = feature.reshape(4, -1)
    occupancy = geometry.target.occupancy.reshape(-1)
    target_indices = np.flatnonzero(occupancy > 0)
    target_weights = occupancy[target_indices]
    target_weights = target_weights / target_weights.sum()
    logits = weight @ flat + 0.3
    expected_target = np.sum(logits[target_indices] * target_weights)
    expected_background = np.mean(
        logits[geometry.target.background_flat_indices]
    )
    assert result.target_mean_logit == pytest.approx(expected_target)
    assert result.background_mean_logit == pytest.approx(expected_background)
    assert result.mean_logit_margin == pytest.approx(
        expected_target - expected_background
    )


def test_head_reversal_changes_utilization_and_signed_margin_sign():
    target, geometry = _geometry()
    rng = np.random.default_rng(5)
    feature = rng.normal(scale=0.2, size=(2, 48, 48))
    feature[:, target] += np.asarray([5.0, 2.0])[:, None]
    weight = np.asarray([1.0, 0.5])

    forward = evaluate_linear_head_conversion(
        feature, geometry, head_weight=weight
    )
    reversed_head = evaluate_linear_head_conversion(
        feature, geometry, head_weight=-weight
    )

    assert reversed_head.mean_margin_availability == pytest.approx(
        forward.mean_margin_availability
    )
    assert reversed_head.head_sensitivity == pytest.approx(
        forward.head_sensitivity
    )
    assert reversed_head.utilization_cosine == pytest.approx(
        -forward.utilization_cosine
    )
    assert reversed_head.mean_logit_margin == pytest.approx(
        -forward.mean_logit_margin
    )


def test_unsigned_survival_cannot_distinguish_head_direction():
    target, geometry = _geometry()
    feature = np.zeros((1, 48, 48), dtype=np.float64)
    feature[0, target] = 5.0

    unsigned = evaluate_feature_survival(feature, geometry)
    positive = evaluate_linear_head_conversion(
        feature, geometry, head_weight=np.asarray([1.0])
    )
    negative = evaluate_linear_head_conversion(
        feature, geometry, head_weight=np.asarray([-1.0])
    )

    assert unsigned.state == "distinct"
    assert positive.utilization_cosine == 1.0
    assert negative.utilization_cosine == -1.0


def test_inverse_channel_reparameterization_preserves_all_factors():
    target, geometry = _geometry()
    rng = np.random.default_rng(17)
    feature = rng.normal(size=(3, 48, 48))
    feature[:, target] += np.asarray([2.0, -4.0, 3.0])[:, None]
    weight = np.asarray([0.5, -1.5, 0.8])
    scales = np.asarray([2.0, 0.25, 4.0])

    original = evaluate_linear_head_conversion(
        feature, geometry, head_weight=weight, head_bias=-0.7
    )
    transformed = evaluate_linear_head_conversion(
        feature * scales[:, None, None],
        geometry,
        head_weight=weight / scales,
        head_bias=-0.7,
    )

    for field in (
        "mean_margin_availability",
        "head_sensitivity",
        "utilization_cosine",
        "mean_logit_margin",
        "target_mean_logit",
        "target_peak_logit",
    ):
        assert getattr(transformed, field) == pytest.approx(
            getattr(original, field)
        )


def test_zero_head_is_explicit_and_factorization_remains_exact():
    target, geometry = _geometry()
    feature = np.zeros((2, 48, 48), dtype=np.float64)
    feature[:, target] = np.asarray([2.0, -3.0])[:, None]

    result = evaluate_linear_head_conversion(
        feature,
        geometry,
        head_weight=np.zeros(2),
        head_bias=4.0,
        scalar_threshold=0.0,
    )

    assert result.reason == "zero_head_sensitivity"
    assert result.head_sensitivity == 0.0
    assert result.utilization_cosine is None
    assert result.mean_logit_margin == 0.0
    assert result.reconstructed_margin is None
    assert result.reconstruction_error == 0.0
    assert result.target_peak_logit == 4.0
    assert result.absolute_peak_margin == 4.0


def test_tiny_nonzero_contrast_keeps_exact_factorization():
    target, geometry = _geometry()
    feature = np.zeros((1, 48, 48), dtype=np.float64)
    feature[0, target] = 1e-14

    result = evaluate_linear_head_conversion(
        feature,
        geometry,
        head_weight=np.asarray([1e8]),
    )

    assert result.reason is None
    assert result.mean_logit_margin == pytest.approx(1e-6)
    assert result.reconstructed_margin == pytest.approx(1e-6)


def test_missing_geometry_and_nonfinite_inputs_fail_closed():
    unavailable = evaluate_linear_head_conversion(
        np.zeros((2, 4, 4)),
        None,
        head_weight=np.ones(2),
    )
    assert not unavailable.available
    assert unavailable.reason == "insufficient_geometry_controls"

    _, geometry = _geometry()
    feature = np.zeros((2, 48, 48))
    feature[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        evaluate_linear_head_conversion(
            feature, geometry, head_weight=np.ones(2)
        )
