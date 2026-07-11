import numpy as np
import pytest

from utils.feature_survival import (
    build_translation_control_set,
    evaluate_feature_survival,
    fractional_project,
    project_geometry_controls,
)


def _singleton_geometry(*, required_controls=64):
    target = np.zeros((64, 64), dtype=bool)
    target[31, 33] = True
    controls = build_translation_control_set(
        target,
        target,
        sample_key="synthetic-singleton",
        min_translation_radius=8,
        max_translation_radius=24,
        max_candidate_controls=192,
    )
    geometry = project_geometry_controls(
        controls,
        (64, 64),
        required_controls=required_controls,
        background_radii=(6.0, 8.0, 12.0, 16.0, 24.0),
        minimum_background_cells=12,
        maximum_background_cells=64,
    )
    return target, controls, geometry


def test_fractional_projection_keeps_a_single_pixel_at_one_sixteenth_scale():
    mask = np.zeros((32, 32), dtype=bool)
    mask[7, 19] = True

    projected = fractional_project(mask, (2, 2))

    assert np.count_nonzero(projected) == 1
    assert projected.max() == pytest.approx(1 / 256)
    assert projected.sum() == pytest.approx(1 / 256)


def test_geometry_control_rank_distinguishes_signal_from_constant_background():
    target, _, geometry = _singleton_geometry()
    assert geometry is not None
    rng = np.random.default_rng(7)
    feature = rng.normal(scale=0.15, size=(3, 64, 64))
    feature[:, target] += np.asarray([8.0, -6.0, 5.0])[:, None]

    signal = evaluate_feature_survival(feature, geometry)
    constant = evaluate_feature_survival(np.zeros_like(feature), geometry)

    assert signal.available
    assert signal.state == "distinct"
    assert signal.rank == 1.0
    assert signal.robust_effect > 0
    assert constant.state == "background_like"
    assert constant.rank == pytest.approx(1 / 65)


def test_scalar_companion_diagnostic_separates_reverse_from_positive_evidence():
    target, _, geometry = _singleton_geometry()
    assert geometry is not None
    negative = np.zeros((1, 64, 64), dtype=np.float64)
    negative[0, target] = -4.0

    result = evaluate_feature_survival(
        negative,
        geometry,
        scalar_threshold=0.0,
    )

    assert result.state == "distinct"
    assert result.directional_auc == 0.0
    assert result.target_peak_margin == -4.0


def test_rank_is_invariant_to_positive_per_channel_affine_changes():
    target, _, geometry = _singleton_geometry()
    assert geometry is not None
    rng = np.random.default_rng(19)
    feature = rng.normal(size=(3, 64, 64))
    feature[:, target] += np.asarray([4.0, 2.0, -3.0])[:, None]
    transformed = (
        feature * np.asarray([2.0, 0.5, 5.0])[:, None, None]
        + np.asarray([7.0, -4.0, 1.5])[:, None, None]
    )

    first = evaluate_feature_survival(feature, geometry)
    second = evaluate_feature_survival(transformed, geometry)

    assert second.rank == first.rank
    assert second.observed_score == pytest.approx(first.observed_score)
    assert second.robust_effect == pytest.approx(first.robust_effect)


def test_insufficient_controls_are_explicitly_undefined():
    target, controls, _ = _singleton_geometry()
    truncated = type(controls)(
        component_mask=controls.component_mask,
        all_target_mask=controls.all_target_mask,
        guarded_target_mask=controls.guarded_target_mask,
        translated_masks=controls.translated_masks[:2],
        sample_key=controls.sample_key,
    )
    geometry = project_geometry_controls(
        truncated,
        (64, 64),
        required_controls=8,
        background_radii=(6.0, 8.0, 12.0),
        minimum_background_cells=8,
    )

    result = evaluate_feature_survival(
        np.zeros((2, 64, 64)), geometry
    )

    assert geometry is None
    assert not result.available
    assert result.state == "undefined"
    assert result.reason == "insufficient_geometry_controls"
    assert target.sum() == 1


def test_nonfinite_features_fail_closed():
    _, _, geometry = _singleton_geometry()
    feature = np.zeros((2, 64, 64))
    feature[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        evaluate_feature_survival(feature, geometry)
