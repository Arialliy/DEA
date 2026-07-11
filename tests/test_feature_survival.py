import numpy as np
import pytest

from utils.feature_survival import (
    TranslationControlSet,
    build_translation_control_set,
    evaluate_feature_survival,
    fractional_project,
    project_geometry_controls,
    select_context_matched_controls,
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


def _spaced_context_controls():
    shape = (96, 96)
    target = np.zeros(shape, dtype=bool)
    target[40, 40] = True
    guarded = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    guarded[(yy - 40) ** 2 + (xx - 40) ** 2 <= 2**2] = True
    candidate_positions = [
        (row, column)
        for row in (16, 40, 64, 80)
        for column in (16, 40, 64, 80)
        if (row, column) != (40, 40)
    ]
    candidates = []
    for row, column in candidate_positions:
        mask = np.zeros(shape, dtype=bool)
        mask[row, column] = True
        candidates.append(mask)
    controls = TranslationControlSet(
        component_mask=target,
        all_target_mask=target,
        guarded_target_mask=guarded,
        translated_masks=tuple(candidates),
        sample_key="spaced-context-controls",
    )
    rows, columns = np.indices(shape)
    image = (
        0.012 * rows
        + 0.007 * columns
        + 0.3 * np.sin(rows / 7.0)
        + 0.2 * np.cos(columns / 9.0)
        + 0.1 * np.sin((rows + columns) / 5.0)
    )
    return image, controls


def _select_spaced_context(image, controls, **kwargs):
    return select_context_matched_controls(
        image,
        controls,
        context_inner_radius=2,
        context_ring_width=6,
        num_controls=6,
        minimum_ring_pixels=40,
        minimum_stencil_pixels=24,
        minimum_covariance_candidates=10,
        **kwargs,
    )


def test_context_matching_never_uses_target_or_candidate_footprint_values():
    image, controls = _spaced_context_controls()
    first = _select_spaced_context(image, controls)
    assert first.available

    changed = image.copy()
    changed[controls.component_mask] = 1e12
    for candidate in controls.translated_masks:
        changed[candidate] = -1e12
    second = _select_spaced_context(changed, controls)

    assert second.available
    assert second.target_descriptor == pytest.approx(first.target_descriptor)
    assert second.descriptor_center == pytest.approx(first.descriptor_center)
    assert second.descriptor_scale == pytest.approx(first.descriptor_scale)
    assert second.regularized_covariance == pytest.approx(
        first.regularized_covariance
    )
    assert [item.mask_digest for item in second.selected] == [
        item.mask_digest for item in first.selected
    ]
    assert [item.mahalanobis_distance for item in second.selected] == pytest.approx(
        [item.mahalanobis_distance for item in first.selected]
    )


def test_context_matching_never_uses_any_protected_pixel_intensity():
    image, controls = _spaced_context_controls()
    protection = controls.guarded_target_mask.copy()
    protection[8:11, 72:75] = True
    first = _select_spaced_context(
        image,
        controls,
        protection_mask=protection,
    )
    assert first.available

    changed = image.copy()
    changed[protection] = np.linspace(
        -1e12,
        1e12,
        int(protection.sum()),
    )
    second = _select_spaced_context(
        changed,
        controls,
        protection_mask=protection,
    )

    assert second.available
    assert second.target_descriptor == pytest.approx(first.target_descriptor)
    assert second.descriptor_center == pytest.approx(first.descriptor_center)
    assert second.descriptor_scale == pytest.approx(first.descriptor_scale)
    assert second.regularized_covariance == pytest.approx(
        first.regularized_covariance
    )
    assert [item.mask_digest for item in second.selected] == [
        item.mask_digest for item in first.selected
    ]
    assert [item.mahalanobis_distance for item in second.selected] == pytest.approx(
        [item.mahalanobis_distance for item in first.selected]
    )


def test_context_matching_is_order_invariant_and_exposes_deterministic_ties():
    image, controls = _spaced_context_controls()
    constant = np.zeros_like(image)
    first = _select_spaced_context(constant, controls)
    reversed_controls = TranslationControlSet(
        component_mask=controls.component_mask,
        all_target_mask=controls.all_target_mask,
        guarded_target_mask=controls.guarded_target_mask,
        translated_masks=tuple(reversed(controls.translated_masks)),
        sample_key=controls.sample_key,
    )
    second = _select_spaced_context(constant, reversed_controls)

    assert first.available and second.available
    assert first.covariance_condition_number == pytest.approx(1.0)
    assert all(item.mahalanobis_distance == 0 for item in first.selected)
    first_digests = [item.mask_digest for item in first.selected]
    second_digests = [item.mask_digest for item in second.selected]
    assert first_digests == sorted(first_digests)
    assert second_digests == first_digests


def test_context_matching_is_invariant_to_positive_affine_image_scaling():
    image, controls = _spaced_context_controls()
    first = _select_spaced_context(image, controls)
    transformed = _select_spaced_context(3.5 * image + 17.0, controls)

    assert first.available and transformed.available
    assert [item.mask_digest for item in transformed.selected] == [
        item.mask_digest for item in first.selected
    ]
    assert [
        item.mahalanobis_distance for item in transformed.selected
    ] == pytest.approx(
        [item.mahalanobis_distance for item in first.selected],
        rel=1e-9,
        abs=1e-10,
    )


def test_context_matching_enforces_protection_and_control_independence():
    image, controls = _spaced_context_controls()
    protection = controls.guarded_target_mask.copy()
    protection |= controls.translated_masks[0]
    result = _select_spaced_context(
        image,
        controls,
        protection_mask=protection,
    )

    assert result.available
    assert dict(result.rejected_candidate_counts)["protection_overlap"] == 1
    selected_masks = [item.component_mask for item in result.selected]
    assert all(not np.any(mask & protection) for mask in selected_masks)
    for index, mask in enumerate(selected_masks):
        for previous in selected_masks[:index]:
            assert not np.any(mask & previous)


def test_context_matching_rejects_same_area_nontranslation_controls():
    image, singleton_controls = _spaced_context_controls()
    component = np.zeros_like(singleton_controls.component_mask)
    component[40, 40:42] = True
    guarded = np.zeros_like(component)
    rows, columns = np.indices(component.shape)
    guarded[
        np.minimum(
            (rows - 40) ** 2 + (columns - 40) ** 2,
            (rows - 40) ** 2 + (columns - 41) ** 2,
        )
        <= 2**2
    ] = True

    translated = []
    for singleton in singleton_controls.translated_masks:
        row, column = np.argwhere(singleton)[0]
        candidate = np.zeros_like(component)
        candidate[row, column : column + 2] = True
        translated.append(candidate)
    deformation = np.zeros_like(component)
    deformation[8:10, 8] = True
    controls = TranslationControlSet(
        component_mask=component,
        all_target_mask=component,
        guarded_target_mask=guarded,
        translated_masks=(deformation, *translated),
        sample_key="same-area-deformation",
    )

    result = select_context_matched_controls(
        image,
        controls,
        context_inner_radius=2,
        context_ring_width=6,
        num_controls=6,
        minimum_ring_pixels=40,
        minimum_stencil_pixels=24,
        minimum_covariance_candidates=10,
        maximum_selected_iou=0.99,
    )

    assert result.available
    assert dict(result.rejected_candidate_counts)["geometry_mismatch"] == 1
    assert all(
        np.array_equal(
            np.argwhere(item.component_mask) - np.argwhere(component),
            np.repeat(
                (np.argwhere(item.component_mask)[0] - np.argwhere(component)[0])[
                    None, :
                ],
                int(component.sum()),
                axis=0,
            ),
        )
        for item in result.selected
    )


def test_context_matching_fails_closed_without_an_exterior_ring():
    image, controls = _spaced_context_controls()
    result = _select_spaced_context(
        image,
        controls,
        protection_mask=np.ones_like(controls.component_mask),
    )

    assert not result.available
    assert result.reason == "insufficient_target_exterior_context"
    assert result.control_set is None


def test_context_matching_fails_closed_when_target_context_has_no_support():
    image, controls = _spaced_context_controls()
    rows, columns = np.indices(image.shape)
    distance = np.sqrt((rows - 40) ** 2 + (columns - 40) ** 2)
    unmatched = image.copy()
    ring = (distance > 2) & (distance <= 8)
    unmatched[ring] += 100.0 * ((rows[ring] + columns[ring]) % 2)

    result = _select_spaced_context(unmatched, controls)

    assert not result.available
    assert result.reason == "target_context_out_of_candidate_support"
    assert result.context_distance_caliper is not None
    assert dict(result.rejected_candidate_counts)["context_distance_caliper"] > 0


def test_context_matching_rejects_a_protection_mask_that_omits_targets():
    image, controls = _spaced_context_controls()

    with pytest.raises(ValueError, match="include every target"):
        _select_spaced_context(
            image,
            controls,
            protection_mask=np.zeros_like(controls.component_mask),
        )
