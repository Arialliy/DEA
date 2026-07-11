from __future__ import annotations

import numpy as np
import pytest

from model.ccsr.pixel_edit_reference import (
    PixelEditConfig,
    build_pixel_edit_state,
    enumerate_pixel_edit_states,
    mask_to_bits,
    reconstruct_edited_logits,
)


def test_raw_threshold_mask_is_the_unique_zero_edit_without_ties() -> None:
    logits = np.array([[-2.0, 0.4], [1.2, -0.7]])
    states = enumerate_pixel_edit_states(logits)
    zero_states = [state for state in states if state.edit_energy == 0.0]

    assert len(states) == 16
    assert len(zero_states) == 1
    assert np.array_equal(zero_states[0].mask(), logits > 0.0)
    assert zero_states[0].num_actions == 0
    assert zero_states[0].infimum_attained


def test_no_pixel_outside_the_action_set_can_change_for_free() -> None:
    logits = np.array([[2.0, 1.0], [-2.0, -1.0]])
    raw = logits > 0
    for index in range(logits.size):
        desired = raw.copy().reshape(-1)
        desired[index] = ~desired[index]
        state = build_pixel_edit_state(logits, desired.reshape(logits.shape))
        assert state.num_actions == 1
        assert state.edit_energy > 0.0


def test_suppressing_component_charges_every_active_pixel() -> None:
    logits = np.array([[1.5, 0.8]])
    empty = np.zeros_like(logits, dtype=bool)
    state = build_pixel_edit_state(logits, empty)

    assert state.deactivation_indices == (0, 1)
    assert state.edit_energy == pytest.approx(2.3)
    assert state.num_components == 0
    reconstructed = reconstruct_edited_logits(logits, state)
    assert np.array_equal(reconstructed > 0.0, empty)


def test_two_path_plateau_requires_a_multi_pixel_vertex_cut() -> None:
    # A 2x3 active rectangle connects its left and right columns through two
    # parallel rows. Under 8-connectivity, deleting either middle pixel alone
    # still leaves a path; both middle pixels must be lowered.
    logits = np.ones((2, 3), dtype=np.float64)
    left_and_right = np.array(
        [[1, 0, 1], [1, 0, 1]],
        dtype=bool,
    )
    split = build_pixel_edit_state(logits, left_and_right)

    assert split.num_components == 2
    assert split.deactivation_indices == (1, 4)
    assert split.edit_energy == 2.0

    for middle_index in (1, 4):
        one_cut = np.ones((2, 3), dtype=bool).reshape(-1)
        one_cut[middle_index] = False
        one_cut_state = build_pixel_edit_state(
            logits,
            one_cut.reshape(2, 3),
        )
        assert one_cut_state.num_components == 1


def test_strict_activation_is_an_unattained_infimum() -> None:
    logits = np.array([[-0.25]])
    active = np.ones((1, 1), dtype=bool)
    infimum = build_pixel_edit_state(logits, active)

    assert infimum.edit_energy == 0.25
    assert not infimum.infimum_attained
    with pytest.raises(RuntimeError, match="infimum is not attained"):
        reconstruct_edited_logits(logits, infimum)

    margin_config = PixelEditConfig(
        activation_semantics="margin",
        activation_margin=0.1,
    )
    margin = build_pixel_edit_state(logits, active, config=margin_config)
    assert margin.infimum_attained
    assert margin.edit_energy == pytest.approx(0.35)
    reconstructed = reconstruct_edited_logits(
        logits,
        margin,
        config=margin_config,
    )
    assert reconstructed[0, 0] == pytest.approx(0.1)
    assert bool(reconstructed[0, 0] > 0.0)


def test_margin_target_must_be_finite_and_representably_above_threshold() -> None:
    with pytest.raises(ValueError, match="strictly above"):
        PixelEditConfig(
            threshold_logit=1.0,
            activation_semantics="margin",
            activation_margin=1e-20,
        ).validate()
    with pytest.raises(ValueError, match="strictly above"):
        PixelEditConfig(
            threshold_logit=1e308,
            activation_semantics="margin",
            activation_margin=1e308,
        ).validate()


def test_state_is_bound_to_its_config_and_source_logits() -> None:
    logits = np.array([[-0.5]])
    active = np.ones((1, 1), dtype=bool)
    config = PixelEditConfig(
        threshold_logit=0.0,
        activation_semantics="margin",
        activation_margin=0.1,
    )
    state = build_pixel_edit_state(logits, active, config=config)

    with pytest.raises(ValueError, match="config must equal"):
        reconstruct_edited_logits(
            logits,
            state,
            config=PixelEditConfig(
                activation_semantics="margin",
                activation_margin=0.2,
            ),
        )
    with pytest.raises(ValueError, match="different source logits"):
        reconstruct_edited_logits(np.array([[-0.4]]), state)


def test_raising_one_peak_realizes_only_the_requested_pixel_support() -> None:
    logits = np.full((3, 3), -1.0)
    desired = np.zeros((3, 3), dtype=bool)
    desired[1, 1] = True
    config = PixelEditConfig(
        activation_semantics="margin",
        activation_margin=0.2,
    )
    state = build_pixel_edit_state(logits, desired, config=config)
    reconstructed = reconstruct_edited_logits(logits, state, config=config)

    assert state.activation_indices == (4,)
    assert state.num_components == 1
    assert np.count_nonzero(reconstructed > 0.0) == 1


def test_mask_bit_round_trip() -> None:
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    state = build_pixel_edit_state(
        np.where(mask, 1.0, -1.0),
        mask,
    )
    assert state.mask_bits == mask_to_bits(mask)
    assert np.array_equal(state.mask(), mask)


@pytest.mark.parametrize(
    ("logits", "config", "message"),
    [
        (np.zeros((1, 1, 1)), PixelEditConfig(), "non-empty 2-D"),
        (np.array([[np.nan]]), PixelEditConfig(), "finite"),
        (np.zeros((5, 5)), PixelEditConfig(max_pixels=16), "at most"),
        (np.zeros((1, 1)), PixelEditConfig(connectivity=1), "8-connectivity"),
        (
            np.zeros((1, 1)),
            PixelEditConfig(activation_semantics="margin", activation_margin=0.0),
            "finite and positive",
        ),
    ],
)
def test_pixel_edit_reference_fails_closed(logits, config, message) -> None:
    with pytest.raises(ValueError, match=message):
        enumerate_pixel_edit_states(logits, config=config)
