import pytest
import torch

from utils.phase_intervention import (
    UNIT_PHASE_OFFSETS,
    PhaseInterventionError,
    aggregate_aligned_scores,
    align_translated_scores,
    phase_preserving_offsets,
    residue_shifted_offsets,
    translate_reflect,
)


def test_reflection_translation_is_exact_on_integer_lattice() -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    shifted = translate_reflect(value, (1, 1))
    expected = torch.tensor(
        [[[[5.0, 4.0, 5.0, 6.0], [1.0, 0.0, 1.0, 2.0], [5.0, 4.0, 5.0, 6.0]]]]
    )
    torch.testing.assert_close(shifted, expected, rtol=0.0, atol=0.0)


def test_alignment_recovers_valid_identity_model_region() -> None:
    value = torch.arange(30, dtype=torch.float32).reshape(1, 1, 5, 6)
    shifted = translate_reflect(value, (1, 1))
    aligned, valid = align_translated_scores(shifted, (1, 1))
    assert valid.shape == (1, 1, 5, 6)
    assert bool(valid[..., :4, :5].all())
    assert not bool(valid[..., 4:, :].any())
    assert not bool(valid[..., :, 5:].any())
    torch.testing.assert_close(aligned[..., :4, :5], value[..., :4, :5], rtol=0.0, atol=0.0)
    assert bool(torch.isneginf(aligned[..., 4:, :]).all())
    assert bool(torch.isneginf(aligned[..., :, 5:]).all())


def test_max_aggregate_keeps_zero_view_and_records_validity() -> None:
    baseline = torch.zeros(2, 1, 5, 6)
    views = [translate_reflect(baseline, offset) for offset in UNIT_PHASE_OFFSETS]
    views[-1] = views[-1] + 2.0
    aggregate, stack, validity = aggregate_aligned_scores(views, UNIT_PHASE_OFFSETS)
    assert stack.shape == (4, 2, 1, 5, 6)
    assert validity.shape == (4, 1, 1, 5, 6)
    assert bool(torch.isfinite(aggregate).all())
    torch.testing.assert_close(aggregate[..., :4, :5], torch.full((2, 1, 4, 5), 2.0))
    torch.testing.assert_close(aggregate[..., 4, :], torch.zeros(2, 1, 6))
    torch.testing.assert_close(aggregate[..., :, 5], torch.zeros(2, 1, 5))


def test_phase_preserving_offsets_and_fail_closed_validation() -> None:
    assert phase_preserving_offsets(16) == ((0, 0), (0, 16), (16, 0), (16, 16))
    assert residue_shifted_offsets(16) == ((0, 0), (0, 17), (17, 0), (17, 17))
    assert residue_shifted_offsets(16, -1) == (
        (0, 0),
        (0, 15),
        (15, 0),
        (15, 15),
    )
    with pytest.raises(PhaseInterventionError):
        phase_preserving_offsets(0)
    with pytest.raises(PhaseInterventionError):
        residue_shifted_offsets(16, 0)
    with pytest.raises(PhaseInterventionError):
        residue_shifted_offsets(16, 16)
    with pytest.raises(PhaseInterventionError):
        residue_shifted_offsets(True)
    with pytest.raises(PhaseInterventionError):
        translate_reflect(torch.zeros(1, 1, 4, 4), (4, 0))
    with pytest.raises(PhaseInterventionError):
        aggregate_aligned_scores([torch.zeros(1, 1, 4, 4)], [(1, 0)])
