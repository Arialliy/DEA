from __future__ import annotations

import pytest
import torch

from model.trace_run_semiring import (
    RootCellRunSemiring,
    TraceSemiringError,
    brute_force_reference,
    exact_sum_product,
    zero_score_cardinality,
    zero_score_log_cardinality,
)


def _masks(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    support = torch.ones((2, height, width), dtype=torch.bool)
    root = torch.zeros_like(support)
    root[0, : min(2, height), : min(2, width)] = True
    root[1, 0, : min(3, width)] = True
    root[1, min(1, height - 1), 1 : min(3, width)] = True

    # Exercise non-rectangular boundary masks.  Root validity is deliberately
    # supplied independently; the solver must also require support validity.
    support[0, height - 1, width - 1] = False
    support[1, 0, 0] = False
    if height == 4 and width == 4:
        support[1, 3, 0] = False
        support[1, 2, 3] = False
    return support, root


@pytest.mark.parametrize("height,width", [(3, 3), (4, 4)])
def test_fp64_exact_dp_matches_independent_brute_force(
    height: int, width: int
) -> None:
    generator = torch.Generator().manual_seed(20260713 + 10 * height + width)
    root_energy = torch.randn(
        (2, height, width), generator=generator, dtype=torch.float64
    ).requires_grad_(True)
    support_energy = torch.randn(
        (2, height, width), generator=generator, dtype=torch.float64
    ).requires_grad_(True)
    valid_support, valid_root = _masks(height, width)

    solver = RootCellRunSemiring(cardinality_correction=True)
    exact = solver(
        root_energy,
        support_energy,
        valid_support,
        valid_root,
    )
    brute = brute_force_reference(
        root_energy,
        support_energy,
        valid_support,
        valid_root,
    )

    assert exact.logZ_positive.dtype == torch.float64
    torch.testing.assert_close(exact.logZ_positive, brute.logZ_positive, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(exact.logZ_total, brute.logZ_total, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(exact.p_nonempty, brute.p_nonempty, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(exact.map_energy, brute.map_energy, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        exact.map_log_joint_posterior,
        brute.map_log_joint_posterior,
        atol=2e-12,
        rtol=2e-12,
    )
    assert torch.equal(exact.map_root, brute.map_root)
    assert torch.equal(exact.map_intervals, brute.map_intervals)
    assert torch.equal(exact.map_support, brute.map_support)

    cardinality = zero_score_cardinality(
        valid_support, valid_root, dtype=torch.float64
    )
    assert torch.equal(cardinality.round().to(torch.long), brute.state_count)
    torch.testing.assert_close(
        zero_score_log_cardinality(valid_support, valid_root),
        torch.log(brute.state_count.to(torch.float64)),
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(exact.cardinality, cardinality, atol=2e-9, rtol=2e-12)

    exact_gradient = torch.autograd.grad(
        exact.logZ_total.sum(), (root_energy, support_energy), retain_graph=True
    )
    brute_gradient = torch.autograd.grad(
        brute.logZ_total.sum(), (root_energy, support_energy)
    )
    for actual, expected in zip(exact_gradient, brute_gradient):
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=5e-12, rtol=5e-12)


def test_no_logk_ablation_is_the_same_state_space_without_base_measure() -> None:
    root_energy = torch.tensor(
        [[[0.2, -0.7, 0.4], [0.1, 0.3, -0.2], [-0.8, 0.5, 0.9]]],
        dtype=torch.float64,
    )
    support_energy = torch.tensor(
        [[[0.4, 0.2, -0.1], [-0.5, 0.8, 0.1], [0.3, -0.2, 0.6]]],
        dtype=torch.float64,
    )
    valid_support = torch.ones((1, 3, 3), dtype=torch.bool)
    valid_root = torch.tensor(
        [[[True, True, False], [True, False, False], [False, False, False]]]
    )

    corrected = exact_sum_product(
        root_energy,
        support_energy,
        valid_support,
        valid_root,
        cardinality_correction=True,
    )
    uncorrected = exact_sum_product(
        root_energy,
        support_energy,
        valid_support,
        valid_root,
        cardinality_correction=False,
    )
    brute_uncorrected = brute_force_reference(
        root_energy,
        support_energy,
        valid_support,
        valid_root,
        cardinality_correction=False,
    )

    torch.testing.assert_close(
        corrected.logZ_positive,
        uncorrected.logZ_positive - corrected.log_cardinality,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        uncorrected.logZ_positive,
        brute_uncorrected.logZ_positive,
        atol=1e-12,
        rtol=1e-12,
    )


def test_cardinality_correction_removes_shape_count_from_existence_prior() -> None:
    valid_support, valid_root = _masks(4, 4)
    bias = torch.tensor([-1.25, 0.85], dtype=torch.float64)
    root_energy = bias[:, None, None].expand(2, 4, 4).clone()
    support_energy = torch.zeros_like(root_energy)

    result = RootCellRunSemiring()(
        root_energy,
        support_energy,
        valid_support,
        valid_root,
        return_map=False,
    )

    assert result.cardinality[0] != result.cardinality[1]
    torch.testing.assert_close(result.logZ_positive, bias, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(result.p_nonempty, torch.sigmoid(bias), atol=2e-12, rtol=2e-12)


def test_marginals_are_gradients_of_empty_inclusive_logZ_total() -> None:
    # There is one legal non-empty atom, so its unconditional root/support
    # marginal is sigmoid(energy).  A logZ_positive derivative would be 1 and
    # would fail this test.
    root_energy = torch.tensor([[[0.3]]], dtype=torch.float64)
    support_energy = torch.tensor([[[-0.8]]], dtype=torch.float64)
    mask = torch.ones((1, 1, 1), dtype=torch.bool)
    expected = torch.sigmoid(torch.tensor(-0.5, dtype=torch.float64))

    output = RootCellRunSemiring()(
        root_energy,
        support_energy,
        mask,
        mask,
        return_marginals=True,
    )

    torch.testing.assert_close(output.root_marginal, expected.reshape(1, 1, 1))
    torch.testing.assert_close(output.support_marginal, expected.reshape(1, 1, 1))
    torch.testing.assert_close(output.p_nonempty, expected.reshape(1))
    torch.testing.assert_close(
        output.root_marginal.sum(dim=(1, 2)), output.p_nonempty
    )


def test_rows_before_first_allowed_root_have_zero_finite_gradient() -> None:
    generator = torch.Generator().manual_seed(91)
    root_energy = torch.randn(
        (1, 3, 3), generator=generator, dtype=torch.float64
    ).requires_grad_(True)
    support_energy = torch.randn(
        (1, 3, 3), generator=generator, dtype=torch.float64
    ).requires_grad_(True)
    valid_support = torch.ones((1, 3, 3), dtype=torch.bool)
    valid_root = torch.zeros_like(valid_support)
    valid_root[0, 2, 1] = True

    exact = RootCellRunSemiring()(
        root_energy,
        support_energy,
        valid_support,
        valid_root,
        return_map=False,
    )
    brute = brute_force_reference(
        root_energy,
        support_energy,
        valid_support,
        valid_root,
    )
    exact_gradient = torch.autograd.grad(
        exact.logZ_total.sum(), (root_energy, support_energy), retain_graph=True
    )
    brute_gradient = torch.autograd.grad(
        brute.logZ_total.sum(), (root_energy, support_energy)
    )
    for actual, expected in zip(exact_gradient, brute_gradient):
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(exact_gradient[1][0, :2]) == 0


def test_map_backtrace_is_a_consecutive_connected_chain_with_canonical_root() -> None:
    root_energy = torch.full((1, 4, 4), -4.0, dtype=torch.float64)
    root_energy[0, 1, 1] = 3.0
    support_energy = torch.full((1, 4, 4), -2.0, dtype=torch.float64)
    support_energy[0, 1, 1:3] = 2.0
    support_energy[0, 2, 2:4] = 2.0
    support_energy[0, 3, 3] = 2.0
    valid_support = torch.ones((1, 4, 4), dtype=torch.bool)
    valid_root = torch.zeros_like(valid_support)
    valid_root[0, 1, 1] = True

    result = RootCellRunSemiring()(root_energy, support_energy, valid_support, valid_root)
    occupied_rows = torch.nonzero(result.map_support[0].any(dim=1), as_tuple=False).flatten()
    assert torch.equal(
        occupied_rows,
        torch.arange(occupied_rows[0], occupied_rows[-1] + 1),
    )
    first_row = int(occupied_rows[0])
    first_left = int(torch.nonzero(result.map_support[0, first_row])[0])
    assert result.map_root[0].tolist() == [first_row, first_left]
    assert bool(valid_root[0, first_row, first_left])
    for previous_y, current_y in zip(occupied_rows[:-1], occupied_rows[1:]):
        previous_x = torch.nonzero(result.map_support[0, previous_y]).flatten()
        current_x = torch.nonzero(result.map_support[0, current_y]).flatten()
        assert int(current_x[0]) <= int(previous_x[-1]) + 1
        assert int(current_x[-1]) >= int(previous_x[0]) - 1


def test_masks_fail_closed_when_no_nonempty_atom_exists() -> None:
    energy = torch.zeros((1, 2, 2), dtype=torch.float64)
    support = torch.ones((1, 2, 2), dtype=torch.bool)
    no_roots = torch.zeros_like(support)
    with pytest.raises(TraceSemiringError, match="at least one legal atom"):
        RootCellRunSemiring()(energy, energy, support, no_roots)
