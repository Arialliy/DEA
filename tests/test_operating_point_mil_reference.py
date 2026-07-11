import numpy as np
import pytest
import torch

from model.operating_point_mil_reference import (
    exact_operating_point_mil_reference,
)
from utils.metric import match_components_hungarian


def test_feasible_instance_peaks_and_background_budget_have_zero_loss():
    logits = torch.full((1, 1, 3, 5), -2.0)
    labels = torch.zeros((1, 1, 3, 5), dtype=torch.long)
    labels[0, 0, 1, 1] = 1
    labels[0, 0, 1, 3] = 2
    logits[0, 0, 1, 1] = 2.0
    logits[0, 0, 1, 3] = 1.5

    result = exact_operating_point_mil_reference(
        logits,
        labels,
        positive_margin=1.0,
        negative_margin=1.0,
    )

    assert result["loss"].item() == 0.0
    assert result["positive_violation"].item() == 0.0
    assert result["background_violation"].item() == 0.0


def test_positive_margin_closes_the_strict_threshold_failure_at_equality():
    logits = torch.tensor([[[[0.0, -2.0]]]])
    labels = torch.tensor([[[[1, 0]]]], dtype=torch.long)

    result = exact_operating_point_mil_reference(
        logits,
        labels,
        threshold=0.0,
        positive_margin=0.25,
        negative_margin=0.0,
    )

    assert result["positive_violation"].item() == pytest.approx(0.25)


def test_background_order_statistic_exempts_exactly_b_pixels():
    logits = torch.tensor([[[[3.0, 2.0, 1.0, -2.0]]]])
    labels = torch.zeros_like(logits, dtype=torch.long)

    result = exact_operating_point_mil_reference(
        logits,
        labels,
        positive_margin=1.0,
        negative_margin=0.0,
        allowed_background_exceedances=2,
    )

    assert result["positive_violation"].item() == 0.0
    assert result["background_violation"].item() == 1.0


def test_empty_image_and_all_background_exempt_are_well_defined():
    logits = torch.randn(2, 1, 2, 2)
    labels = torch.zeros_like(logits, dtype=torch.long)

    result = exact_operating_point_mil_reference(
        logits,
        labels,
        positive_margin=1.0,
        allowed_background_exceedances=4,
        reduction="none",
    )

    assert torch.equal(result["loss"], torch.zeros(2))


def test_one_pixel_per_instance_is_a_zero_loss_shape_degeneracy():
    labels = torch.zeros((1, 1, 5, 5), dtype=torch.long)
    labels[0, 0, 1:4, 1:4] = 1
    logits = torch.full((1, 1, 5, 5), -2.0)
    logits[0, 0, 2, 2] = 2.0

    result = exact_operating_point_mil_reference(
        logits,
        labels,
        positive_margin=1.0,
        negative_margin=1.0,
    )
    prediction = logits[0, 0].numpy() > 0
    target = labels[0, 0].numpy() > 0

    assert result["loss"].item() == 0.0
    assert np.logical_and(prediction, target).sum() / np.logical_or(
        prediction, target
    ).sum() == pytest.approx(1 / 9)


def test_one_allowed_bridge_pixel_breaks_component_pd_at_zero_loss():
    labels = torch.zeros((1, 1, 1, 5), dtype=torch.long)
    labels[0, 0, 0, 1] = 1
    labels[0, 0, 0, 3] = 2
    logits = torch.full((1, 1, 1, 5), -2.0)
    logits[0, 0, 0, 1:4] = 2.0

    result = exact_operating_point_mil_reference(
        logits,
        labels,
        positive_margin=1.0,
        negative_margin=1.0,
        allowed_background_exceedances=1,
    )
    component_match = match_components_hungarian(
        logits[0, 0].numpy() > 0,
        labels[0, 0].numpy() > 0,
    )

    assert result["loss"].item() == 0.0
    assert len(component_match.target_regions) == 2
    assert len(component_match.prediction_regions) == 1
    assert len(component_match.matches) == 1


def test_gradient_routes_only_to_current_worst_constraints():
    logits = torch.tensor(
        [[[[0.0, -2.0, 2.0, -2.0]]]], requires_grad=True
    )
    labels = torch.tensor([[[[1, 0, 2, 0]]]], dtype=torch.long)

    result = exact_operating_point_mil_reference(
        logits,
        labels,
        positive_margin=1.0,
        negative_margin=0.0,
    )
    result["loss"].backward()

    assert logits.grad[0, 0, 0, 0].item() == -1.0
    assert torch.count_nonzero(logits.grad).item() == 1


@pytest.mark.parametrize("margin", [0.0, -1.0])
def test_nonpositive_positive_margin_is_rejected(margin):
    with pytest.raises(ValueError, match="positive_margin"):
        exact_operating_point_mil_reference(
            torch.zeros((1, 1, 1, 1)),
            torch.ones((1, 1, 1, 1), dtype=torch.long),
            positive_margin=margin,
        )
