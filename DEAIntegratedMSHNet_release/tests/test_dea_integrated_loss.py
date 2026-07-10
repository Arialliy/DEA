from __future__ import annotations

import torch

from model.dea_integrated_loss import (
    residual_action_distribution,
    residual_aligned_route_loss,
)


def test_residual_action_distribution_has_exact_correction_semantics():
    logits = torch.tensor([[[[-100.0, 100.0], [100.0, -100.0]]]])
    target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    teacher = residual_action_distribution(logits, target)

    # false negative -> increase; true positive -> keep
    assert teacher[0, 0, 0, 0] > 0.999
    assert teacher[0, 2, 0, 1] > 0.999
    # false positive -> decrease; true negative -> keep
    assert teacher[0, 1, 1, 0] > 0.999
    assert teacher[0, 2, 1, 1] > 0.999
    assert torch.all(teacher >= 0)
    assert torch.allclose(teacher.sum(dim=1), torch.ones_like(target[:, 0]))

    probability = torch.sigmoid(logits)
    expected_residual = target - probability
    assert torch.allclose(
        teacher[:, 0:1] - teacher[:, 1:2],
        expected_residual,
    )


def test_route_loss_detaches_teacher_logit_but_trains_every_route():
    z_base = torch.zeros(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(z_base)
    target[:, :, 2:4, 3:5] = 1.0

    route_logits = []
    routes = []
    for size in (8, 4, 2, 1):
        logits = torch.randn(2, 3, size, size, requires_grad=True)
        route_logits.append(logits)
        routes.append({"probabilities": torch.softmax(logits, dim=1)})
    output = {
        "routes": routes,
        "scale_fusion": {"z_base": z_base},
    }

    loss, log = residual_aligned_route_loss(output, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert z_base.grad is None
    for logits in route_logits:
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0.0
    assert set(
        ["teacher_target_mass", "teacher_clutter_mass", "teacher_keep_mass"]
    ).issubset(log)


def test_route_loss_is_finite_for_empty_target_and_extreme_logits():
    z_base = torch.full((1, 1, 8, 8), -100.0, requires_grad=True)
    target = torch.zeros_like(z_base)
    routes = []
    logits_list = []
    for size in (8, 4, 2, 1):
        logits = torch.full((1, 3, size, size), -100.0, requires_grad=True)
        logits_list.append(logits)
        routes.append({"probabilities": torch.softmax(logits, dim=1)})

    loss, _ = residual_aligned_route_loss(
        {"routes": routes, "scale_fusion": {"z_base": z_base}},
        target,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(logits.grad).all() for logits in logits_list)
