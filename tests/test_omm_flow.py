from __future__ import annotations

import torch

from model.MSHNet import MSHNet
from model.mshnet_evidence_view import forward_mshnet_evidence
from model.omm_flow import (
    experimental_categorical_odds_fusion,
    instance_balanced_logistic_risk,
    label_target_components,
    omm2d_identity_risk,
)


def _logit(probability: float) -> float:
    value = torch.tensor(float(probability))
    return float(torch.logit(value))


def test_rejected_signed_softmax_lift_has_zero_sum_gauge() -> None:
    base = torch.zeros(1, 4, 1, 1)
    shifted = base.clone()
    shifted[:, 0] = 12.0
    shifted[:, 1] = -12.0

    base_final = base.sum(dim=1, keepdim=True)
    shifted_final = shifted.sum(dim=1, keepdim=True)
    base_probability = torch.sigmoid(base_final)
    shifted_probability = torch.sigmoid(shifted_final)
    base_responsibility = torch.softmax(base, dim=1)
    shifted_responsibility = torch.softmax(shifted, dim=1)

    assert torch.equal(base_final, shifted_final)
    assert torch.equal(base_probability, shifted_probability)
    assert torch.allclose(base_responsibility, torch.full_like(base, 0.25))
    assert float(shifted_responsibility[:, 0]) > 0.9999


def test_experimental_odds_fusion_is_probability_conservative() -> None:
    torch.manual_seed(101)
    contributions = torch.randn(2, 4, 5, 7, requires_grad=True)
    bias = torch.tensor([0.3], requires_grad=True)

    output = experimental_categorical_odds_fusion(contributions, bias=bias)

    assert torch.all(output["source_mass"] >= 0)
    assert torch.allclose(
        output["joint_probability"].sum(dim=1, keepdim=True),
        torch.ones_like(output["probability"]),
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(
        output["source_mass"].sum(dim=1, keepdim=True),
        output["probability"],
        atol=1e-7,
        rtol=1e-6,
    )
    output["source_mass"].square().mean().backward()
    assert contributions.grad is not None
    assert bias.grad is not None
    assert torch.isfinite(contributions.grad).all()
    assert torch.isfinite(bias.grad).all()

    extreme = experimental_categorical_odds_fusion(
        torch.full((1, 4, 1, 1), -100.0)
    )
    assert torch.isfinite(extreme["joint_probability"]).all()
    assert torch.allclose(
        extreme["scale_responsibility"],
        torch.full_like(extreme["scale_responsibility"], 0.25),
    )


def test_experimental_odds_fusion_matches_linear_fusion_at_scale_consensus() -> None:
    contributions = torch.full(
        (1, 4, 2, 3),
        0.2,
        dtype=torch.float64,
        requires_grad=True,
    )
    bias = torch.tensor([-0.35], dtype=torch.float64, requires_grad=True)
    output = experimental_categorical_odds_fusion(contributions, bias=bias)

    assert torch.allclose(
        output["fused_logit"],
        output["linear_logit"],
        atol=1e-12,
        rtol=1e-12,
    )
    gradient = torch.autograd.grad(
        output["fused_logit"].sum(),
        contributions,
    )[0]
    assert torch.allclose(gradient, torch.ones_like(gradient))


def test_experimental_odds_fusion_exposes_zero_sum_scale_disagreement() -> None:
    base = torch.zeros(1, 4, 1, 1)
    shifted = base.clone()
    shifted[:, 0] = 2.0
    shifted[:, 1] = -2.0

    base_output = experimental_categorical_odds_fusion(base)
    shifted_output = experimental_categorical_odds_fusion(shifted)

    assert torch.equal(base_output["linear_logit"], shifted_output["linear_logit"])
    assert float(shifted_output["fused_logit"]) > float(base_output["fused_logit"])
    assert float(shifted_output["scale_responsibility"][:, 0]) > 0.999


def test_component_labelling_matches_eight_connectivity() -> None:
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, 0] = 1
    target[0, 0, 1, 1] = 1
    target[0, 0, 3, 3] = 1

    labels = label_target_components(target)

    assert int(labels.max()) == 2
    assert int(labels[0, 0, 0, 0]) == int(labels[0, 0, 1, 1])
    assert int(labels[0, 0, 3, 3]) != int(labels[0, 0, 0, 0])


def test_omm2d_empty_target_has_exact_positive_gradient() -> None:
    logits = torch.zeros(2, 1, 3, 5, dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(logits)

    output = omm2d_identity_risk(logits, target)
    output["loss"].backward()

    expected_loss = torch.tensor(0.5, dtype=logits.dtype)
    expected_gradient = torch.full_like(logits, 0.25 / logits.numel())
    assert output["num_instances"] == 0
    assert torch.allclose(output["loss"], expected_loss)
    assert torch.allclose(logits.grad, expected_gradient, atol=1e-12, rtol=0)


def test_omm2d_gives_each_instance_equal_miss_weight_despite_area() -> None:
    target = torch.zeros(1, 1, 4, 6)
    instance_labels = torch.zeros_like(target, dtype=torch.long)
    target[0, 0, 0, 0] = 1
    instance_labels[0, 0, 0, 0] = 1
    target[0, 0, 2:4, 3:6] = 1
    instance_labels[0, 0, 2:4, 3:6] = 2

    logits = torch.full_like(target, _logit(0.01))
    logits[instance_labels == 2] = _logit(0.99)
    output = omm2d_identity_risk(
        logits,
        target,
        instance_labels=instance_labels,
    )

    assert output["instance_areas"].tolist() == [1, 6]
    assert torch.allclose(
        output["per_instance_miss"],
        torch.tensor([0.99, 0.01]),
        atol=1e-6,
        rtol=0,
    )
    assert torch.allclose(
        output["miss_risk"],
        torch.tensor(0.5),
        atol=1e-6,
        rtol=0,
    )


def test_omm2d_weak_target_response_is_monotone() -> None:
    target = torch.zeros(1, 1, 3, 3)
    target[0, 0, 1, 1] = 1
    instance_labels = target.long()

    weak = torch.full_like(target, _logit(0.001))
    stronger = weak.clone()
    stronger[0, 0, 1, 1] = _logit(0.2)
    weak_loss = omm2d_identity_risk(
        weak,
        target,
        instance_labels=instance_labels,
    )["loss"]
    stronger_loss = omm2d_identity_risk(
        stronger,
        target,
        instance_labels=instance_labels,
    )["loss"]

    assert stronger_loss < weak_loss


def test_omm2d_uses_global_batch_instance_weighting_and_is_id_permutation_invariant() -> None:
    target = torch.zeros(2, 1, 4, 5)
    labels = torch.zeros_like(target, dtype=torch.long)
    target[0, 0, 0, 0] = 1
    labels[0, 0, 0, 0] = 7
    target[1, 0, 0, 0] = 1
    labels[1, 0, 0, 0] = 3
    target[1, 0, 3, 4] = 1
    labels[1, 0, 3, 4] = 11

    logits = torch.full_like(target, _logit(0.01))
    logits[0, 0, 0, 0] = _logit(0.1)
    logits[1, 0, 0, 0] = _logit(0.4)
    logits[1, 0, 3, 4] = _logit(0.7)
    output = omm2d_identity_risk(logits, target, instance_labels=labels)

    expected = torch.tensor(((1 - 0.1) + (1 - 0.4) + (1 - 0.7)) / 3)
    assert torch.allclose(output["miss_risk"], expected, atol=1e-6, rtol=0)

    permuted = labels.clone()
    permuted[labels == 7] = 101
    permuted[labels == 3] = 53
    permuted[labels == 11] = 2
    permuted_output = omm2d_identity_risk(
        logits,
        target,
        instance_labels=permuted,
    )
    assert torch.allclose(
        output["loss"],
        permuted_output["loss"],
        atol=1e-7,
        rtol=0,
    )


def test_omm2d_double_precision_gradcheck() -> None:
    target = torch.zeros(1, 1, 2, 3, dtype=torch.float64)
    target[0, 0, 0, 1] = 1
    labels = target.long()
    logits = torch.randn(1, 1, 2, 3, dtype=torch.float64, requires_grad=True)

    def scalar_loss(value: torch.Tensor) -> torch.Tensor:
        return omm2d_identity_risk(
            value,
            target,
            instance_labels=labels,
        )["loss"]

    assert torch.autograd.gradcheck(scalar_loss, (logits,), eps=1e-6, atol=1e-5)


def test_vectorized_instance_reduction_matches_reference_and_fast_path() -> None:
    torch.manual_seed(109)
    logits = torch.randn(2, 1, 4, 5, dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(logits)
    labels = torch.zeros_like(target, dtype=torch.long)
    target[0, 0, 0, 0:2] = 1
    labels[0, 0, 0, 0:2] = 9
    target[1, 0, 1:3, 1:4] = 1
    labels[1, 0, 1:3, 1:4] = 4
    target[1, 0, 3, 4] = 1
    labels[1, 0, 3, 4] = 17

    output = omm2d_identity_risk(
        logits,
        target,
        instance_labels=labels,
        validate_instance_labels=False,
    )
    probability = torch.sigmoid(logits)
    reference = torch.stack(
        [
            1 - probability[0, 0, 0, 0:2].mean(),
            1 - probability[1, 0, 1:3, 1:4].mean(),
            1 - probability[1, 0, 3, 4],
        ]
    )
    assert torch.allclose(output["per_instance_miss"], reference)
    assert output["instance_areas"].tolist() == [2, 6, 1]
    assert output["instance_batch_indices"].tolist() == [0, 1, 1]
    assert output["instance_ids"].tolist() == [9, 4, 17]

    gradient = torch.autograd.grad(output["loss"], logits)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_logistic_control_keeps_high_confidence_error_gradients() -> None:
    empty_target = torch.zeros(1, 1, 1, 1)
    false_alarm_logit = torch.tensor([[[[20.0]]]], requires_grad=True)
    identity = omm2d_identity_risk(false_alarm_logit, empty_target)["loss"]
    identity.backward()
    identity_gradient = float(false_alarm_logit.grad)

    false_alarm_logit.grad = None
    logistic = instance_balanced_logistic_risk(
        false_alarm_logit,
        empty_target,
    )["loss"]
    logistic.backward()
    logistic_gradient = float(false_alarm_logit.grad)

    assert identity_gradient == 0.0
    assert logistic_gradient > 0.999

    target = torch.ones(1, 1, 1, 1)
    missed_target_logit = torch.tensor([[[[-20.0]]]], requires_grad=True)
    result = instance_balanced_logistic_risk(missed_target_logit, target)
    result["loss"].backward()
    assert float(missed_target_logit.grad) < -0.999


def test_mshnet_canonical_fusion_and_omm2d_backward_smoke() -> None:
    torch.manual_seed(107)
    model = MSHNet(3).eval()
    image = torch.randn(1, 3, 32, 32, requires_grad=True)
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 14:17, 15:18] = 1

    evidence = forward_mshnet_evidence(model, image)
    risk = omm2d_identity_risk(evidence["z_base"], target)
    risk["loss"].backward()

    assert image.grad is not None
    assert model.final.weight.grad is not None
    assert model.final.bias.grad is not None
    for side_head in (
        model.output_0,
        model.output_1,
        model.output_2,
        model.output_3,
    ):
        assert side_head.weight.grad is not None
        assert torch.isfinite(side_head.weight.grad).all()
        assert float(side_head.weight.grad.abs().sum()) > 0.0
    assert torch.isfinite(image.grad).all()
    assert torch.isfinite(model.final.weight.grad).all()
    assert torch.isfinite(model.final.bias.grad).all()
    assert float(image.grad.abs().sum()) > 0.0
    assert float(model.final.weight.grad.abs().sum()) > 0.0


if __name__ == "__main__":
    test_rejected_signed_softmax_lift_has_zero_sum_gauge()
    test_experimental_odds_fusion_is_probability_conservative()
    test_experimental_odds_fusion_matches_linear_fusion_at_scale_consensus()
    test_experimental_odds_fusion_exposes_zero_sum_scale_disagreement()
    test_component_labelling_matches_eight_connectivity()
    test_omm2d_empty_target_has_exact_positive_gradient()
    test_omm2d_gives_each_instance_equal_miss_weight_despite_area()
    test_omm2d_weak_target_response_is_monotone()
    test_mshnet_canonical_fusion_and_omm2d_backward_smoke()
