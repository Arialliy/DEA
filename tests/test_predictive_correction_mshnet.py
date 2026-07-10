import torch

from model.predictive_correction_mshnet import (
    PredictiveCorrectionMSHNet,
    TiedPredictionOperator,
)


def test_tied_operator_implements_numerical_adjoint():
    torch.manual_seed(3)
    operator = TiedPredictionOperator(8).double().eval()
    with torch.no_grad():
        # Cover the learned case rather than only the identity-like spatial
        # initialization.  A non-symmetric kernel catches an incorrect
        # forward-order reuse in the adjoint path.
        operator.depthwise_weight.normal_(mean=0.0, std=0.4)
        operator.pointwise_weight.normal_(mean=0.0, std=0.3)
    weights = operator.bounded_weights()

    for height, width in ((17, 20), (16, 19)):
        x = torch.randn(2, 8, height, width, dtype=torch.float64)
        y = torch.randn_like(x)
        lhs = (operator(x, weights=weights) * y).sum()
        rhs = (x * operator.adjoint(y, weights=weights)).sum()
        relative_error = (lhs - rhs).abs() / (
            lhs.abs() + rhs.abs() + 1e-12
        )

        assert torch.allclose(lhs, rhs, atol=2e-5, rtol=2e-5)
        assert relative_error.item() < 1e-6


def test_group_norm_supports_non_multiple_of_four_state_width():
    model = PredictiveCorrectionMSHNet(3, state_channels=10).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 48), True, return_dict=True)
    assert output["pred"].shape == (1, 1, 32, 48)
    assert model.observation_norm.num_groups == 2


def test_model_has_one_shared_decoder_trajectory_and_expected_shapes():
    model = PredictiveCorrectionMSHNet(3, state_channels=16).eval()
    x = torch.randn(1, 3, 64, 80)

    with torch.no_grad():
        output = model(x, True, return_dict=True, return_details=True)

    assert [tuple(item.shape) for item in output["state_logits"]] == [
        (1, 1, 4, 5),
        (1, 1, 8, 10),
        (1, 1, 16, 20),
        (1, 1, 32, 40),
        (1, 1, 64, 80),
    ]
    assert output["pred"].shape == (1, 1, 64, 80)
    assert len(output["states"]) == 5
    assert len(output["corrections"]) == 5

    forbidden = ("decoder_", "output_", "final", "decidability")
    assert not any(
        token in name
        for name, _ in model.named_modules()
        for token in forbidden
    )
    assert sum(
        1 for name, _ in model.named_modules()
        if name == "prediction_operator"
    ) == 1
    assert sum(1 for name, _ in model.named_modules() if name == "readout") == 1


def test_warm_flag_changes_only_auxiliary_loss_contract():
    torch.manual_seed(4)
    model = PredictiveCorrectionMSHNet(3, state_channels=16).eval()
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        cold = model(x, False, return_dict=True)
        warm = model(x, True, return_dict=True)

    assert torch.equal(cold["pred"], warm["pred"])
    assert cold["aux_enabled"] is False
    assert warm["aux_enabled"] is True


def test_all_new_parameters_receive_finite_gradient():
    torch.manual_seed(5)
    model = PredictiveCorrectionMSHNet(3, state_channels=16).train()
    output = model(
        torch.randn(2, 3, 32, 32),
        True,
        return_dict=True,
        return_details=True,
    )
    loss = sum(logit.square().mean() for logit in output["state_logits"])
    loss.backward()

    for name, parameter in model.named_parameters():
        if not (
            name.startswith("scale_adapters")
            or name.startswith("observation_norm")
            or name.startswith("prediction_operator")
            or name.startswith("readout")
            or name in ("state_prior", "raw_delta")
        ):
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum().item() > 0.0, name


def test_pseudo_huber_influence_is_bounded_per_residual_channel():
    model = PredictiveCorrectionMSHNet(
        3, state_channels=8, delta_init=0.7, delta_min=0.05
    )
    residual = torch.linspace(-1e4, 1e4, 101).view(1, 1, 1, -1)
    residual = residual.expand(1, 8, 1, -1)
    influence = model._influence(residual)

    assert torch.all(influence.abs() <= model.delta + 1e-5)


def test_each_initial_correction_reduces_its_scale_observation_energy():
    torch.manual_seed(6)
    model = PredictiveCorrectionMSHNet(3, state_channels=16).eval()
    with torch.no_grad():
        model.prediction_operator.depthwise_weight.normal_(mean=0.0, std=0.4)
        model.prediction_operator.pointwise_weight.normal_(mean=0.0, std=0.3)
    with torch.no_grad():
        output = model(
            torch.randn(1, 3, 64, 64),
            True,
            return_dict=True,
            return_details=True,
        )

    for before, after in zip(
        output["local_observation_energies_before"],
        output["local_observation_energies_after"],
    ):
        assert torch.all(after <= before + 1e-6)
