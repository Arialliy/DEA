import pytest
import torch

from model.signed_local_reference import (
    AnnularReference,
    CenteredLocalReferenceProbe,
    RawUnitLinearProbe,
    SignedStandardizedLocalReferenceProbe,
    UnsignedStandardizedNormControl,
    UnsignedStandardizedProjectionControl,
)


def test_annulus_excludes_inner_guard_and_uses_valid_border_counts() -> None:
    reference = AnnularReference(1, outer_size=5, inner_size=3)
    features = torch.zeros(1, 1, 7, 7)
    features[:, :, 3, 3] = 1000.0
    mean, variance, count = reference.statistics(features)
    assert mean[0, 0, 3, 3].item() == 0.0
    assert variance[0, 0, 3, 3].item() == 0.0
    assert count[0, 0, 3, 3].item() == 16.0
    assert 0.0 < count[0, 0, 0, 0].item() < 16.0


def test_constant_field_standardizes_to_finite_zero_everywhere() -> None:
    reference = AnnularReference(4, outer_size=9, inner_size=3)
    features = torch.full((2, 4, 13, 15), 7.25)
    standardized = reference(features)
    assert bool(torch.isfinite(standardized).all())
    assert torch.equal(standardized, torch.zeros_like(standardized))


def test_standardization_removes_channel_offsets_and_positive_scales() -> None:
    torch.manual_seed(7)
    reference = AnnularReference(3, outer_size=7, inner_size=3)
    features = torch.randn(2, 3, 17, 19, dtype=torch.float64)
    offsets = torch.tensor([2.0, -5.0, 11.0], dtype=torch.float64).view(1, 3, 1, 1)
    scales = torch.tensor([0.5, 2.0, 7.0], dtype=torch.float64).view(1, 3, 1, 1)
    expected = reference(features)
    observed = reference(features * scales + offsets)
    assert torch.allclose(observed, expected, atol=2e-9, rtol=2e-9)


def test_float32_standardization_is_stable_under_large_additive_offset() -> None:
    torch.manual_seed(8)
    reference = AnnularReference(3, outer_size=7, inner_size=3)
    features = torch.randn(2, 3, 17, 19, dtype=torch.float32)
    expected = reference(features)
    observed = reference(features + 10_000.0)
    assert torch.allclose(observed, expected, atol=5e-3, rtol=5e-3)
    assert float(observed.abs().max()) < 50.0


def test_signed_probe_is_odd_around_its_bias() -> None:
    torch.manual_seed(11)
    probe = SignedStandardizedLocalReferenceProbe(5, initialization_seed=3)
    probe.readout.bias.data.fill_(0.7)
    features = torch.randn(2, 5, 21, 21)
    positive = probe(features)
    negative = probe(-features)
    bias = probe.readout.bias.view(1, 1, 1, 1)
    assert torch.allclose(negative - bias, -(positive - bias), atol=2e-5, rtol=2e-5)


def test_unsigned_control_is_even() -> None:
    torch.manual_seed(13)
    control = UnsignedStandardizedNormControl(4)
    features = torch.randn(2, 4, 17, 17)
    assert torch.allclose(control(features), control(-features), atol=1e-6, rtol=1e-6)


def test_parameter_matched_unsigned_projection_is_even() -> None:
    torch.manual_seed(14)
    control = UnsignedStandardizedProjectionControl(4)
    features = torch.randn(2, 4, 17, 17)
    assert torch.allclose(control(features), control(-features), atol=1e-6, rtol=1e-6)
    assert sum(parameter.numel() for parameter in control.parameters()) == 5


def test_probe_parameter_budget_and_gradient_scope() -> None:
    probes = (
        RawUnitLinearProbe(16),
        CenteredLocalReferenceProbe(16),
        SignedStandardizedLocalReferenceProbe(16),
        UnsignedStandardizedProjectionControl(16),
    )
    for probe in probes:
        parameters = [parameter for parameter in probe.parameters() if parameter.requires_grad]
        assert sum(parameter.numel() for parameter in parameters) == 17
        features = torch.randn(2, 16, 15, 17, requires_grad=True)
        loss = probe(features).square().mean()
        loss.backward()
        assert features.grad is not None
        assert all(parameter.grad is not None for parameter in parameters)


def test_probe_weight_scale_is_learnable_and_not_hidden_by_normalization() -> None:
    probe = RawUnitLinearProbe(6)
    features = torch.randn(1, 6, 5, 5)
    original = probe(features)
    probe.readout.weight.data.mul_(2.0)
    doubled = probe(features)
    assert torch.allclose(doubled, 2.0 * original, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_annular_reference_rejects_nonfinite_features(value) -> None:
    reference = AnnularReference(1)
    features = torch.zeros(1, 1, 9, 9)
    features[0, 0, 4, 4] = value
    with pytest.raises(ValueError, match="finite"):
        reference(features)


def test_variance_floor_must_be_strictly_positive() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        AnnularReference(1, variance_floor_scale=0.0)
