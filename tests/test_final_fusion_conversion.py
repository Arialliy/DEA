from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from utils.final_fusion_conversion import factorize_final_fusion_margin


def _weighted_mean(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    normalized = weights / weights.sum()
    return torch.sum(value[:, 0] * normalized)


def _fixture():
    generator = torch.Generator().manual_seed(917)
    scale_logits = torch.randn(1, 4, 5, 7, generator=generator, dtype=torch.float64)
    fusion_weight = torch.randn(1, 4, 3, 3, generator=generator, dtype=torch.float64)
    fusion_bias = torch.randn(1, generator=generator, dtype=torch.float64)
    patch_scale = 0.25 + torch.rand(
        4, 3, 3, generator=generator, dtype=torch.float64
    )
    target = torch.zeros(1, 5, 7, dtype=torch.float64)
    control = torch.zeros_like(target)
    # Fractional border mass makes the zero-padding convention observable.
    target[0, 0, 0] = 0.25
    target[0, 0, 1] = 0.75
    control[0, 2, 3] = 0.4
    control[0, 1, 5] = 0.6
    return scale_logits, fusion_weight, fusion_bias, patch_scale, target, control


def test_final_unfold_reconstructs_direct_zero_padded_convolution_and_margin() -> None:
    inputs, weight, bias, scale, target, control = _fixture()

    result = factorize_final_fusion_margin(
        inputs,
        target,
        control,
        fusion_weight=weight,
        fusion_bias=bias,
        patch_scale=scale,
        padding=1,
    )
    direct = F.conv2d(inputs, weight, bias, padding=1)
    direct_target = _weighted_mean(direct, target)
    direct_control = _weighted_mean(direct, control)

    assert result.output_shape == (5, 7)
    assert torch.allclose(result.reconstructed_logits, direct, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result.target_logit_mean, direct_target, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result.control_logit_mean, direct_control, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        result.signed_margin,
        direct_target - direct_control,
        atol=1e-12,
        rtol=1e-12,
    )
    assert result.utilization_cosine is not None
    assert result.reconstructed_margin is not None
    assert torch.allclose(
        result.reconstructed_margin,
        result.signed_margin,
        atol=1e-12,
        rtol=1e-12,
    )


def test_normalized_target_control_margin_cancels_fusion_bias_exactly() -> None:
    inputs, weight, bias, scale, target, control = _fixture()
    shifted_bias = bias + 7.25

    base = factorize_final_fusion_margin(
        inputs,
        target,
        control,
        fusion_weight=weight,
        fusion_bias=bias,
        patch_scale=scale,
    )
    shifted = factorize_final_fusion_margin(
        inputs,
        target,
        control,
        fusion_weight=weight,
        fusion_bias=shifted_bias,
        patch_scale=scale,
    )

    assert torch.equal(base.patch_difference, shifted.patch_difference)
    assert torch.equal(base.signed_margin, shifted.signed_margin)
    assert torch.equal(base.available_contrast, shifted.available_contrast)
    assert torch.equal(base.head_sensitivity, shifted.head_sensitivity)
    assert torch.equal(base.utilization_cosine, shifted.utilization_cosine)
    assert torch.allclose(
        shifted.target_logit_mean - base.target_logit_mean,
        torch.tensor(7.25, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert torch.allclose(
        shifted.control_logit_mean - base.control_logit_mean,
        torch.tensor(7.25, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_channel_rescaling_and_sign_flip_preserve_conversion_factorization() -> None:
    inputs, weight, bias, scale, target, control = _fixture()
    gains = torch.tensor([2.0, -0.5, 4.0, -3.0], dtype=torch.float64)

    base = factorize_final_fusion_margin(
        inputs,
        target,
        control,
        fusion_weight=weight,
        fusion_bias=bias,
        patch_scale=scale,
    )
    transformed = factorize_final_fusion_margin(
        inputs * gains.view(1, 4, 1, 1),
        target,
        control,
        fusion_weight=weight / gains.view(1, 4, 1, 1),
        fusion_bias=bias,
        patch_scale=scale * gains.abs().view(4, 1, 1),
    )

    assert torch.allclose(
        transformed.reconstructed_logits,
        base.reconstructed_logits,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        transformed.signed_margin, base.signed_margin, atol=1e-12, rtol=1e-12
    )
    assert torch.allclose(
        transformed.available_contrast,
        base.available_contrast,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        transformed.head_sensitivity,
        base.head_sensitivity,
        atol=1e-12,
        rtol=1e-12,
    )
    assert transformed.utilization_cosine is not None
    assert base.utilization_cosine is not None
    assert torch.allclose(
        transformed.utilization_cosine,
        base.utilization_cosine,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("zero_source", ["contrast", "head"])
def test_zero_factor_is_explicitly_undefined(zero_source: str) -> None:
    inputs, weight, bias, scale, target, control = _fixture()
    if zero_source == "contrast":
        control = target.clone()
    else:
        weight = torch.zeros_like(weight)

    result = factorize_final_fusion_margin(
        inputs,
        target,
        control,
        fusion_weight=weight,
        fusion_bias=bias,
        patch_scale=scale,
    )

    assert result.utilization_cosine is None
    assert result.reconstructed_margin is None
    assert torch.equal(result.signed_margin, torch.zeros_like(result.signed_margin))


def test_nonzero_padding_modes_fail_closed() -> None:
    inputs, weight, bias, scale, target, control = _fixture()
    with pytest.raises(ValueError, match="zero padding"):
        factorize_final_fusion_margin(
            inputs,
            target,
            control,
            fusion_weight=weight,
            fusion_bias=bias,
            patch_scale=scale,
            padding_mode="reflect",
        )


def test_target_level_factorization_rejects_cross_sample_batch_mixing() -> None:
    inputs, weight, bias, scale, target, control = _fixture()
    with pytest.raises(ValueError, match="batch size one"):
        factorize_final_fusion_margin(
            inputs.repeat(2, 1, 1, 1),
            target.repeat(2, 1, 1),
            control.repeat(2, 1, 1),
            fusion_weight=weight,
            fusion_bias=bias,
            patch_scale=scale,
        )
