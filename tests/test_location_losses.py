from __future__ import annotations

import torch
import pytest

from model.loss import SLSIoULoss
from model.location_losses import (
    GlobalMassLocationLoss,
    legacy_location_loss,
    mass_normalized_centroid,
    normalized_xy_grid,
)


def _reference_sls_iou_loss(
    pred_log: torch.Tensor,
    target: torch.Tensor,
    warm_epoch: int,
    epoch: int,
    with_shape: bool = True,
) -> torch.Tensor:
    """Frozen snapshot of SLSIoULoss before the Phase-0 refactor."""
    pred = torch.sigmoid(pred_log)
    smooth = 0.0

    intersection = pred * target
    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))

    dis = torch.pow((pred_sum - target_sum) / 2, 2)
    alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (
        torch.max(pred_sum, target_sum) + dis + smooth
    )
    iou = (intersection_sum + smooth) / (
        pred_sum + target_sum - intersection_sum + smooth
    )
    location = _reference_legacy_location_loss(pred, target)

    if epoch > warm_epoch:
        siou = alpha * iou
        if with_shape:
            return 1 - siou.mean() + location
        return 1 - siou.mean()
    return 1 - iou.mean()


def _reference_legacy_location_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Frozen snapshot of model.loss.LLoss before the Phase-0 refactor."""
    h = pred.shape[2]
    w = pred.shape[3]
    x_index = torch.arange(0, w, 1, device=pred.device, dtype=pred.dtype).view(1, 1, 1, w) / w
    y_index = torch.arange(0, h, 1, device=pred.device, dtype=pred.dtype).view(1, 1, h, 1) / h
    smooth = 1e-8

    pred_centerx = (x_index * pred).mean(dim=(1, 2, 3))
    pred_centery = (y_index * pred).mean(dim=(1, 2, 3))
    target_centerx = (x_index * target).mean(dim=(1, 2, 3))
    target_centery = (y_index * target).mean(dim=(1, 2, 3))

    angle_loss = (4 / (torch.pi ** 2)) * torch.square(
        torch.atan(pred_centery / (pred_centerx + smooth))
        - torch.atan(target_centery / (target_centerx + smooth))
    )

    pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
    target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)
    length_loss = torch.minimum(pred_length, target_length) / (
        torch.maximum(pred_length, target_length) + smooth
    )

    return (1 - length_loss + angle_loss).mean()


def _target_cases(dtype: torch.dtype = torch.float32) -> list[torch.Tensor]:
    empty = torch.zeros(2, 1, 9, 13, dtype=dtype)
    single = empty.clone()
    single[:, :, 3:5, 7:9] = 1
    multiple = empty.clone()
    multiple[:, :, 1:3, 2:4] = 1
    multiple[:, :, 6:8, 9:12] = 1
    return [empty, single, multiple]


def _point_map(
    coordinates: list[tuple[int, int]],
    *,
    height: int = 16,
    width: int = 16,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    result = torch.zeros(1, 1, height, width, dtype=dtype)
    for y, x in coordinates:
        result[0, 0, y, x] = 1
    return result


def test_normalized_xy_grid_matches_legacy_coordinates() -> None:
    reference = torch.empty(2, 1, 3, 4, dtype=torch.float64)
    x, y = normalized_xy_grid(reference)

    assert x.shape == (1, 1, 1, 4)
    assert y.shape == (1, 1, 3, 1)
    assert x.dtype == reference.dtype
    assert y.dtype == reference.dtype
    assert torch.equal(x.flatten(), torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=x.dtype))
    assert torch.equal(y.flatten(), torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=y.dtype))


def test_legacy_forward_exact() -> None:
    generator = torch.Generator().manual_seed(20260711)
    for target in _target_cases():
        pred = torch.rand(target.shape, generator=generator, dtype=target.dtype)
        actual = legacy_location_loss(pred, target)
        expected = _reference_legacy_location_loss(pred, target)
        assert torch.equal(actual, expected)
        assert torch.isfinite(actual)


def test_legacy_gradient_exact() -> None:
    generator = torch.Generator().manual_seed(20260712)
    for target in _target_cases():
        reference_logits = torch.randn(
            target.shape,
            generator=generator,
            dtype=target.dtype,
            requires_grad=True,
        )
        actual_logits = reference_logits.detach().clone().requires_grad_(True)

        expected = _reference_legacy_location_loss(
            torch.sigmoid(reference_logits),
            target,
        )
        actual = legacy_location_loss(torch.sigmoid(actual_logits), target)
        expected_gradient = torch.autograd.grad(expected, reference_logits)[0]
        actual_gradient = torch.autograd.grad(actual, actual_logits)[0]

        assert torch.equal(actual, expected)
        assert torch.equal(actual_gradient, expected_gradient)
        assert torch.isfinite(actual_gradient).all()


def test_default_sls_forward_and_gradient_match_frozen_reference() -> None:
    generator = torch.Generator().manual_seed(20260713)
    criterion = SLSIoULoss()

    for target in _target_cases():
        for epoch, with_shape in ((0, True), (5, True), (6, True), (6, False)):
            reference_logits = torch.randn(
                target.shape,
                generator=generator,
                dtype=target.dtype,
                requires_grad=True,
            )
            actual_logits = reference_logits.detach().clone().requires_grad_(True)

            expected = _reference_sls_iou_loss(
                reference_logits,
                target,
                warm_epoch=5,
                epoch=epoch,
                with_shape=with_shape,
            )
            actual = criterion(
                actual_logits,
                target,
                warm_epoch=5,
                epoch=epoch,
                with_shape=with_shape,
            )
            expected_gradient = torch.autograd.grad(expected, reference_logits)[0]
            actual_gradient = torch.autograd.grad(actual, actual_logits)[0]

            assert torch.equal(actual, expected)
            assert torch.equal(actual_gradient, expected_gradient)
            assert torch.isfinite(actual)
            assert torch.isfinite(actual_gradient).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_default_sls_cuda_forward_and_gradient_match_frozen_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260713)
    criterion = SLSIoULoss().cuda()

    for cpu_target in _target_cases():
        target = cpu_target.cuda()
        for epoch, with_shape in ((0, True), (5, True), (6, True), (6, False)):
            reference_logits = torch.randn(
                target.shape,
                generator=generator,
                device=target.device,
                dtype=target.dtype,
                requires_grad=True,
            )
            actual_logits = reference_logits.detach().clone().requires_grad_(True)
            expected = _reference_sls_iou_loss(
                reference_logits,
                target,
                warm_epoch=5,
                epoch=epoch,
                with_shape=with_shape,
            )
            actual = criterion(
                actual_logits,
                target,
                warm_epoch=5,
                epoch=epoch,
                with_shape=with_shape,
            )
            expected_gradient = torch.autograd.grad(expected, reference_logits)[0]
            actual_gradient = torch.autograd.grad(actual, actual_logits)[0]

            assert torch.equal(actual, expected)
            assert torch.equal(actual_gradient, expected_gradient)
            assert torch.isfinite(actual)
            assert torch.isfinite(actual_gradient).all()


def test_none_mode_is_post_warm_segmentation_only() -> None:
    generator = torch.Generator().manual_seed(20260714)
    criterion = SLSIoULoss(location_mode="none", lambda_location=7.0)

    for target in _target_cases():
        logits = torch.randn(
            target.shape,
            generator=generator,
            dtype=target.dtype,
            requires_grad=True,
        )
        actual, breakdown = criterion(
            logits,
            target,
            warm_epoch=5,
            epoch=6,
            return_breakdown=True,
        )
        expected = _reference_sls_iou_loss(
            logits,
            target,
            warm_epoch=5,
            epoch=6,
            with_shape=False,
        )

        assert torch.equal(actual, expected)
        assert torch.equal(breakdown["total"], actual)
        assert torch.equal(breakdown["segmentation"], expected.detach())
        assert torch.equal(breakdown["location"], logits.new_zeros(()))
        assert torch.equal(breakdown["location_weighted"], logits.new_zeros(()))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_sls_location_weight_must_be_finite(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        SLSIoULoss(lambda_location=value)


def test_mass_centroid_amplitude_invariance() -> None:
    support = torch.zeros(2, 1, 7, 11, dtype=torch.float64)
    support[0, 0, 1, 2] = 0.25
    support[0, 0, 5, 8] = 0.75
    support[1, 0, 2:5, 4:7] = 0.5

    low_center, low_mass = mass_normalized_centroid(0.25 * support)
    high_center, high_mass = mass_normalized_centroid(0.5 * support)

    torch.testing.assert_close(low_center, high_center, atol=0, rtol=0)
    torch.testing.assert_close(high_mass, 2.0 * low_mass, atol=0, rtol=0)

    for metric in ("polar", "cartesian"):
        criterion = GlobalMassLocationLoss(metric=metric)
        low_loss, _ = criterion(0.25 * support, support)
        high_loss, _ = criterion(0.5 * support, support)
        torch.testing.assert_close(low_loss, high_loss, atol=1e-15, rtol=0)


def test_empty_target_is_finite_and_zero() -> None:
    target = torch.zeros(3, 1, 8, 10, dtype=torch.float64)
    for metric in ("polar", "cartesian"):
        pred = torch.rand_like(target, requires_grad=True)
        loss, logs = GlobalMassLocationLoss(metric=metric)(pred, target)
        gradient = torch.autograd.grad(loss, pred)[0]

        assert torch.equal(loss, pred.new_zeros(()))
        assert torch.equal(gradient, torch.zeros_like(gradient))
        assert all(torch.isfinite(value).all() for value in logs.values())
        assert torch.equal(logs["location_valid_ratio"], torch.tensor(0.0))


def test_tiny_prediction_is_finite() -> None:
    target = _point_map([(6, 9)])
    for metric in ("polar", "cartesian"):
        pred = torch.full_like(target, torch.finfo(target.dtype).tiny, requires_grad=True)
        loss, logs = GlobalMassLocationLoss(metric=metric)(pred, target)
        gradient = torch.autograd.grad(loss, pred)[0]

        assert torch.isfinite(loss)
        assert torch.isfinite(gradient).all()
        assert all(torch.isfinite(value).all() for value in logs.values())


def test_cartesian_common_translation_invariance() -> None:
    criterion = GlobalMassLocationLoss(metric="cartesian", beta=0.02)

    target_before = _point_map([(3, 4)])
    pred_before = _point_map([(5, 5)])
    target_after = _point_map([(7, 9)])
    pred_after = _point_map([(9, 10)])

    loss_before, logs_before = criterion(pred_before, target_before)
    loss_after, logs_after = criterion(pred_after, target_after)

    torch.testing.assert_close(loss_before, loss_after, atol=0, rtol=0)
    torch.testing.assert_close(
        logs_before["global_centroid_l1"],
        logs_after["global_centroid_l1"],
        atol=0,
        rtol=0,
    )
    assert loss_before > 0


def test_global_multitarget_cancellation_remains() -> None:
    target = _point_map([(4, 4), (12, 12)])
    # The left target moves right and the right target moves left by the same
    # amount. Their global centroids agree although neither instance is localised.
    pred = _point_map([(4, 6), (12, 10)])

    target_center, _ = mass_normalized_centroid(target)
    pred_center, _ = mass_normalized_centroid(pred)
    global_loss, logs = GlobalMassLocationLoss(metric="cartesian")(pred, target)
    per_instance_l1 = 2.0 / target.shape[-1] + 2.0 / target.shape[-1]

    assert not torch.equal(pred, target)
    assert per_instance_l1 > 0
    assert torch.equal(pred_center, target_center)
    assert torch.equal(global_loss, target.new_zeros(()))
    assert torch.equal(logs["global_centroid_l1"], target.new_zeros(()))
