from __future__ import annotations

import pytest
import torch

from model.MSHNet import MSHNet
from model.mshnet_d0_backbone import MSHNetD0Backbone, MSHNetD0BackboneError


def _canonical_d0(model: MSHNet, image: torch.Tensor) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    handle = model.decoder_0.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        model(image, True)
    finally:
        handle.remove()
    assert len(captured) == 1
    return captured[0]


def test_headless_front_is_bitwise_equal_to_canonical_mshnet_d0() -> None:
    torch.manual_seed(20260713)
    canonical = MSHNet(3).eval()
    headless = MSHNetD0Backbone(3).eval()
    headless.load_mshnet_front_state_dict(canonical.state_dict())
    image = torch.randn(2, 3, 64, 80)
    with torch.no_grad():
        expected = _canonical_d0(canonical, image)
        observed = headless(image)
    assert torch.equal(observed, expected)


def test_headless_front_physically_contains_no_prediction_or_fusion_modules() -> None:
    model = MSHNetD0Backbone(3)
    names = set(dict(model.named_modules()))
    assert not {"output_0", "output_1", "output_2", "output_3", "final"}.intersection(
        names
    )
    assert all(not name.startswith("output_") for name in model.state_dict())
    assert all(not name.startswith("final") for name in model.state_dict())


def test_front_loader_fails_closed_on_missing_or_malformed_tensor() -> None:
    canonical_state = MSHNet(3).state_dict()
    model = MSHNetD0Backbone(3)
    missing = dict(canonical_state)
    missing.pop(model.front_state_keys[0])
    with pytest.raises(MSHNetD0BackboneError, match="lacks"):
        model.load_mshnet_front_state_dict(missing)

    malformed = dict(canonical_state)
    malformed[model.front_state_keys[0]] = torch.zeros(1)
    with pytest.raises(MSHNetD0BackboneError, match="incompatible"):
        model.load_mshnet_front_state_dict(malformed)


@pytest.mark.parametrize("shape", ((1, 3, 63, 64), (1, 3, 64, 79)))
def test_front_rejects_shapes_that_break_the_frozen_four_pooling_levels(
    shape: tuple[int, int, int, int],
) -> None:
    model = MSHNetD0Backbone(3)
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.zeros(shape))
