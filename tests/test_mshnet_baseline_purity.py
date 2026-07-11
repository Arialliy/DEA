from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from model.MSHNet import MSHNet
from model.mshnet_checkpoint import (
    LEGACY_DEA_LITE_HEAD_STATE_SPECS,
    strip_legacy_dea_lite_head,
)


CANONICAL_PARAMETER_COUNT = 4_065_513
DEA_LITE_PARAMETER_COUNT = 521


def _make_pure_and_dea_lite(seed: int = 1701):
    torch.manual_seed(seed)
    pure = MSHNet(3).eval()
    torch.manual_seed(seed)
    dea_lite = MSHNet(3, enable_dea_lite=True).eval()
    return pure, dea_lite


def test_default_mshnet_is_parameter_pure_and_dea_lite_is_explicit() -> None:
    pure, dea_lite = _make_pure_and_dea_lite()

    assert pure.enable_dea_lite is False
    assert not hasattr(pure, "decidability_head")
    assert not any(
        key.startswith("decidability_head.") for key in pure.state_dict()
    )
    assert sum(parameter.numel() for parameter in pure.parameters()) == (
        CANONICAL_PARAMETER_COUNT
    )

    assert dea_lite.enable_dea_lite is True
    assert hasattr(dea_lite, "decidability_head")
    assert sum(parameter.numel() for parameter in dea_lite.parameters()) == (
        CANONICAL_PARAMETER_COUNT + DEA_LITE_PARAMETER_COUNT
    )
    assert sum(
        parameter.numel()
        for parameter in dea_lite.decidability_head.parameters()
    ) == DEA_LITE_PARAMETER_COUNT


def test_pure_mshnet_dea_requests_fail_closed() -> None:
    model = MSHNet(3).eval()
    x = torch.zeros(1, 3, 16, 16)
    scale_logits = torch.zeros(1, 4, 16, 16)
    z_full = torch.zeros(1, 1, 16, 16)

    with pytest.raises(RuntimeError, match="enable_dea_lite=True"):
        model(x, True, return_dea=True)
    with pytest.raises(RuntimeError, match="enable_dea_lite=True"):
        model.build_dea_lite_outputs(scale_logits, z_full)


def test_opted_in_dea_lite_head_builds_expected_outputs() -> None:
    model = MSHNet(3, enable_dea_lite=True).eval()
    scale_logits = torch.randn(1, 4, 8, 8)
    z_full = model.final(scale_logits)

    output = model.build_dea_lite_outputs(scale_logits, z_full)

    assert set(output) == {
        "scale_logits",
        "z_empty",
        "z_only",
        "z_only_max",
        "z_only_var",
        "decidability_logit",
    }
    assert output["decidability_logit"].shape == (1, 1, 8, 8)


def test_pure_and_opted_in_models_have_bitwise_identical_canonical_state_and_outputs() -> None:
    pure, dea_lite = _make_pure_and_dea_lite(seed=1703)
    pure_state = pure.state_dict()
    dea_state = dea_lite.state_dict()
    common_keys = set(pure_state).intersection(dea_state)

    assert common_keys == set(pure_state)
    assert set(dea_state) - common_keys == set(
        LEGACY_DEA_LITE_HEAD_STATE_SPECS
    )
    for key in common_keys:
        assert torch.equal(pure_state[key], dea_state[key]), key

    generator = torch.Generator().manual_seed(1705)
    x = torch.randn(1, 3, 16, 16, generator=generator)
    with torch.no_grad():
        pure_cold = pure(x, False)
        dea_cold = dea_lite(x, False)
        pure_warm = pure(x, True)
        dea_warm = dea_lite(x, True)

    assert pure_cold[0] == dea_cold[0] == []
    assert torch.equal(pure_cold[1], dea_cold[1])
    for pure_tensor, dea_tensor in zip(pure_warm[0], dea_warm[0]):
        assert torch.equal(pure_tensor, dea_tensor)
    assert torch.equal(pure_warm[1], dea_warm[1])


def test_exact_legacy_head_filter_loads_strictly_into_pure_mshnet() -> None:
    pure, dea_lite = _make_pure_and_dea_lite(seed=1707)
    legacy_state = dea_lite.state_dict()

    filtered = strip_legacy_dea_lite_head(legacy_state)

    assert set(filtered) == set(pure.state_dict())
    assert all(key in legacy_state for key in LEGACY_DEA_LITE_HEAD_STATE_SPECS)
    assert not any(
        "decidability_head" in key for key in filtered._metadata
    )
    pure.load_state_dict(filtered, strict=True)
    for key, value in pure.state_dict().items():
        assert torch.equal(value, legacy_state[key]), key


def test_legacy_head_filter_supports_data_parallel_prefix() -> None:
    _, dea_lite = _make_pure_and_dea_lite(seed=1709)
    prefixed = OrderedDict(
        ("module." + key, value) for key, value in dea_lite.state_dict().items()
    )

    filtered = strip_legacy_dea_lite_head(prefixed)

    assert not any("decidability_head." in key for key in filtered)
    assert all(key.startswith("module.") for key in filtered)


def test_legacy_head_filter_rejects_fake_or_partial_keys() -> None:
    _, dea_lite = _make_pure_and_dea_lite(seed=1711)
    legacy_state = dea_lite.state_dict()

    fake = OrderedDict(legacy_state.items())
    fake["decidability_head.fake"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        strip_legacy_dea_lite_head(fake)

    partial = OrderedDict(legacy_state.items())
    del partial["decidability_head.2.bias"]
    with pytest.raises(RuntimeError, match="missing"):
        strip_legacy_dea_lite_head(partial)


def test_legacy_head_filter_rejects_wrong_shape() -> None:
    _, dea_lite = _make_pure_and_dea_lite(seed=1713)
    wrong_shape = OrderedDict(dea_lite.state_dict().items())
    wrong_shape["decidability_head.0.weight"] = torch.zeros(8, 7, 1, 1)

    with pytest.raises(RuntimeError, match="has shape"):
        strip_legacy_dea_lite_head(wrong_shape)


def test_legacy_head_filter_is_idempotent_for_pure_state() -> None:
    pure = MSHNet(3)
    state = pure.state_dict()

    filtered = strip_legacy_dea_lite_head(state)

    assert filtered is not state
    assert list(filtered) == list(state)
    for key in state:
        assert filtered[key] is state[key]
