"""Strict compatibility helpers for legacy MSHNet state dictionaries.

DEA-lite historically registered its optional decidability head on every
MSHNet instance.  Canonical MSHNet does not use that head.  The helper below
removes only the four tensors created by that exact legacy implementation; it
rejects partial, renamed, or shape-incompatible variants instead of silently
weakening checkpoint validation.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch


LEGACY_DEA_LITE_HEAD_STATE_SPECS = OrderedDict(
    (
        ("decidability_head.0.weight", (8, 7, 3, 3)),
        ("decidability_head.0.bias", (8,)),
        ("decidability_head.2.weight", (1, 8, 1, 1)),
        ("decidability_head.2.bias", (1,)),
    )
)


def _copy_state_dict(state_dict: Mapping[str, torch.Tensor]):
    copied = OrderedDict(state_dict.items())
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        copied._metadata = metadata.copy()
    return copied


def strip_legacy_dea_lite_head(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Return a copy without the exact legacy DEA-lite head tensors.

    Both ordinary and ``nn.DataParallel`` state dictionaries are supported.
    A state dictionary without any legacy head key is returned unchanged as a
    copy.  If a head-like key is present, all four expected keys and their
    historical shapes must match exactly.
    """

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    if not all(isinstance(key, str) for key in state_dict):
        raise TypeError("state_dict keys must be strings")

    prefixes = ("", "module.")
    candidate_keys_by_prefix = {
        prefix: {
            key
            for key in state_dict
            if key.startswith(prefix + "decidability_head.")
        }
        for prefix in prefixes
    }
    present_prefixes = [
        prefix for prefix, keys in candidate_keys_by_prefix.items() if keys
    ]

    if not present_prefixes:
        return _copy_state_dict(state_dict)
    if len(present_prefixes) != 1:
        raise RuntimeError(
            "legacy DEA-lite head uses mixed prefixed and unprefixed keys"
        )

    prefix = present_prefixes[0]
    actual_keys = candidate_keys_by_prefix[prefix]
    expected_keys = {
        prefix + key for key in LEGACY_DEA_LITE_HEAD_STATE_SPECS
    }
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "malformed legacy DEA-lite head: missing=%s unexpected=%s"
            % (missing, unexpected)
        )

    for relative_key, expected_shape in LEGACY_DEA_LITE_HEAD_STATE_SPECS.items():
        key = prefix + relative_key
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise RuntimeError(
                "malformed legacy DEA-lite head: %s is not a tensor" % key
            )
        actual_shape = tuple(value.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                "malformed legacy DEA-lite head: %s has shape %s, expected %s"
                % (key, actual_shape, expected_shape)
            )

    filtered = _copy_state_dict(state_dict)
    for key in actual_keys:
        del filtered[key]
    metadata = getattr(filtered, "_metadata", None)
    if metadata is not None:
        module_name = prefix + "decidability_head"
        for key in list(metadata):
            if key == module_name or key.startswith(module_name + "."):
                del metadata[key]
    return filtered


__all__ = [
    "LEGACY_DEA_LITE_HEAD_STATE_SPECS",
    "strip_legacy_dea_lite_head",
]
