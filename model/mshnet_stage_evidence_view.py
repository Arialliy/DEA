"""Read-only stage evidence for the unmodified MSHNet computation graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from model.mshnet_evidence_view import forward_mshnet_evidence


_OUTPUT_STAGES = (
    ("stem", "conv_init"),
    ("e0", "encoder_0"),
    ("e1", "encoder_1"),
    ("e2", "encoder_2"),
    ("e3", "encoder_3"),
    ("m", "middle_layer"),
    ("d3", "decoder_3"),
    ("d2", "decoder_2"),
    ("d1", "decoder_1"),
    ("d0", "decoder_0"),
)

_INPUT_STAGES = (
    ("p0", "encoder_1"),
    ("p1", "encoder_2"),
    ("p2", "encoder_3"),
    ("p3", "middle_layer"),
    ("j3", "decoder_3"),
    ("j2", "decoder_2"),
    ("j1", "decoder_1"),
    ("j0", "decoder_0"),
)

_PATH_ORDER = (
    "input",
    "stem",
    "e0",
    "p0",
    "e1",
    "p1",
    "e2",
    "p2",
    "e3",
    "p3",
    "m",
    "j3",
    "d3",
    "j2",
    "d2",
    "j1",
    "d1",
    "j0",
    "d0",
)


def _detach_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    return value


def _require_tensor(value: Any, *, stage: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 4:
        raise RuntimeError("stage %s did not expose a 4-D tensor" % stage)
    return value


def forward_mshnet_stage_evidence(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    detach: bool = False,
) -> dict[str, Any]:
    """Run the native warm path once and expose its actual DAG tensors.

    Ordinary hooks capture module outputs while pre-hooks capture the exact
    pooled encoder inputs (``p*``) and skip-concatenated decoder inputs
    (``j*``).  No stage is recomputed and the direct final prediction from the
    model remains the authoritative output.
    """

    if not torch.is_tensor(x) or x.ndim != 4:
        raise ValueError("x must be a 4-D tensor")
    required_modules = {
        module_name
        for _, module_name in (*_OUTPUT_STAGES, *_INPUT_STAGES)
    }
    missing = sorted(name for name in required_modules if not hasattr(model, name))
    if missing:
        raise TypeError(
            "model does not expose the required MSHNet modules: %s"
            % ", ".join(missing)
        )

    captured: dict[str, torch.Tensor] = {"input": x}
    handles = []
    for stage, module_name in _OUTPUT_STAGES:
        module = getattr(model, module_name)

        def capture_output(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            key: str = stage,
        ) -> None:
            captured[key] = _require_tensor(output, stage=key)

        handles.append(module.register_forward_hook(capture_output))

    for stage, module_name in _INPUT_STAGES:
        module = getattr(model, module_name)

        def capture_input(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            *,
            key: str = stage,
        ) -> None:
            if len(inputs) != 1:
                raise RuntimeError(
                    "stage %s expected exactly one module input" % key
                )
            captured[key] = _require_tensor(inputs[0], stage=key)

        handles.append(module.register_forward_pre_hook(capture_input))

    try:
        base = forward_mshnet_evidence(model, x, detach=False)
    finally:
        for handle in handles:
            handle.remove()

    missing_stages = [stage for stage in _PATH_ORDER if stage not in captured]
    if missing_stages:
        raise RuntimeError(
            "MSHNet stage hooks did not observe: %s"
            % ", ".join(missing_stages)
        )

    path = {stage: captured[stage] for stage in _PATH_ORDER}
    native_sides = {
        "mask%d" % index: _require_tensor(mask, stage="mask%d" % index)
        for index, mask in enumerate(base["masks"])
    }
    full_sides = {
        "s%d" % index: base["scale_logits"][:, index : index + 1]
        for index in range(4)
    }
    contributions = {
        "c%d" % index: base["contributions"][:, index : index + 1]
        for index in range(4)
    }
    output: dict[str, Any] = {
        "pred": base["pred"],
        "path": path,
        "native_sides": native_sides,
        "full_sides": full_sides,
        "contributions": contributions,
        "side_heads": {
            "mask%d" % index: {
                "weight": getattr(model, "output_%d" % index).weight,
                "bias": getattr(model, "output_%d" % index).bias,
            }
            for index in range(4)
        },
        "final_head": {
            "weight": model.final.weight,
            "bias": model.final.bias,
            "stride": model.final.stride,
            "padding": model.final.padding,
            "dilation": model.final.dilation,
        },
        "fusion_bias": base["fusion_bias"],
        "z_reconstructed": base["z_reconstructed"],
    }
    return _detach_tree(output) if detach else output


__all__ = ["forward_mshnet_stage_evidence"]
