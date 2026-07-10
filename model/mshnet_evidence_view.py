"""Read-only evidence view for a trained MSHNet.

The helper calls MSHNet's original forward path exactly once, captures the
four decoder outputs with temporary hooks, and reuses the repository's exact
linear-fusion decomposer.  It never replaces the direct ``model.final``
prediction with a reconstructed sum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from model.dea_evidence import ExactScaleContributionDecomposer


_DECODER_NAMES = ("decoder_0", "decoder_1", "decoder_2", "decoder_3")


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


def _validate_mshnet_interface(model: torch.nn.Module) -> None:
    required = (*_DECODER_NAMES, "up", "up_4", "up_8", "final")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(
            "model does not expose the required MSHNet interface: %s"
            % ", ".join(missing)
        )


def forward_mshnet_evidence(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    detach: bool = False,
) -> dict[str, Any]:
    """Run the unmodified MSHNet path and expose diagnostic evidence.

    Args:
        model: An MSHNet-compatible module, not a decoder-replacement model.
        x: Input tensor accepted by ``model``.
        detach: Detach every returned tensor for offline manifest generation.

    Returns:
        A dictionary containing the original direct prediction, native side
        logits, full-resolution scale logits, exact bias-free contributions,
        leave-one-scale-out interventions, and decoder features ordered
        ``(d0, d1, d2, d3)``.
    """

    _validate_mshnet_interface(model)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    for name in _DECODER_NAMES:
        module = getattr(model, name)

        def capture_output(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            key: str = name,
        ) -> None:
            captured[key] = output

        handles.append(module.register_forward_hook(capture_output))

    try:
        masks, z_base = model(x, True)
    finally:
        for handle in handles:
            handle.remove()

    if not isinstance(masks, Sequence) or len(masks) != 4:
        raise RuntimeError(
            "MSHNet warm forward must return four native side logits"
        )
    missing_features = [name for name in _DECODER_NAMES if name not in captured]
    if missing_features:
        raise RuntimeError(
            "decoder hooks did not observe: %s" % ", ".join(missing_features)
        )

    scale_logits = torch.cat(
        [
            masks[0],
            model.up(masks[1]),
            model.up_4(masks[2]),
            model.up_8(masks[3]),
        ],
        dim=1,
    )

    decomposition = ExactScaleContributionDecomposer(scale_channels=4)(
        scale_logits=scale_logits,
        z_base=z_base,
        fusion_weight=model.final.weight,
        fusion_bias=model.final.bias,
        stride=model.final.stride,
        padding=model.final.padding,
        dilation=model.final.dilation,
    )

    if model.final.bias is None:
        fusion_bias = z_base.new_zeros((1, 1, 1, 1))
    else:
        fusion_bias = model.final.bias.view(1, 1, 1, 1)

    output: dict[str, Any] = {
        "pred": z_base,
        "z_base": z_base,
        "masks": tuple(masks),
        "scale_logits": scale_logits,
        "contributions": decomposition["scale_contributions"],
        "fusion_bias": fusion_bias,
        "z_reconstructed": decomposition["z_reconstructed"],
        "z_without_scale": decomposition["z_without_scale"],
        "z_only_scale": decomposition["z_only_scale"],
        "scale_statistics": decomposition["scale_statistics"],
        "decoder_features": tuple(captured[name] for name in _DECODER_NAMES),
    }
    return _detach_tree(output) if detach else output


__all__ = ["forward_mshnet_evidence"]
