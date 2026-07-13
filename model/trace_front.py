"""Strictly loaded and immutable MSHNet input-to-d0 evidence front."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from model.mshnet_d0_backbone import MSHNetD0Backbone
from utils.trace_provenance import (
    BaselineFrontProvenance,
    TraceProvenanceError,
    load_clean_mshnet_checkpoint,
    state_dict_sha256,
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenMSHNetD0(nn.Module):
    """A physically headless, checkpoint-authenticated, immutable front."""

    output_channels = MSHNetD0Backbone.output_channels

    def __init__(
        self,
        baseline_checkpoint: str | Path,
        *,
        input_channels: int = 3,
        expected_dataset: str | None = None,
        expected_seed: int | None = None,
        expected_train_split_sha256: str | None = None,
        expected_val_split_sha256: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone = MSHNetD0Backbone(input_channels=input_channels)
        state, provenance, _ = load_clean_mshnet_checkpoint(
            baseline_checkpoint,
            front_state_keys=self.backbone.front_state_keys,
            expected_dataset=expected_dataset,
            expected_seed=expected_seed,
            expected_train_split_sha256=expected_train_split_sha256,
            expected_val_split_sha256=expected_val_split_sha256,
        )
        self.backbone.load_mshnet_front_state_dict(state)
        self.provenance: BaselineFrontProvenance = provenance
        self._freeze()
        self._loaded_state_sha256 = state_dict_sha256(
            self.backbone.state_dict(), keys=self.backbone.front_state_keys
        )
        if self._loaded_state_sha256 != provenance.front_tensor_sha256:
            raise TraceProvenanceError(
                "loaded front tensors differ from authenticated checkpoint tensors"
            )

    def _freeze(self) -> None:
        super().train(False)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenMSHNetD0":
        # The wrapper itself may be nested in a training model, but its BatchNorm
        # buffers and stochastic semantics are permanently held in eval mode.
        super().train(False)
        self.backbone.eval()
        return self

    @property
    def loaded_state_sha256(self) -> str:
        return self._loaded_state_sha256

    def assert_integrity(self) -> str:
        if self.training or self.backbone.training:
            raise TraceProvenanceError("frozen MSHNet front entered training mode")
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise TraceProvenanceError("frozen MSHNet front contains a trainable parameter")
        current = state_dict_sha256(
            self.backbone.state_dict(), keys=self.backbone.front_state_keys
        )
        if current != self._loaded_state_sha256:
            raise TraceProvenanceError("frozen MSHNet front parameters or BN buffers changed")
        return current

    @torch.no_grad()
    def anchor(self, image: torch.Tensor) -> dict[str, str | list[int]]:
        self.assert_integrity()
        d0 = self.backbone(image)
        return {
            "input_sha256": tensor_sha256(image),
            "d0_sha256": tensor_sha256(d0),
            "input_shape": list(image.shape),
            "d0_shape": list(d0.shape),
            "front_state_sha256": self._loaded_state_sha256,
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            return self.backbone(image)


__all__ = ["FrozenMSHNetD0", "tensor_sha256"]
