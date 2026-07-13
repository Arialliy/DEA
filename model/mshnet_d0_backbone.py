"""MSHNet's frozen input-to-d0 graph without any prediction heads.

The research contract freezes the canonical MSHNet representation through
``d0`` while requiring the later component-native model to *replace* the four
scalar side heads and terminal fusion.  This class makes that separation
physical: no ``output_0..3`` or ``final`` module is constructed, so a future
forward pass cannot silently execute a parallel pixel-prediction branch.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from model.MSHNet import ResNet


class MSHNetD0BackboneError(RuntimeError):
    """The supplied baseline state cannot reproduce the frozen d0 graph."""


class MSHNetD0Backbone(nn.Module):
    """Canonical MSHNet encoder/decoder through the full-resolution ``d0``."""

    output_channels = 16

    def __init__(self, input_channels: int = 3, block: type[nn.Module] = ResNet):
        super().__init__()
        if isinstance(input_channels, bool) or not isinstance(input_channels, int):
            raise TypeError("input_channels must be an integer")
        if input_channels < 1:
            raise ValueError("input_channels must be positive")

        channels = (16, 32, 64, 128, 256)
        blocks = (2, 2, 2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        self.conv_init = nn.Conv2d(input_channels, channels[0], 1, 1)
        self.encoder_0 = self._make_layer(channels[0], channels[0], block)
        self.encoder_1 = self._make_layer(
            channels[0], channels[1], block, blocks[0]
        )
        self.encoder_2 = self._make_layer(
            channels[1], channels[2], block, blocks[1]
        )
        self.encoder_3 = self._make_layer(
            channels[2], channels[3], block, blocks[2]
        )
        self.middle_layer = self._make_layer(
            channels[3], channels[4], block, blocks[3]
        )
        self.decoder_3 = self._make_layer(
            channels[3] + channels[4], channels[3], block, blocks[2]
        )
        self.decoder_2 = self._make_layer(
            channels[2] + channels[3], channels[2], block, blocks[1]
        )
        self.decoder_1 = self._make_layer(
            channels[1] + channels[2], channels[1], block, blocks[0]
        )
        self.decoder_0 = self._make_layer(
            channels[0] + channels[1], channels[0], block
        )

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        block: type[nn.Module],
        block_num: int = 1,
    ) -> nn.Sequential:
        layers = [block(in_channels, out_channels)]
        layers.extend(block(out_channels, out_channels) for _ in range(block_num - 1))
        return nn.Sequential(*layers)

    @property
    def front_state_keys(self) -> tuple[str, ...]:
        """Return the exact state keys required from a canonical MSHNet."""

        return tuple(self.state_dict())

    def load_mshnet_front_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Load only input-to-d0 tensors from an exact, unprefixed MSHNet state."""

        if not isinstance(state_dict, Mapping) or not state_dict:
            raise MSHNetD0BackboneError("MSHNet state_dict must be a non-empty mapping")
        required = self.state_dict()
        missing = sorted(set(required).difference(state_dict))
        if missing:
            raise MSHNetD0BackboneError(
                f"MSHNet state_dict lacks {len(missing)} front tensors, e.g. {missing[:3]}"
            )
        selected: dict[str, torch.Tensor] = {}
        for name, reference in required.items():
            value = state_dict[name]
            if not torch.is_tensor(value) or value.shape != reference.shape:
                raise MSHNetD0BackboneError(
                    f"MSHNet front tensor {name!r} has an incompatible shape/type"
                )
            selected[name] = value
        self.load_state_dict(selected, strict=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape [batch, channels, height, width]")
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError("image height and width must be divisible by 16")
        e0 = self.encoder_0(self.conv_init(image))
        e1 = self.encoder_1(self.pool(e0))
        e2 = self.encoder_2(self.pool(e1))
        e3 = self.encoder_3(self.pool(e2))
        middle = self.middle_layer(self.pool(e3))
        d3 = self.decoder_3(torch.cat((e3, self.up(middle)), dim=1))
        d2 = self.decoder_2(torch.cat((e2, self.up(d3)), dim=1))
        d1 = self.decoder_1(torch.cat((e1, self.up(d2)), dim=1))
        d0 = self.decoder_0(torch.cat((e0, self.up(d1)), dim=1))
        expected = (image.shape[0], self.output_channels, image.shape[2], image.shape[3])
        if tuple(d0.shape) != expected or not bool(torch.isfinite(d0).all()):
            raise MSHNetD0BackboneError("frozen MSHNet front produced an invalid d0")
        return d0


__all__ = ["MSHNetD0Backbone", "MSHNetD0BackboneError"]
