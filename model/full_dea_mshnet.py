from __future__ import annotations

import torch

from model.MSHNet import MSHNet, ResNet
from model.full_dea_head import FullDEAHead


class FullDEAMSHNet(MSHNet):
    """MSHNet wrapper with a prototype Full DEA head.

    The base encoder and decoder are inherited unchanged. The Full DEA head is
    inserted after the final decoder feature x_d0 and before the final output.
    """

    def __init__(self, input_channels: int, block=ResNet):
        super().__init__(input_channels, block=block)
        self.full_dea_head = FullDEAHead(in_channels=16)

    def forward(
        self,
        x: torch.Tensor,
        warm_flag: bool = True,
        return_full_dea: bool = False,
    ):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        if return_full_dea:
            full_dea_out = self.full_dea_head(x_d0)
            if warm_flag:
                scale_logits = self._build_scale_logits(x_d0, x_d1, x_d2, x_d3)
                return scale_logits, full_dea_out["y_final"], full_dea_out
            return [], full_dea_out["y_final"], full_dea_out

        if warm_flag:
            scale_logits = self._build_scale_logits(x_d0, x_d1, x_d2, x_d3)
            z_full = self.final(torch.cat(scale_logits, dim=1))
            return scale_logits, z_full

        output = self.output_0(x_d0)
        return [], output

    def _build_scale_logits(
        self,
        x_d0: torch.Tensor,
        x_d1: torch.Tensor,
        x_d2: torch.Tensor,
        x_d3: torch.Tensor,
    ) -> list[torch.Tensor]:
        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)
        return [
            mask0,
            self.up(mask1),
            self.up_4(mask2),
            self.up_8(mask3),
        ]
