from __future__ import annotations

import torch

from model.MSHNet import MSHNet, ResNet
from model.full_dea_head import (
    FullDEAHeadV2,
    FullDEAHeadV3,
    FullDEAHeadV4,
    FullDEAHeadV5,
    FullDEAHeadV6,
)


class FullDEAMSHNet(MSHNet):
    """MSHNet with Full DEA at the multi-scale evidence fusion point."""

    def __init__(self, input_channels: int, block=ResNet, full_dea_version: str = "v3"):
        super().__init__(input_channels, block=block)
        if full_dea_version == "v2":
            self.full_dea_head = FullDEAHeadV2(hidden_channels=32)
        elif full_dea_version == "v3":
            self.full_dea_head = FullDEAHeadV3(hidden_channels=32)
        elif full_dea_version == "v4":
            self.full_dea_head = FullDEAHeadV4(hidden_channels=32)
        elif full_dea_version == "v5":
            self.full_dea_head = FullDEAHeadV5(hidden_channels=32)
        elif full_dea_version == "v6":
            self.full_dea_head = FullDEAHeadV6(hidden_channels=32)
        else:
            raise ValueError("Unsupported Full DEA version: %s" % full_dea_version)
        self.full_dea_version = full_dea_version

    def _forward_features(self, x: torch.Tensor):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], dim=1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], dim=1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], dim=1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], dim=1))

        return x_d0, x_d1, x_d2, x_d3

    def _build_masks_and_scale_logits(self, x_d0, x_d1, x_d2, x_d3):
        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)

        masks = [mask0, mask1, mask2, mask3]
        scale_logits_full = torch.cat(
            [
                mask0,
                self.up(mask1),
                self.up_4(mask2),
                self.up_8(mask3),
            ],
            dim=1,
        )
        return masks, scale_logits_full

    def forward(
        self,
        x: torch.Tensor,
        warm_flag: bool = True,
        return_full_dea: bool = False,
        return_dict: bool = False,
    ):
        x_d0, x_d1, x_d2, x_d3 = self._forward_features(x)

        if not warm_flag:
            pred = self.output_0(x_d0)
            if return_dict:
                return {
                    "masks": [],
                    "pred": pred,
                    "z_base": pred,
                    "scale_logits_full": None,
                    "full_dea": None,
                }
            return [], pred

        masks, scale_logits_full = self._build_masks_and_scale_logits(
            x_d0,
            x_d1,
            x_d2,
            x_d3,
        )
        z_base = self.final(scale_logits_full)
        head_kwargs = {
            "x_d0": x_d0,
            "x_d1": x_d1,
            "x_d2": x_d2,
            "x_d3": x_d3,
            "scale_logits_full": scale_logits_full,
            "z_base": z_base,
        }
        if self.full_dea_version in ("v3", "v4", "v5", "v6"):
            head_kwargs.update(
                {
                    "fusion_weight": self.final.weight,
                    "fusion_bias": self.final.bias,
                    "fusion_stride": self.final.stride,
                    "fusion_padding": self.final.padding,
                    "fusion_dilation": self.final.dilation,
                }
            )
        full_dea_out = self.full_dea_head(**head_kwargs)
        pred = full_dea_out["z_final"]

        if return_dict:
            return {
                "masks": masks,
                "pred": pred,
                "z_base": z_base,
                "scale_logits_full": scale_logits_full,
                "full_dea": full_dea_out,
            }

        if return_full_dea:
            return masks, pred, full_dea_out

        return masks, pred
