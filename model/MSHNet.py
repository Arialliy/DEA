import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class MSHNet(nn.Module):
    def __init__(
        self,
        input_channels,
        block=ResNet,
        *,
        enable_dea_lite=False,
    ):
        super().__init__()
        if not isinstance(enable_dea_lite, bool):
            raise TypeError("enable_dea_lite must be a bool")
        self.enable_dea_lite = enable_dea_lite
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])
     
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])
        
        self.decoder_3 = self._make_layer(param_channels[3]+param_channels[4], param_channels[3], block, param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2]+param_channels[3], param_channels[2], block, param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1]+param_channels[2], param_channels[1], block, param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0]+param_channels[1], param_channels[0], block)

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)
        if self.enable_dea_lite:
            self.decidability_head = nn.Sequential(
                nn.Conv2d(7, 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=1),
            )


    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []        
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def build_dea_lite_outputs(self, scale_logits, z_full, detach_evidence=False):
        if not self.enable_dea_lite or not hasattr(self, "decidability_head"):
            raise RuntimeError(
                "DEA-lite outputs require MSHNet(enable_dea_lite=True)"
            )

        if detach_evidence:
            cf_scale_logits = scale_logits.detach()
            dec_scale_logits = scale_logits.detach()
            dec_z_full = z_full.detach()
        else:
            cf_scale_logits = scale_logits
            dec_scale_logits = scale_logits
            dec_z_full = z_full

        if self.final.bias is None:
            z_empty = torch.zeros_like(z_full)
            bias = None
        else:
            bias = self.final.bias.view(1, 1, 1, 1)
            z_empty = bias.expand_as(z_full)

        only_weight = self.final.weight.permute(1, 0, 2, 3).contiguous()
        z_only = F.conv2d(
            cf_scale_logits,
            only_weight,
            bias=None,
            stride=self.final.stride,
            padding=self.final.padding,
            dilation=self.final.dilation,
            groups=cf_scale_logits.shape[1],
        )
        if bias is not None:
            z_only = z_only + bias

        z_only_max = z_only.max(dim=1, keepdim=True)[0]
        z_only_var = z_only.var(dim=1, keepdim=True, unbiased=False)

        d_input = torch.cat([
            dec_z_full,
            z_only_max.detach() if detach_evidence else z_only_max,
            z_only_var.detach() if detach_evidence else z_only_var,
            dec_scale_logits,
        ], dim=1)

        d_logit = self.decidability_head(d_input)

        return {
            "scale_logits": scale_logits,
            "z_empty": z_empty,
            "z_only": z_only,
            "z_only_max": z_only_max,
            "z_only_var": z_only_var,
            "decidability_logit": d_logit,
        }

    def forward(self, x, warm_flag, return_dea=False, dea_detach_evidence=False):
        if return_dea and not self.enable_dea_lite:
            raise RuntimeError(
                "return_dea=True requires MSHNet(enable_dea_lite=True)"
            )

        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        
        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)

            s0 = mask0
            s1 = self.up(mask1)
            s2 = self.up_4(mask2)
            s3 = self.up_8(mask3)

            scale_logits = torch.cat([s0, s1, s2, s3], dim=1)
            z_full = self.final(scale_logits)

            if return_dea:
                dea_out = self.build_dea_lite_outputs(
                    scale_logits,
                    z_full,
                    detach_evidence=dea_detach_evidence,
                )
                return [mask0, mask1, mask2, mask3], z_full, dea_out

            return [mask0, mask1, mask2, mask3], z_full
    
        else:
            output = self.output_0(x_d0)
            return [], output

       
    
