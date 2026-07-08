from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FullDEAHead(nn.Module):
    """Minimal structural Full DEA head.

    This module intentionally exposes the tensors required by the predeclare
    protocol: target evidence, clutter evidence, counterfactual feature, real
    prediction, counterfactual prediction, and final evidence-calibrated output.
    """

    def __init__(self, in_channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = hidden_channels or max(16, in_channels)

        self.shared = nn.Sequential(
            ConvBlock(in_channels, hidden),
            ConvBlock(hidden, hidden),
        )
        self.evidence_head = nn.Conv2d(hidden, 2, kernel_size=1)
        self.real_head = nn.Sequential(
            ConvBlock(in_channels + 2, hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.counterfactual_head = nn.Sequential(
            ConvBlock(in_channels, hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.gate_head = nn.Sequential(
            ConvBlock(3, hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, decoder_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(decoder_feature)
        evidence_logits = self.evidence_head(shared)
        target_evidence_logit, clutter_evidence_logit = torch.chunk(
            evidence_logits,
            chunks=2,
            dim=1,
        )
        target_evidence = torch.sigmoid(target_evidence_logit)
        clutter_evidence = torch.sigmoid(clutter_evidence_logit)

        cf_gate = torch.sigmoid(clutter_evidence_logit - target_evidence_logit)
        counterfactual_feature = decoder_feature * cf_gate

        real_input = torch.cat(
            [decoder_feature, target_evidence, clutter_evidence],
            dim=1,
        )
        y_real = self.real_head(real_input)
        y_cf = self.counterfactual_head(counterfactual_feature)

        gate_input = torch.cat([y_real, target_evidence, clutter_evidence], dim=1)
        evidence_gate = torch.sigmoid(self.gate_head(gate_input))
        y_final = evidence_gate * y_real + (1.0 - evidence_gate) * y_cf

        return {
            "decoder_feature": decoder_feature,
            "target_evidence_logit": target_evidence_logit,
            "clutter_evidence_logit": clutter_evidence_logit,
            "target_evidence": target_evidence,
            "clutter_evidence": clutter_evidence,
            "counterfactual_feature": counterfactual_feature,
            "counterfactual_gate": cf_gate,
            "y_real": y_real,
            "y_cf": y_cf,
            "evidence_gate": evidence_gate,
            "y_final": y_final,
        }
