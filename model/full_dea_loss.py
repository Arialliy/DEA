from __future__ import annotations

import torch
import torch.nn.functional as F

from model.loss import SoftIoULoss, build_safe_bg


def full_dea_loss(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    lambda_cf: float = 0.1,
    lambda_bg: float = 0.05,
    lambda_sep: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Finite prototype loss for structural Full DEA tests.

    The loss uses y_final for segmentation and y_cf for the counterfactual path,
    so shape tests can verify that the counterfactual branch participates in
    optimization before any long training is attempted.
    """

    y_final = full_dea_out["y_final"]
    y_cf = full_dea_out["y_cf"]
    target_evidence = full_dea_out["target_evidence"]
    clutter_evidence = full_dea_out["clutter_evidence"]

    seg_loss = SoftIoULoss(y_final, target)

    cf_loss_map = F.binary_cross_entropy_with_logits(
        y_cf,
        torch.zeros_like(y_cf),
        reduction="none",
    )
    cf_loss = (cf_loss_map * target.float()).sum() / (target.float().sum() + 1e-6)

    safe_bg = build_safe_bg(target)
    bg_loss_map = F.binary_cross_entropy_with_logits(
        y_final,
        torch.zeros_like(y_final),
        reduction="none",
    )
    bg_loss = (bg_loss_map * safe_bg).sum() / (safe_bg.sum() + 1e-6)

    sep_loss = (target_evidence * clutter_evidence).mean()

    total = seg_loss + lambda_cf * cf_loss + lambda_bg * bg_loss + lambda_sep * sep_loss
    log_vars = {
        "loss_seg": seg_loss.detach(),
        "loss_cf": cf_loss.detach(),
        "loss_bg": bg_loss.detach(),
        "loss_sep": sep_loss.detach(),
        "target_evidence_mean": target_evidence.detach().mean(),
        "clutter_evidence_mean": clutter_evidence.detach().mean(),
    }
    return total, log_vars
