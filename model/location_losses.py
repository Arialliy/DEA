from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def legacy_location_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Preserve the original ``model.loss.LLoss`` computation exactly."""
    h = pred.shape[2]
    w = pred.shape[3]
    x_index = torch.arange(0, w, 1, device=pred.device, dtype=pred.dtype).view(1, 1, 1, w) / w
    y_index = torch.arange(0, h, 1, device=pred.device, dtype=pred.dtype).view(1, 1, h, 1) / h
    smooth = 1e-8

    pred_centerx = (x_index * pred).mean(dim=(1, 2, 3))
    pred_centery = (y_index * pred).mean(dim=(1, 2, 3))
    target_centerx = (x_index * target).mean(dim=(1, 2, 3))
    target_centery = (y_index * target).mean(dim=(1, 2, 3))

    angle_loss = (4 / (torch.pi ** 2)) * torch.square(
        torch.atan(pred_centery / (pred_centerx + smooth))
        - torch.atan(target_centery / (target_centerx + smooth))
    )

    pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
    target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)
    length_loss = torch.minimum(pred_length, target_length) / (
        torch.maximum(pred_length, target_length) + smooth
    )

    return (1 - length_loss + angle_loss).mean()


def normalized_xy_grid(reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return broadcastable x/y grids in [0, 1), as used by legacy LLoss."""
    if reference.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got {tuple(reference.shape)}")
    h, w = reference.shape[-2:]
    x = torch.arange(w, device=reference.device, dtype=reference.dtype)
    y = torch.arange(h, device=reference.device, dtype=reference.dtype)
    x = (x / float(w)).view(1, 1, 1, w)
    y = (y / float(h)).view(1, 1, h, 1)
    return x, y


def probability_mass(prob: torch.Tensor) -> torch.Tensor:
    """Return one total probability mass per BCHW sample."""
    return prob.sum(dim=(1, 2, 3))


def mass_normalized_centroid(
    prob: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Bx2 centroids [x, y] and the unmodified B-vector mass."""
    x, y = normalized_xy_grid(prob)
    mass = probability_mass(prob)
    safe_mass = mass.clamp_min(eps)
    cx = (prob * x).sum(dim=(1, 2, 3)) / safe_mass
    cy = (prob * y).sum(dim=(1, 2, 3)) / safe_mass
    return torch.stack((cx, cy), dim=-1), mass


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Average valid entries, returning differentiable zero if none are valid."""
    valid_f = valid.to(dtype=values.dtype)
    return (values * valid_f).sum() / valid_f.sum().clamp_min(1.0)


class GlobalMassLocationLoss(nn.Module):
    """Mass-normalized global location diagnostic.

    ``metric='polar'`` isolates mass normalization while retaining the legacy
    polar form. ``metric='cartesian'`` compares x/y displacements with
    Smooth-L1 and is invariant to a common translation that stays in-frame.
    Empty target samples are excluded because they have no defined location.
    """

    def __init__(
        self,
        metric: str = "cartesian",
        eps: float = 1e-6,
        beta: float = 0.02,
    ) -> None:
        super().__init__()
        if metric not in {"polar", "cartesian"}:
            raise ValueError(f"unknown metric: {metric}")
        self.metric = metric
        self.eps = float(eps)
        self.beta = float(beta)

    def forward(
        self,
        pred_prob: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_center, pred_mass = mass_normalized_centroid(pred_prob, self.eps)
        target_center, target_mass = mass_normalized_centroid(target, self.eps)
        valid = target_mass > 0

        if self.metric == "cartesian":
            per_axis = F.smooth_l1_loss(
                pred_center,
                target_center,
                reduction="none",
                beta=self.beta,
            )
            per_sample = per_axis.sum(dim=-1)
        else:
            px, py = pred_center.unbind(dim=-1)
            tx, ty = target_center.unbind(dim=-1)
            smooth = self.eps
            angle = (4.0 / (torch.pi ** 2)) * torch.square(
                torch.atan(py / (px + smooth))
                - torch.atan(ty / (tx + smooth))
            )
            plen = torch.sqrt(px.square() + py.square() + smooth)
            tlen = torch.sqrt(tx.square() + ty.square() + smooth)
            length_ratio = torch.minimum(plen, tlen) / (
                torch.maximum(plen, tlen) + smooth
            )
            per_sample = 1.0 - length_ratio + angle

        loss = masked_mean(per_sample, valid)
        valid_f = valid.to(dtype=pred_center.dtype)
        logs = {
            "location_valid_ratio": valid.float().mean().detach(),
            "pred_mass_mean": pred_mass.mean().detach(),
            "target_mass_mean": target_mass.mean().detach(),
            "global_centroid_l1": (
                (pred_center - target_center).abs().sum(dim=-1) * valid_f
            ).sum().detach()
            / valid_f.sum().clamp_min(1.0),
        }
        return loss, logs
