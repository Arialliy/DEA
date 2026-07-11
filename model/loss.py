import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
from skimage import measure
import math

from model.location_losses import GlobalMassLocationLoss, legacy_location_loss


def SoftIoULoss( pred, target):
        pred = torch.sigmoid(pred)
  
        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        loss = (intersection_sum + smooth) / \
                    (pred_sum + target_sum - intersection_sum + smooth)
    
        loss = 1 - loss.mean()

        return loss

def Dice( pred, target,warm_epoch=1, epoch=1, layer=0):
        pred = torch.sigmoid(pred)
  
        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))

        loss = (2*intersection_sum + smooth) / \
            (pred_sum + target_sum + intersection_sum + smooth)

        loss = 1 - loss.mean()

        return loss

class SLSIoULoss(nn.Module):
    VALID_LOCATION_MODES = (
        "legacy",
        "none",
        "mass_polar",
        "mass_cartesian",
    )

    def __init__(
        self,
        location_mode="legacy",
        lambda_location=1.0,
        return_breakdown=False,
    ):
        super(SLSIoULoss, self).__init__()
        if location_mode not in self.VALID_LOCATION_MODES:
            raise ValueError("unknown location mode: %s" % location_mode)
        if not math.isfinite(float(lambda_location)) or float(lambda_location) < 0.0:
            raise ValueError("lambda_location must be finite and non-negative")
        self.location_mode = location_mode
        self.lambda_location = float(lambda_location)
        self.return_breakdown = bool(return_breakdown)
        self.mass_polar = GlobalMassLocationLoss(metric="polar")
        self.mass_cartesian = GlobalMassLocationLoss(metric="cartesian")

    def forward(
        self,
        pred_log,
        target,
        warm_epoch,
        epoch,
        with_shape=True,
        *,
        location_mode=None,
        with_location=None,
        return_breakdown=None,
    ):
        mode = self.location_mode if location_mode is None else location_mode
        if mode not in self.VALID_LOCATION_MODES:
            raise ValueError("unknown location mode: %s" % mode)
        if with_location is None:
            with_location = with_shape
        if return_breakdown is None:
            return_breakdown = self.return_breakdown

        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        dis = torch.pow((pred_sum-target_sum)/2, 2)
        
        
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth) 
        
        loss = (intersection_sum + smooth) / \
                (pred_sum + target_sum - intersection_sum  + smooth)       
        location_log = {}
        if epoch>warm_epoch:
            siou_loss = alpha * loss
            segmentation_loss = 1 - siou_loss.mean()
            if not with_location or mode == "none":
                location_loss = pred.new_zeros(())
            elif mode == "legacy":
                location_loss = legacy_location_loss(pred, target)
            elif mode == "mass_polar":
                location_loss, location_log = self.mass_polar(pred, target)
            elif mode == "mass_cartesian":
                location_loss, location_log = self.mass_cartesian(pred, target)
            else:
                raise AssertionError("unreachable location mode: %s" % mode)
            loss = segmentation_loss + self.lambda_location * location_loss
        else:
            # Preserve canonical MSHNet warm-up exactly: ordinary soft IoU only.
            segmentation_loss = 1 - loss.mean()
            location_loss = pred.new_zeros(())
            loss = segmentation_loss

        if not return_breakdown:
            return loss
        breakdown = {
            "total": loss,
            "segmentation": segmentation_loss.detach(),
            "location": location_loss.detach(),
            "location_weighted": (
                self.lambda_location * location_loss
            ).detach(),
        }
        breakdown.update(location_log)
        return loss, breakdown
    
    

# Historical public name retained for downstream imports and checkpoint-era tools.
LLoss = legacy_location_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build_safe_bg(gt, kernel_size=15):
    pad = kernel_size // 2
    gt_dilate = F.max_pool2d(
        gt.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )
    safe_bg = (gt_dilate < 0.5).float()
    return safe_bg


def single_scale_anti_sufficiency_loss(z_only_max, gt, tau=0.5):
    safe_bg = build_safe_bg(gt)

    with torch.no_grad():
        hard_bg = safe_bg * (torch.sigmoid(z_only_max.detach()) > tau).float()

    loss_map = F.binary_cross_entropy_with_logits(
        z_only_max,
        torch.zeros_like(z_only_max),
        reduction="none",
    )

    loss = (loss_map * hard_bg).sum() / (hard_bg.sum() + 1e-6)
    log_vars = {
        "hard_bg_ratio": hard_bg.mean().detach(),
        "z_only_prob_mean": torch.sigmoid(z_only_max.detach()).mean(),
        "z_only_prob_max": torch.sigmoid(z_only_max.detach()).max(),
    }
    return loss, log_vars


def empty_evidence_loss(z_empty):
    return F.binary_cross_entropy_with_logits(
        z_empty,
        torch.zeros_like(z_empty),
    )


def decidability_loss(d_logit, z_full, gt, tau=0.5):
    safe_bg = build_safe_bg(gt)
    pos = gt.float()

    with torch.no_grad():
        hard_bg = safe_bg * (torch.sigmoid(z_full.detach()) > tau).float()

    valid = torch.clamp(pos + hard_bg, max=1.0)
    label = pos

    loss_map = F.binary_cross_entropy_with_logits(
        d_logit,
        label,
        reduction="none",
    )

    loss = (loss_map * valid).sum() / (valid.sum() + 1e-6)
    log_vars = {
        "d_pos_ratio": pos.mean().detach(),
        "d_hard_bg_ratio": hard_bg.mean().detach(),
        "d_prob_mean": torch.sigmoid(d_logit.detach()).mean(),
        "d_prob_max": torch.sigmoid(d_logit.detach()).max(),
    }
    return loss, log_vars


def dea_lite_loss(
    dea_out,
    z_full,
    gt,
    lambda_single=0.0,
    lambda_dec=0.0,
    lambda_empty=0.0,
    tau=0.5,
):
    total_loss = torch.tensor(0.0, device=z_full.device)
    log_vars = {}

    if lambda_single > 0:
        loss_single, single_log = single_scale_anti_sufficiency_loss(
            dea_out["z_only_max"],
            gt,
            tau=tau,
        )
        total_loss = total_loss + lambda_single * loss_single
        log_vars["loss_single_raw"] = loss_single.detach()
        log_vars["loss_single_weighted"] = (lambda_single * loss_single).detach()
        log_vars.update(single_log)

    if lambda_empty > 0:
        loss_empty = empty_evidence_loss(dea_out["z_empty"])
        total_loss = total_loss + lambda_empty * loss_empty
        log_vars["loss_empty_raw"] = loss_empty.detach()
        log_vars["loss_empty_weighted"] = (lambda_empty * loss_empty).detach()

    if lambda_dec > 0:
        loss_dec, dec_log = decidability_loss(
            dea_out["decidability_logit"],
            z_full,
            gt,
            tau=tau,
        )
        total_loss = total_loss + lambda_dec * loss_dec
        log_vars["loss_dec_raw"] = loss_dec.detach()
        log_vars["loss_dec_weighted"] = (lambda_dec * loss_dec).detach()
        log_vars.update(dec_log)
    elif "decidability_logit" in dea_out:
        log_vars["d_prob_mean"] = torch.sigmoid(
            dea_out["decidability_logit"].detach()
        ).mean()
        log_vars["d_prob_max"] = torch.sigmoid(
            dea_out["decidability_logit"].detach()
        ).max()

    return total_loss, log_vars
