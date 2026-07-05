import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
from skimage import measure


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
    def __init__(self):
        super(SLSIoULoss, self).__init__()


    def forward(self, pred_log, target,warm_epoch, epoch, with_shape=True):
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
        lloss = LLoss(pred, target)

        if epoch>warm_epoch:       
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 -siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss
    
    

def LLoss(pred, target):
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


def single_scale_anti_sufficiency_loss(z_only_max, z_full, gt, tau=0.3):
    safe_bg = build_safe_bg(gt)

    with torch.no_grad():
        hard_bg_from_full = (torch.sigmoid(z_full) > tau).float()
        hard_bg_from_only = (torch.sigmoid(z_only_max) > tau).float()
        hard_bg = safe_bg * torch.clamp(
            hard_bg_from_full + hard_bg_from_only,
            max=1.0,
        )

    loss_map = F.binary_cross_entropy_with_logits(
        z_only_max,
        torch.zeros_like(z_only_max),
        reduction="none",
    )

    loss = (loss_map * hard_bg).sum() / (hard_bg.sum() + 1e-6)
    return loss


def empty_evidence_loss(z_empty):
    return F.binary_cross_entropy_with_logits(
        z_empty,
        torch.zeros_like(z_empty),
    )


def decidability_loss(d_logit, z_full, gt, tau=0.3):
    safe_bg = build_safe_bg(gt)
    pos = gt.float()

    with torch.no_grad():
        hard_bg = safe_bg * (torch.sigmoid(z_full) > tau).float()

    valid = torch.clamp(pos + hard_bg, max=1.0)
    label = pos

    loss_map = F.binary_cross_entropy_with_logits(
        d_logit,
        label,
        reduction="none",
    )

    loss = (loss_map * valid).sum() / (valid.sum() + 1e-6)
    return loss


def dea_lite_loss(dea_out, z_full, gt,
                  lambda_single=0.10,
                  lambda_dec=0.05,
                  lambda_empty=0.01,
                  tau=0.3):
    loss_single = single_scale_anti_sufficiency_loss(
        dea_out["z_only_max"],
        z_full,
        gt,
        tau=tau,
    )

    loss_dec = decidability_loss(
        dea_out["decidability_logit"],
        z_full,
        gt,
        tau=tau,
    )

    loss_empty = empty_evidence_loss(dea_out["z_empty"])

    loss = (
        lambda_single * loss_single
        + lambda_dec * loss_dec
        + lambda_empty * loss_empty
    )

    log_vars = {
        "loss_single": loss_single.detach(),
        "loss_dec": loss_dec.detach(),
        "loss_empty": loss_empty.detach(),
    }

    return loss, log_vars
