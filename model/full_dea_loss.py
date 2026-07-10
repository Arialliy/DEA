from __future__ import annotations

import torch
import torch.nn.functional as F
from skimage import measure

from model.dea_evidence import gather_chosen_relation_endpoints
from model.loss import SoftIoULoss, build_safe_bg


def _masked_mean(
    loss_map: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return (loss_map * mask).sum() / (mask.sum() + eps)


def _topk_safe_mask(
    score: torch.Tensor,
    safe_bg: torch.Tensor,
    ratio: float = 0.001,
    min_score: float = 0.45,
) -> torch.Tensor:
    b, _, h, w = score.shape
    score = score.detach() * safe_bg
    flat_score = score.view(b, -1)
    flat_safe = safe_bg.view(b, -1)
    k = max(1, int(h * w * ratio))

    masks = []
    for i in range(b):
        valid_idx = torch.nonzero(
            (flat_safe[i] > 0.5) & (flat_score[i] >= min_score),
            as_tuple=False,
        ).view(-1)
        m = torch.zeros_like(flat_safe[i])
        if valid_idx.numel() > 0:
            local_score = flat_score[i, valid_idx]
            local_k = min(k, valid_idx.numel())
            top_idx_local = torch.topk(local_score, k=local_k, largest=True).indices
            m[valid_idx[top_idx_local]] = 1.0
        masks.append(m)

    return torch.stack(masks, dim=0).view(b, 1, h, w)


def _limit_safe_mask_by_score(
    mask: torch.Tensor,
    score: torch.Tensor,
    max_ratio: float,
) -> torch.Tensor:
    if max_ratio <= 0:
        return mask

    b, _, h, w = mask.shape
    flat_mask = mask.view(b, -1)
    flat_score = score.detach().view(b, -1)
    max_k = max(1, int(h * w * max_ratio))

    limited = []
    for i in range(b):
        valid_idx = torch.nonzero(flat_mask[i] > 0.5, as_tuple=False).view(-1)
        out = torch.zeros_like(flat_mask[i])
        if valid_idx.numel() > 0:
            local_k = min(max_k, valid_idx.numel())
            local_score = flat_score[i, valid_idx]
            top_idx_local = torch.topk(local_score, k=local_k, largest=True).indices
            out[valid_idx[top_idx_local]] = 1.0
        limited.append(out)

    return torch.stack(limited, dim=0).view(b, 1, h, w)


def _limit_component_mask_by_score(
    mask: torch.Tensor,
    score: torch.Tensor,
    max_ratio: float,
) -> torch.Tensor:
    """Select complete high-score components under a per-image pixel budget.

    Unlike pixel top-k, this function never cuts holes into a selected clutter
    component. Components that do not fit in the remaining budget are skipped.
    """
    if max_ratio <= 0:
        return mask

    b, _, h, w = mask.shape
    max_pixels = max(1, int(h * w * max_ratio))
    mask_cpu = mask.detach().cpu().numpy()
    score_cpu = score.detach().cpu().numpy()
    limited = []

    for i in range(b):
        labels = measure.label(mask_cpu[i, 0] > 0.5, connectivity=2)
        regions = list(
            measure.regionprops(labels, intensity_image=score_cpu[i, 0])
        )
        regions.sort(
            key=lambda region: (float(region.max_intensity), float(region.mean_intensity)),
            reverse=True,
        )
        out = torch.zeros((h, w), device=mask.device, dtype=mask.dtype)
        used = 0
        for region in regions:
            area = int(region.area)
            if used + area > max_pixels:
                continue
            coords = region.coords
            out[coords[:, 0], coords[:, 1]] = 1.0
            used += area
        limited.append(out.view(1, h, w))

    return torch.stack(limited, dim=0)


def _dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    return F.max_pool2d(
        mask.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )


def build_tps_region_masks(
    target: torch.Tensor,
    protect_kernel: int = 9,
    safe_kernel: int = 15,
) -> dict[str, torch.Tensor]:
    target_core = (target.float() > 0.5).float()
    target_protect = (_dilate_mask(target_core, protect_kernel) > 0.5).float()
    target_safe_dilate = (_dilate_mask(target_core, safe_kernel) > 0.5).float()
    safe_bg = (target_safe_dilate < 0.5).float()
    ignore_ring = torch.clamp(target_safe_dilate - target_core, min=0.0, max=1.0)
    return {
        "target_core": target_core,
        "target_protect": target_protect,
        "ignore_ring": ignore_ring,
        "safe_bg": safe_bg,
    }


def _component_hard_mask(
    candidate: torch.Tensor,
    score: torch.Tensor,
    target_protect: torch.Tensor,
    min_area: int,
    max_area: int,
) -> torch.Tensor:
    b, _, h, w = candidate.shape
    masks = []
    candidate_cpu = candidate.detach().cpu().numpy()
    score_cpu = score.detach().cpu().numpy()
    protect_cpu = target_protect.detach().cpu().numpy()

    for i in range(b):
        labels = measure.label(candidate_cpu[i, 0] > 0.5, connectivity=2)
        out = torch.zeros((h, w), device=candidate.device, dtype=candidate.dtype)
        for region in measure.regionprops(labels, intensity_image=score_cpu[i, 0]):
            area = int(region.area)
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue
            coords = region.coords
            if protect_cpu[i, 0, coords[:, 0], coords[:, 1]].max() > 0:
                continue
            out[coords[:, 0], coords[:, 1]] = 1.0
        masks.append(out.view(1, h, w))

    return torch.stack(masks, dim=0)


def build_component_hard_clutter_label(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    tau_base: float = 0.45,
    tau_target: float = 0.45,
    tau_scale: float = 0.45,
    protect_kernel: int = 9,
    safe_kernel: int = 15,
    min_area: int = 1,
    max_area: int = 256,
    max_hard_bg_ratio: float = 0.003,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    regions = build_tps_region_masks(
        target=target,
        protect_kernel=protect_kernel,
        safe_kernel=safe_kernel,
    )
    safe_bg = regions["safe_bg"]

    z_base = full_dea_out["z_base"]
    scale_logits = full_dea_out["scale_logits_full"]
    z_without_scale = full_dea_out["z_without_scale"]

    with torch.no_grad():
        p_base = torch.sigmoid(z_base.detach())
        p_scale = torch.sigmoid(scale_logits.detach()).max(dim=1, keepdim=True)[0]
        p_counterfactual = torch.sigmoid(z_without_scale.detach()).max(
            dim=1,
            keepdim=True,
        )[0]

        # A clutter candidate is either a baseline false alarm or a small
        # target-like component exposed by a single-scale/leave-one-scale-out
        # intervention. The latter makes the pseudo label consistent with
        # DEA's exact scale-fragility mechanism and avoids an empty clutter
        # class when a strong frozen baseline has few training-set errors.
        hard_score = torch.maximum(
            torch.maximum(p_base, p_scale),
            p_counterfactual,
        )
        candidate = safe_bg * (
            (p_base > tau_base)
            | (p_counterfactual > tau_target)
            | (p_scale > tau_scale)
        ).float()
        hard_clutter = _component_hard_mask(
            candidate=candidate,
            score=hard_score,
            target_protect=regions["target_protect"],
            min_area=max(1, int(min_area)),
            max_area=int(max_area),
        )
        hard_clutter = hard_clutter * safe_bg
        hard_clutter = _limit_component_mask_by_score(
            hard_clutter,
            hard_score,
            max_ratio=max_hard_bg_ratio,
        )

    return hard_clutter, regions


def _paired_component_scores(
    value: torch.Tensor,
    reference: torch.Tensor,
    component_mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable log-mean-exp scores for every binary component."""
    labels_cpu = component_mask.detach().cpu().numpy()
    value_scores = []
    reference_scores = []
    temp = max(float(temperature), 1e-6)

    for batch_index in range(component_mask.shape[0]):
        labels = measure.label(labels_cpu[batch_index, 0] > 0.5, connectivity=2)
        for component_id in range(1, int(labels.max()) + 1):
            component = torch.from_numpy(labels == component_id).to(
                device=value.device,
                dtype=torch.bool,
            )
            value_pixels = value[batch_index, 0][component]
            reference_pixels = reference[batch_index, 0][component]
            normalizer = torch.log(
                value_pixels.new_tensor(float(max(1, value_pixels.numel())))
            )
            value_scores.append(
                temp * (torch.logsumexp(value_pixels / temp, dim=0) - normalizer)
            )
            reference_scores.append(
                temp
                * (torch.logsumexp(reference_pixels / temp, dim=0) - normalizer)
            )

    if not value_scores:
        zero = value.sum().reshape(()) * 0.0
        return zero.view(1), zero.view(1)
    return torch.stack(value_scores), torch.stack(reference_scores)


def _component_keep_loss(
    value: torch.Tensor,
    reference: torch.Tensor,
    component_mask: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    value_scores, reference_scores = _paired_component_scores(
        value,
        reference.detach(),
        component_mask,
    )
    return F.relu(reference_scores + margin - value_scores).mean()


def _component_suppress_loss(
    value: torch.Tensor,
    reference: torch.Tensor,
    component_mask: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    value_scores, reference_scores = _paired_component_scores(
        value,
        reference.detach(),
        component_mask,
    )
    return F.relu(value_scores - reference_scores + margin).mean()


def build_hard_clutter_label(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    tau_base: float = 0.45,
    tau_target: float = 0.45,
    tau_scale: float = 0.45,
    safe_kernel: int = 15,
    topk_ratio: float = 0.001,
    topk_min_score: float = 0.45,
    max_hard_bg_ratio: float = 0.003,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.float()
    safe_bg = build_safe_bg(target, kernel_size=safe_kernel)

    z_base = full_dea_out["z_base"]
    z_target = full_dea_out["z_target"]
    scale_logits_full = full_dea_out["scale_logits_full"]

    with torch.no_grad():
        p_base = torch.sigmoid(z_base.detach())
        p_target = torch.sigmoid(z_target.detach())
        p_scale = torch.sigmoid(scale_logits_full.detach()).max(dim=1, keepdim=True)[0]
        hard_score = torch.maximum(torch.maximum(p_base, p_target), p_scale)

        hard_by_threshold = safe_bg * (
            (p_base > tau_base).float()
            + (p_target > tau_target).float()
            + (p_scale > tau_scale).float()
        )
        hard_by_threshold = (hard_by_threshold > 0).float()

        hard_by_topk = _topk_safe_mask(
            hard_score,
            safe_bg,
            ratio=topk_ratio,
            min_score=topk_min_score,
        )
        hard_bg = torch.maximum(hard_by_threshold, hard_by_topk) * safe_bg
        hard_bg = _limit_safe_mask_by_score(
            hard_bg,
            hard_score,
            max_ratio=max_hard_bg_ratio,
        )

    return hard_bg, safe_bg


def full_dea_aux_loss_v2(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    epoch: int,
    warm_epoch: int,
    seg_criterion=None,
    tau_base: float = 0.45,
    tau_target: float = 0.45,
    tau_scale: float = 0.45,
    margin: float = 1.0,
    lambda_target_aux: float = 0.30,
    lambda_ev_target: float = 0.20,
    lambda_ev_clutter: float = 0.20,
    lambda_clutter_pred: float = 0.20,
    lambda_suppress_gate: float = 0.10,
    lambda_margin: float = 0.05,
    lambda_hard_bg_final: float = 0.10,
    lambda_suppress_order: float = 0.05,
    safe_kernel: int = 15,
    topk_ratio: float = 0.001,
    topk_min_score: float = 0.45,
    max_hard_bg_ratio: float = 0.003,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = target.float()
    device = target.device

    z_base = full_dea_out["z_base"]
    z_target = full_dea_out["z_target"]
    z_final = full_dea_out["z_final"]
    z_clutter = full_dea_out["z_clutter"]
    e_t_logit = full_dea_out["target_evidence_logit"]
    e_c_logit = full_dea_out["clutter_evidence_logit"]
    suppression_logit = full_dea_out["suppression_logit"]

    hard_bg, safe_bg = build_hard_clutter_label(
        full_dea_out=full_dea_out,
        target=target,
        tau_base=tau_base,
        tau_target=tau_target,
        tau_scale=tau_scale,
        safe_kernel=safe_kernel,
        topk_ratio=topk_ratio,
        topk_min_score=topk_min_score,
        max_hard_bg_ratio=max_hard_bg_ratio,
    )
    valid_ev = torch.clamp(target + hard_bg, 0.0, 1.0)

    if seg_criterion is None:
        loss_target_aux = SoftIoULoss(z_target, target)
    else:
        loss_target_aux = seg_criterion(z_target, target, warm_epoch, epoch)

    loss_ev_t = _masked_mean(
        F.binary_cross_entropy_with_logits(e_t_logit, target, reduction="none"),
        valid_ev,
    )
    loss_ev_c = _masked_mean(
        F.binary_cross_entropy_with_logits(e_c_logit, hard_bg, reduction="none"),
        valid_ev,
    )
    loss_clutter_pred = _masked_mean(
        F.binary_cross_entropy_with_logits(z_clutter, hard_bg, reduction="none"),
        valid_ev,
    )
    loss_suppress_gate = _masked_mean(
        F.binary_cross_entropy_with_logits(
            suppression_logit,
            hard_bg,
            reduction="none",
        ),
        valid_ev,
    )

    loss_margin = _masked_mean(
        F.relu(margin - (e_t_logit - e_c_logit)),
        target,
    ) + _masked_mean(
        F.relu(margin - (e_c_logit - e_t_logit)),
        hard_bg,
    )

    loss_hard_bg_final = _masked_mean(
        F.binary_cross_entropy_with_logits(
            z_final,
            torch.zeros_like(z_final),
            reduction="none",
        ),
        hard_bg,
    )
    loss_suppress_order = _masked_mean(
        F.relu(z_final - z_base.detach()),
        hard_bg,
    )

    total = torch.tensor(0.0, device=device)
    total = total + lambda_target_aux * loss_target_aux
    total = total + lambda_ev_target * loss_ev_t
    total = total + lambda_ev_clutter * loss_ev_c
    total = total + lambda_clutter_pred * loss_clutter_pred
    total = total + lambda_suppress_gate * loss_suppress_gate
    total = total + lambda_margin * loss_margin
    total = total + lambda_hard_bg_final * loss_hard_bg_final
    total = total + lambda_suppress_order * loss_suppress_order

    target_sum = target.sum() + 1e-6
    hard_sum = hard_bg.sum() + 1e-6
    log_vars = {
        "full_dea_loss_target_aux": loss_target_aux.detach(),
        "full_dea_loss_ev_t": loss_ev_t.detach(),
        "full_dea_loss_ev_c": loss_ev_c.detach(),
        "full_dea_loss_clutter_pred": loss_clutter_pred.detach(),
        "full_dea_loss_suppress_gate": loss_suppress_gate.detach(),
        "full_dea_loss_margin": loss_margin.detach(),
        "full_dea_loss_hard_bg_final": loss_hard_bg_final.detach(),
        "full_dea_loss_suppress_order": loss_suppress_order.detach(),
        "hard_bg_ratio": hard_bg.detach().mean(),
        "safe_bg_ratio": safe_bg.detach().mean(),
        "target_evidence_on_gt": (
            torch.sigmoid(e_t_logit).detach() * target
        ).sum()
        / target_sum,
        "clutter_evidence_on_gt": (
            torch.sigmoid(e_c_logit).detach() * target
        ).sum()
        / target_sum,
        "target_evidence_on_hard_bg": (
            torch.sigmoid(e_t_logit).detach() * hard_bg
        ).sum()
        / hard_sum,
        "clutter_evidence_on_hard_bg": (
            torch.sigmoid(e_c_logit).detach() * hard_bg
        ).sum()
        / hard_sum,
        "suppression_on_gt": (
            torch.sigmoid(suppression_logit).detach() * target
        ).sum()
        / target_sum,
        "suppression_on_hard_bg": (
            torch.sigmoid(suppression_logit).detach() * hard_bg
        ).sum()
        / hard_sum,
        "alpha": full_dea_out["alpha"].detach(),
    }
    return total, log_vars


def full_dea_aux_loss_v3(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    epoch: int,
    warm_epoch: int,
    seg_criterion=None,
    tau_base: float = 0.45,
    tau_target: float = 0.45,
    tau_scale: float = 0.45,
    margin: float = 1.0,
    margin_target: float = 0.0,
    margin_bg: float = 0.5,
    lambda_target_aux: float = 0.10,
    lambda_keep_target: float = 0.50,
    lambda_protect: float = 0.20,
    lambda_ev_target: float = 0.15,
    lambda_ev_clutter: float = 0.15,
    lambda_clutter_pred: float = 0.10,
    lambda_suppress_gate: float = 0.15,
    lambda_margin: float = 0.05,
    lambda_hard_bg_final: float = 0.05,
    lambda_suppress_order: float = 0.50,
    lambda_bridge: float = 0.20,
    lambda_bridge_recover: float = 0.50,
    lambda_bridge_leak: float = 0.10,
    protect_kernel: int = 9,
    safe_kernel: int = 15,
    min_component_area: int = 1,
    max_component_area: int = 256,
    max_hard_bg_ratio: float = 0.003,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = target.float()
    device = target.device

    z_base = full_dea_out["z_base"]
    z_target = full_dea_out["z_target"]
    z_final = full_dea_out["z_final"]
    decision_logits = full_dea_out["decision_logits"]
    e_t_logit = full_dea_out["target_evidence_logit"]
    e_c_logit = full_dea_out["clutter_evidence_logit"]
    target_prob = full_dea_out["target_prob"]
    clutter_prob = full_dea_out["clutter_prob"]
    uncertain_prob = full_dea_out["uncertain_prob"]
    target_amount = full_dea_out["target_amount"]
    clutter_amount = full_dea_out["clutter_amount"]
    bridge_evidence_logit = full_dea_out["bridge_evidence_logit"]
    bridge_delta = full_dea_out["bridge_delta"]
    topology_prior = full_dea_out["topology_prior"]

    hard_clutter, regions = build_component_hard_clutter_label(
        full_dea_out=full_dea_out,
        target=target,
        tau_base=tau_base,
        tau_target=tau_target,
        tau_scale=tau_scale,
        protect_kernel=protect_kernel,
        safe_kernel=safe_kernel,
        max_hard_bg_ratio=max_hard_bg_ratio,
        min_area=min_component_area,
        max_area=max_component_area,
    )
    target_core = regions["target_core"]
    target_protect = regions["target_protect"]
    safe_bg = regions["safe_bg"]
    uncertain_bg = safe_bg * (1.0 - hard_clutter)
    with torch.no_grad():
        missing_target_weight = (
            target_core
            * torch.sigmoid(-z_base.detach())
            * topology_prior.detach()
        )
        # A topology proposal is positive only inside the same annotated
        # component. Proposals in the protection ring remain negative so the
        # bridge cannot merge two nearby targets into one predicted instance.
        bridge_negative_weight = (
            (1.0 - target_core) * topology_prior.detach()
        )
        bridge_valid = torch.clamp(
            target_core + bridge_negative_weight,
            min=0.0,
            max=1.0,
        )

    if seg_criterion is None:
        loss_target_aux = SoftIoULoss(z_target, target_core)
    else:
        loss_target_aux = seg_criterion(z_target, target_core, warm_epoch, epoch)

    decision_target = torch.zeros_like(target_core, dtype=torch.long).squeeze(1)
    decision_clutter = torch.ones_like(target_core, dtype=torch.long).squeeze(1)
    decision_uncertain = torch.full_like(
        target_core,
        fill_value=2,
        dtype=torch.long,
    ).squeeze(1)
    decision_ce_target = F.cross_entropy(
        decision_logits,
        decision_target,
        reduction="none",
    ).unsqueeze(1)
    decision_ce_clutter = F.cross_entropy(
        decision_logits,
        decision_clutter,
        reduction="none",
    ).unsqueeze(1)
    decision_ce_uncertain = F.cross_entropy(
        decision_logits,
        decision_uncertain,
        reduction="none",
    ).unsqueeze(1)
    # Target-core recognition and topology bridging are separate operations.
    # Supervising the full dilated protection ring as target duplicates the
    # bridge path and causes boundary expansion on dense-target datasets.
    loss_decision_target = _masked_mean(decision_ce_target, target_core)
    loss_decision_clutter = _masked_mean(decision_ce_clutter, hard_clutter)
    loss_decision_uncertain = _masked_mean(decision_ce_uncertain, uncertain_bg)
    loss_decision = (
        loss_decision_target
        + loss_decision_clutter
        + 0.25 * loss_decision_uncertain
    )
    loss_bridge_decision = _masked_mean(
        F.binary_cross_entropy_with_logits(
            bridge_evidence_logit,
            target_core,
            reduction="none",
        ),
        bridge_valid,
    )
    loss_bridge_recover = _masked_mean(
        1.0 - torch.sigmoid(z_target),
        missing_target_weight,
    )
    loss_bridge_leak = _masked_mean(bridge_delta, 1.0 - target_core)

    # Object-level constraints mirror the object-level PD/FA evaluation. A
    # component is kept/suppressed by its differentiable log-mean-exp score,
    # rather than by independently editing a few easy pixels.
    loss_keep_target = _component_keep_loss(
        value=z_final,
        reference=z_base,
        component_mask=target_core,
        margin=margin_target,
    )
    loss_suppress_component = _component_suppress_loss(
        value=z_final,
        reference=z_base,
        component_mask=hard_clutter,
        margin=margin_bg,
    )

    # Asymmetric calibration must not leak across semantic states: target
    # enhancement is penalized on clutter, while clutter suppression is
    # penalized throughout the target protection region.
    loss_target_leak = _masked_mean(target_amount, hard_clutter)
    loss_clutter_leak = _masked_mean(clutter_amount, target_protect)
    loss_uncertain_identity = _masked_mean(
        torch.abs(z_final - z_base.detach()),
        uncertain_bg,
    )
    loss_margin = _masked_mean(
        F.relu(margin - (e_t_logit - e_c_logit)),
        target_core,
    ) + _masked_mean(
        F.relu(margin - (e_c_logit - e_t_logit)),
        hard_clutter,
    )
    loss_hard_bg_final = _masked_mean(
        F.binary_cross_entropy_with_logits(
            z_final,
            torch.zeros_like(z_final),
            reduction="none",
        ),
        hard_clutter,
    )

    total = torch.zeros((), device=device)
    total = total + lambda_target_aux * loss_target_aux
    total = total + lambda_keep_target * loss_keep_target
    total = total + lambda_protect * loss_decision
    total = total + lambda_ev_target * loss_target_leak
    total = total + lambda_ev_clutter * loss_clutter_leak
    total = total + lambda_clutter_pred * loss_uncertain_identity
    total = total + lambda_suppress_gate * loss_decision_clutter
    total = total + lambda_margin * loss_margin
    total = total + lambda_hard_bg_final * loss_hard_bg_final
    total = total + lambda_suppress_order * loss_suppress_component
    total = total + lambda_bridge * loss_bridge_decision
    total = total + lambda_bridge_recover * loss_bridge_recover
    total = total + lambda_bridge_leak * loss_bridge_leak

    target_sum = target_core.sum() + 1e-6
    protect_sum = target_protect.sum() + 1e-6
    hard_sum = hard_clutter.sum() + 1e-6

    def mean_on(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return (value.detach() * mask).sum() / (mask.sum() + 1e-6)

    delta_final_base = z_final - z_base.detach()
    log_vars = {
        "full_dea_loss_target_aux": loss_target_aux.detach(),
        "full_dea_loss_keep_target": loss_keep_target.detach(),
        "full_dea_loss_component_keep": loss_keep_target.detach(),
        "full_dea_loss_component_suppress": loss_suppress_component.detach(),
        "full_dea_loss_decision": loss_decision.detach(),
        "full_dea_loss_decision_target": loss_decision_target.detach(),
        "full_dea_loss_decision_clutter": loss_decision_clutter.detach(),
        "full_dea_loss_decision_uncertain": loss_decision_uncertain.detach(),
        "full_dea_loss_target_leak": loss_target_leak.detach(),
        "full_dea_loss_clutter_leak": loss_clutter_leak.detach(),
        "full_dea_loss_uncertain_identity": loss_uncertain_identity.detach(),
        "full_dea_loss_bridge_decision": loss_bridge_decision.detach(),
        "full_dea_loss_bridge_recover": loss_bridge_recover.detach(),
        "full_dea_loss_bridge_leak": loss_bridge_leak.detach(),
        # Compatibility names retained for existing log parsers.
        "full_dea_loss_protect": loss_decision_target.detach(),
        "full_dea_loss_ev_t": loss_decision_target.detach(),
        "full_dea_loss_ev_c": loss_decision_clutter.detach(),
        "full_dea_loss_clutter_pred": loss_clutter_leak.detach(),
        "full_dea_loss_suppress_gate": loss_decision_clutter.detach(),
        "full_dea_loss_margin": loss_margin.detach(),
        "full_dea_loss_hard_bg_final": loss_hard_bg_final.detach(),
        "full_dea_loss_suppress_order": loss_suppress_component.detach(),
        "hard_clutter_ratio": hard_clutter.detach().mean(),
        "safe_bg_ratio": safe_bg.detach().mean(),
        "target_protect_ratio": target_protect.detach().mean(),
        "delta_final_base_on_gt": mean_on(target_core, delta_final_base),
        "delta_final_base_on_target_protect": mean_on(target_protect, delta_final_base),
        "delta_final_base_on_hard_clutter": mean_on(hard_clutter, delta_final_base),
        "target_prob_on_gt": (target_prob.detach() * target_core).sum() / target_sum,
        "target_prob_on_target_protect": (
            target_prob.detach() * target_protect
        ).sum()
        / protect_sum,
        "target_prob_on_hard_clutter": (
            target_prob.detach() * hard_clutter
        ).sum()
        / hard_sum,
        "clutter_prob_on_gt": (
            clutter_prob.detach() * target_core
        ).sum()
        / target_sum,
        "clutter_prob_on_hard_clutter": (
            clutter_prob.detach() * hard_clutter
        ).sum()
        / hard_sum,
        "uncertain_prob_on_gt": (
            uncertain_prob.detach() * target_core
        ).sum()
        / target_sum,
        "uncertain_prob_on_safe_bg": (
            uncertain_prob.detach() * uncertain_bg
        ).sum()
        / (uncertain_bg.sum() + 1e-6),
        "target_amount_on_gt": (
            target_amount.detach() * target_core
        ).sum()
        / target_sum,
        "target_amount_on_hard_clutter": (
            target_amount.detach() * hard_clutter
        ).sum()
        / hard_sum,
        "clutter_amount_on_gt": (
            clutter_amount.detach() * target_core
        ).sum()
        / target_sum,
        "clutter_amount_on_hard_clutter": (
            clutter_amount.detach() * hard_clutter
        ).sum()
        / hard_sum,
        "topology_prior_on_gt": mean_on(target_core, topology_prior),
        "topology_prior_on_safe_bg": mean_on(safe_bg, topology_prior),
        "missing_target_bridge_ratio": missing_target_weight.detach().mean(),
        "bridge_gate_on_gt": mean_on(target_core, full_dea_out["bridge_gate"]),
        "bridge_gate_on_safe_bg": mean_on(safe_bg, full_dea_out["bridge_gate"]),
        "bridge_gate_off_gt": mean_on(
            1.0 - target_core,
            full_dea_out["bridge_gate"],
        ),
        "bridge_delta_on_gt": mean_on(target_core, bridge_delta),
        "bridge_delta_on_safe_bg": mean_on(safe_bg, bridge_delta),
        "bridge_delta_off_gt": mean_on(1.0 - target_core, bridge_delta),
        "reconstruction_error": torch.abs(
            full_dea_out["z_reconstructed"].detach() - z_base.detach()
        ).max(),
        # Compatibility diagnostics.
        "protect_on_gt": mean_on(target_core, target_prob),
        "protect_on_target_protect": mean_on(target_protect, target_prob),
        "protect_on_hard_clutter": mean_on(hard_clutter, target_prob),
        "raw_suppression_on_gt": mean_on(target_core, clutter_prob),
        "raw_suppression_on_hard_clutter": mean_on(hard_clutter, clutter_prob),
        "suppression_on_gt": mean_on(target_core, clutter_prob),
        "suppression_on_target_protect": mean_on(target_protect, clutter_prob),
        "suppression_on_hard_clutter": mean_on(hard_clutter, clutter_prob),
        "target_boost_on_gt": mean_on(target_core, target_amount),
        "target_boost_on_hard_clutter": mean_on(hard_clutter, target_amount),
        "target_evidence_on_gt": mean_on(target_core, target_prob),
        "clutter_evidence_on_gt": mean_on(target_core, clutter_prob),
        "target_evidence_on_hard_clutter": mean_on(hard_clutter, target_prob),
        "clutter_evidence_on_hard_clutter": mean_on(hard_clutter, clutter_prob),
        "alpha": full_dea_out["alpha"].detach(),
        "beta": full_dea_out.get("beta", torch.tensor(0.0, device=device)).detach(),
        "gamma": full_dea_out.get("gamma", torch.tensor(0.0, device=device)).detach(),
    }
    return total, log_vars


def full_dea_aux_loss_v4(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    epoch: int,
    warm_epoch: int,
    seg_criterion=None,
    tau_base: float = 0.45,
    tau_target: float = 0.45,
    tau_scale: float = 0.45,
    protect_kernel: int = 9,
    safe_kernel: int = 15,
    min_component_area: int = 1,
    max_component_area: int = 256,
    max_hard_bg_ratio: float = 0.003,
    lambda_relation: float = 0.20,
    lambda_relation_leak: float = 0.10,
    lambda_relation_identity: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train pair-level reconnect/suppress/identity operations for DEA v4."""

    total, log_vars = full_dea_aux_loss_v3(
        full_dea_out=full_dea_out,
        target=target,
        epoch=epoch,
        warm_epoch=warm_epoch,
        seg_criterion=seg_criterion,
        tau_base=tau_base,
        tau_target=tau_target,
        tau_scale=tau_scale,
        protect_kernel=protect_kernel,
        safe_kernel=safe_kernel,
        min_component_area=min_component_area,
        max_component_area=max_component_area,
        max_hard_bg_ratio=max_hard_bg_ratio,
        # V4 supervises reconnection as one class of the joint relation
        # decision. Retaining the V3 binary bridge classifier would give the
        # same logits two incompatible objectives.
        lambda_bridge=0.0,
    )

    target_core = (target.float() > 0.5).float()
    pair_choice = full_dea_out["relation_pair_choice"]
    max_offset = 4
    positive_target, negative_target = gather_chosen_relation_endpoints(
        target_core,
        pair_choice,
        max_offset=max_offset,
    )
    positive_target = (positive_target > 0.5).float()
    negative_target = (negative_target > 0.5).float()

    # Reconnection is positive only if the gap center and both selected
    # endpoints lie in the annotated target. If exactly one endpoint lies in a
    # target, the other endpoint is supervised as satellite clutter. All other
    # configurations preserve the baseline through identity.
    reconnect_mask = target_core * positive_target * negative_target
    suppress_positive_mask = (
        negative_target * (1.0 - positive_target)
    )
    suppress_negative_mask = (
        positive_target * (1.0 - negative_target)
    )
    non_identity = torch.clamp(
        reconnect_mask + suppress_positive_mask + suppress_negative_mask,
        min=0.0,
        max=1.0,
    )
    identity_mask = 1.0 - non_identity

    relation_target = torch.full_like(
        target_core,
        fill_value=3,
        dtype=torch.long,
    )
    relation_target = torch.where(
        reconnect_mask.bool(),
        torch.zeros_like(relation_target),
        relation_target,
    )
    relation_target = torch.where(
        suppress_positive_mask.bool(),
        torch.ones_like(relation_target),
        relation_target,
    )
    relation_target = torch.where(
        suppress_negative_mask.bool(),
        torch.full_like(relation_target, 2),
        relation_target,
    )
    relation_ce = F.cross_entropy(
        full_dea_out["relation_logits"],
        relation_target.squeeze(1),
        reduction="none",
    ).unsqueeze(1)
    pair_weight = full_dea_out["relation_pair_prior"].detach()
    loss_relation_reconnect = _masked_mean(
        relation_ce,
        pair_weight * reconnect_mask,
    )
    loss_relation_suppress_positive = _masked_mean(
        relation_ce,
        pair_weight * suppress_positive_mask,
    )
    loss_relation_suppress_negative = _masked_mean(
        relation_ce,
        pair_weight * suppress_negative_mask,
    )
    loss_relation_identity_class = _masked_mean(
        relation_ce,
        pair_weight * identity_mask,
    )
    loss_relation = (
        loss_relation_reconnect
        + loss_relation_suppress_positive
        + loss_relation_suppress_negative
        + 0.25 * loss_relation_identity_class
    )

    relation_suppress_delta = full_dea_out["relation_suppress_delta"]
    loss_relation_target_leak = _masked_mean(
        relation_suppress_delta,
        target_core,
    )
    loss_relation_identity = _masked_mean(
        torch.abs(full_dea_out["z_final"] - full_dea_out["z_base"].detach()),
        pair_weight * identity_mask,
    )
    total = total + lambda_relation * loss_relation
    total = total + lambda_relation_leak * loss_relation_target_leak
    total = total + lambda_relation_identity * loss_relation_identity

    pair_mass = pair_weight.sum() + 1e-6
    reconnect_mass = (pair_weight * reconnect_mask).sum()
    suppress_positive_mass = (pair_weight * suppress_positive_mask).sum()
    suppress_negative_mass = (pair_weight * suppress_negative_mask).sum()
    identity_mass = (pair_weight * identity_mask).sum()
    relation_probabilities = full_dea_out["relation_probabilities"].detach()
    log_vars.update(
        {
            "full_dea_loss_relation": loss_relation.detach(),
            "full_dea_loss_relation_reconnect": loss_relation_reconnect.detach(),
            "full_dea_loss_relation_suppress_positive": (
                loss_relation_suppress_positive.detach()
            ),
            "full_dea_loss_relation_suppress_negative": (
                loss_relation_suppress_negative.detach()
            ),
            "full_dea_loss_relation_identity_class": (
                loss_relation_identity_class.detach()
            ),
            "full_dea_loss_relation_target_leak": (
                loss_relation_target_leak.detach()
            ),
            "full_dea_loss_relation_identity": loss_relation_identity.detach(),
            "relation_pair_prior_mean": pair_weight.mean(),
            "relation_reconnect_mass_ratio": reconnect_mass / pair_mass,
            "relation_suppress_positive_mass_ratio": (
                suppress_positive_mass / pair_mass
            ),
            "relation_suppress_negative_mass_ratio": (
                suppress_negative_mass / pair_mass
            ),
            "relation_identity_mass_ratio": identity_mass / pair_mass,
            "relation_reconnect_probability": (
                relation_probabilities[:, 0:1] * pair_weight
            ).sum()
            / pair_mass,
            "relation_suppress_probability": (
                relation_probabilities[:, 1:3].sum(dim=1, keepdim=True)
                * pair_weight
            ).sum()
            / pair_mass,
            "relation_identity_probability": (
                relation_probabilities[:, 3:4] * pair_weight
            ).sum()
            / pair_mass,
            "relation_reconnect_map_mean": full_dea_out[
                "relation_reconnect_map"
            ].detach().mean(),
            "relation_suppress_map_mean": full_dea_out[
                "relation_suppress_map"
            ].detach().mean(),
            "relation_suppress_delta_on_gt": _masked_mean(
                relation_suppress_delta.detach(),
                target_core,
            ),
            "relation_suppress_scale": full_dea_out[
                "relation_suppress_scale"
            ].detach(),
        }
    )
    return total, log_vars


def build_component_relation_targets(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    """Assign graph-pair operations from GT component identity and overlap."""

    label_map = full_dea_out["component_relation_label_map"].detach().cpu().numpy()
    pair_batch = full_dea_out[
        "component_relation_pair_batch"
    ].detach().cpu().tolist()
    pair_first = full_dea_out[
        "component_relation_pair_first_id"
    ].detach().cpu().tolist()
    pair_second = full_dea_out[
        "component_relation_pair_second_id"
    ].detach().cpu().tolist()
    target_np = target.detach().cpu().numpy()
    gt_labels = [
        measure.label(target_np[index, 0] > 0.5, connectivity=2)
        for index in range(target.shape[0])
    ]
    assignment_cache: dict[tuple[int, int], int] = {}

    def assigned_gt(batch_index: int, component_id: int) -> int:
        key = (batch_index, component_id)
        if key in assignment_cache:
            return assignment_cache[key]
        overlap_labels = gt_labels[batch_index][
            label_map[batch_index, 0] == component_id
        ]
        overlap_labels = overlap_labels[overlap_labels > 0]
        if overlap_labels.size == 0:
            assignment = 0
        else:
            counts = np.bincount(overlap_labels.astype(np.int64))
            assignment = int(np.argmax(counts))
        assignment_cache[key] = assignment
        return assignment

    targets = []
    for batch_index, first_id, second_id in zip(
        pair_batch,
        pair_first,
        pair_second,
    ):
        first_gt = assigned_gt(int(batch_index), int(first_id))
        second_gt = assigned_gt(int(batch_index), int(second_id))
        if first_gt > 0 and first_gt == second_gt:
            relation = 0
        elif first_gt == 0 and second_gt > 0:
            relation = 1
        elif first_gt > 0 and second_gt == 0:
            relation = 2
        else:
            relation = 3
        targets.append(relation)
    return torch.tensor(
        targets,
        device=target.device,
        dtype=torch.long,
    )


def full_dea_aux_loss_v6(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    epoch: int,
    warm_epoch: int,
    seg_criterion=None,
    tau_base: float = 0.45,
    tau_target: float = 0.45,
    tau_scale: float = 0.45,
    protect_kernel: int = 9,
    safe_kernel: int = 15,
    min_component_area: int = 1,
    max_component_area: int = 256,
    max_hard_bg_ratio: float = 0.003,
    lambda_component_relation: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train exact component-pair operations while retaining DEA pixel states."""

    total, log_vars = full_dea_aux_loss_v3(
        full_dea_out=full_dea_out,
        target=target,
        epoch=epoch,
        warm_epoch=warm_epoch,
        seg_criterion=seg_criterion,
        tau_base=tau_base,
        tau_target=tau_target,
        tau_scale=tau_scale,
        protect_kernel=protect_kernel,
        safe_kernel=safe_kernel,
        min_component_area=min_component_area,
        max_component_area=max_component_area,
        max_hard_bg_ratio=max_hard_bg_ratio,
        lambda_bridge=0.0,
        lambda_bridge_recover=0.0,
        lambda_bridge_leak=0.0,
    )
    relation_logits = full_dea_out["component_relation_logits"]
    relation_targets = build_component_relation_targets(full_dea_out, target)
    if relation_logits.shape[0] == 0:
        loss_component_relation = full_dea_out["z_final"].sum() * 0.0
        class_losses = [loss_component_relation for _ in range(4)]
        class_ratios = [loss_component_relation.detach() for _ in range(4)]
    else:
        relation_ce = F.cross_entropy(
            relation_logits,
            relation_targets,
            reduction="none",
        )
        class_losses = []
        class_ratios = []
        for relation_class in range(4):
            class_mask = relation_targets == relation_class
            if class_mask.any():
                class_losses.append(relation_ce[class_mask].mean())
            else:
                class_losses.append(relation_ce.sum() * 0.0)
            class_ratios.append(class_mask.float().mean().detach())
        loss_component_relation = (
            class_losses[0]
            + class_losses[1]
            + class_losses[2]
            + 0.25 * class_losses[3]
        )
    total = total + lambda_component_relation * loss_component_relation

    probabilities = full_dea_out["component_relation_probabilities"].detach()
    if probabilities.shape[0] == 0:
        probability_means = [
            full_dea_out["z_final"].new_tensor(0.0) for _ in range(4)
        ]
    else:
        probability_means = [
            probabilities[:, index].mean() for index in range(4)
        ]
    log_vars.update(
        {
            "full_dea_loss_component_relation_graph": (
                loss_component_relation.detach()
            ),
            "full_dea_loss_relation_reconnect": class_losses[0].detach(),
            "full_dea_loss_relation_suppress_first": class_losses[1].detach(),
            "full_dea_loss_relation_suppress_second": class_losses[2].detach(),
            "full_dea_loss_relation_identity": class_losses[3].detach(),
            "component_relation_pair_count": full_dea_out[
                "component_relation_pair_count"
            ].detach(),
            "component_relation_reconnect_ratio": class_ratios[0],
            "component_relation_suppress_first_ratio": class_ratios[1],
            "component_relation_suppress_second_ratio": class_ratios[2],
            "component_relation_identity_ratio": class_ratios[3],
            "component_relation_reconnect_probability": probability_means[0],
            "component_relation_suppress_first_probability": (
                probability_means[1]
            ),
            "component_relation_suppress_second_probability": (
                probability_means[2]
            ),
            "component_relation_identity_probability": probability_means[3],
            "component_relation_bridge_pixels": full_dea_out[
                "component_relation_bridge_mask"
            ].detach().sum(),
            "component_relation_suppress_pixels": full_dea_out[
                "component_relation_suppress_mask"
            ].detach().sum(),
        }
    )
    return total, log_vars


def full_dea_loss(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    lambda_cf: float = 0.1,
    lambda_bg: float = 0.05,
    lambda_sep: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return full_dea_aux_loss_v3(
        full_dea_out=full_dea_out,
        target=target,
        epoch=1,
        warm_epoch=0,
        lambda_target_aux=1.0,
        lambda_ev_target=lambda_sep,
        lambda_ev_clutter=lambda_sep,
        lambda_clutter_pred=lambda_cf,
        lambda_suppress_gate=lambda_bg,
        lambda_hard_bg_final=lambda_bg,
    )
