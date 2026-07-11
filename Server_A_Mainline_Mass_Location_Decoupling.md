# Server A 主线：MSHNet 实例级质量—位置解耦

> **研究方向**：Mass–Location Decoupling for Infrared Small Target Segmentation  
> **执行服务器**：Server A  
> **建议分支**：`research/mass-location-decoupling`  
> **代码基线**：`Arialliy/DEA` 的 `main`，文档分析时最新提交为 `9a363799baf904127a07cd05a24b8e526d4511c4`（短 SHA `9a36379`，2026-07-11）。开始实验前必须把本机完整 SHA 写入运行清单。  
> **核心约束**：本线不改 MSHNet 网络结构，不启用历史 DEA/Full-DEA/Integrated-DEA 模块，只研究 SLS 中位置监督的语义与实例化。

---

## 0. 一页执行摘要

### 0.1 本线要回答的唯一问题

当前 `model/loss.py::LLoss` 使用：

```python
pred_centerx = (x_index * pred).mean(dim=(1, 2, 3))
pred_centery = (y_index * pred).mean(dim=(1, 2, 3))
```

其数学形式是：

\[
\widetilde{\mu}(p)=\frac{1}{HW}\sum_i p_i x_i
=\frac{M(p)}{HW}\mu(p),\qquad M(p)=\sum_i p_i.
\]

所以它不是纯位置质心，而是“预测质量/面积/置信度 × 位置”的一阶矩。主线要验证：

> **MSHNet 的 FA–PD 权衡是否部分来自质量与位置耦合，以及多目标场景中的全局一阶矩抵消。**

### 0.2 严格执行顺序

1. 建立 `legacy` 精确等价路径；
2. 只做四个全局诊断：`legacy / none / mass_polar / mass_cartesian`；
3. 只有全局诊断通过 GO 条件，才实现实例级版本；
4. 实例级版本依次加入：`instance_loc → + instance_mass → + unmatched_mass`；
5. 最终只保留一个候选，做 3 数据集 × 3 seed；
6. 官方 test 在方法冻结前保持 sealed。

### 0.3 第一批应运行的实验

| ID | Final head | Side heads | 目的 |
|---|---|---|---|
| A0 | legacy SLS | legacy SLS | 精确 baseline |
| A1 | 无 location | 无 location | 判断 LLoss 的净作用 |
| A2 | mass-normalized polar | 同 final | 只隔离“除数错误” |
| A3 | mass-normalized Cartesian | 同 final | 同时消除原点极坐标耦合 |
| A4 | mass-normalized Cartesian | scale-IoU，无 location | 判断 side-head LLoss 是否是主要来源 |

第一批只允许修改 loss 和日志；不得加入 component loss、hard negative、attention 或推理模块。

### 0.4 主线停止条件

若 A1–A4 在三个数据集上均不能稳定改变 FA/PD 方向，或者变化完全可由阈值平移解释，则立即 NO-GO，不再开发实例级复杂损失。

---

## 1. 当前代码事实与问题定义

### 1.1 当前 MSHNet 训练图

当前 `model/MSHNet.py` 在完整阶段输出：

```text
mask0: 1×H×W
mask1: 1×H/2×W/2
mask2: 1×H/4×W/4
mask3: 1×H/8×W/8
pred : 四个 side logits 上采样拼接后，经 4→1 卷积融合
```

当前 `main.py` 对 final loss 和所有 side-head loss 相加后除以 `len(masks)+1`；side target 通过逐次 `MaxPool2d(2,2)` 获得。warm 阶段 `masks=[]`，只训练单一 full-resolution 输出；warm 结束后切换到完整多尺度图。

### 1.2 当前 LLoss 的三个结构性问题

#### 问题 A：质量—位置耦合

若 `pred = c * target` 且空间位置完全相同，真正的质心不变，但当前一阶矩会随 `c` 线性变化。于是置信度变化会被解释为“位置变化”。

#### 问题 B：面积—位置重复约束

SLS 的 scale-sensitive IoU 已经使用 `pred_sum` 与 `target_sum`；LLoss 又通过一阶矩隐式依赖 `pred_sum`。两个项可能重复控制质量，并产生梯度冲突。

#### 问题 C：多目标全局抵消

一张图中多个目标的偏移可以彼此抵消；远处小虚警也可能改变全局一阶矩。全局中心接近不代表每个目标均正确定位。

### 1.3 本线不声称什么

本线第一阶段只是机制诊断，不应提前声称：

- “现有 MSHNet 的 LLoss 是 bug”；
- “mass-normalized centroid 必然优于原方法”；
- “首次使用实例级 loss”；
- “具有因果解释”。

论文级结论必须来自跨数据集、跨 seed、component-level 的稳定证据。

---

## 2. 研究假设与可证伪预测

### H1：当前 LLoss 对幅度不具有不变性

**预测**：固定空间支持，仅改变概率幅度时，legacy LLoss 显著变化；mass-normalized centroid 基本不变。

### H2：全局位置监督在多目标中发生抵消

**预测**：构造两个目标朝相反方向偏移，global centroid loss 接近零，但 component-wise loss 明显非零。

### H3：原 LLoss 会改变预测质量与虚警梯度

**预测**：移除 LLoss 或质量归一化后，背景预测总质量、远距 FP component 数量及 FA 会系统性变化，而不仅是 centroid error 变化。

### H4：实例级解耦能改善 Pareto，而不是单纯更保守

**预测**：实例级方法在相同阈值和 threshold sweep 下同时保持 Pd/IoU，并减少 unmatched FP mass；若只降低所有 logits，则不满足 H4。

---

## 3. 分支、目录和提交纪律

### 3.1 初始化

```bash
git checkout main
git pull --ff-only
git rev-parse HEAD
git checkout -b research/mass-location-decoupling
```

把完整 SHA 写入：

```text
repro_runs/location_decoupling/<batch_id>/source_commit.txt
```

### 3.2 新增文件

```text
model/location_losses.py
utils/instance_targets.py
tools/audit_location_losses.py
tools/run_location_decoupling.py
tools/summarize_location_decoupling.py
tests/test_location_losses.py
tests/test_component_decoupling.py
```

### 3.3 修改文件

```text
model/loss.py
utils/data.py
main.py
```

### 3.4 禁止事项

- 不删除历史 DEA 文件；
- 不改 `model/MSHNet.py` 的 backbone/decoder/fusion；
- 不从最佳 baseline checkpoint 微调后与 from-scratch baseline 比较；
- 不使用 official test 选 loss mode、权重、ownership radius 或 checkpoint；
- 不把 A0 以外的模型结果写回 clean baseline 目录；
- 不在同一 run-dir 续跑不同语义的 loss。

---

## 4. Phase 0：建立精确 legacy 路径

### 4.1 目标

重构 `SLSIoULoss` 时，必须证明以下调用与当前代码完全一致：

```python
criterion = SLSIoULoss(location_mode="legacy")
loss = criterion(logits, target, warm_epoch, epoch)
```

### 4.2 必须通过的 identity 测试

固定随机种子，在 CPU 与 CUDA 各测试：

1. forward loss 完全一致；
2. `d(loss)/d(logits)` 完全一致；
3. warm 前、warm 后均一致；
4. 空目标、单目标、多目标均无 NaN；
5. float32 至少 `atol=0, rtol=0`；若 CUDA 算子导致不可避免差异，最多允许 `1e-7`，并在测试中写明原因；
6. A0 一个 epoch 的 batch loss 序列与原分支一致。

### 4.3 推荐重构原则

不要直接替换现有 `LLoss`。先保留原函数并改名：

```python
def legacy_location_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # 原 LLoss 逐行复制，不做“顺手优化”
    ...

# 兼容历史 import
LLoss = legacy_location_loss
```

---

## 5. 代码修改一：新增 `model/location_losses.py`

下面是建议骨架。第一阶段先实现 global modes；component mode 在 Phase 2 再补。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_xy_grid(reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return x/y grids in [0, 1), matching the legacy coordinate convention."""
    if reference.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got {tuple(reference.shape)}")
    h, w = reference.shape[-2:]
    x = torch.arange(w, device=reference.device, dtype=reference.dtype)
    y = torch.arange(h, device=reference.device, dtype=reference.dtype)
    x = (x / float(w)).view(1, 1, 1, w)
    y = (y / float(h)).view(1, 1, h, 1)
    return x, y


def probability_mass(prob: torch.Tensor) -> torch.Tensor:
    return prob.sum(dim=(1, 2, 3))


def mass_normalized_centroid(
    prob: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return B×2 centroid [x, y] and B-vector mass."""
    x, y = normalized_xy_grid(prob)
    mass = probability_mass(prob)
    safe_mass = mass.clamp_min(eps)
    cx = (prob * x).sum(dim=(1, 2, 3)) / safe_mass
    cy = (prob * y).sum(dim=(1, 2, 3)) / safe_mass
    return torch.stack((cx, cy), dim=-1), mass


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_f = valid.to(dtype=values.dtype)
    return (values * valid_f).sum() / valid_f.sum().clamp_min(1.0)


class GlobalMassLocationLoss(nn.Module):
    """Mass-normalized global location diagnostic.

    metric='polar' isolates normalization while retaining the legacy polar form.
    metric='cartesian' uses translation-stable x/y Smooth-L1 distance.
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
            # 保留 legacy 的 atan(y / x) 形式，用于只隔离 mass normalization。
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
        logs = {
            "location_valid_ratio": valid.float().mean().detach(),
            "pred_mass_mean": pred_mass.mean().detach(),
            "target_mass_mean": target_mass.mean().detach(),
            "global_centroid_l1": (
                (pred_center - target_center).abs().sum(dim=-1) * valid.float()
            ).sum().detach() / valid.float().sum().clamp_min(1.0),
        }
        return loss, logs
```

### 5.1 关键实现选择

- 坐标范围第一阶段保持 legacy 的 `[0,1)`，不要同时改成 `[-1,1]`；
- `mass_normalized_centroid` 使用 `clamp_min`，而不是分母统一加 `eps`，避免对正常质量破坏幅度不变性；
- target 为空时，location loss 为 0；背景约束仍由 segmentation loss 承担；
- `mass_polar` 是机制隔离 control，不是最终推荐形式；
- `mass_cartesian` 才是主候选，因为 legacy polar 距离依赖图像左上角原点。

---

## 6. 代码修改二：重构 `model/loss.py`

### 6.1 新接口

```python
class SLSIoULoss(nn.Module):
    def __init__(
        self,
        location_mode: str = "legacy",
        lambda_location: float = 1.0,
        return_breakdown: bool = False,
    ):
        super().__init__()
        self.location_mode = location_mode
        self.lambda_location = float(lambda_location)
        self.return_breakdown = bool(return_breakdown)
        ...

    def forward(
        self,
        pred_log,
        target,
        warm_epoch,
        epoch,
        with_shape=True,          # 保留历史位置参数/关键字，兼容现有调用
        *,
        location_mode=None,
        with_location=None,
        instance_map=None,
        return_breakdown=None,
    ):
        if with_location is None:
            with_location = with_shape
        ...
```

### 6.2 必须把 segmentation 与 location 拆开

建议增加私有函数：

```python
def sls_scale_iou_term(pred_prob, target):
    smooth = 0.0
    intersection_sum = (pred_prob * target).sum(dim=(1, 2, 3))
    pred_sum = pred_prob.sum(dim=(1, 2, 3))
    target_sum = target.sum(dim=(1, 2, 3))
    dis = torch.pow((pred_sum - target_sum) / 2, 2)
    alpha = (torch.minimum(pred_sum, target_sum) + dis + smooth) / (
        torch.maximum(pred_sum, target_sum) + dis + smooth
    )
    iou = (intersection_sum + smooth) / (
        pred_sum + target_sum - intersection_sum + smooth
    )
    return 1.0 - (alpha * iou).mean(), {
        "scale_iou_alpha_mean": alpha.mean().detach(),
        "soft_iou_mean": iou.mean().detach(),
    }
```

注意：为了 A0 精确一致，实际 legacy 分支应保持原运算顺序。上面的函数可用于新模式，但 A0 必须走逐行复制的旧表达式。此外，现有仓库内若有代码使用 `with_shape=False`，新接口必须通过 `with_shape`→`with_location` 兼容映射继续支持，不能直接删除该参数。

### 6.3 推荐 dispatch

```python
mode = location_mode or self.location_mode
pred = torch.sigmoid(pred_log)

# warm 阶段严格保持原行为：仅普通 soft IoU，不加入 scale/location。
if epoch <= warm_epoch:
    seg_loss = legacy_warm_iou(pred, target)
    loc_loss = pred.new_zeros(())
else:
    seg_loss = sls_scale_iou(...)
    if not with_location or mode == "none":
        loc_loss = pred.new_zeros(())
    elif mode == "legacy":
        loc_loss = legacy_location_loss(pred, target)
    elif mode == "mass_polar":
        loc_loss, loc_log = self.mass_polar(pred, target)
    elif mode == "mass_cartesian":
        loc_loss, loc_log = self.mass_cartesian(pred, target)
    elif mode == "component_decoupled":
        if instance_map is None:
            raise ValueError("component_decoupled requires instance_map")
        loc_loss, loc_log = self.component_loss(pred, target, instance_map)
    else:
        raise ValueError(f"unknown location mode: {mode}")

total = seg_loss + self.lambda_location * loc_loss
```

### 6.4 返回 breakdown

训练日志需要能够区分：

```python
{
    "total": total,
    "seg": seg_loss.detach(),
    "location": loc_loss.detach(),
    "location_weighted": (self.lambda_location * loc_loss).detach(),
    **loc_log,
}
```

默认仍返回 scalar，避免破坏历史调用：

```python
if return_breakdown:
    return total, breakdown
return total
```

---

## 7. 代码修改三：`main.py` 的 CLI 和训练逻辑

### 7.1 新增参数

```python
parser.add_argument(
    "--location-loss",
    default="legacy",
    choices=[
        "legacy",
        "none",
        "mass_polar",
        "mass_cartesian",
        "component_decoupled",
    ],
)
parser.add_argument(
    "--side-location-loss",
    default="same",
    choices=["same", "legacy", "none", "mass_polar", "mass_cartesian"],
)
parser.add_argument("--lambda-location", type=float, default=1.0)
parser.add_argument("--lambda-instance-location", type=float, default=1.0)
parser.add_argument("--lambda-instance-mass", type=float, default=0.0)
parser.add_argument("--lambda-unmatched-mass", type=float, default=0.0)
parser.add_argument("--component-radius-min", type=float, default=5.0)
parser.add_argument("--component-radius-scale", type=float, default=3.0)
parser.add_argument("--component-min-area", type=int, default=1)
parser.add_argument("--loss-log-interval", type=int, default=50)
```

### 7.2 参数校验

```python
if args.location_loss == "component_decoupled":
    if args.model_type != "mshnet":
        raise ValueError("mainline component loss is restricted to canonical mshnet")
    if args.lambda_instance_location < 0:
        raise ValueError(...)

if args.side_location_loss == "same" and args.location_loss == "component_decoupled":
    # 第一版不在低分辨率 side head 做 component loss，避免组件合并混杂。
    args.side_location_loss = "none"
```

### 7.3 Criterion 初始化

```python
self.loss_fun = SLSIoULoss(
    location_mode=args.location_loss,
    lambda_location=args.lambda_location,
)
```

### 7.4 训练 batch 兼容 2/3 元组

```python
for i, batch in enumerate(tbar):
    if len(batch) == 2:
        data, mask = batch
        instance_map = None
    elif len(batch) == 3:
        data, mask, instance_map = batch
    else:
        raise RuntimeError(f"unexpected batch length: {len(batch)}")

    data = data.to(self.device, non_blocking=True)
    labels = mask.to(self.device, non_blocking=True)
    if instance_map is not None:
        instance_map = instance_map.to(self.device, non_blocking=True)
```

### 7.5 替换当前统一 loss 段

为避免历史 DEA 分支受影响，建议抽出新函数，只在：

```python
args.model_type == "mshnet"
```

时使用：

```python
def canonical_mshnet_loss(self, pred, masks, labels, instance_map, epoch):
    final_loss, final_log = self.loss_fun(
        pred,
        labels,
        self.warm_epoch,
        epoch,
        location_mode=self.args.location_loss,
        instance_map=instance_map,
        return_breakdown=True,
    )

    losses = [final_loss]
    logs = {f"final/{k}": v for k, v in final_log.items() if k != "total"}

    side_mode = (
        self.args.location_loss
        if self.args.side_location_loss == "same"
        else self.args.side_location_loss
    )

    labels_for_scale = labels
    for j, side_logit in enumerate(masks):
        if j > 0:
            labels_for_scale = self.down(labels_for_scale)
        side_loss, side_log = self.loss_fun(
            side_logit,
            labels_for_scale,
            self.warm_epoch,
            epoch,
            location_mode=side_mode,
            # component mode 第一版只用于 final；side 不传 instance map。
            instance_map=None,
            return_breakdown=True,
        )
        losses.append(side_loss)
        for key, value in side_log.items():
            if key != "total":
                logs[f"side{j}/{key}"] = value

    # A0 不仅要“数学等价”，还要保留当前浮点求和顺序。
    total = losses[0]
    for side_loss in losses[1:]:
        total = total + side_loss
    total = total / (len(losses))
    return total, logs
```

### 7.6 运行元数据

必须把以下字段加入 `get_method_metadata(args)` 和 checkpoint semantic check：

```text
location_loss
side_location_loss
lambda_location
lambda_instance_location
lambda_instance_mass
lambda_unmatched_mass
component_radius_min
component_radius_scale
component_min_area
```

否则 resume 时可能把不同损失语义混在同一目录。

### 7.7 run 名称

示例：

```text
mshnet__loc_mass_cartesian__side_same__nuaa-sirst__seed_20260711
mshnet__loc_component_decoupled__lm_0.2__lu_0.05__nudt-sirst__seed_20260712
```

---

## 8. Phase 1：全局 root-cause 实验

### 8.1 实验矩阵

| ID | `--location-loss` | `--side-location-loss` | 解释 |
|---|---|---|---|
| A0 | `legacy` | `same` | 当前 canonical 语义 |
| A1 | `none` | `same` | 去掉所有 location |
| A2 | `mass_polar` | `same` | 仅修复 mass normalization，尽量保留 polar |
| A3 | `mass_cartesian` | `same` | 推荐全局形式 |
| A4 | `mass_cartesian` | `none` | final 改进；side 仅 scale-IoU |
| A5-control | `legacy` | `none` | 隔离 side LLoss 的影响 |

### 8.2 漏斗式调度

#### Gate 1：smoke

每个模式：

- NUAA-SIRST；
- seed `20260711`；
- 2 epoch；
- deterministic；
- 检查 loss finite、梯度 finite、checkpoint metadata、resume。

#### Gate 2：方向筛选

- 三个数据集；
- seed `20260711`；
- 完整训练协议；
- A0–A5 全部运行；
- 只看 internal validation。

#### Gate 3：复现

只保留至多两个非 baseline 模式：

- 三个数据集；
- seeds `20260711,20260712,20260713`；
- paired initialization、paired split、paired data order；
- 完成 clustered paired bootstrap。

### 8.3 示例命令

建议复制 `tools/run_clean_baselines.py` 为 `tools/run_location_decoupling.py`，增加 `--modes`，而不是手写九组 shell。

```bash
python tools/run_location_decoupling.py \
  --batch-id loc_root_cause_v1 \
  --datasets NUAA-SIRST,NUDT-SIRST,IRSTD-1K \
  --seeds 20260711 \
  --modes legacy,none,mass_polar,mass_cartesian,mass_cartesian_final_only \
  --gpus 0,1 \
  --epochs 400 \
  --deterministic true
```

Gate 3：

```bash
python tools/run_location_decoupling.py \
  --batch-id loc_root_cause_repl_v1 \
  --datasets NUAA-SIRST,NUDT-SIRST,IRSTD-1K \
  --seeds 20260711,20260712,20260713 \
  --modes legacy,mass_cartesian_final_only \
  --gpus 0,1 \
  --epochs 400 \
  --deterministic true
```

---

## 9. 新增 `tools/audit_location_losses.py`

该脚本不训练网络，先对损失的语义进行合成验证。

### 9.1 必须包含的六组测试

#### T1：幅度缩放

```text
GT：固定 3×3 component
Prediction：0.2×GT、0.5×GT、0.8×GT
```

记录 legacy、mass-polar、mass-Cartesian。预期 mass-normalized 两者近似不变。

#### T2：相同质心、不同面积

构造 1×1、3×3、5×5，同一质心；检查位置 loss 是否被面积改变。

#### T3：同向平移

目标向右平移 1、2、4、8 像素；检查 loss 单调性。

#### T4：双目标抵消

两个目标分别向左/右移动相同距离；global loss 与 component loss 对比。

#### T5：远距离虚警

正确目标外增加 1×1、3×3 FP；记录 loss、预测质量和 centroid 漂移。

#### T6：空目标与近零预测

确保无 NaN、无 Inf，梯度有限。

### 9.2 输出

```text
repro_runs/location_decoupling/audits/location_loss_semantics.json
repro_runs/location_decoupling/audits/location_loss_semantics.md
```

JSON 至少包含：

```json
{
  "source_commit": "...",
  "torch_version": "...",
  "device": "...",
  "cases": [...],
  "all_finite": true
}
```

---

## 10. Phase 2：实例级质量—位置解耦

只有 Phase 1 达到 GO 条件后实施。

### 10.1 目标形式

对每个 GT component `k`，定义固定的 GT-derived ownership region `R_k`：

\[
\widehat m_k=\sum_{i\in R_k}p_i,
\qquad
\widehat\mu_k=\frac{\sum_{i\in R_k}p_i x_i}{\widehat m_k+\epsilon}.
\]

三类损失分开：

\[
L_{\text{instance-loc}}
=\frac{1}{K}\sum_k
\operatorname{Huber}(\widehat\mu_k,\mu_k),
\]

\[
L_{\text{instance-mass}}
=\frac{1}{K}\sum_k
\left|\log(1+\widehat m_k)-\log(1+m_k^{gt})\right|,
\]

\[
L_{\text{unmatched}}
=\frac{\sum_i p_i\mathbf{1}[i\notin\cup_kR_k]}
{\sum_i\mathbf{1}[i\notin\cup_kR_k]+\epsilon}.
\]

最终：

\[
L=L_{\text{SLS-no-global-location}}
+\lambda_lL_{\text{instance-loc}}
+\lambda_mL_{\text{instance-mass}}
+\lambda_uL_{\text{unmatched}}.
\]

### 10.2 为什么先不用 predicted connected components

对预测做阈值和连通域会：

- 非可微；
- 引入阈值超参数；
- 使训练语义受组件匹配算法影响；
- 容易把目标分裂/合并问题混在一起。

第一版 ownership 必须完全由 GT component 生成，预测只以连续概率参与求和。

### 10.3 ownership region 的推荐构造

每个 component 获取：

- GT centroid `c_k`；
- area `A_k`；
- 等效半径 `r_k=sqrt(A_k/pi)`；
- 支持半径：

```python
R_k = max(component_radius_min, component_radius_scale * r_k)
```

对每个像素：

1. 计算到所有 component centroid 的距离；
2. 仅保留距离小于该 component 支持半径的候选；
3. 若多个 component 覆盖同一像素，分配给归一化距离 `d/R_k` 最小者；
4. 无候选者为 unmatched region 0。

这使 ownership 互斥，避免预测质量被多次计算。

### 10.4 `utils/data.py`：训练时返回 instance map

当前数据集在 augmentation 后将 mask 转 tensor。必须在所有几何变换之后做 connected components。

```python
# imports
import numpy as np
from skimage import measure

# __init__
self.return_instance_map = (
    bool(getattr(args, "return_instance_map", False))
    and mode == "train"
)

# __getitem__
img_tensor = self.transform(img)
mask_tensor = transforms.ToTensor()(mask)
mask_tensor = (mask_tensor > 0.5).to(torch.float32)

if not self.return_instance_map:
    return img_tensor, mask_tensor

instance_np = measure.label(
    mask_tensor[0].numpy().astype(np.uint8),
    connectivity=2,
    background=0,
).astype(np.int32)
instance_map = torch.from_numpy(instance_np)
return img_tensor, mask_tensor, instance_map
```

注意：

- 只在 train 返回 instance map，保持 val/test 两元组兼容；
- connectivity 与当前 component metric 保持一致；
- crop 将一个原目标切成可见 fragment 时，把 fragment 当作当前训练样本中的实例；
- 不默认删除 1-pixel component。

### 10.5 `utils/instance_targets.py` 推荐接口

```python
@dataclass
class OwnershipBatch:
    ownership: torch.Tensor       # B×H×W, 0=unmatched, 1..K=component
    centroids_xy: list[torch.Tensor]
    areas: list[torch.Tensor]
    component_counts: torch.Tensor


def build_component_ownership(
    instance_map: torch.Tensor,
    radius_min: float,
    radius_scale: float,
    min_area: int = 1,
) -> OwnershipBatch:
    ...
```

实现时允许对 batch 和 component 做 Python loop，因为每幅红外图目标数通常较少；像素距离计算应在 GPU 上向量化。

### 10.6 Component loss 骨架

```python
class ComponentMassLocationLoss(nn.Module):
    def __init__(
        self,
        lambda_location=1.0,
        lambda_mass=0.0,
        lambda_unmatched=0.0,
        radius_min=5.0,
        radius_scale=3.0,
        eps=1e-6,
    ):
        super().__init__()
        ...

    def forward(self, pred_prob, target, instance_map):
        ownership = build_component_ownership(...)
        x_grid, y_grid = normalized_xy_grid(pred_prob)

        loc_terms = []
        mass_terms = []
        unmatched_terms = []

        for b in range(pred_prob.shape[0]):
            p = pred_prob[b, 0]
            owner = ownership.ownership[b]
            ids = torch.unique(instance_map[b])
            ids = ids[ids > 0]

            for component_id in ids:
                gt_component = instance_map[b] == component_id
                region = owner == component_id
                gt_area = gt_component.sum().to(p.dtype)

                q = p * region.to(p.dtype)
                pred_mass = q.sum()
                safe_mass = pred_mass.clamp_min(self.eps)
                pred_cx = (q * x_grid[0, 0, 0]).sum() / safe_mass
                pred_cy = (q * y_grid[0, 0, :, 0]).sum() / safe_mass

                gt_weight = gt_component.to(p.dtype)
                gt_cx = (gt_weight * x_grid[0, 0, 0]).sum() / gt_area
                gt_cy = (gt_weight * y_grid[0, 0, :, 0]).sum() / gt_area

                loc_terms.append(
                    F.smooth_l1_loss(
                        torch.stack([pred_cx, pred_cy]),
                        torch.stack([gt_cx, gt_cy]),
                        beta=0.02,
                    )
                )
                mass_terms.append(
                    F.smooth_l1_loss(
                        torch.log1p(pred_mass),
                        torch.log1p(gt_area),
                    )
                )

            unmatched = owner == 0
            if unmatched.any():
                unmatched_terms.append(p[unmatched].mean())

        zero = pred_prob.new_zeros(())
        loc = torch.stack(loc_terms).mean() if loc_terms else zero
        mass = torch.stack(mass_terms).mean() if mass_terms else zero
        unmatched = (
            torch.stack(unmatched_terms).mean() if unmatched_terms else zero
        )

        total = (
            self.lambda_location * loc
            + self.lambda_mass * mass
            + self.lambda_unmatched * unmatched
        )
        return total, {
            "instance_location": loc.detach(),
            "instance_mass": mass.detach(),
            "unmatched_mass": unmatched.detach(),
        }
```

上面是接口骨架，不应直接复制后跳过测试。尤其要检查 grid broadcasting、component id 与 ownership id 的一致性。

### 10.7 不要一开始引入的复杂项

- Sinkhorn/optimal transport；
- learned ownership；
- predicted component matching；
- SCR weighting；
- focal hard-negative；
- boundary ring suppression；
- 多尺度 component assignment。

这些会使根因不可辨识，并与已有工作高度重叠。

---

## 11. Phase 2 消融矩阵

以 Phase 1 最佳 global control 为共同基线；推荐 final 使用 component loss，side 使用 scale-IoU、无 location。

| ID | Instance loc | Instance mass | Unmatched mass | 目的 |
|---|---:|---:|---:|---|
| C0 | 0 | 0 | 0 | 无 global location 的共同 control |
| C1 | ✓ | 0 | 0 | 只纠正实例位置 |
| C2 | ✓ | ✓ | 0 | 位置与目标质量解耦 |
| C3 | ✓ | 0 | ✓ | 位置与未匹配虚警解耦 |
| C4 | ✓ | ✓ | ✓ | 完整候选 |
| C5-control | 0 | 0 | ✓ | 判断收益是否仅来自全局背景压制 |

### 11.1 权重搜索原则

先固定：

```text
lambda_instance_location = 1.0
lambda_instance_mass ∈ {0.02, 0.05, 0.1}
lambda_unmatched_mass ∈ {0.01, 0.02, 0.05}
```

只在 development holdout 上选择一次。不要每个数据集独立选择权重；最终使用同一组超参数跨数据集。

### 11.2 权重健康检查

每 50 batch 记录：

```text
weighted_instance_loc / segmentation_loss
weighted_instance_mass / segmentation_loss
weighted_unmatched / segmentation_loss
```

建议初始目标范围：每一附加项中位数不超过 segmentation loss 的 10%；若长期超过 30%，先降权重而不是继续训练解释结果。

---

## 12. 日志与分析字段

### 12.1 batch/epoch loss

```text
final/seg
final/location
final/location_weighted
side0/seg ... side3/seg
instance/location
instance/mass
instance/unmatched
pred_mass_mean
target_mass_mean
global_centroid_l1
component_centroid_l1
component_mass_log_error
unmatched_probability_mass
```

### 12.2 validation 输出

复用 clean mechanism audit，并增加：

```text
image_id
dataset
seed
target_component_count
single_or_multi_target
target_area_min/mean/max
predicted_positive_mass
fp_component_count
fp_component_area
fn_component_count
matched_centroid_error_mean
unmatched_probability_mass
failure_taxonomy
```

### 12.3 必须报告的指标

- IoU；
- Pd；
- FA；
- FP component count/image；
- FP component area/image；
- FN component count；
- matched centroid error；
- threshold sweep 下的 Pd–FA 曲线；
- single-target 与 multi-target 子集；
- 按目标面积分层；
- paired image-level difference；
- 以 `(dataset, image_id)` 为 cluster 的 bootstrap CI。

单一阈值 0.5 的 FA 降低不够，必须排除纯 calibration shift。

---

## 13. 测试清单

### `tests/test_location_losses.py`

- `test_legacy_forward_exact`；
- `test_legacy_gradient_exact`；
- `test_mass_centroid_amplitude_invariance`；
- `test_cartesian_translation_monotonicity`；
- `test_empty_target_is_finite`；
- `test_tiny_prediction_is_finite`；
- `test_no_location_zero_gradient_contribution`；
- `test_return_breakdown_does_not_change_loss`。

### `tests/test_component_decoupling.py`

- component id permutation invariance；
- ownership regions mutually exclusive；
- every retained component has non-empty ownership；
- overlapping support assigned by normalized nearest distance；
- empty-target batch；
- mixed empty/non-empty batch；
- one-pixel component；
- two nearby components；
- component loss detects opposite-direction cancellation；
- unmatched mass reacts to a far FP；
- gradients finite；
- `lambda_* = 0` gives exact control loss。

### 完整回归

```bash
pytest -q tests/test_location_losses.py tests/test_component_decoupling.py
pytest -q
```

历史测试全部通过后才能启动 full runs。

---

## 14. GO / NO-GO 预注册

### 14.1 Phase 1 诊断 GO

满足以下全部条件：

1. 至少两个数据集上，A1/A2/A3/A4 相对 A0 的 FA 或 FP-component 指标呈相同方向；
2. 该变化在 threshold sweep 中仍存在，不只是固定阈值 calibration；
3. multi-target 子集变化大于 single-target 子集，或远距 FP/centroid audit 提供一致机制证据；
4. 至少两个 seed 复现；
5. 没有明显的训练不稳定或预测质量整体坍缩。

### 14.2 Phase 1 NO-GO

任一情形触发：

- 变化在数据集间随机翻转；
- FA 降低完全伴随等比例 Pd/IoU 降低；
- 所有收益在重新选择阈值后消失；
- mass normalization 对 loss 和梯度几乎无影响；
- 只有某一个 43-image holdout 出现微小提升。

### 14.3 最终方法 GO

建议冻结以下判定，不在看到结果后修改：

- 3 数据集 × 3 seed 中，至少 2/3 数据集的 median paired `ΔIoU >= 0`；
- 聚合 `ΔPd >= -0.002`；
- 至少 2/3 数据集相对 FA 或 FP-component count 降低 ≥ 10%；
- 任一数据集 FA 不得恶化超过 10%，除非 IoU/Pd 有清晰且预注册的 Pareto 补偿；
- multi-target 与 FP-heavy 子集的机制指标方向与主结果一致；
- 参数量和推理 FLOPs 与 canonical MSHNet 完全相同；
- official test 只在最终配置冻结后运行一次。

这些阈值可在首次运行前根据 baseline 方差调整一次，之后不得改动。

---

## 15. 主要风险与对应控制

| 风险 | 表现 | 必做控制 |
|---|---|---|
| 只是改 calibration | 0.5 阈值变好，曲线不变 | threshold sweep、固定 FA budget |
| unmatched loss 退化为全局背景抑制 | FA↓、Pd↓ | C5-control、目标邻域召回 |
| component crop 泄露 GT | 训练好、泛化差 | ownership 仅作 loss support，推理完全不使用 |
| 目标邻近时 ownership 错配 | 多目标性能波动 | 互斥 Voronoi、邻近目标专门分层 |
| mass target 等于面积过强 | 预测变厚/halo | log-mass、低权重、boundary/compactness 只做分析不加新 loss |
| 与 TDA 等实例 loss 重叠 | 审稿人认为只是 reweight | 强调 mass/location/unmatched 三变量显式解耦与机制证明 |
| 只修 MSHNet 实现细节 | 外部有效性不足 | 最终在至少两个额外 backbone 上验证 |

---

## 16. Server A 的提交序列

建议每一步独立 commit：

```text
A-00  pin source commit and add experiment manifest schema
A-01  preserve exact legacy SLS/LLoss path with identity tests
A-02  add mass-normalized global location losses and semantic audit
A-03  add CLI, metadata, logging, and root-cause runner
A-04  run smoke tests and commit only configs/manifests, not weights
A-05  add training-time instance maps and ownership utility
A-06  add component location loss and unit tests
A-07  add mass and unmatched terms with controls
A-08  add summarizer, paired bootstrap, and frozen GO/NO-GO report
```

每个 commit 的 PR/记录应包含：

- 改动文件；
- 新增测试；
- `pytest` 输出；
- exact CLI；
- source SHA；
- 是否改变 inference graph；
- 是否改变参数量/FLOPs。

---

## 17. 预期论文叙事

若主线成立，论文不应写成“我们进一步改进 MSHNet”。建议问题化表达：

> Existing location-sensitive objectives conflate where a target is with how much foreground mass is predicted, and global moments permit cancellation across multiple targets. We decouple instance localization, target mass, and unmatched foreground mass.

候选题目：

```text
Decoupling Target Mass and Localization for Infrared Small Target Segmentation
```

核心贡献应是：

1. 形式化 mass–location entanglement；
2. 合成和真实数据证明 global-moment cancellation；
3. 提出实例位置、实例质量、未匹配响应的可辨识分解；
4. 在多 backbone、多数据集上改善 Pd–FA Pareto，且零推理开销。

若最终只得到“将 `.mean()` 改为 `/sum(prob)` 有小幅提升”，不要作为 AAAI 主论文；它只能作为 baseline correction 或附录观察。

---

## 18. Server A 最终交付物

```text
1. source_commit.txt
2. location_loss_semantics.json/.md
3. A0–A5 root-cause manifest and summary
4. component ablation manifest and summary
5. per-image/component ledgers
6. paired bootstrap report
7. exact legacy identity report
8. final GO/NO-GO decision.md
9. frozen final config.yaml
10. official-test unlock record（仅最终 GO 后）
```

### 最重要的执行原则

> **先证明“位置损失确实制造了可重复的机制问题”，再提出实例级方法；不要先写复杂 loss，再从小验证集寻找支持它的现象。**
