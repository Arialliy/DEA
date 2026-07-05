# CERA: Counterfactual Evidence Regularization for Multi-Scale Infrared Small Target Detection

## 0. 一句话总结

**CERA** 不是简单改进 MSHNet，也不是给 MSHNet 继续加一个辅助 loss，而是从 MSHNet 的静态多尺度融合结构中抽象出一个更通用的问题：

> 多尺度 dense detector 的最终预测存在 **evidence ambiguity**：一个高置信目标响应可能来自真实目标的多尺度一致证据，也可能来自某个尺度分支的偶发背景激活。CERA 通过对多尺度 evidence branches 做反事实干预，约束最终预测必须由尺寸匹配、反事实可验证的尺度证据支撑，从而减少 hard clutter 下的 false alarms。

---

## 1. Motivation: 从 MSHNet 源码中发现的问题

MSHNet 的多尺度 head 结构大致如下：

```python
mask0 = self.output_0(x_d0)
mask1 = self.output_1(x_d1)
mask2 = self.output_2(x_d2)
mask3 = self.output_3(x_d3)

output = self.final(torch.cat([
    mask0,
    up(mask1),
    up_4(mask2),
    up_8(mask3)
], dim=1))
```

其中：

- `mask0` 来自高分辨率 decoder branch；
- `mask1 / mask2 / mask3` 来自更低分辨率 decoder branches；
- 四个尺度预测被上采样到同一尺寸后 concat；
- `final conv` 对四个 scale logits 做静态融合。

这个结构对小目标召回友好，但在 hard clutter 场景下存在一个隐含缺陷：

> final prediction 可以由某个尺度分支的局部强响应触发，而模型没有显式验证这个候选目标是否被多个尺度的合理证据支持。

因此，局部亮斑、边缘纹理、噪声块、背景碎片可能在某个尺度上强激活，并通过 final fusion 变成 false alarm。

---

## 2. Problem Definition

### 2.1 英文定义

> Multi-scale dense detectors suffer from evidence ambiguity: a high-confidence prediction can be caused either by target-compatible multi-scale evidence or by spurious single-branch background activation. Existing segmentation losses supervise output correctness but do not regularize whether the final prediction is counterfactually supported by scale-compatible evidence.

### 2.2 中文定义

多尺度 dense detector 的最终预测缺少 evidence source constraint。模型知道 final mask 对不对，但不知道：

- 这个预测是否由真实目标的多尺度一致证据支撑；
- 这个预测是否只是某个尺度分支的偶发背景激活；
- 去掉某些尺度证据后，该预测是否仍然成立；
- 只保留某个单一尺度时，背景 false alarm 是否仍然会被生成。

CERA 要解决的不是普通 test split 上 mIoU 再涨一点，而是：

> 在保持 Pd / Recall 的前提下，抑制 hard clutter 下由 scale-inconsistent evidence 引起的 false alarms。

---

## 3. Core Idea

设多尺度 evidence branches 为：

\[
E = \{e_0, e_1, e_2, e_3\}
\]

在 MSHNet 中：

\[
e_0 = mask0
\]

\[
e_1 = up(mask1)
\]

\[
e_2 = up_4(mask2)
\]

\[
e_3 = up_8(mask3)
\]

final prediction 为：

\[
z = F(E)
\]

其中 \(F\) 是 final fusion conv。

CERA 的核心是构造反事实输出：

\[
z_S = F(E_S, N_{\bar{S}})
\]

其中：

- \(S\) 是保留的尺度证据集合；
- \(\bar{S}\) 是被替换掉的尺度集合；
- \(N\) 是 neutral evidence；
- \(z_S\) 表示只使用尺度集合 \(S\) 时的反事实预测。

CERA 通过这些 counterfactual predictions 约束三件事：

1. **Positive Sufficiency**  
   真目标只保留尺寸匹配的尺度证据时，预测仍应成立。

2. **Positive Necessity**  
   真目标去掉尺寸匹配的尺度证据后，响应应明显下降。

3. **Background Anti-Sufficiency**  
   safe background 中，任何单一尺度证据都不应独立生成高置信目标。

---

## 4. 为什么不是 MSHNet Trick

CERA 从 MSHNet 的源码中发现问题，但抽象的是多尺度 dense detector 的通用问题。

只要模型满足：

\[
\{e_1, e_2, ..., e_K\} \rightarrow Fusion \rightarrow Final\ Prediction
\]

就可以使用 CERA。

适用对象包括：

- MSHNet：4 个 scale logits；
- FPN：不同 pyramid levels；
- U-Net side outputs：多个 decoder side predictions；
- DNANet / UIU-Net：nested multi-level outputs；
- Transformer-FPN hybrid：多层 decoder predictions。

论文中可以写：

> We instantiate CERA on MSHNet because its source code exposes a clean multi-scale head, but the formulation applies to any multi-branch dense detector with explicit fusion.

---

## 5. Method Overview

### 5.1 原始 MSHNet

原始 MSHNet 多尺度融合：

\[
z = F(e_0, e_1, e_2, e_3)
\]

其中四个尺度 evidence 被静态 concat，然后通过 final conv 得到最终 logit。

### 5.2 CERA-MSHNet

训练阶段，CERA 额外构造多个反事实输出：

\[
z_{\emptyset}, z_{\{0\}}, z_{\{1\}}, z_{\{2\}}, z_{\{3\}}, ..., z_{\{0,1,2,3\}}
\]

对于 4 个尺度，总共有 \(2^4=16\) 个 subset。

推理阶段，不需要 counterfactual branches，仍然使用原始输出：

\[
z = F(e_0, e_1, e_2, e_3)
\]

因此 CERA 可以做到：

> training-time regularization, no extra inference cost.

---

## 6. Neutral Evidence 设计

CERA 需要定义被移除尺度的 neutral evidence。

### 6.1 Zero-logit neutral

最简单版本：

\[
N_i = 0
\]

logit 为 0 对应 probability 0.5，可视为中性 evidence。

优点：

- 实现简单；
- 不增加参数；
- 适合第一版验证。

缺点：

- 可能和真实背景 logit 分布不完全一致。

### 6.2 Background-mean neutral

使用 safe background 区域的平均 logit：

\[
N_i = \operatorname{Mean}_{p \in B_{safe}} e_i(p)
\]

优点：

- 更接近真实背景 evidence；
- 反事实干预更自然。

### 6.3 Learnable neutral token

定义可学习 neutral token：

```python
self.neutral = nn.Parameter(torch.zeros(1, 4, 1, 1))
```

并加入 empty-evidence loss：

\[
L_{empty} = BCE(F(N), 0)
\]

防止 neutral token 自己变成 target evidence。

### 6.4 推荐落地版本

第一版建议使用：

> zero-logit neutral + empty evidence loss

原因是实现最小、最稳、容易 debug。

---

## 7. Target-Compatible Scale Set

对每个 GT component \(C\)，根据面积定义其 target-compatible scale set。

设目标面积为：

\[
A(C)=|C|
\]

MSHNet 四个尺度的 stride 近似为：

\[
r_0=1,\quad r_1=2,\quad r_2=4,\quad r_3=8
\]

目标在第 \(s\) 个尺度上的有效面积为：

\[
A_s(C)=\frac{A(C)}{r_s^2}
\]

定义：

\[
S(C)=\{s \mid A_s(C) \in [a_{min}, a_{max}]\}
\]

实践中可以用简化规则：

| Target Type | Compatible Scales |
|---|---|
| extremely tiny target | `{0, 1}` |
| small target | `{0, 1, 2}` |
| larger / fuzzy target | `{1, 2, 3}` |

注意不要写成“真小目标只依赖最高分辨率尺度”。更稳的说法是：

> A target should be supported by a compact set of scale-compatible branches rather than by an isolated incompatible branch.

---

## 8. Loss Design

总损失：

\[
L = L_{seg} + \lambda_s L_{suff} + \lambda_n L_{nec} + \lambda_a L_{anti} + \lambda_e L_{empty}
\]

其中：

- \(L_{seg}\)：原始 MSHNet segmentation / SLS loss；
- \(L_{suff}\)：positive sufficiency loss；
- \(L_{nec}\)：positive necessity loss；
- \(L_{anti}\)：background anti-sufficiency loss；
- \(L_{empty}\)：empty evidence suppression loss。

---

### 8.1 Positive Sufficiency Loss

对于真目标 component \(C\)，只保留 compatible scales \(S(C)\)：

\[
z_{S(C)} = F(E_{S(C)}, N_{\bar{S}(C)})
\]

要求真目标仍然被预测出来：

\[
z_{S(C)}(C) \rightarrow 1
\]

loss：

\[
L_{suff}
=
\sum_C BCEWithLogits(z_{S(C)}, Y) \quad \text{on pixels in } C
\]

含义：

> 真目标不能只靠完整多尺度融合才成立；只保留尺寸匹配的尺度证据时，它仍应成立。

---

### 8.2 Positive Necessity Loss

去掉 compatible scales：

\[
z_{\bar{S}(C)} = F(N_{S(C)}, E_{\bar{S}(C)})
\]

要求响应下降：

\[
m(z_{\bar{S}(C)}, C) < m(z, C)
\]

其中：

\[
m(z,C)=\frac{1}{|C|}\sum_{p\in C}z(p)
\]

margin loss：

\[
L_{nec}
=
\sum_C
\operatorname{softplus}
(
 m(z_{\bar{S}(C)}, C)
 -
 m(z, C)
 +
 \gamma_{nec}
)
\]

含义：

> 如果去掉目标尺寸匹配的尺度证据，模型就不应该还能高置信预测该目标。

---

### 8.3 Background Anti-Sufficiency Loss

对 safe background，任何单一尺度证据都不应该独立生成高置信目标。

单尺度反事实输出：

\[
z_{\{i\}} = F(E_i, N_{\bar{i}})
\]

对 safe background \(B\)：

\[
L_{anti}
=
\sum_i BCEWithLogits(z_{\{i\}}(B), 0)
\]

为了避免 easy background 淹没 loss，应只选择 hard safe background：

```python
safe_bg = 1 - dilate(gt)
hard_bg = safe_bg & (sigmoid(final_logit).detach() > tau)
```

或者：

```python
hard_bg = safe_bg & (
    sigmoid(z_only_0).detach() > tau |
    sigmoid(z_only_1).detach() > tau |
    sigmoid(z_only_2).detach() > tau |
    sigmoid(z_only_3).detach() > tau
)
```

含义：

> 背景亮斑、边缘纹理、噪声块即使在某个尺度上强响应，也不能通过单尺度 evidence 被 final fusion 当成目标。

---

### 8.4 Empty Evidence Loss

当所有尺度 evidence 都被替换为 neutral evidence 时，不应生成目标：

\[
z_{\emptyset} = F(N_0,N_1,N_2,N_3)
\]

\[
L_{empty}=BCEWithLogits(z_{\emptyset}, 0)
\]

含义：

> neutral evidence 本身不能携带 target prior。

---

## 9. MSHNet 源码修改方案

### 9.1 修改目标

把原来的 final fusion：

```python
output = self.final(torch.cat([mask0, up(mask1), up_4(mask2), up_8(mask3)], dim=1))
```

封装成：

```python
scale_logits = torch.cat([s0, s1, s2, s3], dim=1)
output = self.fuse(scale_logits)
```

训练时额外返回：

```python
cf_logits = self.counterfactual_fuse(scale_logits)
```

---

### 9.2 `__init__` 修改

```python
self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

self.fuse_conv = nn.Conv2d(4, 1, 3, 1, 1)

self.register_buffer("cf_masks", self._build_cf_masks(4))
```

---

### 9.3 添加 subset mask 构造函数

```python
def _build_cf_masks(self, k):
    masks = []
    for subset in range(2 ** k):
        m = [(subset >> i) & 1 for i in range(k)]
        masks.append(m)
    return torch.tensor(masks).float()
```

subset id 规则：

| subset id | active scales |
|---:|---|
| 0 | `{}` |
| 1 | `{0}` |
| 2 | `{1}` |
| 3 | `{0,1}` |
| 4 | `{2}` |
| 7 | `{0,1,2}` |
| 8 | `{3}` |
| 15 | `{0,1,2,3}` |

---

### 9.4 添加 fuse 函数

```python
def fuse(self, scale_logits):
    return self.fuse_conv(scale_logits)
```

---

### 9.5 添加 counterfactual_fuse 函数

```python
def counterfactual_fuse(self, scale_logits, neutral_type="zero"):
    """
    Args:
        scale_logits: Tensor, [B, 4, H, W]

    Returns:
        cf_logits: Tensor, [16, B, 1, H, W]

    subset id uses bit mask:
        0  = empty evidence
        1  = only scale 0
        2  = only scale 1
        3  = scales {0, 1}
        ...
        15 = full evidence
    """
    B, K, H, W = scale_logits.shape
    device = scale_logits.device

    masks = self.cf_masks.to(device).view(16, 1, K, 1, 1)

    if neutral_type == "zero":
        neutral = torch.zeros_like(scale_logits)
    else:
        raise NotImplementedError

    x_cf = scale_logits.unsqueeze(0) * masks + neutral.unsqueeze(0) * (1.0 - masks)
    x_cf = x_cf.reshape(16 * B, K, H, W)

    z_cf = self.fuse(x_cf)
    z_cf = z_cf.view(16, B, 1, H, W)

    return z_cf
```

---

### 9.6 forward 修改

```python
def forward(self, x, warm_flag, return_counterfactual=False):
    # encoder / decoder code omitted

    mask0 = self.output_0(x_d0)

    if not warm_flag:
        return [mask0], mask0

    mask1 = self.output_1(x_d1)
    mask2 = self.output_2(x_d2)
    mask3 = self.output_3(x_d3)

    s0 = mask0
    s1 = self.up(mask1)
    s2 = self.up_4(mask2)
    s3 = self.up_8(mask3)

    scale_logits = torch.cat([s0, s1, s2, s3], dim=1)

    output = self.fuse(scale_logits)

    if return_counterfactual:
        cf_logits = self.counterfactual_fuse(scale_logits)
        return [mask0, mask1, mask2, mask3], output, {
            "scale_logits": scale_logits,
            "cf_logits": cf_logits
        }

    return [mask0, mask1, mask2, mask3], output
```

---

## 10. CERA Loss 伪代码

```python
import torch
import torch.nn.functional as F


def cera_loss(
    final_logit,
    cf_logits,
    target,
    lambda_s=0.2,
    lambda_n=0.1,
    lambda_a=0.1,
    lambda_e=0.05,
    tau=0.3,
    margin=1.0,
):
    """
    Args:
        final_logit: [B, 1, H, W]
        cf_logits:   [16, B, 1, H, W]
        target:      [B, 1, H, W]

    Returns:
        loss_cera: scalar
        loss_dict: dict
    """

    target = target.float()

    z_empty = cf_logits[0]
    z_full = cf_logits[15]

    # --------------------------------------------------
    # 1. Empty evidence loss
    # --------------------------------------------------
    loss_empty = F.binary_cross_entropy_with_logits(
        z_empty,
        torch.zeros_like(target)
    )

    # --------------------------------------------------
    # 2. Safe-background hard mining
    # --------------------------------------------------
    target_dilated = F.max_pool2d(
        target,
        kernel_size=15,
        stride=1,
        padding=7
    )
    safe_bg = (target_dilated < 0.5).float()

    with torch.no_grad():
        hard_bg = ((torch.sigmoid(final_logit) > tau).float() * safe_bg)

    # --------------------------------------------------
    # 3. Background anti-sufficiency loss
    # --------------------------------------------------
    only_ids = [1, 2, 4, 8]
    loss_anti = 0.0
    denom_bg = hard_bg.sum() + 1e-6

    for sid in only_ids:
        z_only = cf_logits[sid]
        loss_map = F.binary_cross_entropy_with_logits(
            z_only,
            torch.zeros_like(target),
            reduction="none"
        )
        loss_anti = loss_anti + (loss_map * hard_bg).sum() / denom_bg

    loss_anti = loss_anti / len(only_ids)

    # --------------------------------------------------
    # 4. Positive sufficiency loss
    # 简化版：所有小目标默认 compatible set = {0,1,2}
    # paper-ready 版本应改成 component-adaptive compatible set
    # --------------------------------------------------
    pos = target
    denom_pos = pos.sum() + 1e-6

    sid_suff = 7          # {0,1,2}
    sid_nec = 15 ^ sid_suff

    z_suff = cf_logits[sid_suff]
    z_nec = cf_logits[sid_nec]

    loss_suff_map = F.binary_cross_entropy_with_logits(
        z_suff,
        target,
        reduction="none"
    )
    loss_suff = (loss_suff_map * pos).sum() / denom_pos

    # --------------------------------------------------
    # 5. Positive necessity loss
    # --------------------------------------------------
    full_pos_score = (final_logit * pos).sum() / denom_pos
    nec_pos_score = (z_nec * pos).sum() / denom_pos

    loss_nec = F.softplus(nec_pos_score - full_pos_score + margin)

    # --------------------------------------------------
    # 6. Total CERA loss
    # --------------------------------------------------
    loss_cera = (
        lambda_s * loss_suff
        + lambda_n * loss_nec
        + lambda_a * loss_anti
        + lambda_e * loss_empty
    )

    loss_dict = {
        "loss_cera_suff": loss_suff.detach(),
        "loss_cera_nec": loss_nec.detach(),
        "loss_cera_anti": loss_anti.detach(),
        "loss_cera_empty": loss_empty.detach(),
    }

    return loss_cera, loss_dict
```

---

## 11. 训练策略

### 11.1 Warm-up 阶段

MSHNet 原始训练中可能有 warm-up 阶段，只训练高分辨率输出 `mask0`。

CERA 不建议在 warm-up 阶段启用，因为此时没有完整多尺度 evidence。

策略：

```python
if warm_flag:
    enable CERA
else:
    only use original segmentation loss
```

---

### 11.2 Loss 权重建议

第一版建议：

```python
lambda_s = 0.2
lambda_n = 0.1
lambda_a = 0.1
lambda_e = 0.05
```

如果出现 Pd 下降：

- 降低 `lambda_n`；
- 降低 `lambda_a`；
- 使用 residual gate 或延迟启用 CERA。

如果 FA 没有下降：

- 提高 `lambda_a`；
- 提高 hard background threshold；
- 加入 branch-only hard mining。

---

### 11.3 延迟启用 CERA

建议前若干 epoch 只训练原始 MSHNet，然后再启用 CERA：

```python
if epoch >= cera_start_epoch:
    loss = loss_seg + loss_cera
else:
    loss = loss_seg
```

原因：

- early training 阶段 scale logits 不稳定；
- 太早强约束 counterfactual evidence 可能误伤 recall；
- 先让模型学到基本 target response，再做 evidence regularization 更稳。

---

## 12. 实验设计

### 12.1 主实验

主 baseline：

```text
MSHNet
MSHNet + safe-bg BCE
MSHNet + focal / hard negative loss
MSHNet + random scale dropout
MSHNet + scale consistency gate
MSHNet + CERA
```

目的：

- 证明 CERA 不是普通 hard negative loss；
- 证明 CERA 不是 scale dropout；
- 证明 CERA 的 counterfactual evidence regularization 更有效。

---

### 12.2 泛化实验

至少再做一个非 MSHNet baseline：

```text
U-Net side-output baseline + CERA
或
DNANet / UIU-Net style detector + CERA
```

目的：

> 证明 CERA 不是 MSHNet-specific trick，而是适用于 multi-branch dense detector 的通用训练原则。

---

### 12.3 主要指标

不要只看 mIoU / nIoU。

主指标排序：

1. `FA_ppm ↓`
2. `FP_components ↓`
3. `Precision ↑`
4. `Pd / Recall ≈ maintained`
5. `mIoU / nIoU ≥ baseline or comparable`

尤其要报告：

```text
HC-Val:
    FA_ppm
    FP_components
    Precision
    Pd
    mIoU / nIoU

Full Test:
    FA_ppm
    FP_components
    Precision
    Pd
    mIoU / nIoU
```

---

### 12.4 Fixed-Pd / Fixed-FA 评估

为了避免 reviewer 质疑“只是调阈值”，必须加：

```text
FA at fixed Pd
Precision at fixed Pd
Pd-FA curve
PR curve
```

主结论应该是：

> At matched Pd, CERA significantly reduces FA and FP components.

---

## 13. 消融实验

### 13.1 Loss ablation

```text
Baseline MSHNet
+ L_suff
+ L_suff + L_nec
+ L_suff + L_anti
+ L_suff + L_nec + L_anti
+ L_suff + L_nec + L_anti + L_empty
```

预期：

- `L_anti` 对 FA_ppm / FP_components 最关键；
- `L_suff` 对 Pd preservation 重要；
- `L_nec` 提高 evidence dependence；
- `L_empty` 防止 neutral evidence 退化。

---

### 13.2 Counterfactual subset ablation

```text
only single-scale subsets
only compatible-scale subsets
all 16 subsets
random subsets per iteration
```

预期：

- all 16 subsets 最完整，但训练稍慢；
- random subsets 更省显存；
- compatible-scale subsets 对正样本更有效；
- single-scale subsets 对背景 anti-sufficiency 更有效。

---

### 13.3 Neutral evidence ablation

```text
zero-logit neutral
background-mean neutral
learnable neutral token
```

预期：

- zero-logit neutral 是最简单强 baseline；
- background-mean neutral 可能更稳；
- learnable neutral token 需要 empty loss，否则容易退化。

---

### 13.4 与其他方法对比

```text
MSHNet + threshold tuning
MSHNet + area filter
MSHNet + safe-background BCE
MSHNet + focal loss
MSHNet + random scale dropout
MSHNet + final-logit-only hard negative suppression
MSHNet + CERA
```

关键比较：

> CERA should reduce FA at matched Pd better than threshold tuning, area filtering, and hard-negative pixel losses.

---

## 14. Diagnostic Analysis

CERA 论文需要一个 diagnostic section，证明 scale evidence ambiguity 确实存在。

### 14.1 分析对象

在 HC-Val 上，将 MSHNet baseline 的 predicted components 分成：

```text
TP components
FP components
```

### 14.2 分析指标

对每个 component，统计四个 scale logits 上的：

- peak response；
- mean response；
- response variance；
- centroid shift；
- area expansion ratio；
- single-scale dominance score；
- local background contrast。

### 14.3 预期发现

TP components：

```text
cross-scale response more stable
centroid shift smaller
area trajectory smoother
compatible scales mutually support each other
```

FP components：

```text
single-scale burst more frequent
centroid drift larger
area expansion or fragmentation stronger
background contrast abnormal
```

### 14.4 可视化

必须展示：

```text
input image
GT
MSHNet prediction
CERA prediction
scale logits from e0/e1/e2/e3
counterfactual only-scale predictions
counterfactual without-compatible-scale prediction
```

目标是让 reviewer 看到：

> CERA suppresses false alarms because those false alarms are not counterfactually supported by scale-compatible evidence.

---

## 15. 和 STV 的关系

STV 可以作为 motivation 和 diagnostic tool，但不建议作为最终主方法名。

推荐关系：

```text
Scale-Trajectory Analysis: 用于观察和证明 false alarms 的 scale evidence 不稳定。
CERA: 用反事实 evidence regularization 解决这个问题。
```

也就是说：

- STV 是现象分析和 baseline；
- CERA 是主方法；
- CERA 比 STV 更像 AAAI idea，因为它不依赖 handcrafted verifier，而是定义了一个通用训练原则。

---

## 16. 论文 Contribution 写法

### Contribution 1: Evidence Ambiguity Diagnosis

> We identify evidence ambiguity as a failure mode of multi-scale dense IR small target detectors: high-confidence predictions may arise from either target-compatible multi-scale evidence or spurious single-branch background activation.

### Contribution 2: Counterfactual Evidence Regularization

> We propose CERA, a training-time regularization framework that intervenes on scale evidence branches and enforces sufficiency, necessity, and background anti-sufficiency constraints.

### Contribution 3: MSHNet Instantiation with No Inference Overhead

> We instantiate CERA on MSHNet by counterfactually masking its four scale logits before final fusion. CERA is used only during training and preserves the original inference pipeline.

### Contribution 4: Hard-Clutter Reliability

> Experiments show that CERA suppresses hard-clutter false alarms and FP components while preserving Pd, improving the reliability of multi-scale IR small target detection.

---

## 17. 推荐论文标题

### Option 1

**CERA: Counterfactual Evidence Regularization for Reliable Infrared Small Target Detection**

### Option 2

**Counterfactual Evidence Regularization for Multi-Scale Infrared Small Target Detection**

### Option 3

**Mitigating Scale-Fragile False Alarms in Infrared Small Target Detection via Counterfactual Evidence Regularization**

### Option 4

**Evidence Matters: Counterfactual Multi-Scale Regularization for Infrared Small Target Detection**

最推荐：

> **CERA: Counterfactual Evidence Regularization for Reliable Infrared Small Target Detection**

---

## 18. 推荐 Abstract 草稿

Infrared small target detection remains challenging in hard-clutter scenes, where local bright structures and noise fragments can be easily confused with true targets. Recent multi-scale dense detectors such as MSHNet improve small-target recall by fusing predictions from multiple decoder scales. However, their static multi-scale fusion introduces an evidence ambiguity problem: a high-confidence prediction can be supported by target-compatible multi-scale evidence or triggered by spurious single-branch background activation. To address this issue, we propose Counterfactual Evidence Regularization (CERA), a training-time framework that intervenes on scale evidence branches before final fusion. CERA constructs counterfactual predictions by selectively preserving or neutralizing scale logits and enforces three constraints: positive sufficiency, positive necessity, and background anti-sufficiency. These constraints encourage true targets to be supported by scale-compatible evidence while preventing hard background regions from becoming false alarms through isolated scale responses. CERA introduces no additional inference cost and can be instantiated on any multi-branch dense detector with explicit fusion. Experiments on infrared small target detection benchmarks show that CERA significantly reduces hard-clutter false alarms and false-positive components while preserving detection probability and maintaining competitive segmentation accuracy.

---

## 19. 最终定位

这篇论文不要讲成：

> 我们改进了 MSHNet，让 mIoU 更高。

应该讲成：

> 我们从 MSHNet 的静态多尺度融合中发现 evidence ambiguity 问题，并提出一种通用的 counterfactual evidence regularization，使多尺度 dense detector 的预测必须由尺寸匹配、反事实可验证的尺度证据支撑，从而在保持 Pd 的同时显著减少 hard clutter false alarms。

最终主 claim：

> CERA reduces hard-clutter false alarms by preventing scale-fragile background activations from becoming final predictions, while preserving target detection probability.

---

## 20. 最小可行实验版本

第一阶段只需要完成：

```text
1. 修改 MSHNet forward，返回 scale_logits 和 cf_logits。
2. 实现 zero-logit neutral。
3. 实现 all-16-subset counterfactual fusion。
4. 实现 L_suff + L_anti + L_empty。
5. 在 HC-Val 上比较：
   - MSHNet
   - MSHNet + safe-bg BCE
   - MSHNet + random scale dropout
   - MSHNet + CERA
6. 报告 FA_ppm、FP_components、Precision、Pd、mIoU/nIoU。
```

如果第一阶段结果满足：

```text
FA_ppm 明显下降；
FP_components 明显下降；
Precision 明显提升；
Pd 基本不下降；
mIoU / nIoU 持平或小幅提升；
```

则 CERA 可以继续作为 AAAI 主线推进。

如果结果变成：

```text
FA 下降但 Pd 明显下降；
```

则需要降低 `lambda_a`、延迟启用 CERA，或者加强 positive sufficiency。

如果结果变成：

```text
Pd 上升但 FA 也上升；
```

则说明又回到了 CGA-v2 的失败模式，不能作为主线。

---

## 21. 最终建议

CERA 是比 CGA-v2.1 和 STV-MSHNet 更适合投稿 AAAI 的方向。

原因：

1. 它从 MSHNet 源码结构中发现真实问题；
2. 它把问题抽象成多尺度 dense detector 的通用 evidence ambiguity；
3. 它不是简单外挂模块，而是反事实训练原则；
4. 它训练时生效、推理时无额外开销；
5. 它的主指标与 IRST 部署需求一致：降低 FA、减少 FP components、保持 Pd；
6. 它的消融清晰，可以直接验证 sufficiency、necessity、anti-sufficiency 三个约束是否有效。

最终推荐主线：

> **CERA: Counterfactual Evidence Regularization for Reliable Multi-Scale Infrared Small Target Detection**
