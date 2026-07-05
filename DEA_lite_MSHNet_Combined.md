# DEA-lite on Lliu666/MSHNet：作用、源码改法、第一版验证路线（结合版）

> 适配源码：`https://github.com/Lliu666/MSHNet`  
> 第一版目标：不重写 backbone、不重写 decoder、不改变测试推理流程，只在 MSHNet 原始四尺度融合位置加入 **single-scale counterfactual evidence paths** 和一个轻量 **decidability head**。  
> 当前版本重点：模型结构、源码修改、loss 接口、训练接入、第一版验证逻辑、数据集顺序和后续指标方向。  
> 第一版名称建议：`DEA-lite`。完整论文概念可以继续叫 `DEA: Decidable Evidence Attribution`。

---

## 0. 一句话结论

这个方案是对的，第一版可以直接在 `Lliu666/MSHNet` 的 `model/MSHNet.py` 上做最小修改。

**DEA-lite 的作用不是换 backbone，而是验证并抑制 MSHNet 中一种具体 failure mode：**

> 一部分 hard-background false alarms 可能具有 **single-scale sufficiency**：只保留某一个尺度分支时，背景区域仍然能被 final fusion 预测成目标。DEA-lite 通过训练时反事实 evidence paths 抑制这种单尺度误触发，使 MSHNet 的 final fusion 不再过度依赖单个尺度分支的偶发背景激活。

更严谨地说，DEA-lite 不是严格证明所有 false alarm 都来自单尺度激活，而是检查：

```text
这个 false alarm 是否在只保留某个尺度 evidence 时仍然成立？
```

如果成立，则它可以被视为：

```text
evidence-underdetermined / scale-fragile false alarm candidate
```

---

## 1. all-16 subset 是什么？为什么第一版不要做？

`all-16 subset` 不是数据集，也不是 NUAA / NUDT / IRSTD-1K。

它指的是：MSHNet 有 4 个尺度分支：

```text
s0, s1, s2, s3
```

如果每个尺度都做“保留 / 去掉”二选一，就有：

\[
2^4 = 16
\]

种组合，例如：

```text
{}
{s0}
{s1}
{s2}
{s3}
{s0, s1}
{s0, s2}
{s1, s2}
...
{s0, s1, s2, s3}
```

第一版不建议把 all-16 subset 作为主方法，因为：

```text
1. 它像暴力 regularization；
2. 推广到 K 个尺度会变成 2^K；
3. 很多组合没有明确语义，比如 {s0, s3}；
4. 训练失败时很难定位是哪类 counterfactual path 出问题。
```

所以 DEA-lite 第一版只保留最有语义的路径：

```text
1. full evidence:    z_full
2. empty evidence:   z_empty
3. single-scale:     z_only_0, z_only_1, z_only_2, z_only_3
```

后续 DEA-full 再考虑：

```text
component-compatible evidence: z_comp
scale attribution decomposition
inference-time decidability gate
```

---

## 2. 原始 MSHNet 结构

`Lliu666/MSHNet` 的 `model/MSHNet.py` 已经有四个尺度输出头：

```python
self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

self.final = nn.Conv2d(4, 1, 3, 1, 1)
```

`warm_flag=True` 时，原始 MSHNet 会做：

```python
mask0 = self.output_0(x_d0)
mask1 = self.output_1(x_d1)
mask2 = self.output_2(x_d2)
mask3 = self.output_3(x_d3)

output = self.final(torch.cat([
    mask0,
    self.up(mask1),
    self.up_4(mask2),
    self.up_8(mask3)
], dim=1))
```

可以抽象成：

```text
Input image
    ↓
Encoder / Decoder
    ↓
x_d0, x_d1, x_d2, x_d3
    ↓
output_0, output_1, output_2, output_3
    ↓
mask0, mask1, mask2, mask3
    ↓
upsample to same resolution
    ↓
concat [mask0, up(mask1), up_4(mask2), up_8(mask3)]
    ↓
self.final
    ↓
output
```

这正好给 DEA-lite 提供了天然入口：

```text
把上采样后的四个 scale logits 显式保存下来，然后对它们做反事实干预。
```

---

## 3. DEA-lite 的核心作用

原始 MSHNet 只做完整多尺度融合：

```text
s0, s1, s2, s3 → self.final → z_full
```

DEA-lite 额外检查：

```text
只用 s0 会不会也预测成目标？
只用 s1 会不会也预测成目标？
只用 s2 会不会也预测成目标？
只用 s3 会不会也预测成目标？
完全不用尺度 evidence，会不会也预测成目标？
```

对应输出：

```text
z_only_0 = final([s0, 0,  0,  0])
z_only_1 = final([0,  s1, 0,  0])
z_only_2 = final([0,  0,  s2, 0])
z_only_3 = final([0,  0,  0,  s3])
z_empty  = final([0,  0,  0,  0])
```

其中：

```text
z_only_i: 检查单个尺度 evidence 是否足以独立触发目标响应。
z_empty:  检查 zero-neutral evidence 或 final conv bias 是否会自己产生目标响应。
```

训练时，如果 safe background 区域在 `z_only_i` 上也出现高响应，就说明它可能是：

```text
single-scale sufficient false alarm
```

DEA-lite 用辅助 loss 压掉这种响应。

---

## 4. 第一版不要做什么

第一版先不要做：

```text
1. all-16 subset
2. component-compatible scale selector
3. positive necessity loss
4. candidate-level MLP verifier
5. inference-time d gate
6. 改 encoder / decoder
7. 替换原始 self.final
8. 复杂 scale attribution loss
```

原因：第一版目标是快速验证 hypothesis，而不是一次性做完整 DEA。

第一版应该验证：

```text
MSHNet 的一部分 false alarms 是否具有 single-scale sufficient response？
用辅助 loss 压制这种 single-scale response 后，false alarm 是否下降且 Pd 不明显掉？
```

---

## 5. DEA-lite 模型结构

原始：

```text
scale_logits = concat[s0, s1, s2, s3]
z_full = self.final(scale_logits)
```

DEA-lite：

```text
scale_logits = concat[s0, s1, s2, s3]
z_full = self.final(scale_logits)

z_empty  = self.final([0, 0, 0, 0])
z_only_0 = self.final([s0, 0, 0, 0])
z_only_1 = self.final([0, s1, 0, 0])
z_only_2 = self.final([0, 0, s2, 0])
z_only_3 = self.final([0, 0, 0, s3])
```

然后把这些 evidence-related signals 输入一个轻量 decidability head：

```text
d_input = concat[
    z_full,
    max(z_only_0, z_only_1, z_only_2, z_only_3),
    var(z_only_0, z_only_1, z_only_2, z_only_3),
    s0, s1, s2, s3
]
```

通道数是：

```text
z_full      : 1 channel
z_only_max  : 1 channel
z_only_var  : 1 channel
s0~s3       : 4 channels
-----------------------------
total       : 7 channels
```

所以第一层必须是：

```python
nn.Conv2d(7, 8, 3, padding=1)
```

不是：

```python
nn.Conv2d(6, 8, 3, padding=1)
```

---

## 6. `model/MSHNet.py` 修改方案

### 6.1 修改 `__init__`

在原始：

```python
self.final = nn.Conv2d(4, 1, 3, 1, 1)
```

后面新增：

```python
self.decidability_head = nn.Sequential(
    nn.Conv2d(7, 8, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 1, kernel_size=1)
)
```

第一版使用 zero neutral：

```python
neutral = torch.zeros_like(scale_logits)
```

后续可以再做 ablation：

```text
zero neutral vs learnable neutral vs background-mean neutral
```

---

### 6.2 新增 `build_dea_lite_outputs`

在 `MSHNet` 类里新增：

```python
def build_dea_lite_outputs(self, scale_logits, z_full):
    """
    Build DEA-lite counterfactual outputs.

    Args:
        scale_logits: Tensor, [B, 4, H, W]
        z_full:       Tensor, [B, 1, H, W]

    Returns:
        dict with z_empty, z_only, z_only_max, z_only_var, decidability_logit
    """
    neutral = torch.zeros_like(scale_logits)

    # Empty evidence path: final([0, 0, 0, 0])
    z_empty = self.final(neutral)

    # Single-scale evidence paths
    z_only_list = []
    for i in range(4):
        e_only = neutral.clone()
        e_only[:, i:i+1] = scale_logits[:, i:i+1]
        z_only_i = self.final(e_only)
        z_only_list.append(z_only_i)

    # z_only: [B, 4, H, W]
    z_only = torch.cat(z_only_list, dim=1)

    # Single-scale statistics
    z_only_max = z_only.max(dim=1, keepdim=True)[0]
    z_only_var = z_only.var(dim=1, keepdim=True, unbiased=False)

    # Decidability input: 1 + 1 + 1 + 4 = 7 channels
    d_input = torch.cat([
        z_full,
        z_only_max,
        z_only_var,
        scale_logits,
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
```

---

### 6.3 修改 forward 函数签名

原始：

```python
def forward(self, x, warm_flag):
```

改成：

```python
def forward(self, x, warm_flag, return_dea=False):
```

默认：

```python
return_dea=False
```

所以原始测试代码不受影响。

---

### 6.4 修改 `warm_flag=True` 分支

原始：

```python
if warm_flag:
    mask0 = self.output_0(x_d0)
    mask1 = self.output_1(x_d1)
    mask2 = self.output_2(x_d2)
    mask3 = self.output_3(x_d3)

    output = self.final(torch.cat([
        mask0,
        self.up(mask1),
        self.up_4(mask2),
        self.up_8(mask3)
    ], dim=1))

    return [mask0, mask1, mask2, mask3], output
```

改成：

```python
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
        dea_out = self.build_dea_lite_outputs(scale_logits, z_full)
        return [mask0, mask1, mask2, mask3], z_full, dea_out

    return [mask0, mask1, mask2, mask3], z_full
```

---

### 6.5 `warm_flag=False` 分支保持不变

原始：

```python
else:
    output = self.output_0(x_d0)
    return [], output
```

保持不变：

```python
else:
    output = self.output_0(x_d0)
    return [], output
```

原因：

```text
warm_flag=False 是原始 MSHNet 的 warm-up / single-output 阶段。
这个阶段没有四尺度 fusion，不适合启用 DEA-lite。
```

---

## 7. `model/loss.py` 增加 DEA-lite loss

第一版不要替换原始 `SLSIoULoss`，只增加辅助 loss。

### 7.1 safe background mask

```python
import torch
import torch.nn.functional as F


def build_safe_bg(gt, kernel_size=15):
    """
    Args:
        gt: Tensor, [B, 1, H, W], binary mask
    Returns:
        safe_bg: Tensor, [B, 1, H, W]
    """
    pad = kernel_size // 2
    gt_dilate = F.max_pool2d(
        gt.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )
    safe_bg = (gt_dilate < 0.5).float()
    return safe_bg
```

---

### 7.2 single-scale anti-sufficiency loss

```python
def single_scale_anti_sufficiency_loss(z_only_max, z_full, gt, tau=0.3):
    """
    Penalize single-scale evidence that independently produces high response
    on hard safe-background regions.

    Args:
        z_only_max: Tensor, [B, 1, H, W]
        z_full:     Tensor, [B, 1, H, W]
        gt:         Tensor, [B, 1, H, W]
    """
    safe_bg = build_safe_bg(gt)

    with torch.no_grad():
        hard_bg_from_full = (torch.sigmoid(z_full) > tau).float()
        hard_bg_from_only = (torch.sigmoid(z_only_max) > tau).float()
        hard_bg = safe_bg * torch.clamp(hard_bg_from_full + hard_bg_from_only, max=1.0)

    loss_map = F.binary_cross_entropy_with_logits(
        z_only_max,
        torch.zeros_like(z_only_max),
        reduction="none",
    )

    loss = (loss_map * hard_bg).sum() / (hard_bg.sum() + 1e-6)
    return loss
```

---

### 7.3 empty evidence suppression loss

```python
def empty_evidence_loss(z_empty):
    """
    Empty / zero-neutral evidence should not produce target response.
    """
    return F.binary_cross_entropy_with_logits(
        z_empty,
        torch.zeros_like(z_empty),
    )
```

---

### 7.4 decidability loss

```python
def decidability_loss(d_logit, z_full, gt, tau=0.3):
    """
    Positive:
        GT target regions -> decidability = 1
    Negative:
        hard safe-background predicted regions -> decidability = 0
    Ignore:
        easy background
    """
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
```

---

### 7.5 DEA-lite total auxiliary loss

```python
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
```

---

## 8. `main.py` 训练部分怎么改

原始训练大致是：

```python
masks, pred = self.model(data, tag)
loss = 0
loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
for j in range(len(masks)):
    if j > 0:
        labels = self.down(labels)
    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch)
loss = loss / (len(masks) + 1)
```

### 8.1 warm-up 阶段不启用 DEA-lite

当：

```python
epoch <= self.warm_epoch
```

此时：

```python
tag = False
```

模型只输出 `output_0`，不要启用 DEA-lite。

---

### 8.2 warm-up 之后启用 DEA-lite

```python
tag = epoch > self.warm_epoch

if tag:
    masks, pred, dea_out = self.model(data, tag, return_dea=True)
else:
    masks, pred = self.model(data, tag)
    dea_out = None
```

原始 loss 保持，但注意不要覆盖 full-resolution `labels`：

```python
loss = 0
loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)

labels_for_scale = labels
for j in range(len(masks)):
    if j > 0:
        labels_for_scale = self.down(labels_for_scale)
    loss = loss + self.loss_fun(masks[j], labels_for_scale, self.warm_epoch, epoch)

loss = loss / (len(masks) + 1)
```

然后加 DEA-lite loss：

```python
if tag:
    loss_dea, dea_log = dea_lite_loss(
        dea_out,
        pred,
        labels,
        lambda_single=0.10,
        lambda_dec=0.05,
        lambda_empty=0.01,
        tau=0.3,
    )
    loss = loss + loss_dea
```

完整训练片段：

```python
for i, (data, mask) in enumerate(tbar):
    data = data.to(self.device)
    labels = mask.to(self.device)

    tag = epoch > self.warm_epoch

    if tag:
        masks, pred, dea_out = self.model(data, tag, return_dea=True)
    else:
        masks, pred = self.model(data, tag)
        dea_out = None

    # Original MSHNet loss
    loss = 0
    loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)

    labels_for_scale = labels
    for j in range(len(masks)):
        if j > 0:
            labels_for_scale = self.down(labels_for_scale)
        loss = loss + self.loss_fun(masks[j], labels_for_scale, self.warm_epoch, epoch)

    loss = loss / (len(masks) + 1)

    # DEA-lite auxiliary loss
    if tag:
        loss_dea, dea_log = dea_lite_loss(
            dea_out,
            pred,
            labels,
            lambda_single=0.10,
            lambda_dec=0.05,
            lambda_empty=0.01,
            tau=0.3,
        )
        loss = loss + loss_dea

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    losses.update(loss.item(), pred.size(0))
    tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
```

---

## 9. 测试阶段第一版不用改

第一版推理仍然只用：

```python
_, pred = self.model(data, tag)
```

不要启用：

```python
return_dea=True
```

也不要直接用：

```python
p_final = sigmoid(pred) * sigmoid(d_logit)
```

原因：

```text
第一版 d_logit 只作为 auxiliary evidence-decidability supervision 和诊断输出。
如果推理时直接 gate，弱小真目标可能被 d 压掉，导致 Pd / Recall 下降。
```

稳定后再尝试：

```python
p = torch.sigmoid(pred)
d = torch.sigmoid(dea_out["decidability_logit"])
p_final = p * (0.5 + 0.5 * d)
```

这不是第一版内容。

---

## 10. 数据集顺序建议

原始 Markdown 里暂不讨论数据集和实验表，这是对的；第一版应该先把源码和 loss 跑通。

但结合 `Lliu666/MSHNet` 仓库和常见 IR small target 数据集，建议实际运行顺序是：

```text
1. IRSTD-1K
2. NUDT-SIRST
3. NUAA-SIRST，可选
```

### 10.1 为什么先 IRSTD-1K？

```text
1. 仓库 README 的训练 / 测试示例默认使用 IRSTD-1k；
2. README 里报告了 IRSTD-1k 的 mIoU / Pd / Fa；
3. 先用官方默认数据集最容易排除工程错误。
```

第一阶段目标不是刷榜，而是确认：

```text
代码能跑；
loss 不爆；
z_only_i 有实际响应；
z_empty 不异常；
d_logit 能学出 target / hard background 区分。
```

### 10.2 为什么第二个跑 NUDT-SIRST？

```text
1. README 也报告了 NUDT-SIRST；
2. 如果 DEA-lite 只在 IRSTD-1K 有效，泛化 claim 不够；
3. NUDT-SIRST 可以验证 single-scale anti-sufficiency 是否跨数据集有效。
```

### 10.3 NUAA-SIRST 怎么看？

```text
NUAA-SIRST 可以作为后续补充泛化数据集。
第一版时间紧时可以不优先做。
```

如果本地 `repro_data` 确实是：

```text
IRSTD-1K:   trainval 800, test 201
NUDT-SIRST: trainval 697, test 664
```

那么第一版实验顺序可以定为：

```text
先 IRSTD-1K 跑通 DEA-lite；
再 NUDT-SIRST 验证 false alarm 是否下降；
最后考虑 NUAA-SIRST。
```

---

## 11. 后续评估指标方向

虽然第一版文档重点是源码改法，不做完整实验表，但后续验证 DEA-lite 时不能只看 mIoU。

至少要看：

```text
mIoU
Pd
Fa
Precision
F1
FP components
```

其中最关键的是：

```text
Fa 是否下降；
FP components 是否下降；
Pd 是否基本保持；
Precision / F1 是否提升；
mIoU 是否不明显下降。
```

DEA-lite 的核心目标不是单纯提高 mIoU，而是：

```text
在不明显牺牲 Pd 的前提下，降低 single-scale false alarms。
```

所以后续最好报告：

```text
1. Full test set: mIoU / Pd / Fa / F1
2. Hard clutter subset: Fa / FP components / Precision / Pd
3. 可视化: z_full, z_only_i, z_empty, d_logit
```

---

## 12. loss 权重建议

第一版建议：

```python
lambda_single = 0.10
lambda_dec    = 0.05
lambda_empty  = 0.01
```

如果训练不稳定，改成：

```python
lambda_single = 0.05
lambda_dec    = 0.02
lambda_empty  = 0.005
```

也可以做 warm-up ramp：

```python
ratio = min(1.0, (epoch - self.warm_epoch) / 20.0)

lambda_single = 0.10 * ratio
lambda_dec    = 0.05 * ratio
lambda_empty  = 0.01 * ratio
```

推荐第一版使用 ramp，因为 DEA-lite 是 auxiliary objective，不要一开始就强压主分割学习。

---

## 13. 第一版实现 checklist

### `model/MSHNet.py`

```text
[ ] forward 加 return_dea=False 参数
[ ] warm_flag=True 分支中保存 s0/s1/s2/s3
[ ] 构造 scale_logits
[ ] z_full = self.final(scale_logits)
[ ] 新增 decidability_head: Conv2d(7, 8, 3, padding=1)
[ ] 新增 build_dea_lite_outputs
[ ] return_dea=True 时返回 dea_out
[ ] warm_flag=False 分支保持不变
```

### `model/loss.py`

```text
[ ] build_safe_bg
[ ] single_scale_anti_sufficiency_loss
[ ] empty_evidence_loss
[ ] decidability_loss
[ ] dea_lite_loss
```

### `main.py`

```text
[ ] warm-up 前保持原始训练
[ ] warm-up 后调用 model(data, tag, return_dea=True)
[ ] 原始 SLSIoULoss 保持不变
[ ] 加 dea_lite_loss
[ ] 测试阶段不改
[ ] 不要覆盖 full-resolution labels
```

### 可视化 / Debug

```text
[ ] 保存 z_full
[ ] 保存 z_only_0~z_only_3
[ ] 保存 z_empty
[ ] 保存 d_logit
[ ] 对比 FP component 上的 z_only_i 响应
```

---

## 14. 第一版预期验证什么？

DEA-lite 第一版应该回答三个问题：

### Q1：MSHNet 的 false alarms 是否存在 single-scale sufficient response？

看：

```text
FP component 上，z_only_i 是否仍然高响应？
哪个尺度最容易单独误触发？
```

### Q2：single-scale anti-sufficiency loss 是否能压掉这种响应？

看：

```text
训练后 z_only_max 在 safe background 上是否降低？
FP components 是否减少？
Fa 是否下降？
```

### Q3：压 false alarm 时是否误伤 target detection？

看：

```text
Pd 是否基本保持？
Recall 是否明显下降？
小目标是否被 d_logit 错误判为 undecidable？
```

如果答案是：

```text
z_only_i 能解释一部分 FP；
DEA-lite 压低 z_only_i 后 FP / Fa 降；
Pd 基本不掉；
```

那么 DEA-lite 的核心 hypothesis 成立。

---

## 15. 后续升级路线

### 15.1 component-compatible sufficiency

新增：

```text
scale_mask = component_adaptive_scale_selector(gt)
z_comp = final(scale_mask * scale_logits + (1 - scale_mask) * neutral)
loss_comp: target 区域上 z_comp -> 1
```

这是完整 CERA / DEA 的正样本证据约束。

---

### 15.2 learnable neutral token

把：

```python
neutral = torch.zeros_like(scale_logits)
```

替换为：

```python
self.neutral_token = nn.Parameter(torch.zeros(1, 4, 1, 1))
neutral = self.neutral_token.expand_as(scale_logits)
```

再比较：

```text
zero neutral
learnable neutral
background-mean neutral
```

---

### 15.3 scale attribution decomposition

因为 `self.final` 是：

```python
nn.Conv2d(4, 1, 3, 1, 1)
```

可以拆成四个尺度 contribution map：

```text
z = bias + c0 + c1 + c2 + c3
```

后续 DEA 完整版可以让 `decidability_head` 输入：

```text
z_full
z_empty
z_only_max
z_only_var
attr_max
attr_entropy
scale_logits
```

---

### 15.4 inference-time decidability gate

稳定后再尝试：

```python
p_final = sigmoid(z_full) * (0.5 + 0.5 * sigmoid(d_logit))
```

或：

```python
z_final = z_full + beta * logit(sigmoid(d_logit))
```

第一版不要直接上。

---

## 16. 最终版本总结

第一版最推荐实现为：

```text
DEA-lite on MSHNet
```

具体做法：

```text
1. 直接在 Lliu666/MSHNet 的 model/MSHNet.py 里改；
2. 保留原始 output_0~output_3 和 self.final；
3. 在 warm_flag=True 分支里构造 scale_logits；
4. z_full = self.final(scale_logits)；
5. 训练时额外构造 z_only_0~z_only_3 和 z_empty；
6. 用 7-channel input 预测 decidability_logit；
7. loss.py 加 single-scale anti-sufficiency loss、empty loss、decidability loss；
8. main.py 只在 warm-up 之后启用 return_dea=True；
9. 推理阶段保持原始 z_full，不使用 d gate；
10. 先在 IRSTD-1K 跑通，再在 NUDT-SIRST 验证，NUAA-SIRST 后续可选。
```

工程难度：

```text
MSHNet.py 修改：低
loss.py 修改：中等
main.py 修改：中等
整体难度：3 / 5
```

这一版足够验证 DEA 的核心想法：

> 高置信小目标预测不应仅由某个单尺度 evidence 独立支撑；模型应学习区分 target-supported prediction 和 evidence-underdetermined prediction。

---

## 17. 最推荐写进论文 / 开题文档的一句话

中文：

> DEA-lite 在 MSHNet 原始四尺度融合结构上增加训练时反事实 evidence paths，通过 `z_only_i` 检查并抑制 single-scale sufficient background responses，通过 `z_empty` 抑制 zero-neutral evidence 下的伪响应，并用轻量 decidability head 学习预测响应是否具有可靠证据。第一版不改变 backbone，不改变推理输出，主要用于验证和降低 MSHNet 的单尺度误触发型 false alarms。

英文：

> DEA-lite augments the original four-scale fusion of MSHNet with training-time counterfactual evidence paths. By preserving only one scale branch at a time, it diagnoses and suppresses single-scale sufficient background responses, while a lightweight decidability head learns whether a target response is supported by reliable evidence. The first version keeps the original backbone and inference output unchanged, and focuses on reducing single-scale false alarms without sacrificing target detection probability.
