# DEA-lite on Lliu666/MSHNet：基于官方源码的最小实现方案

> 适配仓库：`Lliu666/MSHNet`  
> 目标：第一版直接在 `model/MSHNet.py` 上最小修改，实现 DEA-lite / CERA-lite。  
> 当前只讨论 **模型结构、源码改法、forward 返回、loss 接口、train.py 适配**。  
> 暂不讨论数据集、实验表格、Pd、mIoU、F1、FA 曲线等实验设计。

---

## 0. 总结判断

这个方案是对的，而且非常适合直接在 `Lliu666/MSHNet` 的源码上实现。

原因是该仓库的 `model/MSHNet.py` 已经天然提供了 DEA-lite 需要的结构：

```python
self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

self.final = nn.Conv2d(4, 1, 3, 1, 1)
```

forward 里也已经有：

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

所以第一版不需要重写 backbone，不需要重写 decoder，也不需要换掉原始 `self.final`。只需要：

```text
1. 把上采样后的 s0/s1/s2/s3 保存成 scale_logits；
2. 用同一个 self.final 生成 z_full；
3. 训练时额外生成 z_only_i 和 z_empty；
4. 加一个很小的 decidability_head；
5. loss.py / main.py 中加入 DEA-lite loss。
```

---

## 1. 第一版方法名

建议代码层面叫：

```text
DEA-lite
```

论文概念层面可以叫：

```text
DEA: Decidable Evidence Attribution
```

第一版代码只实现 DEA 的最小可运行原型：

```text
DEA-lite = MSHNet + single-scale counterfactual outputs + decidability head
```

---

## 2. 第一版不要做什么

第一版先不要做：

```text
1. all-16 subset
2. component-compatible scale selector
3. positive necessity loss
4. candidate-level MLP verifier
5. inference-time d gate
6. 修改 encoder / decoder
7. 替换原始 self.final
```

原因：

```text
第一版的目标是快速验证：
MSHNet 的 false alarms 是否部分来自 single-scale evidence 的偶发激活。
```

如果一开始加入 all-16 subset、component selector、decidability gate，训练风险会显著升高，且很难定位问题来源。

---

## 3. 原始 MSHNet 的结构

原始结构可以抽象为：

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

DEA-lite 改成：

```text
s0 = mask0
s1 = up(mask1)
s2 = up_4(mask2)
s3 = up_8(mask3)

scale_logits = concat[s0, s1, s2, s3]
z_full = self.final(scale_logits)
```

其中：

```python
scale_logits.shape = [B, 4, H, W]
z_full.shape       = [B, 1, H, W]
```

---

## 4. DEA-lite 的核心结构

训练时额外构造：

```text
z_empty  = final([0, 0, 0, 0])
z_only_0 = final([s0, 0, 0, 0])
z_only_1 = final([0, s1, 0, 0])
z_only_2 = final([0, 0, s2, 0])
z_only_3 = final([0, 0, 0, s3])
```

然后构造 evidence decidability input：

```text
d_input = concat[
    z_full,
    max(z_only_0, z_only_1, z_only_2, z_only_3),
    var(z_only_0, z_only_1, z_only_2, z_only_3),
    s0, s1, s2, s3
]
```

注意通道数：

```text
z_full      : 1 channel
z_only_max  : 1 channel
z_only_var  : 1 channel
s0~s3       : 4 channels
-----------------------------
total       : 7 channels
```

所以 `decidability_head` 的第一层应该是：

```python
nn.Conv2d(7, 8, 3, padding=1)
```

不是：

```python
nn.Conv2d(6, 8, 3, padding=1)
```

---

## 5. `model/MSHNet.py` 修改方案

### 5.1 修改 `__init__`

在原始：

```python
self.final = nn.Conv2d(4, 1, 3, 1, 1)
```

后面加：

```python
self.decidability_head = nn.Sequential(
    nn.Conv2d(7, 8, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 1, kernel_size=1)
)
```

第一版使用 zero neutral，不额外加 learnable neutral token。

后续可以再做：

```text
zero neutral vs learnable neutral vs background-mean neutral
```

但第一版先不要复杂化。

---

### 5.2 新增 `build_dea_lite_outputs`

在 `MSHNet` 类里面新增：

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

### 5.3 修改 forward 函数签名

原始是：

```python
def forward(self, x, warm_flag):
```

改成：

```python
def forward(self, x, warm_flag, return_dea=False):
```

这样默认行为不变：

```python
return_dea=False
```

所以原始 test 代码不会受影响。

---

### 5.4 修改 warm_flag=True 分支

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

### 5.5 warm_flag=False 分支保持不变

原始：

```python
else:
    output = self.output_0(x_d0)
    return [], output
```

建议保持不变：

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

## 6. 完整 `MSHNet.py` 关键补丁版本

下面不是完整文件，只是关键改动片段。

```python
class MSHNet(nn.Module):
    def __init__(self, input_channels, block=ResNet):
        super().__init__()

        # 原始 MSHNet 初始化部分保持不变
        # ...

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)

        # DEA-lite head
        self.decidability_head = nn.Sequential(
            nn.Conv2d(7, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def build_dea_lite_outputs(self, scale_logits, z_full):
        neutral = torch.zeros_like(scale_logits)

        z_empty = self.final(neutral)

        z_only_list = []
        for i in range(4):
            e_only = neutral.clone()
            e_only[:, i:i+1] = scale_logits[:, i:i+1]
            z_only_i = self.final(e_only)
            z_only_list.append(z_only_i)

        z_only = torch.cat(z_only_list, dim=1)
        z_only_max = z_only.max(dim=1, keepdim=True)[0]
        z_only_var = z_only.var(dim=1, keepdim=True, unbiased=False)

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

    def forward(self, x, warm_flag, return_dea=False):
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
                dea_out = self.build_dea_lite_outputs(scale_logits, z_full)
                return [mask0, mask1, mask2, mask3], z_full, dea_out

            return [mask0, mask1, mask2, mask3], z_full

        else:
            output = self.output_0(x_d0)
            return [], output
```

---

## 7. `model/loss.py` 增加 DEA-lite loss

该仓库已有 `SLSIoULoss`。第一版不要替换它，只额外加 DEA-lite loss。

建议在 `model/loss.py` 末尾增加以下函数。

### 7.1 safe background mask

```python
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
    Empty evidence should not produce target response.
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

该仓库的训练逻辑大致是：

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

### 8.1 保持 warm-up 阶段不变

当：

```python
epoch <= self.warm_epoch
```

此时 `tag=False`，模型只输出 `output_0`，不要启用 DEA-lite。

### 8.2 warm-up 之后启用 DEA-lite

把训练阶段改成：

```python
if epoch > self.warm_epoch:
    tag = True
else:
    tag = False

if tag:
    masks, pred, dea_out = self.model(data, tag, return_dea=True)
else:
    masks, pred = self.model(data, tag)
```

然后原始 loss 保持：

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

注意：不要直接修改原始 `labels`，否则后面 DEA loss 需要 full-resolution GT 时会出错。用 `labels_for_scale`。

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

## 9. `main.py` 测试部分不用改

测试阶段保持：

```python
_, pred = self.model(data, tag)
```

第一版推理仍然只用 `z_full`，不使用 `decidability_logit` gate。

所以测试阶段不需要：

```python
return_dea=True
```

也不需要改变 evaluation。

---

## 10. 为什么第一版推理不使用 d？

第一版 `decidability_logit` 的作用是：

```text
1. 训练 final fusion 学会不要依赖单尺度虚假 evidence；
2. 提供 evidence decidability 的辅助监督；
3. 用于可视化和诊断。
```

暂时不让它直接调节 prediction，是为了避免：

```text
弱小真目标被 d 压掉，导致 Pd / Recall 下降。
```

等 DEA-lite loss 稳定后，再尝试：

```python
p = torch.sigmoid(pred)
d = torch.sigmoid(dea_out["decidability_logit"])
p_final = p * (0.5 + 0.5 * d)
```

但这不是第一版内容。

---

## 11. 训练权重建议

第一版建议权重：

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

---

## 12. 第一版实现 checklist

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
```

---

## 13. 后续升级路线

DEA-lite 跑通后，再依次加：

### 13.1 component-compatible sufficiency

新增：

```text
scale_mask = component_adaptive_scale_selector(gt)
z_comp = final(scale_mask * scale_logits + (1 - scale_mask) * neutral)
loss_comp: target 区域上 z_comp -> 1
```

这是完整 CERA / DEA 的正样本证据约束。

---

### 13.2 learnable neutral token

替换：

```python
neutral = torch.zeros_like(scale_logits)
```

为：

```python
self.neutral_token = nn.Parameter(torch.zeros(1, 4, 1, 1))
neutral = self.neutral_token.expand_as(scale_logits)
```

---

### 13.3 scale attribution decomposition

因为 `self.final` 是 `Conv2d(4, 1, 3, 1, 1)`，可以把四个输入通道拆成 contribution map：

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

### 13.4 inference-time decidability gate

稳定后可以尝试：

```python
p_final = sigmoid(z_full) * (0.5 + 0.5 * sigmoid(d_logit))
```

或：

```python
z_final = z_full + beta * logit(sigmoid(d_logit))
```

但第一版不要直接上。

---

## 14. 最终回答

这个方案对。

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
7. loss.py 加 single-scale anti-sufficiency loss 和 decidability loss；
8. main.py 只在 warm-up 之后启用 return_dea=True；
9. 推理阶段保持原始 z_full，不使用 d gate。
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
