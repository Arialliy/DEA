# FullDEA-v3-TPS 下一步修改方案：优先模块与网络结构

当前不建议继续跑 P3，也不建议继续在 FullDEA-v2 上大范围调参。已有结果说明：

```text
FullDEA-v2 已经能降 FA；
但它没有可靠保护弱目标 / 目标边界 / 低置信目标；
所以降 FA 的同时把 PD 和 IoU 一起压下去了。
```

matched MSHNet finetune 也出现了类似方向：

```text
matched MSHNet finetune:
IoU -0.000477
PD  -0.003777
FA  -2.923377

FullDEA-v2:
IoU -0.000877
PD  -0.007577
FA  -5.989477
```

这说明有两层问题：

```text
第一层：从 baseline checkpoint 继续 finetune，本身就会把模型往低 FA / 低 PD 方向推；
第二层：FullDEA-v2 的 suppression 进一步增强了这种趋势。
```

因此下一步核心不是“继续增强 clutter suppression”，而是做：

> **target-protected suppression：只允许 DEA 在可靠假警区域扣分，不允许它在可能是真目标的位置扣分。**

---

# 1. 当前最应该判定为有问题的结构

## 1.1 问题核心：v2 的 suppression 没有 target protection

当前 v2 结构是：

```python
z_final = z_target - alpha * suppression_gate * softplus(z_clutter)
```

这个方向比旧版：

```python
y_final = gate * y_real + (1 - gate) * y_cf
```

合理，但它仍然有一个致命问题：

```text
suppression_gate 可以在任何位置生效，包括真实目标、弱目标、目标边界、标注不精确区域。
```

现在的实验结果正好符合这个现象：

```text
FA 明显下降；
PD 下降；
IoU 略低于 baseline；
说明模型学会了扣掉高响应区域，但没有足够能力区分 hard clutter 和 weak target。
```

当前 FullDEA-v2 的基础方向是对的：

```text
它已经放在多尺度 fusion 处；
它使用 z_base；
它使用 scale_logits_full；
它有 target_evidence；
它有 clutter_evidence；
它有 suppression_gate；
它采用 subtractive calibration。
```

但是结构里还缺少显式 target-protection 约束。

---

## 1.2 第二个结构问题：`target_delta` 是自由正负残差

当前 v2 里：

```python
target_delta = self.target_delta_head(target_input)
z_target = z_base + target_delta
```

这意味着：

```text
target_delta 可以为正，也可以为负。
```

如果 `target_delta < 0` 出现在真实目标区域，那么即使 suppression 没有压目标，`z_target` 本身也可能已经把目标压低。

所以下一步必须先审计：

```text
z_base 的 PD
z_target 的 PD
z_final 的 PD
```

判断方式：

```text
如果 z_base 正常，z_target 掉 PD：
    说明 target branch 本身在伤害 recall。

如果 z_target 正常，z_final 掉 PD：
    说明 suppression branch 在过抑制。
```

---

## 1.3 第三个结构问题：hard_bg 仍然是 pixel-level，而不是 component-level

当前 FullDEA-v2 loss 使用 safe background 里的高响应像素 / top-k 像素作为 hard clutter，并且加了 hard_bg ratio 上限。

这个做法对 P0 / P1 是可以的，但要冲主结果还不够。

问题是 IRSTD 的 FA 指标本质上更接近：

```text
component-level false alarm
```

而不是 pixel-level BCE。

一个小 clutter component 里只标 top-k 像素，会导致网络学到：

```text
局部打洞
```

而不是学到：

```text
这个连通区域是杂波。
```

这会带来两个副作用：

```text
1. suppression 学得碎；
2. target 边缘、弱目标邻域、标注误差区域容易被误选成 hard clutter。
```

所以下一版 hard clutter 应该从 pixel-level 改成：

```text
component-level pseudo clutter
```

---

## 1.4 第四个问题：冻结 backbone 时必须确认 BN 没有漂移

matched MSHNet finetune 也掉 PD，说明即使没有 FullDEA，也会出现 FA-PD trade-off。

这里必须排查一个常见问题：

```text
requires_grad=False 不会冻结 BatchNorm running_mean / running_var。
```

如果在 head-only 阶段调用了：

```python
model.train()
```

即使 backbone 参数被冻结，backbone 里的 BN running statistics 仍可能更新。

所以 P1 的 head-only training 必须满足：

```text
backbone 参数 requires_grad=False；
backbone BN 处于 eval()；
只有 full_dea_head 处于 train()。
```

否则你以为在训练 DEA head，实际上 baseline feature distribution 也在动。

---

# 2. 下一步不要跑长实验，先做 P1.5：logit decomposition audit

在改 v3 前，先用现有失败 checkpoint 做一次审计。

这个审计不用训练，只跑验证集。

需要分别评估：

```text
z_base
z_target
z_final
```

保存三组指标：

```text
Metric(z_base)
Metric(z_target)
Metric(z_final)
```

判断规则：

```text
Case A:
z_base 已经低于原始 baseline
=> checkpoint load / freeze / BN drift / finetune protocol 有问题。

Case B:
z_base 正常，但 z_target 掉 PD
=> target_delta 结构有问题，应该改为 target-only non-negative boost。

Case C:
z_target 正常，但 z_final 掉 PD
=> suppression branch 过抑制，必须加 target protection。

Case D:
z_base、z_target、z_final 都掉
=> backbone 或 BN 确实发生了漂移，先修训练 contract。
```

这个审计非常关键。

不要直接改一堆东西，否则不知道 v2 到底是哪里伤了 PD。

---

# 3. 推荐直接进入 FullDEA-v3：Target-Protected Asymmetric DEA

下一版不要叫 v2 patch，建议定义为：

```text
FullDEA-v3: Target-Protected Counterfactual Evidence Suppression
```

核心变化：

```text
v2:
    z_final = z_target - alpha * suppression_gate * clutter

v3:
    z_final = z_base + target_boost - protected_suppression
```

也就是说：

```text
target branch 只能 boost，不能压低目标；
clutter branch 只能在非保护区域 suppression；
final output 必须以 z_base 为 identity anchor。
```

---

# 4. FullDEA-v3 的推荐结构

## 4.1 总体结构

```text
输入：
    x_d0, x_d1, x_d2, x_d3
    scale_logits_full
    z_base

模块：
    1. Multi-scale Evidence Fusion
    2. Target Protection Head
    3. Target Boost Head
    4. Clutter Verification Head
    5. Protected Suppression Head
    6. Asymmetric Logit Calibration

输出：
    protect_logit
    target_boost
    clutter_logit
    raw_suppression_gate
    protected_suppression_gate
    z_final
```

最终形式：

```python
protect_prob = sigmoid(protect_logit)

target_boost = protect_prob * softplus(target_boost_logit)

clutter_amount = softplus(clutter_logit)

raw_suppression_gate = sigmoid(suppression_logit)

protected_suppression_gate = raw_suppression_gate * (1.0 - protect_prob.detach())

z_final = z_base + beta * target_boost - alpha * protected_suppression_gate * clutter_amount
```

关键点：

```text
1. target_boost 非负；
2. suppression 不能直接作用在 protect_prob 高的位置；
3. protect_prob 在 final calibration 中 detach，避免 segmentation loss 把 protect_prob 推成全 1；
4. protect_prob 只由 target-protection supervision 训练。
```

---

# 5. 具体模块修改建议

## 5.1 把 `target_delta_head` 改成 `target_boost_head`

不要再用：

```python
target_delta = self.target_delta_head(...)
z_target = z_base + target_delta
```

改成：

```python
target_boost_logit = self.target_boost_head(target_input)
target_boost = protect_prob * F.softplus(target_boost_logit)
z_target = z_base + beta * target_boost
```

这样 target branch 只能做：

```text
增强目标响应
```

不能做：

```text
降低目标响应
```

这一步是优先级最高的结构改动之一。

---

## 5.2 新增 `TargetProtectionHead`

新增一个保护头：

```python
self.protect_head = nn.Sequential(
    ConvGNAct(protect_in_channels, h, kernel_size=3),
    ConvGNAct(h, h, kernel_size=3),
    nn.Conv2d(h, 1, kernel_size=1),
)
```

输入建议包含：

```text
fused_feature
scale_logits_full
z_base
scale_mean
scale_max
scale_var
```

输出：

```python
protect_logit = self.protect_head(protect_input)
protect_prob = torch.sigmoid(protect_logit)
```

它的作用不是预测最终 mask，而是判断：

```text
这个位置是否应该禁止 clutter suppression。
```

---

## 5.3 把 BN 改成 GN

当前 v2 的新 head 使用 `BatchNorm2d`。

在 batch size=4、IRSTD 小数据集、checkpoint finetune 场景下，BN 很容易引入不稳定性。

建议 FullDEA-v3 新增模块全部使用：

```python
GroupNorm
```

例如：

```python
class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=8):
        super().__init__()
        padding = kernel_size // 2
        g = min(groups, out_channels)
        while out_channels % g != 0:
            g -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(g, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
```

这不是调参，而是结构稳定性修复。

---

## 5.4 suppression gate 改成“验证式 suppression”，不是自由 suppression

当前 v2 suppression gate 是一个自由 head：

```python
suppression_gate = sigmoid(suppression_logit)
```

v3 建议改成：

```python
raw_suppression_gate = sigmoid(suppression_logit)

evidence_suppression_gate = sigmoid(
    clutter_evidence_logit - target_evidence_logit - margin
)

protected_suppression_gate = (
    raw_suppression_gate
    * evidence_suppression_gate
    * (1.0 - protect_prob.detach())
)
```

语义是：

```text
只有当 clutter evidence 明显强于 target evidence，
并且 target protection 不认为这里是目标，
才允许 suppression。
```

这可以显著降低“弱目标被当作 clutter 扣掉”的概率。

---

# 6. hard_bg 生成方式要从 pixel-level 改成 component-level

## 6.1 新的三类区域

训练 FullDEA-v3 时，不要只有：

```text
target
hard_bg
safe_bg
```

应该改成四类：

```text
target_core:
    原始 GT target。

target_protect:
    GT dilated 区域，用来训练 protect head。

ignore_ring:
    target_protect - target_core。
    这个区域不训练 clutter，不训练 suppression。

hard_clutter:
    safe_bg 中由 baseline / online prediction 产生的 false-alarm connected components。
```

关系：

```text
target_core      = gt
target_protect   = dilate(gt, kernel=7 or 9)
ignore_ring      = dilate(gt, kernel=15) - target_core
safe_bg          = 1 - dilate(gt, kernel=15)
hard_clutter     = false-positive components inside safe_bg
```

关键是：

```text
ignore_ring 不能当 hard clutter。
```

小目标数据集标注经常有 1~2 pixel 偏差，target 周边区域如果被当成 clutter，会直接损害 IoU / PD。

---

## 6.2 component-level hard clutter

不要只取 top-k pixels。

建议用：

```text
p_base > low_threshold
或者 p_scale_max > low_threshold
```

得到 candidate components，然后过滤：

```text
与 target_protect 有重叠的 component 丢弃；
面积过大 / 过小的 component 可单独记录；
剩下 component 作为 hard_clutter。
```

伪代码：

```python
def build_component_hard_clutter(p_score, gt, threshold=0.35):
    target_core = gt.float()
    target_protect = dilate(gt, kernel_size=9)
    safe_bg = 1.0 - dilate(gt, kernel_size=15)

    candidate = (p_score > threshold).float() * safe_bg

    hard_clutter = zeros_like(candidate)

    for each image:
        components = connected_components(candidate)
        for comp in components:
            if overlap(comp, target_protect) > 0:
                continue
            if area(comp) < min_area:
                continue
            if area(comp) > max_area:
                continue
            hard_clutter[comp] = 1

    return hard_clutter, target_core, target_protect, safe_bg
```

这个改动比继续调 `topk_ratio` 更重要。

---

# 7. Loss 结构也要改：从“压背景”改成“保护目标 + 验证杂波”

## 7.1 新增 target-preserving rank loss

在 target 区域强制：

```text
z_final 不得低于 z_base 太多
```

loss：

```python
loss_keep_target = mean(
    relu(z_base.detach() + margin_target - z_final) * target_core
)
```

更保守一点可以先设：

```python
margin_target = 0.0
```

含义：

```text
真实目标上，DEA 至少不能比原始 MSHNet 更不自信。
```

这是解决 PD 掉的关键 loss。

---

## 7.2 新增 protection loss

```python
loss_protect = BCEWithLogits(protect_logit, target_protect)
```

但不要在全图监督，否则 protect head 可能变成普通 segmentation head。

建议只在以下区域监督：

```text
target_protect + hard_clutter
```

监督标签：

```text
protect_label:
    target_protect -> 1
    hard_clutter   -> 0
```

loss：

```python
valid_protect = clamp(target_protect + hard_clutter, 0, 1)

loss_protect_map = BCEWithLogits(protect_logit, target_protect)
loss_protect = masked_mean(loss_protect_map, valid_protect)
```

---

## 7.3 suppression gate 只在 hard_clutter 和 target_protect 上监督

```text
target_protect:
    suppression_gate = 0

hard_clutter:
    suppression_gate = 1
```

```python
suppress_label = hard_clutter
valid_suppress = clamp(target_protect + hard_clutter, 0, 1)

loss_suppress = masked_mean(
    BCEWithLogits(suppression_logit, suppress_label),
    valid_suppress,
)
```

不要把所有 safe background 都当 suppression=0。

否则模型会过度学习“背景都是负类”，弱目标附近会被牵连。

---

## 7.4 hard clutter 上保留 suppression order loss

这个可以保留：

```python
loss_suppress_order = mean(
    relu(z_final - z_base.detach() + margin_bg) * hard_clutter
)
```

含义：

```text
在确认的 hard clutter 上，DEA 输出应该低于 baseline。
```

---

## 7.5 新的 FullDEA-v3 loss 结构

建议总 loss：

```text
L_total =
    L_seg(z_final, gt)
  + λ_base_keep * L_keep_target
  + λ_protect * L_protect
  + λ_ev_target * L_target_evidence
  + λ_ev_clutter * L_clutter_evidence
  + λ_suppress * L_suppress_gate
  + λ_order * L_suppress_order
  + λ_clutter * L_clutter_pred
```

但从结构角度，最关键的是这三个：

```text
L_keep_target
L_protect
L_suppress_order
```

它们分别对应：

```text
保护 PD；
阻止 suppression 打到目标；
让 clutter 区域确实低于 baseline。
```

---

# 8. 需要立刻加的审计指标

v2 只看 final 指标不够。

下一轮必须记录：

```text
Metric(z_base)
Metric(z_target)
Metric(z_final)

PD(z_base)
PD(z_target)
PD(z_final)

FA(z_base)
FA(z_target)
FA(z_final)
```

再加：

```text
mean(z_final - z_base) on target_core
mean(z_final - z_base) on target_protect
mean(z_final - z_base) on hard_clutter

protect_prob_on_gt
protect_prob_on_hard_clutter

suppression_gate_on_gt
suppression_gate_on_target_protect
suppression_gate_on_hard_clutter

target_boost_on_gt
target_boost_on_hard_clutter

hard_clutter_component_count
hard_clutter_pixel_ratio
```

P2 条件也要升级。

原来是：

```text
target_evidence_on_gt ↑
clutter_evidence_on_hard_bg ↑
suppression_on_gt ↓
suppression_on_hard_bg ↑
```

现在应该加：

```text
z_final_on_gt >= z_base_on_gt
protect_on_gt ↑
protect_on_hard_clutter ↓
suppression_on_target_protect ↓
```

否则即使 evidence 指标看起来对，PD 仍可能掉。

---

# 9. 立即执行顺序

## Step 1：用现有失败 checkpoint 做三输出审计

不要训练。

评估：

```text
z_base
z_target
z_final
```

得到：

```text
IoU / PD / FA for each output
```

然后判断是哪一支伤了 PD。

这是下一步第一优先级。

---

## Step 2：修 freeze / BN contract

确认 P1 的 head-only 阶段真的只训练 head。

在每个 epoch `model.train()` 后调用：

```python
def freeze_backbone_train_head(model):
    core = model.module if hasattr(model, "module") else model

    for name, p in core.named_parameters():
        p.requires_grad = name.startswith("full_dea_head")

    for name, m in core.named_modules():
        if not name.startswith("full_dea_head"):
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    core.full_dea_head.train()
```

注意：这要在每次 `self.model.train()` 之后调用，因为 `model.train()` 会把 BN 重新切回 train mode。

---

## Step 3：实现 FullDEA-v3 head

核心替换：

```python
target_delta
```

改成：

```python
target_boost = protect_prob * softplus(target_boost_logit)
```

最终输出从：

```python
z_final = z_target - alpha * suppression_gate * softplus(z_clutter)
```

改成：

```python
z_final = (
    z_base
    + beta * target_boost
    - alpha * protected_suppression_gate * clutter_amount
)
```

---

## Step 4：hard_bg 改成 component-level pseudo clutter

先做 offline 版本，不要一开始全 online。

流程：

```text
1. 用 NUAA baseline checkpoint 跑 train set；
2. 取 p_base > 0.35 / 0.40 的 connected components；
3. 去掉与 GT dilated 区域重叠的 components；
4. 保存 pseudo_clutter masks；
5. FullDEA-v3 训练时读取 pseudo_clutter；
6. online hard_clutter 只作为补充，不作为主监督。
```

这一步能显著降低 early training 抖动。

---

## Step 5：重新跑 P0

P0 要新增以下测试：

```text
z_final ≈ z_base 初始化
target_boost 初始接近 0
protected_suppression 初始接近 0
protect_prob 不全 1
suppression_gate_on_synthetic_target 低
suppression_gate_on_synthetic_clutter 高
native masks 尺度正确
partial checkpoint load 正确
loss finite
component hard_clutter 不碰 target_protect
BN freeze contract 正确
```

---

## Step 6：重新跑 P1，但只看机制指标

P1 不看主表结论，只看：

```text
z_base 是否保持 baseline；
z_target 是否不损害 PD；
z_final 是否只在 hard_clutter 上低于 z_base；
protect_on_gt 是否高；
suppression_on_gt 是否低。
```

如果 P1 仍然出现：

```text
FA 降，PD 掉
```

就看：

```text
z_target 掉还是 z_final 掉。
```

这样可以知道是 target_boost 还是 suppression 出问题。

---

# 10. 如果要进一步做 SOTA，建议加入 Local Contrast Evidence Module

仅靠 scale logits 和 decoder feature 可能不够区分 NUAA 上的 hard clutter。

建议 FullDEA-v3 加一个轻量 local contrast module。

小目标和假警的区别经常不只是点本身，而是：

```text
中心响应
周围背景
局部对比度
局部结构紧凑性
```

可以在 head 中加入：

```python
center_feat = avg_pool(fused_feature, kernel_size=3)
context_feat = avg_pool(fused_feature, kernel_size=15)
contrast_feat = center_feat - context_feat
```

然后把：

```python
contrast_feat
```

concat 到：

```text
protect_head
clutter_head
suppression_head
```

输入里。

结构上是：

```text
Multi-scale Evidence Fusion
+ Local Contrast Evidence
+ Target Protection
+ Component-level Clutter Supervision
+ Protected Suppression
```

这比继续堆普通 conv head 更有意义。

---

# 11. 建议的新版本命名和判断标准

建议不要继续叫 v2 patch。

命名：

```text
FullDEA-v3-TPS
Target-Protected Suppression
```

核心 claim：

```text
FullDEA-v2 showed that counterfactual clutter suppression can reduce FA,
but unconstrained suppression hurts weak target recall.

FullDEA-v3 introduces target-protected asymmetric calibration,
where target evidence can only boost the base detector and clutter evidence
can only suppress verified non-target components.
```

P2 gate 也要改成：

```text
机制 gate:
    z_base 接近 baseline；
    z_target PD >= z_base PD；
    z_final FA < z_base FA；
    z_final PD >= baseline_PD - small_tol；
    protect_on_gt > protect_on_hard_clutter；
    suppression_on_hard_clutter > suppression_on_gt；
    mean(z_final - z_base) on GT >= -epsilon；
    mean(z_final - z_base) on hard_clutter < 0。
```

只有这些成立，再跑 NUAA-first gate。

---

# 12. 最终建议

下一步按这个优先级做：

```text
1. 先审计现有 checkpoint：z_base / z_target / z_final 谁导致 PD 掉。
2. 修正 head-only freeze 的 BN contract。
3. 把 FullDEA-v2 改成 FullDEA-v3-TPS：
       signed target_delta -> non-negative target_boost
       free suppression -> target-protected suppression
       pixel hard_bg -> component-level hard clutter
       BN head -> GN head
       no target-protection -> explicit protect head
4. 加 target-preserving rank loss。
5. 加 local contrast evidence module。
6. 重新跑 P0 和 P1。
7. P1 机制指标不过，不跑 NUAA gate；NUAA gate 不过，不跑 NUDT/IRSTD。
```

一句话总结：

> **v2 已证明“能压 FA”，但没证明“只压假警”。v3 的核心任务不是更强 suppression，而是 target-protected suppression：目标只能被 boost，杂波只能在被验证为非目标时被扣除。**
