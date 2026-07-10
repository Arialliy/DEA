# DEA：尺度证据语义混叠的问题诊断、创新性否证与性能安全边界

## 结论

**DEA v0 作为最终主模型的设计已经失败。**

更准确地说，失败的是下面这条结构假设：

> 用同一个 \(K/K^\ast\) 在五个尺度上执行单步预测误差回写，就足以替代 MSHNet 的层级 decoder，并获得更好的目标—杂波判别。

完整训练已经否定了该假设：DEA v0 的最佳 IoU 为 0.7274，MSHNet 为 0.7471；二者 PD 相同，但 DEA v0 的 FA 更高。

但是，**DEA 的研究问题并没有因此失败**。真正的问题是：v0 解决错了问题。

v0 解决的是“如何受约束地把新尺度特征写入状态”；当前反事实尺度分析提出了一个更贴近实际错误的候选问题：

> **目标和杂波可能产生相似的尺度贡献模式；是否必须利用压缩前上下文改变 decoder state，仍需通过 OOF probe、terminal controls 和受控状态干预验证。**

这两个问题并不相同。

### 2026-07-10 严格收缩后的结论

当前方案还不能称为合格的 DEA 主模型：

- `ComponentEvidenceControl` 是终端动态尺度门控，只能做能力上界和因果 control；
- `Decoder-Jacobian` 确实改变了 forward 计算图，但“通过 decoder gradient/VJP 修正 latent state 再重解码”已有强近邻工作，不能将 Jacobian/VJP 本身当作顶会级主创新；
- 继续叠加 persistent state、ODE、primal--dual、learned corrector 不会自动增加创新性，反而会同时增加 prior-art 重合和性能风险；
- 因此，当前项目状态是 **存在一个待确认的问题，但还没有同时通过问题资格、结构创新和性能门槛的 DEA v1**。

“保证性能”也必须分成两件事：

1. 可以保证新结构在关闭校正时逐元素退化为已训练的 MSHNet；
2. 不可以在没有训练和独立 holdout 结果时保证开启校正后一定提高 IoU/PD/FA；任何这样的承诺都不是可证明的模型性质。

---

# 一、DEA 到底要解决什么问题

## 1. 不再以“可解释、可判定、可执行”为出发点

“让预测具有可归因、可判定、可执行的证据依据”只能作为方法性质，不能作为研究动机。

DEA 的出发点应该是一个现有方法确实无法稳定处理、且能够被反事实实验验证的具体问题：

## **尺度证据语义混叠**

设 MSHNet 四个尺度对最终预测的精确贡献为：

\[
c_s=W_s*s_s,
\]

最终预测为：

\[
z_{\mathrm{base}}
=
b+\sum_{s=0}^{3}c_s.
\]

对于一个候选组件 \(R\)，定义它的尺度证据为：

\[
C_R=
\left\{
c_s(x)\mid
x\in\mathcal N(R),\ s=0,\ldots,3
\right\}.
\]

尺度证据语义混叠的待验证定义是：

\[
C_{R_{\mathrm{target}}}
\approx
C_{R_{\mathrm{clutter}}},
\]

但二者具有不同的语义标签。这里的“近似”目前尚未由 OOF probe、置信区间和跨 checkpoint 结果建立，不能把它写成已确认事实。

也就是说，真实目标和背景杂波经过多尺度 decoder、单通道 side heads 和最终融合后，可能落入相似的尺度证据表示空间。

MSHNet 的 decoder states 原本是多通道的，但每个尺度先被独立压缩成一个单通道 logit，随后四个 logits 才进入固定的 \(3\times3\) final convolution。因此，terminal fusion 阶段不再直接访问：

- 组件形态；
- 组件内部结构；
- 目标与周围背景的关系；
- 相邻碎片或卫星响应；
- decoder feature 中未被 side logit 保留的判别信息。

这是需要验证的 `side-logit bottleneck` 假设。它不等于“MSHNet 完全没有使用上下文”：decoder 可能已经把部分上下文编码进 side logits；只有 P1--P3 的条件增益才能判断单通道压缩是否真的造成不可恢复的判别损失。

---

## 2. 正式问题定义

建议将 DEA 的问题定义写成：

> **待验证问题：真实目标、背景杂波和目标附近碎片可能产生相似或冲突的尺度贡献模式。需要判断这种混叠究竟只是固定 terminal fusion 的容量不足，还是 side-logit 压缩确实丢失了只能通过 decoder context 与 state evolution 利用的条件信息。**

因此，当前 DEA 研究需要先回答两个相互排斥的分支：

1. 若局部非线性或 terminal context weighting 已足够，则应停止结构扩张，采用更简单的 terminal 方法；
2. 只有 terminal controls 仍不足、而 pre-logit decoder context 有稳定条件增益时，才有理由改变保留强 decoder 前提下的跨尺度 state evolution。

组件条件的

\[
z=b+\sum_s r_sc_s
\]

只是一项 terminal control 假设，不再作为 DEA 主模型定义；其数学类别仍是 dynamic scale gating。

---

# 二、当前反事实发现说明了什么（探索性、单 checkpoint）

对完整训练的 MSHNet 最佳 checkpoint 做反事实尺度分解后，发现：

- 63 个目标中漏检 1 个；
- 剩余 4 个 FP 组件，共 27 个像素；
- 4 个 FP 全部受到最细尺度 side0 的强正贡献；
- 其中 3 个 FP 同时受到半分辨率尺度的负贡献，但最终融合仍被细尺度正响应压过；
- FP 的尺度贡献符号模式与 TP 高度重合；
- 唯一 FN 中，最细尺度贡献约为 \(-33.87\)，半尺度贡献约为 \(+3.98\)，粗尺度还提供弱正证据，但被细尺度强负证据覆盖。

例如，TP 的主要模式可能是：

\[
\text{细尺度正}
/
\text{半尺度负}
/
\text{四分之一尺度正},
\]

而 FP 中也会出现完全相同的符号模式。

因此，当前发现支持下面这个更具体的判断：

> **固定融合不一定缺少尺度响应，而是缺少在相似尺度响应组合下区分目标与杂波的语义依据。**

但这还只是初步证据，不能仅凭 4 个 FP 和 1 个 FN 直接建立普遍性结论。

---

# 三、当前问题定义还需要两项修正

## 1. 不能只比较贡献符号模式

“细尺度正、半尺度负、四分之一尺度正”只是三位符号模式。

两个组件符号相同，不代表它们的完整证据相同。它们还可能在以下方面不同：

- 贡献绝对幅值；
- \(3\times3\) contribution patch；
- 组件内贡献分布；
- 组件边界处贡献变化；
- 不同尺度响应的中心偏移；
- 局部非线性组合；
- decoder feature 中的上下文差异。

因此必须依次比较：

\[
\text{符号统计}
\rightarrow
\text{贡献标量统计}
\rightarrow
4\times k\times k\text{ 完整局部 patch}
\rightarrow
\text{组件级统计}
\rightarrow
\text{压缩前特征上下文}.
\]

若一个小型非线性局部网络只看 \(4\times k\times k\) contribution patch 就能稳定区分 TP/FP，那么问题只是：

> MSHNet 的最终线性卷积能力不足。

此时不需要组件级 DEA。

只有当局部非线性融合仍无法区分，而加入组件或上下文信息后显著改善，才能支持“尺度证据混叠需要组件裁决”的论点。

---

## 2. 当前错误样本太少，仍属于假设生成

当前验证集只有：

- 63 个目标；
- 1 个 FN；
- 4 个 FP 组件。

这组案例非常有价值，因为它提供了具体反例，但还不足以证明稳定规律。

不能只依赖：

- 一个 checkpoint；
- 一个阈值；
- 一个 seed；
- 一个数据集；
- 4 个 FP 和 1 个 FN。

下一步应先扩大证据样本，而不是立即实现新主模型。

---

# 四、DEA 研究应由四个可证伪假设驱动

## H1：尺度证据确实存在语义混叠

定义组件标签：

\[
Y_R\in\{\text{target},\text{clutter}\}.
\]

只使用尺度贡献 \(C_R\) 预测 \(Y_R\)。

若 TP 和 FP 的 contribution representation 高度重叠，则：

\[
\operatorname{AUC}(Y_R\mid C_R)
\]

应该相对有限。

需要比较多个 probe：

| Probe | 输入 | 回答的问题 |
|---|---|---|
| Linear-local | contribution 标量统计 | 固定线性组合是否足够 |
| Nonlinear-local | \(4\times k\times k\) contribution patch | 是否只缺局部非线性 |
| Component-scale | 组件内尺度统计 | 是否需要组件聚合 |
| Contextual | contribution + morphology/context | 上下文是否消除混叠 |

---

## H2：组件与上下文确实提供条件增益

令 \(H_R\) 表示组件与上下文信息。

需要验证：

\[
I(Y_R;H_R\mid C_R)>0.
\]

实际实验可使用 probe 性能差表示：

\[
\Delta_{\mathrm{context}}
=
\operatorname{AUC}(C_R,H_R)
-
\operatorname{AUC}(C_R).
\]

只有当该增益满足下面条件，才能说明上下文是必要变量：

- 在 image-grouped cross-validation 中稳定；
- bootstrap 置信区间下界高于零；
- 超过 seed/checkpoint 波动；
- 在不同数据集或不同模型 checkpoint 上方向一致。

应依次加入：

1. 组件形态；
2. 原图局部与外围 ring；
3. 压缩前 decoder features。

这样还能定位区分信息到底存在于哪一层。

---

## H3：重新选择现有尺度证据能够修复错误

四个尺度只有：

\[
2^4=16
\]

种保留子集。

对每个组件枚举：

\[
z_{R,S}
=
b+\sum_{s\in S}c_s,
\qquad
S\subseteq\{0,1,2,3\}.
\]

需要回答：

- 当前 FP 是否存在某个子集能将其压回背景；
- 当前 recoverable FN 是否存在某个子集能将其恢复；
- 修复该组件时是否破坏已有 TP；
- 仅执行 retain/drop 是否足够；
- 是否必须使用连续增益或引入新证据。

该 oracle 决定下一版模型的能力边界。

### 三种可能结果

| Oracle 结果 | 结论 |
|---|---|
| retain/drop 已能修复大部分错误 | 只需证据拒绝机制 |
| 连续非负权重才能修复 | 需要证据可信度重标定 |
| 任意重加权仍不能修复 | 当前尺度贡献基底不足，必须回到 feature representation |

### 当前 global fixed-subset 预审计

在固定 43-image validation split（SHA256：`e9156e9386b4cd15b587536f20e5e6ab7db04b41c00a01e2b59c5a49673ca86f`）上，对 MSHNet epoch-258 checkpoint 运行 16 个**全图固定**尺度子集：

- all-scale direct baseline：IoU 0.747056、PD 0.984127、FA 9.5811；
- 全图删除 scale3：IoU 0.747266、PD 0.984127、FA 9.5811；
- 全图删除 scale1：IoU 0.747679、PD 0.984127、FA 9.9360。

因此，全图固定删尺度最多只带来约 \(+0.00062\) IoU，且最佳 IoU 配置增加 FA；唯一严格支配 baseline 的“删除 scale3”也只增加约 \(+0.00021\) IoU。这个结果说明**静态全局子集不是主模型答案**，但它不是 component oracle：不同图像、不同组件选择不同子集的可修复上界仍未计算。该 split 已参与设计，结果只能用于 mechanics，不是确认性证据。

### 当前 prediction-only component oracle

使用 final/side logits 与 \(0.1,0.2,0.3,0.5\) 阈值生成候选；候选生成完全不读取 GT，GT 只在之后按项目原 PD/FA 规则标注和计算 oracle。在同一个 43-image design-used validation split 上：

- 生成 69 个 prediction-only 候选；
- 覆盖 baseline 的 4/4 个 FP component；
- 39/69 个候选存在严格更好的局部尺度子集；
- 3/4 个 baseline FP 具有 terminal retain/drop 修复空间；
- 唯一 FN 附近没有任何 prediction-only 候选，因此 0/1 FN 可被 terminal component selection 恢复。

这个结果同时给出两个边界：

1. terminal evidence control 对部分 FP 有真实局部能力，因此是必须的 strong control；
2. 它无法恢复当前 FN，若要提高 recall，干预必须发生在候选消失之前的 encoder/decoder representation 层。

但 4 FP/1 FN 仍然只是 mechanics 样本，不能据此声称可泛化性能增益。

---

## H4：terminal evidence control 是否不足，decoder-state intervention 是否必要

H1--H3 成立只说明“存在可诊断、可能可修复的错误”，并不能推出必须设计新 decoder 机制。还需要区分三层能力上界：

\[
U_{\mathrm{static}}
\le
U_{\mathrm{terminal\text{-}context}}
\le
U_{\mathrm{decoder\text{-}state}}.
\]

只有同时观察到以下事实，才允许进入 decoder-state 结构候选：

- P3 中的 decoder context 在 image-grouped OOF 上稳定优于 contribution-only probe；
- subset oracle 或连续权重 oracle 显示错误具有可修复空间；
- local nonlinear fusion、pixel attention 和冻结的组件上下文控制仍不能闭合该空间；
- 对 decoder state 的受控干预提供 terminal controls 不具备的额外修复能力。

若简单 terminal control 已经足够，则不应为了“结构创新”继续增加复杂度。

---

# 五、必须区分两类 FN

## 1. 可裁决 FN

至少一个 side scale 已经产生目标候选或正贡献，但 final fusion 将它压掉。

例如：

\[
c_0\approx-33.87,\qquad
c_1\approx+3.98,
\]

其他粗尺度还有弱正证据。

这种 FN 可能通过 evidence adjudication 修复。

## 2. 不可裁决 FN

所有尺度都没有产生候选，或者各尺度都没有目标证据。

这种错误不是融合或裁决问题，而是：

- encoder 没有提取到目标；
- decoder 没有保留目标；
- side prediction 已经完全丢失目标。

DEA 的 terminal evidence adjudication 无法凭空修复这种 FN。后文的 Decoder-Jacobian 候选也只能处理 decoder 内仍然存在可利用证据的漏检，不能恢复 encoder 已完全遗漏的目标。

因此候选生成必须完全不使用 GT：

\[
M_{\mathrm{candidate}}
=
\operatorname{Union}
\left(
\operatorname{Candidates}(s_0),
\dots,
\operatorname{Candidates}(s_3),
\operatorname{Candidates}(z)
\right).
\]

GT 只用于给候选打标签，不能用于生成候选。

---

# 六、当前 DEA v0 为什么偏离了更有支持的候选问题

伴随预测误差回写 v0 解决的是：

> 如何受约束地吸收新的尺度 observation。

但它没有直接解决：

- TP 与 FP 的尺度证据模式重合；
- 组件级目标/杂波区分；
- 细尺度证据错误支配；
- 目标附近碎片和卫星响应；
- 单通道 side logit 压缩前的判别信息丢失。

同时，它还删除了 MSHNet 的强层级 decoder。

因此 v0 偏离了当前更有支持的候选问题。它应冻结为：

> **DEA-v0 mechanics control / structural control**

而不是继续通过增加状态宽度、步长、鲁棒阈值或额外模块修补。

---

# 七、下一步不要先改 `model/dea_mshnet.py`

合理的开发顺序应当是：

```text
证明 aliasing
→ 判断 local/context 条件信息增益
→ 计算 terminal oracle
→ 运行 terminal controls
→ 证明 terminal control 不足
→ 做 decoder-state intervention mechanics
→ 才判断是否存在 DEA v1
```

而不是：

```text
直接往 DEA v0 中加入组件模块
```

DEA v0 已经是一个完整、可复现的负控制。继续在它上面添加 context、attention 或 component head，会重新混淆失败原因。

---

# 八、第一阶段代码修改：建立证据审计接口

## 1. 保持 baseline 预测路径不变

建议新增：

```text
model/mshnet_evidence_view.py
```

或者在 `MSHNet` 中增加一个不改变默认返回值的：

```python
forward_evidence(...)
```

不要把 baseline 的：

```python
z_base = self.final(scale_logits)
```

改成：

```python
z_base = bias + contributions.sum(dim=1)
```

两者代数等价，但浮点归约顺序可能不同。baseline 指标必须始终使用原始 direct convolution。

新增分解函数：

```python
import torch
import torch.nn.functional as F


def decompose_final_contributions(final, scale_logits):
    # Return bias-free per-scale contributions.
    if scale_logits.ndim != 4 or scale_logits.shape[1] != 4:
        raise ValueError(
            f"expected scale_logits [B,4,H,W], got {tuple(scale_logits.shape)}"
        )

    per_scale_weight = final.weight.permute(1, 0, 2, 3).contiguous()

    contributions = F.conv2d(
        scale_logits,
        per_scale_weight,
        bias=None,
        stride=final.stride,
        padding=final.padding,
        dilation=final.dilation,
        groups=4,
    )

    if final.bias is None:
        bias = scale_logits.new_zeros(1, 1, 1, 1)
    else:
        bias = final.bias.view(1, 1, 1, 1)

    return contributions, bias
```

diagnostic 输出应包含：

```python
{
    "pred": z_base,                 # 原始 direct convolution
    "scale_logits": scale_logits,
    "contributions": contributions,
    "fusion_bias": bias,
    "decoder_features": (
        x_d0, x_d1, x_d2, x_d3,
    ),
}
```

---

## 2. 添加严格重建测试

新增：

```text
tests/test_mshnet_evidence_decomposition.py
```

至少检查：

```python
z_direct = model.final(scale_logits)
z_reconstructed = bias + contributions.sum(dim=1, keepdim=True)

assert torch.allclose(
    z_direct,
    z_reconstructed,
    atol=1e-4,
    rtol=1e-5,
)
```

这里的 `float32` grouped convolution 与原始四通道 convolution 具有不同的浮点归约顺序；真实 checkpoint 上误差可达到约 \(9.16\times10^{-5}\)。因此诊断采用 `atol=1e-4, rtol=1e-5`，若需要更严格代数检查则转为 `float64`。同时保证普通 `forward()` 输出逐元素不变。

---

# 九、第二阶段：建立组件级证据数据集

新增：

```text
utils/component_evidence.py
tools/build_component_evidence_manifest.py
```

每个组件记录：

```text
image_id
candidate_id
threshold_source
component_mask / bbox
matched_label
is_tp / is_fp / is_recoverable_fn
per-scale contribution statistics
local contribution patches
component morphology
ring context
pooled decoder features
```

---

## 1. 候选生成

候选必须来自：

- final prediction；
- 四个 side predictions；
- 多个低阈值；
- 必要时各尺度局部极大值。

建议至少使用一个 threshold bank：

```python
thresholds = (0.1, 0.2, 0.3, 0.5)
```

候选合并后去重。

组件标签应复用当前项目 PD/FA 的匹配规则，不能重新定义另一套匹配标准，否则诊断和最终指标不一致。

---

## 2. 组件特征建议

### 尺度贡献特征

- 每尺度贡献 mean/max/min/sum；
- 正贡献比例；
- 负贡献比例；
- contribution cancellation；
- scale-wise peak；
- 各尺度质心偏移；
- 完整 \(4\times7\times7\) 或 \(4\times15\times15\) patch。

### 组件形态

- area；
- bbox 长宽；
- compactness；
- eccentricity；
- peak count；
- 边界长度；
- 组件内部响应离散程度。

### 周边上下文

- component 与 ring 的原图均值差；
- ring 方差；
- 局部对比度；
- 边缘密度；
- 最近邻组件距离；
- 最近更强组件距离；
- 是否属于强目标附近的卫星碎片。

“卫星碎片”不能作为人工模糊标签，应转成可计算量，例如：

\[
d_{\mathrm{nearest}},
\quad
\frac{\text{当前峰值}}{\text{邻近主峰值}},
\quad
\text{两组件间响应连通度}.
\]

### 压缩前特征

从 \(x_{d0},\ldots,x_{d3}\) 提取：

- component pooling；
- surrounding-ring pooling；
- 两者差值；
- channel-wise mean/max。

这一组实验将直接判断：

> target/clutter 区分信息是否存在于 decoder feature 中，只是在单通道 side head 处丢失了。

---

# 十、训练 probe 时必须使用 OOF 候选

不能直接使用训练完成的 MSHNet 在自己的训练图像上产生组件数据，再训练 adjudicator。

原因包括：

- 训练图上的错误分布不真实；
- FP 可能过少；
- 组件 classifier 会利用 backbone 过拟合后的特征；
- 结果会过度乐观。

正确做法：

1. 将训练集分成 \(K\) 折；
2. 每次用 \(K-1\) 折训练 MSHNet；
3. 在未参与训练的一折生成候选组件；
4. 汇总得到 out-of-fold component manifest；
5. validation 仅用于模型选择；
6. test 完全不参与。

这一步比立即启动一轮新 DEA 长训练更重要。

---

# 十一、最关键的四组 probe

新增：

```text
tools/probe_evidence_aliasing.py
```

按 image 分组进行交叉验证。

| Probe | 输入 |
|---|---|
| P0 | contribution 标量统计 |
| P1 | 完整局部 contribution patch |
| P2 | contribution + component morphology/ring |
| P3 | contribution + pooled decoder context |

每个 probe 必须同时报告 image 数、target/clutter 事件数、class prevalence 和分组方式。FP/FN 事件过少时，AUC 只能作为描述量，不能强行给出 PASS/FAIL。

结果解释如下。

## P1 明显优于 P0

说明问题主要是：

> 固定线性 final convolution 太弱。

此时先做参数量匹配的局部非线性 fusion control，不要直接上组件模型。

## P1 与 P0 都弱，P2 明显提高

支持：

> 组件结构是解除证据混叠的必要变量。

## P2 改善有限，P3 明显提高

支持：

> side-logit bottleneck 确实丢失了 target/clutter 判别信息。

此时 DEA 应保留并利用 pre-logit decoder features。

## P3 也没有改善

说明当前 decoder representation 本身无法区分这些 FP/TP，问题应转向 encoder/decoder 表征，而不是继续做证据裁决。

---

# 十二、Terminal ComponentEvidenceControl：能力与因果 control，不是 DEA v1

建议第一版先命名为：

```text
ComponentEvidenceControl
```

它只能命名为 control，不能在成功后自动升级为 DEA v1。

新增：

```text
model/component_evidence_control.py
```

它冻结 MSHNet 的强 encoder、hierarchical decoder 和尺度贡献，只在 terminal decision 上施加受约束干预，用于回答 H2/H3；它不改变 decoder state evolution。

完整路径是：

```text
threshold candidates
→ connected components
→ morphology/ring/decoder pooling
→ component token
→ adjudicator
→ component-mask contribution reassembly
```

因此它是一个后验组件管线。即使限制输出范围，它在数学上仍属于 component-conditioned dynamic scale gating，不能满足“非模块堆叠统一主模型”的要求。

---

## 1. 最小机制：组件条件证据拒绝

对组件 \(R_k\)，构造组件 token：

\[
u_k
=
\operatorname{Pool}
\left(
C_{R_k},
H_{R_k}^{\mathrm{morph}},
H_{R_k}^{\mathrm{ring}},
H_{R_k}^{\mathrm{decoder}}
\right).
\]

一个共享 adjudicator 输出：

\[
a_k=\sigma(\phi_a(u_k)),
\]

\[
q_{k,s}=\sigma(\phi_s(u_k)).
\]

定义尺度保留率：

\[
r_{k,s}
=
1-a_kq_{k,s},
\qquad
0\le r_{k,s}\le1.
\]

最终：

\[
z_{\mathrm{control}}(x)
=
z_{\mathrm{base}}(x)
+
m_k(x)
\sum_s
\left(r_{k,s}-1\right)c_s(x),
\]

其中 \(m_k\) 是组件区域。

该 control 的干预边界具有明确意义：

- \(a_k=0\)：完全保持 baseline；
- \(r_{k,s}\approx1\)：信任该尺度；
- \(r_{k,s}\approx0\)：拒绝该尺度证据；
- 去除正贡献会降低 logit，抑制 FP；
- 去除负贡献会提高 logit，恢复可裁决 FN。

因此不需要显式 Increase/Decrease/Keep 三分类：

\[
\Delta z_k
=
\sum_s(r_{k,s}-1)c_s
\]

的符号自然决定最终动作。

最重要的是：

> adjudicator 不能产生任意 residual，只能连续衰减或近似保留已有的精确尺度贡献。

有限 sigmoid logits 下 \(0<r_{k,s}<1\)，因此它不是严格的 binary retain/drop，而是 soft attenuation；16-subset binary oracle 只是能力上界。这个限制使 control 的干预边界清楚，但不会改变它属于 dynamic gating 的数学类别。

此外，多个候选组件重叠时必须先定义唯一归属或显式合并规则，否则不同 \(m_k\) 会对同一像素重复修改贡献。它也不能修复所有尺度均无候选的 FN。

---

## 2. 为什么初版先限制 \(r\in[0,1]\)

先用 16-subset oracle 判断“删除有害证据”是否足够。

若 retain/drop oracle 已经能修复主要错误，就没有理由允许 arbitrary amplification。

只有 oracle 明确表明需要放大正证据时，才扩展为：

\[
0\le r_{k,s}\le2.
\]

若必须允许负权重或完全自由 residual 才能获得提升，说明现有 contribution basis 本身不足，继续称为“证据裁决”就不再可信。

---

# 十三、该控制模型的训练顺序

第一轮只训练 adjudicator：

```text
freeze MSHNet encoder
freeze MSHNet decoder
freeze side heads
freeze original final weights
train component adjudicator
```

这样可以回答一个单一问题：

> 在现有 MSHNet 表征和尺度贡献完全不变的情况下，组件上下文是否能改善证据决策？

若该控制失败，不应直接进行 end-to-end 微调。

若成功，只能证明 terminal context 有用。若它已经达到性能目标，应优先停留在更简单的 terminal 方法，而不是为了结构叙事继续扩张。

可选的解冻实验最多作为 capacity upper bound：

```text
unfreeze decoder features
keep contribution-reassembly constraint
jointly fine-tune
```

不能一开始就全网络微调，否则提升无法归因于 evidence adjudication；该 upper bound 也不能自动命名为 DEA 主模型。

---

# 十四、必须加入的对照

terminal controls 至少比较：

| 对照 | 验证内容 |
|---|---|
| MSHNet static final conv | baseline |
| local nonlinear fusion | 是否只缺非线性 |
| large-receptive-field conv | 是否只缺更大局部感受野 |
| pixelwise soft attention | 是否普通动态权重已足够 |
| component morphology only | 组件结构贡献 |
| decoder context only | pre-logit 信息贡献 |
| component + decoder context | 完整假设 |
| unconstrained residual refiner | 证据约束是否有价值 |
| exact contribution reassembly | 受约束 terminal control |

如果普通 pixelwise attention 已经达到相同结果，则没有必要使用组件裁决。

feature-state controls 另行比较：

| 对照 | 验证内容 |
|---|---|
| scalar/logit Kalman gate | 动态尺度插值是否足够 |
| \(H_s^\top\) rank-1 projection | 最小范数 head 回写；其 side-logit 严格等价 gate |
| direct multichannel residual injection | 是否只需注入残差 |
| Decoder-Jacobian VJP | 原 nonlinear decoder 的空间、通道与上下文 Jacobian 是否必要 |

---

# 十五、Decoder-Jacobian：仅保留为低风险 mechanics candidate

这一节定义的是一个待否证的 mechanics candidate，不是已经成立的 DEA v1。其目的不是增加 learned block，而是检验：**相同的尺度 logit 冲突能否借助原 decoder 自身的输入相关 Jacobian，在不同上下文中产生不同的多通道状态修正。**

严格创新性结论已经是 **NO-GO as main novelty**：

- [Predify, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/75c58d36157505a600e0695ed0b3a22d-Abstract.html) 已使用 reconstruction-error gradient 反复更新中间 representation；
- [PR-MaGIC, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_PR-MaGIC_Prompt_Refinement_Via_Mask_Decoder_Gradient_Flow_For_In-Context_CVPR_2026_paper.html) 已在无额外训练/结构修改的设置下，通过 mask-decoder gradient flow 修正 embedding；
- [Deep Back-Projection Networks, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.html) 已建立跨尺度 projection-error feedback 拓扑；
- IRSTD 中的 [DFINet, Pattern Recognition 2026](https://www.sciencedirect.com/science/article/pii/S0031320325006181) 也已覆盖“历史预测反馈、迭代纠错、动态高低层语义融合”这一宽泛叙事。

因此，不能声称创新点是“首次 decoder-gradient refinement”、“首次 latent correction”或“首次 predictive-coding decoder”。它仅剩下一个待验证的局部差异：**MSHNet 特定的相邻尺度 side residual，经下一个原生 nonlinear decoder stage 的 exact VJP 回写，并直接改变后续 coarse-to-fine state evolution**。这个差异只有在严格对照上显示特异能力和稳定性能收益后，才可能升级。

## 1. 完整保留 MSHNet 强路径

以下部分全部保留：

- encoder 与 `middle_layer`；
- `decoder_3 ... decoder_0` 的全部 ResNet、BN、ChannelAttention、SpatialAttention 与 skip；
- `output_3 ... output_0`；
- 原 `final` 的 \(4\to1\) \(3\times3\) convolution；
- 原多尺度监督与训练协议。

不增加新 decoder、component extractor、attention/router 或 learned correction head。

## 2. 一步前瞻 decoder-state 校正

设已经得到较粗尺度 decoder state \(d_s\)。原 MSHNet 的下一尺度映射写为：

\[
G_s(d;F_{s-1})
=
H_{s-1}
\left(
D_{s-1}[F_{s-1},U(d)]
\right),
\]

其中 \(D_{s-1}\) 和 \(H_{s-1}\) 都是原 MSHNet 已有模块。

先按 baseline 做一次 probe：

\[
\tilde d_{s-1}=D_{s-1}[F_{s-1},U(d_s)],
\qquad
o_{s-1}=H_{s-1}(\tilde d_{s-1}).
\]

将当前粗尺度 side logit 固定上采样为前瞻 target：

\[
t_{s-1}
=
\operatorname{sg}
\left[U(H_s(d_s))\right],
\qquad
r_{s-1}=t_{s-1}-o_{s-1}.
\]

`sg` 表示 stop-gradient；没有它时 target 同样依赖 \(d_s\)，完整梯度会多出 target 路径，不能再称为下面这条单向前瞻更新。

使用一个全尺度共享的固定鲁棒影响函数 \(\psi_\tau\)，通过 autograd VJP 复用原 decoder 的 Jacobian：

\[
v_s
=
J_{G_s}(d_s;F_{s-1})^\top
\psi_\tau(r_{s-1}).
\]

只更新原本就要传给下一层的多通道 state：

\[
d_s^+
=
d_s+
\frac{\eta}{L_s+\epsilon}v_s,
\]

然后用同一个原 decoder 重新解码：

\[
d_{s-1}
=
D_{s-1}[F_{s-1},U(d_s^+)].
\]

对 `d3 -> d2`、`d2 -> d1`、`d1 -> d0` 三个 transition 依次执行一次。\(L_s\) 可由 1--2 步 JVP/VJP power iteration 估计局部 \(\|J\|_2^2\)，mechanics pilot 也可以使用回溯确保当前 probe energy 不增加。\(\eta,\tau\) 必须全尺度共享。

## 3. 为什么它不是已否决的 rank-1 gate

若直接使用 side head 伴随：

\[
d'=d+\eta H_s^\top(b-H_sd),
\]

更新后的 side logit 可以严格化简成 \((1-g)H_sd+gb\)，因此只是动态 gate control。

Decoder-Jacobian VJP 则穿过原 `decoder_{s-1}` 的空间卷积、BN、ReLU、CA、SA 和 residual path。它对多通道 \(d_s\) 的修正是输入相关且空间非对角的；只有 decoder 退化为逐点对角线性映射时才接近普通插值。相同的 side residual 是否会在 TP/FP 上产生不同的空间修正，是该候选唯一需要证明的核心能力。

这仍不能被提前解释为“已经解除语义混叠”。它也可能只是昂贵的 iterative refinement。

## 4. 训练边界

第一阶段只做冻结 checkpoint 的 forward mechanics：

```text
freeze complete MSHNet
create_graph = False
test shared eta/tau or fixed backtracking
do not call the result end-to-end DEA
```

若把 \(v_s\) detach，则只能验证 forward correction；模型不会学习如何改变 Jacobian 方向。真正端到端训练需要 `create_graph=True`，最终损失反传会产生 Hessian-vector 或 mixed derivative，必须单独报告显存、速度与数值稳定性。

\(\eta=0\) 时必须逐元素恢复完整 MSHNet，包括四个 side logits 与 `final` 输出。

## 5. GO/NO-GO

| Gate | 进入条件 | 未通过时 |
|---|---|---|
| J0：问题资格 | P3 稳定优于 P1/P2，且 terminal controls 不能闭合 oracle gap | 不实现 Jacobian candidate |
| J1：非 gate 性 | VJP 明显优于 scalar gate、\(H_s^\top\) rank-1 projection 和等范数 direct residual | 判为普通残差/门控变体 |
| J2：mechanics | \(\eta=0\) identity、VJP 内积、有限梯度、局部稳定、可接受开销全部通过 | 冻结为失败控制 |
| J3：冻结因果试验 | 在未参与设计的 OOF/holdout 上优于完整 MSHNet 和 terminal controls | 禁止靠 unfreeze rescue |
| J4：完整训练 | 多 seed 同协议稳定提升，且粗到细优于 reverse/random/parallel | 不称 DEA v1 |
| J5：近邻区分 | 显著优于 PR-MaGIC-style decoder-logit gradient、Predify-style reconstruction gradient 与 DFINet-style feedback control | 不得把 VJP candidate 当主结构 |

关键否证对照是用匹配 correction norm 的直接残差注入替代 \(J_G^\top\psi(r)\)。若二者相同，收益并不来自 decoder Jacobian context。

---

# 十六、性能安全边界：可保证 baseline identity，不伪造涨点承诺

## 1. 唯一可严格保证的性能底座

任何候选结构必须写成包含完整 MSHNet 的 baseline-embedded family：

\[
z_\alpha(x;\theta_B,\theta_C)
=
z_{\mathrm{MSHNet}}(x;\theta_B)
+
\Delta z_\alpha(x;\theta_B,\theta_C),
\]

并严格满足：

\[
\Delta z_0\equiv0,
\qquad
z_0\equiv z_{\mathrm{MSHNet}}.
\]

这不是要把新模块并联到 baseline 后面，而是要求新的 decoder transition 本身带有一个可证明的 MSHNet 不动点。对 Decoder-Jacobian mechanics，这个开关是全尺度共享的 \(\eta\)；对任何后续结构，必须给出同等强度的恒等证明和数值测试。

可严格验收的是：

- 原 MSHNet encoder、`middle_layer`、四个 decoder、四个 side head 与 `final` 全部加载；
- 校正关闭时，side logits、final logits、loss 与评估指标与 baseline 一致；
- 新路径零初始化或使用 homotopy 从 identity 出发；
- 训练不稳定或候选无效时，可无损返回原 checkpoint。

这些保证的是**实现不破坏 baseline 且项目有退路**，不是活性候选在未见测试集上必然优于 baseline。若要对任意未知 GT 保证指标不下降，唯一通用做法就是永远输出 baseline，那样也不会产生新能力。

## 2. 禁止“一起训练看运气”的分级性能门

| Gate | 数据与训练状态 | 只能回答什么 | 进入条件 |
|---|---|---|---|
| B0 identity | 无训练 | 实现是否严格包含 MSHNet | 输出/梯度/参数加载检查通过 |
| B1 frozen mechanics | 完整冻结 MSHNet | 校正方向是否具有局部修复能力 | 在未参与调参的 holdout 上优于 gate/direct-residual controls |
| B2 short paired run | 同 seed、同 split、同 optimizer 短训练 | 是否有早期正信号 | IoU/PD/FA 的改变方向稳定，无数值或速度崩溃 |
| B3 full paired | 与 MSHNet 完全同协议长训练 | 单 seed 下是否超过 baseline | 预注册主指标达标，不许只报最好 epoch 的偶然点 |
| B4 confirmation | 多 seed + 独立数据集/最终 holdout | 是否可称稳定性能收益 | 成对增益和置信区间超过 seed/checkpoint 波动 |

B1 失败时不允许靠 end-to-end unfreeze 救活；B2 失败时不跑 400 epoch；B3 失败时不继续叠模块。官方 test 始终不用于选模型、选阈值或选 \(\eta\)。

## 3. DEA 主模型的双重准入标准

一个候选只有同时满足下列两类条件，才可能叫 DEA v1：

**结构准入：**

- 改变 decoder state transition/topology，而不是只改 final fusion 点；
- 不能化简为 \(a_s(x)\odot e_s\) 或凸组合 gate；
- 不依赖额外 attention/router/component pipeline 堆叠；
- 结构中的每个新变量都必须对应已诊断的 MSHNet 信息瓶颈；
- 严格包含 MSHNet identity 子模型。

**性能准入：**

- 先在 frozen holdout 显示超过 terminal/gate/direct-residual 强对照的修复能力；
- 再在 paired full training 下提高预注册主指标；
- 最后在多 seed 与至少一个独立数据集上超过方差；
- 收益必须集中在预先诊断的 FP/FN/冲突尺度事件，而不是只由参数量或更多计算量产生。

当前 `ComponentEvidenceControl` 不通过结构准入；`Decoder-Jacobian` 通过 baseline identity 与非门控初检，但不通过主创新性准入，只允许进入 B1 mechanics。

---

# 十七、建议的仓库改动顺序

```text
model/
├── MSHNet.py                       # 默认行为保持不变
├── mshnet_evidence_view.py         # 导出 decoder/features/contributions
├── dea_mshnet.py                   # 冻结 DEA v0，不继续修改
├── component_evidence_control.py   # control-only；H1-H3 后才允许新增
└── decoder_jacobian_control.py     # 只有 J0-J2 通过后才允许新增

utils/
└── component_evidence.py

tools/
├── build_component_evidence_manifest.py
├── probe_evidence_aliasing.py
├── component_subset_oracle.py
├── audit_component_context.py
└── audit_decoder_jacobian_vjp.py   # 先做冻结 mechanics，不先造主模型

tests/
├── test_mshnet_evidence_decomposition.py
├── test_candidate_generation_without_gt.py
├── test_component_matching_consistency.py
├── test_component_grouped_split.py
└── test_subset_oracle_identity.py
```

必须保证：

- all-scale subset 等于 baseline；
- 候选生成不读取 GT；
- component matching 与官方 PD/FA 相同；
- train/val/test image id 无交叉；
- baseline direct output 完全不变。

---

# 十八、最终决策树

| 结果 | 下一步 |
|---|---|
| 局部 nonlinear contribution probe 已能区分 | 做局部 nonlinear fusion，不做组件 DEA |
| context 不增加判别能力 | 放弃 evidence adjudication |
| context 有增益，但 subset oracle 无法修复 | 现有证据基底不足，改 feature representation |
| context 有增益，subset oracle 有明显空间 | 实现冻结 ComponentEvidenceControl |
| 控制模型不能超过 MSHNet | 停止该方向 |
| terminal control 已经足够 | 停留在更简单 terminal 方法，不做 Jacobian 主模型 |
| terminal control 不足且 P3/context 有效 | 进入 J0，检查 Decoder-Jacobian 资格 |
| J1/J2/J3 任一失败 | 当前不存在合格 DEA v1，不允许靠堆模块补救 |
| J1/J2/J3 均通过 | 只说明 mechanics 有效；先做 J5 近邻区分，不直接长训练 |
| J5 也通过 | 重新评估问题+结构组合是否具有足够新意，再决定 B2--B4 |
| J5 失败 | 将 Decoder-Jacobian 固化为 causal control，不称 DEA 主模型 |
| 只在 MSHNet 出现 | 定位为 MSHNet-specific limitation |
| 其他多尺度 baseline 也复现 | 才可宣称一般性的尺度证据混叠 |

---

# 十九、DEA 的最终出发点

最准确的一句话是：

> **DEA 研究多尺度红外小目标检测中的尺度证据语义混叠是否真实存在，以及解除这种混叠究竟只需要 terminal context weighting，还是必须在保留强 decoder 的前提下改变多通道 decoder state 的跨尺度演化。**

“可归因、可判定、可执行”可以保留，但只能放在方法性质中：

- 精确贡献分解使结果可归因；
- 条件上下文实验检验证据是否可判别；
- terminal control 与 decoder-state intervention 检验错误是否可修复。

它们不是研究问题本身。

---

# 二十、当前阶段的最终判断

当前最准确的项目状态是：

> **DEA v0 已被完整训练否定为最终主模型。当前只有一个由 4 FP、1 FN 和尺度贡献分析产生的 aliasing 假设。Prediction-only component oracle 说明 3/4 FP 可被 terminal retain/drop 局部修复，但 0/1 FN 可恢复；因此 ComponentEvidenceControl 是 strong causal control，不是主结构。Decoder-Jacobian 保留 MSHNet 强路径且具有 identity 退路，但主机制创新性已被强近邻压缩，只能做 B1 mechanics。目前不存在已经确定的 DEA v1 主模型，也不存在未经训练就能承诺的性能增益。**

因此，下一步不是继续修改 DEA v0，也不是立即堆叠组件模块，而是：

1. DEA v0 继续冻结，不再修补；
2. MSHNet 精确尺度贡献导出、global 16-subset 与 prediction-only component oracle 已完成；
3. 下一步是构建 OOF 组件证据数据集，完成 P0--P3 局部—组件—上下文 probe；
4. 只有 OOF context 增益稳定时，才实现冻结 `ComponentEvidenceControl` 作为 strong control；
5. 只有 terminal controls 不足且 J0 成立时，才运行 Decoder-Jacobian B1 mechanics；
6. Decoder-Jacobian 若无法通过 J5，则作为失败/因果对照归档，不进行 400-epoch 长训练；
7. 只有新候选同时通过结构准入和 B0--B4，才可定义为 DEA v1。
