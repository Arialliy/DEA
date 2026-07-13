# TRACE-MSHNet：以原子组件为预测变量的 MSHNet 统一重构方案

> **面向投稿**：AAAI-27 Main Technical Track
> **设计日期**：2026-07-13
> **代码审计基准**：`Arialliy/DEA` 当前公开 `main`，HEAD `a9435e8`（`Add protocol-locked evidence gates and CRWD controls`）
> **Baseline**：CVPR 2024 MSHNet
> **候选方法**：**TRACE-MSHNet**
> **全称**：**Tractable Root-cell Atomic Component Exponential-family MSHNet**
> **建议论文标题**：**From Pixels to Atomic Components: Exact Run-Semiring Prediction for Infrared Small Target Detection**
> **模型状态**：`T0/T1 IMPLEMENTATION GO`；长训练仅在预注册机械门通过后开放
> **性能声明状态**：尚无新模型实验，不允许填写或暗示虚构提升

---

## 0. 最终决策

本方案不再改 MSHNet 的 encoder、decoder、attention、multi-scale feature 或 SLS，也不再增加 gate、refinement、auxiliary head、ranking loss、topology loss、NMS 或 test-time solver。

当前仓库证据把第一个稳定异常边界定位在：

\[
\boxed{d_0\rightarrow\text{scalar prediction}}
\]

因此，唯一有充分代码依据的改动是：

> **冻结并复用 MSHNet 的 `input → d0`，物理删除四个逐像素 side heads 与最终 scalar fusion；将 `d0` 后的预测变量整体替换为一个可归一化、可精确求解的“空状态或完整组件”随机变量。**

锁定模型为：

\[
\boxed{\text{TRACE-MSHNet}}
\]

其核心不是一个可插拔模块，而是一次**预测范式替换**：

| MSHNet | TRACE-MSHNet |
|---|---|
| 每个像素一个 scalar logit | 每个根单元一个 `empty-or-component` 结构化变量 |
| SLS/IoU 类像素损失 | 一个精确条件似然 NLL |
| 四尺度 scalar maps 后融合 | 单一组件指数族，无后融合 |
| 阈值改变组件内部像素 | 阈值只选择或拒绝整个原子组件 |
| 组件在二值化后偶然形成 | 连通组件是模型的基本预测单位 |
| 组件置信度由像素场间接产生 | 组件存在概率由精确 partition function 得到 |

### 0.1 冻结结论

```text
input → d0                         FROZEN
MSHNet output_0...output_3         REMOVED FROM TRACE FORWARD
MSHNet final 4→1 convolution       REMOVED FROM TRACE FORWARD
SLS / BCE / IoU / topology mix     NOT USED BY TRACE
TRACE atom family                  SPEC FROZEN v1.0
TRACE exact run-semiring solver    IMPLEMENTATION GO
TRACE full long training           CONDITIONAL ON T0/T1/T2
```

### 0.2 这次如何避免继续失败

不能保证任何尚未训练的模型一定涨点，也不能保证录用；真正可以保证的是**不再用长训练验证未经证明的故事**。TRACE 在 GPU 长跑前设置三个廉价、可判死刑的门：

1. **几何门**：真实训练目标是否能被该原子族无损表示；
2. **数学门**：DP 的 `logZ`、MAP、梯度是否与穷举严格一致；
3. **微拟合门**：冻结 `d0` 后，组件变量是否至少能在小样本上辨认存在、根与支持。

任一门失败，方案立即停止，禁止再以增加模块、loss 或分支补救。

---

## 1. 截止日期与执行边界

AAAI-27 官方时间均按 UTC−12：

- 摘要截止：**2026-07-21 23:59 UTC−12**；
- 正文截止：**2026-07-28 23:59 UTC−12**；
- 补充材料与代码截止：**2026-07-31 23:59 UTC−12**。[R2]

摘要截止换算为台北时间约为 **2026-07-22 19:59**。本项目不使用这段时区缓冲，内部硬截止仍定为：

\[
\boxed{2026\text{-}07\text{-}21\ \text{台北时间前完成摘要与方法图}}
\]

### 1.1 摘要前必须得到的最低证据

摘要不是完整论文，但至少要在 7 月 20 日前得到：

- T0-A 几何门通过；
- T0-B 精确推断门通过；
- T1 frozen-`d0` 微拟合通过；
- T2 至少一个数据集、一个 seed、严格 train/dev 协议下的机制信号；
- 实测参数、显存、延迟；
- 不含虚构数字的摘要版本，以及可在 T2 后替换的数值占位符。

若 T2 未过，摘要不得把 TRACE 写成已验证有效的模型，只能停止该路线，而不是在最后两天拼接新模块。

---

## 2. 对当前 DEA/MSHNet 代码的严格判断

## 2.1 当前仓库状态

当前公开仓库的 README 已明确记录：

- Gate I 将首个稳定异常边界定位到 `d0 → scalar prediction`；
- `input → d0` 已冻结；
- Gate K 的 signed/unsigned post-`d0` readout 均为 NO-GO；
- RCP、ROOT–ADD–STOP、typed forest、CEMC、component-tree filtration、bounded-treewidth connected-subset CRF、hard-core polymer field、anytime component process、closed-walk occupancy readout等路线均被否决；
- OHR 已撤回并永久 NO-GO；
- 当前没有已获授权的顶会主模型或长训练路线。[R1]

因此，新设计必须同时满足：

1. 不重复仓库已否决机制；
2. 不把旧失败路线换名；
3. 不恢复 signed coordinate；
4. 不改冻结的前端；
5. 不保留 pixel branch 再旁接 component branch。

## 2.2 MSHNet 的实际 forward

MSHNet 先经过 U-Net 型 encoder–middle–decoder 得到：

\[
d_3,d_2,d_1,d_0.
\]

然后分别用四个 `1×1 Conv` 标量化：

```python
self.output_0 = nn.Conv2d(16,  1, 1)
self.output_1 = nn.Conv2d(32,  1, 1)
self.output_2 = nn.Conv2d(64,  1, 1)
self.output_3 = nn.Conv2d(128, 1, 1)
self.final    = nn.Conv2d(4, 1, 3, 1, 1)
```

推理中，低分辨率 mask logits 被双线性上采样，四路拼接后进入 `3×3 Conv(4,1)`：[R3]

\[
z=\operatorname{Conv}_{3\times3}
\left[
M_0(d_0),
\uparrow M_1(d_1),
\uparrow M_2(d_2),
\uparrow M_3(d_3)
\right].
\]

这不是 noisy-OR，也不是显式的组件模型；上采样 logits 只是 final convolution 的可学习输入特征。此前 OHR 把中间 sigmoid 值当作独立 occupancy probability，因此被否决是正确的。

## 2.3 MSHNet 真正强在哪里

仓库 Gate I 的 hard-core 观测显示：

```text
d0 distinct:      44 / 48
mask0 distinct:   21 / 48
final z distinct: 22 / 48

d0 → mask0 distinct-state drops: 23 / 48
d0 → z     distinct-state drops: 22 / 48
matched-success controls:          0 drops
```

这说明当前最有价值的结论不是“decoder 不好”或“attention 不够”，而是：

> **困难目标信息通常仍存在于 `d0` 的 16 维高分辨率表征中，但在压缩成独立像素 scalar field 时大量消失。**

同时，历史证据已否决“只换 final fusion”：71 个 miss 中 `d0` 与 final mean-margin 有 69 个同号，scale cancellation 在成功对照中反而更常见，因此 fusion suppression 不是统一根因。[R4]

## 2.4 为什么普通新 head 也不成立

Gate K 已用冻结前端、参数匹配的 readout 验证：

```text
refit_signed_standardized Pd @ FA=1/5/10/20:
0.003802 / 0.003802 / 0.003802 / 0.003802

original_final_z Pd:
0.148289 / 0.422053 / 0.798479 / 0.904943

fixed-logit-0 IoU:
signed = 0.005778
native = 0.723586
```

因此以下解释已不可再用：

- 只差一个 signed projection；
- 只差一个更强 `1×1/3×3` head；
- 加 attention 就能把 `d0` 读出来；
- 再做一个 scalar ranking loss 即可。

下一步必须改变的是**输出变量和训练归一化结构**，而不是 scalar readout 的函数形式。

## 2.5 评价协议与当前模型的错位

仓库官方 legacy 指标流程是：

1. 对 dense logits 做 sigmoid 与阈值化；
2. 使用 8-connectivity 得到预测组件；
3. 预测组件与 GT 组件只有在质心距离严格 `<3 px` 时才匹配；
4. 未匹配预测组件的面积计入 FA；
5. 目标级 Pd 由匹配组件数决定。[R6]

也就是说，训练基本单位是 pixel，最终评价基本单位却是 connected component。对于极小目标，几个像素的排序改变就可能造成：

- 完整目标变成多个碎片；
- 两个相邻目标粘连；
- 噪声像素形成孤立假组件；
- 组件质心偏移超过 3 像素；
- 阈值变化时组件形状和数量同时突变。

TRACE 的目标不是简单说“训练–评价不一致”，而是给出一个**可归一化、可训练、可精确推断、可按整组件阈值化**的替代预测变量。

---

## 3. 明确研究问题

### 3.1 研究问题

> 当 MSHNet 的 `d0` 已保留多数困难目标的可区分信息，但独立像素标量化在低组件虚警预算下无法稳定形成正确组件时，能否把输出变量改成一个可精确归一化的完整组件，使组件存在、组件支持和组件置信度由同一个概率模型联合产生？

### 3.2 可证伪假设

\[
\mathcal H_{\text{TRACE}}:
\]

在冻结相同 MSHNet `input→d0` 的条件下，若把逐像素 scalar field 替换成 exact normalized atomic component field，则：

1. 弱但空间一致的 `d0` evidence 可在合法组件内累积；
2. 所有错误组件候选进入同一 partition，因此空单元 likelihood 会直接惩罚假组件；
3. 阈值只决定整个组件是否输出，不再逐像素改变组件内部；
4. 在固定组件 FA 预算下，目标 Pd 应优于相同容量的 dense scalar control；
5. 改善必须跨相邻预算、matcher、seed 与数据集稳定，而非单点 oracle。

这五条中任一核心机制未出现，TRACE 不成立。

---

# 4. TRACE-MSHNet：完整模型定义

## 4.1 总体计算图

\[
I
\xrightarrow[\text{frozen}]{\text{MSHNet input→d0}}
D\in\mathbb R^{16\times H\times W}
\xrightarrow{f_\theta}
\Theta=(A,U)
\xrightarrow{\text{exact run-semiring}}
\{P(Y_g=\varnothing),P(Y_g=C)\}_{g,C}.
\]

其中：

- `D=d0`；
- `fθ` 是唯一可训练 natural-parameter map；
- `A` 是 root sufficient-statistic 的自然参数场；
- `U` 是 support sufficient-statistic 的自然参数场；
- 两者不是两个预测 head，而是同一个指数族状态的两种自然坐标；
- `Y_g` 是根单元 `g` 的唯一预测随机变量。

模型不再计算：

```text
output_0
output_1
output_2
output_3
final
```

## 4.2 根单元与全局变量

把图像划分为互不重叠的根单元：

\[
\mathcal G=\{g\},\qquad
\operatorname{core}(g)\subset\Omega.
\]

首选单元边长：

\[
s=4.
\]

每个单元有一个随机变量：

\[
Y_g\in\{\varnothing\}\cup\mathcal C_g,
\]

其中：

- `∅` 表示该根单元不拥有目标；
- `C∈C_g` 表示该根单元拥有一个完整目标组件；
- 每个组件通过一个唯一 canonical root 归属一个单元。

全图条件分布采用一个明确的因子化：

\[
P(Y\mid D)=\prod_{g\in\mathcal G}P(Y_g\mid D).
\]

这是模型当前唯一的独立性假设。它不声称解决全局实例排斥；跨单元 overlap/contact 是预注册失败指标，而不是用 NMS 临时修复。

## 4.3 原子组件状态族

一个非空原子组件由连续若干行上的水平 run 构成：

\[
C=(I_{y_0},I_{y_0+1},\ldots,I_{y_1}),
\qquad
I_y=[l_y,r_y],\quad l_y\le r_y.
\]

要求：

1. 每一行恰有一个连续区间，即 row-convex；
2. 非空行在垂直方向连续；
3. 相邻两行的 runs 满足 8-connectivity：

\[
l_{y+1}\le r_y+1,
\qquad
r_{y+1}\ge l_y-1.
\]

等价地，两条 run 在水平扩张 1 像素后相交。

### 4.3.1 Canonical root

定义：

\[
\operatorname{root}(C)=(y_0,l_{y_0}),
\]

即组件最上方非空行的最左像素。

并要求：

\[
\operatorname{root}(C)\in\operatorname{core}(g).
\]

由此，每个可表示组件在该状态族中只有一个 owner cell。

### 4.3.2 诚实的表达能力边界

TRACE v1.0 只建模：

> **连续行、每行单 run、相邻 run 8-connected 的 row-run chain。**

它不声称可表达任意拓扑、空洞、分叉或非 row-convex mask。是否适合 IRSTD 必须由 T0-A 在真实训练 mask 上验证，不能靠“小目标通常简单”这类经验判断替代。

## 4.4 局部组件窗口

每个根单元只在局部窗口 `Ω_g` 中生成组件。窗口大小不得用 test mask 决定。

从训练集 GT 统计 root-relative extents：

\[
\Delta_{\text{up}},
\Delta_{\text{down}},
\Delta_{\text{left}},
\Delta_{\text{right}}.
\]

固定窗口为训练最大值加预声明 margin：

\[
\Omega_g=
[-m_u,m_d]\times[-m_l,m_r],
\]

边界处裁剪并用合法状态 mask 处理。

建议 margin：1–2 px；最终值在 T0-A 前冻结。

## 4.5 唯一可训练映射

建议最小实现：

```python
self.potential_map = nn.Sequential(
    nn.Conv2d(16, 16, kernel_size=1, bias=True),
    nn.GELU(),
    nn.Conv2d(16, 2, kernel_size=1, bias=True),
)
```

参数量：

\[
(16\times16+16)+(16\times2+2)=306.
\]

原 MSHNet 四个 side heads 与 final fusion 的参数量为：

\[
17+33+65+129+37=281.
\]

因此 TRACE 的可训练映射只多 25 个参数，约为原预测端的同量级替换，而不是大容量 decoder。

输出：

\[
f_\theta(D_p)=(a_p,u_p).
\]

重要语义：

- `a_p`、`u_p` 是 energy/natural parameters；
- 它们不是概率；
- 不对其单独做 sigmoid；
- 不为它们分别定义 BCE、IoU 或 auxiliary target；
- 概率只在完整 `empty-or-component` 状态空间归一化后出现。

## 4.6 联合 root/support 指数族

对组件 `C` 定义未归一化能量：

\[
E_g(C;D)
=
 a_{\operatorname{root}(C)}
+
\sum_{p\in C}u_p
-
\log K_g.
\]

空状态能量固定为：

\[
E_g(\varnothing;D)=0.
\]

其中：

\[
K_g=|\mathcal C_g|
\]

是该根单元合法组件状态数，由同一个 DP 在全零 score 下精确计算并缓存。

### 4.6.1 为什么必须同时有 root 与 support sufficient statistics

若只用：

\[
E(C)=\sum_{p\in C}u_p,
\]

一个真实目标可能被邻近根单元解释为多个后缀、截断或局部复制组件。加入 canonical root statistic 后：

- 正确 cell 的 positive NLL 增强正确 root；
- 同 cell 的 partition 抑制其他 root 与其他 shape；
- 邻接 cell 作为 empty cell，其 likelihood 抑制从目标中部开始的错误后缀组件。

这不是“root head + mask head”后融合；root 与 support 是**同一个随机变量的联合充分统计量**。

## 4.7 Cardinality-corrected base measure

若不减 `log K_g`，合法形状更多的边界/窗口会天然拥有更大的 positive partition，导致组件存在先验被状态空间大小污染。

使用：

\[
E_g(C)=\tilde E_g(C)-\log K_g
\]

后，若所有 root energy 都为常数 `b`，support energy 为 0，则：

\[
\sum_{C\in\mathcal C_g}\exp E_g(C)
=K_g\exp(b-\log K_g)
=\exp b.
\]

因此：

\[
P(Y_g\ne\varnothing)=\sigma(b),
\]

与候选形状数量无关。

这是一个**base-measure correction**，不是新增 loss 或 calibration module。

## 4.8 归一化条件分布

正状态 partition：

\[
Z_{g,+}
=
\sum_{C\in\mathcal C_g}\exp E_g(C;D).
\]

完整 partition：

\[
Z_g=1+Z_{g,+}.
\]

于是：

\[
P(Y_g=\varnothing\mid D)=\frac{1}{Z_g},
\]

\[
P(Y_g=C\mid D)=\frac{\exp E_g(C;D)}{Z_g}.
\]

组件存在 log-odds 为：

\[
o_g=\log Z_{g,+},
\]

组件存在概率为：

\[
p_g=P(Y_g\ne\varnothing\mid D)=\sigma(o_g).
\]

这里的 `p_g` 才是 proper normalized probability。TRACE 不再重解释 MSHNet 的 SLS side logits。

---

# 5. Exact Run-Semiring Dynamic Programming

## 5.1 Run score

在局部窗口的一行 `y`，区间 `[l,r]` 的 support score：

\[
\rho_y(l,r)=\sum_{x=l}^{r}u_{y,x}.
\]

利用行前缀和可在 `O(1)` 得到每个 run score。

## 5.2 Sum-product 状态

定义：

\[
F_y(l,r)
\]

为所有在第 `y` 行以 run `[l,r]` 结束的合法组件链的 log-sum unnormalized mass。

启动项：

\[
S_y(l,r)=
\begin{cases}
 a_{y,l}-\log K_g,& (y,l)\in\operatorname{core}(g),\\
-\infty,&\text{otherwise}.
\end{cases}
\]

递推：

\[
F_y(l,r)
=
\rho_y(l,r)
+
\operatorname{LSE}
\left(
S_y(l,r),
T(F_{y-1};l,r)
\right),
\]

其中：

\[
T(F;l,r)
=
\operatorname{LSE}_{\substack{a\le r+1\\b\ge l-1\\a\le b}}
F(a,b).
\]

`a≤r+1` 且 `b≥l−1` 正好编码相邻两行 run 的 8-connectivity。

一个组件可以在任意合法根行开始，并在任意行结束。因此：

\[
\log Z_{g,+}
=
\operatorname{LSE}_{y,l,r}F_y(l,r).
\]

## 5.3 从朴素 `O(W^4)` 到精确 `O(W^2)` 转移

若每个当前 run 都枚举所有上一行 run，单行转移为 `O(W^4)`。

对上一行状态矩阵 `F[a,b]`，先把无效 `a>b` 置 `−∞`，再计算：

### 第一步：对右端点做 suffix log-sum-exp

\[
Q[a,q]=\operatorname{LSE}_{b\ge q}F[a,b].
\]

可用反向 `logcumsumexp` 实现。

### 第二步：对左端点做 prefix log-sum-exp

\[
P[p,q]=\operatorname{LSE}_{a\le p}Q[a,q].
\]

可用正向 `logcumsumexp` 实现。

### 第三步：常数时间查询

\[
T(F;l,r)
=
P[\min(W-1,r+1),\max(0,l-1)].
\]

每行只需处理一个 `W×W` 状态矩阵：

\[
\boxed{O(W^2)\text{ per row}}
\]

每个局部 field 的复杂度：

\[
\boxed{O(H_{\text{loc}}W_{\text{loc}}^2)}.
\]

## 5.4 Max-semiring MAP

把 `LSE/logcumsumexp` 换成 `max/cummax`，并保存 backpointer，即得：

\[
\hat C_g
=
\arg\max_{C\in\mathcal C_g}E_g(C;D).
\]

同一套状态、兼容关系和边界 mask 被复用；不能为 MAP 再写一套近似 grower。

## 5.5 Marginals

由于 `logZ` 可微：

\[
\frac{\partial\log Z_g}{\partial a_p}
=P(\operatorname{root}(Y_g)=p\mid D),
\]

\[
\frac{\partial\log Z_g}{\partial u_p}
=P(p\in Y_g\mid D).
\]

因此 root marginal 与 support marginal 由同一 partition 自动得到，只用于诊断与可视化，不作为额外输出分支或额外 loss。

## 5.6 必须证明的正确性

论文与单元测试需要覆盖：

1. **唯一编码**：每个状态族内组件对应唯一 run chain；
2. **不漏不重**：递推恰好枚举每个合法 atom 一次；
3. **连通性**：兼容条件与 8-connectivity 等价；
4. **复杂度**：orthant cumulative transform 与朴素枚举等价，且为 `O(HW²)`；
5. **base measure**：`−log K_g` 消除 shape-cardinality 对存在先验的影响；
6. **renderer 等价性**：dense compatibility output 的阈值结果等于被选 atom 的并集。

---

# 6. 单一训练目标

给定 GT，每个根单元有：

\[
Y_g^*\in\{\varnothing\}\cup\mathcal C_g.
\]

## 6.1 Empty cell

\[
\ell_g
=-\log P(Y_g=\varnothing\mid D)
=\log Z_g.
\]

## 6.2 Positive cell

\[
\ell_g
=-\log P(Y_g=C_g^*\mid D)
=\log Z_g-E_g(C_g^*;D).
\]

## 6.3 全图目标

\[
\boxed{
\mathcal L_{\text{TRACE}}
=\sum_{g\in\mathcal G}\ell_g
}
\]

允许除以固定的 cell 数或 batch 大小作为数值尺度，但不增加：

- SLS；
- BCE；
- soft-IoU；
- Dice；
- center loss；
- ranking loss；
- topology loss；
- FA surrogate；
- distillation；
- consistency loss。

## 6.4 稀有正样本与初始化

令训练集中非空根单元先验为：

\[
\pi_{\text{train}}
=
\frac{\#\text{positive root cells}}
{\#\text{all root cells}}.
\]

初始化 root-channel 最后 bias：

\[
b_\pi=\operatorname{logit}(\pi_{\text{train}}),
\]

support-channel bias 初始化为 0，其他权重用小随机值。

由于 `−logK` correction，初始：

\[
P(Y_g\ne\varnothing)\approx\pi_{\text{train}}.
\]

对于每张约一个目标、数千 empty cells 的情形，单个 empty cell 梯度约为 `π`，聚合后的负样本梯度与正 cell 的 `O(1)` 梯度自然同量级，减少一开始全空坍缩的风险。

## 6.5 负单元采样

默认优先使用全 cell 精确 NLL。

若显存或吞吐不足，只允许：

- 所有 positive cells 全保留；
- 按已知 inclusion probability 采样 negative cells；
- 用 Horvitz–Thompson 权重得到原全 cell NLL 的无偏估计。

这仍是同一个 likelihood 的随机估计，不得改成 hard-negative ranking 或第二个目标。

---

# 7. 原子级推理与阈值语义

## 7.1 原生输出

模型原生输出是 atom list：

```text
(root_cell_id, existence_probability, MAP_component)
```

对每个 cell：

\[
p_g=\sigma(\log Z_{g,+}),
\qquad
\hat C_g=\arg\max_CE_g(C).
\]

## 7.2 阈值只选择整个组件

\[
\mathcal A_\tau
=
\{\hat C_g:p_g\ge\tau\}.
\]

阈值变化只决定 `C_hat_g` 是否进入输出，不改变其内部支持。

因此：

\[
\boxed{\text{support is threshold-invariant}}
\]

这与 dense pixel logits 的关键差别是：阈值不再逐像素侵蚀、膨胀或打碎一个候选目标。

## 7.3 与仓库 dense metric 的兼容渲染

仓库指标需要 dense logit map。定义：

\[
z_p
=
\max_{g:p\in\hat C_g}\log Z_{g,+},
\]

没有 atom 覆盖的位置置固定低值，例如 `−30`。

则对任意阈值 `τ`：

\[
\mathbf1[\sigma(z_p)\ge\tau]
=
\mathbf1\left[
 p\in\bigcup_{g:p_g\ge\tau}\hat C_g
\right].
\]

这不是 learned refinement，而是 atom list 到官方 dense interface 的确定性序列化。

## 7.4 不允许 NMS 或事后 set packing

不同根单元的 MAP atom 可能重叠、接触或在 union 后粘连。v1.0 不增加：

- NMS；
- Hungarian set packing；
- graph arbitration；
- learned overlap resolver；
- morphology opening；
- connected-component repair。

这些现象被直接计入 failure metrics。若超门，说明当前独立 cell factorization 不成立，模型停止，而不是补丁式续命。

---

# 8. 为什么 TRACE 不是模块堆叠

TRACE 的所有新增计算都由同一个概率定义必然推出：

```text
一个冻结证据场 D=d0
       ↓
一个 natural-parameter map fθ
       ↓
一个 empty-or-component 指数族变量 Yg
       ↓
一个 exact partition / MAP semiring
       ↓
一个 exact NLL
       ↓
一个 atom threshold
```

它没有：

- backbone enhancement module；
- attention module；
- frequency branch；
- edge branch；
- root proposal branch；
- mask refinement branch；
- multi-loss training；
- auxiliary supervision；
- NMS；
- post-processing optimizer。

### 8.1 两个 natural channels 为什么不算两个 head

`a_p` 与 `u_p` 类似指数族中不同 sufficient statistics 的自然参数：

\[
E(C)=\langle\theta(D),T(C)\rangle.
\]

其中：

\[
T(C)=
\left(
\mathbf1[p=\operatorname{root}(C)],
\mathbf1[p\in C]
\right).
\]

模型没有分别输出“root probability”和“mask probability”，也没有把两种预测再融合；它只输出同一状态能量所需的两个坐标。

### 8.2 代码层硬约束

以下测试必须成立：

```python
assert not trace_forward_called("output_0")
assert not trace_forward_called("output_1")
assert not trace_forward_called("output_2")
assert not trace_forward_called("output_3")
assert not trace_forward_called("final")
```

保留原类文件用于 baseline 对照可以，但 TRACE forward 不能执行这些层。

---

# 9. 预期论文创新点

以下是可防守、但仍需文献最终复核的创新声明。

## 创新点 1：原子组件条件指数族

把红外小目标检测的输出从 pixel scalar field 改为：

\[
Y_g\in\{\varnothing\}\cup\mathcal C_g.
\]

组件存在、root、support 与置信度属于同一归一化随机变量，而非 objectness head、mask head 和后处理的组合。

## 创新点 2：二维 run-chain 的精确 semiring 推断

对 row-run 8-connected atoms，设计可在 sum-product 与 max-product 间切换的正交累积 DP：

\[
O(HW^2)
\]

同时得到 exact partition、MAP、root/support marginals，而不是 beam search、autoregressive growth 或近似 connected-set inference。

## 创新点 3：shape-cardinality corrected base measure

通过：

\[
-\log K_g
\]

使组件存在先验与可用 shape 数量解耦，避免边界 cell 或不同窗口因候选数不同而产生先验漂移。

## 创新点 4：原子阈值语义

阈值作用于完整 atom 的 normalized existence probability，组件内部支持在阈值扫描中保持不变，并有一个与 dense metric **精确等价**的确定性 renderer。

## 创新点 5：证据定位驱动的物理替换

不是在 MSHNet 上堆模块，而是依据跨 seed 的代码级证据，冻结 `input→d0`，仅替换已定位的 `d0→scalarization` 边界；新预测端参数量与旧端接近。

### 9.1 不能写进论文的夸大表述

禁止使用：

- “first connected-component predictor”；
- “first row-convex model”；
- “first exact CRF”；
- “first mask-level detector”；
- “models arbitrary connected masks”；
- “guarantees lower false alarms”；
- “guarantees AAAI acceptance”；
- “the first ever” 等绝对首创措辞。

安全表述应为：

> 在截至 2026-07-13 的有限检索中，尚未发现把 **root-cell ownership、empty-or-run-chain normalized exponential family、cardinality correction、exact sum/max semiring inference 与 atomic threshold renderer** 同时用于 IRSTD 的直接先例。该判断不是穷尽性首创证明。

---

# 10. 与近期模型的明确区分

## 10.1 IRSTD 直接竞争路线

| 工作 | 主要机制 | 与 TRACE 的本质区别 |
|---|---|---|
| MSHNet, CVPR 2024 [R7] | 多尺度 side prediction + SLS | 仍是 dense pixel logits；TRACE 冻结其前端并替换输出随机变量 |
| IRMamba, AAAI 2025 [R8] | Pixel Difference Mamba + layer restoration | 改 feature extractor/encoder-decoder；TRACE 不改 feature backbone |
| PConv + SD Loss, AAAI 2025 [R9] | 低层 pinwheel convolution + 动态尺度损失 | 卷积与 loss 增强；TRACE 是单一组件 likelihood |
| DEFANet, AAAI 2026 [R10] | edge-target 双路径、frequency-aware enhancement | 双路径 feature collaboration；TRACE 无 edge/frequency 分支 |
| NS-FPN, CVPR 2026 [R11] | low-frequency purification + spiral sampling FPN | 可插拔 feature pyramid；TRACE 删除多尺度 scalar head，不做 feature plugin |
| InvDet, CVPR 2026 [R12] | invertible encoder + reconstruction guidance | 针对 encoder 信息保留；本仓库证据已显示 `input→d0` 强，TRACE 不重做 encoder |

因此 TRACE 的竞争叙事不能写成“更强特征增强”，而应写成：

> 近期 IRSTD 主要仍通过 backbone、frequency、edge、state-space、sampling 或 loss 改善 dense predictor；TRACE 研究的是 evidence 已存在之后，预测变量如何与组件级低 FA 决策对齐。

## 10.2 与 MaskFormer/Mask2Former 的区别

MaskFormer 把 segmentation 改写为一组 mask-classification outputs；Mask2Former 使用 query、Transformer decoder、masked attention 与 bipartite matching。[R13][R14]

TRACE 不使用：

- learned object queries；
- fixed `N` query set；
- Hungarian training assignment；
- mask/class 两个独立输出；
- cross-attention decoder；
- query competition。

TRACE 使用的是固定空间根单元，每个单元一个 proper normalized `empty-or-atom` variable，并在受限 shape family 上精确求 partition。

## 10.3 与中心/半径、StarDist 和 one-stage instance methods 的区别

StarDist 等方法由 object probability 与若干 radial distances 参数化 star-convex object，通常需要候选抑制。[R18]

TRACE：

- 不回归中心半径；
- 不使用 objectness + geometry 两套 loss；
- 不通过 NMS 仲裁；
- shape confidence 来自对所有合法 run chains 的 exact normalization；
- MAP shape 不是固定维参数回归，而是结构化状态解码。

## 10.4 与 autoregressive growth / flood filling 的区别

自回归 region growing 依赖：

- 起点 recall；
- 动作顺序；
- teacher forcing；
- stop action；
- exposure bias；
- beam/search budget。

仓库此前 RCP/ROOT–ADD–STOP 已被否决。TRACE 不预测动作序列，而是一次性对所有合法 run chains 求和或取最大值；相同组件不会因不同生长顺序重复计数。

## 10.5 与 semi-Markov CRF 的区别

Semi-Markov CRF 以一维 span 为基本单元并在序列上做 segment-level inference。[R17]

TRACE 借鉴“结构单元而非 token/pixel”的一般思想，但其状态是二维连续行上的 interval chain；相邻 interval 的 8-connectivity 和二维 orthant cumulative transform 是当前问题的关键。

## 10.6 与经典 row-convex/connected-row-convex 算法的区别

图像分割中 row-convex shape 的多项式算法，以及 CSP 中 connected row-convex constraints 的可解性都早已有研究。[R15][R16]

因此：

- row-convex 本身不是创新；
- “使用 DP”本身不是创新；
- “连通约束可解”本身不是创新。

TRACE 需要主张的窄组合是：

1. 神经条件自然参数场；
2. 根单元所有权；
3. 一个 empty-or-2D-run-chain 指数族；
4. shape-cardinality base measure；
5. exact log-partition/MAP/marginal 的 semiring 实现；
6. 与组件 FA 阈值语义一致的原子输出。

---

# 11. 直接先例压力测试与新颖性风险

## 11.1 高风险相邻领域

投稿前必须继续检索：

- exact structured prediction over connected subgraphs；
- neural CRF over convex/orthogonally-convex shapes；
- scanline/run-length instance segmentation；
- weighted automata / semiring parsing for images；
- object-centric local partition functions；
- anchored shape grammars；
- root-cell mask distributions；
- exact differentiable DP for polyominoes；
- row-convex binary image enumeration；
- segmental energy-based instance detection。

## 11.2 一旦发现直接先例后的判定

若找到论文已同时包含：

```text
anchored empty-or-component variable
+ row-run 2D state family
+ exact log-partition and MAP
+ neural conditional potentials
+ component-level threshold semantics
```

则 TRACE 的方法新颖性应判 **NO-GO**，不能只换术语后投稿。

若先例只包含其中一部分，应在 Related Work 中明确承认，并把贡献压缩到可证明不同的部分。

---

# 12. GT 编码规范

## 12.1 组件提取

- 使用与官方 metric 一致的 8-connectivity；
- 每个 GT connected component 独立编码；
- canonical root 为 topmost row 的 leftmost pixel；
- 按 root 所属 cell 分配变量。

## 12.2 Root-cell collision

如果两个 GT components 的 roots 落入同一个 cell，当前变量无法同时表示二者。

预声明处理顺序只有一次：

1. 首选 `s=4`；
2. 若训练集出现 collision，允许切换到预声明 fallback `s=2`；
3. 若 `s=2` 仍有 collision，TRACE v1.0 立即 NO-GO。

禁止临时加入：

- 每 cell 两个 slots；
- query set；
- collision head；
- post-hoc split；
- dataset-specific cell size 搜索。

## 12.3 Row-convex exactness

对每个 GT component，逐行检查 foreground x 坐标是否为一个无间断区间，并检查非空行连续、相邻 run 8-connected。

主模型不允许把非 row-convex GT 投影成 row hull，因为那会改变标签和 IoU。T0-A 要求训练 GT **100% 精确可表示**；否则停止。

## 12.4 Window coverage

训练组件相对 root 的所有 support 必须落入预声明局部 window，并保留 margin。任何训练组件被裁剪即失败。

---

# 13. 工程实现方案

## 13.1 新增文件

```text
model/
  trace_mshnet.py
  trace_run_semiring.py

utils/
  trace_codec.py
  trace_geometry.py

 tools/
  audit_trace_geometry.py
  verify_trace_dp.py
  train_trace.py
  evaluate_trace.py

 tests/
  test_trace_codec.py
  test_trace_semiring.py
  test_trace_frozen_front.py
  test_trace_renderer.py
  test_trace_prior_calibration.py
```

保持：

```text
model/MSHNet.py
```

不被修改，作为 canonical baseline。

## 13.2 模型骨架

```python
from __future__ import annotations

import torch
from torch import nn

from model.mshnet_d0_backbone import MSHNetD0Backbone
from model.trace_run_semiring import RootCellRunSemiring


class TRACEMSHNet(nn.Module):
    """Frozen MSHNet evidence front plus one atomic-component field."""

    def __init__(
        self,
        *,
        input_channels: int,
        cell_size: int,
        local_height: int,
        local_width: int,
        log_cardinality: torch.Tensor,
    ) -> None:
        super().__init__()
        self.front = MSHNetD0Backbone(input_channels=input_channels)
        self.potential_map = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=1, bias=True),
        )
        self.field = RootCellRunSemiring(
            cell_size=cell_size,
            local_height=local_height,
            local_width=local_width,
            log_cardinality=log_cardinality,
        )
        self._freeze_front()

    def _freeze_front(self) -> None:
        self.front.eval()
        for parameter in self.front.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "TRACEMSHNet":
        super().train(mode)
        # Prevent BatchNorm buffers from drifting.
        self.front.eval()
        return self

    def forward(self, image: torch.Tensor):
        with torch.no_grad():
            d0 = self.front(image)
        natural = self.potential_map(d0.float())
        return self.field(
            root_energy=natural[:, 0],
            support_energy=natural[:, 1],
        )
```

## 13.3 DP 实现原则

- interval score 用 prefix sums；
- 全部无效 `l>r` 状态为 `−inf`；
- sum semiring 用 `torch.logcumsumexp`；
- max semiring用 `cummax` 并保留 index；
- cells 作为 batch 维向量化；
- local windows 用 `unfold` 或显式 gather；
- 对 cells 分 chunk，避免一次展开全图导致显存峰值；
- partition 计算至少 FP32；T0-B reference 使用 FP64；
- AMP 只允许作用在 frozen front 与 pointwise map，DP 默认关闭 autocast，直至稳定性通过；
- 边界 cell 用合法 root/support mask，不通过 padding 生成伪状态；
- `log K_g` 缓存需绑定 image size、cell size、window、boundary policy 与代码 hash。

## 13.4 初始化细节

不得将两层 `potential_map` 的所有权重初始化为精确 0，否则第一层可能无梯度。建议：

- 两层 weights：小方差 Xavier/normal；
- hidden bias：0；
- support output bias：0；
- root output bias：`logit(pi_train)`；
- 在初始化测试中验证全图平均 `p_nonempty≈pi_train`。

## 13.5 Checkpoint 与 provenance

每个 TRACE run 必须保存：

```text
repo_commit
baseline_checkpoint_sha256
front_state_key_hash
front_tensor_sha256
BN_buffer_sha256
fixed_d0_anchor_sha256
trace_spec_version
geometry_manifest_sha256
cell_size
window_extents
logK_cache_sha256
split_manifest_sha256
seed
matcher
threshold_grid_or_exact_score_policy
```

---

# 14. Fail-closed 实验门

# 14.1 T0-A：几何可表示性门（CPU，当天完成）

必须输出每个数据集：

```text
number_of_gt_components
row_convex_exact_count / total
consecutive_row_count / total
unique_root_count / total
root_collision_count(s=4)
root_collision_count(s=2)
max_relative_extents
window_coverage
encode_decode_bit_exact_count / total
```

### PASS 条件

- 所有训练 GT components 100% row-convex exact；
- 非空行 100% 连续；
- encode/decode bit-exact 100%；
- chosen cell size 下 root collision = 0；
- local window coverage = 100%；
- 每个组件唯一 owner/root；
- 所有统计仅来自 train split。

### FAIL

任一失败：

\[
\boxed{\text{TRACE-R NO-GO before GPU}}
\]

不允许用 hull projection、slot expansion 或 morphology 补救。

# 14.2 T0-B：数学与数值精确性门

在 `4×4`、`5×5` tiny windows 穷举所有合法 chains，与 DP 对比。

### 必测

1. `logZ`：
   \[
   |\log Z_{DP}-\log Z_{brute}|<10^{-6}\quad(\text{FP64})
   \]
2. MAP energy 与 MAP mask 完全一致；
3. 随机无 tie 时 backpointer 完全一致；
4. autograd 与 finite difference 最大误差 `<10^{-4}`；
5. zero-score count-DP 的 `K` 与穷举计数一致；
6. prior calibration：常数 root bias `b`、zero support 下，`p_nonempty=sigmoid(b)`，误差 `<10^-6`；
7. score 范围 `[-30,30]` 无 NaN/Inf；
8. 阈值扫描不改变单 atom support；
9. renderer 的 dense threshold 与 atom union bit-exact；
10. frozen front 参数与 BN buffers 在 train step 前后 hash 不变。

### 初始工程预算

在 batch 1、`256×256` 输入上：

- 新预测端 added latency 不超过 canonical MSHNet 总延迟的 2 倍；
- 新增峰值显存不超过 2 GB；
- 不出现 Python per-cell loop。

若只因 kernel/chunk 不达标，允许优化同一个 DP 实现；不允许改状态族或增加近似 proposal branch。

# 14.3 T1：冻结 `d0` 微拟合门

固定小 manifest，例如：

```text
64 positive root cells
256 hard negative root cells
```

只训练 `potential_map`，不训练 front。

### PASS 条件

- exact NLL 相对初始化下降至少 80%；
- positive cell median `p_nonempty ≥ 0.95`；
- negative cell median `p_nonempty ≤ 0.05`；
- positive conditional MAP median IoU `≥0.95`；
- 至少 95% 正样本质心距离 `<3 px`；
- wrong-root、suffix-copy、duplicate rate 接近 0，并明确报告；
- 无 all-empty collapse；
- front/BN/d0 anchors hash 不变。

### FAIL

T1 失败代表：冻结 `d0` 与当前 atom family/energy 无法完成最基本的存在和支持辨认。

禁止通过：

- 加 attention；
- 加 root BCE；
- 加 IoU loss；
- 加 decoder；
- 增大 MLP；
- 解冻 front；
- 引入 teacher model

进行补救。

# 14.4 T2：单数据集、单 seed 机制门

先选择一个数据集和一个 seed，使用**train-only dev selection**，不能继续依赖 test-selected checkpoint 得出论文性能。

### 对照

| ID | 对照 | 目的 |
|---|---|---|
| A | canonical MSHNet | 原 baseline |
| B | frozen d0 + 同容量 dense potential map | 排除只是小 MLP/重新训练 head |
| C | support-only atom field | 验证 root statistic 是否抑制截断副本 |
| D | point atoms | 验证完整 shape accumulation 是否必要 |
| E | rectangle atoms | 排除只是简单 bounding shape prior |
| F | independent per-row intervals | 验证 inter-row connectivity 是否必要 |
| G | full TRACE | 完整方法 |

所有对照必须遵守同一 front、split、seed、selection、训练预算与测试协议。

### 硬门

相对 A 与 B，full TRACE 必须同时满足：

1. official legacy Pd 在 `FA=1/5/10/20` 中至少两个相邻预算绝对提升 `≥0.02`；
2. audit Hungarian 同方向；
3. fixed-threshold IoU 下降不超过 `0.005`；
4. 恢复预注册的 `d0-distinct / final-missed` targets，而非只增加容易目标；
5. 实际 achieved FA 不超预算；
6. overlap、contact、duplicate、wrong-root 各自 `<0.5%`；
7. 无 test geometry、test threshold 或 test checkpoint selection leakage。

任一不满足：

\[
\boxed{\text{TRACE MAIN ROUTE STOP}}
\]

# 14.5 T3：论文级放行门

至少：

```text
3 datasets × 3 seeds
```

### 放行标准

- 至少 2 个数据集；
- 至少 2 个相邻 FA budgets；
- 每个满足数据集在至少 2/3 seeds 同方向；
- pooled absolute Pd gain `≥0.02`；
- IoU drop `≤0.005`；
- official legacy 与 Hungarian 结论一致；
- latency、显存、参数量完整报告；
- 不使用 best-seed 表格；
- 不把 oracle threshold 当 deployable result。

---

# 15. 论文评估协议

## 15.1 旧 artifact 的用途边界

仓库已明确指出，full-train/no-val/periodic-test 的 selected checkpoint 是 test-selected，不能作为 untouched-test 的无偏估计。[R1]

这些 artifact 可用于：

- 定位瓶颈；
- 构造 hard-core manifest；
- 因果审计；
- 预注册失败样本。

不能直接用于：

- 论文主表；
- 最终模型选择；
- 超参数选择；
- 窗口/单元大小选择；
- 摘要性能数字。

## 15.2 `paper_protocol_v1`

建议立即建立：

1. 从 canonical train manifest 内确定性划分 train/dev；
2. 所有模型使用相同 split seed；
3. epoch、early stop、learning rate schedule、threshold 仅由 dev 决定；
4. test 在规格和 checkpoint 冻结后只评一次；
5. baseline 在同协议下重训；
6. 保存 exact unique-score operating curves；
7. 同时报告 requested 与 achieved FA；
8. 主指标为 official legacy；Hungarian 为机制审计；
9. IoU 使用预声明固定阈值和完整曲线，不挑最优 test threshold；
10. 所有 seed 均报告，禁止 cherry-pick。

## 15.3 主指标

- Pd at `1/5/10/20 FA/Mpix`；
- achieved FA/Mpix；
- IoU；
- nIoU（若原 benchmark 使用）；
- component precision/recall；
- centroid distance；
- false component area；
- atom overlap/contact/merge；
- threshold-induced support change，TRACE 应严格为 0；
- latency、FPS、peak memory、trainable parameters。

---

# 16. 机制诊断

除了最终 Pd/FA，必须证明方法按预期工作。

## 16.1 Evidence recovery

对预注册 `d0-distinct / z-missed` targets，记录：

- native z rank；
- dense control rank；
- TRACE `p_nonempty` rank；
- MAP IoU；
- centroid distance；
- 是否在目标 FA budget 下被选中。

## 16.2 Partition decomposition

对每个 cell 记录：

\[
\log Z_{+},\quad
E(C^*),\quad
E(\hat C),\quad
\log Z_{+}-E(\hat C).
\]

最后一项反映 alternative-shape uncertainty。

## 16.3 Root duplication

统计同一 GT 周围：

- 多少 cell 输出与其 IoU>0.1 的 atom；
- top root 是否为 canonical root cell；
- support-only control 是否更易产生 suffix copy；
- empty-cell NLL 是否抑制邻近错误根。

## 16.4 Atomic threshold stability

对 `τ` 全扫描，单 cell 的 `C_hat` 应保持完全不变；只有 active flag 改变。若实现中 shape 随阈值改变，说明代码错误或偷偷加入了 threshold-conditioned decoder。

---

# 17. 风险清单与止损动作

| 风险 | 早期检测 | 唯一允许动作 | 禁止动作 |
|---|---|---|---|
| GT 非 row-convex | T0-A | 终止 TRACE-R | row-hull 改标签、加拓扑分支 |
| root-cell collision | T0-A | 仅一次 `s=4→2` fallback；仍碰撞则终止 | 多 slot/query/NMS |
| window 截断 | T0-A | 用 train-only 最大 extent 重新冻结一次 | 看 test 后扩大 |
| DP 错误 | T0-B | 修正同一 recurrence/implementation | 近似 grower 替代 |
| numerical overflow | T0-B | FP32/64、稳定 LSE、chunk | clip 到失去概率语义 |
| all-empty collapse | T1 | 检查 prior bias、NLL 实现、采样无偏性 | positive BCE/额外 loss |
| duplicate suffix atoms | T1/T2 | 判 root statistic 假设是否失败 | NMS/duplicate loss |
| overlap/contact merge | T2 | 若超门，停止独立-cell factorization | post-hoc arbitration |
| d0 不足以解码 shape | T1/T2 | 停止 frozen-output 路线 | 解冻 front 或堆 decoder |
| 只提升 IoU、不提升低 FA Pd | T2 | NO-GO | 改门槛或挑预算 |
| 只单 seed 有效 | T3 | NO-GO | best seed 报告 |
| 出现直接 prior art | 文献审计 | 重定贡献或 NO-GO | 换名宣称首创 |
| 延迟过高 | T0-B/T3 | 仅优化同一 semiring kernel/chunk | proposal branch/近似两阶段 |

---

# 18. 最小消融矩阵

论文不应堆十几个模块消融。所有消融围绕一个概率模型的必要组成。

## 18.1 必须消融

1. **Dense scalar control**：同 frozen `d0`、同容量 map；
2. **Point atom**：状态族只允许单像素；
3. **Rectangle atom**：排除任意结构化输出都有效；
4. **No root statistic**：只保留 support energy；
5. **No cardinality correction**：去掉 `−log K_g`；
6. **Independent rows**：去掉 inter-row compatibility；
7. **Full TRACE**。

## 18.2 各消融预期验证的问题

| 变化 | 验证问题 |
|---|---|
| dense → point atom | proper empty/atom normalization 是否已有作用 |
| point → rectangle/run-chain | 完整支持累积是否必要 |
| no-root → root/support | root ownership 是否抑制截断副本 |
| no-`logK` → corrected | shape-space cardinality 是否污染存在概率 |
| independent rows → connected chains | 连通推断是否减少碎片/错误质心 |
| full vs MSHNet | 新预测范式是否优于原 scalarization |

不能把 Mamba、attention、frequency、edge、topology loss 加进消融，因为它们不是 TRACE 定义的一部分。

---

# 19. 预期理论命题

## 命题 1：唯一表示

在 TRACE atom family 内，每个非空二值 mask 可由其逐行最小/最大前景坐标唯一表示为 run chain；canonical root 唯一。

## 命题 2：DP 完备性与无重复

递推中的 start 分支生成且仅生成长度为 1 的合法 chain；transition 分支把每条合法 chain 唯一延长一行。对行数归纳可得所有合法 atom 被枚举一次且仅一次。

## 命题 3：正交累积查询等价

对任意当前 run `[l,r]`，兼容上一状态集合为二维索引域：

\[
\{(a,b):a\le r+1,b\ge l-1,a\le b\}.
\]

在无效状态置 `−∞` 后，先对 `b` 后缀聚合，再对 `a` 前缀聚合，恰好得到该域上的 semiring sum/max。

## 命题 4：存在先验不受状态数量影响

当所有 root energy 为 `b` 且 support energy 为 0 时，cardinality-corrected positive partition 为 `exp(b)`，故 nonempty probability 为 `sigmoid(b)`。

## 命题 5：Atomic renderer 等价

定义每个 atom 内所有像素 dense score 为其 cell existence log-odds的最大值，则对任何阈值，dense 二值图恰好是被选 atom 的并集。

这些命题都应有完整证明或补充材料证明，而不是仅写“显然”。

---

# 20. 代码开发顺序

## 第 1 步：只做 T0-A codec

实现：

```text
extract_8_connected_components
canonical_root
mask_to_run_chain
run_chain_to_mask
is_exact_row_convex_chain
assign_root_cell
measure_root_collision
measure_window_extents
```

不写模型，不占 GPU。

## 第 2 步：CPU 穷举器

对 tiny window 枚举：

- 所有 start roots；
- 所有合法 horizontal intervals；
- 所有相邻兼容 transitions；
- 所有可能 ending rows。

输出 brute-force `K/logZ/MAP/marginals`。

## 第 3 步：向量化 sum/max semiring

先 FP64 通过 tiny exact tests，再上 GPU FP32。

## 第 4 步：接 `MSHNetD0Backbone`

仓库已有精确 headless front 与严格 state loading，可直接复用。[R5]

锁定：

- parameters `requires_grad=False`；
- front 永远 eval；
- BN buffers 不变；
- 固定 anchor 输入的 `d0` hash 不变。

## 第 5 步：T1 microfit

只在 micro manifest 上验证可学习性，不看 test，不做长训练。

## 第 6 步：T2 paired run

同一 baseline checkpoint/front、同一 split/seed/budget，比较 dense control 与 full TRACE。

---

# 21. 7 月 13–21 日执行表

| 日期 | 必须完成 | GO/NO-GO 输出 |
|---|---|---|
| 7/13 | 规格冻结；T0-A geometry audit；tiny brute enumerator | GT 是否可表示、cell/window 是否成立 |
| 7/14 | exact sum-semiring、max-semiring、`logK`；T0-B tests | 数学和数值是否完全正确 |
| 7/15 | 接入 frozen `d0`；renderer；hash/provenance；性能 profiling | 计算预算是否可接受 |
| 7/16 | T1 microfit | 当前 energy/state 是否可学习 |
| 7/17 | T2 单数据集单 seed：MSHNet、dense control、TRACE | 是否出现最低机制增益 |
| 7/18 | 补机制 controls：no-root、point、rectangle、independent rows | 核心组成是否必要 |
| 7/19 | 第二/第三数据集及更多 seeds；延迟与失败分析 | 是否有跨数据集信号 |
| 7/20 | 主图、方法图、表格骨架；冻结摘要数字 | 摘要是否有真实可写结论 |
| 7/21 | 台北内部摘要硬截止；OpenReview 信息复核 | 提交或停止，不临时换模块 |

正文截止前：

- 7/22–7/24：3×3 主矩阵；
- 7/25：理论证明与补充；
- 7/26：related work 和 failure analysis；
- 7/27：全文内部冻结；
- 7/28：正文提交；
- 7/29–7/31：代码、supplement、environment 和 hashes。

---

# 22. 论文结构

## 22.1 Introduction 逻辑

1. IRSTD 训练通常是 dense pixel prediction；
2. 实际低 FA 评价依赖 connected components；
3. 对 MSHNet 的代码级审计显示 evidence 大多存活到 `d0`，首次稳定损失发生在 scalarization；
4. 普通 readout/fusion/threshold 已被因果控制否决；
5. 提出 TRACE：一个可精确归一化的原子组件场；
6. 贡献是 prediction unit、exact inference、base measure 和 atomic threshold—not another feature module。

## 22.2 Method 章节

1. Evidence-preserving frozen front；
2. Root-cell atomic component variables；
3. Joint root/support exponential family；
4. Exact run-semiring partition and MAP；
5. Exact NLL and atomic renderer；
6. Complexity and proofs。

## 22.3 实验章节

1. Protocol and metrics；
2. Main low-FA results；
3. Mechanism controls；
4. Evidence-recovery analysis；
5. Calibration/threshold stability；
6. Complexity；
7. Failure cases。

---

# 23. 图表规划

## Figure 1：证据边界与整体替换

左：MSHNet `input→d0→side scalar heads→fusion`；
中：Gate I 的 distinct survival/drop；
右：冻结 `input→d0`，删除 scalar heads，接一个 atomic field。

## Figure 2：Run-chain 状态与 exact DP

- canonical root；
- 每行 interval；
- 8-connectivity；
- `F_y(l,r)` lattice；
- suffix-then-prefix orthant transform。

## Figure 3：Pixel threshold 与 atomic threshold

同一目标在 dense logits 下随阈值碎裂/消失；TRACE 的 atom support 不变，只改变 active/inactive。

## Table 1：三数据集三 seed 主结果

- IoU/nIoU；
- Pd@1/5/10/20 FA/Mpix；
- achieved FA；
- mean±std。

## Table 2：机制对照

MSHNet、dense control、point、rectangle、no-root、no-logK、independent rows、TRACE。

## Table 3：复杂度

参数、GMAC、latency、FPS、peak memory、DP local size、number of cells。

## Figure 4：Failure analysis

- non-row-convex（若测试出现，但训练门要求无）；
- overlapping atoms；
- root ambiguity；
- weak d0 evidence；
- window boundary cases。

---

# 24. AAAI 摘要草案

## 建议标题

**From Pixels to Atomic Components: Exact Run-Semiring Prediction for Infrared Small Target Detection**

## 英文摘要（实验前占位版）

> Infrared small target detectors are commonly optimized as dense pixel predictors, although deployment performance is assessed through connected target components under stringent false-alarm budgets. A protocol-locked, cross-seed audit of MSHNet localizes the first stable evidence loss to its high-resolution feature-to-scalar prediction boundary, while its input-to-feature front remains strongly discriminative for most persistent misses. We introduce TRACE, a tractable root-cell atomic component exponential family that replaces all scalar side heads and fusion while freezing the evidence-preserving MSHNet front. Each root cell predicts one normalized empty-or-component variable whose nonempty states are unique-root chains of 8-connected horizontal runs. A semiring-generic orthant dynamic program computes the exact partition function, component MAP, and root/support marginals in \(O(HW^2)\) time per local field. This enables a single exact negative log-likelihood and threshold-invariant whole-component decisions, without auxiliary losses, object queries, NMS, or refinement. A cardinality-corrected base measure further decouples component existence from the number of admissible shapes. On **[DATASETS]**, TRACE improves detection probability by **[X]** at **[BUDGETS]** while changing IoU by **[Y]** and adding **[LATENCY]**. These results indicate that replacing pixel scalarization with a normalized atomic prediction unit can convert preserved small-target evidence into more reliable low-false-alarm component decisions.

### 摘要填写纪律

只有 T2/T3 真实结果出来后才能替换：

```text
[DATASETS]
[X]
[BUDGETS]
[Y]
[LATENCY]
```

若只得到单数据集单 seed 信号，摘要不能写“consistently”或“across datasets”。

---

# 25. 审稿人可能的攻击与预答辩

## Q1：这不就是 CRF 吗？

**回答边界**：它属于 conditional energy-based structured prediction 的广义范畴；不应否认 CRF/半马尔可夫传统。区别在于输出状态是 root-cell-owned 2D run-chain atom，使用一个 empty-inclusive local normalization、cardinality-corrected base measure、`O(HW²)` orthant semiring，并直接定义低 FA 的 whole-atom threshold。

## Q2：row-convex 很早就有，创新在哪里？

承认 row-convex tractability 是经典结果。创新不在“发现 row-convex”，而在把其构造成神经条件 atomic component field，并统一 exact likelihood、existence、MAP mask、marginals 和 threshold semantics，且针对代码审计定位的 `d0→scalarization` 边界实施物理替换。

## Q3：为什么不直接用 Mask2Former？

Mask2Former 使用 learned queries、attention decoder、mask/class outputs 与 matching，计算和数据需求更大，也不是对当前 16-channel `d0` 的参数匹配因果替换。TRACE 是固定空间 ownership 与 exact local partition，不需要 query assignment 或 NMS。

## Q4：为什么冻结 backbone？

因为仓库证据显示 `input→d0` 是当前最稳定的 evidence-preserving 部分；解冻会破坏因果定位，并使任何增益无法归因于 prediction-unit change。若 TRACE 通过后，可把 joint fine-tuning 作为未来工作，但不进入 v1.0。

## Q5：独立 cells 会不会重复预测？

会，这是明确风险。联合 root/support likelihood 与 empty cells 旨在抑制错误 roots，但不是全局排斥定理。论文必须报告 overlap/contact/duplicate；若超过预注册门，方法停止，不用 NMS 掩盖。

## Q6：为何不用任意 connected subset？

任意 connected-subset partition 通常不可承受，仓库相关路线也已因可解性/机械门失败。TRACE 有意收窄为可由真实训练几何验证、且 partition/MAP 可精确计算的状态族。

## Q7：两通道是不是 root head 加 mask head？

不是。没有两个概率、两个 loss、两个 decoder 或后融合；它们是同一 energy 的两个 natural parameters，只有完整 atom state 有概率。

---

# 26. 最终冻结规范 v1.0

## 不可改

- canonical root 定义；
- cell size `s=4`，仅一次预声明 `s=2` fallback；
- row-run 8-connected atom family；
- joint root/support energy；
- `−log K_g` base measure；
- one exact NLL；
- frozen `input→d0`；
- no side heads/final fusion；
- no auxiliary losses；
- no NMS/refinement；
- atomic threshold renderer；
- T0/T1/T2/T3 gates。

## 允许的工程修复

- tensor layout；
- chunk size；
- CUDA/vectorization；
- stable log-sum-exp；
- boundary masks；
- bug fixes；
- deterministic dataloader；
- checkpoint/provenance handling。

## 不允许借口“工程优化”引入的结构变化

- learned proposal；
- top-k root prefilter；
- approximate beam search；
- extra feature branch；
- dataset-specific shape head；
- morphology；
- learned duplicate resolver；
- joint fine-tune front。

---

# 27. 当前 GO/NO-GO 状态

\[
\boxed{
\begin{aligned}
&\text{OHR-MSHNet} &&:\ \text{FINAL NO-GO}\\
&\text{signed/unsigned readout} &&:\ \text{FINAL NO-GO}\\
&\text{final-fusion/attention/gate replacement} &&:\ \text{NO-GO}\\
&\text{threshold/solver/calibration route} &&:\ \text{NO-GO}\\
&\text{RCP/forest/CEMC/global connected-set variants} &&:\ \text{NO-GO}\\
&\text{TRACE geometry audit} &&:\ \text{GO}\\
&\text{TRACE exact solver implementation} &&:\ \text{GO}\\
&\text{TRACE microfit} &&:\ \text{GO after T0}\\
&\text{TRACE one-seed paired run} &&:\ \text{GO after T1}\\
&\text{TRACE long multi-dataset training} &&:\ \text{CONDITIONAL after T2}
\end{aligned}}
\]

### 一句话主线

> **保留 MSHNet 已被证据证明有效的 `input→d0`，删除被证据定位为首次稳定损失点的逐像素标量化，将输出改为一个根单元拥有、可精确归一化和精确解码的完整组件随机变量。**

这已经是一个可以立即开始写代码、当天判定几何、次日判定数学、三天内判定可学习性、五天内判定最低性能信号的完整模型方案；不是再等待灵感，也不是继续堆模块。

---

# 参考资料

- **[R1]** Arialliy/DEA current repository and research status.
  <https://github.com/Arialliy/DEA>
- **[R2]** AAAI-27 official author timetable.
  <https://aaai.org/conference/aaai/aaai-27/>
- **[R3]** Current MSHNet implementation in DEA.
  <https://github.com/Arialliy/DEA/blob/main/model/MSHNet.py>
- **[R4]** MSHNet baseline strength, bottleneck, and sequential-freeze audit.
  <https://github.com/Arialliy/DEA/blob/main/MSHNet_Baseline_Strength_Bottleneck_and_Sequential_Freeze_2026-07-13.md>
- **[R5]** Exact headless canonical `input→d0` implementation.
  <https://github.com/Arialliy/DEA/blob/main/model/mshnet_d0_backbone.py>
- **[R6]** DEA component matching and PD/FA metric implementation.
  <https://github.com/Arialliy/DEA/blob/main/utils/metric.py>
- **[R7]** Liu et al., *Infrared Small Target Detection with Scale and Location Sensitivity*, CVPR 2024.
  <https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html>
- **[R8]** Zhang et al., *IRMamba: Pixel Difference Mamba with Layer Restoration for Infrared Small Target Detection*, AAAI 2025.
  <https://ojs.aaai.org/index.php/AAAI/article/view/33085>
- **[R9]** Yang et al., *Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection*, AAAI 2025.
  <https://ojs.aaai.org/index.php/AAAI/article/view/32996>
- **[R10]** Du et al., *DEFANet: Dual-Path Edge-Target Collaboration with Frequency-Aware Enhancement for Infrared Small Target Detection*, AAAI 2026.
  <https://ojs.aaai.org/index.php/AAAI/article/view/37368>
- **[R11]** Yuan et al., *Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective*, CVPR 2026.
  <https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html>
- **[R12]** Yan et al., *Target-Aware Invertible Encoder with Reconstruction Guidance for Infrared Small Target Detection*, CVPR 2026.
  <https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html>
- **[R13]** Cheng et al., *Per-Pixel Classification is Not All You Need for Semantic Segmentation (MaskFormer)*, NeurIPS 2021.
  <https://arxiv.org/abs/2107.06278>
- **[R14]** Cheng et al., *Masked-Attention Mask Transformer for Universal Image Segmentation (Mask2Former)*, CVPR 2022.
  <https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html>
- **[R15]** Asano et al., *Polynomial-time Solutions to Image Segmentation*, SODA 1996.
  <https://dl.acm.org/doi/10.5555/313852.313893>
- **[R16]** Zhang, *Fast Algorithm for Connected Row Convex Constraints*, IJCAI 2007.
  <https://www.ijcai.org/Proceedings/07/Papers/029.pdf>
- **[R17]** Sarawagi and Cohen, *Semi-Markov Conditional Random Fields for Information Extraction*, NeurIPS 2004.
  <https://papers.neurips.cc/paper_files/paper/2004/file/eb06b9db06012a7a4179b8f3cb5384d3-Paper.pdf>
- **[R18]** Schmidt et al., *Cell Detection with Star-Convex Polygons (StarDist)*, MICCAI 2018.
  <https://doi.org/10.1007/978-3-030-00934-2_30>

---

## 文档完整性声明

本文档给出的是一个**已冻结、可直接实现、可在长训练前严格证伪**的研究方案。数学定义、代码接口、实验门槛与停止规则已明确；但在 T2/T3 完成前，任何性能提升、SOTA 或录用概率都仍是未知量。真正避免再次失败的方式不是承诺结果，而是让每个核心前提在最小成本阶段接受可复现检验，并在失败时立即停止，而不是继续增加模块掩盖问题。
