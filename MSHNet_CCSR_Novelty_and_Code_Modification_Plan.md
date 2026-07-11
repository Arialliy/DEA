# MSHNet 基线下的 Component-Cut Structured Risk：历史方案与 NO-GO 审计索引

> **档案状态（2026-07-12）**：本文保留为被否证方案的审计轨迹，不再是可执行
> 的代码修改或训练计划。本文第 5–6、9、16、20 节中的原始
> `max_{F,M}` structured hinge、critical-pixel edit energy 与
> `O(|T|3^K)` DP 尚未通过形式门，不能据此实现或训练。已发现“完美预测仍
> 正损失且零梯度”、critical set 外免费编辑、strict-threshold activation
> 不取到最小值，以及 plateau/多路径无法由单 saddle pixel 切断的反例。
> Gate C1/C2 仍有效；C3–C5 以
> `MSHNet_CCSR_Formal_Audit_and_Corrected_Spec.md` 为准。
> 后续审计还证明 C2 风险是 centroid-gated maximum-weight overlap
> matching，并发现 2012/2015 年已有“extremal/merge tree + one-to-one
> correspondence + structured SVM + exact DP”的直接先例；本文原先的
> 条件性 `7/10–8/10` 新颖性判断已失效。
> 三数据集 × 三 paired seeds 的 internal-validation 账本进一步显示：在
> `p=0.5` 的 79 次 miss 中，71 次为 no-response，只有 4 次属于 one-to-one
> assignment conflict；bridge proxy 仅出现在 1008 次 paired image evaluation
> 中的 5 次。因此本文严格版 CCSR 已在 Gate C1 判为 **NO-GO**，不得进入
> max-tree/DP/训练阶段。后续多尺度审计进一步得到：71 个 final no-response
> 中只有 6 个有任一 raw-side support、3 个能被任一 side 直接匹配、15 个能被
> GT-conditioned contribution-subset oracle 恢复；65 个在四个 raw side 中全部
> 缺失，53 个既无 side support 也不能被任何已测 subset 恢复。因此 fusion/cut
> 不是主导突破口。随后 71 miss + 71 matched-control 的 feature audit 显示：
> D0 对 65/71 miss 仍有无方向局部区分度，但 mask0/final 仅剩 33/35；final
> `AUC+>0.5` 为 48/71，而 71/71 target peaks 均低于固定阈值。该现象是
> readout/operating-point 诊断，不是方法：weakest-side 严格退化为 worst-head
> MIL，best-side 退化为 max-MIL/selector，output-only margin 又与同一 MSHNet
> 上的 AC-SLSIoU 直接重合。因此 path-wise survival 主线也 **NO-GO**。所有后续判断以
> `MSHNet_CCSR_Formal_Audit_and_Corrected_Spec.md` 为准；下方方法、solver、训练、
> 消融与 claim 章节均是 historical proposal，不得按正文继续实现。

> 面向 CVPR / ICCV / AAAI 方法论文的第三轮研究方案  
> 日期：2026-07-11  
> 基线仓库：[Arialliy/DEA](https://github.com/Arialliy/DEA)  
> 暂定方法名：**CCSR — Component-Cut Structured Risk**  
> 论文标题候选：**One Component, One Match: Metric-Calibrated Structured Cut Learning for Infrared Small Target Detection**
>
> 说明：本文以当前公开 `main` 分支可见代码与用户报告的本地未提交状态共同为依据；具体行号可能随本地 `git diff` 漂移，因此代码修改按文件职责和函数接口描述，不假定未提交工作树已完整同步到 GitHub。

---

## 0. 原始设计结论（已否证，仅作档案）

本节记录审计前的设计假设。其“条件 GO”“7/10–8/10”和“精确 tree DP”判断
均已失效，不能作为当前投稿判断。

### 0.1 它是不是“模块堆叠”

**严格按本文方案实现时，不是模块堆叠。**

原因不是“没有新增网络层”这么简单，而是训练语义必须同时满足以下六条：

1. **CCSR 完整替换 SLS、LLoss、四路 side supervision 和 OMM loss**，而不是写成
   \[
   L=L_{\mathrm{SLS}}+\lambda_1L_{\mathrm{topo}}+\lambda_2L_{\mathrm{match}}+\lambda_3L_{\mathrm{FA}}.
   \]
2. 网络仍只输出 MSHNet 原有 final logit；不增加可学习 head、attention、gate 或 refinement branch。
3. max-tree、组件切分、one-to-one assignment 和动态规划是**同一个结构化推断问题的内部变量**，不是四个可独立开关的模块。
4. 训练只回传**一个结构化标量目标**，不设置 `lambda_merge`、`lambda_split`、`lambda_center`、`lambda_shape`、`lambda_fa`。
5. miss、clutter、形状修正不是多个附加 loss，而是一个 component edit risk 中不同匹配状态的固定代价。
6. 推理阶段原则上仍采用 MSHNet final probability、固定阈值和 connected components；不依赖 GT，不增加 learned inference branch。

以下任何实现都会重新退化为模块堆叠，必须判为 **NO-GO**：

```text
SLS + CCSR
BCE/Dice + topology loss + Hungarian loss
CCSR + center loss
CCSR + boundary loss
CCSR + false-alarm focal loss
CCSR + side-head auxiliary supervision
```

### 0.2 当前创新性够不够

需要区分两个版本。

| 版本 | 创新性判断 | 顶会判断 |
|---|---:|---|
| `max-tree + Hungarian + bridge penalty` 的朴素组合 | 4/10–5/10 | 不够，直接近邻过强 |
| 本文收紧后的 `threshold-consistent cut + atomic one-to-one assignment + metric-derived component edit risk + structured upper-bound + exact tree DP` | 条件性 7/10–8/10 | 有机会，必须通过理论和强基线门 |

所以答案不是“现在已经够”，而是：

> **这条路线可以做成非堆叠且具备顶会创新性的统一方法，但创新性来自新的结构化风险、阈值一致可行域、理论界和精确求解器；不是来自 max-tree、连通组件、Hungarian 或拓扑梯度这些单独元素。**

如果最终只做到“找 bridge 像素并加惩罚”，即使数值提升明显，也很可能被评价为已有 topology/component loss 在 IRSTD 上的应用。

---

## 1. 当前研究状态与不可逆结论

当前公开仓库已经包含：

- `model/omm_flow.py` 中的 `omm2d_identity_risk()`；
- 同实例权重的 `instance_balanced_logistic_risk()`；
- DataLoader worker 内生成增强后实例标签；
- `main.py` 中 fail-closed 的 MSHNet objective 路由；
- OMM 数学审计、bridge 反例和 from-scratch smoke 记录；
- 全量回归 `129 passed`。

当前两条结论应永久保留，不再反复“救活”旧路线：

### 1.1 OMM-2D 永久降级为诊断基线

OMM-2D 精确等价于 batch-dependent instance-weighted MAE：

\[
\mathcal L_{\mathrm{OMM2D}}
=
\sum_{b,x}w_{bx}(Y)\left|p_{bx}-Y_{bx}\right|.
\]

它适合验证实例重权、empty gradient 和 properness，但不能包装为新的 OT。

### 1.2 Pixel OMM-Spatial 永久 NO-GO

pixel flow 可拆分同一预测组件的质量。三像素 bridge 构造中，两个 GT 可被不同 source pixel 解释，而阈值输出仍只有一个 connected component，因此 component Pd 只能命中一个目标。

这不是 Sinkhorn 精度、熵正则或稀疏 solver 的问题，而是 prediction unit 错了：

\[
\text{pixel unit}\neq\text{component unit}.
\]

因此第三轮不能再优化 transport solver；必须先把**不可拆分预测组件**写进学习问题。

---

## 2. 为什么朴素 CCSR 仍不够新

以下工作构成直接新颖性压力：

| 近邻 | 已经覆盖的核心能力 | CCSR 不能再声称的内容 |
|---|---|---|
| Component Tree Loss, Perret & Cousty | component-tree node altitude 对像素可微；按属性选择/删除 maxima | 首次把 component tree 用于可训练 loss |
| Betti Matching, ICML 2023 | 预测与 GT 拓扑特征的空间正确匹配 | 首次做空间一致的拓扑匹配 |
| Topograph, ICLR 2025 Spotlight | prediction–GT component graph、局部 critical region、严格拓扑保证 | 首次构造组件图或提供 topology guarantee |
| Supervoxel-Based Loss, AAAI 2025 | 识别导致 split/merge 的 critical false-positive/negative components | 首次惩罚关键 split/merge 组件 |
| SCNP, CVPR 2026 | 用最差同类邻域传播改善组件数量和拓扑准确性 | 首次用邻域最弱响应改善 component count |
| WPRF, 2026-07 | max-min widest path，把梯度送到 connectivity bottleneck | 首次把梯度送到 bridge/saddle |
| Generalized UOT, CVPR 2021 | dense response 到稀疏目标、背景与质量失配 | 首次统一定位、质量和背景拒绝 |
| UOT Detection, CVPR 2023 | GT assignment 与 background rejection 的统一运输框架 | 首次做预测—GT—背景联合匹配 |

因此论文不能写：

- “我们首次利用 component tree”；
- “我们首次对 bridge 建模”；
- “我们首次通过 one-to-one matching 保证实例唯一性”；
- “我们首次将拓扑与分割联合优化”；
- “我们首次把梯度路由到 saddle/bottleneck”；
- “我们首次用一个统一目标处理 miss 与 false alarm”。

### 真正可能成立的新颖性组合

CCSR 必须同时具备以下四点：

1. **Threshold-consistent component cut**  
   不是任意 max-tree antichain，而是能够通过固定阈值下的峰值、鞍点和 branch 临界编辑真实实现的组件前沿。

2. **Atomic one-to-one assignment**  
   一个候选预测组件只能整体匹配一个 GT，不能像 pixel transport 一样分流。

3. **Metric-derived component edit risk**  
   matched、missed、clutter 和形状修正都来自同一个组件编辑问题，使用 Pd/FA 与实例均衡语义确定单位，不设置独立 λ。

4. **Structured upper-bound and exact solver**  
   给出 decoded component risk 的 excess-risk upper bound，并在树结构上给出可验证的精确动态规划，而不是启发式 penalty。

这四点缺一，创新性都会明显下降。

---

## 3. 研究问题的重新定义

MSHNet 输出 final logit：

\[
z\in\mathbb R^{H\times W},\qquad p=\sigma(z).
\]

固定推理阈值为 \(\tau\)，logit 阈值为：

\[
\theta=\operatorname{logit}(\tau).
\]

仓库通常使用 \(\tau=0.5\)，对应 \(\theta=0\)。定义 8-connectivity 阈值前景：

\[
\widehat Y_\theta(z)=\mathbf 1[z>\theta],
\]

其预测连通组件集合为：

\[
\mathcal C_\theta(z)=\{C_1,\ldots,C_J\}.
\]

GT connected components 为：

\[
\mathcal G(Y)=\{G_1,\ldots,G_K\}.
\]

CCSR 的基本公理是：

> **预测的原子单位是阈值 connected component，而不是像素、概率质量或局部 patch。一个预测组件只能被解释一次：匹配一个 GT，或者作为 clutter 被拒绝。**

---

## 4. 单一组件编辑风险

### 4.1 One-to-one 可行匹配

令 \(M\subseteq\mathcal C\times\mathcal G\) 为一对一匹配：

\[
\deg_M(C_j)\le1,\qquad \deg_M(G_k)\le1.
\]

只有满足质心容忍条件的边才允许进入匹配图：

\[
\|\mu(C_j)-\mu(G_k)\|_2<\delta,
\qquad \delta=3\ \text{pixels}.
\]

正式论文应把主定理建立在 order-invariant Hungarian assignment 上；legacy greedy Pd/FA 继续作为 headline benchmark metric 报告，但不能把其 target-order dependence 写进理论。

### 4.2 统一 matched component edit cost

对于匹配对 \((C_j,G_k)\)，定义：

\[
d_{\mathrm{edit}}(C_j,G_k)
=
\frac{|G_k\setminus C_j|}{K|G_k|}
+
\frac{|C_j\setminus G_k|}{HW}.
\]

两项不是两个可调 loss：

- 第一项是该实例未被解释的比例，完整漏掉一个实例恰为 \(1/K\)；
- 第二项是该预测组件超出 GT 的图像归一化面积，与 FA area fraction 使用相同单位。

未匹配 GT 的代价：

\[
d_{\mathrm{miss}}(G_k)=\frac1K.
\]

未匹配预测组件的代价：

\[
d_{\mathrm{clutter}}(C_j)=\frac{|C_j|}{HW}.
\]

### 4.3 Component edit risk

\[
\begin{aligned}
\Delta(\mathcal C,M;Y)
=&
\sum_{(C_j,G_k)\in M}
\left[
\frac{|G_k\setminus C_j|}{K|G_k|}
+
\frac{|C_j\setminus G_k|}{HW}
\right]\\
&+
\sum_{G_k\notin M}\frac1K
+
\sum_{C_j\notin M}\frac{|C_j|}{HW}.
\end{aligned}
\]

并定义：

\[
\Delta(\mathcal C,Y)
=
\min_{M\in\mathcal M_{1:1}}\Delta(\mathcal C,M;Y).
\]

空 GT 时：

\[
K=0\Longrightarrow
\Delta(\mathcal C,Y)=\sum_j\frac{|C_j|}{HW}.
\]

### 4.4 为什么这个风险不会出现 one-pixel 退化

若一个面积为 \(A_k\) 的 GT 只在质心预测一个像素，即使满足 Pd 质心条件，其 matched cost 仍至少为：

\[
\frac{A_k-1}{K A_k}.
\]

所以 CCSR 不需要再附加 IoU/Dice 来阻止 one-pixel hit；形状完整性已经进入同一个 component edit risk。

### 4.5 需要诚实说明的边界

该风险不是官方 Pd/FA 的逐项精确连续替代：官方 Pd 对已匹配组件不检查形状，而论文还要报告 IoU/nIoU。CCSR 的准确表述应是：

> **它在保持 component atomicity 和官方 centroid tolerance 的同时，以固定单位统一 detection miss、unmatched prediction area 与 matched shape correction。**

不能写成“与官方 Pd/FA 完全等价”。

---

## 5. Max-tree 只是可行域，不是创新本身

### 5.1 Component hierarchy

对 final logit \(z\) 构建 8-connectivity max-tree：

\[
T(z).
\]

每个 branch/node 保存：

- peak/birth critical pixel \(r_v^{\mathrm{peak}}\)；
- merge/death saddle pixel \(r_v^{\mathrm{saddle}}\)；
- parent 与 children；
- 该 branch 的像素支持、面积和质心统计；
- 在阈值 \(\theta\) 下是否 active；
- 与 GT 候选边。

### 5.2 关键修正：不能直接优化任意 antichain

max-tree 中的任意 antichain 并不自动等于当前固定阈值下真实存在的预测组件集合。

因此候选状态不能只写成：

\[
F\in\mathcal A(T).
\]

正确状态应为：

\[
q=(F,M,A),
\]

其中：

- \(F\) 是候选 component frontier；
- \(M\) 是 one-to-one assignment；
- \(A\) 是让 \(F\) 在固定阈值 \(\theta\) 下可实现所需的临界编辑动作。

可行性要求：

1. selected component 的 peak 必须高于阈值；
2. 要把 parent merge 拆成多个 child component，相应 saddle 必须低于阈值；
3. 被拒绝的孤立 branch 必须将其 peak 压到阈值以下；
4. 同一路径不能同时选择 ancestor 与 descendant；
5. 重建后的二值组件必须与 \(F\) 一一对应；
6. 当前原始阈值组件集合是一个零编辑可行状态。

因此本文称其为：

> **threshold-consistent repair frontier**，而不是普通 antichain。

---

## 6. 结构化 score 与单一训练目标

### 6.1 最小临界编辑能量

对候选状态 \(q\)，定义其最小临界编辑能量：

\[
E_\theta(q;z)
=
\min_{\widetilde z:\;\mathcal C_\theta(\widetilde z)=F(q)}
\frac{1}{\gamma}
\sum_{r\in\mathcal K(q)}
|\widetilde z_r-z_r|.
\]

其中：

- \(\mathcal K(q)\) 只包含该切分所需的 peak、saddle 和必要 boundary critical pixels；
- \(\gamma\) 是统一的 logit margin 尺度，论文主实验固定一个值，不按数据集调节；
- 当前 raw threshold frontier 的编辑能量为 0；
- 其他状态必须支付真实的 peak 激活、branch 抑制或 saddle 切断代价。

结构化 score：

\[
S_z(q)=-E_\theta(q;z).
\]

### 6.2 Oracle state

在当前候选树内，最优任务状态为：

\[
\Delta^*(z,Y)
=
\min_{q\in\mathcal Q(T(z))}\Delta(q,Y),
\]

\[
\mathcal Q_Y^*(z)
=
\arg\min_{q\in\mathcal Q(T(z))}\Delta(q,Y).
\]

若 max-tree 候选能够表达正确组件切分，则 \(\Delta^*=0\)；否则它表示当前候选层次的不可约 approximation error，必须单独记录。

### 6.3 CCSR latent structured hinge

定义：

\[
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{CCSR}}(z,Y)
=&
\max_{q\in\mathcal Q(T(z))}
\left[
S_z(q)+\Delta(q,Y)-\Delta^*(z,Y)
\right]\\
&-
\max_{q\in\mathcal Q_Y^*(z)}S_z(q).
\end{aligned}
}
\]

该式只有一个结构化目标：

- 第一项是 loss-augmented inference；
- 第二项是最高分 oracle inference；
- 两者共享同一棵树、同一 score、同一 component risk 和同一 solver；
- 不存在 `loc/mass/topo/FA` 四个独立梯度源。

### 6.4 中心理论目标

令：

\[
\widehat q(z)=\arg\max_{q\in\mathcal Q(T(z))}S_z(q).
\]

在固定候选树、确定性 tie-break 和精确求解条件下，应证明：

\[
\boxed{
\Delta(\widehat q(z),Y)-\Delta^*(z,Y)
\le
\mathcal L_{\mathrm{CCSR}}(z,Y).
}
\]

若进一步证明 raw fixed-threshold components 是唯一零编辑 decoder，且 \(\Delta^*=0\)，则得到：

\[
\Delta(\mathcal C_\theta(z),Y)
\le
\mathcal L_{\mathrm{CCSR}}(z,Y).
\]

这才是顶会主方法真正需要的理论核心。

### 6.5 可微性表述

max-tree 组合结构对 logit 排序和 tie 非连续，因此不能声称“全局光滑可微”。正确表述是：

> 在树结构、最优状态和 critical-pixel identity 不变化的区域内，CCSR 对被选择的 peak/saddle logits 是分段线性且可求次梯度；树和 argmax 发生切换时采用确定性次梯度。

实现上：

1. 对树拓扑和离散 solver `stop_gradient`；
2. solver 返回 critical pixel indices 与带符号系数；
3. 用原始 `logits.flatten()[indices]` 重新构造 score；
4. 让 PyTorch 只对这些 logit 值反传；
5. 在无 tie 小图上做 finite-difference 验证。

---

## 7. Bridge、miss、clutter 和 shape 在同一问题中的行为

### 7.1 两个 GT 被一个 bridge 合并

当前 threshold mask 只有一个 parent component。由于 assignment 是一对一：

- 保留 parent：最多匹配一个 GT，另一个支付 \(1/K\) miss；
- 选择两个 child：必须切断对应 merge saddle，结构化 score 对 saddle 产生向下梯度。

同一组件不可能把不同像素分给两个 GT，因此 pixel OMM 的反例被表示层面消除。

### 7.2 孤立虚警组件

该 branch 只能：

- 作为 unmatched component 支付 \(|C|/HW\)；或
- 把 peak 压到阈值以下。

梯度集中在能够消除整个 false component 的 peak/critical support，而不是对全背景平均稀释。

### 7.3 漏检目标

若目标附近存在 subthreshold branch，oracle state 会激活最低编辑代价的 peak；若完全没有候选 branch，则该样本产生不可约 \(\Delta^*>0\)，需要扩大 candidate construction，而不是悄悄产生错误梯度。

### 7.4 一像素命中但形状不完整

虽然 centroid edge 可行，matched component edit cost 仍保留大部分 GT missing-area 风险，因此不会依靠一个中心像素获得零损失。

### 7.5 Halo 与大块粘连

匹配组件超出 GT 的区域进入 \(|C\setminus G|/HW\)，同时若 halo 连接多个目标，atomic assignment 还会产生未匹配 GT 风险或 saddle cut 动作。

---

## 8. 为什么它不是“拓扑 loss + 匹配 loss”

CCSR 中各对象的角色如下：

| 对象 | 角色 | 是否独立 loss |
|---|---|---|
| max-tree | 枚举 fixed-threshold 可修复组件层次 | 否 |
| frontier/cut | 结构化隐变量 | 否 |
| one-to-one matching | frontier 状态的一部分 | 否 |
| miss/clutter/edit cost | 单一 task risk 的状态代价 | 否 |
| peak/saddle margin | 结构化 score | 否 |
| DP solver | 同时做 loss-augmented 与 oracle inference | 否 |

最终训练代码只能看到：

```python
result = ccsr_structured_risk(final_logit, target, instance_labels, config)
loss = result.loss
loss.backward()
```

而不能出现：

```python
loss = (
    sls_loss
    + lambda_topo * topology_loss
    + lambda_match * assignment_loss
    + lambda_fa * false_alarm_loss
)
```

---

## 9. 求解器设计

### 9.1 先做 reference，不先做 CUDA

第一阶段只支持单图、小图、CPU 精确求解：

1. 构建 deterministic max-tree；
2. 显式枚举可行 repair frontiers；
3. 对每个 frontier 枚举/求解 one-to-one matching；
4. 计算 \(E_\theta\)、\(\Delta\) 和 structured hinge；
5. 与二值 mask brute force 或 exhaustive action enumeration 对照。

### 9.2 Tree DP

IRSTD 每图实例数通常较少时，可以采用 bitmask DP。定义：

\[
\mathrm{DP}[v,m,s],
\]

其中：

- \(v\)：component-tree node；
- \(m\in[0,2^K)\)：该子树已占用的 GT 集合；
- \(s\)：当前 node 被抑制、作为一个组件保留、或向 children 分裂的状态。

节点决策：

1. **suppress**：压低该 branch peak；
2. **select-as-clutter**：保留一个 unmatched component；
3. **select-and-match-k**：整个 component 匹配一个 GT；
4. **split-to-children**：支付 saddle cut，并组合 children 的互斥 GT masks。

两个 child mask 合并需要 disjoint constraint：

\[
m_1\cap m_2=\varnothing.
\]

朴素复杂度约为：

\[
O(|T|3^K).
\]

必须先统计三个数据集增强后的 \(K\) 分布；不能直接假设所有图都可行。v0 对超过 `--ccsr-max-instances` 的样本 fail closed，不得静默退化成 greedy。

### 9.3 两次共享 solver

structured hinge 需要：

- loss-augmented inference；
- oracle-constrained inference。

两者应共享同一个 DP engine，只改变 node/assignment state cost，避免实现语义漂移。

### 9.4 Exactness 验证

所有生产优化必须满足：

```text
DP objective == brute-force objective
DP frontier == brute-force frontier
DP matching == brute-force matching（允许声明的 tie 除外）
critical gradient == finite-difference subgradient
```

在这些测试通过前，不应启动长周期训练。

---

## 10. 代码目录设计

不要继续把新方法塞进 `model/omm_flow.py`。OMM 文件应冻结为诊断/负对照。

建议新增：

```text
model/
└── ccsr/
    ├── __init__.py
    ├── types.py
    ├── max_tree.py
    ├── frontier.py
    ├── task_risk.py
    ├── reference_solver.py
    ├── dp_solver.py
    ├── loss.py
    └── diagnostics.py

utils/
├── metric.py
└── component_ledger.py

tests/
├── test_ccsr_max_tree.py
├── test_ccsr_frontier.py
├── test_ccsr_task_risk.py
├── test_ccsr_reference_solver.py
├── test_ccsr_dp_solver.py
├── test_ccsr_gradients.py
├── test_ccsr_main_args.py
└── test_component_ledger.py
```

---

## 11. 逐文件代码修改

## 11.1 `model/ccsr/types.py`

定义不可变数据结构，禁止 solver 随意传递匿名 tuple。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class CriticalPoint:
    pixel_index: int
    level: float
    kind: Literal["peak", "saddle", "boundary"]


@dataclass(frozen=True)
class TreeNode:
    node_id: int
    parent_id: int
    children: tuple[int, ...]
    level: float
    peak: CriticalPoint
    saddle: CriticalPoint | None
    area: int
    centroid_yx: tuple[float, float]
    support_start: int
    support_length: int


@dataclass(frozen=True)
class ComponentTree:
    height: int
    width: int
    root_id: int
    nodes: tuple[TreeNode, ...]
    support_indices: Tensor
    raw_frontier: tuple[int, ...]
    threshold_logit: float


@dataclass(frozen=True)
class MatchEdge:
    node_id: int
    gt_id: int
    centroid_distance: float
    edit_cost: float


@dataclass(frozen=True)
class RepairAction:
    pixel_index: int
    direction: Literal[-1, 1]
    target_logit: float
    normalized_cost: float
    reason: Literal["activate", "suppress", "split", "shape"]


@dataclass(frozen=True)
class StructuredState:
    frontier: tuple[int, ...]
    matches: tuple[tuple[int, int], ...]
    actions: tuple[RepairAction, ...]
    edit_energy: float
    task_risk: float


@dataclass
class CCSRResult:
    loss: Tensor
    loss_augmented_state: StructuredState
    oracle_state: StructuredState
    raw_task_risk: float
    oracle_task_risk: float
    approximation_gap: float
    num_tree_nodes: int
    num_gt: int
    diagnostics: dict[str, float | int | str]
```

实现要求：

- `ComponentTree` 内不保存需要梯度的 logit tensor；
- 只保存 detached topology、索引与统计；
- `loss.py` 根据 action index 从原始 logits 重新 gather 值。

---

## 11.2 `model/ccsr/max_tree.py`

建议接口：

```python
def build_max_tree_reference(
    logits_2d: Tensor,
    *,
    threshold_logit: float = 0.0,
    connectivity: int = 8,
) -> ComponentTree:
    """Build a deterministic CPU reference max-tree.

    The topology is detached. Equal-valued plateaus must be processed as a
    level set, not broken into artificial one-pixel nodes by arbitrary sorting.
    """
```

v0 可使用 `skimage.morphology.max_tree` 或自写 union-find reference，但必须补充：

- 8-connectivity；
- plateau canonicalization；
- deterministic node ordering；
- parent support contains child support；
- raw frontier 与 `measure.label(logits > threshold, connectivity=2)` 完全一致；
- peak 与 saddle critical pixel 可复现；
- NaN/Inf fail closed。

必须增加：

```python
def reconstruct_frontier_mask(
    tree: ComponentTree,
    frontier: tuple[int, ...],
    actions: tuple[RepairAction, ...],
) -> Tensor:
    """Reconstruct the binary mask realized by a repair frontier."""
```

不能直接把 node 的静态 support 当成 fixed-threshold repaired component；必须通过重建验证候选状态的真实组件。

---

## 11.3 `model/ccsr/frontier.py`

负责可行性，不计算 loss。

```python
def validate_threshold_consistent_frontier(
    tree: ComponentTree,
    frontier: tuple[int, ...],
    actions: tuple[RepairAction, ...],
) -> None:
    """Raise on ancestor conflicts or unrealizable fixed-threshold states."""


def enumerate_reference_frontiers(
    tree: ComponentTree,
    *,
    max_nodes: int,
) -> list[tuple[tuple[int, ...], tuple[RepairAction, ...]]]:
    """Exhaustive small-tree reference only."""
```

验证项：

- ancestor/descendant 不能同时 selected；
- split 必须对应至少一个 saddle-down action；
- suppressed branch 必须对应 peak-down action；
- activated branch 必须对应 peak-up action；
- 重建后 component 数和 selected frontier 数一致；
- raw frontier 必须存在且 edit energy 为 0。

---

## 11.4 `model/ccsr/task_risk.py`

实现纯离散 risk，不负责 autograd。

```python
def build_gt_components(
    target: Tensor,
    instance_labels: Tensor,
) -> list[Tensor]:
    ...


def component_pair_cost(
    pred_component: Tensor,
    gt_component: Tensor,
    *,
    num_gt: int,
    num_pixels: int,
    centroid_radius: float,
) -> float:
    ...


def exact_component_edit_risk(
    pred_components: list[Tensor],
    gt_components: list[Tensor],
    *,
    centroid_radius: float = 3.0,
) -> tuple[float, tuple[tuple[int, int], ...]]:
    """Minimum one-to-one risk with miss/clutter dummy states."""
```

建议将 dummy assignment 写成一个统一方阵：

- real prediction → real GT：`component_pair_cost`；
- real prediction → clutter dummy：`area / HW`；
- miss dummy → real GT：`1 / K`；
- illegal centroid edge：`+inf`；
- dummy → dummy：0。

reference 可使用 `scipy.optimize.linear_sum_assignment`，但正式 tree DP 不能依赖逐 frontier 调 Hungarian，否则开销过大。

特殊情况：

- `K == 0`；
- `J == 0`；
- 单像素 GT；
- 相同质心；
- centroid distance 恰为 3，需遵循仓库严格 `< 3` 规则；
- 一个 prediction 同时邻近多个 GT；
- 多个 prediction 邻近同一 GT。

---

## 11.5 `model/ccsr/reference_solver.py`

该文件是论文语义的金标准，速度不重要。

```python
def solve_reference(
    logits_2d: Tensor,
    target_2d: Tensor,
    instance_labels_2d: Tensor,
    *,
    threshold_logit: float,
    centroid_radius: float,
    margin_scale: float,
    loss_augmented: bool,
    oracle_only: bool,
) -> StructuredState:
    ...
```

必须显式计算：

```python
score = -edit_energy
objective = score + task_risk - oracle_task_risk  # loss-augmented
```

oracle 模式限制在最小 task-risk states 中，再选择 score 最大者。

不要在 reference solver 中使用软化、Sinkhorn、straight-through estimator 或 greedy pruning。

---

## 11.6 `model/ccsr/dp_solver.py`

建议先定义通用 semiring-like state cost，再复用两次推断：

```python
@dataclass(frozen=True)
class DPConfig:
    threshold_logit: float = 0.0
    centroid_radius: float = 3.0
    margin_scale: float = 1.0
    max_instances: int = 12


def solve_tree_dp(
    tree: ComponentTree,
    gt_components: list[Tensor],
    *,
    config: DPConfig,
    mode: Literal["loss_augmented", "oracle"],
    oracle_task_risk: float | None = None,
) -> StructuredState:
    ...
```

实现要求：

- 结果与 reference exhaustive solver 等值；
- tie-break 顺序固定，例如：objective、edit energy、task risk、lexicographic frontier；
- 不得因为超出 `max_instances` 自动改用 greedy；
- 内存不足或状态数爆炸时抛出带样本名的异常；
- 日志记录实际状态数、pruned states、solve time。

优化顺序：

1. exact dense bitmask DP；
2. 按 node 的可达 GT edge 做局部 mask 压缩；
3. dominance pruning；
4. CPU 并行；
5. 最后才考虑 C++/CUDA。

---

## 11.7 `model/ccsr/loss.py`

核心公开 API：

```python
def ccsr_structured_risk(
    logits: Tensor,
    target: Tensor,
    instance_labels: Tensor,
    *,
    threshold_logit: float = 0.0,
    centroid_radius: float = 3.0,
    margin_scale: float = 1.0,
    solver: str = "reference",
    max_instances: int = 12,
) -> CCSRResult:
    """Single structured objective; no auxiliary segmentation loss."""
```

batch reduction 建议采用 image mean：

\[
\mathcal L_{\mathcal B}
=
\frac1B\sum_b\mathcal L_{\mathrm{CCSR}}^{(b)}.
\]

原因：结构化风险本身已在图内使用 \(1/K_b\) 和 \(1/HW\) 归一化。不要再对 batch 中所有实例做全局 denominator，否则会改变每图结构化上界的含义。

梯度重建伪代码：

```python
def state_score_from_logits(
    logits_2d: Tensor,
    state: StructuredState,
    margin_scale: float,
) -> Tensor:
    flat = logits_2d.reshape(-1)
    score = flat.sum() * 0.0

    for action in state.actions:
        value = flat[action.pixel_index]
        target = value.new_tensor(action.target_logit)

        if action.direction < 0:      # lower a peak/saddle
            edit = torch.relu(value - target)
        else:                         # raise a peak
            edit = torch.relu(target - value)

        score = score - edit / margin_scale

    return score
```

最终 loss：

```python
loss_aug_score = state_score_from_logits(logits_2d, loss_aug_state, gamma)
oracle_score = state_score_from_logits(logits_2d, oracle_state, gamma)

loss_i = (
    loss_aug_score
    + (loss_aug_state.task_risk - oracle_state.task_risk)
    - oracle_score
)
loss_i = torch.relu(loss_i)  # only for numerical guard; exact solver should be >= 0
```

注意：

- 不能直接把 solver 返回的 Python float `edit_energy` 当 loss；那样没有梯度；
- 不能 detach transport/assignment 后再对另一个 pixel loss 反传；
- 若 loss 因浮点误差小于 0，记录 violation 并只允许极小 tolerance；大量 violation 说明 solver 或 score 重建不一致。

---

## 11.8 `model/ccsr/diagnostics.py`

每个 batch 返回：

```text
raw_task_risk
oracle_task_risk
approximation_gap
loss_augmented_task_risk
num_tree_nodes
num_raw_components
num_selected_components
num_gt
num_peak_up
num_peak_down
num_saddle_down
num_shape_actions
num_solver_states
solve_time_ms
num_ties
```

论文需要证明改善来自 component repair，而不是整体 logit 下移。

---

## 11.9 `main.py`

当前公开代码已有：

```python
MSHNET_OBJECTIVES = (
    'sls',
    'omm2d_identity',
    'instance_balanced_logistic',
)
```

修改为：

```python
MSHNET_OBJECTIVES = (
    'sls',
    'omm2d_identity',
    'instance_balanced_logistic',
    'ccsr',
)
```

新增参数：

```python
parser.add_argument(
    '--ccsr-solver',
    choices=('reference', 'dp'),
    default='reference',
)
parser.add_argument('--ccsr-threshold-logit', type=float, default=0.0)
parser.add_argument('--ccsr-centroid-radius', type=float, default=3.0)
parser.add_argument('--ccsr-margin-scale', type=float, default=1.0)
parser.add_argument('--ccsr-max-instances', type=int, default=12)
parser.add_argument('--ccsr-log-ledger', action='store_true')
```

论文正式配置应固定：

```text
threshold_logit = 0.0
centroid_radius = 3.0
margin_scale = 1.0
connectivity = 8
```

其中前三项不是按数据集调优的 loss 权重。`margin_scale` 只允许做一次全局 sensitivity，不允许对每个数据集搜索。

### Fail-closed 语义

```python
def validate_ccsr_args(args) -> None:
    if args.mshnet_objective != 'ccsr':
        return

    if args.model_type != 'mshnet':
        raise ValueError('CCSR v0 supports plain MSHNet only')
    if args.mshnet_side_supervision != 'none':
        raise ValueError('CCSR must replace side supervision')
    if args.mshnet_train_graph != 'full':
        raise ValueError('CCSR requires the full final-logit graph from epoch 0')
    if args.dea_enabled or args.full_dea_enabled:
        raise ValueError('CCSR cannot be combined with DEA losses/modules')
    if args.multi_gpus:
        raise ValueError('CCSR v0 is single-process until DDP semantics are tested')
    if args.ccsr_centroid_radius != 3.0:
        raise ValueError('paper protocol fixes centroid radius to 3 px')
```

### 训练路由

```python
if objective == 'ccsr':
    outputs = model(image, warm_flag=True)
    final_logit = extract_final_logit(outputs)

    result = ccsr_structured_risk(
        final_logit,
        label,
        instance_labels,
        threshold_logit=args.ccsr_threshold_logit,
        centroid_radius=args.ccsr_centroid_radius,
        margin_scale=args.ccsr_margin_scale,
        solver=args.ccsr_solver,
        max_instances=args.ccsr_max_instances,
    )
    loss = result.loss
else:
    ...
```

必须明确：

```python
# 禁止
loss = ccsr_loss + sls_loss
loss = ccsr_loss + final_bce
loss = ccsr_loss + side_losses
```

### Run manifest

记录：

```json
{
  "objective": "ccsr",
  "objective_version": "ccsr_structured_hinge_v1",
  "network_inference_changed": false,
  "side_supervision": "none",
  "train_graph": "full",
  "threshold_logit": 0.0,
  "centroid_radius": 3.0,
  "connectivity": 8,
  "margin_scale": 1.0,
  "solver": "reference",
  "from_scratch": true,
  "git_commit": "...",
  "dirty_worktree": false
}
```

正式实验不得在 dirty worktree 下运行。

---

## 11.10 `utils/data.py`

当前 worker 端在几何增强后生成 instance labels 的方向是正确的，应保留。

需要新增 sample metadata，区分：

- 原图本身无目标；
- crop 后合法 empty；
- resize 后目标被消失；
- mask interpolation/threshold 导致组件数改变；
- 多个 GT 因 resize 合并。

建议返回：

```python
sample_meta = {
    'sample_name': sample_name,
    'num_instances_before_resize': int(...),
    'num_instances_after_resize': int(...),
    'is_source_empty': bool(...),
    'is_crop_empty': bool(...),
    'num_resize_erased': int(...),
    'num_resize_merged': int(...),
}
```

CCSR 模式下：

```python
return image, mask, instance_labels, sample_meta
```

协议：

- 合法 crop-empty 保留，CCSR 会把所有预测组件视为 clutter；
- resize-erased 样本不得静默当 background；
- 在修复 resize 策略前，至少从主训练 risk 中 fail closed 并记录；
- 数据协议改变后，SLS、logistic、OMM 和 CCSR 全部重新训练。

---

## 11.11 `utils/metric.py`

官方 legacy metric 保持不动，保证可比性。

新增：

```python
def match_components_hungarian(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    centroid_radius: float = 3.0,
    connectivity: int = 2,
) -> ComponentMatchResult:
    ...
```

同时增加 threshold sweep：

```python
def evaluate_component_curve(
    logits_or_probs,
    target,
    thresholds,
) -> list[dict[str, float]]:
    ...
```

正式报告：

- legacy greedy Pd/FA；
- Hungarian Pd/FA；
- Pd@fixed-FA；
- FA@fixed-Pd；
- curve AUC；
- best threshold 与 threshold drift。

主定理针对 Hungarian 风险；legacy 只作为官方 benchmark。

---

## 11.12 `utils/component_ledger.py`

每个 threshold 和 sample 记录：

```text
num_gt
num_pred_components
legacy_matches
hungarian_matches
unmatched_gt
unmatched_pred_components
unmatched_pred_area
no_response_gt
centroid_miss_gt
merged_gt_count
split_prediction_count
multi_gt_per_pred_component
pred_components_per_gt
bridge_candidate_count
mean_bridge_saddle_margin
raw_component_edit_risk
```

推荐定义：

- `merged_gt_count`：同一预测组件 3 px 扩张邻域内包含至少两个 GT centroid；
- `split_prediction_count`：同一 GT 的 3 px 邻域内存在多个预测组件；
- `no_response_gt`：GT 3 px 邻域内无正响应；
- `centroid_miss_gt`：有邻近组件但质心条件失败；
- `bridge_candidate_count`：一个 active parent 的不同 child branches 分别邻近不同 GT。

这能区分：

- 整体 logit 下移；
- false component 抑制；
- merge 减少；
- split 增加；
- centroid drift；
- 阈值迁移。

---

## 12. 测试计划

## 12.1 `tests/test_ccsr_max_tree.py`

必须覆盖：

- 4/8 connectivity；正式只允许 8；
- parent support 包含 child；
- plateau 不产生人工链；
- deterministic node IDs；
- raw frontier 等于 `measure.label(logits > 0, connectivity=2)`；
- peak/saddle index 可复现；
- 空图、全前景、单像素峰、等值双峰；
- NaN/Inf fail closed。

## 12.2 `tests/test_ccsr_frontier.py`

构造：

1. 两个独立 peak；
2. 两个 peak 由一个单像素 bridge 连接；
3. 三个 peak 共用层次 saddle；
4. ancestor/descendant 冲突；
5. false branch suppression；
6. subthreshold missed branch activation。

验证：

- arbitrary invalid antichain 被拒绝；
- split state 必须有 saddle action；
- raw frontier 零编辑；
- 重建组件与 frontier 一一对应。

## 12.3 `tests/test_ccsr_task_risk.py`

必须验证：

- 完美预测风险 0；
- 完整漏掉一个 GT 为 \(1/K\)；
- 空图 false component 为 `area/HW`；
- one-pixel centroid hit 仍有 shape miss；
- bridge component 只能匹配一个 GT；
- 一个 prediction 不可重复匹配；
- GT ID 置换不变；
- prediction component 顺序置换不变；
- centroid distance `2.999...` 可行、`3.0` 不可行。

## 12.4 `tests/test_ccsr_reference_solver.py`

- exhaustive frontier objective；
- oracle risk；
- loss-augmented state；
- non-negative hinge；
- zero-loss implication；
- approximation gap；
- deterministic tie-break。

## 12.5 `tests/test_ccsr_dp_solver.py`

在随机 `4×4`、`6×6`、`8×8` 小图：

```text
DP == exhaustive reference
```

比较：

- objective；
- frontier；
- assignment；
- critical actions；
- task risk；
- edit energy。

至少运行 1000 个随机无 tie case 和专项 tie case。

## 12.6 `tests/test_ccsr_gradients.py`

用 double precision finite difference 检查：

| 构造 | 预期梯度 |
|---|---|
| 两 GT 被 bridge 合并 | merge saddle 向下 |
| 空图孤立 false component | false peak 向下 |
| missed GT 有 subthreshold peak | peak 向上 |
| one-pixel hit、GT 形状缺失 | 必要 shape/boundary critical point 得梯度 |
| 完美 margin state | 梯度为 0 |
| irrelevant far background | 梯度为 0 |

不能只检查 finite；必须检查符号、support 与 finite-difference 数值。

## 12.7 `tests/test_ccsr_main_args.py`

非法组合必须失败：

```text
ccsr + canonical side supervision
ccsr + canonical warm graph
ccsr + DEA
ccsr + Full-DEA
ccsr + SLS auxiliary
ccsr + DDP（v0）
ccsr + unsupported radius
ccsr + dirty protocol metadata missing
```

## 12.8 `tests/test_component_ledger.py`

使用 bridge、split、no-response、centroid-miss 和 empty 图，验证 ledger 分类互斥关系与计数。

---

## 13. 分阶段实施门

## Gate C0：冻结 OMM 否证成果

建议先提交不可变节点：

```text
diagnostic/omm2d-weighted-mae-no-go
```

包含：

- OMM-2D identity；
- logistic 强对照；
- bridge theorem；
- fail-closed CLI；
- worker instance labels；
- 129 tests；
- from-scratch smoke。

不要在同一提交里加入 CCSR 原型。

## Gate C1：组件误差账本

先完成 legacy/Hungarian matching、threshold sweep 和 component ledger。

通过标准：

- 现有 SLS、OMM、logistic checkpoint 均可离线分析；
- 能把 Pd 下降拆成 suppression、merge、centroid drift 和 threshold migration；
- bridge 统计定义稳定。

## Gate C2：离散 task risk

只实现 `exact_component_edit_risk()`。

通过标准：

- 所有合成反例通过；
- one-pixel degeneracy 被定量惩罚；
- bridge 只能匹配一个 GT；
- permutation invariant。

## Gate C3：Max-tree 与 threshold-consistent frontier

通过标准：

- raw frontier 与实际 threshold components 完全一致；
- 任意候选 frontier 都能重建；
- bridge split 必须支付 saddle edit；
- 无 silent invalid antichain。

## Gate C4：Reference structured hinge

通过标准：

- 中心 excess-risk upper-bound 在 exhaustive 小图成立；
- zero-loss implication 成立；
- finite-difference gradient 通过；
- tie 行为明确。

## Gate C5：Exact tree DP

通过标准：

- 1000+ 随机小图与 exhaustive reference 一致；
- 状态复杂度在真实 K 分布上可接受；
- 无 greedy fallback。

## Gate C6：单数据集短训练

只在 IRSTD-1K train/dev，从头训练，对比：

1. canonical SLS；
2. SLS final-only/full-graph；
3. instance-balanced logistic；
4. OMM-2D/weighted MAE；
5. generic Component Tree Loss；
6. Betti/Topograph 类强拓扑对照；
7. CCSR。

每 epoch 做 threshold sweep 和 ledger。

进入长训练的最低条件：

- loss 可稳定下降；
- Pd–FA 曲线外移，而非仅 `τ=0.5` 改变；
- `merged_gt_count` 或 `unmatched_gt` 有机制一致改善；
- IoU 不因 one-pixel/过度切分退化；
- solver approximation gap 受控。

## Gate C7：完整论文实验

- 三数据集 × 三 paired seeds；
- 全部 from scratch；
- paired initialization、split、data order；
- paired bootstrap CI；
- 至少两个额外 backbone；
- legacy 与 Hungarian component metrics；
- 固定 FA-budget 曲线；
- size、contrast、empty、multi-target、close-pair 分层。

---

## 14. 必须纳入的强基线

### IRSTD 基线

- canonical MSHNet + SLS；
- mass-normalized SLS correction；
- SLS + matched empty penalty；
- final-only/full-graph SLS；
- instance-balanced logistic；
- OMM-2D/instance-weighted MAE；
- TDA；
- ICI/blob 类 instance supervision；
- AC-SLSIoU。

### 跨领域近邻

- Generalized UOT Loss；
- UOT Detection adaptation；
- Component Tree Loss；
- Betti Matching；
- Topograph；
- AAAI 2025 Supervoxel-Based Loss；
- SCNP；
- WPRF 或等价 bottleneck-gradient control。

最关键的不是击败所有方法的单点数值，而是证明：

```text
generic component-tree regularization
< component matching only
< threshold-consistent cut only
< full CCSR joint structured risk
```

---

## 15. 正确的消融方式

不能再做一组 λ 消融。应做结构限制：

| 消融 | 删除/限制什么 | 证明什么 |
|---|---|---|
| CCSR w/o atomic assignment | 允许一个组件匹配多个 GT | bridge atomicity 的必要性 |
| CCSR w/ arbitrary antichain | 不验证 threshold consistency | 可实现前沿的必要性 |
| CCSR matching-only | 固定 raw components，不允许 repair cut | component cut 的必要性 |
| CCSR cut-only | 不做 GT one-to-one assignment | assignment 的必要性 |
| CCSR count-risk | matched pair 不含 shape edit | one-pixel 退化控制 |
| CCSR greedy | greedy 状态选择 | exact DP 的价值 |
| CCSR fixed components | 不使用 max-tree候选 | 层次修复的价值 |
| CCSR + SLS | 仅作为负对照 | 证明叠加不必要，而非主方法 |

主表中的完整 CCSR 只能有一个目标和一个正式配置。

---

## 16. 理论清单

投稿前至少需要完成：

### Proposition 1：Atomicity

任何可行状态中，每个 selected component 至多匹配一个 GT，每个 GT 至多匹配一个 component。

### Proposition 2：Raw-frontier consistency

当前 fixed-threshold connected components 对应唯一零编辑状态，除非存在显式定义的 logit tie。

### Proposition 3：Bridge sensitivity

若一个 raw component 覆盖两个只能由不同 child branches 解释的 GT，则任何同时匹配二者的可行状态必须包含至少一个 separating saddle-down action。

### Proposition 4：One-pixel non-degeneracy

单像素 centroid hit 对面积 \(A_k>1\) 的 GT 具有严格正 component edit risk。

### Theorem 1：Structured excess-risk upper bound

\[
\Delta(\widehat q,Y)-\Delta^*
\le
\mathcal L_{\mathrm{CCSR}}.
\]

### Theorem 2：DP exactness

tree DP 返回与完整可行状态枚举相同的 loss-augmented 和 oracle objective。

### Proposition 5：Almost-everywhere critical subgradient

在 tree/order/argmax 唯一的局部区域内，solver 返回的 peak/saddle coefficient 等于结构化 value function 对对应 logit 的次梯度。

如果这些性质无法严格成立，应降低论文定位，不应通过增加多个 loss 项掩盖。

---

## 17. 审稿风险与预设回答

### 17.1 “这只是 Component Tree Loss 的 IRSTD 版本”

必须用以下证据回答：

- Component Tree Loss 只做 node attribute selection，不定义 IRSTD one-to-one component edit risk；
- CCSR 的可行变量是 threshold-consistent repair frontier + assignment；
- 有 structured excess-risk bound；
- generic component-tree baseline 明显弱于完整 CCSR。

### 17.2 “这只是 Topograph/Betti Matching 的组合”

回答重点：

- Betti/Topograph 关注拓扑特征/组件图的一致性；
- CCSR 直接优化阈值 prediction components、centroid feasible edges、instance miss 和 unmatched area；
- shape、miss、clutter 在一个 assignment risk 中；
- exact tree-cut solver 与 task-risk theorem 是核心。

### 17.3 “critical bridge gradient 已被 SCNP/WPRF 做过”

承认该事实，不把 bottleneck gradient 作为 claim。强调：

- gradient location 是 joint solver 的结果；
- 一个组件是否应 split、match、miss 或 reject 由同一 one-to-one structured inference 决定；
- WPRF/SCNP 是必要强对照。

### 17.4 “这是多个代价项相加，仍然是 compound loss”

回答：任何 structured risk 都由状态成本组成。CCSR 与 compound loss 的区别是：

- 所有成本由同一个 matching/edit problem 定义；
- 它们共享一个优化变量和一个 solver；
- 没有独立 λ；
- 删除其中一项会改变任务距离定义，而不是关闭一个网络模块；
- 主方法不与旧 segmentation/topology loss 相加。

### 17.5 “hard tree 不可微”

不能回避。正确回答：

- 离散结构 stop-gradient；
- value function 对选中 critical logits 分段线性；
- 提供 almost-everywhere subgradient 命题；
- 小图 finite difference 和 solver exactness；
- 不声称全局光滑。

### 17.6 “只为 IRSTD 指标过拟合”

需要：

- 至少两个额外 backbone；
- 可选增加一个使用 component matching 的外部小目标/细胞实例数据集；
- 展示 CCSR 的通用输入仅为 dense logit + instance mask + component tolerance；
- 但不要牺牲主线去做大规模模块扩展。

---

## 18. 可以写与不能写的论文 claim

### 可以写，前提是理论与实验实际完成

- We formulate dense IR small-target segmentation as threshold-component structured prediction rather than divisible pixel attribution.
- We introduce a metric-calibrated component edit risk with atomic one-to-one assignment and fixed miss/clutter units.
- We derive a threshold-consistent component-tree cut formulation whose repair actions act on prediction-critical peaks and saddles.
- We provide an exact tree dynamic program for loss-augmented and oracle inference.
- The structured hinge upper-bounds the excess component edit risk under stated fixed-tree and tie conditions.
- The method replaces the original training objective and adds no learned inference branch.

### 不能写

- first component-tree loss；
- first topology-aware IRSTD method；
- first differentiable connected-component matching；
- exact equivalence to official greedy Pd/FA；
- globally differentiable max-tree；
- no hyperparameters；
- no inference change，除非最终实现确实保持原 threshold path；
- SOTA，除非三数据集、多 seed 和完整曲线支持；
- bridge 已解决，除非合成门、ledger 和真实多目标结果同时支持。

---

## 19. GO / NO-GO 判定

| 项目 | 判定 |
|---|---|
| OMM-2D 作为论文主创新 | 永久 NO-GO |
| Pixel OMM-Spatial | 永久 NO-GO |
| 继续优化 Sinkhorn/UOT solver | 停止 |
| `SLS + topology + matching` | NO-GO，模块/损失堆叠 |
| `max-tree + bridge penalty` | NO-GO，创新性不足 |
| `max-tree + Hungarian` | NO-GO，明显组合式 |
| 朴素 CCSR antichain | NO-GO，未保证 fixed-threshold 可实现 |
| 本文严格版 CCSR | **NO-GO**：形式门、先例与机制覆盖率均失败 |
| vertex-cut 扩展 | **NO-GO**：求解代价与真实覆盖率不匹配 |
| C1 ledger、C2/C3a/corrected-C4 reference | 保留为诊断与形式负结果 |
| max-tree、production DP、CCSR from-scratch 训练 | 停止 |
| 只在单数据集单 seed 提升 | 不足 |
| feature-survival audit | PASS as exploratory diagnostic，不是方法证据 |
| weakest-side / best-side path risk | **NO-GO**：退化为 worst-/max-MIL |
| output-only logit margin + FA control | **NO-GO**：AC-SLSIoU/NP/pAUC 直接先例 |
| operating-point MIL | 只保留精确负对照，不接训练主路 |

### 最终证据门

CCSR 与 path-survival 均已终止，不再设置“补齐后转 GO”的实现清单。Feature
audit 的正确结论被限定为：多数 miss 在 late decoder 中仍是局部异常，但现有
readout 没有把它稳定映射到正确方向和 fixed threshold。它不能推出“路径信息
保存”或“新校准理论”。下一立题前必须先复现 AC-SLSIoU、TDA、REEM、
output-only NP/pAUC 和 HED-style learned fusion；只有这些强对照训练后的残余
错误出现新的、跨 seed 稳定的 component mechanism，才重新开始 idea gate。

---

## 20. 当前执行顺序

已完成并冻结为诊断：

1. component ledger、order-invariant Hungarian metric 与 threshold sweep；
2. C2 exact inner matching risk 及其 maximum-weight overlap closure；
3. C3a tiny all-pixel repair reference 与 plateau/多路径反例；
4. corrected exhaustive hinge reference；
5. 三数据集 × 三 seed paired Gate-C1 audit；
6. 四 side 与 14 contribution subsets 的 no-response audit；
7. encoder/pool/skip/decoder/readout 的 geometry-null feature audit；
8. weakest-/best-side path risk 的代数退化与 gauge 反例；
9. output-only operating-point MIL + background order-statistic 精确负对照。

明确停止：deterministic max-tree 产品化、frontier/DP solver、CCSR autograd route、
CCSR smoke/长训练，以及新的 scale selector/gate/distillation 模块。

当前不授权新方法训练。若继续实验，第一项只能是独立复现上述强 loss/readout
baselines，并在新的冻结 development split 上重做 residual ledger；不得继续给
MSHNet 叠加 target-background margin、side calibration 或 FA penalty 后换名投稿。

---

## 21. 参考工作

1. Liu et al., **Infrared Small Target Detection with Scale and Location Sensitivity**, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html

2. Perret and Cousty, **Component Tree Loss Function: Definition and Optimization**, DGMM 2022 / arXiv:2101.08063.  
   https://arxiv.org/abs/2101.08063

3. Stucki et al., **Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes**, ICML 2023.  
   https://proceedings.mlr.press/v202/stucki23a.html

4. Lux et al., **Topograph: An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation**, ICLR 2025 Spotlight.  
   https://openreview.net/forum?id=Q0zmmNNePz

5. Grim et al., **Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function**, AAAI 2025.  
   https://ojs.aaai.org/index.php/AAAI/article/view/32326

6. Valverde et al., **Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels**, CVPR 2026.  
   https://openaccess.thecvf.com/content/CVPR2026/papers/Valverde_Towards_High-Quality_Image_Segmentation_Improving_Topology_Accuracy_by_Penalizing_Neighbor_CVPR_2026_paper.pdf

7. Zong et al., **Widest-Path Reachability Fields for Connectivity-Preserving Slender Structure Segmentation**, arXiv:2607.07123, 2026.  
   https://arxiv.org/abs/2607.07123

8. Wan et al., **A Generalized Loss Function for Crowd Counting and Localization**, CVPR 2021.  
   https://openaccess.thecvf.com/content/CVPR2021/html/Wan_A_Generalized_Loss_Function_for_Crowd_Counting_and_Localization_CVPR_2021_paper.html

9. De Plaen et al., **Unbalanced Optimal Transport: A Unified Framework for Object Detection**, CVPR 2023.  
   https://openaccess.thecvf.com/content/CVPR2023/html/De_Plaen_Unbalanced_Optimal_Transport_A_Unified_Framework_for_Object_Detection_CVPR_2023_paper.html

10. Zeng et al., **Boosting Infrared Small Target Detection via Logit-Domain Contrast and Adaptive Shape Refinement**, arXiv:2607.01555, 2026.  
    https://arxiv.org/abs/2607.01555

11. Arteta et al., **Learning to Detect Cells Using Non-overlapping Extremal Regions**, MICCAI 2012.  
    https://www.robots.ox.ac.uk/~vgg/publications/2012/Arteta12/

12. Funke et al., **Learning to Segment: Training Hierarchical Segmentation under a Topological Loss**, MICCAI 2015.  
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/1960706384/funke_15_learning.pdf

13. Turaga et al., **MALIS: A Maximum Affinity Learning Approach to Segmentation**, structured-threshold extension.  
    https://arxiv.org/abs/1709.02974

14. **Maximum Matching Accuracy**, 2026.  
    https://arxiv.org/abs/2606.10107

15. Xie and Tu, **Holistically-Nested Edge Detection**, ICCV 2015.  
    https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html

16. Zeng et al., **Boosting Infrared Small Target Detection via Logit-Domain Contrast and Adaptive Shape Refinement**, 2026.

    https://arxiv.org/html/2607.01555

17. Shoji et al., **Target Driven Adaptive Loss for Infrared Small Target Detection**, ICIP 2025.

    https://arxiv.org/html/2506.01349

18. Sevim and Töreyin, **SCR-Guided Difficulty-Aware Optimization for Infrared Small Target Detection**, CVPRW 2026.

    https://openaccess.thecvf.com/content/CVPR2026W/PBVS/html/Sevim_SCR-Guided_Difficulty-Aware_Optimization_for_Infrared_Small_Target_Detection_CVPRW_2026_paper.html

19. Lee et al., **Deeply-Supervised Nets**, AISTATS 2015.

    https://proceedings.mlr.press/v38/lee15a.html

20. Rigollet and Tong, **Neyman-Pearson Classification under a Strict Constraint**, COLT 2011.

    https://proceedings.mlr.press/v19/rigollet11a.html

---

## 22. 一句话判断

> **CCSR 与 path-wise survival 顶会主线均最终 NO-GO。Feature audit 发现 D0
> 对 65/71 miss 仍有局部区分度，却不能自动生成新颖方法：weakest-side 是
> worst-head MIL，best-side 是 max-MIL，output-only logit margin/FA control 又被
> 同 baseline 的 AC-SLSIoU 直接覆盖。当前只保留诊断与负对照；下一步应先复现
> 强并发 loss/readout baselines，再从它们的残余错误重新立题。**
