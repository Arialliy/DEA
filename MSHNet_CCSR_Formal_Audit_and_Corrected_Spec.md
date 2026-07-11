# CCSR 形式审计与修正规范

> 日期：2026-07-12  
> 适用文档：`MSHNet_CCSR_Novelty_and_Code_Modification_Plan.md`  
> 当前结论：**C1/C2/C3a/C4 仅保留为诊断与形式负结果；C3b FAIL；
> CCSR 主方法、vertex-cut 扩展、production solver 与训练均 NO-GO。**

## 1. 审计结论

原方案抓住了正确的问题：阈值预测的原子单位应是 connected component，
而不是可任意分流的 pixel mass；主方法也应替换旧目标，而不是叠加多个
loss/module。但是，原 structured hinge 和 edit score 目前存在四个形式错误：

1. `min_M` 的评测 assignment 被错误放入 loss-augmented 外层 `max`；
2. edit energy 只收费 critical set，却允许集合外像素免费变化；
3. strict `z > theta` 的向上激活一般只有 infimum，没有 minimum；
4. 单 peak/saddle pixel 不足以抑制或切分一般 component。

这四点不是实现细节。若不先修复，后续 max-tree、finite difference、DP 和
训练结果都不能支持论文中的 structured-risk claim。

## 2. Fatal counterexample：assignment 方向错误

### 2.1 原文的两个不兼容定义

离散风险先被定义为：

\[
\bar\Delta(F,Y)=\min_{M\in\mathcal M_{1:1}(F,Y)}
\Delta(F,M;Y).
\]

但随后状态写成 `q=(F,M,A)`，并在 loss-augmented inference 中计算：

\[
\max_{F,M,A}\left[S_z(F,A)+\Delta(F,M;Y)-\Delta^*\right].
\]

后者会主动选择风险最大的 matching，而不是评测定义中的最小风险 matching。

### 2.2 最小反例

令一张图只有一个 GT component `G`，预测 frontier 也只有完全相同的
component `C=G`。不需要任何 edit，所以同一 frontier 下：

\[
S_z(F,A)=0.
\]

存在两个 matching state：

- 正确匹配 `M_good={(C,G)}`，风险为 0；
- 空匹配 `M_bad=empty`，同时支付一次 miss 和一次 clutter：

\[
\Delta(F,M_{bad};Y)=1+\frac{|C|}{HW}>0.
\]

由于两者的 frontier 和 action 完全相同，score 与 score gradient 也完全相同。
原 loss-augmented `max_M` 必然选择 `M_bad`，得到正损失；但两个 score 相减后
对 logits 的梯度为 0。因此：

```text
perfect prediction -> positive loss -> zero gradient
```

这同时否定原规范的 zero-loss implication、calibration 叙事和 joint DP 语义。

### 2.3 唯一可接受的修正

Assignment 必须是每个 frontier 的内部评测证书，而不是 adversarial output：

\[
\boxed{
\bar\Delta(F,Y)=
\min_M\Delta(F,M;Y)
}
\]

结构状态只包含可被模型解码的 `(F,A)`。若需要保存 matching，保存
`M^*(F,Y)` 作为 inner argmin certificate，不能让外层自由选择它。

## 3. 修正版 structured hinge

令可行 repair state 为 `u=(F,A)`，定义：

\[
\Delta^*(z,Y)=\min_{u\in\mathcal U(T(z))}\bar\Delta(F(u),Y),
\]

\[
\mathcal U_Y^*(z)=
\arg\min_{u\in\mathcal U(T(z))}\bar\Delta(F(u),Y).
\]

修正版 hinge 为：

\[
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{CCSR}}^{\mathrm{corr}}(z,Y)
=&\max_{u\in\mathcal U(T(z))}
\left[S_z(u)+\bar\Delta(F(u),Y)-\Delta^*(z,Y)\right]\\
&-\max_{u\in\mathcal U_Y^*(z)}S_z(u).
\end{aligned}
}
\]

令 `u_hat=argmax_u S_z(u)`，并令 `u_star` 是 oracle 集中 score 最大者，则：

\[
\begin{aligned}
\mathcal L_{\mathrm{CCSR}}^{\mathrm{corr}}
&\ge S_z(\widehat u)+\bar\Delta(F(\widehat u),Y)-\Delta^*-S_z(u^*)\\
&\ge \bar\Delta(F(\widehat u),Y)-\Delta^*.
\end{aligned}
\]

第二个不等式来自 `u_hat` 最大化 score。这个 pointwise upper bound 成立，但它
是标准 margin-rescaling 推导；论文创新不能只依靠该两行不等式。

## 4. DP 结论必须降级

正确的 loss-augmented inference 是：

\[
\max_F\left[S_z(F)+\min_M\Delta(F,M;Y)\right],
\]

即 max–min，而不是：

\[
\max_{F,M}\left[S_z(F)+\Delta(F,M;Y)\right].
\]

原 `DP[v,m,s]` 若把 `match-k / clutter / miss` 当作与 frontier 同方向的
max 决策，计算的是错误的第二式；它会偏好空 matching。`O(|T|3^K)` 只可能
在以下额外条件下成立：

1. matching 与 frontier 的优化方向一致；
2. 每个 node 的 component support 固定；
3. shape/miss/clutter 与 edit score 对树分支局部可加；
4. 树已二叉化且所有 tie-break 固定。

当前正确风险不满足第 1 条。因此在新证明出现前：

```text
Gate C4: exhaustive frontier + inner Hungarian only
Gate C5: exact tree DP claim suspended
```

可研究但尚未获准的求解路线包括 assignment-LP dual、MILP、Pareto-state DP
或改变 structured surrogate。任何路线都必须逐图与 exhaustive max–min
reference 对齐，不能用 greedy fallback 掩盖方向错误。

## 5. Edit energy 的形式错误

### 5.1 Critical set 外免费编辑

原式为：

\[
E_\theta(u;z)=
\min_{\widetilde z:\mathcal C_\theta(\widetilde z)=F(u)}
\sum_{r\in\mathcal K(u)}|\widetilde z_r-z_r|/\gamma.
\]

约束作用于整幅 `z_tilde`，但目标只对 `K(u)` 收费。因此 solver 可以任意修改
`Omega \ K(u)` 来构造目标组件，代价仍为 0。此时“最小 critical edit”没有被
定义出来。

至少必须二选一：

\[
\widetilde z_{\Omega\setminus A(u)}=z_{\Omega\setminus A(u)},
\]

并只允许显式 action set `A(u)` 改动；或者对所有改变的像素收取全图代价：

\[
\|\widetilde z-z\|_1/\gamma.
\]

### 5.2 Strict threshold 的 minimum 不存在

仓库 decoder 使用 `z > theta`。对一个 `z_r <= theta` 的像素做 activation 时：

\[
\inf_{\widetilde z_r>\theta}|\widetilde z_r-z_r|=\theta-z_r,
\]

但该值在 strict inequality 下不取到。因此原文中的 `min` 一般不存在。

可选修复只有：

- 明确写 `inf`，并证明 value function/subgradient；
- 使用固定正 margin `z_r >= theta + m`；
- 改成闭阈值 decoder，并同步改变整个评测协议。

不能一边保留 strict decoder，一边把 `theta-z_r` 称作已实现的最小 edit。

### 5.3 单 peak 不能抑制一般组件

若 active component 中有多个像素高于阈值，只降低最高 peak 并不会消除其余
active pixels。要让整个 component 消失，pixel-logit edit 至少要使其所有
active support 不再高于阈值，除非方法明确定义了一个“改变 node altitude 会
同步重建整块 support”的不同参数化。后者已非常接近已有 Component Tree Loss，
必须精确区分并引用。

### 5.4 单 saddle 不能切开 plateau 或多路径 bridge

以下两类构造直接反驳“一个 saddle-down action 必然 split”：

1. 两峰之间有宽度大于 1 的等值 plateau；降低一个 canonical saddle pixel
   后仍有相邻 active pixel 连通；
2. 两峰之间存在两条 vertex-disjoint active path；切断任意单点后另一条路径
   仍保持连接。

一般 split action 必须是一个真实 vertex cut（可能包含多个像素），并通过
重建后的 8-connected mask 验证。max-tree merge level 只能指出合并发生在哪个
level，不能自动证明某一个代表 pixel 就是充分 cut set。

### 5.5 Subthreshold peak activation 只产生一像素

把一个局部 peak 提到阈值上方，通常只会产生一个 active pixel，而不会自动
激活静态 max-tree node 的完整 support。若 frontier 把该 node support 当成预测
component，score reconstruction 与实际 fixed-threshold reconstruction 不一致。

## 6. Gate C1/C2 的保留语义

### 6.1 C1：评测 matching 与风险 matching 必须分开

目前保留两种 assignment：

- headline Hungarian metric：先最大化合法匹配数，再最小化质心距离；
- component edit risk：最小化 matched shape + miss + clutter 的完整风险。

二者目的不同，不能复用同一个 matching 后再声称 risk exact。组件账本中的
`raw_component_edit_risk` 必须调用第二种 inner minimization。

### 6.2 C2：离散风险是 fixed-frontier metric，不是训练 loss

当前可接受定义：

\[
\bar\Delta(F,Y)=\min_M\left[
\sum_{(C,G)\in M}
\left(
\frac{|G\setminus C|}{K|G|}+\frac{|C\setminus G|}{HW}
\right)
+\frac{|\mathrm{missed\ GT}|}{K}
+\frac{|\mathrm{unmatched\ pred\ pixels}|}{HW}
\right].
\]

令 `P` 是所有预测组件的并集，`I_jk=|C_j intersect G_k|`。从“全部 unmatched”
状态出发，每加入一条合法匹配边 `(j,k)`，风险恰好减少：

\[
I_{jk}\left(\frac{1}{K|G_k|}+\frac{1}{HW}\right).
\]

因此对 `K>0` 有精确恒等式：

\[
\boxed{
\bar\Delta(F,Y)=
1+\frac{|P|}{HW}
-\max_{M\in\mathcal M_{1:1}}
\sum_{(j,k)\in M}
I_{jk}\left(\frac{1}{K|G_k|}+\frac{1}{HW}\right)
}
\]

并继续施加 strict centroid admissibility。由此可知 inner problem 是标准
maximum-weight bipartite matching。它与 instance-weighted pixel MAE 的差别是：
pixel MAE 会给所有 overlap credit；这里仅给最大权 one-to-one edges credit，
所以 bridge component 不能同时解释两个 GT。

它已经通过 perfect、miss、empty、one-pixel、bridge atomicity、strict radius、
component/GT permutation、matching-credit identity 和 exhaustive partial-assignment
检查。正确称呼应是 **task-specific centroid-gated hybrid matching risk**；不能把
风险本身包装成新的一般 component edit distance，也不能把 Hungarian 当创新。

## 7. 修订后的 Gate

### Gate C1 — 组件账本

- legacy greedy metric 保持原样；
- Hungarian metric 最大 cardinality 后最小 distance；
- threshold 明确区分 logit/probability domain；
- bridge/split/no-response/centroid-miss/empty 合成测试通过；
- 三数据集、三 seed、冻结 checkpoint 的 paired ledger 与多尺度离线
  aggregator 已完成；C1 code-level gate PASS，但真实错误覆盖率使 CCSR
  mechanism gate NO-GO。

### Gate C2 — Fixed-frontier task risk

- assignment 是 inner minimum；
- bridge component 最多匹配一个 GT；
- one-pixel hit 保留严格正 shape risk；
- 置换不变；
- 与 exhaustive partial assignments 对齐；
- 不接 autograd，不接 main training route。

### Gate C3a — Pixel-edit reference（新增）

先不构建 max-tree。在不超过 `4x4` 的小图上全枚举 binary masks；更大图只能
枚举事先声明的 restricted action set，不能声称 `6x6` 的 `2^36` 全枚举：

1. action 外 pixels 固定；
2. 每个候选 action 重建 fixed-threshold mask；
3. 代价使用明确的 infimum 或固定 margin；
4. split 必须通过真实 8-connectivity；
5. plateau 与双路径 bridge 必须覆盖。

### Gate C3b — Tree representation

C3b 必须拆成两个不同性质：

1. **soundness**：每个 tree state 都有同 mask、同 action set、同全 L1 energy
   的 C3a pixel certificate；
2. **subset completeness**：只相对于事先声明的 tree candidate subset 检查
   是否漏状态，不能声称普通 tree cut 覆盖全部 `2^N` pixel masks。

只有 soundness 和所声明的 subset completeness 都通过，才允许声称：

- raw frontier 唯一零编辑；
- suppress/activate/split action 可实现；
- tree support 与重建 component 一一对应。

当前探针已给出 plain max-tree 的反例：`2x3` 双路径 plateau 的 split 需要两个
pixel vertex cut；`2x5` plateau 的最小切分会留下不属于任何原 node support 的
残余；全负平坦图激活中心产生的 singleton 也不存在于原树。因此
`static antichain + single saddle` 在 C3b 判定为 FAIL。

### Gate C4 — Corrected hinge exhaustive reference

- outer enumerate `(F,A)`；
- 每个 frontier 内调用 exact component risk；
- 验证 corrected upper bound；
- 专门测试旧 `max_M` perfect counterexample；
- finite difference 只在 action identity 和 inner assignment 均稳定的邻域检查。

### Gate C5 — Solver

在给出 max–min exactness 证明之前，不实现或宣传 `O(|T|3^K)` DP。候选 solver
必须与 C4 exhaustive reference 在 objective、frontier、inner matching、action
和 score 上全部一致。

### Gate C6 — Training

只有 C3a、C3b、C4、C5 全通过后才允许 from-scratch smoke。禁止用训练 loss
下降反向证明 solver 正确。

## 8. 新颖性压力补充

原文列出的 Component Tree Loss、Betti Matching、Topograph、SCNP、UOT 和
WPRF 均应保留。此外至少补入：

1. **MALIS/structured affinity segmentation**：已有工作直接围绕 threshold 后的
   connected segmentation，把梯度路由到 maximin critical edges；因此
   “structured threshold loss / bottleneck gradient”本身不是新意。  
   https://arxiv.org/abs/1709.02974
2. **Every Component Counts, AAAI 2025**：已有 proximity-based component matching
   与局部 overlap metric，限制“首次 component-aware metric”的 claim。  
   https://ojs.aaai.org/index.php/AAAI/article/view/32408
3. **WPRF, 2026**：已明确使用 Max-Min dynamic programming 将梯度集中到
   connectivity bottleneck；CCSR 不能把 critical bridge gradient 作为主创新。  
   https://arxiv.org/abs/2607.07123
4. **Arteta et al., MICCAI 2012**：extremal-region/MSER tree、non-overlap
   region selection、与 GT dots 的 one-to-one correspondence loss、
   margin-rescaled structured SVM 与 exact tree DP 已形成直接组合先例。  
   https://www.robots.ox.ac.uk/~vgg/publications/2012/Arteta12/
5. **Funke et al., Learning to Segment: Training Hierarchical Segmentation
   under a Topological Loss, MICCAI 2015**：merge-tree candidates、
   non-overlap cut、split/merge/FP/FN topological loss 与 margin-rescaled
   structured SVM 已直接覆盖“hierarchy + structured hinge + exact constrained
   inference”的主骨架。  
   https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/1960706384/funke_15_learning.pdf
6. **Maximum Matching Accuracy, 2026**：globally optimal one-to-one matching、
   continuous overlap 与 global pixel normalization 已直接限制 C2 metric 的
   新颖性。C2 的差异只剩 instance-balanced term 与 IRSTD centroid gate。  
   https://arxiv.org/abs/2606.10107

以下曾是修正后方法成立所需的反事实条件：

> fixed-threshold、atomic component prediction 下，一个可实现的 edit space、
> 内部最小 one-to-one task risk，以及对该 max–min inference 的精确求解。

缺少任何一项，都应降级为 component metric、morphological tree filtering 或
bottleneck loss 的组合应用。当前 C3b 已失败，精确 max–min solver 未成立，
Arteta/Funke/MMA 已构成直接先例，且真实 bridge 覆盖率极低；因此该反事实
组合不再是当前投稿路线。

## 9. 当前执行决定

```text
C1 code-level gate: PASS
C2 hybrid matching risk: PASS (reference/diagnostic; novelty downgraded)
C3a all-pixel repair reference: PASS on declared tiny state space
C3b static max-tree + single saddle: FAIL
corrected pixel-exhaustive C4 reference: PASS
original C4 max_{F,M} hinge: FAIL
original C5 O(|T|3^K) DP: UNPROVEN / PAUSED
training: NOT AUTHORIZED BY FORMAL GATES
```

后续已完成 C3a 小图 pixel-edit reference、三数据集 paired ledger 和
no-response scale audit。它们共同支持停止 max-tree、solver 与 CCSR 训练；
不得再通过增加辅助 loss 绕过形式门或机制覆盖率门。

## 10. 三数据集 paired Gate-C1 结果与最终判定

### 10.1 协议

- 数据集：IRSTD-1K、NUAA-SIRST、NUDT-SIRST；
- seeds：`20260711 / 20260712 / 20260713`；
- checkpoint：已冻结 clean baseline 的 best-IoU checkpoint；
- split：各 run 原始 internal validation，未读取 official test；
- operating point：主表 `p=0.5`，另检查 `p=0.1/0.9`；
- 单位：下表为 paired observations，同一 validation image 在三个 seed 中分别
  计数，不能解释为 unique images。

完整机器可读结果位于：

```text
repro_runs/ccsr_gate_c1/*_component_ledger.json
repro_runs/ccsr_gate_c1/*_scale_ledger.json
repro_runs/ccsr_gate_c1/summary_p050.json
repro_runs/ccsr_gate_c1/summary_p050.md
repro_runs/ccsr_gate_c1/summary_scale_p050.json
repro_runs/ccsr_gate_c1/summary_scale_p050.md
```

### 10.2 `p=0.5` 汇总

| Dataset | Paired images | GT | Pd | Miss | No-response | Centroid miss | Assignment conflict | Bridge images | Split excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | 480 | 756 | 0.9299 | 53 | 48 | 2 | 3 | 3 | 8 |
| NUAA-SIRST | 129 | 162 | 0.9630 | 6 | 3 | 2 | 1 | 2 | 13 |
| NUDT-SIRST | 399 | 561 | 0.9643 | 20 | 20 | 0 | 0 | 0 | 16 |
| **All** | **1008** | **1479** | **0.9466** | **79** | **71** | **4** | **4** | **5** | **37** |

关键比例：

- `71/79 = 89.87%` 的 miss 是 no-response；
- `4/79 = 5.06%` 是具有合法邻近响应但被 one-to-one 竞争留下的 assignment
  conflict；
- bridge proxy 只出现在 `5/1008 = 0.50%` 的 paired image evaluations；
- NUDT-SIRST 三个 seed 没有出现 bridge proxy。

阈值敏感性不能解释该结论：

- `p=0.1`：no-response `68/77 = 88.31%`，assignment conflict `4/77`，
  bridge image rate `0.50%`；
- `p=0.9`：no-response `71/80 = 88.75%`，assignment conflict `4/80`，
  bridge image rate `0.50%`。

### 10.3 最终 GO/NO-GO

CCSR 最具区分度、同时也是代价最高的机制是：把 merged prediction component
作为原子单位，通过 threshold-consistent cut 分裂后再一对一匹配。但 paired
账本显示，该机制只覆盖极少数 baseline errors；绝大多数 miss 在目标附近根本
没有 threshold response。抑制 false component 或激活 missed response 虽仍有
价值，却已分别落入已有 component filtering、structured detection、MALIS/
topology 与 ordinary localization supervision 的强 prior-art 区域，不能依靠
CCSR 名称重新包装。

同时：

1. plain max-tree + single saddle 已被 C3b 反例否证；
2. 改为真实 vertex-cut 会失去简单 tree-DP，并把求解推向更困难的 graph-cut/
   multiway-cut 问题；
3. Arteta 2012 与 Funke 2015 已覆盖 hierarchy + one-to-one/topological loss +
   structured SVM + exact inference 的大部分论文骨架；
4. C2 风险本身只是 maximum-weight overlap matching closure。

因此截至当前证据：

```text
CCSR as top-conference main method: NO-GO
vertex-cut extension for these three IRSTD benchmarks: NO-GO (cost/coverage mismatch)
C1 ledger + C2/C3a/C4 tiny references: retain as diagnostics and formal negative results
max-tree/production DP/from-scratch CCSR training: stop
```

### 10.4 No-response 多尺度审计

对同一组九个冻结 checkpoint，在 `p=0.5` 下提取 MSHNet 四个原始 side
logit、final logit 和 final convolution 的精确逐尺度 contribution。71 个
final no-response paired observations 的结果为：

| Dataset | Final no-response | Any raw-side support | Any side matched | Any contribution subset recovers | Absent from all raw sides |
|---|---:|---:|---:|---:|---:|
| IRSTD-1K | 48 | 5 | 3 | 8 | 43 |
| NUAA-SIRST | 3 | 1 | 0 | 2 | 2 |
| NUDT-SIRST | 20 | 0 | 0 | 5 | 20 |
| **All** | **71** | **6** | **3** | **15** | **65** |

其中 raw-side support 表示任一 side threshold component 在 GT support 的
strict 3-pixel 邻域内出现；side matched 还要求通过同一 centroid-gated
one-to-one matching。逐目标交叉分解为：

```text
raw-side support and subset recovery: 3
raw-side support only:                 3
subset recovery only:                12
neither:                             53
```

因此：

- 只有 `6/71 = 8.45%` 有任一 raw side support，只有 `3/71 = 4.23%`
  能被任一 side 直接匹配；
- `65/71 = 91.55%` 在四个 raw side outputs 中全部缺失；
- 14 个非空、非全集 contribution subsets 的 GT-conditioned oracle 最多恢复
  `15/71 = 21.13%`，仍有 `56/71 = 78.87%` 无法恢复；
- `53/71 = 74.65%` 既无 raw-side support，也不能被任何已测 contribution
  subset 恢复。

14-subset 结果使用 GT 仅在生成后判断“哪个 subset 能恢复该目标”，是诊断
上界，不是可部署 selector，也不是模型性能；side/subset frequency 可在同一
目标上重叠。所有计数均为跨 seed 的 paired observations，不是 unique targets。

这排除了“已有正确 side prediction 普遍被 final fusion 吞掉”作为主解释。
15 个 subset-recoverable case 表明少数目标存在 contribution cancellation，
但覆盖率不足以支持新 scale gate、subset selector 或蒸馏模块；这些路线还直接
受到 dynamic scale fusion、multiple-choice learning 和 online self-distillation
先例约束。主问题转向 side prediction 之前的 representation/optimization。

下一硬门是只读 feature-level audit：用带 geometry-matched null control 的统一
统计，定位 no-response GT 在 encoder/decoder path 上何处出现或丢失
target-versus-local-background 区分度。该门只做方向筛选，不增加网络模块、
不设计复合 loss、不启动新方法训练。

## 11. Feature-survival audit

### 11.1 协议与解释边界

九个冻结 checkpoint 均在原 internal validation 上执行一次原生 MSHNet warm
forward。临时 hooks 只读取以下真实 DAG 张量，不重写 forward，也不以重建结果
替代 direct prediction：

```text
input -> stem -> e0 -> p0 -> e1 -> p1 -> e2 -> p2 -> e3 -> p3 -> m
      -> j3 -> d3 -> j2 -> d2 -> j1 -> d1 -> j0 -> d0
      -> native masks / full sides / exact contributions -> final z
```

对每个 GT component，在每个空间尺度使用 area projection 保留 fractional
occupancy；在排除所有 GT 及 3-pixel guard 后，生成 64 个同形状平移
pseudo-target。主统计是 target 相对其局部背景的逐通道 robust-standardized
contrast norm，再以 64 个 geometry-matched null scores 做 rank calibration：

```text
distinct:        rank >= 0.95
background-like: rank <= 0.50
uncertain:       otherwise
undefined:       geometry/background controls insufficient
```

这是无方向的 GT-conditioned 探索性统计，不是线性可分性证明、可部署分类器、
模型性能或因果归因。对 scalar logits 另报 `AUC+ = P(z_target > z_background)`
和 target peak 到固定阈值 0 的 margin，以区分正确方向、反向响应和仅低于阈值。

审计 cohort 固定为前述 71 个 `p=0.5` final no-response paired observations，
并从每个 checkpoint 的 matched GT 中按面积/边界距离选取 71 个对照。完整结果：

```text
repro_runs/ccsr_gate_c1/*_feature_survival.json
repro_runs/ccsr_gate_c1/summary_feature_survival_p050.json
repro_runs/ccsr_gate_c1/summary_feature_survival_p050.md
```

### 11.2 主要结果

| Dataset | No-response | Input distinct | Middle distinct | D0 distinct | Any native side distinct | Mask0 distinct | Final distinct | Final AUC+ > 0.5 | D0→final drop | Matched D0/final distinct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | 48 | 42 | 25/45 | 44 | 41 | 27 | 29 | 28 | 16 | 47/48 |
| NUAA-SIRST | 3 | 3 | 3/3 | 3 | 3 | 2 | 2 | 3 | 1 | 3/3 |
| NUDT-SIRST | 20 | 13 | 10/20 | 18 | 15 | 4 | 4 | 17 | 14 | 20/20 |
| **All** | **71** | **58** | **38/68** | **65** | **59** | **33** | **35** | **48** | **31** | **70/71** |

补充事实：

- 任一 decoder output distinct：`68/71`；D0 distinct：`65/71 = 91.55%`；
- 任一 native side logit distinct：`59/71 = 83.10%`，任一 native side
  `AUC+ > 0.5`：`62/71`；
- `d0 distinct -> mask0 non-distinct` 有 33 个，反向 recovery 仅 1 个；
- final `z` distinct：`35/71 = 49.30%`，但 `AUC+ > 0.5` 有
  `48/71 = 67.61%`；
- final `z` target peak margin 对 71 个目标全部小于 0；其
  `min / median / max = -63.28 / -29.28 / -0.515`；
- area/border-matched 正对照中，D0 distinct `70/71`、final distinct
  `71/71`。

### 11.3 修正后的机制判断

多尺度 threshold audit 中“65/71 raw sides 全缺失”不能再解释为 latent
information 已经消失。训练-free audit 显示，多数 no-response 目标在 late
decoder feature 中仍是局部异常，部分 scalar side/final logits 也保留正确方向的
相对排序；失败主要表现为 readout 后区分度降低、响应反向，或绝对 logit 远低于
固定阈值。Middle 只有 `38/68` distinct，而 D0 回升到 `65/71`，说明 skip path
存在大量 recovery；不能把单个最早 drop edge 解释成因果瓶颈。

同时必须保留三项限制：

1. raw input 本身已有 `58/71` distinct，故无方向统计可能捕获局部亮度/纹理，
   不能直接等同于 target semantics；
2. `distinct` 不提供可部署方向，只有 scalar `AUC+` 与 fixed-threshold margin
   能说明读出方向和校准；
3. best-IoU checkpoint 由同一 internal validation 选择，本审计又在该 split 上
   定义 cohort 和解释特征，因此只能用于探索性方向筛选。方法一旦确定，必须在
   新冻结 development split 上做确认，official test 仍不能提前读取。

当前允许进入的只剩理论/先例门：检查能否定义一个**单一的 path-wise
fixed-threshold survival/calibration risk**，约束 target-local-background 的正确
方向与背景 FA budget，同时不新增 head、selector 或 inference branch。若它退化
为 per-layer loss 求和、MIL/max pooling、hard-positive weighting、ordinary ranking
或 deep supervision，则继续 NO-GO；在该门通过前仍不启动训练。

## 12. Path-wise survival 的数学与先例门

### 12.1 Weakest-side 与 best-side 都发生精确退化

令第 `s` 个 side 对实例 `k` 的 bag margin 为：

\[
m_{ks}=\max_{u\in A_{ks}}a_s(u)-\tau_s-\gamma_s.
\]

若所谓 path survival 是所有 side 的最弱 margin，则对任意单调 hinge：

\[
[-\min_s m_{ks}]_+=\max_s[-m_{ks}]_+.
\]

它严格等于“每实例 max-pooling MIL + worst-head/OHEM”，只是把 canonical
deep-supervision 的均值改成最大值。若改用 best side：

\[
[ -\max_s m_{ks}]_+=\min_s[-m_{ks}]_+,
\]

则退化为 latent best-expert/max-MIL；再要求 final 模仿 best side，就是已有
self-distillation/learned fusion 的变体。用路径概率乘积也不能避免：取负对数后
重新变成逐阶段损失求和，并额外假设阶段独立。

真实记录也不支持 weakest-side：

```text
any side AUC+ > 0.5:        62 / 71
all four sides AUC+ > 0.5:  27 / 71
all four sides distinct:    17 / 71
any side target peak > 0:    4 / 71
all side target peaks > 0:   0 / 71
```

Middle 的低 distinctness 又可被 skip path 恢复，因此强制所有阶段越阈值会惩罚
大量最终函数不需要的内部状态。

### 12.2 Raw side logits 不是可识别的路径证书

MSHNet final readout 满足：

\[
z=b+\sum_s K_s*a_s.
\]

对任意非零常数 `c_s`，变换：

\[
a'_s=c_sa_s,\qquad K'_s=K_s/c_s
\]

保持 deployed `z` 完全不变；`c_s<0` 甚至会翻转某个 side 的符号。因此跨 raw
side 比较 margin、阈值或“最弱阶段”不是 final function 的可识别属性。Canonical
side loss 可以人为固定 gauge，但新 path term 随即成为另一个辅助 loss。

使用精确 contributions `c_s=K_s*a_s` 可消除该 gauge，却暴露另一问题：它们是
并行加和项，不是依次必须存活的 stages。GT-conditioned 14-subset oracle 只恢复
`15/71`，不能支撑一个覆盖主错误的 contribution path 方法。

### 12.3 唯一干净 reference 仍是已知目标的闭包

删除不可信的内部 path 后，剩下的 output-only 参考式是：

\[
\boxed{
\mathcal L_\infty^{out}=
\max\left\{
\max_k[\tau+\gamma_+-\max_{u\in A_k}z(u)]_+,\;
[z^{\mathcal B}_{[B+1]}-(\tau-\gamma_-)]_+
\right\}.
}
\]

其中 `B` 是允许越阈值的 safe-background pixels 数，`z_[B+1]` 是背景第
`B+1` 大 logit。正 margin `gamma_+>0` 是必要的；推理使用 strict `z>tau`，若
margin 为 0，则 `z=tau` 会得到零 loss 却仍判为背景。空目标图只保留背景项；
`B` 覆盖全部背景时背景项为 0。

该式只是以下已知元素的 `L_inf` 聚合：

- per-instance max-pooling MIL existence；
- instance/head OHEM 或 worst-group scalarization；
- background exact top-k/order statistic；
- fixed type-I-error budget 下的 Neyman–Pearson learning。

精确负对照已实现于：

```text
model/operating_point_mil_reference.py
tests/test_operating_point_mil_reference.py
```

它不接 `main.py`，不作为方法。测试显式展示两个零损失退化：每个大 GT 只保留
一个正 pixel 时 IoU 可低至 `1/|G|`；允许一个 bridge background pixel 时，两个
GT 可连成一个 component，one-to-one Pd 最多为 `1/2`。

### 12.4 直接先例使 output/readout 路线失去主创新空间

最直接的并发工作是 2026-07-02 的 **AC-SLSIoU**：同样以 MSHNet 为
baseline，其 Eq. 6 已用 Softplus 做 target-logit 对 OHNM hard-negative-logit
的 margin ranking，Eq. 10–11 又对超过固定置信阈值的背景使用 FA focal：

https://arxiv.org/html/2607.01555

此外：

- TDA Loss 已做逐目标 local patch、scale/local-contrast adaptive weighting；
  https://arxiv.org/html/2506.01349
- REEM 已在相同 MSHNet 上用 GT local SCR 重权困难目标且不改推理；
  https://openaccess.thecvf.com/content/CVPR2026W/PBVS/papers/Sevim_SCR-Guided_Difficulty-Aware_Optimization_for_Infrared_Small_Target_Detection_CVPRW_2026_paper.pdf
- MSHNet、Deeply-Supervised Nets 与 HED 已覆盖 side supervision 和 learned
  fusion；
- strict Neyman–Pearson、partial-AUC/DRO、OHEM 与 differentiable top-k 已覆盖
  fixed-FA constraint 和 hardest negatives。

因此不能声称“首次实例局部 loss”“首次 logit target-background margin”“首次
低 FA 学习”“首次跨尺度生存”或“首次不增加推理模块改善弱目标”。

### 12.5 最终判定

全局 operating threshold 的现有 `p=0.1...0.9` sweep 中，no-response 始终为
`68...71`；简单阈值变化不能修复主错误。`p=0.8` 虽在 paired aggregate 上相对
`p=0.5` 多 1 个 match 且 unmatched area 更低，但 no-response 仍为 71，说明
它只改变边缘 assignment/FA，不改变主病灶。Contribution-subset oracle 的上界又
只有 `15/71`。

```text
weakest-side/path survival as top-tier method: NO-GO
best-side selector/distillation: NO-GO
output-only target/background margin as main novelty: NO-GO
operating-point MIL reference: retain as negative control only
new path/from-scratch training: NOT AUTHORIZED
```

当前最可信的新发现只是：MSHNet 在 D0 对多数 missed targets 仍保留无方向局部
区分度，但原生 readout 与 fixed operating point 出现系统性坍缩。这是机制诊断，
不是论文方法。下一次方法探索前，必须先把 AC-SLSIoU、TDA、REEM、output-only
NP/pAUC 和 HED-style learned fusion 当强对照；只在它们训练完成后的**残余错误**
呈现新的、稳定的 component mechanism 时，才重新立题。
