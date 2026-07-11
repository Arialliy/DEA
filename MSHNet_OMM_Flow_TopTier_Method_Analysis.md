# MSHNet → OMM-Flow 顶会方法审计与修订版

> 修订日期：2026-07-11  
> 评审对象：当前仓库中的 canonical MSHNet、SLS/LLoss、Pd/FA 实现，以及原稿提出的 OMM-Flow  
> 当前结论：**原 OMM-Scale 公式 NO-GO；OMM-2D 仅作为可运行的机制探针 GO；整条 OMM 顶会主线尚未 GO。**

---

## 0. 执行结论

原稿的方向——用“一份预测质量只能解释一次”取代 `seg + loc + mass + FA` 的模块化复合 loss——比继续堆损失项更有研究价值。但原稿中最关键的尺度提升

\[
a_{xs}=p_x\operatorname{softmax}_s(e_{xs}),
\qquad
e_{xs}=(W_s*z_s)(x)
\]

并不成立为 MSHNet 的可信尺度归因。原因不是实现细节，而是**不可辨识性**：在最终 logit 和预测概率完全不变时，四个 signed contribution 可以通过正负抵消产生任意尺度责任。

因此，修订后的路线必须拆成三个严格 gate：

| 阶段 | 当前判断 | 允许做什么 | 禁止声称什么 |
|---|---:|---|---|
| OMM-2D identity | GO，机制探针 | 作为 instance-weighted MAE 验证 empty、实例均衡和梯度语义 | 顶会主方法、完整 OT、metric-equivalent |
| 当前 pixel OMM-Spatial | NO-GO | 作为 pixel/point-hit relaxation baseline | component calibration、one-to-one attribution |
| Component-Preserving rescue | 未设计，条件 GO | 先解决不可分割 prediction-unit representation | 只换 solver 就能修复 bridge 反例 |
| OMM-Scale | 原公式 NO-GO | 只有原生联合前景—尺度概率模型通过 gate 后才可恢复 | 从 signed MSHNet branch 唯一恢复 causal attribution |

原工作标题只能保留为 rescue 成功后的候选，当前不能用于投稿：

> **One Mass, One Match: Capacity-Constrained Instance Attribution for Infrared Small Target Detection**

在证明 surrogate 与 Pd/FA 的一致性或上界以前，标题中不应出现 `Metric-Aligned`；在尺度路径通过可辨识性和实证 gate 以前，不应出现 `Scale-Space`。

---

## 1. 代码与协议事实

### 1.1 MSHNet 的最终融合是 signed additive logit，而不是非负质量混合

canonical 完整路径为：

\[
z_f(x)=b+\sum_{s=0}^{3} e_s(x),
\qquad
e_s(x)=(W_s*z_s)(x).
\]

仓库已经有精确、无重复 bias 的分解器：

- [`model/MSHNet.py`](model/MSHNet.py)：四个 side logits 和 `4→1` final convolution；
- [`model/dea_evidence.py`](model/dea_evidence.py)：`ExactScaleContributionDecomposer`；
- [`model/mshnet_evidence_view.py`](model/mshnet_evidence_view.py)：不改变原 forward 的只读证据视图；
- [`tests/test_mshnet_evidence_decomposition.py`](tests/test_mshnet_evidence_decomposition.py)：重建与 autograd 测试。

这些 (e_s) 可以是正、负或相互抵消的。它们是 **logit terms**，不是概率质量。

当前 DEA-lite 的 `z_only` 还把 final bias 加到每个尺度一次，因此四路相加含四份 bias。它适合 counterfactual control，不适合作为守恒分解。

### 1.2 canonical 参数统计仍被 DEA-lite head 污染

当前 `MSHNet.__init__()` 无条件实例化 `decidability_head`，默认训练虽不调用，但会增加 521 个参数：

- 当前类总参数：4,066,034；
- canonical 可执行主干：4,065,513。

这不影响本轮 OMM 原型的数学测试，但正式论文前必须把它做成显式 DEA-only 选项，否则“完全不增加参数”和 checkpoint 语义都不干净。

### 1.3 SLS/LLoss 的真实问题

[`model/loss.py`](model/loss.py) 中：

- `LLoss` 计算的是 `(coord * pred).mean()`，不是除以前景质量的 centroid；
- 位置与总质量仍然耦合；
- 空 target 上 SLS 主项没有有效 foreground-suppression gradient；
- post-warm 的 legacy `LLoss` 才间接提供空图梯度。

mass-normalized centroid 是 paper-faithful correction，可以作为强基线，但不能包装成论文创新。

### 1.4 仓库 Pd/FA 不是像素 FP/FN

[`utils/metric.py`](utils/metric.py) 的实际口径是：

1. 预测和 GT 分别做 8-connectivity connected components；
2. 按 GT 的 `regionprops` 顺序，贪心匹配最近的未匹配预测组件；
3. 质心距离必须严格 `< 3 px`；
4. Pd 是成功匹配 GT 数 / GT 总数；
5. FA 只累计**完全未匹配预测组件**的面积，再除以总图像面积。

所以：

- 一个已匹配组件的 halo 面积不进入 FA；
- 一个像素命中也可能得到 Pd=1；
- 一个 bridge blob 在 flow 中可分给两个 GT，但离散指标中一个组件只能匹配一次；
- legacy 贪心匹配具有 target-order dependence。

任何 pixel transport 都只能称 **metric-informed surrogate**，不能称为该指标的严格连续化。

### 1.5 数据协议会制造空图和目标消失

本地增强审计估计，随机 crop 产生空 target 的比例约为：

| 数据集 | crop-empty 比例 |
|---|---:|
| NUAA-SIRST | 14.5% |
| NUDT-SIRST | 22.4% |
| IRSTD-1K | 17.5% |

此外，统一 resize 到 `256×256` 会擦除少数极小 IRSTD-1K 标注；本地检查至少观察到 4 张，包含 `XDU807` 与 `XDU665`。合法 crop-empty 与错误 resize-erased 必须分开记账。

### 1.6 本项目的真实数据根目录与划分口径

本轮已按用户指定位置实际解析：

```text
/home/ly/DEA/datasets/
├── IRSTD-1K/img_idx/{train_IRSTD-1K,test_IRSTD-1K}.txt
├── NUAA-SIRST/img_idx/{train_NUAA-SIRST,test_NUAA-SIRST}.txt
└── NUDT-SIRST/img_idx/{train_NUDT-SIRST,test_NUDT-SIRST}.txt
```

数据集官方只提供 train/test 两部分，不存在官方 validation split：

| 数据集 | Train | Test | 备注 |
|---|---:|---:|---|
| IRSTD-1K | 800 | 201 | train/test 无重叠 |
| NUAA-SIRST | 213 | 214 | train/test 无重叠 |
| NUDT-SIRST | 663 | 664 | train/test 无重叠 |

三个数据集的 train/test manifest 均无重复项、无交集，且清单中的 image/mask 均存在。

仓库 `IRSTD_Dataset` 当前为了训练期间选 checkpoint，会在官方 train manifest 内部按 `split_seed=20260706, val_fraction=0.2` 派生一个**开发 holdout**：IRSTD 为 `640/160`、NUAA 为 `170/43`、NUDT 为 `530/133`。这只是本项目的实验协议，不得写成数据集划分。对应派生 split 的 SHA-256 前缀为：

| 数据集 | derived fit | derived validation | official test |
|---|---|---|---|
| IRSTD-1K | `8bf05e7f61a2` | `1447a3d88c53` | `8c71e474358a` |
| NUAA-SIRST | `267152a32aae` | `a27f06dfcc17` | `395eecd6bf0e` |
| NUDT-SIRST | `2e5910daa4f2` | `983e9768da47` | `cec44220c69d` |

顶会实验必须在论文中明确选择以下一种协议：

1. 从官方 train 派生固定 validation，只用它选模型与调参，official test 最终只评一次；或
2. 用 train-only 规则预先固定训练轮数/模型选择，再用全部官方 train 重训，official test 最终只评一次。

不能把 official test 当作逐 epoch validation；否则即使 train/test 样本没有重叠，也会发生 test-set selection leakage。

`NUDT-SIRST/img_idx/hcval_NUDT-SIRST.txt` 的 6 个样本全部属于 test manifest，和 train 交集为 0。它只能作为 test 内 hard-case diagnostic，**不能**用于选 checkpoint 或调参，否则会造成 test leakage。正式运行无需手写 `trainval.txt`；当前 loader 会从 `img_idx/train_<dataset>.txt` 与 `img_idx/test_<dataset>.txt` 自动解析。

---

## 2. 原 OMM-Scale 的致命退化

### 2.1 Zero-sum gauge 反例

固定任意最终 logit：

\[
z_f=b+\sum_s e_s.
\]

对任意满足 \(\sum_s h_s=0\) 的场，令：

\[
e'_s=e_s+h_s.
\]

则：

\[
z'_f=z_f,
\qquad
p'=p,
\]

但通常：

\[
\operatorname{softmax}(e')
\ne
\operatorname{softmax}(e).
\]

最简单的两尺度反例：

\[
e=(0,0),
\qquad
e'=(M,-M).
\]

两者最终 logit 都为 0、概率都为 0.5；但当 \(M\to\infty\) 时，后者的尺度责任趋于 one-hot。网络可以用无限增大的正负 branch logit 编码尺度标签，而不改变最终预测。

[`tests/test_omm_flow.py`](tests/test_omm_flow.py) 已把该反例写成防回归测试。

### 2.2 `sum_s p*rho_s=p` 只是归一化恒等式

只要 \(\rho\) 位于 simplex，定义 \(a_s=p\rho_s\) 就必然满足守恒。这个等式不能证明 \(\rho_s\) 是：

- 对最终预测的因果贡献；
- 唯一的尺度解释；
- 与原 final convolution 的 probability contribution 等价。

原稿把“人为分账后相加仍为总账”误写成了“从原模型恢复出真实分账”。

### 2.3 bias、负证据和温度都没有闭环

- final bias 参与 \(p\)，却未被任何尺度真实拥有；
- 抑制性负贡献仍被 softmax 分配正质量；
- softmax 的单位温度依赖 branch logit 的任意数值尺度；
- scale loss 会鼓励正负抵消和 branch logit 爆炸。

因此，**不能直接实现原稿的 OMM-Scale，也不能把 exact signed contribution 指数化后称为 faithful attribution。**

---

## 3. “Metric-Aligned、无语义权重”的表述也必须撤回

### 3.1 Pd 与 FA 本来就是两个目标

Pd 和 FA 没有天然可相加的单位。把两个上界不超过 1 的 surrogate 相加，仍然是一种明确的 fixed scalarization。更准确的名称是 `unit-bounded fixed scalarization`：(R_{\mathrm{miss}}) 的上界为 1，而 (R_{\mathrm{FA}}) 的上界是 batch 中背景像素比例，只有全空 GT 且全图预测为 1 时才等于 1。

修订后统一写：

> We optimize a fixed scalarization of a unit-bounded, instance-normalized miss surrogate and an image-normalized foreground-on-background surrogate.

### 3.2 原稿的 `1/(K_image A_k)` 与数据集 Pd 不一致

若先在每张图内部除以 \(K_b\)，再按 batch 图像平均，那么单目标图中的一个目标比四目标图中的一个目标权重大四倍。数据集 Pd 则对所有 GT 实例全局等权。

用 batch 总实例数代替每图实例数，可以消除**同一个 batch 内**的 per-image weighting error：

\[
r_{bk}^{\mathrm{miss}}=\frac{1}{K_{\mathcal B}A_{bk}},
\qquad
K_{\mathcal B}=\sum_b K_b.
\]

但这仍是 self-normalized batch ratio，不是数据集实例平均的无偏估计。反例：两张图的实例数为 `[1,9]`，唯一 miss 位于单目标图；若 batch size 为 1，两个 batch loss 的期望为 0.5，而数据集 target mean 为 0.1。DDP 单 step all-reduce 只能修复 rank-local 偏差，不能修复跨 batch 的 ratio bias。

因此本轮代码必须把它标为 `batch_global_instance_mean` 研究控制。若要严格估计固定数据集的 target average，需要依据数据集总实例数 (K_D) 和采样概率构造无偏 estimator；随机 crop 又会改变每次增强后的 (K)，此时必须预先定义 ratio-of-expectations 或运行 normalizer，不能声称已经与 dataset Pd denominator 等价。

### 3.3 pixel mass 同时在学 shape，不等价于 Pd

把每个 GT pixel 都设为 target capacity，完整解释一个面积为 \(A_k\) 的目标需要约 \(A_k\) 单位预测概率。它更接近 instance-balanced segmentation，而不是只要求一个 component centroid 命中的 Pd。

这不是坏事，因为论文还要报告 IoU/nIoU；但必须诚实称为**检测—分割联合 surrogate**，不能把收益全部解释成 Pd/FA 对齐。

---

## 4. OMM-2D Identity 的真实身份：Instance-Weighted MAE

### 4.1 精确定义与等价定理

这里的 `S=1` 指 canonical MSHNet 的**最终四尺度融合 logit** \(z_f\)，不是 `warm_flag=False` 时只训练 `output0`。

令 batch 大小为 \(B\)，图像面积为 \(HW\)，batch 中 GT components 总数为 \(K_{\mathcal B}\)。定义：

\[
R_{\mathrm{FA}}
=
\frac{1}{BHW}
\sum_b\sum_{x\notin G_b}p_{bx},
\]

\[
R_{\mathrm{miss}}
=
\frac{1}{K_{\mathcal B}}
\sum_b\sum_{k=1}^{K_b}
\frac{1}{A_{bk}}
\sum_{u\in G_{bk}}(1-p_{bu}),
\]

\[
\mathcal L_{\mathrm{id}}=R_{\mathrm{FA}}+R_{\mathrm{miss}}.
\]

对固定 batch，定义：

\[
w_{bx}(Y)=
\begin{cases}
(BHW)^{-1}, &Y_{bx}=0,\\
(K_{\mathcal B}A_{bk})^{-1}, &x\in G_{bk}.
\end{cases}
\]

则有精确恒等式：

\[
\boxed{
\mathcal L_{\mathrm{id}}(p,Y)
=\sum_{b,x}w_{bx}(Y)|p_{bx}-Y_{bx}|
}.
\]

它也是 partial-flow LP 只保留 `(x,x)` 零成本 identity edge 时的精确最优值：foreground 上最优流为 \(\pi_{xx}=p_x\)，background probability 全部 reject。

这给出了数学闭环，也同时否定了把它包装成新 OT 的可能：**OMM-2D identity 就是 instance-weighted MAE。** 若允许 \(p\in[0,1]\)，零风险当且仅当 \(p=Y\) 逐点成立；若 \(p=\sigma(z)\) 且 logit 有限，则只有 \(\inf\mathcal L=0\)，不存在有限 logit 的零风险解。

### 4.2 能成立的阈值误差上界

仓库使用严格阈值 \(\hat Y_\tau=\mathbf1[p>\tau]\)。定义 instance-balanced pixel error：

\[
E_{\mathrm{pix}}^\tau
=
\frac1{BHW}\sum_{Y=0}\mathbf1[p>\tau]
+
\frac1{K_{\mathcal B}}\sum_k\frac1{A_k}
\sum_{x\in G_k}\mathbf1[p_x\le\tau].
\]

由 \(\mathbf1[p>\tau]\le p/\tau\) 与 \(\mathbf1[p\le\tau]\le(1-p)/(1-\tau)\)，可得：

\[
\boxed{
E_{\mathrm{pix}}^\tau
\le
\frac{R_{\mathrm{FA}}}{\tau}
+\frac{R_{\mathrm{miss}}}{1-\tau}
}.
\]

每实例 `no-hit` rate 也不超过上式的 foreground 项；当 \(\tau=0.5\) 时，\(E_{\mathrm{pix}}^{0.5}\le2\mathcal L_{\mathrm{id}}\)。这是当前能诚实成立的 pixel/hit relaxation bound，**不是 connected-component Pd/FA theorem**。

### 4.3 它不是 probability-proper，且两端梯度都会消失

条件风险具有线性形式：

\[
\mathcal R(p\mid X)=\alpha p+\beta(1-p).
\]

其 Bayes 解只取 0、1 或整段并列解，不恢复 \(P(Y=1\mid X)\)。它 elicited 的是 cost-sensitive hard decision，不是 calibrated probability。

空背景 logit 梯度为：

\[
\frac{p(1-p)}{BHW},
\]

GT foreground 梯度为：

\[
-\frac{p(1-p)}{K_{\mathcal B}A_k}.
\]

所以 persistent false alarm 的 \(p\to1\) 和真正 weak/missed target 的 \(p\to0\) 都会饱和。原稿的 `anti-weak-response` 只有损失值单调性，没有有效梯度保证。

### 4.4 仓库正式 component metric 的两个构造性反例

1. **Pd/FA 完美但 MAE 很大**：5×5 单目标只在 GT centroid 预测一个正像素。仓库 metric 得 `Pd=1, FA=0`，但 \(R_{\mathrm{miss}}=24/25=0.96\)。
2. **MAE 极小但 Pd 很差**：256² 图中两个相隔一格的单像素 GT，再预测中间 bridge 像素。三个正像素形成一个 8-connected component；仓库 greedy metric 只能匹配一个 GT，故 `Pd=0.5, FA=0`，但 \(R_{\mathrm{miss}}=0,R_{\mathrm{FA}}=1/65536\approx1.53\times10^{-5}\)。

第二个反例也原样否定当前 pixel-capacity OMM-Spatial：两个 GT capacity 可由同位置 source 零成本填满，bridge source 只支付 \(1/HW\) reject cost，但预测拓扑仍是一个 component。因此不存在分辨率无关的 component miss \(\le C\mathcal L\) 上界。

### 4.5 必须加入同归一化 logistic 强对照

本轮新增：

\[
R_{\mathrm{bg-log}}
=\frac1{BHW}\sum_{Y=0}\operatorname{softplus}(z),
\]

\[
R_{\mathrm{fg-log}}
=\frac1{K_{\mathcal B}}\sum_k\frac1{A_k}
\sum_{x\in G_k}\operatorname{softplus}(-z).
\]

它使用与 OMM-2D 完全相同的实例/背景权重，但错误高置信 background 的梯度趋于 `+1/(BHW)`，missed foreground 的梯度趋于 `-1/(K A_k)`，不会在错误端饱和。它是 reweighted distribution 上的 proper-composite control，不是论文贡献。

若 OMM-2D 不能超过该 logistic control，其任何收益都应解释为实例重加权或 deep-supervision 改变，而不是 transport/attribution 创新。

---

## 5. 空间 partial flow：当前 pixel-capacity 表示先于求解器 NO-GO

### 5.1 修订后的目标

在 `S=1` 下，source capacity 为 \(a_i=p_x\)，每个 GT pixel 的 target capacity 为 1。令：

\[
r_i^{\mathrm{FA}}=\frac{1}{BHW},
\qquad
r_j^{\mathrm{miss}}=\frac{1}{K_{\mathcal B}A_k}.
\]

使用 3 px 仅作为 surrogate 的空间尺度：

\[
d_{ij}^2=\min\left(\frac{\|x-u\|_2^2}{3^2},1\right),
\]

\[
c_{ij}=(r_i^{\mathrm{FA}}+r_j^{\mathrm{miss}})d_{ij}^2.
\]

部分归属目标为：

\[
\begin{aligned}
\min_{\pi\ge0}\quad
&\sum_{ij}c_{ij}\pi_{ij}
+\sum_i r_i^{\mathrm{FA}}
\left(a_i-\sum_j\pi_{ij}\right)\\
&+\sum_j r_j^{\mathrm{miss}}
\left(b_j-\sum_i\pi_{ij}\right),
\end{aligned}
\]

满足：

\[
\sum_j\pi_{ij}\le a_i,
\qquad
\sum_i\pi_{ij}\le b_j.
\]

该 LP 约束的是可拆分 pixel mass。一个 source 可以分流到多个 target，同一预测 connected component 的不同像素也可服务不同 GT；它是 `non-reusing fractional mass attribution`，不是 one-to-one component assignment。

### 5.2 Bridge 反例已经否定 component calibration

第 4.4 节的三像素 bridge 构造在这个 Spatial LP 中仍只有 \(1/HW\) 风险：两个 GT pixel 被同位置 source 零成本填满，中间 bridge pixel 被 reject。但阈值后仍只有一个 connected component，仓库 metric 只能匹配一个 GT，Pd=0.5。

因此 Gate 2 不能再写成“求解器正确以后证明 component theorem”。在现有 pixel-capacity representation 下，任何分辨率无关的 component-Pd upper bound 已被构造性否定。若论文核心必须对齐正式 Pd/FA，必须先引入真正的 predicted-group/component representation、integer/group capacity，或一个能证明保持 component exclusivity 的可微松弛。

这不等于再堆一个 component module；它意味着研究问题必须从“运输更多像素质量”重构为“什么可微表示能够保留 threshold component 的不可分割身份”。在解决该 representation barrier 前，Spatial LP 只能作为 pixel/point-hit relaxation baseline。

### 5.3 稀疏边结论可以保留，但不能当主要 novelty

展开目标可得 maximum-savings 形式。若：

\[
c_{ij}\ge r_i^{\mathrm{FA}}+r_j^{\mathrm{miss}},
\]

则该边没有正 savings，未正则化 LP 中总存在一个最优解不使用它。因此 3 px 外边可剪除。

这是正确的 reduced-cost/dummy-rejection 结论，但 partial OT、dummy nodes 和 sparse plan 已有成熟理论；它只能作为效率性质，不能独立支撑顶会创新。

当边 savings 恰为 0 时，dense problem 可能存在使用或不使用该边的多个最优 plan。因此只能要求 dense/sparse **optimal value** 与选定 value-subgradient 一致；不能无条件要求 plan 和 gradient 完全相同，除非预先规定 tie-break 或让外边具有严格负 savings。

### 5.4 即使表示被修复，也不能直接写 Sinkhorn

未正则化 LP 的最优值是分段线性的。若求出 \(\pi\) 后 `detach()`，再把它代回 PyTorch objective，梯度会漏掉 capacity 约束对应的 dual contribution，甚至可能把 target 附近预测也一律向下压。

若使用 entropic Sinkhorn：

- 优化的是不同的正则化目标；
- \(\varepsilon\) 不是“纯数值参数”；
- exact sparse-edge proposition 不再原样成立；
- identity reduction 和 zero-saving pruning 都会改变。

若未来先解决 component representation，求解器 gate 仍必须完成：

1. 小图 exact LP reference；
2. primal/dual objective gap；
3. 对 supply 的 dual subgradient；
4. finite difference / gradcheck；
5. dense/sparse optimal value 与正确 value-subgradient 等价；
6. entropic solver 只能作为另一个有明确定义的消融。

---

## 6. 尺度路径：原方案删除，联合概率模型只保留为实验对照

### 6.1 本轮实现的 categorical-odds 对照

为验证“什么样的 source measure 才合法”，原型保留了一个**实验性、非主线**的联合概率融合：

\[
h_s=S e_s+b-\log S,
\]

\[
[P(\mathrm{bg}),a_1,\dots,a_S]
=
\operatorname{softmax}([0,h_1,\dots,h_S]),
\]

\[
p=\sum_s a_s,
\qquad
\rho_s=\frac{a_s}{p}.
\]

这里 \(a_s=P(\mathrm{foreground},\mathrm{scale}=s)\) 是合法的联合概率，不再是对 signed final logit 的后验分账。

但它**不是 baseline-preserving attribution**：

\[
z_{\mathrm{joint}}
=b+\log\frac1S\sum_s\exp(S e_s)
\ge
b+\sum_s e_s=z_{\mathrm{linear}},
\]

且只有所有 \(e_s\) 相等时取等号。例：\(S=4,e=(1,-1,0,0)\) 时，linear contribution sum 为 0，但 joint logit 额外增加约 2.65，可能显著抬高 FA。

因此 [`model/omm_flow.py`](model/omm_flow.py) 将其明确命名为 `experimental_categorical_odds_fusion()`；它没有接入 `main.py`，只能用于研究：

- 合法 joint source measure 是否必要；
- logit inflation 是否导致 FA；
- scale occupancy、calibration 与梯度塌缩；
- 与直接原生 `S+1` 类输出 head 的差别。

### 6.2 真要做尺度，应该原生建模而不是解释旧 signed branches

更干净的版本应直接输出：

\[
[a_0,a_1,\ldots,a_S]
=
\operatorname{softmax}([g_{\mathrm{bg}},g_1,\ldots,g_S]),
\]

其中 \(a_s=P(\mathrm{foreground},\mathrm{scale}=s)\)。这会改变最终输出几何，必须从头训练或做明确蒸馏；不能再声称推理图、最终预测和 baseline 完全不变。

尺度融合也不能单列为创新：逐像素尺度 attention、scale-associated side supervision、LSE/noisy-OR fusion 都有强先例。只有“联合事件测度是 partial attribution theorem 的必要条件”被证明后，尺度维才可能进入主贡献。

### 6.3 当前面积尺度分布并不理想

原稿使用：

\[
\xi_k=\tfrac12\log_2 A_k,
\]

本地 train-mask 统计为：

| 数据集 | 实例数 | 面积中位数 | hard scale `[0,1,2,3]` | `<6 px` 近邻对 / 检查对 |
|---|---:|---:|---:|---:|
| NUAA-SIRST | 270 | 21.5 | `[0, 23, 169, 78]` | `4 / 99` |
| NUDT-SIRST | 918 | 21 | `[1, 52, 539, 326]` | `1 / 284` |
| IRSTD-1K | 1192 | 8 | `[157, 474, 481, 80]` | `20 / 503` |

NUAA/NUDT 几乎没有 scale 0，极易出现 branch starvation；IRSTD-1K 的分布又明显不同。面积到 branch 的映射仍是手工先验，不是由 MSHNet receptive-field semantics 推出的定理。

---

## 7. 近邻工作：当前新颖性风险为高

### 7.1 与 MSHNet 直接相关

- [MSHNet / SLS, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html) 已对每个尺度预测独立施加 SLS；这是必须公平复现的 baseline。
- [TDA loss, arXiv 2025](https://arxiv.org/abs/2506.01349) 已覆盖 target patch、尺度和局部对比度自适应。
- [AC-SLSIoU, arXiv 2026](https://arxiv.org/abs/2607.01555) 已覆盖 logit margin、边界/halo 和 false-alarm penalty；该文目前是预印本，但会直接削弱“给 SLS 加几项 loss”的故事。

### 7.2 OT / matching 的直接威胁

- [Generalized UOT Loss, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Wan_A_Generalized_Loss_Function_for_Crowd_Counting_and_Localization_CVPR_2021_paper.html)：预测 density、GT points、未匹配预测和遗漏 annotation 的联合 UOT；是最接近的强基线。
- [Optimal Correction Cost, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Otani_Optimal_Correction_Cost_for_Object_Detection_Evaluation_CVPR_2022_paper.html)：用 OT、dummy detection/GT 和 FP/FN correction cost 做全局评价。
- [UOT for Object Detection, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/De_Plaen_Unbalanced_Optimal_Transport_A_Unified_Framework_for_Object_Detection_CVPR_2023_paper.html)：统一 prediction-to-GT、GT-to-prediction、Hungarian 和 discard/background。
- [Partial OT, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/1e6e25d952a0d639b676ee20d0519ee2-Abstract.html)：exact partial OT、dummy points 和稀疏计划已有系统理论。

### 7.3 尺度与正质量融合的直接威胁

- [Attention to Scale, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Chen_Attention_to_Scale_CVPR_2016_paper.html)：逐像素尺度 softmax 权重与跨尺度 score fusion。
- [FSDS, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Shen_Object_Skeleton_Extraction_CVPR_2016_paper.html)：GT 尺度量化、scale-associated side outputs 与尺度特定融合。
- [UNO, ICRA 2020](https://arxiv.org/abs/1911.05611)：语义分割中使用 probabilistic noisy-OR 融合多个 expert。

### 7.4 Component-Preserving rescue 同样不是空白方向

- [Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes](https://openreview.net/forum?id=vlaPdKdbGK) 已提出可微 Betti matching，并强调 segmentation persistence barcode 的空间正确匹配。
- [SCNP, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Valverde_Towards_High-Quality_Image_Segmentation_Improving_Topology_Accuracy_by_Penalizing_Neighbor_CVPR_2026_paper.html) 已针对 connected-component/topology error 提出高效邻域 logit penalty，并覆盖 semantic/instance segmentation。
- [Deep Watershed Transform, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Bai_Deep_Watershed_Transform_CVPR_2017_paper.pdf) 已把 watershed-style instance separation 接入深度网络。
- [Topograph](https://openreview.net/forum?id=Q0zmmNNePz) 已用 component graph 构造 topology-preserving segmentation metric/loss。

因此 rescue 也不能简单变成“OMM + topology loss”或“OMM + watershed”，否则再次落入模块堆叠。可守住的研究问题必须更窄：**能否从 IRSTD 的 `<3 px centroid + unmatched component area` 协议推导一种 threshold-component group 表示及其 calibration 条件，而不是再添加一个通用拓扑正则项。**

因此，“positive fusion + scale labels + partial OT + rejection”简单组合起来，仍然会被评为模块拼接。真正需要新增的是：

1. 对 IRSTD 组件协议有意义的 calibration / upper-bound theorem；
2. 证明联合事件测度为何是该 theorem 的必要表示；
3. 显著超过 `MSHNet + Generalized UOT Loss`；
4. 不是只靠 empty gradient 或阈值移动获得收益。

---

## 8. 本轮代码修改及其边界

### 8.1 已新增

[`model/omm_flow.py`](model/omm_flow.py)：

- `label_target_components()`：在几何增强后按 8-connectivity 生成 GT instance map；
- `omm2d_identity_risk()`：canonical final logit 上的 instance-weighted MAE control；
- `instance_balanced_logistic_risk()`：完全相同 reduction 下的 proper-composite 强对照；
- per-instance reduction 改为 differentiable `scatter_add`，不再逐实例 GPU `.item()`；
- `experimental_categorical_odds_fusion()`：仅用于检验合法 joint measure 的实验对照。

[`utils/data.py`](utils/data.py)：

- 仅在非 SLS train mode 下，由 DataLoader worker 在所有几何增强后生成 `[1,H,W]` long instance labels；
- 默认 train、validation 和 test 的二元组接口保持不变；
- connectivity 与仓库 metric 固定为 2。

[`main.py`](main.py)：

- 新增 `--mshnet-objective {sls,omm2d_identity,instance_balanced_logistic}`；
- 新增 `--mshnet-side-supervision {canonical,none}`；
- 新增 `--mshnet-train-graph {canonical_warm,full}`；
- 非默认 objective fail-closed：只能用于 plain MSHNet，必须 `final-only + full-graph`，禁止与 DEA-lite/SLS 叠加；
- checkpoint metadata 固化 objective、graph、side policy、reduction、connectivity、threshold 和 split hashes；
- 非默认实验只保存含 metadata 的 checkpoint，不写来源不可辨识的 raw `weight.pkl`；
- 每 epoch 写 `instance_objective_train.jsonl`，同时区分 optimization batch mean 与跨 epoch numerator/denominator diagnostic。

[`tests/test_omm_flow.py`](tests/test_omm_flow.py)：

- signed-softmax zero-sum gauge 反例；
- 空图损失与解析梯度；
- 大小实例等权；
- batch 全局实例等权；
- instance ID 置换不变；
- 弱响应单调性；
- double-precision gradcheck；
- joint probability 守恒；
- categorical-odds 的 logit inflation 可观察性；
- canonical MSHNet final fusion 到四个 side heads 的端到端反传 smoke。
- vectorized reduction 与逐实例 reference 的值/梯度一致；
- weighted-MAE 饱和与同权重 logistic 错误端梯度；
- worker instance-map support、dtype 与 8-connectivity；
- OMM 非法 CLI 组合、full graph、final-only 路由和 checkpoint 语义 fail-close。

### 8.2 本轮故意保留的边界

- 没有实现已被 bridge 反例否定的 pixel-capacity Spatial solver；
- 没有把实验性 scale fusion 接入训练或包装成正式方法；
- 没有改变 canonical MSHNet inference path；
- 没有使用 baseline checkpoint 微调；本轮训练从随机初始化开始；
- 没有启动 400 epoch 或用 2-epoch smoke 做性能 claim；
- 没有在 official test 上选模型或报告 smoke test 指标。

### 8.3 本轮运行记录

训练环境：`$HOME/BasicIRSTD/infrarenet/bin/python`。

定向回归：

```bash
$HOME/BasicIRSTD/infrarenet/bin/python -m pytest -q \
  tests/test_omm_flow.py tests/test_data_splits.py \
  tests/test_full_dea_main_args.py
```

结果：`39 passed in 13.50s`。

全量回归：

```bash
$HOME/BasicIRSTD/infrarenet/bin/python -m pytest -q
```

结果：`129 passed in 125.88s`。

float32 错误端梯度压力检查：

| 错误类型 | logit | OMM/MAE 梯度 | 同权重 logistic 梯度 |
|---|---:|---:|---:|
| empty/background false alarm | `+20` | `0.0` | `≈+1` |
| foreground missed target | `-20` | `≈-2.06e-9` | `≈-1` |

这直接验证 OMM-2D/weighted-MAE 的两端饱和，而不是只存在 empty-background 问题。

三个真实数据集的单 batch forward/backward 均已通过；exact contribution 重建误差均小于 `9e-8`，final 与四个 side heads 均有有限非零梯度。

### 8.4 IRSTD-1K from-scratch 两轮工程 smoke

本轮按用户要求从随机初始化重训，**没有加载 baseline checkpoint**。配置为：official train 内固定 `640/160` fit/development holdout、seed `20260711`、split seed `20260706`、batch 4、Adagrad lr `1e-3`、full graph、final-only OMM identity、2 epochs。official test 未参与训练或选择。

运行目录：`weight/omm/gate0_irstd_fromscratch_s20260711_20260711_224941/`。

| epoch | optimization loss image mean | global BG risk | global instance miss | empty augmented images | augmented instances | dev IoU | dev Pd | dev FA/Mpix |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.306531` | `0.233089` | `0.071161` | 122 | 740 | `0.0032` | `0.2381` | `90964.22` |
| 1 | `0.131504` | `0.085061` | `0.050677` | 112 | 752 | `0.0052` | `0.1032` | `56188.01` |

工程 gate 通过：无 NaN/OOM，344 个 checkpoint tensors 全有限，split hashes 与配置完整，未生成缺少 metadata 的 raw weights。

但这个 smoke 同时给出一个重要负信号：objective、背景风险与 instance miss 都下降，dev Pd 却从 0.2381 降到 0.1032。两轮训练过短，不能作性能结论；方向上却完全符合 bridge/metric-mismatch 分析，说明继续把 pixel MAE 训练更久不能替代 component-aware representation 设计。

---

## 9. 正确的实验推进顺序

### Gate −1：数据与 baseline 协议

- 区分合法 crop-empty 与 resize-erased；
- 修复或显式排除 resize-erased 样本；
- 去掉 canonical 参数统计中的 DEA head 污染；
- canonical MSHNet 三数据集 × 三 paired seeds；
- mass-normalized SLS 作为 correction baseline；
- SLS + matched empty penalty，隔离 empty-gradient 收益。

### Gate 0：OMM-2D 数值语义

- 129 项全量测试已通过；
- float32 两端 gradient stress 已完成，并确认 MAE 饱和；
- mixed empty/non-empty batch 与 worker instance map 已验证；
- DDP 全局 instance normalization 仍待实现；
- full graph from epoch 0 与 canonical warm schedule 必须做成对控制；
- 所有性能比较从头训练，使用相同 seed、初始化顺序和 data order；
- 禁止把 `warm_flag=False/output0` 当作 OMM 的 `S=1`。

### Gate 1：OMM-2D 机制筛选

至少比较：

1. canonical SLS；
2. SLS final-only + full graph；
3. instance-balanced logistic final-only + full graph；
4. mass-normalized SLS 与 matched empty penalty；
5. OMM-2D/weighted-MAE final-only + full graph；
6. MSHNet + Generalized UOT Loss adaptation。

这组实验必须隔离三种混杂：实例重权、properness/错误端梯度、去掉 side deep supervision。2-epoch smoke 不能进入结果表；完整 epoch、三 paired seeds 和全部从头训练后，OMM-2D 仍只作为诊断 baseline。

### Gate 2：Component-Preserving Representation Rescue

- 先定义不可分割 predicted group/component unit，而不是 pixel mass；
- bridge 构造必须产生两个可独立匹配单元，或显式支付一个 component miss；
- 给出 group relaxation 到 threshold connected components 的条件与失败边界；
- 再讨论 integer/partial assignment、dual-aware gradient 和 sparse solver；
- legacy greedy 与 Hungarian diagnostic 同时报告；
- 与 pixel UOT、topological/component losses 做公平近邻比较。

若无法构造 component-preserving 可微表示，OMM 主线应停止，而不是继续优化 pixel transport solver。

### Gate 3：联合尺度概率

只有 Component-Preserving 主线已经成立后再做：

- 原生 `background + S scale` joint output；
- no-scale vs hard scale vs soft scale；
- calibration、branch occupancy、entropy 和 gradient starvation；
- NUAA/NUDT scale-0 稀缺压力测试；
- 明确报告推理图已改变；
- 必须证明 scale 维提升的不只是训练复杂度。

---

## 10. 最终 GO / NO-GO 门

### OMM-2D 作为诊断方法

**GO。** 它数学闭环、代码可测，适合回答：

- empty gradient 是否是主要瓶颈；
- 全局 LLoss cancellation 是否真实存在；
- instance-balanced miss 是否改善小目标；
- 收益是否只是类别重加权；
- weighted-MAE 与同权重 logistic 的 properness/梯度差异。

### OMM-2D 作为顶会最终方法

**永久 NO-GO 作为独立创新。** 它已被精确化为 conventional instance-weighted MAE；即使完整训练结果很好，也不能改写这一数学身份。

### 当前 pixel-capacity OMM-Spatial

**NO-GO。** Bridge 反例已经构造性否定其 component-Pd calibration；实现更精确的 solver 无法修复 representation mismatch。

### Component-Preserving Attribution Rescue

**条件 GO，尚未设计完成。** 至少必须满足：

1. prediction unit 保留 threshold component 的不可分割身份；
2. bridge、halo、close-pair 上有 component-level guarantee 或明确条件；
3. group assignment 的可微松弛与 solver 正确；
4. 超过同权重 logistic、component/topological loss 与 Generalized UOT；
5. 三数据集 from-scratch paired runs 方向一致；
6. Pd–FA 曲线外移而不是阈值迁移。

### OMM-Scale

**原 signed-softmax 版本永久 NO-GO。** 原生 joint-probability 版本只有在 Component-Preserving 主线成立且尺度维有独立增益后才可重新申请 GO。

---

## 11. 可以写与不能写的 claim

### 当前可以写

- MSHNet 的 signed branch contributions 不支持 post-hoc non-negative scale attribution；
- OMM-2D identity 精确等价于 batch-dependent instance-weighted MAE；
- 它对 instance-balanced pixel/hit relaxation 有阈值上界，但不对 component Pd/FA calibrated；
- empty target 的解析梯度存在，但 foreground/background 错误端都会数值饱和；
- 两轮 from-scratch smoke 只证明工程路径稳定，并观察到 loss 下降而 dev Pd 下降；
- categorical joint output 能构造合法 source measure，但不是原 MSHNet 的 faithful attribution；
- component-preserving representation、尺度增益和 benchmark improvement 尚待验证。

### 当前不能写

- “首次将 OT/UOT 用于 IRSTD，因此具有创新性”；
- “与 Pd/FA 严格对齐”；
- “没有任何语义权重”；
- “从 MSHNet final convolution 唯一恢复尺度贡献”；
- “Sinkhorn ε 只是数值参数”；
- “空图上工程梯度永远非零”；
- “anti-weak response 已解决 missed-target optimization”；
- “当前 fractional pixel flow 是 one-to-one attribution”；
- “batch-global instance mean 等价于 dataset Pd denominator”；
- “不改变推理图”——若最终采用 joint scale output，此句不成立；
- 任何尚未运行得到的性能提升数字。

若救援后的最终方法能成立，论文核心不应是若干组件名，而应是一个可证明的单一陈述：

> Prediction groups preserve component identity under a stated threshold regime, and a capacity-feasible group assignment upper-bounds a precisely defined relaxation of component miss and unmatched-component area.

在这条 representation theorem 和对应实证出现以前，OMM-2D/Spatial 只能作为诊断与负对照，不是已经完成的顶会方法。
