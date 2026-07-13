# DEA 当前 `main` 严格审计与 AAAI 模型决策

> 审计日期：2026-07-12
> 审计对象：`https://github.com/Arialliy/DEA` 公开 `main` 分支
> 当前公开 HEAD：`30874cc7ba85540a42d84a6cd2541f665a721bc3` — **Record Gate F transport audit**
> Integrated DEA 引入提交：`3d5436a9280ab897dcce05b37bf14d99a455b118` — **Integrate DEA routing release**

---

## 0. 最终结论

本次更新中最值得审计的新增候选不是 OHR，而是仓库中的：

- `DEAIntegratedMSHNet_release/model/dea_integrated_mshnet.py`
- 论文定位名：**Decidable Evidence Recursion，DER**
- 实现类：`DEAIntegratedMSHNet`

严格结论如下。

| 审计层级 | 结论 | 说明 |
|---|---:|---|
| 代码机械完整性 | **PASS** | checkpoint 可加载、初始输出严格保持、梯度可到达、测试通过 |
| baseline exact embedding | **PASS** | 全部 route 为 uncertain 时，forward 等于 MSHNet |
| 三动作语义可辨识性 | **FAIL** | 无显式语义时 target/increase 占用为 0；残差教师控制变成 100% keep |
| “recursion” 数学主张 | **FAIL / 必须降级命名** | terminal closure 是可交换加法，顺序不影响输出，不构成状态依赖递归 |
| 与组件级 Pd–FA North Star 对齐 | **FAIL** | 当前监督仍是像素级 BCE 残差，不解释低 FA 组件错误 |
| 与当前部署约束兼容 | **FAIL** | 新增 20,988 路由参数与约 0.193069 GMAC，改变推理图和成本 |
| 作为 AAAI 主模型 | **NO-GO** | 目前没有泄漏-free、paired、multi-seed 的方法证据 |
| 启动长训练 | **NO-GO** | 必须先通过冻结特征的 route-identifiability gate |
| 保留方式 | **隔离对照** | 可作为 attention / residual-routing / identity-initialization control |

因此，当前状态应冻结为：

\[
\boxed{
\text{DER mechanics PASS, method hypothesis FAIL, AAAI main-model NO-GO}
}
\]

这不是说代码“不能运行”，而是说它现在还不是一个可以承担顶会核心贡献的科学机制。

---

## 1. 本次公开仓库更新的实际含义

### 1.1 当前公开 `main`

GitHub 当前公开提交历史显示：

- `main` 有 39 个提交；
- 当前 HEAD 为 `30874cc`；
- 最新提交只修改一份 Gate F 文档，变更约 `+21/-6`；
- `DEAIntegratedMSHNet_release` 由更早的 `3d5436a` 提交引入，包含模型、loss、测试、验证报告和论文定位文档。

上一轮审计涉及的 `af4a920...` 提交仍可直接访问，但当前公开 `main` 历史页面显示的 HEAD 是 `30874cc...`。因此本文件以**当前公开 main 的可见树**为唯一审计基准，不依赖旧分支/旧 HEAD 推断。

### 1.2 仓库自己的当前状态声明

仓库 README 已明确：

- 默认训练路径仍是原始 MSHNet topology 和 SLS objective；
- Integrated-DEA、predictive correction 等仍属于历史实验或控制；
- Gate F/G 的审计结果没有自动授权新 loss、architecture 或 long run；
- OHR 仍只是未验证设计说明，不能作为有效方法陈述。

因此，`DEAIntegratedMSHNet_release` 出现在代码树中，并不等价于它已经通过项目方法门。

---

## 2. DER 实际做了什么

### 2.1 路由证据

在尺度 \(s\)，encoder 和 coarse decoder 特征分别投影并限制到 \([-1,1]\)：

\[
\bar e_s=\tanh(P_s^e e_s),\qquad
\bar d_s=\tanh(P_s^d d_{s+1}).
\]

路由输入不是有符号 innovation，而是：

\[
a_s=\bar e_s\odot\bar d_s,
\qquad
r_s=|\bar e_s-\bar d_s|.
\]

然后一个 `1×1` predictor 输出三类：

\[
\{	ext{target/increase},\text{clutter/decrease},\text{uncertain/keep}\}.
\]

### 2.2 decoder correction

原始 MSHNet decoder 保留为 baseline operator：

\[
B_s(e_s,d_{s+1}).
\]

新增 update branch 产生有界更新：

\[
U_s=0.25\tanh\bigl(PW(DW([a_s,r_s]))\bigr).
\]

输出为：

\[
D_s=B_s+(g_s^t-g_s^c)U_s.
\]

### 2.3 terminal scale closure

原始 `Conv2d(4,1,3)` 仍直接计算：

\[
z_{\mathrm{base}}=	ext{Conv}_{4\rightarrow1}([m_0,m_1,m_2,m_3]).
\]

其每尺度贡献被分解为：

\[
c_s=W_s*m_s.
\]

最终输出为：

\[
z_{\mathrm{DER}}
=z_{\mathrm{base}}+
\sum_{s=0}^{3}(g_s^t-g_s^c)|c_s|.
\]

同一个 route 同时控制 decoder correction 和 terminal correction，这是该实现声称的统一性来源。

---

## 3. 工程上真正成立的部分

以下内容可以保留，并且值得肯定。

### 3.1 exact baseline embedding 成立

初始状态强制 uncertain 获胜，因此：

\[
g_s^t=g_s^c=0,
\]

从而：

\[
D_s=B_s,
\qquad
z_{\mathrm{DER}}=z_{\mathrm{MSHNet}}.
\]

仓库报告：

- 四个 side masks 与最终 prediction 对真实 NUAA、NUDT checkpoint 的最大误差均为 `0.0`；
- 初始四尺度 route 均为 uncertain；
- 每个 routing parameter 都能获得非零梯度；
- 完整仓库测试为 `45 passed`。

这证明实现具有较好的 checkpoint 兼容性和结构测试覆盖。

### 3.2 direct baseline convolution 的保留是正确的

把四个 kernel slice 分组卷积后再求和，在实数代数中等价，但 GPU 浮点 reduction order 不同。仓库测得误差最高约 `9.1553e-5`，因此继续使用原始 direct `4→1` convolution 才能维持 bitwise identity。

这是合理的数值工程选择。

### 3.3 uncertain 是 forward identity，而非小权重近似

在 hard winner 为 uncertain 时，forward correction 严格为零，而不是“接近零”。这个代数性质比普通 soft attention 的弱权重更清楚。

但它只能证明初始化安全，不能证明方法新颖或有效。

---

## 4. 致命科学问题

## 4.1 路由证据丢失了动作方向

当前证据算子为：

\[
F(\bar e,\bar d)
=
[\bar e\odot\bar d,\ |\bar e-\bar d|].
\]

它满足两个精确不变性：

\[
F(\bar e,\bar d)=F(\bar d,\bar e),
\]

以及：

\[
F(\bar e,\bar d)=F(-\bar e,-\bar d).
\]

也就是说，一旦进入投影空间，当前接口同时丢掉：

1. 两个 projected sources 的显式顺序；
2. joint sign orientation。

而 router 却要决定有方向的动作：

- increase；
- decrease；
- keep。

需要精确说明：`P_e` 与 `P_d` 是不同投影，所以这不是对原始输入映射的完整“不可能性定理”；投影器可能间接编码 source identity。但在路由接口本身，没有保留一个显式的 signed innovation，动作方向只能依赖投影器学习出脆弱的约定，而不是由证据代数保证。

这与实际 smoke 的动作坍缩一致：当前接口能描述“相似/不相似”，却没有天然表达“应该向上修正还是向下修正”。

### 直接后果

将 `agreement + absolute disagreement` 称为 target/clutter evidence 不充分。它更接近一个**无符号一致性描述符**，不是有向 correction statistic。

---

## 4.2 当前三动作教师会系统性偏向 keep

可选训练控制定义：

\[
q_+=y(1-p),\qquad
q_-=(1-y)p,
\qquad
q_0=yp+(1-y)(1-p),
\]

其中：

- \(q_+\)：increase；
- \(q_-\)：decrease；
- \(q_0\)：keep；
- \(p=\sigma(z_{\mathrm{base}})\)。

它满足：

\[
q_+-q_-=y-p,
\]

即 BCE 对 logit 的负梯度方向。

但三类 argmax 语义存在一个直接问题。

### 前景像素 \(y=1\)

\[
q_+=1-p,
\qquad q_0=p.
\]

只要 baseline 已经给出 \(p>0.5\)，keep 就是主类。

### 背景像素 \(y=0\)

\[
q_-=p,
\qquad q_0=1-p.
\]

只要 baseline 已经给出 \(p<0.5\)，keep 同样是主类。

因此，在一个已训练 MSHNet checkpoint 上：

- 大多数正确前景教 keep；
- 大多数正确背景也教 keep；
- 只有被 baseline 错分或严重欠置信的少量像素教 increase/decrease。

foreground/background 分区平均只能缓解背景像素数量不平衡，不能改变每个区域内部 keep 的主导关系。再加上 uncertain 初始化 bias 和 hard winner，最终进入 100% keep 不是偶然异常，而是当前 teacher geometry 下高度可预期的吸引点。

仓库的一轮 smoke 正好观察到：

- 无 route supervision：target/increase occupancy 为 `0`；
- residual-alignment control：hard keep occupancy 为 `1.0`。

所以当前三态动作没有被识别出来。

---

## 4.3 “硬三态动作”实际上仍带连续概率幅度

代码并不是简单执行：

\[
g\in\{-1,0,+1\}.
\]

实际为：

\[
g_t=\mathbf 1[k=t]\,p_t,
\qquad
g_c=\mathbf 1[k=c]\,p_c.
\]

因此 signed gate 实际属于：

\[
[-1,1],
\]

只是非零方向由 hard winner 决定，幅度仍由 soft probability 决定。

这不一定是错误，但论文不能把它描述成纯离散 \(-1/0/+1\) 干预。更准确的说法是：

> hard-selected, confidence-modulated signed residual。

如果继续强调“decidable hard action”，审稿人很容易指出动作幅度仍是连续 attention confidence。

---

## 4.4 terminal “recursion” 是可交换加法，不是真正递归

代码按 coarse-to-fine 顺序执行：

\[
Z_s=Z_{s+1}+\Delta_s.
\]

但 \(\Delta_s\) 不依赖前一个状态 \(Z_{s+1}\)，所以：

\[
Z_0
=z_{\mathrm{base}}+\Delta_3+\Delta_2+\Delta_1+\Delta_0.
\]

加法交换律意味着：

\[
\Delta_3+\Delta_2+\Delta_1+\Delta_0
=
\Delta_{\pi(3)}+\Delta_{\pi(2)}+\Delta_{\pi(1)}+\Delta_{\pi(0)}
\]

对任意排列 \(\pi\) 都成立。

所以 coarse-to-fine order 对最终结果没有任何可执行意义。`recursive_states` 只是中间诊断轨迹，并没有状态依赖 update。

因此当前方法不应命名为 **Recursion**。更准确的描述是：

> route-coupled additive multi-scale correction。

如果论文仍把 terminal closure 作为“递归”核心创新，审稿风险很高。

---

## 4.5 `±|c_s|` 本质是尺度贡献的取消或加倍

考虑 baseline 中某个尺度贡献 \(c_s\)。

### target/increase action

\[
c_s+|c_s|
=
\begin{cases}
2c_s,& c_s>0,\\
0,& c_s<0.
\end{cases}
\]

### clutter/decrease action

\[
c_s-|c_s|
=
\begin{cases}
0,& c_s>0,\\
2c_s,& c_s<0.
\end{cases}
\]

因此它不是生成新的有向证据，而是：

- 对与动作方向一致的原始贡献做 doubling；
- 对方向相反的原始贡献做 cancellation。

这确实保证最终 logit 单调增加或减少，但也暴露两个问题：

1. correction magnitude 由原 final kernel 的既有贡献决定，而不是由路由不确定度或 residual need 决定；
2. baseline 已经使用了 \(c_s\)，然后又按 route 再加一次 \(|c_s|\)，存在明显的 evidence reuse / double-counting 解释风险。

审稿人可以合理地把它解释为一个带 hard gate 的 scale-contribution reweighting，而不是新的 evidence recursion。

---

## 4.6 同一路由使用两次，不自动证明“单一机制”

同一个 route 同时作用于：

1. decoder residual；
2. final scale correction。

这是 parameter sharing / decision reuse，确实比两个完全独立模块更紧。

但它仍不足以证明二者构成不可分解的统一机制。必须用仓库自己提出的 `2×2` factorial interaction 验证：

\[
I_M
=M_{11}-M_{10}-M_{01}+M_{00}.
\]

只有在多 seed 上稳定出现正 interaction，且完整模型优于两个 partial variants，才能排除“两个残差模块简单叠加”的解释。

当前没有这组有效实验。

---

## 4.7 与项目 North Star 直接冲突

当前冻结的 North Star 明确要求：

\[
\mathrm{InferenceCost}_{\theta}
=
\mathrm{InferenceCost}_{\mathrm{baseline}},
\]

并写明：

> 在不改变 MSHNet 推理图的前提下改善组件级 Pd–FA frontier。

DER 当前增加：

- 四个 routing cells；
- 20,988 个显式路由参数；
- 约 0.190710 GMAC 的 routing cells；
- 约 0.002359 GMAC 的 grouped scale decomposition；
- 合计约 0.193069 added GMAC；
- 动态 route、update branch 与 terminal correction。

因此它在进入性能讨论前，就已经违反当前项目宪法。

只有两种诚实处理方式：

1. **维持 North Star**：DER 只能作为隔离架构对照，不能成为主线；
2. **正式修订 North Star**：公开承认允许少量推理开销，再重新做模型选择。

不能一边声称 inference graph invariance，一边把 DER 写成正式方法。

---

## 4.8 当前监督没有对准组件级低虚警错误

项目目标是：

\[
\text{more matched GT components at the same unmatched-component area}.
\]

但 residual-action teacher 只使用：

\[
y-p
\]

的像素级 BCE geometry，并把同一 full-resolution teacher 双线性缩放后监督所有尺度 route。

它没有区分：

- calibration-selection-sensitive miss；
- peak-order-limited miss；
- component-conversion-limited miss；
- unmatched false-alarm component；
- 仅边界像素误差但不影响 component matching 的普通 segmentation error。

因此，即使 action loss 下降，也不能推出低 FA Pd–FA frontier 改善。

这正是此前 Gate D/E/F 一直避免的 surrogate leap。

---

## 5. 与近期方法的创新冲突

DER 不能把以下 primitives 单独当创新：

- elementwise product；
- absolute difference；
- hard/soft routing；
- straight-through estimator；
- abstention/reject option；
- residual adapter；
- function-preserving initialization；
- multi-scale dynamic fusion。

### 5.1 通用先例

- **Net2Net** 已系统提出 function-preserving network expansion；
- **Network Morphism** 也研究了保持原网络函数不变的结构扩展；
- hard routing + STE、mixture-of-experts gating、selective/abstaining prediction 都是成熟方向。

因此“初始等于 baseline”“uncertain 为 identity”“hard route 可反传”属于良好工程约束，不足以单独构成 AAAI 级贡献。

### 5.2 近期 IRSTD 方法压力

近期直接相关方法已经覆盖：

- **MSHNet, CVPR 2024**：scale/location sensitivity 与多尺度 side supervision；
- **PConv + Scale-based Dynamic Loss, AAAI 2025**：目标空间形态和尺度敏感动态损失；
- **SAIST, CVPR 2025**：对比式 vision-language / SAM 引导；
- **DEFANet, AAAI 2026**：edge-target dual path 与 frequency-aware enhancement；
- 2026 年工作还进一步覆盖 noise suppression、invertible encoder、物理/频率建模等方向。

DER 唯一可能保留的差异不是“又一个动态融合模块”，而只能是：

> 一个动作可辨识、方向有定义、与 baseline 精确嵌套，并被组件级证据支持的单一干预算子。

当前实现尚未满足其中最关键的“动作可辨识”和“组件级证据支持”。

---

## 6. 当前模型冻结决策

### 6.1 立即冻结

```text
OHR-MSHNet                     FINAL NO-GO
Generic risk-control route     NO-GO
Unique-logit / exact-event     NO-GO
Current DER implementation     NO-GO as main model
DER long training              NO-GO
DER mechanics/control usage    KEEP, quarantined
Gate G / route identifiability GO
```

### 6.2 不允许的下一步

在没有新证据前，不做：

- 400 epoch DER 训练；
- 调 uncertain margin 直到 test 指标变好；
- 继续叠加 edge/frequency/Mamba/attention；
- 给 route 再加多个 auxiliary losses；
- 用一轮含 test leakage 的 smoke 指标写摘要；
- 把 `45 passed` 当成模型有效性证据；
- 把 terminal additive correction 称为 recursion。

---

## 7. 唯一应先执行的最小门：G-RI（Route Identifiability）

这个门不训练完整网络，不碰 official test，不需要长时间运行。

## 7.1 目标

回答一个最小问题：

> MSHNet 已有冻结特征中，是否存在能够稳定区分 increase、decrease、keep 的有向信息？

如果连冻结 probe 都不能识别动作，端到端 hard router 更没有理由成功。

## 7.2 数据与标签

对每个 dataset × seed 的 clean baseline checkpoint：

1. 冻结 MSHNet；
2. 只缓存 encoder feature、decoder feature、side logits、final logit；
3. 不更新 backbone；
4. 不访问 sealed test；
5. 只在 fit/validation manifest 上构建 probe。

使用两级标签。

### A. 信息充分性诊断标签

以 detached baseline residual 为简单诊断：

\[
r=y-\sigma(z).
\]

按预注册 dead zone \(\epsilon\) 定义：

\[
a=
\begin{cases}
+1,&r>\epsilon,\\
-1,&r<-\epsilon,\\
0,&|r|\le\epsilon.
\end{cases}
\]

这只用于判断 evidence 是否携带方向信息，不是最终方法目标。

### B. 组件级相关标签

在预注册 operating thresholds 下，通过 exact component matcher 标记：

- `+1`：与 persistent missed GT 支持相关、需要提高局部响应的区域；
- `-1`：属于 unmatched predicted component、需要压低的区域；
- `0`：已正确匹配且不影响组件决定的区域。

该层决定 probe 信息是否真正覆盖 North Star 错误，而不仅是普通像素残差。

## 7.3 必须比较的证据接口

保持 probe 参数预算一致，至少比较：

### A. 当前 DER 证据

\[
[\bar e\odot\bar d,\ |\bar e-\bar d|].
\]

### B. signed innovation

\[
[\bar e-\bar d,\ \bar e+\bar d].
\]

### C. ordered pair

\[
[\bar e,\bar d].
\]

### D. logit-only control

只使用 baseline side/final logits，判断复杂 feature router 是否真的比原分数更有信息。

probe 只用 linear / `1×1` classifier，禁止添加深层网络掩盖 evidence 接口本身的问题。

## 7.4 建议预注册的硬门

以下阈值应在查看结果前写入 registry：

1. increase-vs-decrease balanced accuracy 在每个 eligible dataset 上均高于 `0.60`；
2. 三类 macro-AUROC 在每个 eligible dataset 上均不低于 `0.70`；
3. increase 和 decrease 两个非 keep 类都必须有可评估支持，不能用 all-keep 获得表面高 accuracy；
4. 结果方向需在至少 3 seeds 中一致；
5. 组件级标签下，probe 必须覆盖可观比例的 persistent misses 和 unmatched false-alarm components；推荐用 baseline-seed variance 冻结最小 material-coverage threshold，而不是事后挑值；
6. 当前 DER evidence 若显著低于 signed innovation / ordered pair，则当前 DER algebra 直接否决。

## 7.5 决策树

```text
A. 当前 DER evidence 通过，且组件覆盖通过
   -> 才允许一个短 paired routing run；仍不能直接 long-run。

B. 当前 DER evidence 失败，但 signed innovation 通过
   -> 当前 DER 保持 NO-GO；只允许改写 evidence algebra。

C. 所有 evidence 都失败
   -> 整条 routing 路线 FINAL NO-GO，停止加 gate/module/loss。

D. 像素标签通过但组件标签失败
   -> 说明只学到 ordinary segmentation residual，不能作为低-FA 方法。
```

这个 gate 的价值是把“又一次模型失败”压缩成一个低成本、可证伪的信息测试，而不是再赌一轮 400 epoch。

---

## 8. 条件修复方向：Signed Innovation Routing

只有当 G-RI 证明 signed evidence 明显优于当前无符号证据时，才考虑下面的单点修复。

### 8.1 只改错误位置

删除：

\[
[\bar e\odot\bar d,|\bar e-\bar d|]
\rightarrow \text{3-way softmax}.
\]

改为显式 signed innovation：

\[
u_s=\bar e_s-\bar d_s.
\]

用一个标量方向量：

\[
q_s=K_s*u_s.
\]

再用有 dead zone 的 signed action：

\[
g_s=
\operatorname{sign}(q_s)
\frac{\operatorname{ReLU}(|q_s|-\tau_s)}
{|q_s|+\varepsilon}.
\]

其动作含义天然为：

- \(g_s>0\)：increase；
- \(g_s<0\)：decrease；
- \(g_s=0\)：keep。

这样不需要三类 label permutation，也不需要把 keep 作为 softmax 主类；方向直接来自 signed innovation。

### 8.2 为什么它仍只是 conditional candidate

它仍会引入 input-dependent inference routing，因此仍违反当前 North Star 的 inference-graph invariance。

所以只有在以下二选一后才可推进：

1. 项目正式放宽部署约束，允许小幅 inference overhead；
2. 找到可以在训练后严格折叠到原 MSHNet 参数中的静态变换。

动态 sample-dependent gate 通常不能被普通 weight folding 消除。因此，在当前冻结宪法下，它不能直接成为主模型。

---

## 9. 7 月 21 日摘要前的执行安排

### 7 月 12 日

- 冻结本审计；
- current DER 不启动 long run；
- 固定 G-RI manifests、labels、metrics 和阈值。

### 7 月 13 日

- 缓存 clean baseline features；
- 完成 current / signed / ordered / logit-only 四组 probe；
- 形成 dataset × seed 结果表。

### 7 月 14 日

- 按 G-RI 决策树做唯一一次路线选择；
- 若所有 evidence 失败，routing 路线当天关闭；
- 若 signed innovation 通过，明确决定是否修订 inference constraint。

### 7 月 15–16 日

只在 gate 通过时：

- 实现单一 signed operator；
- 先做 NUDT、NUAA paired short run；
- 同 checkpoint、同 optimizer policy、同 epochs、同 seed；
- 不看 sealed test。

### 7 月 17–18 日

只在两个 short gates 都通过时：

- 扩到第三 dataset；
- 至少 3 seeds；
- 运行 decoder-only / closure-only / full 的 `2×2` interaction；
- 报告 official legacy 与 audit Hungarian 的完整 frontier。

### 7 月 19 日

- 冻结论文 claim；
- claim 必须与实际证据一致；
- 若 unified interaction 不为正，删除“统一递归”叙事。

### 7 月 20 日

- 摘要、方法图、核心公式、结果表交叉审计；
- 检查数据泄漏、checkpoint selection 和 threshold protocol。

### 7 月 21 日

- 提交摘要；
- 不使用 smoke result 作为 paper number；
- 不使用未经 gate 验证的模型名作为既成贡献。

---

## 10. 摘要可写与不可写的边界

### 当前不可写

- “DER significantly improves MSHNet”；
- “three actions are semantically identifiable”；
- “recursive coarse-to-fine closure”；
- “no additional inference cost”；
- “validated on NUAA/NUDT”；
- “target/clutter routing is learned without auxiliary supervision”。

### 只有通过全部门后才可写

- exact baseline embedding；
- action-identifiable signed intervention；
- paired improvement over continued MSHNet；
- positive decoder–closure interaction；
- cross-seed component-level Pd–FA frontier gain；
- mask-quality non-inferiority。

---

## 11. 最终项目状态

\[
\boxed{
\begin{aligned}
&\text{OHR: FINAL NO-GO}\\
&\text{Current DER: AAAI MAIN-MODEL NO-GO}\\
&\text{DER mechanics: PASS, quarantine as control}\\
&\text{Route identifiability probe: GO}\\
&\text{Signed innovation repair: CONDITIONAL}\\
&\text{Long training: NOT AUTHORIZED}
\end{aligned}
}
\]

最关键的不是立即再写一个复杂网络，而是先回答一个极小但决定性的科学问题：

> **当前 MSHNet 特征里到底有没有稳定、跨数据集的有向 correction information？**

只有答案为“有”，才值得把它变成模型；否则任何 router、abstention、attention 或 auxiliary loss 都只是继续堆叠。

---

## 12. 主要审计来源

### DEA 仓库

- [Current repository root](https://github.com/Arialliy/DEA)
- [Current main commit history](https://github.com/Arialliy/DEA/commits/main/)
- [Current HEAD: 30874cc](https://github.com/Arialliy/DEA/commit/30874cc7ba85540a42d84a6cd2541f665a721bc3)
- [Integrated DEA release commit: 3d5436a](https://github.com/Arialliy/DEA/commit/3d5436a9280ab897dcce05b37bf14d99a455b118)
- [Integrated model implementation](https://github.com/Arialliy/DEA/blob/main/DEAIntegratedMSHNet_release/model/dea_integrated_mshnet.py)
- [Residual action loss](https://github.com/Arialliy/DEA/blob/main/DEAIntegratedMSHNet_release/model/dea_integrated_loss.py)
- [Paper positioning](https://github.com/Arialliy/DEA/blob/main/DEAIntegratedMSHNet_release/docs/PAPER_POSITIONING.md)
- [Experiment protocol](https://github.com/Arialliy/DEA/blob/main/DEAIntegratedMSHNet_release/docs/EXPERIMENT_PROTOCOL.md)
- [Local smoke report](https://github.com/Arialliy/DEA/blob/main/DEAIntegratedMSHNet_release/validation/local_smoke_report.json)
- [North Star](https://github.com/Arialliy/DEA/blob/main/MSHNet_North_Star_Objective_and_Gate_E_Positioning.md)

### 近期 IRSTD 与通用先例

- [MSHNet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html)
- [PConv and Scale-based Dynamic Loss, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32996)
- [SAIST, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_SAIST_Segment_Any_Infrared_Small_Target_Model_Guided_by_Contrastive_CVPR_2025_paper.html)
- [DEFANet, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37368)
- [Net2Net](https://arxiv.org/abs/1511.05641)
- [Network Morphism](https://proceedings.mlr.press/v48/wei16.html)
