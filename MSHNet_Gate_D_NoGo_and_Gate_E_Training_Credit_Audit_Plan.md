# MSHNet Gate D 冻结结论与 Gate E v2.2 可执行审计方案

> 状态：**Gate E v2.2，2026-07-12。E−1a 与 E−1b PASS，E−1c FAIL，因此训练信用路线的 Gate E0 NO-GO。v2.2 的运行前协议哈希保存在 E−1b/E−1c provenance；本文后续结果段是运行后的审计记录。**
>
> 上位约束见 [MSHNet_North_Star_Objective_and_Gate_E_Positioning.md](./MSHNet_North_Star_Objective_and_Gate_E_Positioning.md)。
>
> 本文只授权只读 Gate E−1。Gate E0、LOO/Shapley、metric-constrained solver 和长周期方法训练均未授权。

---

# 0. 当前执行判定

| 阶段 | 当前判定 | 可以做 | 禁止做 |
|---|---|---|---|
| Gate D0/D1 | **冻结 PASS / 方法 NO-GO** | 保留证据和审计工具 | 不进入 D2 训练 |
| Gate E−1a fixed | **PASS** | 保留 immutable 主账本 | 不作训练因果结论 |
| checkpoint policy | **PASS** | 保留 best-IoU sensitivity 与转移矩阵 | 不把 best 与 fixed 合并 |
| Gate E−1b | **PASS** | 保留 immutable prediction-free bundle | 不把低 AUROC 写成因果排除 |
| Gate E−1c | **FAIL** | 保留双 matcher cross-fitted ledger | 不放宽零超调或事后换预算 |
| Gate E0 | **NO-GO** | 暂无 | 不实现、不训练 local-influence probe |
| LOO / Shapley | **PAUSED** | 暂无 | 不先做昂贵归因 |
| 受约束更新器 | **NO-GO** | 仅保留隔离的数学草图 | 不接入 main.py，不跑 solver |
| 顶会方法训练 | **NO-GO** | 暂无 | 不以诊断量包装新 loss 或模块 |

唯一立即执行路径为：

\[
\boxed{
\text{E−1a PASS}
\rightarrow
\text{E−1b PASS}
\rightarrow
\text{E−1c FAIL}
\rightarrow
\text{E0 NO-GO}
}
\]

---

# 1. Gate D 的冻结结论

## 1.1 能够成立的结论

九组 pristine MSHNet checkpoint 的只读审计未发现能够统一解释 no-response 的 readout、final-fusion 或简单 calibration 瓶颈：

- context-matched 后，distinct 在 no-response 和成功对照中都很常见；
- final availability 与 utilization 同时变化，不能归因于纯 utilization；
- d0 到 final 的有向符号绝大多数保持；
- fusion cancellation 在成功对照中也常见；
- nominal cross-fitted FA budget 几乎不能恢复既有 no-response；
- \(AHU=w^\top\Delta\)，优化该乘积退化为普通有向 logit contrast；
- \(U=1\) 不保证绝对响应或 component match。

因此：

> **Evidence-to-Decision Utilization 不适合作为当前顶会主创新，停止进入 D2。**

## 1.2 不能成立的结论

Gate D 没有观察训练过程，因而不能推出：

- representation availability 是 miss 的首要根因；
- 训练首先破坏了 availability；
- 稠密背景梯度已经被证明压倒稀疏目标梯度；
- 困难目标已被证明获得不足的“训练信用”。

推荐总括为：

> Gate D0/D1 排除了统一的末端 readout、fusion 和简单 calibration 解释；训练期 margin 错位只是下一步需要检验的假设，不是 Gate D 已证明的原因。

---

# 2. Gate E 与 North Star 的关系

固定外部目标是低组件虚警区域的 Pd–FA frontier，而不是 target logit、loss 或某个内部指标。

Gate E 只检验一条候选机制链：

\[
\text{真实训练更新}
\rightarrow
\text{冻结 target--clutter margin 的局部变化}
\rightarrow
\text{exact component match}
\rightarrow
\text{Pd--FA frontier}.
\]

其中：

- exact component match 是离散事实；
- margin 是连续诊断桥梁；
- local influence 是模型状态附近的一阶量；
- 三者都不能在没有干预实验时写成因果训练信用；
- Gate E 即使通过，也只提供方法动机，不自动产生新颖方法。

---

# 3. 已冻结的运行与 checkpoint 事实

## 3.1 Clean protocol

主 manifest：

    repro_runs/clean/clean_baseline_holdout_v1/manifest.json

manifest SHA256：

    d483d959c774fd5bf20289c0e29edef5a57a399788c5819f321304506c3b9b92

运行网格：

- datasets：IRSTD-1K、NUAA-SIRST、NUDT-SIRST；
- seeds：20260711、20260712、20260713；
- 共 9 个 job，return code 全为 0；
- 每个 epoch_metric.log 恰有 400 行，末轮为 epoch 399；
- 同一数据集三个 seed 的 validation split hash 完全一致。

## 3.2 Primary 与 sensitivity checkpoint

**fixed-epoch primary**

- 文件：checkpoint.pkl；
- 9 个文件均记录 epoch 399；
- 与配置的 epochs−1 一致；
- 用于回答“相同训练预算结束时是否重复失败”。

**retrospective sensitivity**

- 文件：checkpoint_best_iou.pkl；
- best epoch 跨 seed 为 201–397；
- 受同一 validation 上的 checkpoint selection 影响；
- 只能回答“被选中模型的错误是否相似”，不能作为优化稳定性 primary。

## 3.3 Resume 披露

以下两次运行发生过断点恢复：

- IRSTD-1K / seed 20260712：从 checkpoint epoch 234 后恢复；
- NUDT-SIRST / seed 20260712：从 checkpoint epoch 390 后恢复。

最终 epoch 与 optimizer step 数完整，可作为完成 400 epoch 的 seed realization；但 checkpoint 没有保存 RNG/DataLoader state，禁止写成 9 个 uninterrupted、bitwise-reproducible trajectories。provenance 必须记录 resumed 状态。

## 3.4 Formal-run 前的 rejected preflight bundle

2026-07-12 首次 fixed-epoch CLI 在独立 preflight 结论返回前已经完成，并暴露 `N3=18/493`。该 bundle 因以下 provenance/协议缺口被拒绝，不作为正式 primary：

- source hash 未在推理前冻结并在输出前复核；
- Git/optimizer provenance 不完整；
- registry 构造晚于 checkpoint metadata load；
- target row 缺少完整 run/config/split/resume assertions；
- matcher Pd/count 与 policy recurrence comparison 不完整。

拒绝产物保留于：

    repro_runs/gate_e/persistence_v2/fixed_epoch_rejected_preflight_20260712

因此，修复后的 fixed-epoch rerun 是对已经可见结果的证据链复现，不再称 outcome-blind primary 或 preregistration。routing thresholds 在首次结果前已经写定，但所有正式结论必须以修复后 immutable bundle 为准。

---

# 4. 两套匹配口径必须隔离

| 名称 | 规则 | 本阶段用途 |
|---|---|---|
| official_legacy | target-order greedy，严格质心距离 \(<3\)，8-connectivity | 与原始 MSHNet 指标保持可比 |
| audit_hungarian | 最大匹配数优先，再最小总质心距离；同样严格 \(<3\)，8-connectivity | 稳定逐目标身份和状态 |

Gate E−1 的逐目标 primary 使用 audit_hungarian，因为它不依赖目标遍历顺序。每个 checkpoint 同时汇总 official_legacy 作为 matcher sensitivity。

阈值规则：

- primary：logit 严格大于 0，等价于概率严格大于 0.5；
- tie：logit 恰为 0 判为背景；
- 每个 `dataset × seed × matcher × checkpoint policy` 同时报告 achieved FA/Mpix；该点只称 official-point，除非它实际落入冻结低 FA 区间，否则不能称 low-FA operating point；
- `no_response`：Hungarian-unmatched，且目标 support 的严格 3 像素邻域内没有预测组件；该类别优先于合法质心边检查，以保持现有 component ledger 语义；
- `centroid_miss`：Hungarian-unmatched，存在 support-near 预测组件，但不存在质心距离严格小于 3 的合法边；
- `assignment_residual`：Hungarian-unmatched，同时存在 support-near 预测组件与至少一条合法质心边，但因全局一对一分配仍未匹配。

三个 subtype 互斥并覆盖所有 Hungarian-unmatched targets；实现必须断言这一 partition，不能靠名称推断。

---

# 5. Gate E−1：failure support、跨 seed recurrence 与 seed variation

## 5.1 研究问题

Gate E−1 把 recurrence 当作待检验结果，而不是前提。它只回答：

1. 同一 canonical resized GT component 在三次 fixed-budget 训练后分别是 matched 还是 unmatched？
2. miss 事件集中在 observed 3/3 miss，还是大量发生 seed variation？
3. 该结构是否由单一数据集或少量图像主导？
4. best-IoU checkpoint selection 是否改变 primary 结论？

它不能单独回答：

- miss 是输入难度、表示、监督还是优化造成；
- seed flip 一定是梯度竞争；
- 3/3 miss 一定是监督长期失效；
- 某个 target 应在训练时被赋予更高权重。

## 5.2 Prediction unit 与稳定身份

prediction unit 冻结为：

> canonical validation pipeline 产生的 \(256\times256\) 最近邻 resize 标签中的 8-connected GT component。

每张图先得到二值完整标签 \(Y_i\)。对每个 component \(C_{ik}\)：

1. 构造与完整图同尺寸的 uint8 component mask；
2. 使用 versioned shape encoding、row-major bytes 和 SHA256；
3. 以 dataset、image_name、component-mask digest 构成 stable key；
4. full-label SHA256、bbox、area、centroid、canonical index 和 source index 只作 assertion metadata。

在加载任何 checkpoint 前，必须直接从 canonical validation split 与 resize 后标签建立一次权威 expected registry。registry 覆盖 split 中的 **每张图**，包括没有 GT component 的 target-free image。checkpoint inference 产生的每个 seed 账本只能与该 registry 比对，不能让三个 seed 互相相等就视为完整。

严禁使用以下字段作为主身份：

- 浮点 centroid；
- bbox 字符串；
- component index；
- area；
- seed 内 regionprops 顺序。

合并 seed 前必须验证：

- 每张图的 full-label hash 一致；
- 每个 seed 的 image universe 与 expected registry 完全相同，target-free image 也不能缺失；
- 三个 seed 的完整 stable target set 完全相同；
- shape 恰为 \(256\times256\)；
- connectivity 恰为 2（skimage 的二维 8 邻接）；
- split hash、sample order、resize 和 threshold 配置一致。

任一项不一致均 fail-closed。

## 5.3 每个 target 的完整状态行

每个 policy、dataset、seed、target 必须恰有一行，不能只记录失败目标。必需字段：

| 类别 | 字段 |
|---|---|
| identity | dataset、image_name、stable_target_id（即 StableTargetId.stable_key）、component digest、label digest |
| assertions | height、width、pixel connectivity=8、skimage connectivity=2、canonical/source index、bbox、area、centroid；全部必须与 expected registry 相等 |
| run | seed、checkpoint policy/name/hash/epoch、run/config/split hash、resumed |
| decision | threshold semantics、threshold=0、matcher、radius=3、connectivity=8 |
| outcome | matched、miss、matched prediction index、centroid distance |
| subtype | no_response、centroid_miss、assignment_residual 或 matched |

对每个 policy 要动态断言：

\[
N_{\mathrm{rows}}
=
N_{\mathrm{unique\ targets}}\times3.
\]

当前验证集合预计约有 493 个 target，但实现不得把 493 硬编码为真值。

## 5.4 复现记账

对目标 \(k\)：

\[
m_k^{(r)}
=
\mathbf 1[\text{target }k\text{ unmatched in seed }r],
\qquad
c_k=\sum_{r=1}^{3}m_k^{(r)}.
\]

定义：

- \(N_0\)：0/3 miss；
- \(N_1\)：1/3 miss；
- \(N_2\)：2/3 miss；
- \(N_3\)：3/3 miss。

总 miss 事件：

\[
N_{\mathrm{event}}
=
N_1+2N_2+3N_3.
\]

observed 3/3 targets 对 miss 事件的占比：

\[
R_{3/3}
=
\frac{3N_3}{N_{\mathrm{event}}}.
\]

若 \(N_{\mathrm{event}}=0\)，\(R_{3/3}\) 必须为 null/NA，不能填 0。

报告层级：

- 每个 dataset 的 \(N_0,N_1,N_2,N_3\)；
- overall micro；
- 每 seed 原始 miss 数；
- unique 3/3 target images；
- no-response subtype 的独立 \(N_0\)–\(N_3\)；
- 三个 dataset rate 的未加权 macro，仅作为域平衡描述，不与 micro 混称。

dataset macro 冻结为各 dataset 的 \(N_3/N\) 等权平均；\(R_{3/3}\) macro 只在 event count 大于 0 的 dataset 上平均，并同时报告 defined/undefined dataset 数。overall target-micro 与 dataset-macro 必须使用不同字段名。

只允许写“observed 3/3 miss in these runs”，不能写成总体 persistent、always missed 或 seed-invariant。

## 5.5 不确定性

primary CI 使用 dataset-stratified image-cluster percentile bootstrap：

- 在每个 dataset 内、仅对 registry 中至少含一个 target 的 images 重采样；target-free images 仍用于完整性验证，但不属于 target-level recurrence estimand；
- 同一 image 的全部 targets 与三个 seed 状态一起移动；
- overall 在 dataset 内分层后合并；
- 2,000 replicates；
- RNG seed 20260712；
- 报告 \(N_3/N\) 与 \(R_{3/3}\) 的 95% percentile interval。

若 point estimate 有 miss、但某些 bootstrap replicate 的 \(N_{\mathrm{event}}=0\)，这些 replicate 必须计入 replicates_undefined；事件占比只能报告明确标注 conditioning=
\(N_{\mathrm{event}}>0\) 的 conditional interval，不能静默丢弃后仍命名普通 CI。dataset macro 必须在同一组联合分层重采样中逐 replicate 计算，不能平均各 dataset 的 CI 端点。

该 CI 只覆盖 image sampling：

- 不覆盖只有 3 个 training seed 的不确定性；
- 不覆盖只有 3 个 dataset 的 domain uncertainty；
- 不支持跨 seed 或跨域总体显著性声明。

## 5.6 Prediction-free image-and-annotation 难度分析属于 E−1b

这些变量使用 GT component geometry，不能称为纯 input-only。准确名称是 prediction-free image-and-annotation covariates。先冻结完整账本并作 E−1a 描述，再运行已在 fixed-epoch outcome 生成前写定的 E−1b；不得根据 E−1a 的相关性重新选变量。

分析单位是一行一个 stable target；primary response 为：

\[
y_k^{\mathrm{any}}=\mathbf 1[c_k>0],
\]

即三个 seed 中是否至少一次 miss。secondary response 为：

\[
y_k^{3/3}=\mathbf 1[c_k=3].
\]

最多使用四个 prediction-free covariates：

1. \(\log(1+\mathrm{area})\)；
2. border distance：
   \[
   \min_{(y,x)\in C}\min(y,x,H-1-y,W-1-x);
   \]
3. local robust SCR；
4. local ring robust dispersion。

强度 \(I\) 必须复现 validation 输入：源 PNG 经 PIL convert RGB，双线性 resize 到 \(256\times256\)，通道值除以 255，再固定为：

\[
I=0.2126R+0.7152G+0.0722B.
\]

若源数据经 convert RGB 后不能稳定落到 8-bit 三通道，E−1b fail-closed。对 component \(C\)，背景 ring 为到 \(C\) 的 Euclidean distance 满足 \(2\le d<9\) 且不属于任何 GT 的像素。定义：

\[
\mathrm{MAD}_{\mathrm{ring}}
=
1.4826\,
\operatorname{median}_{u\in R}
\left|I_u-\operatorname{median}_{v\in R}I_v\right|,
\]

\[
\mathrm{SCR}
=
\frac{\operatorname{mean}_{u\in C}I_u-\operatorname{median}_{v\in R}I_v}
{\mathrm{MAD}_{\mathrm{ring}}+10^{-6}}.
\]

ring 像素少于 16 时变量标记 unavailable，不插值；若任一变量 unavailable 的 targets 超过 5%，E−1b 标记 unresolved。不得加入 model logit、feature、best epoch、seed outcome 或事后选出的 clutter 指标。

两个模型严格分开：

- association：target-level binomial descriptive model，dataset fixed blocking，image-cluster robust uncertainty；只作关联描述；
- LODO：训练 fold 内标准化、固定 L2 logistic \(C=1\)、无 dataset dummy、无数据驱动调参；报告 AUROC、average precision、held-out positive/negative image clusters。

每个 LODO training fold 与 held-out fold 都必须至少有 10 张 positive images 和 10 张 negative images；否则该 fold ineligible。少于两个 eligible held-out datasets 时，难度解释状态为 unresolved，不能把“未拟合”解释成“难度不能解释”。完全分离时不报告普通 MLE 系数。

v2.2 将实现自由度进一步冻结如下；这些定义在 E−1b AUROC 产生前写定：

- 只读取 formal fixed-epoch ledger；每个 stable target 的三条 seed 状态先折叠。routing 只由 primary \(y^{\mathrm{any}}\) 决定；\(y^{3/3}\) 是 secondary sensitivity，association 不进入 routing；
- 四个变量必须联合进入模型，不作 winsorization、clipping、取对数、缺失插补、missing indicator 或事后删变量；`ring_pixel_count` 与 `MAD=0` 只作 QA；
- association 与 LODO 都只用 complete cases；除 overall unavailable rate 外，同时报告各 dataset 缺失率。任一训练折变量标准差为零/非有限或优化不收敛，该 fold unresolved，禁止临时删变量；
- positive-support image 定义为至少含一个 complete-case \(y=1\) target，negative-support image 定义为至少含一个 complete-case \(y=0\) target；两集合允许在多目标图像上重叠，cluster key 为 `(dataset, image_name)`，target-free image 不计；
- association 在全体 complete cases 上对四变量按 population std（`ddof=0`）标准化，`IRSTD-1K` 为 dataset reference，普通无惩罚 binomial MLE；先用线性规划检测 complete/quasi separation，协方差固定为 image-cluster sandwich CR0、无 small-sample correction；
- LODO 只在两个 training datasets 上拟合 mean/std（`ddof=0`），再应用于 held-out dataset；无 dummy、class weight、变量选择或调参。目标固定为
  \[
  \sum_i\ell_{\mathrm{logistic}}(y_i,b+x_i^\top\beta)
  +\frac12\lVert\beta\rVert_2^2,
  \]
  即截距不惩罚、\(C=1\)、float64、零初始化、确定性 L-BFGS 与解析梯度；
- AUROC/AP 是 target-level 指标，image clusters 只审计支持度；eligible folds 的 AUROC 作 dataset-unweighted macro mean。只有 `mean AUROC >= 0.90` 且每个 eligible fold `AUROC >= 0.85` 才触发 prediction-free difficulty 的 E0 NO-GO；这仍是项目路线判据，不是因果解释。

## 5.7 E−1c：与低 FA North Star 的桥梁

fixed logit 0 的 E−1a 是稳定 official-point audit，不自动属于低 FA。若 E−1a 达到样本支持，E0 开放前还必须生成 matcher-specific cross-fitted ledger：

- nominal budgets 固定为 \(\{1,5,10,20\}\) FA/Mpix；
- primary 只用 fixed-epoch checkpoint；best-IoU 不进入 E−1c；
- 固定为 deterministic two-fold image-disjoint cross-fit：
  \[
  h(i)=\operatorname{uint64be}\left(
  \operatorname{SHA256}(\texttt{mshnet-decision-operating-fold-v1\\0}+i)_{0:8}
  \right)\bmod 2,
  \]
  完整 image-to-fold mapping 写入 provenance；dataset 内 mapping 在三个 seed 间完全共享；
- audit_hungarian 与 official_legacy 分别校准；
- threshold grid 只包含固定 logit `0` 与
  `sorted({0.5,0.75,0.85,1.0} ∪ {1-x | x∈geomspace(1e-1,1e-7,49)})`
  在 calibration logits 上用 NumPy `quantile(method="linear")` 得到的 53 个 tail-quantile 候选；不加入旧 Gate D 的 13 个 probability-logit 候选；quantile `1.0` 即 calibration maximum/all-off candidate；
- 所有阈值严格使用 `logit > tau`；calibration tie-break 固定为匹配数更多、未匹配预测面积更小、阈值更高；held-out logits/labels 不参与 grid 或 threshold selection；
- 每个 held-out target 记录所用 calibration threshold、matched status 与 achieved aggregate FA；
- target-free images 必须进入 held-out pixel/FA 分母；fold 先累计整数 matches/targets/unmatched area/pixels，`dataset × seed × matcher × budget` 再池化，禁止对 fold 比率求平均；
- 预算可行使用无 epsilon 的整数零超调：
  \[
  A_{\mathrm{unmatched}}10^6\le\alpha A_{\mathrm{image}}.
  \]

原门槛“低 FA 下至少 12 个 2/3 miss”可被 all-off 阈值平凡满足，也没有要求这些 targets 与 E−1a failure 相同，因此 v2.2 在 E−1c 结果生成前将其强化。对 matcher \(g\) 定义同一 target 的 fixed-logit-0 与 low-FA miss count 为 \(c^0_{k,g}\) 和 \(c^\alpha_{k,g}\)。只有存在同一个 \(\alpha\)，同时满足以下全部条件，E−1c 才 PASS：

1. 至少两个 datasets 在 official_legacy 与 audit_hungarian 下的三个 seeds 都精确预算可行；
2. 这些 eligible datasets 的每个 `dataset × seed × matcher` 至少匹配一个 target，任何 pooled all-off operating point 都被 veto；
3. eligible datasets 合计至少 12 个 unique bridge targets 同时满足
   \[
   c^0_{k,\mathrm{legacy}}\ge2,
   \quad c^\alpha_{k,\mathrm{legacy}}\ge2,
   \quad c^0_{k,\mathrm{hungarian}}\ge2,
   \quad c^\alpha_{k,\mathrm{hungarian}}\ge2;
   \]
4. 同一 eligible datasets 至少有 12 个 stable controls，在两个 matcher 的 fixed 与 low-FA 下都至多 1/3 miss。

该 joint gate 使 legacy North Star primary 与 order-invariant audit matcher 必须同向，并把“低 FA 下很多目标都关掉”与“同一 official-point failure 在可行低-FA点持续存在”分开。否则 E0 不能宣称研究低-FA frontier。

## 5.8 Pilot-visible prospective routing gate

这不是完全独立的 preregistration：Gate D 与 best-checkpoint pilot 已经可见。routing thresholds 在首次 fixed-epoch 尝试前冻结；但 v2.1 provenance 修复发生在 rejected preflight outcome `N3=18/493` 可见后。正式 rerun 只能称 outcome-visible evidence-chain replication。运行前 provenance 必须记录两份协议文档的 SHA256、冻结时间、rejected bundle 与全部已知 pilot 清单。

### Integrity gate

以下全部通过才可解释结果：

- 9 个 primary checkpoint 均为 epoch 399；
- 9 个 job 完整；
- 目标集合三 seed 完全一致；
- 每 target 恰有三行；
- fixed threshold、matcher、resize 与 hashes 完整；
- 两个 resumed run 已披露；
- official 与 audit matcher 差异已量化。
- 每个组合的 achieved FA/Mpix 已报告，official-point 与 low-FA 表述没有混用。

任一失败：**E−1 BLOCKED**，先修证据链。

### Failure-support gate

在前述 image/target 总量条件之外，至少满足一个 arm：

**recurrence arm**

- observed 3/3 targets 至少 12 个；
- 分布在至少两个 datasets，且每个不少于 4 个；
- \(R_{3/3}\ge0.25\)。

**seed-variation arm**

- observed 1/3 或 2/3 miss targets 合计至少 18 个；
- 分布在至少两个 datasets，且每个不少于 6 个；
- 这些 targets 贡献的 miss events 占全部 miss events 至少 25%。

这些数值只是 ahead-of-primary 的工程 probe-support gate，不是总体显著性阈值。

### Policy-transition gate

必须生成每个 target 的 \((c_{\mathrm{fixed}},c_{\mathrm{best}})\) \(4\times4\) 转移矩阵，并报告：

- fixed 与 best 的 unique missed-target set Jaccard；
- fixed observed-3/3 targets 在 best 下仍为 2/3 或 3/3 的 retention；
- 两个 policy 各自的 \(N_3/N\) 与 \(R_{3/3}\)。

reversal 冻结定义为：

- missed-set Jaccard 低于 0.50；或
- 3/3-to-at-least-2/3 retention 低于 0.50。

出现任一项即认为 checkpoint selection 改变主要 recurrence support，E0 NO-GO。

### Exploratory E0 gate

只有 fixed-epoch primary 同时满足以下条件，才允许最小 E0：

1. 至少两个 dataset 各有不少于 10 张包含 miss 的 unique images；
2. 所有 miss 事件不由单一 dataset 贡献超过 80%；
3. 至少有 24 个 unique missed targets，可构造按 dataset 和 \(c_k\) 分层的固定 probe panel；
4. failure-support arm 至少一个通过；
5. policy-transition gate 通过；
6. E−1b 至少有两个 eligible LODO folds，且不能同时出现“eligible folds 平均 AUROC 不低于 0.90 且每个 fold 不低于 0.85”；该条件只作项目路线 gate，不作显著性或因果判据；
7. E−1c low-FA bridge 通过。

通过时结论只能是：

> **允许一次最小、受控、from-scratch local-update influence audit。**

它不授权新方法，不授权 solver，也不证明训练信用是根因。

E−1b 样本不足时是 **UNRESOLVED/BLOCKED**，不能按没有解释力处理。若只在一个 dataset 出现足够事件、failure-support 两个 arm 均失败、policy reversal 成立或 low-FA bridge 失败：**Gate E0 NO-GO**。

## 5.9 已冻结的 E−1a 正式结果

以下结果来自 v2 immutable bundles；它们在本 v2.2 amendment 前已可见，因此不能称 outcome-blind preregistration：

- fixed-epoch：493 个 canonical targets、3 seeds、1479 个 target-run statuses；\(N_3=18\)，其中 IRSTD-1K 13、NUDT-SIRST 5、NUAA-SIRST 0；80 个 miss events 中 3/3 targets 贡献 54 个，\(R_{3/3}=0.675\)；
- fixed-epoch failure units：36 个 unique missed targets、35 张包含至少一次 miss 的 images；IRSTD-1K 与 NUDT-SIRST 分别为 17 与 15 张，单一 dataset 的 miss-event share 未超过 80%；
- best-IoU sensitivity：\(N_3=15/493\)，persistent-event share 为 0.5696；
- policy transition：missed-set Jaccard 为 0.6383；fixed 的 18 个 3/3 targets 中有 17 个在 best 下仍至少 2/3 miss，retention 为 0.9444；overall policy-transition gate PASS；
- fixed logit 0 的 overall audit_hungarian achieved FA 为 31.925/Mpix、Pd 为 0.9459；该点明确不是 budget-matched low-FA result。

因此 integrity、failure-support、image/target concentration 与 policy-transition 条件已经通过；**E0 仍为 PAUSED**，剩余阻断是 E−1b 与 v2.2 强化后的 E−1c。

## 5.10 E−1b/E−1c 正式结果与最终路由

E−1b immutable bundle 覆盖 493/493 targets，四个 covariates 的 unavailable rate 均为 0；16 个 targets 的 ring MAD 为 0，按预协议保留 `+1e-6` 分母且只作 QA，没有事后裁剪。primary \(y^{\mathrm{any}}\) 的 eligible LODO folds 为：

| held-out dataset | AUROC | AP | 判定 |
|---|---:|---:|---|
| IRSTD-1K | 0.5304 | 0.2368 | eligible |
| NUAA-SIRST | — | — | 仅 3 张 positive-support images，ineligible |
| NUDT-SIRST | 0.6190 | 0.1815 | eligible |

eligible macro AUROC 为 0.5747，未达到 `mean >=0.90` 且 `each >=0.85`，因此 E−1b PASS；这只表示四个冻结变量没有近乎完全跨域预测 any-seed miss，不证明 image/annotation 难度与失败无关。secondary \(y^{3/3}\) 没有 eligible fold，保持 unresolved；association MLE 的不稳定/分离结果不进入 routing。

E−1c 生成 11832 个 target rows、8064 个 image rows（其中 target-free rows 恰为 \(2\times3\times2\times4=48\)）与 36 个 calibration records。两个 matcher 在本批 selected operating points 的逐 target 状态一致，但由独立 matcher 路径计算。零超调结果为：

| nominal budget \(\alpha\) | 三 seeds、双 matcher 全部可行的数据集数 | eligible datasets | joint 判定 |
|---:|---:|---|---|
| 1 | 0 | — | FAIL |
| 5 | 0 | — | FAIL |
| 10 | 0 | — | FAIL |
| 20 | 1 | IRSTD-1K | FAIL |

在 \(\alpha=20\) 时，IRSTD-1K 三个 seeds 的 achieved FA/Mpix 为 19.6457、14.8773、12.1117；但 NUAA-SIRST seed 20260712 为 24.4850，NUDT-SIRST seed 20260711 为 25.2401，均精确超调。该点虽在唯一 eligible dataset 中有 16 个 joint bridge targets 与 236 个 controls，仍因 `eligible datasets >=2` 失败。更低预算没有 dataset 的三个 seeds 全部可行。

独立 post-run 复核重新计算了 artifact hashes、行数/唯一性、fold disjointness、53 tail quantiles + fixed 0、整数池化、target-free 分母、E−1a Hungarian fixed-status 对齐与 joint gate；未发现能反转结论的实现错误。故最终路由是：

\[
\boxed{\text{E−1c FAIL}\Longrightarrow\text{Gate E0 NO-GO}}
\]

禁止通过增加 held-out 容差、删去失败 seed、改用 best checkpoint、只保留 IRSTD-1K、放宽到单 matcher 或事后更换预算来“救”E0。

---

# 6. Gate E−1 代码与产物

## 6.1 新增模块

    utils/target_identity.py

职责：

- canonical binary target；
- versioned component/full-label SHA256；
- stable target set；
- 跨 seed identity assertions。

    utils/cross_seed_persistence.py

职责：

- 完整 row validation；
- 恰好 3 seeds 与目标集合相等；
- \(N_0\)–\(N_3\)、event count、\(R_{3/3}\)；
- image-cluster bootstrap；
- zero-event 与重复 row fail-closed。

    tools/audit_cross_seed_failure_persistence.py

职责：

- manifest/checkpoint/config/split 审计；
- 一次 run 一次 validation inference；
- 完整逐目标 ledger；
- policy 隔离；
- JSONL/JSON/Markdown/provenance 原子导出；
- 已存在 output directory 时拒绝覆盖。

    tools/compare_gate_e_checkpoint_policies.py

职责：

- 精确核验 fixed/best target universe、seed set 与 assertion metadata；
- 生成 \((c_{\mathrm{fixed}},c_{\mathrm{best}})\) 4×4 转移矩阵；
- 计算 missed-set Jaccard 与 c=3 到 c≥2 retention；
- 按冻结 0.50/0.50 routing threshold 给出 policy-transition gate；
- 原子导出且拒绝覆盖。

    utils/prediction_free_difficulty.py
    tools/audit_gate_e_prediction_free_difficulty.py

职责：严格复现四个 prediction-free covariates；fixed-ledger 三 seed 折叠；association/CR0 与 fixed L2 LODO；缺失、separation、cluster-support、source-data hash 与 immutable E−1b bundle。

    utils/cross_fitted_low_fa.py
    tools/audit_gate_e_low_fa_bridge.py

职责：deterministic two-fold；calibration-only 53-point tail grid；legacy/Hungarian 独立校准；整数零超调；target-free image ledger；fixed0/low-FA target overlap、all-off veto、stable controls 与 immutable E−1c bundle。

Gate E−1 不修改：

- main.py；
- model/MSHNet.py；
- model/loss.py；
- optimizer；
- baseline checkpoint。

## 6.2 输出结构

    repro_runs/gate_e/persistence_v2/
      fixed_epoch/
        target_persistence.jsonl
        target_persistence_summary.json
        target_persistence_summary.md
        provenance.json
      best_iou/
        target_persistence.jsonl
        target_persistence_summary.json
        target_persistence_summary.md
        provenance.json
      policy_transition/
        policy_transition.json
        policy_transition.md
        provenance.json
      prediction_free_difficulty/
        target_difficulty.jsonl
        difficulty_summary.json
        difficulty_summary.md
        provenance.json
      low_fa_bridge/
        target_low_fa.jsonl
        image_low_fa.jsonl
        calibration.json
        low_fa_bridge_summary.json
        low_fa_bridge_summary.md
        provenance.json

provenance 至少包含：

- git HEAD 与 dirty status；
- 完整命令；
- Python、PyTorch、CUDA、NumPy、SciPy、skimage 版本；
- manifest/config/split/checkpoint SHA256；
- checkpoint epoch 与 optimizer metadata；
- resume ledger；
- image/mask resize；
- threshold 语义；
- matcher、radius、connectivity；
- bootstrap seed/replicates；
- output schema version。

单个 target_persistence.jsonl 同时包含每个 seed 的 image-envelope rows（用于证明含 target-free images 的完整 image universe）和完整 target-status rows；target rows 已回填 miss_count、miss seed IDs 与 no-response recurrence，因此不再维护两个可能漂移的 JSONL。

## 6.3 执行命令

fixed-epoch primary：

    $HOME/BasicIRSTD/infrarenet/bin/python \
      tools/audit_cross_seed_failure_persistence.py \
      --batch-id clean_baseline_holdout_v1 \
      --checkpoint-policy fixed_epoch \
      --batch-size 8 \
      --num-workers 2 \
      --device cuda:0 \
      --bootstrap-replicates 2000 \
      --bootstrap-seed 20260712 \
      --output-dir repro_runs/gate_e/persistence_v2/fixed_epoch

只有 primary 完成并冻结后，才运行 best-IoU sensitivity：

    $HOME/BasicIRSTD/infrarenet/bin/python \
      tools/audit_cross_seed_failure_persistence.py \
      --batch-id clean_baseline_holdout_v1 \
      --checkpoint-policy best_iou \
      --batch-size 8 \
      --num-workers 2 \
      --device cuda:0 \
      --bootstrap-replicates 2000 \
      --bootstrap-seed 20260712 \
      --output-dir repro_runs/gate_e/persistence_v2/best_iou

两套 immutable ledger 完成后，运行 policy transition：

    $HOME/BasicIRSTD/infrarenet/bin/python \
      tools/compare_gate_e_checkpoint_policies.py \
      --fixed-dir repro_runs/gate_e/persistence_v2/fixed_epoch \
      --best-dir repro_runs/gate_e/persistence_v2/best_iou \
      --output-dir repro_runs/gate_e/persistence_v2/policy_transition

policy gate 通过后运行 E−1b：

    $HOME/BasicIRSTD/infrarenet/bin/python \
      tools/audit_gate_e_prediction_free_difficulty.py \
      --fixed-dir repro_runs/gate_e/persistence_v2/fixed_epoch \
      --batch-id clean_baseline_holdout_v1 \
      --batch-size 8 \
      --num-workers 2 \
      --output-dir repro_runs/gate_e/persistence_v2/prediction_free_difficulty

随后运行 E−1c；它只读取 fixed-epoch checkpoints：

    $HOME/BasicIRSTD/infrarenet/bin/python \
      tools/audit_gate_e_low_fa_bridge.py \
      --batch-id clean_baseline_holdout_v1 \
      --fixed-dir repro_runs/gate_e/persistence_v2/fixed_epoch \
      --batch-size 8 \
      --num-workers 2 \
      --device cuda:0 \
      --output-dir repro_runs/gate_e/persistence_v2/low_fa_bridge

---

# 7. Gate E0：条件性最小 local-update influence audit

本节只是数学骨架，不是完整预注册协议，也不是当前执行项。即使 E−1 routing gate 通过，仍需在任何 E0 训练前新增一次协议 amendment，冻结 probe/support/control/run-grid/statistics；在该 amendment 完成前 E0 保持 PAUSED。

## 7.1 索引与主量

- \(a\)：固定 validation probe target；
- \(j\)：若以后开放 source attribution，首版固定为当前 training batch 中的一张完整 training image 及其完整标签，不是单个 GT component；
- \(t\)：真实 optimizer step。

绝对 target score influence 仅作 auxiliary。primary 使用冻结 target–hard-background margin：

\[
m_a(\theta,b)
=
s_a^{\mathrm{target}}(\theta,b)
-
s_a^{\mathrm{hard-bg}}(\theta,b).
\]

target 与 hard-bg 支持必须在 E0 训练开始前写入不可变 probe registry，并在所有 step、(B,D,I) 与 finite difference 中完全相同。仅在每个 step 前重新选择不够：那会让不同 step 的 (m_a) 变成不同函数，禁止用于累计或 learned/forgotten 轨迹。

E0 amendment 还必须冻结 target support、hard-background 候选域、GT guard band、像素/组件数、排序 tie-break、支持选择模型与时刻、温度 \(\rho\) 和空 support 的 fail-closed 行为。当前这些值未定义，因此 E0 尚不可执行。

局部分数使用 normalized log-mean-exp：

\[
\operatorname{LME}_\rho(z_{\mathcal S})
=
\rho\log
\left(
\frac{1}{|\mathcal S|}
\sum_{u\in\mathcal S}e^{z_u/\rho}
\right).
\]

原始 log-sum-exp 因含 \(\rho\log|\mathcal S|\) 面积偏置而禁止。

## 7.2 BN buffer 与 parameter effect

一次 train-mode step：

\[
(\theta_t,b_t^-)
\rightarrow
(\theta_t,b_t^+)
\rightarrow
(\theta_{t+1},b_t^+).
\]

必须分别记录：

\[
B_{a,t}
=m_a(\theta_t,b_t^+)-m_a(\theta_t,b_t^-),
\]

\[
D_{a,t}
=m_a(\theta_{t+1},b_t^+)-m_a(\theta_t,b_t^+),
\]

\[
I^m_{a,t}
=
\nabla_\theta m_a(\theta_t,b_t^+)^\top
(\theta_{t+1}-\theta_t).
\]

总 exact change 满足恒等式：

\[
m_a(\theta_{t+1},b_t^+)-m_a(\theta_t,b_t^-)
=B_{a,t}+D_{a,t}.
\]

\(I^m\) 只近似 \(D\)，不能拿它拟合 \(B+D\)。所有 supervision-component gradients 必须来自同一次真实 train-mode forward；禁止为每个 component 重跑 forward 并再次更新 BN buffers。

probe forward 必须：

- 使用 stateless/functional clone；
- 固定 full inference graph；
- eval mode；
- 克隆并冻结 buffer；
- 不改变参数、buffer、RNG 和原 module training flags；
- 从 warm 首步到末步始终使用同一 probe graph。

## 7.3 真实 Adagrad 重构

首版仅支持当前 baseline 的 dense Adagrad：

- lr 0.05；
- lr_decay 0；
- eps \(10^{-10}\)；
- weight_decay 0；
- 无 AMP；
- 无 clipping；
- sparse gradient 直接拒绝；
- None gradient 按真实 optimizer 语义处理并显式记录。

对参数 \(i\)：

\[
u_{i,t}
=
\sum_q w_qg^{(q)}_{i,t},
\qquad
v^+_{i,t}
=
v_{i,t-1}+u_{i,t}^2,
\]

\[
\alpha_{i,t}
=
\frac{\mathrm{lr}_i}
{1+(\mathrm{step}_{i,t}-1)\mathrm{lr\_decay}_i},
\]

\[
\Delta\theta^{\mathrm{math}}_{i,t}
=
-\alpha_{i,t}
\frac{u_{i,t}}
{\sqrt{v^+_{i,t}}+\epsilon_i}.
\]

关键断言：

- denominator 包含当前 total gradient square；
- actual parameter delta 与 math delta 的差异作为 float32 reconstruction residual；
- dot product 与 norm 用 float64 累加；
- 不把逐元素 bitwise 相等当作数学正确性要求。

## 7.4 Baseline loss registry

warm phase：

- 只返回 output0；
- 一个普通 Soft-IoU；
- 权重 1。

post-warm：

- final、side0、side1、side2、side3；
- 五项各权重 0.2；
- 总梯度是五项加权梯度之和。

分量影响必须共享真实 total-gradient 产生的 Adagrad denominator。SLS 中乘法项若再拆，只能称为 gradient-path attribution，不能称为彼此独立的 scalar loss。

## 7.5 Exact effect、Taylor fidelity 与测试

\[
\Delta m_{a,t}^{\mathrm{parameter}}
=
m_a(\theta_{t+1},b_t^+)
-
m_a(\theta_t,b_t^+).
\]

报告：

- absolute residual；
- relative residual；
- sign agreement；
- 每个 probe 的完整分布；
- 不静默删除高误差 hard targets。

实现正确性另用真实方向的单边有限差分：

\[
\frac{
m_a(\theta_t+\varepsilon\Delta\theta_t,b_t^+)
-
m_a(\theta_t,b_t^+)
}{\varepsilon}
\rightarrow
I^m_{a,t},
\qquad
\varepsilon\downarrow0.
\]

ReLU/MaxPool kink 必须标记，不强迫中心差分穿越不光滑点。

E0 amendment 必须再冻结 epsilon 序列、relative-residual denominator 和 zero-neighborhood sign tolerance；当前未定义时不得给 Taylor PASS/FAIL。

## 7.6 LOO、source attribution 与 Shapley

总 influence 通过前，不运行 LOO/Shapley。

以后若开放，source unit 固定为一张 training image。coalition loss 使用原 batch size (B) 作固定 denominator：保留集合 (T) 的 per-image objectives 求和后除以 (B)，空 coalition 的 data-gradient 为 0；不因移除 image 改成除以 (B-1)。这避免把 batch-renormalization 变化混入 source contribution。

令：

\[
P_t=\operatorname{diag}\left[(\sqrt{v_t^+}+\epsilon)^{-1}\right],
\]

并令 (A_t) 是按 parameter group 展开的 Adagrad learning-rate diagonal。若开放，必须使用不同索引：

\[
C^{\mathrm{LOO}}_{a\leftarrow j,t}
=
-\nabla m_a^\top A_tP_t
\left[
g(\mathcal G)-g(\mathcal G\setminus\{j\})
\right].
\]

负号和 group-specific learning rate 不能遗漏。

Shapley 的标量游戏应定义为：

\[
v_a(T)
=
-\nabla m_a^\top A_tP_tg(T),
\]

且：

\[
\sum_j\phi_{a,j}
=
v_a(\mathcal G)-v_a(\varnothing).
\]

必须报告 base term \(v_a(\varnothing)\)。这只是固定真实 preconditioner 下的 gradient game，不是为每个 coalition 重新运行 Adagrad 的真实反事实。

## 7.7 Sampling 口径

若每 100 step 记录一次：

- 名称必须是 sampled-update audit；
- sampled sum 不得命名 cumulative training credit；
- 不声称覆盖每次更新；
- 不用采样和重构最终 margin 变化。

若要全轨迹，只允许小型冻结 probe panel 每 step 记录，并同时累计 BN effect、parameter exact effect 与 Taylor residual。

## 7.8 E0 amendment 必须补齐的实验设计

E−1 通过后、E0 训练前，必须另行冻结：

- datasets：至少两个通过 E−1c low-FA bridge 的 datasets；
- run grid：每个 dataset 至少 3 个新 from-scratch baseline seeds，seed 在结果前写死；
- failed probe panel：来自 E−1c 的 low-FA repeated-miss targets；
- matched control panel：同 dataset 1:1 配对，并按 area、SCR、border distance 的预定义距离匹配；
- panel size、无可用 control 的处理和所有 stable target IDs；
- sampled-update 频率或 every-step 小面板，二者只能选一；
- epoch-end learned/forgotten 状态使用的 matcher、calibration threshold 与 exact component rule；
- primary estimand，例如每个 probe 的负 margin-influence rate及 failed-minus-control paired difference；
- image/probe/seed 层级、cluster rule、CI 与多重比较策略；
- Taylor fidelity、sign tolerance 和 missing-step 的 fail-closed 阈值。

只观察 failed probes 不足以识别 target-specific phenomenon；没有 matched controls 时 E0 NO-GO。

进入方法设计的正向门也必须在 E0 结果前冻结，至少要求：

1. failed-minus-control 的负 margin-influence excess 在至少两个 datasets 方向一致；
2. 三个新 seeds 中不是由单一 seed 主导；
3. BN effect、parameter effect 与 Taylor residual 可分离并达到冻结 fidelity 标准；
4. sampled influence 与 epoch-end exact miss/forgetting 有预定义关联，而不是仅在绝对 target score 上成立；
5. 现象覆盖达到预定义的 low-FA miss event 比例。

本版本尚未冻结这些数值，因此 E0 仍为 PAUSED，不能因 E−1 PASS 自动启动训练。

## 7.9 Gate E0 退出条件

出现任一情况即停止该方向：

- BN effect 与 parameter effect 无法稳定分离；
- actual Adagrad delta 无法重构；
- finite-difference 不收敛且原因不可定位；
- 负向 margin influence 只出现在单一 dataset/seed；
- local influence 不能解释最终 exact miss；
- 现象被 prediction-free image-and-annotation 难度或 checkpoint selection 近乎完全解释；
- failed probes 与 matched controls 没有 amendment 中前瞻冻结的差异。

---

# 8. 方法与新颖性 Gate

Gate E−1/E0 是诊断，不是论文主创新。

当前 closest-work 压力表：

| 本项目构造 | 直接压力 | 当前定位 |
|---|---|---|
| 一阶训练更新对 test probe 的影响 | [TracIn, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html) | 真实 Adagrad delta 提高审计保真度，但未形成新的 influence 原理 |
| never learned / forgotten 轨迹 | [Example Forgetting, ICLR 2019](https://openreview.net/pdf?id=BJlxm30cKm)、[Dataset Cartography, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.746/) | component/fixed-FA 版本可作为领域发现，不自动成为方法创新 |
| 半空间更新投影 | [GEM, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)、[CAGrad, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) | 普通 QP 结构重合 |
| fixed-FA / rate constraint | [Implicit Rate-Constrained Optimization, ICML 2021](https://proceedings.mlr.press/v139/kumar21b.html) | 仅加入 FA 半空间不够新 |
| IRSTD difficulty-aware supervision | [MSHNet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html) 及后续 scale/SCR/hard-negative 方法 | “关注困难目标”或“不增加推理开销”不是 novelty |

这些来源构成新颖性压力，不表示它们与未来 component-aware finite-step 方法完全等价；真正的差异仍需在方法形成后重新检索和证明。

直接风险：

- 一阶 test-point influence 与 TracIn / influence-family 接近；
- never learned / forgotten 与 example forgetting、dataset cartography 接近；
- LOO/Shapley 是归因工具；
- 半空间 QP 与 GEM/OGD/PCGrad/CAGrad 接近；
- fixed-FA 约束与 Neyman–Pearson / rate-constrained optimization 接近；
- IRSTD 已有 scale、SCR、difficulty、hard-negative 与 false-alarm aware loss。

因此以下候选继续隔离：

\[
\min_d
\frac12\|d-d_0\|_{P^{-1}}^2
\quad
\text{s.t.}
\quad
\nabla m_k^\top d\ge-\epsilon_k,
\quad
\nabla R_{\mathrm{FA}}^\top d\le\beta.
\]

当前阻断：

1. \(R_{\mathrm{FA}}\) 尚未定义成与 unmatched-component area 对齐的量；
2. validation probe 不能进入训练；
3. 一阶约束没有 finite-step guarantee；
4. 可行性、slack 和优先级未定义；
5. 普通 QP 本身缺乏 novelty；
6. 尚无跨 backbone 的统一负向 margin evidence。

只有同时出现以下 novelty delta 才重新评审：

- component-aware target/FA bridge；
- 离散 metric 的反例边界或有限步安全保证；
- 非平凡、高效的 dual/active-set 推导；
- 训练 batch 可观测 active set 到 held-out frontier 的桥梁；
- 对直接优化与 IRSTD baselines 的公平优势；
- 外部 Pd–FA frontier 与内部机制同时改善。

---

# 9. 测试清单

## Gate E−1

- strict logit tie；
- 8-connectivity；
- strict 3-pixel centroid radius；
- Hungarian maximum-cardinality/minimum-distance；
- component digest 对 region order 不变；
- 一像素 mask 变化导致 digest 变化；
- label/component envelope drift fail-closed；
- 缺 seed、重复 row、target-set drift fail-closed；
- \(N_0\)–\(N_3\)、event count、\(R_{3/3}\) 手算；
- zero-event 返回 null；
- image-cluster bootstrap deterministic；
- fixed epoch mismatch fail-closed；
- best/fixed policy 不混合；
- existing output directory 拒绝覆盖；
- CPU fixture 可完整生成 ledger。

E−1b 还必须测试：RGB uint8/resize/luminance；`2<=d<9` ring 边界；component-support border distance；ring<16 缺失；不插补 complete-case；target-class-support image cluster；train-only standardization；固定 \(C=1\) objective；complete/quasi separation LP；CR0；AUROC/AP tie；source hash 与原子输出。

E−1c 还必须测试：held-out logits 不进入 threshold grid；deterministic two-fold/shared mapping；53-point tail quantiles；strict tie 与 all-off candidate；legacy/Hungarian trap；fold 整数计数池化而非比率平均；整数预算边界；target-free image inclusion；三 seed/same-budget gate；E−1a Hungarian fixed-status 对齐；fixed0/low-FA overlap；all-off veto；stable controls；immutable provenance。

## 条件性 Gate E0

- probe forward 无状态污染；
- BN buffer/parameter effect 精确分离；
- Adagrad denominator 使用当前 total gradient square；
- warm/post-warm registry 与权重正确；
- float64 dot/norm；
- LOO 符号正确；
- Shapley 加 base term 后重构；
- target/background margin influence；
- 单边有限差分收敛；
- sampled influence 不得标成 full cumulative。

---

# 10. 允许与禁止的论文表述

## E−1 允许

- “In these three completed fixed-budget runs, target \(k\) was missed in 3/3 seeds.”
- “The image-cluster interval quantifies image-sampling uncertainty only.”
- “Best-IoU selection was evaluated as a retrospective sensitivity.”

## E−1 禁止

- “The target is inherently unlearnable.”
- “Persistent miss proves supervision failure.”
- “Seed variation proves gradient competition.”
- “The cross-fitted point is Pd@FA≤α”——除非 achieved held-out FA 对预算零超调；带正容差的结果仍只能称 nominal sensitivity。

## E0 允许

- “At this frozen model state, the actual parameter update had negative first-order influence on the probe margin.”
- “The first-order quantity approximated the parameter-only finite-step effect within the reported error.”

## E0 禁止

- “Training sample \(j\) caused validation target \(a\) to fail.”
- “Target-logit influence is detectability credit.”
- “Sampled influence is total cumulative credit.”
- “A first-order non-decrease guarantees finite-step Pd improvement.”

---

# 11. 当前修改边界与下一步

本 Gate 已完成：

1. 修订 North Star 与 Gate E v2；
2. 实现 target identity；
3. 实现跨 seed persistence aggregation；
4. 实现完整 E−1 audit CLI；
5. 跑定向与全量回归测试；
6. 跑 fixed-epoch primary（已完成）；
7. 冻结结果后跑 best-IoU sensitivity 与 policy transition（已完成）；
8. 跑 E−1b prediction-free alternative（PASS）；
9. 跑 E−1c 双 matcher low-FA bridge（FAIL）；
10. 按前瞻规则关闭 E0，不起草 amendment，不训练。

下一步不是继续修改 Gate E，而是返回 North Star 做方向重置。当前唯一新证据是 **component operating threshold 的 calibration-to-held-out transport 不稳定**；它尚不是创新，也不能直接包装成 calibration loss。允许的新阶段只能是只读 Gate F：

1. 把 E−1c 超调分解为 score-tail shift、component-count/area concentration、fold threshold migration 与 dataset/seed dependence；
2. 检索 risk-control、conformal calibration、low-FPR detection 与 component-level operating-point training 的直接先例；
3. 只有在找到现有方法不能覆盖的机制缺口，并能提出单一原理而非模块堆叠时，才冻结新的方法 Gate；
4. Gate F 之前不启动新的 long-run 训练。

本轮不做：

- 修改 main.py；
- 修改 MSHNet forward；
- 修改 loss；
- 修改 optimizer；
- 使用 baseline checkpoint 继续训练；
- 启动 from-scratch E0；
- 运行 Shapley；
- 实现 metric-constrained solver；
- 宣称形成顶会方法。

最终原则：

\[
\boxed{
\text{先证明稳定的 component-frontier 错位，再设计方法；诊断量永远不能替代外部终点。}
}
\]
