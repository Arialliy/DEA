# MSHNet 顶会主线继续分析：Gate F、精确组件事件曲线与创新性判断（审计修订版）

## 0. 核心结论

> 文档状态：**2026-07-12，经 Gate F0 rare-event 下界和九个冻结 checkpoint 的事件规模复核后修订。本文不再把 full unique-logit curve 作为当前必做项，也不再据此重开 risk-control 主线。**

基于当前代码、冻结 E−1c bundle 和已完成的 Gate F0，结论如下：

> **现在不应继续设计 MSHNet 网络模块或复合 loss。54 点分位网格确实不是 event-complete，但 full exact curve 只能审计经验网格离散化，不能突破 Gate F0 对 generic distribution-free HB-LTT / unit-bound CRC 所证明的 rare-event 信息下界。当前先执行严格嵌套的 Q0/Q1/Q2 calibration-grid sensitivity；只有出现预注册的跨数据集、跨 seed、双 matcher 实质翻转，才对受影响区间做 targeted unique-logit sweep。**

但必须明确：

\[
\boxed{\text{当前仓库还没有形成创新性足够的顶会主方法。}}
\]

仓库已经形成了较强的否证、协议和诊断基础设施。它的价值在于排除了 OMM、CCSR、Evidence-to-Decision Utilization、final-fusion suppression、简单 calibration、训练信用等多条错误路线。

Gate F0 已经在 optimistic iid bounded-loss assumptions 下、比任何事件搜索都更有利的情形——**一个预设候选且 calibration empirical loss 为零**——证明 generic distribution-free HB-LTT / unit-bound CRC 用当前样本量无法认证 1--20 FA/Mpix。因此，减少或重排 candidate complexity 最多影响多重选择代价，不能消除该设定下的单候选信息下界；该结论不排除未来引入额外、可验证结构假设的方法。

当前最合理的推进顺序是：

\[
\boxed{
\text{Q0/Q1/Q2 nested-grid sensitivity}
\rightarrow
\text{targeted exact interval（仅在触发时）}
\rightarrow
\text{表示级失败分解}
\rightarrow
\text{Gate G 方向判定}
}
\]

这一路径只产生 `alternative-grid sensitivity`，不会追溯修改冻结的 E−1c formal decision。Generic LTT/CRC 已由 Gate F0 判定 NO-GO，不再重复生成全候选 outcome cache。

---

# 1. 当前 MSHNet baseline 的准确代码语义

## 1.1 网络结构

当前 `model/MSHNet.py` 的 canonical 路径可概括为：

\[
\text{Encoder}
\rightarrow
\text{Decoder}
\rightarrow
\{z_0,z_1,z_2,z_3\}
\rightarrow
\operatorname{Conv}_{3\times3}^{4\to1}
\rightarrow z_f.
\]

主要结构包括：

- 五级通道：\(16,32,64,128,256\)；
- 四个 decoder side head，均为 `1×1 Conv → 1 channel`；
- side logits 上采样到输入分辨率；
- 四路 side logits 拼接后，由 `Conv2d(4,1,3,padding=1)` 融合；
- `decidability_head` 只应在 `enable_dea_lite=True` 时构造；
- pristine MSHNet 的正式 baseline 不应携带 DEA 参数或诊断 head。

这意味着，canonical baseline 目前在推理结构上已经比较干净。

## 1.2 原始训练目标

canonical MSHNet 仍然对 final output 和四个 side outputs 分别计算 SLS，并等权平均：

\[
L_{\mathrm{MSHNet}}
=
\frac{
L(z_f,Y)+\sum_{s=0}^{3}L(z_s,Y_s)
}{5}.
\]

这意味着：

- 训练阶段存在五个监督出口；
- 推理阶段只使用 final output；
- side supervision 与 final fusion 之间存在训练—推理语义差异。

但是，Gate D 已经否定了“final fusion 系统性吞噬已有 side evidence”这一统一根因，因此不能仅凭结构直觉重新设计：

- dynamic fusion；
- scale gate；
- attention；
- target boost；
- side reliability head。

这些设计都缺乏当前机制证据，而且容易重新成为模块堆叠。

## 1.3 正式组件指标

正式 legacy evaluator 的核心语义为：

- 8-connected components；
- target 按 `regionprops` 顺序处理；
- 每个 target 选择最近的未匹配 prediction；
- 质心距离严格小于 3 px 才合法；
- FA 是未匹配预测组件的总面积；
- threshold 语义为：

\[
B_\tau(x)=\mathbf 1[z(x)>\tau].
\]

仓库同时实现了 Hungarian 审计版本，其目标为：

1. 最大化匹配组件数量；
2. 在匹配数相同条件下最小化总质心距离。

因此项目真正的外部目标是：

\[
\max_{\tau}\mathrm{Pd}(\tau)
\quad
\text{s.t.}
\quad
\mathrm{FA}(\tau)\le\alpha.
\]

而不是单独优化：

- pixel IoU；
- target logit；
- feature distinct；
- utilization；
- 某个内部 loss。

---

# 2. 当前 Gate F 的核心问题

## 2.1 当前阈值曲线不是全阈值曲线

当前 `DEFAULT_TAIL_QUANTILES` 由以下部分组成：

- 4 个显式分位点：`0.5, 0.75, 0.85, 1.0`；
- 49 个高尾分位点，由 `1 - geomspace(1e-1,1e-7,49)` 生成；
- 再加固定 logit threshold `0.0`。

因此最多只评估约 54 个不同阈值。

阈值来自 pooled calibration logits 的线性分位数，而不是所有能够改变以下状态的真实 score event：

- 二值 mask；
- 组件连通性；
- 组件面积；
- 组件质心；
- 合法匹配边；
- Hungarian assignment；
- unmatched component area。

当前 E−1c 在有限分位数网格中选择：

1. matched components 最多；
2. unmatched prediction area 最小；
3. threshold 最高。

因此，一个 alternative-grid sensitivity 中观察到的结果差异可能同时受三类因素影响：

\[
\boxed{
\text{observed failure}
=
\text{grid discretization}
+
\text{empirical selection}
+
\text{calibration generalization}
}
\]

这里既不是严格的加性统计恒等式，也不能从一次 threshold flip 中唯一识别三者的因果贡献。特别地，稀疏网格可能改变被选择的 threshold，但 held-out 超调仍然是 calibration-to-held-out generalization 事件；不能因为存在未采样阈值，就把超调归因于 discretization。

## 2.2 为什么稀疏分位数网格会误导结论

两个相邻分位数阈值之间可能发生：

1. 目标组件面积缩小，但仍保持匹配；
2. 组件质心跨过 3 px 合法边界；
3. clutter component 消失；
4. 一个组件发生 split；
5. Hungarian assignment 改变；
6. 某个组件从 matched 变为 unmatched；
7. 某个目标从 miss 变为 hit；
8. unmatched area 出现局部回升。

如果只观察区间两端，就可能：

- 漏掉合法的低-FA、高-Pd operating point；
- 把多个正负事件合并成一次回升；
- 错判最优 threshold；
- 错判 curve non-monotonicity 的幅度；
- 改变 alternative-grid 下的 held-out feasibility sensitivity。

因此，当前 Gate F v1 是对**冻结 54 点协议**的有效正式诊断，而不是全阈值 oracle。更密网格可以检查该结论对候选族的敏感性，但只能命名为 `alternative-grid sensitivity`；即使出现翻转，也不能回写或重判 E−1c，更不能直接作为新方法动机。

---

# 3. Event-Complete Component Curve 的数学定义与用途边界

下述 exact-state 定义在数学上成立，适合作为小图 correctness oracle 或被触发区间的 targeted sweep。它不等于“当前必须在全部真实 logits 上物化完整曲线”。

## 3.1 精确定义

对图像 \(i\) 的 logit map，设不同分数为：

\[
\nu_{i,1}>\nu_{i,2}>\cdots>\nu_{i,m_i}.
\]

由于 threshold 采用严格大于：

\[
B_i(\tau)=\mathbf 1[z_i>\tau],
\]

二值 mask 只会在阈值跨越某个实际 logit value 时改变。

精确状态应至少包括：

- \(\tau=\max z\) 的 all-off 状态；
- 每个相等分数组整体激活后的状态；
- 必要时包括最低分数以下的 all-on 状态。

相同分数的像素必须原子激活，不能逐像素激活。否则会构造实际 threshold 永远无法产生的虚假中间 mask。

推荐实现：

```python
threshold_after_group = np.nextafter(
    score_value,
    -np.inf,
    dtype=score_dtype,
)
```

或者在 event sweep 内直接记录状态，不强行物化 threshold。

## 3.2 不能只枚举 max-tree 的 birth/merge 事件

仅从连通拓扑看，重要事件似乎只有：

- component birth；
- component merge；
- component disappearance；
- split 的反向事件。

但正式 metric 还依赖：

\[
\operatorname{area}(C),
\qquad
\mu(C),
\qquad
\|\mu(C)-\mu(G_k)\|_2<3.
\]

即使拓扑完全不变，仅向已有组件加入一个边界像素，也可能：

- 改变面积；
- 改变质心；
- 使合法匹配边出现或消失；
- 改变最近预测组件；
- 改变 Hungarian assignment；
- 改变 unmatched component area。

因此：

\[
\boxed{
\text{topology-complete}
\neq
\text{metric-event-complete}
}
\]

任何 event compression 都必须证明，被跳过的像素激活不会改变：

1. 组件支持域；
2. 组件面积；
3. 组件质心；
4. 3 px 合法边；
5. 一对一 assignment；
6. unmatched component area。

没有这一证明，不能把 component-tree critical events 当成精确曲线。

---

# 4. 修订后的工程可行性与代码范围

当前阶段仍不得修改：

- `model/MSHNet.py`；
- `model/loss.py`；
- `main.py` 的正式训练路径；
- optimizer 或 checkpoint。

但也不应直接实现全数据 `ComponentEventState` 物化。

## 4.1 E−1c 没有 frozen logit bundle

冻结 E−1c 目录只包含：

```text
target_low_fa.jsonl
image_low_fa.jsonl
calibration.json
low_fa_bridge_summary.json
low_fa_bridge_summary.md
provenance.json
```

logits 只在 `collect_job_predictions` 内存中生成，传给 `cross_fit_job` 后即被丢弃。因此，“读取 frozen E−1c logit bundle”不是当前仓库中存在的操作。任何新 grid 或 exact-event 审计都必须：

1. 从九个 frozen fixed-epoch checkpoint 重新推理；
2. 再次校验 checkpoint、split、target registry 与 E−1c artifact hashes；
3. 写成新的 sensitivity bundle；
4. 明确 official test 仍封存。

## 4.2 实测 unique-event 规模否定全量物化

对九个 checkpoint 的 1008 个 development checkpoint--image forward pairs（336 张 registry images × 3 seeds）做只读复核后：

| 数量 | 实测值 |
|---|---:|
| 总像素分数 | 66,060,288 |
| \(\sum_j|\mathrm{unique}(z_j)|\)：image-local distinct float32 score groups | 65,600,032 |
| distinct-event / pixel 比例 | 99.303% |
| naive matcher-event evaluations \(2\sum_jm_j\) | 约 131,200,064 |

以上数字当前来自不落盘的只读 preflight measurement；正式引用前必须由 nested-grid bundle 连同 checkpoint、registry、source 与 artifact hashes 重新固化。preflight 只决定工程路由，不构成独立实验结果。

这里统计的是 image-local score groups 之和，不是跨图像 union 后的全局 unique threshold 数。它仍说明当前 final logits 几乎处处唯一。即使使用 union-find，正式 unmatched-area 在一个未匹配组件增加一个像素时也会改变，所以 metric-changing updates 仍可能是 \(O(P)\)，不能用少量 birth/merge 事件代替。

若每个 image-local 状态只用 96 bytes 的紧凑二进制记录，单 matcher 也约为 6.30 GB（5.86 GiB），双 matcher 约为 11.73 GiB；Python dataclass/JSON 会显著更大。逐 event 重新调用 `measure.label/regionprops` 或完整 Hungarian 更不可行。故：

\[
\boxed{\text{full unique-logit curve materialization：NO-GO}}
\]

小图 brute-force 仍可作为实现 correctness oracle，但不能冒充真实 256×256 production solver。

## 4.3 当前先实现 Q0/Q1/Q2 嵌套网格

新增只读 sensitivity，而不是 full event curve：

```text
utils/nested_component_grid.py
tools/audit_gate_f_nested_grid_sensitivity.py
```

候选族只由 calibration logits 构造：

- `Q0`：当前 53 个 `DEFAULT_TAIL_QUANTILES` 加 fixed logit `0`，最多 54 thresholds；
- `Q1`：在 Q0 相邻 quantile probabilities 之间插入算术中点，共 105 probabilities，加 `0` 后最多 106 thresholds；
- `Q2`：对 Q1 再做一次同样细分，共 209 probabilities，加 `0` 后最多 210 thresholds；
- probability ties 或 quantile threshold 与 `0` 重复时，实际 threshold 数可以更少；
- 每个 matcher 只计算一次 Q2 superset curve，Q0/Q1 按 exact threshold membership 投影；
- 三层共享 strict `logit > threshold`、calibration maximum all-off candidate、整数预算判定与原 tie-break；
- zero-overshoot 始终用 `unmatched_area*1_000_000 <= budget*total_pixels` 判定；
- Q0 必须逐字段重放现有 E−1c calibration selections、image counts 和 target status，否则 fail closed。

Q1/Q2 selected thresholds 只应用一次到 held-out fold。报告：

- calibration matched count、Pd 与 FA；
- selected tuple 是否改变；
- held-out pooled unmatched pixels、Pd 与 zero-overshoot feasibility；
- Q0→Q1、Q1→Q2、Q0→Q2 的方向；
- matcher-specific 结果；
- `alternative-grid sensitivity flip`，而不是 `formal failure changed`。

## 4.4 targeted exact sweep 的触发门

只有满足以下任一预注册条件，才允许在**受影响 fold 的受影响 Q0 区间**枚举 unique scores：

1. 同一 budget 下，至少两个 dataset-seed、覆盖至少两个 datasets，Q0→Q2 在 legacy 与 Hungarian 中出现同向 zero-overshoot feasibility flip，且非 all-off；
2. 同一 budget 下，至少两个 datasets 各有至少 2/3 seeds，在两个 matcher 中都满足 Q0→Q2 `Δmatched_count≥3` 且 `ΔPd≥5 percentage points`。

未触发时：

\[
\boxed{\text{targeted exact sweep = NO-GO，full exact sweep 同时 NO-GO}}
\]

触发后也只证明原候选网格存在协议敏感性；它不重判 E−1c，不重开 distribution-free risk-control 主线。

## 4.5 correctness tests

嵌套网格至少验证：

```text
test_q0_exactly_reproduces_default_quantiles
test_q0_is_subset_of_q1_and_q1_is_subset_of_q2
test_equal_score_quantiles_remain_atomic
test_exact_integer_budget_selection
test_selection_tie_break_matches_e1c
test_q0_replays_frozen_e1c_records
test_grid_projection_matches_direct_evaluation
test_bundle_is_atomic_and_refuses_overwrite
```

若后续触发 targeted event implementation，再在 `8×8`/`16×16` 上补充 equal-score、centroid `<3` boundary、legacy/Hungarian assignment、split/merge、all-off/all-on 与 byte-determinism exhaustive tests。

---

# 5. 统计口径需要进一步修正

## 5.1 FA 与 Pd 的随机单位不同

当前定义为：

\[
\mathrm{FA}
=
\frac{
\sum_i A_i^{\mathrm{unmatched}}
}{
\sum_i H_iW_i
}
\times10^6,
\]

\[
\mathrm{Pd}
=
\frac{
\sum_i M_i
}{
\sum_i K_i
}.
\]

因此：

- FA 是按图像面积加权的 unmatched-area ratio；
- Pd 是按 GT 实例数加权的 ratio-of-sums；
- empty-target images 对 FA 有贡献，但对 Pd denominator 无贡献。

未来做风险保证时，不能把所有像素视为独立样本，也不能把 pooled Pd 当成普通 image-mean loss。

更严格的 population 量是：

\[
R_{\mathrm{FA}}(\tau)
=
\mathbb E\left[
\frac{A^{\mathrm{unmatched}}(\tau)}{HW}
\right],
\]

而 Pd 应写成 ratio-of-expectations：

\[
U_{\mathrm{Pd}}(\tau)
=
\frac{
\mathbb E[M(\tau)]
}{
\mathbb E[K]
}.
\]

若要构造 Pd 下置信界，需要 ratio-of-means 方法，不能直接套单均值 Hoeffding bound。

## 5.2 当前 two-fold cross-fit 只是诊断

两折互相校准并池化 out-of-fold 结果，可以估计 operating-point generalization，但不能自动产生一个可部署的最终 threshold certificate。

最终方法必须明确：

- candidate 用哪一部分数据生成；
- certificate 用哪一部分数据计算；
- 最终 threshold 如何使用全部 calibration 数据；
- candidate family 是否依赖 calibration labels；
- 是否有独立 proposal/certification split；
- cross-fitting 是否有严格定理支持。

不能把“held-out achieved FA 较稳定”直接写成有限样本保证。

---

# 6. “Component-Event Risk Control” 当前不成立

该名称暂不启用。它在形式上确实不是网络模块或 compound loss，但“不是堆叠”不等于“可认证”或“有主创新”。Gate F0 已给出更强的必要条件：对每图 bounded loss

\[
\ell_\tau=\frac{A_{\mathrm{unmatched}}}{65536}\in[0,1],
\qquad
\alpha=\frac{b}{10^6},
\]

即使候选 \(\tau\) 预先固定且 calibration 上观测到零 loss，HB-LTT 在 \(\delta=0.1\) 时仍至少需要：

| FA/Mpix | single-candidate 所需图像数 | 当前最大 calibration n |
|---:|---:|---:|
| 1 | 2,302,584 | 82 |
| 5 | 460,516 | 82 |
| 10 | 230,258 | 82 |
| 20 | 115,129 | 82 |

标准 unit-bound CRC 的 correction floor 在四个预算下也分别需要 999,999、199,999、99,999、49,999 张图。该下界在 candidate complexity 为 1 时已经成立，因此 event compression、assignment transition count 或更高效 solver 都不能从当前样本中创造缺失的 rare-event 信息。

任何未来 risk-control 路线必须先提供**额外、可验证的信息来源**，例如新的独立样本单位、确定性小 loss bound 或明确的分布结构假设，并重新证明其有效性。仅把 threshold family 换成 metric events 不满足这一条件。

---

# 7. 为什么“应用 risk control”仍然不够新

已有研究已经覆盖：

- Learn-Then-Test；
- Conformal Risk Control；
- 非单调、多维损失的风险控制；
- 有限候选网格上的 multiple testing；
- 目标检测中的 threshold 与定位风险控制；
- 低 FPR 区域 partial-AUC 优化。

因此，下列方案创新性不足：

- MSHNet + LTT；
- MSHNet + CRC；
- component metric + conformal；
- event graph + 通用 multiple testing；
- 低 FA partial-AUC；
- 将分位数阈值换成 unique logits；
- 给 non-monotonic curve 加 UCB。

这些可以作为协议修正或强 baseline，但不能单独成为顶会主创新。

---

# 8. 顶会方法仍需什么：先增加信息，而不是改写 candidate complexity

Gate F0 已否定“只靠 event structure 获得非平凡 distribution-free certificate”的当前版本。未来若重新提出理论路线，至少必须同时回答四个问题。

## 8.1 新信息从哪里来

必须明确超出当前 82 张 calibration images 的可识别信息，例如：

- 可证明有效的独立或 block-independent observation units；
- 预测前成立的确定性 component-area upper bound；
- 经独立数据验证的 random-field/point-process 结构假设；
- 额外 calibration 数据，而不是把同一图像的 pixels/peaks 伪装成 iid 样本。

没有这一项，任何结构复杂度界都会受单候选 Bernoulli lower bound 限制。

## 8.2 结构量必须控制风险估计误差，而不只是 solver 速度

`component birth count`、`upward variation`、`assignment changes` 可以帮助计算或描述曲线，但只有在定理中真正缩小风险估计误差、且不违反 lower bound 时才有统计价值。把 \(|\mathcal E|\) 换名为 \(B_{\mathrm{effective}}\) 不构成新界。

## 8.3 表示级机制必须有真实覆盖率

拓扑持久性、component-tree、Hungarian metric loss 和 black-box solver differentiation 都有强先例。若 exact/sensitivity audit 最终只是发现目标峰低于背景尾部，则它仍可能退化为 hard-negative、pAUC 或 CFAR；若 component conversion 只覆盖少量 miss，则 topology-changing frontier 也不能成为主方法。

任何机制必须先在至少两个 datasets、三个 seeds、两个 matcher 的实际 operating neighborhood 中达到预注册覆盖门，才允许进入最小训练原型。

## 8.4 外部终点不变

即使未来获得新的理论或表示机制，最终仍必须证明：

- cross-fitted achieved `Pd@FA≤α` 改善；
- 非 all-off；
- IoU/nIoU non-regression；
- 多 seed、数据集与 backbone 稳定；
- 不改变当前 MSHNet inference graph。

---

# 9. 创新性最终判断

| 路线 | 当前判断 |
|---|---|
| 增加 attention/gate/head | 永久 NO-GO |
| side/final dynamic fusion | 机制依据不足，NO-GO |
| SLS + ranking + FA + topology | loss 堆叠，永久 NO-GO |
| OMM / CCSR / utilization | 已否证，永久 NO-GO |
| 当前分位数 Gate F | 诊断工具，不是方法 |
| Q0/Q1/Q2 嵌套网格 | 当前应做的低成本 sensitivity，不是方法 |
| full unique-logit 精确曲线 | 全量物化 NO-GO；仅在 gate 触发后做 targeted interval |
| Generic LTT/CRC on component FA | Gate F0 因 rare-event sample size NO-GO |
| Event graph + generic LTT | 创新性不足 |
| 仅靠结构复杂度理论 + 新 solver 获得 certificate | 被单候选下界否定 |
| 带新可验证信息假设的 component-risk 理论 | 未立项，需重新过 novelty 与 identifiability gate |
| 多 backbone、多 connected-component 任务泛化 | 可能达到顶会口径 |

综合评价：

\[
\boxed{
\text{当前创新性属于“强诊断工作”，尚未达到完整顶会方法。}
}
\]

若未来在新增可验证信息假设后完成“结构定理 + 非平凡风险界 + solver + 非 all-off 实证”，则：

- AAAI：理论与优化算法扎实时较匹配；
- CVPR/ICCV：还需多个 backbone，最好再覆盖一个依赖 connected-component decision 的视觉任务；
- 仅在三个 IRSTD 数据集上做阈值校准，通常不足以支撑 CVPR/ICCV 主方法。

---

# 10. 仓库工程结构建议

公开仓库当前仍保留多代 DEA、OMM、CCSR 与诊断路线。为了避免审稿人将最终方法理解为从大量失败分支中事后筛选，应重构目录和叙事。

建议结构：

```text
model/
  MSHNet.py
  loss.py

training/
  train_pristine_mshnet.py

evaluation/
  official_metric.py
  component_operating_point.py
  nested_component_grid.py

diagnostics/
  gate_d/
  gate_e/
  gate_f/

archive/
  dea/
  omm/
  ccsr/
  negative_controls/
```

冻结 canonical 配置：

```yaml
model_type: mshnet
mshnet_objective: sls
mshnet_side_supervision: canonical
mshnet_train_graph: canonical_warm
location_loss: legacy
lambda_location: 1.0
dea_lambda_single: 0.0
dea_lambda_dec: 0.0
dea_lambda_empty: 0.0
```

README 应同步改为：

1. pristine MSHNet baseline；
2. diagnosis gates；
3. archived negative directions；
4. nested-grid sensitivity 与 targeted exact protocol；
5. Gate G representation-level direction search。

不要继续把 DEA-lite 当作仓库主身份。

---

# 11. GO / NO-GO 执行顺序

## Gate F−1a：嵌套网格 sensitivity

必须先完成：

- Q0 对 frozen E−1c 的逐字段 replay；
- Q0/Q1/Q2 calibration-only candidate construction；
- legacy/Hungarian 独立 selection；
- selected threshold 的 held-out 一次应用；
- pooled integer counts、target ledger、artifact hashes 与 byte-deterministic summary。

### GO 条件

- 同一 budget 下至少两个 dataset-seeds、覆盖至少两个 datasets，在两个 matcher 中出现同向、非 all-off 的 Q0→Q2 feasibility flip；或
- 至少两个 datasets 各有 2/3 seeds，在两个 matcher 中同时满足 Q0→Q2 `Δmatched_count≥3` 与 `ΔPd≥5 percentage points`。

### NO-GO 条件

- Q1/Q2 只改变 threshold 数值而不产生上述实质变化；
- 改善只在单 dataset、单 seed 或单 matcher 出现；
- 所谓改善来自 all-off；
- Q0 replay 与 frozen E−1c 不一致。

GO 只授权 targeted exact interval；不授权方法、训练或 risk theorem。

## Gate F−1b：targeted unique-logit interval

只有 Gate F−1a 通过时才执行：

- 只枚举触发结果所在 Q0 相邻区间；
- calibration scores 原子分组；
- streaming 保留各预算 best tuple，不物化完整曲线；
- small-image brute-force 与 incremental state 逐状态一致；
- held-out 仍只应用 calibration 选出的 threshold。

若 targeted sweep 不能稳定复现 Q2 的方向，或 Q2 已经收敛到同一 selected tuple，则 full exact route NO-GO。

## Gate G0：表示级失败分解

在冻结 Q2 family 上做 development-only post-hoc diagnosis，并显式标注看过 held-out labels。互斥分类为：

1. `calibration-selection-sensitive`：存在 pooled-budget-feasible threshold pair 可 exact-match，但 cross-fitted selected point miss；
2. `peak-order-limited`：不存在 feasible pair 使固定 target support peak 严格超过 threshold；
3. `component-conversion-limited`：存在 active target peak，但所有 feasible pair 都无法 exact-match。

target-wise oracle 不能相加成一个可实现 Pd。该诊断只用于选择 Gate G 方向，不得写成 deployable/generalization claim。

## 已关闭：Generic risk control / structure-only theorem

Gate F0 已完成，无需再跑 Bonferroni/Holm LTT、generic CRC 或 finite-grid non-monotonic CRC 的 outcome cache。没有额外可验证信息假设时，F1/F2 式结构复杂度定理和 certificate solver 均不开放。

---

# 12. 最终建议

当前最正确的研究路线不是继续给 MSHNet 增加网络模块，而是：

> **保持 MSHNet 完全不变，先用 Q0/Q1/Q2 判断 54 点候选族是否产生实质协议敏感性；只有 sensitivity gate 被跨数据集、跨 seed、双 matcher 触发，才对局部区间做 exact sweep。无论结果如何，它都不能突破 Gate F0 的 rare-event 样本下界；下一主线仍必须回到有真实覆盖率、且不退化为 topology/hard-negative/CFAR 的表示级机制。**

最终纪律是：

1. 不再新增 head、gate、attention 或 dynamic fusion；
2. 不再构造 `SLS + ranking + FA + topology`；
3. 不物化 6560 万 unique-score states；
4. 先完成 Q0/Q1/Q2 sensitivity，并把任何 flip 写成 `alternative-grid sensitivity`；
5. 未触发 gate 时停止 exact curve；触发时只扫受影响区间；
6. Generic LTT/CRC 与 structure-only certificate 保持 NO-GO；
7. 在新的表示级机制通过先例、覆盖率与可证伪门之前，不命名方法、不训练。
