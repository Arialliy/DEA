# DEA 更新仓库严格复核与 AAAI 模型路线决策

> 审计锁定点：`af4a9200dd1f0974dbb2269a35068362c87c6552`（2026-07-12）  
> 基线：MSHNet（CVPR 2024）  
> 目标：在低组件虚警预算下提升目标实例检出率，同时保持 IoU/nIoU，不把诊断现象包装成模型贡献。

---

## 0. 执行结论

更新后的仓库没有给出一个可以直接冻结、训练并投稿的模型；相反，它进一步排除了两条路线：

1. **通用 operating-point risk control：NO-GO**。即使假设单个候选阈值预先固定且校准经验虚警为零，当前图像样本量仍不足以对 `1–20 FA/Mpix` 给出非平凡的 distribution-free 认证。
2. **Q0/Q1/Q2 嵌套阈值网格与 full unique-logit curve：NO-GO**。网格细化确实频繁改变所选阈值，但没有形成跨数据集、跨 seed、双 matcher 一致的实质增益；因此 targeted/full exact sweep 未获授权。

因此，当前严格判定为：

| 路线 | 判定 | 说明 |
|---|---|---|
| OHR-MSHNet | **FINAL NO-GO** | 概率语义不成立，且与实际 MSHNet 推理图和既有失败证据不一致 |
| Dynamic fusion / scale gate / reliability head | **NO-GO** | Gate D 已否定 final fusion suppression 是统一根因 |
| SLS + ranking + FA + topology | **PERMANENT NO-GO** | 复合 loss 堆叠，且已有直接先例压力 |
| Generic LTT / CRC / conformal thresholding | **NO-GO** | rare-event 样本量下界导致非平凡认证不可行 |
| Event graph + full unique-logit solver | **NO-GO** | 约 6560 万 image-local score events，且不能突破单候选信息下界 |
| DER / Integrated DEA | **HOLD，不能升为主方法** | 结构上不是简单堆叠，但 action identifiability 尚未解决，已有 hard-route collapse 证据 |
| Predictive-Correction MSHNet | **CONDITIONAL RESEARCH CANDIDATE** | 是仓库中唯一较像“单一算法替换”而非模块叠加的结构，但尚无机制覆盖率、稳定收益和新颖性放行 |
| Gate G0 表示级失败分解 | **GO** | 当前唯一被证据支持的下一步；只读、development-only、不训练 |

**当前仓库仍没有创新性足够的顶会主方法。** 这不是悲观判断，而是该仓库最新审计文档本身的正式结论。

---

## 1. 本次更新到底增加了什么

最新提交 `af4a920` 修改 12 个文件，增加约 4181 行，核心新增内容是：

- `MSHNet_Gate_F_Event_Complete_Component_Risk_Analysis.md`
- `utils/risk_control_feasibility.py`
- `utils/nested_component_grid.py`
- `tools/audit_gate_f0_risk_control_feasibility.py`
- `tools/audit_gate_f_nested_grid_sensitivity.py`
- 对应的四组单元测试与协议回放测试
- README 对当前研究状态、历史失败路线和 baseline 纯净配置的重新定位

这次更新主要是**否证与协议基础设施更新**，不是网络模型更新。

### 1.1 Gate F0 的关键结果

对每图 bounded false-area loss

\[
\ell_\tau=\frac{A_{\mathrm{unmatched}}}{65536}\in[0,1],
\qquad
\alpha=\frac{b}{10^6},
\]

在最乐观条件——候选预先固定、经验 loss 为零、置信失效概率 \(\delta=0.1\)——下，HB-LTT 单候选所需图像数为：

| FA/Mpix | 最少图像数 | 当前最大 calibration 图像数 |
|---:|---:|---:|
| 1 | 2,302,584 | 82 |
| 5 | 460,516 | 82 |
| 10 | 230,258 | 82 |
| 20 | 115,129 | 82 |

标准 unit-bound CRC 的 `1/(n+1)` correction floor 对应至少需要：

- FA=1：999,999 张图；
- FA=5：199,999 张图；
- FA=10：99,999 张图；
- FA=20：49,999 张图。

这说明问题不是候选阈值数量太多，也不是 solver 不够聪明；在 candidate complexity 等于 1 时，信息已经不足。把候选改为 component event、max-tree event 或 matching transition，不能从同一批 82 张图中创造缺失的 rare-event 信息。

### 1.2 Q0/Q1/Q2 嵌套网格的关键结果

更新实现了严格嵌套的三个 calibration-only 网格：

- `Q0`：原 53 个尾部分位点加固定 logit 0，最多 54 个阈值；
- `Q1`：在相邻 quantile probability 间插入一次中点，最多 106 个阈值；
- `Q2`：再次细化，最多 210 个阈值。

Q0 对冻结 E−1c 的 calibration selection、逐图计数、逐目标状态和 pooled aggregate 做了逐字段 replay。Q0→Q2 在四个预算上频繁改变 selected threshold，但没有达到预注册的跨数据集 material-gain 门：

| FA/Mpix | 观察到的 feasibility flip | 结论 |
|---:|---|---|
| 1 | IRSTD-1K / seed 20260711：pass→fail | FAIL |
| 5 | IRSTD-1K / seed 20260713：pass→fail | FAIL |
| 10 | NUAA-SIRST / seed 20260712：fail→pass | FAIL |
| 20 | NUDT-SIRST / seed 20260713：pass→fail | FAIL |

唯一的 fail→pass 只出现在一个 dataset-seed。网格敏感性真实存在，但不是跨域、跨 seed 的统一模型机制。因此 full/targeted unique-logit sweep 被停止。

---

## 2. OHR 为什么仍然是 FINAL NO-GO

仓库更新没有修复 OHR 的三个根本问题。

### 2.1 Side sigmoid 不是校准的块占用概率

MSHNet 的 side outputs 使用 SLS/soft-IoU 与 location term 监督。该目标不是 proper scoring rule，也没有使

\[
\sigma(z_s)
\]

成为某个粗网格内“至少有一个目标”的校准概率。因此，基于该解释建立 hazard conservation 或 noisy-OR 等式没有统计语义基础。

### 2.2 Baseline 从未执行 noisy-OR

实际推理链是：

\[
\{z_0,z_1,z_2,z_3\}
\xrightarrow{\text{bilinear upsample}}
\operatorname{concat}
\xrightarrow{\operatorname{Conv}_{3\times3}^{4\to1}}
z_f.
\]

上采样后的四个值只是 final convolution 的输入特征，不被当作四个独立 Bernoulli 事件再计算 noisy-OR。因此“0.9 插值到四个像素后 noisy-OR 变成 0.9999”不是 MSHNet 的实际计算反例。

### 2.3 既有失败审计不支持 fusion suppression 根因

现有 checkpoint 证据已经表明，最终融合并不是大多数 miss 的统一原因。更新后的 Gate F 文档也再次明确：不得仅凭 side/final 结构直觉重新设计 dynamic fusion、scale gate、attention、target boost 或 reliability head。

因此：

\[
\boxed{\text{OHR 不实现、不训练、不进入消融表、不保留为候选主方法。}}
\]

---

## 3. 当前 MSHNet 真正已知的结构事实

Canonical MSHNet 的推理图是：

\[
\text{Encoder}
\rightarrow
\text{four-stage decoder}
\rightarrow
\{z_0,z_1,z_2,z_3\}
\rightarrow
3\times3\;\text{fusion}
\rightarrow z_f.
\]

训练时 final 与四个 side outputs 各计算一次 SLS，并等权平均；推理时只使用 final output。由此可以提出“训练—推理出口不一致”的结构疑问，但现有证据只允许把它称作**待检验现象**，不能直接推导出 dynamic fusion 方法。

当前缺少的不是另一个模块，而是对 low-FA miss 的表示级类型进行互斥分解：模型究竟是选错 operating point、目标排序本来就低于 clutter，还是目标峰存在但无法转化成合法组件匹配。

---

## 4. 唯一合理的下一门：Gate G0

Gate G0 使用冻结的 Q2 candidate family，在 development-only 数据上做 post-hoc 表示级失败分解。它看过 held-out labels，因此只用于选择研究方向，不能当作部署性能。

对每个低-FA miss target，互斥分类为：

### A. `calibration-selection-sensitive`

存在 pooled-budget-feasible 的阈值对可以 exact-match 该目标，但 cross-fitted selector 选中的阈值没有匹配。

含义：主要问题在 operating-point selection，而不是网络表示。若该类占主导，**不应设计新网络**。

### B. `peak-order-limited`

不存在任何 budget-feasible 阈值，使冻结 target support 的 peak 严格超过阈值。

含义：目标响应在低-FA operating neighborhood 内始终排在 clutter tail 之下。只有该类具有稳定、跨数据集覆盖率时，才有理由设计表示/排序机制。

### C. `component-conversion-limited`

存在 active target peak，但所有 budget-feasible 阈值都无法形成满足连通组件与质心 `<3 px` 匹配规则的预测。

含义：问题在从局部响应到合法组件的几何转化，而不是简单 target-score 不足。只有该类稳定占主导，才有理由研究几何/状态演化机制。

必须强调：target-wise oracle 不能相加成一个实际可部署的 Pd；该分类只用于根因路由。

### 建议现在就冻结的覆盖门

以下数值是项目的**前瞻资源分配规则**，不是统计定理：

1. 同一类别在至少两个数据集上成立；
2. 每个数据集至少 2/3 seeds 同向；
3. official legacy 与 audit Hungarian 均成立；
4. 每个符合条件的数据集内，该类别至少覆盖 1/3 的 repeated low-FA misses；
5. 结果不能由单个 image、单个 target size bin 或单一预算主导。

未达到该门：不允许命名模型，不允许长训练。

---

## 5. 模型路线的严格决策树

```text
Gate G0
├── calibration-selection-sensitive 占主导
│   └── 模型路线 NO-GO；转向协议/数据，不伪装成网络创新
│
├── peak-order-limited 通过覆盖门
│   └── 开放“表示排序”模型门 G1-P
│       ├── 禁止：focal、OHEM、普通 hard-negative、pAUC、logit margin 堆叠
│       └── 需要：一个改变表示生成原理的单一结构机制
│
├── component-conversion-limited 通过覆盖门
│   └── 开放“组件形成”模型门 G1-C
│       ├── 禁止：topology loss + SLS + FA 的复合堆叠
│       └── 需要：一个直接改变状态演化/组件形成原理的单一机制
│
└── 无类别通过覆盖门
    └── 当前 MSHNet 线不具备可靠顶会模型动机，停止继续试模块
```

---

## 6. 现有 Predictive-Correction 分支应如何定位

`model/predictive_correction_mshnet.py` 用一个共享 coarse-to-fine 状态替换完整 decoder：每个尺度把 encoder feature 映射为 observation，由同一个有界预测算子 \(K\) 预测 observation，再用参数严格共享的伴随算子 \(K^*\) 回投 robust residual：

\[
r_s = y_s-Kh_s,
\qquad
h_{s+1}=\operatorname{up}(h_s)+\eta K^*\psi_\delta(r_s).
\]

它没有 per-scale decoder blocks、side heads 或 terminal scale fusion，因此从结构形式看，确实比“MSHNet + attention/gate/head”更接近单一算法替换。

但是当前只能判为 **conditional research candidate**，原因是：

1. 它改变推理图，违反 North Star v2 的 inference-graph invariance；
2. 现有 Gate G0 尚未证明 miss 主要属于 peak-order-limited 或 component-conversion-limited；
3. 当前仓库没有提供跨数据集、跨 seed 的 paired improvement；
4. “共享预测—残差—伴随回投”与 predictive coding、learned iterative inference、deep unrolling、adjoint/backprojection 网络存在明显先例压力；
5. 没有机制证据时直接训练，只会成为另一次大结构试错。

若必须走“尽快形成模型”的独立路线，应明确创建 **North Star v3-Architecture**，不能在 v2 中静默启用该分支。其最小放行条件应是：

- Gate G0 中 peak-order-limited 或 component-conversion-limited 通过覆盖门；
- 明确说明 predictive correction 对应哪一种失败类型；
- 先做 matched-capacity control，而不是只和原 MSHNet 比；
- 逐尺度报告 residual energy、target/clutter ordering 与 exact component conversion；
- 在 development split 上完成 paired baseline/candidate，official test 保持封存；
- 未通过最小门立即停止，不再叠加 gate、attention 或额外 loss 挽救。

---

## 7. DER / Integrated DEA 为什么不能作为当前主方案

DER 的优点是同一 tri-state route 同时控制 decoder correction 与 terminal sign-monotone correction，形式上不是互不相关模块的简单相加；并且初始化可以嵌入 baseline。

但其核心语义仍未识别：

- 无 action supervision 时，target/increase action 未成为 hard winner；
- residual-action control 曾收敛到 100% keep/abstain；
- 这说明 route 可以保持 baseline，却未证明三个 action 表示了可重复的 target/clutter 决策。

因此 DER 只能保留在 `archive/experimental`，不能因为设计漂亮就升为 AAAI 主模型。

---

## 8. 仓库当前应立即整理的工程问题

### 8.1 README 已改善，但目录仍需隔离

建议公开主目录只保留：

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
  gate_g/
archive/
  dea/
  der/
  predictive_correction/
  omm/
  ccsr/
  negative_controls/
```

否则审稿人会看到大量失败变体与开关，容易将最终方案理解为事后筛选。

### 8.2 `main.py` 的模型别名需要清理

当前 `dea` 与 `predictive_correction` 等历史别名容易混淆正式方法身份。正式 baseline、诊断控制和候选模型应使用互斥名称，并在 checkpoint metadata 中写入完整结构语义。

### 8.3 OHR 文档必须标记为撤回

原 OHR 文档建议在首行加入：

```text
STATUS: RETRACTED / FINAL NO-GO
Reason: invalid probabilistic occupancy interpretation and mismatch with actual MSHNet inference/evidence.
```

避免后续协作者误实现。

---

## 9. 对 AAAI 路线的最终建议

### 正式主线

先完成 Gate G0。它是现有九 checkpoint 与 Q2 outcome 上的只读分解，不需要新训练，也不会浪费一个长周期实验。结果将决定是否存在值得设计模型的统一失败类型。

### 模型主线

只有在 Gate G0 通过覆盖门后，才从以下两条中选择**一条**：

- `G1-P`：表示排序机制；
- `G1-C`：组件形成机制。

不能同时做两条，也不能把两条再与 attention、gate、topology loss、FA loss 拼接。

### 当前冻结判断

\[
\boxed{
\begin{aligned}
&\text{OHR：FINAL NO-GO}\\
&\text{Risk-control / exact-event：NO-GO}\\
&\text{DER：HOLD}\\
&\text{Predictive Correction：CONDITIONAL，仅作为独立 v3 候选}\\
&\text{Gate G0：GO，且是下一步唯一主任务}
\end{aligned}}
\]

现在最危险的不是“模型还没命名”，而是在没有表示级覆盖证据时再命名一个模型。更新后的仓库已经明确证明：继续凭结构直觉改 fusion、上采样、gate 或 loss，失败概率最高。

---

## 10. 审计依据

本文件依据以下仓库内容复核：

- commit `af4a9200dd1f0974dbb2269a35068362c87c6552`
- `model/MSHNet.py`
- `model/loss.py`
- `model/predictive_correction_mshnet.py`
- `MSHNet_North_Star_Objective_and_Gate_E_Positioning.md`
- `MSHNet_Gate_D_NoGo_and_Gate_E_Training_Credit_Audit_Plan.md`
- `MSHNet_Gate_F_Event_Complete_Component_Risk_Analysis.md`
- `utils/risk_control_feasibility.py`
- `utils/nested_component_grid.py`
- `README.md`

