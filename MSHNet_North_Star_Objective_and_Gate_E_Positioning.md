# MSHNet 项目的固定目标、当前验证任务与未来方法边界（审计修订版）

> 文档状态：**North Star v2，2026-07-12 冻结评测口径；Gate F v1 只读诊断已完成，方法仍未授权。**
>
> 这是一份项目宪法和证据门说明，不是论文创新声明，也不是可直接反向传播的训练目标。后续任何 surrogate、诊断量或优化器改动都必须回到这里定义的外部终点接受检验。

## 一句话结论

模型最终应始终朝一个固定的外部目标前进：

> **在给定组件级虚警预算下，检出尽可能多的真实目标实例，同时不牺牲基本的掩膜质量。**

真正需要持续优化的不是某个内部 loss、某个特征指标或固定阈值下的 Pd，而是：

\[
\boxed{
\text{在相同 FA 预算下获得更高 Pd，并使完整 Pd--FA 曲线整体外移}
}
\]

整个研究项目应区分三个层次：

| 层级 | 是否固定 | 作用 |
|---|---:|---|
| 最终评价目标 | 固定 | 决定项目始终朝哪里前进 |
| 当前根因假设 | 可被否证和替换 | 解释 baseline 为什么没有达到目标 |
| 训练目标或优化算法 | 等根因明确后再设计 | 将诊断证据转化为最终方法 |

因此，不是选择一个 loss 后持续让它下降，而是让所有代理目标、诊断指标和方法设计始终服从同一个外部评价目标。

---

# 1. 模型真正的目标应该如何定义

## 1.1 先冻结两套不可混用的匹配口径

本仓库目前同时存在两套组件匹配规则，后续报告必须并列命名，不能把审计口径静默写成官方结果：

| 口径 | 匹配器 | 用途 |
|---|---|---|
| `official_legacy` | `utils.metric.PD_FA` 的 target-order greedy、严格质心距离 `<3`、8-connectivity | 与原始 MSHNet 结果保持可比；固定概率阈值 `0.5` 是当前 `main.py` 的官方点 |
| `audit_hungarian` | 最大匹配数优先、再最小总质心距离的 Hungarian；同样使用严格 `<3` 与 8-connectivity | 消除目标遍历顺序影响，用于逐目标身份、持久性和机制审计 |

两者都把 FA 定义为未匹配预测组件面积占总像素的比例，并以 FA/Mpix 报告：

\[
\mathrm{FA}
=
\frac{\sum_i A_{i,\mathrm{unmatched\ pred}}}
{\sum_i A_{i,\mathrm{image}}}
\times 10^6.
\]

最终方法必须首先在 `official_legacy` 上与论文 baseline 公平比较；`audit_hungarian` 是前瞻冻结的稳健性与机制口径。只有两种口径方向一致，才允许声称结论不是匹配器顺序造成的。

以下统一令 (g\in\{\mathrm{legacy},\mathrm{hungarian}\}) 表示 matcher。由于不同 assignment 会同时改变匹配数与未匹配预测面积，所有 (mathrm{Pd})、(mathrm{FA})、(F) 与 AUC 都必须带 (g)；禁止跨 matcher 共用一个未标注的 frontier 数值。

## 1.2 区分 oracle 曲线与可部署的 cross-fitted 曲线

设模型参数为 \(\theta\)，在阈值 \(\tau\) 下得到二值预测及其连通组件。按照正式评测协议定义：

- \(\mathrm{Pd}_{\theta,g}(\tau)\)：在 matcher (g) 下被成功匹配的 GT 实例比例；
- \(\mathrm{FA}_{\theta,g}(\tau)\)：在 matcher (g) 下未匹配预测组件面积除以图像总面积。

对给定评测集 \(D\) 和虚警预算 \(\alpha\)，同一集合上扫描阈值得到的是描述性的 oracle frontier：

\[
\widehat F_{\theta,D,g}^{\mathrm{oracle}}(\alpha)
=
\max_{\tau\in\mathcal T(D)}
\left\{
\widehat{\mathrm{Pd}}_{\theta,D,g}(\tau):
\widehat{\mathrm{FA}}_{\theta,D,g}(\tau)\le \alpha
\right\}.
\]

它适合描述模型分数的排序上界，但阈值看过同一批标签，不能作为可部署的 primary endpoint。可部署口径必须把阈值选择与评估分开。将数据确定性划分为 \(K\) 个 fold；对第 \(h\) 个 held-out fold，阈值只能在其余数据上选择：

\[
\tau_{-h,g}(\alpha)
\in
\arg\max_{\tau\in\mathcal T(D_{-h})}
\left\{
\widehat{\mathrm{Pd}}_{\theta,D_{-h},g}(\tau):
\widehat{\mathrm{FA}}_{\theta,D_{-h},g}(\tau)\le\alpha
\right\}.
\]

随后只在 \(D_h\) 上应用该阈值，并把各 held-out fold 的匹配数、目标数、未匹配预测面积和像素数先求和再求比值，得到：

\[
\widehat F_{\theta,g}^{\mathrm{CF}}(\alpha)
=
\frac{\sum_h M_g(D_h,\tau_{-h,g}(\alpha))}
{\sum_h N_{\mathrm{GT}}(D_h)},
\]

\[
\widehat{\mathrm{FA}}_{\theta,g}^{\mathrm{CF}}(\alpha)
=
\frac{\sum_h A_{\mathrm{unmatched},g}(D_h,\tau_{-h,g}(\alpha))}
{\sum_h A_{\mathrm{image}}(D_h)}10^6.
\]

cross-fit 实现同时冻结：

- 在每个 dataset 内按 image 划 fold；同一 image 的全部 targets 不拆分；
- fold assignment 在 seed、method 和 backbone 间共享；
- 每个 matcher 各自在 calibration fold 选阈值，得到各自的 frontier；若要隔离 matcher 本身的影响，另报告把 legacy 选出的同一阈值同时应用于两种 matcher 的 shared-threshold sensitivity；
- 阈值网格只由 calibration logits 的冻结 tail-quantiles、固定 logit 0 和最大 logit all-off 候选构成，held-out logits/labels 不参与构造；
- 始终使用 strict score (>\tau)；
- calibration tie-break 依次为：匹配数更多、未匹配预测面积更小、阈值更高；
- cross-fitted 结果是校准程序的 held-out 估计，不是一个在全数据上统一部署的阈值。

这里的 \(\alpha\) 只是 **nominal calibration budget**。held-out 实际 FA 不会自动等于 \(\alpha\)。数学上的预算可行性使用零超调：只有每个 `dataset × seed` 聚合后的 achieved held-out FA 都满足 \(\mathrm{FA}\le\alpha\)，才可写成 `Pd@FA≤α`。若另设工程容差 \(\delta_\alpha>0\)，必须在结果前冻结，并只能写成 nominal-budget sensitivity，不能写成精确预算可行：

```text
cross-fitted Pd at nominal FA budget α
+ achieved held-out FA/Mpix
```

当前已有 Gate D cross-fit 在若干预算上发生 held-out FA 超调，因此它只能作为 nominal-budget 敏感性分析，不能覆盖固定 logit `0` 的主审计口径。

## 1.3 固定的外部目标与当前可估计量

总体研究目标仍然是提高未知数据分布上的低 FA frontier；项目 primary 绑定 (g=\mathrm{legacy})，(g=\mathrm{hungarian}) 作为审计稳健性：

\[
\max_\theta F_{\theta,\mathrm{legacy}}(\alpha).
\]

但实验中只能报告上面明确区分的 `official_legacy`、`audit_hungarian`、oracle 或 cross-fitted 估计量，不能把经验最大值直接写成总体目标已经被优化。

其直观含义是：

> 对当前模型，在虚警不超过 \(\alpha\) 的前提下，通过选择最合适的阈值能够达到的最高 Pd。

模型的外部评价目标可写成：

\[
\boxed{
\max_\theta F_{\theta,\mathrm{legacy}}(\alpha)
}
\]

即：

\[
\boxed{
\max_\theta \mathrm{Pd}_{\mathrm{legacy}}@\mathrm{FA}_{\mathrm{legacy}}\le\alpha
}
\]

但只报告一个 FA 预算容易过拟合单一 operating point。更可靠的外部评价覆盖多个预注册低虚警预算：

\[
\boxed{
\max_\theta
\sum_{\alpha\in\mathcal A}
w_\alpha F_{\theta,\mathrm{legacy}}(\alpha)
}
\]

其中 \(\mathcal A\) 只能用独立 baseline calibration 或领域协议确定并在方法结果产生前冻结。`{1,5,10,20}` FA/Mpix 目前只是候选，不是已经验证可达的正式预算：

\[
\mathcal A=\{1,5,10,20\}\ \text{FA/Mpix}.
\]

连续形式可写为：

\[
\max_\theta
\int_{\alpha_{\min}}^{\alpha_{\max}}
w(\alpha)F_{\theta,\mathrm{legacy}}(\alpha)\,d\log\alpha.
\]

它对应于低虚警区域的 Pd--FA frontier 面积，而不是单个阈值点。该量离散、数据依赖且通常不可微；它是评测终点，不是可以不经论证直接塞入训练的 loss。

---

# 2. 为什么不能只让目标响应不断提高

简单提高目标 logit 并不必然改善模型能力。

假设目标和背景 clutter 的 logit 同时增加相同常数：

\[
z_{\mathrm{target}}'=z_{\mathrm{target}}+1,
\qquad
z_{\mathrm{clutter}}'=z_{\mathrm{clutter}}+1.
\]

此时固定阈值 \(0.5\) 下的 Pd 和 FA 可能变化，但目标与 clutter 的排序没有改变，完整 Pd--FA 曲线也不会真正改善。

在固定 FA 预算 \(\alpha\) 下，更关键的是目标相对于 clutter operating threshold 的余量。但这里必须区分 **连续诊断 margin** 与 **离散组件匹配事件**。

对 validation probe 目标 \(a\)，可在预先冻结的固定支持 \(\mathcal N_a\) 上定义归一化局部分数：

\[
s_a(\theta)
=
\rho\log
\left(
\frac{1}{|\mathcal N_a|}
\sum_{u\in\mathcal N_a}
\exp\frac{z_u(\theta)}{\rho}
\right).
\]

使用 log-mean-exp 而不是原始 log-sum-exp，避免支持面积带来 \(\rho\log|\mathcal N_a|\) 偏置。对由独立 calibration fold 冻结的阈值 \(t_{-h}^{(\alpha)}\)，定义局部 operating-point audit margin：

\[
m_{a,h}^{(\alpha)}
=
s_a-t_{-h}^{(\alpha)}.
\]

还可以在 E0 运行前写入不可变 registry 的 hard-background 支持 \(\mathcal H_a\) 上定义 `target − hard-bg` margin，作为对共同 logit shift 不敏感的主要局部影响量。支持不能只在每个 step 前重选，也不能根据更新后的结果追选。

必须强调：

- \(m_{a,h}^{(\alpha)}>0\) 只表示某个连续局部分数超过冻结阈值；
- 它既不保证阈值后像素形成独立组件，也不保证预测组件质心满足匹配半径；
- \(t_{-h}^{(\alpha)}\) 对模型和 calibration 数据是离散、数据依赖的，不能当作普通可微参数；
- 逐目标成功/失败的 primary truth 始终来自实际阈值化后的 exact component matching。

因此真正需要改善的是离散 component frontier；连续 margin 只是在 Gate E 中检验训练方向是否与它局部一致的桥梁：

\[
\boxed{
\text{真实目标相对于高风险 clutter 的局部排序余量，并最终转化为更多 exact component matches}
}
\]

而不是目标 logit 的绝对大小。

---

# 3. 为什么还需要掩膜质量约束

单独追求组件 Pd 和低 FA 存在退化解：

> 模型可能只在每个 GT 质心附近输出一个像素，从而获得较高 Pd 和极低 FA，但目标形状、IoU 和分割质量很差。

因此，最终目标不能简单写成多个加权损失：

\[
L
=
L_{\mathrm{Pd}}
+\lambda_{\mathrm{FA}}L_{\mathrm{FA}}
+\lambda_{\mathrm{IoU}}L_{\mathrm{IoU}}.
\]

这种形式会重新回到复合 loss 堆叠。

更合理的表达是一个具有明确主次关系的约束优化问题：

\[
\begin{aligned}
\max_\theta\quad&
\sum_{\alpha\in\mathcal A}
w_\alpha F_{\theta,\mathrm{legacy}}(\alpha)
\\
\text{s.t.}\quad&
\mathrm{IoU}_\theta
\ge
\mathrm{IoU}_{\mathrm{baseline}}-\varepsilon,
\\
&
\mathrm{nIoU}_\theta
\ge
\mathrm{nIoU}_{\mathrm{baseline}}-\varepsilon_n,
\\
&
\mathrm{InferenceCost}_\theta
=
\mathrm{InferenceCost}_{\mathrm{baseline}}.
\end{aligned}
\]

即：

1. **主目标：改善组件级 Pd--FA frontier；**
2. **质量约束：IoU 与 nIoU 不允许显著退化；**
3. **部署约束：不增加推理结构和推理成本。**

这不是把多个模块相加，而是对最终方法设置明确的优化优先级和非劣约束。

---

# 4. 项目应该始终朝什么方向前进

整个项目可以固定成下面一句话：

> **在不改变 MSHNet 推理图的前提下，改善跨阈值的组件级 Pd--FA frontier，尤其提高低 FA 区域的目标检出率，同时保持掩膜质量不低于 baseline。**

进一步压缩为：

\[
\boxed{
\text{More matched GT components at the same unmatched-component area.}
}
\]

中文表述为：

\[
\boxed{
\text{在相同未匹配预测面积下，匹配更多真实目标组件。}
}
\]

这应成为所有诊断、理论、算法和实验的统一 North Star。

---

# 5. 哪些量只是诊断，不能成为最终目标

前面几轮研究中出现的内部量都可以用于解释问题，但不能替代最终评价目标。

| 指标或机制 | 正确定位 |
|---|---|
| LLoss | 原始 baseline surrogate |
| mass-normalized centroid | paper-faithful baseline correction |
| OMM mass | 已被数学退化否定的中间构造 |
| bridge/merge | 真实覆盖率较低的错误类型 |
| feature distinct | 无符号特征差异诊断 |
| availability \(A\) | 特征差异强度诊断 |
| head sensitivity \(H\) | 原生 head 响应尺度诊断 |
| utilization \(U\) | 特征差异与 head 方向的对齐诊断 |
| signed margin \(AHU\) | 普通有向 logit contrast |
| local update influence \(I^m_{a,t}\) | 固定模型状态与 probe margin 上的一阶机制诊断，不是因果信用 |
| 固定阈值 Pd | 单一 operating point 指标 |
| Pd--FA frontier | 最终主评价目标 |

最重要的研究纪律是：

> **任何内部量的改善，都必须最终同时接受 official_legacy primary endpoint 与 calibrated audit_hungarian frontier 的检验；否则它只能作为解释变量。**

---

# 6. Gate E 在总目标中的位置

Gate E 不是最终模型目标，也不是已经形成的新方法。

它正在验证：

> **baseline 的训练更新是否系统性地偏离最终 Pd--FA 目标。**

其分析链条是：

\[
\text{训练更新}
\longrightarrow
\text{冻结 target--clutter margin 的局部变化}
\longrightarrow
\text{组件匹配状态}
\longrightarrow
\text{Pd--FA frontier}.
\]

Gate E 具体需要回答：

- 困难目标的冻结 margin 是否在采样更新中反复获得负向局部影响；
- 是否存在 learned-then-forgotten；
- 是否某个监督分量在真实 Adagrad preconditioner 下反复降低困难目标 margin；
- 是否稠密背景梯度压倒了稀疏目标梯度；
- 这些现象是否能够解释大部分最终漏检；
- 同一现象是否跨数据集、跨 seed 一致。

因此 Gate E 的定位是：

\[
\boxed{
\text{寻找 baseline 训练方向与最终 component Pd--FA 目标之间的稳定错位}
}
\]

只有找到这种稳定错位，才有理由继续研究新的训练更新规则；局部一阶影响本身仍不足以证明改变更新会改善 finite-step component frontier。

---

# 7. Gate E−1：跨 seed 失败持久性验证什么

对同一个 GT 实例 \(k\)，在不同随机种子下定义：

\[
m_k^{(r)}
=
\mathbf 1[
\text{target }k\text{ is missed in seed }r
].
\]

跨 seed miss 频率为：

\[
q_k
=
\frac{1}{R}
\sum_{r=1}^{R}m_k^{(r)}.
\]

以三个 seed 为例：

| 类别 | 定义 | 初步解释 |
|---|---:|---|
| observed 0/3 miss | \(q_k=0\) | 在这 3 次运行中均被匹配；不能外推为始终可学习 |
| observed seed-varying | \(q_k=1/3\) 或 \(2/3\) | 在这 3 次运行中状态发生变化；原因尚未识别 |
| observed 3/3 miss | \(q_k=1\) | 在这 3 次运行中重复失败；不能外推为总体“持久” |

身份与状态口径必须先于解释冻结：

- 在 canonical `256×256` 最近邻 resize 标签上使用 8-connectivity；
- 每个 GT 的主身份是 full-size component-mask SHA256；
- bbox、area、centroid 和 component index 只用于一致性断言；
- 三个 seed 的完整 target set 必须完全相同，否则 fail-closed；
- primary 状态是固定 logit `0` 下 `audit_hungarian` 的 matched/unmatched；
- 每个 `dataset × seed × matcher × checkpoint policy` 必须同时报告该固定点的 achieved FA/Mpix；固定 logit 0 只是无标签调阈值的 official-point audit，不自动属于冻结的低 FA 区间；
- `no-response` 只是 unmatched 的次级 subtype；
- fixed-epoch checkpoint 是优化稳定性分析的 primary，best-IoU checkpoint 只能称作 retrospective selected-model sensitivity；
- cross-fitted nominal FA 只有在 achieved held-out FA 零超调、即满足 \(\mathrm{FA}\le\alpha\) 后才升级为预算可行结果；带正容差的结果仍只能称 nominal sensitivity。

Gate E−1 只能估计 **失败复现结构**，不能单独区分输入难度、表示缺陷、监督缺陷、优化竞争、checkpoint 选择或阈值迁移。`3/3 miss` 不证明监督失效，seed flip 也不证明信用分配问题。

它最多给出进入 E0 的候选 probe support。真正开放 E0 前还必须经过两个只读替代解释/外部终点门：

1. E−1b 只用预先冻结的 image-and-annotation covariates，检查跨域 prediction-free 难度是否已近乎完全预测失败；
2. E−1c 在 fixed-epoch checkpoint 上分别按 official_legacy 与 audit_hungarian 做 image-disjoint cross-fit，且 held-out FA 对 nominal budget 零超调。

E−1c 的 target support 必须是真正的桥接交集，而不是“阈值升高后新增的大量 miss”：同一 target 必须在 fixed logit 0 与 low-FA 下、两个 matcher 中都至少 2/3 miss。每个 eligible `dataset × seed × matcher` 必须至少匹配一个 target，以 veto all-off；同时需要足够的稳定成功 controls。否则 E0 只能研究 official-point failure，不能宣称解释 North Star 的低-FA frontier。

正式结果是：E−1b 的两个 eligible LODO AUROC 为 0.5304/0.6190，未达到“prediction-free variables 近乎完全预测 failure”的 NO-GO 门；但 E−1c 在 \(\alpha\in\{1,5,10,20\}\) 均未找到至少两个 datasets 的三个 seeds、两个 matchers 全部零超调的共同预算。\(\alpha=20\) 仅 IRSTD-1K 全部可行，NUAA-SIRST seed 20260712 与 NUDT-SIRST seed 20260711 分别达到 24.4850 与 25.2401 FA/Mpix。因此训练信用路线的 E0 已按协议 NO-GO。

---

# 8. Gate E0：真实训练更新的局部影响验证什么

设第 \(t\) 次更新前后的模型参数分别为：

\[
\theta_t,
\qquad
\theta_{t+1}.
\]

真实参数更新为：

\[
\Delta\theta_t
=
\theta_{t+1}-\theta_t.
\]

必须把两个索引分开：\(a\) 是固定 validation probe target，\(j\) 是当前 training batch 中的 source instance。先定义整个真实 step 对 probe margin 的一阶局部影响：

\[
I^m_{a,t}
=
\nabla_\theta m_a(\theta_t)^\top
\Delta\theta_t.
\]

绝对 target-score influence \(\nabla s_a^\top\Delta\theta_t\) 只能作为辅助；primary 必须是 `target − hard-bg` 或冻结 operating threshold 下的 margin influence，否则共同抬高目标和背景会产生虚假的正“信用”。这里统一使用 **local update influence**，不把它命名为因果训练信用。

MSHNet 含有 BatchNorm。一次 train-mode step 同时改变 buffer \(b\) 和参数 \(\theta\)，因此 exact effect 必须拆为：

\[
B_{a,t}
=
m_a(\theta_t,b_t^+)-m_a(\theta_t,b_t^-),
\]

\[
D_{a,t}
=
m_a(\theta_{t+1},b_t^+)-m_a(\theta_t,b_t^+),
\]

\[
I^m_{a,t}
=
\nabla_\theta m_a(\theta_t,b_t^+)^\top
(\theta_{t+1}-\theta_t).
\]

\(I^m\) 只近似 parameter effect \(D\)，不能拿它拟合总变化 \(B+D\)。probe 始终在 stateless clone 上使用固定完整 inference graph、`eval()` 和冻结 buffer，不能改变参数、buffer、RNG 或原模块 mode。

对当前 dense、无 AMP、无 clipping、无 weight decay 的 Adagrad 首版审计，真实数学步必须按安装版本重构：

\[
u_{i,t}=\sum_q w_qg^{(q)}_{i,t},
\qquad
v^+_{i,t}=v_{i,t-1}+u_{i,t}^2,
\]

\[
\Delta\theta^{\mathrm{math}}_{i,t}
=
-\alpha_{i,t}
\frac{u_{i,t}}{\sqrt{v^+_{i,t}}+\epsilon_i}.
\]

分母必须包含 **当前总梯度平方**。warm phase 只有 `output0`、权重 1；post-warm 是 final 与 side0…3 五项、每项权重 `0.2`。任何分量影响分解都必须在同一个真实 denominator 下进行。

其含义是：

| \(I^m_{a,t}\) | 仅限局部一阶解释 |
|---:|---|
| \(>0\) | 本次参数更新在该状态附近倾向于提高冻结 probe margin |
| \(<0\) | 本次参数更新在该状态附近倾向于降低冻结 probe margin |
| \(\approx0\) | 对该局部 margin 的一阶参数影响很小 |

同时需要计算真实分数变化：

\[
\Delta m_{a,t}^{\mathrm{parameter}}
=
m_a(\theta_{t+1},b_t^+)-m_a(\theta_t,b_t^+),
\]

并检查一阶近似误差：

\[
E_{a,t}
=
\left|
\Delta m_{a,t}^{\mathrm{parameter}}-I^m_{a,t}
\right|.
\]

该误差是 full-step Taylor fidelity，不是实现正确性的充分测试。实现还必须通过沿真实更新方向的单边有限差分收敛测试；所有 dot product 与 norm 使用 float64 累加，并同时报告相对误差与 sign agreement。

Gate E0 要验证：

- 漏检目标是否从未被学会；
- 是否曾被检出但在后期被遗忘；
- 是否 final 与 side supervision 对同一 probe margin 给出相反局部影响；
- 是否 LLoss 或其他分量在同一真实 Adagrad preconditioner 下系统性降低目标 operating-point margin；
- 是否背景相关更新反复对冻结 target–clutter margin 产生负向局部影响；
- 这些训练轨迹是否能够解释最终 miss。

Gate E0 仍然是诊断，不改变模型 forward、loss 或 optimizer。若只每隔若干 step 采样，就必须称为 `sampled-update audit`，不得把采样和写成全训练累计信用。LOO/Shapley 在核心 total influence、BN 分离和 Adagrad 重构通过之前全部推迟。

---

# 9. 未来方法真正应该优化什么（当前隔离，未授权实现）

如果 Gate E 最终证明：

\[
\text{困难目标的 fixed-support margin 在训练中反复受到负向局部影响},
\]

并且该现象能够跨 seed、跨数据集解释足够比例的 exact component miss，那么未来方法的直接目标不应是再增加一个 target loss，而应是：

> **在尽量保留 baseline 更新方向的前提下，阻止关键目标的 fixed-FA margin 被训练更新系统性降低，并限制组件级 FA 风险增长。**

设 baseline Adagrad 更新为：

\[
d_0.
\]

目标 \(k\) 在 FA 预算 \(\alpha\) 下的 margin 为：

\[
m_k^{(\alpha)}.
\]

一个条件候选形式为：

\[
\begin{aligned}
d^*
=
\arg\min_d\quad&
\frac12
\|d-d_0\|_{P^{-1}}^2
\\
\text{s.t.}\quad&
\nabla_\theta m_k^{(\alpha)\top}d
\ge
-\epsilon_k,
\qquad k\in\mathcal H,
\\
&
\nabla_\theta R_{\mathrm{FA}}^\top d
\le
\beta.
\end{aligned}
\]

其中：

- \(d_0\)：原始 MSHNet/Adagrad 更新；
- \(P\)：Adagrad preconditioner；
- \(\mathcal H\)：只由当前 training batch 可观测信息构造的困难目标集合；validation probe 绝不能参与训练；
- \(m_k^{(\alpha)}\)：目标在 fixed-FA operating point 下的 margin；
- \(R_{\mathrm{FA}}\)：尚待定义并证明与离散 unmatched-component area 对齐的训练期风险；普通背景 BCE 或 pixel-FA 不能冒充该量；
- \(d^*\)：满足安全约束后、最接近 baseline 的更新。

这一形式不是：

```python
loss = baseline_loss
loss += lambda_target * target_loss
loss += lambda_fa * fa_loss
loss += lambda_rank * ranking_loss
```

而是：

```python
baseline_update = adagrad_step(baseline_loss)
safe_update = project_to_metric_feasible_set(baseline_update)
apply(safe_update)
```

形式上它属于单一受约束更新规则，而不是模块或 loss 堆叠；但“不是 loss 堆叠”不等于“具有方法新颖性”。当前 QP 与 [GEM](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html) 式半空间投影、[CAGrad](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) 式冲突梯度协调以及 [rate-constrained optimization](https://proceedings.mlr.press/v139/kumar21b.html) 存在直接结构重合，不能作为顶会主创新直接实现和命名。

重新开放方法设计至少需要同时满足：

1. Gate E 证明 primary margin influence 的稳定负向现象，而不是仅有绝对 target logit 变化；
2. 定义 component-aware target/FA bridge，并给出与 exact Pd--FA 事件的反例边界或可验证保证；
3. 用 smoothness remainder、trust region 或 line search 把一阶约束提升为 finite-step 安全条件；
4. 预定义约束不可行时的 slack、优先级和 fail-closed 行为；
5. 给出利用“目标数远小于参数数”的非平凡 dual/active-set 机制，而不是调用通用 QP；
6. 把 GEM/OGD/PCGrad/CAGrad、逐样本约束、rate-constrained optimization 及 IRSTD difficulty-aware loss 纳入直接基线。

在这些条件满足前，该 QP 只保留为 **quarantined design sketch**：不接入 `main.py`，不运行 solver，不启动方法训练。

---

# 10. 判断新方案是否朝正确目标前进的五个问题

以后任何候选方案都应先通过以下检查。

## 10.1 它覆盖了多少真实错误

不能因为一个理论反例漂亮，就投入复杂方法。

bridge 路线已经说明：

- 理论反例成立；
- 但真实账本只解释极少量 miss；
- 因而不适合作为主方法。

`30%–40%` 只能作为项目资源分配的先验启发，不能在看到当前结果后当作统计定律。正式 Gate 必须在结果产生前冻结分母（全部 miss、no-response 或独立 target）、checkpoint policy、数据集聚合方式和最低覆盖率，并同时报告各数据集原始计数。

## 10.2 它改善的是完整曲线还是固定阈值

必须同时比较 official fixed-threshold 结果与预注册的 cross-fitted 结果：

\[
\widehat F_{\theta}^{\mathrm{CF}}(\alpha)
\]

而不是只比较：

\[
\mathrm{Pd}(\tau=0.5).
\]

并报告 achieved held-out FA。如果改变阈值就能得到同样结果，则只是 calibration 或 operating-point migration；若 held-out FA 超过预算，也不能写成 `Pd@FA≤α`。

## 10.3 它是否只是优化已有代理量

下列方向即使有效，也通常不足以形成新的顶会主贡献：

- 普通 logit contrast；
- hard-negative ranking；
- feature cosine；
- instance reweighting；
- focal weighting；
- 单纯 target boost；
- 普通梯度投影。

## 10.4 它是否会退化为已有目标

每个新目标在实现前都应进行：

- 闭式化简；
- 极端样例；
- 退化情况分析；
- 与 MAE、BCE、Dice、ranking 等常见目标的恒等性检查；
- 与正式 component metric 的反例检查。

OMM-2D 的经验说明，数学退化检查必须早于长周期训练。

## 10.5 它是否真正改变 error frontier

最终硬判据应是：

\[
\Delta F_{\theta,\mathrm{legacy}}(\alpha)>0
\]

在多个 \(\alpha\)、多个 seed、多个数据集和多个 backbone 上成立。

---

# 11. 建议固定的项目成功标准

## 11.1 Primary endpoint

每个数据集首先报告原始 `official_legacy` 固定阈值结果，随后报告 `audit_hungarian` 下的 oracle 曲线与 cross-fitted held-out 点。候选 nominal budget 为：

\[
\mathrm{Pd}_{g}@\mathrm{FA}_{g}
\in
\{1,5,10,20\}\ \text{/Mpix},
\]

具体预算只能在独立 baseline calibration 上确认可达区间后、在任何方法结果产生前预注册。每个点必须同时给出 nominal budget 和 achieved held-out FA/Mpix；发生 held-out FA 超调的预算标记 infeasible/misaligned，不强行插值成成功点。

同时报告低 FA 区间的：

\[
\mathrm{AUC}_{\mathrm{Pd-FA}},
\]

建议在冻结的正数区间 \([\alpha_{\min},\alpha_{\max}]\) 上，以预注册阈值网格、单调 frontier envelope 和梯形规则对 **achieved FA** 的 \(\log\) 坐标积分；nominal budget 不能充当横坐标。FA=0 的处理、端点外推、重复 FA tie-break 和 matcher (g) 必须在看结果前定义，避免 AUC 被实现细节操纵。

## 11.2 Secondary non-regression constraints

要求：

\[
\Delta\mathrm{IoU}\ge-\varepsilon,
\qquad
\Delta\mathrm{nIoU}\ge-\varepsilon_n.
\]

容忍范围应在训练前根据独立 baseline 重复运行与领域可接受差异预注册，不能在结果出来后选择。非劣结论必须使用 paired method–baseline 运行和明确的 non-inferiority rule；仅凭均值没有下降不构成非劣证明。

## 11.3 Robustness requirements

最终方法证据包至少要求：

- 三个数据集方向一致；
- 每个 seed 使用 paired baseline/method 初始条件并报告逐 seed effect；
- 当前 3 个 seed 只足以做工程与方向性 pilot，image-cluster CI 不覆盖 seed/domain uncertainty；若要作显著性或非劣声明，应预先做功效/精度设计并增加独立训练重复；
- 至少两个额外 backbone；
- 冻结阈值网格上的完整 sweep，并同时报告 official 与 audit matcher；
- 不新增推理参数、算子或分支；训练期开销单独报告；
- pristine baseline 与诊断/方法分支严格隔离。

## 11.4 Mechanism endpoint

方法必须真正改善它声称解决的机制指标。

若声称解决负向 margin influence，应同步观察到：

- 负向 primary margin influence 比例下降；
- learned-then-forgotten 减少；
- observed 3/3 miss 减少；
- fixed-FA margin 提高；
- Pd--FA frontier 同时改善。

若只有内部指标改善而外部 frontier 不变，则机制不能成立为方法贡献。

---

# 12. 当前阶段的固定目标与当前任务

## 永久不变的研究目标

> **提高 MSHNet 在低组件虚警预算下的目标实例检出率，并使这种改善跨阈值、跨 seed、跨数据集和跨 backbone 稳定成立。**

数学上：

\[
\boxed{
\max_\theta
\sum_{\alpha\in\mathcal A}
w_\alpha
\mathrm{Pd}_{\mathrm{legacy}}@\mathrm{FA}_{\mathrm{legacy}}\le\alpha
}
\]

并满足：

- mask-quality non-regression；
- inference-graph invariance（不新增推理参数、算子或分支；训练期开销单独审计）。

## 当前阶段任务

> **固定 epoch baseline 的 exact component miss 存在跨 seed 支撑，但它没有通过双 matcher、零超调的低-FA桥接；因此不再执行训练信用 E0。Gate F v1 已证明当前现象既不是单一 hard-image 机制，也不足以形成通用 calibration 创新。下一步只能先用强 risk-control baseline 证伪剩余的 topology-and-matching 特有问题，不能直接设计新 loss。**

当前执行顺序应为：

\[
\boxed{
\text{E−1a recurrence PASS}
\longrightarrow
\text{E−1b alternative PASS}
\longrightarrow
\text{E−1c low-FA bridge FAIL}
\longrightarrow
\text{停止训练信用路线}
\longrightarrow
\text{Gate F v1：只读分解与 prior art（完成）}
\longrightarrow
\text{Gate F0：通用 risk-control falsification}
\longrightarrow
\text{条件性方法设计}
}
\]

Gate E 已在进入 margin-influence 之前因低-FA bridge 失败而终止。不得通过新增 loss、模块、普通梯度投影、事后容差或单数据集结果维持该路线。

## 12.1 Gate F v1：operating-point generalization 分解

Gate F v1 只读取冻结的 E−1c bundle，不重新推理、不加载 checkpoint，也不改变正式 gate。工具会先后校验 E−1c 的五个 artifact hashes、完整 calibration curve、原始非单调 tie-break、fold mapping、逐图整数计数、fold aggregate 与双折池化 aggregate。独立输出位于：

```text
$HOME/DEA/repro_runs/gate_f/operating_transport_v1
```

生成的数据契约为：

- 36 个原始 calibration records；
- 8064 个原始 image rows；
- 144 个 `dataset × seed × matcher × budget × held-out fold` transport rows；
- 72 个 `dataset × seed × matcher × budget` 双折池化 rows；
- official legacy 与 Hungarian 仍独立复算；本 bundle 的 72/72 fold groups 与 36/36 pooled groups 恰好一致只是经验结果，不能在实现中合并 matcher。

按 matcher 折叠后，结果为：

| nominal FA/Mpix | 正式池化失败的 dataset-seeds | held-out 超调 folds | 移除 pooled top-1 后翻转的失败数 | 两折阈值差中位数 / 最大值 |
|---:|---:|---:|---:|---:|
| 1 | 7/9 | 10/18 | 2/7 | 13.17 / 96.69 |
| 5 | 6/9 | 8/18 | 5/6 | 7.87 / 78.90 |
| 10 | 6/9 | 11/18 | 3/6 | 15.17 / 34.79 |
| 20 | 2/9 | 6/18 | 2/2 | 9.83 / 55.26 |

这里的 top-1 removal 是看过结果后的 concentration sensitivity，只用于判断机制，不允许删除图像、修改 denominator 或重判 E−1c。

在 α=20 时，两个正式失败分别是：

| dataset / seed | achieved FA/Mpix | pooled 非零 FA 图像 | top-1 图像与面积 | top-1 share | leave-one-image-out FA/Mpix |
|---|---:|---:|---|---:|---:|
| NUAA-SIRST / 20260712 | 24.4850 | 5 | `Misc_229`, 37 px | 0.5362 | 11.6257 |
| NUDT-SIRST / 20260711 | 25.2401 | 21 | `000982`, 67 px | 0.3045 | 17.6863 |

二者都会被单张图像翻转，但不能归为同一种机制：NUAA 是明显集中尾部，NUDT 是中度集中叠加较小预算裕量；在 α=1/10 时，大部分失败也不能由 top-1 解释。因此：

\[
\boxed{\text{统一 hard-image tail hypothesis：FAIL}}
\]

另一个稳定事实是，36/36 matcher-specific calibration curves 都出现“阈值升高、未匹配组件面积反而局部增加”，最大单步增加 93 pixels。这说明 matching 后的 component risk 不能用单调 ROC crossing、插值或二分搜索代替完整候选曲线。但它还不证明存在新的学习机制：当前阈值选择器只是贴着经验 FA 上界最大化匹配数，没有有限样本置信控制，held-out 超调本身可能是普通抽样波动。

本实验的 folds 来自同一 dataset，且每个 dataset 分别校准。故本阶段只能称 **calibration-to-held-out operating-point generalization**；在真正执行跨数据集、跨传感器或跨场景阈值迁移前，禁止使用 cross-domain、cross-environment transport claim。

## 12.2 Gate F prior-art NO-GO map

以下区分“论文直接提供的事实”和“本项目据此作出的路由判断”：

| 直接先例 | 已覆盖内容 | 对本项目的路由结论 |
|---|---|---|
| [Learn then Test](https://arxiv.org/abs/2110.01052) | 在有限候选/超参数上用 hypothesis testing 做风险控制选择 | 把现有 threshold grid 接上通用检验不是主创新 |
| [Conformal Risk Control](https://research.google/pubs/conformal-risk-control/) | 对单调风险族给出分布无关的期望风险控制框架 | 单纯把 FA 换成 conformal risk 属于应用 |
| [Conformal prediction with limited false positives](https://proceedings.mlr.press/v162/fisch22a.html) | 直接研究 false-positive 数量控制 | “控制虚警”本身不构成 gap |
| [SeqCRC for object detection](https://arxiv.org/abs/2505.24038) | 已把 matching、空图像与置信阈值纳入检测风险控制 | matching-aware calibration 不能单独声称新颖 |
| [Conformal Risk Control with Non-Monotonic Loss Functions](https://arxiv.org/abs/2602.20151) | 直接处理非单调、多维风险控制 | “component FA 非单调”本身仍不是算法贡献 |
| [Rethinking Evaluation Metrics of IRSTD](https://papers.nips.cc/paper_files/paper/2025/hash/a81051ae2c8b1e46bd51480917b8ab84-Abstract-Datasets_and_Benchmarks_Track.html) | 已研究 IRSTD matching、错误类型和跨数据集评价；阈值设计被留在其范围外 | 仍有 operating-point 研究空间，但单做评价 taxonomy 高度拥挤 |
| [Deep partial AUC optimization](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ca7998666c2e53cc1e882b7268414d8a-Abstract-Conference.html) | 已直接优化低 FPR 区间的排序 | 普通 pAUC、DRO 或 rank consistency 不是新原理 |
| [SLS/MSHNet](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html)、[REEM](https://openaccess.thecvf.com/content/CVPR2026W/PBVS/html/Sevim_SCR-Guided_Difficulty-Aware_Optimization_for_Infrared_Small_Target_Detection_CVPRW_2026_paper.html) | scale/location sensitivity 与 SCR difficulty weighting 已直接用于 MSHNet | 不能回到面积、对比度、难度加权或训练 loss 堆叠 |

因此当前四条直接路线均关闭：

1. generic empirical threshold calibration：NO-GO；
2. generic CRC/LTT 在 IRSTD 上的直接应用：NO-GO；
3. SCR/area/contrast/hard-negative/logit-margin loss stacking：NO-GO；
4. 当前三模型种子规模的 benchmark/negative-result paper：NO-GO。

唯一尚未被现有证据完全覆盖的窄问题是：

> **对由 score superlevel-set 连通性变化与一对一 matching 共同诱导的非单调 component risk，能否利用其临界事件结构，获得比 black-box LTT/CRC 更紧、且不会退化为 all-off 的有限样本 `Pd@FA≤α` 控制？**

这只是待证伪的问题，不是方法声明。它必须同时提供新的结构刻画、理论界或 solver；如果只是 threshold grid、置信上界、pAUC、GroupDRO、rank consistency、CFAR 或已有 risk-control 的领域替换，则立即 NO-GO。

## 12.3 下一门：Gate F0 generic risk-control falsification

方法设计前只允许再做一个无训练 gate：

1. 在 fixed-epoch development holdout 上生成逐图、全阈值的 component outcome cache；同一 logits 只做一次推理，两种 matcher 独立计算，official test 继续封存；
2. 先实现 current empirical selector、LTT 与适用的 CRC/non-monotonic risk-control 强 baseline，明确区分各自控制的是期望风险、高概率风险还是经验风险；
3. 所有方法必须报告 achieved FA、Pd、all-off rate、置信水平、校准样本量和阈值稳定性，禁止只报 nominal budget；
4. 如果通用方法已在至少两个 datasets、三个 seeds、两个 matchers 上给出非 all-off 的预算控制，则结构化新方法路线 NO-GO；
5. 只有当通用方法稳定地 vacuous/过度保守，并能证明原因来自 component topology/matching 而不是样本量不足时，才允许冻结 structure-aware theorem/solver Gate；
6. Gate F0 前后都不修改 MSHNet forward、loss、optimizer，也不启动训练。

当前状态：

\[
\boxed{
\text{Gate F v1 diagnosis PASS}
\;\land\;
\text{generic calibration novelty NO-GO}
\;\land\;
\text{method authorization = NO}
}
\]

---

# 13. 最终结论

需要一直坚持的是外部评价目标，而不是内部代理目标。

## 应始终不变的目标

\[
\boxed{
\text{提升 component-level Pd--FA frontier}
}
\]

## 可以随证据改变的内容

- 根因假设；
- loss 形式；
- 优化算法；
- 诊断指标；
- 方法名称；
- 训练期优化规则与 surrogate。

当前 North Star v2 固定“不新增推理参数、算子或分支”。任何推理结构变化都必须显式升级 North Star 版本，并作为另一条项目路线重新建立 baseline、成本和证据门，不能在本路线中静默改变。

前几轮路线被依次否定并不意味着项目失去方向：

- OMM 因数学退化停止；
- CCSR 因真实覆盖率过低停止；
- Evidence Utilization 因无法唯一归因且会退化为普通 contrast 停止；
- 经验阈值 calibration 因不能满足 cross-fitted 零超调、且缺少风险保证而停止。

这些路线都围绕同一个外部目标进行证伪。

最终应坚持：

\[
\boxed{
\text{始终朝 Pd--FA frontier 前进；所有内部 surrogate、诊断和方法都必须服从该目标，并允许被证据替换。}
}
\]
