# MSHNet Decision-Conversion Gate 与 Evidence Utilization 研究方案

> **2026-07-12 执行校正。** 原稿在第 4.3 节把 ε 加进余弦分母，却在第 4.4 节继续声称精确恒等式；两者不相容。当前实现与后文统一改为：仅当 \(A H>0\) 时定义无 ε 的 \(U\)，退化情形返回 undefined。fixed-FA 改为图像级交叉拟合与 Hungarian component matching，禁止用背景像素分位数替代 component FA。failure class 只保留为待证的多轴诊断，不再强制单标签归因。

## 1. 当前结论

现有三数据集 component ledger、多尺度证据审计与九组 decision-conversion audit 支持三个明确决定：

1. **CCSR / component-cut 路线永久归档为 NO-GO。**
2. **当前主要矛盾不是 bridge/merge，而是目标特征差异没有稳定转化为最终检测决策。**
3. **该转换失败并不集中在单一 utilization 或 final-fusion 机制，standalone Evidence-to-Decision Utilization 已在 Gate D1 判为 NO-GO。**

早期单轮三数据集固定阈值账本中，共计 493 个 GT、27 个 miss，只有 2 个 miss 可以明确归因于 merge/bridge。九组完整账本的统计口径见第 15 节。因此，围绕 split、vertex-cut 或 component-tree 构造主方法，会把最复杂的机制投入最次要的错误来源。

同时，多尺度诊断显示：71 个 final no-response 目标中，65 个在四个原始 side 输出上也没有阈值以上的邻近响应。但后续只读 feature-survival pilot 又发现，IRSTD-1K seed-20260711 的 18 个 no-response 中，17 个在 `d0` 特征上仍与同形状局部背景平移控制可区分。

这说明：

> side output 在固定阈值下缺失，不等价于 latent feature 信息彻底消失。

更准确的研究问题应改写为：

> **对于 no-response 目标，网络内部是否仍保留可用特征差异；如果保留，原生线性 readout 为什么没有将其转化为稳定、正向、可排序的最终分数？**

因此，下一阶段进入：

# Decision-Conversion Gate：可用特征差异如何转化为最终决策

---

## 2. 必须修正 feature-survival 的解释口径

当前 `evaluate_feature_survival()` 使用无符号差异分数：

\[
S=
\sqrt{
\frac1C\sum_c
\left(
\frac{\mu^+_c-\mu^-_c}{\sigma_c}
\right)^2
}.
\]

由于通道差值被平方，`state="distinct"` 只能说明目标位置与背景控制存在差异，不能说明差异方向对检测有利。

以下情况都会得到较高的 unsigned distinct score：

- 目标特征高于背景；
- 目标特征低于背景；
- 不同通道方向冲突；
- 差异与实际 side-head 权重方向正交。

因此，目前最严格的结论应写为：

> IRSTD-1K seed-20260711 中，18 个 no-response 目标有 17 个在 `d0` 保留了相对于几何平移背景控制的显著特征偏离；但该偏离可能是正向目标证据、反向编码、与输出头正交的异常性，或局部背景结构差异。

不能直接写成：

> 17/18 的目标语义在 `d0` 中仍然存活。

后者需要有方向的原生 readout 分析或严格跨图像 probe 支持。

---

## 3. 当前平移控制的局限

现有 control policy 保证：

- 控制区域与 GT 形状一致；
- 位于目标保护区外；
- 平移距离满足预设范围；
- 控制选择确定、可复现。

但它没有按背景难度匹配局部上下文。目标位置 rank 较高可能有两种解释：

1. 网络确实保留目标可检测信息；
2. 目标所在背景本身比随机平移位置更特殊。

因此，后续应并行保留两类 control。

### 3.1 Geometry control

保留现有同形状随机平移控制，用于检测一般位置异常。

### 3.2 Context-matched control

仅使用目标 footprint 外部背景 ring 构造上下文描述量：

\[
r(q)=
[
\operatorname{median},
\operatorname{MAD},
\text{gradient energy},
\text{Laplacian energy},
\text{local entropy}
].
\]

在候选背景位置中选择：

\[
q^*
=
\arg\min_{q\in\mathcal Q}
\|r(q)-r(G_k)\|_{\Sigma^{-1}}.
\]

必须满足：

- 匹配过程不能读取目标 footprint 内部强度；
- control 与目标保护区域不重叠；
- context distance 及选择顺序可复现；
- geometry control 与 context-matched control 均需保留，不能相互覆盖。

只有在 context-matched controls 下仍然 distinct，才更接近“目标位置特异性信息”。

---

## 4. 原生 head 的精确 Decision-Conversion 分解

MSHNet 的四个 side heads 是 `1×1 Conv`，final fusion 是对四个全分辨率 side logits 使用 `4→1` 的 `3×3 Conv`。因此，从 decoder feature 到 side logit，以及从 side logits 到 final logit，均可做精确线性分解，无需先训练新 probe。

### 4.1 可用特征差异

对目标实例 \(k\)、阶段 \(s\)，定义目标与背景控制的通道均值差：

\[
\Delta_{ks}
=
\mu_{ks}^{\mathrm{target}}
-
\mu_{ks}^{\mathrm{background}}.
\]

令：

\[
D_{ks}
=
\operatorname{diag}(\sigma_{ks,1},\ldots,\sigma_{ks,C})
\]

为局部背景 MAD/RMS 得到的稳健通道尺度。

定义可用差异：

\[
A_{ks}
=
\left\|
D_{ks}^{-1}\Delta_{ks}
\right\|_2.
\]

该量是 **mean-margin availability**：目标占据加权均值减去所选局部背景的算术均值，再按稳健通道尺度归一化。它不等于当前 unsigned feature-survival statistic；后者使用局部背景中位数，并通过 translated-control null rank 判断是否 distinct。两者必须并列报告，不能互换分母或结论。

### 4.2 原生输出头敏感度

side head 的 `1×1` 权重记为：

\[
w_s\in\mathbb R^C.
\]

定义 head sensitivity：

\[
H_{ks}
=
\|D_{ks}w_s\|_2.
\]

它表示当前 head 在局部背景尺度下对通道方向的总体敏感度。

### 4.3 证据利用率

定义：

\[
U_{ks}
=
\frac{
w_s^\top\Delta_{ks}
}{
\|D_{ks}w_s\|_2
\|D_{ks}^{-1}\Delta_{ks}\|_2
}.
\]

该定义仅在 \(A_{ks}>0\) 且 \(H_{ks}>0\) 时成立。若任一因子为零，\(U_{ks}\) 必须记为 undefined，同时单独记录有向 margin 为零；不能把 undefined 写成 0，也不能解释成“正交”。

这是 whitened feature difference 与原生 head direction 的余弦：

\[
U_{ks}\in[-1,1].
\]

解释如下：

| \(U_{ks}\) | 含义 |
|---:|---|
| 接近 \(+1\) | 输出头充分沿正确方向读取差异 |
| 正但接近 0 | 特征中有差异，但输出头利用很弱 |
| 接近 0 | 在 \(A,H\) 非退化且尺度 floor 未激活时，差异近似与 head 正交 |
| 小于 0 | 输出头将目标差异投影到错误方向 |
| 接近 \(-1\) | 强烈反向编码 |

### 4.4 精确 margin factorization

存在恒等式：

\[
\boxed{
w_s^\top\Delta_{ks}
=
A_{ks}\,H_{ks}\,U_{ks}
}
\]

即：

\[
\text{实际有向 logit margin}
=
\text{可用差异}
\times
\text{head sensitivity}
\times
\text{utilization}.
\]

因此，每个 miss 可以拆分为：

- `availability failure`：\(A\) 低；
- `head sensitivity failure`：\(H\) 低；
- `alignment failure`：\(U\le 0\) 或过低；
- `absolute calibration failure`：局部 margin 为正，但最终 logit 仍低于阈值；
- `global clutter competition`：目标高于局部背景，但低于全图 hard clutter；
- `fusion suppression`：side 可用但 final readout 对齐度下降或反转。

这里的 \(D\) 是对角稳健尺度，因此重参数化不变性只覆盖带逆 head 变换的逐通道缩放/符号翻转；不覆盖一般线性换基。固定 absolute floor 激活时，该不变性也不再严格成立，必须报告 floor-active channels。不同 stage 的维数与局部尺度不同，原始 \(A\) 或 \(U\) 不能直接跨 stage 比大小并据此宣称 fusion suppression。

---

## 5. Final `3×3` fusion 的精确分解

对四通道 side-logit map 做 `unfold`，将每个位置的 \(4\times3\times3\) 邻域展开为：

\[
q_x\in\mathbb R^{36}.
\]

将 final kernel 展开为：

\[
w_f\in\mathbb R^{36}.
\]

随后对 \(q_x\) 使用与 side head 相同的：

\[
A_f,
H_f,
U_f.
\]

这样可精确区分：

- decoder 到 side projection 已经失败；
- side 中存在弱正向差异，但 final kernel 与其不对齐；
- final bias 导致所有响应整体低于阈值；
- 不同 side 贡献在 final fusion 中发生正负抵消；
- final 只改变绝对标定，没有改变排序；
- final 同时破坏排序与绝对标定。

必须验证：

\[
\text{unfold}(\text{side logits})\cdot w_f+b_f
\]

与原生 `final` 卷积输出在数值误差范围内一致。

---

## 6. 目标级 failure ledger

每个 no-response target 建议输出以下字段：

```text
target_available_d0
target_available_side0
d0_head_sensitivity
d0_head_utilization
d0_signed_head_margin
side0_peak_logit
side0_local_directional_auc
final_available
final_head_sensitivity
final_head_utilization
final_signed_margin
final_peak_logit
final_peak_margin_to_zero
global_target_rank
fixed_fa_margin
recovered_threshold
failure_class
```

### 6.1 固定阈值绝对 margin

由于 `sigmoid(z)=0.5` 对应 \(z=0\)，定义：

\[
M_k^{(0.5)}
=
\max_{x\in N_3(G_k)}z(x).
\]

若：

\[
M_k^{(0.5)}\le 0,
\]

则该目标在严格 `logit > 0` 语义下无邻近响应。这里的邻域必须是到 GT support 的严格欧氏距离小于 3，不是只取目标 footprint。即使局部 peak 大于阈值，也不自动等价于 component Pd；长桥或质心偏移仍可能导致 Hungarian component miss。

### 6.2 固定 FA budget margin

定义目标分数：

\[
s_k
=
\max_{x\in N_3(G_k)}z(x).
\]

令 \(t_{\mathrm{FA}=\alpha}\) 为在**图像不相交校准集**上、使用严格阈值与 Hungarian component matching 选择的阈值，定义：

\[
M_k^{(\alpha)}
=
s_k-t_{\mathrm{FA}=\alpha}.
\]

该量用于区分：

- 只是固定阈值不合适；
- 目标与 clutter 的全局排序确实失败。

必须同时保存校准集达到的 FA、独立评估集实际 FA、exact component matched 状态与局部 peak margin。若评估集实际 FA 明显越过 budget，应报告 operating-point transfer instability，而不能仍写成“Pd@该 fixed-FA”。

### 6.3 failure taxonomy

| failure class | 判定原则 |
|---|---|
| `representation_absent` | 多种 peak/mean 统计及 image-disjoint probe 均失败；单个 mean \(A\) 低不足以判定 |
| `projection_orthogonal` | \(A,H\) 非退化、floor 稳定且 \(U_{d0}\approx0\)；否则不得使用该标签 |
| `projection_reversed` | \(A_{d0}\) 高，\(U_{d0}<0\) |
| `absolute_underconfidence` | 局部 margin 正、fixed-FA margin 正，但 \(z_{\max}<0\) |
| `clutter_competition` | 局部 margin 正，但 fixed-FA margin 不正 |
| `fusion_suppression` | 同一输出位置的 exact per-scale contribution 显示正证据被 final 权重抵消；禁止仅比较跨 stage 的 raw \(U\) |
| `threshold_only` | 调整阈值可恢复，Pd–FA 曲线不变 |
| `mixed_or_undefined` | 控制不足、统计不稳定或多个因素同时发生 |

分类逻辑必须 fail-closed，并保留多轴布尔/连续字段。`threshold_only` 是数据集 operating-point 现象，不是固有 target class。任何不满足严格判据、多个因素并存或校准外推不稳的样本均归入 `mixed_or_undefined`，不能强制分配到有利类别。

---

## 7. 固定阈值账本必须升级为 operating-point 账本

阈值 0.5 的结果可能混合三类问题：

1. 排序正确，只是整体 logit 偏低；
2. 目标与背景排序错误；
3. 目标确实没有可用响应。

必须同时报告：

- 完整 threshold sweep；
- Pd–FA 曲线；
- Pd@fixed-FA；
- FA@fixed-Pd；
- 每个目标首次被检出的阈值；
- 每个目标被检出时对应的 FA；
- threshold stability；
- affine calibration 前后对比。

### 7.1 Bias-only calibration

\[
z'=z+b.
\]

若只拟合一个 bias 就能恢复固定阈值指标，而完整 Pd–FA 曲线不变，则主要是绝对标定问题。

### 7.2 Monotonic affine calibration

\[
z'=az+b,
\qquad a>0.
\]

单调 affine 变换不改变 score 排序和 threshold-free Pd–FA 曲线。若只修复 \(\tau=0.5\) 指标，则不能作为顶会方法动机。

---

## 8. 冻结模型 probe 的正确顺序

在设计新训练目标之前，应按以下顺序完成冻结诊断。

### Probe 0：原生 head

直接计算 \(A,H,U\)，不训练任何新参数。

目的：判断原生线性 readout 是否对现有特征差异利用不足或方向错误。

### Probe 1：bias-only

冻结全部网络，只拟合一个标量 bias。

目的：验证固定阈值绝对标定问题。

### Probe 2：原生 side-head / final-head refit

冻结 encoder 和 decoder，仅重新拟合已有：

- `output_0`；
- `output_1`；
- `output_2`；
- `output_3`；
- `final` convolution。

这不是新方法，而是诊断：

> 特征保持不变时，重新学习原有线性 readout 是否能恢复 no-response targets？

若能够稳定恢复，说明问题主要在 readout utilization，而非 representation absence。

### Probe 3：跨图像线性 probe

在 fit images 上训练 ridge logistic probe，在完全不同的 held-out images 上评估：

- target vs geometry controls；
- target vs context-matched controls；
- no-response target vs matched-target controls。

必须满足：

- 图像级 train/eval 隔离；
- 同一图像不得同时出现在训练和评估；
- target/control 数量平衡；
- 随机标签 control；
- 随机位置 control；
- area、SCR、边缘距离分层；
- 所有超参数只在 fit split 内确定。

### Probe 4：小型非线性 probe

仅当线性 probe 失败时运行。

若非线性 probe 成功而线性 probe 失败，说明特征中存在信息，但其结构不适配 MSHNet 当前线性 side head。这属于 representation/readout compatibility 问题，而不只是监督目标不足。

---

## 9. 建议的代码修改

## 9.1 `utils/feature_survival.py`

新增结果结构：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadConversionResult:
    mean_margin_availability: float | None
    head_sensitivity: float | None
    utilization_cosine: float | None
    mean_logit_margin: float | None
    reconstructed_margin: float | None
    reconstruction_error: float | None
    absolute_scale_floor_active_channels: int | None
    reparameterization_stable: bool | None
    absolute_peak_margin: float | None
```

新增 API：

```python
def evaluate_linear_head_conversion(
    feature,
    geometry,
    *,
    head_weight,
    head_bias=None,
    scalar_threshold=None,
) -> HeadConversionResult:
    """Evaluate availability, head sensitivity, utilization and margin.

    This is a read-only diagnostic. It must not modify the model graph,
    feature tensor, or head parameters.
    """
    ...
```

核心实现：

```python
delta = target_mean - background_arithmetic_mean
scale = robust_channel_scale(background_values)

available_vector = delta / scale
head_vector = weight * scale

available = np.linalg.norm(available_vector)
sensitivity = np.linalg.norm(head_vector)

utilization = None
if available > 0 and sensitivity > 0:
    utilization = np.dot(available_vector, head_vector) / (
        available * sensitivity
    )

signed_margin = np.dot(weight, delta)
reconstructed = (
    available * sensitivity * utilization
    if utilization is not None
    else None
)
```

必须处理：

- 零方差通道；
- 全零 head；
- 单通道输出；
- NaN/Inf；
- target/control 样本不足；
- 正比例 channel rescaling；
- 负比例 channel flipping；
- mixed precision 下的稳定性。

同时新增 context-matched control selector，例如：

```python
def select_context_matched_controls(
    image,
    controls,
    *,
    protection_mask,
    context_ring_width: float,
    num_controls: int,
):
    ...
```

要求该函数只读取目标 footprint 外的 context ring，不得使用目标内部强度进行匹配。

---

## 9.2 `model/mshnet_stage_evidence_view.py`

继续保持 read-only，并且只执行一次原生 forward。

新增导出：

```python
{
    "side_heads": {
        "s0": {
            "weight": model.output_0.weight.detach(),
            "bias": model.output_0.bias.detach(),
        },
        "s1": {
            "weight": model.output_1.weight.detach(),
            "bias": model.output_1.bias.detach(),
        },
        "s2": {
            "weight": model.output_2.weight.detach(),
            "bias": model.output_2.bias.detach(),
        },
        "s3": {
            "weight": model.output_3.weight.detach(),
            "bias": model.output_3.bias.detach(),
        },
    },
    "final_head": {
        "weight": model.final.weight.detach(),
        "bias": model.final.bias.detach(),
    },
}
```

禁止：

- 修改 `forward()` 返回语义；
- 增加训练 head；
- 替换原生 activation；
- 重算第二次 forward；
- 在 hook 中改变 tensor；
- 将 diagnostic tensor 注册为模型参数。

---

## 9.3 `tools/audit_mshnet_feature_survival.py`

新增 CLI：

```text
--control-policy geometry
--control-policy context_matched
--context-control-count 64
--context-ring-width 5
--fixed-fa-budgets ...
--compute-head-conversion
--compute-final-unfold-conversion
--compute-affine-calibration
--failure-taxonomy-version v1
```

输出 schema 增加：

```json
{
  "availability_rate": {},
  "positive_utilization_rate": {},
  "orthogonal_utilization_rate": {},
  "reversed_utilization_rate": {},
  "calibration_only_rate": {},
  "clutter_competition_rate": {},
  "fusion_suppression_rate": {},
  "failure_class_counts": {},
  "control_policy_comparison": {},
  "fixed_fa_operating_points": {}
}
```

每条 target record 应保存：

```json
{
  "sample_name": "...",
  "target_index": 0,
  "area": 5,
  "outcome": "no_response",
  "controls": {
    "geometry": {},
    "context_matched": {}
  },
  "conversion": {
    "d0": {},
    "s0": {},
    "final": {}
  },
  "operating_point": {},
  "failure_class": "mixed_or_undefined"
}
```

---

## 9.4 新增 `utils/head_conversion.py`

建议将线性分解与 final unfold 分解从 audit script 中拆出，形成纯函数模块：

```python
def robust_channel_scale(...):
    ...


def factorize_linear_margin(...):
    ...


def extract_final_fusion_vectors(...):
    ...


def classify_conversion_failure(...):
    ...
```

该文件只包含：

- 无状态数值函数；
- 类型与形状校验；
- fail-closed 分类；
- 无模型训练逻辑；
- 无 CLI；
- 无数据集特例。

---

## 9.5 新增 `tools/audit_mshnet_decision_conversion.py`

为了避免原 `feature_survival` 工具过度膨胀，可单独建立 decision-conversion audit：

```text
tools/audit_mshnet_decision_conversion.py
```

职责：

1. 加载 pristine checkpoint；
2. 通过 stage evidence view 运行单次 forward；
3. 读取 geometry/context controls；
4. 计算各 stage 的 \(A,H,U\)；
5. 对 final `3×3` kernel 做 unfold factorization；
6. 运行 threshold sweep 与 fixed-FA audit；
7. 输出 target-level ledger 与 dataset summary；
8. 不训练任何参数；
9. 不修改 checkpoint；
10. 不改变 baseline 推理结果。

---

## 9.6 测试修改

建议新增：

```text
tests/test_head_conversion.py
tests/test_context_matched_controls.py
tests/test_decision_conversion_audit.py
```

至少覆盖：

```text
test_unsigned_survival_cannot_distinguish_sign
test_head_utilization_changes_sign_under_head_reversal
test_margin_factorization_is_exact
test_factorization_is_invariant_to_positive_channel_rescaling
test_orthogonal_head_has_zero_utilization
test_zero_head_is_fail_closed
test_final_3x3_unfold_matches_direct_convolution_margin
test_affine_calibration_preserves_score_order
test_bias_only_calibration_preserves_score_order
test_context_matching_never_uses_target_footprint
test_context_controls_respect_protection_mask
test_control_selection_is_deterministic
test_failure_taxonomy_is_fail_closed
test_mixed_cases_are_not_forced_into_named_class
test_read_only_audit_does_not_change_model_state
test_read_only_audit_reproduces_original_logits
```

---

## 10. 创新性判断

### 10.1 当前 feature-survival audit

它是高价值诊断工具，但不能作为论文主方法。

以下元素单独都不足以构成顶会创新：

- 同形状平移 control；
- feature contrast；
- hard-negative comparison；
- logit margin；
- 中间层 probe；
- side-head 重训；
- bias calibration。

因此，不能发展为：

```text
SLS
+ target-background contrast
+ hard-negative ranking
+ feature consistency
```

这种设计仍然是典型 compound loss stacking，并且会与已有 target–hard-negative ranking、false-alarm suppression、boundary loss 等工作形成直接重叠。

### 10.2 原条件候选：Evidence-to-Decision Utilization

只有在九组 checkpoint 一致证明：

\[
A_{d0}\text{ 高，但 }U_{d0}\text{ 系统性低或为负}
\]

并且同时满足：

- bias-only 无法解释；
- monotonic affine calibration 无法解释；
- frozen linear head refit 可以稳定恢复；
- context-matched controls 下结论保持；
- 三数据集、多个 seed 方向一致；
- 问题主要出现在同一 stage 或同一转换环节；

才有资格发展统一主方法：

# Evidence-to-Decision Utilization

其核心不能是“再加一个 contrast loss”，而应是：

> 对原生线性 readout，显式优化可用特征差异被同一个决策方向利用的比例，而不是仅优化输出概率、未经归一化的 logit 间距或额外 hard-negative 项。

潜在贡献应围绕：

1. **Availability–utilization separation**
   区分“没有信息”与“有信息但 readout 不使用”。

2. **受限的 channel-reparameterization invariance**
   仅在 absolute floor 未激活、并同步逆变换 head 时，\(U\) 对逐通道非零缩放保持不变；不覆盖一般线性换基。

3. **Instance-wise worst-case conversion**
   防止高对比目标主导平均目标，而弱目标方向仍与 head 不一致。

4. **No added inference module**
   只优化原有 MSHNet 参数，推理图不增加 head、gate、attention 或 refinement branch。

5. **Single risk, not additive modules**
   若最终方法成立，应以统一 decision-conversion risk 完整替换旧的 side-loss 组合，而不是附加：

   ```text
   SLS + utilization + contrast + calibration
   ```

### 10.3 当前仍不能宣称的内容

现阶段不能声称：

- MSHNet 的主要问题已经确定为 head misalignment；
- no-response 目标的语义在 decoder 中普遍保留；
- side supervision 系统性破坏了 readout；
- final fusion 是主要瓶颈；
- utilization objective 已具备顶会创新性；
- 一个归一化余弦项即可解决 Pd–FA 权衡。

九组精确分解现已完成；上述主方法声明仍不成立，具体证据见第 15 节。

---

## 11. 决策树

| 九组审计结果 | 结论 | 后续 |
|---|---|---|
| 大部分 miss 可由 affine calibration 恢复 | 固定阈值校准问题 | 不发展顶会方法 |
| \(A\) 低，线性/非线性 probe 均失败 | 表征确实缺失 | loss-only 路线 NO-GO |
| \(A\) 高，\(U\) 低，head-only refit 成功 | readout utilization failure | 条件发展统一 utilization risk |
| \(A\) 高，但仅非线性 probe 成功 | 特征存在但线性不可读 | 需要改 head/架构，不满足无模块约束 |
| 局部排序正确，全局 fixed-FA margin 失败 | hard clutter competition | 研究 operating-point risk |
| 同一输出位置的 exact contribution 显示 final 系统性压制正证据 | fusion suppression | 研究统一 fusion/readout 训练原则 |
| 各数据集失败原因不一致 | 无统一根因 | 停止 MSHNet 单机制主线 |

---

## 12. 当前 GO / NO-GO

| 路线 | 判断 |
|---|---|
| CCSR / component cut | 永久 NO-GO |
| bridge/merge loss | 永久 NO-GO |
| side threshold absence = feature disappearance | 结论不成立 |
| 当前 unsigned feature-survival | 诊断 GO |
| context-matched control | GO |
| 原生 head margin factorization | GO |
| final `3×3` unfold factorization | GO |
| 直接训练新 feature head | 暂停 |
| 普通 logit hard-negative ranking | 新颖性 NO-GO |
| SLS + contrast + ranking + calibration | 模块堆叠 NO-GO |
| Decision-conversion factorization | 诊断 GO |
| Evidence-utilization 独立主方法 | D1 后 NO-GO：并非统一根因，且纯 \(1-U\) 退化 |
| final-fusion suppression 主线 | NO-GO：同号传递 69/71，抵消在成功对照中同样普遍 |
| D2 head-refit / 长周期主方法训练 | 当前 gate 不授权 |

---

## 13. 下一步执行顺序

### Gate D0：审计工具正确性（已通过）

必须先通过：

- margin factorization exact identity；
- final unfold 与原生 convolution 一致；
- read-only audit 不改变 logits；
- geometry/context controls 可复现；
- fail-closed taxonomy；
- 全部数值 finite。

### Gate D1：九组 checkpoint 完整账本（已完成）

三数据集 × 三 seed，逐目标输出：

- availability；
- head sensitivity；
- utilization；
- absolute calibration；
- fixed-FA margin；
- exact per-scale contribution 与 sign cancellation；
- fail-closed 多轴字段（不自动强制 failure class）。

### Gate D2：冻结 probe（未触发）

仅在 D1 出现一致性机制后运行：

- bias-only；
- affine calibration；
- existing-head-only refit；
- image-disjoint linear probe；
- 必要时 small nonlinear probe。

### Gate D3：方法资格判断（未通过）

只有同时满足以下条件，才进入新目标函数设计：

1. 三数据集方向一致；
2. 多 seed 稳定；
3. context-matched controls 下仍成立；
4. affine calibration 不能解释；
5. 主要错误集中在同一个 conversion mechanism；
6. 原生 head refit 能恢复；
7. 不需要增加 inference module；
8. 可构造一个统一风险，而不是多项附加 loss。

---

## 14. 最终结论

当前最合理的研究主线已经不是：

- mass transport；
- component cut；
- bridge suppression；
- side-output threshold recovery；
- 常规 hard-negative ranking。

该诊断步骤现已完成：

> **在九组 pristine MSHNet checkpoint 上，把每个 no-response target 精确分解为 availability、head sensitivity、utilization、absolute calibration、global clutter competition 与 fusion suppression。**

九组结果见下一节。它没有给出一个跨数据集、跨 seed 的单一 conversion 根因，因此不进入新目标函数设计。

Evidence-to-Decision Utilization 从“条件候选”降级为**诊断量**；当前不能作为主方法立项。

---

## 15. Gate D0/D1 实际执行结果（2026-07-12）

### 15.1 协议与完整性

- 三数据集 × 三 seed，共 9 个 pristine MSHNet checkpoint；
- 固定 internal-validation split，共 1008 次图像观测、1479 个 GT component；
- 所有 cohort、operating-point 与 stage-evidence forward 固定 `batch_size=1`，避免 CUDA 不同 batch 归约带来的阈值漂移；
- 共得到 71 个 fixed-threshold no-response target，并按面积/边界距离一对一选择 71 个 matched target control；
- side `1×1` head、final `3×3` unfold 与四尺度贡献均使用原生权重，只读且不训练；
- 710 个已定义 factorization 项的最大数值重构误差为 \(1.53\times10^{-5}\)（float32）；退化项保留 undefined，不写成零；
- fixed-FA 使用确定性两折 image-disjoint cross-fitting、预注册的 fixed-logit 与 pooled-tail finite threshold grid、严格 `logit > threshold` 和 Hungarian component matching；它不是对全部 unique logits 的穷举最优曲线。
- NUAA-SIRST seed-20260713 由当前代码完整重跑后 JSON byte-identical，确定性复核通过。

原始九组账本位于 `repro_runs/ccsr_gate_c1/*_decision_conversion.json`；汇总为：

- `repro_runs/ccsr_gate_c1/summary_decision_conversion_v1.json`
- `repro_runs/ccsr_gate_c1/summary_decision_conversion_v1.md`

### 15.2 Context control 结论

| cohort | geometry D0 distinct | context-matched D0 distinct |
|---|---:|---:|
| no-response | 65/71 | 58/61（10 个 fail-closed unavailable） |
| matched control | 70/71 | 56/57（14 个 fail-closed unavailable） |

因此，“多数 miss 在 `d0` 仍存在相对局部背景的无符号偏离”在 context matching 后仍成立；但成功目标也几乎同样 distinct。该统计支持“不能把 side threshold absence 等同于 feature disappearance”，却不能单独定位 readout 根因或证明目标语义完整保留。

### 15.3 精确 conversion 结论

| final 指标 | no-response | matched control |
|---|---:|---:|
| mean logit margin 正/负 | 49/22 | 71/0 |
| median utilization \(U\) | 0.0836 | 0.2212 |
| median mean-margin availability \(A\) | 11.67 | 62.15 |
| median head sensitivity \(H\) | 5.33 | 4.12 |

配对差值统一按 `miss − control` 计算：final mean margin 在 71/71 对中更低，中位差为 -53.09；final \(U\) 在 58/71 对中更低，中位差为 -0.1356；final \(A\) 在 70/71 对中更低。由此不能把缺口唯一归因于 utilization：availability magnitude 同时、且更一致地下降，而 head sensitivity 并未系统性偏低。

更关键的是，`d0` 与 final mean-margin 的符号在 69/71 个 miss 上保持一致：48 个正→正、21 个负→负，仅 1 个正→负和 1 个负→正。四尺度贡献存在正负抵消的比例是 miss 60/71、成功对照 70/71；它在成功样本中反而更常见。因此：

> **final fusion suppression 不是这批 no-response 的统一主因。**

### 15.4 Operating-point 结论

在跨折阈值下，71 个 fixed-threshold no-response 的 exact component 恢复数为：

| calibration FA budget（/Mpix） | no-response 恢复 | 汇总后的实际 eval FA/Mpix | 全部 GT eval Pd |
|---:|---:|---:|---:|
| 1 | 0/71 | 3.45 | 0.2596 |
| 10 | 0/71 | 19.80 | 0.7181 |
| 20 | 1/71 | 23.04 | 0.8607 |

这排除了“大部分只是 0.5 阈值不合适”的解释。同时，校准 budget 到独立折的实际 FA 存在明显外推偏差，尤其小样本 fold；所以这些点只能写成 cross-fitted operating-point 诊断，不能冒充精确命中的固定 FA 测试点。

### 15.5 D1 门控决定

当前证据同时出现两类 miss：22/71 的 final local mean margin 为负，49/71 为正但绝对 peak 仍低于固定阈值且在实用 FA budget 下几乎不恢复。它们不是同一个 conversion failure。

纯 utilization 风险也有结构性退化：\(U=1\) 可以与任意小的 \(A H\) 共存；而 \(A H U=w^\top\Delta\) 又退化为普通有向 logit contrast。若再补 availability、absolute calibration、hard clutter 与 component operating-point 项，就重新变成已经禁止的多损失堆叠与常规 ranking 路线。

因此本轮决定是：

1. context-matched feature survival、linear factorization 与 final unfold：继续保留为诊断工具；
2. final-fusion suppression：NO-GO；
3. threshold-only calibration：NO-GO；
4. standalone Evidence-to-Decision Utilization 主方法：NO-GO；
5. D2 existing-head refit 与长周期训练：当前不运行，因为 D1 没有通过“主要错误集中于同一 conversion mechanism”这一前置门；
6. 不把 \(1-U\)、margin、hard-negative、FA calibration 拼成新 loss。

这不是 baseline 性能结论，也不是说 decoder 完全没有信息；结论仅是：**现有九组证据不足以支撑一个不堆模块、以 utilization 为唯一原理的顶会主方法。**
