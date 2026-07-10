# MSHNet / DEAIntegratedMSHNet 诊断与主模型设计评审

> **范围说明（2026-07-10）**：MSHNet 是开源 baseline；本文中的 DEAIntegratedMSHNet/CSIR 是历史诊断与控制原型，不是当前 DEA 主模型，也不是 DEA-lite。当前 DEA 结构、完整 400 epoch 结果和去留判断以 `MSHNet_伴随预测误差回写主模型设计评审.md` 为准。

## 总体判断

**你的主判断是对的：现在应停止继续调 `lambda/margin`，冻结 DEAIntegratedMSHNet 为控制模型，先完成 frozen-baseline 的尺度诊断。**

但这份表述不能原样定稿，有四个关键修正：

1. 当前 Integrated 模型的 **decoder feature path 确实是递归的**；不是真递归的是 terminal scale-fusion 中命名为 `recursive_states` 的那段累加。
2. 四个 DEA cell 是 **参数不共享**，但其输入并非彼此独立：较粗尺度经过路由后的 decoder feature 会进入下一尺度。
3. 原 CSIR 公式在代数上就是一个单门 GRU 更新，仍不足以支撑主模型创新。
4. 你提出的三个诊断基本正确，但“顺序干预”和“跨数据集尺度排序反转”的逻辑需要拆开。

---

# 一、对当前代码定位的准确判断

## 1. MSHNet 不能描述成“四尺度完全独立”

代码中 decoder 明确是：

\[
x_{d3}\rightarrow x_{d2}\rightarrow x_{d1}\rightarrow x_{d0},
\]

后一层 decoder 使用前一层上采样结果，所以四个 decoder 状态具有粗到细的层级依赖。随后才从四个状态分别生成四个 side logits，并用固定的 `Conv2d(4,1,3)` 汇总。

因此论文里最准确的描述应是：

> MSHNet 从层级耦合的 decoder states 生成四个分别监督的 side predictions，再使用一个输入无关、空间平稳的 \(3\times3\) 仿射算子完成最终聚合。

这里还要再加一个限制：

**固定 final fusion 不等于整个 MSHNet 没有条件适应能力。** Decoder features 和 side logits 本身都是输入相关的，它们已经可能在上游隐式抑制不可靠尺度。真正需要验证的是：

> 在上游 side prediction 已经完成条件编码后，是否仍存在 final fusion 无法处理的、随当前预测状态和场景变化的剩余尺度边际效用。

这是诊断必须存在的原因。

---

## 2. 对 DEAIntegratedMSHNet 的判断基本正确，但要区分两种“递归”

四个 DEA cell 的确是四个单独实例，没有共享参数。

但 feature decoder 部分确实递推：

- `route_3` 修正后的 \(x_{d3}\) 进入 `dea_cell_2`；
- 修正后的 \(x_{d2}\) 进入 `dea_cell_1`；
- 再进入 `dea_cell_0`。

代码注释本身也明确指出，路由后的 feature 会被更细尺度 decoder 递归消费。

相反，terminal fusion 中：

\[
z_{\mathrm{DEA}}
=
z_{\mathrm{base}}+\sum_s \Delta_s
\]

每个 \(\Delta_s\) 在进入所谓的 `recursive_states` 之前已经计算完成，后续只是按顺序相加。给定这些预计算的 delta 后，终点与累加顺序无关。

因此应写成：

> DEAIntegratedMSHNet 的 feature decoder routing 是真实的粗到细递推；但其 terminal logit closure 不是 state-conditioned recursion，而是对预计算修正项的顺序记账。各 delta 可能通过 decoder feature path 间接耦合，但它们不依赖正在更新的 terminal logit state。

也不能再说它“只改变融合点”，因为启用 `decoder_routing` 时，它同时改变每个 decoder stage 的 feature。更合理的定位是冻结为三组控制：

- **DecoderOnly**：测试 decoder feature routing；
- **ScaleOnly**：测试 terminal routing；
- **Both**：测试二者组合；
- 连续 attention 模式作为 tri-state routing 的参数匹配对照。

所以最终定位成立：

> **DEAIntegratedMSHNet 是 decoder-routing / terminal-routing control family，不是顶会主模型。**

---

# 二、建议采用的正式问题定义

可以直接改成下面这版：

> MSHNet 的层级 decoder states 是粗到细相关的，并产生内容相关的多尺度 side predictions；然而，最终预测由一个输入无关、空间平稳的仿射算子统一聚合。该聚合规则能够学习平均意义上的跨尺度组合，却没有显式建模某一尺度相对于当前预测状态所提供的是新增证据、冲突证据还是冗余证据。  
>   
> 我们假设，在上游 side predictions 已具有输入适应性的前提下，尺度的剩余边际效用仍可能随目标尺寸、局部对比度和背景杂波条件变化。该假设必须先通过冻结模型的反事实尺度干预得到验证，再决定是否需要状态递归解码。

这比“尺度完全独立”和“NUAA/NUDT 已证明尺度可靠性反转”严谨得多。

DEA-lite 在 NUAA 上的负结果目前只能证明：

> 该正则化假设或优化方式不具有数据集鲁棒性。

它不能直接证明 baseline 存在尺度可靠性反转，更不能直接证明递归 decoder 必要。

---

# 三、原始 CSIR 目前确实不够新

你给出的更新：

\[
z_s
=
\bar z_s+g_s(e_s-\bar z_s)
=
(1-g_s)\bar z_s+g_se_s
\]

从代数上看，就是标准的 update-gate interpolation：

- \(\bar z_s\)：previous state；
- \(e_s\)：candidate state；
- \(g_s\)：update gate。

这与 GRU 的核心更新形式相同；而 CRDN 已经把卷积递归单元用于层级 decoder feature fusion。

此外还有三个具体问题。

### 1. `Keep` 在当前公式中不一定真实存在

若 \(g_s=\operatorname{sigmoid}(\cdot)\)，则理论上：

\[
0<g_s<1,
\]

所以不会精确等于零。此时所谓 Keep 只是“很小的更新”，并非结构性拒绝证据。

要获得精确 Keep，需要硬门、稀疏映射、dead zone 或解析阈值，而这些本身也已有大量先例。

### 2. \(e_s\) 的“尺度观测”语义不成立

你当前定义：

\[
(e_s,g_s)=\Phi_\theta(A_sF_s,\bar z_s)
\]

意味着 \(e_s\) 可以任意依赖当前 state。它既可以利用尺度特征，也可以直接复制、翻转或重写 \(\bar z_s\)。

所以 \(e_s-\bar z_s\) 只是一个 learned residual，不自动等于统计意义上的“新尺度证据”。

若要坚持 innovation 解释，至少要限制：

- observation \(e_s\) 主要由当前尺度 \(F_s\) 产生；
- 当前 state 只用于判断该 observation 是否可信、是否冗余。

### 3. “不会过冲”只能相对于 learned candidate 成立

更新确实位于 \(\bar z_s\) 和 \(e_s\) 之间，但 \(e_s\) 本身是无界网络输出。因此只能说：

> 更新不会越过当前网络产生的 candidate。

不能说它不会相对于 ground truth 或合理 posterior 发生过度修正。

---

## 相关工作判断

你对 prior-art 风险的总体判断是正确的：

- CRDN 已使用递归 decoding cells 串联层级特征；
- Dynamic Routing 已根据输入尺度分布产生数据相关路径；
- SegRefiner 已进行逐像素迭代状态转移；
- RRCANet 已在 IRSTD 中使用 recurrent reusable convolution；
- SeRankDet 也已在 IRSTD 中使用动态 feature fusion。

因此，**递归、共享参数、动态路由、I/D/K 命名都不能单独作为创新点。**

文献表中 `RRN (CVPR 2018)` 的具体论文条目建议重新核对；仅凭该简称和年份，尚未核实到唯一匹配项。SegRefiner 的 venue 信息也应在正式 related work 中再次核准，不要只依据简称和年份落笔。

---

# 四、三个诊断应该保留，但需要这样修改

## D1：错误—尺度冲突关系

你的方向正确，但主要证据不应首先是 side-logit 的投票熵或 JS divergence。

四个 side heads 的校准程度可能不同，直接对它们做 Bernoulli JS 容易把“校准差异”误认为“尺度冲突”。更稳妥的主证据是最终卷积的精确贡献：

\[
c_s=W_s*s_s,\qquad
z=b+\sum_{s=0}^{3}c_s.
\]

当前代码已经计算了 scale-only outputs，但有一个实现陷阱：`z_only` 的每个通道都加了一次 final bias。

所以纯尺度贡献应写成：

\[
c_s=z_{\mathrm{only},s}-z_{\mathrm{empty}},
\]

并验证：

\[
z_{\mathrm{full}}
\approx
z_{\mathrm{empty}}+\sum_s c_s.
\]

由于 grouped convolution 与原 direct convolution 的浮点归约顺序不同，允许 \(10^{-5}\) 左右的数值误差，或者用 float64 做诊断。

### 建议的证据优先级

**一级证据：**

- 精确贡献符号和幅值；
- 贡献抵消度：

\[
C(R)=
1-\frac{|\sum_s c_s(R)|}
{\sum_s|c_s(R)|+\epsilon};
\]

- leave-one-scale-out 的损失和检测变化；
- 由全部 16 个尺度子集计算出的 exact Shapley marginal utility。

四个尺度只有 \(2^4=16\) 个 coalition，计算 exact Shapley 几乎没有额外成本。它比单独 leave-one-out 更好，因为它能反映某一尺度在不同已有尺度组合下的平均边际作用。

**二级证据：**

- side probability 的 entropy；
- pairwise Bernoulli JS；
- sign disagreement。

这些只能作为描述性结果，最好先在 validation set 对各 side head 做 temperature calibration。

### 分析单元

不要只做像素级统计。主分析应对应官方 PD/FA 定义：

- detected target 与 FN target；
- matched predicted component 与 FP component；
- image-level 作为 bootstrap cluster。

当前评测按连通域质心距离小于 3 像素匹配，并把未匹配预测连通域面积计入 FA；诊断应完全复用这一规则。

### GO 条件

你给出的 AUROC \(\ge 0.60\)、bootstrap 下界 \(>0.50\) 可以作为内部工程门槛，但应同时要求：

- FN 与 FP 两个任务分别成立；
- AUROC 和 AUPRC 都报告；
- 按 image bootstrap，而不是按像素 bootstrap；
- 至少在多个 baseline seed 上方向一致；
- 不能只在其中一个数据集成立后就写成通用规律。

---

## D2：冻结贡献下的动态上界

需要把它准确称为：

> **frozen-contribution intervention oracle**

它只是固定 MSHNet features、side logits 和 final kernels 后，对尺度贡献进行选择的上界，不是所有动态 decoder 的理论上界。

建议形成四级 oracle：

| Oracle 层级 | 回答的问题 |
|---|---|
| 全数据集固定子集 | 删除某些尺度是否已经足够 |
| 每图 16 子集 oracle | 是否需要 image-conditioned selection |
| 每组件子集 oracle | 是否需要 component-local selection |
| 每像素子集/连续权重 oracle | 是否需要 pixel-conditioned routing |

还应加入：

- best singleton scale；
- 四个 leave-one-out；
- validation 选择的 global best subset；
- 每个 subset 自己校准 threshold 的结果。

最后一项很重要：若 oracle gap 在重新校准 threshold 后消失，问题主要是输出校准，而不是尺度条件冲突。

### FN、FP 应分开报告

不要只报告 oracle \(\Delta\)IoU：

- **FN recoverability**：baseline FN 中，有多少能被某个尺度子集恢复；
- **FP suppressibility**：baseline FP 中，有多少能在不损失已有 TP 的前提下被某个子集消除；
- **jointly correctable images**：是否存在同时减少 FN 和 FP 的子集；
- PD–FA Pareto frontier。

你提出的 \(\Delta\)IoU \(\ge 0.01\) 可以保留作内部门槛，但还应要求 gain 大于 baseline 多 seed 波动。

### 顺序干预不能直接混在 subset oracle 里

对精确贡献：

\[
z=b+\sum_s c_s
\]

最终和是可交换的，所以粗到细、细到粗、随机顺序最终完全相同。直接改变求和顺序没有诊断意义。

顺序诊断应另定义为 **greedy policy oracle**：

\[
a_s^\star
=
\arg\max_{a\in\{0,1\}}
U(z_{\mathrm{current}}+a\,c_s),
\]

然后分别按：

- coarse-to-fine；
- fine-to-coarse；
- 多个随机顺序

执行。

只有这种“当前决策依赖已有 state”的 greedy intervention 才会产生顺序差异。它检验的是：

> 自然尺度顺序是否为状态依赖决策提供了额外价值。

这才是递归必要性的直接证据。

---

## D3：条件可靠性需要拆成 D3a 和 D3b

### D3a：尺度边际效用是否可预测

这是动态模型的必要条件。

输入条件可包括：

- target area / equivalent diameter；
- local SCR 或 CNR；
- target ring 的梯度能量、边缘密度；
- Gaussian low-pass 后的局部非均匀性；
- LoG 或 top-hat 局部峰值密度；
- 背景局部方差和方向性。

预测目标最好使用：

- exact Shapley utility；
- leave-one-out helpfulness；
- scale inclusion 是否修复 FN／抑制 FP。

并使用按 image 分组的 out-of-fold 预测。若这些客观条件不能预测尺度 helpfulness，oracle gap 很可能不可学习。

### D3b：跨数据集 transport 和排序反转

这是跨域叙事的加强证据，但**不是递归成立的必要条件**。

建议做完整的 \(2\times2\)：

| 训练 checkpoint | NUAA 测试 | NUDT 测试 |
|---|---:|---:|
| NUAA-trained | ✓ | ✓ |
| NUDT-trained | ✓ | ✓ |

这样可以区分：

- 固定模型换数据后尺度排序变化：更支持场景条件效应；
- 只在重新训练模型后变化：更像优化或参数适配效应；
- 两者都不变化：DEA-lite 的反向结果不能归因于尺度可靠性反转。

原来的判断：

> D3 不成立就不支持条件递归

过于严格。应改为：

- **D3a 不成立**：没有可学习的条件动态性，停止动态模型；
- **D3a 成立但 D3b 不成立**：仍可支持数据集内部的条件动态融合，但不能宣称跨数据集尺度排序反转；
- **D3b 成立**：可以进一步支撑跨域鲁棒性叙事。

手工“背景类型”只能作为盲标、次级分析。主证据应使用上述客观描述量。

---

# 五、真正的 GO / NO-GO 决策树

| 诊断结果 | 下一步 |
|---|---|
| D1 不成立 | 转向 encoder 小目标信息丢失、监督或数据域问题 |
| D1 成立、D2 gap 很小 | 存在冲突，但现有尺度选择无法修复；考虑校准或 encoder |
| D1/D2 成立、D3a 不成立 | oracle gap 存在但不可预测，不做动态主模型 |
| D3a 成立，但 image-level gate 已闭合 gap | 做简单动态融合，不做递归 |
| local oracle 明显高于 image oracle | 才需要逐像素或逐组件控制 |
| greedy 顺序无差异 | 动态 set fusion 足够，不支持递归 |
| coarse-to-fine 明显优于反向/随机，且 attention 未闭合 gap | 才进入真正的状态递归 decoder |

可以定义 gap closure ratio：

\[
R_{\mathrm{close}}
=
\frac{U_{\mathrm{model}}-U_{\mathrm{base}}}
{U_{\mathrm{oracle}}-U_{\mathrm{base}}}.
\]

若轻量 pixel-wise soft attention 已达到约 \(70\%-80\%\) 的 oracle gap，且与递归模型的差异落在置信区间内，就没有足够理由使用递归。

---

# 六、当前主模型：共享鲁棒预测校正 Decoder

原 CSIR 和 PSIR 都不再作为主模型。二者都可逐元素改写为：

\[
h'=(1-g)h+ge,
\]

本质仍是 candidate interpolation。当前主结构改为 **多通道状态上的预测—残差—伴随回写**：状态不接收一个候选预测，而是预测当前尺度 encoder observation，只用无法解释的部分校正自身。

## 1. 保留和删除的结构

保留 MSHNet 五级 encoder：

| 尺度 | 特征 | 通道 | 输入为 256 时的分辨率 |
|---|---|---:|---:|
| 4 | \(F_4=x_m\) | 256 | \(16\times16\) |
| 3 | \(F_3=x_{e3}\) | 128 | \(32\times32\) |
| 2 | \(F_2=x_{e2}\) | 64 | \(64\times64\) |
| 1 | \(F_1=x_{e1}\) | 32 | \(128\times128\) |
| 0 | \(F_0=x_{e0}\) | 16 | \(256\times256\) |

完整删除：

- `decoder_3 ... decoder_0`；
- `output_3 ... output_0`；
- `final`；
- 仅为旧 DEA-lite 服务的 `decidability_head`。

新 decoder 只包含：

1. 五个必要的线性 \(1\times1\) adapter \(A_s\)，把不同 encoder 通道映射到共同状态维度 \(C\)；
2. 一个在五个尺度严格共享的预测算子 \(K_\theta\) 及其参数绑定的伴随 \(K_\theta^*\)；
3. 一个共享的鲁棒影响函数；
4. 一个共享的 \(1\times1\) readout。

adapter 只解决输入坐标维度不同，不承担各自独立的 decoder 功能。

## 2. 多通道状态和尺度观测

状态不再是一通道 logit，而是：

\[
h_s\in\mathbb R^{B\times C\times H_s\times W_s},
\]

首版取 \(C=32\)，同时训练 \(C=64\) 容量控制。

各尺度观测为：

\[
o_s=N(A_sF_s),
\]

其中 \(N\) 是五个尺度共享的 GroupNorm。最粗尺度从可学习通道先验开始：

\[
\bar h_4=b.
\]

其余尺度只做无参数上采样：

\[
\bar h_s=\operatorname{BilinearUp}
(h_{s+1},\operatorname{size}(F_s)),
\qquad s=3,2,1,0.
\]

## 3. 唯一共享的状态更新

共享预测算子写为：

\[
K_\theta=M_\theta D_\theta,
\]

其中 \(D_\theta\) 是 depthwise \(3\times3\) 空间算子，\(M_\theta\) 是 \(1\times1\) 通道混合。反投影不是另一个网络，而是严格复用同一组参数的伴随：

\[
K_\theta^*=D_\theta^*M_\theta^*.
\]

当前状态对尺度观测的预测残差为：

\[
r_s=o_s-K_\theta\bar h_s.
\]

采用跨尺度共享的 pseudo-Huber influence：

\[
\psi_\delta(r)
=
\frac{r}{\sqrt{1+(r/\delta)^2}},
\qquad \delta_c>0.
\]

最终更新为：

\[
\boxed{
h_s
=
\bar h_s+\eta K_\theta^*
\psi_\delta
\left(o_s-K_\theta\bar h_s\right)
}
\]

当前实现固定 \(0<\eta\le1\)，并约束 depthwise 和 pointwise 部分的增益。五个尺度执行的是同一个更新，不存在 per-scale cell、attention、router 或 terminal fusion。

这个更新也可以理解为对当前尺度鲁棒预测能量做一步校正：

\[
E_s(h)
=
\sum_{p,c}
\delta_c^2
\left(
\sqrt{1+\left(
\frac{o_{s,p,c}-[K_\theta h]_{p,c}}
{\delta_c}
\right)^2}-1
\right).
\]

它解决问题的方式是：

- 已被当前 state 解释的尺度信息满足 \(r_s\approx0\)，自然产生近零更新；
- 新增或冲突信息产生带符号的 feature-space correction；
- 极端残差的影响被 \(\psi_\delta\) 有界化，不能自由覆盖已有 state；
- correction 经空间—通道耦合的 \(K^*\) 回写，不受限于 state 和某个 candidate 之间的逐点线段；
- 最细状态直接输出，不再进行四尺度静态后融合。

## 4. 共享输出和训练闭环

所有尺度使用同一个 readout：

\[
z_s=Q_\theta h_s.
\]

最终预测为：

\[
z_{\mathrm{final}}=z_0.
\]

当前 MSHNet 的五项平均损失等价于 full-resolution 权重 0.4，以及 \(1/2,1/4,1/8\) 三个尺度各 0.2。新模型不复制 full-resolution head，而是保持相同的有效分辨率权重：

\[
\mathcal L
=
0.4\mathcal L_{\mathrm{SLS}}(z_0,Y_0)
+0.2\mathcal L_{\mathrm{SLS}}(z_1,Y_1)
+0.2\mathcal L_{\mathrm{SLS}}(z_2,Y_2)
+0.2\mathcal L_{\mathrm{SLS}}(z_3,Y_3).
\]

其中 \(Y_1,Y_2,Y_3\) 复用当前代码的递归 max-pooling。\(z_4\) 只作为最粗 prefix，不单独增加损失。warm-up 期间与 MSHNet 一样，只监督最终 \(z_0\)。

## 5. 退化路径和必须记录的状态量

必须记录每个尺度：

\[
R_s=
\frac{\|h_s-\bar h_s\|_2}
{\|h_s\|_2+\epsilon},
\]

以及 residual mean、correction mean、\(\delta\) 的最小值/均值/最大值。

需要识别的退化包括：

- \(A_s\to0\) 或 correction 全零：模型忽略尺度；
- \(\delta\to0\)：所有 correction 被压没；
- \(\delta\to\infty\)：鲁棒函数退化为线性残差；
- \(K\) 退化为恒等映射：模型接近逐点插值控制；
- 非共享 \(K_s\) 明显优于共享 \(K\)：说明五个尺度没有形成共同状态坐标系；
- correction ratio 长期接近或大于 1：后尺度仍可能主导状态，需要降低 \(\eta\) 或重新约束 \(K\)。

---

# 七、当前代码与训练状态

实现位置：

```text
model/predictive_correction_mshnet.py
tests/test_predictive_correction_mshnet.py
```

训练入口：

```text
--model-type predictive_correction
--predictive-state-channels 32
--predictive-step-size 1.0
```

已经通过的 mechanics：

- 五个状态/logit 尺寸正确；
- 数值内积验证 \(K^*\) 是当前 \(K\) 的伴随；
- `warm_flag` 只改变辅助监督契约，不改变 `pred`；
- adapter、共享 \(K/K^*\)、\(\delta\)、prior 和 readout 均有非零有限梯度；
- 模型不存在旧 `decoder_*`、`output_*`、`final`；
- 256 输入、batch 2 的一次 forward/backward 有限，峰值显存约 482 MiB；
- C32 总参数 2,837,267，其中新 decoder 17,345；原 MSHNet 总参数 4,066,034。

当前正在同一 NUAA clean split、同一 seed、同一 optimizer/lr/warm-up 下并行训练：

| 运行 | 作用 |
|---|---|
| MSHNet | 原结构对照 |
| PredictiveCorrection-C32, \(\eta=1\) | 主结构 |
| PredictiveCorrection-C64, \(\eta=1\) | 容量控制 |
| PredictiveCorrection-C32, \(\eta=0.5\) | 更新步长控制 |

epoch 0 只证明可优化，不能作为最终结论：C32 的 validation IoU 为 0.2764，MSHNet 为 0.0006。是否保留该结构必须看完整训练趋势和同一 selection rule 下的最佳 validation IoU、PD、FA。

---

# 八、当前结论

- MSHNet 问题诊断继续保留；DEAIntegrated 继续作为旧控制，不再扩展。
- 原 CSIR/PSIR 已废弃，因为它们仍属于 candidate interpolation。
- 当前主模型已经形成完整 forward graph：五级 encoder observation 驱动一个共享多通道 predictive-correction state，最细状态直接输出。
- 这不是在 MSHNet 后面叠模块，也不是只替换 final fusion，而是删除完整原 decoder 后建立新的单轨状态 decoder。
- 当前只能确认结构闭合、可反向传播且真实数据可进入有效解；能否提升性能等待正在运行的配对训练结果。
