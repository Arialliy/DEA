# 伴随预测误差回写主模型设计评审

## 结论

**这个想法在机制上是对的，而且比“候选预测 \(e_s\)+门控 \(g_s\)”的 CSIR 更统一、更可解释。**

但要准确地说：

> 它不是一个“递归融合模块”，而是一个对 encoder 多尺度观测执行的、尺度共享的鲁棒预测误差回写过程。

数学上，它对应一次**广义鲁棒 Landweber／预测编码更新**。不过，**机制正确不等于已经构成顶会创新**：预测误差回写、伴随算子、跨阶段参数共享和展开更新都有明确先例。你的潜在新意必须集中在：

> **把 encoder pyramid 定义为同一个潜在状态的有序多分辨率观测，并用唯一的、严格前向—伴随一致的更新规律完成解码，而不是设计新的融合模块。**

这个定位是成立的。

---

## 零、这个模型究竟要解决 MSHNet 的什么问题

当前工作的直接目标不是先追一个 SOTA 数字，而是检验并解决下面这个具体结构问题：

> **MSHNet 虽然具有真正的粗到细层级 decoder，但每一级通过不同的 decoder block 自由重写特征，随后从四个 decoder states 生成四个 side logits，再由一个固定的 \(3\times3\) convolution 完成终端融合。网络没有显式约束“当前累计状态对新尺度已经解释了什么”，也没有把未解释部分定义成唯一允许写入状态的 innovation。**

因此不能把 MSHNet 描述成“四尺度独立”或“完全不会自适应”。它的 encoder、decoder features 和 side predictions 都依赖输入。更准确的潜在缺陷是：

1. 每个 decoder stage 都有自己的非共享空间变换，新尺度可以自由改写已有状态；
2. 四个 side logits 分别监督后才被终端卷积组合，最终融合发生在已完成的预测上；
3. 冗余、支持和冲突证据没有由同一个状态相关准则定义；
4. 最终融合能学习平均组合规律，但没有显式的 observation-consistency 约束。

本模型针对的正是“**多尺度证据缺乏受约束的状态吸收规律**”，而不是泛泛地增加注意力或融合能力。它作出的核心假设是：

> 如果五级 encoder features 能被映射成同一潜在场景状态的多分辨率观测，那么每一级只需回写当前状态无法预测的鲁棒残差；已被解释的证据自然不重复写入，极端冲突的证据受到有界影响函数限制。

这仍是可被否证的模型假设，而不是已经确认的 MSHNet 缺陷。模型是否真正解决问题，至少要同时满足：

- 优于同协议 MSHNet，而不是只更容易优化；
- 自然粗到细递推优于并行、反向或随机次序；
- exact \(K/K^\ast\) 优于参数量匹配的自由 back-projection；
- 每次回写确实降低当前尺度 observation energy；
- PD 提升不能以不可接受的 FA 增长为代价。

---

## 一、应当写成下面这个严格模型

MSHNet 编码器本身正好提供五个分辨率：

\[
F_4=x_m,\quad F_3=x_{e3},\quad F_2=x_{e2},\quad
F_1=x_{e1},\quad F_0=x_{e0},
\]

通道数分别为：

\[
256,\;128,\;64,\;32,\;16.
\]

现有代码随后才通过四级 decoder、四个 side heads 和 \(4\to1\) final convolution 生成输出，因此删除这些部分、保留五级 encoder observation 在工程上是清晰可行的。

令 \(h_s\) 为固定通道数 \(C\) 的共享潜在状态。当前实现令 observation 与 state 位于同一个 \(C\) 维坐标空间，即 \(C_y=C_h=C\)，并令每个尺度只有一个必要的线性坐标适配器：

\[
y_s=N(A_sF_s),\qquad
A_s:\mathbb R^{C_s}\rightarrow\mathbb R^{C}.
\]

其中 \(A_s\) 是每个尺度不可避免的 \(1\times1\) 线性坐标适配器，不提供空间建模；\(N\) 是五个尺度共用的同一个 GroupNorm，而不是五套尺度专属模块。需要准确区分：\(A_s\) 是线性的，但使用样本与空间统计量的 \(N\circ A_s\) 整体不是线性映射；后文局部能量只定义在这一归一化后的学习坐标中。

粗尺度从常数先验开始：

\[
\bar h_4=b,
\]

其余尺度使用固定上采样：

\[
\bar h_s=U(h_{s+1}),\qquad s=3,2,1,0.
\]

然后所有尺度严格执行同一个更新：

\[
\hat y_s=K\bar h_s,
\]

\[
r_s=y_s-\hat y_s,
\]

\[
h_s
=
\bar h_s+\eta K^\ast\psi_\delta(r_s),
\]

最终只输出：

\[
z=Oh_0+o.
\]

这里：

- \(K:\mathbb R^C\rightarrow\mathbb R^C\) 是共享的方形线性卷积；
- \(K^\ast:\mathbb R^C\rightarrow\mathbb R^C\) 是 \(K\) 的**严格数学伴随**；
- \(\psi_\delta\) 是共享鲁棒影响函数；
- \(\eta>0\) 是共享步长；
- \(O:C\rightarrow1\) 和 bias \(o\) 构成唯一的仿射输出映射；
- \(U\) 应当是固定的 bilinear 或 nearest upsampling，而不是另一套 learned decoder。

这才是真正的“单一机制”。

---

## 二、为什么这个更新在数学上成立

定义当前尺度的鲁棒观测一致性目标：

\[
E_s(h)
=
\sum_{p,c}
\rho_\delta
\left(
y_s-Kh
\right)_{p,c},
\]

并令：

\[
\psi_\delta(r)=\rho_\delta'(r).
\]

则：

\[
\nabla_hE_s(h)
=
-K^\ast\psi_\delta(y_s-Kh).
\]

所以：

\[
h_s
=
\bar h_s
-
\eta\nabla_hE_s(\bar h_s)
=
\bar h_s+\eta K^\ast\psi_\delta(r_s)
\]

恰好是从上一尺度状态出发，对当前尺度观测一致性执行的一步梯度修正。

因此，“先预测当前尺度 encoder observation，再把预测不了的残差通过 \(K^\ast\) 回写”不是比喻，而是有明确优化含义的更新。

它相较原 CSIR 有四个实质性改进：

1. 不再产生一个自由的 candidate prediction；
2. 不再学习 Increase／Decrease／Keep 三分类；
3. 状态修正必须具有与前向权重绑定的 \(K^\ast\psi(y-Kh)\) 形式，不能由独立 residual head 任意生成；由于当前 \(K\) 是方阵且可能满秩，不能把这一约束夸大成低维子空间限制；
4. 当前残差依赖正在递推的 \(h_s\)，因此是真正的 state-conditioned recursion，而不是对预计算 delta 求和。

---

## 三、必须修正的几个关键点

### 1. \(K^\ast\) 必须是真正的伴随，不能是另一个可学习卷积

这是这个设计是否成立的分界线。

若代码中写成：

```python
prediction = conv_k(state)
correction = conv_back(residual)
```

但 `conv_k` 与 `conv_back` 参数独立，那么它只是一个普通 residual decoder，不能称为伴随回写。

正确实现应当是：

```python
prediction = F.conv2d(state, weight_k, padding=padding)
correction = F.conv_transpose2d(
    residual,
    weight_k,
    padding=padding,
)
```

对于 stride \(=1\)、匹配 padding 的情况，`conv_transpose2d` 使用同一权重实现 `conv2d` 的线性伴随。

必须加入单元测试：

\[
\frac{
\left|
\langle Kh,r\rangle-\langle h,K^\ast r\rangle
\right|
}{
|\langle Kh,r\rangle|+
|\langle h,K^\ast r\rangle|+\epsilon
}
<10^{-6}.
\]

只要加入 BatchNorm、ReLU、SiLU 或其他非线性到 \(K\) 内部，就不再存在这个简单的严格伴随。因此：

> \(K\) 应保持为纯线性卷积，所有非线性只出现在残差的影响函数 \(\psi\) 中。

---

### 2. 多通道状态不应称为“目标状态”

如果 \(h_s\) 只是 target-mask belief，它通常没有足够信息预测完整的 encoder feature，因为 encoder feature 同时包含：

- 背景结构；
- 局部纹理；
- 目标响应；
- 传感器噪声；
- 深层上下文。

因此更准确的语义是：

> \(h_s\) 是一个共享的 latent scene-evidence state，最终由 \(O\) 从中读取 target logit。

否则审稿人会提出一个合理质疑：

> 为什么仅表示目标的状态能够重建背景占主导的 encoder observation？

另一种说法是：\(A_sF_s\) 不是原始 encoder feature，而是由线性适配器投影出的**任务相关观测坐标**。但这时必须承认，\(A_s\) 承担了将 encoder feature 转换成可预测目标证据的职责。

---

### 3. 五个 \(A_s\) 仍然是尺度特定参数

它们虽然只是 \(1\times1\) coordinate adapters，但不能写成“模型完全没有尺度特定参数”。

准确表述应是：

> 除了处理输入通道数差异所必需的线性坐标映射 \(A_s\)，所有具有空间建模或非线性推理能力的算子均在尺度间共享。

建议限制 \(A_s\)：

- 只允许 \(1\times1\)；
- 不允许后接尺度特定的 \(3\times3\) convolution；
- 不允许尺度特定 attention；
- 不允许尺度特定 MLP；
- 参数量和计算量单独报告；
- 最好对输出范数进行约束或校准。

原因是各 encoder scale 的特征幅值和统计分布可能不同。如果不校准，而 \(\psi_\delta\) 的阈值又严格共享，那么某些尺度会仅仅因为数值幅值更大而主导更新。

可以让适配器自身完成线性增益校准，或者加入**所有尺度共用的同一个规范化**；若规范化包含 affine 参数，这些参数也必须共享，不能另外设计五套 normalization block。当前实现采用一个共享 GroupNorm。

---

### 4. 鲁棒影响函数不能再变成一个自由网络

初版建议使用明确的奇函数、单调函数和 Lipschitz 函数，例如 pseudo-Huber：

\[
\psi_\delta(r)
=
\frac{r}{
\sqrt{1+(r/\delta)^2}
}.
\]

或者 Huber influence：

\[
\psi_\delta(r)
=
\operatorname{clip}(r,-\delta,\delta).
\]

其中 \(\delta\) 可以是少量共享的 channel-wise 参数，但不能由每个尺度的独立网络预测。

这意味着：

- 小残差近似线性传递；
- 大残差被限制幅值；
- 更新符号由 observation innovation 决定；
- 不需要再命名 Increase／Decrease／Keep。

需要注意，Huber 并没有结构性的精确 Keep；只有残差为零时更新才为零。不要为了保留旧的 I/D/K 叙事，强行加入 hard dead zone。是否需要稀疏拒绝更新应当作为独立消融，而不是先验结论。

---

### 5. 步长和 \(K\) 的谱范数必须受到约束

如果 \(\psi\) 是 \(L_\psi\)-Lipschitz，当前尺度目标的梯度 Lipschitz 常数满足：

\[
L_E
\le
L_\psi\|K\|_2^2.
\]

因此使用：

\[
0<\eta<
\frac{2}{
L_\psi\|K\|_2^2
}
\]

可以保证一次更新不会增加当前尺度的观测一致性目标。

工程上可用：

- spectral normalization；
- power iteration 估计 \(\|K\|_2\)；
- 受限共享步长；
- identity-like initialization。

但只能声称：

> 每一步对当前尺度的 observation fidelity 具有下降性质。

不能声称整个五尺度网络全局收敛，因为每一步使用的 \(y_s\) 不同，而且尺度之间还有固定上采样。

---

## 四、严格共享 \(K/K^\ast\) 是合理假设，但不是天然事实

同一个 \(3\times3\) kernel 在不同 feature resolution 上对应不同的原图感受野。因此“所有尺度共享 \(K\)”是一个很强的建模假设：

> 五个 \(A_s\) 能否把不同 encoder levels 映射到一个具有统一局部统计和统一误差尺度的 observation space？

这必须通过消融验证。

关键对照是：

\[
\text{shared }K
\quad\text{vs.}\quad
\{K_s\}_{s=0}^{4}.
\]

可能出现三种结果：

- shared \(K\) 与 non-shared \(K_s\) 接近或更好：支持统一观测规律；
- non-shared 明显更好：说明尺度间不存在足够统一的预测规律；
- 两者都不如简单 attention：说明问题不适合用预测误差回写解决。

不能在 non-shared 明显优于 shared 后，再通过增加 scale embeddings、scale-specific threshold 和 scale-specific correction blocks 来“补救”；那会重新退化为模块堆叠。

---

## 五、创新性判断：比 CSIR 更有希望，但仍不能直接称为新

这个设计涉及的基本成分都有先例：

- DBPN 已通过 projection error 和 back-projection 实现误差反馈；
- Learned Primal-Dual 把 forward operator 和相应反投影写入展开网络；
- MoDL 已强调跨迭代共享网络参数和显式 data consistency；
- RPCANet 已把深度展开用于红外小目标检测。

还要特别注意：**LCPNet 于 2026 年 7 月 6 日发布**，已经在 IRSTD 中提出 latent-domain unfolding、连续 latent-state 更新和 shared optimization memory。它与这里的设计不是同一机制，但使“潜在状态＋共享递推＋优化解释”这一宽泛创新表述基本失效。

你和 LCPNet 的差异必须明确限定为：

- LCPNet 建模背景、目标、噪声的 latent decomposition；
- 你的模型不做 RPCA decomposition；
- 你的五个输入是 encoder pyramid 的异构尺度观测；
- 只有一个共同状态，而不是多个分解变量；
- 每一步由同一 \(K\) 预测当前尺度观测；
- 回写算子是与 \(K\) 权重严格绑定的 \(K^\ast\)；
- 五个尺度是有序的观测序列，而不是对同一分解目标重复执行的普通 unfolding stages。

因此，潜在贡献不能写成：

> 我们提出了共享递归 decoder 或预测误差反馈。

而应写成：

> 我们将 encoder pyramid 重新解释为对同一个潜在场景状态的有序多分辨率观测，并通过尺度共享、严格伴随一致的预测误差更新替代传统 decoder 和多尺度输出融合。

---

## 六、这套机制成立所需的实验事实

在把该结构确立为主模型并形成正式机制结论前，至少需要以下证据。当前 400 epoch 训练属于 mechanics exploratory run，不代表这些 gate 已被越过。

首先，原来的 D1/D2 仍然必须成立：

- MSHNet 的错误确实与尺度贡献冲突相关；
- frozen scale intervention 确实存在可修复的 FN 和 FP 空间。

其次，需要证明**顺序递推有价值**。否则这个模型可以被替换成并行残差求和：

\[
h_{\mathrm{parallel}}
=
U_{4\rightarrow0}b
+\sum_{s=0}^{4}
U_{s\rightarrow0}
\left[
\eta K^\ast\psi\!\left(y_s-Kb_s\right)
\right],
\]

其中 \(b_s\) 是同一先验在尺度 \(s\) 的广播，\(U_{s\rightarrow0}\) 将各尺度 correction 映射到共同的最细分辨率。reverse/random 对照也必须为任意相邻处理尺度显式定义无参数 resize \(R_{t\rightarrow s}\)，不能把不同空间尺寸的张量直接相加。

应比较：

- coarse-to-fine；
- fine-to-coarse；
- 随机顺序；
- 并行一次性聚合。

只有当粗到细顺序稳定优于反向和随机顺序时，才能说明“当前状态预测下一尺度 observation”具有必要性。

最后，必须包含以下机制消融：

- \(K^\ast\) vs 独立 learned back-projection \(B_s\)；
- shared \(K\) vs non-shared \(K_s\)；
- robust \(\psi\) vs identity；
- exact adjoint vs 参数量匹配的普通 convolution；
- state-conditioned residual vs 不读取当前状态的直接 encoder aggregation；
- 本方法 vs soft attention；
- 本方法 vs ConvGRU/CRDN；
- 本方法 vs 原 CSIR interpolation；
- coarse-to-fine vs reverse/random；
- 四尺度 encoder vs 加入 `middle_layer` 的五尺度 encoder。

还应记录逐图、逐尺度的局部机制指标：

\[
E_s(h_s)-E_s(\bar h_s),
\]

用于检查实现是否违反当前尺度的解析下降约束，并记录最大能量增量与 violation fraction；以及

\[
\frac{
\|y_s-Kh_s\|
}{
\|y_s-K\bar h_s\|
},
\]

用于描述当前 observation residual 被校正的程度。此外还应记录

\[
\frac{\|h_s-\bar h_s\|_2}{\|\bar h_s\|_2+\epsilon}
\quad\text{和}\quad
\cos(h_s-\bar h_s,\bar h_s),
\]

分别量化新尺度对既有状态的覆盖强度，以及该更新是在支持、正交补充还是抵消既有状态。

必须明确：局部 observation energy 下降是当前 \(K\)、\(K^\ast\)、\(\psi\) 和步长约束带来的结构性质，只能作为数值与实现 sanity check。它不是五尺度联合能量、分割损失、旧证据保留度，也不能用来选择 checkpoint 或证明模型有效；更新完全可能在降低当前局部能量的同时拟合噪声或遗忘粗尺度信息。

---

## 七、当前实际实现与第一轮训练事实

### 7.1 它已经不是概念图，而是完整替换了 MSHNet decoder

当前实现位于 `model/predictive_correction_mshnet.py`，训练入口为 `main.py --model-type predictive_correction`。实际网络保留 `conv_init`、四级 encoder 和 `middle_layer`，完整删除以下 MSHNet 路径：

- `decoder_0` 至 `decoder_3`；
- `output_0` 至 `output_3`；
- `final` 的四尺度预测融合；
- 与本主模型无关的 `decidability_head`。

替代路径不是在旧 decoder 旁外挂多个独立增强块，而是五个必要的线性坐标适配器加一套复用五次的共享状态转移。五个 \(A_s\) 是唯一的尺度特定参数：

\[
F_s\rightarrow A_s\rightarrow N\rightarrow y_s,
\]

\[
\bar h_s\rightarrow K\bar h_s\rightarrow r_s
\rightarrow\psi_\delta(r_s)\rightarrow K^\ast\psi_\delta(r_s)
\rightarrow h_s.
\]

其中：

- 五个 \(A_s\) 只负责把 \(256/128/64/32/16\) 通道映射到同一个状态坐标；
- GroupNorm、\(K\)、\(K^\ast\)、\(\delta\)、\(\eta\) 和 readout 在所有尺度共享；
- \(K=MD\)，\(D\) 是 depthwise \(3\times3\)，\(M\) 是 pointwise \(1\times1\)；
- \(K^\ast=D^\ast M^\ast\) 直接复用前向权重，通过 `conv_transpose2d` 实现，不存在独立 learned back-projection；
- \(D\) 使用逐通道 \(\ell_1\) 增益上界，\(M\) 使用精确矩阵谱范数上界，当前主配置 \(\eta=1\)；
- 鲁棒影响函数为逐通道可学习阈值的 pseudo-Huber：

\[
\psi_\delta(r)=\frac{r}{\sqrt{1+(r/\delta)^2}}.
\]

推理输出仅采用同一个 readout 对 \(h_0\) 的读取。训练期该 readout 还读取 \(h_1,h_2,h_3\) 做辅助监督；\(z_4\) 虽被计算但不监督。这不是五个独立 side heads。warm-up 后的损失为：

\[
0.4\mathcal L(z_0,Y_0)
+0.2\mathcal L(z_1,Y_1)
+0.2\mathcal L(z_2,Y_2)
+0.2\mathcal L(z_3,Y_3),
\]

最粗的 \(z_4\) 只作为递推前缀，不直接监督。这保持了原 MSHNet 各分辨率的有效监督权重，但没有复制其 full-resolution head。

### 7.2 结构和数学约束已经通过的检查

截至当前，相关定向命令为 22 passed：其中 11 项直接覆盖 predictive-correction，另 11 项验证其复用的主训练入口与共享行为。直接覆盖项包括：

- 非方形输入的五级张量尺寸和最终输出尺寸；
- 所有新增参数均获得有限且非零的梯度；
- pseudo-Huber influence 的幅值受 \(\delta\) 约束；
- 在随机非对称 depthwise kernel 和随机 channel mixing 下，数值内积检验仍满足 \(\langle Kx,y\rangle=\langle x,K^\ast y\rangle\)，相对误差小于 \(10^{-6}\)；
- 在非 identity 随机算子下，每一级实际回写后的鲁棒 observation energy 均不高于回写前；
- checkpoint 元数据、CLI 约束和 baseline encoder 部分加载路径。

另有 35 项数据划分、PD/FA、Full-DEA 路径及 release 回归测试通过。因此当前明确运行的结果为 57 passed、0 failed。

在状态宽度 \(C=32\) 时，模型总参数量为 2,837,267，其中替换 decoder 的路径共 17,345 个参数；本仓库 MSHNet 为 4,066,034 个参数。17,345 个参数的透明拆分为：五个 adapter 15,872，shared GroupNorm 64，shared \(K/K^\ast\) 权重 1,312，prior 32，\(\delta\) 32，readout 33。adapter 占该路径的 91.5%，但它们只有 \(1\times1\) 线性坐标变换能力。这一变化是 decoder 拓扑替换，不是给 MSHNet 旁挂注意力或路由模块。

### 7.3 NUAA 划分与 100 epoch 中期结果

NUAA-SIRST 共 427 张图像：官方训练列表 213 张、官方测试列表 214 张。本轮模型选择只把官方 213 张训练集固定拆为 170 train / 43 validation；官方 214 张 test 完全没有参与模型选择。因此“170/43”不是 NUAA 的总规模。

四组实验使用相同 split、seed、batch size、Adagrad、学习率、warm-up 和确定性设置。100 epoch 的 validation 最优结果为：

| 模型 | 最优 epoch | IoU | PD | FA |
|---|---:|---:|---:|---:|
| MSHNet baseline | 88 | 0.6490 | 0.9048 | 10.2908 |
| 本模型，\(C_h=32,\eta=1\) | 98 | **0.7192** | 0.9683 | 14.5491 |
| 本模型，\(C_h=64,\eta=1\) | 98 | 0.7099 | **0.9841** | 26.6142 |
| 本模型，\(C_h=32,\eta=0.5\) | 91 | 0.6811 | 0.9524 | 18.0976 |

相对同协议 baseline，当前主配置 \(C_h=32,\eta=1\) 的中期 IoU 提升 0.0702、PD 提升 0.0635，但 FA 增加 4.2583。它说明该结构已经产生了明显不同且更有效的优化行为，也说明“抑制虚警”尚未被证明，不能仅凭 IoU 宣布模型成立。

四组训练目前正在同协议续跑至 400 epoch。最终去留要依据完整性能曲线与多指标 Pareto 关系，以及自然顺序/反向顺序、exact adjoint/独立 back-projection 等关键对照决定。逐级局部 energy 只负责检查实现是否遵守所定义的更新，不参与性能判优。

---

## 最终判断

**对，这可以收敛为一个单一、清晰且可实现的主机制。**

而且它比原 CSIR 更强，因为：

- 不再有自由 candidate；
- 不再有三分类 router；
- 不再有任意 residual update；
- 不再保留 baseline fusion 旁路；
- 五个尺度确实通过正在变化的状态发生递推；
- 原 decoder、四 heads 和 final fusion 可以全部删除。

但需要采用下面这句作为最终机制定义：

> **模型将五级 encoder features 视为同一潜在场景状态的有序多分辨率观测。每个尺度先由共享线性观测算子预测其坐标对齐后的 encoder observation，再将鲁棒化的预测误差仅通过该算子的严格伴随回写到共享状态；所有尺度执行同一更新律，最终只从最细状态读取一次目标预测。**

它目前是一个**合理且比 CSIR 更有潜力的主模型假设**，还不是已经成立的顶会贡献。决定它是否值得实现的关键，不是结构看起来是否漂亮，而是：

1. baseline 是否确实存在可修复的尺度冲突；
2. 自然粗到细顺序是否比并行或反向处理更有效；
3. 严格 \(K/K^\ast\) 约束是否优于普通 learned residual decoder；
4. shared \(K\) 是否没有被 non-shared \(K_s\) 明显击败。
