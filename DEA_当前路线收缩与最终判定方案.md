# DEA 当前路线收缩与最终判定方案

## 结论（修订后）

你说得对。**按现在“提出一个 correction 分支—跑控制—失败—再提出另一个分支”的方式，确实会无限慢。**  
现在需要停止结构发散，而不是继续提高设计复杂度。

但必须同时纠正一个关键判定：

> **“mean-anchor SIED 信号稳定”只能证明当前问题假设值得继续，不能直接证明 SIED 就是最终 DEA。**

 原因有四个：

1. 当前最好结果只是 43 张 design-val 上的 `+0.002135 IoU`，它只是开发信号，不能代表官方 214-test 的最终性能；
2. 只改 `decoder_0` 不会再影响更细的 decoder state，因此它是“预测头之前的最终特征修正”，还不是完整的 coarse-to-fine recurrence 重写；
3. 若把 \(\alpha\) 改成可学习标量，\(q^{11}+\alpha(j-p)\) 仍可被审稿人合理地归类为 global residual gate/ReZero 变体，不足以单独承担顶会结构创新；
4. 在 `1×3×256×256` 上的 Conv-MAC 实测为 MSHNet `6.0015392 G`、decoder0-only SIED `7.983741248 G`，增量 `33.03%`，已高于已约定的 25% 计算量上界。

因此，这份方案的用途应改为：

```text
用一次性审计判定“Mean-Centered Decoder Interaction Imbalance”是否真实稳定
→ 稳定：问题假设 H-GO，只允许再设计一个真正重写 decoder fusion/recurrence 的 DEA
→ 不稳定：问题假设 H-NO-GO，终止整条路线
```

**当前 SIED 的定位是最后一个问题诊断探针，不是已经设计完成的 DEA 主模型。**

---

## 一、为什么模型设计会这么慢

### 1. 当前 NUAA 验证集几乎已经饱和

baseline 是 63 个目标中检出 62 个：

\[
0.984127=\frac{62}{63}.
\]

PD 每变化一个目标就跳变：

\[
\frac{1}{63}=0.015873.
\]

也就是说，当前所有方案实际上都在围绕：

- 1 个 FN；
- 4 个 FP；

做架构选择。

例如：

- shared stencil 从 0.984127 降到 0.968254，本质上就是多丢了一个目标；
- mean-anchor SIED 的 \(+0.002135\) IoU，也可能只来自极少数像素或一个组件。

因此，这个验证集只适合做机制筛选和错误定位，**不适合做结构搜索，也不能再用于最终确认**。它已经参与 anchor、\(\alpha\) 和多个候选的选择，后续在该 split 上的结果都必须标注为 `design-used mechanics evidence`。

### 2. 43 张不是完整 NUAA，也不能代表最终 PD 分辨率

本地数据清单已核对：

| NUAA-SIRST 角色 | 图像数 | 当前用途 |
|---|---:|---|
| 官方 train | 213 | 开发阶段再分为 170 train / 43 validation |
| 官方 test | 214 | 最终独立测试，当前应封存 |
| 总计 | 427 | 不能将训练图和测试图混在一起报告泛化性能 |

仓库中：

- `train_NUAA-SIRST.txt` 为 213 张；
- `test_NUAA-SIRST.txt` 为 214 张；
- 两个清单交集为 0，并集为 427；
- `utils/data.py` 在训练模式下对 213 张按 `val_fraction=0.2` 做确定性拆分，所以得到 170/43。

因此，“PD 每丢一个目标变化 \(1/63\)”只是当前 43 张开发集的局部事实，不是 NUAA 数据集或最终官方 test 的性能上限。最终 PD 必须用 214 张 test 中的实际目标数重新计算。

### 3. 开发 baseline 与官方 baseline 必须分账本

当前存在两种 MSHNet 结果，它们不能直接横向比较：

| 账本 | 训练/评估划分 | 已核对的代表 checkpoint | 用途 |
|---|---|---|---|
| `MSHNet-dev-historical` | 170 train / 43 validation | epoch 258，IoU `0.747056`；checkpoint 已清理 | 仅解释旧 mechanics，不进入新主表 |
| `MSHNet-official-historical` | 213 official train / 214 official test | 旧 run epoch 381，IoU `0.746177`；checkpoint 已清理 | 仅作历史协议说明，不再复用 |
| `MSHNet-clean-v1` | 三数据集 official-train 内固定 80/20 holdout，3 seeds | 2026-07-11 起从零重训 | 新模型设计与 paired 开发比较的唯一 baseline |

epoch-258 checkpoint 的 `method_meta` 和 split manifests 明确记录了 170/43；旧版 MSHNet 数据加载代码则使用全部 213 张训练清单，并把 214 张 test 作为训练期 `val` 逐 epoch 评估。

从此比较规则锁定为：

1. 开发阶段的 DEA/control 只能和**同一 170/43 split、同 checkpoint、同 threshold** 的 `MSHNet-dev` 比较；
2. 最终 DEA 只能和**同一 213/214 split、同 seed、同 optimizer、同 augment、同 epoch/选模规则**重新训练的 `MSHNet-official` 配对比较；
3. 不得用 170 张训练出的 DEA 对比 213 张训练出的 MSHNet，也不得用 43-val 的 `0.747056` 对比 214-test 的任何数字；
4. 213-train 内部 5-fold OOF 是稳定性诊断账本，不能直接和论文中 214-test 的 MSHNet 数字比较。

原开源协议在训练期反复查看 214 test，因此严格说它存在 test-based checkpoint selection。最终应同时保留两种可审计结果：

- **protocol-reproduction**：DEA 和 MSHNet 都完全复刻原 213/214 选模流程，用于与开源 baseline 对齐；
- **leakage-resistant paired result**：在 170/43 开发阶段锁定 epoch 和全部超参，再用 213 张重训 MSHNet/DEA，对 214 test 只评估一次。这一账本作为更严格的主消融证据。

### Clean baseline v1 已冻结的开发协议

历史 `weight/`、`repro_runs/` 和 `results/` 生成物已清空。当前 baseline 不加载任何旧 checkpoint，统一从随机初始化训练：

| 数据集 | official train | 固定 fit | 固定 validation | official test |
|---|---:|---:|---:|---:|
| NUAA-SIRST | 213 | 170 | 43 | 214 |
| NUDT-SIRST | 663 | 530 | 133 | 664 |
| IRSTD-1K | 800 | 640 | 160 | 201 |

固定配置为：`split_seed=20260711`，训练 seeds 为 `20260711/20260712/20260713`，400 epochs，Adagrad，`lr=0.05`，warm-up 5 epochs，batch size 4，threshold 0.5。三个数据集均从仓库各自的 `img_idx/train_*.txt` 读取 official train；`test_*.txt` 在本阶段只用于 ID 交集/hash 审计，不构造 test forward，不参与 checkpoint 选择。

该账本的目的，是用三数据集、三 seed 的 clean predictions 定义 MSHNet 的真实失败模式并设计 DEA；它仍属于 design-used holdout evidence。候选锁定后的 `official-full` fixed-last 重训和一次性 test 必须另立最终账本。

---

### 4. 当前候选同时背负了过多要求

每个候选都被要求同时满足：

- 严格退化为 MSHNet；
- 在预测头之前改变 decoder state；
- 不是 gate、attention 或普通 residual；
- 有足够的新颖性；
- 还必须立即涨点。

这些标准本身是正确的，但不能每一轮都重新发明一个模型来同时碰运气。

当前 90 项测试通过，只能说明：

> 工程契约可靠。

它不能增加统计证据，也不能证明结构假设成立。

---

### 5. 已有结果已经足够进行一次硬裁剪

当前应正式停止：

- MA-PCID；
- Shared Discrepancy Stencil；
- CEV；
- SMEI；
- Decoder-Jacobian；
- 其他新的 correction branch。

当前真实结果是：

| 方案 | IoU | PD | FA/M | 判断 |
|---|---:|---:|---:|---|
| MSHNet | 0.747056 | 0.984127 | 9.5811 | baseline |
| mean-anchor SIED，\(\alpha=0.2\) | 0.749191 | 0.984127 | 9.5811 | 43-val 微弱诊断信号，不是最终性能 |
| MA-PCID，\(\alpha=1\) | 0.201576 | 0.714286 | 17.0331 | 强 NO-GO |
| 8 参数 shared stencil 最佳学习态 | 0.745749 | 0.984127 | 9.9360 | 未超过 baseline |
| stencil 安全门触发时 | 0.746856 | 0.968254 | 32.6467 | NO-GO |

因此，目前不能诚实地说 DEA 已经设计完成。当前 SIED 只有 design-val 信号，官方性能门尚未评估；而它的原样实现已经超出计算量边界，所以不具备“最终主模型”资格。

---

# 二、在问题假设判定前，不再设计新模型

当前唯一没有被完全否定的假设是：

> **在最细 decoder transition 中，以保留通道 DC、去除空间证据的 mean anchor 定义独立项和联合项，可能存在一个稳定的交互失衡。**

注意，这已经不再是宽泛的：

- 尺度证据混叠；
- 所有尺度交互；
- 任意 anchor 下的 interaction；
- 通用 correction-state 理论。

它应该被收缩为：

## Mean-Centered Decoder Interaction Imbalance

即：

> **最细 decoder transition 中，当前尺度独立空间响应与粗细联合响应之间，可能存在稳定的失衡。**

现有 SIED 代码已经足够验证这一点，不需要再写新模型。这一阶段的输出是“问题假设 GO/NO-GO”，不是新模型命名或论文故事。

---

# 三、当前唯一剩余的候选公式

设当前尺度 encoder skip 为 \(e\)，继承的粗尺度 decoder state 为 \(u\)，原 decoder 为：

\[
D(e,u).
\]

使用 mean anchor：

\[
\bar e=\operatorname{sg}\left[\operatorname{Mean}_{xy}(e)\right],
\]

\[
\bar u=\operatorname{sg}\left[\operatorname{Mean}_{xy}(u)\right].
\]

计算四个 coalition：

\[
q^{11}=D(e,u),
\]

\[
q^{10}=D(e,\bar u),
\]

\[
q^{01}=D(\bar e,u),
\]

\[
q^{00}=D(\bar e,\bar u).
\]

定义当前尺度独立项：

\[
p=q^{10}-q^{00},
\]

继承粗尺度主效应：

\[
c=q^{01}-q^{00},
\]

联合非加性交互：

\[
j=q^{11}-q^{10}-q^{01}+q^{00}.
\]

代数上：

\[
q^{11}=q^{00}+p+c+j.
\]

SIED transition 为：

\[
\boxed{
\hat d
=
q^{11}+\alpha(j-p)
}
\]

即：

\[
\hat d
=
q^{00}+c+(1-\alpha)p+(1+\alpha)j.
\]

当前唯一存在弱正信号的是：

```text
active stage = decoder_0
anchor = mean
alpha = 0.2
MSHNet = frozen
```

不要把它强行推广到四个 decoder stage。

---

# 四、为什么现在的问题不再是“下一个模型长什么样”

现在真正需要回答的是：

> **mean-anchor SIED 的唯一正信号，是否跨 checkpoint、跨 seed、跨数据集稳定存在？**

若稳定，只说明“mean-centered interaction imbalance”成为唯一保留的问题假设。

若不稳定，说明这个信号只是：

- 单 checkpoint 偶然性；
- anchor artifact；
- 少量像素变化；
- 验证集过小造成的波动；
- 对当前 NUAA split 的过拟合。

这时应结束整条路线。不论是否稳定，当前 `decoder_0 + fixed alpha=0.2` 都不会自动升格为最终 DEA。

---

# 五、三层数据协议：开发、OOF 稳定性、官方测试

## A. 快速筛选：170/43

当前 170/43 split 继续用于：

- identity/梯度/BN 工程检查；
- frozen mechanics audit；
- 淘汰明显降低 IoU/PD 或恶化 FA 的结构；
- 确定是否值得为唯一候选付出 OOF 成本。

这一层已经否定 MA-PCID、shared stencil 和 zero-anchor SIED。其数字不进入最终主表。

## B. 稳定性确认：213-train 内部 5-fold cross-fitting

对唯一剩余的 mean-anchor SIED，固定：

```text
active stage = decoder_0
anchor = mean
alpha = 0.2
threshold = fixed
training recipe = paired MSHNet recipe
metric protocol = fixed
```

然后对官方 213 张 train 做确定性 5 折 cross-fitting：

1. 每折用约 170 张训练 MSHNet，只在未参与该次训练的 42/43 张上生成 baseline 与 frozen-SIED 配对预测；
2. 五折的预测汇总后覆盖全部 213 张，每张图都由一个未训练过它的模型预测；
3. 五个 validation fold 在预测样本上互斥，但它们的训练集相互重叠，因此不称为“五个统计独立 seed”；
4. 当前已用于选择 \(\alpha=0.2\) 的 43 张折单独标记为 design fold；必须另外报告剩余四个 unseen folds 的结果，防止已选折主导 pooled 增益；
5. 不扫描新 \(\alpha\)、anchor、stage 或 threshold；否则需要 nested CV，不能再称为锁定假设审计；
6. 每张图保存 intersection、union、TP/FN 目标数、unmatched-FA area/组件数、changed pixels 和 conflict-region mask；
7. image-paired bootstrap 必须对重采样后的 intersection/union 重新求 ratio-of-sums，不能简单对 per-image IoU 取平均。

这是充分利用 213 张官方训练数据的**问题假设确认**，不是官方 test 结果。

## C. 最终确认：213 全训练 / 214 一次性测试

只有问题 H-GO 且最终 DEA 结构完全锁定后，才进入：

1. 用全部 213 张分别重训 paired MSHNet 和 DEA；
2. 在开发/OOF 阶段预先固定 seed、epoch、学习率、augment、threshold 和 checkpoint 规则；
3. 不根据 214 test 结果重选 epoch，不回改 anchor/结构；
4. 对 214 test 一次性评估，并报告至少 3 个 paired seeds；
5. 若还需完全对齐开源 MSHNet 的“逐 epoch 看 test”协议，单独报告 `protocol-reproduction` 结果，不与一次性 test 账本混用。

不把 427 张全部用于训练后又在同样 427 张上测试。如果要让 427 张都获得未见预测，只能额外做 427-image K-fold OOF，且必须标记为 cross-validation result，不是 official-test result。

---

# 六、必须分开 H-Gate 与 M-Gate

## H-Gate：问题假设是否成立

mean-anchor SIED 只有同时满足以下条件，才能将“interaction imbalance”判为 H-GO：

1. 213-train 5-fold OOF 中至少 4/5 折的配对 \(\Delta\mathrm{IoU}>0\)；
2. 全部 213 张 cross-fitted predictions 汇总后 pooled \(\Delta\mathrm{IoU}>0\)，且 image-paired bootstrap 的 \(\Delta\mathrm{IoU}\) 95% CI 下界 \(>0\)；
3. \(\Delta\mathrm{PD}\) 的 CI 下界 \(\ge0\)，且每折不减少检出目标数；
4. \(\Delta\mathrm{FA/M}\) 的 CI 上界 \(\le0\)，不允许用 pooled PD/FA 遮盖某折的明显恶化；
5. leave-one-fold-out 与 leave-one-image-out 后方向不反转；
6. 只使用四个 unseen folds 时仍为正，不由已参与 \(\alpha\) 选择的 design fold 主导；
7. 改善集中在预先诊断的细尺度冲突区域，且 mean anchor 明显优于 zero-anchor control；
8. 稳定性不能被 static scaling 或等范数 direct residual 完整复现，否则只能证明普通 calibration/residual 效果。

H-GO 只回答：

> **MSHNet 的最细尺度融合中确实存在可复现的 mean-centered interaction imbalance。**

它不回答“SIED 是否具备最终性能”，也不回答“结构创新是否足够”。

## M-Gate：DEA 主模型是否成立

任何最终 DEA 必须另外满足下列分层合同：

- P0：关闭新机制时与 MSHNet bitwise identity；
- P1：三数据集固定开发 holdout（NUAA 170/43、NUDT 530/133、IRSTD-1K 640/160）存活门；每个数据集均要求 paired \(\Delta\mathrm{IoU}>0\)、\(\Delta\mathrm{PD}\ge0\)、\(\Delta\mathrm{FA/M}\le0\)；通过只代表允许进入 OOF；
- P2：213-train 5-fold OOF 通过上述 H-Gate；
- P3a（最低存活门）：官方 test paired 至少 3 seeds，mean \(\Delta\mathrm{IoU}\ge+0.005\)，每个 seed \(\Delta\mathrm{PD}\ge0\)，mean \(\mathrm{FA}_{DEA}/\mathrm{FA}_{MSHNet}\le0.90\)，至少 2/3 seeds Pareto 支配 baseline；
- P3b（顶会级主结果门）：mean \(\Delta\mathrm{IoU}\ge+0.008\)，目标区间为 \(+0.008\sim+0.015\)；mean \(\Delta\mathrm{PD}>0\)；mean \(\mathrm{FA}_{DEA}/\mathrm{FA}_{MSHNet}\le0.85\)，目标下降区间为 15%--30%；paired bootstrap 的 IoU 增益 95% CI 下界 \(>0\)；
- P4：NUAA-SIRST、NUDT-SIRST、IRSTD-1K 必须全部运行，三个数据集的 mean \(\Delta\mathrm{IoU}\) 全为正，至少两个数据集 \(\Delta\mathrm{IoU}\ge+0.008\)，PD 不降，跨数据集平均 FA 下降至少 15%；
- P5：参数/MACs 或 FLOPs/延迟增量分别不超过 10%/25%/30%。

当前 fixed SIED 在 43-val 上只能算 P1 开发信号，P2--P4 均未评估；其 Conv-MAC 增量 `33.03%` 已不符合 P5。因此，即使问题 H-GO，当前 SIED 也只保留为探针，不能直接升格为 DEA。

### 顶会级机制归因合同

P3b 通过仍不足以单独证明 DEA 的收益来自所声称机制。最终模型还必须同时给出以下真实结果：

1. TP preservation：baseline 的 TP 区域和已检出目标基本保持，不以牺牲原有目标换取平均 IoU；
2. FP removal：baseline FP 中有显著比例被消除，并报告按图像、组件和面积统计的消除率；
3. FN recovery：预先定义的 recoverable FN 中确有目标被恢复，而不是只改变少量边界像素；
4. conflict localization：改善显著集中于开发阶段预先定义的 decoder-interaction 冲突图像/区域；
5. predictive validity：在不读取 GT 的前提下，mean-anchor interaction 指标能够预测哪些图像受益，并在独立 fold 上报告效应量、置信区间和排序指标；
6. structural specificity：DEA 必须显著优于参数量匹配的 ordinary residual、attention 和 final-fusion control；若这些通用对照完整复现收益，则不能声称 decoder-interaction 机制成立。

上述数值均是预注册的准入目标，不是预设结果。若完整实验只得到微弱 IoU 增益或机制归因失败，应诚实判为 NO-GO，不把它包装为顶会级贡献。

---

# 七、问题假设审计只有两种结果

## 结果 A：mean-anchor 信号稳定

这时只将问题假设判为 H-GO。固定的 SIED 仍保留为诊断探针：

\[
\boxed{
T^{\mathrm{probe}}_0(e,u)
=
q^{11}+\alpha(j-p)
}
\]

但不把它直接命名为 \(D^{\mathrm{DEA}}\)，因为它尚未通过 P1，且 `decoder_0` 的输出不再传向更细 decoder。

只允许再做**一次**主模型结构定稿，且设计稿必须先满足：

- 改写的是 `encoder skip + inherited decoder state` 的 fusion rule 或真正会传向更细尺度的 state transition，而不是 final decoder 后的权重修正；
- 最终推理不以可学习全局 \(\alpha\) 、像素 gate、router 或 correction head 作为主机制；
- 不新增第二套 encoder/decoder，不用 adapter 堆叠救性能；
- 上线前同时通过结构特异性对照：static scaling、direct residual、parameter-matched generic fusion；
- 先给出 FLOPs 上界和 P0 identity 实现，再写主模型代码；
- 只做一个原型；首次完整运行不过 P1 就停止，不再连续发明 correction branch。

不要为了模型看起来完整而把当前 SIED 盲目扩展到四级，也不再加 persistent state、stencil、router、attention 或 correction head。

### 删除“从 0 学习一个 \(\alpha\)”的训练路线

当前 `ScaleInteractionExchangeMSHNet` 中 \(\alpha\) 是 Python float，不是参数；测试也明确要求 `named_parameters()` 中没有 alpha。同时，\(\alpha=0\) 走 hard-baseline fast path，不计算 coalition branches。因此：

- 现有代码不支持“只学习 \(\alpha\)”；
- 即使把它改成 Parameter，hard path，\(\alpha=0\) 时也没有这条参数的梯度；
- 若强制在 \(\alpha=0\) 也计算 branches 以获得梯度，它仍只是一个可学习 global residual gate，不解决主创新问题。

所以 fixed \(\alpha=0.2\) 只用于 H-Gate 诊断，不进入“学 alpha—解冻 D0—全网微调”的救火链条。

---

## 结果 B：mean-anchor 信号不稳定

则正式结束整个主线：

```text
scale-evidence aliasing
decoder interaction exchange
evidence correction
```

此时不要继续尝试：

- median anchor；
- learned anchor；
- local anchor；
- 不同窗口 mean；
- 更多 \(\alpha\)；
- 多 stage SIED；
- 新 recurrent state；
- 新 correction kernel；
- 更复杂的 interaction decomposition。

因为这只是在拟合一个不稳定的小样本信号。

随后应转向具有独立证据的新问题，例如：

- encoder 对极小目标的信息丢失；
- side supervision 与 final objective 的梯度冲突；
- NUAA/NUDT 的数据域和标注差异；
- 选择错误空间更充分的 baseline；
- 改变训练目标，而不是继续修正 decoder；
- 使用更大的验证错误集进行结构研究。

---

# 八、为什么这种流程会快很多

旧流程是：

```text
发明模型
→ 写测试
→ 跑 frozen control
→ 短训
→ 失败
→ 再发明模型
```

修订后的新流程是：

```text
固定唯一剩余假设
→ 跨 checkpoint / 跨数据集零训练验证
→ H-NO-GO：整条研究问题终止
→ H-GO：只允许一次完整结构定稿
→ P0/P1 失败：终止
→ P0/P1 通过：进入 P2--P4
```

它不再允许：

- 失败后换一个 correction branch；
- 失败后改 anchor；
- 失败后增加状态；
- 失败后增加门控；
- 失败后解冻整个 MSHNet；
- 用训练能力把结构错误“救活”。

---

# 九、当前代码定位

现有原型继续隔离保留：

```text
model/dea_scale_interaction_exchange.py
model/dea_persistent_conditional_increment.py
model/dea_shared_discrepancy_stencil.py
tools/train_shared_discrepancy_stencil.py
```

定位如下：

| 原型 | 定位 |
|---|---|
| SIED | 唯一剩余弱信号 control |
| MA-PCID | 强负结果 |
| Shared Discrepancy Stencil | 负结果 |
| CEV | terminal gate control |
| SMEI | oracle/control |
| DEA v0 | 结构负控制 |
| DEAIntegrated / FullDEA | 历史原型或控制 |

在 clean baseline 与逐图机制审计完成前，不新增主模型文件。

历史失败机制、负证据和新设计禁区统一记录在：

```text
DEA_失败模型谱系与新设计禁区.md
```

baseline 完成后的任何新结构提案必须逐条说明其如何避开该文档中的已证伪路径。

---

# 十、最终停止规则

应把下面的规则写入设计稿，并严格执行：

## 继续条件

只有 clean MSHNet baseline 的三数据集、三 seed 逐图证据能够定义一个跨数据集可复现的失败问题，才允许设计 DEA。若问题仍指向 decoder interaction，则还必须满足：

- 冲突指标在多 checkpoint、多 seed 和三个数据集上方向稳定；
- mean-anchor interaction 在不读取 GT 时能预测高风险或受益图像；
- 错误确实集中于预先定义的 conflict 区域，而不是全图 calibration；
- baseline 的 FP 有可消除空间，且至少部分 FN 在现有 encoder/side evidence 中可恢复；
- ordinary residual、attention 和 final-fusion control 不能完整解释该现象。

满足后只允许设计一个保留 factual MSHNet path、真正改写 decoder fusion/recurrence 的主模型。

## 停止条件

只要出现以下任一情况，就停止：

- 增益只在一个 checkpoint 出现；
- zero/mean anchor 方向高度不一致且无法解释；
- leave-one-image-out 后增益消失；
- PD 减少一个目标；
- FA 明显恶化；
- 其他数据集不复现；
- 需要重新搜索 \(\alpha\) 才有正结果；
- 需要解冻整个 MSHNet 才能涨点。

上述停止条件终止的是当前问题/机制，不是要求把同一 correction 思路换名后继续。若 interaction 假设失败，只能根据 clean baseline 的另一项独立、跨数据集失败证据重新定义问题；不存在这类证据时，整个 DEA 设计阶段 NO-GO。

---

# 最终决策

当前按以下顺序执行，不能跳步：

```text
三数据集 × 三 seed 从零重训 clean MSHNet baseline
→ 冻结每个 seed 的 best-IoU holdout checkpoint
→ 导出逐图 TP/FP/FN、component、mean-anchor interaction 与 conflict ledger
→ 结合失败模型谱系，定义唯一的 problem / gap / root cause
→ 设计一个非模块堆叠、保留 factual path 的 DEA
→ 先做 P0/复杂度/参数匹配 controls，再做三数据集 paired training
→ 达到 P3b/P4 与机制归因合同后，才进入 official-full / official-test
```

baseline 证据之后的决策有三种，而不是预设 SIED 必须成为最终结构：

\[
\boxed{
\text{interaction failure 稳定且可预测}
\Rightarrow
\text{围绕该根因设计一次新的 decoder fusion/recurrence}
}
\]

\[
\boxed{
\text{interaction 不成立，但存在另一项跨数据集独立失败证据}
\Rightarrow
\text{明确换问题，不复活 correction/gate 路线}
}
\]

\[
\boxed{
\text{不存在稳定问题证据}
\Rightarrow
\text{DEA 设计 NO-GO}
}
\]

真正拖慢项目的，不是模型设计能力不足，而是此前没有为失败路线设置最终停止条件。

现在锁死的是证据顺序、单一机制约束和停止条件，而不是提前锁死某个尚未被三数据集证明的结构。
