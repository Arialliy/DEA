# DEA 失败模型谱系与新设计禁区

## 文档用途

本文件将仓库中历史原型转化为**负证据和设计约束**。旧 checkpoint、日志和结果已清空；下列历史数字只用于解释哪些机制已经被否定，不能进入新的三数据集统一重训主表。

新模型设计必须从 clean MSHNet baseline 的真实失败模式出发，并显式说明它如何避开本文件中的已证伪路径。

## 失败谱系

| 路线 | 原机制与问题假设 | 已有负证据 | 失败类型 | 对新设计的约束 |
|---|---|---|---|---|
| SIED | 在同一 decoder 上计算 mean/zero-anchor 四个 coalition，以 \(q^{11}+\alpha(j-p)\) 交换当前独立项和联合项；当前仅作用于 `decoder_0` | mean-anchor \(\alpha=0.2\) 仅有 `+0.002135` 的 43-val 弱信号，PD/FA 不变且只改 37 个阈值像素；zero-anchor 方向相反；decoder0-only Conv-MAC 增量 `33.03%` | 小样本、anchor-sensitive、效率失败 | 只保留 mean-anchor interaction 作为诊断变量；禁止扩展四 stage、搜索更多 anchor、学习全局 \(\alpha\) 或把 Möbius 分解本身当创新。证据见 `DEA_尺度证据语义混叠_问题定义与下一步设计.md:1537`、`:1597` 及 `DEA_当前路线收缩与最终判定方案.md:14`。 |
| Decoder-Jacobian / VJP | 通过 decoder gradient/VJP 修正 latent state 后重新解码 | 已有强近邻类别；仓库只有设计审查，没有可作为性能证据的完整实现与训练结果 | 主创新性解析 NO-GO，不是经验性能失败 | Jacobian、VJP、adjoint 或 side-head rank-1 feedback 不能单独作为新 DEA 主创新；不得把解析上等价的动态 gate 改名复活。见问题定义文档 `:957-1100`。 |
| MA-PCID | 用 mean-anchor conditional increment 替代 factual inherited state，并跨 decoder 持久传播 | \(\alpha=1\) 时 IoU `0.201576`、PD `0.714286`；centered 变体仍强失败 | 破坏完整粗尺度语义的强 NO-GO | factual inherited decoder state 不得被差分、interaction 或 persistent state 替换；禁止用小 gate 把失败结构改成 ReZero。见 `DEA_尺度证据语义混叠_问题定义与下一步设计.md:1850`。 |
| Persistent adapter / \(A_s\!\to K\!\to B_s\) | 用多层 adapter 生成并写回 persistent conditional state | 设计审查已指出其属于普通 recurrent residual/adapter 链；两层同时零初始化还会形成一阶死梯度 | 结构类别与可训练性 NO-GO，不是经验性能失败 | 禁止用 adapter 链“修复” MA-PCID，也不得把 persistent corrector、primal--dual 或 ODE 包装加入新主模型。见问题定义文档 `:1624-1685`。 |
| Shared Discrepancy Stencil | 对 alternate-factual discrepancy 使用全尺度/全通道共享、zero-DC 的 8 参数 stencil，并写回 recurrence | 最佳学习态 `0.745749 < 0.747056`；另有 PD 降、FA 恶化的安全门失败；不是约束饱和 | 共享线性修正缺少语义条件能力 | 禁止继续给 stencil 增加 adapter、attention 或 channel mixer；zero-sum/Laplacian 不能承担创新。可借鉴 zero-init identity 和一阶梯度检查。见同一问题定义文档 `:1993`。 |
| CEV | final fusion 后对 exact per-scale contribution 做共享局部 attenuation | harmfulness teacher 在解析上退化为删除 gate 的标签×贡献符号；缺少独立因果证据，且存在强近邻 | 因果语义与创新性 NO-GO | 仅作 final-fusion/gating control；禁止 causal/Granger claim，禁止把普通 gate 包装为证据裁决。见同一问题定义文档 `:1258`。 |
| DEA-lite / CERA-lite | 原 MSHNet 推理图不变，增加 single-scale anti-sufficiency、empty-evidence、decidability auxiliary head/loss | 已有跨数据集负证据；最终预测仍是原 `final(scale_logits)`，改变的是训练损失而非 decoder mechanism | loss-level 路线 NO-GO | 禁止继续搜索 `lambda/tau/ramp`，或把 anti-sufficiency、necessity/sufficiency loss、decidability head 政名为新 DEA；仅可作 loss-level control。实现见 `model/MSHNet.py:98-160`、`model/loss.py:135-236`、`main.py:1051-1063,1415-1425,1539-1566`。 |
| SMEI / component subset oracle | 从 final/side prediction 生成组件，尝试 16 种尺度子集操作 | 43-val 上 4/4 baseline FP 有候选覆盖、3/4 有 oracle 修复空间；在**当前候选生成器、3 像素邻近和 16-subset 编辑规则下** 0/1 FN 可恢复；这是使用 GT 事后评分的能力上界 | 有局部 FP 信号，但不是可部署模型；当前规则未恢复该 representation-level FN | 禁止组件提取 + 16-way router 作为主模型，也不得把 oracle 当模型结果；可用于界定特定编辑空间的可修复上界，不能推广成所有 FN 原理上不可恢复。见同一问题定义文档 `:356`、`:1292` 及 `tools/component_subset_oracle.py:167-244`。 |
| PredictiveCorrection / DEA v0 | 删除原 decoder/side heads/final，改为跨尺度共享的 tied-adjoint/Landweber 单步回写 | 400 epoch：MSHNet `0.7471/0.9841/9.5811`；主配置 `0.7274/0.9841/14.5491`，其他宽度/步长也被 baseline 支配 | 完整性能 NO-GO，强 decoder 被容量更弱的统一更新替换 | 禁止删除 MSHNet hierarchical decoder；禁止再通过 state width、步长或 delta 调参救活。adjoint 只可作诊断。见 `MSHNet_伴随预测误差回写主模型设计评审.md:555`。 |
| DEAIntegrated | 四个不共享参数的 tri-state routing cell 与 terminal additive closure | 无显式 action semantics 时 target action occupancy 为 0；加入 residual-action loss 后 100% hard keep；历史 `0.7706` 使用过 test-selected 初始化，不能作论文证据 | action 不可识别、机制不统一、协议污染 | 禁止复活 tri-state router、四个 cell 或 terminal route closure；旧 DecoderOnly/ScaleOnly/attention 只能作旧 family control。见 `DEAIntegratedMSHNet_release/README.md:62`。 |
| CSIR / PSIR | 以“创新/保留/抑制”命名的 candidate-state update | 解析上可化简为 \(h'=(1-g)h+ge\)，即单门 GRU/candidate interpolation；没有独立实证支持 | 解析类别 NO-GO | 禁止把 candidate-state interpolation 或单门 update 重新命名为证据决策创新。见 `MSHNet_CSIR_诊断与主模型设计评审.md:101-145,408-416`。 |
| FullDEA prototype / v2 | `x_d0` 后 target/clutter/counterfactual head；v2 改为 final-fusion target residual 与 clutter suppression | 原型插入点绕开真正四尺度 fusion并有全零退化；v2 历史结果呈 FA 降但 IoU/PD 同时下降 | 插入点错误、语义混合、过抑制 | 禁止在 `x_d0` 后外挂二值 attention/head；禁止将 clutter alternative prediction 混回 final；FA 降不能代替 TP/PD 保持。见 `dea_full_v2_modification_plan.md:23`、`full_dea_v3_tps_next_steps (1).md:1`。 |
| FullDEA-v3 TPS / topology bridge | exact scale attribution、tri-state target/clutter/identity、topology bridge 与 component hard negatives | 历史 NUAA 仅 `+0.0009086`，NUDT 为 `-0.0011794`；阈值改变主要来自 target/topology path；延迟约 `1.44×` | 单数据集微增、跨数据集退化、机制归因不符、效率风险 | 禁止 morphology/topology bridge、operation selector 和组件关系模块继续堆叠；exact attribution 只作审计。见 `full_dea_v3_structural_iteration_results.md:101`、`:124`、`:158`。 |
| FullDEA-v4/v5/v6 | relation selector、hard transport 及未完整接通的后续原型 | 只有实现与 shape/identity 测试，没有可审计 paired 性能；v4 源码已注明 soft relation residual 易 collapse；v6 类已实现但未接 CLI | 未验证或退化原型 | 不得把 identity/shape 测试当性能证据，不再为历史原型投入三数据集统一重训。见 `model/full_dea_head.py:564-570,628-710`、`model/full_dea_mshnet.py:28-29`、`main.py:168-169,479-483`、`tests/test_full_dea_shapes.py:217`。 |

## 新模型的五条硬约束

1. **显式保留 factual path**：MSHNet 的 encoder、inherited decoder state 和原 decoder recurrence 必须始终存在；新机制关闭时 bitwise identity。
2. **只允许一个核心机制**：不做 terminal gate、router、component pipeline、correction head、adapter 链或多模块拼接。
3. **不能靠缩放伪装创新**：不使用可学习全局 \(\alpha\)、普通 residual、ReZero 或 persistent state 作为主贡献。
4. **问题必须由 clean baseline 证据定义**：先在 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 的三 seed holdout prediction 中复现同一失败模式，再决定结构。
5. **性能与机制同时成立**：除三数据集 paired 性能外，必须证明 TP 保持、FP 消除、recoverable FN 恢复、冲突区域富集以及 mean-anchor 指标对独立样本收益的预测有效性。

## 允许继承的内容

- MSHNet 原始 factual decoder 与多尺度监督合同；
- SIED 的 mean-anchor interaction 作为无 GT 诊断指标，而不是最终 transition；
- CEV、attention、ordinary residual、final fusion 作为结构特异性对照；
- component oracle 作为能力上界和失败分类工具；
- zero-init/P0 identity、split hash、checkpoint metadata 和 paired seed 工程合同。

## 对照实现与参数口径补充

历史代码只能提供 control 的实现骨架，不能自动成为未来 DEA 的公平对照：

- ordinary residual 可借鉴
  `DEAIntegratedMSHNet_release/model/dea_integrated_mshnet.py:294-315` 或
  `model/full_dea_head.py:196-240`，但仓库目前没有独立、干净、与未来 DEA
  参数量匹配的 ordinary-residual control；
- attention 可复用旧 Integrated 的 `routing_mode=attention`，但它只与旧四-cell
  family 参数匹配；新插入点或新预算必须重新构造 parameter-matched control；
- final-fusion 可参考旧 ScaleOnly 或 CEV，但 CEV 只有 883 个新增参数且冻结
  baseline；若未来 DEA 全网训练，必须匹配 freeze/unfreeze、optimizer、epoch 和
  checkpoint-selection policy；
- identity 的训练对照应使用同 checkpoint、同预算继续训练的
  `MSHNet-Continued`，而不是仍会构造/执行 route cells 的旧
  `Integrated-Identity`；
- fixed SIED 只复用为无新增参数的冻结诊断 probe，不在新 holdout 上重新搜索
  alpha、anchor 或 stage。

当前 `MSHNet` 类总参数为 4,066,034，其中 legacy `decidability_head` 占 521 个
参数，但不参与 clean baseline 的正式预测。后续参数表必须先冻结 canonical
MSHNet 口径，并明确是否排除这 521 个未使用参数；否则 parameter matching 会被
无关 legacy head 污染。

复杂度和 identity 还必须遵守：

1. 保留原 direct final convolution；把 grouped per-scale contributions 求和只保证
   代数等价，不保证 bitwise identity；
2. 关闭机制时必须 hard bypass，不能用 `0 * branch`、相消 gate 或极小 bias；
3. 除 forward 外，还要比较 baseline 梯度、BN buffers、optimizer step/state；
4. 重复 decoder call 会重复 BN/梯度并显著增加 execution-trace MAC，不能只按
   module tree 计 FLOPs；
5. 两层新增路径同时零初始化会产生死梯度；必须逐参数检查一阶梯度；
6. `warm_flag=False` 会绕开四尺度 final fusion，不能用于 P0、复杂度或正式推理；
7. CPU component pipeline、离散 connected-components 和 `.cpu().numpy()` 路径必须
   进入真实 latency，不能用卷积 MAC 表掩盖；
8. bias-free scale contributions 可求和，但包含 final bias 的 per-scale logits 不能
   再跨尺度求和，否则会重复加 bias。

## 进入模型设计前的准入问题

baseline 完成后，只有以下问题都能由真实逐图证据回答，才允许写新模型：

1. 三个数据集是否存在方向一致的 MSHNet decoder failure，而不是 NUAA 单 split 偶然性？
2. 该 failure 主要造成 FP、FN、边界偏差还是组件碎裂？
3. mean-anchor interaction 是否在不读取 GT 时预测高风险图像/区域？
4. 失败位置是否位于会继续传向更细尺度的 decoder transition，而非 final head 后处理？
5. 为什么 ordinary residual、attention、final-fusion gate 和已有 correction state 不能表达所需约束？
6. 一个单一、可退化、低开销的 fusion/recurrence 改写能否直接对应上述根因？

若这些问题没有共同答案，应缩窄或放弃 decoder-interaction 路线，而不是再发明一个模块。
