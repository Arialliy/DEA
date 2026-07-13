# MSHNet baseline 强项、首个瓶颈与顺序冻结判定

> 日期：2026-07-13
> 目标：提高低组件虚警预算下的实例检出率，并使改善跨阈值、seed、数据集与 backbone 稳定成立。
> 约束：只允许一个统一预测范式；禁止 attention/head/loss/refinement 的模块堆叠。

## 1. 结论先行

当前能被证据支持的最窄结论是：

1. MSHNet 的主要强项位于 `input → d0`。多尺度编码、残差表征、跳连恢复和多尺度监督共同产生了一个对绝大多数困难目标仍保留区分信息的高分辨率特征场。
2. 第一个稳定定位的异常边界是 `d0 → scalar prediction`。在 48 个 hard-core 观测中，`d0` 有 44 个保持 distinct，但 `mask0` 和 `z` 分别只剩 21 和 22 个。
3. 这还不能证明“换 final fusion”“加 attention”或“加 component head”会成功；现有结果只授权冻结 `input → d0`，并把第一个可修改边界放在 `d0` 之后。
4. 全局阈值调节已被 Gate G 否决为主路线；低 FA 下的主要操作性问题是目标证据没有稳定进入可行的前景排序，不是只选错一个阈值。
5. 冻结 `d0` 的历史 holdout signed-readout smoke 已给出负向证据，正式 full-train/test-only smoke 仍在复核。该 probe 只负责定位可读性，不是论文模块；在正式证据放行前，任何 signed branch 都禁止进入模型。

## 2. 为什么 MSHNet 强

### 2.1 原论文已给出的因果消融

MSHNet 的原始设计不是靠复杂 backbone 取胜，而是把 plain U-Net 的四个 decoder 尺度分别预测、上采样和融合，并对 final 与四个 side predictions 施加 SLS。论文的尺度数消融从 1 个尺度增加到 4 个尺度时，IoU 从 63.10% 提升到 67.16%，Pd 从 86.73% 提升到 93.88%，FA 从 19.21 降到 15.03/Mpix。这支持“多尺度预测与监督”是 baseline 强度来源之一，而不是仅凭结构外观推测。[MSHNet CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.pdf)

SLS 的作用具有明确取舍：相对普通 IoU，完整 SLS 把 MSHNet 的 IoU 从 64.83% 提升到 67.16%、Pd 从 91.16% 提升到 93.88%，但 FA 从 5.28 增至 15.03/Mpix；其中 location term 把 Pd 从 89.46% 提升到 93.88%，同时把 FA 从 4.06 增至 15.03/Mpix。原论文也明确承认 location sensitivity 可能把噪声当作目标。因此，SLS 解释了 baseline 为何有较高 Pd，也暴露了它在低 FA 目标下的内生矛盾。

### 2.2 当前代码中的实际生成链

当前 canonical 路径是：

```text
input
  → 1×1 stem
  → 4-stage encoder + middle
  → skip-connected symmetric decoder
  → d3 → d2 → d1 → d0
  → four scalar side logits
  → upsample + 4→1 spatial fusion
  → z
```

- 每个 ResNet block 内含残差、channel attention 和 spatial attention：[model/MSHNet.py](model/MSHNet.py#L34-L65)。
- encoder、middle 和 skip decoder 的真实路径见 [model/MSHNet.py](model/MSHNet.py#L79-L100) 与 [model/MSHNet.py](model/MSHNet.py#L184-L194)。
- 四个 `1×1` side readout 与 `4→1, 3×3` final fusion 见 [model/MSHNet.py](model/MSHNet.py#L101-L106) 和 [model/MSHNet.py](model/MSHNet.py#L197-L219)。
- canonical 训练对 final 与四个 side outputs 等权计算 SLS，粗尺度标签逐级 max-pool，见 [main.py](main.py#L2025-L2053)。

这些结构事实解释了机制，但不能把 channel attention、spatial attention 或某一级 decoder 单独写成“已证明的强项”，因为当前仓库没有逐项干预消融。

### 2.3 当前仓库的逐层生存证据

Gate I 对 16 个 Q2/FA20 hard-core targets、3 个固定 seed，共 48 个正式观测进行逐层审计，并为每个观测构造同数据集、同 seed 的成功匹配对照：

| 位置 | hard-core distinct | 成功对照 distinct |
|---|---:|---:|
| `p3` | 36/48 | 48/48 |
| `middle` | 31/48 | 45/48 |
| `j3` | 41/48 | 48/48 |
| `d0` | 44/48 | 48/48 |
| `mask0` | 21/48 | 48/48 |
| `z` | 22/48 | 48/48 |

coarse path 的下降经 skip decoder 恢复到 `d0=44/48`，说明 MSHNet 前端确实能把多数微弱目标证据带回高分辨率。这是目前冻结前端最直接的本地证据，详见 [Gate I summary](repro_runs/gate_i/front_freeze_confirmatory_v1/front_freeze_summary.md)。

## 3. 哪部分拖了后腿

### 3.1 已定位：`d0 → scalar prediction`

Gate I 的最大、不可恢复下降发生在：

```text
d0 → mask0 : 23/48 distinct-state drops
d0 → z     : 22/48 distinct-state drops
```

成功对照对应下降均为 0。相比之下，第一个 pooling 边界 `e0 → p0` 只有 2/48 drops。因此，当前第一个可靠可修改边界是 prediction conversion，而不是 encoder、pooling 或 decoder。

这个证据是“无符号可区分性”证据，尚不能证明 `d0` 中存在方向一致、可泛化的 target-positive readout。另有 1 个 hard-core target 在三个 seed 的 `d0` 都不 distinct，所以 output-only 路线不可能被宣称覆盖全部失败目标。

### 3.2 已否决：只改 final fusion

历史 Gate D 显示，`d0` 与 final mean-margin 在 71 个 miss 中有 69 个保持同号；scale cancellation 在 miss 中为 60/71，在成功对照中反而为 70/71。因而 final-fusion suppression 不是统一根因。不能把 `4→1` fusion 换成 attention/gate 后就称为机制创新。

### 3.3 已否决：只调阈值或求更强 solver

Gate G 在 FA=1/5/10/20 四个预算下，component-conversion direction 全部不通过，comparable joint-oracle gain 也不通过。大量目标属于“预算可行集合中没有局部峰响应”，不是阈值挑选误差。详见 [Gate G summary](repro_runs/gate_g/frontier_decomposition_v2/frontier_decomposition_summary.md)。

因此以下路线永久 NO-GO：

- 更密的 threshold grid；
- 换 Sinkhorn/solver；
- test-time calibration；
- 在原 pixel field 后堆 connected-component repair；
- SLS + ranking + FA + topology 的 compound loss。

### 3.4 初步否定 readout compatibility，正式 test-selected 复核中

现在仍有两个可证伪解释：

1. `readout compatibility`：`d0` 有一致的 target-positive 方向，只是原 `1×1` head 没读出来；
2. `prediction-unit mismatch`：即便像素 evidence 可读，独立像素 scalarization 在低组件 FA 下仍产生碎裂、合并和噪声排序，必须改变预测基本单位。

历史 holdout smoke（NUAA-SIRST/seed 20260711）没有支持解释 1。在 cross-fitted Q2 下，signed-standardized 在 FA=1/5/10/20 仅匹配 `0/0/1/1` 个目标，参数匹配 unsigned control 为 `1/1/1/2`；该数据集唯一预注册 hard-core 目标在全部六种读出下均为 0 恢复。更严重的是，原 Gate K finalizer 存在真实 JSON variant 顺序必拒绝、漏 centered control、跨预算 seed 不取交集、hard-core 不参与门控等缺陷，所以该 smoke 只能作为工程否定信号，不能作为正式科学结论。

新的 [full-train/test-only probe](tools/run_signed_readout_probe_full_train_test.py) 已改为：完整 canonical train 训练、canonical test 评测、验证集数量为 0，并严格绑定每 10 epoch test-selected 的 best-IoU checkpoint。若这个复核仍失败，则解释 1 被停止，不允许通过增加 readout/attention/loss 补救；下一步只能在保持 `input→d0` 冻结的同时，研究一个整体替换 scalarization 的统一表示。

## 4. 不堆模块的硬约束

最终方法只能有一个论文级核心定义。候选机制必须满足：

```text
一个训练目标
↕
一个概率分解
↕
一个推理预测单位
↕
一个与 component Pd/FA 对齐的阈值语义
```

以下情况直接判为模块堆叠：

- 把 signed branch、root head、growth module、topology loss、NMS 分别列为贡献；
- 仍保留 pixel segmentation 主分支，再增加 component refinement；
- 每个失败现象对应增加一个 loss；
- 通过最终 joint fine-tune 重新改变已冻结的前段；
- 不能用同一概率模型解释新增计算为何必需。

代码层的判据同样固定：候选模型必须让 canonical forward 不再执行原 `output_0...output_3 + final` 标量融合；它不能一边保留原像素主分支，一边旁接结构化分支再融合。即使参数量很小，只要存在“两套预测变量 + 后融合”，仍按模块堆叠否决。

这一约束已经落实为 [MSHNetD0Backbone](model/mshnet_d0_backbone.py)：它只构造 canonical `input→d0` 模块，物理上不存在 `output_0...output_3` 和 `final`。定向测试证明，从同一 MSHNet state 加载前段后，随机输入上的 `d0` 与 canonical forward hook **逐位相等**，并验证缺失/错形 state fail closed；当前 5 项测试全部通过。后续完整候选只能从这个 headless front 接出唯一预测变量。

如果采用组件值预测，ROOT、support generation、STOP 和 component mark 必须是同一个组件随机变量的不可分割因子，而不是四个可插拔模块。signed local reference 若通过，只是固定输入坐标系，不作为独立创新点。

### 4.1 性能判定必须是各模型独立选优

`每 10 个 completed epochs 在 canonical test 上评测` 只定义候选 checkpoint 集合，不定义 baseline 与新模型的配对 epoch。正式比较固定为：

1. baseline 与每个候选都独立完成 400 epochs；
2. 各自在 `10, 20, ..., 400` 的评测点中，按完全相同的预注册规则独立选出 best-IoU 与 constrained best-Pd/FA checkpoint；
3. 只比较“baseline 自己的最优”与“候选自己对应准则的最优”，并同时报告各自最优 epoch；
4. 禁止把候选固定到 baseline 最优 epoch，禁止为了同 epoch 对齐而丢弃候选后续最优结果，也禁止从多个准则中事后挑对候选最有利的一项。

因此，后续所谓 paired comparison 只表示相同 dataset、seed、训练预算、数据划分与选择规则；**不表示相同 epoch**。跨 seed 主结论仍报告全部预注册 seed 的分布与逐 seed 胜负。最终交付模型可以按你的要求，从三个 seed 各自已经独立选出的最优 checkpoint 中再选总体最优，但必须明确标注为 test-selected best-seed；它可以是最终权重，不能替代论文中的跨-seed 稳定性证据。

## 5. 不可逆顺序冻结表

| 顺序 | 边界 | 当前状态 | 通过依据 | 后续是否允许改动 |
|---:|---|---|---|---|
| 0 | 数据与评估协议 | 已冻结 | canonical `img_idx`、full train、无 val、每 10 epoch test-selected | 否 |
| 1 | `input → d0` | 已冻结 | Gate I 48 个观测与 matched controls | 否 |
| 2 | signed decision coordinate | **已 NO-GO 并冻结** | full-train/test-only、参数匹配 signed/centered/unsigned/raw/native 因果对照 | 否；不得重新加入后续模型 |
| 3 | component-valued prediction unit | RCP、typed forest、CEMC 均已 NO-GO | 必须先通过新颖性、可解性与 mechanical gate | 只能整体替换 `d0` 后 scalar heads/fusion |
| 4 | root/支持生成 | 待验证 | root recall 上界 + GT-root free-running growth | 通过后永久冻结 |
| 5 | 多组件仲裁与渲染 | 待验证 | 无重叠、无接触、低 blocking、跨阈值整组件选择 | 通过后永久冻结 |
| 6 | 全模型 | 待验证 | 跨阈值/seed/数据集/backbone + latency/参数/失败分析 | 不做 joint 回调前段 |

## 6. 下一项最小因果门

该门已经关闭。冻结 MSHNet 与 BN buffers、只从 `d0` 训练参数完全匹配的 17 参数 readout：

```text
native final z
native output0
refit raw d0
annulus-centered d0
signed-standardized d0
parameter-matched unsigned |wᵀu| + b
```

所有 trainable probes 使用相同 class-balanced BCE、相同 Q2、相同 matcher 和相同 FA=1/5/10/20。旧 fit/dev Gate K 已废止。新的 full-train/test-only、明确 test-selected 的单数据集单 seed 工程 smoke 已完成，artifact 位于 [Gate K smoke](repro_runs/gate_k/full_train_test_signed_readout_smoke_v1/NUAA-SIRST/seed_20260711/summary.json)。它原本只有在以下条件同时满足时才允许放行 signed coordinate：

- signed-standardized 同时严格优于 native、raw、annulus-centered 和参数匹配 unsigned；
- Hungarian 与 legacy matcher 方向一致；
- candidate 与所有参与 dominance 的 comparator 在 pooled 与两个 held-out folds 均不超预算，否则该点不可比较并 fail closed；
- 至少两个相邻 FA budgets 同向；
- 相邻 budgets 必须由同一组至少 2/3 seed 同时通过，不能分别凑 seed；
- 至少两个数据集满足上述同-seed 相邻预算条件；
- 相对 paired baseline 的 mask quality 退化不超过 0.005；
- 必须恢复既定 baseline-missed target，而非只在容易目标上增加 matched count；
- backbone 参数、BN buffers 和固定 anchor `d0` hash 不变；
- bundle、checkpoint、split、target authority 与全部直接依赖源码哈希必须重新计算一致。

实际结果远未过门：在 official legacy matcher 下，`refit_signed_standardized` 的 cross-fit Pd 在 FA=1/5/10/20 四个预算均为 `0.003802`；`original_final_z` 对应为 `0.148289/0.422053/0.798479/0.904943`。signed 的固定 logit-0 IoU 也只有 `0.005778`，原生 final 为 `0.723586`。参数匹配 unsigned 同样退化，说明标准化投影本身没有提供可用的后端坐标。由于单 seed 必要条件已经失败，不再浪费 GPU 扩大到九任务矩阵；signed/unsigned coordinate 均永久删除，不得以辅助分支或额外 loss 形式回填。

解释规则固定：

- signed 胜、unsigned 不胜：冻结有符号局部参照这一 prediction primitive；
- raw 与 signed 同胜：只是旧 head/监督不足，不能作为 AAAI 主创新；
- signed 与 unsigned 同胜：收益来自强度，不是方向；
- 全部失败：停止这类线性 post-`d0` readout，不追加模块补救；保持已冻结的 `input→d0`，改换 `d0` 后的唯一预测变量，而不是回改前段。

## 7. 当前研究判断

MSHNet 强在“把微弱、多尺度目标信息保留到高分辨率 `d0`”，弱在“把该信息压成独立像素标量并用全局阈值转换成组件决策”。前半句已有较强本地证据；后半句中的“组件决策单位”仍是待验证假设，而不是既定创新。

因此现在正确的动作已经从“继续调 readout”变为：只在冻结 `d0` 之后定义一个统一、组件值的预测变量；原 scalar heads/fusion 与失败的 signed/unsigned readout 都不得保留。该顺序满足从前到后冻结，也防止用模块堆叠制造虚假的性能改进。

## 8. 新颖性处置：不能把结构保证本身包装成模块

截至 2026-07-13 的 primary-source 压力检索给出两项不可逆处置：

1. `ROOT–ADD–STOP/RCP` 作为 AAAI 主创新为 **NO-GO**。PointGroup 已包含 BFS 连通候选与单簇评分，Region Growing CNN 已有邻接增长与停止，CGVAE 已处理 BFS ADD/STOP 与生成 trace，Graph Unpooling 已给出连通保持与任意连通图可表达；StarDist/MaskFormer 又占据了“support + 单 mark + 整实例选择”。exact prefix connectivity 只是邻接增长的直接归纳，不能撑起主贡献。RCP 仅保留为内部 codec、机械连通测试和消融对照。
2. Conservative Absorbing Forest / typed ghost-root forest 作为主方法为 **NO-GO**。它满足单一结构变量、不是代码模块堆叠，但机制几乎就是 random rooted forest、Random Walker prior、Matrix-Tree structured learning 与 spanning-forest clustering 的组合。两类 ghost-root 边在分母只以 `rT+rB` 出现；其 canonical-component numerator 不是 binary-mask likelihood，而真正 mask likelihood又允许同一 GT 组件拆成多个 forest units。maximum spanning tree 只是 latent-tree MAP，也不是边缘化 partition 的 MAP。256² 精确 logdet/selected-inverse 训练还没有可用的 batched sparse 路径。因此 typed root、ghost、Matrix-Tree 和 MST 都不能作为创新点，也不进入模型实现。
3. Component-Exclusion Margin Cut（CEMC）为 **NO-GO**。对任意 binary submodular Potts MAP 前景组件 `C`，其 constrained exclusion min-marginal 精确等于删除该组件的能量差，也就是 `Σ_C[D(0)-D(1)] - boundary(C)`；所谓每组件再做一次 constrained mincut 完全冗余。当前 CPU artifact 已穷举 5,696 个 8-neighbour 小网格能量、2,765 个 MAP 组件且无反例，见 [CEMC collapse audit](repro_runs/gate_l/cemc_collapse_audit_v1/summary.md)。它只会从既有 MAP 中删组件，不能恢复 MAP 中不存在的漏检目标，最多保留为 classical control。
4. Learned component-tree filtration + budgeted antichain 为 **NO-GO**。它要么退化成 `scalar field → max-tree` 形态学后处理，要么需要另一个 region scorer 而重新构成模块堆叠；extremal-region tree、区域打分、exact tree DP 与 structured output learning 又已被 CVPR 2013/MedIA 2015 等直接覆盖。各预算分别求最优 antichain 一般也不嵌套，不能同时宣称“每个预算最优”和“跨预算集合单调”。
5. Bounded-treewidth rooted connected-subset CRF 为 **NO-GO**。单 root、固定小窗口的 exact frontier DP 虽然成立，但 11×11 的 root-tagged frontier 状态包络已约 462 万，33×33 约 `6.38e21`；全图逐 root 更不可行。多 root 后不可避免出现重复/重叠 support，统一选择就是 NP-hard set packing，或退化为 proposal + NMS。
6. Neural hard-core polymer field 为 **NO-GO**。它在定义上确实是单一 component-set 随机变量，但 connected polymer 数指数爆炸、log-partition 一般 #P-hard、MAP 是 MWIS/set packing。更致命的是，标准 Kotecký–Preiss low-activity 证书蕴含每个 activity `<1`，从而空配置必为 MAP；一旦 activity `>1` 允许非空检测，可证 cluster-expansion 区间立即失效。它不能以“稀疏目标”同时解决训练归一化与检测 MAP。
7. Anytime-valid e-component process 为 **NO-GO**。对模型自己生成或 `argmax` 的 ADD/STOP 动作使用 `q1/q0`，不是对零假设下新观测构造的 e-factor；若整张 `d0` 已在初始时刻可见，后续因子也不再构成 predictable reveal。把动作改为 `q0` 自采样虽可令条件期望为 1，却只检验内部随机数；把 root 先验做成数据依赖 proposal 又破坏固定 wealth。去掉 anytime 保证后只剩普通自回归序列似然，与 FFN/region growing/GraphRNN 等强先例重合，不能作为理论创新。
8. Closed-Walk Occupancy Readout（CWOR）为 **NO-GO**。六阶闭合游走读出精确等价于截断 Estrada subgraph centrality，是半径 3 的局部图多项式，输出仍是逐像素标量。非负边权下 bridge 的往返游走只会单调增加证据；单位 7×7 网格内部的六阶值 `5.05556` 高于二像素目标端点的 `1.54306`，同形 target/clutter 更不可分。可计算的截断版本没有组件语义，完整矩阵指数又不具备 256² 可训练性，且 subgraph centrality/communicability/heat-kernel/graph-filter 与 IRSTD graph 方法已有直接先例。

所以“非堆叠”不是把多个步骤改成一个名字。最终候选必须同时满足：单一随机变量、单一目标、可计算训练与推理、相对强先例有不可拆分的新统计或算法性质，并在进入 GPU 前先过小网格 exhaustive gate。
