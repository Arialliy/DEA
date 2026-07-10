# Full-DEA v3：结构迭代、双数据集验证与下一步模型设计

日期：2026-07-10

## 1. 当前结论

本轮没有继续做超参数搜索，而是围绕 MSHNet 的最终融合输出进行了结构级修改。当前 Full-DEA v3 已形成一条可解释的正式路径：

1. 对 MSHNet 最终 logit 做严格可加的尺度贡献分解；
2. 用多半径局部对比描述目标与杂波证据；
3. 用 target / clutter / uncertain 三态决策选择增强、抑制或恒等操作；
4. 用双向响应和尺度归因一致性寻找断裂目标的候选连接；
5. 要求候选连接两端都由 target 状态拥有，阻断 target-clutter 和 target-uncertain 的直接连接；
6. 用完整连通域难负样本和组件级损失约束错误抑制。

结果证明该结构在 NUAA-SIRST 上确实缓解了 MSHNet 的近目标碎裂/对象级虚警问题，但还不能宣称 SOTA 或跨数据集成立：

- NUAA-SIRST：PD 不变，IoU 小幅提高，FA 降低约 41.3%；
- NUDT-SIRST：PD 和 FA 改善，但 IoU 低于基线 0.00118；
- 因而当前结构通过了“机制有效性”验证，但没有通过“跨数据集主指标不退化”门槛。

## 2. 为什么重点从杂波抑制转向拓扑与关系建模

对修正评估器后的 NUAA 验证集进行逐组件诊断发现：

- 基线共有 22 个假阳性组件、356 个假阳性像素；
- 218 个假阳性像素（61.2%）距离最近 GT 质心不超过 8 像素；
- 256 个假阳性像素（71.9%）距离最近 GT 质心不超过 15 像素；
- 真正远离目标的独立杂波组件只有 4 个；
- `Misc_60` 中，同一个目标被分成两个预测碎片，两个碎片都被对象级评估计为虚警，同时该目标被计为漏检。

这说明 MSHNet 在 NUAA 上的主要失败并非普通背景误激活，而是目标附近的错误拓扑：目标碎裂、卫星响应以及目标边缘和杂波混淆。仅做更强的负向抑制会同时伤害目标，因此 Full-DEA 的核心不应是一个附加注意力块，而应是基于证据归因的操作选择器。

训练集伪杂波和验证集真实错误还存在明显域差异：伪杂波的通道统计与真实 FP、TP 不同。由此否决了“依靠训练伪杂波原型直接识别验证杂波”的路线。

## 3. 当前正式结构

### 3.1 精确尺度反事实分解

把 MSHNet 最终融合层写成：

```text
z_base = bias + sum_s contribution_s
```

由此可以精确构造：

- leave-one-scale-out：`z_base - contribution_s`；
- branch-only：`bias + contribution_s`；
- 每个位置的归一化尺度归因向量。

该分解不是近似 attention，也不改变 MSHNet 基线输出；审计中 `z_base` 与修正后的 MSHNet 输出一致。

### 3.2 三态操作语义

每个位置产生 target、clutter、uncertain 三种状态：

- target：只允许目标增强或拓扑连接；
- clutter：只允许抑制；
- uncertain：执行恒等映射。

该设计把“是否修改”纳入模型表达，避免所有不确定响应都被强制增强或抑制。

### 3.3 双向归因一致的拓扑桥

候选 gap 只有同时满足以下条件才允许形成 bridge：

1. gap 两侧存在响应支持，而不是单侧目标边界；
2. 两侧归一化尺度归因相容；
3. gap 位置的三态决策倾向 target；
4. 两个端点都由 target 状态拥有。

归因相容度定义为：

```text
consistency = clamp(1 - 0.5 * L1(a_left, a_right), 0, 1)
```

最终连接为：

```text
bridge_gate = target_gate_at_gap
              * sigmoid(bridge_evidence)
              * endpoint_target_prior

bridge_delta = gamma * bridge_gate * bridge_amount
```

形态学 closing 只作为描述符，不再直接成为执行先验。单侧边界、归因不一致、target-clutter 和 target-uncertain 端点组合均被结构性阻断。

### 3.4 组件级训练约束

训练时使用完整连通域难负样本，并补充：

- exact counterfactual hard clutter mining；
- component target keep loss；
- component clutter suppress loss；
- bridge supervision；
- frozen-BN 协议，保证冻结的 MSHNet 主干不发生统计漂移。

## 4. 双数据集结果

所有结果使用修正后的同一评估器：sigmoid 概率阈值 0.5、真实概率阈值扫描、按组件身份而非等面积值进行匹配，并在每次评估前重置 ROC 状态。

| 数据集 | 模型 | IoU | PD | FA | 相对基线变化 |
|---|---|---:|---:|---:|---|
| NUAA-SIRST | MSHNet | 0.7461767 | 0.9619772 | 25.3838 | — |
| NUAA-SIRST | Full-DEA v3 endpoint-owned | **0.7470853** | **0.9619772** | **14.9023** | IoU +0.0009086；PD 不变；FA -10.4815（-41.29%） |
| NUDT-SIRST | MSHNet | **0.7538766** | 0.9449735 | 24.3819 | — |
| NUDT-SIRST | Full-DEA v3 endpoint-owned | 0.7526972 | **0.9470899** | **22.9341** | IoU -0.0011794；PD +0.0021164；FA -1.4477（-5.94%） |

NUAA 最优 checkpoint：

```text
weight/FullDEA-v3-TPS-2026-07-10-03-51-25/
```

NUDT 最优 checkpoint：

```text
weight/FullDEA-v3-TPS-2026-07-10-03-47-14/
```

## 5. 机制审计

两个数据集的最优 checkpoint 均满足：

- `z_base` 精确复现修正后的 MSHNet 基线；
- 二值评估下 `z_target == z_final`；
- 因而本轮可见的预测变化来自目标/拓扑路径，而不是 clutter suppression 路径；
- endpoint target prior 和 bridge delta 在 GT 区域显著高于非 GT 区域。

| 数据集 | endpoint prior（GT / 非 GT） | bridge delta（GT / 非 GT） | 解释 |
|---|---:|---:|---|
| NUAA | 0.009699 / 0.0000247 | 0.087364 / 0.000189 | 连接明显集中在目标区域，主指标和 FA 同时改善 |
| NUDT | 0.013051 / 0.0000532 | 0.061385 / 0.000213 | 机制集中性成立，但拓扑修改仍造成少量 IoU 损失 |

NUDT 的失败不是“杂波抑制过强”。更准确的诊断是：像素级 bridge 无法稳定判断两个邻近组件究竟是同一目标的碎片，还是目标与卫星杂波。端点 ownership 降低了风险，但没有消除这种语义歧义。

## 6. 已否决或降级的结构路线

### 6.1 软区域池化

没有带来 NUAA 改善，不能构成正式模块。

### 6.2 目标原型记忆

一度减少 2 个 FP 像素，但最优 checkpoint 和主指标未改善；同时训练伪杂波与验证真实 FP 的统计不匹配，因此已从正式结构移除。

### 6.3 无端点归属的双向桥

NUAA 有效，但操作语义不完整：只验证 gap 的局部 target 倾向，不能阻断 target-clutter 端点组合。endpoint ownership 是必要的结构约束。

### 6.4 保守拓扑传输幅度

曾把 bridge 幅度上限绑定到较弱端点，试图减少过连接；NUDT 4 epoch 最优仅为 IoU 0.7488，明显恶化，已回滚。说明简单压低连接幅度不能解决关系误判。

## 7. 复杂度、测试与实现状态

| 模型 | 参数量 | FVCore 统计的卷积 FLOPs | 单张 256×256 延迟 |
|---|---:|---:|---:|
| MSHNet | 4,066,034 | 6.053 G | 7.19 ms |
| Full-DEA v3 | 4,071,741 | 6.479 G | 10.37 ms |

- 参数增加 5,707，约 0.14%；
- FVCore 可计数 FLOPs 增加约 7.0%；该工具没有覆盖 padding、pooling、reduction 和逐元素操作；
- 当前本机实测延迟约为 MSHNet 的 1.44 倍，不能直接外推到其他硬件；
- 双向配对已经从重复 padding 改为预 padding + 张量化组合，和朴素参考实现在最大 offset 1/2/4 下完全一致；
- 当前 18 项测试通过，包含精确分解、单侧边界拒绝、归因不一致拒绝和双端点 ownership 等测试；
- `git diff --check` 通过。

## 8. 下一步：从像素桥升级为组件关系操作选择器

下一轮不应再搜索 bridge/loss 权重，也不应继续堆叠局部注意力。应把现有 pixel-wise topology bridge 升级为 **Attribution-Guided Component Relation Reasoning（归因引导的组件关系推理）**。

### 8.1 需要解决的核心判断

对空间邻近的两个候选组件，显式判别：

1. same-target fragments：属于同一目标的碎片，执行 reconnect；
2. target + satellite clutter：一个是目标、另一个是卫星杂波，执行 suppress satellite；
3. uncertain：证据不足，执行 identity。

输出不再是每个像素独立的 bridge gate，而是组件对级别的离散操作分布，再把选定操作投影回像素域。

### 8.2 组件对特征

优先使用与当前因果分解一致的特征，而不是再引入通用 attention：

- 两组件的精确尺度归因向量及相似度；
- 组件面积、质心距离、相对方向、长宽比和沿连接轴的 elongation；
- gap 区域的多半径局部对比与响应连续性；
- 两端 baseline confidence、target/clutter/uncertain ownership；
- leave-one-scale-out 后组件是否共同消失或分化。

### 8.3 建议实现形态

首选一个轻量、可微的候选组件图：

```text
soft components / proposals
        -> K-nearest spatial pairs
        -> pair feature encoder
        -> {reconnect, suppress-satellite, identity}
        -> operation-specific decoder
        -> residual correction of z_base
```

为了避免“组件提取不可微”成为实现障碍，可以先用局部峰值/软掩码生成少量 proposal，并只在 proposal 对上做关系分类；不要直接在全图建立稠密图。identity 必须保留为显式安全操作。

### 8.4 下一轮硬门槛

保持当前冻结主干、评估器和训练协议不变，先做 NUAA + NUDT 双门验证：

- NUAA：PD 不下降，IoU 高于 MSHNet，FA 明显低于 MSHNet；
- NUDT：IoU 至少不低于 MSHNet，同时保留 PD/FA 改善；
- 若 NUDT IoU 仍退化，则不能进入多种子和全 benchmark；
- 双门通过后才做 3 个以上种子、更多公开数据集、复杂度和显著性分析。

建议结构消融顺序：

1. exact decomposition；
2. + tri-state identity；
3. + whole-component hard mining；
4. + bidirectional attribution prior；
5. + endpoint ownership；
6. + component relation operation selector。

## 9. 可复现实验与代码入口

当前正式结构：

```text
model/dea_evidence.py
model/full_dea_head.py
model/full_dea_loss.py
model/full_dea_mshnet.py
```

修正评估器与测试：

```text
utils/metric.py
tests/test_metric_pd_fa.py
tests/test_full_dea_counterfactual_path.py
tests/test_full_dea_shapes.py
```

当前 endpoint-owned 实验：

```text
repro_runs/dea_v3_nuaa_endpoint_owned_bridge_6e/
repro_runs/dea_v3_nudt_endpoint_owned_bridge_4e/
```

中间与否决实验：

```text
repro_runs/dea_v3_nuaa_bidirectional_attribution_bridge_6e/
repro_runs/dea_v3_nudt_bidirectional_attribution_bridge_4e/
repro_runs/dea_v3_nudt_conservative_topology_transport_4e/
repro_runs/dea_v3_nuaa_decoupled_topology_6e/
repro_runs/dea_v3_nudt_decoupled_topology_4e/
```

分解审计：

```text
repro_runs/dea_v3_nuaa_endpoint_owned_bridge_6e/decomposition_best_iou.json
repro_runs/dea_v3_nudt_endpoint_owned_bridge_4e/decomposition_best_iou.json
```

每个正式新实验目录都保留了 `command.txt`、`train.log` 和可用时的 `decomposition_best_iou.json`。

## 10. 投稿定位的谨慎表述

当前证据只足以支持：Full-DEA 针对 MSHNet 的最终融合决策进行精确尺度归因，并用归因一致的操作选择处理近目标碎裂与卫星杂波。它不同于只做边缘/形状监督、普通多尺度融合、频域净化或简单 miss-vs-false-alarm loss balancing。

但在组件关系模块和双数据集主指标门槛完成之前，不应使用“通用提升”“SOTA”或“解决红外小目标检测”的表述。最终创新性还需要在完整实现后做系统检索和同类方法对照。
