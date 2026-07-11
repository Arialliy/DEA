# DEA Clean Baseline 运行进度

> 快照日期：2026-07-11  
> 批次：`clean_baseline_holdout_v1`  
> 状态：运行中；本文件记录的是开发 holdout 进度，不是最终论文结果。

## 一、当前总体进度

- 计划任务：`3 datasets × 3 seeds = 9` 个 MSHNet clean baseline；
- 每个任务：400 epochs；
- 已完成：2/9；
- 正在运行：2/9；
- 等待调度：5/9；
- 已记录 epoch：1186/3600，约 **33%**；
- GPU 0/1 运行正常；
- official test 仅用于 split ID 交集/hash 审计，**未执行 test forward，未参与 checkpoint 选择**。

## 二、逐任务快照

下表中的 `Best` 是当前 holdout best-IoU checkpoint；运行中任务的数字仍可能变化，不能提前作为结论。

| 数据集 | Seed | 状态 | 当前 epoch | 当前/最终 epoch IoU | 当前 best epoch | 当前 best IoU | 同 epoch PD | 同 epoch FA/M |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 20260711 | 已完成 | 399 | 0.5479 | 227 | 0.6742 | 0.9630 | 77.3585 |
| NUDT-SIRST | 20260711 | 已完成 | 399 | 0.7382 | 383 | 0.7532 | 0.9679 | 27.9936 |
| IRSTD-1K | 20260711 | 运行中 | 247 | 0.6301 | 200 | 0.6384 | 0.9167 | 9.3460 |
| NUAA-SIRST | 20260712 | 运行中 | 137 | 0.4400 | 85 | 0.6852 | 0.9259 | 12.7748 |
| NUDT-SIRST | 20260712 | 待运行 | — | — | — | — | — | — |
| IRSTD-1K | 20260712 | 待运行 | — | — | — | — | — | — |
| NUAA-SIRST | 20260713 | 待运行 | — | — | — | — | — | — |
| NUDT-SIRST | 20260713 | 待运行 | — | — | — | — | — | — |
| IRSTD-1K | 20260713 | 待运行 | — | — | — | — | — | — |

## 三、冻结的开发协议

| 数据集 | Official train | Fit | Validation | Official test |
|---|---:|---:|---:|---:|
| NUAA-SIRST | 213 | 170 | 43 | 214 |
| NUDT-SIRST | 663 | 530 | 133 | 664 |
| IRSTD-1K | 800 | 640 | 160 | 201 |

统一配置：

```text
model                 = MSHNet
initialization         = from scratch
split_seed             = 20260711
training_seeds         = 20260711, 20260712, 20260713
epochs                 = 400
optimizer              = Adagrad
learning_rate          = 0.05
warmup_epochs          = 5
batch_size             = 4
input/crop size        = 256 × 256
operating threshold    = 0.5
checkpoint rule        = best holdout IoU
official test policy   = sealed; no forward in this stage
```

## 四、已经完成的工程工作

1. 清空旧 `weight/`、`repro_runs/` 和 `results/` 历史生成物，避免混用旧 checkpoint；
2. 建立三数据集可恢复调度器 `tools/run_clean_baselines.py`；
3. 三数据集各完成 1 epoch smoke test，3/3 成功；
4. 为每个任务固定 dataset、seed、split hash、run label 和独立输出目录；
5. 建立实时查看工具 `tools/summarize_clean_baselines.py`；
6. 建立完成后 fail-closed 收口器 `tools/finalize_clean_baselines.py`；
7. 整理历史失败路线及新设计禁区：`DEA_失败模型谱系与新设计禁区.md`；
8. 在 `DEA_当前路线收缩与最终判定方案.md` 中预注册顶会级性能门与机制归因合同。

## 五、Baseline 完成后的固定流程

```text
9 个 clean MSHNet baseline 全部完成
→ 校验 3 datasets × 3 seeds × 400 epochs 与 checkpoint metadata
→ 输出三 seed mean ± std 的 holdout baseline 表
→ 导出逐图 TP / FP / FN、connected components、mean-anchor interaction 和 conflict ledger
→ 用真实证据定义唯一 problem / gap / root cause
→ 结合历史失败模型禁区，设计一个非模块堆叠的 DEA
→ 与 MSHNet 做同 split、同 seed、同训练配方的三数据集 paired retraining
→ 通过 P0--P5、参数匹配 controls 和机制归因后，再进入 official-full / official-test
```

## 六、DEA 的预注册目标

最终模型以 MSHNet 为唯一 baseline。顶会级主结果目标为：

- mean IoU 提升 `+0.008～+0.015`；
- PD 上升；
- FA/M 下降 `15%～30%`；
- 三个数据集方向一致；
- TP 区域基本保持；
- baseline FP 中有显著比例被消除；
- recoverable FN 中有目标被恢复；
- 改善集中在预定义的 decoder-interaction conflict 样本/区域；
- mean-anchor interaction 指标能预测独立样本是否受益；
- 显著优于参数量匹配的 ordinary residual、attention 和 final-fusion controls。

这些是准入目标，不是预设结果。若只得到微弱 IoU 增益或机制归因失败，应报告 NO-GO，不能将其包装为顶会级贡献。

## 七、查看实时进度

在项目根目录执行：

```bash
"$PYTHON" tools/summarize_clean_baselines.py
```

正式批次完成后执行：

```bash
"$PYTHON" tools/finalize_clean_baselines.py \
  --batch-id clean_baseline_holdout_v1
```

收口器只有在 9/9 任务全部成功、每个任务恰有 400 条连续 epoch、checkpoint/split/seed 元数据全部一致时才会生成最终 holdout 汇总；否则 fail closed。
