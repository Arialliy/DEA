# TRACE-MSHNet 实现与门禁报告（2026-07-13）

## 结论先行

TRACE-MSHNet v1 已经按“**替换预测变量，而非堆叠模块**”的原则完成核心实现、严格 provenance、数学验证、整模型验证、复杂度审计、训练/恢复与 dev 评估代码路径。但当前结论必须是：

> **NO-GO，不应进入正式训练，更不能作为 AAAI 2027 投稿模型宣称成功。**

这是两个相互独立的失败，不是调参问题：

1. **表达族失败**：train-only GT 中存在大量非 row-convex 组件，TRACE-v1 的单 horizontal-run-per-row 状态族无法无损表示；
2. **工程预算失败**：即使只看可表示状态，exact sum/max semiring 的实际延迟也远超预声明预算。

因此没有启动 T1/T2 正式训练，没有生成性能提升数字，也没有用 hull、morphology、额外 slot、proposal、NMS 或新 head 掩盖失败。

---

## 1. 对 MSHNet baseline 的真实改动边界

保留的语义路径是：

```text
input -> encoder -> middle -> decoder -> d0
```

删除并且不实例化：

```text
output_0, output_1, output_2, output_3, final
```

替换后的唯一可训练映射为：

```text
Conv1x1(16,16) -> GELU -> Conv1x1(16,2)
```

- TRACE natural-parameter map：306 个参数；
- 被替换的 canonical MSHNet side heads + fusion：281 个参数；
- matched dense Bernoulli control：307 个参数。

两个输出通道不是两个独立 head，而是同一个 empty-or-component 指数族变量的 root/support natural coordinates。训练只允许一个 empty-inclusive exact NLL；没有 BCE、Dice、IoU、ranking、topology、distillation 或 auxiliary loss。

核心实现：

- `model/mshnet_d0_backbone.py`：物理 headless 的 canonical `input -> d0`；
- `model/trace_front.py`：只接受 clean、dev-selected、完整 provenance 的 MSHNet checkpoint；
- `model/trace_mshnet.py`：306 参数 potential map、chunked atomic field、exact NLL、whole-atom renderer；
- `model/trace_run_semiring.py`：同一状态族上的 exact sum-product、max-product、marginals 与 brute-force reference。

---

## 2. 一个必须修正的概率语义

原设计把 `p_nonempty` 当成 emitted MAP atom 的置信度，这是不成立的。实现现已明确区分：

\[
p_g=P(Y_g\neq\varnothing\mid D)
\]

和

\[
q_g=P(Y_g=\hat C_g\mid D)
=\exp(E_g(\hat C_g)-\log Z_g).
\]

如果 100 个正形状几乎等概率，即使 `p_nonempty=0.9`，单个 MAP atom 的联合后验也可能只有约 `0.009`。因此：

- `p_nonempty` 只报告组件存在概率；
- renderer 与 whole-atom threshold 使用 `map_log_joint_posterior = log q_g`；
- 同一个 atom 内所有像素共享同一个有限 score；
- 阈值操作严格使用仓库约定 `score > threshold`；
- 为兼容 evaluator，背景使用有限 sentinel，合法阈值域显式限制为 `[background_score, +inf)`。

不同 cell 的 atoms 仍可能重叠或接触；保证的是**每个 cell 内部 support 对阈值不变**，而不是全图 union 永远保持组件数不变。

---

## 3. T0-A：表达能力门（正式结论为 NO-GO）

以下数字只来自 seed `20260711` 的 clean fit split；test assets 没有参与几何选择。

| Dataset | Fit images | GT components | Exact row-run chains | Exact fraction | max-runs histogram | selected cell | collisions | local field |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| IRSTD-1K | 640 | 940 | 845 | 0.898936 | 1:845, 2:94, 3:1 | 2 | 0 | 25×28 |
| SIRST（目录名 NUAA-SIRST） | 170 | 216 | 203 | 0.939815 | 1:203, 2:13 | 4 | 0 | 31×39 |
| NUDT-SIRST | 530 | 731 | 482 | 0.659371 | 1:482, 2:178, 3:63, 4:7, 5:1 | 4 | 0 | 18×44 |

预声明门槛是 100% exact encode/decode、零 collision、零 window failure。三套数据均在第一项失败，尤其 NUDT-SIRST 只有 65.94% 可表示。

这不是 codec bug：失败样本的原始 mask 确实在同一行含多个不相连 runs。把它们做 hull、填洞或丢像素会改变监督标签，因此被明确禁止。

认证报告：

- `repro_runs/trace/t0_a/IRSTD-1K_fit_seed_20260711.json`
- `repro_runs/trace/t0_a/NUAA-SIRST_fit_seed_20260711.json`
- `repro_runs/trace/t0_a/NUDT-SIRST_fit_seed_20260711.json`

---

## 4. T0-B-DP：数学核心通过

exact solver 在 FP64/CPU 上通过：

- 3×3、4×4、5×5 tiny fields；
- nominal 与 `[-30,30]` extreme energies；
- DP 与独立 exhaustive enumeration 的 `logZ_positive/logZ_total`；
- MAP energy、root、interval backpointer、完整 support bit-exact；
- empty-inclusive root/support marginals；
- autograd 与 central finite difference `<1e-4`；
- zero-score `K/logK` 与穷举计数；
- `-logK` 后常数 root bias 的 `p_nonempty=sigmoid(b)`。

认证报告：`repro_runs/trace/t0_b/trace_t0_b_dp.json`。

这说明 T0-A 失败不是因为 DP 写错；数学实现正确并不能挽救不合适的状态族。

---

## 5. T0-B-INTEGRATION：功能通过，延迟门失败

在 RTX 5090、batch 1、256×256、SIRST fit geometry、chunk 1024 上，完整 integration gate 的功能项全部通过：

- clean checkpoint 严格加载；
- old side heads/final 物理不存在；
- 一次 optimizer step 前后 front parameter + BN hash 不变；
- exact NLL 有有限且非零的 head gradient；
- renderer 与手工 atom union bit-exact；
- threshold 不改变 atom support；
- renderer 使用 MAP joint posterior，而非存在概率；
- 无 Python per-cell loop。

但工程结果为：

| Quantity | Measured | Predeclared budget | Result |
|---|---:|---:|---|
| canonical MSHNet total median latency | 7.025 ms | reference | — |
| TRACE new prediction-end median latency | 173.083 ms | ≤14.050 ms（2× canonical total） | **FAIL** |
| latency ratio | 24.64× | ≤2× | **FAIL** |
| added peak allocated memory | 1,951,294,464 bytes | ≤2 GiB | PASS（但余量很小） |

认证报告：`repro_runs/trace/t0_b/NUAA-SIRST_seed20260711_integration.json`。

独立的 sum/max profiler 得到约 89.08 ms + 82.90 ms，与整合测量一致，排除了偶然计时异常。

---

## 6. 静态复杂度

fit geometry、batch 1 的核心 work proxy 为 `N × H × W²`：

| Dataset | Cells N | Local H×W | Work proxy | boundary/logK patterns |
|---|---:|---:|---:|---:|
| IRSTD-1K | 16,384 | 25×28 | 321,126,400 | 195 |
| SIRST | 4,096 | 31×39 | 193,130,496 | 88 |
| NUDT-SIRST | 4,096 | 18×44 | 142,737,408 | 55 |

所以“只有 306 个参数”不能被写成“计算轻量”。参数复杂度很小，但结构化状态推断的运行复杂度很大。

认证静态/运行报告位于 `repro_runs/trace/complexity/`。

---

## 7. 已完成的防泄漏与可复现路径

### 数据

- fit/dev 排名现在 byte-exact 复用 clean MSHNet 的 `SHA256(seed + NUL + name)` 协议；
- 三数据集 seed 20260711 的 fit/dev hashes 已与 clean checkpoint metadata 完全一致；
- train 模式默认 `include_test=False`，只读取 test manifest 名称用于 overlap audit，不访问 test image/mask；
- fixed resize，mask nearest-neighbor；正式 paired pipeline 不做会改变已冻结几何的 augmentation；若未来启用 deterministic flip，T0-A 必须先同时认证原图与翻转 mask；
- `drop_last=False`。

### Checkpoint

- 拒绝 raw weights、test-selected checkpoint、test-evaluated checkpoint 和无 internal holdout checkpoint；
- 保存 model/head、optimizer、epoch、best dev、全部 RNG、loader state、geometry/logK/front/data/gate/source/git/runtime hashes；
- resume 时逐项重新认证，不能只做 `strict=False` 权重加载。

### 评估

- threshold 只从 dev 的 exact unique finite scores 产生；
- 同一个 locked threshold 同时用于 dev/test、legacy/Hungarian matcher；
- 严格 `score > threshold`；
- 报告 Pd、FA/Mpix、global IoU、per-image nIoU、false component count/area 和 empty-image policy；
- 现有三数据集 test 已被历史探索反复使用，应视为 adaptive pilot，不应再描述为完全 untouched confirmatory evidence。

---

## 8. 训练入口为何必须拒绝当前实验

训练入口：`tools/train_trace.py`。

它同时要求：

1. T0-A `PASS`；
2. T0-B-DP `PASS` 且源码 hash 新鲜；
3. T0-B-INTEGRATION `PASS`；
4. geometry、baseline checkpoint、fit/dev manifests、logK cache 与 source hashes 完全一致。

当前 T0-A 和 integration 均为 `NO-GO`，因此真实训练命令必须在进入 optimizer/GPU 长训练前退出。这是预期行为，不是未完成。

dev-only inference 已能输出 evaluator-compatible NPZ：

- TRACE：joint-MAP atom log-score dense render；
- matched dense：Bernoulli logits。

正式 test export 已放在独立入口 `tools/infer_trace_locked_test.py`：它要求精确 unlock phrase、best-dev checkpoint、已冻结的 dev bundle metadata，并在完成全部认证后才第一次调用 `include_test=True`。它不做 threshold、epoch、checkpoint 或超参数选择。

---

## 9. 软件验证状态

- 使用项目运行环境执行全仓测试：`700 passed, 1 warning`，耗时 630.00 s；唯一 warning 是 Python 3.12 对多线程进程中 `fork()` 的弃用提示；
- TRACE 源文件通过 `compileall`，工作区已通过 `git diff --check`；
- `repro_runs/trace/` 下 12 份 T0-A、T0-B 与复杂度 JSON 均通过嵌入内容哈希和当前源码清单校验；
- 真实训练入口用当前 `NO-GO` 报告验证为 fail-closed：在创建训练输出或进入 optimizer 前拒绝运行。

这些测试证明的是实现、门禁和复现约束按设计工作，不把失败的研究假设改写成成功结果。

---

## 10. 对顶会创新性的判断

原方向中真正有研究价值的不是 `1×1 head`、`-logK` 或若干组件的组合，而是一个统一命题：

> 用一个 normalized component-native prediction unit 同时定义 existence、shape、confidence、likelihood 与 whole-atom decision。

这比继续给 MSHNet 叠 attention/gate/refinement 更干净，也更像可审稿的算法贡献。但 v1 的具体 row-convex family 已被真实训练几何证伪，且当前 exact kernel 不满足工程预算。因此论文不能继续围绕 TRACE-v1 包装。

下一步若继续该研究主题，应先提出一个**新的单一状态族及其 exact/controlled-complexity algorithm**，而不是加多个模块。一个可以研究、但尚未通过证明与实现门的方向是：

- 由 train-only taxonomy 约束的 bounded multi-run connected frontier state；
- 仍只有一个 empty-or-component 变量、一个 partition、一个 MAP、一个 NLL；
- 在写神经网络前先证明无重复、可表示率、复杂度上界，并在 NUDT 的 `R=1..5` 上做 train-only state-count audit；
- 若 exact frontier 在 R=5 时不可承受，应直接放弃，而不是退化成 proposal + refinement 堆叠。

在找到同时通过“100% 表达门 + 数学门 + 2× latency/2 GiB 工程门”的新状态族之前，最诚实的项目状态是：

> **实现闭环完成；TRACE-v1 科学假设被证伪；停止正式训练与投稿包装。**
