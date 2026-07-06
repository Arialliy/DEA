# DEA-lite 0.005 跑完后的下一步方案与实际代码修改

> 适配仓库：`https://github.com/Arialliy/DEA`  
> 当前阶段：**结果固化、复测、汇总、跨数据集验证**。  
> 当前不建议继续改 `model/MSHNet.py`、`model/loss.py` 或 `main.py` 的训练逻辑。  
> 代码修改仅新增实验管理工具脚本，不改变模型、loss、训练和测试逻辑。

---

## 1. 当前结果判断

### 1.1 Paired DEA-off baseline

| Setting | Checkpoint | IoU | PD | FA |
|---|---:|---:|---:|---:|
| DEA-off baseline | best-IoU | 0.6705 | 0.9150 | 9.2616 |

### 1.2 DEA `lambda_single=0.01`

| Setting | Checkpoint | IoU | PD | FA | Interpretation |
|---|---:|---:|---:|---:|---|
| DEA 0.01 | best-IoU | 0.6705 | 0.9116 | 7.8951 | IoU 持平，FA 下降，PD 微降 |
| DEA 0.01 | PD/FA-best | 0.6639 | 0.9286 | 7.8951 | PD 提升，FA 下降，IoU 小幅下降 |

### 1.3 DEA `lambda_single=0.005, tau=0.6`

| Setting | Checkpoint | IoU | PD | FA | 相对 baseline |
|---|---:|---:|---:|---:|---|
| DEA 0.005 | best-IoU, epoch 282 | 0.6718 | 0.9014 | 6.4527 | IoU +0.0013, PD -0.0136, FA -2.8089 |
| DEA 0.005 | PD/FA-best, epoch 367 | 0.6637 | 0.9218 | 6.6805 | IoU -0.0068, PD +0.0068, FA -2.5811 |

### 1.4 结论

`lambda_single=0.005` 是当前最强的 **FA-control operating point**：

```text
baseline:       IoU 0.6705 / PD 0.9150 / FA 9.2616
DEA 0.005 P/F:  IoU 0.6637 / PD 0.9218 / FA 6.6805
```

它满足：

```text
IoU >= baseline IoU - 0.01 ，即 0.6637 >= 0.6605
PD  >= baseline PD         ，即 0.9218 >= 0.9150
FA  <  baseline FA         ，即 6.6805 < 9.2616
```

所以这不是失败结果，而是可以作为主线证据的结果：

> DEA-lite 的 single-scale anti-sufficiency 约束可以显著降低 FA，并在合理 IoU trade-off 下保持甚至提升 PD。

---

## 2. 为什么现在不继续改模型

当前 `/home/ly/DEA` 已经不是原始 MSHNet，而是 DEA-lite fork。它已经具备：

```text
1. z_only_i / z_empty / z_only_max / d_logit
2. conservative single-scale anti-sufficiency loss
3. DEA 开关、ramp、detach evidence
4. best-IoU / latest / PD-FA-best checkpoint 保存
5. seed / deterministic / paired-IoU threshold
```

现在继续改模型会破坏变量隔离。当前最干净的实验变量是：

```text
lambda_single = 0, 0.005, 0.01, 0.02
```

其他保持一致：

```text
same seed
same dataset
same backbone
same decoder
same final fusion
same inference path
```

这使得结论很清楚：

> FA 的降低主要来自训练时 single-scale counterfactual anti-sufficiency，而不是来自额外推理 gate 或新模块。

因此当前阶段不要改：

```text
model/MSHNet.py
model/loss.py
main.py 的训练逻辑
lambda_dec
lambda_empty
inference-time d gate
component selector
all-16 subset
positive necessity
candidate verifier
learnable neutral token
```

---

## 3. 下一步执行优先级

```text
Step 1. 归档 0.005 run 的 best-IoU 和 PD/FA-best checkpoint。
Step 2. 单独 test 复测两个 0.005 权重。
Step 3. 新增两个实验管理脚本。
Step 4. 生成 IRSTD-1K 汇总表。
Step 5. 在 NUDT-SIRST 上跑 paired baseline 和 DEA 0.005。
Step 6. 再决定是否把 0.005 设为默认配置。
```

---

## 4. 归档当前 0.005 run

如果 0.005 是最新跑完的 run，可以先自动取最新目录：

```bash
cd /home/ly/DEA

export RUN_DIR=$(ls -td /home/ly/DEA/weight/MSHNet-* | head -n 1)
echo "$RUN_DIR"
```

先检查 checkpoint 内容：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python - <<'PY'
import os
import torch

run_dir = os.environ["RUN_DIR"]

def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

for name in ["checkpoint_best_iou.pkl", "checkpoint_pd_fa_best.pkl", "checkpoint.pkl"]:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        print(name, "MISSING")
        continue
    ckpt = load(path)
    print(name)
    print("  epoch:", ckpt.get("epoch"))
    print("  iou:  ", ckpt.get("iou"))
    print("  pd:   ", ckpt.get("pd"))
    print("  fa:   ", ckpt.get("fa"))
PY
```

预期看到：

```text
checkpoint_best_iou.pkl:   epoch 282, IoU≈0.6718, PD≈0.9014, FA≈6.4527
checkpoint_pd_fa_best.pkl: epoch 367, IoU≈0.6637, PD≈0.9218, FA≈6.6805
checkpoint.pkl:            epoch 399, latest/final
```

归档：

```bash
cp "$RUN_DIR/weight.pkl" \
   "$RUN_DIR/weight_lambda_single_0p005_best_iou_e282.pkl"

cp "$RUN_DIR/checkpoint_best_iou.pkl" \
   "$RUN_DIR/checkpoint_lambda_single_0p005_best_iou_e282.pkl"

cp "$RUN_DIR/weight_pd_fa_best.pkl" \
   "$RUN_DIR/weight_lambda_single_0p005_pdfa_best_e367.pkl"

cp "$RUN_DIR/checkpoint_pd_fa_best.pkl" \
   "$RUN_DIR/checkpoint_lambda_single_0p005_pdfa_best_e367.pkl"
```

---

## 5. 复测 0.005 best-IoU 权重

```bash
cd /home/ly/DEA

export RUN_DIR=$(ls -td /home/ly/DEA/weight/MSHNet-* | head -n 1)

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/IRSTD-1K \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --weight-path "$RUN_DIR/weight_lambda_single_0p005_best_iou_e282.pkl"
```

预期：

```text
IoU ≈ 0.6718
PD  ≈ 0.9014
FA  ≈ 6.4527
```

---

## 6. 复测 0.005 PD/FA-best 权重

```bash
cd /home/ly/DEA

export RUN_DIR=$(ls -td /home/ly/DEA/weight/MSHNet-* | head -n 1)

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/IRSTD-1K \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --weight-path "$RUN_DIR/weight_lambda_single_0p005_pdfa_best_e367.pkl"
```

预期：

```text
IoU ≈ 0.6637
PD  ≈ 0.9218
FA  ≈ 6.6805
```

---

# 7. 代码修改：新增实验管理工具

这部分不修改模型、loss 或训练逻辑，只新增 `tools/` 下的实验管理脚本。

使用下面的 patch：

```bash
cd /home/ly/DEA

git apply <<'PATCH'
diff --git a/tools/archive_dea_checkpoints.sh b/tools/archive_dea_checkpoints.sh
new file mode 100755
index 0000000..c8bb001
--- /dev/null
+++ b/tools/archive_dea_checkpoints.sh
@@ -0,0 +1,115 @@
+#!/usr/bin/env bash
+set -euo pipefail
+
+if [[ $# -lt 1 ]]; then
+  echo "Usage: bash tools/archive_dea_checkpoints.sh RUN_DIR [TAG]" >&2
+  echo "Example: bash tools/archive_dea_checkpoints.sh /home/ly/DEA/weight/MSHNet-xxxx lambda_single_0p005" >&2
+  exit 1
+fi
+
+RUN_DIR="$1"
+TAG="${2:-lambda_single_0p005}"
+PYTHON_BIN="${PYTHON_BIN:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
+
+if [[ ! -d "$RUN_DIR" ]]; then
+  echo "Run directory not found: $RUN_DIR" >&2
+  exit 1
+fi
+
+get_epoch() {
+  local ckpt_path="$1"
+  "$PYTHON_BIN" - "$ckpt_path" <<'PY'
+import sys
+import torch
+
+path = sys.argv[1]
+try:
+    ckpt = torch.load(path, map_location="cpu", weights_only=False)
+except TypeError:
+    ckpt = torch.load(path, map_location="cpu")
+
+epoch = ckpt.get("epoch")
+if epoch is None:
+    raise SystemExit(f"No epoch field in {path}")
+print(int(epoch))
+PY
+}
+
+print_metric() {
+  local ckpt_path="$1"
+  "$PYTHON_BIN" - "$ckpt_path" <<'PY'
+import sys
+import torch
+
+path = sys.argv[1]
+try:
+    ckpt = torch.load(path, map_location="cpu", weights_only=False)
+except TypeError:
+    ckpt = torch.load(path, map_location="cpu")
+
+print(f"epoch={ckpt.get('epoch')} iou={ckpt.get('iou')} pd={ckpt.get('pd')} fa={ckpt.get('fa')}")
+PY
+}
+
+BEST_CKPT="$RUN_DIR/checkpoint_best_iou.pkl"
+BEST_WEIGHT="$RUN_DIR/weight.pkl"
+PDFA_CKPT="$RUN_DIR/checkpoint_pd_fa_best.pkl"
+PDFA_WEIGHT="$RUN_DIR/weight_pd_fa_best.pkl"
+
+if [[ ! -f "$BEST_CKPT" ]]; then
+  echo "Missing $BEST_CKPT" >&2
+  exit 1
+fi
+
+if [[ ! -f "$BEST_WEIGHT" ]]; then
+  echo "Missing $BEST_WEIGHT" >&2
+  exit 1
+fi
+
+BEST_EPOCH=$(get_epoch "$BEST_CKPT")
+BEST_WEIGHT_OUT="$RUN_DIR/weight_${TAG}_best_iou_e${BEST_EPOCH}.pkl"
+BEST_CKPT_OUT="$RUN_DIR/checkpoint_${TAG}_best_iou_e${BEST_EPOCH}.pkl"
+
+cp "$BEST_WEIGHT" "$BEST_WEIGHT_OUT"
+cp "$BEST_CKPT" "$BEST_CKPT_OUT"
+
+echo "Archived best-IoU checkpoint:"
+echo "  $BEST_WEIGHT_OUT"
+echo "  $BEST_CKPT_OUT"
+print_metric "$BEST_CKPT"
+
+if [[ -f "$PDFA_CKPT" && -f "$PDFA_WEIGHT" ]]; then
+  PDFA_EPOCH=$(get_epoch "$PDFA_CKPT")
+  PDFA_WEIGHT_OUT="$RUN_DIR/weight_${TAG}_pdfa_best_e${PDFA_EPOCH}.pkl"
+  PDFA_CKPT_OUT="$RUN_DIR/checkpoint_${TAG}_pdfa_best_e${PDFA_EPOCH}.pkl"
+
+  cp "$PDFA_WEIGHT" "$PDFA_WEIGHT_OUT"
+  cp "$PDFA_CKPT" "$PDFA_CKPT_OUT"
+
+  echo "Archived PD/FA-best checkpoint:"
+  echo "  $PDFA_WEIGHT_OUT"
+  echo "  $PDFA_CKPT_OUT"
+  print_metric "$PDFA_CKPT"
+else
+  echo "PD/FA-best checkpoint not found. Skipping PD/FA archive."
+fi
+
+echo "Archive done."
diff --git a/tools/dea_run_report.py b/tools/dea_run_report.py
new file mode 100644
index 0000000..1a7a002
--- /dev/null
+++ b/tools/dea_run_report.py
@@ -0,0 +1,203 @@
+#!/usr/bin/env python3
+import argparse
+import csv
+import os
+from pathlib import Path
+
+import torch
+
+
+SLOTS = [
+    ("best_iou", "checkpoint_best_iou.pkl"),
+    ("pd_fa_best", "checkpoint_pd_fa_best.pkl"),
+    ("latest", "checkpoint.pkl"),
+]
+
+
+def load_torch_file(path):
+    try:
+        return torch.load(path, map_location="cpu", weights_only=False)
+    except TypeError:
+        return torch.load(path, map_location="cpu")
+
+
+def parse_run_spec(spec):
+    if ":" not in spec:
+        path = spec
+        label = Path(spec).name
+    else:
+        path, label = spec.split(":", 1)
+    return Path(path), label
+
+
+def fmt(value, digits=4):
+    if value is None:
+        return ""
+    if isinstance(value, int):
+        return str(value)
+    try:
+        return f"{float(value):.{digits}f}"
+    except (TypeError, ValueError):
+        return str(value)
+
+
+def collect_rows(run_specs, baseline):
+    rows = []
+    for spec in run_specs:
+        run_dir, label = parse_run_spec(spec)
+        for slot, filename in SLOTS:
+            ckpt_path = run_dir / filename
+            if not ckpt_path.exists():
+                continue
+            ckpt = load_torch_file(str(ckpt_path))
+            iou = ckpt.get("iou")
+            pd = ckpt.get("pd")
+            fa = ckpt.get("fa")
+            row = {
+                "label": label,
+                "slot": slot,
+                "run_dir": str(run_dir),
+                "checkpoint": filename,
+                "epoch": ckpt.get("epoch"),
+                "iou": iou,
+                "pd": pd,
+                "fa": fa,
+                "delta_iou": None if iou is None else float(iou) - baseline["iou"],
+                "delta_pd": None if pd is None else float(pd) - baseline["pd"],
+                "delta_fa": None if fa is None else float(fa) - baseline["fa"],
+            }
+            rows.append(row)
+    return rows
+
+
+def write_csv(rows, out_csv):
+    fieldnames = [
+        "label",
+        "slot",
+        "epoch",
+        "iou",
+        "pd",
+        "fa",
+        "delta_iou",
+        "delta_pd",
+        "delta_fa",
+        "checkpoint",
+        "run_dir",
+    ]
+    out_csv.parent.mkdir(parents=True, exist_ok=True)
+    with out_csv.open("w", newline="") as f:
+        writer = csv.DictWriter(f, fieldnames=fieldnames)
+        writer.writeheader()
+        for row in rows:
+            writer.writerow(row)
+
+
+def write_markdown(rows, out_md, baseline):
+    out_md.parent.mkdir(parents=True, exist_ok=True)
+    lines = []
+    lines.append("# DEA-lite Run Report")
+    lines.append("")
+    lines.append("## Baseline")
+    lines.append("")
+    lines.append("| IoU | PD | FA |")
+    lines.append("|---:|---:|---:|")
+    lines.append(f"| {baseline['iou']:.4f} | {baseline['pd']:.4f} | {baseline['fa']:.4f} |")
+    lines.append("")
+    lines.append("## Checkpoints")
+    lines.append("")
+    lines.append("| Run | Slot | Epoch | IoU | PD | FA | ΔIoU | ΔPD | ΔFA |")
+    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
+    for row in rows:
+        lines.append(
+            "| {label} | {slot} | {epoch} | {iou} | {pd} | {fa} | {diou} | {dpd} | {dfa} |".format(
+                label=row["label"],
+                slot=row["slot"],
+                epoch=fmt(row["epoch"], 0),
+                iou=fmt(row["iou"]),
+                pd=fmt(row["pd"]),
+                fa=fmt(row["fa"]),
+                diou=fmt(row["delta_iou"]),
+                dpd=fmt(row["delta_pd"]),
+                dfa=fmt(row["delta_fa"]),
+            )
+        )
+    lines.append("")
+    lines.append("## Suggested interpretation")
+    lines.append("")
+    lines.append("- Use `best_iou` rows to compare IoU-preserving behavior.")
+    lines.append("- Use `pd_fa_best` rows for PD/IoU-constrained false-alarm control.")
+    lines.append("- A useful FA-control point should satisfy: IoU within 0.01 of baseline, PD no lower than baseline, and FA lower than baseline.")
+    lines.append("")
+    out_md.write_text("\n".join(lines), encoding="utf-8")
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--baseline-iou", type=float, required=True)
+    parser.add_argument("--baseline-pd", type=float, required=True)
+    parser.add_argument("--baseline-fa", type=float, required=True)
+    parser.add_argument("--runs", nargs="+", required=True, help="Run specs: /path/to/run:label")
+    parser.add_argument("--out-md", type=str, default="results/dea_run_report.md")
+    parser.add_argument("--out-csv", type=str, default="results/dea_run_report.csv")
+    args = parser.parse_args()
+
+    baseline = {
+        "iou": args.baseline_iou,
+        "pd": args.baseline_pd,
+        "fa": args.baseline_fa,
+    }
+    rows = collect_rows(args.runs, baseline)
+    if not rows:
+        raise SystemExit("No checkpoint rows collected. Check --runs paths.")
+
+    write_csv(rows, Path(args.out_csv))
+    write_markdown(rows, Path(args.out_md), baseline)
+    print(f"Wrote {args.out_csv}")
+    print(f"Wrote {args.out_md}")
+
+
+if __name__ == "__main__":
+    main()
+PATCH
```

然后设置执行权限并做语法检查：

```bash
cd /home/ly/DEA

chmod +x tools/archive_dea_checkpoints.sh

/home/ly/BasicIRSTD/infrarenet/bin/python -m py_compile \
  tools/dea_run_report.py \
  main.py \
  model/MSHNet.py \
  model/loss.py
```

---

## 8. 使用工具脚本归档 0.005

```bash
cd /home/ly/DEA

export RUN_DIR=$(ls -td /home/ly/DEA/weight/MSHNet-* | head -n 1)

bash tools/archive_dea_checkpoints.sh \
  "$RUN_DIR" \
  lambda_single_0p005
```

---

## 9. 生成 IRSTD-1K 汇总报告

如果 0.01 run 目录是：

```text
/home/ly/DEA/weight/MSHNet-2026-07-06-18-56-58
```

0.005 是最新目录，则执行：

```bash
cd /home/ly/DEA

export RUN_0P01=/home/ly/DEA/weight/MSHNet-2026-07-06-18-56-58
export RUN_0P005=$(ls -td /home/ly/DEA/weight/MSHNet-* | head -n 1)

/home/ly/BasicIRSTD/infrarenet/bin/python tools/dea_run_report.py \
  --baseline-iou 0.6705 \
  --baseline-pd 0.9150 \
  --baseline-fa 9.2616 \
  --runs \
    "$RUN_0P01:dea_0p01" \
    "$RUN_0P005:dea_0p005" \
  --out-md results/irstd1k_dea_lite_report.md \
  --out-csv results/irstd1k_dea_lite_report.csv
```

查看报告：

```bash
cat /home/ly/DEA/results/irstd1k_dea_lite_report.md
```

---

## 10. 下一步：NUDT-SIRST paired baseline

如果 IRSTD-1K 复测和汇总都正常，下一步不要继续在 IRSTD-1K 上微调更多 `lambda`，而是去 NUDT-SIRST 验证泛化。

先跑 DEA-off paired baseline：

```bash
cd /home/ly/DEA

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/NUDT-SIRST \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --epochs 400 \
  --lr 0.05 \
  --mode train \
  --seed 20260706 \
  --deterministic true \
  --dea-lambda-single 0 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0
```

记录 NUDT baseline 的：

```text
best-IoU IoU / PD / FA
```

---

## 11. 下一步：NUDT-SIRST DEA 0.005

假设 NUDT baseline 的 best-IoU 是：

```text
NUDT_BASE_IOU
NUDT_BASE_PD
NUDT_BASE_FA
```

需要把下面三个值替换成真实 baseline 结果后再跑。

```bash
cd /home/ly/DEA

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/NUDT-SIRST \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --epochs 400 \
  --lr 0.05 \
  --mode train \
  --seed 20260706 \
  --deterministic true \
  --dea-lambda-single 0.005 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0 \
  --dea-ramp-epochs 80 \
  --dea-tau 0.6 \
  --dea-detach-evidence \
  --save-dea-debug \
  --paired-baseline-iou NUDT_BASE_IOU \
  --pd-fa-min-pd NUDT_BASE_PD \
  --pd-fa-min-iou NUDT_BASE_IOU_MINUS_0P01 \
  --pd-fa-iou-margin 0.01
```

实际替换规则：

```text
NUDT_BASE_IOU_MINUS_0P01 = NUDT_BASE_IOU - 0.01
```

例子：如果 NUDT baseline 是：

```text
IoU=0.7000 / PD=0.9200 / FA=8.0000
```

则命令中使用：

```bash
--paired-baseline-iou 0.7000 \
--pd-fa-min-pd 0.9200 \
--pd-fa-min-iou 0.6900 \
--pd-fa-iou-margin 0.01
```

---

## 12. 0.005 在 IRSTD-1K 上的论文表述

可以写成：

> On the paired IRSTD-1K setting, DEA-lite with `lambda_single=0.005` provides the strongest false-alarm control. Under the PD/IoU-constrained checkpoint selection, it reduces FA from 9.2616 to 6.6805 while improving PD from 0.9150 to 0.9218, with a small IoU trade-off from 0.6705 to 0.6637.

中文：

> 在 paired IRSTD-1K 设置下，`lambda_single=0.005` 的 DEA-lite 表现出最强的虚警控制能力。在 PD/IoU 约束的 checkpoint 选择下，它将 FA 从 9.2616 降到 6.6805，同时 PD 从 0.9150 提升到 0.9218，IoU 仅从 0.6705 小幅下降到 0.6637。

---

## 13. 最终当前决策

当前不要再改模型。

当前应执行：

```text
1. 归档并复测 0.005 checkpoint。
2. 新增实验管理脚本。
3. 生成 0 / 0.005 / 0.01 / 0.02 汇总表。
4. 在 NUDT-SIRST 上跑 paired baseline。
5. 在 NUDT-SIRST 上跑 DEA 0.005。
```

当前暂时不执行：

```text
1. 不开 lambda_dec。
2. 不开 lambda_empty。
3. 不加 inference-time d gate。
4. 不加 component selector。
5. 不做 all-16 subset。
6. 不改 MSHNet.py / loss.py 的训练逻辑。
```

当前主配置建议：

```text
DEA-lite FA-control main setting:
--dea-lambda-single 0.005
--dea-tau 0.6
--dea-detach-evidence
--dea-lambda-dec 0
--dea-lambda-empty 0
```
