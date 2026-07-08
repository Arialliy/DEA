# DEA-lite 0.005 NUDT 正结果归档、复测与 NUAA 下一步计划

> Canonical repo root: `/home/ly/DEA`  
> 当前结论：**先归档 + 复测 NUDT DEA-lite 0.005；然后补 NUAA paired baseline / DEA-lite 0.005。不要现在改模型，也不要现在跑 0.01。**

---

## 0. 先判断：你提出的顺序是对的

你的判断成立：

```text
1. 先归档 NUDT-SIRST DEA-lite 0.005 正结果。
2. 复测 best-IoU 和 PD/FA-best 两个 checkpoint。
3. 整理 IRSTD-1K + NUDT-SIRST 当前主结果。
4. 下一轮优先跑 NUAA paired evidence。
5. 暂时不要改模型，不要跑 0.01。
```

原因是这次 NUDT-SIRST 结果已经不是“只提升 PD、FA 变差”的失败模式，而是：

```text
MSHNet best-IoU:
  IoU 0.7539 / PD 0.9450 / FA 24.38

DEA-lite 0.005 best-IoU:
  IoU 0.7632 / PD 0.9513 / FA 17.33

DEA-lite 0.005 PD/FA-best:
  IoU 0.7567 / PD 0.9481 / FA 12.27
```

这个信号对 AAAI rescue 比 CGA-v2 更有价值，因为它同时改善了 IoU、PD 和 FA。CGA-v2 的旧路线已经被 P2 gate 打掉，原始 summary 里 `decision=P2_FAIL_IMPL_AUDIT_ALLOWED`，且预声明的 seed42 gate 失败；所以现在应该把重心转向 DEA-lite 这条正向路线，而不是继续救 CGA-v2。  

---

## 1. 当前 evidence 状态

| 数据集 | 状态 | 当前解释 |
|---|---|---|
| IRSTD-1K | 已有 DEA-lite 0.005 正向信号 | 主要体现 FA-control / PD-FA trade-off |
| NUDT-SIRST | DEA-lite 0.005 已跑完 400 epochs，结果正向 | best-IoU 与 PD/FA-best 均优于 MSHNet baseline |
| NUAA | 下一步缺口 | 需要 paired MSHNet baseline，然后跑 DEA-lite 0.005 |
| 0.01 | 暂停 | 现在调 0.01 会打断当前 0.005 的证据链 |

结论：

```text
DEA-lite 0.005 should become the current main experimental setting.
0.01 is deferred until NUAA 0.005 paired evidence is complete.
```

---

## 2. 仓库事实与约束

当前 GitHub `Arialliy/DEA` 是一个 DEA-lite MSHNet 仓库，README 说明它在 MSHNet 训练流基础上加入了 lightweight Decidable Evidence Aggregation losses、checkpoint utilities、multi-GPU options 和 local path defaults。仓库也说明 DEA-lite loss 权重可以通过 `--dea-lambda-single / --dea-lambda-dec / --dea-lambda-empty / --dea-tau / --dea-ramp-epochs` 调整。

仓库默认使用 project-local `datasets/` 和 `weight/`，并且 `datasets/`、`weight/`、`repro_runs/` 是本地目录，不应直接提交大文件。训练输出包括 `checkpoint.pkl`、`weight.pkl`、`metric.log`、`epoch_metric.log` 和可选 `dea_debug/*.pt`。因此当前可以先执行命令归档/复测；如果后续新增代码，只应新增 **archive / retest / summarize scripts**，不应该改 `model/MSHNet.py` 或 `model/loss.py`。

---

## 3. 当前决策

```text
GO:
  archive NUDT DEA-lite 0.005 checkpoints
  retest NUDT best-IoU and PD/FA-best
  write machine-readable manifest
  compare against MSHNet baseline
  run NUAA paired baseline next

NO-GO:
  modify DEA model
  modify loss
  run lambda 0.01 now
  start broad hyperparameter sweep
  write AAAI claim before NUAA paired evidence
```

---

## 4. 目录与命名约定

```bash
ROOT=/home/ly/DEA
PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python
DATASET_NUDT=/home/ly/DEA/datasets/NUDT-SIRST
DATASET_NUAA=/home/ly/DEA/datasets/NUAA-SIRST
NUDT_0P005=/home/ly/DEA/weight/MSHNet-2026-07-07-03-24-31
REPRO_DIR=/home/ly/DEA/repro_runs/dea_lite_0p005_nudt_archive_retest
```

命名规则：

```text
weight_nudt_lambda_single_0p005_best_iou_e368.pkl
checkpoint_nudt_lambda_single_0p005_best_iou_e368.pkl
weight_nudt_lambda_single_0p005_pdfa_best_e367.pkl
checkpoint_nudt_lambda_single_0p005_pdfa_best_e367.pkl
```

---

## 5. 立即执行命令：先固化 NUDT 正结果

这一节是当前优先执行版。**不需要先改模型，也不需要先新增脚本**。先把 NUDT-SIRST DEA-lite 0.005 的两个正结果 checkpoint 固化并复测。

不要修改：

```text
model/MSHNet.py
model/loss.py
utils/data.py
utils/metric.py
main.py
```

### 5.1 归档 NUDT DEA-lite 0.005 checkpoint

```bash
cd /home/ly/DEA

export ROOT=/home/ly/DEA
export PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python
export DATASET_NUDT=/home/ly/DEA/datasets/NUDT-SIRST
export NUDT_0P005=/home/ly/DEA/weight/MSHNet-2026-07-07-03-24-31
export NUDT_OUT=/home/ly/DEA/repro_runs/dea_lite_0p005_nudt_archive_retest

mkdir -p "$NUDT_OUT"

test -s "$NUDT_0P005/weight.pkl"
test -s "$NUDT_0P005/checkpoint_best_iou.pkl"
test -s "$NUDT_0P005/weight_pd_fa_best.pkl"
test -s "$NUDT_0P005/checkpoint_pd_fa_best.pkl"

cp -n "$NUDT_0P005/weight.pkl" \
  "$NUDT_0P005/weight_nudt_lambda_single_0p005_best_iou_e368.pkl"
cp -n "$NUDT_0P005/checkpoint_best_iou.pkl" \
  "$NUDT_0P005/checkpoint_nudt_lambda_single_0p005_best_iou_e368.pkl"
cp -n "$NUDT_0P005/weight_pd_fa_best.pkl" \
  "$NUDT_0P005/weight_nudt_lambda_single_0p005_pdfa_best_e367.pkl"
cp -n "$NUDT_0P005/checkpoint_pd_fa_best.pkl" \
  "$NUDT_0P005/checkpoint_nudt_lambda_single_0p005_pdfa_best_e367.pkl"

sha256sum \
  "$NUDT_0P005/weight_nudt_lambda_single_0p005_best_iou_e368.pkl" \
  "$NUDT_0P005/checkpoint_nudt_lambda_single_0p005_best_iou_e368.pkl" \
  "$NUDT_0P005/weight_nudt_lambda_single_0p005_pdfa_best_e367.pkl" \
  "$NUDT_0P005/checkpoint_nudt_lambda_single_0p005_pdfa_best_e367.pkl" \
  > "$NUDT_OUT/nudt_dea_lite_0p005_archived_artifacts.sha256"

cat "$NUDT_OUT/nudt_dea_lite_0p005_archived_artifacts.sha256"
ls -lh --time-style=long-iso \
  "$NUDT_0P005/weight_nudt_lambda_single_0p005_best_iou_e368.pkl" \
  "$NUDT_0P005/checkpoint_nudt_lambda_single_0p005_best_iou_e368.pkl" \
  "$NUDT_0P005/weight_nudt_lambda_single_0p005_pdfa_best_e367.pkl" \
  "$NUDT_0P005/checkpoint_nudt_lambda_single_0p005_pdfa_best_e367.pkl"
```

注意：这里使用 `cp -n` 是为了避免覆盖旧归档；如果目标文件已经存在，`cp -n` 会静默保留旧文件。所以归档后必须看 `sha256` 和 `mtime`，确认归档文件确实对应当前这轮 NUDT 0.005 结果。

### 5.2 复测 NUDT best-IoU 权重

```bash
cd /home/ly/DEA

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u main.py \
  --dataset-dir "$DATASET_NUDT" \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --seed 20260706 \
  --deterministic true \
  --weight-path "$NUDT_0P005/weight_nudt_lambda_single_0p005_best_iou_e368.pkl" \
  2>&1 | tee "$NUDT_OUT/nudt_dea_lite_0p005_best_iou_e368_retest.log"
```

预期接近：

```text
mIoU 0.7632385950 / Pd 0.9513227513 / Fa 17.3269984234
```

### 5.3 复测 NUDT PD/FA-best 权重

```bash
cd /home/ly/DEA

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u main.py \
  --dataset-dir "$DATASET_NUDT" \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --seed 20260706 \
  --deterministic true \
  --weight-path "$NUDT_0P005/weight_nudt_lambda_single_0p005_pdfa_best_e367.pkl" \
  2>&1 | tee "$NUDT_OUT/nudt_dea_lite_0p005_pdfa_best_e367_retest.log"
```

预期接近：

```text
mIoU 0.7566753346 / Pd 0.9481481481 / Fa 12.2713755412
```

---

## 6. 可选脚本 1：归档 + 复测 NUDT DEA-lite 0.005

本节是可选脚本版。**如果运行这个脚本，必须先创建第 7 节的 `tools/official/parse_dea_test_log.py`；如果还要计算 delta，也要创建第 8 节的 `tools/official/compare_dea_lite_against_baseline.py`。** 否则脚本会在复测完成后的解析阶段失败。若只执行第 5 节的手动命令，则不依赖这些 parser / compare 脚本。

新建：

```text
scripts/official/archive_retest_nudt_dea_lite_0p005.sh
```

内容如下：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
CUDA_DEVICE=${CUDA_DEVICE:-0}
DATASET_DIR=${DATASET_DIR:-${ROOT}/datasets/NUDT-SIRST}
RUN_DIR=${RUN_DIR:-${ROOT}/weight/MSHNet-2026-07-07-03-24-31}
OUT_DIR=${OUT_DIR:-${ROOT}/repro_runs/dea_lite_0p005_nudt_archive_retest}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PIN_MEMORY=${PIN_MEMORY:-false}

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR does not exist: ${RUN_DIR}" >&2
  exit 2
fi

for f in weight.pkl checkpoint_best_iou.pkl weight_pd_fa_best.pkl checkpoint_pd_fa_best.pkl; do
  if [[ ! -s "${RUN_DIR}/${f}" ]]; then
    echo "ERROR: required artifact missing or empty: ${RUN_DIR}/${f}" >&2
    exit 3
  fi
done

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "ERROR: DATASET_DIR does not exist: ${DATASET_DIR}" >&2
  exit 4
fi

BEST_IOU_WEIGHT="${RUN_DIR}/weight_nudt_lambda_single_0p005_best_iou_e368.pkl"
BEST_IOU_CKPT="${RUN_DIR}/checkpoint_nudt_lambda_single_0p005_best_iou_e368.pkl"
PDFA_WEIGHT="${RUN_DIR}/weight_nudt_lambda_single_0p005_pdfa_best_e367.pkl"
PDFA_CKPT="${RUN_DIR}/checkpoint_nudt_lambda_single_0p005_pdfa_best_e367.pkl"

cp -n "${RUN_DIR}/weight.pkl" "${BEST_IOU_WEIGHT}"
cp -n "${RUN_DIR}/checkpoint_best_iou.pkl" "${BEST_IOU_CKPT}"
cp -n "${RUN_DIR}/weight_pd_fa_best.pkl" "${PDFA_WEIGHT}"
cp -n "${RUN_DIR}/checkpoint_pd_fa_best.pkl" "${PDFA_CKPT}"

sha256sum \
  "${BEST_IOU_WEIGHT}" \
  "${BEST_IOU_CKPT}" \
  "${PDFA_WEIGHT}" \
  "${PDFA_CKPT}" \
  > "${OUT_DIR}/nudt_dea_lite_0p005_archived_artifacts.sha256"

cat > "${OUT_DIR}/nudt_dea_lite_0p005_archive_manifest.json" <<JSON
{
  "project": "DEA-lite MSHNet",
  "dataset": "NUDT-SIRST",
  "lambda_single": 0.005,
  "source_run_dir": "${RUN_DIR}",
  "dataset_dir": "${DATASET_DIR}",
  "best_iou": {
    "epoch": 368,
    "reported_iou": 0.7632,
    "reported_pd": 0.9513,
    "reported_fa": 17.33,
    "weight_path": "${BEST_IOU_WEIGHT}",
    "checkpoint_path": "${BEST_IOU_CKPT}"
  },
  "pdfa_best": {
    "epoch": 367,
    "reported_iou": 0.7567,
    "reported_pd": 0.9481,
    "reported_fa": 12.27,
    "weight_path": "${PDFA_WEIGHT}",
    "checkpoint_path": "${PDFA_CKPT}"
  },
  "baseline_reference": {
    "model": "MSHNet",
    "best_iou_iou": 0.7538765773118212,
    "best_iou_pd": 0.944973544973545,
    "best_iou_fa": 24.381890354386297
  },
  "decision": "ARCHIVED_PENDING_RETEST",
  "next_step": "retest best-IoU and PD/FA-best weights at fixed test configuration"
}
JSON

BEST_IOU_LOG="${OUT_DIR}/nudt_dea_lite_0p005_best_iou_e368_retest.log"
PDFA_LOG="${OUT_DIR}/nudt_dea_lite_0p005_pdfa_best_e367_retest.log"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" -u main.py \
  --dataset-dir "${DATASET_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --pin-memory "${PIN_MEMORY}" \
  --mode test \
  --weight-path "${BEST_IOU_WEIGHT}" \
  2>&1 | tee "${BEST_IOU_LOG}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" -u main.py \
  --dataset-dir "${DATASET_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --pin-memory "${PIN_MEMORY}" \
  --mode test \
  --weight-path "${PDFA_WEIGHT}" \
  2>&1 | tee "${PDFA_LOG}"

"${PYTHON}" tools/official/parse_dea_test_log.py \
  --log "${BEST_IOU_LOG}" \
  --dataset NUDT-SIRST \
  --method DEA-lite-0.005 \
  --checkpoint_role best_iou \
  --checkpoint_epoch 368 \
  --weight_path "${BEST_IOU_WEIGHT}" \
  --output "${OUT_DIR}/nudt_dea_lite_0p005_best_iou_e368_retest_summary.json"

"${PYTHON}" tools/official/parse_dea_test_log.py \
  --log "${PDFA_LOG}" \
  --dataset NUDT-SIRST \
  --method DEA-lite-0.005 \
  --checkpoint_role pdfa_best \
  --checkpoint_epoch 367 \
  --weight_path "${PDFA_WEIGHT}" \
  --output "${OUT_DIR}/nudt_dea_lite_0p005_pdfa_best_e367_retest_summary.json"

echo "DONE: archive + retest outputs written to ${OUT_DIR}"
```

授权：

```bash
chmod +x scripts/official/archive_retest_nudt_dea_lite_0p005.sh
```

运行：

```bash
cd /home/ly/DEA

CUDA_DEVICE=0 \
ROOT=/home/ly/DEA \
PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python \
DATASET_DIR=/home/ly/DEA/datasets/NUDT-SIRST \
RUN_DIR=/home/ly/DEA/weight/MSHNet-2026-07-07-03-24-31 \
OUT_DIR=/home/ly/DEA/repro_runs/dea_lite_0p005_nudt_archive_retest \
bash scripts/official/archive_retest_nudt_dea_lite_0p005.sh
```

---

## 7. 新增脚本 2：解析 test log 为 JSON

新建：

```text
tools/official/parse_dea_test_log.py
```

内容如下：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_metric(text: str, names: list[str]) -> float | None:
    for name in names:
        patterns = [
            rf"\b{name}\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            rf"\b{name}\b\s+([0-9]+(?:\.[0-9]+)?)",
        ]
        for pat in patterns:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            if matches:
                return float(matches[-1])
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--checkpoint_role", required=True, choices=["best_iou", "pdfa_best", "final", "baseline"])
    p.add_argument("--checkpoint_epoch", required=True, type=int)
    p.add_argument("--weight_path", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    weight_path = Path(args.weight_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.is_file():
        raise SystemExit(f"missing log: {log_path}")
    if not weight_path.is_file():
        raise SystemExit(f"missing weight: {weight_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")

    iou = parse_metric(text, ["IoU", "iou", "mIoU", "miou"])
    pd = parse_metric(text, ["PD", "Pd", "pd"])
    fa = parse_metric(text, ["FA", "Fa", "fa"])

    metrics_found = all(v is not None for v in (iou, pd, fa))

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "method": args.method,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint_epoch": args.checkpoint_epoch,
        "weight_path": str(weight_path),
        "weight_sha256": sha256_file(weight_path),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "metrics_found": metrics_found,
        "IoU": iou,
        "PD": pd,
        "FA": fa,
    }

    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if not metrics_found:
        raise SystemExit(
            "Could not parse IoU/PD/FA from log. Inspect log manually and update parser patterns. "
            f"Output written to {output_path}"
        )


if __name__ == "__main__":
    main()
```

授权 / 编译检查：

```bash
chmod +x tools/official/parse_dea_test_log.py
python3 -m py_compile tools/official/parse_dea_test_log.py
```

---

## 8. 新增脚本 3：与 MSHNet baseline 计算 delta

新建：

```text
tools/official/compare_dea_lite_against_baseline.py
```

内容如下：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: str) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"missing json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def metric(d: dict[str, Any], key: str) -> float:
    value = d.get(key)
    if value is None:
        raise SystemExit(f"missing metric {key} in {d}")
    return float(value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_json", required=True)
    p.add_argument("--candidate_json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min_delta_iou", type=float, default=0.0)
    p.add_argument("--min_delta_pd", type=float, default=0.0)
    p.add_argument("--max_delta_fa", type=float, default=0.0)
    args = p.parse_args()

    base = read_json(args.baseline_json)
    cand = read_json(args.candidate_json)

    b_iou, c_iou = metric(base, "IoU"), metric(cand, "IoU")
    b_pd, c_pd = metric(base, "PD"), metric(cand, "PD")
    b_fa, c_fa = metric(base, "FA"), metric(cand, "FA")

    delta = {
        "IoU": c_iou - b_iou,
        "PD": c_pd - b_pd,
        "FA": c_fa - b_fa,
    }

    gate_pass = bool(
        delta["IoU"] >= args.min_delta_iou
        and delta["PD"] >= args.min_delta_pd
        and delta["FA"] <= args.max_delta_fa
    )

    result = {
        "baseline": base,
        "candidate": cand,
        "delta": delta,
        "thresholds": {
            "min_delta_iou": args.min_delta_iou,
            "min_delta_pd": args.min_delta_pd,
            "max_delta_fa": args.max_delta_fa,
        },
        "gate_pass": gate_pass,
        "decision": "DEA_LITE_POSITIVE" if gate_pass else "DEA_LITE_GATE_FAIL",
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if not gate_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
```

授权 / 编译检查：

```bash
chmod +x tools/official/compare_dea_lite_against_baseline.py
python3 -m py_compile tools/official/compare_dea_lite_against_baseline.py
```

---

## 9. NUDT baseline JSON 手动固化

先把你已经给出的 MSHNet baseline 写成 JSON，作为 NUDT 对比锚点。

```bash
cd /home/ly/DEA
mkdir -p repro_runs/dea_lite_0p005_nudt_archive_retest

cat > repro_runs/dea_lite_0p005_nudt_archive_retest/nudt_mshnet_baseline_best_iou_summary.json <<'JSON'
{
  "dataset": "NUDT-SIRST",
  "method": "MSHNet-baseline",
  "checkpoint_role": "best_iou",
  "checkpoint_epoch": null,
  "IoU": 0.7538765773118212,
  "PD": 0.944973544973545,
  "FA": 24.381890354386297,
  "source": "user-reported new MSHNet baseline result",
  "source_run_dir": "/home/ly/DEA/weight/MSHNet-2026-07-07-04-42-06",
  "evidence_role": "reported_anchor_pending_retest",
  "note": "Use as a temporary comparison anchor. For a paper main table, archive and retest the MSHNet baseline checkpoint as well."
}
JSON
```

完成复测后，计算 best-IoU delta：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python tools/official/compare_dea_lite_against_baseline.py \
  --baseline_json repro_runs/dea_lite_0p005_nudt_archive_retest/nudt_mshnet_baseline_best_iou_summary.json \
  --candidate_json repro_runs/dea_lite_0p005_nudt_archive_retest/nudt_dea_lite_0p005_best_iou_e368_retest_summary.json \
  --output repro_runs/dea_lite_0p005_nudt_archive_retest/nudt_dea_lite_0p005_best_iou_vs_mshnet_delta.json \
  --min_delta_iou 0.0 \
  --min_delta_pd 0.0 \
  --max_delta_fa 0.0
```

计算 PD/FA-best delta：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python tools/official/compare_dea_lite_against_baseline.py \
  --baseline_json repro_runs/dea_lite_0p005_nudt_archive_retest/nudt_mshnet_baseline_best_iou_summary.json \
  --candidate_json repro_runs/dea_lite_0p005_nudt_archive_retest/nudt_dea_lite_0p005_pdfa_best_e367_retest_summary.json \
  --output repro_runs/dea_lite_0p005_nudt_archive_retest/nudt_dea_lite_0p005_pdfa_best_vs_mshnet_delta.json \
  --min_delta_iou 0.0 \
  --min_delta_pd 0.0 \
  --max_delta_fa 0.0
```

---

## 10. NUDT retest 通过标准

复测允许有极小浮动，但不应改变结论。

| Checkpoint | 预期角色 | 通过条件 |
|---|---|---|
| `best_iou_e368` | 主表 IoU checkpoint | IoU ≥ baseline，PD ≥ baseline，FA ≤ baseline |
| `pdfa_best_e367` | FA-control 诊断 checkpoint | IoU 不显著低于 baseline，PD 不明显下降，FA 明显低于 baseline |

推荐主表写法：

```text
Main NUDT table:
  MSHNet baseline best-IoU
  DEA-lite 0.005 best-IoU

Diagnostic / FA-control table:
  DEA-lite 0.005 PD/FA-best
```

不要把 `final epoch 399` 作为主结果。它可作为 training stability note，但主表应优先使用预先保存的 best checkpoint。

---

## 11. 下一步：NUAA paired evidence，不跑 0.01

### 11.1 先跑 NUAA MSHNet baseline

确认数据集目录：

```bash
ls -la /home/ly/DEA/datasets
find /home/ly/DEA/datasets -maxdepth 2 -type f | grep -Ei 'nuaa|train|test|img_idx' | head -50
```

如果目录名是 `NUAA-SIRST`：

```bash
cd /home/ly/DEA

export RUN_TAG=nuaa_mshnet_baseline_seed20260706_$(date +%Y%m%d_%H%M%S)
export LOG_DIR=/home/ly/DEA/repro_runs/$RUN_TAG
mkdir -p "$LOG_DIR"

find /home/ly/DEA/weight -maxdepth 1 -type d -name 'MSHNet-*' -printf '%f\n' \
  | sort > "$LOG_DIR/weight_dirs_before.txt"

CUDA_VISIBLE_DEVICES=0 nohup /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/NUAA-SIRST \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --epochs 400 \
  --lr 0.05 \
  --warm-epoch 5 \
  --mode train \
  --seed 20260706 \
  --deterministic true \
  --dea-lambda-single 0.0 \
  --dea-lambda-dec 0.0 \
  --dea-lambda-empty 0.0 \
  > "$LOG_DIR/train.log" 2>&1 &

echo $! > "$LOG_DIR/pid.txt"
echo "$LOG_DIR"
```

这就是 paired MSHNet baseline：DEA loss 全关，训练代码和随机种子与后续 DEA-lite 0.005 保持一致。查看进度：

```bash
tail -f "$LOG_DIR/train.log"
```

### 11.2 NUAA baseline 训练完成后立即归档

```bash
cd /home/ly/DEA

export PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python
if [[ -z "${LOG_DIR:-}" ]]; then
  echo "ERROR: set LOG_DIR to the NUAA baseline repro run dir first, e.g. /home/ly/DEA/repro_runs/nuaa_mshnet_baseline_seed20260706_YYYYmmdd_HHMMSS" >&2
  exit 2
fi

find /home/ly/DEA/weight -maxdepth 1 -type d -name 'MSHNet-*' -printf '%f\n' \
  | sort > "$LOG_DIR/weight_dirs_after.txt"

mapfile -t NEW_RUNS < <(comm -13 "$LOG_DIR/weight_dirs_before.txt" "$LOG_DIR/weight_dirs_after.txt")
if [[ "${#NEW_RUNS[@]}" -ne 1 ]]; then
  printf 'ERROR: expected exactly one new MSHNet run dir, got %d\n' "${#NEW_RUNS[@]}" >&2
  printf '%s\n' "${NEW_RUNS[@]}" >&2
  exit 3
fi

export NUAA_BASE_RUN="/home/ly/DEA/weight/${NEW_RUNS[0]}"
echo "$NUAA_BASE_RUN"

cp "$NUAA_BASE_RUN/weight.pkl" "$NUAA_BASE_RUN/weight_nuaa_mshnet_baseline_best_iou.pkl"
cp "$NUAA_BASE_RUN/checkpoint_best_iou.pkl" "$NUAA_BASE_RUN/checkpoint_nuaa_mshnet_baseline_best_iou.pkl"

read NUAA_BASE_EPOCH NUAA_BASE_IOU NUAA_BASE_PD NUAA_BASE_FA < <( "$PYTHON" - <<'PY'
import os, torch
run = os.environ["NUAA_BASE_RUN"]
ck = torch.load(os.path.join(run, "checkpoint_best_iou.pkl"), map_location="cpu", weights_only=False)
print(ck["epoch"], ck["iou"], ck["pd"], ck["fa"])
PY
)
export NUAA_BASE_EPOCH NUAA_BASE_IOU NUAA_BASE_PD NUAA_BASE_FA

echo "NUAA baseline best-IoU: epoch=$NUAA_BASE_EPOCH IoU=$NUAA_BASE_IOU PD=$NUAA_BASE_PD FA=$NUAA_BASE_FA"

sha256sum \
  "$NUAA_BASE_RUN/weight_nuaa_mshnet_baseline_best_iou.pkl" \
  "$NUAA_BASE_RUN/checkpoint_nuaa_mshnet_baseline_best_iou.pkl" \
  > repro_runs/nuaa_mshnet_baseline_artifacts.sha256
```

### 11.3 复测 NUAA baseline

```bash
cd /home/ly/DEA

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/NUAA-SIRST \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --seed 20260706 \
  --deterministic true \
  --weight-path "$NUAA_BASE_RUN/weight_nuaa_mshnet_baseline_best_iou.pkl" \
  2>&1 | tee repro_runs/nuaa_mshnet_baseline_best_iou_retest.log
```

解析：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python tools/official/parse_dea_test_log.py \
  --log repro_runs/nuaa_mshnet_baseline_best_iou_retest.log \
  --dataset NUAA-SIRST \
  --method MSHNet-baseline \
  --checkpoint_role baseline \
  --checkpoint_epoch "$NUAA_BASE_EPOCH" \
  --weight_path "$NUAA_BASE_RUN/weight_nuaa_mshnet_baseline_best_iou.pkl" \
  --output repro_runs/nuaa_mshnet_baseline_best_iou_summary.json
```

---

## 12. NUAA DEA-lite 0.005 训练命令

只有 NUAA baseline 完成并复测后，再跑 DEA-lite 0.005：

```bash
cd /home/ly/DEA

export PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python
export DATASET_NUAA=/home/ly/DEA/datasets/NUAA-SIRST

if [[ -z "${NUAA_BASE_RUN:-}" ]]; then
  echo "ERROR: set NUAA_BASE_RUN to the completed NUAA MSHNet baseline run dir first." >&2
  exit 2
fi

read NUAA_BASE_EPOCH NUAA_BASE_IOU NUAA_BASE_PD NUAA_BASE_FA < <( "$PYTHON" - <<'PY'
import os, torch
run = os.environ["NUAA_BASE_RUN"]
ck = torch.load(os.path.join(run, "checkpoint_best_iou.pkl"), map_location="cpu", weights_only=False)
print(ck["epoch"], ck["iou"], ck["pd"], ck["fa"])
PY
)
export NUAA_BASE_EPOCH NUAA_BASE_IOU NUAA_BASE_PD NUAA_BASE_FA

export NUAA_PDFA_MIN_IOU=$("$PYTHON" - <<'PY'
import os
print(float(os.environ["NUAA_BASE_IOU"]) - 0.01)
PY
)

export RUN_TAG=nuaa_dea_lite_0p005_seed20260706_$(date +%Y%m%d_%H%M%S)
export LOG_DIR=/home/ly/DEA/repro_runs/$RUN_TAG
mkdir -p "$LOG_DIR"

find /home/ly/DEA/weight -maxdepth 1 -type d -name 'MSHNet-*' -printf '%f\n' \
  | sort > "$LOG_DIR/weight_dirs_before.txt"

CUDA_VISIBLE_DEVICES=0 nohup "$PYTHON" -u main.py \
  --dataset-dir "$DATASET_NUAA" \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --epochs 400 \
  --lr 0.05 \
  --warm-epoch 5 \
  --mode train \
  --seed 20260706 \
  --deterministic true \
  --dea-lambda-single 0.005 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0 \
  --dea-tau 0.6 \
  --dea-ramp-epochs 80 \
  --dea-detach-evidence \
  --paired-baseline-iou "$NUAA_BASE_IOU" \
  --pd-fa-min-iou "$NUAA_PDFA_MIN_IOU" \
  --pd-fa-min-pd "$NUAA_BASE_PD" \
  --pd-fa-iou-margin 0.01 \
  > "$LOG_DIR/train.log" 2>&1 &

echo $! > "$LOG_DIR/pid.txt"
echo "$LOG_DIR"
```

查看进度：

```bash
tail -f "$LOG_DIR/train.log"
```

训练完成后按 NUDT 同样流程归档：

```bash
cd /home/ly/DEA

export PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python
if [[ -z "${LOG_DIR:-}" ]]; then
  echo "ERROR: set LOG_DIR to the NUAA DEA-lite 0.005 repro run dir first, e.g. /home/ly/DEA/repro_runs/nuaa_dea_lite_0p005_seed20260706_YYYYmmdd_HHMMSS" >&2
  exit 2
fi

find /home/ly/DEA/weight -maxdepth 1 -type d -name 'MSHNet-*' -printf '%f\n' \
  | sort > "$LOG_DIR/weight_dirs_after.txt"

mapfile -t NEW_RUNS < <(comm -13 "$LOG_DIR/weight_dirs_before.txt" "$LOG_DIR/weight_dirs_after.txt")
if [[ "${#NEW_RUNS[@]}" -ne 1 ]]; then
  printf 'ERROR: expected exactly one new MSHNet run dir, got %d\n' "${#NEW_RUNS[@]}" >&2
  printf '%s\n' "${NEW_RUNS[@]}" >&2
  exit 3
fi

export NUAA_DEA_RUN="/home/ly/DEA/weight/${NEW_RUNS[0]}"
echo "$NUAA_DEA_RUN"

cp "$NUAA_DEA_RUN/weight.pkl" "$NUAA_DEA_RUN/weight_nuaa_lambda_single_0p005_best_iou.pkl"
cp "$NUAA_DEA_RUN/checkpoint_best_iou.pkl" "$NUAA_DEA_RUN/checkpoint_nuaa_lambda_single_0p005_best_iou.pkl"

read NUAA_DEA_EPOCH NUAA_DEA_IOU NUAA_DEA_PD NUAA_DEA_FA < <( "$PYTHON" - <<'PY'
import os, torch
run = os.environ["NUAA_DEA_RUN"]
ck = torch.load(os.path.join(run, "checkpoint_best_iou.pkl"), map_location="cpu", weights_only=False)
print(ck["epoch"], ck["iou"], ck["pd"], ck["fa"])
PY
)
export NUAA_DEA_EPOCH NUAA_DEA_IOU NUAA_DEA_PD NUAA_DEA_FA

echo "NUAA DEA-lite 0.005 best-IoU: epoch=$NUAA_DEA_EPOCH IoU=$NUAA_DEA_IOU PD=$NUAA_DEA_PD FA=$NUAA_DEA_FA"

if [[ -s "$NUAA_DEA_RUN/weight_pd_fa_best.pkl" ]]; then
  cp "$NUAA_DEA_RUN/weight_pd_fa_best.pkl" "$NUAA_DEA_RUN/weight_nuaa_lambda_single_0p005_pdfa_best.pkl"
fi

if [[ -s "$NUAA_DEA_RUN/checkpoint_pd_fa_best.pkl" ]]; then
  cp "$NUAA_DEA_RUN/checkpoint_pd_fa_best.pkl" "$NUAA_DEA_RUN/checkpoint_nuaa_lambda_single_0p005_pdfa_best.pkl"
fi
```

复测 NUAA DEA-lite 0.005 best-IoU：

```bash
cd /home/ly/DEA

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/NUAA-SIRST \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --seed 20260706 \
  --deterministic true \
  --weight-path "$NUAA_DEA_RUN/weight_nuaa_lambda_single_0p005_best_iou.pkl" \
  2>&1 | tee repro_runs/nuaa_dea_lite_0p005_best_iou_retest.log
```

解析：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python tools/official/parse_dea_test_log.py \
  --log repro_runs/nuaa_dea_lite_0p005_best_iou_retest.log \
  --dataset NUAA-SIRST \
  --method DEA-lite-0.005 \
  --checkpoint_role best_iou \
  --checkpoint_epoch "$NUAA_DEA_EPOCH" \
  --weight_path "$NUAA_DEA_RUN/weight_nuaa_lambda_single_0p005_best_iou.pkl" \
  --output repro_runs/nuaa_dea_lite_0p005_best_iou_summary.json
```

NUAA paired delta：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python tools/official/compare_dea_lite_against_baseline.py \
  --baseline_json repro_runs/nuaa_mshnet_baseline_best_iou_summary.json \
  --candidate_json repro_runs/nuaa_dea_lite_0p005_best_iou_summary.json \
  --output repro_runs/nuaa_dea_lite_0p005_vs_mshnet_delta.json \
  --min_delta_iou 0.0 \
  --min_delta_pd -0.002 \
  --max_delta_fa 0.0
```

---

## 13. 什么时候再考虑 0.01

现在不要跑 0.01。

只有满足下面条件时再开 0.01：

```text
1. NUDT 0.005 复测通过；
2. IRSTD-1K 0.005 结果已整理；
3. NUAA baseline 完成；
4. NUAA DEA-lite 0.005 完成；
5. 三个数据集的 0.005 paired result 均为正或至少不负；
6. 论文叙事确定为 DEA-lite 0.005，而不是 lambda sweep。
```

0.01 的角色只能是：

```text
secondary sensitivity analysis
```

不能让它打断当前主线。

---

## 14. 当前 paper claim 建议

如果 NUAA 也正，claim 可以写成：

```text
DEA-lite is a lightweight evidence-aware regularization add-on for MSHNet-style infrared small target detection. Across NUDT-SIRST, IRSTD-1K, and NUAA, lambda_single=0.005 improves false-alarm control while preserving or improving target detection performance.
```

如果 NUAA 不正，但 NUDT + IRSTD-1K 正，则 claim 降级为：

```text
DEA-lite 0.005 shows promising false-alarm control on NUDT-SIRST and IRSTD-1K, while NUAA reveals dataset-dependent behavior.
```

不要写：

```text
DEA-lite universally improves all IRSTD datasets.
DEA-lite is full SOTA.
DEA-lite solves false alarms.
0.005 is globally optimal.
```

---

## 15. 最终执行顺序

```text
R0. 不改模型，不改 loss，不跑 0.01。
R1. 归档 NUDT DEA-lite 0.005 权重和 checkpoint。
R2. 复测 NUDT best-IoU 和 PD/FA-best。
R3. 生成 NUDT retest summary JSON 和 delta JSON。
R4. 整理 IRSTD-1K + NUDT 主表草稿。
R5. 训练 NUAA MSHNet baseline。
R6. 复测 NUAA baseline。
R7. 训练 NUAA DEA-lite 0.005。
R8. 复测 NUAA DEA-lite 0.005。
R9. 计算 NUAA paired delta。
R10. NUAA 也正，再考虑 seed / dataset extension / ablation。
R11. 只有三数据集 0.005 主线稳定后，再考虑 0.01 sensitivity。
```

---

## 16. 提交建议

当前只需要提交本文档；如果后续真的新增 `scripts/official/` 或 `tools/official/` 脚本，再单独提交脚本。不要提交 `datasets/`、`weight/`、checkpoint、prediction 或大型日志。

```bash
cd /home/ly/DEA

git checkout -b dea-lite-0p005-nudt-archive-nuaa-next

git add DEA_lite_0p005_NUDT_archive_retest_NUAA_next_plan.md

git status --short | grep -E 'weight/|datasets/|\.pkl|\.pth|\.tar' && {
  echo 'ERROR: large/data artifacts are staged or visible for commit. Unstage them.' >&2
  exit 1
} || true

git commit -m "Update DEA-lite 0.005 NUDT archive and NUAA execution plan"
```

---

## 17. 一句话结论

```text
你的判断是对的：现在不是继续调 0.01，也不是改模型。
现在应该保护 NUDT 0.005 正结果，复测确认可复现，然后补 NUAA paired baseline + DEA-lite 0.005。
```
