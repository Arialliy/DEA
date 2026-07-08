# Full DEA NUAA-first 接入方案与代码修改

> Canonical repo root: `/home/ly/DEA`  
> Current branch: `full-dea-prototype-nuaa-first`  
> Current state: `model/full_dea_mshnet.py` and `model/full_dea_loss.py` exist; 4 tests passed; `main.py` has not yet integrated Full DEA train/test paths.  
> Current decision: **进入接入阶段；不要启动 400-epoch NUAA 训练。**

---

## 0. Verdict

当前审计结果满足进入 **Full DEA 接入阶段**。

但是接入阶段只允许做：

```text
1. main.py CLI 接入
2. FullDEAMSHNet 构建路径接入
3. full_dea_loss 训练路径接入
4. test/eval 使用 y_final 路径接入
5. checkpoint / log metadata 接入
6. smoke runner 和 contract tests
```

仍然不允许做：

```text
1. 启动 400-epoch NUAA 正式训练
2. 改 NUAA split / metric / threshold
3. 把 Full DEA 写成已验证方法
4. 把 DEA-lite 结果冒充 Full DEA 结果
5. 在 Full DEA 接入未通过前跑 NUDT / IRSTD-1K
```

当前目标是：

```text
FullDEAMSHNet and full_dea_loss become reachable from main.py train/test without breaking existing MSHNet / DEA-lite paths.
```

---

## 1. Why this is the correct next step

之前已经明确：DEA-lite 不能包装成 Full DEA。只有新增显式 evidence decomposition、counterfactual path、inference evidence gate 的结构实现，才允许命名为 Full DEA。

当前分支已经是：

```text
full-dea-prototype-nuaa-first
```

并且已经有：

```text
model/full_dea_mshnet.py
model/full_dea_loss.py
```

所以现在不是继续写 protocol，也不是继续做 DEA-lite lambda sensitivity，而是把已有 Full DEA prototype 接入训练主入口。

---

## 2. Integration contract

### 2.1 Required model forward contract

`model/full_dea_mshnet.py` 必须导出：

```python
FullDEAMSHNet
```

`FullDEAMSHNet.forward(...)` 在 `return_dict=True` 时必须返回 dict，并至少包含：

```text
y_final             final inference-time evidence-calibrated logits
y_real              real prediction logits before or inside evidence calibration
y_cf                counterfactual prediction logits
target_evidence     target evidence tensor
clutter_evidence    clutter/background evidence tensor
masks               auxiliary multi-scale masks/list, compatible with existing MSHNet training loss
```

允许额外字段：

```text
feature
counterfactual_feature
evidence_gate
method_meta
```

### 2.2 Required loss contract

`model/full_dea_loss.py` 必须导出：

```python
full_dea_loss
```

`full_dea_loss(...)` 必须支持下面调用方式：

```python
loss_full_dea, log_dict = full_dea_loss(
    output=full_dea_output,
    gt=labels,
    epoch=epoch,
    warm_epoch=self.warm_epoch,
    lambda_evidence=self.args.full_dea_lambda_evidence,
    lambda_cf=self.args.full_dea_lambda_cf,
    lambda_gate=self.args.full_dea_lambda_gate,
    ramp_epochs=self.args.full_dea_ramp_epochs,
)
```

返回要求：

```text
loss_full_dea: torch.Tensor scalar
log_dict: dict[str, float | int | bool]
```

如果当前 `full_dea_loss.py` 不是这个签名，应优先改 `full_dea_loss.py` 适配这个 contract，而不是在 `main.py` 里写很多分支兼容。

---

## 3. Modify `main.py`

### 3.1 Add Full DEA imports

在现有 import 区域加入：

```python
from model.full_dea_mshnet import FullDEAMSHNet
from model.full_dea_loss import full_dea_loss
```

### 3.2 Add method validation helper

把下面代码加入 `str2bool(...)` 之后、`parse_args()` 之前：

```python
def validate_method_args(args):
    if args.use_full_dea:
        lite_lambdas = [
            args.dea_lambda_single,
            args.dea_lambda_dec,
            args.dea_lambda_empty,
        ]
        if any(float(v) != 0.0 for v in lite_lambdas):
            raise ValueError(
                'Full DEA and DEA-lite losses must not be enabled together. '
                'Set --dea-lambda-single 0 --dea-lambda-dec 0 --dea-lambda-empty 0.'
            )
        if args.full_dea_protocol != 'nuaa_first_v0':
            raise ValueError(
                'Full DEA prototype currently requires --full-dea-protocol nuaa_first_v0.'
            )
        for name in (
            'full_dea_lambda_evidence',
            'full_dea_lambda_cf',
            'full_dea_lambda_gate',
        ):
            if float(getattr(args, name)) < 0.0:
                raise ValueError('%s must be non-negative.' % name)
        if int(args.full_dea_ramp_epochs) < 0:
            raise ValueError('--full-dea-ramp-epochs must be non-negative.')
    return args


def build_model_from_args(args):
    if args.use_full_dea:
        return FullDEAMSHNet(3)
    return MSHNet(3)


def get_method_name(args):
    if args.use_full_dea:
        return 'FullDEA'
    if (
        args.dea_lambda_single > 0
        or args.dea_lambda_dec > 0
        or args.dea_lambda_empty > 0
    ):
        return 'DEA-lite'
    return 'MSHNet'


def get_method_metadata(args):
    return {
        'method': get_method_name(args),
        'use_full_dea': bool(args.use_full_dea),
        'full_dea_protocol': args.full_dea_protocol if args.use_full_dea else None,
        'full_dea_lambda_evidence': float(args.full_dea_lambda_evidence),
        'full_dea_lambda_cf': float(args.full_dea_lambda_cf),
        'full_dea_lambda_gate': float(args.full_dea_lambda_gate),
        'full_dea_ramp_epochs': int(args.full_dea_ramp_epochs),
        'dea_lambda_single': float(args.dea_lambda_single),
        'dea_lambda_dec': float(args.dea_lambda_dec),
        'dea_lambda_empty': float(args.dea_lambda_empty),
        'dataset_dir': args.dataset_dir,
        'seed': int(args.seed),
        'deterministic': bool(args.deterministic),
        'mode': args.mode,
    }


def validate_full_dea_output(output):
    if not isinstance(output, dict):
        raise RuntimeError('FullDEAMSHNet must return a dict when return_dict=True.')

    required = [
        'y_final',
        'y_real',
        'y_cf',
        'target_evidence',
        'clutter_evidence',
        'masks',
    ]
    missing = [key for key in required if key not in output]
    if missing:
        raise RuntimeError('Full DEA output missing keys: %s' % missing)

    return output


def forward_full_dea_model(model, data, tag):
    output = model(data, tag, return_dict=True)
    output = validate_full_dea_output(output)
    return output


def compute_multiscale_seg_loss(loss_fun, down, pred, masks, labels, warm_epoch, epoch):
    loss = loss_fun(pred, labels, warm_epoch, epoch)
    labels_for_scale = labels
    for j in range(len(masks)):
        if j > 0:
            labels_for_scale = down(labels_for_scale)
        loss = loss + loss_fun(masks[j], labels_for_scale, warm_epoch, epoch)
    return loss / (len(masks) + 1)
```

### 3.3 Add CLI arguments

在 `parse_args()` 里 DEA-lite 参数后面加入：

```python
    parser.add_argument('--use-full-dea', action='store_true')
    parser.add_argument('--full-dea-protocol', type=str, default='')
    parser.add_argument('--full-dea-lambda-evidence', type=float, default=1.0)
    parser.add_argument('--full-dea-lambda-cf', type=float, default=1.0)
    parser.add_argument('--full-dea-lambda-gate', type=float, default=0.5)
    parser.add_argument('--full-dea-ramp-epochs', type=int, default=80)
    parser.add_argument('--full-dea-debug', action='store_true')
```

把 `parse_args()` 末尾改成：

```python
    args = parser.parse_args()
    args = validate_method_args(args)
    return args
```

### 3.4 Replace model construction

在 `Trainer.__init__` 里，把：

```python
model = MSHNet(3)
```

替换成：

```python
model = build_model_from_args(args)
```

在 `self.model = model` 后加入：

```python
self.method_meta = get_method_metadata(args)
```

### 3.5 Modify training forward path

在 `Trainer.train(...)` 里，找到当前这段逻辑：

```python
tag = epoch > self.warm_epoch
use_dea = self.use_dea(epoch)
if use_dea:
    masks, pred, dea_out = self.model(
        data,
        tag,
        return_dea=True,
        dea_detach_evidence=self.args.dea_detach_evidence,
    )
else:
    masks, pred = self.model(data, tag)
    dea_out = None
loss = 0
loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
labels_for_scale = labels
for j in range(len(masks)):
    if j>0:
        labels_for_scale = self.down(labels_for_scale)
    loss = loss + self.loss_fun(masks[j], labels_for_scale, self.warm_epoch, epoch)
loss = loss / (len(masks)+1)
```

替换为：

```python
tag = epoch > self.warm_epoch
use_dea = self.use_dea(epoch)
full_dea_out = None

if self.args.use_full_dea:
    full_dea_out = forward_full_dea_model(self.model, data, tag)
    masks = full_dea_out['masks']
    pred = full_dea_out['y_final']
    dea_out = None
elif use_dea:
    masks, pred, dea_out = self.model(
        data,
        tag,
        return_dea=True,
        dea_detach_evidence=self.args.dea_detach_evidence,
    )
else:
    masks, pred = self.model(data, tag)
    dea_out = None

loss = compute_multiscale_seg_loss(
    self.loss_fun,
    self.down,
    pred,
    masks,
    labels,
    self.warm_epoch,
    epoch,
)
```

在现有 `if use_dea:` DEA-lite loss 块之前加入 Full DEA loss 块：

```python
if self.args.use_full_dea:
    loss_full_dea, full_dea_log = full_dea_loss(
        output=full_dea_out,
        gt=labels,
        epoch=epoch,
        warm_epoch=self.warm_epoch,
        lambda_evidence=self.args.full_dea_lambda_evidence,
        lambda_cf=self.args.full_dea_lambda_cf,
        lambda_gate=self.args.full_dea_lambda_gate,
        ramp_epochs=self.args.full_dea_ramp_epochs,
    )
    loss = loss + loss_full_dea

    if self.args.full_dea_debug and i == 0:
        msg = [
            'full_dea_loss=%.6f' % float(loss_full_dea.detach().cpu()),
        ]
        for key, value in full_dea_log.items():
            try:
                msg.append('%s=%.6f' % (key, float(value)))
            except (TypeError, ValueError):
                msg.append('%s=%s' % (key, str(value)))
        print('[FULL DEA DEBUG] ' + ' | '.join(msg))
```

保留现有 `if use_dea:` DEA-lite loss 块，但改成 `elif use_dea:`，避免 Full DEA 和 DEA-lite 同时加 loss：

```python
elif use_dea:
    ramp = get_dea_ramp(epoch, self.warm_epoch, self.args.dea_ramp_epochs)
    cur_lambda_single = self.args.dea_lambda_single * ramp
    cur_lambda_dec = self.args.dea_lambda_dec * ramp
    cur_lambda_empty = self.args.dea_lambda_empty * ramp
    loss_dea, dea_log = dea_lite_loss(
        dea_out=dea_out,
        z_full=pred,
        gt=labels,
        lambda_single=cur_lambda_single,
        lambda_dec=cur_lambda_dec,
        lambda_empty=cur_lambda_empty,
        tau=self.args.dea_tau,
    )
    loss = loss + loss_dea
    self.save_dea_debug(epoch, i, data, labels, pred, dea_out)
    if self.args.save_dea_debug and self.args.dea_debug_interval > 0 and i % self.args.dea_debug_interval == 0:
        dea_ratio = (loss_dea.detach() / (loss_seg_for_debug + 1e-6)).item()
        msg = [
            'dea_ratio=%.4f' % dea_ratio,
            'lambda_single=%.6f' % cur_lambda_single,
            'lambda_empty=%.6f' % cur_lambda_empty,
            'lambda_dec=%.6f' % cur_lambda_dec,
        ]
        for key, value in dea_log.items():
            try:
                msg.append('%s=%.6f' % (key, float(value)))
            except (TypeError, ValueError):
                pass
        print('[DEA DEBUG] ' + ' | '.join(msg))
```

### 3.6 Modify evaluation forward path

在 `Trainer.test(...)` 里，把：

```python
_, pred = self.model(data, tag)
```

替换成：

```python
if self.args.use_full_dea:
    full_dea_out = forward_full_dea_model(self.model, data, tag)
    pred = full_dea_out['y_final']
else:
    _, pred = self.model(data, tag)
```

这样 test/eval 只使用 `y_final`，而不是 `y_real` 或 `y_cf`。

### 3.7 Add method metadata to checkpoints

在 `checkpoint_best_iou.pkl`、`checkpoint_pd_fa_best.pkl` 和 `checkpoint.pkl` 三个 states dict 里加入：

```python
"method_meta": self.method_meta,
"use_full_dea": bool(self.args.use_full_dea),
"full_dea_protocol": self.args.full_dea_protocol if self.args.use_full_dea else None,
```

例如 `best_iou_states` 应包含：

```python
best_iou_states = {
    "net": self.model.state_dict(),
    "optimizer": self.optimizer.state_dict(),
    "epoch": epoch,
    "iou": mean_IoU,
    "pd": current_pd,
    "fa": current_fa,
    "best_iou": self.best_iou,
    "best_pd_fa": self.best_pd_fa,
    "best_pd_fa_iou": self.best_pd_fa_iou,
    "best_pd_fa_pd": self.best_pd_fa_pd,
    "best_pd_fa_epoch": self.best_pd_fa_epoch,
    "method_meta": self.method_meta,
    "use_full_dea": bool(self.args.use_full_dea),
    "full_dea_protocol": self.args.full_dea_protocol if self.args.use_full_dea else None,
}
```

对 `pd_fa_states` 和 `latest_states` 做同样修改。

---

## 4. Add integration smoke runner

Create:

```text
scripts/official/run_full_dea_nuaa_first_smoke.sh
```

Content:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
CUDA_DEVICE=${CUDA_DEVICE:-0}
DATASET_DIR=${DATASET_DIR:-${ROOT}/datasets/NUAA-SIRST}
OUT_DIR=${OUT_DIR:-${ROOT}/repro_runs/full_dea_nuaa_first_smoke}

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "ERROR: missing DATASET_DIR: ${DATASET_DIR}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" -u main.py \
  --dataset-dir "${DATASET_DIR}" \
  --batch-size 2 \
  --num-workers 0 \
  --pin-memory false \
  --epochs 1 \
  --lr 0.05 \
  --warm-epoch 0 \
  --mode train \
  --seed 20260706 \
  --deterministic true \
  --use-full-dea \
  --full-dea-protocol nuaa_first_v0 \
  --full-dea-lambda-evidence 1.0 \
  --full-dea-lambda-cf 1.0 \
  --full-dea-lambda-gate 0.5 \
  --full-dea-ramp-epochs 1 \
  --full-dea-debug \
  --dea-lambda-single 0.0 \
  --dea-lambda-dec 0.0 \
  --dea-lambda-empty 0.0 \
  2>&1 | tee "${OUT_DIR}/train_smoke.log"

echo "DONE: Full DEA NUAA-first smoke completed."
```

Authorize and check:

```bash
cd /home/ly/DEA
chmod +x scripts/official/run_full_dea_nuaa_first_smoke.sh
bash -n scripts/official/run_full_dea_nuaa_first_smoke.sh
```

Run smoke only:

```bash
cd /home/ly/DEA

CUDA_DEVICE=0 \
ROOT=/home/ly/DEA \
PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python \
DATASET_DIR=/home/ly/DEA/datasets/NUAA-SIRST \
OUT_DIR=/home/ly/DEA/repro_runs/full_dea_nuaa_first_smoke \
bash scripts/official/run_full_dea_nuaa_first_smoke.sh
```

This is not the 400-epoch experiment. It is only a train/eval path integration smoke.

---

## 5. Add main-path contract tests

Create:

```text
tests/test_full_dea_main_integration.py
```

Content:

```python
from argparse import Namespace

import pytest

from main import validate_method_args, get_method_name


def make_args(**overrides):
    args = dict(
        use_full_dea=False,
        full_dea_protocol='',
        full_dea_lambda_evidence=1.0,
        full_dea_lambda_cf=1.0,
        full_dea_lambda_gate=0.5,
        full_dea_ramp_epochs=80,
        dea_lambda_single=0.0,
        dea_lambda_dec=0.0,
        dea_lambda_empty=0.0,
        dataset_dir='/home/ly/DEA/datasets/NUAA-SIRST',
        seed=20260706,
        deterministic=True,
        mode='train',
    )
    args.update(overrides)
    return Namespace(**args)


def test_full_dea_requires_protocol():
    args = make_args(use_full_dea=True, full_dea_protocol='')
    with pytest.raises(ValueError):
        validate_method_args(args)


def test_full_dea_rejects_dea_lite_lambdas():
    args = make_args(
        use_full_dea=True,
        full_dea_protocol='nuaa_first_v0',
        dea_lambda_single=0.005,
    )
    with pytest.raises(ValueError):
        validate_method_args(args)


def test_full_dea_method_name():
    args = make_args(use_full_dea=True, full_dea_protocol='nuaa_first_v0')
    args = validate_method_args(args)
    assert get_method_name(args) == 'FullDEA'


def test_dea_lite_method_name_still_works():
    args = make_args(dea_lambda_single=0.005)
    args = validate_method_args(args)
    assert get_method_name(args) == 'DEA-lite'


def test_mshnet_method_name_still_works():
    args = make_args()
    args = validate_method_args(args)
    assert get_method_name(args) == 'MSHNet'
```

Run:

```bash
cd /home/ly/DEA
python3 -m pytest tests/test_full_dea_main_integration.py
```

---

## 6. Add checkpoint metadata audit helper

Create:

```text
tools/official/check_full_dea_checkpoint_meta.py
```

Content:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--expect_full_dea', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        raise SystemExit(f'missing checkpoint: {ckpt_path}')

    ckpt = load_torch(ckpt_path)
    meta = ckpt.get('method_meta', {}) if isinstance(ckpt, dict) else {}

    checks = {
        'has_method_meta': isinstance(meta, dict) and len(meta) > 0,
        'use_full_dea_matches': bool(meta.get('use_full_dea', False)) == bool(args.expect_full_dea),
    }

    if args.expect_full_dea:
        checks.update(
            {
                'method_is_full_dea': meta.get('method') == 'FullDEA',
                'protocol_is_nuaa_first_v0': meta.get('full_dea_protocol') == 'nuaa_first_v0',
            }
        )

    result = {
        'checkpoint': str(ckpt_path),
        'expect_full_dea': bool(args.expect_full_dea),
        'method_meta': meta,
        'checks': checks,
        'pass': all(bool(v) for v in checks.values()),
    }

    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')

    if not result['pass']:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
```

Check:

```bash
cd /home/ly/DEA
chmod +x tools/official/check_full_dea_checkpoint_meta.py
python3 -m py_compile tools/official/check_full_dea_checkpoint_meta.py
```

---

## 7. Required validation before any 400-epoch NUAA training

Run these first:

```bash
cd /home/ly/DEA

python3 -m py_compile \
  main.py \
  model/full_dea_mshnet.py \
  model/full_dea_loss.py \
  tools/official/check_full_dea_checkpoint_meta.py

python3 -m pytest \
  tests/test_full_dea_main_integration.py \
  tests/test_full_dea_shapes.py \
  tests/test_full_dea_counterfactual_path.py

bash -n scripts/official/run_full_dea_nuaa_first_smoke.sh

git diff --check
```

Then run only smoke:

```bash
cd /home/ly/DEA

CUDA_DEVICE=0 \
ROOT=/home/ly/DEA \
PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python \
DATASET_DIR=/home/ly/DEA/datasets/NUAA-SIRST \
OUT_DIR=/home/ly/DEA/repro_runs/full_dea_nuaa_first_smoke \
bash scripts/official/run_full_dea_nuaa_first_smoke.sh
```

After smoke, inspect:

```bash
cd /home/ly/DEA

tail -50 repro_runs/full_dea_nuaa_first_smoke/train_smoke.log
ls -td weight/MSHNet-* | head -3
```

If the smoke creates a checkpoint, audit metadata:

```bash
cd /home/ly/DEA

SMOKE_RUN=$(ls -td weight/MSHNet-* | head -1)

python3 tools/official/check_full_dea_checkpoint_meta.py \
  --checkpoint "$SMOKE_RUN/checkpoint.pkl" \
  --expect_full_dea \
  --output repro_runs/full_dea_nuaa_first_smoke/checkpoint_meta_check.json
```

---

## 8. Gate after integration smoke

### PASS

接入阶段通过条件：

```text
1. py_compile pass
2. pytest pass
3. bash -n pass
4. smoke train executes one epoch without crash
5. test path uses y_final without crash
6. checkpoint metadata says method=FullDEA and protocol=nuaa_first_v0
7. DEA-lite and MSHNet paths still pass existing tests
```

### FAIL

任一失败时：

```text
1. Do not start 400-epoch NUAA training.
2. Fix integration bug.
3. Re-run smoke.
4. Do not change NUAA gate or baseline numbers.
```

---

## 9. Only after integration PASS: prepare NUAA-first official run

Only after the integration smoke passes, create a separate official runner:

```text
scripts/official/run_full_dea_nuaa_first_official.sh
```

Do not create or run it before smoke passes.

The official NUAA-first gate remains:

```text
Baseline MSHNet NUAA best-IoU:
  IoU 0.7461767423
  PD  0.9619771863
  FA  25.3124771831

Full DEA first gate:
  IoU >= 0.7461767423
  PD  >= 0.9569771863
  FA  <= 25.3124771831

Also:
  Full DEA must outperform DEA-lite 0.005 on NUAA.
```

---

## 10. Commit plan

Commit only integration code and tests:

```bash
cd /home/ly/DEA

git status --short | grep -E '(^|/)(datasets|weight|repro_runs)/|\.pkl|\.pth|\.tar|\.pt$' && {
  echo "ERROR: data/weight/repro artifacts visible. Do not commit them." >&2
  exit 1
} || true

git add \
  main.py \
  model/full_dea_mshnet.py \
  model/full_dea_loss.py \
  scripts/official/run_full_dea_nuaa_first_smoke.sh \
  tools/official/check_full_dea_checkpoint_meta.py \
  tests/test_full_dea_main_integration.py \
  tests/test_full_dea_shapes.py \
  tests/test_full_dea_counterfactual_path.py

git commit -m "Integrate Full DEA prototype into main train test path"
```

Do not commit:

```text
datasets/
weight/
repro_runs/
*.pkl
*.pth
*.tar
*.pt
```

---

## 11. One-line conclusion

```text
Yes, the audit satisfies entering integration stage.
Next: wire FullDEAMSHNet/full_dea_loss into main.py, add metadata and smoke tests, run only integration smoke, and postpone 400-epoch NUAA training until the smoke path passes.
```
