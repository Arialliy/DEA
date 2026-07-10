#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
CUDA_DEVICE=${CUDA_DEVICE:-0}
SRC_DATASET=${SRC_DATASET:-${ROOT}/datasets/NUAA-SIRST}
OUT_DIR=${OUT_DIR:-${ROOT}/repro_runs/full_dea_v4_crr_smoke}
SMOKE_DATASET="${OUT_DIR}/NUAA-SIRST-smoke"
INIT_FROM_BASELINE=${INIT_FROM_BASELINE:-}

cd "${ROOT}"
mkdir -p "${OUT_DIR}" "${SMOKE_DATASET}/img_idx"

ln -sfn "${SRC_DATASET}/images" "${SMOKE_DATASET}/images"
ln -sfn "${SRC_DATASET}/masks" "${SMOKE_DATASET}/masks"

head -n 8 "${SRC_DATASET}/img_idx/train_NUAA-SIRST.txt" \
  > "${SMOKE_DATASET}/img_idx/train_NUAA-SIRST-smoke.txt"
head -n 4 "${SRC_DATASET}/img_idx/test_NUAA-SIRST.txt" \
  > "${SMOKE_DATASET}/img_idx/test_NUAA-SIRST-smoke.txt"

CMD=(
  "${PYTHON}" -u main.py
  --model-type full_dea
  --full-dea-version v4
  --dataset-dir "${SMOKE_DATASET}"
  --mode train
  --epochs 1
  --warm-epoch 0
  --batch-size 2
  --num-workers 0
  --pin-memory false
  --lr 0.005
  --seed 20260710
  --deterministic true
  --full-dea-lambda 0.5
  --full-dea-ramp-epochs 1
  --full-dea-start-epoch 0
  --full-dea-freeze-backbone-epochs 1
  --full-dea-tau-base 0.45
  --full-dea-tau-target 0.45
  --full-dea-tau-scale 0.45
  --full-dea-max-hard-bg-ratio 0.003
  --full-dea-safe-kernel 15
  --full-dea-protect-kernel 9
  --full-dea-hard-min-area 1
  --full-dea-hard-max-area 256
  --full-dea-debug
  --dea-debug-interval 1
)

if [[ -n "${INIT_FROM_BASELINE}" ]]; then
  CMD+=(--init-from-baseline "${INIT_FROM_BASELINE}")
fi

printf '%q ' "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}" "${CMD[@]}" \
  > "${OUT_DIR}/command.txt"
printf '\n' >> "${OUT_DIR}/command.txt"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}" 2>&1 \
  | tee "${OUT_DIR}/train_smoke.log"

echo "DONE: FullDEA-v4-CRR smoke outputs written to ${OUT_DIR}"
