#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
CUDA_DEVICE=${CUDA_DEVICE:-0}
DATASET_DIR=${DATASET_DIR:-${ROOT}/datasets/NUAA-SIRST}
OUT_DIR=${OUT_DIR:-${ROOT}/repro_runs/full_dea_v3_tps_nuaa_first_gate}
INIT_FROM_BASELINE=${INIT_FROM_BASELINE:-${ROOT}/weight/MSHNet-2026-07-08-02-11-49/checkpoint_nuaa_mshnet_baseline_best_iou_e381.pkl}
BASELINE_SUMMARY=${BASELINE_SUMMARY:-}
DEA_LITE_SUMMARY=${DEA_LITE_SUMMARY:-}

EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PIN_MEMORY=${PIN_MEMORY:-false}
LR=${LR:-0.005}
SEED=${SEED:-20260710}
DETERMINISTIC=${DETERMINISTIC:-true}

FULL_DEA_LAMBDA=${FULL_DEA_LAMBDA:-0.5}
FULL_DEA_RAMP_EPOCHS=${FULL_DEA_RAMP_EPOCHS:-10}
FULL_DEA_START_EPOCH=${FULL_DEA_START_EPOCH:-0}
FULL_DEA_FREEZE_BACKBONE_EPOCHS=${FULL_DEA_FREEZE_BACKBONE_EPOCHS:-5}
FULL_DEA_TAU_BASE=${FULL_DEA_TAU_BASE:-0.45}
FULL_DEA_TAU_TARGET=${FULL_DEA_TAU_TARGET:-0.45}
FULL_DEA_TAU_SCALE=${FULL_DEA_TAU_SCALE:-0.45}
FULL_DEA_MAX_HARD_BG_RATIO=${FULL_DEA_MAX_HARD_BG_RATIO:-0.003}
FULL_DEA_SAFE_KERNEL=${FULL_DEA_SAFE_KERNEL:-15}
FULL_DEA_PROTECT_KERNEL=${FULL_DEA_PROTECT_KERNEL:-9}
FULL_DEA_HARD_MIN_AREA=${FULL_DEA_HARD_MIN_AREA:-1}
FULL_DEA_HARD_MAX_AREA=${FULL_DEA_HARD_MAX_AREA:-256}
FULL_DEA_DEBUG=${FULL_DEA_DEBUG:-true}
FULL_DEA_DEBUG_INTERVAL=${FULL_DEA_DEBUG_INTERVAL:-60}

BASELINE_IOU=${BASELINE_IOU:-0.7461767422765062}
BASELINE_PD=${BASELINE_PD:-0.9619771863117871}
BASELINE_FA=${BASELINE_FA:-25.312477183119157}
PD_TOLERANCE=${PD_TOLERANCE:-0.005}
PD_MIN=${PD_MIN:-0.9569771863117871}
DEA_LITE_IOU=${DEA_LITE_IOU:-0.7126024590163934}
DEA_LITE_PD=${DEA_LITE_PD:-0.935361216730038}
DEA_LITE_FA=${DEA_LITE_FA:-27.522862514602807}

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

for path in "${DATASET_DIR}" "${INIT_FROM_BASELINE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing path: ${path}" >&2
    exit 2
  fi
done

if [[ -z "${BASELINE_SUMMARY}" ]]; then
  BASELINE_SUMMARY="${OUT_DIR}/nuaa_mshnet_reference_summary.json"
  cat > "${BASELINE_SUMMARY}" <<JSON
{
  "dataset": "NUAA-SIRST",
  "method": "MSHNet-baseline",
  "checkpoint_role": "frozen_reference_best_iou",
  "checkpoint_epoch": 381,
  "IoU": ${BASELINE_IOU},
  "PD": ${BASELINE_PD},
  "FA": ${BASELINE_FA}
}
JSON
fi

if [[ -z "${DEA_LITE_SUMMARY}" ]]; then
  DEA_LITE_SUMMARY="${OUT_DIR}/nuaa_dea_lite_0p005_reference_summary.json"
  cat > "${DEA_LITE_SUMMARY}" <<JSON
{
  "dataset": "NUAA-SIRST",
  "method": "DEA-lite-0.005",
  "checkpoint_role": "frozen_negative_reference_best_iou",
  "checkpoint_epoch": 324,
  "IoU": ${DEA_LITE_IOU},
  "PD": ${DEA_LITE_PD},
  "FA": ${DEA_LITE_FA}
}
JSON
fi

CMD=(
  "${PYTHON}" -u main.py
  --model-type full_dea
  --full-dea-version v3
  --dataset-dir "${DATASET_DIR}"
  --mode train
  --epochs "${EPOCHS}"
  --warm-epoch -1
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --pin-memory "${PIN_MEMORY}"
  --lr "${LR}"
  --seed "${SEED}"
  --deterministic "${DETERMINISTIC}"
  --init-from-baseline "${INIT_FROM_BASELINE}"
  --full-dea-lambda "${FULL_DEA_LAMBDA}"
  --full-dea-ramp-epochs "${FULL_DEA_RAMP_EPOCHS}"
  --full-dea-start-epoch "${FULL_DEA_START_EPOCH}"
  --full-dea-freeze-backbone-epochs "${FULL_DEA_FREEZE_BACKBONE_EPOCHS}"
  --full-dea-tau-base "${FULL_DEA_TAU_BASE}"
  --full-dea-tau-target "${FULL_DEA_TAU_TARGET}"
  --full-dea-tau-scale "${FULL_DEA_TAU_SCALE}"
  --full-dea-max-hard-bg-ratio "${FULL_DEA_MAX_HARD_BG_RATIO}"
  --full-dea-safe-kernel "${FULL_DEA_SAFE_KERNEL}"
  --full-dea-protect-kernel "${FULL_DEA_PROTECT_KERNEL}"
  --full-dea-hard-min-area "${FULL_DEA_HARD_MIN_AREA}"
  --full-dea-hard-max-area "${FULL_DEA_HARD_MAX_AREA}"
  --pd-fa-min-pd "${PD_MIN}"
  --pd-fa-min-iou "${BASELINE_IOU}"
  --paired-baseline-iou "${BASELINE_IOU}"
  --pd-fa-iou-margin 0.0
)

if [[ "${FULL_DEA_DEBUG}" == "true" ]]; then
  CMD+=(--full-dea-debug --dea-debug-interval "${FULL_DEA_DEBUG_INTERVAL}")
fi

printf '%q ' "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}" "${CMD[@]}" > "${OUT_DIR}/command.txt"
printf '\n' >> "${OUT_DIR}/command.txt"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/train.log"

RUN_DIR="$(
  find "${ROOT}/weight" -maxdepth 1 -type d -name 'FullDEA-v3-TPS-*' -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
)"

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: could not locate latest FullDEA-v3-TPS run dir" >&2
  exit 3
fi

ln -sfn "${RUN_DIR}" "${OUT_DIR}/run_dir"
echo "${RUN_DIR}" > "${OUT_DIR}/run_dir.txt"

"${PYTHON}" tools/official/write_dea_checkpoint_summary.py \
  --checkpoint "${RUN_DIR}/checkpoint_best_iou.pkl" \
  --weight "${RUN_DIR}/weight.pkl" \
  --dataset NUAA-SIRST \
  --method FullDEA-v3-TPS \
  --checkpoint_role best_iou \
  --source_run_dir "${RUN_DIR}" \
  --output "${OUT_DIR}/full_dea_v3_tps_nuaa_best_iou_summary.json"

"${PYTHON}" tools/official/compare_full_dea_gate.py \
  --candidate_json "${OUT_DIR}/full_dea_v3_tps_nuaa_best_iou_summary.json" \
  --baseline_json "${BASELINE_SUMMARY}" \
  --reference_json "${DEA_LITE_SUMMARY}" \
  --require_better_reference \
  --iou_min "${BASELINE_IOU}" \
  --pd_min "${PD_MIN}" \
  --fa_max "${BASELINE_FA}" \
  --allow_gate_fail \
  --output "${OUT_DIR}/full_dea_v3_tps_nuaa_best_iou_gate.json"

"${PYTHON}" tools/official/analyze_dea_epoch_metrics.py \
  --epoch_metric_log "${RUN_DIR}/epoch_metric.log" \
  --baseline_iou "${BASELINE_IOU}" \
  --baseline_pd "${BASELINE_PD}" \
  --baseline_fa "${BASELINE_FA}" \
  --min_delta_iou 0.0 \
  --min_delta_pd "-${PD_TOLERANCE}" \
  --max_delta_fa 0.0 \
  --output "${OUT_DIR}/full_dea_v3_tps_nuaa_epoch_gate_audit.json"

"${PYTHON}" tools/official/audit_full_dea_decomposition.py \
  --dataset-dir "${DATASET_DIR}" \
  --weight "${RUN_DIR}/weight.pkl" \
  --full-dea-version v3 \
  --batch-size 1 \
  --num-workers 0 \
  --base-size 256 \
  --tau-base "${FULL_DEA_TAU_BASE}" \
  --tau-target "${FULL_DEA_TAU_TARGET}" \
  --tau-scale "${FULL_DEA_TAU_SCALE}" \
  --safe-kernel "${FULL_DEA_SAFE_KERNEL}" \
  --protect-kernel "${FULL_DEA_PROTECT_KERNEL}" \
  --hard-min-area "${FULL_DEA_HARD_MIN_AREA}" \
  --hard-max-area "${FULL_DEA_HARD_MAX_AREA}" \
  --max-hard-bg-ratio "${FULL_DEA_MAX_HARD_BG_RATIO}" \
  --baseline-iou "${BASELINE_IOU}" \
  --baseline-pd "${BASELINE_PD}" \
  --baseline-fa "${BASELINE_FA}" \
  --pd-tolerance "${PD_TOLERANCE}" \
  --output "${OUT_DIR}/full_dea_v3_tps_nuaa_decomposition_audit.json"

TEST_LOG="${OUT_DIR}/full_dea_v3_tps_nuaa_best_iou_test.log"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" -u main.py \
  --model-type full_dea \
  --full-dea-version v3 \
  --dataset-dir "${DATASET_DIR}" \
  --mode test \
  --warm-epoch -1 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --pin-memory "${PIN_MEMORY}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --weight-path "${RUN_DIR}/weight.pkl" \
  2>&1 | tee "${TEST_LOG}"

"${PYTHON}" tools/official/parse_dea_test_log.py \
  --log "${TEST_LOG}" \
  --dataset NUAA-SIRST \
  --method FullDEA-v3-TPS \
  --checkpoint_role best_iou \
  --checkpoint_epoch "$("${PYTHON}" - <<PY
import torch
ck = torch.load("${RUN_DIR}/checkpoint_best_iou.pkl", map_location="cpu", weights_only=False)
print(int(ck["epoch"]))
PY
)" \
  --weight_path "${RUN_DIR}/weight.pkl" \
  --extra run_dir="${RUN_DIR}" \
  --output "${OUT_DIR}/full_dea_v3_tps_nuaa_best_iou_test_summary.json"

cat > "${OUT_DIR}/MANIFEST.json" <<JSON
{
  "stage": "P1_NUAA_FIRST_GATE",
  "method": "FullDEA-v3-TPS",
  "dataset": "NUAA-SIRST",
  "run_dir": "${RUN_DIR}",
  "train_log": "${OUT_DIR}/train.log",
  "baseline_summary": "${BASELINE_SUMMARY}",
  "dea_lite_negative_reference": "${DEA_LITE_SUMMARY}",
  "gate_best_iou_json": "${OUT_DIR}/full_dea_v3_tps_nuaa_best_iou_gate.json",
  "epoch_gate_audit_json": "${OUT_DIR}/full_dea_v3_tps_nuaa_epoch_gate_audit.json",
  "decomposition_audit_json": "${OUT_DIR}/full_dea_v3_tps_nuaa_decomposition_audit.json",
  "epochs": ${EPOCHS},
  "lr": ${LR},
  "seed": ${SEED},
  "deterministic": ${DETERMINISTIC},
  "init_from_baseline": "${INIT_FROM_BASELINE}"
}
JSON

echo "DONE: FullDEA-v3-TPS NUAA first-gate outputs written to ${OUT_DIR}"
