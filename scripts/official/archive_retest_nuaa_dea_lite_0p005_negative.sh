#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
CUDA_DEVICE=${CUDA_DEVICE:-0}
DATASET_DIR=${DATASET_DIR:-${ROOT}/datasets/NUAA-SIRST}
OUT_DIR=${OUT_DIR:-${ROOT}/repro_runs/dea_lite_0p005_nuaa_negative_archive}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PIN_MEMORY=${PIN_MEMORY:-false}
SEED=${SEED:-20260706}
DETERMINISTIC=${DETERMINISTIC:-true}

: "${BASE_RUN:?BASE_RUN is required, e.g. /home/ly/DEA/weight/MSHNet-... for NUAA baseline}"
: "${DEA_RUN:?DEA_RUN is required, e.g. /home/ly/DEA/weight/MSHNet-... for NUAA DEA-lite 0.005}"
export BASE_RUN DEA_RUN

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

for d in "${BASE_RUN}" "${DEA_RUN}" "${DATASET_DIR}"; do
  if [[ ! -e "${d}" ]]; then
    echo "ERROR: missing path: ${d}" >&2
    exit 2
  fi
done

for f in weight.pkl checkpoint_best_iou.pkl epoch_metric.log; do
  if [[ ! -s "${BASE_RUN}/${f}" ]]; then
    echo "ERROR: baseline artifact missing or empty: ${BASE_RUN}/${f}" >&2
    exit 3
  fi
  if [[ ! -s "${DEA_RUN}/${f}" ]]; then
    echo "ERROR: DEA artifact missing or empty: ${DEA_RUN}/${f}" >&2
    exit 4
  fi
done

read -r BASE_EPOCH BASE_IOU BASE_PD BASE_FA < <("${PYTHON}" - <<'PY'
import os
import torch

ck = torch.load(os.path.join(os.environ["BASE_RUN"], "checkpoint_best_iou.pkl"), map_location="cpu", weights_only=False)
print(int(ck["epoch"]), float(ck["iou"]), float(ck["pd"]), float(ck["fa"]))
PY
)

read -r DEA_EPOCH DEA_IOU DEA_PD DEA_FA < <("${PYTHON}" - <<'PY'
import os
import torch

ck = torch.load(os.path.join(os.environ["DEA_RUN"], "checkpoint_best_iou.pkl"), map_location="cpu", weights_only=False)
print(int(ck["epoch"]), float(ck["iou"]), float(ck["pd"]), float(ck["fa"]))
PY
)

BASE_WEIGHT="${BASE_RUN}/weight_nuaa_mshnet_baseline_best_iou_e${BASE_EPOCH}.pkl"
BASE_CKPT="${BASE_RUN}/checkpoint_nuaa_mshnet_baseline_best_iou_e${BASE_EPOCH}.pkl"
DEA_WEIGHT="${DEA_RUN}/weight_nuaa_lambda_single_0p005_best_iou_e${DEA_EPOCH}.pkl"
DEA_CKPT="${DEA_RUN}/checkpoint_nuaa_lambda_single_0p005_best_iou_e${DEA_EPOCH}.pkl"

cp -n "${BASE_RUN}/weight.pkl" "${BASE_WEIGHT}"
cp -n "${BASE_RUN}/checkpoint_best_iou.pkl" "${BASE_CKPT}"
cp -n "${DEA_RUN}/weight.pkl" "${DEA_WEIGHT}"
cp -n "${DEA_RUN}/checkpoint_best_iou.pkl" "${DEA_CKPT}"

PDFA_STATUS="absent"
if [[ -s "${DEA_RUN}/checkpoint_pd_fa_best.pkl" || -s "${DEA_RUN}/weight_pd_fa_best.pkl" ]]; then
  PDFA_STATUS="present"
fi

sha256sum \
  "${BASE_WEIGHT}" "${BASE_CKPT}" \
  "${DEA_WEIGHT}" "${DEA_CKPT}" \
  > "${OUT_DIR}/nuaa_dea_lite_0p005_archived_artifacts.sha256"

cat > "${OUT_DIR}/nuaa_dea_lite_0p005_archive_manifest.json" <<JSON
{
  "dataset": "NUAA-SIRST",
  "candidate": "DEA-lite-0.005",
  "baseline": "MSHNet-baseline",
  "baseline_run_dir": "${BASE_RUN}",
  "candidate_run_dir": "${DEA_RUN}",
  "dataset_dir": "${DATASET_DIR}",
  "baseline_best_iou": {"epoch": ${BASE_EPOCH}, "IoU": ${BASE_IOU}, "PD": ${BASE_PD}, "FA": ${BASE_FA}},
  "candidate_best_iou": {"epoch": ${DEA_EPOCH}, "IoU": ${DEA_IOU}, "PD": ${DEA_PD}, "FA": ${DEA_FA}},
  "candidate_pdfa_best_artifact_status": "${PDFA_STATUS}",
  "decision": "ARCHIVED_PENDING_RETEST",
  "interpretation": "NUAA DEA-lite 0.005 is reported negative; retest will confirm evidence status."
}
JSON

BASE_LOG="${OUT_DIR}/nuaa_mshnet_baseline_best_iou_e${BASE_EPOCH}_retest.log"
DEA_LOG="${OUT_DIR}/nuaa_dea_lite_0p005_best_iou_e${DEA_EPOCH}_retest.log"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" -u main.py \
  --dataset-dir "${DATASET_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --pin-memory "${PIN_MEMORY}" \
  --mode test \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --weight-path "${BASE_WEIGHT}" \
  2>&1 | tee "${BASE_LOG}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" -u main.py \
  --dataset-dir "${DATASET_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --pin-memory "${PIN_MEMORY}" \
  --mode test \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --weight-path "${DEA_WEIGHT}" \
  2>&1 | tee "${DEA_LOG}"

"${PYTHON}" tools/official/parse_dea_test_log.py \
  --log "${BASE_LOG}" \
  --dataset NUAA-SIRST \
  --method MSHNet-baseline \
  --checkpoint_role baseline \
  --checkpoint_epoch "${BASE_EPOCH}" \
  --weight_path "${BASE_WEIGHT}" \
  --extra run_dir="${BASE_RUN}" \
  --output "${OUT_DIR}/nuaa_mshnet_baseline_best_iou_e${BASE_EPOCH}_retest_summary.json"

"${PYTHON}" tools/official/parse_dea_test_log.py \
  --log "${DEA_LOG}" \
  --dataset NUAA-SIRST \
  --method DEA-lite-0.005 \
  --checkpoint_role best_iou \
  --checkpoint_epoch "${DEA_EPOCH}" \
  --weight_path "${DEA_WEIGHT}" \
  --extra run_dir="${DEA_RUN}" \
  --extra pdfa_best_artifact_status="${PDFA_STATUS}" \
  --output "${OUT_DIR}/nuaa_dea_lite_0p005_best_iou_e${DEA_EPOCH}_retest_summary.json"

"${PYTHON}" tools/official/compare_dea_lite_against_baseline.py \
  --baseline_json "${OUT_DIR}/nuaa_mshnet_baseline_best_iou_e${BASE_EPOCH}_retest_summary.json" \
  --candidate_json "${OUT_DIR}/nuaa_dea_lite_0p005_best_iou_e${DEA_EPOCH}_retest_summary.json" \
  --output "${OUT_DIR}/nuaa_dea_lite_0p005_vs_mshnet_delta.json" \
  --min_delta_iou 0.0 \
  --min_delta_pd 0.0 \
  --max_delta_fa 0.0 \
  --allow_gate_fail

"${PYTHON}" tools/official/analyze_dea_epoch_metrics.py \
  --epoch_metric_log "${DEA_RUN}/epoch_metric.log" \
  --baseline_iou "${BASE_IOU}" \
  --baseline_pd "${BASE_PD}" \
  --baseline_fa "${BASE_FA}" \
  --output "${OUT_DIR}/nuaa_dea_lite_0p005_epoch_metric_audit.json" \
  --min_delta_iou 0.0 \
  --min_delta_pd 0.0 \
  --max_delta_fa 0.0

cat > "${OUT_DIR}/NUAA_NEGATIVE_README.md" <<MD
# NUAA-SIRST DEA-lite 0.005 negative evidence

Baseline best-IoU:

\`\`\`text
IoU=${BASE_IOU}
PD=${BASE_PD}
FA=${BASE_FA}
epoch=${BASE_EPOCH}
\`\`\`

DEA-lite 0.005 best-IoU:

\`\`\`text
IoU=${DEA_IOU}
PD=${DEA_PD}
FA=${DEA_FA}
epoch=${DEA_EPOCH}
\`\`\`

PD/FA-best artifact status:

\`\`\`text
${PDFA_STATUS}
\`\`\`

Decision:

\`\`\`text
NUAA-SIRST fails the DEA-lite 0.005 paired gate.
Treat as dataset-dependent negative evidence.
Do not claim universal DEA-lite improvement.
Do not run 0.01 as an immediate post-hoc rescue.
\`\`\`
MD

echo "DONE: NUAA negative archive/retest outputs written to ${OUT_DIR}"
