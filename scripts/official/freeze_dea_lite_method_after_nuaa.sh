#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-python3}

cd "${ROOT}"

EXPECTED_BRANCH=${EXPECTED_BRANCH:-dea-lite-0p005-nuaa-negative-archive}
CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "ERROR: wrong branch for DEA-lite freeze." >&2
  echo "  expected: ${EXPECTED_BRANCH}" >&2
  echo "  current : ${CURRENT_BRANCH}" >&2
  exit 6
fi

METHOD_DIR="docs/internal/dea_lite"
EVIDENCE_DIR="docs/internal/dea_lite_0p005"
DECISION_JSON="${METHOD_DIR}/DEA_LITE_FREEZE_DECISION.json"
CLAIM_GUARD_JSON="${METHOD_DIR}/DEA_LITE_FULL_DEA_CLAIM_GUARD.json"

mkdir -p "${METHOD_DIR}"

if [[ ! -f "${METHOD_DIR}/DEA_LITE_METHOD_FREEZE_AFTER_NUAA.md" ]]; then
  echo "ERROR: missing method freeze document: ${METHOD_DIR}/DEA_LITE_METHOD_FREEZE_AFTER_NUAA.md" >&2
  exit 2
fi

if [[ ! -d "${EVIDENCE_DIR}" ]]; then
  echo "ERROR: missing evidence directory: ${EVIDENCE_DIR}" >&2
  exit 3
fi

REQUIRED_EVIDENCE_FILES=(
  "docs/internal/dea_lite_0p005/evidence_status_after_nuaa.md"
  "docs/internal/dea_lite_0p005/evidence_status_after_nuaa.json"
  "docs/internal/dea_lite_0p005/no_universal_positive_claims_check.json"
)

for f in "${REQUIRED_EVIDENCE_FILES[@]}"; do
  if [[ ! -s "${f}" ]]; then
    echo "ERROR: missing required DEA-lite 0.005 evidence file: ${f}" >&2
    exit 7
  fi
done

if [[ ! -f "tools/official/check_no_full_dea_claim_from_dea_lite.py" ]]; then
  echo "ERROR: missing claim guard tool." >&2
  exit 4
fi

# Protect method code from accidental freeze-commit contamination.
# This catches staged, unstaged, and untracked implementation changes.
PROTECTED_CHANGED="$(git status --short -- main.py model utils 2>/dev/null || true)"
if [[ -n "${PROTECTED_CHANGED}" ]]; then
  echo "ERROR: protected implementation files have uncommitted changes:" >&2
  echo "${PROTECTED_CHANGED}" >&2
  echo "Freeze commit must not modify model/loss/main/utils implementation." >&2
  exit 5
fi

"${PYTHON}" tools/official/check_no_full_dea_claim_from_dea_lite.py \
  --root "${ROOT}" \
  --output "${CLAIM_GUARD_JSON}"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
GIT_STATUS_SHORT="$(git status --short 2>/dev/null || true)"
GIT_DIRTY_COUNT="$(printf "%s\n" "${GIT_STATUS_SHORT}" | sed '/^$/d' | wc -l | tr -d ' ')"
GIT_STATUS_SHORT_JSON="$(printf "%s" "${GIT_STATUS_SHORT}" | "${PYTHON}" -c 'import json, sys; print(json.dumps(sys.stdin.read()))')"

cat > "${DECISION_JSON}" <<JSON
{
  "decision": "DEA_LITE_FROZEN_AS_PILOT_ABLATION_LIMITATION_EVIDENCE",
  "root": "${ROOT}",
  "branch": "${GIT_BRANCH}",
  "commit": "${GIT_COMMIT}",
  "git_dirty_entry_count_at_decision_write": ${GIT_DIRTY_COUNT},
  "git_status_short_at_decision_write": ${GIT_STATUS_SHORT_JSON},
  "method_status": "DEA-lite is not full DEA",
  "evidence_status": {
    "NUDT-SIRST": "positive_anchor",
    "IRSTD-1K": "fa_control_positive_signal",
    "NUAA-SIRST": "stable_negative_dataset_dependent_failure"
  },
  "nuaa_negative": {
    "gate_pass": false,
    "decision": "DEA_LITE_NEGATIVE_DATASET_DEPENDENT",
    "delta_iou": -0.0336,
    "delta_pd": -0.0266,
    "delta_fa": 2.2104,
    "num_gate_pass_epochs": 0
  },
  "allowed_next_step": "open full-dea-predeclare-design branch and add protocol document only",
  "forbidden_next_steps": [
    "claim DEA-lite validates full DEA",
    "claim universal DEA-lite improvement",
    "run lambda 0.01 as DEA-main rescue",
    "mix Full DEA implementation code into the DEA-lite freeze branch"
  ],
  "claim_guard_json": "${CLAIM_GUARD_JSON}"
}
JSON

echo "DONE: wrote ${DECISION_JSON}"
