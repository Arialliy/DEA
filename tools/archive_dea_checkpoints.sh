#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash tools/archive_dea_checkpoints.sh RUN_DIR [TAG]" >&2
  echo "Example: bash tools/archive_dea_checkpoints.sh /home/ly/DEA/weight/MSHNet-xxxx lambda_single_0p005" >&2
  exit 1
fi

RUN_DIR="$1"
TAG="${2:-lambda_single_0p005}"
PYTHON_BIN="${PYTHON_BIN:-/home/ly/BasicIRSTD/infrarenet/bin/python}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

get_epoch() {
  local ckpt_path="$1"
  "$PYTHON_BIN" - "$ckpt_path" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(path, map_location="cpu")

epoch = ckpt.get("epoch")
if epoch is None:
    raise SystemExit(f"No epoch field in {path}")
print(int(epoch))
PY
}

print_metric() {
  local ckpt_path="$1"
  "$PYTHON_BIN" - "$ckpt_path" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(path, map_location="cpu")

print(f"epoch={ckpt.get('epoch')} iou={ckpt.get('iou')} pd={ckpt.get('pd')} fa={ckpt.get('fa')}")
PY
}

BEST_CKPT="$RUN_DIR/checkpoint_best_iou.pkl"
BEST_WEIGHT="$RUN_DIR/weight.pkl"
PDFA_CKPT="$RUN_DIR/checkpoint_pd_fa_best.pkl"
PDFA_WEIGHT="$RUN_DIR/weight_pd_fa_best.pkl"

if [[ ! -f "$BEST_CKPT" ]]; then
  echo "Missing $BEST_CKPT" >&2
  exit 1
fi

if [[ ! -f "$BEST_WEIGHT" ]]; then
  echo "Missing $BEST_WEIGHT" >&2
  exit 1
fi

BEST_EPOCH=$(get_epoch "$BEST_CKPT")
BEST_WEIGHT_OUT="$RUN_DIR/weight_${TAG}_best_iou_e${BEST_EPOCH}.pkl"
BEST_CKPT_OUT="$RUN_DIR/checkpoint_${TAG}_best_iou_e${BEST_EPOCH}.pkl"

cp "$BEST_WEIGHT" "$BEST_WEIGHT_OUT"
cp "$BEST_CKPT" "$BEST_CKPT_OUT"

echo "Archived best-IoU checkpoint:"
echo "  $BEST_WEIGHT_OUT"
echo "  $BEST_CKPT_OUT"
print_metric "$BEST_CKPT"

if [[ -f "$PDFA_CKPT" && -f "$PDFA_WEIGHT" ]]; then
  PDFA_EPOCH=$(get_epoch "$PDFA_CKPT")
  PDFA_WEIGHT_OUT="$RUN_DIR/weight_${TAG}_pdfa_best_e${PDFA_EPOCH}.pkl"
  PDFA_CKPT_OUT="$RUN_DIR/checkpoint_${TAG}_pdfa_best_e${PDFA_EPOCH}.pkl"

  cp "$PDFA_WEIGHT" "$PDFA_WEIGHT_OUT"
  cp "$PDFA_CKPT" "$PDFA_CKPT_OUT"

  echo "Archived PD/FA-best checkpoint:"
  echo "  $PDFA_WEIGHT_OUT"
  echo "  $PDFA_CKPT_OUT"
  print_metric "$PDFA_CKPT"
else
  echo "PD/FA-best checkpoint not found. Skipping PD/FA archive."
fi

echo "Archive done."
