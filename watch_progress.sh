#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/home/ly/MSHNet}"
RUN_DIR="${2:-}"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -td "$PROJECT_DIR"/repro_runs/dea_lite_irstd1k_host_4gpu_* 2>/dev/null | head -1 || true)"
fi

if [[ -z "$RUN_DIR" || ! -f "$RUN_DIR/train.log" ]]; then
  echo "No train.log found. Pass RUN_DIR as the second argument."
  exit 1
fi

PID_FILE="$RUN_DIR/train.pid"
PID=""
if [[ -f "$PID_FILE" ]]; then
  PID="$(tr -dc '0-9' < "$PID_FILE")"
fi

if [[ -z "$PID" ]]; then
  PID="$(pgrep -f "$PROJECT_DIR.*main.py.*--mode train" | head -1 || true)"
fi

while true; do
  clear 2>/dev/null || true
  echo "Run dir: $RUN_DIR"
  echo

  if [[ -n "$PID" ]] && ps -p "$PID" >/dev/null 2>&1; then
    ps -p "$PID" -o pid,stat,etime,cmd
  else
    echo "Training process is not running."
  fi

  echo
  echo "GPU:"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true

  echo
  echo "Checkpoint:"
  stat -c '%y %s %n' "$PROJECT_DIR"/weight/MSHNet-*/checkpoint.pkl 2>/dev/null | sort | tail -1 || true

  echo
  echo "Latest epoch metrics:"
  tail -n 6 "$PROJECT_DIR"/weight/MSHNet-*/epoch_metric.log 2>/dev/null || true

  echo
  echo "Latest best metrics:"
  python3 - "$PROJECT_DIR" <<'PY'
from pathlib import Path
import re
import sys

project = Path(sys.argv[1])
metric_logs = sorted((project / "weight").glob("MSHNet-*/metric.log"), key=lambda p: p.stat().st_mtime)
if not metric_logs:
    raise SystemExit

rows = []
for line in metric_logs[-1].read_text(errors="replace").splitlines():
    match = re.search(
        r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\s+-\s+(\d+).*?IoU\s+([0-9.]+).*?PD\s+([0-9.]+).*?FA\s+([0-9.]+)",
        line,
    )
    if match:
        rows.append(match.groups())

if rows:
    time_s, epoch, iou, pd, fa = rows[-1]
    print(f"epoch {int(epoch)} | IoU {float(iou):.4f} | PD {float(pd):.4f} | FA {float(fa):.4f} | {time_s}")
PY

  echo
  echo "Best metric log:"
  tail -n 8 "$PROJECT_DIR"/weight/MSHNet-*/metric.log 2>/dev/null || true

  echo
  echo "Latest progress:"
  python3 - "$RUN_DIR/train.log" <<'PY'
from pathlib import Path
import sys

log = Path(sys.argv[1])
text = log.read_text(errors="replace").replace("\r", "\n")
lines = [line.strip() for line in text.splitlines() if line.strip()]
progress = [
    line for line in lines
    if "Epoch " in line and ("loss" in line or "IoU" in line)
]
for line in progress[-16:]:
    print(line)

errors = [
    line for line in lines
    if any(token in line.lower() for token in ("traceback", "error", "exception", "cuda out of memory", "nan"))
]
if errors:
    print()
    print("Recent warnings/errors:")
    for line in errors[-6:]:
        print(line)
PY

  sleep 2
done
