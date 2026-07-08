#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

LINE_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\s+-\s+"
    r"(?P<epoch>\d+)\s+-\s+IoU\s+(?P<iou>[0-9.]+)\s+-\s+"
    r"PD\s+(?P<pd>[0-9.]+)\s+-\s+FA\s+(?P<fa>[0-9.]+)"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch_metric_log", required=True)
    p.add_argument("--baseline_iou", type=float, required=True)
    p.add_argument("--baseline_pd", type=float, required=True)
    p.add_argument("--baseline_fa", type=float, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min_delta_iou", type=float, default=0.0)
    p.add_argument("--min_delta_pd", type=float, default=0.0)
    p.add_argument("--max_delta_fa", type=float, default=0.0)
    args = p.parse_args()

    path = Path(args.epoch_metric_log).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"missing epoch_metric.log: {path}")

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        normalized = line.replace("\t", " ")
        m = LINE_RE.search(normalized)
        if not m:
            continue
        rec = {
            "epoch": int(m.group("epoch")),
            "IoU": float(m.group("iou")),
            "PD": float(m.group("pd")),
            "FA": float(m.group("fa")),
        }
        rec["delta_IoU"] = rec["IoU"] - args.baseline_iou
        rec["delta_PD"] = rec["PD"] - args.baseline_pd
        rec["delta_FA"] = rec["FA"] - args.baseline_fa
        rec["gate_pass"] = bool(
            rec["delta_IoU"] >= args.min_delta_iou
            and rec["delta_PD"] >= args.min_delta_pd
            and rec["delta_FA"] <= args.max_delta_fa
        )
        records.append(rec)

    if not records:
        raise SystemExit(f"no metric records parsed from {path}")

    best_iou = max(records, key=lambda x: x["IoU"])
    lowest_fa = min(records, key=lambda x: x["FA"])
    gate_pass_epochs = [r for r in records if r["gate_pass"]]

    result = {
        "epoch_metric_log": str(path),
        "num_records": len(records),
        "baseline": {
            "IoU": args.baseline_iou,
            "PD": args.baseline_pd,
            "FA": args.baseline_fa,
        },
        "thresholds": {
            "min_delta_iou": args.min_delta_iou,
            "min_delta_pd": args.min_delta_pd,
            "max_delta_fa": args.max_delta_fa,
        },
        "best_iou_epoch": best_iou,
        "lowest_fa_epoch": lowest_fa,
        "num_gate_pass_epochs": len(gate_pass_epochs),
        "gate_pass_epochs": gate_pass_epochs[:20],
        "decision": "HAS_GATE_PASS_EPOCH" if gate_pass_epochs else "NO_GATE_PASS_EPOCH",
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
