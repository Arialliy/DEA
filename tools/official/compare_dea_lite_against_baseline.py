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
    p.add_argument(
        "--allow_gate_fail",
        action="store_true",
        help="Write negative evidence JSON and return 0 even if the gate fails.",
    )
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

    decision = "DEA_LITE_POSITIVE" if gate_pass else "DEA_LITE_NEGATIVE_DATASET_DEPENDENT"
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
        "decision": decision,
        "interpretation": (
            "candidate improves the paired baseline under the declared gate"
            if gate_pass
            else "candidate fails the paired gate; treat as dataset-dependent negative evidence unless audit invalidates the run"
        ),
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if (not gate_pass) and (not args.allow_gate_fail):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
