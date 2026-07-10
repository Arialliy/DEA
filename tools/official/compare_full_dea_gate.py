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
    p.add_argument("--candidate_json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline_json", default="")
    p.add_argument("--reference_json", default="")
    p.add_argument("--iou_min", type=float, required=True)
    p.add_argument("--pd_min", type=float, required=True)
    p.add_argument("--fa_max", type=float, required=True)
    p.add_argument("--require_better_reference", action="store_true")
    p.add_argument("--allow_gate_fail", action="store_true")
    args = p.parse_args()

    candidate = read_json(args.candidate_json)
    baseline = read_json(args.baseline_json) if args.baseline_json else None
    reference = read_json(args.reference_json) if args.reference_json else None

    c_iou = metric(candidate, "IoU")
    c_pd = metric(candidate, "PD")
    c_fa = metric(candidate, "FA")
    method = candidate.get("method", "FullDEA")

    main_gate = {
        "IoU": c_iou >= args.iou_min,
        "PD": c_pd >= args.pd_min,
        "FA": c_fa <= args.fa_max,
    }

    reference_gate: dict[str, bool] | None = None
    if reference is not None and args.require_better_reference:
        r_iou = metric(reference, "IoU")
        r_pd = metric(reference, "PD")
        r_fa = metric(reference, "FA")
        reference_gate = {
            "IoU": c_iou > r_iou,
            "PD": c_pd > r_pd,
            "FA": c_fa < r_fa,
        }

    gate_pass = all(main_gate.values())
    if reference_gate is not None:
        gate_pass = gate_pass and all(reference_gate.values())

    result: dict[str, Any] = {
        "candidate": candidate,
        "baseline": baseline,
        "reference": reference,
        "thresholds": {
            "iou_min": args.iou_min,
            "pd_min": args.pd_min,
            "fa_max": args.fa_max,
            "require_better_reference": bool(args.require_better_reference),
        },
        "main_gate": main_gate,
        "reference_gate": reference_gate,
        "gate_pass": bool(gate_pass),
        "decision": (
            "FULL_DEA_NUAA_FIRST_GATE_PASS"
            if gate_pass
            else "FULL_DEA_NUAA_FIRST_GATE_FAIL"
        ),
        "interpretation": (
            f"{method} passes the declared NUAA first gate for this checkpoint."
            if gate_pass
            else f"{method} fails at least one declared NUAA first-gate condition for this checkpoint."
        ),
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if (not gate_pass) and (not args.allow_gate_fail):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
