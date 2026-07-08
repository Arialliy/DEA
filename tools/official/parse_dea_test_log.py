#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_metric(text: str, names: list[str]) -> float | None:
    for name in names:
        patterns = [
            rf"\b{name}\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            rf"\b{name}\b\s+([0-9]+(?:\.[0-9]+)?)",
        ]
        for pat in patterns:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            if matches:
                return float(matches[-1])
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument(
        "--checkpoint_role",
        required=True,
        choices=["best_iou", "pdfa_best", "final", "baseline"],
    )
    p.add_argument("--checkpoint_epoch", required=True, type=int)
    p.add_argument("--weight_path", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    weight_path = Path(args.weight_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.is_file():
        raise SystemExit(f"missing log: {log_path}")
    if not weight_path.is_file():
        raise SystemExit(f"missing weight: {weight_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")

    iou = parse_metric(text, ["IoU", "iou", "mIoU", "miou"])
    pd = parse_metric(text, ["PD", "Pd", "pd"])
    fa = parse_metric(text, ["FA", "Fa", "fa"])
    metrics_found = all(v is not None for v in (iou, pd, fa))

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "method": args.method,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint_epoch": args.checkpoint_epoch,
        "weight_path": str(weight_path),
        "weight_sha256": sha256_file(weight_path),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "metrics_found": metrics_found,
        "IoU": iou,
        "PD": pd,
        "FA": fa,
    }

    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if not metrics_found:
        raise SystemExit(
            "Could not parse IoU/PD/FA from log. Inspect log manually and update parser patterns. "
            f"Output written to {output_path}"
        )


if __name__ == "__main__":
    main()
