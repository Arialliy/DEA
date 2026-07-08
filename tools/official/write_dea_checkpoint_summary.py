#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_torch(path: Path) -> Any:
    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"torch is required to read checkpoint: {exc}") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--weight", default="")
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--checkpoint_role", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source_run_dir", default="")
    p.add_argument("--lambda_single", type=float, default=None)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt_path}")

    data = load_torch(ckpt_path)
    if not isinstance(data, dict):
        raise SystemExit(f"checkpoint is not a dict: {ckpt_path}")

    def f(key: str) -> float | None:
        value = data.get(key)
        return None if value is None else float(value)

    epoch = data.get("epoch")
    if epoch is None:
        raise SystemExit(f"checkpoint missing epoch: {ckpt_path}")

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "method": args.method,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_epoch": int(epoch),
        "IoU": f("iou"),
        "PD": f("pd"),
        "FA": f("fa"),
        "best_iou": f("best_iou"),
        "best_pd_fa": f("best_pd_fa"),
        "best_pd_fa_iou": f("best_pd_fa_iou"),
        "best_pd_fa_pd": f("best_pd_fa_pd"),
        "best_pd_fa_epoch": data.get("best_pd_fa_epoch"),
        "source_run_dir": args.source_run_dir,
        "lambda_single": args.lambda_single,
    }

    if args.weight:
        weight_path = Path(args.weight).expanduser().resolve()
        if not weight_path.is_file():
            raise SystemExit(f"missing weight: {weight_path}")
        result["weight_path"] = str(weight_path)
        result["weight_sha256"] = sha256_file(weight_path)

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
