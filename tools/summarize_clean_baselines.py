#!/usr/bin/env python3
"""Summarize the clean-baseline scheduler without reading official test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PROJECT_DIR = Path(__file__).resolve().parents[1]
METRIC_RE = re.compile(
    r"-\s+(?P<epoch>\d+)\s+\t?\s*- IoU (?P<iou>[0-9.]+)"
    r"\s+\t?\s*- PD (?P<pd>[0-9.]+)\s+\t?\s*- FA (?P<fa>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def parse_metrics(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = METRIC_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "iou": float(match.group("iou")),
                "pd": float(match.group("pd")),
                "fa": float(match.group("fa")),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    report_root = PROJECT_DIR / "repro_runs" / "clean" / args.batch_id
    manifest_path = report_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summaries = []
    for job in manifest["jobs"]:
        run_dir = Path(job["run_dir"])
        rows = parse_metrics(run_dir / "epoch_metric.log")
        result_path = Path(job["result_file"])
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else None
        )
        if result is not None:
            status = "completed" if result.get("returncode") == 0 else "failed"
        elif rows:
            status = "running_or_interrupted"
        else:
            status = "pending"
        latest = rows[-1] if rows else None
        best = max(rows, key=lambda row: row["iou"]) if rows else None
        summaries.append(
            {
                "job_id": job["job_id"],
                "dataset": job["dataset"],
                "seed": job["seed"],
                "status": status,
                "epochs_recorded": len(rows),
                "latest": latest,
                "best": best,
                "returncode": result.get("returncode") if result else None,
            }
        )

    payload = {
        "batch_id": args.batch_id,
        "stage": manifest["stage"],
        "official_test_policy": manifest["official_test_policy"],
        "jobs": summaries,
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0

    print("| Dataset | Seed | Status | Epoch | Latest IoU | Best IoU | Best PD | Best FA/M |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for row in summaries:
        latest = row["latest"] or {}
        best = row["best"] or {}
        print(
            "| {dataset} | {seed} | {status} | {epoch} | {latest_iou} | "
            "{best_iou} | {best_pd} | {best_fa} |".format(
                dataset=row["dataset"],
                seed=row["seed"],
                status=row["status"],
                epoch=latest.get("epoch", "-"),
                latest_iou=("%.4f" % latest["iou"]) if latest else "-",
                best_iou=("%.4f" % best["iou"]) if best else "-",
                best_pd=("%.4f" % best["pd"]) if best else "-",
                best_fa=("%.4f" % best["fa"]) if best else "-",
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
