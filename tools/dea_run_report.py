#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch


SLOTS = [
    ("best_iou", "checkpoint_best_iou.pkl"),
    ("pd_fa_best", "checkpoint_pd_fa_best.pkl"),
    ("latest", "checkpoint.pkl"),
]


def load_torch_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_run_spec(spec):
    if ":" not in spec:
        path = spec
        label = Path(spec).name
    else:
        path, label = spec.split(":", 1)
    return Path(path), label


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def collect_rows(run_specs, baseline):
    rows = []
    for spec in run_specs:
        run_dir, label = parse_run_spec(spec)
        for slot, filename in SLOTS:
            ckpt_path = run_dir / filename
            if not ckpt_path.exists():
                continue
            ckpt = load_torch_file(str(ckpt_path))
            iou = ckpt.get("iou")
            pd = ckpt.get("pd")
            fa = ckpt.get("fa")
            row = {
                "label": label,
                "slot": slot,
                "run_dir": str(run_dir),
                "checkpoint": filename,
                "epoch": ckpt.get("epoch"),
                "iou": iou,
                "pd": pd,
                "fa": fa,
                "delta_iou": None if iou is None else float(iou) - baseline["iou"],
                "delta_pd": None if pd is None else float(pd) - baseline["pd"],
                "delta_fa": None if fa is None else float(fa) - baseline["fa"],
            }
            rows.append(row)
    return rows


def write_csv(rows, out_csv):
    fieldnames = [
        "label",
        "slot",
        "epoch",
        "iou",
        "pd",
        "fa",
        "delta_iou",
        "delta_pd",
        "delta_fa",
        "checkpoint",
        "run_dir",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows, out_md, baseline):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# DEA-lite Run Report")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append("| IoU | PD | FA |")
    lines.append("|---:|---:|---:|")
    lines.append(f"| {baseline['iou']:.4f} | {baseline['pd']:.4f} | {baseline['fa']:.4f} |")
    lines.append("")
    lines.append("## Checkpoints")
    lines.append("")
    lines.append("| Run | Slot | Epoch | IoU | PD | FA | Delta IoU | Delta PD | Delta FA |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {label} | {slot} | {epoch} | {iou} | {pd} | {fa} | {diou} | {dpd} | {dfa} |".format(
                label=row["label"],
                slot=row["slot"],
                epoch=fmt(row["epoch"], 0),
                iou=fmt(row["iou"]),
                pd=fmt(row["pd"]),
                fa=fmt(row["fa"]),
                diou=fmt(row["delta_iou"]),
                dpd=fmt(row["delta_pd"]),
                dfa=fmt(row["delta_fa"]),
            )
        )
    lines.append("")
    lines.append("## Suggested Interpretation")
    lines.append("")
    lines.append("- Use `best_iou` rows to compare IoU-preserving behavior.")
    lines.append("- Use `pd_fa_best` rows for PD/IoU-constrained false-alarm control.")
    lines.append("- A useful FA-control point should satisfy: IoU within 0.01 of baseline, PD no lower than baseline, and FA lower than baseline.")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-iou", type=float, required=True)
    parser.add_argument("--baseline-pd", type=float, required=True)
    parser.add_argument("--baseline-fa", type=float, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="Run specs: /path/to/run:label")
    parser.add_argument("--out-md", type=str, default="results/dea_run_report.md")
    parser.add_argument("--out-csv", type=str, default="results/dea_run_report.csv")
    args = parser.parse_args()

    baseline = {
        "iou": args.baseline_iou,
        "pd": args.baseline_pd,
        "fa": args.baseline_fa,
    }
    rows = collect_rows(args.runs, baseline)
    if not rows:
        raise SystemExit("No checkpoint rows collected. Check --runs paths.")

    write_csv(rows, Path(args.out_csv))
    write_markdown(rows, Path(args.out_md), baseline)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
