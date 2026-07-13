#!/usr/bin/env python3
"""Run TRACE T0-A against train-only masks and emit a reproducible report."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.trace_codec import component_records, root_cell_collisions
from utils.trace_geometry import TraceGeometrySpec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def provenance_path(path: Path) -> str:
    """Avoid embedding a user-specific absolute workspace path in artifacts."""

    resolved = path.resolve()
    try:
        return "repo:" + resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "external:" + resolved.name


def read_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"empty split file: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate sample names in split file: {path}")
    return names


def resolve_train_split(dataset_dir: Path, explicit: str) -> Path:
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = dataset_dir / candidate
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate.resolve()
    candidates = sorted((dataset_dir / "img_idx").glob("train_*.txt"))
    if (dataset_dir / "trainval.txt").is_file():
        candidates.insert(0, dataset_dir / "trainval.txt")
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one canonical train split under {dataset_dir}, found {candidates}"
        )
    return candidates[0].resolve()


def known_test_names(dataset_dir: Path) -> set[str]:
    candidates = list((dataset_dir / "img_idx").glob("test_*.txt"))
    if (dataset_dir / "test.txt").is_file():
        candidates.append(dataset_dir / "test.txt")
    names: set[str] = set()
    for path in candidates:
        current = set(read_names(path))
        if names.intersection(current):
            raise RuntimeError("canonical test manifests overlap one another")
        names.update(current)
    return names


def resized_mask(path: Path, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.resize((width, height), Image.Resampling.NEAREST)
        array = np.asarray(resized)
    while array.ndim > 2:
        array = array[..., 0]
    return np.ascontiguousarray(array > 0)


def collision_summary(
    roots_by_image: dict[str, tuple[tuple[int, int], ...]],
    cell_size: int,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    colliding_cells = 0
    excess_components = 0
    for name, roots in roots_by_image.items():
        collisions = root_cell_collisions(roots, cell_size)
        if not collisions:
            continue
        colliding_cells += len(collisions)
        excess_components += sum(len(items) - 1 for items in collisions.values())
        failures.append(
            {
                "sample": name,
                "cells": [
                    {"cell": list(cell), "roots": [list(root) for root in roots]}
                    for cell, roots in sorted(collisions.items())
                ],
            }
        )
    return {
        "cell_size": cell_size,
        "images_with_collision": len(failures),
        "colliding_cells": colliding_cells,
        "excess_components": excess_components,
        "failures": failures,
    }


def _source_inventory() -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "utils" / "trace_codec.py",
        PROJECT_ROOT / "utils" / "trace_geometry.py",
        Path(__file__).resolve(),
    ]
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def audit(args: argparse.Namespace) -> dict[str, object]:
    dataset_dir = Path(args.dataset_dir).resolve()
    split_path = resolve_train_split(dataset_dir, args.train_split_file)
    names = read_names(split_path)
    overlap = sorted(set(names).intersection(known_test_names(dataset_dir)))
    if overlap:
        raise RuntimeError(
            f"train split overlaps canonical test names ({len(overlap)}), e.g. {overlap[:5]}"
        )

    masks_dir = dataset_dir / "masks"
    roots_by_image: dict[str, tuple[tuple[int, int], ...]] = {}
    failures: list[dict[str, object]] = []
    error_counts: Counter[str] = Counter()
    max_runs_histogram: Counter[int] = Counter()
    raw_sizes: Counter[str] = Counter()
    mask_hashes: dict[str, str] = {}
    total_components = 0
    exact_components = 0
    encode_decode_exact = 0
    empty_images = 0
    max_down = max_left = max_right = 0
    area_values: list[int] = []

    for name in names:
        mask_path = masks_dir / f"{name}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        mask_hashes[name] = sha256_file(mask_path)
        with Image.open(mask_path) as raw_image:
            raw_sizes[f"{raw_image.height}x{raw_image.width}"] += 1
        mask = resized_mask(mask_path, args.image_height, args.image_width)
        records = component_records(mask)
        if not records:
            empty_images += 1
        roots_by_image[name] = tuple(record.root for record in records)
        for record in records:
            total_components += 1
            area_values.append(record.area)
            max_runs_histogram[record.max_runs_per_row] += 1
            top, left, bottom, right = record.bbox
            root_y, root_x = record.root
            max_down = max(max_down, bottom - 1 - root_y)
            max_left = max(max_left, root_x - left)
            max_right = max(max_right, right - 1 - root_x)
            if record.exact:
                exact_components += 1
                encode_decode_exact += 1
            else:
                error_counts[record.error_code or "unknown"] += 1
                failures.append(
                    {
                        "sample": name,
                        "component_label": record.label,
                        "area": record.area,
                        "bbox": list(record.bbox),
                        "canonical_root": list(record.root),
                        "max_runs_per_row": record.max_runs_per_row,
                        "error_code": record.error_code,
                        "error_message": record.error_message,
                    }
                )

    collision_reports = {
        str(cell_size): collision_summary(roots_by_image, cell_size)
        for cell_size in args.cell_sizes
    }
    chosen_cell_size: int | None = None
    for cell_size in args.cell_sizes:
        if collision_reports[str(cell_size)]["excess_components"] == 0:
            chosen_cell_size = cell_size
            break

    geometry: TraceGeometrySpec | None = None
    window_exact = 0
    window_failures: list[dict[str, object]] = []
    if chosen_cell_size is not None:
        geometry = TraceGeometrySpec(
            image_height=args.image_height,
            image_width=args.image_width,
            cell_size=chosen_cell_size,
            max_down=max_down,
            max_left=max_left,
            max_right=max_right,
            margin=args.margin,
        )
        for name in names:
            mask_path = masks_dir / f"{name}.png"
            for record in component_records(
                resized_mask(mask_path, args.image_height, args.image_width)
            ):
                if record.chain is None:
                    continue
                try:
                    geometry.chain_to_local_mask(record.chain)
                    window_exact += 1
                except Exception as exc:
                    window_failures.append(
                        {
                            "sample": name,
                            "component_label": record.label,
                            "error": str(exc),
                        }
                    )

    exact_fraction = exact_components / total_components if total_components else 1.0
    collision_pass = chosen_cell_size is not None
    row_chain_pass = exact_components == total_components
    window_pass = window_exact == exact_components and not window_failures
    t0_a_pass = row_chain_pass and collision_pass and window_pass
    quantiles = {}
    if area_values:
        quantiles = {
            str(q): float(np.quantile(np.asarray(area_values), q))
            for q in (0.0, 0.5, 0.95, 1.0)
        }

    report: dict[str, object] = {
        "schema_version": "trace_t0_a_geometry_report_v1",
        "gate": "T0-A",
        "status": "PASS" if t0_a_pass else "NO-GO",
        "dataset": dataset_dir.name,
        "dataset_dir": provenance_path(dataset_dir),
        "train_only": True,
        "train_split": {
            "path": provenance_path(split_path),
            "raw_sha256": sha256_file(split_path),
            "ordered_names_sha256": hashlib.sha256(
                ("\n".join(names) + "\n").encode("utf-8")
            ).hexdigest(),
            "number_of_images": len(names),
            "canonical_test_overlap": 0,
        },
        "resize": {
            "height": args.image_height,
            "width": args.image_width,
            "interpolation": "PIL.Image.Resampling.NEAREST",
            "raw_size_histogram": dict(sorted(raw_sizes.items())),
        },
        "component_geometry": {
            "number_of_gt_components": total_components,
            "empty_images": empty_images,
            "row_run_chain_exact_count": exact_components,
            "row_run_chain_exact_fraction": exact_fraction,
            "encode_decode_bit_exact_count": encode_decode_exact,
            "error_counts": dict(sorted(error_counts.items())),
            "max_runs_per_row_histogram": {
                str(key): value for key, value in sorted(max_runs_histogram.items())
            },
            "area_quantiles": quantiles,
            "max_relative_extents": {
                "up": 0,
                "down": max_down,
                "left": max_left,
                "right": max_right,
            },
            "failures": failures,
        },
        "root_cell_collision": collision_reports,
        "selected_cell_size": chosen_cell_size,
        "window_coverage": {
            "exact_components_checked": exact_components,
            "covered": window_exact,
            "failures": window_failures,
        },
        "candidate_geometry_spec": geometry.to_dict() if geometry is not None else None,
        "candidate_geometry_sha256": geometry.sha256 if geometry is not None else None,
        "criteria": {
            "all_components_exact_row_run_chains": row_chain_pass,
            "zero_root_cell_collisions": collision_pass,
            "all_exact_components_inside_frozen_window": window_pass,
            "train_only_no_test_overlap": True,
        },
        "source_sha256": _source_inventory(),
        "mask_manifest_sha256": canonical_json_sha256(mask_hashes),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--train-split-file", default="")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--cell-sizes", type=int, nargs="+", default=[4, 2])
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="emit a NO-GO report with exit status 0 instead of the fail-closed status 2",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.cell_sizes or any(value < 1 for value in args.cell_sizes):
        raise ValueError("--cell-sizes must contain positive integers")
    report = audit(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["status"] != "PASS" and not args.report_only:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
