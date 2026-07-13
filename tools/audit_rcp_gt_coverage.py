#!/usr/bin/env python3
"""Development-only coverage audit for Rooted Component Programs (RCP).

Only the fit and internal development manifests from clean baseline runs are
opened.  Official test manifests are deliberately never instantiated.  The
audit asks two representation questions before any neural RCP head exists:

1. does the canonical BFS codec round-trip every observed 8-connected target;
2. how many program nodes are required by the fit/dev target-area distribution.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image
from skimage import measure
from torchvision.transforms import ToTensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data import IRSTD_Dataset  # noqa: E402
from utils.rooted_component_program import (  # noqa: E402
    encode_rooted_component,
    render_rooted_component,
)
from utils.target_identity import build_stable_target_set  # noqa: E402


SCHEMA = "dea.rcp.gt_coverage.v1"
DEFAULT_CAPS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024)


class RCPGTCoverageError(RuntimeError):
    """Raised when the development-only representation audit cannot be trusted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RCPGTCoverageError(
                        f"{path}:{line_number} is not a JSON object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RCPGTCoverageError(f"cannot read {path}: {exc}") from exc
    return rows


def formal_hard_core_ids(path: Path) -> frozenset[tuple[str, str]]:
    """Return the formal Q2/FA20 three-seed no-activation target panel."""

    grouped: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in _jsonl(path):
        if (
            row.get("grid_level") == "Q2"
            and int(row.get("nominal_budget_fa_per_mpix", -1)) == 20
            and row.get("category_core") == "no_feasible_local_peak_activation"
        ):
            key = (str(row["dataset"]), str(row["stable_target_id"]))
            grouped[key].add(int(row["seed"]))
    expected_seeds = {20260711, 20260712, 20260713}
    result = frozenset(key for key, seeds in grouped.items() if seeds == expected_seeds)
    if len(result) != 16:
        raise RCPGTCoverageError(
            f"formal Q2/FA20 hard-core panel must contain 16 targets, got {len(result)}"
        )
    return result


def summarize_areas(
    rows: Iterable[Mapping[str, Any]],
    *,
    caps: tuple[int, ...] = DEFAULT_CAPS,
) -> dict[str, Any]:
    selected = list(rows)
    areas = np.asarray([int(row["target_area"]) for row in selected], dtype=np.int64)
    if not selected or bool(np.any(areas <= 0)):
        raise RCPGTCoverageError("coverage group must contain positive target areas")
    if not caps or tuple(sorted(set(caps))) != caps or caps[0] < 1:
        raise RCPGTCoverageError("caps must be unique sorted positive integers")
    quantile_levels = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    quantiles = {
        f"q{int(level * 100):02d}": float(np.quantile(areas, level))
        for level in quantile_levels
    }
    return {
        "target_count": len(selected),
        "image_count": len({(row["dataset"], row["split"], row["image_name"]) for row in selected}),
        "area_mean": float(areas.mean()),
        "area_quantiles": quantiles,
        "maximum_area": int(areas.max()),
        "node_cap_coverage": {
            str(cap): {
                "covered_targets": int(np.sum(areas <= cap)),
                "coverage": float(np.mean(areas <= cap)),
                "uncovered_targets": int(np.sum(areas > cap)),
            }
            for cap in caps
        },
        "exact_roundtrip_targets": sum(bool(row["exact_roundtrip"]) for row in selected),
    }


def _load_run(run_dir: Path) -> tuple[str, Namespace, dict[str, tuple[str, ...]], dict[str, Any]]:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise RCPGTCoverageError(f"missing run_config.json: {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stored = config.get("args")
    if not isinstance(stored, dict):
        raise RCPGTCoverageError(f"run config has no args mapping: {config_path}")
    if stored.get("mode") != "train" or stored.get("model_type") != "mshnet":
        raise RCPGTCoverageError("coverage sources must be clean MSHNet training runs")
    args = Namespace(**stored)
    fit = IRSTD_Dataset(args, mode="train")
    dev = IRSTD_Dataset(args, mode="val")
    fit_names = tuple(fit.names)
    dev_names = tuple(dev.names)
    if set(fit_names) & set(dev_names):
        raise RCPGTCoverageError("fit and development manifests overlap")
    dataset = Path(args.dataset_dir).resolve().name
    split_hashes = {
        "fit": fit.split_sha256,
        "dev": dev.split_sha256,
    }
    expected = {
        "fit": str(stored.get("train_split_sha256", "")),
        "dev": str(stored.get("val_split_sha256", "")),
    }
    if split_hashes != expected:
        raise RCPGTCoverageError(
            f"dataset split hashes disagree with run metadata for {dataset}"
        )
    return dataset, args, {"fit": fit_names, "dev": dev_names}, {
        "run_dir": str(run_dir),
        "run_config": str(config_path),
        "run_config_sha256": _sha256_file(config_path),
        "split_sha256": split_hashes,
        "fit_images": len(fit_names),
        "dev_images": len(dev_names),
    }


def _canonical_mask(path: Path, *, size: int) -> np.ndarray:
    if not path.is_file():
        raise RCPGTCoverageError(f"missing target mask: {path}")
    with Image.open(path) as image:
        resized = image.resize((size, size), resample=Image.Resampling.NEAREST)
        tensor = ToTensor()(resized)
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RCPGTCoverageError(f"target mask is not single-channel: {path}")
    return (tensor[0].numpy() > 0.5)


def _records_for_run(
    dataset: str,
    args: Namespace,
    split_names: Mapping[str, tuple[str, ...]],
    hard_core: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    dataset_dir = Path(args.dataset_dir).resolve()
    base_size = int(args.base_size)
    records: list[dict[str, Any]] = []
    for split, names in split_names.items():
        for image_name in names:
            mask = _canonical_mask(
                dataset_dir / "masks" / f"{image_name}.png",
                size=base_size,
            )
            target_set = build_stable_target_set(
                mask,
                dataset=dataset,
                image_name=image_name,
                connectivity=2,
            )
            labels = measure.label(mask, connectivity=2)
            regions = measure.regionprops(labels)
            if len(regions) != len(target_set.targets):
                raise RCPGTCoverageError("component count disagrees with stable identities")
            for target in target_set.targets:
                component = labels == target.source_label
                program = encode_rooted_component(component)
                rendered = render_rooted_component(program, component.shape)
                exact = bool(np.array_equal(rendered, component))
                if not exact or program.node_count != target.area:
                    raise RCPGTCoverageError("canonical RCP failed an exact round-trip")
                records.append(
                    {
                        "schema_version": SCHEMA,
                        "dataset": dataset,
                        "split": split,
                        "image_name": image_name,
                        "stable_target_id": target.stable_key,
                        "component_mask_sha256": target.component_mask_sha256,
                        "target_area": target.area,
                        "bbox": list(target.bbox),
                        "centroid_y": target.centroid_y,
                        "centroid_x": target.centroid_x,
                        "root_y": program.root_y,
                        "root_x": program.root_x,
                        "program_nodes": program.node_count,
                        "program_commands": len(program.parent_indices),
                        "exact_roundtrip": exact,
                        "formal_hard_core": (dataset, target.stable_key) in hard_core,
                    }
                )
    return records


def _git_state() -> dict[str, Any]:
    def command(*values: str) -> str:
        result = subprocess.run(
            values,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "head": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "branch", "--show-current"),
        "status_porcelain": command("git", "status", "--short").splitlines(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument(
        "--hard-core-source",
        default="repro_runs/gate_g/frontier_decomposition_v2/target_decomposition.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    hard = summary["formal_hard_core"]
    lines = [
        "# Rooted Component Program GT coverage gate",
        "",
        "development-only fit/dev representation audit; official test sealed; no model-performance claim",
        "",
        f"- Targets: {overall['target_count']}",
        f"- Exact canonical BFS round-trips: {overall['exact_roundtrip_targets']}/{overall['target_count']}",
        f"- Maximum observed area/program nodes: {overall['maximum_area']}",
        f"- Formal hard-core targets found: {hard['target_count']}/16",
        f"- Formal hard-core maximum area: {hard['maximum_area']}",
        "",
        "| Node cap | all fit/dev coverage | uncovered | hard-core coverage |",
        "|---:|---:|---:|---:|",
    ]
    for cap, values in overall["node_cap_coverage"].items():
        hard_values = hard["node_cap_coverage"][cap]
        lines.append(
            f"| {cap} | {values['coverage']:.6f} | {values['uncovered_targets']} | {hard_values['coverage']:.6f} |"
        )
    lines.extend(
        [
            "",
            "An exact codec round-trip proves representation completeness only for the observed masks. It does not prove that a neural decoder can learn the programs or improve the component Pd–FA frontier.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    run_dirs = tuple(Path(value).expanduser().resolve() for value in args.run_dir)
    hard_core_source = Path(args.hard_core_source).expanduser()
    if not hard_core_source.is_absolute():
        hard_core_source = (ROOT / hard_core_source).resolve()
    hard_core = formal_hard_core_ids(hard_core_source)

    runs = []
    records: list[dict[str, Any]] = []
    datasets = set()
    for run_dir in run_dirs:
        dataset, dataset_args, split_names, run_record = _load_run(run_dir)
        if dataset in datasets:
            raise RCPGTCoverageError(f"duplicate dataset source: {dataset}")
        datasets.add(dataset)
        runs.append({"dataset": dataset, **run_record})
        records.extend(
            _records_for_run(dataset, dataset_args, split_names, hard_core)
        )
    if datasets != {"IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"}:
        raise RCPGTCoverageError("coverage gate requires the three clean datasets")
    observed_hard_core = {
        (row["dataset"], row["stable_target_id"])
        for row in records
        if row["formal_hard_core"]
    }
    if observed_hard_core != set(hard_core):
        missing = sorted(set(hard_core) - observed_hard_core)
        raise RCPGTCoverageError(
            f"fit/dev manifests do not contain the formal hard-core panel: {missing}"
        )

    by_dataset = {
        dataset: summarize_areas(row for row in records if row["dataset"] == dataset)
        for dataset in sorted(datasets)
    }
    by_split = {
        split: summarize_areas(row for row in records if row["split"] == split)
        for split in ("fit", "dev")
    }
    hard_records = [row for row in records if row["formal_hard_core"]]
    summary = {
        "schema_version": SCHEMA,
        "scope": "development-only fit/dev GT representation audit; official test sealed",
        "codec": {
            "root": "nearest component pixel to centroid; lexicographic tie-break",
            "traversal": "BFS",
            "neighbour_order": "N,NE,E,SE,S,SW,W,NW",
            "connectivity": 8,
            "soundness": "every valid ancestor-closed prefix is 8-connected",
            "completeness": "every finite 8-connected mask has an exact BFS program with one node per pixel",
        },
        "overall": summarize_areas(records),
        "by_dataset": by_dataset,
        "by_split": by_split,
        "formal_hard_core": summarize_areas(hard_records),
    }
    provenance = {
        "schema_version": SCHEMA,
        "runs": runs,
        "hard_core_source": str(hard_core_source),
        "hard_core_source_sha256": _sha256_file(hard_core_source),
        "source_sha256": {
            "codec": _sha256_file(ROOT / "utils" / "rooted_component_program.py"),
            "audit": _sha256_file(Path(__file__).resolve()),
        },
        "git": _git_state(),
        "official_test_opened": False,
    }

    output_dir.mkdir(parents=True)
    with (output_dir / "target_programs.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "rcp_gt_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "rcp_gt_coverage_summary.md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote RCP GT coverage gate: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
