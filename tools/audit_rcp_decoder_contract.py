#!/usr/bin/env python3
"""Audit the neural-decoder contract implied by frozen RCP programs.

The audit is deliberately read-only with respect to the model and codec.  It
opens only the three clean fit/development manifests and their masks, rebuilds
the canonical rooted component programs, and measures the finite capacities a
future neural decoder must support.  Official test manifests and masks are not
opened.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from skimage import measure


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.rooted_component_program import (  # noqa: E402
    NEIGHBOUR_OFFSETS_8,
    RootedComponentProgram,
    encode_rooted_component,
    program_positions,
    render_rooted_component,
)
from utils.target_identity import build_stable_target_set  # noqa: E402


SCHEMA = "dea.rcp.decoder_contract.v1"
EXPECTED_DATASETS = frozenset({"IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"})
ALLOWED_SPLITS = frozenset({"fit", "dev"})
OFFSET_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


class RCPDecoderContractError(RuntimeError):
    """Raised when the decoder-contract evidence cannot be trusted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RCPDecoderContractError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RCPDecoderContractError(f"{path} is not a JSON object")
    return value


def load_target_program_rows(path: Path) -> list[dict[str, Any]]:
    """Load Gate-J rows and fail closed on any non-fit/dev record."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RCPDecoderContractError(
                        f"{path}:{line_number} is not a JSON object"
                    )
                split = row.get("split")
                if split not in ALLOWED_SPLITS:
                    raise RCPDecoderContractError(
                        f"{path}:{line_number} has forbidden split {split!r}"
                    )
                if not bool(row.get("exact_roundtrip")):
                    raise RCPDecoderContractError(
                        f"{path}:{line_number} is not an exact RCP round-trip"
                    )
                if int(row.get("program_nodes", -1)) != int(
                    row.get("target_area", -2)
                ):
                    raise RCPDecoderContractError(
                        f"{path}:{line_number} has inconsistent node/area counts"
                    )
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RCPDecoderContractError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise RCPDecoderContractError("target-program input is empty")
    identities = [(str(row["dataset"]), str(row["stable_target_id"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise RCPDecoderContractError("target-program input contains duplicate identities")
    return rows


def _read_names(path: Path) -> tuple[str, ...]:
    try:
        names = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeError) as exc:
        raise RCPDecoderContractError(f"cannot read fit source manifest {path}: {exc}") from exc
    if not names or len(names) != len(set(names)):
        raise RCPDecoderContractError(f"invalid or duplicate names in {path}")
    return names


def _names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def _resolve_dataset_path(dataset_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RCPDecoderContractError("run config is missing a required path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = dataset_dir / path
    return path.resolve()


def _fit_dev_names(stored: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Reproduce the clean holdout using only the training source manifest."""

    dataset_dir = Path(str(stored["dataset_dir"])).expanduser().resolve()
    source_path = _resolve_dataset_path(dataset_dir, stored.get("train_split_file"))
    source_names = _read_names(source_path)
    explicit_val = stored.get("val_split_file", "")
    if explicit_val:
        fit_names = source_names
        dev_names = _read_names(_resolve_dataset_path(dataset_dir, explicit_val))
    else:
        fraction = float(stored.get("val_fraction", 0.2))
        if not 0.0 < fraction < 1.0 or len(source_names) < 2:
            raise RCPDecoderContractError("invalid clean holdout specification")
        seed = int(stored.get("split_seed", stored.get("seed", 0)))
        ranked = sorted(
            source_names,
            key=lambda name: hashlib.sha256(
                (f"{seed}\0{name}").encode("utf-8")
            ).digest(),
        )
        number_dev = max(
            1,
            min(len(source_names) - 1, int(round(len(source_names) * fraction))),
        )
        dev_set = set(ranked[:number_dev])
        fit_names = tuple(name for name in source_names if name not in dev_set)
        dev_names = tuple(name for name in source_names if name in dev_set)
    if set(fit_names) & set(dev_names):
        raise RCPDecoderContractError("fit and development manifests overlap")
    actual_hashes = {
        "fit": _names_sha256(fit_names),
        "dev": _names_sha256(dev_names),
    }
    expected_hashes = {
        "fit": str(stored.get("train_split_sha256", "")),
        "dev": str(stored.get("val_split_sha256", "")),
    }
    if actual_hashes != expected_hashes:
        raise RCPDecoderContractError(
            "reconstructed fit/dev hashes disagree with clean run metadata"
        )
    return {"fit": fit_names, "dev": dev_names}


def _canonical_mask(path: Path, *, size: int) -> np.ndarray:
    if not path.is_file():
        raise RCPDecoderContractError(f"missing fit/dev target mask: {path}")
    with Image.open(path) as image:
        resized = image.resize((size, size), resample=Image.Resampling.NEAREST)
        array = np.asarray(resized)
    if array.ndim != 2:
        raise RCPDecoderContractError(f"target mask is not single-channel: {path}")
    if array.dtype != np.uint8:
        raise RCPDecoderContractError(f"target mask is not 8-bit: {path}")
    # Match Gate J's torchvision ToTensor()+threshold(0.5) conversion exactly.
    # A few source PNGs contain low-valued antialias/background pixels, so
    # requiring the raw files themselves to be exactly 0/255 would be wrong.
    return np.asarray(array > 127, dtype=np.bool_)


def program_contract(
    program: RootedComponentProgram,
    *,
    bbox: Sequence[int],
    shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Return exact structural statistics for one canonical BFS program."""

    positions = program_positions(program, shape=shape)
    depths = [0]
    parent_lags: list[int] = []
    for child_index, parent_index in enumerate(program.parent_indices, start=1):
        depths.append(depths[parent_index] + 1)
        parent_lags.append(child_index - parent_index)
    offset_counts = [0] * len(NEIGHBOUR_OFFSETS_8)
    for code in program.offset_codes:
        offset_counts[code] += 1
    root_y, root_x = positions[0]
    radius = max(
        max(abs(y - root_y), abs(x - root_x))
        for y, x in positions
    )
    if len(bbox) != 4:
        raise RCPDecoderContractError("bbox must contain four coordinates")
    top, left, bottom, right = (int(value) for value in bbox)
    if top >= bottom or left >= right:
        raise RCPDecoderContractError("bbox must have positive extent")

    # One observation per decoder state: root-only through the complete
    # component (the latter is the state from which STOP is chosen).  The
    # frontier is the set of unique, in-image, unoccupied 8-neighbours of the
    # current prefix.  It deliberately includes GT-background locations.  A
    # root-centred patch can only reduce this full-image upper bound.
    if shape is None:
        height = max(y for y, _ in positions) + 2
        width = max(x for _, x in positions) + 2
    else:
        height, width = shape
    occupied: set[tuple[int, int]] = set()
    frontier: set[tuple[int, int]] = set()
    frontier_sizes: list[int] = []
    for position in positions:
        frontier.discard(position)
        occupied.add(position)
        y, x = position
        for dy, dx in NEIGHBOUR_OFFSETS_8:
            neighbour = (y + dy, x + dx)
            if (
                0 <= neighbour[0] < height
                and 0 <= neighbour[1] < width
                and neighbour not in occupied
            ):
                frontier.add(neighbour)
        frontier_sizes.append(len(frontier))
    return {
        "canonical_bfs_depth": int(max(depths)),
        "node_depth_mean": float(np.mean(depths)),
        "parent_lag_max": int(max(parent_lags)) if parent_lags else 0,
        "parent_lag_mean": float(np.mean(parent_lags)) if parent_lags else 0.0,
        "offset_counts": offset_counts,
        "root_chebyshev_radius": int(radius),
        "bbox_height": bottom - top,
        "bbox_width": right - left,
        "bbox_max_side": max(bottom - top, right - left),
        "frontier_candidate_size_mean": float(np.mean(frontier_sizes)),
        "frontier_candidate_size_max": int(max(frontier_sizes)),
        # Private exact samples are retained for pooled command-level summaries
        # and removed before target JSONL serialization.
        "_node_depths": depths,
        "_parent_lags": parent_lags,
        "_frontier_candidate_sizes": frontier_sizes,
    }


def component_separation_contract(
    component_masks: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Measure pairwise pixel-center Chebyshev distances between components."""

    pair_distances: list[int] = []
    coordinates = [np.argwhere(np.asarray(mask, dtype=np.bool_)) for mask in component_masks]
    if any(points.ndim != 2 or points.shape[1:] != (2,) or len(points) == 0 for points in coordinates):
        raise RCPDecoderContractError("component masks must be non-empty 2-D supports")
    for first_index, first in enumerate(coordinates):
        for second in coordinates[first_index + 1 :]:
            differences = np.abs(first[:, None, :] - second[None, :, :])
            distance = int(np.min(np.max(differences, axis=2)))
            pair_distances.append(distance)
    minimum = min(pair_distances) if pair_distances else None
    return {
        "component_pair_count": len(pair_distances),
        "minimum_distinct_component_chebyshev_distance": minimum,
        "minimum_empty_pixel_gap": (minimum - 1) if minimum is not None else None,
        "any_distinct_components_8_adjacent": bool(
            minimum is not None and minimum <= 1
        ),
        "_pair_distances": pair_distances,
    }


def _numeric_summary(values: Iterable[int | float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise RCPDecoderContractError("numeric summary requires finite observations")
    quantiles = {
        f"q{int(level * 100):02d}": float(np.quantile(array, level))
        for level in QUANTILES
    }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "quantiles": quantiles,
    }


def _coverage(values: Iterable[int], capacity: int) -> dict[str, Any]:
    selected = np.asarray(list(values), dtype=np.int64)
    if selected.size == 0:
        raise RCPDecoderContractError("capacity coverage requires observations")
    covered = int(np.sum(selected <= capacity))
    return {
        "capacity": int(capacity),
        "covered": covered,
        "total": int(selected.size),
        "coverage": float(covered / selected.size),
        "maximum_observed": int(selected.max()),
    }


def summarize_contract(
    target_records: Sequence[Mapping[str, Any]],
    image_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not target_records or not image_records:
        raise RCPDecoderContractError("contract groups must contain targets and images")
    parent_lags = [
        int(value)
        for row in target_records
        for value in row["_parent_lags"]
    ]
    node_depths = [
        int(value)
        for row in target_records
        for value in row["_node_depths"]
    ]
    pair_distances = [
        int(value)
        for row in image_records
        for value in row["_pair_distances"]
    ]
    frontier_sizes = [
        int(value)
        for row in target_records
        for value in row["_frontier_candidate_sizes"]
    ]
    offset_counts = np.sum(
        np.asarray([row["offset_counts"] for row in target_records], dtype=np.int64),
        axis=0,
    )
    command_count = int(offset_counts.sum())
    return {
        "target_count": len(target_records),
        "image_count": len(image_records),
        "program_nodes": _numeric_summary(int(row["program_nodes"]) for row in target_records),
        "canonical_bfs_depth_per_target": _numeric_summary(
            int(row["canonical_bfs_depth"]) for row in target_records
        ),
        "canonical_bfs_node_depth": _numeric_summary(node_depths),
        "parent_lag_per_command": _numeric_summary(parent_lags),
        "parent_lag_max_per_target": _numeric_summary(
            int(row["parent_lag_max"]) for row in target_records
        ),
        "frontier_candidate_size_per_decoder_state": _numeric_summary(
            frontier_sizes
        ),
        "frontier_candidate_size_max_per_target": _numeric_summary(
            int(row["frontier_candidate_size_max"]) for row in target_records
        ),
        "root_chebyshev_radius": _numeric_summary(
            int(row["root_chebyshev_radius"]) for row in target_records
        ),
        "bbox_height": _numeric_summary(int(row["bbox_height"]) for row in target_records),
        "bbox_width": _numeric_summary(int(row["bbox_width"]) for row in target_records),
        "bbox_max_side": _numeric_summary(int(row["bbox_max_side"]) for row in target_records),
        "offset_frequency": {
            name: {
                "count": int(offset_counts[index]),
                "fraction": float(offset_counts[index] / command_count),
                "dy": NEIGHBOUR_OFFSETS_8[index][0],
                "dx": NEIGHBOUR_OFFSETS_8[index][1],
            }
            for index, name in enumerate(OFFSET_NAMES)
        },
        "component_count_per_image": _numeric_summary(
            int(row["component_count"]) for row in image_records
        ),
        "target_free_images": sum(int(row["component_count"]) == 0 for row in image_records),
        "multi_component_images": sum(int(row["component_count"]) >= 2 for row in image_records),
        "component_pair_count": len(pair_distances),
        "pairwise_component_chebyshev_distance": (
            _numeric_summary(pair_distances) if pair_distances else None
        ),
        "minimum_distance_per_multi_component_image": _numeric_summary(
            int(row["minimum_distinct_component_chebyshev_distance"])
            for row in image_records
            if row["minimum_distinct_component_chebyshev_distance"] is not None
        ) if any(
            row["minimum_distinct_component_chebyshev_distance"] is not None
            for row in image_records
        ) else None,
        "distinct_component_8_adjacency_images": sum(
            bool(row["any_distinct_components_8_adjacent"])
            for row in image_records
        ),
    }


def fit_only_preregistration(
    target_records: Sequence[Mapping[str, Any]],
    image_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose exact-capacity decoder constants from pooled fit only."""

    fit_targets = [row for row in target_records if row["split"] == "fit"]
    fit_images = [row for row in image_records if row["split"] == "fit"]
    if not fit_targets or not fit_images:
        raise RCPDecoderContractError("fit-only preregistration has no fit observations")
    fit_maximum_action_states = max(
        int(row["program_nodes"]) for row in fit_targets
    )
    # A fixed power-of-two rule is declared here before dev is inspected.  It
    # gives a static tensor horizon while remaining a deterministic function
    # of fit evidence rather than a response to a dev outlier.
    action_horizon = 1 << (fit_maximum_action_states - 1).bit_length()
    selected = {
        # Root localization is a separate proposal decision.  A component
        # with n pixels then needs n-1 ADD decisions plus one STOP decision,
        # hence exactly n frontier-action states.
        "T": action_horizon,
        "fit_maximum_required_T": fit_maximum_action_states,
        "local_patch_radius": max(
            int(row["root_chebyshev_radius"]) for row in fit_targets
        ),
        "topK": max(int(row["component_count"]) for row in fit_images),
        "parallel_growth_rounds": max(
            int(row["canonical_bfs_depth"]) for row in fit_targets
        ),
    }
    selected["maximum_add_actions"] = selected["T"] - 1
    selected["local_patch_side"] = 2 * selected["local_patch_radius"] + 1

    def report(split: str) -> dict[str, Any]:
        targets = [row for row in target_records if row["split"] == split]
        images = [row for row in image_records if row["split"] == split]
        return {
            "T_target_coverage": _coverage(
                (int(row["program_nodes"]) for row in targets), selected["T"]
            ),
            "patch_target_coverage": _coverage(
                (int(row["root_chebyshev_radius"]) for row in targets),
                selected["local_patch_radius"],
            ),
            "topK_image_coverage": _coverage(
                (int(row["component_count"]) for row in images), selected["topK"]
            ),
        }

    hard_targets = [row for row in target_records if bool(row["formal_hard_core"])]
    return {
        "selection_source": "pooled fit masks from the three clean datasets only",
        "selection_rule": {
            "T": "smallest power of two not below the maximum fit program nodes, where n nodes require n-1 ADD actions plus one STOP after root proposal",
            "parallel_growth_rounds": "maximum canonical BFS depth among fit targets; diagnostic for a non-autoregressive wavefront alternative",
            "local_patch_radius": "maximum root-to-pixel Chebyshev radius among fit targets",
            "topK": "maximum 8-connected component count among fit images, including target-free images",
        },
        "selected": selected,
        "fit_confirmation": report("fit"),
        "dev_report_only": report("dev"),
        "formal_hard_core_report_only": {
            "target_count": len(hard_targets),
            "T_target_coverage": _coverage(
                (int(row["program_nodes"]) for row in hard_targets), selected["T"]
            ),
            "patch_target_coverage": _coverage(
                (int(row["root_chebyshev_radius"]) for row in hard_targets),
                selected["local_patch_radius"],
            ),
        },
        "lock_rule": "development observations are reported but must not alter T, patch radius, or topK",
    }


def _load_sources(
    gate_provenance: Path,
) -> tuple[dict[str, tuple[Path, int, dict[str, tuple[str, ...]], Path]], list[dict[str, Any]]]:
    provenance = _read_json(gate_provenance)
    runs = provenance.get("runs")
    if not isinstance(runs, list):
        raise RCPDecoderContractError("Gate-J provenance has no run list")
    sources: dict[str, tuple[Path, int, dict[str, tuple[str, ...]], Path]] = {}
    run_records: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise RCPDecoderContractError("Gate-J run provenance is malformed")
        dataset = str(run.get("dataset"))
        if dataset in sources:
            raise RCPDecoderContractError(f"duplicate dataset source: {dataset}")
        config_path = Path(str(run.get("run_config"))).expanduser().resolve()
        if _sha256_file(config_path) != str(run.get("run_config_sha256")):
            raise RCPDecoderContractError(f"run config digest drifted: {config_path}")
        config = _read_json(config_path)
        stored = config.get("args")
        if not isinstance(stored, dict):
            raise RCPDecoderContractError(f"run config has no args mapping: {config_path}")
        if stored.get("mode") != "train" or stored.get("model_type") != "mshnet":
            raise RCPDecoderContractError("decoder contract requires clean MSHNet training runs")
        dataset_dir = Path(str(stored["dataset_dir"])).expanduser().resolve()
        if dataset_dir.name != dataset:
            raise RCPDecoderContractError("dataset directory disagrees with provenance")
        size = int(stored["base_size"])
        split_names = _fit_dev_names(stored)
        sources[dataset] = (dataset_dir, size, split_names, config_path)
        run_records.append(
            {
                "dataset": dataset,
                "dataset_dir": str(dataset_dir),
                "base_size": size,
                "run_config": str(config_path),
                "run_config_sha256": _sha256_file(config_path),
                "fit_images": len(split_names["fit"]),
                "dev_images": len(split_names["dev"]),
                "split_sha256": {
                    split: _names_sha256(names) for split, names in split_names.items()
                },
            }
        )
    if frozenset(sources) != EXPECTED_DATASETS:
        raise RCPDecoderContractError("decoder contract requires all three clean datasets")
    return sources, run_records


def build_contract_records(
    target_rows: Sequence[Mapping[str, Any]],
    sources: Mapping[str, tuple[Path, int, Mapping[str, Sequence[str]], Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild and verify every target program, including zero-target images."""

    rows_by_image: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in target_rows:
        key = (str(row["dataset"]), str(row["split"]), str(row["image_name"]))
        rows_by_image.setdefault(key, []).append(row)
    target_records: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    visited_target_ids: set[tuple[str, str]] = set()

    for dataset in sorted(sources):
        dataset_dir, size, split_names, _ = sources[dataset]
        for split in ("fit", "dev"):
            for image_name in split_names[split]:
                mask = _canonical_mask(
                    dataset_dir / "masks" / f"{image_name}.png",
                    size=size,
                )
                target_set = build_stable_target_set(
                    mask,
                    dataset=dataset,
                    image_name=image_name,
                    connectivity=2,
                )
                labels = measure.label(mask, connectivity=2)
                expected_rows = rows_by_image.get((dataset, split, image_name), [])
                expected_by_id = {str(row["stable_target_id"]): row for row in expected_rows}
                actual_ids = {target.stable_key for target in target_set.targets}
                if actual_ids != set(expected_by_id):
                    raise RCPDecoderContractError(
                        f"Gate-J target rows disagree with mask components: {dataset}/{split}/{image_name}"
                    )
                component_masks: list[np.ndarray] = []
                for target in target_set.targets:
                    source_row = expected_by_id[target.stable_key]
                    component = labels == target.source_label
                    component_masks.append(component)
                    program = encode_rooted_component(component)
                    if not np.array_equal(
                        render_rooted_component(program, component.shape), component
                    ):
                        raise RCPDecoderContractError("canonical RCP reconstruction drifted")
                    assertions = {
                        "component_mask_sha256": target.component_mask_sha256,
                        "target_area": target.area,
                        "bbox": list(target.bbox),
                        "root_y": program.root_y,
                        "root_x": program.root_x,
                        "program_nodes": program.node_count,
                    }
                    if any(source_row.get(field) != value for field, value in assertions.items()):
                        raise RCPDecoderContractError(
                            f"Gate-J program metadata drifted for {target.stable_key}"
                        )
                    contract = program_contract(
                        program,
                        bbox=target.bbox,
                        shape=component.shape,
                    )
                    target_records.append(
                        {
                            "schema_version": SCHEMA,
                            "dataset": dataset,
                            "split": split,
                            "image_name": image_name,
                            "stable_target_id": target.stable_key,
                            "formal_hard_core": bool(source_row["formal_hard_core"]),
                            "program_nodes": program.node_count,
                            **contract,
                        }
                    )
                    visited_target_ids.add((dataset, target.stable_key))
                separation = component_separation_contract(component_masks) if component_masks else {
                    "component_pair_count": 0,
                    "minimum_distinct_component_chebyshev_distance": None,
                    "minimum_empty_pixel_gap": None,
                    "any_distinct_components_8_adjacent": False,
                    "_pair_distances": [],
                }
                image_records.append(
                    {
                        "schema_version": SCHEMA,
                        "dataset": dataset,
                        "split": split,
                        "image_name": image_name,
                        "component_count": len(component_masks),
                        **separation,
                    }
                )

    expected_target_ids = {
        (str(row["dataset"]), str(row["stable_target_id"])) for row in target_rows
    }
    if visited_target_ids != expected_target_ids:
        raise RCPDecoderContractError("not all Gate-J targets were visited through fit/dev masks")
    if any(row["any_distinct_components_8_adjacent"] for row in image_records):
        raise RCPDecoderContractError(
            "different 8-connected GT components cannot be 8-neighbour adjacent"
        )
    return target_records, image_records


def _public_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


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


def _markdown(summary: Mapping[str, Any]) -> str:
    selected = summary["fit_only_preregistration"]["selected"]
    fit_eval = summary["fit_only_preregistration"]["fit_confirmation"]
    dev_eval = summary["fit_only_preregistration"]["dev_report_only"]
    hard_eval = summary["fit_only_preregistration"]["formal_hard_core_report_only"]
    fit = summary["by_split"]["fit"]
    dev = summary["by_split"]["dev"]
    hard = summary["formal_hard_core"]
    lines = [
        "# RCP neural decoder contract audit",
        "",
        "Fit-only capacity preregistration; development masks are report-only; official test sealed.",
        "",
        "## Locked decoder capacities",
        "",
        f"- Frontier-action horizon `T = {selected['T']}` decisions after root proposal: the smallest power of two covering the fit requirement `{selected['fit_maximum_required_T']}`; at most `{selected['maximum_add_actions']}` ADD actions plus one STOP.",
        f"- Canonical parallel-growth depth is separately bounded by `{selected['parallel_growth_rounds']}` rounds; this is not the autoregressive `T`.",
        f"- Root-centred local patch radius `R = {selected['local_patch_radius']}` (`{selected['local_patch_side']} x {selected['local_patch_side']}`).",
        f"- Root proposal capacity `topK = {selected['topK']}` per image.",
        "- Rule: pooled three-dataset fit maxima only. Dev and hard-core observations did not select or enlarge these constants.",
        "",
        "| Capacity | fit coverage | dev coverage | hard-core coverage |",
        "|---|---:|---:|---:|",
        f"| frontier-action T / target | {fit_eval['T_target_coverage']['coverage']:.6f} | {dev_eval['T_target_coverage']['coverage']:.6f} | {hard_eval['T_target_coverage']['coverage']:.6f} |",
        f"| patch / target | {fit_eval['patch_target_coverage']['coverage']:.6f} | {dev_eval['patch_target_coverage']['coverage']:.6f} | {hard_eval['patch_target_coverage']['coverage']:.6f} |",
        f"| topK / image | {fit_eval['topK_image_coverage']['coverage']:.6f} | {dev_eval['topK_image_coverage']['coverage']:.6f} | n/a |",
        "",
        "## Structural distributions",
        "",
        "| Scope | targets | images | BFS depth q99 / max | parent lag q99 / max | frontier candidates mean / q95 / max | root radius q99 / max | bbox max-side q99 / max | components/image max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in (("fit", fit), ("dev", dev), ("formal hard-core", hard)):
        depth = group["canonical_bfs_depth_per_target"]["quantiles"]
        lag = group["parent_lag_per_command"]["quantiles"]
        frontier = group["frontier_candidate_size_per_decoder_state"]
        radius = group["root_chebyshev_radius"]["quantiles"]
        bbox = group["bbox_max_side"]["quantiles"]
        components = group["component_count_per_image"]["quantiles"]
        lines.append(
            f"| {name} | {group['target_count']} | {group['image_count']} | "
            f"{depth['q99']:.2f} / {depth['q100']:.0f} | "
            f"{lag['q99']:.2f} / {lag['q100']:.0f} | "
            f"{frontier['mean']:.2f} / {frontier['quantiles']['q95']:.2f} / {frontier['quantiles']['q100']:.0f} | "
            f"{radius['q99']:.2f} / {radius['q100']:.0f} | "
            f"{bbox['q99']:.2f} / {bbox['q100']:.0f} | {components['q100']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Three-dataset contract breakdown",
            "",
            "| Dataset | split | targets | images | program nodes max | BFS depth max | root radius max | bbox max-side max | components/image max | pair distance min | 8-adjacent images |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, split_groups in summary["by_dataset_split"].items():
        for split in ("fit", "dev"):
            group = split_groups[split]
            pair = group["pairwise_component_chebyshev_distance"]
            pair_min = (
                f"{pair['quantiles']['q00']:.0f}" if pair is not None else "n/a"
            )
            lines.append(
                f"| {dataset} | {split} | {group['target_count']} | {group['image_count']} | "
                f"{group['program_nodes']['quantiles']['q100']:.0f} | "
                f"{group['canonical_bfs_depth_per_target']['quantiles']['q100']:.0f} | "
                f"{group['root_chebyshev_radius']['quantiles']['q100']:.0f} | "
                f"{group['bbox_max_side']['quantiles']['q100']:.0f} | "
                f"{group['component_count_per_image']['quantiles']['q100']:.0f} | "
                f"{pair_min} | {group['distinct_component_8_adjacency_images']} |"
            )
    lines.extend(
        [
            "",
            "## Canonical 8-neighbour command frequencies",
            "",
            "| Offset | dy | dx | fit count / fraction | dev count / fraction | hard-core count / fraction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for offset in OFFSET_NAMES:
        fit_value = fit["offset_frequency"][offset]
        dev_value = dev["offset_frequency"][offset]
        hard_value = hard["offset_frequency"][offset]
        lines.append(
            f"| {offset} | {fit_value['dy']} | {fit_value['dx']} | "
            f"{fit_value['count']} / {fit_value['fraction']:.6f} | "
            f"{dev_value['count']} / {dev_value['fraction']:.6f} | "
            f"{hard_value['count']} / {hard_value['fraction']:.6f} |"
        )
    overall_images = summary["overall"]["image_count"]
    adjacent_images = summary["overall"]["distinct_component_8_adjacency_images"]
    pair_summary = summary["overall"]["pairwise_component_chebyshev_distance"]
    lines.extend(
        [
            "",
            "## Component-separation invariant",
            "",
            f"- Distinct-component pairs: {summary['overall']['component_pair_count']} across {overall_images} fit/dev images.",
            f"- Minimum observed pixel-centre Chebyshev distance: {pair_summary['quantiles']['q00']:.0f} pixels (one empty-pixel gap corresponds to distance 2).",
            f"- Images containing distinct 8-neighbour-adjacent GT components: {adjacent_images}. Expected and required value: 0, because components were defined with 8-connectivity.",
            "",
            "`T` is the serialized frontier-action horizon after root proposal: one action per added pixel and one STOP. Canonical BFS depth is reported separately because it would be the number of rounds only for a parallel wavefront decoder. `R` is the exact square support needed around the canonical root. `topK` includes zero-target images when its per-image distribution is measured.",
            "Frontier size is measured after every canonical prefix, including the final STOP state, as unique in-image unoccupied 8-neighbours of the occupied prefix. It includes GT background and is a conservative upper bound for a decoder restricted to the locked local patch.",
            "",
            "This audit establishes a finite decoder capacity contract only. It does not show that a neural RCP decoder is learnable or improves Pd-FA performance.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-programs",
        default="repro_runs/gate_j/rcp_gt_coverage_v1/target_programs.jsonl",
    )
    parser.add_argument(
        "--gate-provenance",
        default="repro_runs/gate_j/rcp_gt_coverage_v1/provenance.json",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_programs = Path(args.target_programs).expanduser()
    if not target_programs.is_absolute():
        target_programs = (ROOT / target_programs).resolve()
    gate_provenance = Path(args.gate_provenance).expanduser()
    if not gate_provenance.is_absolute():
        gate_provenance = (ROOT / gate_provenance).resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    target_rows = load_target_program_rows(target_programs)
    if frozenset(str(row["dataset"]) for row in target_rows) != EXPECTED_DATASETS:
        raise RCPDecoderContractError("target programs must cover the three clean datasets")
    if sum(bool(row["formal_hard_core"]) for row in target_rows) != 16:
        raise RCPDecoderContractError("formal hard-core subset must contain 16 targets")
    sources, run_records = _load_sources(gate_provenance)
    target_records, image_records = build_contract_records(target_rows, sources)

    overall = summarize_contract(target_records, image_records)
    by_split = {
        split: summarize_contract(
            [row for row in target_records if row["split"] == split],
            [row for row in image_records if row["split"] == split],
        )
        for split in ("fit", "dev")
    }
    by_dataset_split = {
        dataset: {
            split: summarize_contract(
                [
                    row for row in target_records
                    if row["dataset"] == dataset and row["split"] == split
                ],
                [
                    row for row in image_records
                    if row["dataset"] == dataset and row["split"] == split
                ],
            )
            for split in ("fit", "dev")
        }
        for dataset in sorted(EXPECTED_DATASETS)
    }
    hard_targets = [row for row in target_records if bool(row["formal_hard_core"])]
    hard_image_keys = {
        (row["dataset"], row["split"], row["image_name"]) for row in hard_targets
    }
    hard_images = [
        row
        for row in image_records
        if (row["dataset"], row["split"], row["image_name"]) in hard_image_keys
    ]
    summary = {
        "schema_version": SCHEMA,
        "scope": "three-dataset fit/dev masks only; official test sealed",
        "definitions": {
            "T": "frontier-action decisions after root proposal: one ADD per non-root node plus one STOP",
            "canonical_bfs_depth": "maximum root-to-node edge count in the frozen canonical BFS tree",
            "parent_lag": "child BFS index minus parent BFS index",
            "frontier_candidate_size": "unique in-image unoccupied 8-neighbours after each canonical prefix, including background and the final STOP state",
            "root_chebyshev_radius": "maximum Chebyshev distance from canonical root to a component pixel",
            "distinct_component_distance": "minimum pixel-centre Chebyshev distance between different 8-connected GT components",
            "eight_adjacent": "distance <= 1; impossible for distinct components under 8-connectivity",
        },
        "fit_only_preregistration": fit_only_preregistration(
            target_records, image_records
        ),
        "overall": overall,
        "by_split": by_split,
        "by_dataset_split": by_dataset_split,
        "formal_hard_core": summarize_contract(hard_targets, hard_images),
        "connectivity_contract_pass": (
            overall["distinct_component_8_adjacency_images"] == 0
        ),
    }
    provenance = {
        "schema_version": SCHEMA,
        "inputs": {
            "target_programs": str(target_programs),
            "target_programs_sha256": _sha256_file(target_programs),
            "gate_provenance": str(gate_provenance),
            "gate_provenance_sha256": _sha256_file(gate_provenance),
        },
        "runs": run_records,
        "source_sha256": {
            "codec": _sha256_file(ROOT / "utils" / "rooted_component_program.py"),
            "audit": _sha256_file(Path(__file__).resolve()),
        },
        "selection_data": ["fit"],
        "report_only_data": ["dev", "formal_hard_core"],
        "official_test_opened": False,
        "git": _git_state(),
    }

    output_dir.mkdir(parents=True)
    with (output_dir / "target_decoder_contract.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in target_records:
            handle.write(json.dumps(_public_record(row), sort_keys=True, allow_nan=False) + "\n")
    with (output_dir / "image_decoder_contract.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in image_records:
            handle.write(json.dumps(_public_record(row), sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "rcp_decoder_contract_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "rcp_decoder_contract_summary.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote RCP decoder contract audit: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
