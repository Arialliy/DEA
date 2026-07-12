#!/usr/bin/env python3
"""Gate F0 analytic precheck for generic finite-sample risk control."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_gate_f_operating_transport import (  # noqa: E402
    _read_json,
    _read_jsonl,
    _verify_input_bundle,
    sha256_file,
)
from utils.risk_control_feasibility import (  # noqa: E402
    DEFAULT_CONFIDENCE_DELTAS,
    RiskControlFeasibilityError,
    analyze_risk_control_feasibility,
)


PROVENANCE_SCHEMA = "dea.gate_f0.risk_feasibility_provenance.v1"
OUTPUT_FILES = (
    "risk_feasibility.jsonl",
    "risk_feasibility_summary.json",
    "risk_feasibility_summary.md",
    "provenance.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether best-case HB-LTT and standard unit-bound CRC can "
            "possibly certify the frozen component FA budgets at the available "
            "image sample sizes."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="repro_runs/gate_e/persistence_v2/low_fa_bridge",
    )
    parser.add_argument(
        "--output-dir",
        default="repro_runs/gate_f/risk_control_feasibility_v1",
    )
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _fold_sizes(
    fold_mappings: Mapping[str, Mapping[str, int]],
    *,
    fold_count: int,
) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for dataset, mapping in fold_mappings.items():
        if not isinstance(dataset, str) or not dataset or not isinstance(
            mapping, Mapping
        ):
            raise RiskControlFeasibilityError("invalid E-1c fold mapping")
        counts = {fold: 0 for fold in range(fold_count)}
        for image_name, fold in mapping.items():
            if (
                not isinstance(image_name, str)
                or not image_name
                or isinstance(fold, bool)
                or not isinstance(fold, int)
                or fold not in counts
            ):
                raise RiskControlFeasibilityError("invalid image fold assignment")
            counts[fold] += 1
        if any(value <= 0 for value in counts.values()):
            raise RiskControlFeasibilityError("dataset lacks a non-empty fold")
        result[dataset] = counts
    if not result:
        raise RiskControlFeasibilityError("fold mapping cannot be empty")
    return result


def _candidate_grid_size(
    calibration: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int],
) -> int:
    if not calibration:
        raise RiskControlFeasibilityError("calibration records cannot be empty")
    sizes: set[int] = set()
    for record in calibration:
        grid = record.get("threshold_grid")
        selections = record.get("selections")
        if (
            record.get("schema_version")
            != "dea.gate_e.low_fa_calibration.v1"
            or not isinstance(grid, list)
            or not grid
            or not isinstance(selections, Mapping)
            or set(selections) != {str(value) for value in budgets}
        ):
            raise RiskControlFeasibilityError("calibration grid contract drifted")
        sizes.add(len(grid))
    if len(sizes) != 1:
        raise RiskControlFeasibilityError("calibration grid sizes are not uniform")
    return sizes.pop()


def _validate_equal_image_area(
    image_rows: Sequence[Mapping[str, Any]],
    *,
    fold_mappings: Mapping[str, Mapping[str, int]],
) -> tuple[int, dict[str, int]]:
    metadata: dict[tuple[str, str], int] = {}
    for row in image_rows:
        dataset = row.get("dataset")
        image_name = row.get("image_name")
        pixels = row.get("total_pixels")
        if (
            not isinstance(dataset, str)
            or not isinstance(image_name, str)
            or isinstance(pixels, bool)
            or not isinstance(pixels, int)
            or pixels <= 0
        ):
            raise RiskControlFeasibilityError("invalid image population metadata")
        key = (dataset, image_name)
        previous = metadata.setdefault(key, pixels)
        if previous != pixels:
            raise RiskControlFeasibilityError("image area changes across ledger rows")
    expected = {
        (dataset, image_name)
        for dataset, mapping in fold_mappings.items()
        for image_name in mapping
    }
    if set(metadata) != expected:
        raise RiskControlFeasibilityError("image ledger/fold universe mismatch")
    areas = set(metadata.values())
    if len(areas) != 1:
        raise RiskControlFeasibilityError(
            "pooled FA is not an unweighted image mean for unequal image areas"
        )
    return areas.pop(), {
        dataset: len(mapping) for dataset, mapping in fold_mappings.items()
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gate F0 generic risk-control sample-size precheck",
        "",
        str(summary["analysis_scope"]),
        "",
        "This is an optimistic zero-empirical-loss bound, not an observed model result.",
        "",
        "| FA/Mpix | max cross-fit n | max development n | HB-LTT n at delta=.1 (single) | HB-LTT n at delta=.1 (54-way Bonferroni) | standard CRC n floor |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in summary["protocol"]["budgets_fa_per_mpix"]:
        record = summary["by_budget"][str(budget)]
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                budget,
                record["maximum_crossfit_calibration_image_count"],
                record["maximum_full_development_image_count"],
                record["minimum_hb_ltt_images_at_delta_0.1_single_candidate"],
                record["minimum_hb_ltt_images_at_delta_0.1_bonferroni"],
                record["minimum_standard_crc_images_for_unit_bound_floor"],
            )
        )
    lines.extend(
        [
            "",
            f"- Decision: {summary['pre_gate']['decision']}",
            "- A deterministic no-prediction rule is safe but has Pd=0 and is vetoed.",
            f"- Scope limit: {summary['scope_limit']}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def write_bundle(
    output_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        ledger_path = temporary / OUTPUT_FILES[0]
        summary_path = temporary / OUTPUT_FILES[1]
        markdown_path = temporary / OUTPUT_FILES[2]
        _write_jsonl(ledger_path, rows)
        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(build_markdown(summary), encoding="utf-8")
        artifacts = (ledger_path, summary_path, markdown_path)
        complete_provenance = {
            **dict(provenance),
            "artifact_sha256": {
                path.name: sha256_file(path) for path in artifacts
            },
        }
        (temporary / OUTPUT_FILES[3]).write_text(
            json.dumps(
                complete_provenance,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if {path.name for path in temporary.iterdir()} != set(OUTPUT_FILES):
            raise RiskControlFeasibilityError("temporary Gate F0 inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise RiskControlFeasibilityError("Gate F0 output cannot modify E-1c")

    input_provenance, formal_summary = _verify_input_bundle(input_dir)
    input_provenance_hash = sha256_file(input_dir / "provenance.json")
    source_hashes = {
        "tool": sha256_file(Path(__file__).resolve()),
        "utility": sha256_file(ROOT / "utils" / "risk_control_feasibility.py"),
        "input_verifier": sha256_file(
            ROOT / "tools" / "analyze_gate_f_operating_transport.py"
        ),
    }
    if formal_summary["joint_gate"].get("pass") is not False:
        raise RiskControlFeasibilityError("Gate F0 requires the failed formal E-1c")
    protocol = input_provenance.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RiskControlFeasibilityError("E-1c protocol is missing")
    budgets = protocol.get("budgets_fa_per_mpix")
    matchers = protocol.get("matchers")
    fold_count = protocol.get("fold_count")
    fold_mappings = protocol.get("fold_mappings")
    if (
        not isinstance(budgets, list)
        or not isinstance(matchers, list)
        or isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or fold_count <= 1
        or not isinstance(fold_mappings, Mapping)
    ):
        raise RiskControlFeasibilityError("E-1c protocol fields are invalid")
    calibration = _read_json(input_dir / "calibration.json")
    if not isinstance(calibration, list):
        raise RiskControlFeasibilityError("calibration artifact must be a list")
    image_rows = _read_jsonl(input_dir / "image_low_fa.jsonl")
    fold_sizes = _fold_sizes(fold_mappings, fold_count=fold_count)
    grid_size = _candidate_grid_size(calibration, budgets=budgets)
    image_pixels, dataset_image_counts = _validate_equal_image_area(
        image_rows, fold_mappings=fold_mappings
    )
    rows, base_summary = analyze_risk_control_feasibility(
        fold_sizes,
        budgets_fa_per_mpix=budgets,
        candidate_grid_size=grid_size,
        confidence_deltas=DEFAULT_CONFIDENCE_DELTAS,
    )
    for row in rows:
        alpha = float(row["target_area_fraction_risk"])
        sample_count = int(row["sample_count"])
        maximum_bound = min(1.0, alpha * (sample_count + 1))
        row["standard_crc"][
            "maximum_deterministic_loss_bound_for_nonempty_zero_loss_set"
        ] = maximum_bound
        row["standard_crc"][
            "equivalent_total_prediction_area_cap_pixels_floor"
        ] = int(maximum_bound * image_pixels)
    structural_escape_by_budget: dict[str, Any] = {}
    for budget in budgets:
        budget_rows = [
            row
            for row in rows
            if row["budget_fa_per_mpix"] == budget
            and row["sample_scope"] == "crossfit_calibration_images"
        ]
        most_favorable = max(budget_rows, key=lambda row: int(row["sample_count"]))
        structural_escape_by_budget[str(budget)] = {
            "most_favorable_crossfit_sample_count": most_favorable["sample_count"],
            "maximum_deterministic_loss_bound": most_favorable["standard_crc"][
                "maximum_deterministic_loss_bound_for_nonempty_zero_loss_set"
            ],
            "equivalent_total_prediction_area_cap_pixels_floor": most_favorable[
                "standard_crc"
            ]["equivalent_total_prediction_area_cap_pixels_floor"],
        }
    summary = {
        **base_summary,
        "formal_gate_effect": "none; Gate E-1c remains FAIL",
        "image_population": {
            "pixels_per_image": image_pixels,
            "equal_image_area_verified": True,
            "dataset_image_counts": dataset_image_counts,
        },
        "candidate_grid_caveat": (
            "the E-1c grids were constructed from calibration logits; the "
            "precheck optimistically treats candidates as independently frozen, "
            "which can only favor certifiability"
        ),
        "structural_escape_condition": {
            "by_budget": structural_escape_by_budget,
            "interpretation": (
                "reducing the loss bound would require an a-priori deterministic "
                "output-area cap; no such predictor exists in the frozen baseline. "
                "Capping total prediction area would change inference behavior, and "
                "capping unmatched area directly would require unavailable GT"
            ),
        },
        "references": {
            "learn_then_test": "https://arxiv.org/abs/2110.01052",
            "conformal_risk_control": "https://arxiv.org/abs/2208.02814",
            "non_monotone_crc_finite_grid": "https://arxiv.org/abs/2604.01502",
        },
    }

    rechecked_provenance, rechecked_summary = _verify_input_bundle(input_dir)
    current_source_hashes = {
        "tool": sha256_file(Path(__file__).resolve()),
        "utility": sha256_file(ROOT / "utils" / "risk_control_feasibility.py"),
        "input_verifier": sha256_file(
            ROOT / "tools" / "analyze_gate_f_operating_transport.py"
        ),
    }
    if (
        input_provenance != rechecked_provenance
        or formal_summary != rechecked_summary
        or input_provenance_hash != sha256_file(input_dir / "provenance.json")
        or source_hashes != current_source_hashes
    ):
        raise RiskControlFeasibilityError("Gate F0 source/input changed during audit")
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "source_sha256": source_hashes,
        "source_bundle": {
            "path": str(input_dir),
            "provenance_sha256": input_provenance_hash,
            "artifact_sha256": dict(input_provenance["artifact_sha256"]),
            "formal_joint_gate_pass": False,
        },
        "analysis_contract": {
            "read_only": True,
            "model_inference_rerun": False,
            "zero_empirical_loss_best_case": True,
            "formal_gate_unchanged": True,
            "deterministic_all_off_vetoed": True,
        },
        "frozen_protocol": {
            "budgets_fa_per_mpix": budgets,
            "candidate_grid_size": grid_size,
            "confidence_deltas": list(DEFAULT_CONFIDENCE_DELTAS),
            "matchers": matchers,
            "fold_count": fold_count,
        },
        "runtime": {"python": sys.version},
    }
    write_bundle(output_dir, rows=rows, summary=summary, provenance=provenance)
    try:
        rechecked_provenance, rechecked_summary = _verify_input_bundle(input_dir)
        current_source_hashes = {
            "tool": sha256_file(Path(__file__).resolve()),
            "utility": sha256_file(
                ROOT / "utils" / "risk_control_feasibility.py"
            ),
            "input_verifier": sha256_file(
                ROOT / "tools" / "analyze_gate_f_operating_transport.py"
            ),
        }
        if (
            input_provenance != rechecked_provenance
            or formal_summary != rechecked_summary
            or input_provenance_hash != sha256_file(input_dir / "provenance.json")
            or source_hashes != current_source_hashes
        ):
            raise RiskControlFeasibilityError(
                "Gate F0 source/input changed before handoff"
            )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return summary, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    summary, output_dir = run(parse_args(argv))
    print(summary["pre_gate"]["decision"])
    print(f"wrote immutable Gate F0 feasibility bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        RiskControlFeasibilityError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"Gate F0 feasibility audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
