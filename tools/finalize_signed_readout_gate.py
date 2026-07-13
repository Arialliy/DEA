#!/usr/bin/env python3
"""Finalize the preregistered nine-job Gate-K signed-readout diagnostic.

This gate decides only whether a fixed signed local-reference coordinate is a
usable post-d0 primitive.  It does not authorize a paper-method claim and it
does not select hyperparameters.  The official test split must remain sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_signed_readout_probe import (  # noqa: E402
    ALL_VARIANTS,
    BUDGETS,
    MATCHERS,
    SCHEMA as JOB_SCHEMA,
)


SCHEMA = "dea.gate_k.signed_readout_final.v1"
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
SEEDS = (20260711, 20260712, 20260713)
SIGNED = "refit_signed_standardized"
COMPARATORS = (
    "original_final_z",
    "original_output0",
    "refit_raw",
    "refit_unsigned_standardized_projection",
)
MAX_PIXEL_IOU_DROP = 0.005
MIN_PASSING_SEEDS_PER_DATASET = 2
MIN_PASSING_DATASETS = 2
ADJACENT_BUDGET_PAIRS = tuple(zip(BUDGETS[:-1], BUDGETS[1:]))


class SignedReadoutFinalizationError(RuntimeError):
    """Raised when a Gate-K job or aggregate violates the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SignedReadoutFinalizationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SignedReadoutFinalizationError(f"JSON is not an object: {path}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SignedReadoutFinalizationError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SignedReadoutFinalizationError(f"{label} is not finite")
    return result


def _job_dir(root: Path, dataset: str, seed: int) -> Path:
    return root / dataset / f"seed_{seed}"


def _validate_point(
    summary: Mapping[str, Any],
    *,
    variant: str,
    matcher: str,
    budget: int,
) -> dict[str, Any]:
    try:
        point = summary["variants"][variant]["crossfit_q2"][matcher][str(budget)]
        pixel = summary["variants"][variant]["crossfit_pixel"][matcher][str(budget)]
    except (KeyError, TypeError) as exc:
        raise SignedReadoutFinalizationError(
            f"missing operating point {variant}/{matcher}/FA{budget}"
        ) from exc
    if not isinstance(point, Mapping) or not isinstance(pixel, Mapping):
        raise SignedReadoutFinalizationError("operating point has invalid schema")
    target_components = int(point.get("target_components", -1))
    matched_components = int(point.get("matched_components", -1))
    if target_components <= 0 or not 0 <= matched_components <= target_components:
        raise SignedReadoutFinalizationError("component counts are invalid")
    achieved_pd = _finite_number(point.get("achieved_pd"), "achieved_pd")
    expected_pd = matched_components / target_components
    if abs(achieved_pd - expected_pd) > 1e-12:
        raise SignedReadoutFinalizationError("achieved Pd disagrees with counts")
    achieved_fa = _finite_number(
        point.get("achieved_fa_per_mpix"), "achieved_fa_per_mpix"
    )
    pooled_feasible = bool(point.get("budget_feasible_zero_overshoot"))
    folds_feasible = bool(point.get("all_held_out_folds_feasible"))
    pixel_iou = _finite_number(pixel.get("iou"), "crossfit pixel IoU")
    if not 0.0 <= pixel_iou <= 1.0:
        raise SignedReadoutFinalizationError("pixel IoU is outside [0,1]")
    if pixel.get("strict_prediction_rule") != "logit > threshold":
        raise SignedReadoutFinalizationError("pixel threshold semantics drifted")
    if achieved_fa < 0.0:
        raise SignedReadoutFinalizationError("achieved FA is negative")
    return {
        "target_components": target_components,
        "matched_components": matched_components,
        "achieved_pd": achieved_pd,
        "achieved_fa_per_mpix": achieved_fa,
        "pooled_budget_feasible": pooled_feasible,
        "all_held_out_folds_feasible": folds_feasible,
        "pixel_iou": pixel_iou,
    }


def load_job(root: Path, dataset: str, seed: int) -> dict[str, Any]:
    job_dir = _job_dir(root, dataset, seed)
    summary_path = job_dir / "summary.json"
    provenance_path = job_dir / "provenance.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        raise SignedReadoutFinalizationError(f"missing plain summary: {summary_path}")
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise SignedReadoutFinalizationError(
            f"missing plain provenance: {provenance_path}"
        )
    summary = _read_object(summary_path)
    provenance = _read_object(provenance_path)
    if (
        summary.get("schema") != JOB_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("protocol") != "formal"
        or summary.get("dataset") != dataset
        or int(summary.get("seed", -1)) != seed
    ):
        raise SignedReadoutFinalizationError(
            f"job identity/protocol drifted for {dataset}/{seed}"
        )
    if tuple(summary.get("variants", {})) != ALL_VARIANTS:
        raise SignedReadoutFinalizationError("variant inventory/order drifted")
    boundary = summary.get("scientific_boundary")
    if not isinstance(boundary, Mapping) or not all(
        boundary.get(key) is True
        for key in (
            "diagnostic_only",
            "same_development_q2_oracle_is_not_deployable_performance",
            "crossfit_is_internal_development_only",
            "does_not_establish_a_paper_method",
        )
    ):
        raise SignedReadoutFinalizationError("scientific boundary is missing")
    if provenance.get("dataset") != dataset or int(provenance.get("seed", -1)) != seed:
        raise SignedReadoutFinalizationError("provenance identity drifted")
    if provenance.get("protocol") != "formal":
        raise SignedReadoutFinalizationError("provenance protocol drifted")
    data_access = provenance.get("data_access")
    if not isinstance(data_access, Mapping) or (
        data_access.get("official_test_dataset_constructed") is not False
        or data_access.get("official_test_sample_iterated") is not False
    ):
        raise SignedReadoutFinalizationError("official test was not sealed")
    freeze = provenance.get("freeze_audit")
    if not isinstance(freeze, Mapping):
        raise SignedReadoutFinalizationError("freeze audit is missing")
    if not (
        freeze.get("model_eval_for_all_fit_and_dev_forwards") is True
        and freeze.get("model_requires_grad_false") is True
        and freeze.get("d0_extracted_under_no_grad") is True
        and freeze.get("shared_d0_once_per_fit_batch") is True
        and freeze.get("backbone_state_sha256_before")
        == freeze.get("backbone_state_sha256_after_training")
        == freeze.get("backbone_state_sha256_after_inference")
        and freeze.get("batchnorm_state_sha256_before")
        == freeze.get("batchnorm_state_sha256_after_training")
        == freeze.get("batchnorm_state_sha256_after_inference")
    ):
        raise SignedReadoutFinalizationError("frozen d0/backbone invariant failed")
    artifact_hashes = provenance.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping) or artifact_hashes.get(
        "summary.json"
    ) != _sha256(summary_path):
        raise SignedReadoutFinalizationError("summary artifact hash drifted")
    source_hashes = provenance.get("source_sha256")
    if (
        provenance.get("source_specific_hashes_unchanged") is not True
        or not isinstance(source_hashes, Mapping)
        or not source_hashes
    ):
        raise SignedReadoutFinalizationError("source freeze record is invalid")
    return {
        "summary": summary,
        "provenance": provenance,
        "summary_sha256": _sha256(summary_path),
        "source_sha256": dict(source_hashes),
    }


def finalize(input_root: Path) -> dict[str, Any]:
    root = input_root.resolve()
    jobs: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            jobs[(dataset, seed)] = load_job(root, dataset, seed)
    source_records = [job["source_sha256"] for job in jobs.values()]
    if any(record != source_records[0] for record in source_records[1:]):
        raise SignedReadoutFinalizationError("formal jobs used different source hashes")

    seed_records: dict[str, dict[str, Any]] = {}
    dataset_budget_records: dict[str, dict[str, Any]] = {
        dataset: {} for dataset in DATASETS
    }
    for dataset in DATASETS:
        for seed in SEEDS:
            summary = jobs[(dataset, seed)]["summary"]
            seed_key = f"{dataset}/seed_{seed}"
            seed_records[seed_key] = {}
            for budget in BUDGETS:
                matcher_records: dict[str, Any] = {}
                for matcher in MATCHERS:
                    signed = _validate_point(
                        summary,
                        variant=SIGNED,
                        matcher=matcher,
                        budget=budget,
                    )
                    comparators = {
                        variant: _validate_point(
                            summary,
                            variant=variant,
                            matcher=matcher,
                            budget=budget,
                        )
                        for variant in COMPARATORS
                    }
                    dominance = {
                        variant: signed["matched_components"]
                        > point["matched_components"]
                        for variant, point in comparators.items()
                    }
                    native_iou = comparators["original_final_z"]["pixel_iou"]
                    pixel_delta = signed["pixel_iou"] - native_iou
                    passed = (
                        signed["pooled_budget_feasible"]
                        and signed["all_held_out_folds_feasible"]
                        and all(dominance.values())
                        and pixel_delta >= -MAX_PIXEL_IOU_DROP - 1e-12
                    )
                    matcher_records[matcher] = {
                        "signed": signed,
                        "comparators": comparators,
                        "strict_matched_component_dominance": dominance,
                        "signed_minus_native_pixel_iou": pixel_delta,
                        "pass": passed,
                    }
                seed_records[seed_key][str(budget)] = {
                    "by_matcher": matcher_records,
                    "both_matchers_pass": all(
                        matcher_records[matcher]["pass"] for matcher in MATCHERS
                    ),
                }
        for budget in BUDGETS:
            passing_seeds = [
                seed
                for seed in SEEDS
                if seed_records[f"{dataset}/seed_{seed}"][str(budget)][
                    "both_matchers_pass"
                ]
            ]
            dataset_budget_records[dataset][str(budget)] = {
                "passing_seeds": passing_seeds,
                "passing_seed_count": len(passing_seeds),
                "pass": len(passing_seeds) >= MIN_PASSING_SEEDS_PER_DATASET,
            }

    adjacent_records: dict[str, Any] = {}
    for lower, upper in ADJACENT_BUDGET_PAIRS:
        passing_datasets = [
            dataset
            for dataset in DATASETS
            if dataset_budget_records[dataset][str(lower)]["pass"]
            and dataset_budget_records[dataset][str(upper)]["pass"]
        ]
        adjacent_records[f"{lower}-{upper}"] = {
            "budgets": [lower, upper],
            "passing_datasets": passing_datasets,
            "passing_dataset_count": len(passing_datasets),
            "pass": len(passing_datasets) >= MIN_PASSING_DATASETS,
        }
    passing_pairs = [
        key for key, record in adjacent_records.items() if bool(record["pass"])
    ]
    passed = bool(passing_pairs)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "gate": "pass" if passed else "no_go",
        "pass": passed,
        "scope": (
            "internal fit/development signed-readout diagnostic only; official "
            "test sealed; not a deployable performance or paper-method claim"
        ),
        "preregistered_rule": {
            "signed_variant": SIGNED,
            "comparators": list(COMPARATORS),
            "strict_component_count_dominance": True,
            "signed_zero_overshoot_required": True,
            "matchers_must_both_pass": list(MATCHERS),
            "maximum_paired_native_pixel_iou_drop": MAX_PIXEL_IOU_DROP,
            "minimum_passing_seeds_per_dataset": MIN_PASSING_SEEDS_PER_DATASET,
            "minimum_passing_datasets": MIN_PASSING_DATASETS,
            "adjacent_budget_pair_required": True,
            "budgets": list(BUDGETS),
        },
        "input_root": str(root),
        "job_inventory": {
            f"{dataset}/seed_{seed}": {
                "summary_sha256": jobs[(dataset, seed)]["summary_sha256"]
            }
            for dataset in DATASETS
            for seed in SEEDS
        },
        "source_sha256": source_records[0],
        "by_seed": seed_records,
        "by_dataset_budget": dataset_budget_records,
        "adjacent_budget_pairs": adjacent_records,
        "passing_adjacent_budget_pairs": passing_pairs,
        "authorization": {
            "freeze_signed_coordinate": passed,
            "implement_component_prediction_unit": passed,
            "claim_signed_coordinate_as_paper_innovation": False,
            "add_parallel_pixel_or_refinement_modules": False,
        },
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gate K signed-readout finalization",
        "",
        "> Internal fit/development diagnostic only; official test sealed; not a paper-method claim.",
        "",
        f"- Decision: **{summary['gate']}**",
        "- Signed coordinate may be frozen: "
        + str(summary["authorization"]["freeze_signed_coordinate"]),
        "- Component prediction implementation authorized: "
        + str(summary["authorization"]["implement_component_prediction_unit"]),
        "- Signed coordinate is a paper innovation: False",
        "",
        "| Dataset | FA1 | FA5 | FA10 | FA20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        values = [
            summary["by_dataset_budget"][dataset][str(budget)] for budget in BUDGETS
        ]
        lines.append(
            "| "
            + dataset
            + " | "
            + " | ".join(
                f"{value['passing_seed_count']}/3 ({'pass' if value['pass'] else 'fail'})"
                for value in values
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Adjacent-budget stability",
            "",
            "| Pair | Passing datasets | Pass |",
            "|---|---|---:|",
        ]
    )
    for key, record in summary["adjacent_budget_pairs"].items():
        lines.append(
            f"| {key} | {', '.join(record['passing_datasets']) or 'none'} | {record['pass']} |"
        )
    lines.extend(
        [
            "",
            "The gate compares matched-component counts at the same nominal FA budget, requires both matchers and both held-out folds to be feasible, and forbids more than 0.005 paired pixel-IoU loss relative to native final z.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = finalize(Path(args.input_root).expanduser())
    output.mkdir(parents=True)
    _atomic_write(
        output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )
    _atomic_write(output / "summary.md", render_markdown(summary))
    print(json.dumps({"gate": summary["gate"], "output_dir": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
