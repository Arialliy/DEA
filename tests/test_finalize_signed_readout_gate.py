import hashlib
import json
from pathlib import Path

import pytest

from tools.finalize_signed_readout_gate import (
    BUDGETS,
    DATASETS,
    SEEDS,
    SignedReadoutFinalizationError,
    finalize,
    render_markdown,
)
from tools.run_signed_readout_probe import ALL_VARIANTS, MATCHERS, SCHEMA


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _point(matched: int, *, feasible: bool = True) -> dict:
    return {
        "target_components": 10,
        "matched_components": matched,
        "prediction_components": matched,
        "unmatched_prediction_components": 0,
        "unmatched_prediction_area": 0,
        "total_pixels": 1_000_000,
        "achieved_pd": matched / 10,
        "achieved_fa_per_mpix": 0.0,
        "budget_feasible_zero_overshoot": feasible,
        "all_held_out_folds_feasible": feasible,
    }


def _variant(matched: int, pixel_iou: float, *, feasible: bool = True) -> dict:
    return {
        "crossfit_q2": {
            matcher: {
                str(budget): _point(matched, feasible=feasible)
                for budget in BUDGETS
            }
            for matcher in MATCHERS
        },
        "crossfit_pixel": {
            matcher: {
                str(budget): {
                    "intersection_pixels": 10,
                    "union_pixels": 20,
                    "prediction_pixels": 15,
                    "target_pixels": 15,
                    "iou": pixel_iou,
                    "strict_prediction_rule": "logit > threshold",
                }
                for budget in BUDGETS
            }
            for matcher in MATCHERS
        },
    }


def _write_job(
    root: Path,
    dataset: str,
    seed: int,
    *,
    signed_matched: int = 8,
    signed_iou: float = 0.699,
    signed_feasible: bool = True,
    source_tag: str = "frozen",
    official_test_accessed: bool = False,
) -> None:
    directory = root / dataset / f"seed_{seed}"
    directory.mkdir(parents=True)
    matched = {
        "original_final_z": 5,
        "original_output0": 4,
        "refit_raw": 6,
        "refit_annulus_centered": 7,
        "refit_signed_standardized": signed_matched,
        "refit_unsigned_standardized_projection": 7,
    }
    pixel_iou = {name: 0.68 for name in ALL_VARIANTS}
    pixel_iou["original_final_z"] = 0.70
    pixel_iou["refit_signed_standardized"] = signed_iou
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "protocol": "formal",
        "dataset": dataset,
        "seed": seed,
        "variants": {
            name: _variant(
                matched[name],
                pixel_iou[name],
                feasible=(
                    signed_feasible
                    if name == "refit_signed_standardized"
                    else True
                ),
            )
            for name in ALL_VARIANTS
        },
        "scientific_boundary": {
            "diagnostic_only": True,
            "same_development_q2_oracle_is_not_deployable_performance": True,
            "crossfit_is_internal_development_only": True,
            "does_not_establish_a_paper_method": True,
        },
    }
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    provenance = {
        "dataset": dataset,
        "seed": seed,
        "protocol": "formal",
        "source_sha256": {"tool": hashlib.sha256(source_tag.encode()).hexdigest()},
        "source_specific_hashes_unchanged": True,
        "artifact_sha256": {"summary.json": _sha256(summary_path)},
        "data_access": {
            "official_test_dataset_constructed": official_test_accessed,
            "official_test_sample_iterated": official_test_accessed,
        },
        "freeze_audit": {
            "model_eval_for_all_fit_and_dev_forwards": True,
            "model_requires_grad_false": True,
            "d0_extracted_under_no_grad": True,
            "shared_d0_once_per_fit_batch": True,
            "backbone_state_sha256_before": "a" * 64,
            "backbone_state_sha256_after_training": "a" * 64,
            "backbone_state_sha256_after_inference": "a" * 64,
            "batchnorm_state_sha256_before": "b" * 64,
            "batchnorm_state_sha256_after_training": "b" * 64,
            "batchnorm_state_sha256_after_inference": "b" * 64,
        },
    }
    (directory / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


def _write_matrix(root: Path, **kwargs) -> None:
    for dataset in DATASETS:
        for seed in SEEDS:
            _write_job(root, dataset, seed, **kwargs)


def test_formal_matrix_passes_only_as_diagnostic(tmp_path: Path) -> None:
    _write_matrix(tmp_path)
    result = finalize(tmp_path)
    assert result["pass"] is True
    assert result["gate"] == "pass"
    assert result["passing_adjacent_budget_pairs"] == ["1-5", "5-10", "10-20"]
    assert result["authorization"] == {
        "freeze_signed_coordinate": True,
        "implement_component_prediction_unit": True,
        "claim_signed_coordinate_as_paper_innovation": False,
        "add_parallel_pixel_or_refinement_modules": False,
    }
    markdown = render_markdown(result)
    assert "internal fit/development diagnostic" in markdown.lower()
    assert "paper innovation: False" in markdown


@pytest.mark.parametrize(
    "kwargs",
    (
        {"signed_matched": 7},
        {"signed_feasible": False},
        {"signed_iou": 0.694},
    ),
)
def test_dominance_feasibility_and_pixel_quality_are_each_vetoes(
    tmp_path: Path, kwargs: dict
) -> None:
    _write_matrix(tmp_path, **kwargs)
    result = finalize(tmp_path)
    assert result["pass"] is False
    assert result["gate"] == "no_go"
    assert result["authorization"]["implement_component_prediction_unit"] is False


def test_two_datasets_and_two_seeds_are_required(tmp_path: Path) -> None:
    for dataset in DATASETS:
        for seed in SEEDS:
            passes = dataset == "IRSTD-1K" or seed == SEEDS[0]
            _write_job(
                tmp_path,
                dataset,
                seed,
                signed_matched=8 if passes else 7,
            )
    result = finalize(tmp_path)
    assert result["pass"] is False
    assert all(
        not record["pass"] for record in result["adjacent_budget_pairs"].values()
    )


def test_source_drift_fails_closed(tmp_path: Path) -> None:
    _write_matrix(tmp_path)
    path = tmp_path / DATASETS[-1] / f"seed_{SEEDS[-1]}" / "provenance.json"
    value = json.loads(path.read_text())
    value["source_sha256"] = {"tool": "c" * 64}
    path.write_text(json.dumps(value))
    with pytest.raises(SignedReadoutFinalizationError, match="different source"):
        finalize(tmp_path)


def test_official_test_access_fails_closed(tmp_path: Path) -> None:
    _write_matrix(tmp_path)
    path = tmp_path / DATASETS[0] / f"seed_{SEEDS[0]}" / "provenance.json"
    value = json.loads(path.read_text())
    value["data_access"]["official_test_dataset_constructed"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(SignedReadoutFinalizationError, match="official test"):
        finalize(tmp_path)
