from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.analyze_clean_baseline_failures import (
    AnalysisError,
    DATASET_NAMES,
    EXPECTED_AUDIT_OFFICIAL_TEST_STATUS,
    EXPECTED_EVALUATION_SCOPE,
    EXPECTED_OFFICIAL_TEST_STATUS,
    OUTPUT_JSON,
    OUTPUT_MARKDOWN,
    analyze_failures,
    auroc,
    build_markdown,
    clustered_bootstrap_ci,
    load_validated_evidence,
    spearman_correlation,
    write_outputs,
)


SEEDS = (20260711, 20260712, 20260713)
PIXELS = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")


def component(
    image_id: str,
    domain: str,
    role: str,
    component_id: int,
    area: int,
    conflict_pixels: int,
    *,
    recoverable: bool | None = None,
) -> dict:
    row = {
        "image_id": image_id,
        "domain": domain,
        "role": role,
        "component_id": component_id,
        "component_index": component_id - 1,
        "area": area,
        "conflict_pixels": conflict_pixels,
        "conflict_fraction": conflict_pixels / area,
        "p_z_mean": 0.2,
        "j_z_mean": -0.1,
        "interaction_ratio_mean": 1.5,
        "interaction_ratio_p95": 2.0,
        "mean_anchor_score_mean": 0.2 if conflict_pixels else 0.01,
        "prediction_logit_mean": 0.5,
    }
    if recoverable is not None:
        row["recoverable"] = recoverable
    return row


def fixture_cases() -> list[tuple[dict, list[dict]]]:
    cases: list[tuple[dict, list[dict]]] = []

    def image(
        image_id: str,
        *,
        inter: int,
        fp: int,
        fn: int,
        target: int,
        tp_components: int,
        fn_components: int,
        prediction: int,
        matched_prediction: int,
        fp_components: int,
        fp_area: int,
        recoverable_fn: int,
        recoverable_area: int,
        candidates: int,
        conflict: int,
        conflict_tp: int,
        conflict_fp: int,
        conflict_fn: int,
        score: float,
        score_tp: float,
        score_fp: float,
        score_fn: float,
    ) -> dict:
        union = inter + fp + fn
        return {
            "image_index": len(cases),
            "image_id": image_id,
            "intersection_pixels": inter,
            "union_pixels": union,
            "ground_truth_positive_pixels": inter + fn,
            "predicted_positive_pixels": inter + fp,
            "false_positive_pixels": fp,
            "false_negative_pixels": fn,
            "target_component_count": target,
            "true_positive_component_count": tp_components,
            "false_negative_component_count": fn_components,
            "prediction_component_count": prediction,
            "matched_prediction_component_count": matched_prediction,
            "false_positive_component_count": fp_components,
            "false_positive_component_area": fp_area,
            "recoverable_fn_component_count": recoverable_fn,
            "recoverable_fn_target_component_area": recoverable_area,
            "candidate_component_count": candidates,
            "conflict_pixels": conflict,
            "conflict_on_true_positive_pixels": conflict_tp,
            "conflict_on_false_positive_pixels": conflict_fp,
            "conflict_on_false_negative_pixels": conflict_fn,
            "iou": inter / max(1, union),
            "mean_anchor_index": score / PIXELS,
            "interaction_ratio_mean": 1.0 + score,
            "interaction_ratio_p95": 2.0 + score,
            "conflict_fraction": conflict / PIXELS,
            "mean_anchor_score_sum_true_positive": score_tp,
            "mean_anchor_score_sum_false_positive": score_fp,
            "mean_anchor_score_sum_false_negative": score_fn,
        }

    perfect = image(
        "perfect",
        inter=4,
        fp=0,
        fn=0,
        target=1,
        tp_components=1,
        fn_components=0,
        prediction=1,
        matched_prediction=1,
        fp_components=0,
        fp_area=0,
        recoverable_fn=0,
        recoverable_area=0,
        candidates=0,
        conflict=1,
        conflict_tp=1,
        conflict_fp=0,
        conflict_fn=0,
        score=0.16,
        score_tp=0.10,
        score_fp=0,
        score_fn=0,
    )
    cases.append(
        (
            perfect,
            [
                component("perfect", "target", "tp_target", 1, 4, 1, recoverable=False),
                component("perfect", "prediction", "matched_pred", 1, 4, 1),
            ],
        )
    )

    localization = image(
        "localization",
        inter=3,
        fp=1,
        fn=1,
        target=1,
        tp_components=1,
        fn_components=0,
        prediction=1,
        matched_prediction=1,
        fp_components=0,
        fp_area=0,
        recoverable_fn=0,
        recoverable_area=0,
        candidates=0,
        conflict=3,
        conflict_tp=1,
        conflict_fp=1,
        conflict_fn=1,
        score=0.48,
        score_tp=0.10,
        score_fp=0.10,
        score_fn=0.10,
    )
    cases.append(
        (
            localization,
            [
                component("localization", "target", "tp_target", 1, 4, 1, recoverable=False),
                component("localization", "prediction", "matched_pred", 1, 4, 1),
            ],
        )
    )

    fp_only = image(
        "fp_only",
        inter=2,
        fp=3,
        fn=0,
        target=1,
        tp_components=1,
        fn_components=0,
        prediction=2,
        matched_prediction=1,
        fp_components=1,
        fp_area=3,
        recoverable_fn=0,
        recoverable_area=0,
        candidates=0,
        conflict=3,
        conflict_tp=1,
        conflict_fp=2,
        conflict_fn=0,
        score=1.28,
        score_tp=0.10,
        score_fp=0.80,
        score_fn=0,
    )
    cases.append(
        (
            fp_only,
            [
                component("fp_only", "target", "tp_target", 1, 2, 0, recoverable=False),
                component("fp_only", "prediction", "matched_pred", 1, 2, 0),
                component("fp_only", "prediction", "fp_pred", 2, 3, 2),
            ],
        )
    )

    fn_only = image(
        "fn_only",
        inter=0,
        fp=0,
        fn=4,
        target=1,
        tp_components=0,
        fn_components=1,
        prediction=0,
        matched_prediction=0,
        fp_components=0,
        fp_area=0,
        recoverable_fn=1,
        recoverable_area=4,
        candidates=1,
        conflict=3,
        conflict_tp=0,
        conflict_fp=0,
        conflict_fn=2,
        score=1.60,
        score_tp=0,
        score_fp=0,
        score_fn=1.00,
    )
    cases.append(
        (
            fn_only,
            [
                component("fn_only", "target", "fn_target", 1, 4, 2, recoverable=True),
                component("fn_only", "candidate", "candidate", 1, 1, 1),
            ],
        )
    )

    mixed = image(
        "mixed",
        inter=2,
        fp=2,
        fn=2,
        target=2,
        tp_components=1,
        fn_components=1,
        prediction=2,
        matched_prediction=1,
        fp_components=1,
        fp_area=2,
        recoverable_fn=1,
        recoverable_area=2,
        candidates=1,
        conflict=4,
        conflict_tp=1,
        conflict_fp=2,
        conflict_fn=1,
        score=1.92,
        score_tp=0.10,
        score_fp=0.80,
        score_fn=0.60,
    )
    cases.append(
        (
            mixed,
            [
                component("mixed", "target", "tp_target", 1, 2, 0, recoverable=False),
                component("mixed", "target", "fn_target", 2, 2, 1, recoverable=True),
                component("mixed", "prediction", "matched_pred", 1, 2, 0),
                component("mixed", "prediction", "fp_pred", 2, 2, 2),
                component("mixed", "candidate", "candidate", 1, 1, 1),
            ],
        )
    )
    return cases


def make_finalized_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "mechanism_audits"
    runs = []
    cases = fixture_cases()
    for dataset in DATASET_NAMES:
        for seed in SEEDS:
            audit_dir = root / "artifacts" / dataset / f"seed_{seed}"
            images_path = audit_dir / "images.jsonl"
            components_path = audit_dir / "components.jsonl"
            images_path.parent.mkdir(parents=True, exist_ok=True)
            image_rows = [dict(row) for row, _ in cases]
            component_rows = [dict(row) for _, rows in cases for row in rows]
            images_path.write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in image_rows),
                encoding="utf-8",
            )
            components_path.write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in component_rows),
                encoding="utf-8",
            )
            manifest = {
                "schema_version": "dea.clean_mechanism_audit.v1",
                "dataset": dataset,
                "seed": seed,
                "split_role": "val",
                "method": "MSHNet",
                "model_type": "mshnet",
                "anchor_mode": "mean",
                "active_stage": 0,
                "official_test_status": EXPECTED_AUDIT_OFFICIAL_TEST_STATUS,
                "base_size": 4,
                "crop_size": 4,
                "checkpoint": {"role": "best_iou"},
                "summary": {"images": len(image_rows), "pixels": PIXELS * len(image_rows)},
                "artifacts": {
                    "images_jsonl": "images.jsonl",
                    "images_sha256": sha256(images_path),
                    "components_jsonl": "components.jsonl",
                    "components_sha256": sha256(components_path),
                },
            }
            manifest_path = audit_dir / "manifest.json"
            write_json(manifest_path, manifest)
            runs.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "audit_manifest": str(manifest_path.resolve()),
                    "audit_manifest_sha256": sha256(manifest_path),
                    "artifact_counts": {
                        "image_rows": len(image_rows),
                        "component_rows": len(component_rows),
                    },
                }
            )
    summary = {
        "schema_version": "dea.clean_mechanism_audit_evidence_summary.v1",
        "batch_id": "fixture",
        "status": "complete_and_validated",
        "evaluation_scope": EXPECTED_EVALUATION_SCOPE,
        "official_test_status": EXPECTED_OFFICIAL_TEST_STATUS,
        "not_for_official_test_or_main_table_claims": True,
        "dea_evaluated": False,
        "dea_gain_claimed": False,
        "causal_mechanism_claimed": False,
        "datasets": list(DATASET_NAMES),
        "seeds": list(SEEDS),
        "runs": runs,
        "interpretation_boundary": {
            "descriptive_baseline_evidence_only": True,
            "does_not_establish_error_causation": True,
            "does_not_establish_dea_benefit": True,
            "does_not_establish_mean_anchor_predictiveness": True,
            "requires_later_paired_dea_and_control_evidence": True,
        },
    }
    summary_path = root / "clean_mechanism_audit_evidence_summary.json"
    write_json(summary_path, summary)
    return summary_path


def test_end_to_end_taxonomy_components_ratio_of_sums_and_outputs(tmp_path: Path) -> None:
    summary_path = make_finalized_fixture(tmp_path)
    analysis = analyze_failures(
        summary_path, bootstrap_replicates=200, bootstrap_seed=17
    )
    assert analysis["input"]["unique_dataset_image_clusters"] == 15
    assert analysis["input"]["seed_image_observations"] == 45
    metrics = analysis["overall"]["baseline_metrics"]
    # One dataset/seed has sum(intersection)=11 and sum(union)=24.  This differs
    # from the unweighted mean of the five per-image IoUs.
    assert metrics["global_iou_ratio_of_sums"]["estimate"] == pytest.approx(11 / 24)
    assert metrics["global_iou_ratio_of_sums"]["estimate"] != pytest.approx(
        (1 + 3 / 5 + 2 / 5 + 0 + 2 / 6) / 5
    )
    taxonomy = analysis["overall"]["failure_taxonomy"]["categories"]
    assert {name: item["seed_image_observations"] for name, item in taxonomy.items()} == {
        "pixel_perfect": 9,
        "matched_component_localization": 9,
        "fp_only": 9,
        "fn_only": 9,
        "mixed_fp_fn": 9,
    }
    components = analysis["overall"]["component_statistics"]
    assert components["fp_prediction"]["component_observations_across_seeds"] == 18
    assert components["fn_target"]["component_observations_across_seeds"] == 18
    assert components["recoverable_fn_target"]["component_observations_across_seeds"] == 18
    assert components["tp_target"]["component_observations_across_seeds"] == 36
    risk = analysis["overall"]["baseline_risk_predictor"]
    assert risk["predictor"]["uses_ground_truth"] is False
    assert risk["auroc_primary_majority_component_error"]["defined"] is True
    assert "cannot estimate or claim future dea benefit" in risk["interpretation"].lower()
    assert analysis["evidence_decision"]["dea_model_or_gain_gate"]["status"] == (
        "NOT_EVALUATED_BASELINE_ONLY"
    )
    gate_ids = {
        item["id"]
        for item in analysis["evidence_decision"]["baseline_problem_gate"]["criteria"]
    }
    assert "fp_component_conflict_enrichment_overall_ci" in gate_ids
    assert "conflict_enrichment_overall_ci" not in gate_ids
    markdown = build_markdown(analysis)
    assert "no DEA evaluation or gain" in markdown
    json_path, md_path = write_outputs(analysis, tmp_path / "output")
    assert json_path.name == OUTPUT_JSON and md_path.name == OUTPUT_MARKDOWN
    assert json.loads(json_path.read_text())["schema_version"] == analysis["schema_version"]
    with pytest.raises(FileExistsError):
        write_outputs(analysis, tmp_path / "output")


def test_clustered_bootstrap_is_deterministic_and_keeps_seed_rows_together() -> None:
    records = [
        {"dataset": "A", "image": "x", "seed": 1, "value": 0.0},
        {"dataset": "A", "image": "x", "seed": 2, "value": 10.0},
        {"dataset": "A", "image": "y", "seed": 1, "value": 20.0},
        {"dataset": "A", "image": "y", "seed": 2, "value": 30.0},
    ]

    def statistic(rows):
        return sum(row["value"] for row in rows) / len(rows)

    first = clustered_bootstrap_ci(
        records,
        statistic,
        cluster_key=("dataset", "image"),
        strata_key="dataset",
        replicates=100,
        seed=9,
    )
    second = clustered_bootstrap_ci(
        records,
        statistic,
        cluster_key=("dataset", "image"),
        strata_key="dataset",
        replicates=100,
        seed=9,
    )
    assert first == second
    assert first["bootstrap"]["cluster_count"] == 2
    assert first["bootstrap"]["observation_count"] == 4
    # Whole-cluster resamples can only have means 5, 15, or 25; an independent
    # row bootstrap would admit many additional means.
    assert first["ci95"]["low"] == pytest.approx(5.0)
    assert first["ci95"]["high"] == pytest.approx(25.0)


def test_auroc_and_spearman_report_undefined_cases_explicitly() -> None:
    assert auroc([0.1, 0.2, 0.3], [1, 1, 1]) == {
        "defined": False,
        "estimate": None,
        "undefined_reason": "single_class_outcome",
        "positive_image_clusters": 3,
        "negative_image_clusters": 0,
    }
    assert spearman_correlation([1, 1, 1], [0, 1, 2])["undefined_reason"] == (
        "constant_predictor"
    )
    assert spearman_correlation([0, 1, 2], [1, 1, 1])["undefined_reason"] == (
        "constant_outcome"
    )
    assert auroc([0, 0, 1, 1], [0, 1, 0, 1])["estimate"] == pytest.approx(0.5)


def test_rejects_nonvalidated_or_official_test_scope(tmp_path: Path) -> None:
    summary_path = make_finalized_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary["evaluation_scope"] = "official test"
    write_json(summary_path, summary)
    with pytest.raises(AnalysisError, match="development-only"):
        load_validated_evidence(summary_path)


def test_rejects_seed_ledgers_with_different_image_sets(tmp_path: Path) -> None:
    summary_path = make_finalized_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    run = next(
        run
        for run in summary["runs"]
        if run["dataset"] == DATASET_NAMES[0] and run["seed"] == SEEDS[-1]
    )
    manifest_path = Path(run["audit_manifest"])
    manifest = json.loads(manifest_path.read_text())
    images_path = manifest_path.parent / manifest["artifacts"]["images_jsonl"]
    rows = [json.loads(line) for line in images_path.read_text().splitlines()]
    rows[-1]["image_id"] = "different_image"
    images_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    components_path = manifest_path.parent / manifest["artifacts"]["components_jsonl"]
    component_rows = [json.loads(line) for line in components_path.read_text().splitlines()]
    for row in component_rows:
        if row["image_id"] == "mixed":
            row["image_id"] = "different_image"
    components_path.write_text(
        "".join(json.dumps(row) + "\n" for row in component_rows), encoding="utf-8"
    )
    manifest["artifacts"]["images_sha256"] = sha256(images_path)
    manifest["artifacts"]["components_sha256"] = sha256(components_path)
    write_json(manifest_path, manifest)
    run["audit_manifest_sha256"] = sha256(manifest_path)
    write_json(summary_path, summary)
    with pytest.raises(AnalysisError, match="identical validation images"):
        load_validated_evidence(summary_path)
