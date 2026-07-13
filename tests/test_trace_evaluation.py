from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from tools.evaluate_trace import main as evaluate_trace_main
from utils.trace_evaluation import (
    InfeasibleFABudgetError,
    TooManyUniqueScoresError,
    TraceOperatingPoint,
    build_dev_threshold_candidates,
    evaluate_operating_point,
    evaluate_trace_protocol,
    select_dev_operating_points,
)


def _maps() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dev_scores = np.full((1, 8, 8), -2.0, dtype=np.float64)
    dev_targets = np.zeros((1, 8, 8), dtype=np.uint8)
    dev_targets[0, 2, 2] = 1
    dev_scores[0, 2, 2] = 2.0
    dev_scores[0, 6, 6] = 1.0

    test_scores = np.full((1, 8, 8), -2.0, dtype=np.float64)
    test_targets = np.zeros((1, 8, 8), dtype=np.uint8)
    test_targets[0, 3, 3] = 1
    test_scores[0, 3, 3] = 2.0
    return dev_scores, dev_targets, test_scores, test_targets


def test_test_scores_cannot_change_dev_selected_threshold() -> None:
    dev_scores, dev_targets, test_scores, test_targets = _maps()
    first = evaluate_trace_protocol(
        dev_scores,
        dev_targets,
        test_scores,
        test_targets,
        fa_budgets_per_million_pixels=(20_000.0,),
        max_unique_scores=10,
    )

    changed_test_scores = np.linspace(-100.0, 100.0, 64).reshape(1, 8, 8)
    second = evaluate_trace_protocol(
        dev_scores,
        dev_targets,
        changed_test_scores,
        test_targets,
        fa_budgets_per_million_pixels=(20_000.0,),
        max_unique_scores=10,
    )

    assert first["operating_points"][0]["locked_threshold"] == second[
        "operating_points"
    ][0]["locked_threshold"]
    assert first["selection_provenance"]["candidate_sha256"] == second[
        "selection_provenance"
    ]["candidate_sha256"]
    assert first["selection_provenance"]["test_used_for_threshold_selection"] is False


def test_all_matchers_and_splits_share_the_dev_locked_threshold() -> None:
    dev_scores, dev_targets, test_scores, test_targets = _maps()
    report = evaluate_trace_protocol(
        dev_scores,
        dev_targets,
        test_scores,
        test_targets,
        fa_budgets_per_million_pixels=(20_000.0,),
        max_unique_scores=10,
    )
    point = report["operating_points"][0]
    locked = point["locked_threshold"]

    assert point["selected_on"] == "dev"
    assert point["selected_with_matcher"] == "legacy"
    assert {
        point[split][matcher]["threshold"]
        for split in ("dev", "test")
        for matcher in ("legacy", "hungarian")
    } == {locked}


def test_strict_greater_than_excludes_score_equal_to_threshold() -> None:
    scores = np.asarray([[0.5, 0.4], [0.0, 0.0]])
    target = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)

    equal = evaluate_operating_point(
        scores, target, 0.5, matching="legacy"
    )
    below = evaluate_operating_point(
        scores, target, np.nextafter(0.5, -np.inf), matching="legacy"
    )

    assert equal.prediction_components == 0
    assert equal.matched_components == 0
    assert below.prediction_components == 1
    assert below.matched_components == 1


def test_too_many_exact_unique_dev_scores_fails_without_quantization() -> None:
    scores = np.arange(6, dtype=np.float64).reshape(2, 3)
    with pytest.raises(TooManyUniqueScoresError, match="without quantization"):
        build_dev_threshold_candidates([scores], max_unique_scores=5)


def _point(
    threshold: float,
    *,
    false_area: int,
    matched: int,
) -> TraceOperatingPoint:
    return TraceOperatingPoint(
        threshold=threshold,
        matching="legacy",
        sample_count=1,
        total_pixels=100,
        target_components=1,
        prediction_components=1,
        matched_components=matched,
        missed_target_components=1 - matched,
        false_component_count=int(false_area > 0),
        false_component_area_pixels=false_area,
        pd=float(matched),
        achieved_fa_per_million_pixels=false_area * 10_000.0,
        global_foreground_iou=0.0,
        per_image_niou=0.0,
        foreground_intersection_pixels=0,
        foreground_union_pixels=1,
        empty_gt_image_count=0,
        empty_gt_and_empty_prediction_count=0,
        empty_gt_with_prediction_count=0,
    )


def test_infeasible_budget_fails_closed() -> None:
    curve = (_point(0.0, false_area=1, matched=1),)
    with pytest.raises(InfeasibleFABudgetError, match="no exact development threshold"):
        select_dev_operating_points(curve, (1.0,))

    empty_target_curve = (replace(curve[0], target_components=0, pd=None),)
    with pytest.raises(InfeasibleFABudgetError, match="no target component"):
        select_dev_operating_points(empty_target_curve, (1.0,))


def test_empty_gt_niou_policy_and_false_component_statistics_are_explicit() -> None:
    scores = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    targets = np.zeros_like(scores, dtype=np.uint8)
    point = evaluate_operating_point(
        scores, targets, 0.5, matching="legacy"
    )

    assert point.per_image_niou == pytest.approx(0.5)
    assert point.global_foreground_iou == 0.0
    assert point.empty_gt_and_empty_prediction_count == 1
    assert point.empty_gt_with_prediction_count == 1
    assert point.false_component_count == 1
    assert point.false_component_area_pixels == 1


def test_cli_requires_separate_bundles_and_writes_hashes(tmp_path) -> None:
    dev_scores, dev_targets, test_scores, test_targets = _maps()
    dev_path = tmp_path / "dev.npz"
    test_path = tmp_path / "test.npz"
    output_path = tmp_path / "report.json"
    np.savez(dev_path, scores=dev_scores, targets=dev_targets)
    np.savez(test_path, scores=test_scores, targets=test_targets)

    exit_code = evaluate_trace_main(
        [
            "--dev-bundle",
            str(dev_path),
            "--test-bundle",
            str(test_path),
            "--output",
            str(output_path),
            "--fa-budgets",
            "20000",
            "--max-unique-scores",
            "10",
        ]
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["inputs"]["dev"]["file_sha256"]
    assert report["inputs"]["dev"]["content_sha256"]
    assert report["inputs"]["test"]["file_sha256"]
    assert report["selection_provenance"]["selection_split"] == "dev"
