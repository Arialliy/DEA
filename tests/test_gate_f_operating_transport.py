from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.analyze_gate_f_operating_transport import (
    GateFTransportError,
    _verify_input_bundle,
    analyze_transport,
    sha256_file,
    write_bundle,
)


def _curve_point(
    *,
    threshold: float,
    area: int,
    matched: int = 1,
    predictions: int | None = None,
) -> dict[str, object]:
    if predictions is None:
        predictions = matched + (1 if area else 0)
    unmatched_predictions = predictions - matched
    return {
        "threshold": threshold,
        "sample_count": 1,
        "total_pixels": 100_000,
        "target_components": 1,
        "matched_components": matched,
        "unmatched_target_components": 1 - matched,
        "prediction_components": predictions,
        "unmatched_prediction_components": unmatched_predictions,
        "unmatched_prediction_area": area,
        "fa_per_million_pixels": area * 10.0,
        "pd": float(matched),
    }


def _calibration_record(
    *,
    evaluation_fold: int,
    selected_threshold: float,
) -> dict[str, object]:
    evaluation_name = f"i{evaluation_fold}"
    calibration_name = f"i{1 - evaluation_fold}"
    infeasible = _curve_point(threshold=0.0, area=5)
    selected = _curve_point(threshold=selected_threshold, area=0)
    return {
        "schema_version": "dea.gate_e.low_fa_calibration.v1",
        "dataset": "D",
        "seed": 1,
        "matcher": "m",
        "evaluation_fold": evaluation_fold,
        "calibration_fold": 1 - evaluation_fold,
        "evaluation_image_names": [evaluation_name],
        "calibration_image_names": [calibration_name],
        "threshold_grid": [0.0, selected_threshold],
        "curve": [infeasible, selected],
        "selections": {"10": copy.deepcopy(selected)},
    }


def _aggregate(
    *,
    area: int,
    pixels: int,
    targets: int,
    matched: int,
    predictions: int,
    budget: int = 10,
) -> dict[str, object]:
    unmatched_predictions = predictions - matched
    return {
        "total_pixels": pixels,
        "target_components": targets,
        "matched_components": matched,
        "prediction_components": predictions,
        "unmatched_prediction_components": unmatched_predictions,
        "unmatched_prediction_area": area,
        "achieved_fa_per_mpix": area * 1_000_000.0 / pixels,
        "achieved_pd": matched / targets,
        "budget_feasible_zero_overshoot": area * 1_000_000 <= budget * pixels,
    }


def _image_row(
    *,
    evaluation_fold: int,
    threshold: float,
    area: int,
) -> dict[str, object]:
    predictions = 1 + (1 if area else 0)
    held = _aggregate(
        area=area,
        pixels=100_000,
        targets=1,
        matched=1,
        predictions=predictions,
    )
    pooled = _aggregate(
        area=3,
        pixels=200_000,
        targets=2,
        matched=2,
        predictions=3,
    )
    return {
        "schema_version": "dea.gate_e.low_fa_image.v1",
        "dataset": "D",
        "seed": 1,
        "matcher": "m",
        "nominal_budget_fa_per_mpix": 10,
        "evaluation_fold": evaluation_fold,
        "calibration_threshold": threshold,
        "image_name": f"i{evaluation_fold}",
        "image_index": evaluation_fold,
        "target_free_image": False,
        "total_pixels": 100_000,
        "target_components": 1,
        "matched_components": 1,
        "prediction_components": predictions,
        "unmatched_prediction_components": predictions - 1,
        "unmatched_prediction_area": area,
        "held_out_fold_aggregate": held,
        "dataset_seed_aggregate": pooled,
    }


def _fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    calibration = [
        _calibration_record(evaluation_fold=0, selected_threshold=1.0),
        _calibration_record(evaluation_fold=1, selected_threshold=2.0),
    ]
    images = [
        _image_row(evaluation_fold=0, threshold=1.0, area=3),
        _image_row(evaluation_fold=1, threshold=2.0, area=0),
    ]
    return calibration, images


def _analyze(
    calibration: list[dict[str, object]], images: list[dict[str, object]]
):
    return analyze_transport(
        calibration,
        images,
        budgets=(10,),
        matchers=("m",),
        fold_count=2,
        fold_mappings={"D": {"i0": 0, "i1": 1}},
    )


def test_transport_recomputes_folds_pooling_and_concentration() -> None:
    calibration, images = _fixture()
    fold_rows, pair_rows, summary = _analyze(calibration, images)

    assert len(fold_rows) == 2
    assert len(pair_rows) == 1
    pair = pair_rows[0]
    assert pair["threshold_transport"]["absolute_span"] == 1.0
    assert pair["held_out_overshooting_fold_count"] == 1
    assert pair["pooled_held_out"]["fa_per_mpix"] == 15.0
    assert pair["pooled_held_out"]["budget_feasible_zero_overshoot"] is False
    concentration = pair["pooled_unmatched_area_concentration"]
    assert concentration["top_k"]["1"]["unmatched_area_share"] == 1.0
    assert concentration["top_k"]["1"]["repairs_original_overshoot"] is True
    assert concentration["top_k"]["1"]["leave_images_out_remaining_fa_per_mpix"] == 0.0
    assert summary["by_budget"]["10"]["pooled_infeasible_count"] == 1


def test_transport_rejects_duplicate_images_and_embedded_aggregate_drift() -> None:
    calibration, images = _fixture()
    with pytest.raises(GateFTransportError, match="duplicate image row"):
        _analyze(calibration, images + [copy.deepcopy(images[0])])

    calibration, images = _fixture()
    images[0]["held_out_fold_aggregate"]["unmatched_prediction_area"] = 2
    with pytest.raises(GateFTransportError, match="held_out_fold_aggregate"):
        _analyze(calibration, images)


def test_transport_rejects_selection_that_violates_frozen_tie_break() -> None:
    calibration, images = _fixture()
    record = calibration[0]
    first = record["curve"][0]
    first["unmatched_prediction_area"] = 0
    first["unmatched_prediction_components"] = 0
    first["prediction_components"] = 1
    first["fa_per_million_pixels"] = 0.0
    record["selections"]["10"] = copy.deepcopy(first)
    images[0]["calibration_threshold"] = 0.0
    with pytest.raises(GateFTransportError, match="violates frozen tie-break"):
        _analyze(calibration, images)


def test_transport_flags_nonmonotone_matching_defined_area() -> None:
    calibration, images = _fixture()
    record = calibration[0]
    extra = _curve_point(threshold=3.0, area=1, matched=0, predictions=1)
    record["threshold_grid"].append(3.0)
    record["curve"].append(extra)
    _, _, summary = _analyze(calibration, images)
    assert summary["calibration_curve_structure"][
        "nonmonotone_unmatched_area_record_count"
    ] == 1


def test_input_bundle_hash_verification_detects_any_registered_artifact_change(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "e1c"
    bundle.mkdir()
    payloads = {
        "calibration.json": "[]\n",
        "image_low_fa.jsonl": "",
        "target_low_fa.jsonl": "",
        "low_fa_bridge_summary.json": json.dumps(
            {"joint_gate": {"pass": False, "passing_budgets": []}}
        )
        + "\n",
        "low_fa_bridge_summary.md": "formal\n",
    }
    for name, payload in payloads.items():
        (bundle / name).write_text(payload, encoding="utf-8")
    provenance = {
        "schema_version": "dea.gate_e.low_fa_bridge_provenance.v1",
        "artifact_sha256": {
            name: sha256_file(bundle / name) for name in payloads
        },
    }
    (bundle / "provenance.json").write_text(
        json.dumps(provenance) + "\n", encoding="utf-8"
    )
    _verify_input_bundle(bundle)
    (bundle / "target_low_fa.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(GateFTransportError, match="hash mismatch"):
        _verify_input_bundle(bundle)


def test_gate_f_bundle_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "gate_f"
    write_bundle(
        output,
        fold_rows=[],
        pair_rows=[],
        summary={
            "analysis_scope": "test",
            "formal_gate_effect": "none",
            "diagnostic_caveat": "test",
            "protocol": {"budgets_fa_per_mpix": []},
            "by_budget": {},
            "by_dataset_budget": {},
        },
        provenance={"schema_version": "test"},
    )
    assert {path.name for path in output.iterdir()} == {
        "fold_transport.jsonl",
        "pair_transport.jsonl",
        "operating_transport_summary.json",
        "operating_transport_summary.md",
        "provenance.json",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_bundle(
            output,
            fold_rows=[],
            pair_rows=[],
            summary={},
            provenance={},
        )
