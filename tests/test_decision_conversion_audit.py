from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.audit_mshnet_decision_conversion import (
    _build_cross_fitted_operating_audit,
    _empty_contribution_margins,
    _empty_conversion,
    _normalized_rgb_to_context_luminance,
    _operating_fold,
    _select_paired_matched_controls,
    _target_operating_record,
)


def test_inverse_normalization_recovers_replicated_grayscale() -> None:
    grayscale = torch.linspace(0.0, 1.0, 20).reshape(1, 1, 4, 5)
    rgb = grayscale.repeat(1, 3, 1, 1)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    recovered = _normalized_rgb_to_context_luminance((rgb - mean) / std)

    assert recovered == pytest.approx(grayscale[0, 0].numpy(), abs=1e-6)


def test_matched_target_controls_keep_the_exact_pair_id() -> None:
    misses = [
        {
            "sample_name": "miss",
            "target_index": 2,
            "area": 3,
            "border_distance": 20.0,
        }
    ]
    matched = [
        {
            "sample_name": "control",
            "target_index": 4,
            "area": 3,
            "border_distance": 20.0,
        }
    ]

    selected = _select_paired_matched_controls(
        misses, matched, controls_per_miss=1
    )

    assert selected[0]["paired_no_response_id"] == "miss:2"
    assert selected[0]["pair_index"] == 0


def _two_fold_fixture():
    names_by_fold = {}
    index = 0
    while set(names_by_fold) != {0, 1}:
        name = "sample-%d" % index
        names_by_fold.setdefault(_operating_fold(name), name)
        index += 1
    names = [names_by_fold[0], names_by_fold[1]]
    logits = []
    targets = []
    for _ in names:
        score = np.full((8, 8), -2.0)
        target = np.zeros((8, 8), dtype=bool)
        score[3, 3] = 2.0
        target[3, 3] = True
        logits.append(score)
        targets.append(target)
    return tuple(logits), tuple(targets), names


def test_operating_thresholds_are_cross_fitted_by_image() -> None:
    logits, targets, names = _two_fold_fixture()

    audit, lookup = _build_cross_fitted_operating_audit(
        logits,
        targets,
        names,
        fixed_threshold=0.0,
        budgets=(0.0,),
    )

    assert audit["protocol"] == "deterministic_two_fold_image_disjoint_cross_fit_v1"
    for fold in (0, 1):
        record = audit["folds"][str(fold)]
        assert set(record["evaluation_names"]).isdisjoint(
            record["calibration_names"]
        )
        assert lookup[fold]["0"]["calibration_operating_point"][
            "fa_per_million_pixels"
        ] == 0.0

    target_record = _target_operating_record(
        logits[0],
        targets[0],
        sample_name=names[0],
        target_index=0,
        fixed_threshold=0.0,
        threshold_lookup=lookup,
    )
    assert target_record["fixed_threshold"]["matched"]
    assert target_record["cross_fitted_fixed_fa"]["0"]["status"]["matched"]


def test_unavailable_schemas_never_encode_missing_values_as_zero() -> None:
    conversion = _empty_conversion("missing")
    contribution = _empty_contribution_margins("missing")

    assert not conversion["available"]
    assert conversion["mean_logit_margin"] is None
    assert conversion["utilization_cosine"] is None
    assert not contribution["available"]
    assert contribution["sum"] is None
    assert contribution["has_sign_cancellation"] is None
