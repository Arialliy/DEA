from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.export_clean_mechanism_audit import (
    analyse_image_components,
    compute_conflict_maps,
    enrich_component_mechanism_fields,
    project_coalitions,
    validate_checkpoint_metadata,
    validate_recomputed_checkpoint_metrics,
)
from model.dea_scale_interaction_exchange import ScaleInteractionExchangeMSHNet


def test_projected_conflict_formula_cpu() -> None:
    p_z = torch.tensor([[[[-2.0, -2.0, 0.0, -0.75]]]])
    j_z = torch.tensor([[[[4.0, 4.0, 1.0, 2.0]]]])
    zero = torch.zeros_like(p_z)
    current_main = torch.tensor([
        [[[3.0, 3.0, 0.0, 3.0]], [[4.0, 4.0, 0.0, 4.0]]]
    ])
    interaction = torch.tensor([
        [[[6.0, 1.5, 1.0, 6.0]], [[8.0, 2.0, 1.0, 8.0]]]
    ])
    maps = compute_conflict_maps(
        p_z + j_z, p_z, zero, zero, current_main, interaction, eps=1.0
    )
    torch.testing.assert_close(maps["p_z"], p_z)
    torch.testing.assert_close(maps["j_z"], j_z)
    assert maps["r"].flatten()[0] > 1
    assert maps["r"].flatten()[1] < 1
    # Pixel 3 verifies the exact |p_z|+|j_z|>eps support definition: |p_z| alone
    # is below eps, but the preregistered sum is above it.
    assert maps["conflict_mask"].flatten().tolist() == [True, False, False, True]
    assert maps["conflict_score"].flatten()[0] > 0
    assert maps["conflict_score"].flatten()[1] > 0
    assert maps["conflict_score"].flatten()[2] == 0


def test_component_recoverable_fn_and_checkpoint_fail_closed_cpu() -> None:
    target = np.zeros((16, 16), bool); prediction = np.zeros_like(target)
    target[2, 2] = target[10, 10] = True
    prediction[2, 2] = prediction[14, 14] = True
    z = np.full((16, 16), -10.0); z[prediction] = 10.0
    scales = np.full((4, 16, 16), -10.0); scales[2, 10, 10] = -1.0
    fields, rows, maps = analyse_image_components(
        image_id="fixture", prediction=prediction, target=target,
        z_base=z, scale_logits=scales,
    )
    assert (fields["target_component_count"], fields["true_positive_component_count"],
            fields["false_negative_component_count"]) == (2, 1, 1)
    assert (fields["prediction_component_count"], fields["matched_prediction_component_count"],
            fields["false_positive_component_count"], fields["false_positive_component_area"]) == (2, 1, 1, 1)
    assert fields["recoverable_fn_target_component_ids"] == [2]
    target_rows = [r for r in rows if r["domain"] == "target"]
    assert [r["role"] for r in target_rows] == ["tp_target", "fn_target"]
    assert [r["recoverable"] for r in target_rows] == [False, True]
    assert [r["component_id"] for r in target_rows] == [1, 2]
    assert [r["component_index"] for r in target_rows] == [0, 1]
    assert maps["recoverable_fn_mask"][10, 10] == 1
    candidate_rows = [r for r in rows if r["domain"] == "candidate"]
    assert len(candidate_rows) == fields["candidate_component_count"]
    assert any(r["supports_recoverable_fn"] for r in candidate_rows)
    ones = np.ones_like(z, dtype=float)
    enrich_component_mechanism_fields(
        rows, maps, p_z=ones, j_z=-ones, ratio=2 * ones,
        conflict_score=0.5 * ones, conflict_mask=np.ones_like(target),
        prediction_logit=z,
    )
    assert all(row["conflict_fraction"] == 1.0 for row in rows)
    assert all(row["interaction_ratio_mean"] == 2.0 for row in rows)

    ckpt = {"method_meta": {"method": "MSHNet", "model_type": "mshnet", "seed": 7,
        "val_split_sha256": "val", "test_split_sha256": "test"}}
    validate_checkpoint_metadata(ckpt, seed=7, val_hash="val", role="val", split_hash="val")
    with pytest.raises(RuntimeError, match="seed"):
        validate_checkpoint_metadata(ckpt, seed=8, val_hash="val", role="val", split_hash="val")
    with pytest.raises(RuntimeError, match="test_split_sha256"):
        validate_checkpoint_metadata(ckpt, seed=7, val_hash="val", role="test", split_hash="wrong")


def test_recomputed_checkpoint_metrics_fail_closed() -> None:
    checkpoint = {"iou": 0.75, "pd": 0.9, "fa": 12.5}
    summary = {"pooled_iou": 0.75, "pd": 0.9, "fa_per_million": 12.5}
    matched = validate_recomputed_checkpoint_metrics(checkpoint, summary)
    assert matched["iou"]["recomputed"] == 0.75
    with pytest.raises(RuntimeError, match="does not reproduce"):
        validate_recomputed_checkpoint_metrics(
            checkpoint, {**summary, "pooled_iou": 0.7501}
        )


def test_projected_factual_logit_matches_native_mshnet_cpu() -> None:
    torch.manual_seed(11)
    model = ScaleInteractionExchangeMSHNet(
        3, alpha=1.0, active_stages=(0,), anchor_mode="mean",
        freeze_bn_statistics=True,
    ).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        _, native = model(image, True, alpha=0.0)
        audit_output = model(image, True, return_dict=True, alpha=1.0)
        factual = project_coalitions(model, audit_output)["z11"]
    assert torch.equal(factual, native)
