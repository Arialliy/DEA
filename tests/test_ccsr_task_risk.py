from __future__ import annotations

import math
import itertools

import numpy as np
import pytest
import torch

from model.ccsr.task_risk import (
    build_components_from_binary,
    build_gt_components,
    component_pair_cost,
    exact_component_edit_risk,
    exact_component_edit_risk_from_masks,
)


def _mask(shape=(9, 11), points=()):
    result = np.zeros(shape, dtype=np.uint8)
    for y, x in points:
        result[y, x] = 1
    return result


def test_perfect_prediction_has_zero_risk() -> None:
    target = _mask(points=((2, 2), (2, 3), (6, 8)))
    result = exact_component_edit_risk_from_masks(target, target)
    assert result.risk == 0.0
    assert len(result.matches) == 2
    assert result.unmatched_prediction_indices == ()
    assert result.unmatched_target_indices == ()
    assert result.risk == pytest.approx(
        result.unmatched_base_risk - result.matching_credit
    )


def test_complete_miss_is_one_over_num_gt() -> None:
    target = _mask(points=((2, 2), (6, 8)))
    prediction = _mask(points=((2, 2),))
    result = exact_component_edit_risk_from_masks(prediction, target)
    assert result.miss_risk == 0.5
    assert result.risk == 0.5


def test_empty_target_false_component_is_area_fraction() -> None:
    prediction = _mask(points=((2, 2), (2, 3), (3, 3)))
    target = np.zeros_like(prediction)
    result = exact_component_edit_risk_from_masks(prediction, target)
    assert result.risk == 3.0 / prediction.size
    assert result.clutter_risk == result.risk
    assert result.unmatched_prediction_indices == (0,)


def test_both_empty_requires_canvas_and_has_zero_risk() -> None:
    result = exact_component_edit_risk([], [], image_shape=(7, 13))
    assert result.risk == 0.0
    assert result.num_pixels == 91
    with pytest.raises(ValueError, match="image_shape is required"):
        exact_component_edit_risk([], [])


def test_one_pixel_centroid_hit_keeps_shape_missing_risk() -> None:
    target = np.zeros((9, 9), dtype=np.uint8)
    target[3:6, 3:6] = 1
    prediction = _mask((9, 9), points=((4, 4),))
    result = exact_component_edit_risk_from_masks(prediction, target)
    assert len(result.matches) == 1
    assert result.matched_missing_risk == 8.0 / 9.0
    assert result.risk == 8.0 / 9.0


def test_bridge_component_is_atomic_and_can_match_only_one_gt() -> None:
    target = _mask((7, 9), points=((3, 2), (3, 6)))
    prediction = _mask(
        (7, 9),
        points=((3, 2), (3, 3), (3, 4), (3, 5), (3, 6)),
    )
    result = exact_component_edit_risk_from_masks(prediction, target)
    assert len(result.matches) == 1
    assert len(result.unmatched_target_indices) == 1
    assert result.miss_risk == 0.5
    assert result.matched_excess_risk == 4.0 / prediction.size


def test_component_and_gt_order_permutations_preserve_risk() -> None:
    target = _mask(points=((2, 2), (6, 8)))
    prediction = _mask(points=((2, 2), (6, 7)))
    pred_components = build_components_from_binary(prediction)
    gt_components = build_components_from_binary(target)
    forward = exact_component_edit_risk(pred_components, gt_components)
    reversed_order = exact_component_edit_risk(
        tuple(reversed(pred_components)),
        tuple(reversed(gt_components)),
    )
    assert forward.risk == reversed_order.risk
    assert len(forward.matches) == len(reversed_order.matches) == 2


def test_centroid_radius_is_strict() -> None:
    prediction = _mask((3, 8), points=((1, 0),))
    target_at_three = _mask((3, 8), points=((1, 3),))
    assert math.isinf(component_pair_cost(
        prediction,
        target_at_three,
        num_gt=1,
        num_pixels=prediction.size,
        centroid_radius=3.0,
    ))

    prediction_at_half = _mask((3, 8), points=((1, 0), (1, 1)))
    assert math.isfinite(component_pair_cost(
        prediction_at_half,
        target_at_three,
        num_gt=1,
        num_pixels=prediction.size,
        centroid_radius=3.0,
    ))


def test_worker_instance_labels_are_validated() -> None:
    target = torch.zeros(6, 7)
    target[1, 1] = 1
    target[4, 5] = 1
    labels = torch.zeros(6, 7, dtype=torch.long)
    labels[1, 1] = 7
    labels[4, 5] = 3
    components = build_gt_components(target, labels)
    assert len(components) == 2

    invalid = labels.clone()
    invalid[0, 0] = 9
    with pytest.raises(ValueError, match="support must equal"):
        build_gt_components(target, invalid)


def test_risk_fails_closed_on_invalid_component_collections() -> None:
    component = torch.zeros(5, 5, dtype=torch.bool)
    component[1, 1] = True
    overlap = component.clone()
    with pytest.raises(ValueError, match="disjoint"):
        exact_component_edit_risk(
            [component, overlap],
            [component],
            image_shape=(5, 5),
        )
    disconnected = component.clone()
    disconnected[4, 4] = True
    with pytest.raises(ValueError, match="connected"):
        exact_component_edit_risk(
            [disconnected],
            [component],
            image_shape=(5, 5),
        )
    with pytest.raises(ValueError, match="8-connectivity"):
        exact_component_edit_risk(
            [component],
            [component],
            image_shape=(5, 5),
            connectivity=1,
        )
    with pytest.raises(ValueError, match="positive integers"):
        exact_component_edit_risk([], [], image_shape=(3.5, 4))


@pytest.mark.parametrize("radius", [0.0, float("nan"), float("inf")])
def test_radius_must_be_finite_and_positive(radius: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        exact_component_edit_risk([], [], image_shape=(3, 3), centroid_radius=radius)


def _brute_force_risk(pred_components, gt_components, *, radius: float) -> float:
    num_gt = len(gt_components)
    num_pixels = int(pred_components[0].numel() if pred_components else gt_components[0].numel())
    if num_gt == 0:
        return sum(float(component.sum()) for component in pred_components) / num_pixels

    best = float("inf")

    def visit(prediction_index, used_targets, running_cost):
        nonlocal best
        if prediction_index == len(pred_components):
            candidate = running_cost + (num_gt - len(used_targets)) / num_gt
            best = min(best, candidate)
            return

        prediction = pred_components[prediction_index]
        visit(
            prediction_index + 1,
            used_targets,
            running_cost + float(prediction.sum()) / num_pixels,
        )
        for target_index, target in enumerate(gt_components):
            if target_index in used_targets:
                continue
            pair_cost = component_pair_cost(
                prediction,
                target,
                num_gt=num_gt,
                num_pixels=num_pixels,
                centroid_radius=radius,
            )
            if math.isfinite(pair_cost):
                visit(
                    prediction_index + 1,
                    used_targets | {target_index},
                    running_cost + pair_cost,
                )

    visit(0, set(), 0.0)
    return best


def test_hungarian_risk_matches_exhaustive_partial_assignments() -> None:
    generator = np.random.default_rng(20260711)
    shape = (6, 7)
    all_positions = list(itertools.product(range(shape[0]), range(shape[1])))

    for _ in range(200):
        num_predictions = int(generator.integers(0, 4))
        num_gt = int(generator.integers(0, 4))
        if num_predictions + num_gt == 0:
            continue
        prediction_positions = generator.choice(
            len(all_positions), size=num_predictions, replace=False
        )
        target_positions = generator.choice(
            len(all_positions), size=num_gt, replace=False
        )
        pred_components = tuple(
            torch.from_numpy(_mask(shape, points=(all_positions[int(index)],))).bool()
            for index in prediction_positions
        )
        gt_components = tuple(
            torch.from_numpy(_mask(shape, points=(all_positions[int(index)],))).bool()
            for index in target_positions
        )
        actual = exact_component_edit_risk(
            pred_components,
            gt_components,
            centroid_radius=3.0,
            image_shape=shape,
        )
        expected = _brute_force_risk(
            pred_components,
            gt_components,
            radius=3.0,
        )
        assert actual.risk == pytest.approx(expected, abs=1e-12, rel=0)
        assert actual.risk == pytest.approx(
            actual.unmatched_base_risk - actual.matching_credit,
            abs=1e-12,
            rel=0,
        )
