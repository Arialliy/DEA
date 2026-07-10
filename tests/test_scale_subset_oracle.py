from __future__ import annotations

import numpy as np
import torch

from utils.metric import PD_FA, match_connected_components
from utils.scale_subset import kept_scale_indices, reconstruct_scale_subset


def test_all_scale_subset_returns_direct_baseline_identity() -> None:
    torch.manual_seed(47)
    contributions = torch.randn(2, 4, 13, 17)
    bias = torch.randn(1, 1, 1, 1)
    direct = torch.randn(2, 1, 13, 17)

    observed = reconstruct_scale_subset(
        contributions,
        bias,
        subset=15,
        z_base=direct,
    )

    assert observed is direct
    assert torch.equal(observed, direct)


def test_subset_reconstruction_uses_bias_free_selected_contributions() -> None:
    contributions = torch.zeros(1, 4, 3, 3)
    for scale in range(4):
        contributions[:, scale] = float(scale + 1)
    bias = torch.tensor([[[[0.5]]]])

    observed = reconstruct_scale_subset(contributions, bias, subset=0b0101)

    assert kept_scale_indices(0b0101) == (0, 2)
    assert torch.equal(observed, torch.full((1, 1, 3, 3), 4.5))


def test_public_component_match_is_identical_to_pd_fa_operating_point() -> None:
    prediction = np.zeros((16, 16), dtype=np.uint8)
    target = np.zeros_like(prediction)
    prediction[3, 3] = 1
    prediction[10, 10] = 1
    target[3, 3] = 1

    matched = match_connected_components(prediction, target)
    assert len(matched.matches) == 1
    assert matched.unmatched_prediction_indices == (1,)
    assert matched.unmatched_target_indices == ()

    logits = torch.full((1, 1, 16, 16), -10.0)
    labels = torch.zeros_like(logits)
    logits[0, 0, 3, 3] = 10.0
    logits[0, 0, 10, 10] = 10.0
    labels[0, 0, 3, 3] = 1.0
    metric = PD_FA(nclass=1, bins=10, size=16)
    metric.update(logits, labels)
    false_alarm, detection_probability = metric.get()

    assert np.isclose(detection_probability[0], 1.0)
    assert np.isclose(false_alarm[0], 1.0 / (16 * 16))


if __name__ == "__main__":
    test_all_scale_subset_returns_direct_baseline_identity()
    test_subset_reconstruction_uses_bias_free_selected_contributions()
    test_public_component_match_is_identical_to_pd_fa_operating_point()
