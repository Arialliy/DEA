from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.metric import PD_FA


def test_pd_fa_uses_probability_half_at_official_index() -> None:
    metric = PD_FA(nclass=1, bins=10, size=16)
    logits = torch.full((1, 1, 16, 16), -10.0)
    labels = torch.zeros_like(logits)
    logits[0, 0, 4, 4] = 10.0
    logits[0, 0, 12, 12] = 10.0
    labels[0, 0, 4, 4] = 1.0

    metric.update(logits, labels)
    false_alarm, detection_probability = metric.get(img_num=999)

    assert metric.thresholds[0] == 0.5
    assert np.isclose(detection_probability[0], 1.0)
    assert np.isclose(false_alarm[0], 1.0 / (16 * 16))


def test_pd_fa_counts_equal_area_unmatched_component_by_identity() -> None:
    metric = PD_FA(nclass=1, bins=10, size=16)
    logits = torch.full((2, 1, 16, 16), -10.0)
    labels = torch.zeros_like(logits)
    logits[0, 0, 3, 3] = 10.0
    labels[0, 0, 3, 3] = 1.0
    logits[1, 0, 10, 10] = 10.0

    metric.update(logits, labels)
    false_alarm, detection_probability = metric.get(img_num=1)

    assert metric.num_images == 2
    assert np.isclose(detection_probability[0], 1.0)
    assert np.isclose(false_alarm[0], 1.0 / (16 * 16 * 2))
