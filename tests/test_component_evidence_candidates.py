from __future__ import annotations

import numpy as np

from utils.component_evidence import (
    candidate_label_map,
    generate_prediction_candidates,
)


def test_candidate_generation_is_prediction_only_and_deduplicates_overlap() -> None:
    z_base = np.full((16, 16), -10.0)
    scale_logits = np.full((4, 16, 16), -10.0)
    z_base[3, 3] = 10.0
    scale_logits[0, 3, 3] = 9.0
    scale_logits[1, 10, 10] = 8.0

    candidates = generate_prediction_candidates(z_base, scale_logits)

    assert len(candidates) == 2
    assert candidates[0].source == "final"
    assert candidates[0].centroid == (3.0, 3.0)
    assert candidates[1].source == "scale1"
    assert candidates[1].centroid == (10.0, 10.0)
    labels = candidate_label_map(candidates, (16, 16))
    assert labels[3, 3] == 1
    assert labels[10, 10] == 2


def test_low_threshold_disjoint_candidate_is_retained() -> None:
    z_base = np.full((12, 12), -10.0)
    scale_logits = np.full((4, 12, 12), -10.0)
    z_base[2, 2] = 10.0
    scale_logits[2, 8, 8] = -1.0  # sigmoid(-1) is between 0.2 and 0.3.

    candidates = generate_prediction_candidates(z_base, scale_logits)

    assert len(candidates) == 2
    assert candidates[1].source == "scale2"
    assert candidates[1].probability_threshold == 0.2
    assert candidates[1].centroid == (8.0, 8.0)


if __name__ == "__main__":
    test_candidate_generation_is_prediction_only_and_deduplicates_overlap()
    test_low_threshold_disjoint_candidate_is_retained()
