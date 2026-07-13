import json

import numpy as np
import pytest

from tools.audit_rcp_decoder_contract import (
    RCPDecoderContractError,
    component_separation_contract,
    fit_only_preregistration,
    load_target_program_rows,
    program_contract,
)
from utils.rooted_component_program import encode_rooted_component


def test_program_contract_reports_exact_tree_geometry_and_frontier() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[2, 2:5] = True
    program = encode_rooted_component(mask)
    contract = program_contract(program, bbox=(2, 2, 3, 5), shape=mask.shape)

    assert contract["canonical_bfs_depth"] == 1
    assert contract["_node_depths"] == [0, 1, 1]
    assert contract["_parent_lags"] == [1, 2]
    assert contract["offset_counts"] == [0, 0, 1, 0, 0, 0, 1, 0]
    assert contract["root_chebyshev_radius"] == 1
    assert contract["bbox_height"] == 1
    assert contract["bbox_width"] == 3
    assert contract["_frontier_candidate_sizes"] == [8, 10, 12]
    assert contract["frontier_candidate_size_mean"] == 10.0
    assert contract["frontier_candidate_size_max"] == 12


def test_component_separation_uses_pixel_chebyshev_distance() -> None:
    first = np.zeros((5, 5), dtype=bool)
    second = np.zeros((5, 5), dtype=bool)
    first[1, 1] = True
    second[1, 3] = True
    separated = component_separation_contract((first, second))
    assert separated["minimum_distinct_component_chebyshev_distance"] == 2
    assert separated["minimum_empty_pixel_gap"] == 1
    assert not separated["any_distinct_components_8_adjacent"]

    second[:] = False
    second[2, 2] = True
    adjacent = component_separation_contract((first, second))
    assert adjacent["minimum_distinct_component_chebyshev_distance"] == 1
    assert adjacent["any_distinct_components_8_adjacent"]


def test_fit_only_preregistration_never_uses_dev_extremes() -> None:
    target_rows = [
        {
            "split": "fit",
            "program_nodes": 6,
            "canonical_bfs_depth": 3,
            "root_chebyshev_radius": 4,
            "formal_hard_core": False,
        },
        {
            "split": "dev",
            "program_nodes": 60,
            "canonical_bfs_depth": 30,
            "root_chebyshev_radius": 40,
            "formal_hard_core": True,
        },
    ]
    image_rows = [
        {"split": "fit", "component_count": 2},
        {"split": "dev", "component_count": 20},
    ]
    result = fit_only_preregistration(target_rows, image_rows)
    assert result["selected"] == {
        "T": 8,
        "fit_maximum_required_T": 6,
        "local_patch_radius": 4,
        "topK": 2,
        "parallel_growth_rounds": 3,
        "maximum_add_actions": 7,
        "local_patch_side": 9,
    }
    assert result["fit_confirmation"]["T_target_coverage"]["coverage"] == 1.0
    assert result["dev_report_only"]["T_target_coverage"]["coverage"] == 0.0
    assert result["dev_report_only"]["patch_target_coverage"]["coverage"] == 0.0
    assert result["dev_report_only"]["topK_image_coverage"]["coverage"] == 0.0


def test_target_program_loader_rejects_official_or_unknown_split(tmp_path) -> None:
    path = tmp_path / "programs.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset": "D",
                "split": "test",
                "stable_target_id": "T",
                "target_area": 1,
                "program_nodes": 1,
                "exact_roundtrip": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RCPDecoderContractError, match="forbidden split"):
        load_target_program_rows(path)
