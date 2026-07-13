from __future__ import annotations

import numpy as np
import pytest

from utils.trace_codec import (
    HorizontalRun,
    TraceCodecError,
    assign_root_cell,
    component_records,
    coordinates_to_run_chain,
    root_cell_collisions,
    run_chain_to_mask,
)


def test_exact_chain_roundtrip_is_bit_exact_and_order_invariant() -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[1, 4:7] = 255
    mask[2, 3:7] = 255
    mask[3, 2:4] = 255
    mask[4, 1:5] = 255
    mask[5, 2:3] = 255

    expected_runs = (
        HorizontalRun(1, 4, 6),
        HorizontalRun(2, 3, 6),
        HorizontalRun(3, 2, 3),
        HorizontalRun(4, 1, 4),
        HorizontalRun(5, 2, 2),
    )
    coordinates = np.argwhere(mask)
    forward = coordinates_to_run_chain(coordinates)
    reverse = coordinates_to_run_chain(coordinates[::-1])

    assert forward == reverse
    assert forward.runs == expected_runs
    assert forward.root == (1, 4)
    assert forward.area == int(np.count_nonzero(mask))
    assert forward.bbox == (1, 1, 6, 7)
    assert forward.relative_extents == (0, 4, 3, 2)
    assert np.array_equal(run_chain_to_mask(forward, mask.shape), mask > 0)

    records = component_records(mask)
    assert len(records) == 1
    assert records[0].exact
    assert records[0].chain == forward
    assert records[0].error_code is None
    assert records[0].max_runs_per_row == 1


def test_connected_component_with_multiple_runs_in_one_row_fails_closed() -> None:
    # The full support is 8-connected through the second row, but the top row
    # has two disjoint runs.  TRACE must reject it instead of filling the hole.
    mask = np.zeros((5, 7), dtype=bool)
    mask[1, 1] = True
    mask[1, 4] = True
    mask[2, 1:5] = True

    records = component_records(mask)
    assert len(records) == 1
    record = records[0]
    assert not record.exact
    assert record.chain is None
    assert record.error_code == "multiple_runs_per_row"
    assert record.max_runs_per_row == 2
    assert record.area == int(mask.sum())

    with pytest.raises(TraceCodecError) as caught:
        coordinates_to_run_chain(np.argwhere(mask))
    assert caught.value.code == "multiple_runs_per_row"


def test_diagonal_runs_use_eight_connectivity_and_larger_gap_is_rejected() -> None:
    diagonal = np.zeros((6, 6), dtype=bool)
    diagonal[1, 1] = True
    diagonal[2, 2] = True
    diagonal[3, 3] = True

    records = component_records(diagonal)
    assert len(records) == 1
    assert records[0].exact
    assert records[0].chain is not None
    assert records[0].chain.runs == (
        HorizontalRun(1, 1, 1),
        HorizontalRun(2, 2, 2),
        HorizontalRun(3, 3, 3),
    )
    assert np.array_equal(
        run_chain_to_mask(records[0].chain, diagonal.shape), diagonal
    )

    disconnected_adjacent_rows = np.asarray(((1, 1), (2, 3)), dtype=np.int64)
    with pytest.raises(TraceCodecError) as caught:
        coordinates_to_run_chain(disconnected_adjacent_rows)
    assert caught.value.code == "disconnected_adjacent_runs"


def test_canonical_root_is_leftmost_pixel_of_the_topmost_row() -> None:
    mask = np.zeros((7, 9), dtype=bool)
    mask[1, 5:7] = True
    mask[2, 2:6] = True
    mask[3, 1:4] = True

    (record,) = component_records(mask)
    assert record.exact
    assert record.root == (1, 5)
    assert record.chain is not None
    assert record.chain.root == (1, 5)
    # In particular, the root is not the globally leftmost support pixel.
    assert int(np.argwhere(mask)[:, 1].min()) == 1


def test_root_cell_assignment_and_collisions_are_exact() -> None:
    roots = ((0, 0), (3, 3), (4, 0), (7, 3), (7, 7))

    assert [assign_root_cell(root, 4) for root in roots] == [
        (0, 0),
        (0, 0),
        (1, 0),
        (1, 0),
        (1, 1),
    ]
    assert root_cell_collisions(roots, 4) == {
        (0, 0): ((0, 0), (3, 3)),
        (1, 0): ((4, 0), (7, 3)),
    }
    assert root_cell_collisions(((0, 0), (4, 0), (4, 4)), 4) == {}


def test_decode_refuses_to_clip_support_outside_the_canvas() -> None:
    chain = coordinates_to_run_chain(np.asarray(((2, 3), (2, 4))))
    with pytest.raises(TraceCodecError) as caught:
        run_chain_to_mask(chain, (3, 4))
    assert caught.value.code == "window_or_image_clip"
