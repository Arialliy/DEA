from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from utils.trace_codec import TraceCodecError, coordinates_to_run_chain
from utils.trace_geometry import TraceGeometrySpec, encode_trace_targets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _spec() -> TraceGeometrySpec:
    return TraceGeometrySpec(
        image_height=8,
        image_width=12,
        cell_size=4,
        max_down=2,
        max_left=2,
        max_right=3,
        margin=1,
    )


def _manual_global_index_grid(spec: TraceGeometrySpec) -> torch.Tensor:
    expected = torch.full(
        (spec.number_of_cells, spec.local_height, spec.local_width),
        -1,
        dtype=torch.long,
    )
    for cell_index in range(spec.number_of_cells):
        cell = spec.cell_coordinates(cell_index)
        origin_y, origin_x = spec.window_origin(cell)
        for local_y in range(spec.local_height):
            for local_x in range(spec.local_width):
                global_y = origin_y + local_y
                global_x = origin_x + local_x
                if 0 <= global_y < spec.image_height and 0 <= global_x < spec.image_width:
                    expected[cell_index, local_y, local_x] = (
                        global_y * spec.image_width + global_x
                    )
    return expected


def test_geometry_indices_masks_and_boundaries_match_the_contract() -> None:
    spec = _spec()
    assert (spec.grid_height, spec.grid_width, spec.number_of_cells) == (2, 3, 6)
    assert (spec.left_radius, spec.right_radius, spec.down_radius) == (3, 4, 3)
    assert (spec.local_height, spec.local_width) == (7, 11)
    assert spec.core_local_bounds == (0, 3, 4, 7)

    for cell_index in range(spec.number_of_cells):
        cell = spec.cell_coordinates(cell_index)
        assert spec.cell_index(cell) == cell_index
    assert spec.window_origin((0, 0)) == (0, -3)
    assert spec.window_origin((1, 2)) == (4, 5)
    with pytest.raises(IndexError):
        spec.cell_coordinates(spec.number_of_cells)
    with pytest.raises(IndexError):
        spec.cell_index((-1, 0))

    expected_indices = _manual_global_index_grid(spec)
    observed_indices = spec.global_index_grid()
    assert torch.equal(observed_indices, expected_indices)
    assert torch.equal(spec.valid_support_mask(), expected_indices.ge(0))

    expected_roots = torch.zeros_like(expected_indices, dtype=torch.bool)
    top, left, bottom, right = spec.core_local_bounds
    expected_roots[:, top:bottom, left:right] = True
    expected_roots &= expected_indices.ge(0)
    assert torch.equal(spec.valid_root_mask(), expected_roots)
    assert int(spec.valid_root_mask().sum()) == spec.image_height * spec.image_width

    # Explicitly lock down padding at all four image boundaries.
    assert observed_indices[0, 0, 0].item() == -1
    assert observed_indices[0, 0, spec.left_radius].item() == 0
    assert observed_indices[3, spec.cell_size, spec.left_radius].item() == -1
    assert observed_indices[5, 0, spec.left_radius + spec.cell_size - 1].item() == 59
    assert observed_indices[5, 0, spec.left_radius + spec.cell_size].item() == -1


def test_chain_window_encoding_preserves_every_global_support_pixel() -> None:
    spec = _spec()
    coordinates = np.asarray(
        (
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
        ),
        dtype=np.int64,
    )
    chain = coordinates_to_run_chain(coordinates)

    cell_index, root_y, root_x, support = spec.chain_to_local_mask(chain)
    assert cell_index == spec.cell_index((0, 1)) == 1
    assert (root_y, root_x) == (1, spec.left_radius)
    assert support.dtype == np.bool_
    assert int(support.sum()) == len(coordinates)

    global_indices = spec.global_index_grid()[cell_index][torch.from_numpy(support)]
    expected_indices = torch.tensor(
        sorted(int(y * spec.image_width + x) for y, x in coordinates),
        dtype=torch.long,
    )
    assert torch.equal(torch.sort(global_indices).values, expected_indices)


def test_chain_outside_frozen_owner_window_fails_instead_of_clipping() -> None:
    spec = TraceGeometrySpec(
        image_height=8,
        image_width=8,
        cell_size=4,
        max_down=0,
        max_left=0,
        max_right=0,
        margin=0,
    )
    chain = coordinates_to_run_chain(np.asarray(((3, 1), (4, 1)), dtype=np.int64))

    with pytest.raises(TraceCodecError) as caught:
        spec.chain_to_local_mask(chain)
    assert caught.value.code == "window_coverage"


def test_encode_trace_targets_empty_mask_has_canonical_empty_shapes() -> None:
    spec = _spec()
    encoded = encode_trace_targets(np.zeros((8, 12), dtype=np.uint8), spec)

    assert encoded.number_of_cells == spec.number_of_cells
    assert encoded.positive_count == 0
    assert encoded.positive_cell_indices.dtype == torch.long
    assert encoded.positive_cell_indices.shape == (0,)
    assert encoded.root_local_y.shape == (0,)
    assert encoded.root_local_x.shape == (0,)
    assert encoded.support_local.dtype == torch.bool
    assert encoded.support_local.shape == (0, spec.local_height, spec.local_width)


def test_encode_trace_targets_positive_components_keep_roots_and_support_exact() -> None:
    spec = TraceGeometrySpec(
        image_height=8,
        image_width=12,
        cell_size=4,
        max_down=2,
        max_left=1,
        max_right=1,
        margin=0,
    )
    mask = np.zeros((8, 12), dtype=bool)
    first = ((1, 1), (1, 2), (2, 0), (2, 1), (2, 2))
    second = ((4, 8), (5, 7), (5, 8), (6, 8), (6, 9))
    for coordinate in first + second:
        mask[coordinate] = True

    encoded = encode_trace_targets(mask, spec)
    assert encoded.positive_count == 2
    assert encoded.positive_cell_indices.tolist() == [0, 5]
    assert encoded.root_local_y.tolist() == [1, 0]
    assert encoded.root_local_x.tolist() == [2, 1]

    expected_by_cell = {0: first, 5: second}
    index_grid = spec.global_index_grid()
    for positive_index, cell_index in enumerate(encoded.positive_cell_indices.tolist()):
        observed = index_grid[cell_index][encoded.support_local[positive_index]]
        expected = torch.tensor(
            sorted(
                y * spec.image_width + x
                for y, x in expected_by_cell[cell_index]
            ),
            dtype=torch.long,
        )
        assert torch.equal(torch.sort(observed).values, expected)


def test_encode_trace_targets_rejects_root_cell_collision() -> None:
    spec = TraceGeometrySpec(
        image_height=8,
        image_width=8,
        cell_size=4,
        max_down=0,
        max_left=0,
        max_right=0,
        margin=0,
    )
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 0] = True
    mask[1, 3] = True

    with pytest.raises(TraceCodecError) as caught:
        encode_trace_targets(mask, spec)
    assert caught.value.code == "root_cell_collision"


def test_audit_cli_rejects_train_test_overlap_even_in_report_only_mode(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "leaky_dataset"
    split_dir = dataset / "img_idx"
    split_dir.mkdir(parents=True)
    (split_dir / "train_fold.txt").write_text("leaked_sample\n", encoding="utf-8")
    (split_dir / "test_fold.txt").write_text("leaked_sample\n", encoding="utf-8")
    output = tmp_path / "must_not_exist.json"

    completed = subprocess.run(
        (
            sys.executable,
            str(PROJECT_ROOT / "tools" / "audit_trace_geometry.py"),
            "--dataset-dir",
            str(dataset),
            "--image-height",
            "8",
            "--image-width",
            "8",
            "--cell-sizes",
            "4",
            "--output",
            str(output),
            "--report-only",
        ),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "train split overlaps canonical test names" in completed.stderr
    assert not output.exists()
