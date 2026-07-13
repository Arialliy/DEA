import numpy as np
import pytest
from skimage import measure

from utils.rooted_component_program import (
    RootedComponentProgram,
    RootedComponentProgramError,
    canonical_component_root,
    encode_rooted_component,
    program_positions,
    render_rooted_component,
    truncate_rooted_component,
)


def _assert_connected(mask: np.ndarray) -> None:
    labels = measure.label(mask, connectivity=2)
    assert int(labels.max()) == 1


def test_single_pixel_roundtrip() -> None:
    mask = np.zeros((7, 9), dtype=bool)
    mask[3, 4] = True
    program = encode_rooted_component(mask)
    assert program.root_y == 3
    assert program.root_x == 4
    assert program.node_count == 1
    assert program.parent_indices == ()
    assert program.offset_codes == ()
    assert np.array_equal(render_rooted_component(program, mask.shape), mask)


def test_non_star_shaped_component_roundtrip_is_exact_and_deterministic() -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    coordinates = (
        (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 2),                         (3, 5),
        (4, 2), (4, 3),                 (4, 5),
        (5, 3), (5, 4), (5, 5),
        (6, 5), (7, 5), (7, 6),
    )
    for coordinate in coordinates:
        mask[coordinate] = 1
    first = encode_rooted_component(mask)
    second = encode_rooted_component(mask.copy())
    assert first == second
    assert first.node_count == int(mask.sum())
    assert all(parent < child for child, parent in enumerate(first.parent_indices, 1))
    assert np.array_equal(render_rooted_component(first, mask.shape), mask.astype(bool))


def test_root_uses_centroid_distance_then_lexicographic_tie_break() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[1, 2] = True
    mask[2, 1] = True
    mask[2, 2] = True
    assert canonical_component_root(mask) == (1, 1)


def test_disconnected_or_nonbinary_masks_fail_closed() -> None:
    disconnected = np.zeros((5, 5), dtype=bool)
    disconnected[1, 1] = True
    disconnected[3, 3] = True
    with pytest.raises(RootedComponentProgramError, match="exactly one"):
        encode_rooted_component(disconnected)
    with pytest.raises(RootedComponentProgramError, match="exactly binary"):
        encode_rooted_component(np.asarray([[0.0, 0.5], [0.0, 1.0]]))
    with pytest.raises(RootedComponentProgramError, match="cannot be empty"):
        encode_rooted_component(np.zeros((3, 3), dtype=bool))


@pytest.mark.parametrize(
    "program,error",
    (
        (
            RootedComponentProgram(1, 1, (1,), (0,)),
            "parent_indices",
        ),
        (
            RootedComponentProgram(1, 1, (0,), (8,)),
            "offset_codes",
        ),
        (
            RootedComponentProgram(1, 1, (0, 0), (0, 0)),
            "duplicates",
        ),
        (
            RootedComponentProgram(0, 0, (0,), (0,)),
            "outside",
        ),
    ),
)
def test_invalid_programs_fail_closed(program, error) -> None:
    with pytest.raises(RootedComponentProgramError, match=error):
        render_rooted_component(program, (4, 4))


def test_every_canonical_prefix_remains_connected() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:8, 3] = True
    mask[7, 3:10] = True
    mask[4:8, 9] = True
    program = encode_rooted_component(mask)
    for max_nodes in range(1, program.node_count + 1):
        prefix = truncate_rooted_component(program, max_nodes)
        rendered = render_rooted_component(prefix, mask.shape)
        assert int(rendered.sum()) == max_nodes
        _assert_connected(rendered)


def test_legal_arbitrary_program_is_connected_by_construction() -> None:
    # A branching program, not one produced by the canonical encoder.
    program = RootedComponentProgram(
        root_y=4,
        root_x=4,
        parent_indices=(0, 0, 1, 1, 2, 4),
        offset_codes=(0, 2, 7, 1, 3, 2),
    )
    positions = program_positions(program, shape=(10, 10))
    assert len(set(positions)) == program.node_count
    rendered = render_rooted_component(program, (10, 10))
    _assert_connected(rendered)


def test_codec_is_complete_for_every_connected_three_by_three_mask() -> None:
    checked = 0
    for bit_pattern in range(1, 1 << 9):
        flat = np.asarray(
            [(bit_pattern >> index) & 1 for index in range(9)],
            dtype=bool,
        )
        mask = flat.reshape(3, 3)
        if int(measure.label(mask, connectivity=2).max()) != 1:
            continue
        program = encode_rooted_component(mask)
        assert np.array_equal(render_rooted_component(program, mask.shape), mask)
        checked += 1
    assert checked == 388
