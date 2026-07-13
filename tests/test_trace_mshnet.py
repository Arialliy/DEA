from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from model.MSHNet import MSHNet
from model.trace_mshnet import (
    MatchedDensePotentialMap,
    TRACEMSHNet,
    TraceAtomicField,
    TraceFieldOutput,
    TraceModelError,
    TracePotentialMap,
    render_trace_atoms,
)
from model.trace_run_semiring import (
    RootCellRunSemiring,
    brute_force_reference,
    zero_score_log_cardinality,
)
from utils.trace_geometry import (
    EncodedTraceTargets,
    TraceGeometrySpec,
    encode_trace_targets,
)


DATASET = "NUAA-SIRST"
SEED = 17
TRAIN_SPLIT_SHA256 = "1" * 64
VAL_SPLIT_SHA256 = "2" * 64
FORBIDDEN_HEADS = {"output_0", "output_1", "output_2", "output_3", "final"}


def _irregular_spec() -> TraceGeometrySpec:
    return TraceGeometrySpec(
        image_height=4,
        image_width=6,
        cell_size=2,
        max_down=1,
        max_left=1,
        max_right=2,
        margin=0,
    )


def _empty_target(spec: TraceGeometrySpec) -> EncodedTraceTargets:
    return EncodedTraceTargets(
        number_of_cells=spec.number_of_cells,
        positive_cell_indices=torch.empty(0, dtype=torch.long),
        root_local_y=torch.empty(0, dtype=torch.long),
        root_local_x=torch.empty(0, dtype=torch.long),
        support_local=torch.empty(
            (0, spec.local_height, spec.local_width), dtype=torch.bool
        ),
    )


def _manual_local_fields(
    image_field: torch.Tensor,
    spec: TraceGeometrySpec,
) -> torch.Tensor:
    indices = spec.global_index_grid(device=image_field.device)
    valid = indices.ge(0)
    gathered = image_field.flatten(1)[:, indices.clamp_min(0)]
    return torch.where(valid.unsqueeze(0), gathered, torch.zeros_like(gathered))


def test_boundary_patterns_and_cached_logk_match_public_geometry() -> None:
    spec = _irregular_spec()
    field = TraceAtomicField(spec, field_chunk_size=4)
    mapping = field.cell_pattern_index

    expected_support = spec.valid_support_mask()
    expected_root = spec.valid_root_mask()
    assert torch.equal(field.pattern_support_mask[mapping], expected_support)
    assert torch.equal(field.pattern_root_mask[mapping], expected_root)
    # This geometry exercises left, right, and bottom clipping combinations.
    assert field.pattern_support_mask.shape[0] == spec.number_of_cells == 6

    direct_logk = zero_score_log_cardinality(
        expected_support,
        expected_root,
        dtype=torch.float64,
    )
    cached_logk = field.pattern_log_cardinality[mapping]
    torch.testing.assert_close(cached_logk, direct_logk, atol=1e-12, rtol=1e-12)
    assert len(field.logk_cache_sha256) == 64


def test_chunked_field_matches_direct_per_cell_semiring_for_batch_and_boundaries() -> None:
    spec = _irregular_spec()
    field = TraceAtomicField(spec, field_chunk_size=4)
    generator = torch.Generator().manual_seed(20260713)
    root = torch.randn(
        (2, spec.image_height, spec.image_width), generator=generator
    )
    support = torch.randn(
        (2, spec.image_height, spec.image_width), generator=generator
    )

    chunked = field(
        root,
        support,
        return_map=True,
        return_marginals=True,
    )

    local_root = _manual_local_fields(root.float(), spec)
    local_support = _manual_local_fields(support.float(), spec)
    batch, cells = local_root.shape[:2]
    support_mask = spec.valid_support_mask().unsqueeze(0).expand(batch, -1, -1, -1)
    root_mask = spec.valid_root_mask().unsqueeze(0).expand(batch, -1, -1, -1)
    cell_logk = zero_score_log_cardinality(
        spec.valid_support_mask(), spec.valid_root_mask(), dtype=torch.float64
    )
    direct = RootCellRunSemiring()(
        local_root.reshape(batch * cells, spec.local_height, spec.local_width),
        local_support.reshape(batch * cells, spec.local_height, spec.local_width),
        support_mask.reshape(batch * cells, spec.local_height, spec.local_width),
        root_mask.reshape(batch * cells, spec.local_height, spec.local_width),
        log_cardinality=cell_logk.unsqueeze(0).expand(batch, -1).reshape(-1),
        return_map=True,
        return_marginals=True,
    )

    for name in (
        "logZ_positive",
        "logZ_total",
        "p_nonempty",
        "log_cardinality",
        "map_energy",
        "map_log_joint_posterior",
    ):
        expected = getattr(direct, name).reshape(batch, cells)
        torch.testing.assert_close(
            getattr(chunked, name), expected, atol=2e-5, rtol=2e-5
        )
    assert torch.equal(chunked.map_root, direct.map_root.reshape(batch, cells, 2))
    assert torch.equal(
        chunked.map_intervals,
        direct.map_intervals.reshape(batch, cells, spec.local_height, 2),
    )
    assert torch.equal(
        chunked.map_support,
        direct.map_support.reshape(
            batch, cells, spec.local_height, spec.local_width
        ),
    )
    for name in ("root_marginal", "support_marginal"):
        expected = getattr(direct, name).reshape(
            batch, cells, spec.local_height, spec.local_width
        )
        torch.testing.assert_close(
            getattr(chunked, name), expected, atol=2e-5, rtol=2e-5
        )


def test_exact_nll_matches_manual_and_brute_force_with_empty_image_gradients() -> None:
    spec = TraceGeometrySpec(
        image_height=2,
        image_width=2,
        cell_size=2,
        max_down=0,
        max_left=0,
        max_right=0,
        margin=0,
    )
    field = TraceAtomicField(spec, field_chunk_size=1)
    root = torch.tensor(
        [
            [[0.3, -0.4], [0.2, 0.7]],
            [[-0.2, 0.1], [0.5, -0.6]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    support = torch.tensor(
        [
            [[0.4, -0.1], [0.8, -0.5]],
            [[0.2, 0.6], [-0.3, 0.9]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    output = field(root, support, return_map=False)

    positive_mask = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    positive = encode_trace_targets(positive_mask, spec)
    empty = _empty_target(spec)
    result_sum = field.exact_nll(output, (positive, empty), reduction="sum")
    result_mean = field.exact_nll(output, (positive, empty), reduction="mean")

    brute = brute_force_reference(
        root,
        support,
        spec.valid_support_mask(),
        spec.valid_root_mask(),
        return_marginals=False,
    )
    torch.testing.assert_close(
        output.logZ_total.reshape(-1), brute.logZ_total, atol=2e-6, rtol=2e-6
    )
    positive_energy = (
        root[0, 0, 0]
        + support[0, 0, 0]
        + support[0, 0, 1]
        - output.log_cardinality[0, 0]
    )
    expected_per_image = torch.stack(
        (
            brute.logZ_total[0] - positive_energy,
            brute.logZ_total[1],
        )
    )
    torch.testing.assert_close(
        result_sum.per_image_sum, expected_per_image, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        result_sum.positive_energy_sum,
        torch.stack((positive_energy, positive_energy.new_zeros(()))),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(result_sum.loss, expected_per_image.sum())
    torch.testing.assert_close(result_mean.loss, expected_per_image.mean())
    assert result_sum.positive_count == 1
    assert result_sum.cell_count == 2

    root_gradient, support_gradient = torch.autograd.grad(
        result_mean.loss, (root, support)
    )
    assert bool(torch.isfinite(root_gradient).all())
    assert bool(torch.isfinite(support_gradient).all())
    # The second image is all-empty supervision.  Its loss is log Z_total,
    # whose non-zero derivatives are the exact Bernoulli-inclusive marginals.
    assert float(root_gradient[1].abs().sum()) > 0.0
    assert float(support_gradient[1].abs().sum()) > 0.0


def test_map_capacity_prior_calibration_and_empty_bernoulli_gradient() -> None:
    prior = 0.07
    potential = TracePotentialMap(prior)
    dense = MatchedDensePotentialMap(0.02)
    assert sum(parameter.numel() for parameter in potential.parameters()) == 306
    assert sum(parameter.numel() for parameter in dense.parameters()) == 307
    assert sum(
        parameter.numel() for parameter in potential.parameters() if parameter.requires_grad
    ) == 306
    assert sum(
        parameter.numel() for parameter in dense.parameters() if parameter.requires_grad
    ) == 307

    spec = _irregular_spec()
    field = TraceAtomicField(spec, field_chunk_size=5)
    features = torch.zeros(
        (2, 16, spec.image_height, spec.image_width), dtype=torch.float32
    )
    natural = potential(features)
    expected_logit = math.log(prior / (1.0 - prior))
    torch.testing.assert_close(
        natural[:, 0], torch.full_like(natural[:, 0], expected_logit)
    )
    assert torch.count_nonzero(natural[:, 1]) == 0

    output = field(natural[:, 0], natural[:, 1], return_map=False)
    torch.testing.assert_close(
        output.logZ_positive,
        torch.full_like(output.logZ_positive, expected_logit),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        output.p_nonempty,
        torch.full_like(output.p_nonempty, prior),
        atol=2e-6,
        rtol=2e-6,
    )

    empty_targets = tuple(_empty_target(spec) for _ in range(2))
    loss = field.exact_nll(output, empty_targets, reduction="mean").loss
    loss.backward()
    final_bias_gradient = potential[2].bias.grad
    assert final_bias_gradient is not None
    assert bool(torch.isfinite(final_bias_gradient).all())
    # A shared constant root shift changes every positive atom once, so the
    # mean all-empty gradient is exactly the calibrated existence prior.
    assert float(final_bias_gradient[0]) == pytest.approx(prior, abs=2e-6)
    assert float(final_bias_gradient[1]) > 0.0


def _fake_renderer_output(
    spec: TraceGeometrySpec,
    map_support: torch.Tensor,
    map_scores: torch.Tensor,
    p_nonempty: torch.Tensor,
) -> TraceFieldOutput:
    batch, cells = map_scores.shape
    image_zeros = torch.zeros(
        (batch, spec.image_height, spec.image_width), dtype=map_scores.dtype
    )
    cell_zeros = torch.zeros_like(map_scores)
    return TraceFieldOutput(
        root_energy=image_zeros,
        support_energy=image_zeros.clone(),
        logZ_positive=cell_zeros,
        logZ_total=cell_zeros,
        p_nonempty=p_nonempty,
        log_cardinality=cell_zeros,
        map_energy=cell_zeros,
        map_log_joint_posterior=map_scores,
        map_root=torch.zeros((batch, cells, 2), dtype=torch.long),
        map_intervals=torch.full(
            (batch, cells, spec.local_height, 2), -1, dtype=torch.long
        ),
        map_support=map_support,
        root_marginal=None,
        support_marginal=None,
        geometry_sha256=spec.sha256,
        logk_cache_sha256="synthetic-renderer-output",
    )


def _put_global_pixels(
    local_support: torch.Tensor,
    spec: TraceGeometrySpec,
    cell: int,
    coordinates: tuple[tuple[int, int], ...],
) -> None:
    index_grid = spec.global_index_grid()[cell]
    for y, x in coordinates:
        positions = torch.nonzero(
            index_grid.eq(y * spec.image_width + x), as_tuple=False
        )
        assert positions.shape == (1, 2)
        local_y, local_x = positions[0].tolist()
        local_support[0, cell, local_y, local_x] = True


def _global_union(
    local_support: torch.Tensor,
    spec: TraceGeometrySpec,
    selected_cells: tuple[int, ...],
) -> torch.Tensor:
    result = torch.zeros((spec.image_height, spec.image_width), dtype=torch.bool)
    indices = spec.global_index_grid()
    for cell in selected_cells:
        result.view(-1)[indices[cell][local_support[0, cell]]] = True
    return result


def test_renderer_uses_joint_map_score_strict_atomic_threshold_and_overlap_max() -> None:
    spec = TraceGeometrySpec(
        image_height=2,
        image_width=4,
        cell_size=2,
        max_down=0,
        max_left=1,
        max_right=1,
        margin=0,
    )
    local_support = torch.zeros(
        (1, spec.number_of_cells, spec.local_height, spec.local_width),
        dtype=torch.bool,
    )
    atom_a = ((0, 0), (1, 1))
    atom_b = ((0, 2), (1, 1), (1, 2))
    _put_global_pixels(local_support, spec, 0, atom_a)
    _put_global_pixels(local_support, spec, 1, atom_b)
    map_scores = torch.tensor([[-2.0, -0.5]])
    # Deliberately reverse the confidence ordering.  If the renderer uses
    # p_nonempty instead of the joint MAP posterior, every score check fails.
    p_nonempty = torch.tensor([[0.99, 0.01]])
    output = _fake_renderer_output(spec, local_support, map_scores, p_nonempty)

    rendered = render_trace_atoms(output, spec, cell_chunk_size=1)
    assert rendered.threshold_operator == "score > threshold"
    assert rendered.threshold_domain == "[background_score, +inf)"
    assert rendered.scores[0, 0, 0].item() == pytest.approx(-2.0)
    assert rendered.scores[0, 0, 2].item() == pytest.approx(-0.5)
    assert rendered.scores[0, 1, 1].item() == pytest.approx(-0.5)
    assert rendered.scores[0, 0, 1].item() == rendered.background_score

    expected_a = _global_union(local_support, spec, (0,))
    expected_b = _global_union(local_support, spec, (1,))
    expected_union = expected_a | expected_b
    assert torch.equal(rendered.binary(-1.0)[0], expected_b)
    assert torch.equal(rendered.binary(-3.0)[0], expected_union)
    # Strict > rejects an atom whose joint score exactly equals the threshold.
    assert not bool(rendered.binary(-0.5).any())
    # The finite sentinel is itself the minimum valid evaluator candidate.
    assert torch.equal(rendered.binary(rendered.background_score)[0], expected_union)
    background = torch.tensor(rendered.background_score, dtype=rendered.scores.dtype)
    below_background = torch.nextafter(background, torch.tensor(-torch.inf))
    assert bool(torch.isfinite(below_background))
    with pytest.raises(TraceModelError, match="background_score"):
        rendered.binary(below_background)


def test_exact_nll_rejects_malformed_target_types_shapes_and_supports() -> None:
    spec = TraceGeometrySpec(
        image_height=2,
        image_width=2,
        cell_size=2,
        max_down=0,
        max_left=0,
        max_right=0,
        margin=0,
    )
    field = TraceAtomicField(spec, field_chunk_size=1)
    output = field(
        torch.zeros((1, 2, 2)),
        torch.zeros((1, 2, 2)),
        return_map=False,
    )
    one_pixel = torch.zeros((1, 2, 2), dtype=torch.bool)
    one_pixel[0, 0, 0] = True
    two_pixels = one_pixel.clone()
    two_pixels[0, 0, 1] = True

    malformed = (
        EncodedTraceTargets(
            number_of_cells=spec.number_of_cells,
            positive_cell_indices=torch.tensor([[0]], dtype=torch.long),
            root_local_y=torch.tensor([0], dtype=torch.long),
            root_local_x=torch.tensor([0], dtype=torch.long),
            support_local=one_pixel,
        ),
        EncodedTraceTargets(
            number_of_cells=spec.number_of_cells,
            positive_cell_indices=torch.tensor([0], dtype=torch.long),
            root_local_y=torch.tensor([0.0]),
            root_local_x=torch.tensor([0], dtype=torch.long),
            support_local=one_pixel,
        ),
        EncodedTraceTargets(
            number_of_cells=True,
            positive_cell_indices=torch.tensor([0], dtype=torch.long),
            root_local_y=torch.tensor([0], dtype=torch.long),
            root_local_x=torch.tensor([0], dtype=torch.long),
            support_local=one_pixel,
        ),
        EncodedTraceTargets(
            number_of_cells=spec.number_of_cells,
            positive_cell_indices=torch.tensor([0], dtype=torch.long),
            root_local_y=torch.tensor([0], dtype=torch.long),
            root_local_x=torch.tensor([0], dtype=torch.long),
            support_local=torch.zeros_like(one_pixel),
        ),
        EncodedTraceTargets(
            number_of_cells=spec.number_of_cells,
            positive_cell_indices=torch.tensor([0], dtype=torch.long),
            root_local_y=torch.tensor([0], dtype=torch.long),
            root_local_x=torch.tensor([1], dtype=torch.long),
            support_local=two_pixels,
        ),
    )
    for target in malformed:
        with pytest.raises(TraceModelError):
            field.exact_nll(output, (target,))


@dataclass(frozen=True)
class _Checkpoint:
    path: Path


def _expanded_state() -> OrderedDict[str, torch.Tensor]:
    with torch.random.fork_rng(devices=[]):
        canonical = MSHNet(3).state_dict()
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for index, (key, reference) in enumerate(canonical.items()):
        if not reference.dtype.is_floating_point:
            value: float | int = 0
        elif key.endswith("running_var") or (
            key.endswith(".weight")
            and (".bn1." in key or ".bn2." in key or ".shortcut.1." in key)
        ):
            value = 1.0
        elif key.endswith(".weight"):
            value = (index % 3 + 1) * 1.0e-4
        else:
            value = 0.0
        result[key] = torch.tensor(value, dtype=reference.dtype).expand(reference.shape)
    return result


@pytest.fixture(scope="module")
def clean_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> _Checkpoint:
    path = tmp_path_factory.mktemp("trace_mshnet") / "canonical_synthetic.pt"
    state = _expanded_state()
    torch.save(
        {
            "net": state,
            "method_meta": {
                "method": "MSHNet",
                "model_type": "mshnet",
                "dataset_dir": f"/synthetic/datasets/{DATASET}",
                "seed": SEED,
                "train_split_sha256": TRAIN_SPLIT_SHA256,
                "val_split_sha256": VAL_SPLIT_SHA256,
                "selection_split": "validation",
                "evaluation_split": "validation",
                "protocol": "internal_dev_holdout_v1",
            },
            "epoch": 23,
            "iou": 0.731,
        },
        path,
    )
    assert all(f"{name}.weight" in state for name in FORBIDDEN_HEADS)
    assert path.stat().st_size < 512 * 1024
    return _Checkpoint(path)


def test_full_trace_model_has_no_old_heads_and_optimizer_cannot_mutate_front(
    clean_checkpoint: _Checkpoint,
) -> None:
    spec = TraceGeometrySpec(
        image_height=16,
        image_width=16,
        cell_size=8,
        max_down=0,
        max_left=0,
        max_right=0,
        margin=0,
    )
    model = TRACEMSHNet(
        baseline_checkpoint=clean_checkpoint.path,
        geometry=spec,
        positive_cell_prior=0.05,
        field_chunk_size=3,
        expected_dataset=DATASET,
        expected_seed=SEED,
        expected_train_split_sha256=TRAIN_SPLIT_SHA256,
        expected_val_split_sha256=VAL_SPLIT_SHA256,
    )
    assert model.trainable_parameter_count == 306
    for name, _module in model.named_modules():
        assert FORBIDDEN_HEADS.isdisjoint(name.split("."))

    model.train()
    assert all(not module.training for module in model.front.modules())
    front_hash = model.assert_front_integrity()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    original_bias = model.potential_map[2].bias.detach().clone()
    image = torch.linspace(-0.5, 0.5, 3 * 16 * 16).reshape(1, 3, 16, 16)
    output = model(image, return_map=False)
    loss = model.exact_nll(output, (_empty_target(spec),)).loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert not torch.equal(model.potential_map[2].bias, original_bias)
    assert model.assert_front_integrity() == front_hash
    assert all(not module.training for module in model.front.modules())
