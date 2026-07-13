#!/usr/bin/env python3
"""Exhaustively falsify Component-Exclusion Margin Cut as a novel primitive.

For a binary submodular Potts energy, removing one foreground connected
component from a MAP solution has an exact closed-form energy increment.  The
group-exclusion min-marginal is not a new structured confidence and does not
require another min-cut: it is exactly the component sum of unary flip costs
minus its cut boundary.  This audit verifies that collapse exhaustively on
small 8-neighbour graphs and on deterministic 3x3 stress cases.

The artifact is a NO-GO record.  It must not be interpreted as a model result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dea.cemc.collapse_audit.v1"


class CEMCCollapseAuditError(RuntimeError):
    """The finite collapse audit found an invalid input or counterexample."""


def grid_edges(height: int, width: int, connectivity: int = 8) -> tuple[tuple[int, int], ...]:
    if height < 1 or width < 1:
        raise ValueError("grid dimensions must be positive")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    directions = [(0, 1), (1, 0)]
    if connectivity == 8:
        directions.extend(((1, -1), (1, 1)))
    edges: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            source = row * width + col
            for drow, dcol in directions:
                target_row = row + drow
                target_col = col + dcol
                if 0 <= target_row < height and 0 <= target_col < width:
                    target = target_row * width + target_col
                    edges.append((source, target))
    return tuple(edges)


def energy(
    state: np.ndarray,
    unary_one_minus_zero: np.ndarray,
    pairwise: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> float:
    labels = np.asarray(state, dtype=np.bool_)
    unary = np.asarray(unary_one_minus_zero, dtype=np.float64)
    weights = np.asarray(pairwise, dtype=np.float64)
    if labels.ndim != 1 or unary.shape != labels.shape:
        raise ValueError("state and unary arrays must be aligned 1-D arrays")
    if weights.shape != (len(edges),):
        raise ValueError("pairwise array does not match the edge inventory")
    if not np.isfinite(unary).all() or not np.isfinite(weights).all():
        raise ValueError("energies must be finite")
    if np.any(weights < 0):
        raise ValueError("Potts pairwise weights must be non-negative")
    value = float(unary[labels].sum())
    value += float(
        sum(
            weights[index]
            for index, (left, right) in enumerate(edges)
            if bool(labels[left]) != bool(labels[right])
        )
    )
    return value


def foreground_components(
    state: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    labels = np.asarray(state, dtype=np.bool_)
    adjacency = [[] for _ in range(int(labels.size))]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    unseen = {index for index, value in enumerate(labels) if bool(value)}
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(tuple(sorted(component)))
    return tuple(components)


def enumerate_states(node_count: int) -> np.ndarray:
    if node_count < 1 or node_count > 20:
        raise ValueError("finite audit supports 1..20 nodes")
    return np.asarray(
        tuple(itertools.product((False, True), repeat=node_count)),
        dtype=np.bool_,
    )


def audit_energy_instance(
    unary: np.ndarray,
    pairwise: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> dict[str, int]:
    unary = np.asarray(unary, dtype=np.float64)
    pairwise = np.asarray(pairwise, dtype=np.float64)
    states = enumerate_states(int(unary.size))
    values = np.asarray(
        [energy(state, unary, pairwise, edges) for state in states],
        dtype=np.float64,
    )
    minimum = float(values.min())
    # Deterministic lexicographic tie semantics.  The theorem applies to every
    # MAP; choosing one makes the finite artifact reproducible.
    map_index = int(np.flatnonzero(values == minimum)[0])
    map_state = states[map_index]
    components = foreground_components(map_state, edges)
    checked_components = 0
    zero_margin_components = 0
    marks: list[tuple[tuple[int, ...], float]] = []
    for component in components:
        component_array = np.asarray(component, dtype=np.int64)
        feasible = np.logical_not(states[:, component_array]).all(axis=1)
        exact_margin = float(values[feasible].min() - minimum)

        removed = map_state.copy()
        removed[component_array] = False
        removal_margin = energy(removed, unary, pairwise, edges) - minimum
        boundary_weight = float(
            sum(
                pairwise[index]
                for index, (left, right) in enumerate(edges)
                if (left in component) != (right in component)
            )
        )
        # ``unary`` stores D_i(1)-D_i(0).  Flipping a foreground node to
        # background therefore contributes D_i(0)-D_i(1) = -unary_i.
        analytic_margin = float(-unary[component_array].sum() - boundary_weight)
        if not (
            abs(exact_margin - removal_margin) <= 1e-10
            and abs(exact_margin - analytic_margin) <= 1e-10
            and exact_margin >= -1e-10
        ):
            raise CEMCCollapseAuditError(
                "component exclusion margin did not collapse to its analytic aggregate"
            )
        checked_components += 1
        zero_margin_components += int(abs(exact_margin) <= 1e-10)
        marks.append((component, max(0.0, exact_margin)))

    # Thresholding the fixed MAP components must form a nested set filtration
    # and can never modify the support of a surviving component.
    thresholds = sorted({-1.0, *(mark for _, mark in marks), *(mark + 1e-9 for _, mark in marks)})
    prior_support: set[int] | None = None
    for threshold in thresholds:
        support = {
            node
            for component, mark in marks
            if mark > threshold
            for node in component
        }
        if prior_support is not None and not support.issubset(prior_support):
            raise CEMCCollapseAuditError("component threshold filtration is not nested")
        prior_support = support
    return {
        "map_tie_count": int(np.count_nonzero(values == minimum)),
        "map_foreground_components": len(components),
        "checked_components": checked_components,
        "zero_margin_components": zero_margin_components,
        "threshold_states_checked": len(thresholds),
    }


def exhaustive_two_by_two() -> dict[str, int]:
    edges = grid_edges(2, 2, connectivity=8)
    configurations = 0
    components = 0
    ties = 0
    threshold_states = 0
    for unary_values in itertools.product((-1.0, 0.0, 1.0), repeat=4):
        unary = np.asarray(unary_values, dtype=np.float64)
        for pairwise_values in itertools.product((0.0, 1.0), repeat=len(edges)):
            result = audit_energy_instance(
                unary,
                np.asarray(pairwise_values, dtype=np.float64),
                edges,
            )
            configurations += 1
            components += result["checked_components"]
            ties += int(result["map_tie_count"] > 1)
            threshold_states += result["threshold_states_checked"]
    return {
        "configurations": configurations,
        "components": components,
        "map_tie_configurations": ties,
        "threshold_states": threshold_states,
    }


def deterministic_three_by_three(cases: int = 512) -> dict[str, int]:
    if cases < 1:
        raise ValueError("cases must be positive")
    edges = grid_edges(3, 3, connectivity=8)
    rng = np.random.default_rng(20260713)
    components = 0
    ties = 0
    threshold_states = 0
    for _ in range(cases):
        unary = rng.integers(-4, 5, size=9).astype(np.float64) / 2.0
        pairwise = rng.integers(0, 5, size=len(edges)).astype(np.float64) / 2.0
        result = audit_energy_instance(unary, pairwise, edges)
        components += result["checked_components"]
        ties += int(result["map_tie_count"] > 1)
        threshold_states += result["threshold_states_checked"]
    return {
        "configurations": cases,
        "components": components,
        "map_tie_configurations": ties,
        "threshold_states": threshold_states,
    }


def run_audit() -> dict[str, object]:
    exhaustive = exhaustive_two_by_two()
    stress = deterministic_three_by_three()
    return {
        "schema": SCHEMA,
        "status": "complete_no_go",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "finite mathematical falsification; no neural model or performance claim",
        "energy": (
            "sum_i D_i(y_i) + sum_(i,j) w_ij |y_i-y_j|, w_ij >= 0"
        ),
        "claim_checked": (
            "for every MAP foreground connected component C, the constrained "
            "exclusion min-marginal equals E(y*\\C)-E(y*) and equals the unary "
            "flip-cost sum minus the Potts boundary weight"
        ),
        "two_by_two_8_neighbour_exhaustive": exhaustive,
        "three_by_three_8_neighbour_deterministic_stress": stress,
        "total_configurations": exhaustive["configurations"] + stress["configurations"],
        "total_components": exhaustive["components"] + stress["components"],
        "counterexamples": 0,
        "decision": {
            "component_exclusion_margin_cut_as_main_method": "NO-GO",
            "reason": (
                "the proposed structured mark is exactly a conventional component "
                "aggregate of unary and boundary terms; it is not a second inference "
                "problem or a new component-valued random variable"
            ),
            "gpu_training_authorized": False,
            "retain_only_as_control": True,
        },
    }


def _markdown(summary: dict[str, object]) -> str:
    exhaustive = summary["two_by_two_8_neighbour_exhaustive"]
    stress = summary["three_by_three_8_neighbour_deterministic_stress"]
    assert isinstance(exhaustive, dict) and isinstance(stress, dict)
    return "\n".join(
        (
            "# Component-Exclusion Margin Cut collapse audit",
            "",
            "Decision: **NO-GO as an AAAI main method**.",
            "",
            "For a local binary submodular Potts energy, the exact group-exclusion min-marginal of a MAP foreground component is its unary flip-cost sum minus its cut-boundary weight. It therefore collapses to a conventional connected-component aggregate and needs no additional constrained min-cut.",
            "",
            f"- Exhaustive 2×2 8-neighbour configurations: {exhaustive['configurations']}",
            f"- Deterministic 3×3 stress configurations: {stress['configurations']}",
            f"- MAP components checked: {summary['total_components']}",
            f"- Counterexamples: {summary['counterexamples']}",
            "- GPU model training authorized: no",
            "",
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="repro_runs/gate_l/cemc_collapse_audit_v1",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    summary = run_audit()
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
