#!/usr/bin/env python3
"""Evaluate saved TRACE/dense score maps under the locked paper protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.trace_evaluation import (  # noqa: E402
    DEFAULT_FA_BUDGETS_PER_MILLION_PIXELS,
    DEFAULT_MAX_UNIQUE_SCORES,
    TraceEvaluationError,
    dump_trace_evaluation_json,
    evaluate_trace_bundles,
    load_npz_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select exact unique-score thresholds on an explicit development "
            "bundle with the legacy matcher, then evaluate the same locked "
            "thresholds on dev/test with legacy and Hungarian matching."
        )
    )
    parser.add_argument(
        "--dev-bundle",
        required=True,
        type=Path,
        help="Development NPZ used for threshold selection (required).",
    )
    parser.add_argument(
        "--test-bundle",
        required=True,
        type=Path,
        help="Untouched test NPZ used only after thresholds are locked (required).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON report (required).",
    )
    parser.add_argument(
        "--score-key",
        default="scores",
        help="NPZ key containing HxW/NxHxW/Nx1xHxW score maps.",
    )
    parser.add_argument(
        "--target-key",
        default="targets",
        help="NPZ key containing shape-matched binary target maps.",
    )
    parser.add_argument(
        "--sample-id-key",
        default="sample_ids",
        help="Optional NPZ key containing one unique sample id per map.",
    )
    parser.add_argument(
        "--fa-budgets",
        nargs="+",
        type=float,
        default=list(DEFAULT_FA_BUDGETS_PER_MILLION_PIXELS),
        metavar="FA_PER_MPIX",
        help="Requested development FA/Mpix budgets (default: 1 5 10 20).",
    )
    parser.add_argument(
        "--max-unique-scores",
        type=int,
        default=DEFAULT_MAX_UNIQUE_SCORES,
        help=(
            "Hard cap for exact pooled development unique scores; exceeding it "
            "fails without quantization (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--centroid-radius",
        type=float,
        default=3.0,
        help="Strict component centroid match radius in pixels (default: 3).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=2,
        help="2-D component connectivity; protocol requires 2 (8-connectivity).",
    )
    parser.add_argument(
        "--run-provenance-json",
        type=Path,
        help="Optional finite JSON object copied into the report (checkpoint/run ids).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly permit replacing an existing output report.",
    )
    return parser


def _load_run_provenance(path: Path | None) -> dict | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceEvaluationError(
            f"failed to load run provenance JSON {resolved}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TraceEvaluationError("run provenance JSON must contain an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dev_bundle = load_npz_bundle(
            args.dev_bundle,
            score_key=args.score_key,
            target_key=args.target_key,
            sample_id_key=args.sample_id_key,
        )
        test_bundle = load_npz_bundle(
            args.test_bundle,
            score_key=args.score_key,
            target_key=args.target_key,
            sample_id_key=args.sample_id_key,
        )
        report = evaluate_trace_bundles(
            dev_bundle,
            test_bundle,
            fa_budgets_per_million_pixels=args.fa_budgets,
            max_unique_scores=args.max_unique_scores,
            centroid_radius=args.centroid_radius,
            connectivity=args.connectivity,
            run_provenance=_load_run_provenance(args.run_provenance_json),
        )
        output = dump_trace_evaluation_json(
            report, args.output, overwrite=args.overwrite
        )
    except TraceEvaluationError as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
