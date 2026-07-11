#!/usr/bin/env python3
"""Aggregate paired CCSR Gate-C1 ledger audits without inventing results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


COUNT_FIELDS = (
    "images",
    "num_gt",
    "hungarian_matches",
    "unmatched_gt",
    "no_response_gt",
    "centroid_miss_gt",
    "bridge_candidate_count",
    "images_with_bridge_candidate",
    "split_prediction_count",
    "unmatched_pred_components",
    "unmatched_pred_area",
)

SCALE_COUNT_FIELDS = (
    "no_response_with_any_side_support",
    "no_response_with_any_side_centroid",
    "no_response_matched_by_any_side",
    "no_response_recoverable_by_global_subset",
    "no_response_absent_from_all_sides",
)

SCALE_INTERSECTION_FIELDS = (
    "no_response_side_and_subset",
    "no_response_side_only",
    "no_response_subset_only",
    "no_response_neither_side_nor_subset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def finalize(counts: dict[str, float]) -> dict[str, float | int]:
    result = {key: int(counts.get(key, 0)) for key in COUNT_FIELDS}
    result["assignment_conflict_gt"] = (
        result["unmatched_gt"]
        - result["no_response_gt"]
        - result["centroid_miss_gt"]
    )
    if result["assignment_conflict_gt"] < 0:
        raise RuntimeError("unmatched GT error taxonomy became inconsistent")
    result["hungarian_pd"] = (
        result["hungarian_matches"] / result["num_gt"]
        if result["num_gt"]
        else 0.0
    )
    result["no_response_share_of_misses"] = (
        result["no_response_gt"] / result["unmatched_gt"]
        if result["unmatched_gt"]
        else 0.0
    )
    result["assignment_conflict_share_of_misses"] = (
        result["assignment_conflict_gt"] / result["unmatched_gt"]
        if result["unmatched_gt"]
        else 0.0
    )
    result["bridge_candidate_image_rate"] = (
        result["images_with_bridge_candidate"] / result["images"]
        if result["images"]
        else 0.0
    )
    return result


def finalize_scale(
    counts: dict[str, float],
    side_support_counts: dict[str, float],
    recovering_subset_counts: dict[str, float],
    intersection_counts: dict[str, float],
    *,
    no_response_gt: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        key: int(counts.get(key, 0)) for key in SCALE_COUNT_FIELDS
    }
    bounded_fields = tuple(
        int(result[key]) for key in SCALE_COUNT_FIELDS
    )
    if any(value < 0 or value > no_response_gt for value in bounded_fields):
        raise RuntimeError("scale-diagnostic counts exceed no-response count")
    if (
        int(result["no_response_with_any_side_support"])
        + int(result["no_response_absent_from_all_sides"])
        != no_response_gt
    ):
        raise RuntimeError(
            "side-support taxonomy does not partition no-response targets"
        )
    if int(result["no_response_matched_by_any_side"]) > int(
        result["no_response_with_any_side_centroid"]
    ):
        raise RuntimeError("side matches exceed centroid-legal side targets")
    result["side_support_counts"] = {
        str(scale): int(side_support_counts.get(str(scale), 0))
        for scale in range(4)
    }
    result["recovering_subset_counts"] = {
        str(subset): int(recovering_subset_counts.get(str(subset), 0))
        for subset in range(1, 15)
    }
    result.update(
        {
            key: int(intersection_counts.get(key, 0))
            for key in SCALE_INTERSECTION_FIELDS
        }
    )
    if sum(int(result[key]) for key in SCALE_INTERSECTION_FIELDS) != no_response_gt:
        raise RuntimeError(
            "side/subset intersections do not partition no-response targets"
        )
    if (
        int(result["no_response_side_and_subset"])
        + int(result["no_response_side_only"])
        != int(result["no_response_with_any_side_support"])
    ):
        raise RuntimeError("intersection counts disagree with side support")
    if (
        int(result["no_response_side_and_subset"])
        + int(result["no_response_subset_only"])
        != int(result["no_response_recoverable_by_global_subset"])
    ):
        raise RuntimeError("intersection counts disagree with subset recovery")
    result["any_side_support_rate"] = (
        int(result["no_response_with_any_side_support"])
        / no_response_gt
        if no_response_gt
        else 0.0
    )
    result["any_side_match_rate"] = (
        int(result["no_response_matched_by_any_side"]) / no_response_gt
        if no_response_gt
        else 0.0
    )
    result["global_subset_recoverable_rate"] = (
        int(result["no_response_recoverable_by_global_subset"])
        / no_response_gt
        if no_response_gt
        else 0.0
    )
    result["absent_from_all_sides_rate"] = (
        int(result["no_response_absent_from_all_sides"])
        / no_response_gt
        if no_response_gt
        else 0.0
    )
    result["neither_side_nor_subset_rate"] = (
        int(result["no_response_neither_side_nor_subset"])
        / no_response_gt
        if no_response_gt
        else 0.0
    )
    return result


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    if output_json.exists() or output_md.exists():
        raise FileExistsError("refusing to overwrite summary outputs")

    runs = []
    grouped = defaultdict(lambda: defaultdict(float))
    overall = defaultdict(float)
    grouped_scale = defaultdict(lambda: defaultdict(float))
    overall_scale = defaultdict(float)
    grouped_side_support = defaultdict(lambda: defaultdict(float))
    overall_side_support = defaultdict(float)
    grouped_subset_recovery = defaultdict(lambda: defaultdict(float))
    overall_subset_recovery = defaultdict(float)
    grouped_intersections = defaultdict(lambda: defaultdict(float))
    overall_intersections = defaultdict(float)
    scale_flags = []
    for input_name in args.inputs:
        path = Path(input_name).resolve()
        payload = json.loads(path.read_text())
        schema = payload.get("schema")
        if schema not in {
            "ccsr_component_ledger_audit_v1",
            "ccsr_component_scale_ledger_audit_v2",
        }:
            raise RuntimeError("unexpected ledger schema in %s" % path)
        has_scale_audit = (
            schema == "ccsr_component_scale_ledger_audit_v2"
            and payload.get("include_scale_audit") is True
        )
        scale_flags.append(has_scale_audit)
        matches = [
            item
            for item in payload["thresholds"]
            if abs(float(item["probability"]) - args.probability) < 1e-12
        ]
        if len(matches) != 1:
            raise RuntimeError("threshold missing or duplicated in %s" % path)
        source_run = Path(payload["source_run_dir"])
        dataset = Path(payload["dataset_dir"]).name
        seed = source_run.name.removeprefix("seed_")
        summary = matches[0]["summary"]
        counts = {key: float(summary[key]) for key in COUNT_FIELDS}
        finalized = finalize(counts)
        scale_finalized = None
        if has_scale_audit:
            scale_counts = {
                key: float(summary.get(key, 0))
                for key in SCALE_COUNT_FIELDS
            }
            side_counts = {
                str(key): float(value)
                for key, value in summary.get(
                    "side_support_counts", {}
                ).items()
            }
            subset_counts = {
                str(key): float(value)
                for key, value in summary.get(
                    "recovering_subset_counts", {}
                ).items()
            }
            selected_records = [
                record
                for record in payload.get("records", ())
                if abs(
                    float(record["probability_threshold"])
                    - args.probability
                )
                < 1e-12
            ]
            diagnostic_records = [
                item
                for record in selected_records
                for item in record.get("no_response_scale_records", ())
            ]
            if len(diagnostic_records) != int(finalized["no_response_gt"]):
                raise RuntimeError(
                    "scale records do not cover all no-response targets in %s"
                    % path
                )
            intersection_counts = {
                "no_response_side_and_subset": sum(
                    bool(item["side_support_scales"])
                    and bool(item["recovering_subsets"])
                    for item in diagnostic_records
                ),
                "no_response_side_only": sum(
                    bool(item["side_support_scales"])
                    and not item["recovering_subsets"]
                    for item in diagnostic_records
                ),
                "no_response_subset_only": sum(
                    not item["side_support_scales"]
                    and bool(item["recovering_subsets"])
                    for item in diagnostic_records
                ),
                "no_response_neither_side_nor_subset": sum(
                    not item["side_support_scales"]
                    and not item["recovering_subsets"]
                    for item in diagnostic_records
                ),
            }
            scale_finalized = finalize_scale(
                scale_counts,
                side_counts,
                subset_counts,
                intersection_counts,
                no_response_gt=int(finalized["no_response_gt"]),
            )
            for key, value in scale_counts.items():
                grouped_scale[dataset][key] += value
                overall_scale[key] += value
            for key, value in side_counts.items():
                grouped_side_support[dataset][key] += value
                overall_side_support[key] += value
            for key, value in subset_counts.items():
                grouped_subset_recovery[dataset][key] += value
                overall_subset_recovery[key] += value
            for key, value in intersection_counts.items():
                grouped_intersections[dataset][key] += value
                overall_intersections[key] += value
        runs.append(
            {
                "dataset": dataset,
                "seed": seed,
                "source": str(path),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "summary": finalized,
                **(
                    {"scale_diagnostic": scale_finalized}
                    if scale_finalized is not None
                    else {}
                ),
            }
        )
        for key, value in counts.items():
            grouped[dataset][key] += value
            overall[key] += value

    if any(scale_flags) and not all(scale_flags):
        raise RuntimeError(
            "refusing to mix scale-audited and non-scale-audited ledgers"
        )
    include_scale = bool(scale_flags and all(scale_flags))
    dataset_summaries = {
        dataset: finalize(counts)
        for dataset, counts in sorted(grouped.items())
    }
    overall_summary = finalize(overall)
    dataset_scale_summaries = {}
    overall_scale_summary = None
    if include_scale:
        dataset_scale_summaries = {
            dataset: finalize_scale(
                grouped_scale[dataset],
                grouped_side_support[dataset],
                grouped_subset_recovery[dataset],
                grouped_intersections[dataset],
                no_response_gt=int(dataset_summaries[dataset]["no_response_gt"]),
            )
            for dataset in sorted(grouped)
        }
        overall_scale_summary = finalize_scale(
            overall_scale,
            overall_side_support,
            overall_subset_recovery,
            overall_intersections,
            no_response_gt=int(overall_summary["no_response_gt"]),
        )
    result = {
        "schema": (
            "ccsr_component_scale_ledger_summary_v2"
            if include_scale
            else "ccsr_component_ledger_summary_v1"
        ),
        "probability_threshold": args.probability,
        "runs": runs,
        "datasets": dataset_summaries,
        "overall": overall_summary,
        **(
            {
                "dataset_scale_diagnostics": dataset_scale_summaries,
                "overall_scale_diagnostic": overall_scale_summary,
            }
            if include_scale
            else {}
        ),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# CCSR Gate-C1 paired component ledger summary",
        "",
        "Operating probability threshold: `%.3f`." % args.probability,
        "",
        "| Dataset | Images | GT | Pd | Miss | No response | Centroid miss | Assignment conflict | Bridge images | Split excess | Unmatched pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, summary in dataset_summaries.items():
        lines.append(
            "| %s | %d | %d | %.4f | %d | %d | %d | %d | %d | %d | %d |"
            % (
                dataset,
                summary["images"],
                summary["num_gt"],
                summary["hungarian_pd"],
                summary["unmatched_gt"],
                summary["no_response_gt"],
                summary["centroid_miss_gt"],
                summary["assignment_conflict_gt"],
                summary["images_with_bridge_candidate"],
                summary["split_prediction_count"],
                summary["unmatched_pred_components"],
            )
        )
    lines.extend(
        [
            "",
            "Overall: %d/%d misses are no-response (%.2f%%); %d/%d are one-to-one assignment conflicts (%.2f%%). Bridge proxies occur in %d/%d paired image evaluations (%.2f%%)."
            % (
                overall_summary["no_response_gt"],
                overall_summary["unmatched_gt"],
                100.0 * overall_summary["no_response_share_of_misses"],
                overall_summary["assignment_conflict_gt"],
                overall_summary["unmatched_gt"],
                100.0 * overall_summary["assignment_conflict_share_of_misses"],
                overall_summary["images_with_bridge_candidate"],
                overall_summary["images"],
                100.0 * overall_summary["bridge_candidate_image_rate"],
            ),
            "",
            "Counts are paired observations across seeds, not unique images.",
        ]
    )
    if include_scale:
        lines.extend(
            [
                "",
                "## No-response scale diagnostic",
                "",
                "| Dataset | Final no response | Any side support | Any side match | Recoverable by global subset | Absent from all sides |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset, summary in dataset_scale_summaries.items():
            lines.append(
                "| %s | %d | %d | %d | %d | %d |"
                % (
                    dataset,
                    dataset_summaries[dataset]["no_response_gt"],
                    summary["no_response_with_any_side_support"],
                    summary["no_response_matched_by_any_side"],
                    summary["no_response_recoverable_by_global_subset"],
                    summary["no_response_absent_from_all_sides"],
                )
            )
        lines.extend(
            [
                "",
                "Overall: %d/%d no-response targets have any raw side-output support (%.2f%%), %d/%d are directly matched by any side (%.2f%%), %d/%d are recoverable by at least one GT-conditioned global contribution subset (%.2f%%), and %d/%d are absent from all four side outputs (%.2f%%)."
                % (
                    overall_scale_summary[
                        "no_response_with_any_side_support"
                    ],
                    overall_summary["no_response_gt"],
                    100.0
                    * overall_scale_summary["any_side_support_rate"],
                    overall_scale_summary[
                        "no_response_matched_by_any_side"
                    ],
                    overall_summary["no_response_gt"],
                    100.0 * overall_scale_summary["any_side_match_rate"],
                    overall_scale_summary[
                        "no_response_recoverable_by_global_subset"
                    ],
                    overall_summary["no_response_gt"],
                    100.0
                    * overall_scale_summary[
                        "global_subset_recoverable_rate"
                    ],
                    overall_scale_summary[
                        "no_response_absent_from_all_sides"
                    ],
                    overall_summary["no_response_gt"],
                    100.0
                    * overall_scale_summary[
                        "absent_from_all_sides_rate"
                    ],
                ),
                "",
                "The 14-subset statistic is a GT-conditioned diagnostic upper bound, not deployable model performance. Side-scale and subset-frequency counts overlap across targets.",
                "",
                "Of the no-response targets, %d/%d (%.2f%%) have neither raw side support nor recovery under any tested contribution subset."
                % (
                    overall_scale_summary[
                        "no_response_neither_side_nor_subset"
                    ],
                    overall_summary["no_response_gt"],
                    100.0
                    * overall_scale_summary[
                        "neither_side_nor_subset_rate"
                    ],
                ),
            ]
        )
    output_md.write_text("\n".join(lines) + "\n")
    print("wrote %s and %s" % (output_json, output_md))


if __name__ == "__main__":
    main()
