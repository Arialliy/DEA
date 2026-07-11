#!/usr/bin/env python3
"""Aggregate exploratory MSHNet feature-survival audits without independence claims."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SCALAR_STAGES = (
    "mask0",
    "mask1",
    "mask2",
    "mask3",
    "s0",
    "s1",
    "s2",
    "s3",
    "c0",
    "c1",
    "c2",
    "c3",
    "z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def summarize_stage(records, stage: str) -> dict[str, object]:
    values = [record["stages"][stage] for record in records]
    available = [value for value in values if value["available"]]
    state_counts = {
        state: sum(value["state"] == state for value in values)
        for state in ("distinct", "uncertain", "background_like", "undefined")
    }
    directional = [
        float(value["directional_auc"])
        for value in available
        if value["directional_auc"] is not None
    ]
    return {
        "targets": len(values),
        "available": len(available),
        "state_counts": state_counts,
        "distinct_rate_among_available": (
            state_counts["distinct"] / len(available) if available else 0.0
        ),
        "median_rank": (
            float(np.median([float(value["rank"]) for value in available]))
            if available
            else None
        ),
        "directionally_positive": sum(value > 0.5 for value in directional),
        "directionally_positive_rate": (
            sum(value > 0.5 for value in directional) / len(directional)
            if directional
            else None
        ),
        "median_directional_auc": (
            float(np.median(directional)) if directional else None
        ),
    }


def summarize_records(records: list[dict[str, object]], stages) -> dict[str, object]:
    outcomes = {}
    for outcome in ("no_response", "matched_control"):
        selected = [record for record in records if record["outcome"] == outcome]
        outcomes[outcome] = {
            stage: summarize_stage(selected, stage) for stage in stages
        }

    no_response = [
        record for record in records if record["outcome"] == "no_response"
    ]
    matched = [
        record for record in records if record["outcome"] == "matched_control"
    ]

    def distinct(record, stage):
        return record["stages"][stage]["state"] == "distinct"

    def directionally_positive(record, stage):
        auc = record["stages"][stage]["directional_auc"]
        return auc is not None and float(auc) > 0.5

    key = {
        "no_response_targets": len(no_response),
        "matched_control_targets": len(matched),
        "no_response_input_distinct": sum(
            distinct(record, "input") for record in no_response
        ),
        "no_response_m_available": sum(
            record["stages"]["m"]["available"] for record in no_response
        ),
        "no_response_m_distinct": sum(
            distinct(record, "m") for record in no_response
        ),
        "no_response_d0_distinct": sum(
            distinct(record, "d0") for record in no_response
        ),
        "no_response_any_decoder_distinct": sum(
            any(distinct(record, stage) for stage in ("d0", "d1", "d2", "d3"))
            for record in no_response
        ),
        "no_response_mask0_distinct": sum(
            distinct(record, "mask0") for record in no_response
        ),
        "no_response_any_native_side_distinct": sum(
            any(distinct(record, stage) for stage in ("mask0", "mask1", "mask2", "mask3"))
            for record in no_response
        ),
        "no_response_any_native_side_directionally_positive": sum(
            any(
                directionally_positive(record, stage)
                for stage in ("mask0", "mask1", "mask2", "mask3")
            )
            for record in no_response
        ),
        "no_response_z_distinct": sum(
            distinct(record, "z") for record in no_response
        ),
        "no_response_z_directionally_positive": sum(
            directionally_positive(record, "z") for record in no_response
        ),
        "no_response_d0_to_mask0_drop": sum(
            distinct(record, "d0") and not distinct(record, "mask0")
            for record in no_response
        ),
        "no_response_d0_to_z_drop": sum(
            distinct(record, "d0") and not distinct(record, "z")
            for record in no_response
        ),
        "matched_d0_distinct": sum(
            distinct(record, "d0") for record in matched
        ),
        "matched_z_distinct": sum(
            distinct(record, "z") for record in matched
        ),
    }
    key["no_response_z_distinct_and_directionally_positive"] = sum(
        distinct(record, "z") and directionally_positive(record, "z")
        for record in no_response
    )
    key["no_response_z_distinct_and_reverse_or_tied"] = sum(
        distinct(record, "z") and not directionally_positive(record, "z")
        for record in no_response
    )
    key["no_response_z_nondistinct_and_directionally_positive"] = sum(
        not distinct(record, "z") and directionally_positive(record, "z")
        for record in no_response
    )
    key["no_response_z_nondistinct_and_reverse_or_tied"] = sum(
        not distinct(record, "z") and not directionally_positive(record, "z")
        for record in no_response
    )
    margins = sorted(
        float(record["stages"]["z"]["target_peak_margin"])
        for record in no_response
        if record["stages"]["z"]["target_peak_margin"] is not None
    )
    key["no_response_z_nonnegative_peak_margin"] = sum(
        margin >= 0 for margin in margins
    )
    key["no_response_z_peak_margin_quantiles"] = (
        {
            "min": float(np.min(margins)),
            "q25": float(np.quantile(margins, 0.25)),
            "median": float(np.median(margins)),
            "q75": float(np.quantile(margins, 0.75)),
            "max": float(np.max(margins)),
        }
        if margins
        else None
    )
    return {"stages": outcomes, "key": key}


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    if output_json.exists() or output_md.exists():
        raise FileExistsError("refusing to overwrite summary outputs")

    runs = []
    all_records = []
    grouped = defaultdict(list)
    protocol = None
    main_path = None
    for input_name in args.inputs:
        path = Path(input_name).resolve()
        payload = json.loads(path.read_text())
        if payload.get("schema") != "mshnet_feature_survival_audit_v1":
            raise RuntimeError("unexpected feature-audit schema in %s" % path)
        current_protocol = (
            float(payload["threshold_probability"]),
            int(payload["required_geometry_controls"]),
            int(payload["max_candidate_controls"]),
            int(payload["matched_controls_per_miss"]),
            payload["split"],
        )
        if protocol is None:
            protocol = current_protocol
            main_path = tuple(payload["main_path"])
        elif current_protocol != protocol or tuple(payload["main_path"]) != main_path:
            raise RuntimeError("feature-audit protocols do not match")
        dataset = Path(payload["dataset_dir"]).name
        seed = Path(payload["source_run_dir"]).name.removeprefix("seed_")
        records = payload["records"]
        if sum(record["outcome"] == "no_response" for record in records) != int(
            payload["num_no_response_targets"]
        ):
            raise RuntimeError("no-response record count drifted in %s" % path)
        runs.append(
            {
                "dataset": dataset,
                "seed": seed,
                "source": str(path),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "no_response_targets": payload["num_no_response_targets"],
                "matched_controls": payload["num_selected_matched_controls"],
            }
        )
        for record in records:
            tagged = dict(record)
            tagged["dataset"] = dataset
            tagged["seed"] = seed
            all_records.append(tagged)
            grouped[dataset].append(tagged)

    if protocol is None or main_path is None:
        raise RuntimeError("no inputs were supplied")
    stages = (
        *main_path,
        *SCALAR_STAGES,
    )
    dataset_summaries = {
        dataset: summarize_records(records, stages)
        for dataset, records in sorted(grouped.items())
    }
    overall = summarize_records(all_records, stages)
    no_response_unique_clusters = {
        (
            record["dataset"],
            record["sample_name"],
            int(record["target_index"]),
        )
        for record in all_records
        if record["outcome"] == "no_response"
    }
    result = {
        "schema": "mshnet_feature_survival_summary_v1",
        "exploratory_only": True,
        "protocol": {
            "threshold_probability": protocol[0],
            "required_geometry_controls": protocol[1],
            "max_candidate_controls": protocol[2],
            "matched_controls_per_miss": protocol[3],
            "split": protocol[4],
            "distinct_rank_threshold": 0.95,
            "background_like_rank_threshold": 0.5,
        },
        "runs": runs,
        "paired_no_response_observations": overall["key"]["no_response_targets"],
        "unique_dataset_sample_target_clusters": len(no_response_unique_clusters),
        "datasets": dataset_summaries,
        "overall": overall,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# MSHNet exploratory feature-survival summary",
        "",
        "Operating probability threshold: `%.3f`; geometry-matched controls per target: `%d`."
        % (protocol[0], protocol[1]),
        "",
        "| Dataset | No-response | Input distinct | Middle distinct | D0 distinct | Any side distinct | Mask0 distinct | Final distinct | Final AUC > 0.5 | D0→final drop | Matched D0/final distinct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, summary in dataset_summaries.items():
        key = summary["key"]
        lines.append(
            "| %s | %d | %d | %d/%d | %d | %d | %d | %d | %d | %d | %d/%d |"
            % (
                dataset,
                key["no_response_targets"],
                key["no_response_input_distinct"],
                key["no_response_m_distinct"],
                key["no_response_m_available"],
                key["no_response_d0_distinct"],
                key["no_response_any_native_side_distinct"],
                key["no_response_mask0_distinct"],
                key["no_response_z_distinct"],
                key["no_response_z_directionally_positive"],
                key["no_response_d0_to_z_drop"],
                key["matched_d0_distinct"],
                key["matched_z_distinct"],
            )
        )
    key = overall["key"]
    total = key["no_response_targets"]
    lines.extend(
        [
            "",
            "Across %d paired no-response observations, D0 is distinct for %d (%.2f%%), at least one native side logit is distinct for %d (%.2f%%), and the final logit is distinct for %d (%.2f%%)."
            % (
                total,
                key["no_response_d0_distinct"],
                100.0 * key["no_response_d0_distinct"] / total if total else 0.0,
                key["no_response_any_native_side_distinct"],
                100.0
                * key["no_response_any_native_side_distinct"]
                / total
                if total
                else 0.0,
                key["no_response_z_distinct"],
                100.0 * key["no_response_z_distinct"] / total if total else 0.0,
            ),
            "",
            "Final target-vs-local-background AUC exceeds 0.5 for %d/%d (%.2f%%), but fixed-threshold peak margin is non-negative for %d/%d. D0→final distinctness drops occur for %d/%d."
            % (
                key["no_response_z_directionally_positive"],
                total,
                100.0
                * key["no_response_z_directionally_positive"]
                / total
                if total
                else 0.0,
                key["no_response_z_nonnegative_peak_margin"],
                total,
                key["no_response_d0_to_z_drop"],
                total,
            ),
            "",
            "This is an exploratory, GT-conditioned diagnostic. Distinctness is an unsigned geometry-null rank, not a deployable classifier or causal attribution. Checkpoints were selected on the same internal validation split, and repeated targets across seeds are paired observations rather than independent samples.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")
    print("wrote %s and %s" % (output_json, output_md))


if __name__ == "__main__":
    main()
