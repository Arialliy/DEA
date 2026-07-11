#!/usr/bin/env python3
"""Aggregate MSHNet decision-conversion audits as diagnostic evidence only.

The summary deliberately keeps each stage separate.  In particular, raw
availability and utilization values are never turned into a cross-stage drop,
gain, or causal attribution.  Component operating points are pooled from raw
counts/areas rather than by averaging reported FA or Pd ratios.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable


AUDIT_SCHEMA = "mshnet_decision_conversion_audit_v2"
SUMMARY_SCHEMA = "mshnet_decision_conversion_summary_v1"
OUTCOMES = ("no_response", "matched_control")
STAGES = ("d0", "d1", "d2", "d3", "final")
CONTROL_POLICIES = ("geometry", "context_matched")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def _finite(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("decision-conversion audit contains non-finite data")
    return result


def _median(values: Iterable[object]) -> float | None:
    finite = [_finite(value) for value in values]
    selected = [value for value in finite if value is not None]
    return float(median(selected)) if selected else None


def _sign_counts(values: list[float]) -> dict[str, int]:
    return {
        "positive": sum(value > 0.0 for value in values),
        "negative": sum(value < 0.0 for value in values),
        "zero": sum(value == 0.0 for value in values),
    }


def _canonical_budget(value: object) -> str:
    number = _finite(value)
    if number is None or number < 0.0:
        raise RuntimeError("FA budgets must be finite and non-negative")
    return "%.12g" % number


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_control_d0(
    records: list[dict[str, object]], policy: str
) -> dict[str, object]:
    """Summarize D0 distinctness with an explicit availability denominator."""

    eligible = []
    policy_available = 0
    for record in records:
        control = record["controls"][policy]
        if bool(control["available"]):
            policy_available += 1
        d0 = control["survival"]["d0"]
        if bool(control["available"]) and bool(d0["available"]):
            eligible.append(d0)
    distinct = sum(value["state"] == "distinct" for value in eligible)
    return {
        "targets": len(records),
        "policy_available": policy_available,
        "d0_distinct": distinct,
        "d0_distinct_denominator": len(eligible),
        "d0_distinct_rate_among_eligible": _safe_rate(distinct, len(eligible)),
    }


def summarize_stage(
    records: list[dict[str, object]], stage: str
) -> dict[str, object]:
    conversions = [record["conversion"][stage] for record in records]
    available = [value for value in conversions if bool(value["available"])]
    margins = [
        value
        for value in (
            _finite(item.get("mean_logit_margin")) for item in available
        )
        if value is not None
    ]
    utilizations = [
        value
        for value in (
            _finite(item.get("utilization_cosine")) for item in available
        )
        if value is not None
    ]
    availability = [
        value
        for value in (
            _finite(item.get("mean_margin_availability")) for item in available
        )
        if value is not None
    ]
    normalized_availability = [
        value
        for value in (
            _finite(item.get("normalized_mean_margin_availability"))
            for item in available
        )
        if value is not None
    ]
    sensitivity = [
        value
        for value in (
            _finite(item.get("head_sensitivity")) for item in available
        )
        if value is not None
    ]
    floor_eligible = [
        item
        for item in available
        if item.get("absolute_scale_floor_active_channels") is not None
        and item.get("reparameterization_stable") is not None
    ]
    floor_stable = sum(
        int(item["absolute_scale_floor_active_channels"]) == 0
        and bool(item["reparameterization_stable"])
        for item in floor_eligible
    )
    margin_signs = _sign_counts(margins)
    utilization_signs = _sign_counts(utilizations)
    return {
        "targets": len(records),
        "available": len(available),
        "mean_margin": {
            "defined": len(margins),
            **margin_signs,
            "median": _median(margins),
        },
        "utilization": {
            "defined": len(utilizations),
            **utilization_signs,
            "median": _median(utilizations),
        },
        "availability": {
            "defined": len(availability),
            "median": _median(availability),
            "normalized_defined": len(normalized_availability),
            "normalized_median": _median(normalized_availability),
        },
        "head_sensitivity": {
            "defined": len(sensitivity),
            "median": _median(sensitivity),
        },
        "floor_stability": {
            "eligible": len(floor_eligible),
            "stable": floor_stable,
            "rate": _safe_rate(floor_stable, len(floor_eligible)),
        },
    }


def summarize_contributions(records: list[dict[str, object]]) -> dict[str, object]:
    eligible = [
        record["contribution_margins"]
        for record in records
        if bool(record["contribution_margins"]["available"])
        and record["contribution_margins"].get("has_sign_cancellation") is not None
    ]
    cancellations = sum(bool(value["has_sign_cancellation"]) for value in eligible)
    return {
        "eligible": len(eligible),
        "sign_cancellation": cancellations,
        "sign_cancellation_rate": _safe_rate(cancellations, len(eligible)),
        "median_positive_sum": _median(value.get("positive_sum") for value in eligible),
        "median_negative_sum": _median(value.get("negative_sum") for value in eligible),
        "median_final_direct_margin": _median(
            value.get("final_direct") for value in eligible
        ),
    }


def _summarize_target_statuses(statuses: list[dict[str, object]]) -> dict[str, object]:
    component = sum(bool(status["matched"]) for status in statuses)
    local_peak = sum(
        bool(status["neighborhood_peak_above_threshold"]) for status in statuses
    )
    margins = [
        value
        for value in (
            _finite(status.get("neighborhood_margin")) for status in statuses
        )
        if value is not None
    ]
    return {
        "targets": len(statuses),
        "exact_component_detected": component,
        "exact_component_detection_rate": _safe_rate(component, len(statuses)),
        "local_peak_above_threshold": local_peak,
        "local_peak_above_threshold_rate": _safe_rate(local_peak, len(statuses)),
        "local_peak_margin_defined": len(margins),
        "median_local_peak_margin": _median(margins),
    }


def summarize_outcome(
    records: list[dict[str, object]], budgets: tuple[str, ...]
) -> dict[str, object]:
    fixed_statuses = [
        record["operating_point"]["fixed_threshold"] for record in records
    ]
    cross_fitted = {}
    for budget in budgets:
        statuses = []
        for record in records:
            fixed_fa = record["operating_point"]["cross_fitted_fixed_fa"]
            if budget not in fixed_fa:
                raise RuntimeError("record is missing cross-fitted FA budget %s" % budget)
            statuses.append(fixed_fa[budget]["status"])
        cross_fitted[budget] = _summarize_target_statuses(statuses)
    return {
        "targets": len(records),
        "controls": {
            policy: summarize_control_d0(records, policy)
            for policy in CONTROL_POLICIES
        },
        "stages": {stage: summarize_stage(records, stage) for stage in STAGES},
        "final_contribution_signs": summarize_contributions(records),
        "fixed_threshold": _summarize_target_statuses(fixed_statuses),
        "cross_fitted_fixed_fa": cross_fitted,
    }


def summarize_pairs(records: list[dict[str, object]]) -> dict[str, object]:
    """Compute paired final differences, always oriented miss minus control."""

    misses = {}
    for record in records:
        if record["outcome"] != "no_response":
            continue
        run_key = str(record.get("_run_key", ""))
        miss_id = "%s:%d" % (record["sample_name"], int(record["target_index"]))
        key = (run_key, miss_id)
        if key in misses:
            raise RuntimeError("duplicate no-response target in paired summary")
        misses[key] = record

    paired_records = [
        record for record in records if record["outcome"] == "matched_control"
    ]
    seen_pairs = set()
    utilization_differences = []
    margin_differences = []
    for control in paired_records:
        run_key = str(control.get("_run_key", ""))
        pair_id = str(control.get("paired_no_response_id", ""))
        pair_index = int(control.get("pair_index", 0))
        unique_pair = (run_key, pair_id, pair_index)
        if unique_pair in seen_pairs:
            raise RuntimeError("duplicate paired-control index in summary")
        seen_pairs.add(unique_pair)
        miss = misses.get((run_key, pair_id))
        if miss is None:
            raise RuntimeError("matched control references an unknown miss")
        miss_final = miss["conversion"]["final"]
        control_final = control["conversion"]["final"]
        if bool(miss_final["available"]) and bool(control_final["available"]):
            miss_u = _finite(miss_final.get("utilization_cosine"))
            control_u = _finite(control_final.get("utilization_cosine"))
            if miss_u is not None and control_u is not None:
                utilization_differences.append(miss_u - control_u)
            miss_margin = _finite(miss_final.get("mean_logit_margin"))
            control_margin = _finite(control_final.get("mean_logit_margin"))
            if miss_margin is not None and control_margin is not None:
                margin_differences.append(miss_margin - control_margin)

    return {
        "difference_orientation": "no_response_minus_matched_control",
        "candidate_pairs": len(paired_records),
        "final_utilization": {
            "defined_pairs": len(utilization_differences),
            "positive_difference": sum(value > 0 for value in utilization_differences),
            "negative_difference": sum(value < 0 for value in utilization_differences),
            "zero_difference": sum(value == 0 for value in utilization_differences),
            "median_difference": _median(utilization_differences),
        },
        "final_mean_logit_margin": {
            "defined_pairs": len(margin_differences),
            "positive_difference": sum(value > 0 for value in margin_differences),
            "negative_difference": sum(value < 0 for value in margin_differences),
            "zero_difference": sum(value == 0 for value in margin_differences),
            "median_difference": _median(margin_differences),
        },
    }


def _pool_component_points(points: list[dict[str, object]]) -> dict[str, object]:
    """Pool component metrics from sufficient statistics, never ratio means."""

    sample_count = sum(int(point["sample_count"]) for point in points)
    total_pixels = sum(int(point["total_pixels"]) for point in points)
    targets = sum(int(point["target_components"]) for point in points)
    predictions = sum(int(point["prediction_components"]) for point in points)
    matches = sum(int(point["matched_components"]) for point in points)
    unmatched_targets = sum(int(point["unmatched_target_components"]) for point in points)
    unmatched_predictions = sum(
        int(point["unmatched_prediction_components"]) for point in points
    )
    unmatched_area = sum(int(point["unmatched_prediction_area"]) for point in points)
    if matches + unmatched_targets != targets:
        raise RuntimeError("component operating-point target counts are inconsistent")
    if matches + unmatched_predictions != predictions:
        raise RuntimeError("component operating-point prediction counts are inconsistent")
    return {
        "source_points": len(points),
        "sample_count": sample_count,
        "total_pixels": total_pixels,
        "target_components": targets,
        "prediction_components": predictions,
        "matched_components": matches,
        "unmatched_target_components": unmatched_targets,
        "unmatched_prediction_components": unmatched_predictions,
        "unmatched_prediction_area": unmatched_area,
        "actual_evaluation_fa_per_million_pixels": (
            1_000_000.0 * unmatched_area / total_pixels if total_pixels else None
        ),
        "actual_evaluation_pd": matches / targets if targets else None,
    }


def summarize_operating_points(
    payloads: list[dict[str, object]], budgets: tuple[str, ...]
) -> dict[str, object]:
    fixed_points = [
        payload["operating_point_audit"]["fixed_threshold_full_validation"]
        for payload in payloads
    ]
    fixed_summary = _pool_component_points(fixed_points)
    fixed_summary["thresholds"] = sorted(
        {_finite(point["threshold"]) for point in fixed_points}
    )
    cross_fitted = {}
    for budget in budgets:
        points = []
        thresholds = []
        for payload in payloads:
            folds = payload["operating_point_audit"]["folds"]
            for fold in sorted(folds, key=int):
                selection = folds[fold]["fixed_fa_selections"][budget]
                points.append(selection["evaluation_operating_point"])
                thresholds.append(_finite(selection["threshold"]))
        pooled = _pool_component_points(points)
        pooled.update(
            fa_budget_per_million_pixels=float(budget),
            calibration_threshold_min=min(thresholds) if thresholds else None,
            calibration_threshold_median=_median(thresholds),
            calibration_threshold_max=max(thresholds) if thresholds else None,
        )
        cross_fitted[budget] = pooled
    return {
        "fixed_threshold_full_validation": fixed_summary,
        "cross_fitted_fixed_fa": cross_fitted,
    }


def summarize_scope(
    records: list[dict[str, object]],
    payloads: list[dict[str, object]],
    budgets: tuple[str, ...],
) -> dict[str, object]:
    return {
        "outcomes": {
            outcome: summarize_outcome(
                [record for record in records if record["outcome"] == outcome],
                budgets,
            )
            for outcome in OUTCOMES
        },
        "paired_final_differences": summarize_pairs(records),
        "operating_points": summarize_operating_points(payloads, budgets),
    }


def _protocol_signature(payload: dict[str, object]) -> tuple[object, ...]:
    operating = payload["operating_point_audit"]
    budgets = tuple(
        _canonical_budget(value)
        for value in operating["fa_budgets_per_million_pixels"]
    )
    return (
        payload["split"],
        _finite(payload["threshold_probability"]),
        _finite(payload["threshold_logit"]),
        int(payload["required_geometry_controls"]),
        int(payload["context_control_count"]),
        _finite(payload["context_ring_width"]),
        operating["protocol"],
        operating["matching"],
        bool(operating["strict_threshold"]),
        _finite(operating["centroid_radius"]),
        budgets,
    )


def _validate_payload(
    payload: dict[str, object], path: Path
) -> tuple[str, str, str, list[dict[str, object]]]:
    if payload.get("schema") != AUDIT_SCHEMA:
        raise RuntimeError("unexpected decision-conversion schema in %s" % path)
    dataset = Path(str(payload["dataset_dir"])).name
    run_id = Path(str(payload["source_run_dir"])).name
    run_key = "%s/%s" % (dataset, run_id)
    records = payload["records"]
    if not isinstance(records, list):
        raise RuntimeError("audit records must be a list in %s" % path)
    counts = {
        outcome: sum(record.get("outcome") == outcome for record in records)
        for outcome in OUTCOMES
    }
    if any(record.get("outcome") not in OUTCOMES for record in records):
        raise RuntimeError("unexpected cohort outcome in %s" % path)
    if counts["no_response"] != int(payload["num_no_response_targets"]):
        raise RuntimeError("no-response record count drifted in %s" % path)
    if counts["matched_control"] != int(payload["num_selected_matched_controls"]):
        raise RuntimeError("matched-control record count drifted in %s" % path)
    tagged = []
    for record in records:
        copied = dict(record)
        copied["_run_key"] = run_key
        tagged.append(copied)
    return dataset, run_id, run_key, tagged


def build_summary(input_names: list[str]) -> dict[str, object]:
    payloads = []
    run_entries = []
    run_keys = set()
    protocol = None
    grouped_records: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_payloads: dict[str, list[dict[str, object]]] = defaultdict(list)
    all_records = []
    for input_name in input_names:
        path = Path(input_name).resolve()
        payload = json.loads(path.read_text())
        dataset, run_id, run_key, records = _validate_payload(payload, path)
        if run_key in run_keys:
            raise RuntimeError("duplicate dataset/run input: %s" % run_key)
        run_keys.add(run_key)
        current_protocol = _protocol_signature(payload)
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise RuntimeError("decision-conversion audit protocols do not match")
        payloads.append(payload)
        grouped_payloads[dataset].append(payload)
        grouped_records[dataset].extend(records)
        all_records.extend(records)
        run_entries.append(
            {
                "dataset": dataset,
                "run": run_id,
                "source": str(path),
                "source_run_dir": payload["source_run_dir"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "summary": summarize_scope(
                    records,
                    [payload],
                    current_protocol[-1],
                ),
            }
        )
    if protocol is None:
        raise RuntimeError("no inputs were supplied")
    budgets = protocol[-1]
    datasets = {
        dataset: {
            "run_count": len(grouped_payloads[dataset]),
            **summarize_scope(
                grouped_records[dataset], grouped_payloads[dataset], budgets
            ),
        }
        for dataset in sorted(grouped_records)
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "exploratory_only": True,
        "diagnostic_note": (
            "Stage statistics are reported independently. Raw availability or "
            "utilization is not compared across stages and no automatic failure "
            "taxonomy or causal attribution is produced."
        ),
        "pair_difference_orientation": "no_response_minus_matched_control",
        "protocol": {
            "split": protocol[0],
            "threshold_probability": protocol[1],
            "threshold_logit": protocol[2],
            "required_geometry_controls": protocol[3],
            "context_control_count": protocol[4],
            "context_ring_width": protocol[5],
            "operating_point_protocol": protocol[6],
            "component_matching": protocol[7],
            "strict_threshold": protocol[8],
            "centroid_radius": protocol[9],
            "fa_budgets_per_million_pixels": [float(value) for value in budgets],
        },
        "runs": sorted(run_entries, key=lambda item: (item["dataset"], item["run"])),
        "datasets": datasets,
        "overall": {
            "run_count": len(payloads),
            **summarize_scope(all_records, payloads, budgets),
        },
    }


def _count_ratio(numerator: int, denominator: int) -> str:
    return "%d/%d" % (numerator, denominator)


def _number(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    number = float(value)
    return ("%%.%dg" % digits) % number


def _scope_rows(summary: dict[str, object]):
    for run in summary["runs"]:
        yield "%s/%s" % (run["dataset"], run["run"]), run["summary"]
    for dataset, value in summary["datasets"].items():
        yield "dataset:%s" % dataset, value
    yield "overall", summary["overall"]


def render_markdown(summary: dict[str, object]) -> str:
    protocol = summary["protocol"]
    lines = [
        "# MSHNet decision-conversion diagnostic summary",
        "",
        "Internal split: `%s`; fixed probability threshold: `%.3f`; component matching: `%s`."
        % (
            protocol["split"],
            protocol["threshold_probability"],
            protocol["component_matching"],
        ),
        "",
        "## Cohort diagnostics",
        "",
        "All distinctness and detection counts show their eligible target denominator.",
        "",
        "| Scope | Outcome | Targets | Geometry D0 distinct | Context D0 distinct | Fixed exact component | Fixed local peak | Final sign cancellation |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope_name, scope in _scope_rows(summary):
        for outcome in OUTCOMES:
            value = scope["outcomes"][outcome]
            geometry = value["controls"]["geometry"]
            context = value["controls"]["context_matched"]
            fixed = value["fixed_threshold"]
            signs = value["final_contribution_signs"]
            lines.append(
                "| %s | %s | %d | %s | %s | %s | %s | %s |"
                % (
                    scope_name,
                    outcome,
                    value["targets"],
                    _count_ratio(
                        geometry["d0_distinct"],
                        geometry["d0_distinct_denominator"],
                    ),
                    _count_ratio(
                        context["d0_distinct"],
                        context["d0_distinct_denominator"],
                    ),
                    _count_ratio(
                        fixed["exact_component_detected"], fixed["targets"]
                    ),
                    _count_ratio(
                        fixed["local_peak_above_threshold"], fixed["targets"]
                    ),
                    _count_ratio(signs["sign_cancellation"], signs["eligible"]),
                )
            )

    lines.extend(
        [
            "",
            "## Stage-wise factorization diagnostics",
            "",
            "A and U are displayed within each named stage only; this table does not define a cross-stage drop or causal mechanism.",
            "",
            "| Scope | Outcome | Stage | Mean margin +/−/0 (defined) | U +/−/0 (defined) | Median U | Median A | Median H | Floor stable |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scope_name, scope in _scope_rows(summary):
        for outcome in OUTCOMES:
            for stage in STAGES:
                value = scope["outcomes"][outcome]["stages"][stage]
                margin = value["mean_margin"]
                utilization = value["utilization"]
                floor = value["floor_stability"]
                lines.append(
                    "| %s | %s | %s | %d/%d/%d (%d) | %d/%d/%d (%d) | %s | %s | %s | %s |"
                    % (
                        scope_name,
                        outcome,
                        stage,
                        margin["positive"],
                        margin["negative"],
                        margin["zero"],
                        margin["defined"],
                        utilization["positive"],
                        utilization["negative"],
                        utilization["zero"],
                        utilization["defined"],
                        _number(utilization["median"]),
                        _number(value["availability"]["median"]),
                        _number(value["head_sensitivity"]["median"]),
                        _count_ratio(floor["stable"], floor["eligible"]),
                    )
                )

    lines.extend(
        [
            "",
            "## Exact component operating points",
            "",
            "Actual evaluation FA and Pd are recomputed from pooled unmatched-prediction area, pixels, matches, and GT components. They are not means of fold-level ratios.",
            "",
            "| Scope | Operating point | Actual eval FA/Mpix | Actual eval Pd | No-response exact/local | Matched-control exact/local |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scope_name, scope in _scope_rows(summary):
        operating = scope["operating_points"]
        fixed = operating["fixed_threshold_full_validation"]
        no_response = scope["outcomes"]["no_response"]["fixed_threshold"]
        matched = scope["outcomes"]["matched_control"]["fixed_threshold"]
        lines.append(
            "| %s | fixed | %s | %s | %d/%d; %d/%d | %d/%d; %d/%d |"
            % (
                scope_name,
                _number(fixed["actual_evaluation_fa_per_million_pixels"]),
                _number(fixed["actual_evaluation_pd"]),
                no_response["exact_component_detected"],
                no_response["targets"],
                no_response["local_peak_above_threshold"],
                no_response["targets"],
                matched["exact_component_detected"],
                matched["targets"],
                matched["local_peak_above_threshold"],
                matched["targets"],
            )
        )
        for budget, point in operating["cross_fitted_fixed_fa"].items():
            no_response = scope["outcomes"]["no_response"][
                "cross_fitted_fixed_fa"
            ][budget]
            matched = scope["outcomes"]["matched_control"][
                "cross_fitted_fixed_fa"
            ][budget]
            lines.append(
                "| %s | FA %s | %s | %s | %d/%d; %d/%d | %d/%d; %d/%d |"
                % (
                    scope_name,
                    budget,
                    _number(point["actual_evaluation_fa_per_million_pixels"]),
                    _number(point["actual_evaluation_pd"]),
                    no_response["exact_component_detected"],
                    no_response["targets"],
                    no_response["local_peak_above_threshold"],
                    no_response["targets"],
                    matched["exact_component_detected"],
                    matched["targets"],
                    matched["local_peak_above_threshold"],
                    matched["targets"],
                )
            )

    lines.extend(
        [
            "",
            "## Paired final diagnostics",
            "",
            "Differences are `no_response − matched_control`; they are paired diagnostics, not causal effects.",
            "",
            "| Scope | Candidate pairs | Final U defined | Median ΔU | Final mean-margin defined | Median Δmargin |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scope_name, scope in _scope_rows(summary):
        paired = scope["paired_final_differences"]
        utilization = paired["final_utilization"]
        margin = paired["final_mean_logit_margin"]
        lines.append(
            "| %s | %d | %d | %s | %d | %s |"
            % (
                scope_name,
                paired["candidate_pairs"],
                utilization["defined_pairs"],
                _number(utilization["median_difference"]),
                margin["defined_pairs"],
                _number(margin["median_difference"]),
            )
        )

    lines.extend(
        [
            "",
            "This is an exploratory, GT-conditioned, checkpoint diagnostic. Local peak is not component Pd. The report does not assign automatic failure classes, infer fusion suppression from stage differences, or claim that paired associations are causal.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    if output_json.exists() or output_md.exists():
        raise FileExistsError("refusing to overwrite summary outputs")
    summary = build_summary(args.inputs)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    output_md.write_text(render_markdown(summary))
    print("wrote %s and %s" % (output_json, output_md))


if __name__ == "__main__":
    main()
