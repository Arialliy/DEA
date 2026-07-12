#!/usr/bin/env python3
"""Compare immutable Gate E-1 fixed-epoch and best-IoU ledgers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_cross_seed_failure_persistence import (  # noqa: E402
    PersistenceAuditError,
    compare_policy_target_recurrence,
    protocol_document_fingerprints,
    read_policy_ledger,
    sha256_file,
    validate_protocol_documents_unchanged,
)


OUTPUT_FILES = (
    "policy_transition.json",
    "policy_transition.md",
    "provenance.json",
)
PROVENANCE_SCHEMA = "dea.gate_e.checkpoint_policy_transition.provenance.v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gate E-1 fixed-epoch primary and retrospective best-IoU "
            "target-recurrence ledgers without pooling their outcomes."
        )
    )
    parser.add_argument(
        "--fixed-dir",
        default="repro_runs/gate_e/persistence_v2/fixed_epoch",
    )
    parser.add_argument(
        "--best-dir",
        default="repro_runs/gate_e/persistence_v2/best_iou",
    )
    parser.add_argument(
        "--output-dir",
        default="repro_runs/gate_e/persistence_v2/policy_transition",
    )
    return parser.parse_args(argv)


def _resolved_ledger(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.is_dir():
        path = path / "target_persistence.jsonl"
    if not path.is_file():
        raise PersistenceAuditError(f"missing policy ledger: {path}")
    return path


def _validate_bundle_ledger_hash(ledger_path: Path) -> dict[str, Any]:
    provenance_path = ledger_path.parent / "provenance.json"
    if not provenance_path.is_file():
        raise PersistenceAuditError(
            f"missing policy-bundle provenance: {provenance_path}"
        )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PersistenceAuditError(
            f"cannot read policy-bundle provenance {provenance_path}: {exc}"
        ) from exc
    if not isinstance(provenance, dict):
        raise PersistenceAuditError("policy-bundle provenance must be an object")
    artifact_hashes = provenance.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise PersistenceAuditError("policy-bundle provenance lacks artifact hashes")
    expected = artifact_hashes.get(ledger_path.name)
    observed = sha256_file(ledger_path)
    if expected != observed:
        raise PersistenceAuditError(
            f"policy ledger hash disagrees with provenance: {ledger_path}"
        )
    return {
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": sha256_file(provenance_path),
        "ledger_sha256": observed,
    }


def _routing_record(scope: Mapping[str, Any]) -> dict[str, Any]:
    jaccard = scope["missed_set_jaccard"]
    retention = scope["fixed_c3_to_best_c_ge2_retention"]
    jaccard_pass = bool(jaccard["defined"] and float(jaccard["value"]) >= 0.50)
    retention_pass = bool(
        retention["defined"] and float(retention["value"]) >= 0.50
    )
    return {
        "thresholds": {
            "missed_set_jaccard_minimum": 0.50,
            "fixed_c3_to_best_c_ge2_retention_minimum": 0.50,
        },
        "missed_set_jaccard_pass": jaccard_pass,
        "fixed_c3_to_best_c_ge2_retention_pass": retention_pass,
        "policy_transition_gate_pass": jaccard_pass and retention_pass,
        "undefined_is_pass": False,
    }


def build_transition_report(
    fixed_rows: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    comparison = compare_policy_target_recurrence(fixed_rows, best_rows)
    return {
        **comparison,
        "analysis_scope": (
            "retrospective checkpoint-policy sensitivity; fixed_epoch remains "
            "the primary policy and best_iou is not pooled with it"
        ),
        "overall_routing_gate": _routing_record(comparison["overall"]),
        "by_dataset_routing_gate": {
            dataset: _routing_record(scope)
            for dataset, scope in comparison["by_dataset"].items()
        },
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    overall = report["overall"]
    matrix = overall["transition_matrix"]["counts"]
    jaccard = overall["missed_set_jaccard"]
    retention = overall["fixed_c3_to_best_c_ge2_retention"]
    gate = report["overall_routing_gate"]
    fixed_recurrence = overall["fixed_epoch_recurrence"]
    best_recurrence = overall["best_iou_recurrence"]
    lines = [
        "# Gate E−1 checkpoint-policy transition",
        "",
        "Fixed epoch is the primary policy; best-IoU is retrospective sensitivity only.",
        "",
        "## Overall 4×4 transition",
        "",
        "| c_fixed \\ c_best | 0 | 1 | 2 | 3 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fixed_count, row in enumerate(matrix):
        lines.append(
            f"| {fixed_count} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |"
        )
    lines.extend(
        [
            "",
            "## Frozen routing metrics",
            "",
            f"- Missed-set Jaccard: {jaccard['value']}",
            f"- Fixed c=3 retained at best c>=2: {retention['value']}",
            f"- Fixed N3/N: {fixed_recurrence['N3_over_N']}",
            f"- Fixed persistent event share: {fixed_recurrence['persistent_event_share']}",
            f"- Best N3/N: {best_recurrence['N3_over_N']}",
            f"- Best persistent event share: {best_recurrence['persistent_event_share']}",
            f"- Policy-transition gate pass: {gate['policy_transition_gate_pass']}",
            "",
            "Undefined metrics fail the routing gate; they are never silently treated as pass.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(
    output_dir: Path,
    *,
    report: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        report_path = temporary / OUTPUT_FILES[0]
        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        markdown_path = temporary / OUTPUT_FILES[1]
        markdown_path.write_text(build_markdown(report), encoding="utf-8")
        complete_provenance = {
            **dict(provenance),
            "artifact_sha256": {
                report_path.name: sha256_file(report_path),
                markdown_path.name: sha256_file(markdown_path),
            },
        }
        (temporary / OUTPUT_FILES[2]).write_text(
            json.dumps(
                complete_provenance,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if {path.name for path in temporary.iterdir()} != set(OUTPUT_FILES):
            raise PersistenceAuditError("policy-transition output inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    frozen_documents = protocol_document_fingerprints()
    fixed_path = _resolved_ledger(args.fixed_dir)
    best_path = _resolved_ledger(args.best_dir)
    if fixed_path == best_path:
        raise PersistenceAuditError("fixed and best policy ledgers must differ")
    fixed_bundle = _validate_bundle_ledger_hash(fixed_path)
    best_bundle = _validate_bundle_ledger_hash(best_path)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    report = build_transition_report(
        read_policy_ledger(fixed_path),
        read_policy_ledger(best_path),
    )
    validate_protocol_documents_unchanged(frozen_documents)
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "protocol_documents": frozen_documents,
        "inputs": {
            "fixed_epoch_ledger": {
                "path": str(fixed_path),
                "sha256": fixed_bundle["ledger_sha256"],
                "bundle_provenance": fixed_bundle,
            },
            "best_iou_ledger": {
                "path": str(best_path),
                "sha256": best_bundle["ledger_sha256"],
                "bundle_provenance": best_bundle,
            },
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    validate_protocol_documents_unchanged(frozen_documents)
    write_bundle(output_dir, report=report, provenance=provenance)
    return report, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    report, output_dir = run(parse_args(argv))
    print(
        "policy transition gate pass: "
        f"{report['overall_routing_gate']['policy_transition_gate_pass']}"
    )
    print(f"wrote immutable checkpoint-policy comparison: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PersistenceAuditError, FileExistsError, OSError) as exc:
        print(f"Gate E-1 policy comparison refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
