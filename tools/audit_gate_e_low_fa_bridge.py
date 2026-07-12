#!/usr/bin/env python3
"""Run the immutable Gate E-1c cross-fitted low-FA bridge audit."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.MSHNet import MSHNet  # noqa: E402
from tools.audit_cross_seed_failure_persistence import (  # noqa: E402
    CANONICAL_SIZE,
    DATASET_NAMES,
    PersistenceAuditError,
    _normalize_state_dict,
    _resolve_device,
    build_authoritative_registries_before_checkpoints,
    git_worktree_provenance,
    load_validated_jobs,
    protocol_document_fingerprints,
    sha256_file,
    sha256_json,
    validate_protocol_documents_unchanged,
)
from tools.compare_gate_e_checkpoint_policies import (  # noqa: E402
    _validate_bundle_ledger_hash,
)
from tools.finalize_clean_baselines import (  # noqa: E402
    FinalizationError,
    load_checkpoint_cpu,
)
from utils.component_operating_point import DEFAULT_TAIL_QUANTILES  # noqa: E402
from utils.cross_fitted_low_fa import (  # noqa: E402
    BUDGETS,
    FOLD_COUNT,
    FOLD_NAMESPACE,
    MATCHERS,
    LowFABridgeError,
    cross_fit_job,
    image_fold,
    summarize_low_fa_bridge,
    validate_hungarian_fixed_alignment,
)
from utils.data import IRSTD_Dataset  # noqa: E402
from utils.target_identity import (  # noqa: E402
    StableTargetSet,
    assert_same_target_set,
    build_stable_target_set,
)


SUMMARY_SCHEMA = "dea.gate_e.low_fa_bridge_bundle.v1"
PROVENANCE_SCHEMA = "dea.gate_e.low_fa_bridge_provenance.v1"
OUTPUT_FILES = (
    "target_low_fa.jsonl",
    "image_low_fa.jsonl",
    "calibration.json",
    "low_fa_bridge_summary.json",
    "low_fa_bridge_summary.md",
    "provenance.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit fixed-epoch MSHNet component thresholds at frozen "
            "nominal low-FA budgets under legacy and Hungarian matching."
        )
    )
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument(
        "--fixed-dir",
        default="repro_runs/gate_e/persistence_v2/fixed_epoch",
    )
    parser.add_argument(
        "--output-dir",
        default="repro_runs/gate_e/persistence_v2/low_fa_bridge",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LowFABridgeError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LowFABridgeError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LowFABridgeError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LowFABridgeError(f"cannot read {path}: {exc}") from exc
    return rows


def collect_job_predictions(
    job: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    expected_registry: Mapping[str, StableTargetSet],
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[str, ...],
    dict[str, Any],
]:
    stored_args = job.get("stored_args")
    if not isinstance(stored_args, dict):
        raise LowFABridgeError("validated job lacks stored args")
    dataset = IRSTD_Dataset(argparse.Namespace(**stored_args), mode="val")
    if dataset.split_sha256 != job["split_hashes"]["validation"]:
        raise LowFABridgeError("validation split hash drifted")
    if dataset.base_size != CANONICAL_SIZE or dataset.crop_size != CANONICAL_SIZE:
        raise LowFABridgeError("E-1c requires canonical 256x256 validation")
    if tuple(expected_registry) != tuple(dataset.names):
        raise LowFABridgeError("validation image universe disagrees with authority")

    checkpoint_path = Path(str(job["checkpoint"])).resolve()
    if sha256_file(checkpoint_path) != job["checkpoint_sha256"]:
        raise LowFABridgeError("checkpoint changed after validation")
    try:
        checkpoint_value = load_checkpoint_cpu(checkpoint_path)
    except FinalizationError as exc:
        raise LowFABridgeError(str(exc)) from exc
    state = checkpoint_value.get("net")
    if not isinstance(state, Mapping) or not state:
        raise LowFABridgeError("checkpoint has no valid network state")
    model = MSHNet(3)
    model.load_state_dict(_normalize_state_dict(state), strict=True)
    del state, checkpoint_value
    model.requires_grad_(False).to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    cursor = 0
    forward_calls = 0
    with torch.inference_mode():
        for images, batch_targets in loader:
            output = model(images.to(device, non_blocking=True), True)
            forward_calls += 1
            if not isinstance(output, tuple) or len(output) != 2:
                raise LowFABridgeError("MSHNet full graph returned an invalid output")
            batch_logits = output[1]
            if (
                not torch.is_tensor(batch_logits)
                or batch_logits.ndim != 4
                or tuple(batch_logits.shape[1:])
                != (1, CANONICAL_SIZE, CANONICAL_SIZE)
                or not bool(torch.isfinite(batch_logits).all())
            ):
                raise LowFABridgeError("MSHNet produced invalid E-1c logits")
            logit_arrays = batch_logits.detach().float().cpu().numpy()[:, 0]
            target_arrays = (batch_targets[:, 0] > 0.5).numpy().astype(
                bool, copy=False
            )
            for batch_index, (scores, target) in enumerate(
                zip(logit_arrays, target_arrays)
            ):
                image_index = cursor + batch_index
                image_name = dataset.names[image_index]
                observed = build_stable_target_set(
                    target,
                    dataset=str(job["dataset"]),
                    image_name=image_name,
                    connectivity=2,
                )
                try:
                    assert_same_target_set(expected_registry[image_name], observed)
                except Exception as exc:
                    raise LowFABridgeError(
                        f"inference target differs from authority: {exc}"
                    ) from exc
                logits.append(scores.astype(np.float64))
                targets.append(target.astype(bool, copy=False))
            cursor += int(logit_arrays.shape[0])
    if cursor != len(dataset) or forward_calls != len(loader):
        raise LowFABridgeError("E-1c inference pass accounting drifted")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    checkpoint_record = {
        "policy": "fixed_epoch",
        "epoch": int(job["checkpoint_summary"]["epoch"]),
        "path": str(checkpoint_path),
        "sha256": job["checkpoint_sha256"],
        "job_id": job["job_id"],
        "run_config_sha256": job["run_config_sha256"],
        "validation_split_sha256": job["split_hashes"]["validation"],
    }
    inference = {
        "job_id": job["job_id"],
        "dataset": job["dataset"],
        "seed": job["seed"],
        "image_count": len(logits),
        "target_count": sum(len(value.targets) for value in expected_registry.values()),
        "target_free_image_count": sum(
            not value.targets for value in expected_registry.values()
        ),
        "forward_calls": forward_calls,
        "full_graph_warm_flag": True,
    }
    return (
        tuple(logits),
        tuple(targets),
        tuple(dataset.names),
        {"checkpoint": checkpoint_record, "inference": inference},
    )


def build_markdown(summary: Mapping[str, Any]) -> str:
    gate = summary["joint_gate"]
    lines = [
        "# Gate E−1c cross-fitted low-FA bridge",
        "",
        "Fixed-epoch development holdout only; official test remains sealed.",
        "",
        f"- Gate pass: {gate['pass']}",
        f"- Passing budgets: {gate['passing_budgets']}",
        f"- Lowest passing budget: {gate['selected_lowest_passing_budget']}",
        "",
    ]
    for budget in BUDGETS:
        record = gate["by_budget"][str(budget)]
        lines.append(
            "- alpha=%s: datasets=%s, bridge targets=%s, controls=%s, pass=%s"
            % (
                budget,
                record["eligible_datasets"],
                record["joint_fixed0_and_low_fa_repeated_miss_target_count"],
                record["joint_stable_control_target_count"],
                record["pass"],
            )
        )
    lines.extend(
        [
            "",
            "A nominal budget is called feasible only under exact zero overshoot.",
            "All-off operating points are vetoed from bridge eligibility.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def write_bundle(
    output_dir: Path,
    *,
    target_rows: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        target_path = temporary / OUTPUT_FILES[0]
        image_path = temporary / OUTPUT_FILES[1]
        calibration_path = temporary / OUTPUT_FILES[2]
        summary_path = temporary / OUTPUT_FILES[3]
        markdown_path = temporary / OUTPUT_FILES[4]
        _write_jsonl(target_path, target_rows)
        _write_jsonl(image_path, image_rows)
        calibration_path.write_text(
            json.dumps(
                list(calibration),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(build_markdown(summary), encoding="utf-8")
        artifact_paths = (
            target_path,
            image_path,
            calibration_path,
            summary_path,
            markdown_path,
        )
        complete_provenance = {
            **dict(provenance),
            "artifact_sha256": {
                path.name: sha256_file(path) for path in artifact_paths
            },
        }
        (temporary / OUTPUT_FILES[5]).write_text(
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
            raise LowFABridgeError("temporary E-1c inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_hashes() -> dict[str, str]:
    paths = {
        "tool": Path(__file__).resolve(),
        "cross_fitted_low_fa": ROOT / "utils" / "cross_fitted_low_fa.py",
        "component_operating_point": ROOT / "utils" / "component_operating_point.py",
        "metric": ROOT / "utils" / "metric.py",
        "target_identity": ROOT / "utils" / "target_identity.py",
        "dataset": ROOT / "utils" / "data.py",
        "mshnet": ROOT / "model" / "MSHNet.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.batch_size < 1 or args.num_workers < 0:
        raise LowFABridgeError("batch size/workers are invalid")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    fixed_dir = _resolve(args.fixed_dir)
    fixed_ledger = fixed_dir / "target_persistence.jsonl"
    fixed_bundle = _validate_bundle_ledger_hash(fixed_ledger)
    fixed_provenance = _read_json(fixed_dir / "provenance.json")
    if (
        fixed_provenance.get("schema_version")
        != "dea.gate_e.cross_seed_failure_persistence.provenance.v2"
        or fixed_provenance.get("checkpoint_policy") != "fixed_epoch"
    ):
        raise LowFABridgeError("E-1c requires the formal fixed-epoch v2 bundle")
    fixed_rows = _read_jsonl(fixed_ledger)

    documents = protocol_document_fingerprints()
    sources = _source_hashes()
    git = git_worktree_provenance()
    batch_dir = ROOT / "repro_runs" / "clean" / args.batch_id
    authoritative, authority_records, registry_order = (
        build_authoritative_registries_before_checkpoints(
            batch_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )
    jobs, batch_provenance = load_validated_jobs(batch_dir, policy="fixed_epoch")
    device = _resolve_device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    target_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    inference_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    for job in jobs:
        logits, targets, names, record = collect_job_predictions(
            job,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            expected_registry=authoritative[str(job["dataset"])],
        )
        job_target_rows, job_image_rows, job_calibration = cross_fit_job(
            logits,
            targets,
            names,
            dataset=str(job["dataset"]),
            seed=int(job["seed"]),
            registry=authoritative[str(job["dataset"])],
            checkpoint=record["checkpoint"],
        )
        target_rows.extend(job_target_rows)
        image_rows.extend(job_image_rows)
        calibration_records.extend(job_calibration)
        inference_records.append(record["inference"])
        checkpoint_records.append(record["checkpoint"])
    validate_hungarian_fixed_alignment(target_rows, fixed_rows)
    authority_target_count = sum(
        len(target_set.targets)
        for dataset_registry in authoritative.values()
        for target_set in dataset_registry.values()
    )
    formal_target_ids = {
        str(row["stable_target_id"])
        for row in fixed_rows
        if row.get("row_kind") == "target"
    }
    seed_count = len({int(job["seed"]) for job in jobs})
    expected_target_rows = (
        authority_target_count * seed_count * len(MATCHERS) * len(BUDGETS)
    )
    if authority_target_count != len(formal_target_ids) or len(
        target_rows
    ) != expected_target_rows:
        raise LowFABridgeError("formal target universe/cardinality drifted")
    expected_images = sum(len(value) for value in authoritative.values())
    if len(image_rows) != (
        expected_images * seed_count * len(MATCHERS) * len(BUDGETS)
    ):
        raise LowFABridgeError("formal image ledger cardinality drifted")
    summary = {
        **summarize_low_fa_bridge(target_rows, image_rows),
        "schema_version": SUMMARY_SCHEMA,
        "analysis_scope": (
            "fixed-epoch development-holdout cross-fit; best-IoU excluded and "
            "official test remains sealed"
        ),
    }
    fold_mappings = {
        dataset: {
            image_name: image_fold(image_name)
            for image_name in authoritative[dataset]
        }
        for dataset in DATASET_NAMES
    }
    validate_protocol_documents_unchanged(documents)
    if _source_hashes() != sources or git_worktree_provenance() != git:
        raise LowFABridgeError("E-1c source/Git freeze changed during execution")
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "git": git,
        "protocol_documents": documents,
        "source_sha256": sources,
        "fixed_epoch_input": {
            "ledger": str(fixed_ledger),
            "ledger_sha256": fixed_bundle["ledger_sha256"],
            "bundle_provenance": fixed_bundle,
        },
        "batch": batch_provenance,
        "registry_precheckpoint_order": registry_order,
        "authoritative_registry_construction": authority_records,
        "jobs": checkpoint_records,
        "inference": inference_records,
        "protocol": {
            "fold_count": FOLD_COUNT,
            "fold_namespace": FOLD_NAMESPACE,
            "fold_mappings": fold_mappings,
            "fold_mappings_sha256": sha256_json(fold_mappings),
            "budgets_fa_per_mpix": list(BUDGETS),
            "matchers": list(MATCHERS),
            "strict_threshold_operator": ">",
            "fixed_thresholds": [0.0],
            "tail_quantiles": [float(value) for value in DEFAULT_TAIL_QUANTILES],
            "tail_quantiles_sha256": sha256_json(
                [float(value) for value in DEFAULT_TAIL_QUANTILES]
            ),
            "quantile_method": "numpy linear",
            "all_off_candidate": "maximum calibration logit under strict >",
            "calibration_tie_break": (
                "maximize matches, minimize unmatched prediction area, "
                "maximize threshold"
            ),
            "zero_overshoot": (
                "unmatched_area*1000000 <= budget*total_pixels using integers"
            ),
            "official_test_policy": "sealed and never opened",
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit_image": importlib.metadata.version("scikit-image"),
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    }
    validate_protocol_documents_unchanged(documents)
    if _source_hashes() != sources or git_worktree_provenance() != git:
        raise LowFABridgeError("E-1c freeze changed before output")
    write_bundle(
        output_dir,
        target_rows=target_rows,
        image_rows=image_rows,
        calibration=calibration_records,
        summary=summary,
        provenance=provenance,
    )
    return summary, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    summary, output_dir = run(parse_args(argv))
    print(
        "E-1c joint low-FA bridge pass: %s; passing budgets=%s"
        % (summary["joint_gate"]["pass"], summary["joint_gate"]["passing_budgets"])
    )
    print(f"wrote immutable Gate E-1c bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        LowFABridgeError,
        PersistenceAuditError,
        FinalizationError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"Gate E-1c audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
