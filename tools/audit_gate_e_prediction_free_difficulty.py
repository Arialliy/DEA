#!/usr/bin/env python3
"""Run the immutable Gate E-1b prediction-free difficulty audit."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
from argparse import Namespace
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_cross_seed_failure_persistence import (  # noqa: E402
    DATASET_NAMES,
    PersistenceAuditError,
    build_authoritative_registries_before_checkpoints,
    git_worktree_provenance,
    protocol_document_fingerprints,
    sha256_file,
    sha256_json,
    validate_protocol_documents_unchanged,
)
from tools.compare_gate_e_checkpoint_policies import (  # noqa: E402
    _validate_bundle_ledger_hash,
)
from utils.data import IRSTD_Dataset  # noqa: E402
from utils.prediction_free_difficulty import (  # noqa: E402
    DifficultyAuditError,
    compute_prediction_free_covariates,
    join_fixed_outcomes,
    summarize_difficulty,
)
from utils.target_identity import assert_same_target_set, build_stable_target_set  # noqa: E402


SCHEMA = "dea.gate_e.prediction_free_difficulty_bundle.v1"
PROVENANCE_SCHEMA = "dea.gate_e.prediction_free_difficulty_provenance.v1"
OUTPUT_FILES = (
    "target_difficulty.jsonl",
    "difficulty_summary.json",
    "difficulty_summary.md",
    "provenance.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether four frozen prediction-free image-and-annotation "
            "covariates nearly explain fixed-epoch cross-seed misses."
        )
    )
    parser.add_argument(
        "--fixed-dir",
        default="repro_runs/gate_e/persistence_v2/fixed_epoch",
    )
    parser.add_argument(
        "--batch-id",
        default="clean_baseline_holdout_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="repro_runs/gate_e/persistence_v2/prediction_free_difficulty",
    )
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
        raise DifficultyAuditError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DifficultyAuditError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DifficultyAuditError(
                        f"{path}:{line_number} must be a JSON object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DifficultyAuditError(f"cannot read {path}: {exc}") from exc
    return rows


def _source_data_fingerprint(
    datasets: Mapping[str, IRSTD_Dataset],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for dataset_name in DATASET_NAMES:
        dataset = datasets[dataset_name]
        files = []
        for image_name in dataset.names:
            image_path = Path(dataset.imgs_dir) / f"{image_name}.png"
            mask_path = Path(dataset.label_dir) / f"{image_name}.png"
            if not image_path.is_file() or not mask_path.is_file():
                raise DifficultyAuditError(
                    f"missing validation source image or mask for {dataset_name}/{image_name}"
                )
            files.append(
                {
                    "image_name": image_name,
                    "image_sha256": sha256_file(image_path),
                    "mask_sha256": sha256_file(mask_path),
                }
            )
        records[dataset_name] = {
            "image_count": len(files),
            "ordered_file_hashes_sha256": sha256_json(files),
        }
    return records


def build_feature_rows(
    *,
    fixed_provenance: Mapping[str, Any],
    authoritative: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, IRSTD_Dataset], dict[str, Any]]:
    source_jobs = fixed_provenance.get("registry_precheckpoint_order", {}).get(
        "source_jobs"
    )
    if not isinstance(source_jobs, list) or len(source_jobs) != len(DATASET_NAMES):
        raise DifficultyAuditError("fixed provenance lacks registry source jobs")
    source_by_dataset = {str(job.get("dataset")): job for job in source_jobs}
    if set(source_by_dataset) != set(DATASET_NAMES):
        raise DifficultyAuditError("fixed provenance registry source datasets drifted")

    datasets: dict[str, IRSTD_Dataset] = {}
    features: list[dict[str, Any]] = []
    resize = getattr(Image, "Resampling", Image).BILINEAR
    for dataset_name in DATASET_NAMES:
        stored_args = source_by_dataset[dataset_name].get("stored_args")
        if not isinstance(stored_args, dict):
            raise DifficultyAuditError("registry source job lacks stored args")
        dataset = IRSTD_Dataset(Namespace(**stored_args), mode="val")
        datasets[dataset_name] = dataset
        if tuple(authoritative[dataset_name]) != tuple(dataset.names):
            raise DifficultyAuditError("validation image universe disagrees with authority")
        for image_index, image_name in enumerate(dataset.names):
            image_path = Path(dataset.imgs_dir) / f"{image_name}.png"
            try:
                with Image.open(image_path) as source:
                    rgb = source.convert("RGB").resize((256, 256), resize)
                    rgb_array = np.asarray(rgb)
            except (OSError, ValueError) as exc:
                raise DifficultyAuditError(f"cannot load source RGB {image_path}: {exc}") from exc
            _, target_tensor = dataset[image_index]
            target = (target_tensor[0].numpy() > 0.5).astype(bool, copy=False)
            observed = build_stable_target_set(
                target,
                dataset=dataset_name,
                image_name=image_name,
                connectivity=2,
            )
            try:
                assert_same_target_set(authoritative[dataset_name][image_name], observed)
            except Exception as exc:
                raise DifficultyAuditError(
                    f"validation target disagrees with authority for {dataset_name}/{image_name}: {exc}"
                ) from exc
            features.extend(
                compute_prediction_free_covariates(
                    rgb_array,
                    target,
                    dataset=dataset_name,
                    image_name=image_name,
                )
            )
    source_fingerprint = _source_data_fingerprint(datasets)
    return features, datasets, source_fingerprint


def build_markdown(summary: Mapping[str, Any]) -> str:
    routing = summary["routing"]
    availability = summary["availability"]
    lines = [
        "# Gate E−1b prediction-free difficulty audit",
        "",
        "This is an association/prediction audit, not a causal explanation.",
        "",
        f"- Targets: {summary['target_count']}",
        f"- Availability unresolved: {availability['unresolved']}",
        f"- Eligible primary LODO folds: {routing['eligible_fold_count']}",
        f"- Eligible mean primary AUROC: {routing['eligible_mean_auroc']}",
        f"- Near-complete explanation: {routing['near_complete_explanation']}",
        f"- Routing decision: {routing['decision']}",
        "",
    ]
    for fold in summary["lodo"]["miss_any_seed"]["folds"]:
        lines.append(
            "- %s: eligible=%s, AUROC=%s, AP=%s"
            % (
                fold["held_out_dataset"],
                fold["eligible"],
                fold["auroc"],
                fold["average_precision"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_bundle(
    output_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
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
        ledger_path = temporary / OUTPUT_FILES[0]
        with ledger_path.open("w", encoding="utf-8") as handle:
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
        summary_path = temporary / OUTPUT_FILES[1]
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
        markdown_path = temporary / OUTPUT_FILES[2]
        markdown_path.write_text(build_markdown(summary), encoding="utf-8")
        complete_provenance = {
            **dict(provenance),
            "artifact_sha256": {
                path.name: sha256_file(path)
                for path in (ledger_path, summary_path, markdown_path)
            },
        }
        (temporary / OUTPUT_FILES[3]).write_text(
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
            raise DifficultyAuditError("temporary E-1b inventory drifted")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_hashes() -> dict[str, str]:
    paths = {
        "tool": Path(__file__).resolve(),
        "difficulty": ROOT / "utils" / "prediction_free_difficulty.py",
        "target_identity": ROOT / "utils" / "target_identity.py",
        "dataset": ROOT / "utils" / "data.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.batch_size < 1 or args.num_workers < 0:
        raise DifficultyAuditError("batch size/workers are invalid")
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
        raise DifficultyAuditError("E-1b requires the formal fixed-epoch v2 bundle")

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
    if registry_order["checkpoint_files_opened_before_registry_complete"] != 0:
        raise DifficultyAuditError("prediction-free registry opened a checkpoint")
    for dataset_name in DATASET_NAMES:
        expected = fixed_provenance.get("target_registry", {}).get(dataset_name, {})
        observed = authority_records[dataset_name]
        if (
            expected.get("target_registry_sha256")
            != observed.get("target_registry_sha256")
        ):
            raise DifficultyAuditError("current authority differs from fixed bundle")

    feature_rows, datasets, source_data = build_feature_rows(
        fixed_provenance=fixed_provenance,
        authoritative=authoritative,
    )
    joined = join_fixed_outcomes(feature_rows, _read_jsonl(fixed_ledger))
    summary = {
        **summarize_difficulty(joined),
        "schema_version": SCHEMA,
        "analysis_scope": (
            "fixed-epoch development-holdout association and LODO prediction; "
            "not causal and official test remains sealed"
        ),
    }
    validate_protocol_documents_unchanged(documents)
    if _source_hashes() != sources:
        raise DifficultyAuditError("E-1b source changed during execution")
    if _source_data_fingerprint(datasets) != source_data:
        raise DifficultyAuditError("validation images or masks changed during execution")
    if git_worktree_provenance() != git:
        raise DifficultyAuditError("Git worktree changed during E-1b execution")
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "git": git,
        "protocol_documents": documents,
        "source_sha256": sources,
        "source_data": source_data,
        "fixed_epoch_input": {
            "ledger": str(fixed_ledger),
            "ledger_sha256": fixed_bundle["ledger_sha256"],
            "bundle_provenance": fixed_bundle,
        },
        "authoritative_registry_construction": authority_records,
        "registry_precheckpoint_order": registry_order,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "pillow": importlib.metadata.version("Pillow"),
        },
        "protocol": {
            "source_image": "PIL convert RGB, uint8, bilinear resize to 256x256",
            "intensity": "(0.2126R+0.7152G+0.0722B)/255",
            "target_mask": "validation nearest resize, tensor >0.5",
            "prediction_inputs_prohibited": True,
            "official_test_policy": "sealed and never opened",
        },
    }
    validate_protocol_documents_unchanged(documents)
    if _source_hashes() != sources or git_worktree_provenance() != git:
        raise DifficultyAuditError("E-1b freeze changed before output")
    write_bundle(output_dir, rows=joined, summary=summary, provenance=provenance)
    return summary, output_dir


def main(argv: Sequence[str] | None = None) -> int:
    summary, output_dir = run(parse_args(argv))
    print(
        "E-1b routing: %s; eligible primary LODO folds=%s"
        % (
            summary["routing"]["decision"],
            summary["routing"]["eligible_fold_count"],
        )
    )
    print(f"wrote immutable Gate E-1b bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        DifficultyAuditError,
        PersistenceAuditError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"Gate E-1b audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
