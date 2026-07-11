#!/usr/bin/env python3
"""Fail-closed finalizer for the three-dataset clean MSHNet baselines.

This program only reads scheduler metadata, development-holdout metric logs,
and locally produced checkpoints.  It never opens dataset images, masks, or
official test split files, and it never constructs a model or CUDA device.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Callable


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
EXPECTED_EPOCHS = 400
EXPECTED_SEED_COUNT = 3
OUTPUT_JSON = "clean_baseline_holdout_summary.json"
OUTPUT_MARKDOWN = "clean_baseline_holdout_summary.md"
EXPECTED_STAGE = "development_holdout_baseline"
EXPECTED_TEST_POLICY = "loaded only for disjoint/hash audit; not iterated"

FLOAT_PATTERN = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
METRIC_RE = re.compile(
    rf"^\S+\s+-\s+(?P<epoch>\d+)\s+-\s+IoU\s+"
    rf"(?P<iou>{FLOAT_PATTERN})\s+-\s+PD\s+(?P<pd>{FLOAT_PATTERN})"
    rf"\s+-\s+FA\s+(?P<fa>{FLOAT_PATTERN})\s*$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FinalizationError(RuntimeError):
    """Raised when any required artifact or invariant fails validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize a complete 3-dataset x 3-seed, 400-epoch "
            "MSHNet development-holdout baseline batch."
        )
    )
    parser.add_argument("--batch-id", default="clean_baseline_holdout_v1")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing summary JSON/Markdown after validation succeeds.",
    )
    return parser.parse_args()


def read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FinalizationError(f"missing {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"invalid {label}: {path}: {exc}") from exc


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} must be a JSON/object mapping")
    return value


def require_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FinalizationError(f"{label} must be an integer, got {value!r}")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FinalizationError(f"{label} must be numeric, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise FinalizationError(f"{label} must be finite, got {value!r}")
    return result


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FinalizationError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def parse_csv_field(value: Any, cast: Callable[[str], Any], label: str) -> list[Any]:
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        raw_values = value
    else:
        raise FinalizationError(f"{label} must be a CSV string or list")
    try:
        values = [cast(item) for item in raw_values]
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"invalid {label}: {value!r}") from exc
    if not values or len(values) != len(set(values)):
        raise FinalizationError(f"{label} must contain unique non-empty values")
    return values


def parse_metrics(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        raise FinalizationError(f"missing epoch metric log: {path}")
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise FinalizationError(f"cannot read epoch metric log {path}: {exc}") from exc

    rows: list[dict[str, float | int]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise FinalizationError(
                f"blank line in epoch metric log {path} at line {line_number}"
            )
        match = METRIC_RE.fullmatch(line)
        if match is None:
            raise FinalizationError(
                f"unparseable epoch metric log {path} at line {line_number}: {line!r}"
            )
        row: dict[str, float | int] = {
            "epoch": int(match.group("epoch")),
            "iou": float(match.group("iou")),
            "pd": float(match.group("pd")),
            "fa": float(match.group("fa")),
        }
        for metric in ("iou", "pd", "fa"):
            value = float(row[metric])
            if not math.isfinite(value):
                raise FinalizationError(
                    f"non-finite {metric} in {path} at line {line_number}"
                )
        if not 0.0 <= float(row["iou"]) <= 1.0:
            raise FinalizationError(f"IoU outside [0, 1] in {path} at line {line_number}")
        if not 0.0 <= float(row["pd"]) <= 1.0:
            raise FinalizationError(f"PD outside [0, 1] in {path} at line {line_number}")
        if float(row["fa"]) < 0.0:
            raise FinalizationError(f"FA below zero in {path} at line {line_number}")
        rows.append(row)
    return rows


def load_checkpoint_cpu(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalizationError(f"missing best-IoU checkpoint: {path}")
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise FinalizationError(
            "PyTorch and NumPy are required to read checkpoints; run this tool "
            "with the repository training environment"
        ) from exc

    # These checkpoints contain NumPy float scalars for the validation metrics.
    # Explicitly allow only their dtype/scalar reconstructors while retaining
    # PyTorch's weights-only unpickler.  map_location='cpu' prevents CUDA use.
    numpy_scalar = np._core.multiarray.scalar  # type: ignore[attr-defined]
    safe_numpy_types = [
        numpy_scalar,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.int32)),
        type(np.dtype(np.int64)),
    ]
    try:
        with torch.serialization.safe_globals(safe_numpy_types):
            checkpoint = torch.load(
                path,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
    except Exception as exc:
        raise FinalizationError(f"cannot safely load checkpoint {path}: {exc}") from exc
    return require_mapping(checkpoint, f"checkpoint {path}")


def command_flag(command: Any, flag: str, label: str) -> str:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise FinalizationError(f"{label}.command must be a list of strings")
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise FinalizationError(f"{label}.command must contain exactly one {flag}")
    return command[positions[0] + 1]


def normalized_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FinalizationError(f"{label} must be a non-empty path string")
    return Path(value).expanduser().resolve()


def validate_result(result: dict[str, Any], job: dict[str, Any]) -> None:
    label = f"job result {job['job_id']}"
    returncode = require_exact_int(result.get("returncode"), f"{label}.returncode")
    if returncode != 0:
        raise FinalizationError(f"{label} has non-zero returncode {returncode}")
    if result.get("job_id") != job["job_id"]:
        raise FinalizationError(f"{label}.job_id does not match manifest")
    if normalized_path(result.get("run_dir"), f"{label}.run_dir") != normalized_path(
        job.get("run_dir"), f"manifest job {job['job_id']}.run_dir"
    ):
        raise FinalizationError(f"{label}.run_dir does not match manifest")

    command = result.get("command")
    expected_flags = {
        "--mode": "train",
        "--model-type": "mshnet",
        "--epochs": str(EXPECTED_EPOCHS),
        "--seed": str(job["seed"]),
        "--run-label": job["job_id"],
        "--run-dir": str(normalized_path(job["run_dir"], f"manifest {job['job_id']}.run_dir")),
    }
    for flag, expected in expected_flags.items():
        actual = command_flag(command, flag, label)
        if flag == "--run-dir":
            if normalized_path(actual, f"{label}.command {flag}") != Path(expected):
                raise FinalizationError(
                    f"{label}.command {flag} mismatch: {actual!r} != {expected!r}"
                )
        elif actual != expected:
            raise FinalizationError(
                f"{label}.command {flag} mismatch: {actual!r} != {expected!r}"
            )


def validate_checkpoint(
    checkpoint: dict[str, Any],
    job: dict[str, Any],
    dataset_meta: dict[str, Any],
    manifest_args: dict[str, Any],
    rows: list[dict[str, float | int]],
) -> dict[str, float | int]:
    label = f"checkpoint {job['job_id']}"
    metadata = require_mapping(checkpoint.get("method_meta"), f"{label}.method_meta")
    method = metadata.get("method")
    if not isinstance(method, str) or method.casefold() != "mshnet":
        raise FinalizationError(f"{label} method must be MSHNet/mshnet, got {method!r}")
    if metadata.get("model_type") != "mshnet":
        raise FinalizationError(
            f"{label} model_type must be 'mshnet', got {metadata.get('model_type')!r}"
        )
    if require_exact_int(metadata.get("seed"), f"{label}.seed") != job["seed"]:
        raise FinalizationError(f"{label} seed does not match manifest job")
    if metadata.get("run_label") != job["job_id"]:
        raise FinalizationError(f"{label} run_label does not match manifest job_id")

    split_seed = require_exact_int(metadata.get("split_seed"), f"{label}.split_seed")
    expected_split_seed = require_exact_int(
        manifest_args.get("split_seed"), "manifest.args.split_seed"
    )
    if split_seed != expected_split_seed:
        raise FinalizationError(f"{label} split_seed does not match manifest")

    expected_hashes = {
        "train_split_sha256": require_sha256(
            dataset_meta.get("fit_sha256"), f"manifest.datasets.{job['dataset']}.fit_sha256"
        ),
        "val_split_sha256": require_sha256(
            dataset_meta.get("val_sha256"), f"manifest.datasets.{job['dataset']}.val_sha256"
        ),
        "test_split_sha256": require_sha256(
            dataset_meta.get("official_test_sha256"),
            f"manifest.datasets.{job['dataset']}.official_test_sha256",
        ),
    }
    for key, expected in expected_hashes.items():
        actual = require_sha256(metadata.get(key), f"{label}.{key}")
        if actual != expected:
            raise FinalizationError(
                f"{label} {key} does not match the frozen manifest hash"
            )

    epoch = require_exact_int(checkpoint.get("epoch"), f"{label}.epoch")
    if not 0 <= epoch < EXPECTED_EPOCHS:
        raise FinalizationError(f"{label}.epoch outside [0, {EXPECTED_EPOCHS - 1}]")
    iou = require_number(checkpoint.get("iou"), f"{label}.iou")
    pd = require_number(checkpoint.get("pd"), f"{label}.pd")
    fa = require_number(checkpoint.get("fa"), f"{label}.fa")
    if not 0.0 <= iou <= 1.0 or not 0.0 <= pd <= 1.0 or fa < 0.0:
        raise FinalizationError(f"{label} contains out-of-range metrics")
    best_iou = require_number(checkpoint.get("best_iou"), f"{label}.best_iou")
    if not math.isclose(best_iou, iou, rel_tol=0.0, abs_tol=1e-12):
        raise FinalizationError(f"{label}.best_iou disagrees with checkpoint IoU")

    row = rows[epoch]
    checkpoint_metrics = {"iou": iou, "pd": pd, "fa": fa}
    for metric, checkpoint_value in checkpoint_metrics.items():
        logged_value = float(row[metric])
        if f"{checkpoint_value:.4f}" != f"{logged_value:.4f}":
            raise FinalizationError(
                f"{label} {metric} disagrees with epoch_metric.log at epoch {epoch}"
            )
    logged_best_iou = max(float(row_["iou"]) for row_ in rows)
    if f"{iou:.4f}" != f"{logged_best_iou:.4f}":
        raise FinalizationError(
            f"{label} is not the best-IoU epoch recorded in epoch_metric.log"
        )
    return {"seed": job["seed"], "best_epoch": epoch, "iou": iou, "pd": pd, "fa": fa}


def metric_stats(runs: list[dict[str, float | int]]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    for key in ("best_epoch", "iou", "pd", "fa"):
        values = [float(run[key]) for run in runs]
        means[key] = statistics.mean(values)
        standard_deviations[key] = statistics.stdev(values)
    return means, standard_deviations


def validate_manifest(manifest: dict[str, Any], batch_dir: Path) -> tuple[list[int], dict[str, Any]]:
    if manifest.get("batch_id") != batch_dir.name:
        raise FinalizationError(
            f"manifest batch_id {manifest.get('batch_id')!r} does not match {batch_dir.name!r}"
        )
    if manifest.get("stage") != EXPECTED_STAGE:
        raise FinalizationError(
            f"manifest stage must be {EXPECTED_STAGE!r}, got {manifest.get('stage')!r}"
        )
    if manifest.get("official_test_policy") != EXPECTED_TEST_POLICY:
        raise FinalizationError(
            "manifest official_test_policy does not certify audit-only, non-iterated test handling"
        )

    args = require_mapping(manifest.get("args"), "manifest.args")
    if require_exact_int(args.get("epochs"), "manifest.args.epochs") != EXPECTED_EPOCHS:
        raise FinalizationError(f"manifest must declare exactly {EXPECTED_EPOCHS} epochs")
    datasets_from_args = parse_csv_field(args.get("datasets"), str, "manifest.args.datasets")
    if set(datasets_from_args) != set(DATASET_NAMES) or len(datasets_from_args) != len(
        DATASET_NAMES
    ):
        raise FinalizationError(f"manifest must contain exactly datasets {DATASET_NAMES}")
    seeds = parse_csv_field(args.get("seeds"), int, "manifest.args.seeds")
    if len(seeds) != EXPECTED_SEED_COUNT:
        raise FinalizationError(
            f"manifest must contain exactly {EXPECTED_SEED_COUNT} unique seeds"
        )

    datasets = require_mapping(manifest.get("datasets"), "manifest.datasets")
    if set(datasets) != set(DATASET_NAMES) or len(datasets) != len(DATASET_NAMES):
        raise FinalizationError(f"manifest.datasets must be exactly {DATASET_NAMES}")
    for dataset_name in DATASET_NAMES:
        dataset_meta = require_mapping(
            datasets[dataset_name], f"manifest.datasets.{dataset_name}"
        )
        if dataset_meta.get("dataset") != dataset_name:
            raise FinalizationError(f"manifest dataset label mismatch for {dataset_name}")
        for field in ("fit_sha256", "val_sha256", "official_test_sha256"):
            require_sha256(dataset_meta.get(field), f"manifest.datasets.{dataset_name}.{field}")

    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise FinalizationError("manifest.jobs must be a list")
    expected_pairs = {(dataset, seed) for dataset in DATASET_NAMES for seed in seeds}
    if len(jobs) != len(expected_pairs):
        raise FinalizationError(
            f"manifest must contain exactly {len(expected_pairs)} jobs (3 datasets x 3 seeds)"
        )
    actual_pairs: set[tuple[str, int]] = set()
    for index, raw_job in enumerate(jobs):
        job = require_mapping(raw_job, f"manifest.jobs[{index}]")
        dataset = job.get("dataset")
        seed = require_exact_int(job.get("seed"), f"manifest.jobs[{index}].seed")
        pair = (dataset, seed)
        if pair not in expected_pairs or pair in actual_pairs:
            raise FinalizationError(f"unexpected or duplicate manifest job pair: {pair!r}")
        expected_id = f"mshnet__{dataset.lower()}__seed_{seed}"
        if job.get("job_id") != expected_id:
            raise FinalizationError(
                f"manifest job_id mismatch for {pair!r}: expected {expected_id!r}"
            )
        for field in ("run_dir", "result_file"):
            normalized_path(job.get(field), f"manifest job {expected_id}.{field}")
        actual_pairs.add(pair)
    if actual_pairs != expected_pairs:
        raise FinalizationError("manifest jobs do not form the complete dataset x seed grid")
    return seeds, datasets


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Clean MSHNet development-holdout baseline summary",
        "",
        "> **Scope guard:** These are internal development-holdout results from the official training sets. "
        "The official test sets remain untouched and were not evaluated by this finalizer. "
        "These numbers are not official-test or paper-main-table results.",
        "",
        f"- Batch: `{summary['batch_id']}`",
        f"- Validated grid: {len(DATASET_NAMES)} datasets × {len(summary['seeds'])} seeds",
        f"- Epochs per run: {summary['epochs_per_run']}",
        "- Selection: best development-holdout IoU checkpoint within each fixed 400-epoch run",
        "- Aggregate dispersion: sample standard deviation across three seeds (n−1 denominator)",
        "- FA/M: unmatched predicted foreground pixels per million image pixels at threshold 0.5",
        "",
        "## Per-seed best checkpoints",
        "",
        "| Dataset | Seed | Best epoch | IoU | PD | FA/M |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name in DATASET_NAMES:
        for run in summary["datasets"][dataset_name]["runs"]:
            lines.append(
                f"| {dataset_name} | {run['seed']} | {run['best_epoch']} | "
                f"{run['iou']:.6f} | {run['pd']:.6f} | {run['fa']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Three-seed aggregate",
            "",
            "| Dataset | IoU mean ± SD | PD mean ± SD | FA/M mean ± SD | Best epoch mean ± SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset_name in DATASET_NAMES:
        aggregate = summary["datasets"][dataset_name]
        mean = aggregate["mean"]
        std = aggregate["std"]
        lines.append(
            f"| {dataset_name} | {mean['iou']:.6f} ± {std['iou']:.6f} | "
            f"{mean['pd']:.6f} ± {std['pd']:.6f} | "
            f"{mean['fa']:.3f} ± {std['fa']:.3f} | "
            f"{mean['best_epoch']:.2f} ± {std['best_epoch']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This artifact closes only the clean MSHNet development baseline. It can be used to "
            "define and screen DEA hypotheses on matched holdouts. It must not be described as "
            "performance on the official test sets; official-test evaluation requires a separately "
            "frozen protocol after model design and selection are complete.",
            "",
        ]
    )
    return "\n".join(lines)


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def finalize_batch(
    batch_dir: Path,
    *,
    force: bool = False,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = load_checkpoint_cpu,
) -> dict[str, Any]:
    batch_dir = batch_dir.expanduser().resolve()
    json_path = batch_dir / OUTPUT_JSON
    markdown_path = batch_dir / OUTPUT_MARKDOWN
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing final summary: "
            + ", ".join(str(path) for path in existing)
            + "; pass --force only when intentional"
        )

    manifest = require_mapping(read_json(batch_dir / "manifest.json", "manifest"), "manifest")
    seeds, datasets_meta = validate_manifest(manifest, batch_dir)
    manifest_args = require_mapping(manifest["args"], "manifest.args")

    runs_by_dataset: dict[str, list[dict[str, float | int]]] = {
        dataset: [] for dataset in DATASET_NAMES
    }
    jobs = sorted(
        manifest["jobs"],
        key=lambda job: (DATASET_NAMES.index(job["dataset"]), seeds.index(job["seed"])),
    )
    for job in jobs:
        result_path = normalized_path(
            job["result_file"], f"manifest job {job['job_id']}.result_file"
        )
        result = require_mapping(read_json(result_path, "job result"), f"job result {job['job_id']}")
        validate_result(result, job)

        run_dir = normalized_path(job["run_dir"], f"manifest job {job['job_id']}.run_dir")
        rows = parse_metrics(run_dir / "epoch_metric.log")
        if len(rows) != EXPECTED_EPOCHS:
            raise FinalizationError(
                f"{job['job_id']} must contain exactly {EXPECTED_EPOCHS} metric rows; "
                f"found {len(rows)}"
            )
        epoch_ids = [int(row["epoch"]) for row in rows]
        if epoch_ids != list(range(EXPECTED_EPOCHS)):
            raise FinalizationError(
                f"{job['job_id']} epoch ids must be exactly 0..{EXPECTED_EPOCHS - 1}"
            )

        checkpoint_path = run_dir / "checkpoint_best_iou.pkl"
        checkpoint = checkpoint_loader(checkpoint_path)
        run_summary = validate_checkpoint(
            checkpoint,
            job,
            require_mapping(datasets_meta[job["dataset"]], f"dataset {job['dataset']}"),
            manifest_args,
            rows,
        )
        run_summary["checkpoint"] = str(checkpoint_path)
        runs_by_dataset[job["dataset"]].append(run_summary)
        del checkpoint

    dataset_summaries: dict[str, Any] = {}
    for dataset_name in DATASET_NAMES:
        runs = sorted(runs_by_dataset[dataset_name], key=lambda run: seeds.index(int(run["seed"])))
        if len(runs) != EXPECTED_SEED_COUNT:
            raise FinalizationError(f"internal error: incomplete summaries for {dataset_name}")
        means, standard_deviations = metric_stats(runs)
        dataset_meta = require_mapping(datasets_meta[dataset_name], f"dataset {dataset_name}")
        dataset_summaries[dataset_name] = {
            "split_hashes": {
                "fit": dataset_meta["fit_sha256"],
                "validation": dataset_meta["val_sha256"],
                "official_test_audit_only": dataset_meta["official_test_sha256"],
            },
            "runs": runs,
            "mean": means,
            "std": standard_deviations,
        }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "batch_id": manifest["batch_id"],
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete_and_validated",
        "method": "mshnet",
        "model_type": "mshnet",
        "evaluation_scope": "official-training-set internal development holdout",
        "official_test_status": "untouched; not evaluated by this finalizer",
        "not_for_official_test_or_main_table_claims": True,
        "epochs_per_run": EXPECTED_EPOCHS,
        "seeds": seeds,
        "statistics": "mean and sample standard deviation across three seeds",
        "metrics": {
            "iou": "foreground intersection over union at threshold 0.5",
            "pd": "component probability of detection at threshold 0.5",
            "fa": "unmatched foreground pixels per million image pixels (FA/M) at threshold 0.5",
        },
        "datasets": dataset_summaries,
    }

    json_text = json.dumps(summary, indent=2, sort_keys=False) + "\n"
    markdown_text = build_markdown(summary)
    write_atomic(json_path, json_text)
    write_atomic(markdown_path, markdown_text)
    return summary


def main() -> int:
    args = parse_args()
    if not args.batch_id or Path(args.batch_id).name != args.batch_id:
        raise FinalizationError("--batch-id must be one directory name, not a path")
    batch_dir = PROJECT_DIR / "repro_runs" / "clean" / args.batch_id
    summary = finalize_batch(batch_dir, force=args.force)
    print(
        f"validated {len(DATASET_NAMES)} datasets x {len(summary['seeds'])} seeds; "
        f"wrote {batch_dir / OUTPUT_JSON} and {batch_dir / OUTPUT_MARKDOWN}"
    )
    print("scope: development holdout only; official test remains untouched")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalizationError, FileExistsError) as exc:
        print(f"finalization refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
