#!/usr/bin/env python3
"""Export a locked test score bundle after dev-only checkpoint selection.

This is intentionally a separate executable from training and dev inference.
It first re-authenticates the complete best-dev checkpoint and the already
exported dev bundle, and only then constructs ``include_test=True`` data.  It
does not select thresholds, checkpoints, epochs, or hyperparameters from test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from argparse import Namespace
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.trace_mshnet import render_trace_atoms  # noqa: E402
from tools import train_trace  # noqa: E402
from utils.trace_data import build_trace_data  # noqa: E402
from utils.trace_provenance import (  # noqa: E402
    canonical_json_sha256,
    provenance_path,
    sha256_file,
)
from utils.trace_training import (  # noqa: E402
    TraceTrainingError,
    load_training_checkpoint,
    restore_training_checkpoint,
)


LOCK_PHRASE = "I_UNDERSTAND_TEST_IS_EVALUATION_ONLY"
TEST_BUNDLE_SCHEMA_VERSION = "trace_locked_test_score_bundle_v1"


class TraceLockedTestError(TraceTrainingError):
    """Locked test access or export does not satisfy the protocol."""


def require_explicit_unlock(value: str) -> None:
    if value != LOCK_PHRASE:
        raise TraceLockedTestError(
            "locked test inference requires the exact --unlock-test phrase"
        )


def _strict_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    def reject_nonfinite(token: str) -> None:
        raise ValueError(token)

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TraceLockedTestError("invalid development bundle metadata") from exc
    if not isinstance(payload, dict):
        raise TraceLockedTestError("development bundle metadata must be an object")
    return source, payload


def validate_dev_bundle_lock(
    metadata_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_method: str,
    expected_data_provenance_sha256: str,
) -> dict[str, Any]:
    """Prove a dev score bundle was frozen from this best-dev checkpoint."""

    path, payload = _strict_json(metadata_path)
    required = {
        "schema_version": train_trace.DEV_BUNDLE_SCHEMA_VERSION,
        "method": expected_method,
        "split": "dev",
        "selection_checkpoint_role": "best_dev",
        "checkpoint_sha256": expected_checkpoint_sha256,
        "data_provenance_sha256": expected_data_provenance_sha256,
        "test_access": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise TraceLockedTestError(f"development bundle lock mismatch: {key}")
    bundle_locator = payload.get("bundle")
    bundle_hash = payload.get("bundle_sha256")
    if not isinstance(bundle_locator, str) or not isinstance(bundle_hash, str):
        raise TraceLockedTestError("development bundle lock lacks bundle provenance")
    if bundle_locator.startswith("repo:"):
        bundle_path = PROJECT_ROOT / bundle_locator[5:]
    elif bundle_locator.startswith("external:"):
        bundle_path = path.parent / bundle_locator[9:]
    else:
        raise TraceLockedTestError("development bundle locator is unsafe")
    if not bundle_path.is_file() or sha256_file(bundle_path) != bundle_hash:
        raise TraceLockedTestError("development bundle bytes differ from locked metadata")
    return payload


def _finite_number(value: Any, *, name: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TraceLockedTestError(f"checkpoint run config has invalid {name}") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise TraceLockedTestError(f"checkpoint run config has invalid {name}")
    return result


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TraceLockedTestError(f"checkpoint run config has invalid {name}")
    return value


def reconstruct_training_args(
    checkpoint_payload: Mapping[str, Any], cli: argparse.Namespace
) -> Namespace:
    """Recover the immutable training config instead of retyping hyperparameters."""

    config = checkpoint_payload.get("run_config")
    if not isinstance(config, Mapping) or config.get("schema_version") != (
        "trace_paired_run_config_v1"
    ):
        raise TraceLockedTestError("checkpoint lacks a valid paired run config")
    method = config.get("method")
    if method not in {"trace", "dense_bernoulli"}:
        raise TraceLockedTestError("checkpoint method is invalid")
    recorded_device = config.get("device")
    if cli.device and cli.device != recorded_device:
        raise TraceLockedTestError(
            "locked inference device must match the authenticated run config"
        )
    return Namespace(
        mode="dev-infer",
        method=method,
        dataset_dir=cli.dataset_dir,
        train_manifest=cli.train_manifest,
        test_manifest=cli.test_manifest,
        baseline_checkpoint=cli.baseline_checkpoint,
        t0_a_report=cli.t0_a_report,
        t0_b_dp_report=cli.t0_b_dp_report,
        t0_b_integration_report=cli.t0_b_integration_report,
        output_dir=str(Path(cli.checkpoint).expanduser().resolve().parent),
        seed=_integer(config.get("seed"), name="seed", minimum=0),
        dev_fraction=_finite_number(config.get("dev_fraction"), name="dev_fraction", positive=True),
        epochs=_integer(config.get("epochs"), name="epochs", minimum=1),
        batch_size=_integer(config.get("batch_size"), name="batch_size", minimum=1),
        num_workers=_integer(config.get("num_workers"), name="num_workers", minimum=0),
        learning_rate=_finite_number(config.get("learning_rate"), name="learning_rate", positive=True),
        weight_decay=_finite_number(config.get("weight_decay"), name="weight_decay"),
        gradient_clip_norm=_finite_number(
            config.get("gradient_clip_norm"), name="gradient_clip_norm"
        ),
        positive_cell_prior=config.get("positive_cell_prior"),
        foreground_pixel_prior=config.get("foreground_pixel_prior"),
        field_chunk_size=(
            _integer(config.get("field_chunk_size"), name="field_chunk_size", minimum=1)
            if method == "trace"
            else 256
        ),
        device=recorded_device,
        resume="",
        checkpoint=cli.checkpoint,
        dev_bundle="",
        overwrite_dev_bundle=False,
    )


def _restore_best_dev(
    args: Namespace, state: dict[str, Any], payload: Mapping[str, Any]
) -> None:
    if payload.get("checkpoint_role") != "best_dev" or payload.get("epoch") != payload.get(
        "best_epoch"
    ):
        raise TraceLockedTestError("test inference requires the best-dev checkpoint")
    common = train_trace._checkpoint_common(
        args=args,
        **{
            key: state[key]
            for key in (
                "model",
                "optimizer",
                "train_generator",
                "dev_generator",
                "run_config",
                "data_provenance",
                "gates",
                "geometry_gate",
                "sources",
                "git",
                "runtime",
            )
        },
    )
    restore_training_checkpoint(
        payload,
        expected_method=args.method,
        expected_run_config_sha256=canonical_json_sha256(state["run_config"]),
        expected_data_provenance_sha256=canonical_json_sha256(
            state["data_provenance"]
        ),
        expected_gates=state["gates"],
        expected_geometry_sha256=state["geometry_gate"].geometry_sha256,
        expected_logk_cache_sha256=common["logk_cache_sha256"],
        expected_front_state_sha256=common["front_state_sha256"],
        expected_sources=state["sources"],
        model=state["model"],
        head=state["model"].potential_map,
        optimizer=state["optimizer"],
        train_generator=state["train_generator"],
        dev_generator=state["dev_generator"],
    )
    state["model"].train(False)


def export_locked_test_bundle(cli: argparse.Namespace) -> dict[str, Any]:
    require_explicit_unlock(cli.unlock_test)
    checkpoint_path = Path(cli.checkpoint).expanduser().resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    payload = load_training_checkpoint(checkpoint_path)
    args = reconstruct_training_args(payload, cli)

    # Gates, fit/dev data, model, and checkpoint are all authenticated before
    # the first include_test=True call.
    state = train_trace._prepare(args)
    _restore_best_dev(args, state, payload)
    dev_lock = validate_dev_bundle_lock(
        cli.dev_bundle_metadata,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_method=args.method,
        expected_data_provenance_sha256=canonical_json_sha256(
            state["data_provenance"]
        ),
    )

    geometry = state["geometry_gate"].geometry
    locked = build_trace_data(
        args.dataset_dir,
        train_manifest=args.train_manifest or None,
        test_manifest=args.test_manifest or None,
        image_size=(geometry.image_height, geometry.image_width),
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        include_test=True,
        train_horizontal_flip=False,
    )
    if locked.test is None:
        raise TraceLockedTestError("locked test dataset was not constructed")
    if (
        locked.train.names != state["bundle"].train.names
        or locked.dev.names != state["bundle"].dev.names
        or locked.test_names != state["bundle"].test_names
    ):
        raise TraceLockedTestError("locked test data changed the paired split assignment")

    loader = torch.utils.data.DataLoader(
        locked.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        persistent_workers=False,
        pin_memory=False,
    )
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for images, masks, names in loader:
            images = images.to(state["device"], dtype=torch.float32)
            if args.method == "trace":
                output = state["model"](
                    images, return_map=True, return_marginals=False
                )
                batch_scores = render_trace_atoms(output, geometry).scores
            else:
                batch_scores = state["model"](images)[:, 0]
            if not bool(torch.isfinite(batch_scores).all()):
                raise TraceLockedTestError("test inference emitted non-finite scores")
            scores.extend(np.asarray(batch_scores.cpu(), dtype=np.float64))
            targets.extend(np.asarray(masks[:, 0], dtype=np.uint8))
            sample_ids.extend(str(name) for name in names)
    if tuple(sample_ids) != locked.test_names:
        raise TraceLockedTestError("test inference order differs from locked manifest")

    destination = Path(cli.test_bundle).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise TraceLockedTestError("test bundle path must end in .npz")
    metadata_path = destination.with_suffix(".json")
    if not cli.overwrite and (destination.exists() or metadata_path.exists()):
        raise TraceLockedTestError("locked test output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                scores=np.stack(scores, axis=0),
                targets=np.stack(targets, axis=0),
                sample_ids=np.asarray(sample_ids, dtype=np.str_),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    metadata = {
        "schema_version": TEST_BUNDLE_SCHEMA_VERSION,
        "method": args.method,
        "split": "test",
        "selection_checkpoint_role": "best_dev",
        "checkpoint": provenance_path(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "dev_bundle_metadata": provenance_path(cli.dev_bundle_metadata),
        "dev_bundle_metadata_sha256": sha256_file(cli.dev_bundle_metadata),
        "dev_bundle_sha256": dev_lock["bundle_sha256"],
        "bundle": provenance_path(destination),
        "bundle_sha256": sha256_file(destination),
        "sample_count": len(sample_ids),
        "sample_ids_sha256": canonical_json_sha256(sample_ids),
        "test_split_sha256": locked.test.split_sha256,
        "geometry_sha256": state["geometry_gate"].geometry_sha256,
        "training_data_provenance_sha256": canonical_json_sha256(
            state["data_provenance"]
        ),
        "locked_data_provenance_sha256": canonical_json_sha256(locked.provenance()),
        "score_semantics": (
            "log P(Y_cell = emitted_MAP_atom | image), max over whole atoms"
            if args.method == "trace"
            else "independent Bernoulli foreground logit"
        ),
        "threshold_selection": "none; evaluate_trace.py uses dev scores only",
        "test_access": True,
        "test_used_for_checkpoint_or_hyperparameter_selection": False,
        "explicit_unlock_phrase_sha256": canonical_json_sha256(LOCK_PHRASE),
        "source_sha256": {
            "tools/infer_trace_locked_test.py": sha256_file(Path(__file__)),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev-bundle-metadata", required=True)
    parser.add_argument("--test-bundle", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--test-manifest", default="")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--t0-a-report", required=True)
    parser.add_argument("--t0-b-dp-report", required=True)
    parser.add_argument("--t0-b-integration-report", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--unlock-test", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metadata = export_locked_test_bundle(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LOCK_PHRASE",
    "TEST_BUNDLE_SCHEMA_VERSION",
    "TraceLockedTestError",
    "build_parser",
    "export_locked_test_bundle",
    "main",
    "reconstruct_training_args",
    "require_explicit_unlock",
    "validate_dev_bundle_lock",
]
