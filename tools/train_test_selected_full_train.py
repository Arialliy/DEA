#!/usr/bin/env python3
"""Run the canonical complete-train/test-selected MSHNet protocol.

The legacy ``main.Trainer`` assumes a fit/dev/test workflow.  This standalone
entry point reuses its model, optimizer, loss, metric, checkpoint and CLI
implementations while narrowly adapting the data contract requested here:

* optimize on every name in canonical ``img_idx/train_<dataset>.txt``;
* expose canonical ``img_idx/test_<dataset>.txt`` through Trainer's historical
  ``val_loader`` evaluation alias, with no independent validation split;
* evaluate after every ten completed epochs and once at the final epoch;
* select best-IoU and constrained best-PD/FA checkpoints on that test alias.

This is explicitly test-selected and must not be described as an unbiased
held-out estimate.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as dea_main  # noqa: E402
from utils.full_train_test_protocol import (  # noqa: E402
    CanonicalDatasetAudit,
    EVALUATION_EPOCH_RULE,
    PROTOCOL_VERSION,
    ProtocolContractError,
    RESUME_RNG_SEMANTICS,
    SELECTION_THRESHOLD,
    SELECTION_TIE_BREAK,
    TEST_INTERVAL,
    audit_canonical_dataset,
    build_protocol_metadata,
    evaluation_epochs,
    require_locked_test_interval,
    should_evaluate_epoch,
)


_ORIGINAL_GET_METHOD_METADATA = dea_main.get_method_metadata
_METADATA_CONTEXTS: Dict[int, Tuple[CanonicalDatasetAudit, int, bool]] = {}


def _protocol_get_method_metadata(args):
    base = _ORIGINAL_GET_METHOD_METADATA(args)
    context = _METADATA_CONTEXTS.get(id(args))
    if context is None:
        return base
    audit, test_interval, resumed_process = context
    return build_protocol_metadata(
        base,
        audit,
        test_interval=test_interval,
        resume=resumed_process,
    )


_protocol_get_method_metadata._full_train_test_protocol = True


def _contains_option(argv: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(option + "=") for token in argv)


def _resolved_split_path(dataset_dir: Path, value: str) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = dataset_dir / path
    return Path(os.path.abspath(os.fspath(path)))


def _exact_float(value: float, expected: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) != float(expected):
        raise ProtocolContractError(
            "%s is locked to %s, got %r" % (name, expected, value)
        )


def parse_protocol_args(argv: Optional[Sequence[str]] = None):
    """Parse main.py's CLI plus the locked protocol-only selector flags."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    protocol_parser = ArgumentParser(add_help=False)
    protocol_parser.add_argument("--test-interval", type=int, default=TEST_INTERVAL)
    protocol_parser.add_argument(
        "--threshold", type=float, default=SELECTION_THRESHOLD
    )
    protocol_parser.add_argument(
        "--selection-tie-break",
        type=str,
        default=SELECTION_TIE_BREAK,
    )
    protocol_only, main_argv = protocol_parser.parse_known_args(raw_argv)
    require_locked_test_interval(protocol_only.test_interval)
    _exact_float(protocol_only.threshold, SELECTION_THRESHOLD, "--threshold")
    if protocol_only.selection_tie_break != SELECTION_TIE_BREAK:
        raise ProtocolContractError(
            "--selection-tie-break is locked to %s, got %r"
            % (SELECTION_TIE_BREAK, protocol_only.selection_tie_break)
        )

    if _contains_option(main_argv, "--val-split-file"):
        raise ProtocolContractError(
            "--val-split-file is forbidden: this protocol has no validation set"
        )

    bootstrap = ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--dataset-dir", type=str, default=dea_main.DEFAULT_DATASET_DIR
    )
    bootstrap_args, _ = bootstrap.parse_known_args(main_argv)
    audit = audit_canonical_dataset(bootstrap_args.dataset_dir)
    dataset_dir = Path(audit.dataset_dir)

    # Bootstrap the legacy Trainer's val-mode dataset with the exact test
    # manifest.  The subclass below audits that alias and erases all persisted
    # val semantics.  Explicit train/test arguments, when supplied, must point
    # to these same canonical files.
    split_bootstrap = ArgumentParser(add_help=False)
    split_bootstrap.add_argument("--train-split-file", type=str, default="")
    split_bootstrap.add_argument("--test-split-file", type=str, default="")
    split_bootstrap.add_argument("--val-fraction", type=float, default=0.0)
    split_args, _ = split_bootstrap.parse_known_args(main_argv)
    _exact_float(split_args.val_fraction, 0.0, "--val-fraction")
    supplied_train = _resolved_split_path(dataset_dir, split_args.train_split_file)
    supplied_test = _resolved_split_path(dataset_dir, split_args.test_split_file)
    if supplied_train is not None and supplied_train != Path(audit.train.path):
        raise ProtocolContractError(
            "--train-split-file must be canonical %s, got %s"
            % (audit.train.path, supplied_train)
        )
    if supplied_test is not None and supplied_test != Path(audit.test.path):
        raise ProtocolContractError(
            "--test-split-file must be canonical %s, got %s"
            % (audit.test.path, supplied_test)
        )

    injected = list(main_argv)
    if not _contains_option(injected, "--dataset-dir"):
        injected.extend(("--dataset-dir", audit.dataset_dir))
    if not _contains_option(injected, "--train-split-file"):
        injected.extend(("--train-split-file", audit.train.path))
    if not _contains_option(injected, "--test-split-file"):
        injected.extend(("--test-split-file", audit.test.path))
    if not _contains_option(injected, "--val-fraction"):
        injected.extend(("--val-fraction", "0"))
    injected.extend(("--val-split-file", audit.test.path))

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + injected
        args = dea_main.parse_args()
    finally:
        sys.argv = old_argv

    if args.mode != "train":
        raise ProtocolContractError("this entry point requires --mode train")
    if Path(os.path.abspath(args.dataset_dir)) != Path(audit.dataset_dir):
        raise ProtocolContractError("main CLI changed the audited dataset directory")

    args.dataset_dir = audit.dataset_dir
    args.train_split_file = audit.train.path
    args.test_split_file = audit.test.path
    # Internal bootstrap only.  FullTrainTestTrainer sanitizes these fields
    # before anything is persisted as public protocol metadata.
    args.val_split_file = audit.test.path
    args.val_fraction = 0.0
    args.test_interval = require_locked_test_interval(protocol_only.test_interval)
    args.threshold = float(protocol_only.threshold)
    args.selection_tie_break = protocol_only.selection_tie_break

    _exact_float(args.pd_fa_min_pd, 0.93, "--pd-fa-min-pd")
    _exact_float(args.pd_fa_min_iou, 0.655, "--pd-fa-min-iou")
    _exact_float(args.paired_baseline_iou, 0.0, "--paired-baseline-iou")
    _validate_baseline_semantics(args)
    return args, audit


def _validate_baseline_semantics(args) -> None:
    """Lock this entry point to the clean, from-scratch MSHNet baseline."""

    expected = {
        "model_type": "mshnet",
        "mshnet_objective": "sls",
        "mshnet_side_supervision": "canonical",
        "mshnet_train_graph": "canonical_warm",
        "location_loss": "legacy",
        "side_location_loss": "same",
    }
    mismatches = [
        "%s=%r (expected %r)" % (key, getattr(args, key), value)
        for key, value in expected.items()
        if getattr(args, key) != value
    ]
    _exact_float(args.lambda_location, 1.0, "--lambda-location")
    _exact_float(args.crwd_lambda, 0.0, "--crwd-lambda")
    for name in (
        "dea_lambda_single",
        "dea_lambda_dec",
        "dea_lambda_empty",
    ):
        _exact_float(getattr(args, name), 0.0, "--" + name.replace("_", "-"))
    if args.init_from_baseline:
        mismatches.append("init_from_baseline must be empty")
    if bool(args.reset_optimizer):
        mismatches.append("reset_optimizer must be false for strict resume")
    if mismatches:
        raise ProtocolContractError(
            "baseline entry point received non-canonical method semantics: %s"
            % "; ".join(mismatches)
        )


def _atomic_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _manifest_payload(names: Iterable[str]) -> str:
    return "".join(name + "\n" for name in names)


def _write_or_verify(path: Path, payload: str) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ProtocolContractError("invalid persisted manifest: %s" % path)
        existing = path.read_text(encoding="utf-8")
        if existing != payload:
            raise ProtocolContractError(
                "refusing to overwrite mismatched persisted manifest: %s" % path
            )
        return
    path.write_text(payload, encoding="utf-8")


def _checkpoint_protocol_mismatches(
    metadata: Dict[str, object],
    audit: CanonicalDatasetAudit,
) -> List[str]:
    expected = build_protocol_metadata({}, audit, TEST_INTERVAL, resume=False)
    keys = (
        "protocol",
        "protocol_version",
        "selection_split",
        "evaluation_split",
        "no_internal_holdout",
        "val_split_file",
        "val_split_sha256",
        "val_split_count",
        "test_interval",
        "evaluation_epoch_rule",
        "train_split_file",
        "train_split_count",
        "train_split_raw_sha256",
        "train_split_normalized_sha256",
        "test_split_file",
        "test_split_count",
        "test_split_raw_sha256",
        "test_split_normalized_sha256",
        "selection_threshold",
        "selection_prediction_rule",
        "selection_tie_break",
        "selection_best_iou_rule",
        "selection_pd_fa_rule",
        "selection_pd_fa_min_pd",
        "selection_pd_fa_min_iou",
        "selection_paired_baseline_iou",
        "train_loader_drop_last",
    )
    return [
        "%s expected=%r actual=%r" % (key, expected[key], metadata.get(key))
        for key in keys
        if metadata.get(key) != expected[key]
    ]


def _validate_resume_before_construction(
    args,
    audit: CanonicalDatasetAudit,
) -> Tuple[int, ...]:
    if not bool(args.if_checkpoint):
        return ()
    if not args.checkpoint_dir:
        raise ProtocolContractError(
            "resume requires explicit --checkpoint-dir; implicit latest-run routing is forbidden"
        )
    checkpoint_dir = Path(os.path.abspath(args.checkpoint_dir))
    if args.run_dir and Path(os.path.abspath(args.run_dir)) != checkpoint_dir:
        raise ProtocolContractError(
            "--run-dir, when provided for resume, must equal --checkpoint-dir"
        )
    checkpoint_path = checkpoint_dir / "checkpoint.pkl"
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ProtocolContractError("missing plain resume checkpoint: %s" % checkpoint_path)
    checkpoint = dea_main.load_torch_file(str(checkpoint_path))
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("method_meta"), dict
    ):
        raise ProtocolContractError("resume checkpoint lacks method_meta")
    mismatches = _checkpoint_protocol_mismatches(
        checkpoint["method_meta"], audit
    )
    if mismatches:
        raise ProtocolContractError(
            "resume checkpoint violates protocol: %s" % "; ".join(mismatches)
        )
    checkpoint_metadata = checkpoint["method_meta"]
    semantic_expected = {
        "model_type": "mshnet",
        "mshnet_objective": "sls",
        "mshnet_side_supervision": "canonical",
        "mshnet_train_graph": "canonical_warm",
        "location_loss": "legacy",
        "side_location_loss": "same",
        "lambda_location": 1.0,
        "warm_epoch": int(args.warm_epoch),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "run_label": args.run_label,
    }
    semantic_mismatches = [
        "%s expected=%r actual=%r"
        % (key, expected, checkpoint_metadata.get(key))
        for key, expected in semantic_expected.items()
        if checkpoint_metadata.get(key) != expected
    ]
    if semantic_mismatches:
        raise ProtocolContractError(
            "resume method/seed semantics mismatch: %s"
            % "; ".join(semantic_mismatches)
        )

    run_config_path = checkpoint_dir / "run_config.json"
    if run_config_path.is_symlink() or not run_config_path.is_file():
        raise ProtocolContractError(
            "resume requires a plain prior run_config.json: %s" % run_config_path
        )
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolContractError("invalid prior run_config.json") from exc
    if not isinstance(run_config, dict) or not isinstance(
        run_config.get("args"), dict
    ) or not isinstance(run_config.get("method_meta"), dict):
        raise ProtocolContractError("prior run_config.json has invalid schema")
    config_protocol_mismatches = _checkpoint_protocol_mismatches(
        run_config["method_meta"], audit
    )
    if config_protocol_mismatches:
        raise ProtocolContractError(
            "prior run_config protocol mismatch: %s"
            % "; ".join(config_protocol_mismatches)
        )
    config_expected = {
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "run_label": args.run_label,
        "model_type": "mshnet",
        "mshnet_objective": "sls",
        "mshnet_side_supervision": "canonical",
        "mshnet_train_graph": "canonical_warm",
        "location_loss": "legacy",
        "side_location_loss": "same",
        "lambda_location": 1.0,
        "warm_epoch": int(args.warm_epoch),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
    }
    config_args = run_config["args"]
    config_mismatches = [
        "%s expected=%r actual=%r" % (key, expected, config_args.get(key))
        for key, expected in config_expected.items()
        if config_args.get(key) != expected
    ]
    if config_mismatches:
        raise ProtocolContractError(
            "resume CLI/prior run_config mismatch: %s"
            % "; ".join(config_mismatches)
        )

    epoch = int(checkpoint.get("epoch", -1))
    if epoch < 0 or epoch >= int(args.epochs):
        raise ProtocolContractError(
            "resume checkpoint epoch %d is invalid for total epochs %d"
            % (epoch, args.epochs)
        )
    if not should_evaluate_epoch(epoch, int(args.epochs), TEST_INTERVAL):
        raise ProtocolContractError(
            "resume checkpoint epoch %d is not a scheduled evaluation boundary"
            % epoch
        )

    summary_path = checkpoint_dir / "protocol_summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ProtocolContractError(
            "resume requires a plain prior protocol_summary.json"
        )
    try:
        prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolContractError("invalid prior protocol_summary.json") from exc
    if prior_summary.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolContractError("prior summary protocol mismatch")
    if prior_summary.get("dataset") != audit.dataset_name:
        raise ProtocolContractError("prior summary dataset mismatch")
    if prior_summary.get("total_epochs") != int(args.epochs):
        raise ProtocolContractError("prior summary total_epochs mismatch")
    raw_executed = prior_summary.get(
        "executed_evaluation_epochs_zero_based", []
    )
    if not isinstance(raw_executed, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_executed
    ):
        raise ProtocolContractError("prior summary executed epochs are invalid")
    prior_executed = tuple(raw_executed)
    full_schedule = evaluation_epochs(0, int(args.epochs), TEST_INTERVAL)
    expected_prefix = tuple(value for value in full_schedule if value <= epoch)
    if prior_executed != expected_prefix:
        raise ProtocolContractError(
            "prior summary evaluation prefix mismatch: expected=%s actual=%s"
            % (expected_prefix, prior_executed)
        )
    if not prior_executed or prior_executed[-1] != epoch:
        raise ProtocolContractError(
            "prior summary does not end at resume checkpoint epoch %d" % epoch
        )
    return prior_executed


class FullTrainTestTrainer(dea_main.Trainer):
    """Legacy Trainer with one strictly audited val==complete-test alias."""

    def __init__(
        self,
        args,
        audit: CanonicalDatasetAudit,
        test_interval: int = TEST_INTERVAL,
    ):
        self.protocol_audit = audit
        self.test_interval = require_locked_test_interval(test_interval)
        self.resumed_process = bool(args.if_checkpoint)
        self.prior_evaluation_epochs = _validate_resume_before_construction(
            args, audit
        )
        _METADATA_CONTEXTS[id(args)] = (
            audit,
            self.test_interval,
            self.resumed_process,
        )
        dea_main.get_method_metadata = _protocol_get_method_metadata
        try:
            super().__init__(args)
        except Exception:
            _METADATA_CONTEXTS.pop(id(args), None)
            raise

        self._assert_loaded_dataset_contract()
        self._rebuild_full_train_loader()
        if (
            not getattr(self.PD_FA, "thresholds", ())
            or float(self.PD_FA.thresholds[0]) != SELECTION_THRESHOLD
        ):
            raise ProtocolContractError(
                "legacy PD/FA evaluator operating point must be probability 0.5"
            )

        # The object was constructed in legacy val mode only to reuse the
        # evaluator.  From this point onward it carries truthful test labels.
        self.val_dataset.mode = "test"
        self.val_dataset.list_dir = audit.test.path
        self.val_dataset.split_source = audit.test.path
        args.internal_evaluation_alias = "val_loader_is_complete_canonical_test"
        args.val_split_file = ""
        args.val_split_sha256 = ""
        args.val_fraction = 0.0
        self._write_protocol_summary(
            status="running",
            current_executed=(),
            cumulative_executed=self.prior_evaluation_epochs,
        )

    @staticmethod
    def assert_disjoint_splits(trainset, valset, testset):
        train_names = tuple(trainset.names)
        val_names = tuple(valset.names)
        test_names = tuple(testset.names)
        overlap = sorted(set(train_names).intersection(test_names))
        if overlap:
            raise ProtocolContractError(
                "train/test split leakage detected (%d), e.g. %s"
                % (len(overlap), overlap[:5])
            )
        if val_names != test_names:
            raise ProtocolContractError(
                "legacy evaluation alias must equal canonical test in exact order"
            )
        if valset.split_sha256 != testset.split_sha256:
            raise ProtocolContractError(
                "legacy evaluation alias/test normalized hashes differ"
            )

    def _assert_loaded_dataset_contract(self) -> None:
        audit = self.protocol_audit
        if tuple(self.train_dataset.names) != audit.train.names:
            raise ProtocolContractError(
                "Trainer train dataset is not the complete canonical train manifest"
            )
        if tuple(self.val_dataset.names) != audit.test.names:
            raise ProtocolContractError(
                "Trainer evaluation dataset is not the complete canonical test manifest"
            )
        if self.train_dataset.split_sha256 != audit.train.normalized_sha256:
            raise ProtocolContractError("Trainer train split hash changed")
        if self.val_dataset.split_sha256 != audit.test.normalized_sha256:
            raise ProtocolContractError("Trainer test split hash changed")

    def _loader_kwargs(self, generator_seed: int) -> Dict[str, object]:
        generator = torch.Generator()
        generator.manual_seed(generator_seed)
        kwargs: Dict[str, object] = {
            "num_workers": self.args.num_workers,
            "pin_memory": self.args.pin_memory,
            "persistent_workers": self.args.num_workers > 0,
            "worker_init_fn": dea_main.seed_worker,
            "generator": generator,
        }
        if self.args.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return kwargs

    def _rebuild_full_train_loader(self) -> None:
        self.train_loader = dea_main.Data.DataLoader(
            self.train_dataset,
            self.args.batch_size,
            shuffle=True,
            drop_last=False,
            **self._loader_kwargs(self.args.seed),
        )
        if self.train_loader.drop_last:
            raise ProtocolContractError("train DataLoader must use drop_last=False")

    def print_split_summary(self):
        print(
            "split train: n=%d normalized_sha256=%s raw_sha256=%s source=%s"
            % (
                len(self.train_dataset),
                self.protocol_audit.train.normalized_sha256[:12],
                self.protocol_audit.train.raw_sha256[:12],
                self.protocol_audit.train.path,
            )
        )
        print(
            "split test (legacy val_loader alias): n=%d normalized_sha256=%s "
            "raw_sha256=%s source=%s"
            % (
                len(self.val_dataset),
                self.protocol_audit.test.normalized_sha256[:12],
                self.protocol_audit.test.raw_sha256[:12],
                self.protocol_audit.test.path,
            )
        )

    def persist_split_manifests(self):
        if self.mode != "train":
            raise ProtocolContractError("protocol trainer must remain in train mode")
        save_folder = Path(self.save_folder)
        stale_val = save_folder / "split_val.txt"
        if stale_val.exists() or stale_val.is_symlink():
            raise ProtocolContractError(
                "split_val.txt is forbidden in a no-holdout protocol run"
            )
        _write_or_verify(
            save_folder / "split_train.txt",
            _manifest_payload(self.protocol_audit.train.names),
        )
        _write_or_verify(
            save_folder / "split_test.txt",
            _manifest_payload(self.protocol_audit.test.names),
        )

        serializable_args = {}
        for key, value in sorted(vars(self.args).items()):
            if key.startswith("_"):
                continue
            if value is None or isinstance(value, (bool, int, float, str)):
                serializable_args[key] = value
            else:
                serializable_args[key] = repr(value)
        serializable_args["val_split_file"] = ""
        serializable_args["val_split_sha256"] = ""
        serializable_args["val_fraction"] = 0.0
        _atomic_json(
            save_folder / "run_config.json",
            {
                "args": serializable_args,
                "method_meta": dea_main.get_method_metadata(self.args),
            },
        )

    def _summary_payload(
        self,
        *,
        status: str,
        current_executed: Sequence[int],
        cumulative_executed: Sequence[int],
    ) -> Dict[str, object]:
        checkpoint_selection = self._checkpoint_selection_summary()
        if status == "complete" and (
            checkpoint_selection["best_iou"]["status"] != "found"
        ):
            raise ProtocolContractError(
                "cannot mark protocol complete without checkpoint_best_iou.pkl"
            )
        return {
            "protocol": PROTOCOL_VERSION,
            "status": status,
            "dataset": self.protocol_audit.dataset_name,
            "run_dir": str(self.save_folder),
            "selection_split": "test",
            "no_internal_holdout": True,
            "test_interval": self.test_interval,
            "evaluation_epoch_rule": EVALUATION_EPOCH_RULE,
            "start_epoch": int(self.start_epoch),
            "total_epochs": int(self.args.epochs),
            "planned_evaluation_epochs_zero_based": list(
                evaluation_epochs(
                    self.start_epoch,
                    self.args.epochs,
                    self.test_interval,
                )
            ),
            "current_process_evaluation_epochs_zero_based": list(
                current_executed
            ),
            "executed_evaluation_epochs_zero_based": list(
                cumulative_executed
            ),
            "last_completed_epoch_zero_based": (
                int(self.args.epochs) - 1 if status == "complete" else None
            ),
            "resumed_process": self.resumed_process,
            "resume_rng_semantics": RESUME_RNG_SEMANTICS,
            "checkpoint_selection": checkpoint_selection,
            "method_meta": dea_main.get_method_metadata(self.args),
        }

    def _write_protocol_summary(
        self,
        *,
        status: str,
        current_executed: Sequence[int],
        cumulative_executed: Sequence[int],
    ) -> None:
        _atomic_json(
            Path(self.save_folder) / "protocol_summary.json",
            self._summary_payload(
                status=status,
                current_executed=current_executed,
                cumulative_executed=cumulative_executed,
            ),
        )

    def _one_checkpoint_selection(
        self,
        filename: str,
        *,
        required: bool,
    ) -> Dict[str, object]:
        path = Path(self.save_folder) / filename
        if not path.exists():
            return {
                "status": "not_found",
                "file": None,
                "reason": (
                    "required_checkpoint_not_yet_available"
                    if required
                    else "no_eligible_epoch"
                ),
            }
        if path.is_symlink() or not path.is_file():
            raise ProtocolContractError("invalid selected checkpoint: %s" % path)
        checkpoint = dea_main.load_torch_file(str(path))
        if not isinstance(checkpoint, dict):
            raise ProtocolContractError("selected checkpoint is not a state dict")
        metadata = checkpoint.get("method_meta")
        if not isinstance(metadata, dict):
            raise ProtocolContractError("selected checkpoint lacks method_meta")
        mismatches = _checkpoint_protocol_mismatches(
            metadata, self.protocol_audit
        )
        if mismatches:
            raise ProtocolContractError(
                "selected checkpoint protocol mismatch: %s"
                % "; ".join(mismatches)
            )
        epoch = checkpoint.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, Integral):
            raise ProtocolContractError("selected checkpoint epoch is invalid")
        if int(epoch) not in evaluation_epochs(
            0, int(self.args.epochs), self.test_interval
        ):
            raise ProtocolContractError(
                "selected checkpoint epoch is outside the evaluation schedule"
            )
        for key in ("iou", "pd", "fa"):
            value = checkpoint.get(key)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ProtocolContractError(
                    "selected checkpoint %s metric is invalid" % key
                )
            if not math.isfinite(float(value)):
                raise ProtocolContractError(
                    "selected checkpoint %s metric is non-finite" % key
                )
        if filename == "checkpoint_pd_fa_best.pkl" and (
            float(checkpoint["pd"]) < 0.93
            or float(checkpoint["iou"]) < 0.655
        ):
            raise ProtocolContractError(
                "constrained minimum-FA checkpoint violates Pd/IoU eligibility"
            )
        return {
            "status": "found",
            "file": filename,
            "sha256": dea_main.sha256_file(str(path)),
            "epoch_zero_based": int(epoch),
            "iou": float(checkpoint["iou"]),
            "pd": float(checkpoint["pd"]),
            "fa": float(checkpoint["fa"]),
        }

    def _checkpoint_selection_summary(self) -> Dict[str, object]:
        return {
            "best_iou": self._one_checkpoint_selection(
                "checkpoint_best_iou.pkl", required=True
            ),
            "constrained_min_fa": self._one_checkpoint_selection(
                "checkpoint_pd_fa_best.pkl", required=False
            ),
        }

    def assert_saved_checkpoint_contracts(self) -> None:
        latest_path = Path(self.save_folder) / "checkpoint.pkl"
        if not latest_path.exists():
            raise ProtocolContractError(
                "scheduled evaluation did not create checkpoint.pkl"
            )
        for name in (
            "checkpoint.pkl",
            "checkpoint_best_iou.pkl",
            "checkpoint_pd_fa_best.pkl",
        ):
            path = Path(self.save_folder) / name
            if not path.exists():
                continue
            if path.is_symlink() or not path.is_file():
                raise ProtocolContractError("invalid checkpoint path: %s" % path)
            checkpoint = dea_main.load_torch_file(str(path))
            metadata = (
                checkpoint.get("method_meta")
                if isinstance(checkpoint, dict)
                else None
            )
            if not isinstance(metadata, dict):
                raise ProtocolContractError("%s lacks method_meta" % path)
            mismatches = _checkpoint_protocol_mismatches(
                metadata, self.protocol_audit
            )
            if mismatches:
                raise ProtocolContractError(
                    "%s protocol metadata mismatch: %s"
                    % (path, "; ".join(mismatches))
                )


def run_training_protocol(
    trainer: FullTrainTestTrainer,
    total_epochs: Optional[int] = None,
    test_interval: int = TEST_INTERVAL,
) -> Tuple[int, ...]:
    """Train from ``trainer.start_epoch`` and evaluate only scheduled epochs."""

    interval = require_locked_test_interval(test_interval)
    total = int(trainer.args.epochs if total_epochs is None else total_epochs)
    if total != int(trainer.args.epochs):
        raise ProtocolContractError(
            "total_epochs must equal the parsed --epochs value"
        )
    planned = evaluation_epochs(trainer.start_epoch, total, interval)
    current_executed: List[int] = []
    cumulative_executed: List[int] = list(trainer.prior_evaluation_epochs)
    for epoch in range(int(trainer.start_epoch), total):
        trainer.train(epoch)
        if should_evaluate_epoch(epoch, total, interval):
            trainer.test(epoch)
            trainer.assert_saved_checkpoint_contracts()
            current_executed.append(epoch)
            cumulative_executed.append(epoch)
            trainer._write_protocol_summary(
                status="running",
                current_executed=current_executed,
                cumulative_executed=cumulative_executed,
            )
    if tuple(current_executed) != planned:
        raise ProtocolContractError(
            "evaluation schedule mismatch: planned=%s executed=%s"
            % (planned, tuple(current_executed))
        )
    full_schedule = evaluation_epochs(0, total, interval)
    if tuple(cumulative_executed) != full_schedule:
        raise ProtocolContractError(
            "cumulative evaluation schedule mismatch: planned=%s executed=%s"
            % (full_schedule, tuple(cumulative_executed))
        )
    trainer._write_protocol_summary(
        status="complete",
        current_executed=current_executed,
        cumulative_executed=cumulative_executed,
    )
    return tuple(current_executed)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, audit = parse_protocol_args(argv)
    dea_main.seed_everything(args.seed, args.deterministic)
    trainer = FullTrainTestTrainer(args, audit, args.test_interval)
    if trainer.resumed_process:
        print("resume RNG notice: %s" % RESUME_RNG_SEMANTICS)
    executed = run_training_protocol(trainer, args.epochs, args.test_interval)
    print(
        "protocol complete: %s dataset=%s eval_epochs_zero_based=%s run_dir=%s"
        % (PROTOCOL_VERSION, audit.dataset_name, list(executed), trainer.save_folder)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
