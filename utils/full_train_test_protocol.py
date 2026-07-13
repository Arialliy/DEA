"""Fail-closed contract for the canonical full-train/test-selected protocol.

This module deliberately contains no dataset or model construction.  It
audits the immutable on-disk contract and provides the small scheduling and
metadata helpers used by ``tools/train_test_selected_full_train.py``.

The protocol is intentionally *test selected*: the complete canonical train
manifest is used for optimization, and the complete canonical test manifest
is evaluated every ten completed epochs (plus the final epoch when needed).
There is no internal fit/dev holdout.  This is useful for reproducing the
historical reporting convention requested for this repository, but it is not
an unbiased model-selection estimate and must always be labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePath
from typing import Dict, Iterable, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DATASETS_ROOT = PROJECT_ROOT / "datasets"
CANONICAL_DATASET_NAMES = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")

# Frozen from the repository's canonical manifests on 2026-07-12.  Both the
# byte-level digest (which preserves CRLF/LF) and the loader-equivalent
# normalized-name digest are locked.  Keeping the contract here gives the
# direct training entry point the same fail-closed protection as any batch
# scheduler importing it.
FROZEN_CONTRACT = {
    "IRSTD-1K": {
        "train": {
            "count": 800,
            "raw_sha256": "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744",
            "normalized_sha256": "b698d2d9dbe9e26e1875978d23450e1e6ec45fd71d56d31415007f56c40bba88",
        },
        "test": {
            "count": 201,
            "raw_sha256": "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579",
            "normalized_sha256": "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579",
        },
    },
    "NUAA-SIRST": {
        "train": {
            "count": 213,
            "raw_sha256": "324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4",
            "normalized_sha256": "815dcca749f087f27f5dad4b447015aee70bd7ae7779d6fdd7d6efa6d5c6943f",
        },
        "test": {
            "count": 214,
            "raw_sha256": "e49023203a323c247306b314f23c8b3b917093a26984067792355adff7a8386e",
            "normalized_sha256": "395eecd6bf0ed2a59f531de9145688597632c68f9d0933359aadcb93ec1a60b5",
        },
    },
    "NUDT-SIRST": {
        "train": {
            "count": 663,
            "raw_sha256": "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3",
            "normalized_sha256": "dc555df66b62dd1ea98d119ace8fe8ae86de94f3e4833d8d81e90c0e1f287922",
        },
        "test": {
            "count": 664,
            "raw_sha256": "a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e",
            "normalized_sha256": "cec44220c69d89a5b3fd245b8ee911404e959fef80bd96b32b6b74f28bb32af0",
        },
    },
}

PROTOCOL_VERSION = "test_selected_full_train_interval_v1"
TEST_INTERVAL = 10
EVALUATION_EPOCH_RULE = (
    "(epoch_zero_based + 1) % test_interval == 0 or "
    "epoch_zero_based == total_epochs - 1"
)
RESUME_RNG_SEMANTICS = (
    "model_and_optimizer_are_restored_but_process_and_dataloader_rng_are_"
    "reseeded_from_cli_seed; resumed_trajectory_is_not_bitwise_equivalent_to_"
    "an_uninterrupted_run"
)
SELECTION_THRESHOLD = 0.5
SELECTION_TIE_BREAK = "earliest_epoch"
SELECTION_BEST_IOU_RULE = "strictly_greater_iou; ties_keep_earliest_epoch"
SELECTION_PD_FA_RULE = (
    "pd>=0.93 and iou>=0.655 then strictly_minimum_fa; "
    "ties_keep_earliest_epoch"
)


class ProtocolContractError(RuntimeError):
    """The requested run does not satisfy the canonical protocol contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalised_payload(names: Iterable[str]) -> bytes:
    return (("\n".join(names)) + "\n").encode("utf-8")


def _require_plain_path(path: Path, *, kind: str) -> None:
    """Require an existing, non-symlink canonical path of the requested kind."""

    if path.is_symlink():
        raise ProtocolContractError("symbolic links are forbidden: %s" % path)
    if kind == "file":
        valid = path.is_file()
    elif kind == "directory":
        valid = path.is_dir()
    else:  # pragma: no cover - private programming error
        raise ValueError("unknown path kind: %s" % kind)
    if not valid:
        raise ProtocolContractError("missing %s: %s" % (kind, path))


def _require_canonical_root(root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(root)))
    expected = Path(os.path.abspath(os.fspath(CANONICAL_DATASETS_ROOT)))
    if root != expected:
        raise ProtocolContractError(
            "dataset root must be exactly %s, got %s" % (expected, root)
        )
    _require_plain_path(root, kind="directory")
    # Comparing the lexical and resolved paths rejects a symlink in any path
    # component, not merely a symlink at the final component.
    if root.resolve(strict=True) != root:
        raise ProtocolContractError(
            "canonical dataset root contains a symbolic-link component: %s"
            % root
        )
    return root


def _safe_sample_name(name: str, manifest_path: Path) -> None:
    if not name:
        raise ProtocolContractError("blank sample name in %s" % manifest_path)
    if name != name.strip():
        raise ProtocolContractError(
            "sample names may not contain surrounding whitespace in %s: %r"
            % (manifest_path, name)
        )
    if "\x00" in name or PurePath(name).name != name or name in (".", ".."):
        raise ProtocolContractError(
            "unsafe sample name in %s: %r" % (manifest_path, name)
        )


@dataclass(frozen=True)
class SplitManifestAudit:
    split: str
    path: str
    count: int
    raw_sha256: str
    normalized_sha256: str
    names: Tuple[str, ...]

    def to_metadata(self) -> Dict[str, object]:
        prefix = "%s_split" % self.split
        return {
            "%s_file" % prefix: self.path,
            "%s_count" % prefix: self.count,
            "%s_raw_sha256" % prefix: self.raw_sha256,
            "%s_normalized_sha256" % prefix: self.normalized_sha256,
        }


@dataclass(frozen=True)
class CanonicalDatasetAudit:
    dataset_name: str
    dataset_dir: str
    train: SplitManifestAudit
    test: SplitManifestAudit
    image_count: int
    mask_count: int

    def to_metadata(self) -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "canonical_datasets_root": str(CANONICAL_DATASETS_ROOT),
            "dataset_name": self.dataset_name,
            "dataset_dir": self.dataset_dir,
            "image_count": self.image_count,
            "mask_count": self.mask_count,
        }
        metadata.update(self.train.to_metadata())
        metadata.update(self.test.to_metadata())
        return metadata


def _read_manifest(path: Path, split: str) -> SplitManifestAudit:
    _require_plain_path(path, kind="file")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolContractError(
            "split manifest is not strict UTF-8: %s" % path
        ) from exc

    # splitlines intentionally normalises the CRLF/LF mixture present in the
    # canonical repository.  Empty interior/trailing logical lines are
    # rejected rather than silently skipped as the legacy loader would do.
    logical_lines = text.splitlines()
    if not logical_lines:
        raise ProtocolContractError("empty split manifest: %s" % path)
    for name in logical_lines:
        _safe_sample_name(name, path)
    if len(logical_lines) != len(set(logical_lines)):
        raise ProtocolContractError("duplicate sample names in %s" % path)

    names = tuple(logical_lines)
    return SplitManifestAudit(
        split=split,
        path=str(path),
        count=len(names),
        raw_sha256=_sha256_bytes(raw),
        normalized_sha256=_sha256_bytes(_normalised_payload(names)),
        names=names,
    )


def _assert_frozen_manifest(
    dataset_name: str,
    audit: SplitManifestAudit,
) -> None:
    try:
        expected = FROZEN_CONTRACT[dataset_name][audit.split]
    except (KeyError, TypeError) as exc:
        raise ProtocolContractError(
            "missing frozen manifest contract for %s/%s"
            % (dataset_name, audit.split)
        ) from exc
    actual = {
        "count": audit.count,
        "raw_sha256": audit.raw_sha256,
        "normalized_sha256": audit.normalized_sha256,
    }
    mismatches = [
        "%s expected=%r actual=%r" % (key, expected.get(key), actual[key])
        for key in actual
        if expected.get(key) != actual[key]
    ]
    if mismatches:
        raise ProtocolContractError(
            "frozen canonical manifest mismatch for %s/%s: %s"
            % (dataset_name, audit.split, "; ".join(mismatches))
        )


def _png_inventory(directory: Path) -> Tuple[str, ...]:
    _require_plain_path(directory, kind="directory")
    inventory = []
    for path in directory.iterdir():
        if path.is_symlink():
            raise ProtocolContractError(
                "symbolic links are forbidden in asset directories: %s" % path
            )
        if path.suffix.lower() != ".png":
            continue
        _require_plain_path(path, kind="file")
        with path.open("rb") as handle:
            signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ProtocolContractError("invalid PNG signature: %s" % path)
        inventory.append(path.stem)
    return tuple(sorted(inventory))


def _resolve_dataset_dir(dataset_dir: os.PathLike | str) -> Tuple[Path, str]:
    requested = Path(os.path.abspath(os.fspath(dataset_dir)))
    root = _require_canonical_root(requested.parent)
    dataset_name = requested.name
    if dataset_name not in CANONICAL_DATASET_NAMES:
        raise ProtocolContractError(
            "dataset must be one of %s, got %r"
            % (CANONICAL_DATASET_NAMES, dataset_name)
        )
    expected = root / dataset_name
    if requested != expected:
        raise ProtocolContractError(
            "dataset directory must be exactly %s, got %s"
            % (expected, requested)
        )
    _require_plain_path(expected, kind="directory")
    if expected.resolve(strict=True) != expected:
        raise ProtocolContractError(
            "dataset directory contains a symbolic-link component: %s"
            % expected
        )
    return expected, dataset_name


def audit_canonical_dataset(
    dataset_dir: os.PathLike | str,
) -> CanonicalDatasetAudit:
    """Audit one exact ``datasets/<name>`` train/test contract.

    Only ``train_<name>.txt`` and ``test_<name>.txt`` are opened.  In
    particular, the NUDT ``hcval`` diagnostic manifest is never discovered or
    read by this protocol.
    """

    dataset_path, dataset_name = _resolve_dataset_dir(dataset_dir)
    img_idx_dir = dataset_path / "img_idx"
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    for directory in (img_idx_dir, images_dir, masks_dir):
        _require_plain_path(directory, kind="directory")

    train_path = img_idx_dir / ("train_%s.txt" % dataset_name)
    test_path = img_idx_dir / ("test_%s.txt" % dataset_name)
    train = _read_manifest(train_path, "train")
    test = _read_manifest(test_path, "test")
    _assert_frozen_manifest(dataset_name, train)
    _assert_frozen_manifest(dataset_name, test)

    overlap = sorted(set(train.names).intersection(test.names))
    if overlap:
        raise ProtocolContractError(
            "canonical train/test overlap (%d samples), e.g. %s"
            % (len(overlap), overlap[:5])
        )

    declared = set(train.names).union(test.names)
    images = _png_inventory(images_dir)
    masks = _png_inventory(masks_dir)
    image_names = set(images)
    mask_names = set(masks)
    if image_names != mask_names:
        missing_masks = sorted(image_names - mask_names)
        missing_images = sorted(mask_names - image_names)
        raise ProtocolContractError(
            "image/mask inventory mismatch: missing_masks=%s missing_images=%s"
            % (missing_masks[:5], missing_images[:5])
        )
    if declared != image_names:
        missing_files = sorted(declared - image_names)
        undeclared_files = sorted(image_names - declared)
        raise ProtocolContractError(
            "img_idx/inventory mismatch: missing=%s undeclared=%s"
            % (missing_files[:5], undeclared_files[:5])
        )

    # Inventory equality proves existence, but retain explicit per-sample
    # checks to reject a race or a symbolic link introduced after listing.
    for name in tuple(train.names) + tuple(test.names):
        _require_plain_path(images_dir / (name + ".png"), kind="file")
        _require_plain_path(masks_dir / (name + ".png"), kind="file")

    return CanonicalDatasetAudit(
        dataset_name=dataset_name,
        dataset_dir=str(dataset_path),
        train=train,
        test=test,
        image_count=len(images),
        mask_count=len(masks),
    )


def audit_all_canonical_datasets() -> Dict[str, CanonicalDatasetAudit]:
    """Audit all three fixed dataset contracts without glob-based routing."""

    root = _require_canonical_root(CANONICAL_DATASETS_ROOT)
    return {
        name: audit_canonical_dataset(root / name)
        for name in CANONICAL_DATASET_NAMES
    }


def require_locked_test_interval(value: int) -> int:
    try:
        interval = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolContractError("test interval must be integer 10") from exc
    if interval != TEST_INTERVAL:
        raise ProtocolContractError(
            "this protocol locks --test-interval to %d, got %r"
            % (TEST_INTERVAL, value)
        )
    return interval


def should_evaluate_epoch(
    epoch: int,
    total_epochs: int,
    test_interval: int = TEST_INTERVAL,
) -> bool:
    interval = require_locked_test_interval(test_interval)
    epoch = int(epoch)
    total_epochs = int(total_epochs)
    if total_epochs <= 0:
        raise ProtocolContractError("total_epochs must be positive")
    if epoch < 0 or epoch >= total_epochs:
        raise ProtocolContractError(
            "epoch must satisfy 0 <= epoch < total_epochs, got %d/%d"
            % (epoch, total_epochs)
        )
    return ((epoch + 1) % interval == 0) or (epoch == total_epochs - 1)


def evaluation_epochs(
    start_epoch: int,
    total_epochs: int,
    test_interval: int = TEST_INTERVAL,
) -> Tuple[int, ...]:
    require_locked_test_interval(test_interval)
    start_epoch = int(start_epoch)
    total_epochs = int(total_epochs)
    if total_epochs <= 0:
        raise ProtocolContractError("total_epochs must be positive")
    if start_epoch < 0 or start_epoch > total_epochs:
        raise ProtocolContractError(
            "start_epoch must satisfy 0 <= start_epoch <= total_epochs"
        )
    return tuple(
        epoch
        for epoch in range(start_epoch, total_epochs)
        if should_evaluate_epoch(epoch, total_epochs, test_interval)
    )


def build_protocol_metadata(
    base_metadata: Mapping[str, object],
    audit: CanonicalDatasetAudit,
    test_interval: int = TEST_INTERVAL,
    resume: bool = False,
) -> Dict[str, object]:
    """Add explicit test-selection semantics to normal Trainer metadata."""

    interval = require_locked_test_interval(test_interval)
    metadata = dict(base_metadata)
    metadata.update(audit.to_metadata())
    metadata.update(
        {
            "protocol": PROTOCOL_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "selection_split": "test",
            "evaluation_split": "test",
            "evaluation_alias": "val_loader_is_complete_canonical_test",
            "no_internal_holdout": True,
            "val_split_file": "",
            "val_split_sha256": "",
            "val_split_count": 0,
            "val_fraction": 0.0,
            "test_interval": interval,
            "evaluation_epoch_rule": EVALUATION_EPOCH_RULE,
            "resume": bool(resume),
            "resumed_process": bool(resume),
            "resume_rng_semantics": RESUME_RNG_SEMANTICS,
            "selection_threshold": SELECTION_THRESHOLD,
            "selection_prediction_rule": "sigmoid(logit) > 0.5",
            "selection_tie_break": SELECTION_TIE_BREAK,
            "selection_best_iou_rule": SELECTION_BEST_IOU_RULE,
            "selection_pd_fa_rule": SELECTION_PD_FA_RULE,
            "selection_pd_fa_min_pd": 0.93,
            "selection_pd_fa_min_iou": 0.655,
            "selection_paired_baseline_iou": 0.0,
            "train_loader_drop_last": False,
            # Main Trainer's normalized split keys remain authoritative for
            # compatibility with its strict resume check.
            "train_split_sha256": audit.train.normalized_sha256,
            "test_split_sha256": audit.test.normalized_sha256,
            "train_split_file": audit.train.path,
            "test_split_file": audit.test.path,
        }
    )
    return metadata
