"""Deterministic, paired data contract for TRACE and dense-head controls.

The legacy :mod:`utils.data` loader intentionally remains untouched.  This
module implements the stricter paper protocol shared by every paired method:

* derive fit/dev only from the canonical training manifest;
* audit fit/dev/test names pairwise before exposing a dataset;
* use a fixed resize and no geometry-changing augmentation by default;
* when a separately geometry-audited flip ablation is explicitly enabled,
  bind it to ``(seed, epoch, sample name)`` rather than process RNG; and
* record path-free hashes for manifests, assigned splits, and asset bytes.

``build_trace_data`` reads the test *manifest* during training so leakage can
be rejected, but defaults to ``include_test=False``.  In that mode test image
and mask files are neither checked, opened, nor hashed.  Final locked
evaluation must opt in explicitly with ``include_test=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path, PurePath
import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


TRACE_DATA_SCHEMA_VERSION = "trace.paired_data.v1"
# This identifier and byte-level key intentionally match the clean canonical
# MSHNet holdout implementation in ``utils.data.IRSTD_Dataset``.  Using a
# superficially similar, namespaced hash would create a different fit/dev
# population and make strict baseline-checkpoint provenance impossible.
SPLIT_ALGORITHM = "sha256_seed_nul_name_rank_clean_mshnet_v1"
FLIP_ALGORITHM = "sha256_seed_epoch_name_bit0_v1"
IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

try:  # Pillow >= 9
    _BILINEAR = Image.Resampling.BILINEAR
    _NEAREST = Image.Resampling.NEAREST
    _FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
except AttributeError:  # pragma: no cover - compatibility with old Pillow
    _BILINEAR = Image.BILINEAR
    _NEAREST = Image.NEAREST
    _FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT


class TraceDataContractError(RuntimeError):
    """Raised when data cannot satisfy the leakage-free paired contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalised_name_payload(names: Sequence[str]) -> bytes:
    return (("\n".join(names)) + "\n").encode("utf-8")


def _update_framed(digest: "hashlib._Hash", payload: bytes) -> None:
    """Update a digest without ambiguous concatenation boundaries."""

    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise TraceDataContractError("missing asset file: %s" % path) from exc
    except OSError as exc:
        raise TraceDataContractError("cannot read asset file: %s" % path) from exc
    return digest.hexdigest()


def _require_integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TraceDataContractError("%s must be an integer" % field)
    result = int(value)
    if result < minimum:
        raise TraceDataContractError(
            "%s must be >= %d, got %d" % (field, minimum, result)
        )
    return result


def _safe_sample_name(name: str, *, source: str) -> None:
    if not isinstance(name, str) or not name:
        raise TraceDataContractError("empty sample name in %s" % source)
    if name != name.strip():
        raise TraceDataContractError(
            "sample names cannot contain surrounding whitespace in %s: %r"
            % (source, name)
        )
    if (
        "\x00" in name
        or "/" in name
        or "\\" in name
        or PurePath(name).name != name
        or name in (".", "..")
    ):
        raise TraceDataContractError(
            "unsafe sample name in %s: %r" % (source, name)
        )


def _validate_names(names: Iterable[str], *, split: str) -> Tuple[str, ...]:
    result = tuple(names)
    if not result:
        raise TraceDataContractError("empty %s split" % split)
    for name in result:
        _safe_sample_name(name, source="%s split" % split)
    if len(result) != len(set(result)):
        raise TraceDataContractError("duplicate sample names in %s split" % split)
    return result


def _resolve_dataset_root(dataset_dir: str | Path) -> Path:
    requested = Path(dataset_dir).expanduser()
    try:
        root = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TraceDataContractError(
            "missing dataset directory: %s" % requested
        ) from exc
    if not root.is_dir():
        raise TraceDataContractError("dataset path is not a directory: %s" % root)
    return root


def _resolve_manifest(root: Path, manifest: str | Path) -> Tuple[Path, str]:
    requested = Path(manifest).expanduser()
    candidate = requested if requested.is_absolute() else root / requested
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TraceDataContractError("missing split manifest: %s" % candidate) from exc
    if not resolved.is_file():
        raise TraceDataContractError("split manifest is not a file: %s" % resolved)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise TraceDataContractError(
            "split manifests must be contained in the dataset directory: %s"
            % resolved
        ) from exc
    return resolved, relative.as_posix()


def _reject_opposite_manifest_role(path: Path, *, role: str) -> None:
    """Reject the common and dangerous train/test argument swap explicitly."""

    stem = path.stem.lower()
    tokens = tuple(token for token in re.split(r"[^a-z0-9]+", stem) if token)
    if role == "canonical_train":
        opposite = "test"
    elif role == "test":
        opposite = "train"
    else:  # pragma: no cover - private programming error
        raise ValueError("unknown manifest role: %s" % role)
    if opposite in tokens or stem == opposite or stem.startswith(opposite + "_"):
        raise TraceDataContractError(
            "%s manifest cannot be used as the %s manifest: %s"
            % (opposite, role, path.name)
        )


@dataclass(frozen=True)
class ManifestAudit:
    """Content audit with a dataset-relative, non-identifying locator."""

    role: str
    locator: str
    count: int
    raw_sha256: str
    normalized_sha256: str
    names: Tuple[str, ...]

    def to_provenance(self) -> Dict[str, object]:
        return {
            "role": self.role,
            "locator": self.locator,
            "count": self.count,
            "raw_sha256": self.raw_sha256,
            "normalized_sha256": self.normalized_sha256,
        }


def _read_manifest(
    root: Path,
    manifest: str | Path,
    *,
    role: str,
) -> Tuple[ManifestAudit, Path]:
    path, locator = _resolve_manifest(root, manifest)
    _reject_opposite_manifest_role(path, role=role)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TraceDataContractError("cannot read split manifest: %s" % path) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TraceDataContractError(
            "split manifest is not strict UTF-8: %s" % path
        ) from exc
    if text.startswith("\ufeff"):
        raise TraceDataContractError("UTF-8 BOM is forbidden in %s" % path)

    logical_lines = text.splitlines()
    if not logical_lines:
        raise TraceDataContractError("empty split manifest: %s" % path)
    for name in logical_lines:
        _safe_sample_name(name, source=locator)
    if len(logical_lines) != len(set(logical_lines)):
        raise TraceDataContractError(
            "duplicate sample names in split manifest: %s" % path
        )
    names = tuple(logical_lines)
    return (
        ManifestAudit(
            role=role,
            locator=locator,
            count=len(names),
            raw_sha256=_sha256_bytes(raw),
            normalized_sha256=_sha256_bytes(_normalised_name_payload(names)),
            names=names,
        ),
        path,
    )


@dataclass(frozen=True)
class SplitAudit:
    split: str
    count: int
    names_sha256: str

    @classmethod
    def from_names(cls, split: str, names: Sequence[str]) -> "SplitAudit":
        validated = _validate_names(names, split=split)
        return cls(
            split=split,
            count=len(validated),
            names_sha256=_sha256_bytes(_normalised_name_payload(validated)),
        )

    def to_provenance(self) -> Dict[str, object]:
        return {
            "count": self.count,
            "names_sha256": self.names_sha256,
        }


def _coerce_split_names(value: object, *, split: str) -> Tuple[str, ...]:
    if hasattr(value, "names"):
        value = getattr(value, "names")
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TraceDataContractError("%s split must provide sample names" % split)
    return _validate_names(value, split=split)


def audit_disjoint_splits(
    train: object,
    dev: object,
    test: object,
) -> None:
    """Fail closed unless train, dev, and test sample names are pairwise disjoint.

    Each argument may be a name iterable or an object exposing ``.names``.
    """

    split_names = {
        "train": _coerce_split_names(train, split="train"),
        "dev": _coerce_split_names(dev, split="dev"),
        "test": _coerce_split_names(test, split="test"),
    }
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    failures = []
    for left, right in pairs:
        overlap = sorted(set(split_names[left]).intersection(split_names[right]))
        if overlap:
            failures.append(
                "%s/%s overlap=%d examples=%s"
                % (left, right, len(overlap), overlap[:5])
            )
    if failures:
        raise TraceDataContractError(
            "train/dev/test split leakage: %s" % "; ".join(failures)
        )


def _deterministic_train_dev_split(
    names: Sequence[str],
    *,
    seed: int,
    dev_fraction: float,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    source = _validate_names(names, split="canonical_train")
    if len(source) < 2:
        raise TraceDataContractError(
            "canonical train manifest needs at least two samples"
        )
    if isinstance(dev_fraction, bool):
        raise TraceDataContractError("dev_fraction must be strictly between 0 and 1")
    try:
        fraction = float(dev_fraction)
    except (TypeError, ValueError) as exc:
        raise TraceDataContractError(
            "dev_fraction must be strictly between 0 and 1"
        ) from exc
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise TraceDataContractError("dev_fraction must be strictly between 0 and 1")

    def rank_key(name: str) -> Tuple[bytes, str]:
        payload = ("%d\0%s" % (seed, name)).encode("utf-8")
        return hashlib.sha256(payload).digest(), name

    ranked = sorted(source, key=rank_key)
    # Half-up rounding is explicit and stable (unlike Python's bankers round).
    dev_count = int(math.floor(len(source) * fraction + 0.5))
    dev_count = max(1, min(len(source) - 1, dev_count))
    dev_set = set(ranked[:dev_count])
    # Preserve canonical manifest order.  Only membership is hash-ranked.
    train_names = tuple(name for name in source if name not in dev_set)
    dev_names = tuple(name for name in source if name in dev_set)
    if set(train_names).union(dev_names) != set(source):  # pragma: no cover
        raise TraceDataContractError("internal train/dev partition failure")
    return train_names, dev_names


@dataclass(frozen=True)
class AssetAudit:
    split: str
    count: int
    names_sha256: str
    images_sha256: str
    masks_sha256: str
    paired_sha256: str

    def to_provenance(self) -> Dict[str, object]:
        return {
            "included": True,
            "count": self.count,
            "audited_count": self.count,
            "names_sha256": self.names_sha256,
            "images_sha256": self.images_sha256,
            "masks_sha256": self.masks_sha256,
            "paired_sha256": self.paired_sha256,
        }


def _audit_assets(root: Path, names: Sequence[str], *, split: str) -> AssetAudit:
    validated = _validate_names(names, split=split)
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.is_dir():
        raise TraceDataContractError("missing images directory: %s" % images_dir)
    if not masks_dir.is_dir():
        raise TraceDataContractError("missing masks directory: %s" % masks_dir)

    image_digest = hashlib.sha256()
    mask_digest = hashlib.sha256()
    paired_digest = hashlib.sha256()
    _update_framed(image_digest, b"trace.images.v1")
    _update_framed(mask_digest, b"trace.masks.v1")
    _update_framed(paired_digest, b"trace.paired_assets.v1")
    for name in validated:
        image_path = images_dir / (name + ".png")
        mask_path = masks_dir / (name + ".png")
        if not image_path.is_file():
            raise TraceDataContractError("missing image asset: %s" % image_path)
        if not mask_path.is_file():
            raise TraceDataContractError("missing mask asset: %s" % mask_path)
        image_hash = bytes.fromhex(_sha256_file(image_path))
        mask_hash = bytes.fromhex(_sha256_file(mask_path))
        encoded_name = name.encode("utf-8")

        _update_framed(image_digest, encoded_name)
        _update_framed(image_digest, image_hash)
        _update_framed(mask_digest, encoded_name)
        _update_framed(mask_digest, mask_hash)
        _update_framed(paired_digest, encoded_name)
        _update_framed(paired_digest, image_hash)
        _update_framed(paired_digest, mask_hash)

    return AssetAudit(
        split=split,
        count=len(validated),
        names_sha256=_sha256_bytes(_normalised_name_payload(validated)),
        images_sha256=image_digest.hexdigest(),
        masks_sha256=mask_digest.hexdigest(),
        paired_sha256=paired_digest.hexdigest(),
    )


def _resolve_image_size(
    *,
    height: Optional[int],
    width: Optional[int],
    image_size: Optional[Sequence[int]],
) -> Tuple[int, int]:
    if image_size is not None:
        if isinstance(image_size, (str, bytes)) or len(image_size) != 2:
            raise TraceDataContractError("image_size must be (height, width)")
        image_height = _require_integer(image_size[0], field="image_size[0]", minimum=1)
        image_width = _require_integer(image_size[1], field="image_size[1]", minimum=1)
        if height is not None and int(height) != image_height:
            raise TraceDataContractError("height conflicts with image_size")
        if width is not None and int(width) != image_width:
            raise TraceDataContractError("width conflicts with image_size")
        return image_height, image_width
    if height is None or width is None:
        raise TraceDataContractError(
            "fixed resize requires height and width (or image_size=(H, W))"
        )
    return (
        _require_integer(height, field="height", minimum=1),
        _require_integer(width, field="width", minimum=1),
    )


def _flip_decision(seed: int, epoch: int, name: str) -> bool:
    payload = (
        "trace.horizontal_flip.v1\0%d\0%d\0%s" % (seed, epoch, name)
    ).encode("utf-8")
    return bool(hashlib.sha256(payload).digest()[0] & 1)


class TracePairedDataset(Dataset):
    """One audited split with stateless, method-independent preprocessing."""

    def __init__(
        self,
        dataset_dir: str | Path,
        names: Sequence[str],
        *,
        split: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image_size: Optional[Sequence[int]] = None,
        seed: int,
        horizontal_flip: bool = False,
        asset_audit: Optional[AssetAudit] = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "dev", "test"):
            raise TraceDataContractError("split must be train, dev, or test")
        self._root = _resolve_dataset_root(dataset_dir)
        self.names = _validate_names(names, split=split)
        self.split = split
        self.height, self.width = _resolve_image_size(
            height=height,
            width=width,
            image_size=image_size,
        )
        self.seed = _require_integer(seed, field="seed", minimum=0)
        if not isinstance(horizontal_flip, bool):
            raise TraceDataContractError("horizontal_flip must be boolean")
        self.horizontal_flip = horizontal_flip
        # A shared-memory scalar keeps set_epoch effective for persistent
        # DataLoader workers as well as fork/spawn workers and resumed runs.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.split_audit = SplitAudit.from_names(split, self.names)
        expected_audit = asset_audit or _audit_assets(
            self._root, self.names, split=split
        )
        if (
            expected_audit.split != split
            or expected_audit.count != len(self.names)
            or expected_audit.names_sha256 != self.split_audit.names_sha256
        ):
            raise TraceDataContractError("asset audit does not match dataset split")
        self.asset_audit = expected_audit
        self._mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    @property
    def split_sha256(self) -> str:
        return self.split_audit.names_sha256

    @property
    def asset_content_sha256(self) -> str:
        return self.asset_audit.paired_sha256

    def set_epoch(self, epoch: int) -> None:
        value = _require_integer(epoch, field="epoch", minimum=0)
        if value > torch.iinfo(torch.int64).max:
            raise TraceDataContractError("epoch exceeds int64 range")
        self._epoch.fill_(value)

    def should_flip(self, name: str) -> bool:
        """Return the exact augmentation decision for the current epoch."""

        _safe_sample_name(name, source="augmentation request")
        if self.split != "train" or not self.horizontal_flip:
            return False
        return _flip_decision(self.seed, self.epoch, name)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        name = self.names[index]
        image_path = self._root / "images" / (name + ".png")
        mask_path = self._root / "masks" / (name + ".png")
        try:
            with Image.open(image_path) as source_image:
                source_image.load()
                image_size = source_image.size
                image = source_image.convert("RGB")
            with Image.open(mask_path) as source_mask:
                source_mask.load()
                mask_size = source_mask.size
                mask = source_mask.copy()
        except (FileNotFoundError, OSError) as exc:
            raise TraceDataContractError(
                "cannot decode image/mask pair for sample %r" % name
            ) from exc
        if image_size != mask_size:
            raise TraceDataContractError(
                "image/mask size mismatch for %r: image=%s mask=%s"
                % (name, image_size, mask_size)
            )

        output_size = (self.width, self.height)
        image = image.resize(output_size, resample=_BILINEAR)
        mask = mask.resize(output_size, resample=_NEAREST)
        if self.should_flip(name):
            image = image.transpose(_FLIP_LEFT_RIGHT)
            mask = mask.transpose(_FLIP_LEFT_RIGHT)

        image_array = np.array(image, dtype=np.float32, copy=True) / 255.0
        if image_array.shape != (self.height, self.width, 3):
            raise TraceDataContractError(
                "unexpected RGB tensor shape for %r: %s"
                % (name, image_array.shape)
            )
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - self._mean) / self._std

        mask_array = np.array(mask, copy=True)
        if mask_array.ndim == 2:
            binary_mask = mask_array != 0
        elif mask_array.ndim == 3:
            binary_mask = np.any(mask_array != 0, axis=2)
        else:
            raise TraceDataContractError(
                "unexpected mask shape for %r: %s" % (name, mask_array.shape)
            )
        mask_tensor = torch.from_numpy(binary_mask.astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor, name


@dataclass
class TraceDataBundle:
    """Train/dev datasets plus an optional, explicitly unlocked test set."""

    train: TracePairedDataset
    dev: TracePairedDataset
    test: Optional[TracePairedDataset]
    test_names: Tuple[str, ...]
    canonical_train_manifest: ManifestAudit
    test_manifest: ManifestAudit
    seed: int
    dev_fraction: float
    height: int
    width: int
    train_horizontal_flip: bool

    def set_epoch(self, epoch: int) -> None:
        """Set the train augmentation epoch; dev/test remain unaugmented."""

        self.train.set_epoch(epoch)

    def _combined_split_hash(self) -> str:
        digest = hashlib.sha256()
        _update_framed(digest, b"trace.split_assignment.v1")
        for split, names in (
            ("train", self.train.names),
            ("dev", self.dev.names),
            ("test", self.test_names),
        ):
            _update_framed(digest, split.encode("ascii"))
            _update_framed(digest, _normalised_name_payload(names))
        return digest.hexdigest()

    def _combined_asset_hash(self) -> str:
        digest = hashlib.sha256()
        _update_framed(digest, b"trace.included_assets.v1")
        datasets = (self.train, self.dev) if self.test is None else (
            self.train,
            self.dev,
            self.test,
        )
        for dataset in datasets:
            _update_framed(digest, dataset.split.encode("ascii"))
            _update_framed(
                digest, bytes.fromhex(dataset.asset_audit.paired_sha256)
            )
        return digest.hexdigest()

    def provenance(self) -> Dict[str, object]:
        """Return JSON-ready provenance without absolute filesystem paths."""

        test_asset: Dict[str, object]
        if self.test is None:
            locked_test_split = SplitAudit.from_names("test", self.test_names)
            test_asset = {
                "included": False,
                "count": len(self.test_names),
                "audited_count": 0,
                "names_sha256": locked_test_split.names_sha256,
                "images_sha256": None,
                "masks_sha256": None,
                "paired_sha256": None,
            }
        else:
            test_asset = self.test.asset_audit.to_provenance()

        splits = {
            "train": self.train.split_audit.to_provenance(),
            "dev": self.dev.split_audit.to_provenance(),
            "test": SplitAudit.from_names("test", self.test_names).to_provenance(),
        }
        assets = {
            "train": self.train.asset_audit.to_provenance(),
            "dev": self.dev.asset_audit.to_provenance(),
            "test": test_asset,
        }
        return {
            "schema_version": TRACE_DATA_SCHEMA_VERSION,
            "dataset": self.train._root.name,
            "canonical_train_manifest": self.canonical_train_manifest.to_provenance(),
            "test_manifest": self.test_manifest.to_provenance(),
            "split_algorithm": SPLIT_ALGORITHM,
            "split_seed": self.seed,
            "dev_fraction": self.dev_fraction,
            "splits": splits,
            "split_assignment_sha256": self._combined_split_hash(),
            "asset_content_sha256": {
                "train": self.train.asset_audit.paired_sha256,
                "dev": self.dev.asset_audit.paired_sha256,
                "test": None if self.test is None else self.test.asset_audit.paired_sha256,
                "all_included": self._combined_asset_hash(),
            },
            "assets": assets,
            "test_assets_included": self.test is not None,
            "resize": {
                "height": self.height,
                "width": self.width,
                "image_interpolation": "bilinear",
                "mask_interpolation": "nearest",
            },
            "normalization": {"mean": list(IMAGE_MEAN), "std": list(IMAGE_STD)},
            "train_augmentation": (
                {
                    "policy": "deterministic_horizontal_flip_only",
                    "algorithm": FLIP_ALGORITHM,
                    "key_fields": ["seed", "epoch", "sample_name"],
                    "geometry_contract": (
                        "T0-A must separately authenticate original and flipped masks"
                    ),
                }
                if self.train_horizontal_flip
                else {
                    "policy": "none",
                    "reason": "preserve the exact train-only T0-A geometry contract",
                }
            ),
            "dev_test_augmentation": "none",
            "drop_last_policy": "trainer_owned",
        }


def build_trace_data(
    dataset_dir: str | Path,
    *,
    train_manifest: Optional[str | Path] = None,
    test_manifest: Optional[str | Path] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    image_size: Optional[Sequence[int]] = None,
    seed: int,
    dev_fraction: float = 0.2,
    include_test: bool = False,
    train_horizontal_flip: bool = False,
) -> TraceDataBundle:
    """Build the shared TRACE/control data contract.

    Relative manifest paths are resolved within ``dataset_dir``.  When a
    manifest is omitted, the exact canonical
    ``img_idx/{train,test}_<dataset>.txt`` path is used.  This function never
    constructs a DataLoader and therefore deliberately owns no ``drop_last``
    decision.
    """

    if not isinstance(include_test, bool):
        raise TraceDataContractError("include_test must be boolean")
    if not isinstance(train_horizontal_flip, bool):
        raise TraceDataContractError("train_horizontal_flip must be boolean")
    root = _resolve_dataset_root(dataset_dir)
    dataset_name = root.name
    train_argument = train_manifest or (
        Path("img_idx") / ("train_%s.txt" % dataset_name)
    )
    test_argument = test_manifest or (
        Path("img_idx") / ("test_%s.txt" % dataset_name)
    )
    canonical_train, train_path = _read_manifest(
        root, train_argument, role="canonical_train"
    )
    held_out_test, test_path = _read_manifest(root, test_argument, role="test")
    if train_path == test_path:
        raise TraceDataContractError(
            "test manifest cannot be used as the canonical train manifest"
        )

    seed_value = _require_integer(seed, field="seed", minimum=0)
    resolved_height, resolved_width = _resolve_image_size(
        height=height,
        width=width,
        image_size=image_size,
    )
    train_names, dev_names = _deterministic_train_dev_split(
        canonical_train.names,
        seed=seed_value,
        dev_fraction=dev_fraction,
    )
    audit_disjoint_splits(train_names, dev_names, held_out_test.names)
    if set(train_names).union(dev_names) != set(canonical_train.names):
        raise TraceDataContractError(
            "train/dev do not exactly partition the canonical train manifest"
        )

    train_assets = _audit_assets(root, train_names, split="train")
    dev_assets = _audit_assets(root, dev_names, split="dev")
    train_dataset = TracePairedDataset(
        root,
        train_names,
        split="train",
        height=resolved_height,
        width=resolved_width,
        seed=seed_value,
        horizontal_flip=train_horizontal_flip,
        asset_audit=train_assets,
    )
    dev_dataset = TracePairedDataset(
        root,
        dev_names,
        split="dev",
        height=resolved_height,
        width=resolved_width,
        seed=seed_value,
        horizontal_flip=False,
        asset_audit=dev_assets,
    )
    test_dataset: Optional[TracePairedDataset] = None
    if include_test:
        test_assets = _audit_assets(root, held_out_test.names, split="test")
        test_dataset = TracePairedDataset(
            root,
            held_out_test.names,
            split="test",
            height=resolved_height,
            width=resolved_width,
            seed=seed_value,
            horizontal_flip=False,
            asset_audit=test_assets,
        )

    return TraceDataBundle(
        train=train_dataset,
        dev=dev_dataset,
        test=test_dataset,
        test_names=held_out_test.names,
        canonical_train_manifest=canonical_train,
        test_manifest=held_out_test,
        seed=seed_value,
        dev_fraction=float(dev_fraction),
        height=resolved_height,
        width=resolved_width,
        train_horizontal_flip=train_horizontal_flip,
    )


__all__ = [
    "AssetAudit",
    "FLIP_ALGORITHM",
    "IMAGE_MEAN",
    "IMAGE_STD",
    "ManifestAudit",
    "SPLIT_ALGORITHM",
    "SplitAudit",
    "TRACE_DATA_SCHEMA_VERSION",
    "TraceDataBundle",
    "TraceDataContractError",
    "TracePairedDataset",
    "audit_disjoint_splits",
    "build_trace_data",
]
