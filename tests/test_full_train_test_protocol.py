from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from tools import run_test_selected_baselines as scheduler
from tools import train_test_selected_full_train as training_entry
from utils import full_train_test_protocol as protocol
from utils.data import IRSTD_Dataset


EXPECTED = {
    "IRSTD-1K": {
        "train": (
            800,
            "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744",
            "b698d2d9dbe9e26e1875978d23450e1e6ec45fd71d56d31415007f56c40bba88",
        ),
        "test": (
            201,
            "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579",
            "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579",
        ),
        "assets": 1001,
    },
    "NUAA-SIRST": {
        "train": (
            213,
            "324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4",
            "815dcca749f087f27f5dad4b447015aee70bd7ae7779d6fdd7d6efa6d5c6943f",
        ),
        "test": (
            214,
            "e49023203a323c247306b314f23c8b3b917093a26984067792355adff7a8386e",
            "395eecd6bf0ed2a59f531de9145688597632c68f9d0933359aadcb93ec1a60b5",
        ),
        "assets": 427,
    },
    "NUDT-SIRST": {
        "train": (
            663,
            "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3",
            "dc555df66b62dd1ea98d119ace8fe8ae86de94f3e4833d8d81e90c0e1f287922",
        ),
        "test": (
            664,
            "a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e",
            "cec44220c69d89a5b3fd245b8ee911404e959fef80bd96b32b6b74f28bb32af0",
        ),
        "assets": 1327,
    },
}


def _synthetic_audit() -> protocol.CanonicalDatasetAudit:
    train = protocol.SplitManifestAudit(
        split="train",
        path="/home/ly/DEA/datasets/NUAA-SIRST/img_idx/train_NUAA-SIRST.txt",
        count=2,
        raw_sha256="1" * 64,
        normalized_sha256="2" * 64,
        names=("a", "b"),
    )
    test = protocol.SplitManifestAudit(
        split="test",
        path="/home/ly/DEA/datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST.txt",
        count=1,
        raw_sha256="3" * 64,
        normalized_sha256="4" * 64,
        names=("c",),
    )
    return protocol.CanonicalDatasetAudit(
        dataset_name="NUAA-SIRST",
        dataset_dir="/home/ly/DEA/datasets/NUAA-SIRST",
        train=train,
        test=test,
        image_count=3,
        mask_count=3,
    )


def _fixture_contract(train_raw: bytes, test_raw: bytes) -> dict[str, object]:
    def one(raw: bytes) -> dict[str, object]:
        names = raw.decode("utf-8").splitlines()
        normalized = ("\n".join(names) + "\n").encode("utf-8")
        return {
            "count": len(names),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        }

    return {"train": one(train_raw), "test": one(test_raw)}


def _make_dataset_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_raw: bytes = b"tr0\ntr1\ntr2\n",
    test_raw: bytes = b"te0\nte1\n",
) -> Path:
    root = tmp_path / "datasets"
    dataset = root / "NUDT-SIRST"
    for directory in ("img_idx", "images", "masks"):
        (dataset / directory).mkdir(parents=True, exist_ok=True)
    (dataset / "img_idx" / "train_NUDT-SIRST.txt").write_bytes(train_raw)
    (dataset / "img_idx" / "test_NUDT-SIRST.txt").write_bytes(test_raw)
    names = train_raw.decode("utf-8").splitlines() + test_raw.decode(
        "utf-8"
    ).splitlines()
    for name in names:
        (dataset / "images" / f"{name}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n"
        )
        (dataset / "masks" / f"{name}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n"
        )
    # This deliberately invalid diagnostic manifest proves that the canonical
    # train/test path never discovers or consumes hcval.
    (dataset / "img_idx" / "hcval_NUDT-SIRST.txt").write_text(
        "../../forbidden\n../../forbidden\n", encoding="utf-8"
    )
    monkeypatch.setattr(protocol, "CANONICAL_DATASETS_ROOT", root)
    monkeypatch.setattr(protocol, "CANONICAL_DATASET_NAMES", ("NUDT-SIRST",))
    monkeypatch.setattr(
        protocol,
        "FROZEN_CONTRACT",
        {"NUDT-SIRST": _fixture_contract(train_raw, test_raw)},
    )
    return dataset


def test_real_canonical_contract_has_exact_counts_raw_and_normalized_hashes() -> None:
    assert protocol.CANONICAL_DATASETS_ROOT == Path("/home/ly/DEA/datasets")
    assert set(protocol.CANONICAL_DATASET_NAMES) == set(EXPECTED)

    audits = protocol.audit_all_canonical_datasets()

    assert set(audits) == set(EXPECTED)
    for dataset, expected in EXPECTED.items():
        audit = audits[dataset]
        for role in ("train", "test"):
            split = getattr(audit, role)
            count, raw_hash, normalized_hash = expected[role]
            assert split.split == role
            assert split.count == count
            assert split.raw_sha256 == raw_hash
            assert split.normalized_sha256 == normalized_hash
            assert len(split.names) == count
            assert len(set(split.names)) == count
            assert Path(split.path).name == f"{role}_{dataset}.txt"
        assert set(audit.train.names).isdisjoint(audit.test.names)
        assert audit.image_count == expected["assets"]
        assert audit.mask_count == expected["assets"]


def test_nudt_hcval_is_never_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes
    opened: list[Path] = []

    def guarded_read_bytes(path: Path) -> bytes:
        path = Path(path)
        if "hcval" in path.name.lower():
            raise AssertionError(f"hcval must not be opened: {path}")
        opened.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    audit = protocol.audit_canonical_dataset(
        "/home/ly/DEA/datasets/NUDT-SIRST"
    )

    assert audit.train.count == 663
    assert audit.test.count == 664
    assert {path.name for path in opened} == {
        "train_NUDT-SIRST.txt",
        "test_NUDT-SIRST.txt",
    }


def test_toy_contract_uses_complete_train_and_complete_test_without_hcval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset_fixture(tmp_path, monkeypatch)

    audit = protocol.audit_canonical_dataset(dataset)

    assert audit.train.names == ("tr0", "tr1", "tr2")
    assert audit.test.names == ("te0", "te1")
    assert audit.train.count == 3
    assert audit.test.count == 2
    assert audit.image_count == audit.mask_count == 5
    assert not any("hcval" in name.lower() for name in audit.train.names)
    assert not any("hcval" in name.lower() for name in audit.test.names)


def test_toy_loader_bootstrap_has_full_train_and_val_alias_is_exact_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset_fixture(tmp_path, monkeypatch)
    train_file = dataset / "img_idx" / "train_NUDT-SIRST.txt"
    test_file = dataset / "img_idx" / "test_NUDT-SIRST.txt"
    args = Namespace(
        dataset_dir=str(dataset),
        train_split_file=str(train_file),
        val_split_file=str(test_file),
        test_split_file=str(test_file),
        val_fraction=0.0,
        split_seed=17,
        seed=17,
        crop_size=16,
        base_size=16,
        return_instance_labels=False,
    )

    trainset = IRSTD_Dataset(args, mode="train")
    evaluation_alias = IRSTD_Dataset(args, mode="val")
    testset = IRSTD_Dataset(args, mode="test")

    assert trainset.names == ["tr0", "tr1", "tr2"]
    assert evaluation_alias.names == testset.names == ["te0", "te1"]
    assert evaluation_alias.split_sha256 == testset.split_sha256
    training_entry.FullTrainTestTrainer.assert_disjoint_splits(
        trainset, evaluation_alias, testset
    )


def test_protocol_cli_routes_only_canonical_train_and_test() -> None:
    args, audit = training_entry.parse_protocol_args(
        [
            "--dataset-dir",
            "/home/ly/DEA/datasets/NUAA-SIRST",
            "--epochs",
            "1",
            "--val-fraction",
            "0",
            "--test-interval",
            "10",
        ]
    )

    assert args.train_split_file == audit.train.path
    assert args.test_split_file == audit.test.path
    # This is an internal construction alias only; persisted metadata erases
    # it and labels the evaluation split truthfully as test.
    assert args.val_split_file == audit.test.path
    assert args.val_fraction == 0.0
    assert args.test_interval == 10
    with pytest.raises(protocol.ProtocolContractError, match="forbidden"):
        training_entry.parse_protocol_args(
            [
                "--dataset-dir",
                "/home/ly/DEA/datasets/NUAA-SIRST",
                "--val-split-file",
                "img_idx/test_NUAA-SIRST.txt",
            ]
        )
    with pytest.raises(protocol.ProtocolContractError, match="val-fraction"):
        training_entry.parse_protocol_args(
            [
                "--dataset-dir",
                "/home/ly/DEA/datasets/NUAA-SIRST",
                "--val-fraction",
                "0.2",
            ]
        )
    with pytest.raises(protocol.ProtocolContractError, match="non-canonical"):
        training_entry.parse_protocol_args(
            [
                "--dataset-dir",
                "/home/ly/DEA/datasets/NUAA-SIRST",
                "--model-type",
                "full_dea",
            ]
        )


def test_toy_contract_rejects_duplicate_overlap_wrong_hash_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset_fixture(tmp_path, monkeypatch)
    train_path = dataset / "img_idx" / "train_NUDT-SIRST.txt"
    test_path = dataset / "img_idx" / "test_NUDT-SIRST.txt"

    duplicate_raw = b"tr0\ntr0\ntr2\n"
    train_path.write_bytes(duplicate_raw)
    protocol.FROZEN_CONTRACT["NUDT-SIRST"]["train"] = _fixture_contract(
        duplicate_raw, test_path.read_bytes()
    )["train"]
    with pytest.raises(protocol.ProtocolContractError, match="duplicate"):
        protocol.audit_canonical_dataset(dataset)

    original_train = b"tr0\ntr1\ntr2\n"
    overlap_test = b"te0\ntr1\n"
    train_path.write_bytes(original_train)
    test_path.write_bytes(overlap_test)
    protocol.FROZEN_CONTRACT["NUDT-SIRST"] = _fixture_contract(
        original_train, overlap_test
    )
    with pytest.raises(protocol.ProtocolContractError, match="overlap"):
        protocol.audit_canonical_dataset(dataset)

    original_test = b"te0\nte1\n"
    test_path.write_bytes(original_test)
    protocol.FROZEN_CONTRACT["NUDT-SIRST"] = _fixture_contract(
        original_train, original_test
    )
    train_path.write_bytes(b"tr0\ntr1\nchanged\n")
    with pytest.raises(protocol.ProtocolContractError, match="frozen canonical"):
        protocol.audit_canonical_dataset(dataset)

    train_path.write_bytes(original_train)
    target = tmp_path / "external.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    image = dataset / "images" / "tr0.png"
    image.unlink()
    image.symlink_to(target)
    with pytest.raises(protocol.ProtocolContractError, match="symbolic links"):
        protocol.audit_canonical_dataset(dataset)


def test_wrong_dataset_path_and_unlocked_interval_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(protocol.ProtocolContractError, match="dataset root"):
        protocol.audit_canonical_dataset(tmp_path / "NUAA-SIRST")

    for interval in (0, 1, 5, 11):
        with pytest.raises(protocol.ProtocolContractError, match="interval"):
            protocol.require_locked_test_interval(interval)
    assert protocol.require_locked_test_interval(10) == 10


def test_epoch_schedule_is_completed_epoch_interval_plus_final() -> None:
    selected = [
        epoch
        for epoch in range(23)
        if protocol.should_evaluate_epoch(epoch, total_epochs=23)
    ]
    assert selected == [9, 19, 22]
    assert protocol.evaluation_epochs(0, 23) == (9, 19, 22)
    assert protocol.evaluation_epochs(10, 23) == (19, 22)
    assert protocol.evaluation_epochs(9, 20) == (9, 19)
    assert protocol.evaluation_epochs(20, 20) == ()


def test_protocol_metadata_overwrites_old_holdout_semantics() -> None:
    metadata = protocol.build_protocol_metadata(
        {
            "method": "MSHNet",
            "val_split_file": "old_val.txt",
            "val_split_sha256": "bad",
            "val_fraction": 0.2,
            "train_split_sha256": "old",
            "test_split_sha256": "old",
        },
        _synthetic_audit(),
        test_interval=10,
        resume=True,
    )

    assert metadata["protocol"] == "test_selected_full_train_interval_v1"
    assert metadata["protocol_version"] == metadata["protocol"]
    assert metadata["selection_split"] == "test"
    assert metadata["evaluation_split"] == "test"
    assert metadata["evaluation_alias"] == (
        "val_loader_is_complete_canonical_test"
    )
    assert metadata["no_internal_holdout"] is True
    assert metadata["val_split_file"] == ""
    assert metadata["val_split_sha256"] == ""
    assert metadata["val_split_count"] == 0
    assert metadata["val_fraction"] == 0.0
    assert metadata["test_interval"] == 10
    assert metadata["evaluation_epoch_rule"] == protocol.EVALUATION_EPOCH_RULE
    assert metadata["train_split_sha256"] == "2" * 64
    assert metadata["test_split_sha256"] == "4" * 64
    assert metadata["resume"] is True


def test_legacy_val_alias_must_equal_complete_test_exactly() -> None:
    train = SimpleNamespace(names=["tr0", "tr1"], split_sha256="train-hash")
    test = SimpleNamespace(names=["te0", "te1"], split_sha256="test-hash")
    alias = SimpleNamespace(names=["te0", "te1"], split_sha256="test-hash")

    training_entry.FullTrainTestTrainer.assert_disjoint_splits(
        train, alias, test
    )

    wrong_order = SimpleNamespace(
        names=["te1", "te0"], split_sha256="test-hash"
    )
    with pytest.raises(protocol.ProtocolContractError, match="exact order"):
        training_entry.FullTrainTestTrainer.assert_disjoint_splits(
            train, wrong_order, test
        )
    wrong_hash = SimpleNamespace(
        names=["te0", "te1"], split_sha256="wrong"
    )
    with pytest.raises(protocol.ProtocolContractError, match="hashes differ"):
        training_entry.FullTrainTestTrainer.assert_disjoint_splits(
            train, wrong_hash, test
        )
    leaking = SimpleNamespace(
        names=["tr0", "te1"], split_sha256="leaking"
    )
    with pytest.raises(protocol.ProtocolContractError, match="leakage"):
        training_entry.FullTrainTestTrainer.assert_disjoint_splits(
            train, leaking, leaking
        )


def test_full_train_loader_keeps_last_partial_batch() -> None:
    trainer = training_entry.FullTrainTestTrainer.__new__(
        training_entry.FullTrainTestTrainer
    )
    trainer.args = Namespace(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        seed=1701,
    )
    trainer.train_dataset = TensorDataset(torch.arange(5))

    trainer._rebuild_full_train_loader()

    assert trainer.train_loader.drop_last is False
    assert len(trainer.train_loader) == 3
    assert sorted(
        int(value)
        for batch in trainer.train_loader
        for value in batch[0].tolist()
    ) == [0, 1, 2, 3, 4]


class _FakeTrainer:
    def __init__(self, *, start_epoch: int, epochs: int):
        self.start_epoch = start_epoch
        self.args = Namespace(epochs=epochs)
        self.prior_evaluation_epochs = tuple(
            epoch
            for epoch in range(start_epoch)
            if protocol.should_evaluate_epoch(epoch, epochs)
        )
        self.events: list[tuple[object, ...]] = []

    def train(self, epoch: int) -> None:
        self.events.append(("train", epoch))

    def test(self, epoch: int) -> None:
        self.events.append(("test", epoch))

    def assert_saved_checkpoint_contracts(self) -> None:
        self.events.append(("checkpoint_audit",))

    def _write_protocol_summary(
        self,
        *,
        status: str,
        current_executed,
        cumulative_executed,
    ) -> None:
        self.events.append(
            (
                "summary",
                status,
                tuple(current_executed),
                tuple(cumulative_executed),
            )
        )


def test_training_loop_orders_train_then_periodic_test_plus_final() -> None:
    trainer = _FakeTrainer(start_epoch=0, epochs=23)

    executed = training_entry.run_training_protocol(trainer)

    assert executed == (9, 19, 22)
    assert [event[1] for event in trainer.events if event[0] == "train"] == list(
        range(23)
    )
    assert [event[1] for event in trainer.events if event[0] == "test"] == [
        9,
        19,
        22,
    ]
    for epoch in executed:
        train_index = trainer.events.index(("train", epoch))
        test_index = trainer.events.index(("test", epoch))
        assert train_index < test_index
        assert trainer.events[test_index + 1] == ("checkpoint_audit",)
    assert trainer.events[-1] == (
        "summary",
        "complete",
        (9, 19, 22),
        (9, 19, 22),
    )


@pytest.mark.parametrize(
    ("start_epoch", "expected_tests"),
    [
        (10, (19, 22)),
        (20, (22,)),
        (23, ()),
    ],
)
def test_training_loop_resume_boundary_does_not_repeat_prior_test(
    start_epoch: int,
    expected_tests: tuple[int, ...],
) -> None:
    trainer = _FakeTrainer(start_epoch=start_epoch, epochs=23)

    executed = training_entry.run_training_protocol(trainer)

    assert executed == expected_tests
    assert [event[1] for event in trainer.events if event[0] == "train"] == list(
        range(start_epoch, 23)
    )
    assert [event[1] for event in trainer.events if event[0] == "test"] == list(
        expected_tests
    )


def test_persisted_run_contract_has_train_test_only_and_exact_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _synthetic_audit()
    args = Namespace(
        val_split_file="internal_bootstrap_should_not_persist.txt",
        val_split_sha256="bad",
        val_fraction=0.2,
        test_interval=10,
    )
    trainer = training_entry.FullTrainTestTrainer.__new__(
        training_entry.FullTrainTestTrainer
    )
    trainer.mode = "train"
    trainer.save_folder = str(tmp_path)
    trainer.args = args
    trainer.protocol_audit = audit
    monkeypatch.setattr(
        training_entry.dea_main,
        "get_method_metadata",
        lambda _args: protocol.build_protocol_metadata({}, audit, 10, False),
    )

    trainer.persist_split_manifests()

    assert (tmp_path / "split_train.txt").read_text() == "a\nb\n"
    assert (tmp_path / "split_test.txt").read_text() == "c\n"
    assert not (tmp_path / "split_val.txt").exists()
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["args"]["val_split_file"] == ""
    assert config["args"]["val_split_sha256"] == ""
    assert config["args"]["val_fraction"] == 0.0
    metadata = config["method_meta"]
    assert metadata["protocol"] == "test_selected_full_train_interval_v1"
    assert metadata["selection_split"] == "test"
    assert metadata["no_internal_holdout"] is True
    assert metadata["val_split_file"] == ""
    assert metadata["val_split_sha256"] == ""
    assert metadata["val_split_count"] == 0
    assert metadata["test_interval"] == 10
    assert metadata["train_loader_drop_last"] is False
    assert metadata["evaluation_epoch_rule"] == protocol.EVALUATION_EPOCH_RULE


def test_checkpoint_protocol_audit_rejects_any_validation_semantics() -> None:
    audit = _synthetic_audit()
    metadata = protocol.build_protocol_metadata({}, audit, 10, False)
    assert training_entry._checkpoint_protocol_mismatches(metadata, audit) == []

    tampered = dict(metadata)
    tampered["val_split_file"] = "dev.txt"
    tampered["no_internal_holdout"] = False
    mismatches = training_entry._checkpoint_protocol_mismatches(tampered, audit)
    assert any(value.startswith("val_split_file ") for value in mismatches)
    assert any(value.startswith("no_internal_holdout ") for value in mismatches)


def test_complete_summary_records_selected_checkpoint_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _synthetic_audit()
    metadata = protocol.build_protocol_metadata({}, audit, 10, False)
    torch.save(
        {
            "epoch": 19,
            "iou": 0.71,
            "pd": 0.95,
            "fa": 12.0,
            "method_meta": metadata,
        },
        tmp_path / "checkpoint_best_iou.pkl",
    )
    trainer = training_entry.FullTrainTestTrainer.__new__(
        training_entry.FullTrainTestTrainer
    )
    trainer.save_folder = str(tmp_path)
    trainer.protocol_audit = audit
    trainer.test_interval = 10
    trainer.start_epoch = 0
    trainer.resumed_process = False
    trainer.args = Namespace(epochs=23)
    monkeypatch.setattr(
        training_entry.dea_main,
        "get_method_metadata",
        lambda _args: metadata,
    )

    summary = trainer._summary_payload(
        status="complete",
        current_executed=(9, 19, 22),
        cumulative_executed=(9, 19, 22),
    )

    best_iou = summary["checkpoint_selection"]["best_iou"]
    assert best_iou["status"] == "found"
    assert best_iou["file"] == "checkpoint_best_iou.pkl"
    assert best_iou["epoch_zero_based"] == 19
    assert best_iou["iou"] == 0.71
    assert best_iou["pd"] == 0.95
    assert best_iou["fa"] == 12.0
    assert len(best_iou["sha256"]) == 64
    assert summary["checkpoint_selection"]["constrained_min_fa"] == {
        "status": "not_found",
        "file": None,
        "reason": "no_eligible_epoch",
    }


def test_training_summary_is_accepted_by_scheduler_resume_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _synthetic_audit()
    job_id = "mshnet__nuaa-sirst__seed_20260711"
    base = {
        "method": "MSHNet",
        "model_type": "mshnet",
        "dataset_dir": audit.dataset_dir,
        "seed": 20260711,
        "deterministic": True,
        "run_label": job_id,
    }
    metadata = protocol.build_protocol_metadata(base, audit, 10, False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(
        {
            "epoch": 19,
            "iou": 0.71,
            "pd": 0.95,
            "fa": 12.0,
            "method_meta": metadata,
        },
        run_dir / "checkpoint_best_iou.pkl",
    )
    trainer = training_entry.FullTrainTestTrainer.__new__(
        training_entry.FullTrainTestTrainer
    )
    trainer.save_folder = str(run_dir)
    trainer.protocol_audit = audit
    trainer.test_interval = 10
    trainer.start_epoch = 0
    trainer.resumed_process = False
    trainer.args = Namespace(epochs=23)
    monkeypatch.setattr(
        training_entry.dea_main,
        "get_method_metadata",
        lambda _args: metadata,
    )
    summary = trainer._summary_payload(
        status="complete",
        current_executed=(9, 19, 22),
        cumulative_executed=(9, 19, 22),
    )
    (run_dir / "protocol_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps(
            {
                "returncode": 0,
                "protocol": protocol.PROTOCOL_VERSION,
                "job_id": job_id,
                "dataset": "NUAA-SIRST",
                "seed": 20260711,
                "run_dir": str(run_dir),
                "total_epochs": 23,
                "test_interval": 10,
                "test_evaluation_epochs": [10, 20, 23],
            }
        ),
        encoding="utf-8",
    )
    job = {
        "job_id": job_id,
        "dataset": "NUAA-SIRST",
        "seed": 20260711,
        "dataset_dir": audit.dataset_dir,
        "train_split_arg": "img_idx/train_NUAA-SIRST.txt",
        "test_split_arg": "img_idx/test_NUAA-SIRST.txt",
        "train_split_sha256": audit.train.normalized_sha256,
        "test_split_sha256": audit.test.normalized_sha256,
        "train_split_raw_sha256": audit.train.raw_sha256,
        "test_split_raw_sha256": audit.test.raw_sha256,
        "deterministic": True,
        "run_dir": str(run_dir),
        "result_file": str(result_file),
        "total_epochs": 23,
        "test_interval": 10,
        "test_evaluation_epochs": [10, 20, 23],
    }

    assert scheduler._is_completed_result(job) is True
