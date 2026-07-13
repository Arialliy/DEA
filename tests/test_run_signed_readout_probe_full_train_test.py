from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from tools import run_signed_readout_probe_full_train_test as probe


def test_protocol_epochs_are_predeclared() -> None:
    assert probe.protocol_epochs("formal", "IRSTD-1K", 17) == 20
    assert probe.protocol_epochs("smoke", "NUAA-SIRST", 20260711) == 10
    with pytest.raises(probe.FullTrainTestProbeError, match="predeclared"):
        probe.protocol_epochs("smoke", "IRSTD-1K", 20260711)


def test_source_inventory_covers_previously_omitted_dependencies() -> None:
    paths = probe._source_paths()
    assert paths["component_operating_point"].name == "component_operating_point.py"
    assert paths["mshnet_checkpoint"].name == "mshnet_checkpoint.py"
    assert paths["baseline_finalizer"].name == "finalize_test_selected_baselines.py"
    assert paths["full_train_test_protocol"].name == "full_train_test_protocol.py"
    assert all(path.is_file() and not path.is_symlink() for path in paths.values())


def test_loader_never_drops_the_last_train_sample() -> None:
    dataset = TensorDataset(torch.arange(5))
    loader = probe._loader(
        dataset,  # type: ignore[arg-type]
        training=True,
        num_workers=0,
        device=torch.device("cpu"),
        seed=9,
    )
    assert loader.drop_last is False
    observed = sum(int(batch[0].shape[0]) for batch in loader)
    assert observed == len(dataset)


def test_bundle_roundtrip_uses_explicit_variant_order(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    variants = {
        name: {"sentinel": index}
        for index, name in enumerate(probe.VARIANT_ORDER)
    }
    summary = {
        "schema": probe.SCHEMA,
        "status": "complete",
        "variant_order": list(probe.VARIANT_ORDER),
        "variants": variants,
    }
    logits = {
        name: (np.zeros((2, 2), dtype=np.float32),)
        for name in probe.VARIANT_ORDER
    }
    probe._write_bundle(
        output,
        summary=summary,
        history=(),
        oracle_rows=(),
        target_rows=(),
        image_rows=(),
        calibration_rows=(),
        logits=logits,
        image_names=("sample",),
        head_payload={"state_dict": {}},
        provenance={"schema": probe.PROVENANCE_SCHEMA},
    )

    loaded = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    # sort_keys=True intentionally changes nested mapping order on disk.  The
    # protocol therefore validates the explicit order field and the exact key
    # set independently, never mapping insertion order.
    assert tuple(loaded["variants"]) != probe.VARIANT_ORDER
    assert loaded["variant_order"] == list(probe.VARIANT_ORDER)
    assert set(loaded["variants"]) == set(probe.VARIANT_ORDER)

    provenance = json.loads(
        (output / "provenance.json").read_text(encoding="utf-8")
    )
    assert set(provenance["artifact_sha256"]) == set(probe.BUNDLE_FILES[:-1])
    assert {path.name for path in output.iterdir()} == set(probe.BUNDLE_FILES)


def test_validate_selected_baseline_wraps_finalizer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "manifest.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        probe.baseline_finalizer,
        "_validate_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            probe.baseline_finalizer.FinalizationError("incomplete grid")
        ),
    )
    with pytest.raises(probe.FullTrainTestProbeError, match="incomplete grid"):
        probe.validate_selected_baseline(batch, "NUAA-SIRST", 20260711)
