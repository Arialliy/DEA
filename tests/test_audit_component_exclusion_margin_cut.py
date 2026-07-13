from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import audit_component_exclusion_margin_cut as audit


def test_component_exclusion_margin_has_correct_unary_sign() -> None:
    # State 110 on a path is the unique MAP.  Removing its foreground component
    # changes the unary term by 4 and removes a unit cut boundary, so m_C = 3.
    edges = ((0, 1), (1, 2))
    result = audit.audit_energy_instance(
        np.asarray((-2.0, -2.0, 2.0)),
        np.asarray((1.0, 1.0)),
        edges,
    )
    assert result["map_tie_count"] == 1
    assert result["checked_components"] == 1


def test_finite_audit_is_a_no_go_record() -> None:
    summary = audit.run_audit()
    assert summary["status"] == "complete_no_go"
    assert summary["counterexamples"] == 0
    assert summary["total_configurations"] == 5696
    assert summary["total_components"] == 2765
    assert summary["decision"] == {
        "component_exclusion_margin_cut_as_main_method": "NO-GO",
        "reason": (
            "the proposed structured mark is exactly a conventional component "
            "aggregate of unary and boundary terms; it is not a second inference "
            "problem or a new component-valued random variable"
        ),
        "gpu_training_authorized": False,
        "retain_only_as_control": True,
    }


def test_main_writes_once_and_refuses_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "cemc"
    assert audit.main(("--output-dir", str(output_dir))) == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == audit.SCHEMA
    assert "NO-GO" in (output_dir / "summary.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit.main(("--output-dir", str(output_dir)))
