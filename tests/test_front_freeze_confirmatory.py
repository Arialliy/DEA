from __future__ import annotations

import pytest

from tools.audit_front_freeze_confirmatory import (
    CONTROL_OUTCOME,
    FORMAL_OUTCOME,
    FrontFreezeAuditError,
    build_transition_ledger,
    select_success_controls,
)


def _target(name: str, area: int, border: float, *, outcome: str):
    return {
        "dataset": "IRSTD-1K",
        "seed": 20260711,
        "outcome": outcome,
        "stable_target_id": name,
        "image_name": name,
        "target_area": area,
        "border_distance": border,
    }


def test_success_controls_are_same_job_unique_and_keep_pair_ledger() -> None:
    formal = [
        _target("f-small", 1, 20.0, outcome=FORMAL_OUTCOME),
        _target("f-large", 20, 80.0, outcome=FORMAL_OUTCOME),
    ]
    candidates = [
        _target("c-small", 1, 21.0, outcome=CONTROL_OUTCOME),
        _target("c-large", 21, 82.0, outcome=CONTROL_OUTCOME),
        _target("c-spare", 5, 100.0, outcome=CONTROL_OUTCOME),
    ]

    selected = select_success_controls(formal, candidates)

    assert {row["stable_target_id"] for row in selected} == {
        "c-small",
        "c-large",
    }
    assert {row["paired_formal_target_id"] for row in selected} == {
        "f-small",
        "f-large",
    }
    assert len({row["stable_target_id"] for row in selected}) == len(selected)


def test_success_control_matching_fails_closed_across_jobs() -> None:
    formal = [_target("f", 1, 20.0, outcome=FORMAL_OUTCOME)]
    candidate = _target("c", 1, 20.0, outcome=CONTROL_OUTCOME)
    candidate["seed"] = 20260712

    with pytest.raises(FrontFreezeAuditError, match="crossed"):
        select_success_controls(formal, [candidate])


def test_transition_ledger_reports_drop_recovery_and_undefined() -> None:
    from tools.audit_front_freeze_confirmatory import MAIN_PATH

    stages = {
        stage: {"available": True, "state": "distinct"} for stage in MAIN_PATH
    }
    stages["p0"] = {"available": True, "state": "background_like"}
    stages["e1"] = {"available": True, "state": "distinct"}
    stages["mask0"] = {"available": True, "state": "background_like"}
    stages["z"] = {"available": False, "state": "undefined"}

    ledger = build_transition_ledger(stages)

    assert ledger["e0_to_p0"] == "distinct_drop"
    assert ledger["p0_to_e1"] == "distinct_recovery"
    assert ledger["d0_to_mask0"] == "distinct_drop"
    assert ledger["d0_to_z"] == "undefined_endpoint"
