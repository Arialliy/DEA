import pytest
import torch

from model.crwd_objective import CRWDError, counterfactual_residue_witness_loss


def _fixture():
    shape = (1, 1, 16, 16)
    student = torch.full(shape, -4.0)
    control = torch.full(shape, -4.0)
    residue = torch.full(shape, -4.0)
    target = torch.zeros(shape)
    instances = torch.zeros(shape, dtype=torch.long)
    target[0, 0, 8, 8] = 1.0
    instances[0, 0, 8, 8] = 1
    student[0, 0, 8, 8] = -1.0
    control[0, 0, 8, 8] = 0.0
    residue[0, 0, 8, 8] = 3.0
    student[0, 0, 2, 2] = 2.0
    control[0, 0, 2, 2] = 2.0
    residue[0, 0, 2, 2] = 1.0
    valid = torch.ones(shape, dtype=torch.bool)
    return (
        student.requires_grad_(),
        control.requires_grad_(),
        residue.requires_grad_(),
        valid,
        target,
        instances,
    )


def _loss(student, control, residue, valid, target, instances):
    return counterfactual_residue_witness_loss(
        student,
        control,
        residue,
        valid,
        valid,
        target,
        instances,
        protect_kernel=3,
    )


def test_qualified_witness_projects_target_and_tail_with_student_only_gradients() -> None:
    student, control, residue, valid, target, instances = _fixture()
    result = _loss(student, control, residue, valid, target, instances)
    assert result["witness_components"] == 1
    assert result["witness_events"] == 4
    assert float(result["loss"].detach()) > 0.0
    result["loss"].backward()
    assert float(student.grad[0, 0, 8, 8]) < 0.0
    assert float(student.grad[0, 0, 2, 2]) > 0.0
    assert control.grad is None
    assert residue.grad is None


@pytest.mark.parametrize("failure", ["equal", "tail_only", "target_down", "not_survived"])
def test_non_causal_or_non_surviving_teacher_pair_receives_no_credit(failure) -> None:
    student, control, residue, valid, target, instances = _fixture()
    with torch.no_grad():
        if failure == "equal":
            residue.copy_(control)
        elif failure == "tail_only":
            residue[0, 0, 8, 8] = 0.0
            residue[0, 0, 2, 2] = 4.0
        elif failure == "target_down":
            control[0, 0, 8, 8] = 2.0
            control[0, 0, 2, 2] = 5.0
            residue[0, 0, 8, 8] = 1.0
            residue[0, 0, 2, 2] = -2.0
        else:
            control[0, 0, 8, 8] = -2.0
            control[0, 0, 2, 2] = 2.0
            residue[0, 0, 8, 8] = 0.0
            residue[0, 0, 2, 2] = 1.0
    result = _loss(student, control, residue, valid, target, instances)
    assert result["witness_components"] == 0
    assert float(result["loss"].detach()) == 0.0
    result["loss"].backward()
    assert bool((student.grad == 0).all())


def test_already_feasible_student_has_exact_zero_loss_and_gradient() -> None:
    student, control, residue, valid, target, instances = _fixture()
    with torch.no_grad():
        student[0, 0, 8, 8] = 3.0
        student[0, 0, 2, 2] = 1.0
    result = _loss(student, control, residue, valid, target, instances)
    assert result["witness_events"] == 4
    assert float(result["loss"].detach()) == 0.0
    result["loss"].backward()
    assert bool((student.grad == 0).all())


def test_empty_and_geometrically_invalid_components_are_safe() -> None:
    student, control, residue, valid, target, instances = _fixture()
    empty = _loss(
        student,
        control,
        residue,
        valid,
        torch.zeros_like(target),
        torch.zeros_like(instances),
    )
    assert empty["eligible_components"] == 0
    assert float(empty["loss"].detach()) == 0.0

    residue_valid = valid.clone()
    residue_valid[0, 0, 8, 8] = False
    skipped = counterfactual_residue_witness_loss(
        student,
        control,
        residue,
        valid,
        residue_valid,
        target,
        instances,
        protect_kernel=3,
    )
    assert skipped["eligible_components"] == 0
    assert skipped["skipped_invalid_components"] == 1
    assert float(skipped["loss"].detach()) == 0.0


def test_component_log_mean_exp_and_reduction_are_area_unbiased() -> None:
    student, control, residue, valid, target, instances = _fixture()
    one = _loss(student, control, residue, valid, target, instances)
    target[0, 0, 11:13, 11:13] = 1.0
    instances[0, 0, 11:13, 11:13] = 2
    with torch.no_grad():
        student[0, 0, 11:13, 11:13] = -1.0
        control[0, 0, 11:13, 11:13] = 0.0
        residue[0, 0, 11:13, 11:13] = 3.0
    two = _loss(student, control, residue, valid, target, instances)
    torch.testing.assert_close(two["loss"], one["loss"], rtol=1e-6, atol=1e-6)


def test_crwd_fails_closed_on_invalid_contracts() -> None:
    student, control, residue, valid, target, instances = _fixture()
    with pytest.raises(CRWDError, match="budgets"):
        counterfactual_residue_witness_loss(
            student, control, residue, valid, valid, target, instances, budgets=(20, 5)
        )
    with pytest.raises(CRWDError, match="positive odd"):
        counterfactual_residue_witness_loss(
            student, control, residue, valid, valid, target, instances, protect_kernel=4
        )
    with pytest.raises(CRWDError, match="foreground differs"):
        counterfactual_residue_witness_loss(
            student,
            control,
            residue,
            valid,
            valid,
            target,
            torch.zeros_like(instances),
        )
