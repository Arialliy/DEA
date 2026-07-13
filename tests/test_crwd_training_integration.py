from argparse import Namespace
import copy

import torch
import torch.nn as nn

from main import Trainer
from model.MSHNet import MSHNet


class TinyMSHNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.readout = nn.Conv2d(4, 1, kernel_size=1)
        self.forward_training_flags = []

    def forward(self, value, warm_flag):
        self.forward_training_flags.append(bool(self.training))
        logits = self.readout(torch.relu(self.bn(self.conv(value))))
        return [logits], logits


def _args(lambda_value: float) -> Namespace:
    return Namespace(
        model_type="mshnet",
        mshnet_objective="sls",
        mshnet_side_supervision="canonical",
        mshnet_train_graph="canonical_warm",
        location_loss="legacy",
        side_location_loss="same",
        lambda_location=1.0,
        warm_epoch=5,
        dea_lambda_single=0.0,
        dea_lambda_dec=0.0,
        dea_lambda_empty=0.0,
        crwd_lambda=lambda_value,
        crwd_ramp_epochs=1,
        crwd_protect_kernel=3,
        crwd_target_temperature=0.25,
        crwd_tail_temperature=0.25,
        crwd_delta_target=0.05,
        crwd_delta_margin=0.05,
        crwd_tail_tolerance=0.05,
        crwd_margin_floor=0.0,
        crwd_max_margin_credit=1.0,
        crwd_confidence_width=0.25,
        crwd_huber_delta=0.25,
        crwd_log_interval=0,
    )


def _trainer(model: nn.Module, lambda_value: float) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.args = _args(lambda_value)
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.warm_epoch = 5
    trainer.train_loader = [None, None, None]
    return trainer


def test_teacher_forwards_are_eval_no_grad_and_restore_all_training_flags() -> None:
    torch.manual_seed(41)
    model = TinyMSHNet().train()
    trainer = _trainer(model, 0.2)
    data = torch.randn(2, 3, 32, 32)
    labels = torch.zeros(2, 1, 32, 32)
    instances = torch.zeros(2, 1, 32, 32, dtype=torch.long)
    labels[:, :, 15, 15] = 1.0
    instances[:, :, 15, 15] = 1

    _, canonical = model(data, True)
    bn_after_student = (
        model.bn.running_mean.clone(),
        model.bn.running_var.clone(),
        model.bn.num_batches_tracked.clone(),
    )
    result = trainer.compute_crwd_training_objective(
        data,
        canonical,
        labels,
        instances,
        epoch=6,
        iteration=0,
    )

    assert model.forward_training_flags == [True, False, False]
    assert model.training and model.bn.training
    for actual, expected in zip(
        (model.bn.running_mean, model.bn.running_var, model.bn.num_batches_tracked),
        bn_after_student,
    ):
        assert torch.equal(actual, expected)
    assert result["direction"] == (0, 1)
    assert result["control_offset"] == (0, 16)
    assert result["residue_offset"] == (0, 17)
    result["loss"].backward()
    assert all(parameter.grad is None for parameter in model.parameters()) is False


def test_direction_schedule_is_deterministic_and_covers_three_directions() -> None:
    trainer = _trainer(TinyMSHNet(), 0.2)
    observed = [trainer.crwd_phase_pair(6, index)[0] for index in range(3)]
    assert observed == [(0, 1), (1, 0), (1, 1)]


def test_lambda_zero_training_step_is_bitwise_baseline_and_never_calls_teacher() -> None:
    torch.manual_seed(43)
    trainer_model = TinyMSHNet()
    manual_model = copy.deepcopy(trainer_model)
    data = torch.randn(2, 3, 24, 24)
    labels = torch.randn(2, 1, 24, 24)

    trainer = _trainer(trainer_model, 0.0)
    trainer.train_loader = [(data.clone(), labels.clone())]
    trainer.optimizer = torch.optim.SGD(trainer_model.parameters(), lr=1e-3)
    trainer.configure_full_dea_trainable = lambda epoch: None
    trainer.compute_plain_mshnet_objective = (
        lambda pred, masks, target, instance_labels, epoch: (
            (pred - target).square().mean(),
            None,
        )
    )
    trainer.compute_crwd_training_objective = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("lambda=0 must not call the CRWD teacher path")
    )

    manual_model.train()
    manual_optimizer = torch.optim.SGD(manual_model.parameters(), lr=1e-3)
    _, manual_pred = manual_model(data, True)
    manual_loss = (manual_pred - labels).square().mean()
    manual_optimizer.zero_grad()
    manual_loss.backward()
    manual_optimizer.step()

    trainer.train(epoch=6)
    assert trainer_model.forward_training_flags == [True]
    manual_state = manual_model.state_dict()
    trainer_state = trainer_model.state_dict()
    assert trainer_state.keys() == manual_state.keys()
    for name in trainer_state:
        assert torch.equal(trainer_state[name], manual_state[name]), name


def test_real_mshnet_contract_runs_without_changing_parameter_schema() -> None:
    torch.manual_seed(47)
    model = MSHNet(3).train()
    parameter_keys = tuple(model.state_dict())
    trainer = _trainer(model, 0.2)
    data = torch.randn(1, 3, 32, 32)
    labels = torch.zeros(1, 1, 32, 32)
    instances = torch.zeros(1, 1, 32, 32, dtype=torch.long)
    labels[0, 0, 14, 14] = 1.0
    instances[0, 0, 14, 14] = 1
    _, canonical = model(data, True)
    result = trainer.compute_crwd_training_objective(
        data,
        canonical,
        labels,
        instances,
        epoch=6,
        iteration=0,
    )
    assert bool(torch.isfinite(result["loss"]))
    assert tuple(model.state_dict()) == parameter_keys
    assert model.training
