# Full DEA Predeclare Protocol

> This document is a protocol declaration only.
> It does not implement Full DEA code.

## Motivation

DEA-lite is frozen as pilot / ablation / limitation evidence.

DEA-lite shows:

```text
NUDT-SIRST: positive anchor
IRSTD-1K: false-alarm-control signal
NUAA-SIRST: stable negative result
```

The NUAA failure suggests that loss-level evidence regularization is insufficient.

## Full DEA Hypothesis

A full DEA method should explicitly model and intervene on evidence, rather than only regularizing training loss.

## Required Structural Components

A Full DEA implementation must include:

```text
1. target evidence branch
2. clutter evidence branch
3. counterfactual intervention path
4. real prediction and counterfactual prediction
5. inference-time evidence gate or evidence-calibrated segmentation head
```

## Non-Goals

Full DEA must not be implemented as merely:

```text
MSHNet + another scalar loss
MSHNet + lambda tuning
DEA-lite with a new lambda
post-hoc threshold adjustment
dataset-specific lambda selection
```

## First Gate Dataset

Use NUAA-SIRST first.

Reason:

```text
NUAA is the dataset where DEA-lite 0.005 failed stably.
A full DEA design should first show that explicit evidence decomposition and counterfactual control can fix this failure mode.
```

## Baselines

Compare:

```text
MSHNet baseline
DEA-lite 0.005 negative result
Full DEA candidate
```

Frozen NUAA-SIRST references:

```text
MSHNet baseline best-IoU:
  epoch 381
  IoU 0.7461767422765062
  PD  0.9619771863117871
  FA  25.312477183119157

DEA-lite 0.005 best-IoU:
  epoch 324
  IoU 0.7126024590163934
  PD  0.935361216730038
  FA  27.522862514602807

DEA-lite 0.005 status:
  gate_pass false
  decision DEA_LITE_NEGATIVE_DATASET_DEPENDENT
  num_gate_pass_epochs 0
```

The frozen references are not Full DEA evidence. They are the baseline and failure context for the first Full DEA gate.

## First Gate

Full DEA on NUAA-SIRST must satisfy:

```text
IoU >= 0.7461767422765062
PD  >= 0.9569771863117871
FA  <= 25.312477183119157

and Full DEA must outperform DEA-lite 0.005 on NUAA:

IoU > 0.7126024590163934
PD  > 0.935361216730038
FA  < 27.522862514602807
```

The PD tolerance is fixed before implementation:

```text
PD tolerance = 0.005 absolute
PD threshold = MSHNet baseline PD - 0.005
```

## Failure Criteria

Full DEA fails the NUAA first gate if any of the following happens:

```text
1. IoU is below the MSHNet baseline IoU.
2. PD is below MSHNet baseline PD - 0.005.
3. FA is above the MSHNet baseline FA.
4. Full DEA does not outperform DEA-lite 0.005 on IoU, PD, and FA.
5. FA improves only because target detections collapse.
```

Partial recovery is not a success claim:

```text
Better than DEA-lite 0.005 but still worse than MSHNet baseline = diagnostic only.
Lower FA with lower IoU/PD than MSHNet baseline = failure, not success.
Single-seed success on NUAA = first-gate pass only, not a broad Full DEA claim.
```

## Fixed Evaluation Settings

The first NUAA gate must use the same paired evaluation context:

```text
dataset_dir = /home/ly/DEA/datasets/NUAA-SIRST
seed = 20260706
deterministic = true
batch_size = 4
num_workers = 4
pin_memory = false
```

Changing dataset split, metric implementation, thresholding policy, or checkpoint selection rule invalidates the paired first gate.

## Prototype Entry Criteria

Before any NUAA full training, the implementation branch must pass:

```text
python compile checks
shape tests for target evidence, clutter evidence, counterfactual path, and prediction heads
finite-loss tests for the full DEA loss
counterfactual-path tests proving the path is used by the loss
tiny subset smoke training only after shape/loss tests pass
```

## Evidence Rules

Do not claim Full DEA works until:

```text
1. Full DEA code is predeclared and committed.
2. NUAA seed result passes the first gate.
3. The same protocol is reproduced on NUDT-SIRST and IRSTD-1K.
4. Failure analysis confirms reduced false alarms without target collapse.
```

## AAAI Route

Full DEA may become an AAAI route only after it has:

```text
explicit architecture contribution
counterfactual/evidence-control mechanism
NUAA recovery evidence
multi-dataset paired evidence
ablation separating target evidence, clutter evidence, and counterfactual control
```

## Implementation Boundary

This branch must not add:

```text
FullDEAHead
full_dea_loss
counterfactual branch code
inference-time evidence gate code
training scripts for Full DEA
```

The next implementation branch may only start after this protocol is reviewed.
