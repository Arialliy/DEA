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

Full DEA is a new structural method on top of the MSHNet source code.
It is not a renamed DEA-lite run, not a new lambda value, and not a loss-only patch.

## MSHNet Source-Level Insertion Rule

The MSHNet encoder / multi-scale decoder should remain the base network.

Full DEA may add modules only at explicit structural insertion points:

```text
1. after multi-scale decoder feature fusion
2. before the final segmentation prediction head
3. optionally as an auxiliary branch from decoder features
```

Full DEA must not change the dataset split, metric implementation, test threshold, or checkpoint selection rule to obtain gains.

## Required Structural Components

A Full DEA implementation must include:

```text
1. target evidence branch E_t
2. clutter/background evidence branch E_c
3. counterfactual intervention path C(F, E_t, E_c)
4. real prediction head y_real
5. counterfactual prediction head y_cf
6. inference-time evidence gate or evidence-calibrated segmentation head
```

Minimum structural contract:

```text
F_dec = MSHNet decoder feature
E_t, E_c = EvidenceHead(F_dec)
F_cf = CounterfactualOperator(F_dec, E_t, E_c)
y_real = SegHead(F_dec, E_t, E_c)
y_cf = SegHead_cf(F_cf)
y_final = EvidenceCalibratedHead(y_real, E_t, E_c)
```

The exact operator can be revised during design review, but the implementation must expose separate target evidence, clutter evidence, and counterfactual prediction tensors.

## Non-Goals

Full DEA must not be implemented as merely:

```text
MSHNet + another scalar loss
MSHNet + lambda tuning
DEA-lite with a new lambda
DEA-lite 0.0025 / 0.005 / 0.01 sensitivity rescue
post-hoc threshold adjustment
dataset-specific lambda selection
```

Loss terms may be used only after the structural branches exist.
If the method can be disabled by setting only `--dea-lambda-single 0`, it is still DEA-lite rather than Full DEA.

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
MSHNet baseline reference:
  IoU 0.7461767423
  PD  0.9619771863
  FA  25.3124771831

DEA-lite 0.005 negative reference:
  IoU 0.7126024590
  PD  0.9353612167
  FA  27.5228625146

Full DEA first gate:
  IoU >= 0.7461767423
  PD  >= 0.9569771863
  FA  <= 25.3124771831

and Full DEA must outperform DEA-lite 0.005 on NUAA:

  IoU > 0.7126024590
  PD  > 0.9353612167
  FA  < 27.5228625146
```

The PD tolerance is fixed before implementation:

```text
PD tolerance = 0.005 absolute
PD threshold = MSHNet baseline PD - 0.005
```

If Full DEA fails this NUAA-first gate, stop and audit the structure before running NUDT-SIRST or IRSTD-1K.

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
