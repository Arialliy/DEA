# DEA-lite 0.005 evidence status after NUAA

## Decision

DEA-lite 0.005 is not a universal positive result across NUDT-SIRST, IRSTD-1K, and NUAA-SIRST.

## Dataset summary

| Dataset | DEA-lite 0.005 status | Paper interpretation |
|---|---|---|
| NUDT-SIRST | Positive | Improves IoU/PD and reduces FA compared with MSHNet baseline. |
| IRSTD-1K | Positive FA-control signal | Use as supportive evidence; report exact paired metrics. |
| NUAA-SIRST | Negative | Dataset-dependent failure: IoU/PD decrease and FA increases. |

## NUAA result

```text
Baseline best-IoU:
  epoch 381
  IoU 0.7461767422765062
  PD  0.9619771863117871
  FA  25.312477183119157

DEA-lite 0.005 best-IoU:
  epoch 324
  IoU 0.7126024590163934
  PD  0.935361216730038
  FA  27.522862514602807

Delta:
  IoU -0.0335742832601128
  PD  -0.026615969581749055
  FA  +2.21038533148365

Gate:
  gate_pass false
  decision DEA_LITE_NEGATIVE_DATASET_DEPENDENT

PD/FA-best:
  absent

Epoch-metric audit:
  num_records 400
  num_gate_pass_epochs 0
  decision NO_GATE_PASS_EPOCH
```

## Evidence paths

```text
Archive/retest directory:
  /home/ly/DEA/repro_runs/dea_lite_0p005_nuaa_negative_archive

Paired delta JSON:
  /home/ly/DEA/repro_runs/dea_lite_0p005_nuaa_negative_archive/nuaa_dea_lite_0p005_vs_mshnet_delta.json

Epoch audit JSON:
  /home/ly/DEA/repro_runs/dea_lite_0p005_nuaa_negative_archive/nuaa_dea_lite_0p005_epoch_metric_audit.json
```

## Forbidden claims

Do not claim:

```text
DEA-lite 0.005 improves all datasets.
DEA-lite universally reduces false alarms.
DEA-lite is globally robust.
NUAA supports the main positive claim.
```

## Allowed claim

```text
DEA-lite 0.005 shows promising false-alarm control on NUDT-SIRST and IRSTD-1K,
while NUAA-SIRST reveals dataset-dependent limitations under the current configuration.
```

## Next action

```text
Treat NUAA-SIRST as archived negative evidence for DEA-lite 0.005.
Keep model/loss/split/metric frozen for this evidence chain.
Do not run lambda=0.01 as an immediate rescue.
```
