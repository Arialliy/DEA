# DEA-lite Run Report

## Baseline

| IoU | PD | FA |
|---:|---:|---:|
| 0.6705 | 0.9150 | 9.2616 |

## Checkpoints

| Run | Slot | Epoch | IoU | PD | FA | Delta IoU | Delta PD | Delta FA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dea_0p01 | best_iou | 291 | 0.6705 | 0.9116 | 7.8951 | 0.0000 | -0.0034 | -1.3665 |
| dea_0p01 | pd_fa_best | 333 | 0.6639 | 0.9286 | 7.8951 | -0.0066 | 0.0136 | -1.3665 |
| dea_0p01 | latest | 399 | 0.6508 | 0.9082 | 8.1988 | -0.0197 | -0.0068 | -1.0628 |
| dea_0p005 | best_iou | 282 | 0.6718 | 0.9014 | 6.4527 | 0.0013 | -0.0136 | -2.8089 |
| dea_0p005 | pd_fa_best | 367 | 0.6637 | 0.9218 | 6.6805 | -0.0068 | 0.0068 | -2.5811 |
| dea_0p005 | latest | 399 | 0.6582 | 0.9116 | 10.4003 | -0.0123 | -0.0034 | 1.1387 |

## Suggested Interpretation

- Use `best_iou` rows to compare IoU-preserving behavior.
- Use `pd_fa_best` rows for PD/IoU-constrained false-alarm control.
- A useful FA-control point should satisfy: IoU within 0.01 of baseline, PD no lower than baseline, and FA lower than baseline.
