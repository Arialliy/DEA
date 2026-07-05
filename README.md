# DEA-lite MSHNet

This repository contains a DEA-lite extension of MSHNet for infrared small target detection. It keeps the original multi-scale MSHNet training flow and adds lightweight Decidable Evidence Aggregation losses, checkpoint utilities, multi-GPU options, and safer local path defaults.

The project is based on the CVPR 2024 MSHNet implementation:

> Infrared Small Target Detection with Scale and Location Sensitivity

## Overview

![Overview](assert/overview.png)

## What Changed

- Adds DEA-lite outputs to `model/MSHNet.py`.
- Adds anti-sufficiency, decidability, and empty-evidence losses in `model/loss.py`.
- Uses project-local defaults for `datasets/` and `weight/`.
- Adds configurable data loading, checkpoint resume, optimizer reset, and multi-GPU selection.
- Saves optional DEA debug tensors for training inspection.
- Ignores local datasets, weights, run logs, and Python caches in Git.

## Repository Layout

```text
.
├── main.py
├── model/
│   ├── MSHNet.py
│   └── loss.py
├── utils/
│   ├── data.py
│   └── metric.py
├── assert/
│   ├── overview.png
│   └── visual_result.png
├── datasets/      # local only, ignored by Git
├── weight/        # local only, ignored by Git
└── repro_runs/    # local only, ignored by Git
```

## Dataset

Put datasets under `datasets/` by default:

```text
datasets/IRSTD-1K/
├── images/
├── masks/
├── trainval.txt
└── test.txt
```

The loader also supports split files under `img_idx/`, for example `img_idx/train_IRSTD-1K.txt` and `img_idx/test_IRSTD-1K.txt`.

## Training

Single GPU:

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --batch-size 4 \
  --epochs 400 \
  --lr 0.05 \
  --mode train
```

Multi-GPU:

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --batch-size 16 \
  --epochs 400 \
  --lr 0.05 \
  --mode train \
  --multi-gpus true \
  --gpu-ids 0,1,2,3
```

DEA-lite loss weights can be adjusted with:

```bash
--dea-lambda-single 0.10 \
--dea-lambda-dec 0.05 \
--dea-lambda-empty 0.01 \
--dea-tau 0.3 \
--dea-ramp-epochs 20
```

## Resume Training

Resume from the latest checkpoint under `weight/MSHNet-*/checkpoint.pkl`:

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --mode train \
  --if-checkpoint true
```

Resume from a specific checkpoint folder and reset optimizer state:

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --mode train \
  --if-checkpoint true \
  --checkpoint-dir weight/MSHNet-YYYY-MM-DD-HH-MM-SS \
  --reset-optimizer true
```

## Testing

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --batch-size 4 \
  --mode test \
  --weight-path weight/IRSTD-1k_weight.tar
```

## Outputs

Training writes checkpoints and logs to `weight/MSHNet-<timestamp>/`:

- `checkpoint.pkl`
- `weight.pkl`
- `metric.log`
- `epoch_metric.log`
- optional `dea_debug/*.pt`

These outputs are ignored by Git. Keep datasets, trained weights, and reproduction logs outside commits unless they are intentionally published through a release or external storage.

## Visual Results

![Visual Results](assert/visual_result.png)

## Citation

If this code is useful for your research, please cite the original MSHNet paper:

```bibtex
@inproceedings{liu2024infrared,
  title={Infrared Small Target Detection with Scale and Location Sensitivity},
  author={Liu, Qiankun and Liu, Rui and Zheng, Bolun and Wang, Hongkui and Fu, Ying},
  booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition},
  year={2024}
}
```
