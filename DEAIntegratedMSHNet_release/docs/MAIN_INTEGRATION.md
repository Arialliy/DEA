# `main.py` integration

The maintained `/home/ly/DEA/main.py` is already integrated. For a historical
clean checkout, copy `model/dea_integrated_mshnet.py` into the repository and
run the portable architecture patcher from the repository root:

```bash
python /path/to/tools/patch_main_for_dea_integrated.py --main main.py
```

The patcher adds:

- `--model-type dea_integrated`;
- a dedicated `DEAIntegratedMSHNet` construction branch;
- baseline loading that allows only four routing-cell prefixes to be missing
  and only the legacy DEA-lite `decidability_head.*` prefix to be unexpected;
- argument validation that prevents mixing Integrated DEA with DEA-lite losses;
- run names and metadata for all required structural ablations.

The maintained entry additionally fixes the old validation leakage: train mode
uses a deterministic holdout from the official training manifest (or explicit
fit/validation manifests), while test mode uses only the official test
manifest. It rejects any train/validation/test name overlap and persists the
exact split manifests and hashes. The portable patcher does not patch
`utils/data.py`; do not use its historical `val=test` behavior for experiments.

## Formal model

```bash
python main.py \
  --model-type dea_integrated \
  --init-from-baseline /path/to/mshnet_checkpoint.tar \
  --integrated-routing-mode dea \
  --integrated-decoder-routing true \
  --integrated-scale-routing true \
  --integrated-route-upsample-mode nearest-exact
```

## Required ablations

Original MSHNet:

```bash
python main.py --model-type mshnet
```

Only replace static scale fusion:

```bash
python main.py --model-type dea_integrated \
  --init-from-baseline /path/to/mshnet_checkpoint.tar \
  --integrated-decoder-routing false \
  --integrated-scale-routing true
```

Only replace decoder concatenation fusion:

```bash
python main.py --model-type dea_integrated \
  --init-from-baseline /path/to/mshnet_checkpoint.tar \
  --integrated-decoder-routing true \
  --integrated-scale-routing false
```

Complete Integrated DEA:

```bash
python main.py --model-type dea_integrated \
  --init-from-baseline /path/to/mshnet_checkpoint.tar
```

Remove uncertain identity:

```bash
python main.py --model-type dea_integrated \
  --init-from-baseline /path/to/mshnet_checkpoint.tar \
  --integrated-routing-mode soft_tri
```

Replace routing with parameter-matched continuous attention:

```bash
python main.py --model-type dea_integrated \
  --init-from-baseline /path/to/mshnet_checkpoint.tar \
  --integrated-routing-mode attention
```

Every paired run must use the same baseline checkpoint, data split, random seed,
epoch count, optimizer, learning-rate schedule, augmentation, and full-network
training policy. Do not enable `--dea-lambda-single`, `--dea-lambda-dec`, or
`--dea-lambda-empty` for Integrated DEA.

For NUAA-SIRST, `--val-fraction 0.2 --split-seed 20260710` yields 170 fit /
43 validation from the 213 official training samples, while the 214 official
test samples remain untouched. Use `--train-split-file`, `--val-split-file`,
and `--test-split-file` when frozen external manifests are available.

`--integrated-route-loss-weight` exposes a training-only residual-action
identifiability control and defaults to `0`. Its first smoke (`0.05`) learned
soft keep confidence but no hard increase/decrease actions, so it is not the
formal default. If used for diagnosis, also record
`--integrated-route-ramp-epochs` and include the supervision-only controls from
`EXPERIMENT_PROTOCOL.md`. The attention comparator requires route-loss weight
zero because its third logit controls amplitude rather than keep probability.
