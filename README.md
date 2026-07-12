# DEA: MSHNet Research and Evidence Auditing

This repository is an experimental research workspace built around the CVPR
2024 MSHNet implementation for infrared small target segmentation. It contains
the canonical MSHNet baseline, historical DEA prototypes, and a growing set of
read-only tools for component-level error analysis, low-false-alarm operating
points, cross-seed persistence, feature survival, and evidence-to-decision
conversion.

The default training path remains the original MSHNet topology and SLS
objective. Experimental objectives and DEA variants are explicit opt-in
controls; they are not silently enabled by the baseline configuration.

## Current Research Status

The fixed external objective is:

> Maximize detected target instances at a specified component-level false-alarm
> budget while preserving basic mask quality.

Current evidence gates are conservative:

- Gate D found no single readout, fusion, or simple calibration bottleneck and
  stopped the corresponding method route.
- Gate E-1a and E-1b passed their evidence checks.
- Gate E-1c failed zero-overshoot low-FA transport, so Gate E0 training is
  currently **NO-GO**.
- Gate F v1 completed a read-only operating-transport decomposition. It does
  not authorize a new loss, architecture, solver, or long training run.

These are project-routing results, not paper novelty or performance claims.
The governing definitions and latest status are documented in:

- [North Star objective and Gate E positioning](MSHNet_North_Star_Objective_and_Gate_E_Positioning.md)
- [Gate D/E audit plan and Gate F record](MSHNet_Gate_D_NoGo_and_Gate_E_Training_Credit_Audit_Plan.md)
- [Decision conversion and evidence utilization](MSHNet_Decision_Conversion_Gate_and_Evidence_Utilization_Plan.md)
- [CCSR formal audit](MSHNet_CCSR_Formal_Audit_and_Corrected_Spec.md)
- [CCSR novelty and modification plan](MSHNet_CCSR_Novelty_and_Code_Modification_Plan.md)

## Metric Semantics

The repository keeps two component-matching rules separate:

- `official_legacy`: the original target-order greedy matcher in
  `utils.metric.PD_FA`, using 8-connectivity and strict centroid distance `< 3`.
- `audit_hungarian`: maximum-cardinality, minimum-distance Hungarian matching
  under the same connectivity and distance rule.

The legacy matcher is required for comparison with the original MSHNet
evaluation. The Hungarian matcher is used for stable target identities,
cross-seed persistence, and mechanism audits. Results from the two matchers
must always be named separately.

## Repository Layout

```text
.
|-- main.py                         # training and evaluation entry point
|-- model/
|   |-- MSHNet.py                   # canonical baseline, optional controls
|   |-- loss.py                     # SLS and explicit location-loss modes
|   |-- ccsr/                       # CCSR reference-only implementations
|   |-- omm_flow.py                 # instance-balanced objective controls
|   `-- *_dea_*.py                  # historical/experimental DEA variants
|-- utils/
|   |-- metric.py                   # legacy and audit component metrics
|   |-- component_ledger.py
|   |-- component_operating_point.py
|   |-- cross_seed_persistence.py
|   |-- feature_survival.py
|   `-- target_identity.py
|-- tools/                           # reproducible audit/finalization CLIs
|-- tests/                           # unit and protocol regression tests
|-- datasets/                        # local only; ignored by Git
|-- weight/                          # local only; ignored by Git
`-- repro_runs/                      # local only; ignored by Git
```

## Dataset Layout

Datasets are expected under `datasets/` by default:

```text
datasets/IRSTD-1K/
|-- images/
|-- masks/
`-- img_idx/
    |-- train_IRSTD-1K.txt
    `-- test_IRSTD-1K.txt
```

`datasets/` is intentionally ignored. Split manifests, images, masks, weights,
and run outputs are not published by ordinary Git pushes.

## Canonical Baseline Training

The following command makes the baseline semantics explicit:

```bash
python main.py \
  --mode train \
  --model-type mshnet \
  --dataset-dir datasets/IRSTD-1K \
  --mshnet-objective sls \
  --mshnet-side-supervision canonical \
  --mshnet-train-graph canonical_warm \
  --location-loss legacy \
  --side-location-loss same \
  --lambda-location 1.0 \
  --epochs 400 \
  --batch-size 4 \
  --lr 0.05
```

For a reproducible named run, also set an empty output directory and label:

```bash
--run-dir weight/clean/example_run \
--run-label example_run \
--seed 20260711 \
--split-seed 20260711 \
--deterministic true
```

`--run-dir` refuses to overwrite a non-empty directory. Resume an existing run
with its exact checkpoint directory:

```bash
python main.py \
  --mode train \
  --model-type mshnet \
  --dataset-dir datasets/IRSTD-1K \
  --if-checkpoint true \
  --checkpoint-dir weight/clean/example_run
```

Use `--reset-optimizer true` only after verifying that checkpoint and model
semantics are intentionally compatible.

## Evaluation

```bash
python main.py \
  --mode test \
  --model-type mshnet \
  --dataset-dir datasets/IRSTD-1K \
  --weight-path weight/IRSTD-1k_weight.tar
```

Audit tools under `tools/` add stricter provenance, target-identity, matcher,
checkpoint-policy, and cross-fitting checks. They should be preferred for
research conclusions; `main.py --mode test` remains the simple legacy entry
point.

## Tests

Run the complete test suite in the project environment:

```bash
python -m pytest -q
```

Focused suites cover baseline purity, checkpoint compatibility, component
matching, target identity, CCSR references, feature survival, decision
conversion, low-FA cross-fitting, and cross-seed persistence.

## Experimental Boundaries

- DEA-lite, Full-DEA, Integrated-DEA, predictive-correction, CEV, and related
  models are historical experiments or controls, not the default baseline.
- OMM, CCSR, operating-point MIL, and constrained edit implementations are
  reference or negative-control code unless a governing gate explicitly
  authorizes training.
- Do not report audit-Hungarian values as official legacy metrics.
- Do not treat oracle threshold sweeps as deployable cross-fitted performance.
- Official test sets remain sealed until the method and evaluation protocol
  are frozen by the relevant project gate.

## Outputs and Version Control

Training and audit outputs normally live in `weight/` and `repro_runs/`.
Datasets, checkpoints, logs, generated arrays, and Python caches are ignored by
Git. Publish large artifacts separately with explicit hashes and provenance.

## Visuals

![MSHNet overview](assert/overview.png)

![MSHNet visual results](assert/visual_result.png)

## Citation

If this repository is useful, cite the original MSHNet paper:

```bibtex
@inproceedings{liu2024infrared,
  title={Infrared Small Target Detection with Scale and Location Sensitivity},
  author={Liu, Qiankun and Liu, Rui and Zheng, Bolun and Wang, Hongkui and Fu, Ying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
