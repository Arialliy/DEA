# DEAIntegratedMSHNet patch package

This package implements an endogenous, baseline-preserving DEA for the current
`Arialliy/DEA` MSHNet codebase.

## Contents

- `model/dea_integrated_mshnet.py` — formal model implementation.
- `model/dea_integrated_loss.py` — optional training-only residual-action
  identifiability control; disabled by default.
- `tools/patch_main_for_dea_integrated.py` — idempotent `main.py` CLI patcher.
- `tools/verify_dea_integrated.py` — checkpoint equivalence, gradient,
  complexity, and hygiene verifier.
- `tests/test_dea_integrated_mshnet.py` — structural identity, gradient,
  interpolation, and ablation tests (11 collected cases).
- `docs/MAIN_INTEGRATION.md` — commands for the formal model and six required
  comparisons.
- `docs/PAPER_POSITIONING.md` — unified mathematical framing and reviewer-risk
  analysis.
- `docs/EXPERIMENT_PROTOCOL.md` — hard gates for NUDT/NUAA paired finetuning.

## Install into the repository

From the root of a clean DEA checkout:

```bash
cp /path/to/DEAIntegratedMSHNet_release/model/dea_integrated_mshnet.py model/
cp /path/to/DEAIntegratedMSHNet_release/model/dea_integrated_loss.py model/
cp /path/to/DEAIntegratedMSHNet_release/tools/verify_dea_integrated.py tools/
cp /path/to/DEAIntegratedMSHNet_release/tests/test_dea_integrated_mshnet.py tests/
python /path/to/DEAIntegratedMSHNet_release/tools/patch_main_for_dea_integrated.py \
  --main main.py
```

The maintained workspace in `/home/ly/DEA` is already integrated more fully
than the portable patcher: its `main.py` also separates fit/validation/test,
audits split overlap, persists split hashes/manifests, validates checkpoint
semantics, and writes route diagnostics. For a clean external checkout, the
patcher wires the architecture into the historical entry point, but the same
data-protocol changes must also be ported before any paper experiment.

Then run from the repository root:

```bash
PYTHONPATH=. pytest -q tests/test_dea_integrated_mshnet.py
python DEAIntegratedMSHNet_release/tools/verify_dea_integrated.py \
  --checkpoint /path/to/mshnet_checkpoint.tar --device cuda
```

## Local smoke-test result

The repaired implementation was exercised in the current repository:

- all four decoder masks and the final prediction: bitwise-identical to MSHNet
  (`0.0` maximum error) for real NUAA and NUDT checkpoints;
- grouped four-scale decomposition diagnostic: `9.1553e-5` (NUAA) and
  `4.5776e-5` (NUDT); the executable baseline uses the original direct 4→1
  convolution because a reordered grouped sum is not bitwise identical;
- all four initial routes: uncertain;
- every named parameter in all four routing cells: nonzero gradient;
- full repository tests: `45 passed`;
- one-epoch NUAA mechanics smoke: 170 fit / 43 validation, IoU `0.7706`,
  PD `1.0000`, FA `6.7423` per million pixels.

The smoke also falsified an important training assumption: without explicit
action semantics, target occupancy stayed zero; a foreground/background-
balanced residual-action loss (`weight=0.05`, margin `0.2`) instead converged to
100% hard keep/abstain after one epoch. The latter remains in the code only as
an **experimental control** and is disabled by default. Neither run establishes
an effective three-action mechanism.

The one-epoch number is **not a publishable comparison**. Its initialization
checkpoint had previously been trained on all 213 official NUAA training
images and selected using the test set. It validates execution only. A clean
paper protocol must retrain MSHNet on the frozen 170-image fit manifest, select
on the 43-image validation manifest, and evaluate the untouched 214-image test
manifest once.
