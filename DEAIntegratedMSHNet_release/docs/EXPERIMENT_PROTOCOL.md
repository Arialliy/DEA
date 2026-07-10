# Structural and paired-finetuning protocol

## Phase 1: hard structural gates

Run before any dataset experiment:

```bash
python tools/verify_dea_integrated.py \
  --checkpoint /path/to/mshnet_checkpoint.tar \
  --device cuda --height 256 --width 256 \
  --output-json verification.json
```

The run is valid only when all of the following hold:

| Gate | Required result |
|---|---:|
| Initial masks and prediction | bitwise exact (`0.0` error) |
| Grouped decomposition diagnostic error | `< 1e-4` |
| Initial target-gate nonzero count | `0` |
| Initial clutter-gate nonzero count | `0` |
| Initial uncertain occupancy at all four scales | `100%` |
| Gradient of every named routing parameter | `> 0` |
| Explicit route parameters at `r=16` | `20,988` |
| Topology/prototype/component/bridge modules | absent |
| Legacy DEA-lite decidability head | absent |

At a 256×256 input, the four route cells add approximately **0.190710 GMAC**.
The grouped scale decomposition adds **0.002359 GMAC** on top of the retained
direct 4→1 baseline convolution, for **0.193069 added GMAC** in total (or
**0.386138 GFLOPs** under the two-FLOPs-per-MAC convention). This excludes
inexpensive elementwise gates and nearest-neighbor interpolation. The direct
baseline convolution is retained deliberately: replacing it with a reordered
sum changes real-checkpoint outputs by up to `9.1553e-5`.

The current repository MSHNet includes a 521-parameter DEA-lite decidability
head. Because the formal model deletes it, the net parameter increase against
that current class is 20,467; against the original MSHNet without that head,
the increase is exactly 20,988.

## Phase 2: paired short finetuning

### Freeze the three-way data protocol first

NUAA-SIRST contains 427 images: 213 official training and 214 official test
images. With `val_fraction=0.2` and `split_seed=20260710`, the maintained loader
freezes the 213 training images into 170 fit and 43 validation images; the 214
test images remain untouched. NUDT-SIRST analogously becomes 530 fit / 133
validation / 664 test. The six-image `hcval_NUDT-SIRST.txt` file is not an
independent validation set because all six names occur in the official test
split.

The saved `split_train.txt`, `split_val.txt`, and SHA-256 fields in checkpoint
metadata are part of the experimental artifact. Retrain a clean MSHNet under
this protocol. Historical checkpoints selected by evaluating the test split
every epoch may be used only for mechanics smoke tests.

Start every pair from the same MSHNet checkpoint.

| Run | Initialization | Epochs | Trainable network | Loss |
|---|---|---:|---|---|
| MSHNet continued | same checkpoint | same | full | original segmentation loss |
| Integrated DEA | same checkpoint | same | full | same segmentation loss |

Do not reuse an optimizer state for only one side of the pair. Either restore
the identical optimizer state for both, or initialize a fresh identical
optimizer for both. Log the decision.

Run NUDT-SIRST first, then NUAA-SIRST.

The abstention margin is a stability parameter, not a harmless implementation
default. Predeclare a validation-only sensitivity set (for example
`{0.11, 0.2, 0.5, 1.0}`), and report route occupancy alongside segmentation
metrics. Short mechanics runs showed that low margins can cause transient
clutter activation while high margins can remain entirely uncertain; do not
select a margin after looking at test performance.

### Hard progression gates

- **NUDT-SIRST:** IoU must not be below paired continued MSHNet; at least one of
  PD or FA must improve.
- **NUAA-SIRST:** PD must not decrease; IoU must exceed paired continued MSHNet;
  FA must decline materially.
- Proceed to full schedules and multiple seeds only after both datasets pass.

“Materially” must be fixed before seeing results. A defensible choice is an
absolute or relative FA threshold based on the variance across the baseline
seeds, not a post-hoc visual judgment.

## Required model table

1. Original MSHNet.
2. Scale closure only (`decoder_routing=false`, `scale_routing=true`).
3. Decoder recursion only (`decoder_routing=true`, `scale_routing=false`).
4. Complete DER (`true`, `true`, `routing_mode=dea`).
5. No structural uncertain identity (`routing_mode=soft_tri`).
6. Parameter-matched continuous attention (`routing_mode=attention`).

Also include **MSHNet continued** in every paired table; “original reported
MSHNet” is not a valid control for additional finetuning.

### Coupling interaction: the anti-stacking test

Treat decoder routing and scale routing as a 2×2 factorial experiment. For IoU
or PD, compute per seed

\[
I_M=M_{\text{full}}-M_{\text{decoder-only}}
    -M_{\text{scale-only}}+M_{\text{MSHNet-continued}}.
\]

Report the mean, confidence interval, and paired sign of \(I_M\). A positive
interaction is the direct experimental support for the claim that one route
couples feature recursion and terminal evidence closure. If the interaction is
zero or negative, the implementation may still improve accuracy, but the
“unified, non-stacked mechanism” claim is weak. For FA, use
\(-\log(FA+\epsilon)\) or another predeclared higher-is-better transform before
computing the interaction.

## Route diagnostics needed for a top-tier submission

For each scale and dataset, log:

- target/clutter/uncertain occupancy overall;
- occupancy inside ground-truth targets, a narrow target boundary, hard
  background false positives, and ordinary background;
- mean signed final-logit intervention conditioned on each action;
- transition matrix of coarse-to-fine route states;
- PD and FA changes attributable to pixels/components whose route changed from
  uncertain to target or clutter;
- route entropy and collapse statistics over training.

The maintained `main.py` writes overall/GT/background occupancy, entropy,
coarse→fine transition counts, and per-scale intervention magnitude to
`route_metric.jsonl`. Boundary, hard-false-positive, and component-attribution
analyses remain required additions for a final paper.

The no-route-supervision form is a required ablation, not an assumed success:
the current one-epoch NUAA smoke produced zero target actions. The optional
`--integrated-route-loss-weight` control derives increase/decrease/keep targets
from a detached pre-closure Bernoulli residual and balances foreground against
background. At weight `0.05` and margin `0.2`, it still converged to 100% hard
keep after one epoch, so it is disabled by default and must not yet be called
the formal method.

Before promoting any semantic regularizer, compare: no supervision, direct
foreground/background supervision, residual sign, soft residual magnitude,
an auxiliary action head that does not route features, and MSHNet with
OHEM/focal loss. Otherwise a gain can be explained by extra mask supervision
rather than the unified route. Also compare online pre-closure, route-off, and
frozen-baseline teachers; the current pre-closure target is dynamic and is not
a strict MSHNet counterfactual after decoder routing activates.

## Seed policy

Use at least three seeds for the progression study and five for the final table
when compute permits. Report mean±standard deviation and paired per-seed deltas.
Because the initialization embeds the exact same baseline, a paired statistical
analysis is more informative than two unpaired aggregate means.
