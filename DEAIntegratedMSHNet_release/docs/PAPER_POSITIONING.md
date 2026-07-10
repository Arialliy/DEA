# Paper-level formulation: one mechanism, not a stack of modules

## Recommended name

Use **Decidable Evidence Recursion (DER)** as the method-level concept and keep
`DEAIntegratedMSHNet` as the implementation class. “Integrated DEA” is a useful
engineering label, but a paper contribution should be named after its governing
operator rather than after two code modules.

## Unified operator

At decoder scale \(s\), let \(e_s\) be the encoder feature and \(d_{s+1}\) the
upsampled coarser decoder feature. Define bounded projected evidence

\[
\bar e_s=\tanh(P^e_s e_s),\qquad
\bar d_s=\tanh(P^d_s d_{s+1}),
\]

\[
a_s=\bar e_s\odot\bar d_s,\qquad
r_s=|\bar e_s-\bar d_s|.
\]

A router maps \([a_s,r_s]\) to three action logits and makes a hard spatial
decision

\[
k_s\in\{\text{target},\text{clutter},\text{uncertain}\}.
\]

The decoder fusion is one action-conditioned operator

\[
D_s(e_s,d_{s+1})
=B_s(e_s,d_{s+1})
+g_s^t U_s(e_s,d_{s+1})
-g_s^c U_s(e_s,d_{s+1}),
\]

where \(B_s\) is exactly the original MSHNet decoder fusion. An uncertain winner
makes \(g_s^t=g_s^c=0\), hence \(D_s=B_s\) by construction.

The same action closes the prediction. Keep the original four-channel final
convolution as the executable baseline and decompose its kernel only to obtain
per-scale interventions:

\[
c_s = W_s * m_s,\qquad
z_{\mathrm{base}}=\operatorname{Conv}_{4\rightarrow1}
([m_0,m_1,m_2,m_3];W,b).
\]

Then apply the route corrections in coarse-to-fine order

\[
Z_4=z_{\mathrm{base}},
\]

\[
Z_s=Z_{s+1}+g_s^t|c_s|-g_s^c|c_s|,
\quad s=3,2,1,0,
\]

and output \(z_{\mathrm{DER}}=Z_0\). There is no second router, learned final
head, topology bridge, prototype bank, or component graph. The terminal step is
the same decision variable acting on an analytically decomposed baseline
quantity.

In exact real arithmetic, `z_base` also equals `b+sum(c_s)`. On GPU floating
point, however, the grouped decomposition changes the reduction order and
differs by up to `9.1553e-5` on the checked checkpoints. The direct convolution
is therefore essential to the bitwise baseline-embedding claim; the grouped
contributions are used only for corrections and diagnostics.

## Four properties that should be stated and tested

### 1. Exact baseline embedding

When every route is uncertain,

\[
D_s=B_s\quad\forall s,\qquad z_{\mathrm{DER}}=z_{\mathrm{MSHNet}}.
\]

Thus MSHNet is not merely a comparison model; it is an exact submodel of DER.
A checkpoint can be transported without changing any encoder, decoder, output
head, or final-kernel parameter.

### 2. Abstention is an algebraic identity

Uncertain is not encouraged by a loss and is not an approximate low attention
weight. It is the zero element of the signed intervention. This distinction is
central to the claim of “decidable” routing.

### 3. Sign-monotone terminal actions

For fixed scale contribution \(c_s\),

\[
\Delta_s^{\mathrm{target}}=g_s^t|c_s|\ge 0,
\qquad
\Delta_s^{\mathrm{clutter}}=-g_s^c|c_s|\le 0.
\]

A target decision cannot lower the final logit and a clutter decision cannot
raise it, regardless of the sign learned by the original final kernel. Generic
attention reweighting does not provide this action-level guarantee.

### 4. Semantic anchoring across feature and logit domains

A generic mixture router has label-permutation symmetry: expert labels can be
swapped without changing meaning. Here, target and clutter actions are anchored
by their terminal monotonic effects, and the same anchored route controls the
recursive decoder update. This coupling is a stronger claim than “we added an
attention block to each skip connection.”

## Necessary gradient correction

A literal branch such as

```python
if argmax(route) == uncertain:
    return baseline
```

has the desired forward identity but blocks the route predictor’s gradient over
the entire uncertain region. The implementation therefore uses a
**hard-forward, soft-backward straight-through route**:

\[
h=\operatorname{onehot}(\arg\max p),\qquad
\tilde h=h+(p-\operatorname{stopgrad}(p)).
\]

Forward decisions remain hard and uncertain remains exact identity; backward
optimization uses the softmax Jacobian. This is an optimization device, not a
new loss.

The update branch needs a second, forward-zero surrogate: when the hard winner
is uncertain, the hard signed gate gives the update convolution zero gradient.
The implementation adds and subtracts the same soft signed residual in the
forward pass, restricted to hard-uncertain pixels. It is exactly zero forward
but gives every update parameter a first-step gradient.

## Guaranteed uncertain initialization

The projected evidence is bounded by `tanh`. Target and clutter router weights
are initialized in a bounded interval whose worst-case response magnitude is
at most 0.1. Any uncertain bias margin strictly greater than 0.1 therefore
makes uncertain win for every input at initialization, rather than only for a
sampled calibration batch. The margin remains an optimization-sensitive
hyperparameter and must be selected on validation data with route-collapse
diagnostics, never on the test set.

Hard target/clutter gates use nearest or nearest-exact resizing. Bilinear or
bicubic resizing can make both mutually exclusive actions nonzero at one pixel
(and bicubic can overshoot), invalidating the sign/action interpretation. Those
continuous modes are allowed only for explicit soft-routing ablations.

## Action identifiability is unresolved, not a hidden success

The first real-data smoke falsified the assumption that segmentation loss alone
would assign both action semantics: target/increase was never the hard winner.
An optional training-only control decomposes a detached pre-closure Bernoulli
residual into

\[
q_+=y(1-p),\qquad q_-=(1-y)p,\qquad
q_0=yp+(1-y)(1-p),
\]

so that

\[
q_+-q_-=y-p=-\frac{\partial \mathcal L_{\mathrm{BCE}}}{\partial z}.
\]

This names the actions as **increase, decrease, keep/abstain**; it does not
prove intrinsic target/clutter recognition or epistemic uncertainty. The
implemented soft cross-entropy balances foreground and background regions and
detaches the teacher logit. In the current one-epoch check it nevertheless
produced 100% hard keep, so its default weight is zero. It is an identifiability
control, not a claimed contribution or a validated fix.

If developed further, it needs controls against OHEM/focal loss, direct mask
supervision, a non-routing auxiliary action head, residual-sign and residual-
magnitude targets, and frozen versus online teachers. Otherwise reviewers can
correctly attribute any gain to ordinary hard-example or auxiliary
supervision.

## What not to claim

Do not claim novelty for elementwise product, absolute difference, softmax
routing, straight-through estimation, function-preserving initialization,
residual adapters, or dynamic skip fusion individually. Those are established
primitives or standard optimization devices. The defensible contribution is
their constrained composition into one route-coupled recursion with exact
baseline embedding, structural abstention, and sign-monotone terminal closure.

Do not present `DecidableEvidenceRoutingCell` and
`IntegratedScaleEvidenceFusion` as two independent proposed modules. They are
two realizations of the same action operator at internal and terminal nodes.
The architecture figure should show each decoder route continuing directly to
its corresponding scale contribution.

## Likely reviewer attacks and required responses

1. **“This is just attention.”** Use the parameter-matched continuous-attention
   ablation and emphasize hard tri-state action, exact abstention, and terminal
   sign monotonicity.
2. **“This is a residual adapter around concat.”** Report the full 2×2
   factorial design over decoder routing and scale routing. For a higher-is-
   better metric \(M\), report the coupling interaction

   \[
   I_M=M_{11}-M_{10}-M_{01}+M_{00}.
   \]

   A consistently positive paired interaction across seeds is direct evidence
   that route reuse is more than the sum of two attached blocks. For FA, apply
   the same analysis to a higher-is-better utility such as \(-\log(FA+\epsilon)\).
   If the complete model does not outperform both partial variants, the unified
   recursion claim is not experimentally supported.
3. **“The identity initialization alone explains the gain.”** Add an
   identity-initialized continuous-attention control and the no-uncertain
   variant.
4. **“Routes have no semantics.”** Report route-conditioned statistics:
   foreground/background occupancy, logit change, false-alarm change, and
   target recall for each action. The terminal sign constraint should make the
   empirical action semantics directly testable.
5. **“The mechanism only works because the baseline was trained longer.”** Use
   paired continued-training controls from the identical checkpoint, with the
   identical optimizer state policy, epochs, seed, and full-network training.
