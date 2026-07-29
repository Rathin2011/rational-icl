# Sparse-Quadratic Composition Experiment — final specification (v7)

## Context

Seven review rounds. Round 1: hidden task identity, non-repeating
coefficients, and the no-atomic control were incorrectly flagged as
weaknesses. Round 2: five precision fixes (checkpoint-level mixture
fitting, per-trial patching gate, fuller Bayesian simulation,
`max_seq_length=224`, identifiability-rate breakdown). Round 3: five
substantive errors in the task family, sampling, sequence lengths, the
primitive gate, and a reuse claim for existing predictor code. Round 4:
the population-stream pilot's own `D` value, the primitive-evaluation
gate, and the Bayesian simulation's budgeting were all fixed, along with
ten smaller precision items. Round 5: the population-stream
pilot's fix from round 4 still misused `D` (a `D=13,770` pool under this
document's own §1 sampling rule has 13,770 tasks, one `S` fixed per `c`,
not 137,700 — the average-exposure and identical-support claims built on
top of that were both consequently wrong); §4's evaluation truncation had
an off-by-one that would have been a real implementation bug in a causal
LM; the held-out check needed a third, genuinely neutral Bayesian
reference; the simulation's patching-identifiability sampling procedure
was never actually specified; plus clarifications to the operative §11
gate, the two-probe joint threshold, calibration scope, small-`D` patching
labeling, control-contrast statistics, and five smaller definitions. After
this round, no further task-family or Bayesian-definition problems were
expected. Round 6 made positional matching mandatory at every final query,
separated clean-latent accuracy from noisy-output NLL, budgeted paired
patching-identifiability trials, removed conflicting layer-selection and
test-time multiplicity procedures, and tightened invalid-token and patching
control statistics.
Round 7 defined the donor/base patch statistics explicitly, separated the
real-patch crossed effect from its shuffled-control contrast, and removed two
small inherited ambiguities in the stopping criteria. Remaining work is the
§13 calibration list.

## 1. Task family — `F_5 = {0,1,2,3,4}`

First stage: `x∈F_5^5`. Support `S={i,j}⊂[5]`, `i<j` fixed by convention
(`g_S(x)=(x_i,x_j)`, smaller index first). 10 possible supports.

Second stage: `c=(c_0..c_5)∈F_5^6`, subject to three non-degeneracy
constraints:
```
(c_3,c_4,c_5) ≠ (0,0,0)     — genuine degree 2, not reducible to linear
(c_1,c_3,c_4) ≠ (0,0,0)     — z_1 actually appears (else S unidentifiable)
(c_2,c_4,c_5) ≠ (0,0,0)     — z_2 actually appears (else S unidentifiable)
```
By inclusion-exclusion: `325` excluded out of `5^6=15625`, **15,300 valid
`c` values**, population `10×15,300 = 153,000` tasks.

```
f_c(z_1,z_2) = c_0 + c_1 z_1 + c_2 z_2 + c_3 z_1^2 + c_4 z_1 z_2 + c_5 z_2^2  (mod 5)
```

**Held-out coefficient reservation** (fixed once, globally, before any
finite pool is sampled): reserve a fixed 10% of the 15,300 valid `c`
values (1,530 `c`'s → 15,300 tasks crossed with all 10 `S`) permanently
for the held-out generalization check (§12). The remaining **13,770 `c`'s
(137,700 tasks)** — call this set `C_train` — are the only ones ever
eligible for any finite training pool `𝒯_D`. 10% is a proposed default
(§13).

**Finite-pool task sampling** (used for the §7-§11 grid): `D` = number of
distinct `c` values sampled without replacement from `C_train`
(`D≤13,770`), then an independent `S` per `c` (`S` may repeat — `g_S` is a
trivial coordinate-selection with no memorization burden). A `D`-pool
therefore has exactly `D` tasks, one `S` per `c`, never all 10. Distinct
`c`'s makes "coefficients don't repeat" a guaranteed property **of finite
pools specifically** — this does not apply to the population-stream pilot
(§12), which uses no frozen pool at all and draws `c` and `S`
independently every episode, so the same `c` recurring across episodes
there is expected and not a violation of anything.

## 2. Observation noise

`P(ṽ=v)=0.98`, `P(ṽ=v')=0.005` per wrong value, applied independently to
every emitted field value, including the final query's answer.

**Input distributions**: `x~Uniform(F_5^5)` i.i.d., `z~Uniform(F_5^2)`
i.i.d., both with replacement across draws within a sequence.

## 3. Vocabulary and example types

```
Vocabulary (12 tokens): VALUE_0 VALUE_1 VALUE_2 VALUE_3 VALUE_4
                        [STAGE1] [STAGE2] [COMPOSE] [QUERY] [OUT] [SEP] [PAD]
```
```
[STAGE1] x1 x2 x3 x4 x5 [OUT] z1 z2 [SEP]     — 10 tokens, target (z1,z2)=g_S(x)
[STAGE2] z1 z2 [OUT] y [SEP]                   — 6 tokens,  target y=f_c(z)
[COMPOSE] x1 x2 x3 x4 x5 [OUT] y [SEP]          — 9 tokens,  target y=f_c(g_S(x))
```

## 4. Training sequences

Each sequence: `τ~Uniform(𝒯_D)`, fresh inputs per block (§2), demonstrations
randomly interleaved, one final marked query: `[QUERY][COMPOSE] x1..x5 [OUT] y [SEP]`
(10 tokens).

**Atomic condition**: 8 `[COMPOSE]` + 4 `[STAGE1]` + 8 `[STAGE2]`.
`8×9 + 4×10 + 8×6 = 160` attended demo tokens. For the full training
sequence, insert 20 masked `[PAD]` tokens immediately before `[QUERY]`,
giving **190 physical tokens total with query**.
**No-atomic control**: 20 `[COMPOSE]`, `180` demo tokens, **190 total**.

| | attended demo tokens | physical pre-query tokens | query position | supervised values/seq |
|---|---|---|---|---|
| atomic | 160 | 180 (includes 20 masked `[PAD]`) | 180 | 25 (8+8+8 demos + 1 query) |
| no-atomic | 180 | 180 | 180 | 21 (20 demos + 1 query) |

Report both examples-seen and supervised-output-tokens-seen per condition
throughout. Positional matching is mandatory because the central comparison
attributes a difference to atomic supervision: merely reporting a different
absolute query position would not control that alternative explanation.
For **every final marked composite query**, including behavioral prefixes at
all `t`, population/held-out evaluation, and patching source/base prompts,
insert enough masked `[PAD]` tokens immediately before `[QUERY]` to place
`[QUERY]` at physical position 180. Thus shorter prefixes contain more masked
padding; the attended evidence still consists of exactly the requested `t`
demonstrations. `[PAD]` is masked from attention and loss. Position IDs count
the physical slots (they are not compressed by the attention mask), so
`[QUERY]` receives position ID 180 in every condition and at every `t`. The
padding convention must be implemented explicitly rather than relying on a
library's default position-ID construction.

`max_seq_length=224`, BOS insertion explicitly disabled unless intended.

**Value-space renormalization** (used for JS mixture fitting, predictive-
shape comparisons, and model-output probability gates):
```
p̄_θ(v) = p_θ(v) / Σ_{u∈F_5} p_θ(u)     for v∈F_5
```
Separately report total **invalid-answer mass** at the query position,
`1−Σ_{v∈F_5}p_θ(v)`, which includes all structural markers **and `[PAD]`**.
Optionally split it into marker mass and `[PAD]` mass for diagnosis. Adding a
masked padding token must not create a probability-mass loophole hidden by
value renormalization.

**Training-loss contract**: loss is output-only cross-entropy, computed
exclusively at value-token positions following an `[OUT]` marker — both
inside demonstration blocks and at the final query — with all other
positions label-masked (`-100`). Training data is generated
online/streaming, no fixed finite dataset.

**Evaluation truncation (fixed — was off-by-one)**: evaluation truncates
the prompt **immediately after the final `[OUT]` token** — equivalently,
right before the answer `y` — and reads the logits produced **at** the
`[OUT]` position. `[OUT]` must be part of the prompt: in a causal LM,
`[OUT]`'s own hidden state is what predicts `y`; truncating *before*
`[OUT]` (v4's wording) would leave the model predicting `[OUT]` itself,
not the answer. This also makes the convention identical to the patching
site in §11 (already correct there — "final query `[OUT]` hidden state").

Four independent random-seed streams: model initialization, task-pool
sampling, online training-data generation, and evaluation-episode
generation.

## 5. Bayesian predictors — new implementation required

```
[STAGE1]:  P_ε(z̃1|x_i)·P_ε(z̃2|x_j)
[STAGE2]:  P_ε(ỹ|f_c(z))
[COMPOSE]: P_ε(ỹ|f_c(g_S(x)))
```
`M`: uniform prior over the `D` tasks in the current finite pool. `G`
(called **`G_train`** from here on): uniform prior over the 137,700
`C_train`-eligible tasks. `G(y|s,x*)=Σ_τ P_G(τ|s)·P_ε(y|h_τ(x*))`. This is
the standard reference used throughout §6-§11, where the true task is
guaranteed `C_train`-eligible by construction. Two more references —
`G_all` (full 153,000) and `G_heldout` (reserved 15,300) — are needed only
for the held-out generalization check, where the evaluation distribution
does include held-out tasks; defined in §12.

`compositional_urns`'s `exact_C_GG_predictive` is not reusable — targets
stochastic Dirichlet urns, composite-only sequence pairs, latent-sequence
enumeration for a stochastic `g`/`f`, short-prefix cap. This task needs a
new finite-hypothesis posterior implementation over mixed demonstration
types with deterministic `g_S`/`f_c` and i.i.d. corruption, to `t=20`. Only
the predictive-vector return convention carries over.

Cache `G_train` lazily per evaluated query.

## 6. Bayesian-only simulation (Section 0 — run first, no Transformer)

Pure enumeration/Monte Carlo, no GPU.

**Prefix construction**: one 20-demonstration random interleaving per
episode, truncated at each `t∈{2,4,8,16,20}` — not independently
regenerated per `t`.

**20 independently sampled task pools per `D`.**

**Monte Carlo budget**: within each pool, draw an initial **200
length-20 trajectories per pool and condition, reused at all `t`** (not
"per `t`" — same correction as §7's budgeting). Report between-pool
variance (across the 20 pools) and within-pool Monte Carlo standard error
(across the 200 trajectories) separately.

**Convergence rule (fixed)**: `ℓ_M` and `ℓ_G` are computed on the *same*
underlying episodes, so they're paired, not independent — comparing
`SE(ℓ_Q)` to `|Δℓ|` (v4) understates the available precision and ignores
the correlation. Instead compute the **paired per-episode difference**
`d_i = ℓ_{G,i} − ℓ_{M,i}` directly and use its standard error,
`SE(d̄)`. If `SE(d̄)` exceeds 5% of `|Δℓ(D,t)|`, double the per-pool
trajectory count (ceiling 2,000) and recheck; at the ceiling, report the
achieved `SE(d̄)` honestly rather than chasing an unreachable relative
precision.

Report, separately for the atomic and no-atomic demonstration structures:

- **Entropy under both priors**: `H_M(S|s)`, `H_G(S|s)`, `H_M(c|s)`,
  `H_G(c|s)`.
- **Expected query loss** — the query's own corruption noise is already
  analytically marginalized by the cross-entropy term itself. Remaining
  expectation: (i) task-pool draw `𝒯_D`, (ii) true task `τ`, (iii) the `t`
  context inputs and their corruption noise, (iv) query input `x*`:
  ```
  ℓ_Q = E_{𝒯_D, τ, context+noise, x*}[ H( P_ε(·|h_τ(x*)), Q(·|s,x*) ) ]
  ```
  for `Q∈{M,G}`.
- **`Δℓ(D,t) = ℓ_G(D,t) − ℓ_M(D,t)`**.
- **`TV(M,G)`** (retention-filter check, §7).
- **Patching-identifiability rate — sampling procedure (previously
  unspecified)**: this quantity needs a source *and* a target context, not
  a single episode. Independently of the ordinary behavioral trajectories,
  draw an initial **200 A/B trial pairs per pool and condition**, reuse each
  pair at every nested `t`, and adapt this budget separately: double the
  number of pairs (ceiling 2,000) while the 95% Wilson interval for the joint
  usable-trial rate has half-width greater than 0.05. At the ceiling, report
  the achieved interval and denominator. Each patching-feasibility pair draws:
  distinct
  `τ_A,τ_B`; independent source and target length-20 contexts; matched
  truncations at each `t` (same nested-truncation convention, applied
  independently to each context); independent `x_A,x_B`. Compute all three
  latent outcomes (source `z_A` identification, target `f_{c_B}(z_A)`
  identification, `y_donor`/`y_base` identification), then apply the
  distinctness test. "Identifiable," under both `M` and `G`: **posterior
  mass on the TRUE value ≥0.8** (not the posterior mode — a confidently
  wrong mode must fail this, not pass it). Report the fraction satisfying,
  separately and jointly:
  1. source context identifies the true `z_A`;
  2. target context identifies the true `f_{c_B}(z_A)`;
  3. `y_base`, `y_donor` each identifiable;
  4. all of `y_cross`, `y_donor`, `y_base` pairwise distinct.

  The joint rate is the real usable-patching-trial rate — check against the
  eval budget before committing GPU time to patching.

## 7. Fixed behavioral evaluation sets

**`m=500` length-20 trajectories per `D` per condition** (not per `(D,t)`
cell) — each generated once, truncated at all five `t` values, reused
fivefold. For model evaluation, retain the first `t` demonstrations and then
apply §4's masked pre-query padding so the final marked query remains at
position 180; padding is not Bayesian evidence. Save exact `M`/`G_train`
distributions at every truncation.
Retain, per `t`, only trajectories with `TV(M,G_train)>0.05`; report
`m_retained(D,t)` and use it directly in §8's mixture-fit objective. If
`m_retained` falls below a working minimum (300), generate additional raw
trajectories until met or a budget ceiling (2,000) is hit.

**Metric target convention**: context outputs are sampled with the stated
corruption and are what each posterior conditions on. Query NLL and all
Bayesian loss comparisons analytically average over query corruption using
`H(P_ε,Q)`, rather than scoring one sampled corrupted query token. For a
model, primary NLL uses its **raw** value probabilities,
`−Σ_{y∈F_5}P_ε(y)log p_θ(y)`, so probability assigned to invalid answer
tokens is properly penalized; value-conditional NLL using `p̄_θ` is a
secondary diagnostic. JS mixture fitting and predictive-shape distances use
`p̄_θ` because their Bayesian references live on `F_5`. Accuracy,
including the competence and "unlearned" checks below, is scored against the
clean latent function value; noisy-emitted-token accuracy may be reported only
as a secondary diagnostic. Unless explicitly stated otherwise, accuracy and
invalid-answer-mass competence checks use all unfiltered fixed trajectories, not the
`TV(M,G_train)`-selected subset used for M/G mixture classification.

**Pairing discipline**: atomic and no-atomic evaluation episodes cannot be
literally identical — their demonstration block types structurally
differ. What is shared is the underlying task (`S,c`), the query (`x*`),
and the random-draw stream that selects them; each condition generates its
own condition-specific demonstration sequence around that same task/query.
Call these **matched** episodes. Same correction applies to training-pool
pairing and, where feasible, a shared model-initialization seed.

## 8. Behavioral classification (checkpoint-level)

`k` = training step count (not the theoretical Wurgaft `N_eff`).
```
a*_{k,D,t} = argmin_{a∈[0,1]}  (1/m_retained) Σ_j JS( p̄_{θ,j}, a·M_j + (1-a)·G_{train,j} )
```
`a*≥0.75`→M-like, `a*≤0.25`→G-like, else mixture.

**"neither"** if `min_a JS(...)` exceeds a threshold calibrated
empirically (not borrowed from the unrelated `TV(M,G)` filter — different
scales). Calibrate on synthetic predictive distributions with known ground
truth (pure `M`, pure `G`, controlled `a`-mixtures, small perturbations),
generated from §6's simulation; freeze the resulting threshold before
touching real checkpoints.

**"Unlearned"**: on all unfiltered fixed trajectories, a 95% Wilson
confidence interval for clean-latent value accuracy (on `p̄_θ`) contains
chance (`0.2`), **OR** the mean invalid-answer mass across those trajectories is
`>0.2`. **Precedence**: check "unlearned" first; only if ruled out, proceed
to the M/G/mixture/neither mixture-fit classification — running the
mixture fit on an essentially-noise output distribution can otherwise
produce a spuriously definite-looking `a*`.

**Cross-seed aggregation for the final figure**: `a*` is
checkpoint/seed-specific. A grid cell's headline label is the majority
label across that cell's seeds if at least 2/3 agree; otherwise the cell
is labeled **"seed-inconsistent"** rather than silently collapsed to a
mode — seed variability is itself reportable, not noise to be hidden.

## 9. Model and training

8 layers, 4 heads, hidden 256, MLP 1024, dropout 0, `max_seq_length=224`.
AdamW, lr `3e-4`, wd `0.1`, batch 128, 2000 warmup, constant LR after.
Checkpoints `{0,100,300,1k,3k,10k,30k,100k,200k,300k,500k}`, extendable to
1-1.5M separately for any condition whose relevant validation NLL is still
improving at 500k in the population-stream pilot (§12). For the atomic
condition, primitive metrics are additional diagnostics; no-atomic has no
direct primitive evaluation (§10). Architecture/precision/optimizer
specifics: open choices, §13.

## 10. Primitive evaluations — three distinct tools, not one

**(a) Patching gate** (operational, used only to license §11 trials):
- calibrated probe confidence for `z_A` — hidden-state decodability, *not*
  the model's own predictive distribution;
- the `x'_B` composite-query construction (`g_{S_B}(x'_B)=z_A`, query
  `[QUERY][COMPOSE] x'_B [OUT]`) — checks that `B`'s *overall* composite
  computation produces the crossed value consistent with internally using
  `f_{c_B}` on `z_A`. Does not isolate the second-stage primitive as a
  separate computation (could be solved monolithically); its role is
  limited to validating `y_cross` as a sensible patching target.

**(b) Atomic-condition primitive evaluation** (atomic-trained models
only): append an ordinary, unmarked `[STAGE1]` or `[STAGE2]` block —
structurally identical to a training demonstration, not `[QUERY]`-marked —
and truncate at its own `[OUT]`. Genuinely in-distribution; yields a real
predictive distribution.
- `[STAGE1]` is a **two-token** output `(z̃_1,z̃_2)` — joint NLL is the
  noise-averaged chain-rule quantity
  `E_{z̃_1,z̃_2}[−log p(z̃_1)−log p(z̃_2|z̃_1)]`, evaluated exactly over the
  25 possible corrupted pairs and teacher-forcing each possible observed
  first token when scoring the second. Report clean latent accuracy at both
  granularities: **per-coordinate** uses teacher-forced token predictions,
  while **exact-pair** uses greedy autoregressive decoding (predict `z_1`,
  then predict `z_2` conditioned on that predicted first token) and succeeds
  only when both equal the clean pair. Noisy-emitted-token accuracy is
  secondary only.
- `[STAGE2]`/composite outputs are single-token. Accuracy is against the clean
  latent value; NLL analytically averages over the corrupted output
  distribution as in §7. **Distance from exact Bayes** is mean JS divergence
  in nats between the model's value-renormalized predictive and the exact
  Bayesian predictive. For `[STAGE1]`, report this separately for the first
  token and for the second-token conditional distribution under teacher
  forcing, averaged over the true corruption distribution of the first token,
  plus their mean.

**(c) No-atomic condition**: no in-distribution way to elicit a first- or
second-stage-specific predictive distribution — the model has never seen
`[STAGE1]`/`[STAGE2]` tokens in any capacity. **Do not report primitive
NLL/accuracy for no-atomic models.** Report only probe-decodability of `z`
and composite counterfactual competence via `x'_B`.

## 11. Crossed z-patching

Source `A=(S_A,c_A,x_A)`, base `B=(S_B,c_B,x_B)`, `τ_A≠τ_B` required
explicitly. `z_A=g_{S_A}(x_A)`, `z_B=g_{S_B}(x_B)`. Candidates
`y_cross=f_{c_B}(z_A)`, `y_donor=f_{c_A}(z_A)`, `y_base=f_{c_B}(z_B)`,
**required pairwise distinct** (moved here explicitly from §6 — it belongs
in the operative gate, not just the simulation's identifiability list).

**Per-trial gate**, using §10(a)'s tools only:
- `z_A` resolved from `A`'s context: **calibrated** probe assigns mass to
  the true `z_A` on **both coordinates independently ≥0.8** (the two
  5-way probes' joint threshold — chosen over the alternative product rule
  `p(z_{A1})·p(z_{A2})≥0.8`, which implicitly assumes independence between
  the two probes and is a strictly different, stricter-in-some-regions
  gate; "both coordinates ≥0.8" is simpler and doesn't make that
  assumption);
- `f_{c_B}(z_A)` correctly computed via the `x'_B` construction:
  **model** value-renormalized `p̄_θ(y)≥0.8` **AND invalid-answer mass `<0.2`**
  (both required — high `p̄_θ` alone can coexist with a model mostly
  unsure whether a value belongs at that position at all, since
  renormalization discards invalid-token mass entirely);
- clean donor (`y_donor`) and base (`y_base`) predictions each: model
  `p̄_θ≥0.8` AND invalid-answer mass `<0.2`.

**Calibration scope (fixed — was ambiguous)**: temperature-scale
calibration applies **only to the probe's** confidence output. Never
temperature-scale the model's own `p̄_θ` — doing so would change the
distribution being compared against the exact Bayes `M`/`G_train`
references throughout §6-§8, corrupting the mixture-fit machinery.
Invalid-answer mass applies **only to model-output checks** (the `x'_B`,
donor, base gates) — the probe's 5-way softmax has no vocabulary-level
invalid classes, so this diagnostic is not meaningful for the probe-based
`z_A` check.

If a gate is inapplicable, the outcome is **"not evaluable,"** distinct
from "routing not detected."

**Probe**: two separate 5-way classifiers (one per `z` coordinate).
Task-level split for train/val/test — partition `𝒯_D` itself (60/20/20
target). **For `D≤64`**, where 60/20/20 leaves too few tasks (e.g. `D=16`
gives roughly 9/3/4), label patching results **exploratory/low-task-coverage**
rather than presenting them at the same statistical standing as the larger-D
cells. (Leave-task-out cross-fitting is available as a more rigorous
alternative — every task gets an out-of-fold probe evaluation — but
produces fold-specific `P_z` projectors, which would need its own
aggregation rule across folds; not adopted by default, kept as an option if
the low-`D` regime becomes a focus later.) Calibrate probe confidence
(temperature scaling fit on the validation fold) before treating 0.8 as
operationally meaningful. Validation used only for layer/site selection;
test trials untouched until final reporting.
Both source A and base B in a final reported patching trial must come from
the probe's untouched test-task partition; a train- or validation-task may
not occupy either side of a test intervention.

**`P_z`**: two trained classifiers' weight matrices (each `5×hidden_dim`),
centered, stacked, orthonormalized (QR/SVD). `P_z` = orthogonal projector
onto that span.

Patch at the final query `[OUT]` hidden state, sweep all 8 layers.
`h_B' = h_B + P_z(h_A - h_B)`. Define the real-patch statistics for
`k∈{cross,donor,base}` as:
```
Δ_k^real = log p_patched^real(y_k) − log p_clean(y_k)
```
These use the model's **raw vocabulary probability `p_θ`**, not its
value-renormalized `p̄_θ`, so an intervention must increase the actual output
probability rather than merely redistribute mass conditionally within the
five value tokens. Report the corresponding value-renormalized shifts and
invalid-answer-mass change as secondary diagnostics.

**Controls**, matched to the actual intervention distribution:
- Gaussian-noise: `(μ,Σ)` estimated empirically on validation data from the
  distribution of real
  `P_z(h_A-h_B)` *difference* vectors across trial pairs (not from
  single-hidden-state covariance, which would understate the true variance
  by roughly half); sample from `N(μ,Σ)` and report the real/control norm
  distributions to verify scale matching;
- probe-orthogonal-subspace: match both rank (same subspace dimensionality
  as `P_z`'s span) and norm, per trial;
- shuffled-source-z, whole-residual patch, same-task/same-`z` sanity:
  unchanged.

Sweep all eight layers on a validation patch set and select exactly one
layer/site using the real-minus-shuffled-source primary contrast. Freeze that
choice before opening the disjoint test set. The confirmatory test is performed
only at this selected layer/site; effects at the other layers are validation
diagnostics, not additional test-set discoveries.

**Contrast statistics (previously unspecified)**:
- **Primary paired contrast**: per trial at the frozen layer/site,
  define
  `C_cross = Δ_cross^real − Δ_cross^shuffled`, where
  `Δ_cross^shuffled = log p_patched^shuffled(y_cross) − log p_clean(y_cross)`.
  This real-patch-minus-**shuffled-source-z** contrast uses the same
  perturbation magnitude but mismatched identity, isolating the
  content/identity contribution specifically, not just the act of
  intervening. It is paired at the trial level, then aggregated at the
  seed level per the inferential-unit convention below. Because the layer/site
  was chosen on independent validation data, this is one preregistered
  confirmatory test, not an eight-layer test family.
- **Secondary negative-control family**: at the same frozen site, compare the
  real patch with the Gaussian and matched probe-orthogonal controls; apply
  Holm family-wise correction at 0.05 across these two secondary contrasts.
  Whole-residual and same-task/same-`z` interventions are sanity controls with
  different expected behavior, not exchangeable null hypotheses and not
  members of this multiplicity family.
- **Donor-effect margin**: "no comparable donor-transplant effect" is not
  established by non-significance alone (absence of evidence ≠ evidence of
  absence). For seed `s`, define
  `R_s = Δ_donor,s^real − ρ·Δ_cross,s^real`, where each term is the
  seed-level mean of the corresponding trial-level **real-patch** statistic
  above—not the shuffled-control-subtracted `C_cross`—and `ρ` is the
  preregistered fraction
  (proposed `ρ=0.5`, open choice §13). Require the upper endpoint of the
  seed-level 95% CI for mean `R_s` to be below zero. This tests the relative
  margin as a single linear contrast rather than comparing two separately
  estimated confidence intervals.

**Inferential unit**: model seed / task-pool seed is the primary unit for
confidence intervals, not individual trials.

**Seed count**: three seeds is exploratory only. Determine the
**confirmatory** seed count from a power calculation using the effect-size
variance observed in the exploratory pilot.

**Positive routing result requires all of**: (1) the primary
real-minus-shuffled contrast has a seed-level 95% CI excluding zero in the
positive direction, (2) both secondary negative-control contrasts survive
their Holm correction, (3) the donor linear-contrast upper confidence bound
is below zero, and (4) the per-trial gate is satisfied.

**Equivalence testing**: "equivalence-tested absence" requires a
preregistered margin `δ` and a proper test (e.g. TOST). Either add this
(propose `δ=0.05` nats — open choice §13) or relabel that figure category
to **"no detected effect."**

## 12. Staged execution order

```
§6 Bayesian-only simulation (feasibility + identifiability rate, free)
  → population-stream learnability pilot (no frozen pool — see below)
  → held-out generalization check (reserved 15,300-task set, §1)
  → atomic D=64 pilot (primitive + composite competence, §10)
  → patching (per-trial gate, §11)
  → full grid: D∈{16,64,256,1024,4096} × {atomic, no-atomic}, 3
    exploratory seeds, confirmatory count from power calculation (§11)
```

**Population-stream learnability pilot — corrected definition (round 5).**
Round 4's fix ("`D=13,770`, the full training-eligible population") was
itself still wrong: under this document's own §1 sampling rule, a
`D`-pool has exactly `D` tasks with **one `S` fixed per `c`** — a
`D=13,770` pool has 13,770 tasks, not 137,700, and is a strict subset of
`G_train`'s support (so `TV(M,G_train)` is not identically zero there
either, contradicting round 4's "identical support" claim). The average-
exposure recomputation follows the same error: `64,000,000/13,770 ≈ 4,648`
per task, not `465`.

The actual fix is to drop the frozen-pool mechanism for this pilot
entirely: **`D` is not defined here — there is no `𝒯_D`.** Every episode
independently samples `c~Uniform(C_train)` and `S~Uniform({S_1..S_10})`,
giving the full 137,700-task population fresh per episode. `G_train` (§5)
is defined over exactly this population, so it is the correct Bayesian
reference for this pilot without modification. Because there is no frozen
pool, `M` isn't a meaningful concept here either — this pilot is used
**only for learnability** (does the architecture approach the `G_train`
oracle loss), never for M/G behavioral classification. Run for **both**
atomic and no-atomic conditions (matched pairing as in §7) — a difference
in learnability between the two conditions, independent of the routing
question, is itself diagnostic.

**Held-out generalization check — three-oracle reference (round 5, was
two).** Round 4 had `G_train` (excludes held-out tasks — assigns zero
prior mass to the true task here, so any observed "failure" could be pure
prior misspecification, not an ICL failure) and `G_heldout` (uniform over
just the reserved 15,300 — privileged knowledge the model doesn't actually
have, since it's never told a given query is guaranteed held-out). Neither
is a fair, neutral benchmark on its own. Add a third:
```
G_all: uniform prior over all 153,000 tasks (train-eligible + held-out)
```
`G_all` is the **primary** unseen-task benchmark — it matches what the
model actually has (raw in-context demonstrations, no side-channel about
train/held-out status). Report `G_train` as the learned-prior/
misspecification baseline (how much of any gap is attributable to the
model having implicitly absorbed a prior confined to `C_train`, vs. a
genuine architectural shortfall) and `G_heldout` as a privileged
evaluation-distribution oracle (upper-bound reference, not the primary
comparison).

**Success criterion, population-stream pilot** (rule form fixed — additive
regret, not relative NLL, since relative NLL is distorted by the
irreducible observation-noise floor baked into `NLL_Bayes` itself):
```
NLL_model − NLL_{G_train} ≤ 0.1 nats
```
at `t=20`, on fresh evaluation episodes whose tasks are sampled from the
training-eligible population. (0.1 is a
preregistration-style constant, open to recalibration, §13 — the *form*
of the rule, additive regret against the matching oracle, is fixed.)

**Success criterion, held-out generalization check**: same rule, primary
comparator `G_all`, evaluated explicitly at **`t=20`**:
```
NLL_model − NLL_{G_all} ≤ 0.1 nats
```
with `NLL_model − NLL_{G_train}` and `NLL_model − NLL_{G_heldout}` reported
alongside for interpretation (prior-misspecification magnitude and
distance from the privileged ceiling, respectively).

Stop at any stage that fails rather than proceeding.

**Expected, not guaranteed**: smaller D + longer training favors `M`;
larger D favors behavioral `G`; atomic supervision enlarges the region
where `G` is routed through a patchable `z`; without it, `G` may be
monolithic.

Final figure: background = M/G/mixture/neither/unlearned/seed-inconsistent
(§8); overlay = positive routing evidence / no detected effect (or
equivalence-tested absence, if §13's margin+test are adopted) /
inconclusive / not evaluable.

## 13. Open implementation choices — resolve before preregistration

The scientific design (task family, noise model, M/G formalization,
evaluation structure, patching design) is fixed. These are
engineering/calibration constants and choices still open:

- exact architecture/precision/positional-encoding/initializer/optimizer
  details (decoder-only, pre-LN; positional-embedding scheme; activation;
  tied embeddings; init scheme; AdamW betas/eps; grad-clip; fp32/bf16;
  the mandatory internal `[PAD]` attention-mask and explicit position-ID
  convention from §4; batch/accumulation convention);
- equivalence-testing margin `δ` for "equivalence-tested absence" (§11),
  or drop that claim for "no detected effect";
- the `0.1`-nat additive-regret constant in both §12 success criteria
  (rule form is fixed; magnitude is a calibration constant);
- JS-divergence "neither" threshold, via the calibration procedure in §8;
- Monte Carlo trajectory and A/B-pair count/convergence outcomes for §6 —
  each starts at 200/pool, adaptive; record the numbers actually reached;
- confirmatory seed count for the full grid (§11), from a power
  calculation using exploratory-pilot variance;
- donor-effect equivalence-margin fraction (§11, proposed 50% of the
  crossed-target effect);
- held-out coefficient reservation fraction (proposed 10% of 15,300 `c`'s
  — 1,530 reserved / 13,770 eligible).
