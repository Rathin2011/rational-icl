# Diagnosis notes: why no compositional signature is detected

Running log of the investigation into whether the locked C_GG protocol (D=64,
Phase1→Phase2 curriculum, cross-task patching) shows genuine compositional
routing. Kept separate from PROTOCOL.md (the locked spec) so the spec stays
clean and this stays a working diagnostic record.

## 1. Behavioral M-vs-G pilot (D=64, comp-only baseline)

Ran `--phase baseline` (comp-only, no curriculum) at D=64, 2 seeds, 60k steps,
scored with `eval_mg_sweep.py` (M vs the TRUE generalizing predictor
`approx_C_GG_predictive`, **not** the misspecified `atomic_G_predictive` that
`relative_distance.py` confusingly labels "G" — see PROTOCOL.md's own
docstring on that function).

- Retention (TV(M,G) > 0.05) at L=32: 98–100% — the metric has signal.
- `d_rel_m` decreases smoothly with training steps at fixed D=64 (e.g. OOD
  L=32: 0.72 → 0.60 from step 1k to 60k) — model drifts toward memorizing
  the longer it trains, sane Wurgaft-style direction for small D.
- Absolute CE stays near chance throughout (~2.98, chance=ln(20)=2.996) —
  checked this isn't a training failure: even an **oracle** handed the exact
  true task achieves only CE≈2.84. The task is intrinsically very diffuse
  (two stacked Dirichlet(1,·) priors); near-chance absolute CE is expected.
- **Conclusion**: harness works, D=64 cell alone doesn't establish the D-axis
  dependence — would need the full D sweep (paused, see §5) to see whether
  large D flips the trend, which is the actual Wurgaft replication.

## 2. Mechanistic patching: found and fixed a real design gap

Existing `probe.py` did **same-task** patching (one task `d`, two queries
`x_a`,`x_b`). Compared against Khandelwal & Pavlick (arXiv:2510.01685,
`apoorvkh/composing-functions`) and found their design is **cross-task**
(source/destination residuals from different task instances) — and this
isn't just a stylistic difference: with one shared task, both `route` and
`short` targets are values that task already produces on its own, so a model
that merely recognizes "this is task `d`" can land near either candidate
with no transportable `z`. Cross-task patching removes the confound: the
routed target `f_dst(z_src)` is a value **neither** task produces alone.

Fixed in this session:
- Verified low-level hook mechanics first (`capture_hidden`/`patched_forward`)
  via direct empirical tests — patch-with-own-value reproduces the unpatched
  forward exactly, no backward-in-time leakage, patching with a different
  vector measurably changes output. Mechanics were already correct.
- Rewrote `probe.py`'s trial construction to cross-task: distinct
  `d_src != d_dst`, three candidates (`routed`, `transplant`, `unaffected`),
  a discriminating-trial filter (all three must be argmax-distinct — trivial
  to satisfy here, accept rate came out 100%).
- Updated `PROTOCOL.md`, `eval_ckpts_scc.sh`, tests accordingly.

**Result on the real D=64 Phase-2 checkpoint (step 30000), both layers 0 and 1:**

| layer | kl_route | kl_transplant | kl_unaffected | patch_margin_ok | z_decode_acc |
|---|---|---|---|---|---|
| 0 | 0.465 | 0.104 | 0.130 | False | 0.24 |
| 1 | 0.475 | 0.100 | 0.141 | False | 0.21 |

Previously (same-task test): `kl_route`≈`kl_short` (~0.10 vs ~0.11) — an
ambiguous near-tie. Now: `kl_route` is clearly the **worst** match at both
layers, not a tie. This is a confident negative, not an underpowered one —
real progress, even though the answer is "no."

Mild secondary signal: patching moves the output slightly *toward*
`transplant` and *away from* `unaffected` relative to clean — a whiff of
"carried the finished answer" rather than genuine routing. Per the reading
convention: interpretable, but explicitly not evidence of routing.

## 3. Root-cause diagnosis for the negative result

Ruled out: patching bugs, same-task confound (both fixed/verified), Phase-1
gate failure (passed cleanly: `gap_g=0.017`, `gap_f=-0.003`, both within the
0.05 tolerance), wrong patch layer (tried both available layers, same story).

**Primary suspect — task has almost no exploitable signal to begin with.**
Even the *exact* Bayes-optimal predictor barely beats chance on the atomic
sub-tasks:

| | model CE | exact Bayes CE | chance |
|---|---|---|---|
| `g`-only (X→Z) | 1.781 | 1.764 | ln(6)=1.792 |
| `f`-only (Z→Y) | 2.960 | 2.963 | ln(20)=2.996 |

Perfect inference only buys ~0.03 nats over chance on either stage. The
composite `ce_id_comp` sits at ~3.02 and is **completely flat from step 1000
through 30000** — consistent with "nothing left to learn," not "model failed
to learn something learnable." Can't expect to find a `z`-then-`f` circuit
that the model had little incentive to build in the first place.

**Secondary suspect — architecture may be too shallow to route even if there
were signal.** Locked to 2 layers. The (unimplemented) junta-protocol
document already reasoned about this: "routing requires at least one layer
to compute z and a later layer to apply f to it, and a two-layer model
leaves no margin." Patching at layer 0 leaves exactly one remaining layer
(layer 1) to do everything else.

**Aside, not directly about compositionality:** `checkpoint-5000`'s
`ce_comp` spikes to 12.8 (vs ~3.02 everywhere else) — a training
instability right at the Phase-1→Phase-2 transition. Worth knowing about if
that run gets revisited, but orthogonal to the routing question.

## 4. Cross-reference: Wang et al., "Grokked Transformers are Implicit
Reasoners" (arXiv:2405.15071)

Different genre of task (weight-resident parametric knowledge, deterministic
KB, no in-context learning at all — not directly transplantable) but two
findings bear directly on §3:

- Their KB is **fully deterministic** (`(entity,relation)→entity`, zero
  entropy) — the opposite of our Dirichlet(1,·) diffuse setup. This is
  probably *the* reason their model has something worth building a circuit
  for. Directly actionable: sharpen `w_g`/`w_f` concentration (or make them
  deterministic) so the composite task has real accuracy headroom.
- They train **far past training-accuracy saturation** before the
  compositional circuit "groks" into existence — saturation at ~14K steps,
  full generalization not until ~700K (~50x longer). Our Phase-2 is only
  60K steps and was already flat by step 1000. Possible we stopped
  observing right where a grokking curve would still look flat — but this
  only matters *after* fixing the entropy issue (can't grok signal that
  isn't there).
- Their bridge-entity signature only becomes legible via logit lens at
  ~60% depth (layer 5 of 8), not layer 0 — independent support for sweeping
  patch layer once the architecture is deeper than 2 layers.

## 5. Paused work

The D-sweep (`run_mg_sweep.sh`, D∈{4,16,64,256,1024}) is implemented and
pilot-verified (§1) but the remaining 4 D-values were not launched — this
compositional-detection investigation took priority. Resume with
`./run_mg_sweep.sh 4 16 256 1024` once/if this line of investigation is
parked.

## 6. Redesign implemented (not yet run)

All four levers from §4 are now implemented as opt-in parameters (defaults
preserve the locked D=64/2-layer/concentration=1.0 behavior exactly —
verified via `test_config.py`):

- `data.sample_tasks`/`get_or_create_tasks`: `g_concentration`/`f_concentration`
  args (default 1.0). `predictors.py`'s `g_predictive`, `f_predictive`,
  `approx_C_GG_predictive`/`exact_C_GG_predictive` now take matching
  concentration args too — this was a real correctness requirement, not
  just plumbing: those functions hardcode a Dirichlet **prior**, and if it
  doesn't match the actual generative concentration they silently stop
  being genuinely Bayes-optimal. `phase1_gate.py`, `eval_checkpoint.py`,
  `eval_mg_sweep.py` all updated to pass concentration through.
- `train.py`: new CLI flags `--g_concentration`, `--f_concentration`,
  `--n_layers`, `--n_heads`, `--d_model`, `--d_ff`, `--save_every`.
  `config.py`'s `resolved_save_steps` auto-computes a log-spaced checkpoint
  schedule when `max_steps` exceeds the original 60k (was previously
  hardcoded, capped at 60k with no checkpoints beyond it).
- `run_extended_train_scc.sh`: chains multiple `phase=2` SCC jobs (weights-
  only resume between links, same mechanism as the existing Phase-1→
  Phase-2 handoff) to reach a step budget beyond one job's ~12h walltime.
- `probe.py`: `select_best_patch_layer` sweeps candidate layers on a
  validation trial set and returns the best by `route_margin`
  (`min(kl_transplant, kl_unaffected) - kl_route`); `eval_checkpoint.py`
  now selects on validation and reports on a disjoint test seed at the
  selected layer, replacing the old locked-to-layer-0 probe call.

All changes covered by tests (`test_predictors.py`, `test_probe_patching.py`,
new `test_config.py`) and a local smoke test (tiny model, D=4, concentration
0.1, 4 layers, `--save_every`) confirming the full pipeline runs end-to-end.

**Not yet run**: Phase-1 pretraining needs to happen fresh under the new
concentration/architecture before any chained Phase-2 run can start — the
existing `phase1_passed.json` checkpoint (D=64, 2 layers, concentration=1.0)
is not compatible with a sharpened/deeper config. That's the next actual
cluster step, followed by a staged pilot (moderate step count, one chain
link) before committing to the full extended run, per the plan's own
staging philosophy.
