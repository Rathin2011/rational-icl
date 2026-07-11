# Compositional Lookup-Table ICL

Train a small transformer to in-context learn composite lookup functions.

## Task

A task is a pair `(g, f)` of deterministic lookup tables:

- `g: X -> Z` with `|X| = 50`, `|Z| = 5`
- `f: Z -> Y` with `|Y| = 50`
- composite `h(x) = f(g(x))`

Priors (drawn independently, once, and held fixed): `g ~ Uniform(Z^50)`,
`f ~ Uniform(Y^5)`.

A pool of `D` tasks is sampled and fixed. Each **training** example:

1. picks a task `d ~ Uniform({1..D})`,
2. emits 16 interleaved `(x, y)` pairs (sequence length 32) with `x_i ~ Uniform(X)`,
   `y_i = f(g(x_i))` (compositional / `comp` only).

**Evaluation** additionally logs `g_only` / `f_only` sequence metrics. The main
composition test is the **intermediate-variable probe** in `probe_composition.py`
(activation patching + `f(z')` check), not those sequence accuracies.

The task is never tokenized; the model infers it from the in-context prefix.

## Vocabulary

Disjoint union `X | Z | Y`, 105 tokens: `X = 0..49`, `Z = 50..54`, `Y = 55..104`.
Because the alphabets are disjoint, the sequence type is inferable from the token
ids, so no explicit type token is needed.

## Training

Matched to the repo’s main **linear-regression** experiment
(`experiments/linear-regression/inverse-sqrt-full-exp`):

- Objective: autoregressive next-token CE on output tokens only (`-100` on inputs).
- Model: GPT-NeoX, **8 layers, 1 head, `d_model=64`, `d_ff=256`**.
- Context: **16** in-context pairs (length 32).
- Optimizer: AdamW, lr **5e-4**, **inverse_sqrt** schedule, warmup **500**.
- Batch size **128**, up to **1e5** steps.
- Diversity sweep: `D in {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}`.

## Files

- `config.py`  vocabulary layout + `CompositionConfig` + paths
- `data.py`    task sampling, sequence construction, train/eval datasets
- `model.py`   `GPTNeoXForCausalLM` builder
- `train.py`   HuggingFace `Trainer` setup, metrics, checkpointing, logging
- `report_errors.py`  print error rates from `logs.csv`
- `run_train.sh`  train one D on GPU, then print error rates
- `probe_composition.py`  intermediate-variable composition probe
- `run_sweep.sh` sweep over `D`
- `generate_sample.py`  print example g, f, h, and sequences

## Composition probe

Tests whether the model stores an intermediate `z' = g(x')` and applies `f`:

1. ICL context of compositional `(x, y)` pairs for a task `(g, f)`.
2. Query `x_a` and `x_b ≠ x_a` (true intermediates `z_a`, `z_b`, targets `y_a`, `y_b`).
3. **Patch:** at bridge layer `L`, replace the residual at the query-`x` position
   from the `x_a` run with that from the `x_b` run. Success if predicted Y becomes
   `y_b = f(z_b)`.
4. **Decode:** from the same residual, take LM-head logits over Z tokens → `ẑ'`,
   check `f(ẑ') == y'`.

```bash
# after training, point at a checkpoint
$PY probe_composition.py \
  --checkpoint $CACHE_DIR/composition/runs/<run_name>/checkpoints/checkpoint-10000 \
  --num_tasks 64 --n_trials 200 --task_pool train

# also try held-out tasks
$PY probe_composition.py --checkpoint ... --num_tasks 64 --task_pool ood
```

Key metrics: `patch_to_y_b_given_diff_y` (vs chance `1/50`) and `f_of_zhat_acc`.
High values support composition (C_GG); high clean OOD accuracy with low patch
success looks like a flat shortcut (G).

## Setup

Uses the shared conda env (Python 3.12, transformers 4.52, torch 2.7):

```bash
conda activate transformer   # or use the interpreter directly:
PY=/projectnb/buinlp/jskyi/miniconda3/envs/transformer/bin/python
```

`CACHE_DIR` is read from the project-root `.env` (or `$CACHE_DIR`). Data,
checkpoints, and logs are written under `$CACHE_DIR/composition/`.

## Run (GPU)

```bash
cd composition
PY=/projectnb/buinlp/jskyi/miniconda3/envs/transformer/bin/python

# one D
qsub -N comp_D64 train_scc.sh 64 100000
./run_train.sh 64 --max_steps 100000

# or call train.py directly
$PY train.py --num_tasks 64 --max_steps 100000

# re-print error rates from an existing run
$PY report_errors.py $CACHE_DIR/composition/runs/<run_name>
$PY report_errors.py $CACHE_DIR/composition/runs/<run_name> --all-steps

# full D sweep (same grid as linear-regression)
./run_sweep.sh
# or submit one job per D, e.g.:
# for D in 4 8 16 32 64 128 256 512 1024 2048 4096; do
#   qsub -N comp_D${D} train_scc.sh $D 100000
# done
```

Error rate = `1 - token accuracy` on scored `y` (or `z`) positions.
Main numbers: `id_comp` (train tasks) vs `ood_comp` (held-out tasks).
Chance error ≈ 98% for 50-way Y.
## Outputs

Per run under `$CACHE_DIR/composition/runs/<run_name>/`:

- `config.json`  the resolved configuration
- `checkpoints/checkpoint-<step>/`  sqrt-spaced model checkpoints
- `logs.csv`  training loss and, per eval set, cross-entropy + token accuracy

Eval sets logged as `eval_{id,ood}_{comp,g_only,f_only}_{ce,accuracy}`:
`id_*` reuse the `D` training tasks (in-distribution fit / memorization), `ood_*`
use held-out tasks (generalization to new `(g, f)`).
