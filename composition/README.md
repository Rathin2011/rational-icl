# Compositional Lookup-Table ICL

Train a small transformer to in-context learn composite lookup functions.

## Task

A task is a pair `(g, f)` of deterministic lookup tables:

- `g: X -> Z` with `|X| = 50`, `|Z| = 5`
- `f: Z -> Y` with `|Y| = 50`
- composite `h(x) = f(g(x))`

Priors (drawn independently, once, and held fixed): `g ~ Uniform(Z^50)`,
`f ~ Uniform(Y^5)`.

A pool of `D` tasks is sampled and fixed. Each training example:

1. picks a task `d ~ Uniform({1..D})`,
2. picks a sequence type `tau ~ Uniform({comp, g_only, f_only})`,
3. emits 12 interleaved `(input, output)` pairs (sequence length 24):
   - `comp`:   `x_i ~ Uniform(X)`, output `f(g(x_i))`
   - `g_only`: `x_i ~ Uniform(X)`, output `g(x_i)`
   - `f_only`: `z_i ~ Uniform(Z)`, output `f(z_i)`

The task is never tokenized; the model infers it from the in-context prefix.

## Vocabulary

Disjoint union `X | Z | Y`, 105 tokens: `X = 0..49`, `Z = 50..54`, `Y = 55..104`.
Because the alphabets are disjoint, the sequence type is inferable from the token
ids, so no explicit type token is needed.

## Training

- Objective: autoregressive next-token cross-entropy, scored **only** on the
  deterministic output tokens (input positions are masked with `-100`).
- Model: GPT-NeoX, 2 layers, 4 heads, `d_model = 128`, `d_ff = 512`, context 128.
- Optimizer: AdamW, lr `1e-3` (constant), batch size 256, up to `1e5` steps.
- Diversity sweep: `D in {4, 8, 16, 32, 64, 128, 256}`.

## Files

- `config.py`  vocabulary layout + `CompositionConfig` + paths
- `data.py`    task sampling, sequence construction, train/eval datasets
- `model.py`   `GPTNeoXForCausalLM` builder
- `train.py`   HuggingFace `Trainer` setup, metrics, checkpointing, logging
- `run_sweep.sh` sweep over `D`

## Setup

Uses the shared conda env (Python 3.12, transformers 4.52, torch 2.7):

```bash
conda activate transformer   # or use the interpreter directly:
PY=/projectnb/buinlp/jskyi/miniconda3/envs/transformer/bin/python
```

`CACHE_DIR` is read from the project-root `.env` (or `$CACHE_DIR`). Data,
checkpoints, and logs are written under `$CACHE_DIR/composition/`.

## Run

```bash
cd composition

# single run
$PY train.py --num_tasks 64

# quick smoke test (writes to a local cache)
$PY train.py --num_tasks 4 --max_steps 200 --eval_steps 50 \
    --n_eval 128 --num_checkpoints 3 --cache_dir ./_smoke_cache

# full D sweep
./run_sweep.sh
```

## Outputs

Per run under `$CACHE_DIR/composition/runs/<run_name>/`:

- `config.json`  the resolved configuration
- `checkpoints/checkpoint-<step>/`  sqrt-spaced model checkpoints
- `logs.csv`  training loss and, per eval set, cross-entropy + token accuracy

Eval sets logged as `eval_{id,ood}_{comp,g_only,f_only}_{ce,accuracy}`:
`id_*` reuse the `D` training tasks (in-distribution fit / memorization), `ood_*`
use held-out tasks (generalization to new `(g, f)`).
