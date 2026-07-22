# Wurgaft Balls & Urns (atomic)

Thin SCC wrappers around the parent-repo **categorical-sequence** pipeline
(Wurgaft et al., *In-Context Learning Strategies Emerge Rationally*).

This does **not** reimplement the task. Training, data generation, M/G
predictors, and relative-distance analysis all live under
`src/code/` at the repo root.

## Paper defaults (canonical slice)

| Knob | Value |
|------|--------|
| `num_dims` (V) | 8 |
| `context_length` (C) | 128 (+ start token `s`) |
| Architecture | GPT-NeoX, 1L / 1H / hidden=64 / mlp×4 |
| Optimizer | AdamW, lr=5e-4, warmup=5000, `constant_with_warmup`, bs=64 |
| Steps | 1e5, sqrt checkpoints, eval every 500 |
| Prior | Dirichlet(1,…,1) |
| Sweep | `D ∈ {4,8,16,32,64,128,256,512,1024,2048,4096}` |

Configs: `experiments/categorical-sequence/wurgaft-baseline/yaml-configs/`.

Relative distance uses the Wurgaft barycentric formula (symmetrized KL):
`r = (d(h,G) - d(h,M)) / d(G,M)`, `d_rel = (r+1)/2` → 0≈G, 1≈M.

## Quick start (SCC)

```bash
cd /projectnb/buinlp/rathin/rational-icl

# 1) Generate Dirichlet train/eval pools (once)
bash composition/urns_and_balls/generate_data.sh

# 2) Submit D sweep
mkdir -p composition/urns_and_balls/logs
cd composition/urns_and_balls
bash run_sweep.sh
# DRY_RUN=1 bash run_sweep.sh   # print only

# Single job:
#   qsub -N bu_D64 train_scc.sh 64

# 3) After training finishes, analyze (relative distance + figures)
bash composition/urns_and_balls/analyze.sh
# or on SCC GPU:
#   qsub -N bu_analyze analyze_scc.sh
#   qsub -N bu_analyze -v ANALYZE_ARGS="--load-saved" analyze_scc.sh
#   bash composition/urns_and_balls/analyze.sh --no-figs
#   bash composition/urns_and_balls/analyze.sh --load-saved
```

Note: analysis must use `PYTHONPATH=<repo>/src` (so `code` is importable).
`analyze.sh` sets this correctly. Do **not** put the repo root on
`PYTHONPATH` — that shadows stdlib `code` and breaks imports.

## Outputs

Under `$CACHE_DIR` (from repo `.env`, default `/projectnb/buinlp/rathin/cache`):

- `categorical-sequence/data/train-data/` — fixed task pools of size D
- `categorical-sequence/data/eval-data/` — OOD eval tasks (seed+1)
- `categorical-sequence/transformers/` — checkpoints + `logs.csv` per run

## Parent modules

- `src.code.generate_data` — Dirichlet pools
- `src.code.train` — HF Trainer
- `src.code.models` — M (`BayesianAveragingCategoricalSequence`), G (`PosteriorMeanCategoricalSequence`)
- `src.code.analysis_pipeline` / `analysis_utils` — relative distance + figures
