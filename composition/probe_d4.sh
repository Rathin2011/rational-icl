#!/bin/bash
# Quick patching probe on existing D=4 / N~10k ID checkpoint.
set -euo pipefail
cd "$(dirname "$0")"

PY="${CONDA_ENV:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer}/bin/python"
CKPT=/projectnb/buinlp/rathin/cache/composition/runs/D4-2L-4H-128d-512ff-lr0.001-bs256-10000steps-seed1/checkpoints/checkpoint-9613

"$PY" probe_composition.py \
  --checkpoint "$CKPT" \
  --num_tasks 4 \
  --task_pool train \
  --n_trials 200
