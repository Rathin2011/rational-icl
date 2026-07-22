#!/bin/bash
# Generate Dirichlet task pools for Wurgaft atomic Balls & Urns
# (categorical-sequence) with paper defaults: num_dims=8, D up to 4096.
#
# Usage (from anywhere):
#   bash composition/urns_and_balls/generate_data.sh
#   bash composition/urns_and_balls/generate_data.sh 8 12 16   # extra dims

set -euo pipefail

REPO="${REPO:-/projectnb/buinlp/rathin/rational-icl}"
cd "${REPO}"

if [[ -z "${CACHE_DIR:-}" && -f "${REPO}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  # shellcheck disable=SC2046
  export $(grep -E '^CACHE_DIR=' "${REPO}/.env" | xargs)
  set +a
fi
export CACHE_DIR="${CACHE_DIR:-/projectnb/buinlp/rathin/cache}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

CONDA_ENV="${CONDA_ENV:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  module load miniconda 2>/dev/null || true
  # shellcheck disable=SC1091
  source "$(dirname "${CONDA_ENV}")/../etc/profile.d/conda.sh" 2>/dev/null \
    || source /projectnb/buinlp/jskyi/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

NUM_DIMS=("${@:-8}")
echo "CACHE_DIR=${CACHE_DIR}"
echo "Generating categorical-sequence data for num_dims=${NUM_DIMS[*]}"

python -m src.code.generate_data \
  --setting categorical-sequence \
  --num_dims "${NUM_DIMS[@]}" \
  --max_num_tasks 4096 \
  --random_seed 1 \
  --num_eval_tasks 500

echo "Done. Train pools under ${CACHE_DIR}/categorical-sequence/data/train-data/"
echo "Eval pools under ${CACHE_DIR}/categorical-sequence/data/eval-data/"
