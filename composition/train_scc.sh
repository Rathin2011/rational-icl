#!/bin/bash -l
#
# SCC job script for compositional ICL training.
#
# Submit from the composition/ directory (or adjust -wd / cd below):
#   mkdir -p logs
#   qsub -N comp_D64 train_scc.sh
#   qsub -N comp_D4  train_scc.sh 4 10000
#
# Args (optional):
#   $1 = D          (default 64)
#   $2 = max_steps  (default 100000)
#   extra args can be set via EXTRA_ARGS env var
#
#$ -P buinlp
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=12:00:00
#$ -j y
#$ -m beas
#$ -o logs/$JOB_ID_$JOB_NAME.log
# 8-layer model + 1e5 steps: allow up to 12h. Override with qsub -l h_rt=...

set -euo pipefail

# --- job args ---
D="${1:-64}"
MAX_STEPS="${2:-100000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- paths ---
REPO="/projectnb/buinlp/rathin/rational-icl"
COMP="${REPO}/composition"
cd "${COMP}"
mkdir -p logs

# CACHE_DIR from project .env (or override here)
if [[ -z "${CACHE_DIR:-}" && -f "${REPO}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  # .env is KEY=VAL; source carefully
  export $(grep -E '^CACHE_DIR=' "${REPO}/.env" | xargs)
  set +a
fi
export CACHE_DIR="${CACHE_DIR:-/projectnb/buinlp/rathin/cache}"

echo "HOST=$(hostname)"
echo "JOB_ID=${JOB_ID:-local}  D=${D}  MAX_STEPS=${MAX_STEPS}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# --- env ---
# Prefer the project transformer env (has torch + transformers).
# To use your chammiv2 env instead, set:
#   CONDA_ENV=/projectnb/ivc-ml/chaupham/.conda/envs/chammiv2
CONDA_ENV="${CONDA_ENV:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer}"

module load miniconda
conda deactivate || true
# shellcheck disable=SC1091
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

python -c "import torch; assert torch.cuda.is_available(), 'No GPU'; print('GPU:', torch.cuda.get_device_name(0)); print('torch', torch.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"

# --- train (model/hparams defaults match linear-regression main exp) ---
python train.py \
  --num_tasks "${D}" \
  --max_steps "${MAX_STEPS}" \
  ${EXTRA_ARGS}

# --- print error rates ---
RUN_DIR="$(ls -dt "${CACHE_DIR}/composition/runs/D${D}-"* 2>/dev/null | head -1 || true)"
if [[ -n "${RUN_DIR}" && -f "${RUN_DIR}/logs.csv" ]]; then
  echo "=== Error rates: ${RUN_DIR} ==="
  python report_errors.py "${RUN_DIR}/logs.csv"
  python report_errors.py "${RUN_DIR}/logs.csv" --all-steps | tail -80
else
  echo "WARNING: could not find logs for D=${D} under ${CACHE_DIR}/composition/runs/"
fi
