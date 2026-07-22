#!/bin/bash -l
#
# SCC job script for Wurgaft atomic Balls & Urns (categorical-sequence).
#
# Submit from composition/urns_and_balls/ (qsub resolves relative paths from cwd):
#   mkdir -p logs
#   qsub -N bu_D64 train_scc.sh 64
#   qsub -N bu_D4  train_scc.sh 4
#
# Args:
#   $1 = D  (num_tasks; default 64)
#   Optional override: pass a yaml path as $1 instead of an integer D.
#
#$ -P buinlp
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=12:00:00
#$ -j y
#$ -m beas
#$ -o logs/$JOB_ID_$JOB_NAME.log

set -euo pipefail

REPO="${REPO:-/projectnb/buinlp/rathin/rational-icl}"
EXP_DIR="${REPO}/experiments/categorical-sequence/wurgaft-baseline"
WRAPPER_DIR="${REPO}/composition/urns_and_balls"

ARG="${1:-64}"

if [[ "${ARG}" == *.yaml ]]; then
  YAML="${ARG}"
  # Resolve relative paths against WRAPPER_DIR if needed
  if [[ "${YAML}" != /* ]]; then
    YAML="${WRAPPER_DIR}/${YAML}"
  fi
  D_LABEL="$(basename "${YAML}" .yaml)"
else
  D="${ARG}"
  YAML="${EXP_DIR}/yaml-configs/8dims-${D}tasks-128context-4expansionfactor-1seed.yaml"
  D_LABEL="D${D}"
fi

cd "${WRAPPER_DIR}"
mkdir -p logs

if [[ -z "${CACHE_DIR:-}" && -f "${REPO}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  # shellcheck disable=SC2046
  export $(grep -E '^CACHE_DIR=' "${REPO}/.env" | xargs)
  set +a
fi
export CACHE_DIR="${CACHE_DIR:-/projectnb/buinlp/rathin/cache}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

echo "HOST=$(hostname)"
echo "JOB_ID=${JOB_ID:-local}  LABEL=${D_LABEL}"
echo "YAML=${YAML}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

if [[ ! -f "${YAML}" ]]; then
  echo "ERROR: yaml not found: ${YAML}" >&2
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer}"
module load miniconda
conda deactivate || true
# shellcheck disable=SC1091
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

python -c "import torch; assert torch.cuda.is_available(), 'No GPU'; print('GPU:', torch.cuda.get_device_name(0)); print('torch', torch.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"

cd "${REPO}"
python -m src.code.train "${YAML}"

echo "Training finished for ${D_LABEL}"
