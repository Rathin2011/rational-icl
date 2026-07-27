#!/bin/bash -l
#
# Evaluate the M-vs-G behavioral sweep on all checkpoints of one baseline run.
#
#   qsub -N cu_mg_eval_D64_s1 -v NUM_TASKS=64,SEED=1 eval_mg_scc.sh
#
#$ -P buinlp
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=02:00:00
#$ -j y
#$ -o logs/$JOB_ID_$JOB_NAME.log

set -euo pipefail

REPO="${REPO:-/projectnb/buinlp/rathin/rational-icl}"
DIR="${REPO}/composition/compositional_urns"
cd "${DIR}"
mkdir -p logs

if [[ -z "${CACHE_DIR:-}" && -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC2046
  export $(grep -E '^CACHE_DIR=' "${REPO}/.env" | xargs)
  set +a
fi
export CACHE_DIR="${CACHE_DIR:-/projectnb/buinlp/rathin/cache}"

CONDA_ENV="${CONDA_ENV:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer}"
module load miniconda
conda deactivate || true
# shellcheck disable=SC1091
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

NUM_TASKS="${NUM_TASKS:?set -v NUM_TASKS=<D>}"
SEED="${SEED:-1}"
N_SEQUENCES="${N_SEQUENCES:-64}"

python eval_mg_sweep.py --num_tasks "${NUM_TASKS}" --seed "${SEED}" --n_sequences "${N_SEQUENCES}"

echo DONE
