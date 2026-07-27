#!/bin/bash -l
#
# SCC job for compositional_urns training.
#
# From compositional_urns/:
#   mkdir -p logs
#   qsub -N cu_phase1 train_scc.sh 1
#   qsub -N cu_base   train_scc.sh baseline
#   qsub -N cu_phase2 -v RESUME_FROM=/path/to/checkpoint-XXXX train_scc.sh 2
#   qsub -N cu_mg_D64_s1 -v NUM_TASKS=64,SEED=1 train_scc.sh baseline
#   qsub -N cu_sharp -v G_CONCENTRATION=0.1,F_CONCENTRATION=0.1,N_LAYERS=4,MAX_STEPS=100000 \
#        -v RESUME_FROM=/path/to/checkpoint,RUN_TAG=comp-replay-sharp train_scc.sh 2
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

PHASE="${1:-1}"
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

EXTRA=()
if [[ -n "${NUM_TASKS:-}" ]]; then
  EXTRA+=(--num_tasks "${NUM_TASKS}")
fi
if [[ -n "${SEED:-}" ]]; then
  EXTRA+=(--seed "${SEED}")
fi
if [[ -n "${RUN_TAG:-}" ]]; then
  EXTRA+=(--run_tag "${RUN_TAG}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${G_CONCENTRATION:-}" ]]; then
  EXTRA+=(--g_concentration "${G_CONCENTRATION}")
fi
if [[ -n "${F_CONCENTRATION:-}" ]]; then
  EXTRA+=(--f_concentration "${F_CONCENTRATION}")
fi
if [[ -n "${N_LAYERS:-}" ]]; then
  EXTRA+=(--n_layers "${N_LAYERS}")
fi
if [[ -n "${N_HEADS:-}" ]]; then
  EXTRA+=(--n_heads "${N_HEADS}")
fi
if [[ -n "${D_MODEL:-}" ]]; then
  EXTRA+=(--d_model "${D_MODEL}")
fi
if [[ -n "${D_FF:-}" ]]; then
  EXTRA+=(--d_ff "${D_FF}")
fi
if [[ -n "${SAVE_EVERY:-}" ]]; then
  EXTRA+=(--save_every "${SAVE_EVERY}")
fi
if [[ "${PHASE}" == "2" ]]; then
  if [[ -z "${RESUME_FROM:-}" ]]; then
    # Auto-pick phase1_passed marker if present
    MARKER=$(ls -t "${CACHE_DIR}"/compositional_urns/runs/phase1-*/checkpoints/phase1_passed.json 2>/dev/null | head -1 || true)
    if [[ -n "${MARKER}" ]]; then
      RESUME_FROM=$(python -c "import json; print(json.load(open('${MARKER}'))['checkpoint'])")
    else
      echo "ERROR: phase 2 needs RESUME_FROM or a phase1_passed.json" >&2
      exit 1
    fi
  fi
  EXTRA+=(--resume_from "${RESUME_FROM}")
fi

echo "PHASE=${PHASE} RESUME_FROM=${RESUME_FROM:-} CACHE_DIR=${CACHE_DIR}"
python train.py --phase "${PHASE}" "${EXTRA[@]}"

echo "DONE phase=${PHASE}"
