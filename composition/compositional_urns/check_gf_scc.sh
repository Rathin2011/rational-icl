#!/bin/bash -l
#
# Short GPU check that Phase-1 g/f are near Bayes before Phase 2.
#
#$ -P buinlp
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=00:30:00
#$ -l nvlink=FALSE
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

CKPT="${1:-}"
if [[ -z "${CKPT}" ]]; then
  MARKER=$(ls -t "${CACHE_DIR}"/compositional_urns/runs/phase1-*/checkpoints/phase1_passed.json 2>/dev/null | head -1)
  CKPT=$(python -c "import json; print(json.load(open('${MARKER}'))['checkpoint'])")
fi

echo "Checking g/f learning at: ${CKPT}"
python check_gf_learn.py --checkpoint "${CKPT}" --n_eval 256
echo DONE
