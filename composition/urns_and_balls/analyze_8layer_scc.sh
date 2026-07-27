#!/bin/bash -l
#$ -P buinlp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=12:00:00
#$ -j y
#$ -m beas
#$ -o logs/$JOB_ID_$JOB_NAME.log

set -euo pipefail
REPO=/projectnb/buinlp/rathin/rational-icl
cd "${REPO}/composition/urns_and_balls"
ARGS="${ANALYZE_ARGS:-}"
if [[ -z "${ARGS}" && "${ANALYZE_LOAD_SAVED:-0}" == "1" ]]; then
  ARGS="--load-saved"
fi
bash analyze_8layer.sh ${ARGS}
