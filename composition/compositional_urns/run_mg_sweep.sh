#!/bin/bash
# Submit the M-vs-G behavioral phase-diagram sweep: comp-only baseline
# training across a D x seed grid, with each eval job held until its
# training job finishes.
#
#   ./run_mg_sweep.sh          # full grid: D in {4,16,64,256,1024}, seeds 1,2
#   ./run_mg_sweep.sh 64       # pilot: just D=64, seeds 1,2
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

D_VALUES=("$@")
if [[ ${#D_VALUES[@]} -eq 0 ]]; then
  D_VALUES=(4 16 64 256 1024)
fi
SEEDS=(1 2)

for D in "${D_VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    TRAIN_JOB="cu_mg_D${D}_s${SEED}"
    echo "Submitting training: D=${D} seed=${SEED} (${TRAIN_JOB})"
    qsub -N "${TRAIN_JOB}" -v NUM_TASKS="${D}",SEED="${SEED}" train_scc.sh baseline
    echo "Submitting eval (held until training completes): D=${D} seed=${SEED}"
    qsub -N "cu_mg_eval_D${D}_s${SEED}" -hold_jid "${TRAIN_JOB}" \
      -v NUM_TASKS="${D}",SEED="${SEED}" eval_mg_scc.sh
  done
done
