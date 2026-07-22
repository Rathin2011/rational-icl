#!/bin/bash
# Submit the canonical Wurgaft Balls & Urns D sweep on SCC.
#
# Canonical slice: 8dims / 128 context / mlp×4 / D in {4..4096}.
#
# Usage (from composition/urns_and_balls/):
#   mkdir -p logs
#   bash run_sweep.sh
#
# Dry-run (print qsub commands only):
#   DRY_RUN=1 bash run_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

D_VALUES=(4 8 16 32 64 128 256 512 1024 2048 4096)
DRY_RUN="${DRY_RUN:-0}"

for D in "${D_VALUES[@]}"; do
  JOB_NAME="bu_D${D}"
  CMD=(qsub -N "${JOB_NAME}" train_scc.sh "${D}")
  echo "${CMD[*]}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${CMD[@]}"
  fi
done

echo "Submitted ${#D_VALUES[@]} jobs (DRY_RUN=${DRY_RUN})."
echo "Monitor with: qstat -u \$USER"
