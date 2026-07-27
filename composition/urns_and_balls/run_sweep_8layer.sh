#!/bin/bash
# Submit the 8-layer depth-control sweep for Wurgaft's M/G phase diagram --
# same D grid, same everything as run_sweep.sh, only num_hidden_layers
# differs (1 -> 8), via the wurgaft-8layer experiment's yaml-configs.
#
# Usage (from composition/urns_and_balls/):
#   mkdir -p logs
#   bash run_sweep_8layer.sh
#
# Dry-run (print qsub commands only):
#   DRY_RUN=1 bash run_sweep_8layer.sh
#
# Pilot (just D=64, recommended first per the plan's staging step):
#   bash run_sweep_8layer.sh 64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

REPO="${REPO:-/projectnb/buinlp/rathin/rational-icl}"
YAML_DIR="${REPO}/experiments/categorical-sequence/wurgaft-8layer/yaml-configs"

D_VALUES=("$@")
if [[ ${#D_VALUES[@]} -eq 0 ]]; then
  D_VALUES=(4 8 16 32 64 128 256 512 1024 2048 4096)
fi
DRY_RUN="${DRY_RUN:-0}"

for D in "${D_VALUES[@]}"; do
  YAML="${YAML_DIR}/8dims-${D}tasks-128context-4expansionfactor-1seed.yaml"
  if [[ ! -f "${YAML}" ]]; then
    echo "ERROR: no yaml for D=${D}: ${YAML}" >&2
    exit 1
  fi
  JOB_NAME="bu8_D${D}"
  CMD=(qsub -N "${JOB_NAME}" train_scc.sh "${YAML}")
  echo "${CMD[*]}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${CMD[@]}"
  fi
done

echo "Submitted ${#D_VALUES[@]} jobs (DRY_RUN=${DRY_RUN})."
echo "Monitor with: qstat -u \$USER"
