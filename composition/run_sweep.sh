#!/bin/bash
# Sweep task diversity D for the compositional lookup-table task.
#
# Usage:
#   ./run_sweep.sh                       # default: full sweep, 1e5 steps each
#   ./run_sweep.sh --max_steps 50000     # extra args are forwarded to train.py
#
# Assumes the `transformer` conda env; override PYTHON to use another interpreter.
set -euo pipefail

PYTHON="${PYTHON:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for D in 4 8 16 32 64 128 256; do
    echo "=================== D=${D} ==================="
    "$PYTHON" "${SCRIPT_DIR}/train.py" --num_tasks "$D" "$@"
done
