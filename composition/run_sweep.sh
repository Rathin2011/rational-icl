#!/bin/bash
# Sweep task diversity D (matched to linear-regression main exp in this repo).
#
# Usage:
#   ./run_sweep.sh                       # full D sweep, 1e5 steps each
#   ./run_sweep.sh --max_steps 50000     # extra args forwarded to train.py
set -euo pipefail

PYTHON="${PYTHON:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same D grid as experiments/linear-regression/inverse-sqrt-full-exp
for D in 4 8 16 32 64 128 256 512 1024 2048 4096; do
    echo "=================== D=${D} ==================="
    "$PYTHON" "${SCRIPT_DIR}/train.py" --num_tasks "$D" "$@"
done
