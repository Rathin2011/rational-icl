#!/usr/bin/env bash
# Train one composition run on GPU and print error rates from logs.
#
# Usage (on a GPU node):
#   ./run_train.sh 64
#   ./run_train.sh 64 --max_steps 50000
#   ./run_train.sh 4 --max_steps 10000 --batch_size 256
#
# Extra args after D are forwarded to train.py.
set -euo pipefail

PYTHON="${PYTHON:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <D> [extra train.py args...]"
  echo "Example: $0 64 --max_steps 100000"
  exit 1
fi

D="$1"
shift

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'No GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"

echo "=================== Training D=${D} ==================="
"$PYTHON" "${SCRIPT_DIR}/train.py" --num_tasks "$D" "$@"

# Resolve run dir from cache + naming convention used by CompositionConfig
CACHE_DIR="${CACHE_DIR:-}"
if [[ -z "$CACHE_DIR" && -f "${SCRIPT_DIR}/../.env" ]]; then
  CACHE_DIR="$(grep -E '^CACHE_DIR=' "${SCRIPT_DIR}/../.env" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
fi
if [[ -z "$CACHE_DIR" ]]; then
  echo "Set CACHE_DIR or put it in ../.env to auto-print error rates."
  exit 0
fi

# Pick the newest matching run for this D
RUN_DIR="$(ls -dt "${CACHE_DIR}/composition/runs/D${D}-"* 2>/dev/null | head -1 || true)"
if [[ -z "$RUN_DIR" || ! -f "${RUN_DIR}/logs.csv" ]]; then
  echo "Could not find logs.csv under ${CACHE_DIR}/composition/runs/D${D}-*"
  exit 0
fi

echo "=================== Error rates ==================="
"$PYTHON" "${SCRIPT_DIR}/report_errors.py" "${RUN_DIR}/logs.csv"
