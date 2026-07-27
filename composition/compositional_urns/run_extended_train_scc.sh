#!/bin/bash
# Chain multiple `phase=2` training jobs (weights-only resume between links,
# same mechanism as the existing Phase-1->Phase-2 handoff) to reach a large
# cumulative step count within SCC's ~12h per-job walltime limit.
#
# Each link trains --steps_per_link NEW steps from the previous link's final
# checkpoint (Trainer step counter resets each link -- this is a weights-only
# warm start, not true optimizer-state resumption; LR warmup restarts each
# link too, a bounded cost relative to steps_per_link).
#
# Usage:
#   ./run_extended_train_scc.sh --initial_resume <phase1_or_phase2_checkpoint> \
#     --num_links 5 --steps_per_link 60000 \
#     [--num_tasks 64] [--seed 1] [--g_concentration 1.0] [--f_concentration 1.0] \
#     [--n_layers 2] [--n_heads 4] [--d_model 128] [--d_ff 512] \
#     [--save_every N] [--run_tag_prefix comp-replay-ext]
#
# Example: 5 links x 60k = 300k cumulative steps, sharpened task, 4 layers:
#   ./run_extended_train_scc.sh --initial_resume \
#     $CACHE_DIR/compositional_urns/runs/phase1-.../checkpoints/checkpoint-5000 \
#     --num_links 5 --steps_per_link 60000 \
#     --g_concentration 0.1 --f_concentration 0.1 --n_layers 4 --save_every 5000

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY="${PY:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer/bin/python}"

# Defaults match config.py's locked values.
NUM_TASKS=64
SEED=1
G_CONCENTRATION=1.0
F_CONCENTRATION=1.0
N_LAYERS=2
N_HEADS=4
D_MODEL=128
D_FF=512
SAVE_EVERY=""
RUN_TAG_PREFIX="comp-replay-ext"
INITIAL_RESUME=""
NUM_LINKS=""
STEPS_PER_LINK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --initial_resume) INITIAL_RESUME="$2"; shift 2 ;;
    --num_links) NUM_LINKS="$2"; shift 2 ;;
    --steps_per_link) STEPS_PER_LINK="$2"; shift 2 ;;
    --num_tasks) NUM_TASKS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --g_concentration) G_CONCENTRATION="$2"; shift 2 ;;
    --f_concentration) F_CONCENTRATION="$2"; shift 2 ;;
    --n_layers) N_LAYERS="$2"; shift 2 ;;
    --n_heads) N_HEADS="$2"; shift 2 ;;
    --d_model) D_MODEL="$2"; shift 2 ;;
    --d_ff) D_FF="$2"; shift 2 ;;
    --save_every) SAVE_EVERY="$2"; shift 2 ;;
    --run_tag_prefix) RUN_TAG_PREFIX="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${INITIAL_RESUME}" || -z "${NUM_LINKS}" || -z "${STEPS_PER_LINK}" ]]; then
  echo "usage: $0 --initial_resume <ckpt> --num_links N --steps_per_link N [options]" >&2
  exit 1
fi

checkpoints_dir_for_link() {
  local run_tag="$1"
  "${PY}" -c "
from config import CompUrnsConfig
cfg = CompUrnsConfig(
    num_tasks=${NUM_TASKS}, seed=${SEED},
    g_concentration=${G_CONCENTRATION}, f_concentration=${F_CONCENTRATION},
    n_layers=${N_LAYERS}, n_heads=${N_HEADS}, d_model=${D_MODEL}, d_ff=${D_FF},
    run_tag='${run_tag}', phase='2',
)
print(cfg.checkpoints_dir)
"
}

RESUME="${INITIAL_RESUME}"
PREV_JOB=""
for i in $(seq 1 "${NUM_LINKS}"); do
  RUN_TAG="${RUN_TAG_PREFIX}-link${i}"
  JOB_NAME="cu_ext_link${i}"

  V_ARGS="RESUME_FROM=${RESUME},RUN_TAG=${RUN_TAG},MAX_STEPS=${STEPS_PER_LINK}"
  V_ARGS+=",NUM_TASKS=${NUM_TASKS},SEED=${SEED}"
  V_ARGS+=",G_CONCENTRATION=${G_CONCENTRATION},F_CONCENTRATION=${F_CONCENTRATION}"
  V_ARGS+=",N_LAYERS=${N_LAYERS},N_HEADS=${N_HEADS},D_MODEL=${D_MODEL},D_FF=${D_FF}"
  if [[ -n "${SAVE_EVERY}" ]]; then
    V_ARGS+=",SAVE_EVERY=${SAVE_EVERY}"
  fi

  echo "Submitting link ${i}/${NUM_LINKS}: resume=${RESUME} run_tag=${RUN_TAG}"
  if [[ -z "${PREV_JOB}" ]]; then
    JOB_ID=$(qsub -N "${JOB_NAME}" -v "${V_ARGS}" -terse train_scc.sh 2)
  else
    JOB_ID=$(qsub -N "${JOB_NAME}" -hold_jid "${PREV_JOB}" -v "${V_ARGS}" -terse train_scc.sh 2)
  fi
  echo "  -> job ${JOB_ID}"
  PREV_JOB="${JOB_ID}"

  CKPT_DIR=$(checkpoints_dir_for_link "${RUN_TAG}")
  RESUME="${CKPT_DIR}/checkpoint-${STEPS_PER_LINK}"
done

echo "Chained ${NUM_LINKS} links (${STEPS_PER_LINK} steps each, ~$((NUM_LINKS * STEPS_PER_LINK)) cumulative)."
echo "Final checkpoint will be at: ${RESUME}"
