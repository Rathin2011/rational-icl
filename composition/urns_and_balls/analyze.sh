#!/bin/bash
# Run Wurgaft relative-distance analysis + figures for wurgaft-baseline.
#
# Important: PYTHONPATH must be <repo>/src (so `code` is the package), not the
# repo root — analysis_utils does `from code.models import ...`.
#
# Usage:
#   bash composition/urns_and_balls/analyze.sh
#   bash composition/urns_and_balls/analyze.sh --no-figs   # skip figure gen
#   bash composition/urns_and_balls/analyze.sh --no-bms    # skip BMS fit
#   bash composition/urns_and_balls/analyze.sh --load-saved  # reuse cached evals/BMS

set -euo pipefail

REPO="${REPO:-/projectnb/buinlp/rathin/rational-icl}"
cd "${REPO}"

if [[ -z "${CACHE_DIR:-}" && -f "${REPO}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  # shellcheck disable=SC2046
  export $(grep -E '^CACHE_DIR=' "${REPO}/.env" | xargs)
  set +a
fi
export CACHE_DIR="${CACHE_DIR:-/projectnb/buinlp/rathin/cache}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONDA_ENV="${CONDA_ENV:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  module load miniconda 2>/dev/null || true
  # shellcheck disable=SC1091
  source /projectnb/buinlp/jskyi/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

MAKE_FIGS=1
FIT_BMS=1
LOAD_SAVED=0
if [[ "${ANALYZE_LOAD_SAVED:-0}" == "1" ]]; then
  LOAD_SAVED=1
fi
for arg in "$@"; do
  case "$arg" in
    --no-figs) MAKE_FIGS=0 ;;
    --no-bms) FIT_BMS=0 ;;
    --load-saved) LOAD_SAVED=1 ;;
  esac
done

echo "CACHE_DIR=${CACHE_DIR}"
echo "MAKE_FIGS=${MAKE_FIGS} FIT_BMS=${FIT_BMS} LOAD_SAVED=${LOAD_SAVED}"

# Write a temp runner (avoids stdin -c frame issues with dotenv).
RUNNER="$(mktemp /tmp/wurgaft_analyze_XXXXXX.py)"
trap 'rm -f "${RUNNER}"' EXIT
cat > "${RUNNER}" <<PY
import os

os.environ.setdefault("CACHE_DIR", "${CACHE_DIR}")

from code.analysis_pipeline import AnalysisPipeline

pipe = AnalysisPipeline(
    "categorical-sequence",
    "wurgaft-baseline",
    num_eval_sequences=500,
)
pipe.main(
    load_saved_evaluation=${LOAD_SAVED},
    fit_Bayesian_model=${FIT_BMS},
    load_saved_Bayesian_model_params=${LOAD_SAVED},
    make_figs=${MAKE_FIGS},
    increase_generalized_code_complexity=False,
)
print("Analysis complete.")
PY

python "${RUNNER}"
