#!/bin/bash
# Submit Phase-1 and baseline in parallel. Phase-2 after Phase-1 passes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

echo "Submitting Phase-1 (g+f gate)..."
qsub -N cu_phase1 train_scc.sh 1

echo "Submitting baseline (scratch comp-only)..."
qsub -N cu_baseline train_scc.sh baseline

echo "When Phase-1 passes, submit Phase-2 with:"
echo "  qsub -N cu_phase2 -v RESUME_FROM=<ckpt> train_scc.sh 2"
echo "Or rely on auto-detect of phase1_passed.json:"
echo "  qsub -N cu_phase2 train_scc.sh 2"
