#!/bin/bash -l
#
# Evaluate patching / d_rel on existing Phase-2 (and Phase-1) checkpoints.
#
#$ -P buinlp
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=02:00:00
#$ -l nvlink=FALSE
#$ -j y
#$ -o logs/$JOB_ID_$JOB_NAME.log

set -euo pipefail

REPO="${REPO:-/projectnb/buinlp/rathin/rational-icl}"
DIR="${REPO}/composition/compositional_urns"
cd "${DIR}"
mkdir -p logs

if [[ -z "${CACHE_DIR:-}" && -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC2046
  export $(grep -E '^CACHE_DIR=' "${REPO}/.env" | xargs)
  set +a
fi
export CACHE_DIR="${CACHE_DIR:-/projectnb/buinlp/rathin/cache}"

CONDA_ENV="${CONDA_ENV:-/projectnb/buinlp/jskyi/miniconda3/envs/transformer}"
module load miniconda
conda deactivate || true
# shellcheck disable=SC1091
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

P2="${CACHE_DIR}/compositional_urns/runs/phase2-D64-X20Z6Y20-2L4H128d-lr0.001-bs256-seed1-comp-replay/checkpoints"
P1="${CACHE_DIR}/compositional_urns/runs/phase1-D64-X20Z6Y20-2L4H128d-lr0.001-bs256-seed1-gf/checkpoints"
OUTDIR="${P2}/eval"
mkdir -p "${OUTDIR}"

# Phase-1 baseline + all Phase-2 ckpts that exist so far
CKPTS=(
  "${P1}/checkpoint-5000"
  "${P2}/checkpoint-1000"
  "${P2}/checkpoint-3000"
  "${P2}/checkpoint-10000"
  "${P2}/checkpoint-30000"
)

for ckpt in "${CKPTS[@]}"; do
  if [[ ! -d "${ckpt}" ]]; then
    echo "SKIP missing ${ckpt}"
    continue
  fi
  name=$(basename "$(dirname "${ckpt}")")_$(basename "${ckpt}")
  out="${OUTDIR}/${name}.json"
  echo "==== Evaluating ${ckpt} -> ${out} ===="
  python eval_checkpoint.py \
    --checkpoint "${ckpt}" \
    --n_eval 128 \
    --out_json "${out}"
done

echo "==== SUMMARY ===="
python - <<'PY'
import json, glob, os
outdir = os.environ["CACHE_DIR"] + "/compositional_urns/runs/phase2-D64-X20Z6Y20-2L4H128d-lr0.001-bs256-seed1-comp-replay/checkpoints/eval"
files = sorted(glob.glob(outdir + "/*.json"))
print(f"{'name':40s}  patch_ok  accept  z_dec  KL_r   KL_tp   KL_un   cgg_id  cgg_ood  ce_comp")
for f in files:
    m = json.load(open(f))
    pr = m.get("probe", {})
    print(
        f"{os.path.basename(f):40s}  "
        f"{str(bool(pr.get('patch_margin_ok'))):7s}  "
        f"{pr.get('trial_accept_rate', float('nan')):5.3f}  "
        f"{pr.get('z_decode_acc', float('nan')):5.3f}  "
        f"{pr.get('kl_route', float('nan')):5.3f}  "
        f"{pr.get('kl_transplant', float('nan')):5.3f}  "
        f"{pr.get('kl_unaffected', float('nan')):5.3f}  "
        f"{str(m.get('cgg_wins_id')):5s}  "
        f"{str(m.get('cgg_wins_ood')):5s}  "
        f"{m.get('ce_id_comp', float('nan')):6.3f}"
    )
PY

echo DONE
