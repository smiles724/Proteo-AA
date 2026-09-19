#!/bin/bash
#SBATCH --job-name=sc-env-eval
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/evaluation/%x-%A_%a.out
#SBATCH --error=logs/evaluation/%x-%A_%a.err
#
# Offline evaluation of the SC->BB feedback arms, on the two measures that can
# actually settle the question. The training logs cannot: the arms have
# different trainable sets, so `like_s3` can lower loss_bb and loss_sc by
# changing the backbone and the packer while `fb_only` cannot -- a lower
# training loss there carries no information about the feedback.
#
#   0  fb_only    geometry probe
#   1  like_s3    geometry probe
#   2  official    geometry probe, the reference re-measured in the SAME job
#                  array so the comparison cannot inherit a stale number
#
# 400 sampling steps and the same probe for all three. That matters: the same
# official weights score 8.04% under this probe at 400 steps and 88% at 20,
# and 0.00% under the OTHER probe (which needs FaMPNN and is unreachable here).
# Mixing probes or step counts is how "we 60% vs official 0%" got quoted as a
# comparison when it never was one.
#
# AlphaProteo-10 designability is a separate launcher
# (submit_alphaproteo_generation.sh -> submit_alphaproteo_scoring.sh) because it
# is generation + AF2 scoring over 3280 designs, not a single GPU hour.
set -euo pipefail

REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix"
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/hai/scratch/shenjm/triton_cache}
PYBIN=${PYBIN:-/hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python}
cd "$REPO"
mkdir -p logs/evaluation

RUN=${RUN:-/hai/scratch/shenjm/proteo_aa_runs/sc_env_feedback/119143}
STEP=${STEP:-final}
OUTROOT=${OUTROOT:-/hai/scratch/shenjm/proteo_aa_runs/sc_env_feedback_eval}
OFFICIAL=${OFFICIAL:-/hai/scratch/shenjm/pxdesign_official/pxdesign_v0.1.0.pt}

pick() {   # newest checkpoint for an arm, or the pinned STEP
  local d="$RUN/$1/checkpoints"
  if [[ "$STEP" == final ]]; then
    ls -t "$d"/step*.pt 2>/dev/null | head -1
  else
    echo "$d/step${STEP}.pt"
  fi
}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) LABEL=fb_only;  CKPT=$(pick fb_only) ;;
  1) LABEL=like_s3;  CKPT=$(pick like_s3) ;;
  2) LABEL=official; CKPT="$OFFICIAL" ;;
  *) echo "unexpected array index" >&2; exit 2 ;;
esac
[[ -f "$CKPT" ]] || { echo "no checkpoint for $LABEL at '$CKPT'" >&2; exit 2; }

OUT="$OUTROOT/$LABEL"
mkdir -p "$OUT"
echo "label=$LABEL ckpt=$CKPT out=$OUT commit=$(git rev-parse --short HEAD)"
"$PYBIN" -c "
import torch,sys
ck=torch.load('$CKPT',map_location='cpu',weights_only=False,mmap=True)
sd=ck.get('model',ck); n=sum(1 for k in sd if k.startswith('sc_env_feedback.'))
print(f'step={ck.get(\"step\")} sc_env_feedback tensors={n}')
"

exec "$PYBIN" scripts/evaluation/probe_monomer_backbone_geometry.py \
  --checkpoint "$CKPT" --output-dir "$OUT" \
  --data-root "$PROTENIX_ROOT_DIR" \
  --min-n-token 80 --max-n-token 200 --limit-index "${N_SAMPLES:-12}" \
  --crop-size 200 --n-step "${N_STEP:-400}" --device cuda "$@"
