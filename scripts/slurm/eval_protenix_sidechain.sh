#!/bin/bash
#SBATCH --job-name=pxf_eval_protenix
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_eval_protenix/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_eval_protenix/%x-%j.out
#
# Side-chain packing metrics on Protenix's recentPDB low-homology eval split.
# 1,818 entries released 2022-05-04 .. 2023-01-11; the training index is cut at
# 2021-09-30 and the intersection is empty, so this measures generalization
# rather than recall. 1,642 have a supervision mask and are scored.
#
# Run it twice -- once on the released weights, once on the fine-tune -- then
# compare. The failure mode being watched for is a fine-tune that degrades
# packing, so the comparison is the point, not either number alone.
#
#   WEIGHTS=0.0 OUT=.../eval_before sbatch scripts/slurm/eval_protenix_sidechain.sh
#   CHECKPOINT=runs/ft/checkpoints/final.pt OUT=.../eval_after \
#       sbatch scripts/slurm/eval_protenix_sidechain.sh
#   python scripts/eval_protenix_sidechain.py --compare .../eval_before .../eval_after
#
# h200 is requested explicitly: the yejin partition also has a b200 whose sm_100
# this env's torch has no kernels for, and it only fails on the first launch.
#
# Submit with SLURM_* cleared, or the job inherits the submitting shell's
# allocation:
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       OUT=/hai/scratch/yfsun/... sbatch scripts/slurm/eval_protenix_sidechain.sh
set -euo pipefail

# sbatch copies this script to /var/lib/slurm/scripts, so BASH_SOURCE does not
# point at the repo under SLURM. SLURM_SUBMIT_DIR is the submission cwd.
if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

OUT="${OUT:?set OUT to the output directory}"
WEIGHTS="${WEIGHTS:-0.0}"
CHECKPOINT="${CHECKPOINT:-}"
MASK_ROOT="${MASK_ROOT:-/hai/scratch/yfsun/protenix_sidechain/out_eval_fampnn_strictB}"
NUM_STEPS="${NUM_STEPS:-50}"
MAX_TARGETS="${MAX_TARGETS:-0}"
LABEL="${LABEL:-recentPDB_low_homology}"
mkdir -p "$OUT" /hai/scratch/yfsun/proteo_aa_runs/pxf_eval_protenix
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/Protenix:$ROOT/fampnn"
# Protenix finds its CCD cache here rather than downloading it.
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} out=${OUT}"
echo "weights=${WEIGHTS} checkpoint=${CHECKPOINT:-<released>} masks=${MASK_ROOT}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

CKPT_ARG=""
if [ -n "$CHECKPOINT" ]; then CKPT_ARG="--checkpoint $CHECKPOINT"; fi

python scripts/eval_protenix_sidechain.py \
    --out "$OUT" \
    --label "$LABEL" \
    --weights "$WEIGHTS" \
    ${CKPT_ARG} \
    --mask-root "$MASK_ROOT" \
    --num-steps "$NUM_STEPS" \
    --max-targets "$MAX_TARGETS" \
    ${EXTRA_ARGS:-}

echo "done -> $OUT/sidechain_metrics.json"
