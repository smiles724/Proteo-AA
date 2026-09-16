#!/bin/bash
#SBATCH --job-name=pxf_train_protenix
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_train_protenix/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_train_protenix/%x-%j.out
#
# Fine-tune FaMPNN's side-chain module on Protenix entries with the per-residue
# crystallographic supervision masks applied.
#
# The masks are composed into FaMPNN's own per-atom mask, never substituted for
# it: FaMPNN handles missing and ghost atoms per atom and knows nothing about
# occupancy or altloc, the per-residue mask handles exactly that, and both are
# wanted. See pxf/train/protenix.py.
#
# Data source defaults to the before-2021-09-30 weighted index, which is
# disjoint from the recentPDB eval split, so eval measures generalization.
#
#   INDEX=<training index> OUT=/hai/scratch/yfsun/... \
#       sbatch scripts/slurm/train_protenix.sh
#
# Set NO_MASK=1 for the ablation that trains on the per-atom mask alone -- that
# is the run that says what the crystallographic filter was worth.
#
# Submit with SLURM_* cleared, or the job inherits the submitting shell's
# allocation:
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       OUT=... sbatch scripts/slurm/train_protenix.sh
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

OUT="${OUT:?set OUT to the output directory}"
INDEX="${INDEX:-/hai/scratch/yfsun/protenix_data/indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz}"
MASK_ROOT="${MASK_ROOT:-/hai/scratch/yfsun/protenix_sidechain/out_fampnn_strictB}"
CONFIG="${CONFIG:-configs/train_protenix.yaml}"
INIT_WEIGHTS="${INIT_WEIGHTS:-0.0}"
TRAINABLE="${TRAINABLE:-scn_denoiser}"
MAX_STEPS="${MAX_STEPS:-}"
mkdir -p "$OUT" /hai/scratch/yfsun/proteo_aa_runs/pxf_train_protenix
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/Protenix:$ROOT/fampnn"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} out=${OUT}"
echo "index=${INDEX}"
echo "masks=${MASK_ROOT} trainable=${TRAINABLE} init=${INIT_WEIGHTS}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

MASK_ARG="--protenix-mask-root $MASK_ROOT"
if [ -n "${NO_MASK:-}" ]; then MASK_ARG="--no-supervision-mask"; fi
STEPS_ARG=""
if [ -n "$MAX_STEPS" ]; then STEPS_ARG="--max-steps $MAX_STEPS"; fi

# Resume this run if it was interrupted.
RESUME_ARG=""
LATEST=$(ls -1 "$OUT"/checkpoints/step*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "$LATEST" ]; then
    echo "resuming from $LATEST"
    RESUME_ARG="--resume $LATEST"
fi

python scripts/train.py \
    --protenix-index "$INDEX" \
    ${MASK_ARG} \
    --out "$OUT" \
    --config "$CONFIG" \
    --init-weights "$INIT_WEIGHTS" \
    --trainable "$TRAINABLE" \
    ${STEPS_ARG} \
    ${RESUME_ARG} \
    ${EXTRA_ARGS:-}

echo "done -> $OUT"
