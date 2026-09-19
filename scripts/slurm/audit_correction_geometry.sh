#!/bin/bash
#SBATCH --job-name=corr_geom
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond/%x-%j.out
#
# What the correction DOES to the coordinates, for each arm on the held-out
# panel. Answers what a maximum displacement cannot: how big the typical
# correction is, how much of it is a rigid pose change, whether the large
# corrections are the ones that help, and whether the movement damages backbone
# geometry.
#
# Run the candidate AND its matched BB-only control. If the two move the same
# way, the movement is a property of the receiving interface rather than of the
# side-chain information -- which is the distinction the experiment exists to
# make and is invisible in a single RMSD column.
#
#   R=/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond
#   ARMS="early_s_full=$R/early_s_full_119661/checkpoints/final.pt \
#         early_s_bb_only=$R/early_s_bb_only_119662/checkpoints/final.pt" \
#   sbatch scripts/slurm/audit_correction_geometry.sh
#
# Submit with SLURM_* cleared; see scripts/slurm/train_early_conditioner.sh.
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

ARMS="${ARMS:?set ARMS to a space-separated list of LABEL=CHECKPOINT}"
CONFIG="${CONFIG:-configs/couple_early_e1.yaml}"
STRUCTURES="${STRUCTURES:-$ROOT/configs/val_structures_afdb.txt}"
DONOR="${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}"
TAG="${TAG:-corr_geom}"
OUT="${OUT:-/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond/${TAG}_${SLURM_JOB_ID:-local}}"
CROP_SIZE="${CROP_SIZE:-512}"
N_SIGMA="${N_SIGMA:-4}"
MAX_TARGETS="${MAX_TARGETS:-64}"
SEED="${SEED:-0}"

mkdir -p "$OUT"
cd "$ROOT"
source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-local} out=${OUT}"
echo "arms=${ARMS}"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

CHECKPOINT_ARGS=""
for arm in $ARMS; do
    path="${arm#*=}"
    if [ ! -f "$path" ]; then echo "missing checkpoint: $path" >&2; exit 2; fi
    CHECKPOINT_ARGS="$CHECKPOINT_ARGS --checkpoint $arm"
done

python scripts/audit_correction_geometry.py \
    --config "$CONFIG" \
    --structures "$STRUCTURES" \
    --out "$OUT" \
    --pxdesign-donor "$DONOR" \
    ${CHECKPOINT_ARGS} \
    --crop-size "$CROP_SIZE" \
    --n-sigma "$N_SIGMA" \
    --max-targets "$MAX_TARGETS" \
    --seed "$SEED" \
    ${EXTRA_ARGS:-}

echo "correction geometry audit done -> $OUT"
