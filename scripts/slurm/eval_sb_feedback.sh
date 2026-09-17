#!/bin/bash
#SBATCH --job-name=sb_eval
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_eval/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_eval/%x-%j.out
#
# The exit decision for one SC -> BB corrective event: bb0 against bb1, with the
# wiring check, the trained controls, the matched-conformation control and the
# comparable-cost sampler baseline all scored on the same held-out panel.
#
# ARMS is a space-separated list of LABEL=CHECKPOINT. Give all three pilot arms
# or the full arm's number cannot be read: beating bb0 shows a correction
# happened, and only beating the trained bb_only and generic arms licenses a
# side-chain-specific claim.
#
#   ARMS="full=$R/full_118215/checkpoints/final.pt \
#         bb_only=$R/bb_only_118216/checkpoints/final.pt \
#         generic=$R/generic_118217/checkpoints/final.pt" \
#   sbatch scripts/slurm/eval_sb_feedback.sh
#
# Submit with SLURM_* cleared; see scripts/slurm/train_sb_pilot.sh.
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

ARMS="${ARMS:?set ARMS to a space-separated list of LABEL=CHECKPOINT}"
CONFIG="${CONFIG:-configs/couple_phase2_pilot.yaml}"
# Held out from training by construction: the pilot pool is drawn from the train
# manifest and this is the val one, two disjoint la-proteina splits.
STRUCTURES="${STRUCTURES:-$ROOT/configs/val_structures_afdb.txt}"
DONOR="${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}"
TAG="${TAG:-sb_exit}"
OUT="${OUT:-/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_eval/${TAG}_${SLURM_JOB_ID:-local}}"
CROP_SIZE="${CROP_SIZE:-512}"
PACK_STEPS="${PACK_STEPS:-50}"
N_SIGMA="${N_SIGMA:-4}"
MAX_TARGETS="${MAX_TARGETS:-64}"
PERTURB_DEGREES="${PERTURB_DEGREES:-60}"
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
    label="${arm%%=*}"; path="${arm#*=}"
    if [ ! -f "$path" ]; then echo "missing checkpoint for $label: $path" >&2; exit 2; fi
    CHECKPOINT_ARGS="$CHECKPOINT_ARGS --checkpoint $arm"
done

python scripts/eval_sb_feedback.py \
    --config "$CONFIG" \
    --structures "$STRUCTURES" \
    --out "$OUT" \
    --pxdesign-donor "$DONOR" \
    ${CHECKPOINT_ARGS} \
    --crop-size "$CROP_SIZE" \
    --pack-steps "$PACK_STEPS" \
    --n-sigma "$N_SIGMA" \
    --max-targets "$MAX_TARGETS" \
    --perturb-degrees "$PERTURB_DEGREES" \
    --seed "$SEED" \
    ${EXTRA_ARGS:-}

echo "sb exit evaluation done -> $OUT"
