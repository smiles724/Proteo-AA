#!/bin/bash
#SBATCH --job-name=pxf_eval_couple
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_eval_couple/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_eval_couple/%x-%j.out
#
# Did the coupling actually improve packing? Held-out AFDB structures, the
# coupled cycle against the same cycle with the adapters switched off.
#
# This is the measurement train_couple.sh does not make. That job reports L_SC
# on the data it is fitting; this one reports packing metrics on the val split,
# which the phase 1 manifest does not touch (verified disjoint: 2,000 train ids
# vs 256 val ids, intersection 0).
#
#   CHECKPOINT=/hai/scratch/yfsun/proteo_aa_runs/pxf_couple/phase1_<jobid>/checkpoints/final.pt \
#       OUT=/hai/scratch/yfsun/proteo_aa_runs/pxf_eval_couple/phase1 \
#       sbatch scripts/slurm/eval_couple.sh
#
# Omit CHECKPOINT to evaluate the untrained adapters. That is the pipeline's own
# sanity check rather than a wasted run: the adapters are zero-initialized, so
# the two arms must come out bit-identical, and any difference is a bug in the
# seeding or the arm switching rather than a result.
#
# The val manifest reaches 485 residues, so CROP_SIZE cannot go below that -- a
# crop breaks correspondence with the atom37 side-chain targets and the script
# refuses. Note that lowering it does NOT reduce memory: the featurizer emits the
# structure's true token count, so cost follows the structure, not the crop.
#
# h200 is requested explicitly: the yejin partition also has a b200 whose sm_100
# this env's torch has no kernels for, and it only fails on the first launch.
#
# Submit with SLURM_* cleared, or the job inherits the submitting shell's
# allocation:
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       CHECKPOINT=... OUT=... sbatch scripts/slurm/eval_couple.sh
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
# denoised: PXDesign proposal -> FaMPNN, coupled vs uncoupled arms.
# native:   deposited backbone -> FaMPNN. No donor, no sigma, no adapters.
MODE="${MODE:-denoised}"
CHECKPOINT="${CHECKPOINT:-}"
STRUCTURES="${STRUCTURES:-$ROOT/configs/val_structures_afdb.txt}"
CONFIG="${CONFIG:-$ROOT/configs/couple_phase1.yaml}"
DONOR="${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}"
CROP_SIZE="${CROP_SIZE:-512}"
PACK_STEPS="${PACK_STEPS:-50}"
N_SIGMA="${N_SIGMA:-5}"
MAX_TARGETS="${MAX_TARGETS:-200}"
FAMPNN_WEIGHTS="${FAMPNN_WEIGHTS:-0.0}"
SEED="${SEED:-0}"
mkdir -p "$OUT" /hai/scratch/yfsun/proteo_aa_runs/pxf_eval_couple
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
# Protenix finds its CCD cache here rather than downloading it.
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} mode=${MODE} out=${OUT}"
echo "checkpoint=${CHECKPOINT:-<untrained adapters>} structures=${STRUCTURES}"
echo "crop=${CROP_SIZE} pack_steps=${PACK_STEPS} n_sigma=${N_SIGMA} targets=${MAX_TARGETS}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

if [ "$MODE" = "native" ]; then
    # No backbone is generated, so the donor, the crop and the sigma sweep have
    # nothing to act on; passing them would only imply they were used.
    python scripts/eval_couple.py \
        --mode native \
        --structures "$STRUCTURES" \
        --out "$OUT" \
        --pack-steps "$PACK_STEPS" \
        --max-targets "$MAX_TARGETS" \
        --fampnn-weights "$FAMPNN_WEIGHTS" \
        --seed "$SEED" \
        ${EXTRA_ARGS:-}
else
    CKPT_ARG=""
    if [ -n "$CHECKPOINT" ]; then CKPT_ARG="--checkpoint $CHECKPOINT"; fi
    python scripts/eval_couple.py \
        --mode denoised \
        --structures "$STRUCTURES" \
        --out "$OUT" \
        --config "$CONFIG" \
        --pxdesign-donor "$DONOR" \
        ${CKPT_ARG} \
        --crop-size "$CROP_SIZE" \
        --pack-steps "$PACK_STEPS" \
        --n-sigma "$N_SIGMA" \
        --max-targets "$MAX_TARGETS" \
        --fampnn-weights "$FAMPNN_WEIGHTS" \
        --seed "$SEED" \
        ${EXTRA_ARGS:-}
fi

echo "done -> $OUT/couple_metrics.json"
