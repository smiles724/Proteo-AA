#!/bin/bash
#SBATCH --job-name=sb_pilot
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_pilot/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_pilot/%x-%j.out
#
# The SC -> BB pilot: one corrective event at a fixed noisy state, fixed native
# sequence, frozen PXDesign / FaMPNN / A_BS. Only the SC readout and A_SB train.
#
#   VARIANT=full      the candidate: predicted side chains in the readout
#   VARIANT=bb_only   trained BB/sequence-only control
#   VARIANT=generic   trained sigma-only control
#
# The three differ ONLY in which feature groups z carries. Same receiving hook,
# same pool, same optimizer budget, same parameter count -- so a difference
# between them is information and not capacity. Run all three or the full arm's
# number cannot be read.
#
# h200 is requested explicitly: the yejin partition also has a b200 node whose
# sm_100 this env's torch has no kernels for, and the failure only surfaces on
# the first kernel launch (see pxf/device.py).
#
# Activation checkpointing stays OFF (the driver's default): Protenix recomputes
# the forward during backward and the feedback-injection hook makes the
# recomputation diverge (CheckpointError: a different number of tensors was
# saved). The adapters are small and the backbone is frozen, so the memory is
# not needed.
#
# Submit with SLURM_* cleared, or the job inherits the submitting shell's
# allocation:
#   for v in $(env | sed -n 's/^\(SLURM[A-Za-z_]*\)=.*/\1/p'); do
#       [ "$v" = SLURM_CONF ] && continue; unset "$v"; done
#   unset CUDA_VISIBLE_DEVICES
#   VARIANT=full sbatch scripts/slurm/train_sb_pilot.sh
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

VARIANT="${VARIANT:?set VARIANT to full, bb_only or generic}"
CONFIG="${CONFIG:-configs/couple_phase2_pilot.yaml}"
STRUCTURES="${STRUCTURES:-$ROOT/configs/phase1_structures_afdb.txt}"
VAL_STRUCTURES="${VAL_STRUCTURES:-$ROOT/configs/val_structures_afdb.txt}"
DONOR="${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}"
TAG="${TAG:-$VARIANT}"
RUNS="${RUNS:-/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_pilot}"
OUT="${OUT:-$RUNS/${TAG}_${SLURM_JOB_ID:-local}}"
# Crop is safe to set below the longest structure here, unlike phase 1: the
# pilot's backbone target and its noisy state both come from the SAME
# featurization, so a crop is self-consistent. Phase 1 refuses a crop because it
# reads side-chain targets from an independent parse of the file, where a crop
# breaks the residue correspondence.
CROP_SIZE="${CROP_SIZE:-256}"
# A FIXED pool with known backbone targets, so every arm trains on the same
# examples and the frozen half is paid for once per example, not per step.
POOL_SIZE="${POOL_SIZE:-512}"
SIGMAS_PER_STRUCTURE="${SIGMAS_PER_STRUCTURE:-4}"
VAL_POOL_SIZE="${VAL_POOL_SIZE:-32}"
CONDITIONING_CACHE="${CONDITIONING_CACHE:-16}"
MAX_STEPS="${MAX_STEPS:-}"
# The upstream cache is shared across variants on purpose: the frozen half does
# not depend on which A_SB arm is training, and its identity records the donors,
# the packing length and the BB->SC policy, so a mismatched cache is refused
# rather than silently reused.
CACHE_UPSTREAM="${CACHE_UPSTREAM:-$RUNS/upstream_cache_crop${CROP_SIZE}_pool${POOL_SIZE}.pt}"
INIT_FROM_PHASE1="${INIT_FROM_PHASE1:-}"

mkdir -p "$OUT" "$RUNS"
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-local} variant=${VARIANT} out=${OUT}"
echo "structures=${STRUCTURES} crop=${CROP_SIZE} pool=${POOL_SIZE}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'arch', torch.cuda.get_arch_list()[-2:])"

# --resume continues THIS run if it was interrupted. --init-from-phase1 is a
# different thing (weights-only start from a previous phase) and the script
# refuses to accept both.
RESUME_ARG=""
LATEST=$(ls -1 "$OUT"/checkpoints/step*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "$LATEST" ]; then
    echo "resuming interrupted run from $LATEST"
    RESUME_ARG="--resume $LATEST"
elif [ -n "$INIT_FROM_PHASE1" ]; then
    echo "initializing A_BS from $INIT_FROM_PHASE1 (weights only, step 0)"
    RESUME_ARG="--init-from-phase1 $INIT_FROM_PHASE1"
fi

STEP_ARG=""
if [ -n "$MAX_STEPS" ]; then STEP_ARG="--max-steps $MAX_STEPS"; fi

python scripts/train_couple.py \
    --config "$CONFIG" \
    --structures "$STRUCTURES" \
    --val-structures "$VAL_STRUCTURES" \
    --out "$OUT" \
    --backbone pxdesign \
    --pxdesign-donor "$DONOR" \
    --crop-size "$CROP_SIZE" \
    --sb-variant "$VARIANT" \
    --pool-size "$POOL_SIZE" \
    --sigmas-per-structure "$SIGMAS_PER_STRUCTURE" \
    --val-pool-size "$VAL_POOL_SIZE" \
    --conditioning-cache "$CONDITIONING_CACHE" \
    --cache-upstream "$CACHE_UPSTREAM" \
    ${STEP_ARG} \
    ${RESUME_ARG} \
    ${EXTRA_ARGS:-}

echo "sb pilot ${VARIANT} done -> $OUT"
