#!/bin/bash
#SBATCH --job-name=early_cond
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond/%x-%j.out
#
# E1 / E2: the SC -> BB correction injected into the CONDITIONING rather than
# into the decoder input. One corrective event at a fixed noisy state, fixed
# native sequence, frozen PXDesign / FaMPNN / A_BS. Only the named conditioner
# trains.
#
#   ARM=early_s_full       E1 candidate: the existing readout -> s_single
#   ARM=early_s_bb_only    E1 trained BB/sequence-only control
#   ARM=early_s_generic    E1 trained sigma-only control
#   ARM=atom_sz_full       E2 candidate: predicted atoms -> s_single and z_pair
#   ARM=atom_sz_bb_only    E2 same-architecture BB-only control
#   ARM=atom_s_full        E2 single-only ablation
#
# Run every arm of an experiment or its candidate's number cannot be read: the
# claim is "predicted side chains beat a matched BB-only conditioner", not
# "beat the uncorrected proposal". The arms share a code path and a pool, so
# they differ in information, not in capacity or in data.
#
# The old late A_SB is never applied alongside these. The payload type enforces
# it -- a ConditioningFeedback is taken by the conditioning hook and declined by
# the decoder hook -- so there is no flag to forget.
#
# THE POOL IS THE LATE PILOT'S OWN, not a fresh one. `pilot_examples` derives
# the 512 (structure, sigma, seed) triples deterministically from the manifest,
# the seed, --pool-size and --sigmas-per-structure, so the defaults below
# reproduce exactly the 128-structure/512-event pool the late arms trained on.
# The upstream cache is that run's file: the frozen half does not depend on
# which arm reads it, `UpstreamCache.compatible` refuses a mismatch rather than
# warning, and reusing it is what makes the new arms comparable to the old ones
# rather than merely similar.
#
# h200 is requested explicitly: the yejin partition also has a b200 node whose
# sm_100 this env's torch has no kernels for, and the failure only surfaces on
# the first kernel launch (see pxf/device.py).
#
# Activation checkpointing stays OFF (the driver's default). Protenix recomputes
# the forward during backward and the injection hooks make the recomputation
# diverge; the conditioners are small and the backbone is frozen, so the memory
# is not needed and correctness is.
#
# Submit with SLURM_* cleared, or the job inherits the submitting shell's
# allocation:
#   for v in $(env | sed -n 's/^\(SLURM[A-Za-z_]*\)=.*/\1/p'); do
#       [ "$v" = SLURM_CONF ] && continue; unset "$v"; done
#   unset CUDA_VISIBLE_DEVICES
#   ARM=early_s_full sbatch scripts/slurm/train_early_conditioner.sh
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

ARM="${ARM:?set ARM to one of early_s_full early_s_bb_only early_s_generic atom_sz_full atom_sz_bb_only atom_s_full}"
case "$ARM" in
    early_s_*) DEFAULT_CONFIG=configs/couple_early_e1.yaml ;;
    atom_*)    DEFAULT_CONFIG=configs/couple_early_e2.yaml ;;
    *) echo "ARM=$ARM is not an early-conditioner arm" >&2; exit 2 ;;
esac
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
STRUCTURES="${STRUCTURES:-$ROOT/configs/phase1_structures_afdb.txt}"
VAL_STRUCTURES="${VAL_STRUCTURES:-$ROOT/configs/val_structures_afdb.txt}"
DONOR="${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}"
TAG="${TAG:-$ARM}"
RUNS="${RUNS:-/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond}"
OUT="${OUT:-$RUNS/${TAG}_${SLURM_JOB_ID:-local}}"
# 512 covers both AFDB manifests (longest 510 train, 485 val). The featurizer
# does not crop a design region, it refuses one larger than the crop.
CROP_SIZE="${CROP_SIZE:-512}"
POOL_SIZE="${POOL_SIZE:-512}"
SIGMAS_PER_STRUCTURE="${SIGMAS_PER_STRUCTURE:-4}"
VAL_POOL_SIZE="${VAL_POOL_SIZE:-32}"
CONDITIONING_CACHE="${CONDITIONING_CACHE:-24}"
MAX_STEPS="${MAX_STEPS:-}"
# The late pilot's cache, reused deliberately. See the header.
CACHE_UPSTREAM="${CACHE_UPSTREAM:-/hai/scratch/yfsun/proteo_aa_runs/pxf_sb_pilot/upstream_cache_v2_crop${CROP_SIZE}_pool${POOL_SIZE}.pt}"

mkdir -p "$OUT" "$RUNS"
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-local} arm=${ARM} config=${CONFIG} out=${OUT}"
echo "structures=${STRUCTURES} crop=${CROP_SIZE} pool=${POOL_SIZE} cache=${CACHE_UPSTREAM}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'arch', torch.cuda.get_arch_list()[-2:])"

# --resume continues THIS run if it was interrupted. There is no --init-from
# here: the conditioner is trained from its zero initialization, and inheriting
# a late adapter's weights would be inheriting a different architecture, which
# the trainer refuses on the recorded identity anyway.
RESUME_ARG=""
LATEST=$(ls -1 "$OUT"/checkpoints/step*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "$LATEST" ]; then
    echo "resuming interrupted run from $LATEST"
    RESUME_ARG="--resume $LATEST"
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
    --sb-arm "$ARM" \
    --pool-size "$POOL_SIZE" \
    --sigmas-per-structure "$SIGMAS_PER_STRUCTURE" \
    --val-pool-size "$VAL_POOL_SIZE" \
    --conditioning-cache "$CONDITIONING_CACHE" \
    --cache-upstream "$CACHE_UPSTREAM" \
    ${STEP_ARG} \
    ${RESUME_ARG} \
    ${EXTRA_ARGS:-}

echo "early conditioner ${ARM} done -> $OUT"
