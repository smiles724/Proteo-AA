#!/bin/bash
#SBATCH --job-name=pxf_eval_joint
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_eval_joint/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_eval_joint/%x-%j.out
#
# Held-out scoring for the joint-refinement arms, backbone mode.
#
#   source /users/yfsun/marlowe_env.sh
#   PANEL=.../heldout_val.jsonl OUT=.../eval_joint/run1 \
#       sbatch scripts/slurm/marlowe/eval_joint_refinement.sh
#
# Scores R0 (the untrained donor) plus every checkpoint named in CHECKPOINTS.
# Backbone metrics only: packing safety is reported as incomplete because
# nothing about packing is measured, which is not the same as unharmed.
#
# EVAL_* names are namespaced away from the bare STRUCTURES / EXTRA_ARGS that
# marlowe_env.sh exports for other jobs -- inheriting either would fail on a
# path or a flag that looks plausible in the log. Same trap that killed 488735.
#
# DATA_ROOT is assigned BEFORE anything expands it. The arm launcher had these
# two lines the other way round and died under set -u on every submission; see
# commit 7b303b6.
#
# Do NOT clear the whole SLURM_* block: SLURM_CONF lives in that namespace on
# Marlowe and unsetting it breaks client config discovery. From inside an
# salloc, keep it:
#   env $(env | grep -o '^SLURM_[^=]*' | grep -v '^SLURM_CONF$' \
#         | sed 's/^/-u /' | tr '\n' ' ') PANEL=... OUT=... sbatch ...
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
if [ ! -f "$ROOT/scripts/eval_joint_refinement.py" ]; then
    echo "ROOT=$ROOT predates the joint evaluator; set PXF_REPO" >&2
    exit 2
fi

DATA_ROOT="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
OUT="${OUT:?set OUT to the output directory}"
PANEL="${PANEL:-$DATA_ROOT/runs/eval_joint/heldout_val.jsonl}"
DONOR="${DONOR:-$DATA_ROOT/component_donors/pxdesign_v0.1.0.pt}"
ARM_ROOT="${ARM_ROOT:-$DATA_ROOT/runs/pxf_joint}"
CHECKPOINTS="${CHECKPOINTS:-B0 B1 B2 BF BS}"
WEIGHTS="${WEIGHTS:-ema}"
CANDIDATE="${CANDIDATE:-B2}"
BASELINE="${BASELINE:-B0}"
EVAL_SIGMAS="${EVAL_SIGMAS:-0.105 0.314 0.847 1.939}"
REPLICATES="${REPLICATES:-1}"
EVAL_SEED="${EVAL_SEED:-17}"
mkdir -p "$OUT" "$DATA_ROOT/runs/logs/pxf_eval_joint"
cd "$ROOT"

PXF_PYTHON_ENV="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}"
export PATH="$PXF_PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
# Its sibling, and the wrappers were all missing it: pxf.eval.canonical
# resolves side-chain metrics separately and falls back to a hardcoded HAI
# path. Sourcing marlowe_env.sh first supplies it, but a wrapper that
# defaults PROTEOAA_ROOT should not then depend on the submitting shell.
export PROTEOAA_METRICS_ROOT="${PROTEOAA_METRICS_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$DATA_ROOT/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-$DATA_ROOT/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

ARGS=(--panel "$PANEL" --donor "$DONOR" --include-donor
      --weights "$WEIGHTS" --mode backbone
      --sigmas $EVAL_SIGMAS --replicates "$REPLICATES"
      --seed "$EVAL_SEED" --candidate "$CANDIDATE" --baseline "$BASELINE"
      --out "$OUT")
for ARM in $CHECKPOINTS; do
    CKPT="$ARM_ROOT/$ARM/checkpoints/final.pt"
    if [ ! -f "$CKPT" ]; then
        echo "missing checkpoint for $ARM: $CKPT" >&2
        exit 2
    fi
    # --expect-arm is an independent label check: the loader also reads the
    # arm out of the checkpoint and refuses a disagreement.
    ARGS+=(--checkpoint "$ARM=$CKPT" --expect-arm "$ARM=$ARM")
done

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} out=${OUT}"
echo "panel=${PANEL} weights=${WEIGHTS} candidate=${CANDIDATE} baseline=${BASELINE}"
echo "sigmas=${EVAL_SIGMAS} replicates=${REPLICATES} seed=${EVAL_SEED}"
echo "arms=${CHECKPOINTS}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

python scripts/eval_joint_refinement.py "${ARGS[@]}" ${EVAL_EXTRA_ARGS:-}
echo "done: $OUT/report.md"
