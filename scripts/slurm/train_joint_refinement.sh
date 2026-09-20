#!/bin/bash
#SBATCH --job-name=pxf_joint
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_joint/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_joint/%x-%j.out
#
# One arm of the side-chain-supervised backbone refinement experiment.
#
#   ARM=B0 OUT=.../B0 sbatch --job-name=pxf_joint_B0 scripts/slurm/train_joint_refinement.sh
#   ARM=B1 OUT=.../B1 PREFLIGHT=.../preflight.json \
#       sbatch --job-name=pxf_joint_B1 scripts/slurm/train_joint_refinement.sh
#
# Every arm shares the donor, the trainable scope, the example order and the
# noise draws; the arm decides only which losses reach the backbone. Run B0
# first -- it is the baseline every other arm is measured against -- and B1/B2
# only once a preflight has produced their coefficients. The entry point
# refuses an auxiliary arm whose coefficient is zero rather than running it as
# an expensive B0.
#
# BS is the compute-matched control and needs no coefficient: it runs the
# side-chain branch with the backbone detached at both entrances and discards
# the gradient. A short BS run against B0 is the cheapest check that the arms
# are paired before a full budget is spent.
#
# Submit with SLURM_* cleared, or the job inherits the submitting shell's
# allocation:
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       ARM=B0 OUT=... sbatch scripts/slurm/train_joint_refinement.sh
#
# h200 explicitly: the partition also has b200s, whose sm_100 this env's torch
# has no kernels for, and that only fails on the first real kernel launch.
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/joint/trainer.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo, or predates pxf/joint; set PXF_REPO" >&2
    exit 2
fi

ARM="${ARM:?set ARM to one of B0 B1 B2 BF BS}"
OUT="${OUT:?set OUT to the output directory}"
CONFIG="${CONFIG:-$ROOT/configs/joint_refinement/$ARM.yaml}"
PREFLIGHT="${PREFLIGHT:-}"
STRUCTURES="${JOINT_STRUCTURES:-}"
DONOR="${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}"
MAX_STEPS="${MAX_STEPS:-}"
SEED="${SEED:-0}"
RESUME="${RESUME:-}"
mkdir -p "$OUT" /hai/scratch/yfsun/proteo_aa_runs/pxf_joint
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} arm=${ARM} out=${OUT}"
echo "config=${CONFIG} preflight=${PREFLIGHT:-<none>} seed=${SEED}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

ARGS=(--arm "$ARM" --out "$OUT" --config "$CONFIG" --donor "$DONOR" --seed "$SEED")
[ -n "$PREFLIGHT" ] && ARGS+=(--preflight "$PREFLIGHT")
[ -n "$STRUCTURES" ] && ARGS+=(--structures "$STRUCTURES")
[ -n "$MAX_STEPS" ] && ARGS+=(--max-steps "$MAX_STEPS")
[ -n "$RESUME" ] && ARGS+=(--resume "$RESUME")

python scripts/train_joint_refinement.py "${ARGS[@]}" ${JOINT_EXTRA_ARGS:-}
echo "done: $OUT"
