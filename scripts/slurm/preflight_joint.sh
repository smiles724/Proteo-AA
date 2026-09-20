#!/bin/bash
#SBATCH --job-name=pxf_preflight_joint
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_preflight_joint/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_preflight_joint/%x-%j.out
#
# The gates the joint-refinement plan puts before any training job.
#
# Two things, both needing the real donor on a real GPU, neither answerable
# from the CPU tests:
#
#   1. The joint test suite, on a GPU. The graph moves tensors between devices
#      in several places -- the noise draws are deliberately made on the CPU and
#      moved so a seed means the same thing anywhere -- and a device mismatch in
#      that path is invisible on a CPU-only run by construction.
#   2. Memory, timing, and the auxiliary loss coefficients. L_BB, L_local and
#      L_place are normalized differently, so their values say nothing about
#      their relative pull; the coefficients are set from each term's gradient
#      AT THE BACKBONE, sampled across four backbone-noise bands.
#
# No optimizer runs and no checkpoint is written. This job produces the numbers
# a training run has to be configured with.
#
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       OUT=/hai/scratch/yfsun/proteo_aa_runs/pxf_preflight_joint/run1 \
#       sbatch scripts/slurm/preflight_joint.sh
#
# Submit with SLURM_* cleared as above, or the job silently inherits the
# submitting shell's allocation instead of the one requested here.
#
# h200 explicitly: the partition also has b200s, whose sm_100 this env's torch
# has no kernels for, and that only fails on the first real kernel launch.
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/joint/model.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo, or predates pxf/joint; set PXF_REPO" >&2
    exit 2
fi

OUT="${OUT:?set OUT to the output directory}"
STRUCTURES="${STRUCTURES:-/hai/scratch/yfsun/afdb_laproteina/cif_phase1}"
DONOR="${DONOR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt}"
N_STRUCTURES="${N_STRUCTURES:-4}"
MULTIPLIER="${MULTIPLIER:-8}"
RATIO="${RATIO:-0.1}"
SEED="${SEED:-0}"
mkdir -p "$OUT" /hai/scratch/yfsun/proteo_aa_runs/pxf_preflight_joint
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} out=${OUT}"
echo "structures=${STRUCTURES} n=${N_STRUCTURES} multiplier=${MULTIPLIER} ratio=${RATIO}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

echo "=== 1. the joint suite, on a GPU ==="
# Not -x: a device failure in one module should not hide the others.
python -m pytest tests/test_joint_data.py tests/test_joint_losses.py \
    tests/test_joint_gradients.py tests/test_joint_randomness.py \
    -q --no-header 2>&1 | tail -25

echo "=== 2. memory, timing and the loss coefficients ==="
python scripts/preflight_joint.py \
    --out "$OUT" \
    --structures "$STRUCTURES" \
    --donor "$DONOR" \
    --n-structures "$N_STRUCTURES" \
    --multiplier "$MULTIPLIER" \
    --ratio "$RATIO" \
    --seed "$SEED" \
    ${EXTRA_ARGS:-}

echo "done: $OUT/preflight.json"
