#!/bin/bash
#SBATCH --job-name=pxf_preflight_joint
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_preflight_joint/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_preflight_joint/%x-%j.out
#
# MARLOWE COPY of scripts/slurm/preflight_joint.sh (HAI job 120170). The HAI
# original is left untouched -- that repo is shared and in-place edits break it.
#
# Two gates the joint-refinement plan puts before any training job:
#   1. the joint test suite on a GPU -- the graph draws noise on the CPU and
#      moves it, so a device mismatch there cannot fail on a CPU-only run;
#   2. memory, timing, and the auxiliary loss coefficients, set from each
#      term's gradient AT THE BACKBONE across four backbone-noise bands.
# No optimizer runs and no checkpoint is written.
#
# Differences from the HAI original, all forced by this cluster:
#   header     yejin/yejin/gpu:h200:1 -> batch / marlowe-m000137-pm06 /
#              --qos=medium / -G 1. Marlowe's gres is UNTYPED ("gpu:8(S:0-1)"),
#              so "gpu:h200:1" is rejected outright; all nodes are H100 80GB
#              (sm_90), which this torch has kernels for.
#   logs       /hai/scratch/... does not exist here and SBATCH paths are static.
#   python     no conda; a relocated conda prefix with no bin/activate,
#              activated by PATH (PXF_PYTHON_ENV).
#   PYTHONPATH appended, not overwritten -- marlowe_env.sh already put the mask
#              dir and afdb-laproteina/src there and they must survive.
#   DONOR / STRUCTURES / PROTENIX_*_DIR defaults repointed off /hai.
#
#   EXTRA_ARGS -> PREFLIGHT_EXTRA_ARGS, and STRUCTURES -> PREFLIGHT_STRUCTURES.
#              Both are load-bearing. marlowe_env.sh exports
#              EXTRA_ARGS="--mmcif-dir ..." for the protenix side-chain eval and
#              preflight_joint.py has no such flag, so argparse would exit 2 --
#              the same trap that killed job 488735. It also exports
#              STRUCTURES=<a manifest FILE> for the training job, while this
#              script wants a DIRECTORY of CIFs; inheriting it would fail on a
#              path that looks plausible in the log.
#
# Usage, from a login shell:
#   source /users/yfsun/marlowe_env.sh
#   OUT=/scratch/m000137-pm06/Proteo-AA/pxf/runs/pxf_preflight_joint/run1 \
#       sbatch scripts/slurm/marlowe/preflight_joint.sh
#
# Do NOT clear the whole SLURM_* block the way the HAI header suggests: on
# Marlowe SLURM_CONF lives in that namespace and unsetting it breaks client
# config discovery. A login shell holds no allocation, so nothing needs
# shedding; from inside an salloc keep SLURM_CONF:
#   env $(env | grep -o '^SLURM_[^=]*' | grep -v '^SLURM_CONF$' \
#         | sed 's/^/-u /' | tr '\n' ' ') OUT=... sbatch ...
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/joint/model.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo, or predates pxf/joint; set PXF_REPO" >&2
    exit 2
fi

OUT="${OUT:?set OUT to the output directory}"
DATA_ROOT="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
# A DIRECTORY of training CIFs, not a manifest -- see the EXTRA_ARGS note above.
STRUCTURES="${PREFLIGHT_STRUCTURES:-$DATA_ROOT/afdb_laproteina/cif_phase1}"
DONOR="${DONOR:-$DATA_ROOT/component_donors/pxdesign_v0.1.0.pt}"
N_STRUCTURES="${N_STRUCTURES:-4}"
MULTIPLIER="${MULTIPLIER:-8}"
RATIO="${RATIO:-0.1}"
SEED="${SEED:-0}"
mkdir -p "$OUT" "$DATA_ROOT/runs/logs/pxf_preflight_joint"
cd "$ROOT"

# Relocated conda prefix, no bin/activate and no conda install behind it, so it
# is activated by PATH. torch 2.7.1+cu126 finds its CUDA runtime through the
# nvidia-*-cu12 wheels. Deliberately NOT loading nvhpc/cudnn modules -- this
# torch ships its own cudnn and a system one can shadow it.
PXF_PYTHON_ENV="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}"
export PATH="$PXF_PYTHON_ENV/bin:$PATH"

export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
# pxf.backbone.proteoaa drives the featurizer out of the pxdesign_train
# checkout; it searches hardcoded HAI paths and honours this override.
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$DATA_ROOT/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-$DATA_ROOT/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} out=${OUT}"
echo "structures=${STRUCTURES} n=${N_STRUCTURES} multiplier=${MULTIPLIER} ratio=${RATIO}"
echo "python=$(command -v python)  proteoaa=${PROTEOAA_ROOT}"
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
    ${PREFLIGHT_EXTRA_ARGS:-}

echo "done: $OUT/preflight.json"
