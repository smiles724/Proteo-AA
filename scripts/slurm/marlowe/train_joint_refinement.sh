#!/bin/bash
#SBATCH --job-name=pxf_joint
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_joint/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_joint/%x-%j.out
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
# MARLOWE COPY of scripts/slurm/train_joint_refinement.sh. The HAI original is
# left untouched -- that repo is shared and in-place edits break it.
#
# Differences forced by this cluster: untyped gres (-G 1; "gpu:h200:1" is
# rejected), batch/--qos=medium, log paths off /hai, no conda (a relocated
# prefix activated by PATH), PYTHONPATH appended so marlowe_env.sh's additions
# survive, and PROTEOAA_ROOT pointed at the pxdesign_train checkout.
#
# JOINT_STRUCTURES and JOINT_EXTRA_ARGS are namespaced on purpose:
# marlowe_env.sh exports bare STRUCTURES and EXTRA_ARGS for other jobs, and
# inheriting either here fails on a path or a flag that looks plausible in the
# log. Same trap that killed job 488735.
#
# Do NOT clear the whole SLURM_* block: on Marlowe SLURM_CONF lives in that
# namespace and unsetting it breaks client config discovery. From inside an
# salloc, keep it:
#   env $(env | grep -o '^SLURM_[^=]*' | grep -v '^SLURM_CONF$' \
#         | sed 's/^/-u /' | tr '\n' ' ') ARM=B0 OUT=... sbatch ...
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/joint/trainer.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo, or predates pxf/joint; set PXF_REPO" >&2
    exit 2
fi

ARM="${ARM:?set ARM to one of B0 B1 B2 BF BS}"
OUT="${OUT:?set OUT to the output directory}"
CONFIG="${CONFIG:-$ROOT/configs/joint_refinement/$ARM.yaml}"
PREFLIGHT="${PREFLIGHT:-}"
STRUCTURES="${JOINT_STRUCTURES:-$DATA_ROOT/afdb_laproteina/cif_phase1}"
DATA_ROOT="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
DONOR="${DONOR:-$DATA_ROOT/component_donors/pxdesign_v0.1.0.pt}"
MAX_STEPS="${MAX_STEPS:-}"
SEED="${SEED:-0}"
RESUME="${RESUME:-}"
mkdir -p "$OUT" "$DATA_ROOT/runs/logs/pxf_joint"
cd "$ROOT"

# Relocated conda prefix, no bin/activate, activated by PATH.
PXF_PYTHON_ENV="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}"
export PATH="$PXF_PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$DATA_ROOT/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-$DATA_ROOT/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} arm=${ARM} out=${OUT}"
echo "config=${CONFIG} preflight=${PREFLIGHT:-<none>} seed=${SEED}"
echo "python=$(command -v python)  structures=${STRUCTURES}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

ARGS=(--arm "$ARM" --out "$OUT" --config "$CONFIG" --donor "$DONOR" --seed "$SEED")
[ -n "$PREFLIGHT" ] && ARGS+=(--preflight "$PREFLIGHT")
[ -n "$STRUCTURES" ] && ARGS+=(--structures "$STRUCTURES")
[ -n "$MAX_STEPS" ] && ARGS+=(--max-steps "$MAX_STEPS")
[ -n "$RESUME" ] && ARGS+=(--resume "$RESUME")

python scripts/train_joint_refinement.py "${ARGS[@]}" ${JOINT_EXTRA_ARGS:-}
echo "done: $OUT"
