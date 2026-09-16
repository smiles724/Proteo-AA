#!/bin/bash
#SBATCH --job-name=pxf_sample_uncond
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_sample_uncond/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_sample_uncond/%x-%j.out
#
# Unconditional backbone draw. Backbones only -- the artefact every downstream
# metric and both packing arms reuse, so it is worth producing once.
#
# Marlowe header rationale is the same as eval_couple.sh: untyped gres so -G 1
# (no GPU type is accepted), --qos=medium required, batch the only partition
# that allows it.
#
#   LENGTHS="100 200 300" NUM_SAMPLES=20 OUT=.../uncond \
#     sbatch scripts/slurm/marlowe/sample_uncond.sh
#
# UNCOND_EXTRA_ARGS, not EXTRA_ARGS: marlowe_env.sh exports EXTRA_ARGS for the
# protenix side-chain eval and sample_uncond.py has no --mmcif-dir. Same trap
# that killed job 488735.
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

OUT="${OUT:?set OUT to the output directory}"
LENGTHS="${LENGTHS:-100 200 300 400 500}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
N_STEP="${N_STEP:-400}"
CROP_SIZE_UNCOND="${CROP_SIZE_UNCOND:-640}"
SEED="${SEED:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
mkdir -p "$OUT"
cd "$ROOT"

PXF_PYTHON_ENV="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}"
export PATH="$PXF_PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} out=${OUT}"
echo "lengths=${LENGTHS} num_samples=${NUM_SAMPLES} n_step=${N_STEP} shard=${SHARD_INDEX}/${NUM_SHARDS}"
echo "python=$(command -v python)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

python scripts/sample_uncond.py \
    --out "$OUT" \
    --lengths ${LENGTHS} \
    --num-samples "$NUM_SAMPLES" \
    --n-step "$N_STEP" \
    --crop-size "$CROP_SIZE_UNCOND" \
    --seed "$SEED" \
    --pxdesign-donor "${DONOR:?set DONOR}" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    ${UNCOND_EXTRA_ARGS:-}

echo "done -> $OUT"
