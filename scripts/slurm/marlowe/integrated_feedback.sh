#!/bin/bash
#SBATCH --job-name=pxf_ifb
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_ifb/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_ifb/%x-%j.out
#
# One parameterised wrapper for every integrated_feedback_v1 stage. CMD is the
# script and ARGS its arguments, so cache/train/preflight/eval/matrix all go
# through the same environment rather than five copies of it that can drift.
set -uo pipefail
ROOT="${PXF_REPO:-/users/yfsun/Proteo-AA-pxdesign-fampnn-pack}"
DATA_ROOT="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
CMD="${CMD:?set CMD to a script under scripts/}"
ARGS="${ARGS:?set ARGS}"
cd "$ROOT"
mkdir -p "$DATA_ROOT/runs/logs/pxf_ifb"
export PATH="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTEOAA_METRICS_ROOT="${PROTEOAA_METRICS_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$DATA_ROOT/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-$DATA_ROOT/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} cmd=$CMD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
eval "python $CMD $ARGS"
echo "EXIT=$?"
