#!/bin/bash
#SBATCH --job-name=pxf_bs_seq_sc_eval
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_bs_seq_sc/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_bs_seq_sc/%x-%j.out
#
# bs_seq_sc_v1: S03 (SC only) and J03 (sequence + SC), 2,000 updates.
#
# ARM and SEED are required and appear in the output path, because two paired
# seeds per arm is the design and a run that cannot say which it is cannot be
# paired with anything. LAMBDA_SEQ is required for J03: the trainer refuses
# `auto`, since a default of 1.0 is actively wrong here.
set -euo pipefail

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
[ -f "$ROOT/scripts/eval_bs_seq_sc.py" ] || { echo "set PXF_REPO" >&2; exit 2; }

DATA_ROOT="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
OUT="${OUT:?set OUT}"
CKPT_ARGS="${CKPT_ARGS:?set CKPT_ARGS to one or more --checkpoint LABEL=PATH}"
mkdir -p "$OUT" "$DATA_ROOT/runs/logs/pxf_bs_seq_sc"
cd "$ROOT"

PXF_PYTHON_ENV="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}"
export PATH="$PXF_PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTEOAA_METRICS_ROOT="${PROTEOAA_METRICS_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$DATA_ROOT/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-$DATA_ROOT/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} out=${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/eval_bs_seq_sc.py --out "$OUT" --device cuda $CKPT_ARGS ${EVAL_EXTRA_ARGS:-}
echo "done: $OUT"
