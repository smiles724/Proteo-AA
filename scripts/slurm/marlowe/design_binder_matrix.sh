#!/bin/bash
#SBATCH --job-name=pxf_design_matrix
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
# One shard = one target. Sharding by target keeps every arm's view of a
# backbone inside a single process, so the paired comparison is never split
# across jobs; TARGET and OUT are required for that reason.
set -uo pipefail
ROOT="${PXF_REPO:-/users/yfsun/Proteo-AA-pxdesign-fampnn-pack}"
DATA_ROOT="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
TARGET="${TARGET:?set TARGET}"
OUT="${OUT:?set OUT}"
ARMS="${ARMS:?set ARMS to one or more --arm LABEL[=CKPT]}"
cd "$ROOT"
export PATH="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTEOAA_METRICS_ROOT="${PROTEOAA_METRICS_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$DATA_ROOT/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-$DATA_ROOT/protenix_data/common}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} target=$TARGET"
python scripts/design_binder_matrix.py \
  --backbones "$DATA_ROOT/runs/binder_bench/backbones" \
  --out "$OUT/$TARGET" --targets "$TARGET" $ARMS --device cuda
echo "EXIT=$?"
