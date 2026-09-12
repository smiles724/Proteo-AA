#!/bin/bash
#SBATCH --job-name=sc-warmup-smoke
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sc-warmup-smoke-%j.out
#SBATCH --error=logs/sc-warmup-smoke-%j.err
set -euo pipefail
REPO=${PROTEOAA_REPO:-$SLURM_SUBMIT_DIR}
DATA_ROOT=${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}
FAMPNN_ROOT=${FAMPNN_ROOT:-/hai/users/y/f/yfsun/Protein Project/fampnn}
PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:$FAMPNN_ROOT"
export PROTENIX_ROOT_DIR="$DATA_ROOT/protenix_data"
export PROTENIX_DATA_ROOT_DIR="$DATA_ROOT/protenix_data/common"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$REPO"
"$PYTHON_BIN" scripts/utilities/smoke_sc_warmup.py \
  --backbone-checkpoint "${BACKBONE_CHECKPOINT:-$REPO/runs/component_donors/pxdesign_v0.1.0.pt}" \
  --fampnn-checkpoint "${FAMPNN_CHECKPOINT:-$FAMPNN_ROOT/weights/fampnn_0_3.pt}" \
  --data-root "$DATA_ROOT/protenix_data" --output "${OUTPUT_DIR:-$REPO/runs/sc_warmup_smoke/$SLURM_JOB_ID}" "$@"
