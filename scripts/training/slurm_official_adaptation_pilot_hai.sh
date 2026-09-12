#!/bin/bash
#SBATCH --job-name=official-adaptation-pilot
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/official-adaptation-pilot-%j.out
#SBATCH --error=logs/official-adaptation-pilot-%j.err
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
"$PYTHON_BIN" scripts/evaluation/pilot_official_adaptation.py \
  --checkpoint "${START_CHECKPOINT:?Set START_CHECKPOINT to a validated integrated SC checkpoint}" \
  --smoke-dir "${SMOKE_DIR:?Set SMOKE_DIR to its strict batch and manifest directory}" \
  --data-root "$DATA_ROOT" --steps-per-phase "${STEPS_PER_PHASE:-20}" \
  --output "${OUTPUT_DIR:-$REPO/runs/official_adaptation_pilot/$SLURM_JOB_ID}"
