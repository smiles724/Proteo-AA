#!/usr/bin/env bash
#SBATCH --job-name=sc-adaptation-smoke
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/training/sc-adaptation-smoke-%j.out
#SBATCH --error=logs/training/sc-adaptation-smoke-%j.err
set -euo pipefail
REPO=${PROTEOAA_REPO:-$SLURM_SUBMIT_DIR}
PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
: "${SC_TEST_CHECKPOINT:?Set a rigid-augmentation checkpoint as a test fixture}"
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:${FAMPNN_ROOT:-/hai/users/y/f/yfsun/Protein Project/fampnn}"
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export OMP_NUM_THREADS=4 LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$REPO"
exec "$PYTHON_BIN" scripts/utilities/smoke_sc_adaptation.py --checkpoint "$SC_TEST_CHECKPOINT" \
  --output "${OUTPUT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/sc_adaptation_smoke/$SLURM_JOB_ID}"
