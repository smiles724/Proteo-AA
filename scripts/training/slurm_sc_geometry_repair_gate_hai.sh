#!/usr/bin/env bash
#SBATCH --job-name=sc-repair-gate
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/training/sc-repair-gate-%j.out
#SBATCH --error=logs/training/sc-repair-gate-%j.err
set -euo pipefail

REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases}
PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
DONOR=${SC_REPAIR_DONOR:-/hai/scratch/yfsun/proteo_aa_runs/official_sc_rigid_warmup/114967/checkpoints/step46000.pt}
CALIBRATION_DIR=${SC_REPAIR_CALIBRATION_DIR:-$REPO/runs/sc_geometry_repair/calibration_v1}
OUTPUT=${OUTPUT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/sc_geometry_repair/gate/${SLURM_JOB_ID}}

export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:${FAMPNN_ROOT:-/hai/users/y/f/yfsun/Protein Project/fampnn}"
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export OMP_NUM_THREADS=4 LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$REPO"
exec "$PYTHON_BIN" scripts/utilities/smoke_sc_geometry_repair.py \
  --checkpoint "$DONOR" --calibration-dir "$CALIBRATION_DIR" \
  --output "$OUTPUT" --device cuda
