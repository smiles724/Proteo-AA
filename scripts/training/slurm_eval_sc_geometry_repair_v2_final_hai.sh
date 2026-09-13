#!/usr/bin/env bash
#SBATCH --job-name=sc-repair-v2-final
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/training/sc-repair-v2-final-%j.out
#SBATCH --error=logs/training/sc-repair-v2-final-%j.err
set -euo pipefail

REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases}
PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
RUN_ROOT=${SC_REPAIR_V2_RUN_ROOT:?Set SC_REPAIR_V2_RUN_ROOT to the directory containing acceptance_v2.json}
OUTPUT=${OUTPUT_DIR:-$RUN_ROOT/final_test}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:${FAMPNN_ROOT:-/hai/users/y/f/yfsun/Protein Project/fampnn}"
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export OMP_NUM_THREADS=4 LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$REPO"
exec "$PYTHON_BIN" scripts/utilities/evaluate_sc_geometry_repair_final.py \
  --acceptance "$RUN_ROOT/acceptance_v2.json" --output "$OUTPUT" --device cuda
