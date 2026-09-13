#!/usr/bin/env bash
#SBATCH --job-name=sc-repair-extend
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/sc-repair-extend-%j.out
#SBATCH --error=logs/training/sc-repair-extend-%j.err
set -euo pipefail

REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases}
PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
RUN_ROOT=${SC_REPAIR_RUN_ROOT:?Set SC_REPAIR_RUN_ROOT to the directory containing acceptance.json}
CHECKPOINT=$(
  "$PYTHON_BIN" -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["approved"]; print(a["selected"]["checkpoint"])' \
    "$RUN_ROOT/acceptance.json"
)
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:${FAMPNN_ROOT:-/hai/users/y/f/yfsun/Protein Project/fampnn}"
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export OMP_NUM_THREADS=4 LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$REPO"
exec "$PYTHON_BIN" scripts/training/train_sc_adaptation.py \
  --resume-checkpoint "$CHECKPOINT" --output-dir "$RUN_ROOT/arm_C_extended" \
  --max-steps 5000
