#!/bin/bash
#SBATCH --job-name=official-regressions
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/official-regressions-%j.out
#SBATCH --error=logs/official-regressions-%j.err
set -euo pipefail
REPO=${PROTEOAA_REPO:-$SLURM_SUBMIT_DIR}
FAMPNN_ROOT=${FAMPNN_ROOT:-/hai/users/y/f/yfsun/Protein Project/fampnn}
PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:$FAMPNN_ROOT"
export FAMPNN_CHECKPOINT=${FAMPNN_CHECKPOINT:-$FAMPNN_ROOT/weights/fampnn_0_3.pt}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export CUDA_VISIBLE_DEVICES="" LAYERNORM_TYPE=torch OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$REPO"
"$PYTHON_BIN" -m pytest -q tests --junitxml="logs/official-regressions-$SLURM_JOB_ID.xml"
