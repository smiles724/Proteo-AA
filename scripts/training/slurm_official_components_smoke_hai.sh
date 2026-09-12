#!/bin/bash
#SBATCH --job-name=official-components-smoke
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/official-components-%j.out
#SBATCH --error=logs/official-components-%j.err
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
SC_ARGS=(--sidechain-checkpoint "${SC_CHECKPOINT:-$DATA_ROOT/proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step6000.pt}")
if [[ ${SC_INIT:-checkpoint} == scratch ]]; then
  SC_ARGS=(--sidechain-init scratch)
fi
"$PYTHON_BIN" scripts/utilities/smoke_stage4_fampnn.py \
  --backbone-checkpoint "${BACKBONE_CHECKPOINT:-$REPO/runs/component_donors/pxdesign_v0.1.0.pt}" \
  "${SC_ARGS[@]}" \
  --fampnn-checkpoint "${FAMPNN_CHECKPOINT:-$FAMPNN_ROOT/weights/fampnn_0_3.pt}" \
  --data-root "$DATA_ROOT" --output "${OUTPUT_DIR:-$REPO/runs/official_components_smoke/$SLURM_JOB_ID}" "$@"
