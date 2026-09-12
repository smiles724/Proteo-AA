#!/bin/bash
#SBATCH --job-name=official-bb-metrics
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/official-bb-metrics-%j.out
#SBATCH --error=logs/official-bb-metrics-%j.err
set -euo pipefail
REPO=${PROTEOAA_REPO:-$SLURM_SUBMIT_DIR}
FAMPNN_ROOT=${FAMPNN_ROOT:-/hai/users/y/f/yfsun/Protein Project/fampnn}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:$FAMPNN_ROOT"
export PROTENIX_ROOT_DIR=/hai/scratch/yfsun/protenix_data
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$REPO"
/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python scripts/evaluation/probe_official_backbone_metrics.py \
  --checkpoint "${CHECKPOINT:-$REPO/runs/official_components_smoke/114891/checkpoints/step1_smoke.pt}" \
  --official-checkpoint "$REPO/runs/component_donors/pxdesign_v0.1.0.pt" \
  --samples-per-source "${SAMPLES_PER_SOURCE:-12}" --steps 20 400 \
  --output "$REPO/runs/official_backbone_metrics/$SLURM_JOB_ID"
