#!/bin/bash
#SBATCH --job-name=proteo-aa-bestckpt
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/validation/checkpoint_selection/%x-%j.out
#SBATCH --error=logs/validation/checkpoint_selection/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_backbone/bb_100k_val/checkpoints}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_backbone/best_checkpoint_eval/${SLURM_JOB_ID:-manual}}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" scripts/evaluation/select_best_checkpoint_protenix.py \
  --training-stage "${TRAINING_STAGE:-backbone_only}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint-pattern "${CHECKPOINT_PATTERN:-step*.pt}" \
  --checkpoint-stride "${CHECKPOINT_STRIDE:-1}" \
  --select-by "${SELECT_BY:-ca_lddt}" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${RUN_ROOT}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  "${@}"
