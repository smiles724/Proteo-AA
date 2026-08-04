#!/bin/bash
#SBATCH --job-name=proteo-aa-aainfer
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
CHECKPOINT="${CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_aa_head_warmup/93519/checkpoints/step17000.pt}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/infer_aa_head_monomer/${SLURM_JOB_ID:-manual}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: AA-head checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" scripts/evaluation/infer_aa_head_protenix_monomer.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-16}" \
  --limit-index "${LIMIT_INDEX:-0}" \
  --start-index "${START_INDEX:-0}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --n-step "${N_STEP:-20}" \
  --seed "${SEED:-20260802}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  "${@}"
