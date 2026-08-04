#!/bin/bash
#SBATCH --job-name=proteo-aa-strict-aa
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
CHECKPOINT="${CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/joint_bb_aa_from_aa_head/from_aa_head_step17000/checkpoints/step50000.pt}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/eval_aa_head_strict_backbone/${SLURM_JOB_ID:-manual}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" scripts/evaluation/eval_aa_head_strict_backbone.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --model-stage joint \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-64}" \
  --limit-index "${LIMIT_INDEX:-0}" \
  --start-index "${START_INDEX:-0}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --conditions "${CONDITIONS:-full_topology_scrub,strict_native,strict_random}" \
  --sigmas "${SIGMAS:-0.04,0.4,4.0}" \
  --diffusion-samples "${DIFFUSION_SAMPLES:-1}" \
  --seed "${SEED:-20260802}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  "${@}"
