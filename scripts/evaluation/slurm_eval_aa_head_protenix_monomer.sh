#!/bin/bash
#SBATCH --job-name=proteo-aa-aaeval
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/validation/aa_head/%x-%j.out
#SBATCH --error=logs/validation/aa_head/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/eval_aa_head_monomer/${SLURM_JOB_ID:-manual}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINTS="${CHECKPOINTS:-}"

if [[ -z "${CHECKPOINT_DIR}" && -z "${CHECKPOINTS}" ]]; then
  echo "ERROR: specify CHECKPOINT_DIR or CHECKPOINTS explicitly." >&2
  exit 2
fi
if [[ -n "${CHECKPOINT_DIR}" && ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "ERROR: CHECKPOINT_DIR does not exist: ${CHECKPOINT_DIR}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ckpt_args=()
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  ckpt_args+=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi
if [[ -n "${CHECKPOINTS}" ]]; then
  read -r -a ckpt_list <<< "${CHECKPOINTS}"
  ckpt_args+=(--checkpoints "${ckpt_list[@]}")
fi

"${PYTHON_BIN}" scripts/evaluation/eval_aa_head_protenix_monomer.py \
  "${ckpt_args[@]}" \
  --checkpoint-pattern "${CHECKPOINT_PATTERN:-step*.pt}" \
  --checkpoint-stride "${CHECKPOINT_STRIDE:-1}" \
  --min-step "${MIN_STEP:-0}" \
  --max-step "${MAX_STEP:-0}" \
  --max-checkpoints "${MAX_CHECKPOINTS:-0}" \
  --select-by "${SELECT_BY:-f1_macro}" \
  --logits-source "${LOGITS_SOURCE:-mean_sample}" \
  --mask-source "${MASK_SOURCE:-design}" \
  --model-stage "${MODEL_STAGE:-backbone_only}" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${RUN_ROOT}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-16}" \
  --limit-index "${LIMIT_INDEX:-0}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  "${@}"
