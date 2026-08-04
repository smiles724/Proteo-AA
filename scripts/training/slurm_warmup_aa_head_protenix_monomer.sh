#!/bin/bash
#SBATCH --job-name=proteo-aa-aa-warm
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
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_aa_head_warmup/${SLURM_JOB_ID:-manual}}"

# BEST_MONOMER_CKPT takes precedence.  Otherwise read the checkpoint chosen by
# scripts/evaluation/slurm_select_best_checkpoint_protenix.sh.
BEST_MONOMER_CKPT="${BEST_MONOMER_CKPT:-}"
BEST_CHECKPOINT_JSON="${BEST_CHECKPOINT_JSON:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_backbone/best_checkpoint_eval/92187/best_checkpoint.json}"

if [[ -z "${BEST_MONOMER_CKPT}" ]]; then
  if [[ ! -f "${BEST_CHECKPOINT_JSON}" ]]; then
    echo "ERROR: set BEST_MONOMER_CKPT or provide BEST_CHECKPOINT_JSON." >&2
    echo "Missing selection file: ${BEST_CHECKPOINT_JSON}" >&2
    exit 2
  fi
  BEST_MONOMER_CKPT="$(
    "${PYTHON_BIN}" -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' \
      "${BEST_CHECKPOINT_JSON}"
  )"
fi

if [[ ! -f "${BEST_MONOMER_CKPT}" ]]; then
  echo "ERROR: selected stage-1 checkpoint does not exist: ${BEST_MONOMER_CKPT}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Stage-1 checkpoint: ${BEST_MONOMER_CKPT}"
echo "AA-head run: ${RUN_ROOT}"

"${PYTHON_BIN}" scripts/training/train_protenix_monomer.py \
  --training-stage aa_head_warmup \
  --data-mode monomer \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${RUN_ROOT}" \
  --load-checkpoint "${BEST_MONOMER_CKPT}" \
  --warm-start-params-only \
  --aa-input-source "${AA_INPUT_SOURCE:-backbone_geometry}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --max-steps "${MAX_STEPS:-20000}" \
  --train-samples-per-epoch "${TRAIN_SAMPLES_PER_EPOCH:-10000}" \
  --lr "${LR:-1e-4}" \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-1000}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --eval-interval "${EVAL_INTERVAL:-1000}" \
  --eval-samples "${EVAL_SAMPLES:-128}" \
  --eval-num-workers "${EVAL_NUM_WORKERS:-0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  "${@}"
