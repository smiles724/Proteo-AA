#!/bin/bash
#SBATCH --job-name=proteo-aa-joint-mono
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
PROTEINMPNN_DIR="${PROTEINMPNN_DIR:-/hai/users/y/f/yfsun/tools/ProteinMPNN}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/joint_bb_aa_monomer_scratch/${SLURM_JOB_ID:-manual}}"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}:${PROTEINMPNN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MAX_STEPS="${MAX_STEPS:-50000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"

echo "Joint BB+AA monomer scratch run: ${RUN_ROOT}"
echo "Warm-start checkpoint: none"

"${PYTHON_BIN}" scripts/training/train_protenix_monomer.py \
  --training-stage joint \
  --data-mode monomer \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${RUN_ROOT}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-64}" \
  --max-steps "${MAX_STEPS}" \
  --train-samples-per-epoch "${TRAIN_SAMPLES_PER_EPOCH:-10000}" \
  --lr "${LR:-1e-4}" \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --eval-interval "${EVAL_INTERVAL:-1000}" \
  --eval-samples "${EVAL_SAMPLES:-128}" \
  --eval-num-workers "${EVAL_NUM_WORKERS:-0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  "${@}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${RUN_ROOT}/checkpoints/step${MAX_STEPS}.pt}"
  if [[ ! -f "${EVAL_CHECKPOINT}" ]]; then
    EVAL_CHECKPOINT="$(find "${RUN_ROOT}/checkpoints" -maxdepth 1 -name 'step*.pt' -type f | sort -V | tail -n 1)"
  fi
  if [[ -z "${EVAL_CHECKPOINT}" || ! -f "${EVAL_CHECKPOINT}" ]]; then
    echo "ERROR: no checkpoint found for post-training evaluation under ${RUN_ROOT}/checkpoints" >&2
    exit 3
  fi
  echo "Post-training evaluation checkpoint: ${EVAL_CHECKPOINT}"
  CHECKPOINT="${EVAL_CHECKPOINT}" \
  RUN_ROOT="${RUN_ROOT}/post_train_eval" \
  DATA_ROOT="${DATA_ROOT}" \
  PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR}" \
  PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR}" \
  PROTEINMPNN_DIR="${PROTEINMPNN_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  LIMIT_INDEX="${EVAL_LIMIT_INDEX:-0}" \
  BB_NUM_SAMPLES="${BB_EVAL_NUM_SAMPLES:-0}" \
  AA_MAX_SAMPLES="${AA_EVAL_MAX_SAMPLES:-0}" \
  MPNN_MAX_SAMPLES="${MPNN_EVAL_MAX_SAMPLES:-0}" \
  CROP_SIZE="${CROP_SIZE:-640}" \
  MAX_N_TOKEN="${MAX_N_TOKEN:-640}" \
  MAX_CROP_RETRIES="${EVAL_MAX_CROP_RETRIES:-64}" \
  DTYPE="${DTYPE:-bf16}" \
  bash scripts/evaluation/slurm_eval_joint_checkpoint_all.sh
fi
