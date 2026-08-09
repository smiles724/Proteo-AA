#!/bin/bash
#SBATCH --job-name=proteo-aa-sc-warm
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/sidechain_warmup/%x-%j.out
#SBATCH --error=logs/training/sidechain_warmup/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/${SLURM_JOB_ID:-manual}}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
WARM_START_PARAMS_ONLY="${WARM_START_PARAMS_ONLY:-1}"

if [[ -z "${LOAD_CHECKPOINT}" ]]; then
  echo "ERROR: LOAD_CHECKPOINT must be specified explicitly." >&2
  exit 2
fi
if [[ ! -f "${LOAD_CHECKPOINT}" ]]; then
  echo "ERROR: LOAD_CHECKPOINT does not exist: ${LOAD_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

LOAD_ARGS=(--load-checkpoint "${LOAD_CHECKPOINT}")
if [[ "${WARM_START_PARAMS_ONLY}" == "1" ]]; then
  LOAD_ARGS+=(--warm-start-params-only)
fi

"${PYTHON_BIN}" scripts/training/train_protenix_monomer.py \
  --training-stage sidechain_warmup \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${RUN_ROOT}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --max-steps "${MAX_STEPS:-50000}" \
  --train-samples-per-epoch "${TRAIN_SAMPLES_PER_EPOCH:-10000}" \
  --lr "${LR:-5e-5}" \
  --warmup-steps "${WARMUP_STEPS:-2000}" \
  --iters-to-accumulate "${ITERS_TO_ACCUMULATE:-8}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-2000}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --eval-interval "${EVAL_INTERVAL:-2000}" \
  --eval-samples "${EVAL_SAMPLES:-491}" \
  --eval-num-workers "${EVAL_NUM_WORKERS:-0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  --template-provider "${TEMPLATE_PROVIDER:-dunbrack_mode}" \
  --sc-frame-aware-head \
  --sc-local-coord-input \
  --sc-template-residual \
  "${LOAD_ARGS[@]}" \
  "${@}"
