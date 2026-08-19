#!/bin/bash
#SBATCH --job-name=proteo-aa-stage3-coev
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/stage3_coevolution/%x-%j.out
#SBATCH --error=logs/training/stage3_coevolution/%x-%j.err

# Paper Stage III: both modules train together and the co-evolution refinement
# pass is live (S_phi -> h_res' -> a second backbone/AA pass, supervised by
# L_bb^post / L_aa^post). Warm-starts from a Stage II side-chain checkpoint.
#
# SMOKE=1 runs a handful of steps with logging every step -- use it to prove the
# stage assembles and the refinement pass runs before committing a 24h slot.

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_stage3_coevolution/${SLURM_JOB_ID:-manual}}"
# Stage II side-chain checkpoint. Empty = start S_phi from scratch, which is only
# useful for smoke-testing the machinery.
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
WARM_START_PARAMS_ONLY="${WARM_START_PARAMS_ONLY:-1}"
SMOKE="${SMOKE:-0}"

if [[ -n "${LOAD_CHECKPOINT}" && ! -f "${LOAD_CHECKPOINT}" ]]; then
  echo "ERROR: LOAD_CHECKPOINT does not exist: ${LOAD_CHECKPOINT}" >&2
  exit 2
fi

if [[ "${SMOKE}" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS:-6}"; LOG_INTERVAL="${LOG_INTERVAL:-1}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-0}"; CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
  TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-64}"
  # The refinement pass runs the backbone twice, so memory is ~2x Stage II at the
  # same crop. Keep the smoke crop small so a smoke failure means a real bug and
  # not just OOM.
  CROP_SIZE="${CROP_SIZE:-256}"
  ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-1}"
else
  MAX_STEPS="${MAX_STEPS:-30000}"; LOG_INTERVAL="${LOG_INTERVAL:-50}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"; CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
  TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-10000}"
  CROP_SIZE="${CROP_SIZE:-384}"
  ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-8}"
fi

mkdir -p "${REPO_ROOT}/logs/training/stage3_coevolution" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

LOAD_ARGS=()
if [[ -n "${LOAD_CHECKPOINT}" ]]; then
  LOAD_ARGS+=(--load-checkpoint "${LOAD_CHECKPOINT}")
  if [[ "${WARM_START_PARAMS_ONLY}" == "1" ]]; then
    LOAD_ARGS+=(--warm-start-params-only)
  fi
fi

"${PYTHON_BIN}" -u scripts/training/train_protenix_monomer.py \
  --training-stage coevolution \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${RUN_ROOT}" \
  --crop-size "${CROP_SIZE}" \
  --max-n-token "${CROP_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --train-samples-per-epoch "${TRAIN_SAMPLES_PER_EPOCH}" \
  --lr "${LR:-1e-5}" \
  --warmup-steps "${WARMUP_STEPS:-500}" \
  --iters-to-accumulate "${ITERS_TO_ACCUMULATE}" \
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-samples "${EVAL_SAMPLES:-491}" \
  --eval-num-workers "${EVAL_NUM_WORKERS:-0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  --template-provider "${TEMPLATE_PROVIDER:-dunbrack_mode}" \
  "${LOAD_ARGS[@]}" \
  "${@}"
