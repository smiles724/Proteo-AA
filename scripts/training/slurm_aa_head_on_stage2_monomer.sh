#!/bin/bash
#SBATCH --job-name=proteo-aa-aahead-s2
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/aa_head_on_stage2/%x-%j.out
#SBATCH --error=logs/training/aa_head_on_stage2/%x-%j.err

# Train ONLY the residue-type (AA) head on a FROZEN Stage II model, in the Stage
# III configuration (side chain on, refinement pass on, predicted frames,
# per-sigma). The backbone and S_phi are loaded from the Stage II checkpoint and
# never updated.
#
# Why not just reuse an aa_head_warmup checkpoint: an AA head is not portable
# across configurations. Grafting a head trained with enable_sidechain and
# enable_coevolution OFF into Stage III (both ON) gave aa_ce 54-76 at chance
# accuracy, even when the donor backbone was byte-identical to the Stage II trunk.
# The head has to be fitted to the feature path it will be used in.
#
# The checkpoint this writes carries all three trained components in one file.
#
# LOAD_CHECKPOINT is required: there is nothing to freeze without it.

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_aa_head_on_stage2/${SLURM_JOB_ID:-manual}}"
# Stage II side-chain checkpoint. Empty = start S_phi from scratch, which is only
# useful for smoke-testing the machinery.
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"
WARM_START_PARAMS_ONLY="${WARM_START_PARAMS_ONLY:-1}"
# A Stage II checkpoint carries the backbone and S_phi but a chance-level AA head
# (Stage II excluded it from the warm start and then froze it). Point this at a
# joint/aa-head run to compose all three trained components.
AA_HEAD_CHECKPOINT="${AA_HEAD_CHECKPOINT:-}"
SMOKE="${SMOKE:-0}"

if [[ -z "${LOAD_CHECKPOINT}" ]]; then
  echo "ERROR: LOAD_CHECKPOINT (a Stage II checkpoint) is required for this stage." >&2
  exit 2
fi
if [[ ! -f "${LOAD_CHECKPOINT}" ]]; then
  echo "ERROR: LOAD_CHECKPOINT does not exist: ${LOAD_CHECKPOINT}" >&2
  exit 2
fi
if [[ -n "${AA_HEAD_CHECKPOINT}" && ! -f "${AA_HEAD_CHECKPOINT}" ]]; then
  echo "ERROR: AA_HEAD_CHECKPOINT does not exist: ${AA_HEAD_CHECKPOINT}" >&2
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
  MAX_STEPS="${MAX_STEPS:-20000}"; LOG_INTERVAL="${LOG_INTERVAL:-50}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"; CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
  TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-10000}"
  CROP_SIZE="${CROP_SIZE:-384}"
  ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-8}"
fi

mkdir -p "${REPO_ROOT}/logs/training/aa_head_on_stage2" "${RUN_ROOT}"
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
if [[ -n "${AA_HEAD_CHECKPOINT}" ]]; then
  LOAD_ARGS+=(--load-aa-head-from "${AA_HEAD_CHECKPOINT}")
fi

"${PYTHON_BIN}" -u scripts/training/train_protenix_monomer.py \
  --training-stage aa_head_on_stage2 \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${RUN_ROOT}" \
  --crop-size "${CROP_SIZE}" \
  --max-n-token "${CROP_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --train-samples-per-epoch "${TRAIN_SAMPLES_PER_EPOCH}" \
  --lr "${LR:-1e-4}" \
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
