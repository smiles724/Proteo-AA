#!/bin/bash
#SBATCH --job-name=eval-aa-sigma-compare
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/validation/pinder_binder_backbone_inputs/%x-%j.out
#SBATCH --error=logs/validation/pinder_binder_backbone_inputs/%x-%j.err

# Run all three checkpoints sequentially in one allocation. This avoids using
# three Slurm submission slots and guarantees identical evaluation settings.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
EVAL_SCRIPT="${REPO_ROOT}/scripts/evaluation/slurm_eval_pinder_binder_backbone_inputs.sh"

UNIFORM_CHECKPOINT="${UNIFORM_CHECKPOINT:?UNIFORM_CHECKPOINT is required}"
PARTIAL_LOW_CHECKPOINT="${PARTIAL_LOW_CHECKPOINT:?PARTIAL_LOW_CHECKPOINT is required}"
ALL_LOW_CHECKPOINT="${ALL_LOW_CHECKPOINT:?ALL_LOW_CHECKPOINT is required}"
COMPARISON_ROOT="${COMPARISON_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/aa_sigma_checkpoint_comparison}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
CROP_SIZE="${CROP_SIZE:-448}"
SIGMAS="${SIGMAS:-0.04,0.4,4.0}"
SEED="${SEED:-42}"

cd "${REPO_ROOT}"

run_one() {
  local label="$1"
  local checkpoint="$2"
  echo "=== evaluating ${label}: ${checkpoint} ==="
  CHECKPOINT="${checkpoint}" \
  RUN_ROOT="${COMPARISON_ROOT}/${label}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  CROP_SIZE="${CROP_SIZE}" \
  bash "${EVAL_SCRIPT}" \
    --conditions inference_style \
    --sigmas "${SIGMAS}" \
    --seed "${SEED}"
}

run_one aa-uniform "${UNIFORM_CHECKPOINT}"
run_one aa-partial-low "${PARTIAL_LOW_CHECKPOINT}"
run_one aa-all-low "${ALL_LOW_CHECKPOINT}"

echo "comparison complete: ${COMPARISON_ROOT}"
