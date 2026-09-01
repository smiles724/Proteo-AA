#!/bin/bash
# Fixed-sigma validation for the three matched 1000-step AA controls.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
EVAL_SCRIPT="${REPO_ROOT}/scripts/evaluation/slurm_eval_aa_sigma_checkpoint_comparison.sh"

UNIFORM_CHECKPOINT="${UNIFORM_CHECKPOINT:-/hai/scratch/shenjm/proteo_aa_runs/aa_low_sigma_controls/uniform_sigma/stage3_binder_coevolution/109122/checkpoints/step1000.pt}"
PARTIAL_LOW_CHECKPOINT="${PARTIAL_LOW_CHECKPOINT:-/hai/scratch/shenjm/proteo_aa_runs/aa_low_sigma_controls/forced_low_sigma/stage3_binder_coevolution/109123/checkpoints/step1000.pt}"
ALL_LOW_CHECKPOINT="${ALL_LOW_CHECKPOINT:-/hai/scratch/shenjm/proteo_aa_runs/aa_all_low_sigma_warmup/stage3_binder_coevolution/109126/checkpoints/step1000.pt}"
COMPARISON_ROOT="${COMPARISON_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/aa_sigma_checkpoint_comparison}"

MAX_SAMPLES="${MAX_SAMPLES:-128}"
CROP_SIZE="${CROP_SIZE:-448}"
SIGMAS="${SIGMAS:-0.04,0.4,4.0}"
SEED="${SEED:-42}"

for path in "${EVAL_SCRIPT}" "${UNIFORM_CHECKPOINT}" "${PARTIAL_LOW_CHECKPOINT}" "${ALL_LOW_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
command -v sbatch >/dev/null || { echo "ERROR: sbatch is not available" >&2; exit 2; }

mkdir -p "${REPO_ROOT}/logs/validation/pinder_binder_backbone_inputs"
cd "${REPO_ROOT}"

job_raw="$({
  UNIFORM_CHECKPOINT="${UNIFORM_CHECKPOINT}" \
  PARTIAL_LOW_CHECKPOINT="${PARTIAL_LOW_CHECKPOINT}" \
  ALL_LOW_CHECKPOINT="${ALL_LOW_CHECKPOINT}" \
  COMPARISON_ROOT="${COMPARISON_ROOT}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  CROP_SIZE="${CROP_SIZE}" \
  SIGMAS="${SIGMAS}" \
  SEED="${SEED}" \
  sbatch --parsable "${EVAL_SCRIPT}"
})"

echo "comparison eval: ${job_raw%%;*}"
echo "results root   : ${COMPARISON_ROOT}"
