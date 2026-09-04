#!/bin/bash
# Submit matched PINDER validation for crop-448 Stage III steps 4k and 6k.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
EVAL_SCRIPT="${REPO_ROOT}/scripts/evaluation/slurm_eval_pinder_binder_backbone_inputs.sh"
RUN_DIR="${RUN_DIR:-/hai/scratch/shenjm/proteo_aa_runs/pinder_validation/stage3_107904_sigma_sweep}"
CKPT_DIR="${CKPT_DIR:-/hai/scratch/shenjm/proteo_aa_runs/stage3_binder_coevolution/107904/checkpoints}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
CROP_SIZE="${CROP_SIZE:-448}"
SIGMAS="${SIGMAS:-0.04,0.1,0.2,0.4,0.8,1.0,4.0}"
SEED="${SEED:-42}"

mkdir -p "${REPO_ROOT}/logs/validation/pinder_binder_backbone_inputs"
cd "${REPO_ROOT}"
for step in 4000 6000; do
  checkpoint="${CKPT_DIR}/step${step}.pt"
  [[ -f "${checkpoint}" ]] || { echo "ERROR: missing ${checkpoint}" >&2; exit 2; }
  raw="$({
    CHECKPOINT="${checkpoint}" \
    RUN_ROOT="${RUN_DIR}/step${step}" \
    MAX_SAMPLES="${MAX_SAMPLES}" \
    CROP_SIZE="${CROP_SIZE}" \
    sbatch --parsable --job-name="pinder-s${step}-sigma" "${EVAL_SCRIPT}" \
      --conditions inference_style --sigmas "${SIGMAS}" --seed "${SEED}"
  })"
  echo "step${step}: ${raw%%;*}"
done
