#!/bin/bash
# Submit two matched AA-head-only PINDER controls and fixed-sigma validation:
#
#   A. uniform: historical EDM sigma sampling and uniform AA CE
#   B. low-sigma: force sigma=0.04,0.4 into every N_sample=8 forward and use
#      inverse-quadratic AA CE weighting
#
# Both arms freeze backbone + S_phi, use the same donor/checkpoint/data/seed,
# and independently clip the AA head.  A separate donor validation supplies the
# step-0 reference.  Default submission count: 5 jobs (2 train, 2 dependent
# eval, 1 donor eval). Set SUBMIT_EVAL=0 on accounts with a small submit quota;
# that submits only the two training jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/training/slurm_stage3_coevolution_binder.sh"
EVAL_SCRIPT="${REPO_ROOT}/scripts/evaluation/slurm_eval_pinder_binder_backbone_inputs.sh"

STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt}"
AA_DONOR_CHECKPOINT="${AA_DONOR_CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt}"
CONTROL_ROOT="${CONTROL_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/aa_low_sigma_controls}"

TRAIN_STEPS="${TRAIN_STEPS:-1000}"
CROP_SIZE="${CROP_SIZE:-448}"
SEED="${SEED:-42}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
EVAL_SIGMAS="${EVAL_SIGMAS:-0.04,0.4,4.0}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
SUBMIT_DONOR_EVAL="${SUBMIT_DONOR_EVAL:-1}"

for path in "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${STAGE2_CHECKPOINT}" "${AA_DONOR_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
command -v sbatch >/dev/null || { echo "ERROR: sbatch is not available" >&2; exit 2; }

mkdir -p "${REPO_ROOT}/logs/training/stage3_binder" \
  "${REPO_ROOT}/logs/validation/pinder_binder_backbone_inputs"
cd "${REPO_ROOT}"

export REPO_ROOT
export PROTEOAA_DATA_ROOT="${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}"
export PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-${REPO_ROOT}/Protenix}"
export PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-${REPO_ROOT}/PXDesign}"
export PINDER_CIF_CACHE="${PINDER_CIF_CACHE:-/hai/scratch/shenjm/pinder/cif_cache}"
export PINDER_PDB_CACHE="${PINDER_PDB_CACHE:-/hai/scratch/shenjm/pinder/2024-02/pdbs}"
export LOAD_CHECKPOINT="${STAGE2_CHECKPOINT}"
export AA_HEAD_CHECKPOINT="${AA_DONOR_CHECKPOINT}"
export WARM_START_PARAMS_ONLY=1
export COMPLEX_PROVIDER=pinder
export STAGE2_START_MONOMER_FRAC=0
export STAGE2_END_MONOMER_FRAC=0
export PINDER_COMPLEX_FRAC=1
export MAX_STEPS="${TRAIN_STEPS}"
export TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-10000}"
export CROP_SIZE
export ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-8}"
export LR="${LR:-1e-4}"
export AA_HEAD_LR="${AA_HEAD_LR:-1e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-200}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export EVAL_INTERVAL=0
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-${TRAIN_STEPS}}"

UNIFORM_ROOT="${CONTROL_ROOT}/uniform_sigma"
LOW_SIGMA_ROOT="${CONTROL_ROOT}/forced_low_sigma"

uniform_raw="$({
  RUNS_ROOT="${UNIFORM_ROOT}" \
  sbatch --parsable --job-name="aa-uniform-c${CROP_SIZE}" "${TRAIN_SCRIPT}" \
    --training-stage aa_head_on_stage2 \
    --aa-head-grad-clip-norm 1.0 \
    --aa-sigma-weight-mode uniform \
    --seed "${SEED}"
})"
uniform_job="${uniform_raw%%;*}"

low_sigma_raw="$({
  RUNS_ROOT="${LOW_SIGMA_ROOT}" \
  sbatch --parsable --job-name="aa-lowsigma-c${CROP_SIZE}" "${TRAIN_SCRIPT}" \
    --training-stage aa_head_on_stage2 \
    --aa-head-grad-clip-norm 1.0 \
    --aa-forced-sigmas 0.04,0.4 \
    --aa-sigma-weight-mode inverse_quadratic \
    --aa-sigma-weight-scale 0.4 \
    --aa-sigma-weight-floor 0.1 \
    --seed "${SEED}"
})"
low_sigma_job="${low_sigma_raw%%;*}"

uniform_ckpt="${UNIFORM_ROOT}/stage3_binder_coevolution/${uniform_job}/checkpoints/step${TRAIN_STEPS}.pt"
low_sigma_ckpt="${LOW_SIGMA_ROOT}/stage3_binder_coevolution/${low_sigma_job}/checkpoints/step${TRAIN_STEPS}.pt"

echo "uniform train       : ${uniform_job}"
echo "forced-low train    : ${low_sigma_job}"
echo "uniform checkpoint  : ${uniform_ckpt}"
echo "low-sigma checkpoint: ${low_sigma_ckpt}"

if [[ "${SUBMIT_EVAL}" == "1" ]]; then
  uniform_eval_raw="$({
    CHECKPOINT="${uniform_ckpt}" \
    RUN_ROOT="${CONTROL_ROOT}/eval_uniform_train${uniform_job}" \
    MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
    CROP_SIZE="${CROP_SIZE}" \
    sbatch --parsable --dependency="afterok:${uniform_job}" \
      --job-name=eval-aa-uniform "${EVAL_SCRIPT}" \
      --conditions inference_style --sigmas "${EVAL_SIGMAS}" --seed "${SEED}"
  })"
  low_sigma_eval_raw="$({
    CHECKPOINT="${low_sigma_ckpt}" \
    RUN_ROOT="${CONTROL_ROOT}/eval_lowsigma_train${low_sigma_job}" \
    MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
    CROP_SIZE="${CROP_SIZE}" \
    sbatch --parsable --dependency="afterok:${low_sigma_job}" \
      --job-name=eval-aa-lowsigma "${EVAL_SCRIPT}" \
      --conditions inference_style --sigmas "${EVAL_SIGMAS}" --seed "${SEED}"
  })"
  echo "uniform eval        : ${uniform_eval_raw%%;*} (afterok:${uniform_job})"
  echo "forced-low eval     : ${low_sigma_eval_raw%%;*} (afterok:${low_sigma_job})"

  if [[ "${SUBMIT_DONOR_EVAL}" == "1" ]]; then
    donor_eval_raw="$({
      CHECKPOINT="${AA_DONOR_CHECKPOINT}" \
      RUN_ROOT="${CONTROL_ROOT}/eval_reference_donor" \
      MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
      CROP_SIZE="${CROP_SIZE}" \
      sbatch --parsable --job-name=eval-aa-donor-sigma "${EVAL_SCRIPT}" \
        --conditions inference_style --sigmas "${EVAL_SIGMAS}" --seed "${SEED}"
    })"
    echo "donor reference eval: ${donor_eval_raw%%;*}"
  fi
else
  echo "eval submission     : skipped (SUBMIT_EVAL=0)"
fi

echo "training logs: ${REPO_ROOT}/logs/training/stage3_binder/aa-{uniform,lowsigma}-c${CROP_SIZE}-<jobid>.out"
echo "eval logs    : ${REPO_ROOT}/logs/validation/pinder_binder_backbone_inputs/eval-aa-*-<jobid>.out"
