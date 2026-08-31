#!/bin/bash
# Submit two matched Stage-III AA diagnostics:
#   A. AA head lr=1e-4, live AA->S_phi gradient path
#   B. AA head lr=1e-5, detached AA->S_phi gradient path
# Optionally submit a short inference-style PINDER binder evaluation after each.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/training/slurm_stage3_coevolution_binder.sh"
EVAL_SCRIPT="${REPO_ROOT}/scripts/evaluation/slurm_eval_pinder_binder_backbone_inputs.sh"

STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt}"
AA_DONOR_CHECKPOINT="${AA_DONOR_CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/hai/scratch/shenjm/proteo_aa_runs/stage3_binder_coevolution/107904/checkpoints/step1000.pt}"
CONTROL_ROOT="${CONTROL_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/stage3_aa_controls}"

TRAIN_STEPS="${TRAIN_STEPS:-1000}"
CROP_SIZE="${CROP_SIZE:-448}"
SEED="${SEED:-42}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
SUBMIT_REFERENCE_EVALS="${SUBMIT_REFERENCE_EVALS:-1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"

for path in "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${STAGE2_CHECKPOINT}" "${AA_DONOR_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
if [[ "${SUBMIT_EVAL}" == "1" && "${SUBMIT_REFERENCE_EVALS}" == "1" ]]; then
  [[ -f "${BASELINE_CHECKPOINT}" ]] || {
    echo "ERROR: missing baseline checkpoint ${BASELINE_CHECKPOINT}" >&2
    exit 2
  }
fi
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
export CROP_SIZE TRAIN_STEPS SEED
export MAX_STEPS="${TRAIN_STEPS}"
export TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-10000}"
export LR="${LR:-1e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-8}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export EVAL_INTERVAL=0
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-${TRAIN_STEPS}}"

LR_RUNS_ROOT="${CONTROL_ROOT}/aa_lr1e4"
DETACH_RUNS_ROOT="${CONTROL_ROOT}/aa_detach"

job_lr_raw="$({
  AA_HEAD_LR=1e-4 \
  DETACH_AA_LOGITS_FOR_SIDECHAIN=0 \
  RUNS_ROOT="${LR_RUNS_ROOT}" \
  sbatch --parsable --job-name="aa-lr1e4-c${CROP_SIZE}" "${TRAIN_SCRIPT}" --seed "${SEED}"
})"
job_lr="${job_lr_raw%%;*}"

job_detach_raw="$({
  AA_HEAD_LR="" \
  DETACH_AA_LOGITS_FOR_SIDECHAIN=1 \
  RUNS_ROOT="${DETACH_RUNS_ROOT}" \
  sbatch --parsable --job-name="aa-detach-c${CROP_SIZE}" "${TRAIN_SCRIPT}" --seed "${SEED}"
})"
job_detach="${job_detach_raw%%;*}"

ckpt_lr="${LR_RUNS_ROOT}/stage3_binder_coevolution/${job_lr}/checkpoints/step${TRAIN_STEPS}.pt"
ckpt_detach="${DETACH_RUNS_ROOT}/stage3_binder_coevolution/${job_detach}/checkpoints/step${TRAIN_STEPS}.pt"

echo "submitted AA-lr control : ${job_lr}"
echo "submitted detach control: ${job_detach}"
echo "expected checkpoint A   : ${ckpt_lr}"
echo "expected checkpoint B   : ${ckpt_detach}"

if [[ "${SUBMIT_EVAL}" == "1" ]]; then
  eval_lr_raw="$({
    CHECKPOINT="${ckpt_lr}" \
    RUN_ROOT="${CONTROL_ROOT}/eval_aa_lr1e4_train${job_lr}" \
    MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
    CROP_SIZE="${CROP_SIZE}" \
    sbatch --parsable --dependency="afterok:${job_lr}" \
      --job-name=eval-aa-lr1e4 "${EVAL_SCRIPT}" --conditions inference_style --seed "${SEED}"
  })"
  eval_detach_raw="$({
    CHECKPOINT="${ckpt_detach}" \
    RUN_ROOT="${CONTROL_ROOT}/eval_aa_detach_train${job_detach}" \
    MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
    CROP_SIZE="${CROP_SIZE}" \
    sbatch --parsable --dependency="afterok:${job_detach}" \
      --job-name=eval-aa-detach "${EVAL_SCRIPT}" --conditions inference_style --seed "${SEED}"
  })"
  echo "submitted dependent eval A: ${eval_lr_raw%%;*}"
  echo "submitted dependent eval B: ${eval_detach_raw%%;*}"
  if [[ "${SUBMIT_REFERENCE_EVALS}" == "1" ]]; then
    eval_donor_raw="$({
      CHECKPOINT="${AA_DONOR_CHECKPOINT}" \
      RUN_ROOT="${CONTROL_ROOT}/eval_reference_donor" \
      MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
      CROP_SIZE="${CROP_SIZE}" \
      sbatch --parsable --job-name=eval-aa-donor "${EVAL_SCRIPT}" \
        --conditions inference_style --seed "${SEED}"
    })"
    eval_baseline_raw="$({
      CHECKPOINT="${BASELINE_CHECKPOINT}" \
      RUN_ROOT="${CONTROL_ROOT}/eval_reference_baseline_step1000" \
      MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
      CROP_SIZE="${CROP_SIZE}" \
      sbatch --parsable --job-name=eval-aa-baseline "${EVAL_SCRIPT}" \
        --conditions inference_style --seed "${SEED}"
    })"
    echo "submitted donor reference eval   : ${eval_donor_raw%%;*}"
    echo "submitted baseline reference eval: ${eval_baseline_raw%%;*}"
  fi
fi

echo "monitor: squeue -j ${job_lr},${job_detach}"
echo "logs   : ${REPO_ROOT}/logs/training/stage3_binder/{aa-lr1e4-c${CROP_SIZE},aa-detach-c${CROP_SIZE}}-<jobid>.out"
