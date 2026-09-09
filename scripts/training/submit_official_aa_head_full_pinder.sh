#!/bin/bash
# Train a fresh AA head on the full PINDER train split from the official
# PXDesign checkpoint, then evaluate held-out PINDER at selected checkpoints.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/training/slurm_stage3_coevolution_binder.sh"
EVAL_SCRIPT="${REPO_ROOT}/scripts/evaluation/slurm_eval_pinder_binder_backbone_inputs.sh"
OFFICIAL_CHECKPOINT="${OFFICIAL_CHECKPOINT:-/hai/scratch/shenjm/pxdesign_official/pxdesign_v0.1.0.pt}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/official_aa_head_full_pinder}"

TRAIN_STEPS="${TRAIN_STEPS:-5000}"
CROP_SIZE="${CROP_SIZE:-448}"
SEED="${SEED:-42}"
EVAL_STEPS="${EVAL_STEPS:-1000 3000 5000}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
EVAL_SIGMAS="${EVAL_SIGMAS:-0.04,0.4,4.0}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
RESUME_RUN_ID="${RESUME_RUN_ID:-}"
RESUME_STEP="${RESUME_STEP:-0}"

for path in "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${OFFICIAL_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
command -v sbatch >/dev/null || { echo "ERROR: sbatch is not available" >&2; exit 2; }
if [[ "${SUBMIT_EVAL}" != "0" && "${SUBMIT_EVAL}" != "1" ]]; then
  echo "ERROR: SUBMIT_EVAL must be 0 or 1" >&2
  exit 2
fi
for step in ${EVAL_STEPS}; do
  [[ "${step}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid EVAL_STEPS value: ${step}" >&2; exit 2; }
  (( step <= TRAIN_STEPS )) || { echo "ERROR: eval step ${step} exceeds TRAIN_STEPS=${TRAIN_STEPS}" >&2; exit 2; }
done

mkdir -p "${REPO_ROOT}/logs/training/stage3_binder" \
  "${REPO_ROOT}/logs/validation/pinder_binder_backbone_inputs"
cd "${REPO_ROOT}"

export REPO_ROOT PROTENIX_CODE_DIR="${REPO_ROOT}/Protenix" PXDESIGN_CODE_DIR="${REPO_ROOT}/PXDesign"
export PROTEOAA_DATA_ROOT="${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}"
export PINDER_CIF_CACHE="${PINDER_CIF_CACHE:-/hai/scratch/shenjm/pinder/cif_cache}"
export PINDER_PDB_CACHE="${PINDER_PDB_CACHE:-/hai/scratch/shenjm/pinder/2024-02/pdbs}"
if [[ -n "${RESUME_RUN_ID}" ]]; then
  [[ "${RESUME_STEP}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: RESUME_STEP must be positive when RESUME_RUN_ID is set" >&2
    exit 2
  }
  checkpoint_dir="${EXPERIMENT_ROOT}/stage3_binder_coevolution/${RESUME_RUN_ID}/checkpoints"
  export LOAD_CHECKPOINT="${checkpoint_dir}/step${RESUME_STEP}.pt"
  [[ -f "${LOAD_CHECKPOINT}" ]] || { echo "ERROR: missing ${LOAD_CHECKPOINT}" >&2; exit 2; }
  export WARM_START_PARAMS_ONLY=0 LOAD_AA_HEAD_FROM=0
  export RUN_ROOT="${EXPERIMENT_ROOT}/stage3_binder_coevolution/${RESUME_RUN_ID}"
  job_suffix="resume${RESUME_RUN_ID}s${RESUME_STEP}"
else
  export LOAD_CHECKPOINT="${OFFICIAL_CHECKPOINT}"
  export WARM_START_PARAMS_ONLY=1 LOAD_AA_HEAD_FROM=0
  checkpoint_dir=""
  job_suffix="fresh"
fi
export COMPLEX_PROVIDER=pinder
export STAGE2_START_MONOMER_FRAC=0 STAGE2_END_MONOMER_FRAC=0 PINDER_COMPLEX_FRAC=1
export MAX_STEPS="${TRAIN_STEPS}" TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-10000}"
export CROP_SIZE ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-8}"
export LR="${LR:-3e-4}" AA_HEAD_LR="${AA_HEAD_LR:-3e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-200}" GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}" EVAL_INTERVAL=0
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
export RUNS_ROOT="${EXPERIMENT_ROOT}"

train_raw="$({
  sbatch --parsable --job-name="aa-official-${job_suffix}-c${CROP_SIZE}" "${TRAIN_SCRIPT}" \
    --training-stage aa_head_warmup \
    --no-ref-pos-augment \
    --aa-head-grad-clip-norm 1.0 \
    --aa-forced-sigmas 0.04,0.4,0.04,0.4,0.04,0.4,0.04,0.4 \
    --aa-sigma-weight-mode uniform \
    --seed "${SEED}"
})"
train_job="${train_raw%%;*}"
if [[ -z "${checkpoint_dir}" ]]; then
  checkpoint_dir="${EXPERIMENT_ROOT}/stage3_binder_coevolution/${train_job}/checkpoints"
fi

echo "train: ${train_job}"
echo "train log: ${REPO_ROOT}/logs/training/stage3_binder/aa-official-${job_suffix}-c${CROP_SIZE}-${train_job}.out"

if [[ "${SUBMIT_EVAL}" == "1" ]]; then
  for step in ${EVAL_STEPS}; do
    checkpoint="${checkpoint_dir}/step${step}.pt"
    eval_root="${EXPERIMENT_ROOT}/validation/train${train_job}/step${step}"
    eval_raw="$({
      CHECKPOINT="${checkpoint}" \
      RUN_ROOT="${eval_root}" \
      MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
      CROP_SIZE="${CROP_SIZE}" \
      sbatch --parsable --dependency="afterok:${train_job}" \
        --job-name="eval-offaa-s${step}" "${EVAL_SCRIPT}" \
        --conditions inference_style \
        --sigmas "${EVAL_SIGMAS}" \
        --seed "${SEED}"
    })"
    echo "eval step${step}: ${eval_raw%%;*} (afterok:${train_job})"
    echo "eval output  : ${eval_root}"
  done
else
  echo "validation submission skipped (SUBMIT_EVAL=0)"
fi
