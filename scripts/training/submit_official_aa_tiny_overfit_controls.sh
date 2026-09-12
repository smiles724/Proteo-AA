#!/bin/bash
# Two matched AA-head-only diagnostics starting from the official PXDesign ckpt:
#   clean   native backbone coordinates, conditioned at sigma=0.04
#   sigma04 coordinates noised at sigma=0.04
# The side-chain/co-evolution path is disabled because the official checkpoint
# does not contain a trained Proteo-AA side-chain module.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/training/slurm_stage3_coevolution_binder.sh"
OFFICIAL_CHECKPOINT="${OFFICIAL_CHECKPOINT:-/hai/scratch/shenjm/pxdesign_official/pxdesign_v0.1.0.pt}"
RUNS_ROOT="${RUNS_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/official_aa_tiny_overfit}"

TRAIN_STEPS="${TRAIN_STEPS:-1000}"
TINY_SAMPLES="${TINY_SAMPLES:-8}"
CROP_SIZE="${CROP_SIZE:-448}"
SEED="${SEED:-42}"
ARMS="${ARMS:-clean sigma04}"

for path in "${TRAIN_SCRIPT}" "${OFFICIAL_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
command -v sbatch >/dev/null || { echo "ERROR: sbatch is not available" >&2; exit 2; }

mkdir -p "${REPO_ROOT}/logs/training/stage3_binder"
cd "${REPO_ROOT}"

export REPO_ROOT PROTENIX_CODE_DIR="${REPO_ROOT}/Protenix" PXDESIGN_CODE_DIR="${REPO_ROOT}/PXDesign"
export PROTEOAA_DATA_ROOT="${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}"
export PINDER_CIF_CACHE="${PINDER_CIF_CACHE:-/hai/scratch/shenjm/pinder/cif_cache}"
export PINDER_PDB_CACHE="${PINDER_PDB_CACHE:-/hai/scratch/shenjm/pinder/2024-02/pdbs}"
export LOAD_CHECKPOINT="${OFFICIAL_CHECKPOINT}"
export WARM_START_PARAMS_ONLY=1 LOAD_AA_HEAD_FROM=0
export COMPLEX_PROVIDER=pinder
export STAGE2_START_MONOMER_FRAC=0 STAGE2_END_MONOMER_FRAC=0 PINDER_COMPLEX_FRAC=1
export MAX_STEPS="${TRAIN_STEPS}" TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-256}"
export CROP_SIZE ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-8}"
export LR="${LR:-3e-4}" AA_HEAD_LR="${AA_HEAD_LR:-3e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-50}" GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}" EVAL_INTERVAL=0
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-${TRAIN_STEPS}}" RUNS_ROOT

common_args=(
  --training-stage aa_head_warmup
  --complex-limit-index "${TINY_SAMPLES}"
  --no-ref-pos-augment
  --aa-head-grad-clip-norm 1.0
  --aa-forced-sigmas 0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04
  --aa-sigma-weight-mode uniform
  --seed "${SEED}"
)

for arm in ${ARMS}; do
  if [[ "${arm}" != "clean" && "${arm}" != "sigma04" ]]; then
    echo "ERROR: ARMS entries must be clean or sigma04; got ${arm}" >&2
    exit 2
  fi
  job_name="aa-official-${arm}-${TINY_SAMPLES}-c${CROP_SIZE}"
  arm_args=()
  if [[ "${arm}" == "clean" ]]; then
    arm_args+=(--aa-clean-coordinate-input)
  fi
  raw="$({
    sbatch --parsable --job-name="${job_name}" "${TRAIN_SCRIPT}" \
      "${common_args[@]}" "${arm_args[@]}"
  })"
  job_id="${raw%%;*}"
  echo "submitted ${arm}: ${job_id}"
  echo "log       : ${REPO_ROOT}/logs/training/stage3_binder/${job_name}-${job_id}.out"
  echo "checkpoint: ${RUNS_ROOT}/stage3_binder_coevolution/${job_id}/checkpoints/step${TRAIN_STEPS}.pt"
done
