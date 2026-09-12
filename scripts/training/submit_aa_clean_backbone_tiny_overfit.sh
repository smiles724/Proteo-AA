#!/bin/bash
# Submit one clean-native-backbone AA tiny-overfit diagnostic.
#
# The native backbone is centre/random-rotation augmented but receives NO
# coordinate noise. The diffusion network is still conditioned at sigma=0.04;
# using literal sigma=0 is unsafe in EDM log-time/preconditioning code.
#
# Choose how much of the main diffusion transformer the AA CE may update:
#   UNFREEZE_LAST_BLOCKS=0   AA head only
#   UNFREEZE_LAST_BLOCKS=1   AA head + final transformer block
#   UNFREEZE_LAST_BLOCKS=16  AA head + all 16 transformer blocks

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/training/slurm_stage3_coevolution_binder.sh"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt}"
AA_DONOR_CHECKPOINT="${AA_DONOR_CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt}"
RUNS_ROOT="${RUNS_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/aa_clean_backbone_tiny_overfit}"

TRAIN_STEPS="${TRAIN_STEPS:-3000}"
TINY_SAMPLES="${TINY_SAMPLES:-32}"
CROP_SIZE="${CROP_SIZE:-448}"
SEED="${SEED:-42}"
UNFREEZE_LAST_BLOCKS="${UNFREEZE_LAST_BLOCKS:-0}"

if ! [[ "${UNFREEZE_LAST_BLOCKS}" =~ ^[0-9]+$ ]] \
  || (( UNFREEZE_LAST_BLOCKS > 16 )); then
  echo "ERROR: UNFREEZE_LAST_BLOCKS must be an integer from 0 through 16" >&2
  exit 2
fi
for path in "${TRAIN_SCRIPT}" "${STAGE2_CHECKPOINT}" "${AA_DONOR_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done

case "${UNFREEZE_LAST_BLOCKS}" in
  0) arm="head" ;;
  1) arm="last1" ;;
  16) arm="all16" ;;
  *) arm="last${UNFREEZE_LAST_BLOCKS}" ;;
esac
job_name="aa-clean-${arm}-${TINY_SAMPLES}-c${CROP_SIZE}"

mkdir -p "${REPO_ROOT}/logs/training/stage3_binder"
cd "${REPO_ROOT}"

export REPO_ROOT PROTENIX_CODE_DIR="${REPO_ROOT}/Protenix" PXDESIGN_CODE_DIR="${REPO_ROOT}/PXDesign"
export PROTEOAA_DATA_ROOT="${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}"
export PINDER_CIF_CACHE="${PINDER_CIF_CACHE:-/hai/scratch/shenjm/pinder/cif_cache}"
export PINDER_PDB_CACHE="${PINDER_PDB_CACHE:-/hai/scratch/shenjm/pinder/2024-02/pdbs}"
export LOAD_CHECKPOINT="${STAGE2_CHECKPOINT}" AA_HEAD_CHECKPOINT="${AA_DONOR_CHECKPOINT}"
export WARM_START_PARAMS_ONLY=1 COMPLEX_PROVIDER=pinder
export STAGE2_START_MONOMER_FRAC=0 STAGE2_END_MONOMER_FRAC=0 PINDER_COMPLEX_FRAC=1
export MAX_STEPS="${TRAIN_STEPS}" TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-256}"
export CROP_SIZE ITERS_TO_ACCUMULATE="${ITERS_TO_ACCUMULATE:-8}"
export LR="${TRUNK_LR:-1e-4}" AA_HEAD_LR="${AA_HEAD_LR:-3e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-50}" GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}" EVAL_INTERVAL=0
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-${TRAIN_STEPS}}" RUNS_ROOT

raw="$({
  sbatch --parsable --job-name="${job_name}" "${TRAIN_SCRIPT}" \
    --training-stage aa_head_on_stage2 \
    --complex-limit-index "${TINY_SAMPLES}" \
    --no-ref-pos-augment \
    --aa-head-grad-clip-norm 1.0 \
    --unfreeze-last-diffusion-blocks "${UNFREEZE_LAST_BLOCKS}" \
    --trunk-grad-scale 1.0 \
    --aa-clean-coordinate-input \
    --aa-forced-sigmas 0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04 \
    --aa-sigma-weight-mode uniform \
    --seed "${SEED}"
})"
job_id="${raw%%;*}"

echo "submitted ${arm}: ${job_id}"
echo "log       : ${REPO_ROOT}/logs/training/stage3_binder/${job_name}-${job_id}.out"
echo "checkpoint: ${RUNS_ROOT}/stage3_binder_coevolution/${job_id}/checkpoints/step${TRAIN_STEPS}.pt"
