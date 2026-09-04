#!/bin/bash
# Tiny PINDER overfit with AA head + final main diffusion-transformer block.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/training/slurm_stage3_coevolution_binder.sh"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/fixed_global_decay_from_50k/checkpoints/step52500.pt}"
AA_DONOR_CHECKPOINT="${AA_DONOR_CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_aa_head_on_stage2/from_stage2_65000/checkpoints/step9000.pt}"
RUNS_ROOT="${RUNS_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/aa_lastblock_tiny_overfit}"
TRAIN_STEPS="${TRAIN_STEPS:-1500}"
TINY_SAMPLES="${TINY_SAMPLES:-32}"
CROP_SIZE="${CROP_SIZE:-448}"
SEED="${SEED:-42}"

for path in "${TRAIN_SCRIPT}" "${STAGE2_CHECKPOINT}" "${AA_DONOR_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
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
# Base LR applies to the newly unfrozen trunk block; AA_HEAD_LR is its own group.
export LR="${TRUNK_LR:-1e-5}" AA_HEAD_LR="${AA_HEAD_LR:-3e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-50}" GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}" EVAL_INTERVAL=0
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-${TRAIN_STEPS}}" RUNS_ROOT

raw="$({
  sbatch --parsable --job-name="aa-lastblk${TINY_SAMPLES}-c${CROP_SIZE}" "${TRAIN_SCRIPT}" \
    --training-stage aa_head_on_stage2 \
    --complex-limit-index "${TINY_SAMPLES}" \
    --no-ref-pos-augment \
    --aa-head-grad-clip-norm 1.0 \
    --unfreeze-last-diffusion-blocks 1 \
    --trunk-grad-scale 0.1 \
    --aa-forced-sigmas 0.04,0.4,0.04,0.4,0.04,0.4,0.04,0.4 \
    --aa-sigma-weight-mode uniform \
    --seed "${SEED}"
})"
job_id="${raw%%;*}"
echo "last-block tiny overfit: ${job_id}"
echo "checkpoint: ${RUNS_ROOT}/stage3_binder_coevolution/${job_id}/checkpoints/step${TRAIN_STEPS}.pt"
