#!/bin/bash
#SBATCH --job-name=official-sc-warmup
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/official_pxdesign/%x-%j.out
#SBATCH --error=logs/training/official_pxdesign/%x-%j.err
set -euo pipefail
export PROTEOAA_REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-official-pxdesign-fampnn}
export SC_INIT=scratch
export STAGE4_PHASE=sc_warmup TRAIN_ROUNDS=0 INFERENCE_ROUNDS=0
export RESUME_CHECKPOINT= WARM_START_CHECKPOINT=
export OUTPUT_DIR=${OUTPUT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/official_sc_warmup/${SLURM_JOB_ID:-dry-run}}
# Fresh SC curriculum: native types/frames, monomers only, frozen BB and AA.
# Later complex/generated-input/feedback phases require validation and a warm start.
export CROP_SIZE=${CROP_SIZE:-384} MAX_STEPS=${MAX_STEPS:-50000}
export WARMUP_STEPS=${WARMUP_STEPS:-2000} EVAL_SAMPLES=${EVAL_SAMPLES:-491}
RUN_OPTIONS=()
if [[ ${1:-} == --dry-run ]]; then
  shift
  RUN_OPTIONS=(--dry-run)
fi
exec bash "$PROTEOAA_REPO/scripts/training/slurm_stage4_fampnn_binder_hai.sh" "${RUN_OPTIONS[@]}" \
  --data-mode monomer --stage2-start-monomer-frac 1 --stage2-end-monomer-frac 1 \
  --stage4-sc-lr "${SC_LR:-5e-5}" --no-ref-pos-augment "$@"
