#!/bin/bash
#SBATCH --job-name=official-sc-scratch
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
export STAGE4_PHASE=sc_adapt TRAIN_ROUNDS=0 INFERENCE_ROUNDS=0
export RESUME_CHECKPOINT= WARM_START_CHECKPOINT=
export OUTPUT_DIR=${OUTPUT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/official_sc_scratch/${SLURM_JOB_ID:-dry-run}}
# Match the donor-initialized comparison: 25% PDB monomers, 75% PINDER,
# crop 384, accumulation 8, SC lr 1e-5, warmup 500, maximum 30000 updates.
exec bash "$PROTEOAA_REPO/scripts/training/slurm_stage4_fampnn_binder_hai.sh" "$@"
