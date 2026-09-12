#!/bin/bash
#SBATCH --job-name=sc-rigid-warmup
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/official_pxdesign/%x-%j.out
#SBATCH --error=logs/training/official_pxdesign/%x-%j.err
set -euo pipefail
export PROTEOAA_REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-sc-rigid-augmentation}
export OUTPUT_DIR=${OUTPUT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/official_sc_rigid_warmup/${SLURM_JOB_ID:-dry-run}}
exec bash "$PROTEOAA_REPO/scripts/training/slurm_official_sc_scratch_hai.sh" "$@" --stage4-native-sc-augmentation
