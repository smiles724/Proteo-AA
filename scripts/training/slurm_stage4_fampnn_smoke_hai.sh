#!/bin/bash
#SBATCH --job-name=stage4-fampnn-smoke
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --output=logs/training/stage4_fampnn/%x-%j.out
#SBATCH --error=logs/training/stage4_fampnn/%x-%j.err

# HAI-cluster bindings for the Stage IV engineering smoke. The command itself,
# and Marlowe's defaults for it, live in `slurm_stage4_fampnn_smoke.sh`; this
# file only supplies this cluster's Slurm directives, roots and interpreter.
#
# This is the release gate the Stage IV doc names: real model, real complex,
# finite losses, both feedback-gradient routes, an optimizer step with the
# frozen set verified frozen, save/resume, and a final mmCIF export. Marlowe
# job 476535 never ran it (batch was down), so it runs here instead.
#
#   mkdir -p logs/training/stage4_fampnn
#   sbatch scripts/training/slurm_stage4_fampnn_smoke_hai.sh
set -euo pipefail

export PROTEOAA_REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-official-pxdesign-fampnn}
export PROTEOAA_DATA_ROOT=${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}
export PROTEOAA_CODE_ROOT=${PROTEOAA_CODE_ROOT:-/hai/users/y/f/yfsun/Protein Project}
export PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
# Smoke the donor a production run really warm-starts from, not the AA-head-only
# stand-in: this cluster has a Stage III co-evolution binder checkpoint.
export SC_CHECKPOINT=${SC_CHECKPOINT:-$PROTEOAA_DATA_ROOT/proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step6000.pt}
export OUTPUT_DIR=${OUTPUT_DIR:-$PROTEOAA_DATA_ROOT/proteo_aa_runs/stage4_fampnn_smoke/${SLURM_JOB_ID:-manual}}

exec bash "$PROTEOAA_REPO/scripts/training/slurm_stage4_fampnn_smoke.sh" "$@"
