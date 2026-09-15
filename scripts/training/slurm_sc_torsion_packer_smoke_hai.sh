#!/bin/bash
#SBATCH --job-name=sc-packer-smoke
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:40:00
#SBATCH --array=0-3
#SBATCH --output=logs/training/sc_torsion_packer/%x-%A_%a.out
#SBATCH --error=logs/training/sc_torsion_packer/%x-%A_%a.err
#
# 30-step GPU smoke for both arms. What it is allowed to establish: the forward
# and backward run on real crops, the torsion metrics appear in the log, memory
# and step time are measured, validation and checkpoint saving work. What it
# CANNOT establish: packing quality, or anything about the a-token. Thirty steps
# from random weights is a plumbing check.
#
#   mkdir -p logs/training/sc_torsion_packer
#   sbatch scripts/training/slurm_sc_torsion_packer_smoke_hai.sh
#
set -euo pipefail
export MAX_STEPS=${MAX_STEPS:-30}
export WARMUP_STEPS=${WARMUP_STEPS:-5}
export EVAL_INTERVAL=${EVAL_INTERVAL:-20}
export EVAL_SAMPLES=${EVAL_SAMPLES:-8}
export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-20}
export LOG_INTERVAL=${LOG_INTERVAL:-1}
RUN_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-smoke}}
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) ARM=none ;;
  1) ARM=a_token ;;
  2) ARM=plm ;;
  3) ARM=both ;;
  *) echo "Unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
export OUTPUT_DIR=${OUTPUT_DIR:-/hai/scratch/shenjm/proteo_aa_runs/sc_torsion_packer_smoke/${RUN_ID}/${ARM}}
# Resolve the sibling launcher from the REPO, not from $BASH_SOURCE: sbatch runs
# a COPY of this file out of the Slurm spool directory, where no sibling exists.
# (First submission, 116935, died in 0 seconds exactly that way.)
export PROTEOAA_REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
exec bash "$PROTEOAA_REPO/scripts/training/slurm_sc_torsion_packer_hai.sh" "$@"
