#!/usr/bin/env bash
#SBATCH --job-name=sc-complex-adapt
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/sc-adaptation-%j.out
#SBATCH --error=logs/training/sc-adaptation-%j.err
set -euo pipefail
export SC_PHASE=sc_complex_adapt
export PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
export PROTEOAA_REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases}
export OUTPUT_DIR=${OUTPUT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/sc_complex_adapt/${SLURM_JOB_ID:-dry-run}}
exec bash "$PROTEOAA_REPO/scripts/training/run_sc_adaptation.sh" "$@"
