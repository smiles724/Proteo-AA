#!/bin/bash
#SBATCH --job-name=cmp-estimators
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/evaluation/%x-%j.out
#SBATCH --error=logs/evaluation/%x-%j.err
set -euo pipefail
REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:${APM_REFERENCE:-/hai/scratch/shenjm/apm_reference}:${PYEXTRA:-/hai/scratch/shenjm/pyextra}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p logs/evaluation
exec /hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python \
  scripts/evaluation/compare_estimators_apmdata.py --device cuda \
  --arms "${ARMS:-none,plm,a_token}" \
  --output "${OUTPUT:-/hai/scratch/shenjm/proteo_aa_runs/apm_testset_eval/estimator_comparison.json}" "$@"
