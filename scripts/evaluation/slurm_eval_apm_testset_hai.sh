#!/bin/bash
#SBATCH --job-name=apm-testset-eval
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluation/%x-%j.out
#SBATCH --error=logs/evaluation/%x-%j.err
#
# APM's released sidechain_model.ckpt and the Dunbrack-mode template, scored on
# APM's own 449 post-2021 held-out monomers, through one estimator.
set -euo pipefail
REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$REPO"
exec /hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python \
  scripts/evaluation/eval_on_apm_testset.py --device cuda \
  --output "${OUTPUT:-/hai/scratch/shenjm/proteo_aa_runs/apm_testset_eval/result.json}" "$@"
