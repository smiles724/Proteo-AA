#!/bin/bash
#SBATCH --job-name=apm-eval-all
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/evaluation/%x-%j.out
#SBATCH --error=logs/evaluation/%x-%j.err
#
# Everything on APM's 449 post-2021 held-out monomers, through ONE estimator.
#
#   apm_ckpt            APM's released sidechain_model.ckpt
#   apmdata_{none,plm}  retrained here on APM data with APM's objective
#   ours_{none,a_token,plm,both}   the earlier arms, trained on PXDesign data
#   dunbrack_template   backbone-conditioned rotamer lookup
#
# One estimator matters more than it sounds: an earlier round compared a
# per-residue mean without symmetry alignment against a symmetry-aligned global
# RMSD and concluded the packer lost to a lookup table. It does not.
#
# Each model is run with the diffuse_mask convention it was TRAINED under --
# that is a network input, not bookkeeping.
set -euo pipefail
REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:${APM_REFERENCE:-/hai/scratch/shenjm/apm_reference}:${PYEXTRA:-/hai/scratch/shenjm/pyextra}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p logs/evaluation

exec /hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python \
  scripts/evaluation/eval_on_apm_testset.py --device cuda \
  --methods "${METHODS:-apm_ckpt,apmdata_none,apmdata_plm,ours_none,ours_a_token,ours_plm,ours_both,dunbrack_template}" \
  --apm-data-run "${APM_DATA_RUN:-/hai/scratch/shenjm/proteo_aa_runs/packer_apm_data}" \
  --apm-data-ckpt "${APM_DATA_CKPT:-final.pt}" \
  --output "${OUTPUT:-/hai/scratch/shenjm/proteo_aa_runs/apm_testset_eval/all_methods.json}" "$@"
