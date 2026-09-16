#!/bin/bash
#SBATCH --job-name=a-token-bridge-check
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/data/%x-%j.out
#SBATCH --error=logs/data/%x-%j.err
#
# Does the APM-pickle -> mmCIF -> Protenix-features -> frozen-trunk path produce
# an a_token that is row-aligned with the packer's residues, and what does
# dropping the coordinate noise to zero actually change?
set -euo pipefail
REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:${APM_REFERENCE:-/hai/scratch/shenjm/apm_reference}:${PYEXTRA:-/hai/scratch/shenjm/pyextra}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
cd "$REPO"
mkdir -p logs/data

exec /hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python \
  scripts/data/build_a_token_cache.py \
  --pkl-dir "${PKL_DIR:-/hai/scratch/yfsun/apm/extracted/data_APM/pdb_monomer}" \
  --out "${OUT:-/hai/scratch/shenjm/proteo_aa_runs/a_token_cache}" \
  --check "${CHECK:-8}" --device cuda "$@"
