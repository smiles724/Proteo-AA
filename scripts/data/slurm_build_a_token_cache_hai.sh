#!/bin/bash
#SBATCH --job-name=a-token-cache
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/data/%x-%A_%a.out
#SBATCH --error=logs/data/%x-%A_%a.err
#
# Precompute a_token for every chain the a_token / both arms will train and
# validate on. At zero coordinate noise with the orientation pinned, a_token is
# a deterministic function of the structure (verified: repeat and rot-aug both
# ~3e-5 over eight chains), so it is computed once and read back.
#
#   0 = the 18,373 APM PDB-monomer training chains
#   1 = the 449 post-2021 validation chains
#
# Resumable: a chain with a .npy already on disk is skipped.
set -euo pipefail
REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:${APM_REFERENCE:-/hai/scratch/shenjm/apm_reference}:${PYEXTRA:-/hai/scratch/shenjm/pyextra}"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
cd "$REPO"
mkdir -p logs/data

ROOT=${ROOT:-/hai/scratch/shenjm/proteo_aa_runs/a_token_cache}
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) SRC=(--pkl-dir /hai/scratch/yfsun/apm/extracted/data_APM/pdb_monomer)
     OUT="$ROOT/train" ;;
  1) SRC=(--pkl-dir /hai/scratch/shenjm/apm_weights/pdb_test
          --ids-csv /hai/scratch/shenjm/apm_weights/metadata_all/test_set_pdb_ids.csv)
     OUT="$ROOT/val" ;;
  *) echo "unexpected array index" >&2; exit 2 ;;
esac

exec /hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python \
  scripts/data/build_a_token_cache.py "${SRC[@]}" \
  --out "$OUT" --cif-dir "$OUT/cif" --device cuda "$@"
