#!/bin/bash
#SBATCH --job-name=sc-input-audit
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sc-input-audit-%j.out
#SBATCH --error=logs/sc-input-audit-%j.err
set -euo pipefail
REPO=${PROTEOAA_REPO:-$SLURM_SUBMIT_DIR}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:/hai/users/y/f/yfsun/Protein Project/fampnn"
export PROTENIX_ROOT_DIR=/hai/scratch/yfsun/protenix_data
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export CUDA_VISIBLE_DEVICES="" LAYERNORM_TYPE=torch OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$REPO"
/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python scripts/utilities/audit_native_sc_inputs.py \
  --index /hai/scratch/yfsun/proteo_aa_runs/official_sc_warmup/114932/cache/protenix_monomer_index.csv.gz \
  --output "runs/sc_input_audit/$SLURM_JOB_ID.json" "$@"
