#!/bin/bash
#SBATCH --job-name=pinder-aa-readout
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/validation/pinder_binder_backbone_inputs/%x-%j.out
#SBATCH --error=logs/validation/pinder_binder_backbone_inputs/%x-%j.err

set -euo pipefail

source /hai/users/s/h/shenjm/miniconda3/etc/profile.d/conda.sh
conda activate proteoaa

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
CHECKPOINT="${CHECKPOINT:-/hai/scratch/shenjm/proteo_aa_runs/stage3_binder_coevolution/107904/checkpoints/step5000.pt}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/pinder_aa_readout/${SLURM_JOB_ID:-manual}}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-${REPO_ROOT}/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-${REPO_ROOT}/PXDesign}"

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"
export PROTENIX_ROOT_DIR="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

/hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python \
  scripts/evaluation/infer_aa_readouts_pinder.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}" \
  --pinder-root "${PINDER_ROOT:-/hai/scratch/yfsun/pinder/2024-02}" \
  --pinder-index-csv "${PINDER_INDEX_CSV:-/hai/scratch/yfsun/pinder/2024-02/indices/pinder_ppi_complex.csv.gz}" \
  --pinder-cif-cache "${PINDER_CIF_CACHE:-/hai/scratch/shenjm/pinder/cif_cache}" \
  --pinder-pdb-cache "${PINDER_PDB_CACHE:-/hai/scratch/shenjm/pinder/2024-02/pdbs}" \
  --pinder-archive "${PINDER_ARCHIVE:-/hai/scratch/yfsun/pinder/2024-02/raw/pdbs.zip}" \
  --crop-size "${CROP_SIZE:-448}" \
  --max-samples "${MAX_SAMPLES:-64}" \
  --n-step "${N_STEP:-20}" \
  --sampler-mode "${SAMPLER_MODE:-pxdesign_native}" \
  --aa-readout-sigma "${AA_READOUT_SIGMA:-0.4}" \
  --seed "${SEED:-42}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda
