#!/bin/bash
#SBATCH --job-name=ppi-binder-inputs
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
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PINDER_ROOT="${PINDER_ROOT:-/hai/scratch/shenjm/pinder/2024-02}"
PINDER_INDEX_CSV="${PINDER_INDEX_CSV:-/hai/scratch/yfsun/pinder/2024-02/indices/pinder_ppi_complex.csv.gz}"
PINDER_CIF_CACHE="${PINDER_CIF_CACHE:-/hai/scratch/shenjm/pinder/cif_cache}"
PINDER_PDB_CACHE="${PINDER_PDB_CACHE:-/hai/scratch/shenjm/pinder/2024-02/pdbs}"
PINDER_ARCHIVE="${PINDER_ARCHIVE:-/hai/scratch/yfsun/pinder/2024-02/raw/pdbs.zip}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-${REPO_ROOT}/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-${REPO_ROOT}/PXDesign}"
CHECKPOINT="${CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/stage2_complex_backbone/from_monomer_step96000_protenix_pinder/checkpoints/step50000.pt}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/pinder_binder_input_comparison/${SLURM_JOB_ID:-manual}}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python}"

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" scripts/evaluation/eval_pinder_binder_backbone_inputs.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}" \
  --split "${SPLIT:-val}" \
  --pinder-root "${PINDER_ROOT}" \
  --pinder-index-csv "${PINDER_INDEX_CSV}" \
  --pinder-cif-cache "${PINDER_CIF_CACHE}" \
  --pinder-pdb-cache "${PINDER_PDB_CACHE}" \
  --pinder-archive "${PINDER_ARCHIVE}" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  "${@}"
