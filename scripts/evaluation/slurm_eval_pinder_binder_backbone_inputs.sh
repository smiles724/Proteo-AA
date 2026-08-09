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

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PINDER_ROOT="${PINDER_ROOT:-/hai/scratch/yfsun/pinder/2024-02}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
CHECKPOINT="${CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/stage2_complex_backbone/from_monomer_step96000_protenix_pinder/checkpoints/step50000.pt}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/pinder_binder_input_comparison/${SLURM_JOB_ID:-manual}}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"

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
  --pinder-index-csv "${PINDER_ROOT}/indices/pinder_ppi_complex.csv.gz" \
  --pinder-cif-cache "${PINDER_ROOT}/cif_cache" \
  --pinder-archive "${PINDER_ROOT}/raw/pdbs.zip" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  "${@}"
