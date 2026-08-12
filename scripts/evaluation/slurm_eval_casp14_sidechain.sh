#!/bin/bash
#SBATCH --job-name=proteo-aa-casp14-sc
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/validation/casp14_sidechain/%x-%j.out
#SBATCH --error=logs/validation/casp14_sidechain/%x-%j.err

# CASP14 side-chain packing benchmark: true backbone + true sequence in,
# side chains reconstructed, side-chain RMSD out. See
# scripts/evaluation/eval_casp14_sidechain.py for the protocol and its caveats
# (notably: CASP14 natives are likely INSIDE the pre-2021-09-30 training index).

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
CASP14_DIR="${CASP14_DIR:-/hai/scratch/yfsun/casp14}"
MAPPING="${MAPPING:-${CASP14_DIR}/map.json}"
CHECKPOINT="${CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/frame_residual_resolvedmask_from_bb_step96000/checkpoints/step12000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${CASP14_DIR}/eval_$(basename "${CHECKPOINT}" .pt)}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
CROP_SIZE="${CROP_SIZE:-1536}"
DTYPE="${DTYPE:-bf16}"

if [[ ! -f "${MAPPING}" ]]; then
  echo "ERROR: target->PDB mapping does not exist: ${MAPPING}" >&2
  echo "       build it first (see eval_casp14_sidechain.py docstring)." >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs/validation/casp14_sidechain" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# -u so progress is not lost to block buffering when stdout is a file.
"${PYTHON_BIN}" -u scripts/evaluation/eval_casp14_sidechain.py \
  --mapping "${MAPPING}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --mmcif-dir "${DATA_ROOT}/mmcif" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --crop-size "${CROP_SIZE}" \
  --dtype "${DTYPE}" \
  --device cuda \
  --report-train-overlap \
  "${@}"
