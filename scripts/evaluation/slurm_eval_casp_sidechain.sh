#!/bin/bash
# Side-chain packing benchmark on a CASP edition: true backbone + true sequence
# in, side chains reconstructed, side-chain RMSD out.
#
# Generalises slurm_eval_casp14_sidechain.sh over the edition. Uses the MANIFEST
# path, which eval_casp14_sidechain.py documents as preferred and which works for
# 14/15/16 alike -- it scores the CASP natives directly, so no target->PDB
# mapping is needed (only CASP14 has one locally anyway).
#
#   CASP=15 CHECKPOINT=/path/step50000.pt sbatch scripts/evaluation/slurm_eval_casp_sidechain.sh
#
#SBATCH --job-name=proteo-aa-casp-sc
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/validation/casp_sidechain/%x-%j.out
#SBATCH --error=logs/validation/casp_sidechain/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
CASP="${CASP:?set CASP to 14, 15 or 16}"
CASP_DIR="${CASP_DIR:-/hai/scratch/yfsun/casp${CASP}}"
MANIFEST="${MANIFEST:-${CASP_DIR}/manifest.json}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
TAG="${TAG:-$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")_$(basename "${CHECKPOINT}" .pt)}"
OUTPUT_DIR="${OUTPUT_DIR:-${CASP_DIR}/eval_${TAG}}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
CROP_SIZE="${CROP_SIZE:-1536}"
DTYPE="${DTYPE:-bf16}"

for f in "${MANIFEST}" "${CHECKPOINT}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done

mkdir -p "${REPO_ROOT}/logs/validation/casp_sidechain" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "CASP${CASP}  manifest=${MANIFEST}"
echo "checkpoint=${CHECKPOINT}"
echo "output=${OUTPUT_DIR}"

"${PYTHON_BIN}" -u scripts/evaluation/eval_casp14_sidechain.py \
  --manifest "${MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --mmcif-dir "${DATA_ROOT}/mmcif" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --crop-size "${CROP_SIZE}" \
  --dtype "${DTYPE}" \
  --device cuda \
  "${@}"
