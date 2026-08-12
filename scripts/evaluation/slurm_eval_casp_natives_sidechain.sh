#!/bin/bash
#SBATCH --job-name=proteo-aa-casp-sc
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/validation/casp_sidechain/%x-%j.out
#SBATCH --error=logs/validation/casp_sidechain/%x-%j.err

# Side-chain packing on CASP NATIVES (true backbone + true sequence in, side
# chains reconstructed, side-chain RMSD out), scoring the CASP coordinates
# directly rather than a mapped PDB entry.
#
# Set CASP=15 or CASP=16. Both post-date the pre-2021-09-30 training index, so
# unlike CASP14 these are genuinely held out.

set -euo pipefail

source ~/.bashrc
conda activate ml

CASP="${CASP:-15}"
REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
CASP_DIR="${CASP_DIR:-/hai/scratch/yfsun/casp${CASP}}"
PDB_GLOB="${PDB_GLOB:-${CASP_DIR}/T/**/*.pdb}"
CIF_DIR="${CIF_DIR:-${CASP_DIR}/cif}"
MANIFEST="${MANIFEST:-${CASP_DIR}/manifest.json}"
CHECKPOINT="${CHECKPOINT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/frame_residual_resolvedmask_from_bb_step96000/checkpoints/step12000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${CASP_DIR}/eval_$(basename "${CHECKPOINT}" .pt)}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
CROP_SIZE="${CROP_SIZE:-1536}"
DTYPE="${DTYPE:-bf16}"
# CASP15 natives were released in 2022; CASP16 in 2024. Only used to fill the
# placeholder pdbx_audit_revision_history the parser demands.
REVISION_DATE="${REVISION_DATE:-2022-12-20}"

mkdir -p "${REPO_ROOT}/logs/validation/casp_sidechain" "${OUTPUT_DIR}" "${CIF_DIR}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== CASP${CASP}: converting natives to parseable mmCIF ==="
"${PYTHON_BIN}" -u scripts/evaluation/casp_natives_to_cif.py \
  --pdb-glob "${PDB_GLOB}" \
  --out-dir "${CIF_DIR}" \
  --manifest "${MANIFEST}" \
  --revision-date "${REVISION_DATE}" \
  --max-residues "$(( CROP_SIZE - 64 ))"

echo "=== CASP${CASP}: scoring side-chain packing ==="
"${PYTHON_BIN}" -u scripts/evaluation/eval_casp14_sidechain.py \
  --manifest "${MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --crop-size "${CROP_SIZE}" \
  --dtype "${DTYPE}" \
  --device cuda \
  "${@}"
