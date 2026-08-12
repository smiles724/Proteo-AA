#!/bin/bash
#SBATCH --job-name=proteo-aa-sc-template-legacy
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/validation/sidechain_template_baseline/%x-%j.out
#SBATCH --error=logs/validation/sidechain_template_baseline/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
FILTERED_INDEX="${FILTERED_INDEX:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/frame_residual_bbctx_absconcat_from_bb_step96000/cache/recentPDB_monomer_validation_index.csv.gz}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup/template_baseline_491_include_unresolved}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"

if [[ ! -f "${FILTERED_INDEX}" ]]; then
  echo "ERROR: validation index does not exist: ${FILTERED_INDEX}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "WARNING: running diagnostic-only legacy baseline with unresolved atoms included."
"${PYTHON_BIN}" scripts/evaluation/eval_sidechain_template_baseline_include_unresolved.py \
  --data-root "${DATA_ROOT}" \
  --filtered-index "${FILTERED_INDEX}" \
  --output-dir "${RUN_ROOT}" \
  --num-samples "${NUM_SAMPLES:-491}" \
  --num-workers 0 \
  --template-provider "${TEMPLATE_PROVIDER:-dunbrack_mode}" \
  --sigma-t "${SIGMA_T:-0.3}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}"

echo "wrote_diagnostic_baseline=${RUN_ROOT}/template_baseline.json"
