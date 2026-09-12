#!/bin/bash
#SBATCH --job-name=alpha10-pxd-smoke
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/validation/alphaproteo10/%x-%j.out
#SBATCH --error=logs/validation/alphaproteo10/%x-%j.err

# Generation-only smoke test of the official PXDesign backbone on all ten
# AlphaProteo targets.  This deliberately does not pretend that the untrained
# Proteo-AA head in an official checkpoint is a sequence model: binders are
# written as poly-Gly backbones for a later ProteinMPNN stage.

set -euo pipefail

source /hai/users/s/h/shenjm/miniconda3/etc/profile.d/conda.sh
conda activate proteoaa

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
CHECKPOINT="${CHECKPOINT:-/hai/scratch/shenjm/pxdesign_official/pxdesign_v0.1.0.pt}"
TARGET_DIR="${TARGET_DIR:-${REPO_ROOT}/benchmarks/alphaproteo10/targets}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/alphaproteo10_pxdesign_smoke/${SLURM_JOB_ID:-manual}}"
N_STEP="${N_STEP:-400}"
SEED="${SEED:-42}"
BINDER_LENGTH="${BINDER_LENGTH:-105}"

[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint ${CHECKPOINT}" >&2; exit 2; }
[[ -d "${TARGET_DIR}" ]] || { echo "ERROR: missing target directory ${TARGET_DIR}" >&2; exit 2; }

mkdir -p "${REPO_ROOT}/logs/validation/alphaproteo10" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/PXDesign:${REPO_ROOT}/Protenix${PYTHONPATH:+:${PYTHONPATH}}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

targets=(bhrf1 h1 il17a il7ra ir pdl1 sc2rbd tnfa trka vegfa)
for i in "${!targets[@]}"; do
  target="${targets[$i]}"
  echo "===== target=${target} seed=$((SEED + i)) n_step=${N_STEP} binder_length=${BINDER_LENGTH} ====="
  python scripts/evaluation/design_binder_from_target.py \
    --target "${TARGET_DIR}/${target}.yaml" \
    --checkpoint "${CHECKPOINT}" \
    --out "${RUN_ROOT}" \
    --n-step "${N_STEP}" \
    --binder-length "${BINDER_LENGTH}" \
    --device cuda \
    --seed "$((SEED + i))" \
    --sampler-mode pxdesign_native \
    --backbone-only
done

echo "PXDesign backbone smoke complete: ${RUN_ROOT}"
