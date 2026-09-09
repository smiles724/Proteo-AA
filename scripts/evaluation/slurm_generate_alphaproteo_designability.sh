#!/bin/bash
#SBATCH --job-name=alpha10-generate
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/validation/alphaproteo10/%x-%j.out
#SBATCH --error=logs/validation/alphaproteo10/%x-%j.err

set -euo pipefail

source /hai/users/s/h/shenjm/miniconda3/etc/profile.d/conda.sh
conda activate proteoaa

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
MODEL_LABEL="${MODEL_LABEL:?set MODEL_LABEL}"
MODEL_MODE="${MODEL_MODE:?set MODEL_MODE to pxdesign or proteoaa}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT shared by all benchmark stages}"

mkdir -p "${REPO_ROOT}/logs/validation/alphaproteo10" "${RUN_ROOT}/generation"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/PXDesign:${REPO_ROOT}/Protenix${PYTHONPATH:+:${PYTHONPATH}}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

fixed_length_args=()
if [[ -n "${FIXED_LENGTH:-}" ]]; then
  fixed_length_args=(--fixed-length "${FIXED_LENGTH}")
fi

python scripts/evaluation/generate_alphaproteo_designs.py \
  --checkpoint "${CHECKPOINT}" \
  --model-label "${MODEL_LABEL}" \
  --mode "${MODEL_MODE}" \
  --targets "${TARGETS:-bhrf1,h1,il17a,il7ra,ir,pdl1,sc2rbd,tnfa,trka,vegfa}" \
  --output-root "${RUN_ROOT}/generation" \
  --num-designs-per-target "${NUM_DESIGNS_PER_TARGET:-1}" \
  --length-min "${LENGTH_MIN:-80}" \
  --length-max "${LENGTH_MAX:-130}" \
  --seed "${SEED:-42}" \
  --n-step "${N_STEP:-400}" \
  --device cuda \
  --sampler-mode pxdesign_native \
  --aa-readout-sigma "${AA_READOUT_SIGMA:-0.4}" \
  --aa-readouts "${AA_READOUTS:-final,target_sigma,confidence_best}" \
  "${fixed_length_args[@]}"
