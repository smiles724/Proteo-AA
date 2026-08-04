#!/bin/bash
#SBATCH --job-name=proteo-aa-s2-sc
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
COMPLEX_PROVIDER="${COMPLEX_PROVIDER:-both}"
PINDER_ROOT="${PINDER_ROOT:-/hai/scratch/yfsun/pinder/2024-02}"
PINDER_MANIFEST="${PINDER_MANIFEST:-${PINDER_ROOT}/indices/pinder_ppi_complex.parquet}"
PINDER_CIF_CACHE="${PINDER_CIF_CACHE:-${PINDER_ROOT}/cif_cache}"
PINDER_ARCHIVE="${PINDER_ARCHIVE:-${PINDER_ROOT}/raw/pdbs.zip}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/stage2_complex_sidechain_warmup/${SLURM_JOB_ID:-manual}}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
BEST_MONOMER_CKPT="${BEST_MONOMER_CKPT:-}"
COMPLEX_PROVIDER_ARGS=(--complex-provider "${COMPLEX_PROVIDER}")
if [[ "${COMPLEX_PROVIDER}" == "pinder" || "${COMPLEX_PROVIDER}" == "both" ]]; then
  COMPLEX_PROVIDER_ARGS+=(
    --pinder-manifest "${PINDER_MANIFEST}"
    --pinder-root "${PINDER_ROOT}"
    --pinder-cif-cache "${PINDER_CIF_CACHE}"
    --pinder-archive "${PINDER_ARCHIVE}"
  )
fi

if [[ -z "${BEST_MONOMER_CKPT}" ]]; then
  echo "ERROR: BEST_MONOMER_CKPT must be specified explicitly." >&2
  exit 2
fi
if [[ ! -f "${BEST_MONOMER_CKPT}" ]]; then
  echo "ERROR: BEST_MONOMER_CKPT does not exist: ${BEST_MONOMER_CKPT}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" scripts/training/train_protenix_monomer.py \
  --training-stage sidechain_warmup \
  --data-mode mixed_monomer_complex \
  "${COMPLEX_PROVIDER_ARGS[@]}" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${RUN_ROOT}" \
  --load-checkpoint "${BEST_MONOMER_CKPT}" \
  --warm-start-params-only \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-64}" \
  --complex-max-n-token "${COMPLEX_MAX_N_TOKEN:-1536}" \
  --complex-max-binder-fraction "${COMPLEX_MAX_BINDER_FRACTION:-0.75}" \
  --stage2-start-monomer-frac "${STAGE2_START_MONOMER_FRAC:-0.90}" \
  --stage2-end-monomer-frac "${STAGE2_END_MONOMER_FRAC:-0.65}" \
  --pinder-complex-frac "${PINDER_COMPLEX_FRAC:-0.5}" \
  --curriculum-stage1-end-step "${CURRICULUM_STAGE1_END_STEP:-0}" \
  --curriculum-stage2-start-step "${CURRICULUM_STAGE2_START_STEP:-10000}" \
  --max-steps "${MAX_STEPS:-30000}" \
  --train-samples-per-epoch "${TRAIN_SAMPLES_PER_EPOCH:-10000}" \
  --lr "${LR:-1e-4}" \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-2000}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --eval-interval "${EVAL_INTERVAL:-1000}" \
  --eval-samples "${EVAL_SAMPLES:-128}" \
  --eval-num-workers "${EVAL_NUM_WORKERS:-0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  --template-provider "${TEMPLATE_PROVIDER:-dunbrack_mode}" \
  "${@}"
