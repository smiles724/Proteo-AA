#!/bin/bash
#SBATCH --job-name=proteo-aa-joint-eval
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
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
PROTEINMPNN_DIR="${PROTEINMPNN_DIR:-/hai/users/y/f/yfsun/tools/ProteinMPNN}"
CHECKPOINT="${CHECKPOINT:-}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/joint_eval/${SLURM_JOB_ID:-manual}}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT must be specified explicitly." >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${PROTEINMPNN_DIR}/protein_mpnn_run.py" ]]; then
  echo "ERROR: protein_mpnn_run.py not found under PROTEINMPNN_DIR=${PROTEINMPNN_DIR}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}:${PROTEINMPNN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

BB_DIR="${RUN_ROOT}/bb_eval"
AA_DIR="${RUN_ROOT}/aa_eval"
MPNN_DIR="${RUN_ROOT}/mpnn_recovery"

echo "checkpoint=${CHECKPOINT}"
echo "run_root=${RUN_ROOT}"

"${PYTHON_BIN}" scripts/evaluation/eval_protenix_monomer.py \
  --checkpoint "${CHECKPOINT}" \
  --training-stage joint \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${BB_DIR}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-16}" \
  --limit-index "${LIMIT_INDEX:-0}" \
  --num-samples "${BB_NUM_SAMPLES:-0}" \
  --num-workers "${BB_NUM_WORKERS:-0}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda

"${PYTHON_BIN}" scripts/evaluation/eval_aa_head_protenix_monomer.py \
  --checkpoints "${CHECKPOINT}" \
  --model-stage joint \
  --select-by "${AA_SELECT_BY:-f1_macro}" \
  --logits-source "${AA_LOGITS_SOURCE:-mean_sample}" \
  --mask-source "${AA_MASK_SOURCE:-design}" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --output-dir "${AA_DIR}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-16}" \
  --limit-index "${LIMIT_INDEX:-0}" \
  --max-samples "${AA_MAX_SAMPLES:-0}" \
  --num-workers "${AA_NUM_WORKERS:-0}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda

CHECKPOINT="${CHECKPOINT}" \
PROTEINMPNN_DIR="${PROTEINMPNN_DIR}" \
RUN_ROOT="${MPNN_DIR}" \
COORD_SOURCE=generated \
LIMIT_INDEX="${LIMIT_INDEX:-0}" \
MAX_SAMPLES="${MPNN_MAX_SAMPLES:-0}" \
CROP_SIZE="${CROP_SIZE:-640}" \
MAX_N_TOKEN="${MAX_N_TOKEN:-640}" \
MAX_CROP_RETRIES="${MAX_CROP_RETRIES:-16}" \
DTYPE="${DTYPE:-bf16}" \
SEED="${SEED:-20260728}" \
bash scripts/evaluation/slurm_eval_mpnn_recovery_monomer.sh

"${PYTHON_BIN}" scripts/evaluation/summarize_joint_checkpoint_eval.py \
  --run-root "${RUN_ROOT}" \
  --checkpoint "${CHECKPOINT}"

echo "bb_metrics=${BB_DIR}/metrics.json"
echo "aa_metrics=${AA_DIR}/aa_head_checkpoint_metrics.csv"
echo "mpnn_summary=${MPNN_DIR}/proteinmpnn/proteinmpnn_recovery_summary.json"
echo "combined_summary=${RUN_ROOT}/combined_eval_summary.json"
