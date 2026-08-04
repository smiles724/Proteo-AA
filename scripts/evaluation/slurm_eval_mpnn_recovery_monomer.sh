#!/bin/bash
#SBATCH --job-name=proteo-aa-mpnnrec
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
CHECKPOINT="${CHECKPOINT:-}"
PROTEINMPNN_DIR="${PROTEINMPNN_DIR:-}"
COORD_SOURCE="${COORD_SOURCE:-generated}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/mpnn_recovery_monomer/${SLURM_JOB_ID:-manual}}"

if [[ "${COORD_SOURCE}" != "generated" && "${COORD_SOURCE}" != "gt" ]]; then
  echo "ERROR: COORD_SOURCE must be 'generated' or 'gt'." >&2
  exit 2
fi
if [[ "${COORD_SOURCE}" == "generated" && -z "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT must be specified explicitly when COORD_SOURCE=generated." >&2
  exit 2
fi
if [[ -n "${CHECKPOINT}" && ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ -z "${PROTEINMPNN_DIR}" ]]; then
  echo "ERROR: PROTEINMPNN_DIR must point to an official ProteinMPNN checkout." >&2
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

EXPORT_DIR="${RUN_ROOT}/proteo_backbones"
MPNN_DIR="${RUN_ROOT}/proteinmpnn"

EXPORT_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  EXPORT_ARGS+=(--checkpoint "${CHECKPOINT}")
fi

"${PYTHON_BIN}" scripts/evaluation/export_protenix_backbones_for_mpnn.py \
  "${EXPORT_ARGS[@]}" \
  --output-dir "${EXPORT_DIR}" \
  --coord-source "${COORD_SOURCE}" \
  --data-root "${DATA_ROOT}" \
  --protenix-code-dir "${PROTENIX_CODE_DIR}" \
  --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
  --crop-size "${CROP_SIZE:-640}" \
  --max-n-token "${MAX_N_TOKEN:-640}" \
  --max-crop-retries "${MAX_CROP_RETRIES:-16}" \
  --limit-index "${LIMIT_INDEX:-0}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --sample-select "${SAMPLE_SELECT:-lowest_sigma}" \
  --sample-index "${SAMPLE_INDEX:-0}" \
  --dummy-resname "${DUMMY_RESNAME:-ALA}" \
  --dtype "${DTYPE:-bf16}" \
  --device cuda \
  --seed "${SEED:-20260728}"

"${PYTHON_BIN}" scripts/evaluation/run_proteinmpnn_recovery.py \
  --manifest "${EXPORT_DIR}/backbone_manifest.csv" \
  --proteinmpnn-dir "${PROTEINMPNN_DIR}" \
  --output-dir "${MPNN_DIR}" \
  --proteinmpnn-python "${PROTEINMPNN_PYTHON:-${PYTHON_BIN}}" \
  --num-seq-per-target "${NUM_SEQ_PER_TARGET:-1}" \
  --sampling-temp "${SAMPLING_TEMP:-0.1}" \
  --batch-size "${MPNN_BATCH_SIZE:-1}" \
  --device "${MPNN_DEVICE:-}" \
  --seed "${SEED:-20260728}" \
  --skip-existing \
  "${@}"

echo "export_manifest=${EXPORT_DIR}/backbone_manifest.csv"
echo "mpnn_metrics=${MPNN_DIR}/proteinmpnn_recovery_metrics.csv"
echo "mpnn_summary=${MPNN_DIR}/proteinmpnn_recovery_summary.json"
