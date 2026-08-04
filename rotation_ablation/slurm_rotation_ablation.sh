#!/bin/bash
#SBATCH --job-name=proteo-aa-rot
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate "${CONDA_ENV:-ml}"

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/rotation_ablation/${SLURM_JOB_ID:-manual}}"

CHECKPOINT="${CHECKPOINT:-}"
BUNDLE="${BUNDLE:-}"
SMOKE="${SMOKE:-0}"
N_ROTATIONS="${N_ROTATIONS:-20}"
SEED="${SEED:-0}"
N_HEADS="${N_HEADS:-16}"
SYNTHETIC_LENGTH="${SYNTHETIC_LENGTH:-6}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-10.0}"
INVARIANT_TOLERANCE="${INVARIANT_TOLERANCE:-1e-4}"
COORDINATE_TOLERANCE="${COORDINATE_TOLERANCE:-1e-3}"

if [[ "${SMOKE}" != "1" && -z "${CHECKPOINT}" ]]; then
  echo "ERROR: set CHECKPOINT=/path/to/sidechain_checkpoint.pt, or SMOKE=1." >&2
  exit 2
fi
if [[ -n "${CHECKPOINT}" && ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ -n "${BUNDLE}" && ! -f "${BUNDLE}" ]]; then
  echo "ERROR: bundle does not exist: ${BUNDLE}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}"
cd "${REPO_ROOT}"

export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --device cuda
  --n-rotations "${N_ROTATIONS}"
  --seed "${SEED}"
  --n-heads "${N_HEADS}"
  --synthetic-length "${SYNTHETIC_LENGTH}"
  --translation-scale "${TRANSLATION_SCALE}"
  --invariant-tolerance "${INVARIANT_TOLERANCE}"
  --coordinate-tolerance "${COORDINATE_TOLERANCE}"
  --output "${RUN_ROOT}/results.json"
)

if [[ -n "${CHECKPOINT}" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${BUNDLE}" ]]; then
  args+=(--bundle "${BUNDLE}")
fi

{
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "host=$(hostname)"
  echo "repo_root=${REPO_ROOT}"
  echo "checkpoint=${CHECKPOINT:-none}"
  echo "bundle=${BUNDLE:-synthetic}"
  echo "n_rotations=${N_ROTATIONS}"
  echo "seed=${SEED}"
  echo "output=${RUN_ROOT}/results.json"
} | tee "${RUN_ROOT}/run_metadata.txt"

"${PYTHON_BIN}" -m rotation_ablation "${args[@]}" "$@"

echo "results=${RUN_ROOT}/results.json"
