#!/bin/bash
#SBATCH --job-name=install-pxdbench
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/setup/%x-%j.out
#SBATCH --error=logs/setup/%x-%j.err

# Install the PXDesignBench software environment in the user's normal conda
# env directory. Large package caches and temporary downloads stay on scratch.
# Model weights are handled separately by
# slurm_download_pxdesignbench_binder_weights.sh.

set -euo pipefail

PXDBENCH_DIR="${PXDBENCH_DIR:-/hai/users/s/h/shenjm/tools/PXDesignBench-v0.1.2}"
CONDA_BASE="${CONDA_BASE:-/hai/users/s/h/shenjm/miniconda3}"
ENV_NAME="${ENV_NAME:-pxdbench}"
ENV_PATH="${CONDA_BASE}/envs/${ENV_NAME}"
CACHE_ROOT="${CACHE_ROOT:-/hai/scratch/shenjm/pxdesign_install_cache}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"

[[ -f "${PXDBENCH_DIR}/install.sh" ]] || {
  echo "ERROR: missing ${PXDBENCH_DIR}/install.sh" >&2
  exit 2
}
mkdir -p "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/conda-pkgs"
export TMPDIR="${CACHE_ROOT}/tmp"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda-pkgs"
export PATH="${CONDA_BASE}/bin:${PATH}"

if [[ -x "${ENV_PATH}/bin/python" ]]; then
  if "${ENV_PATH}/bin/python" -c 'import torch, jax, colabdesign, protenix, pxdbench'; then
    echo "PXDesignBench environment is already complete: ${ENV_PATH}"
    exit 0
  fi
  echo "ERROR: ${ENV_PATH} exists but its import check failed." >&2
  echo "Remove or repair that partial environment before rerunning." >&2
  exit 2
fi

cd "${PXDBENCH_DIR}"
bash install.sh \
  --env "${ENV_NAME}" \
  --pkg_manager conda \
  --cuda-version "${CUDA_VERSION}"

"${ENV_PATH}/bin/python" - <<'PY'
import torch
import jax
import colabdesign
import protenix
import pxdbench
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("jax", jax.__version__)
print("PXDesignBench import check: OK")
PY

echo "environment=${ENV_PATH}"
echo "large_cache=${CACHE_ROOT}"
