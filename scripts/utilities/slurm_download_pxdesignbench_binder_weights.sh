#!/bin/bash
#SBATCH --job-name=pxd-binder-weights
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/setup/%x-%j.out
#SBATCH --error=logs/setup/%x-%j.err

# Download only the large files required by binder designability validation.
# AF2 parameters, ProteinMPNN weights, and all temporary data live on scratch.
# ESMFold is intentionally omitted: PXDesignBench binder evaluation does not use it.

set -euo pipefail

TOOL_WEIGHTS_ROOT="${TOOL_WEIGHTS_ROOT:-/hai/scratch/shenjm/pxdesign_tool_weights}"
SCRATCH_TMP_ROOT="${SCRATCH_TMP_ROOT:-/hai/scratch/shenjm/tmp}"
AF2_DIR="${TOOL_WEIGHTS_ROOT}/af2"
MPNN_DIR="${TOOL_WEIGHTS_ROOT}/mpnn"
AF2_URL="${AF2_URL:-https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar}"
MPNN_REPO="${MPNN_REPO:-https://github.com/dauparas/ProteinMPNN.git}"

mkdir -p "${AF2_DIR}" "${MPNN_DIR}" "${SCRATCH_TMP_ROOT}"
task_tmp="$(mktemp -d "${SCRATCH_TMP_ROOT}/pxd-binder-weights.${SLURM_JOB_ID:-manual}.XXXXXX")"
cleanup() {
  rm -rf "${task_tmp}"
}
trap cleanup EXIT

echo "TOOL_WEIGHTS_ROOT=${TOOL_WEIGHTS_ROOT}"
echo "temporary_directory=${task_tmp}"
df -h "${TOOL_WEIGHTS_ROOT}" "${SCRATCH_TMP_ROOT}"

# The official archive contains five standard, five PTM, and five multimer-v3
# parameter files. Reuse it when all 15 existing files are non-empty.
af2_count="$(find "${AF2_DIR}" -maxdepth 1 -type f -name 'params_model*.npz' -size +100M | wc -l)"
if [[ "${af2_count}" -eq 15 ]]; then
  echo "AF2 weights already complete: 15/15; skipping download"
else
  echo "AF2 weights incomplete (${af2_count}/15); downloading archive on CPU node"
  af2_tar="${task_tmp}/alphafold_params_2022-12-06.tar"
  curl --fail --location --retry 5 --retry-delay 10 \
    "${AF2_URL}" --output "${af2_tar}"
  tar -xf "${af2_tar}" -C "${AF2_DIR}"
fi

# ProteinMPNN is small compared with AF2/ESMFold, but clone it in scratch so
# neither Git objects nor transient checkout data consume the 50-GiB home quota.
required_mpnn=(
  "${MPNN_DIR}/vanilla_model_weights/v_48_020.pt"
  "${MPNN_DIR}/soluble_model_weights/v_48_020.pt"
  "${MPNN_DIR}/ca_model_weights/v_48_020.pt"
)
mpnn_complete=1
for path in "${required_mpnn[@]}"; do
  [[ -s "${path}" ]] || mpnn_complete=0
done
if [[ "${mpnn_complete}" -eq 1 ]]; then
  echo "ProteinMPNN weights already complete; skipping clone"
else
  echo "ProteinMPNN weights incomplete; cloning into scratch temporary directory"
  git clone --depth 1 "${MPNN_REPO}" "${task_tmp}/ProteinMPNN"
  for subdir in ca_model_weights soluble_model_weights vanilla_model_weights; do
    mkdir -p "${MPNN_DIR}/${subdir}"
    cp -a "${task_tmp}/ProteinMPNN/${subdir}/." "${MPNN_DIR}/${subdir}/"
  done
fi

af2_count="$(find "${AF2_DIR}" -maxdepth 1 -type f -name 'params_model*.npz' -size +100M | wc -l)"
[[ "${af2_count}" -eq 15 ]] || {
  echo "ERROR: expected 15 AF2 parameter files, found ${af2_count}" >&2
  exit 2
}
for path in "${required_mpnn[@]}"; do
  [[ -s "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done

echo "Binder scoring weights are complete"
du -sh "${AF2_DIR}" "${MPNN_DIR}"
