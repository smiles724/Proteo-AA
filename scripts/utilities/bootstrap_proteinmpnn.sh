#!/bin/bash
# Clone the official ProteinMPNN inference code.
#
# Usage:
#   PROTEINMPNN_DIR=/hai/users/y/f/yfsun/tools/ProteinMPNN bash scripts/utilities/bootstrap_proteinmpnn.sh

set -euo pipefail

PROTEINMPNN_DIR="${PROTEINMPNN_DIR:-/hai/users/y/f/yfsun/tools/ProteinMPNN}"
REPO_URL="${PROTEINMPNN_REPO_URL:-https://github.com/dauparas/ProteinMPNN.git}"

mkdir -p "$(dirname "${PROTEINMPNN_DIR}")"

if [[ -d "${PROTEINMPNN_DIR}/.git" ]]; then
  echo "ProteinMPNN already exists: ${PROTEINMPNN_DIR}"
  git -C "${PROTEINMPNN_DIR}" rev-parse --short HEAD
else
  git clone "${REPO_URL}" "${PROTEINMPNN_DIR}"
fi

if [[ ! -f "${PROTEINMPNN_DIR}/protein_mpnn_run.py" ]]; then
  echo "ERROR: protein_mpnn_run.py not found under ${PROTEINMPNN_DIR}" >&2
  exit 2
fi

echo "ProteinMPNN_DIR=${PROTEINMPNN_DIR}"
echo "ProteinMPNN script=${PROTEINMPNN_DIR}/protein_mpnn_run.py"
find "${PROTEINMPNN_DIR}" -maxdepth 2 -type f -name '*.pt' | sort | head -20
