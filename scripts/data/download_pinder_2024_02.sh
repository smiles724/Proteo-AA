#!/bin/bash
set -euo pipefail

# Paths derive from PROTEOAA_DATA_ROOT so this runs unmodified elsewhere.
# See docs/datasets.md.
PROTEOAA_DATA_ROOT="${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}"
PINDER_RELEASE="${PINDER_RELEASE:-2024-02}"
PINDER_ROOT="${PINDER_ROOT:-${PROTEOAA_DATA_ROOT}/pinder/${PINDER_RELEASE}}"
BASE_URL="${PINDER_BASE_URL:-https://storage.googleapis.com/pinder/${PINDER_RELEASE}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

mkdir -p \
  "${PINDER_ROOT}/raw" \
  "${PINDER_ROOT}/indices" \
  "${PINDER_ROOT}/manifests" \
  "${PINDER_ROOT}/pdbs"

curl -fL --retry 20 --retry-delay 5 --continue-at - \
  -o "${PINDER_ROOT}/raw/index.parquet" "${BASE_URL}/index.parquet"
curl -fL --retry 20 --retry-delay 5 --continue-at - \
  -o "${PINDER_ROOT}/raw/metadata.parquet" "${BASE_URL}/metadata.parquet"
curl -fL --retry 20 --retry-delay 5 --continue-at - \
  -o "${PINDER_ROOT}/raw/pdbs.zip" "${BASE_URL}/pdbs.zip"
printf '%s  %s\n' \
  "28e6e2724c1597514c7f6b6774ff32b4" \
  "${PINDER_ROOT}/raw/pdbs.zip" | md5sum --check --status

"${PYTHON_BIN}" scripts/data/prepare_pinder_dataset.py \
  --index "${PINDER_ROOT}/raw/index.parquet" \
  --metadata "${PINDER_ROOT}/raw/metadata.parquet" \
  --output-parquet "${PINDER_ROOT}/indices/pinder_ppi_complex.parquet" \
  --output-csv "${PINDER_ROOT}/indices/pinder_ppi_complex.csv.gz" \
  --summary "${PINDER_ROOT}/manifests/dataset_statistics.json" \
  --min-n-token 16 \
  --max-n-token 1536 \
  --crop-size 640 \
  --max-binder-fraction 0.75

if [[ "${EXTRACT_SELECTED:-0}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/data/extract_pinder_structures.py \
    --archive "${PINDER_ROOT}/raw/pdbs.zip" \
    --manifest "${PINDER_ROOT}/indices/pinder_ppi_complex.parquet" \
    --output-root "${PINDER_ROOT}/pdbs" \
    --summary "${PINDER_ROOT}/manifests/extraction_statistics.json"
fi
