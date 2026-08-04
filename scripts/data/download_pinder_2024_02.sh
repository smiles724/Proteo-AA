#!/bin/bash
set -euo pipefail

PINDER_ROOT="${PINDER_ROOT:-/hai/scratch/yfsun/pinder/2024-02}"
BASE_URL="https://storage.googleapis.com/pinder/2024-02"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"

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
