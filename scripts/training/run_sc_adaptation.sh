#!/usr/bin/env bash
# Portable launcher. Cluster wrappers supply resources; all settings are explicit.
set -euo pipefail
REPO=${PROTEOAA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
PHASE=${SC_PHASE:?Set SC_PHASE=sc_geometry_repair, sc_complex_adapt, or sc_adapt}
OUTPUT_DIR=${OUTPUT_DIR:?Set a distinct OUTPUT_DIR}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:${FAMPNN_ROOT:-$REPO/../Protein Project/fampnn}${PYTHONPATH:+:$PYTHONPATH}"
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
export PROTENIX_DATA_ROOT_DIR=${PROTENIX_DATA_ROOT_DIR:-$PROTENIX_ROOT_DIR/common}
export LAYERNORM_TYPE=torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
CHECKPOINT_ARGS=()
if [[ -n ${RESUME_CHECKPOINT:-} ]]; then
  CHECKPOINT_ARGS=(--resume-checkpoint "$RESUME_CHECKPOINT")
else
  : "${ACCEPTED_CHECKPOINT:?Set the accepted preceding-phase integrated checkpoint}"
  CHECKPOINT_ARGS=(--accepted-checkpoint "$ACCEPTED_CHECKPOINT" --phase "$PHASE")
  if [[ "$PHASE" == sc_complex_adapt ]]; then
    : "${SC_REPAIR_ACCEPTANCE:?Set SC_REPAIR_ACCEPTANCE to the accepted repair decision}"
    : "${SC_REPAIR_FINAL_TEST:?Set SC_REPAIR_FINAL_TEST to the completed post-selection final-test artifact}"
    CHECKPOINT_ARGS+=(--repair-acceptance "$SC_REPAIR_ACCEPTANCE" --repair-final-test "$SC_REPAIR_FINAL_TEST")
  fi
fi
exec "$PYTHON_BIN" "$REPO/scripts/training/train_sc_adaptation.py" \
  "${CHECKPOINT_ARGS[@]}" --output-dir "$OUTPUT_DIR" "$@"
