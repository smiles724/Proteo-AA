#!/bin/bash
#SBATCH --job-name=proteo-stage4-fampnn
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --account=marlowe-m000137-pm06
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --output=stage4-fampnn-%j.log
# Run with --dry-run directly on the login node; submit only after it succeeds.
# Medium allocations use batch: https://marlowe-research.stanford.edu/documentation/slurm/
# Live partition availability is reported by sinfo; jobs may wait while batch is down.
set -euo pipefail
PROTEO_REPO=${PROTEOAA_REPO:-/users/yfsun/Proteo-AA-official-pxdesign-fampnn}
export PROTEOAA_DATA_ROOT=${PROTEOAA_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA}
export PROTEOAA_CODE_ROOT=${PROTEOAA_CODE_ROOT:-/users/yfsun/protein-code}
export FAMPNN_ROOT=${FAMPNN_ROOT:-$PROTEOAA_CODE_ROOT/fampnn}
export FAMPNN_CHECKPOINT=${FAMPNN_CHECKPOINT:-$FAMPNN_ROOT/weights/fampnn_0_3.pt}
PYTHON_BIN=${PYTHON_BIN:-/users/yfsun/.venvs/proteoaa-stage4/bin/python}
COMPONENT_ARGS=()
if [[ -n ${RESUME_CHECKPOINT:-} ]]; then
  COMPONENT_ARGS=(--resume-checkpoint "$RESUME_CHECKPOINT")
elif [[ -n ${WARM_START_CHECKPOINT:-} ]]; then
  COMPONENT_ARGS=(--warm-start-checkpoint "$WARM_START_CHECKPOINT")
else
  : "${BACKBONE_CHECKPOINT:?Set BACKBONE_CHECKPOINT to official PXDesign weights}"
  COMPONENT_ARGS=(--backbone-checkpoint "$BACKBONE_CHECKPOINT" --fampnn-checkpoint "$FAMPNN_CHECKPOINT")
  if [[ ${SC_INIT:-checkpoint} == scratch ]]; then
    COMPONENT_ARGS+=(--sidechain-init scratch)
  else
    : "${SC_CHECKPOINT:?Set SC_CHECKPOINT to the compatible SC donor}"
    COMPONENT_ARGS+=(--sidechain-checkpoint "$SC_CHECKPOINT")
  fi
fi
STAGE4_PHASE=${STAGE4_PHASE:-sc_adapt}
TRAIN_ROUNDS=${TRAIN_ROUNDS:-0}
INFERENCE_ROUNDS=${INFERENCE_ROUNDS:-0}
OUTPUT_DIR=${OUTPUT_DIR:-$PROTEO_REPO/runs/stage4-fampnn-${SLURM_JOB_ID:-dry-run}}
export PROTENIX_ROOT_DIR=$PROTEOAA_DATA_ROOT/protenix_data
export LAYERNORM_TYPE=torch
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
PROTENIX_CODE_DIR=${PROTENIX_CODE_DIR:-$PROTEO_REPO/Protenix}
PXDESIGN_CODE_DIR=${PXDESIGN_CODE_DIR:-$PROTEO_REPO/PXDesign}
export PYTHONPATH="$PROTEO_REPO:$PROTENIX_CODE_DIR:$PXDESIGN_CODE_DIR:$FAMPNN_ROOT"
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
cd "$PROTEO_REPO"
mkdir -p "$OUTPUT_DIR"
RUN_OPTIONS=()
if [[ ${1:-} == --dry-run ]]; then
  shift
  RUN_OPTIONS=(--dry-run --device cpu)
else
  # Marlowe needs these; the HAI cluster has no module system at all. Load what
  # exists and keep going, instead of failing the job on a missing modulefile.
  if command -v module >/dev/null 2>&1; then
    for _mod in slurm nvhpc cudnn/cuda12/9.3.0.75 mps; do
      module load "$_mod" 2>/dev/null || echo "note: module $_mod unavailable on this cluster" >&2
    done
  fi
fi
TRAIN_ARGS=( \
  --training-stage stage4_fampnn --stage4-phase "$STAGE4_PHASE" \
  --stage4-train-rounds "$TRAIN_ROUNDS" --stage4-inference-rounds "$INFERENCE_ROUNDS" \
  "${COMPONENT_ARGS[@]}" \
  --protenix-code-dir "$PROTENIX_CODE_DIR" --pxdesign-code-dir "$PXDESIGN_CODE_DIR" \
  --data-root "$PROTEOAA_DATA_ROOT/protenix_data" --data-mode mixed_monomer_complex --complex-provider pinder \
  --pinder-root "$PROTEOAA_DATA_ROOT/pinder/2024-02" \
  --pinder-archive "$PROTEOAA_DATA_ROOT/pinder/2024-02/raw/pdbs.zip" \
  --pinder-manifest "$PROTEOAA_DATA_ROOT/pinder/2024-02/indices/pinder_ppi_complex.parquet" \
  --pinder-cif-cache "$OUTPUT_DIR/pinder_cif_cache" \
  --crop-size "${CROP_SIZE:-128}" --max-n-token "${CROP_SIZE:-128}" --complex-max-n-token "${COMPLEX_MAX_N_TOKEN:-640}" \
  --diffusion-batch-size 1 --iters-to-accumulate 1 --dtype bf16 \
  --stage2-start-monomer-frac 0.25 --stage2-end-monomer-frac 0.25 \
  --max-steps "${MAX_STEPS:-100}" --checkpoint-interval 50 --log-interval 1 \
  --eval-interval 50 --eval-samples "${EVAL_SAMPLES:-8}" --eval-num-workers 0 --num-workers 0 \
  --output-dir "$OUTPUT_DIR" "$@" "${RUN_OPTIONS[@]}"
)
"$PYTHON_BIN" scripts/utilities/preflight_stage4_fampnn.py \
  --output "$OUTPUT_DIR/provenance.json" "${TRAIN_ARGS[@]}"
"$PYTHON_BIN" -m pip freeze > "$OUTPUT_DIR/environment.txt"
"$PYTHON_BIN" scripts/training/train_protenix_monomer.py "${TRAIN_ARGS[@]}"
