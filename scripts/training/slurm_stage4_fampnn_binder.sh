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
PROTEO_REPO=${PROTEOAA_REPO:-/users/yfsun/Proteo-AA-stage4-fampnn}
export PROTEOAA_DATA_ROOT=${PROTEOAA_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA}
export PROTEOAA_CODE_ROOT=${PROTEOAA_CODE_ROOT:-/users/yfsun/protein-code}
export FAMPNN_ROOT=${FAMPNN_ROOT:-$PROTEOAA_CODE_ROOT/fampnn}
export FAMPNN_CHECKPOINT=${FAMPNN_CHECKPOINT:-$FAMPNN_ROOT/weights/fampnn_0_3.pt}
PYTHON_BIN=${PYTHON_BIN:-/users/yfsun/.venvs/proteoaa-stage4/bin/python}
: "${STAGE3_CHECKPOINT:?Set STAGE3_CHECKPOINT to the intended complete Stage III donor; no donor is guessed}"
STAGE4_PHASE=${STAGE4_PHASE:-IV-A}
TRAIN_ROUNDS=${TRAIN_ROUNDS:-1}
INFERENCE_ROUNDS=${INFERENCE_ROUNDS:-3}
OUTPUT_DIR=${OUTPUT_DIR:-$PROTEO_REPO/runs/stage4-fampnn-${SLURM_JOB_ID:-dry-run}}
export PROTENIX_ROOT_DIR=$PROTEOAA_DATA_ROOT/protenix_data
export LAYERNORM_TYPE=torch
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROTEO_REPO:$PROTEOAA_CODE_ROOT/Protenix:$PROTEOAA_CODE_ROOT/11/PXDesign:$FAMPNN_ROOT
cd "$PROTEO_REPO"
mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" scripts/utilities/preflight_stage4_fampnn.py \
  --donor "$STAGE3_CHECKPOINT" --fampnn-checkpoint "$FAMPNN_CHECKPOINT" \
  --data-root "$PROTEOAA_DATA_ROOT" --output "$OUTPUT_DIR/provenance.json" \
  --phase "$STAGE4_PHASE" --train-rounds "$TRAIN_ROUNDS" --inference-rounds "$INFERENCE_ROUNDS"
"$PYTHON_BIN" -m pip freeze > "$OUTPUT_DIR/environment.txt"
RUN_OPTIONS=()
if [[ ${1:-} == --dry-run ]]; then
  shift
  RUN_OPTIONS=(--dry-run --device cpu)
else
  module load slurm
  module load nvhpc
  module load cudnn/cuda12/9.3.0.75
  module load mps
fi
"$PYTHON_BIN" scripts/training/train_protenix_monomer.py \
  --training-stage stage4_fampnn --stage4-phase "$STAGE4_PHASE" \
  --stage4-train-rounds "$TRAIN_ROUNDS" --stage4-inference-rounds "$INFERENCE_ROUNDS" \
  --load-checkpoint "$STAGE3_CHECKPOINT" --warm-start-params-only \
  --fampnn-checkpoint "$FAMPNN_CHECKPOINT" \
  --protenix-code-dir "$PROTEOAA_CODE_ROOT/Protenix" --pxdesign-code-dir "$PROTEOAA_CODE_ROOT/11/PXDesign" \
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
