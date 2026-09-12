#!/bin/bash
#SBATCH --job-name=proteo-fampnn-smoke
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --account=marlowe-m000137-pm06
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --output=/users/yfsun/Proteo-AA-official-pxdesign-fampnn/runs/stage4-smoke-%j.log
set -euo pipefail
# Marlowe defaults. Every path is an override, so the same body runs on the HAI
# cluster too -- see slurm_stage4_fampnn_smoke_hai.sh, which supplies that
# cluster's Slurm directives and paths and then delegates here.
PROTEO_REPO=${PROTEOAA_REPO:-/users/yfsun/Proteo-AA-official-pxdesign-fampnn}
export PROTEOAA_DATA_ROOT=${PROTEOAA_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA}
export PROTEOAA_CODE_ROOT=${PROTEOAA_CODE_ROOT:-/users/yfsun/protein-code}
export FAMPNN_ROOT=${FAMPNN_ROOT:-$PROTEOAA_CODE_ROOT/fampnn}
export FAMPNN_CHECKPOINT=${FAMPNN_CHECKPOINT:-$FAMPNN_ROOT/weights/fampnn_0_3.pt}
PYTHON_BIN=${PYTHON_BIN:-/users/yfsun/.venvs/proteoaa-stage4/bin/python}
# SC donor contributes sidechain_module only; override SC_CHECKPOINT for another compatible packer.
SC_CHECKPOINT=${SC_CHECKPOINT:-$PROTEOAA_DATA_ROOT/proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step6000.pt}
BACKBONE_CHECKPOINT=${BACKBONE_CHECKPOINT:-$PROTEO_REPO/runs/component_donors/pxdesign_v0.1.0.pt}
OUTPUT_DIR=${OUTPUT_DIR:-$PROTEO_REPO/runs/stage4-smoke-${SLURM_JOB_ID:-manual}}
export PROTENIX_ROOT_DIR=$PROTEOAA_DATA_ROOT/protenix_data
export LAYERNORM_TYPE=torch
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
PROTENIX_CODE_DIR=${PROTENIX_CODE_DIR:-$PROTEO_REPO/Protenix}
PXDESIGN_CODE_DIR=${PXDESIGN_CODE_DIR:-$PROTEO_REPO/PXDesign}
export PYTHONPATH="$PROTEO_REPO:$PROTENIX_CODE_DIR:$PXDESIGN_CODE_DIR:$FAMPNN_ROOT"
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
# Marlowe needs these; the HAI cluster has no module system at all. Load what
# exists and keep going, instead of failing the job on a missing modulefile.
if command -v module >/dev/null 2>&1; then
  for _mod in slurm nvhpc cudnn/cuda12/9.3.0.75 mps; do
    module load "$_mod" 2>/dev/null || echo "note: module $_mod unavailable on this cluster" >&2
  done
fi
cd "$PROTEO_REPO"
mkdir -p "$OUTPUT_DIR"
echo "stage4-smoke backbone=$BACKBONE_CHECKPOINT sidechain=$SC_CHECKPOINT"
echo "  fampnn=$FAMPNN_CHECKPOINT"
echo "  output=$OUTPUT_DIR"
"$PYTHON_BIN" scripts/utilities/smoke_stage4_fampnn.py \
  --backbone-checkpoint "$BACKBONE_CHECKPOINT" --sidechain-checkpoint "$SC_CHECKPOINT" \
  --fampnn-checkpoint "$FAMPNN_CHECKPOINT" \
  --data-root "$PROTEOAA_DATA_ROOT" --output "$OUTPUT_DIR" "$@"
