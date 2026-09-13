#!/usr/bin/env bash
#SBATCH --job-name=sc-repair-arm-f
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/sc-repair-arm-f-%j.out
#SBATCH --error=logs/training/sc-repair-arm-f-%j.err
set -euo pipefail

REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases}
PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
DONOR=${SC_REPAIR_DONOR:-/hai/scratch/yfsun/proteo_aa_runs/official_sc_rigid_warmup/114967/checkpoints/step46000.pt}
CALIBRATION_DIR=${SC_REPAIR_CALIBRATION_DIR:-$REPO/runs/sc_geometry_repair/calibration_v1}
PREREGISTRATION=${SC_REPAIR_V2_PREREGISTRATION:-$REPO/runs/sc_geometry_repair/sweep_v2_arm_f/preregistered_gate.json}
OUTPUT=${OUTPUT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/sc_geometry_repair/strength_v2_arm_f_${SLURM_JOB_ID}/arm_F}

read -r BOND_SC BOND_ATTACH ANGLE_SC ANGLE_ATTACH < <(
  "$PYTHON_BIN" -c 'import json,sys; w=json.load(open(sys.argv[1]))["arms"]["F"]["weights"]; print(w["bond_sc"],w["bond_attach"],w["angle_sc"],w["angle_attach"])' \
    "$PREREGISTRATION"
)
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix:${FAMPNN_ROOT:-/hai/users/y/f/yfsun/Protein Project/fampnn}"
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export OMP_NUM_THREADS=4 LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$REPO"
exec "$PYTHON_BIN" scripts/training/train_sc_adaptation.py \
  --accepted-checkpoint "$DONOR" --phase sc_geometry_repair --donor-weights ema \
  --repair-arm F --output-dir "$OUTPUT" \
  --calibration-path "$CALIBRATION_DIR/calibration.yaml" \
  --source-index "$CALIBRATION_DIR/train.csv.gz" \
  --eval-source-index "$CALIBRATION_DIR/validation.csv.gz" \
  --final-test-index "$CALIBRATION_DIR/final_test.csv.gz" \
  --max-steps 2000 --sc-lr 1e-5 --warmup-steps 100 --geometry-ramp-steps 200 \
  --accumulation 8 --eval-interval 500 --checkpoint-interval 500 --eval-samples 491 \
  --weight-bond-sc "$BOND_SC" --weight-bond-attach "$BOND_ATTACH" \
  --weight-angle-sc "$ANGLE_SC" --weight-angle-attach "$ANGLE_ATTACH"
