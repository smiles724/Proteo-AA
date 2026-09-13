#!/usr/bin/env bash
#SBATCH --job-name=sc-repair-v2-select
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/training/sc-repair-v2-select-%j.out
#SBATCH --error=logs/training/sc-repair-v2-select-%j.err
set -euo pipefail

REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases}
PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
RUN_ROOT=${SC_REPAIR_V2_RUN_ROOT:?Set SC_REPAIR_V2_RUN_ROOT to the directory containing screen.json}
PREREGISTRATION=${SC_REPAIR_V2_PREREGISTRATION:-$REPO/runs/sc_geometry_repair/sweep_v2/preregistered_gate.json}
ARM_B=${SC_REPAIR_ARM_B:-/hai/scratch/yfsun/proteo_aa_runs/sc_geometry_repair/arms_115668/arm_B}
DONOR_BASELINE=${SC_REPAIR_DONOR_BASELINE:-/hai/scratch/yfsun/proteo_aa_runs/sc_geometry_repair/donor_baseline/donor_baseline.json}
ARM=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["arm"])' "$RUN_ROOT/screen.json")
cd "$REPO"
exec "$PYTHON_BIN" scripts/utilities/select_sc_geometry_repair_v2.py final \
  --preregistration "$PREREGISTRATION" --screen "$RUN_ROOT/screen.json" \
  --arm-b "$ARM_B" --donor-baseline "$DONOR_BASELINE" \
  --extension "$RUN_ROOT/arm_${ARM}_extended" --output "$RUN_ROOT/acceptance_v2.json"
