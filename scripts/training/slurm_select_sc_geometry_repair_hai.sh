#!/usr/bin/env bash
#SBATCH --job-name=sc-repair-select
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/training/sc-repair-select-%j.out
#SBATCH --error=logs/training/sc-repair-select-%j.err
set -euo pipefail

REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases}
PYTHON_BIN=${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}
RUN_ROOT=${SC_REPAIR_RUN_ROOT:?Set SC_REPAIR_RUN_ROOT to the directory containing arm_A/B/C}
DONOR_BASELINE=${SC_REPAIR_DONOR_BASELINE:?Set SC_REPAIR_DONOR_BASELINE to donor_baseline.json}
cd "$REPO"
exec "$PYTHON_BIN" scripts/utilities/select_sc_geometry_repair.py \
  --arm-a "$RUN_ROOT/arm_A" --arm-b "$RUN_ROOT/arm_B" --arm-c "$RUN_ROOT/arm_C" \
  --donor-baseline "$DONOR_BASELINE" \
  --output "$RUN_ROOT/acceptance.json"
