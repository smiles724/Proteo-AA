#!/bin/bash
#SBATCH --job-name=pxf_codesign_uncond
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_uncond/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_uncond/%x-%j.out
#
# MARLOWE wrapper for codesign_uncond.py. scripts/slurm/codesign_uncond.sh is
# the HAI original and hardcodes /hai paths, so it cannot run here.
#
#   SAMPLES=<dir of L*.cif> OUT=<dir> sbatch scripts/slurm/marlowe/codesign_uncond.sh
#   SAMPLES=... OUT=... ADAPTERS=<coupling ckpt> FAMPNN=0.0 sbatch ...
#
# ADAPTERS and FAMPNN must MATCH. A coupling adapter reads a specific FaMPNN
# representation; couple_phase1 was fit against 0.0 and J03/S03 against 0.3,
# and crossing them loads cleanly while measuring nothing.
set -euo pipefail
ROOT="${PXF_REPO:-/users/yfsun/Proteo-AA-pxdesign-fampnn-pack}"
SAMPLES="${SAMPLES:?set SAMPLES to a directory of L<len>_s<idx>.cif}"
OUT="${OUT:?set OUT}"
ADAPTERS="${ADAPTERS:-}"
FAMPNN="${FAMPNN:-0.0}"
LENGTHS="${LENGTHS:-}"
SEED="${SEED:-0}"
mkdir -p "$OUT" /scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_uncond
cd "$ROOT"
PXF_PYTHON_ENV="${PXF_PYTHON_ENV:-/users/yfsun/.venvs/proteoaa-stage4}"
export PATH="$PXF_PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn${PYTHONPATH:+:$PYTHONPATH}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-/scratch/m000137-pm06/Proteo-AA/pxf/protenix_data/common}"
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTEOAA_METRICS_ROOT="${PROTEOAA_METRICS_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

ARGS=(--samples-dir "$SAMPLES" --out "$OUT" --fampnn-weights "$FAMPNN" --seed "$SEED")
[ -n "$ADAPTERS" ] && ARGS+=(--adapters "$ADAPTERS")
[ -n "$LENGTHS" ] && ARGS+=(--lengths $LENGTHS)

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} fampnn=$FAMPNN adapters=${ADAPTERS:-none}"
python scripts/codesign_uncond.py "${ARGS[@]}"
echo "EXIT=$?"
