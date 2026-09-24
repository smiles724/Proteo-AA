#!/bin/bash
#SBATCH --job-name=pxf_eval_sigma
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_ifb/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_ifb/%x-%j.out
#
# Development evaluation of the adapters retrained at one event sigma, on the
# 31 held-out dimers, per the declared criterion in selection.yaml.
#
#   EVENT_SIGMA=4.1250 sbatch scripts/slurm/marlowe/eval_event_sigma.sh
#
# This is the SIDE-CHAIN evaluation for a feedback adapter. eval_couple.py
# measures packing on bb0 and is the A_BS instrument; here the event's
# sequence is held fixed and repacked onto THIS arm's CORRECTED backbone,
# with one packing seed across arms so a rotamer draw cannot be read as a
# chemistry difference.
#
# --event-sigma MUST match what the arms were trained at. An adapter fit at
# 4.125 and evaluated at 0.429 is being queried off its own conditioning,
# which is the whole failure this retrain exists to avoid.
set -euo pipefail
ROOT="${PXF_REPO:-/users/yfsun/Proteo-AA-pxdesign-fampnn-pack}"
DATA="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
SIGMA="${EVENT_SIGMA:?set EVENT_SIGMA}"
TAG="${TAG:-sig$(printf '%s' "$SIGMA" | tr -d '.')}"
RUNROOT="${RUNROOT:-$DATA/runs/integrated_feedback_sigma/$TAG}"
OUT="${OUT:-$RUNROOT/evaluation}"
DONOR="${DONOR:-$DATA/component_donors/pxdesign_v0.1.0.pt}"
FAMPNN="${FAMPNN:-$ROOT/fampnn/weights/fampnn_0_3.pt}"

cd "$ROOT"
mkdir -p "$OUT"
export PATH="/users/yfsun/.venvs/pxdesign_official/bin:$PATH"
export PYTHONPATH="$ROOT:/users/yfsun/pxdesign_pristine:$ROOT/fampnn"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$DATA/official_release_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-$DATA/official_release_data/ccd_cache}"
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTEOAA_METRICS_ROOT="${PROTEOAA_METRICS_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

RUNS=()
for ARM in E1_full_s0 E1_full_s1 E1_bb_only_s0 E1_bb_only_s1; do
  [ -f "$RUNROOT/$ARM/checkpoints/final.pt" ] && RUNS+=("$RUNROOT/$ARM")
done
if [ ${#RUNS[@]} -eq 0 ]; then
  echo "no trained arms under $RUNROOT; did the retrain finish?" >&2; exit 2
fi
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} sigma=$SIGMA arms=${#RUNS[@]}"

python scripts/eval_integrated_feedback.py \
    --manifest "$DATA/runs/integrated_feedback_v1/data/validation.parquet" \
    --runs "${RUNS[@]}" \
    --selection configs/integrated_feedback/selection.yaml \
    --bs-checkpoint "0=$DATA/runs/bs_seq_sc/J03_seed0/checkpoints/step00000500.pt" \
    --bs-checkpoint "1=$DATA/runs/bs_seq_sc/J03_seed1/checkpoints/step00000500.pt" \
    --pxdesign-donor "$DONOR" \
    --fampnn-checkpoint "$FAMPNN" --fampnn-variant 0.3 \
    --event-sigma "$SIGMA" \
    --out "$OUT"
echo "EXIT=$?"
