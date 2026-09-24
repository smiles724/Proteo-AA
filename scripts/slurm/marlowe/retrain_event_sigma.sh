#!/bin/bash
#SBATCH --job-name=pxf_retrain_sigma
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
# Retrain the feedback adapters at ONE event sigma: two caches (one per A_BS
# seed) then four arms (E1_full / E1_bb_only x seed 0/1). Chained in a single
# job because the caches take ~1 minute each and the QoS caps submissions --
# six sbatch calls per sigma would not fit alongside anything else.
#
#   EVENT_SIGMA=4.125 sbatch scripts/slurm/marlowe/retrain_event_sigma.sh
#
# WHY this exists: A_BS and the feedback adapters are conditioned on log
# sigma_B and were fit at ONE event at sigma 0.429. Running inference at a
# different event sigma extrapolates that conditioning, and the MeanResidual
# control is clamped at its knots outside the training window, so it cannot
# even be computed honestly there. An inference sweep says whether an earlier
# event is worth chasing; only a retrain makes a checkpoint adoptable at it.
set -euo pipefail
ROOT="${PXF_REPO:-/users/yfsun/Proteo-AA-pxdesign-fampnn-pack}"
DATA="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
SIGMA="${EVENT_SIGMA:?set EVENT_SIGMA}"
TAG="${TAG:-sig$(printf '%s' "$SIGMA" | tr -d '.')}"
OUTROOT="${OUTROOT:-$DATA/runs/integrated_feedback_sigma/$TAG}"
MANIFEST="${MANIFEST:-$DATA/runs/integrated_feedback_v1/data/train_pdb.parquet}"
DONOR="${DONOR:-$DATA/component_donors/pxdesign_v0.1.0.pt}"
FAMPNN="${FAMPNN:-$ROOT/fampnn/weights/fampnn_0_3.pt}"
MAX_STEPS="${MAX_STEPS:-2000}"

cd "$ROOT"
mkdir -p "$OUTROOT" "$DATA/runs/logs/pxf_ifb"
export PATH="/users/yfsun/.venvs/pxdesign_official/bin:$PATH"
export PYTHONPATH="$ROOT:/users/yfsun/pxdesign_pristine:$ROOT/fampnn"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$DATA/official_release_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-$DATA/official_release_data/ccd_cache}"
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTEOAA_METRICS_ROOT="${PROTEOAA_METRICS_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"

echo "node=$(hostname) job=${SLURM_JOB_ID:-?} EVENT_SIGMA=$SIGMA out=$OUTROOT"
python -c "from pxf.official.require import official_protenix_available as a; print('official:', a())"

# ---- caches, one per A_BS seed ----------------------------------------
for S in 0 1; do
  BS="$DATA/runs/bs_seq_sc/J03_seed${S}/checkpoints/step00000500.pt"
  CACHE="$OUTROOT/cache/J03_s${S}"
  if [ -f "$CACHE/cache.json" ]; then echo "SKIP cache s$S (exists)"; continue; fi
  echo "=== cache seed $S at sigma $SIGMA ==="
  python scripts/cache_integrated_feedback.py \
      --manifest "$MANIFEST" --out "$CACHE" \
      --bs-checkpoint "$BS" --bs-weights ema \
      --pxdesign-donor "$DONOR" --fampnn-checkpoint "$FAMPNN" --fampnn-variant 0.3 \
      --event-sigma "$SIGMA" --seed "$S"
done

# ---- four arms --------------------------------------------------------
for ARM in E1_full E1_bb_only; do
  for S in 0 1; do
    OUT="$OUTROOT/${ARM}_s${S}"
    if [ -f "$OUT/checkpoints/final.pt" ]; then echo "SKIP $ARM s$S (exists)"; continue; fi
    echo "=== train $ARM seed $S at sigma $SIGMA ==="
    python scripts/train_integrated_feedback.py \
        --config "configs/integrated_feedback/${ARM}.yaml" \
        --train-cache "$OUTROOT/cache/J03_s${S}" \
        --bs-checkpoint "$DATA/runs/bs_seq_sc/J03_seed${S}/checkpoints/step00000500.pt" \
        --pxdesign-donor "$DONOR" \
        --fampnn-checkpoint "$FAMPNN" --fampnn-variant 0.3 \
        --seed "$S" --max-steps "$MAX_STEPS" --out "$OUT"
  done
done

echo "=== checkpoints written ==="
find "$OUTROOT" -name "final.pt" -printf "  %p  %s bytes\n"
echo "EXIT=$?"
