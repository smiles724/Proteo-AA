#!/usr/bin/env bash
# Submit with --dependency=afterok:<warmup-job>. The completed warm-up log is
# scanned for the lowest aggregate validation loss, and that exact checkpoint
# becomes the accepted parent of the SC-only complex-adaptation pilot.
#SBATCH --job-name=sc-complex-adapt
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --output=logs/training/sc-adaptation-%j.out
#SBATCH --error=logs/training/sc-adaptation-%j.err
set -euo pipefail

export PROTEOAA_REPO=${PROTEOAA_REPO:-/hai/users/y/f/yfsun/Proteo-AA-sc-adaptation-phases}
: "${WARMUP_JOB_ID:?Set WARMUP_JOB_ID to the completed sc_warmup Slurm job ID}"
WARMUP_REPO=${WARMUP_REPO:-/hai/users/y/f/yfsun/Proteo-AA-sc-rigid-augmentation}
WARMUP_LOG=${WARMUP_LOG:-$WARMUP_REPO/logs/training/official_pxdesign/sc-rigid-warmup-$WARMUP_JOB_ID.out}
WARMUP_CHECKPOINT_DIR=${WARMUP_CHECKPOINT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/official_sc_rigid_warmup/$WARMUP_JOB_ID/checkpoints}

if [[ ! -r "$WARMUP_LOG" ]]; then
  echo "ERROR: warm-up log is not readable: $WARMUP_LOG" >&2
  exit 1
fi

best_step=$(awk '
  /step=[0-9]+ val_n=/ {
    step = ""; loss = ""
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^step=/) { split($i, a, "="); step = a[2] }
      if ($i ~ /^val_loss=/) { split($i, a, "="); loss = a[2] }
    }
    if (step != "" && loss != "" && (best_loss == "" || loss < best_loss)) {
      best_loss = loss; best_step = step
    }
  }
  END {
    if (best_step == "") exit 1
    print best_step, best_loss
  }
' "$WARMUP_LOG")
read -r selected_step selected_loss <<< "$best_step"

export ACCEPTED_CHECKPOINT=$WARMUP_CHECKPOINT_DIR/step${selected_step}.pt
if [[ ! -r "$ACCEPTED_CHECKPOINT" ]]; then
  echo "ERROR: selected checkpoint is not readable: $ACCEPTED_CHECKPOINT" >&2
  exit 1
fi

export SC_PHASE=sc_complex_adapt
export OUTPUT_DIR=${OUTPUT_DIR:-/hai/scratch/yfsun/proteo_aa_runs/sc_complex_adapt/${SLURM_JOB_ID:-dry-run}}
echo "warmup_job_id=$WARMUP_JOB_ID selected_step=$selected_step selected_val_loss=$selected_loss"
echo "accepted_checkpoint=$ACCEPTED_CHECKPOINT"
echo "output_dir=$OUTPUT_DIR"
exec bash "$PROTEOAA_REPO/scripts/training/run_sc_adaptation.sh" "$@"
