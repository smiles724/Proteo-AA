#!/bin/bash
#SBATCH --job-name=pxf_eval_sweep
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_eval_sweep/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_eval_sweep/%x-%j.out
#
# Denoised-mode coupling eval on EVERY phase-1 checkpoint, so the metric can be
# read as a function of training step rather than only at the end.
#
# WHY ONE JOB AND NOT A JOB ARRAY: qos `medium` caps MaxSubmitPA/MaxJobsPA at
# 32 *per account*, shared by every user on marlowe-m000137-pm06. A 20-task
# array was rejected outright (MaxSubmitJobsPerAccount) and, even under the cap,
# would have consumed the whole team's submit budget. This runs the checkpoints
# sequentially inside a single allocation: one submit slot, one GPU.
#
# It does NOT duplicate the eval logic -- each iteration invokes
# scripts/slurm/marlowe/eval_couple.sh, so the held-out guard, the
# EVAL_STRUCTURES fix and the env handling live in exactly one place.
#
# Required at submit time:
#   CKPT_LIST        file with one checkpoint path per line
#   SWEEP_OUT_ROOT   parent directory; each checkpoint writes <root>/<tag>/
#
# A failing checkpoint is recorded and the sweep continues -- one bad checkpoint
# must not discard the other nineteen. The exit code reflects whether any failed.
set -uo pipefail

LIST="${CKPT_LIST:?set CKPT_LIST to a file of checkpoint paths}"
ROOT_OUT="${SWEEP_OUT_ROOT:?set SWEEP_OUT_ROOT to the parent output directory}"
if [ ! -f "$LIST" ]; then echo "CKPT_LIST not found: $LIST" >&2; exit 2; fi

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi

N=$(grep -c . "$LIST")
echo "=== sweep: $N checkpoint(s), denoised mode, out under $ROOT_OUT ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

ok=0; failed=0; failed_tags=""
i=0
while IFS= read -r CKPT; do
    [ -n "$CKPT" ] || continue
    i=$((i+1))
    TAG=$(basename "$CKPT" .pt)
    if [ ! -f "$CKPT" ]; then
        echo "[$i/$N] $TAG SKIP - checkpoint missing: $CKPT"
        failed=$((failed+1)); failed_tags="$failed_tags $TAG"; continue
    fi
    echo "--------------------------------------------------------------"
    echo "[$i/$N] $TAG  ($(date +%H:%M:%S))"
    if MODE=denoised CHECKPOINT="$CKPT" OUT="$ROOT_OUT/$TAG" \
         bash "$ROOT/scripts/slurm/marlowe/eval_couple.sh"; then
        ok=$((ok+1))
    else
        rc=$?
        echo "[$i/$N] $TAG FAILED (exit $rc) - continuing"
        failed=$((failed+1)); failed_tags="$failed_tags $TAG"
    fi
done < "$LIST"

echo "=============================================================="
echo "sweep done: $ok succeeded, $failed failed"
[ -n "$failed_tags" ] && echo "failed:$failed_tags"
[ "$failed" -eq 0 ] || exit 1
