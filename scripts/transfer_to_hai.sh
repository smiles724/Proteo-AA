#!/bin/bash
# Push Marlowe run artefacts back to HAI.
#
# Marlowe -> HAI over ssh. rsync is resumable and runs without --delete, so a
# re-run after an interrupted transfer picks up where it stopped and never
# removes anything on the far side.
#
#   ./scripts/transfer_to_hai.sh --dry-run        # list, transfer nothing
#   ./scripts/transfer_to_hai.sh                  # adapters + metrics
#   ./scripts/transfer_to_hai.sh adapters         # named groups only
#   ./scripts/transfer_to_hai.sh --all-checkpoints adapters
#
# Groups:
#   adapters   phase-1 coupling checkpoints + train log + run config
#   control    the FaMPNN fine-tune control run (if it has finished)
#   metrics    every eval's couple_metrics.json / per_target.csv
#   uncond     sampled backbones and co-design output
#
# One password prompt per rsync invocation unless you have keys or a
# ControlMaster socket; -o ControlMaster below reuses a single connection.
set -uo pipefail

HAI_HOST="${HAI_HOST:-yfsun@haic.stanford.edu}"
HAI_ROOT="${HAI_ROOT:-/hai/scratch/yfsun}"
RUNS="${RUNS:-/scratch/m000137-pm06/Proteo-AA/pxf/runs}"
DEST="$HAI_ROOT/marlowe_runs"

DRY=""
ALL_CKPT=0
GROUPS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY="--dry-run" ;;
        --all-checkpoints) ALL_CKPT=1 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        -*) echo "unknown flag: $arg" >&2; exit 2 ;;
        *) GROUPS+=("$arg") ;;
    esac
done
[ ${#GROUPS[@]} -eq 0 ] && GROUPS=(adapters metrics)

# Reuse one ssh connection for every rsync so the password is asked once.
CTL="${TMPDIR:-/tmp}/ssh-hai-$$"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$CTL -o ControlPersist=300"
cleanup() { ssh -O exit $SSH_OPTS "$HAI_HOST" 2>/dev/null || true; }
trap cleanup EXIT

echo "host   : $HAI_HOST"
echo "dest   : $DEST"
echo "groups : ${GROUPS[*]}${DRY:+  (dry run)}"

ssh $SSH_OPTS "$HAI_HOST" "mkdir -p '$DEST'" || {
    echo "cannot reach $HAI_HOST or create $DEST" >&2; exit 1
}

send() {  # send <local-path> <remote-subdir> [extra rsync args...]
    local src="$1" sub="$2"; shift 2
    if [ ! -e "$src" ]; then
        echo "  SKIP (absent): $src"; return 0
    fi
    echo "  -> $sub"
    ssh $SSH_OPTS "$HAI_HOST" "mkdir -p '$DEST/$sub'"
    rsync -a --info=stats1 --no-delete $DRY "$@" \
        -e "ssh $SSH_OPTS" "$src" "$HAI_HOST:$DEST/$sub/"
}

for group in "${GROUPS[@]}"; do
    echo "=== $group ==="
    case "$group" in
        adapters)
            # final.pt is what every reported coupled number used. The 20 step
            # checkpoints are 114M and only needed to redo the training curve.
            if [ "$ALL_CKPT" -eq 1 ]; then
                send "$RUNS/couple_phase1/checkpoints/" couple_phase1/checkpoints
            else
                send "$RUNS/couple_phase1/checkpoints/final.pt" couple_phase1/checkpoints
            fi
            send "$RUNS/couple_phase1/train_log.jsonl" couple_phase1
            send "$RUNS/couple_phase1/run_config.json" couple_phase1
            send "$RUNS/couple_phase1/result.json" couple_phase1
            ;;
        control)
            send "$RUNS/ft_fampnn_phase1/checkpoints/final.pt" ft_fampnn_phase1/checkpoints
            send "$RUNS/ft_fampnn_phase1/train_log.jsonl" ft_fampnn_phase1
            send "$RUNS/ft_fampnn_phase1/run_config.json" ft_fampnn_phase1
            ;;
        metrics)
            # Small JSON/CSV only -- no weights.
            for d in "$RUNS"/eval_* "$RUNS"/protenix_* "$RUNS"/ft_eval_*; do
                [ -d "$d" ] || continue
                send "$d/" "metrics/$(basename "$d")" \
                    --include='*/' --include='*.json' --include='*.csv' \
                    --exclude='*'
            done
            ;;
        uncond)
            send "$RUNS/uncond/" uncond
            send "$RUNS/cd_smoke_unc/" uncond_codesign/uncoupled
            send "$RUNS/cd_smoke_cou/" uncond_codesign/coupled
            ;;
        *) echo "  unknown group: $group" >&2; exit 2 ;;
    esac
done

echo
echo "done. On HAI:"
echo "  ls -la $DEST/couple_phase1/checkpoints/"
echo
echo "The adapter checkpoint is scored on its EMA weights: eval_couple.py"
echo "applies state['ema'] over state['adapters']. Loading the raw adapters"
echo "gives un-averaged weights and slightly different numbers."
