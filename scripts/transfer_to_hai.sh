#!/bin/bash
# Push Marlowe run artefacts back to HAI.
#
# Marlowe -> HAI over ssh. rsync is resumable and runs without --delete, so a
# re-run after an interrupted transfer picks up where it stopped and never
# removes anything on the far side.
#
#   ./scripts/transfer_to_hai.sh --check          # validate locally, no ssh
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
# HAI uses Duo two-factor, so every ssh costs an interactive approval. A single
# ControlMaster socket at a STABLE path is shared by every rsync here and kept
# alive for SSH_PERSIST seconds afterwards, so consecutive runs (a --dry-run
# then the real thing) reuse one authentication instead of asking again.
#
# NOTE: the group list is NOT held in a variable called GROUPS. That is a bash
# builtin array of the caller's Unix group IDs and assignments to it are
# silently ignored, which made an earlier version of this script iterate over
# GIDs and abort with "unknown group: 1543400513".
set -uo pipefail

HAI_HOST="${HAI_HOST:-yfsun@haic.stanford.edu}"
HAI_ROOT="${HAI_ROOT:-/hai/scratch/yfsun}"
RUNS="${RUNS:-/scratch/m000137-pm06/Proteo-AA/pxf/runs}"
DEST="$HAI_ROOT/marlowe_runs"
SSH_PERSIST="${SSH_PERSIST:-1800}"

KNOWN_GROUPS="adapters control metrics uncond"

DRY=""
CHECK_ONLY=0
ALL_CKPT=0
SEND_GROUPS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY="--dry-run" ;;
        --check) CHECK_ONLY=1 ;;
        --all-checkpoints) ALL_CKPT=1 ;;
        -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
        -*) echo "unknown flag: $arg" >&2; exit 2 ;;
        *) SEND_GROUPS+=("$arg") ;;
    esac
done
[ ${#SEND_GROUPS[@]} -eq 0 ] && SEND_GROUPS=(adapters metrics)

# Validate before touching ssh: a typo must not cost a Duo approval.
for group in "${SEND_GROUPS[@]}"; do
    case " $KNOWN_GROUPS " in
        *" $group "*) ;;
        *) echo "unknown group: $group (choose from: $KNOWN_GROUPS)" >&2; exit 2 ;;
    esac
done

echo "host   : $HAI_HOST"
echo "dest   : $DEST"
echo "groups : ${SEND_GROUPS[*]}${DRY:+  (dry run)}"

# What each group would send, so --check can report without connecting.
plan_for() {
    case "$1" in
        adapters)
            if [ "$ALL_CKPT" -eq 1 ]; then
                echo "$RUNS/couple_phase1/checkpoints/|couple_phase1/checkpoints"
            else
                echo "$RUNS/couple_phase1/checkpoints/final.pt|couple_phase1/checkpoints"
            fi
            echo "$RUNS/couple_phase1/train_log.jsonl|couple_phase1"
            echo "$RUNS/couple_phase1/run_config.json|couple_phase1"
            echo "$RUNS/couple_phase1/result.json|couple_phase1"
            ;;
        control)
            echo "$RUNS/ft_fampnn_phase1/checkpoints/final.pt|ft_fampnn_phase1/checkpoints"
            echo "$RUNS/ft_fampnn_phase1/train_log.jsonl|ft_fampnn_phase1"
            echo "$RUNS/ft_fampnn_phase1/run_config.json|ft_fampnn_phase1"
            ;;
        metrics)
            for d in "$RUNS"/eval_* "$RUNS"/protenix_* "$RUNS"/ft_eval_*; do
                [ -d "$d" ] || continue
                echo "$d/|metrics/$(basename "$d")|json-csv-only"
            done
            ;;
        uncond)
            echo "$RUNS/uncond/|uncond"
            echo "$RUNS/cd_smoke_unc/|uncond_codesign/uncoupled"
            echo "$RUNS/cd_smoke_cou/|uncond_codesign/coupled"
            ;;
    esac
}

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "--- local check (no ssh) ---"
    missing=0
    for group in "${SEND_GROUPS[@]}"; do
        echo "=== $group ==="
        while IFS='|' read -r src sub filt; do
            [ -n "${src:-}" ] || continue
            if [ -e "$src" ]; then
                size=$(du -sh "$src" 2>/dev/null | cut -f1)
                echo "  ok      $size  $src -> $sub${filt:+  [$filt]}"
            else
                echo "  ABSENT        $src"; missing=$((missing+1))
            fi
        done < <(plan_for "$group")
    done
    echo "--- $missing absent source(s) ---"
    exit 0
fi

# Stable socket path: reused across invocations while the master is alive.
CTL="${TMPDIR:-/tmp}/ssh-hai-${USER}.sock"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$CTL" -o "ControlPersist=$SSH_PERSIST")

if ssh -O check "${SSH_OPTS[@]}" "$HAI_HOST" >/dev/null 2>&1; then
    echo "ssh    : reusing the existing authenticated connection"
else
    echo "ssh    : authenticating (Duo approval expected once)"
fi

ssh "${SSH_OPTS[@]}" "$HAI_HOST" "mkdir -p '$DEST'" || {
    echo "cannot reach $HAI_HOST or create $DEST" >&2; exit 1
}

sent=0; skipped=0
for group in "${SEND_GROUPS[@]}"; do
    echo "=== $group ==="
    while IFS='|' read -r src sub filt; do
        [ -n "${src:-}" ] || continue
        if [ ! -e "$src" ]; then
            echo "  SKIP (absent): $src"; skipped=$((skipped+1)); continue
        fi
        echo "  -> $sub"
        ssh "${SSH_OPTS[@]}" "$HAI_HOST" "mkdir -p '$DEST/$sub'"
        extra=()
        if [ "${filt:-}" = "json-csv-only" ]; then
            # Filter so the metrics group can never drag weights along.
            extra=(--include='*/' --include='*.json' --include='*.csv' --exclude='*')
        fi
        rsync -a --info=stats1 $DRY "${extra[@]}" \
            -e "ssh ${SSH_OPTS[*]}" "$src" "$HAI_HOST:$DEST/$sub/" \
            && sent=$((sent+1))
    done < <(plan_for "$group")
done

echo
echo "sent $sent item(s), skipped $skipped absent"
echo "on HAI:  ls -la $DEST/couple_phase1/checkpoints/"
echo
echo "The adapter checkpoint is scored on its EMA weights: eval_couple.py"
echo "applies state['ema'] over state['adapters']. Loading the raw adapters"
echo "gives un-averaged weights and slightly different numbers."
