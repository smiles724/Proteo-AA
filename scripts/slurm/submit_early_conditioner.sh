#!/bin/bash
# Submit a set of early-conditioner arms, retrying past the account submit cap.
#
# The yejin account has an account-WIDE MaxSubmitJobsPerAccount shared with
# every other user on it, so a batch of six arms is routinely accepted in part
# and refused in part. Refused is not failed: the arm simply never entered the
# queue, and the only outward sign is one fewer job in `squeue`. This retries
# each refused arm until it lands, so "I submitted the experiment" means all of
# it rather than whichever prefix fit.
#
#   ARMS="atom_sz_full atom_sz_bb_only atom_s_full" \
#   AFTEROK=119661:119662:119663 bash scripts/slurm/submit_early_conditioner.sh
#
# AFTEROK is a colon-separated list of job ids the arms wait for. It is resolved
# at submit time rather than passed through blindly: ids that have already
# COMPLETED are dropped (the dependency is satisfied), and if any has FAILED the
# whole batch is abandoned, which is the point of a gate.
#
# WHAT THIS DOES AND DOES NOT ENFORCE. Gating E2 on E1's job ids makes the
# ORDER real -- E2 cannot start before E1 finishes. It does not make E2
# conditional on anyone having READ E1's result, and no Slurm dependency can.
# If E1 reads out badly, `scancel` the E2 jobs; nothing here will do it.
# Submitting E2 and relying on the account's submit cap to hold it back is not
# staging at all: the cap frees at an unrelated moment.
set -uo pipefail

ARMS="${ARMS:?set ARMS to a space-separated list of arm names}"
AFTEROK="${AFTEROK:-}"
INTERVAL="${INTERVAL:-300}"      # seconds between retries
DEADLINE="${DEADLINE:-21600}"    # give up after this long, so it cannot linger
LOG="${LOG:-/hai/scratch/yfsun/proteo_aa_runs/pxf_early_cond/submit.log}"

if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

# sbatch inherits SLURM_* from an interactive allocation and they override the
# script's own --cpus-per-task / --mem. See the launcher's header.
for v in $(env | sed -n 's/^\(SLURM[A-Za-z_]*\)=.*/\1/p'); do
    [ "$v" = SLURM_CONF ] && continue; unset "$v"
done
unset CUDA_VISIBLE_DEVICES

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# Drop satisfied gates, abandon on a failed one, keep the rest. Echoes the
# remaining gate list on stdout; returns 1 if the batch should be abandoned.
resolve_gates() {
    local remaining="" id state
    for id in $(echo "$AFTEROK" | tr ':' ' '); do
        state=$(sacct -n -X -j "$id" -o State 2>/dev/null | head -1 | tr -d ' ')
        case "${state:-UNKNOWN}" in
            COMPLETED) say "gate $id completed" >&2 ;;
            FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                say "gate $id ended $state" >&2; return 1 ;;
            *) remaining="${remaining:+$remaining:}$id" ;;
        esac
    done
    echo "$remaining"
}

pending="$ARMS"
started=$(date +%s)
while [ -n "$pending" ]; do
    if [ -n "$AFTEROK" ]; then
        if ! AFTEROK=$(resolve_gates); then
            say "a gate job ended badly; abandoning [$pending]"; exit 1
        fi
    fi
    dep=""
    [ -n "$AFTEROK" ] && dep="--dependency=afterok:$AFTEROK"

    remaining=""
    for arm in $pending; do
        case "$arm" in
            early_s_*) name="e1_${arm#early_s_}" ;;
            atom_*)    name="e2_${arm#atom_}" ;;
            *) say "unknown arm $arm"; exit 2 ;;
        esac
        if id=$(ARM="$arm" sbatch --parsable --job-name="$name" $dep \
                scripts/slurm/train_early_conditioner.sh 2>>"$LOG"); then
            say "$arm -> job $id ($name)"
        else
            remaining="$remaining $arm"
        fi
    done
    pending="$(echo "$remaining" | xargs || true)"
    [ -z "$pending" ] && break

    if [ $(( $(date +%s) - started )) -ge "$DEADLINE" ]; then
        say "deadline reached with [$pending] still unsubmitted"
        exit 1
    fi
    say "account submit queue full; [$pending] retrying in ${INTERVAL}s"
    sleep "$INTERVAL"
done
say "all arms submitted"
