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
#   AFTEROK=119659 bash scripts/slurm/submit_early_conditioner.sh
#
# AFTEROK gates the arms behind a smoke job. It is resolved at submit time
# rather than passed through blindly: a dependency on an already-COMPLETED job
# is satisfied immediately (so it is dropped), and one on a job that FAILED
# would leave the arms permanently unsatisfiable (so the whole batch is
# abandoned, which is the point of the gate).
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

gate_state() {
    [ -z "$AFTEROK" ] && { echo NONE; return; }
    local state
    state=$(sacct -n -X -j "$AFTEROK" -o State 2>/dev/null | head -1 | tr -d ' ')
    echo "${state:-UNKNOWN}"
}

pending="$ARMS"
started=$(date +%s)
while [ -n "$pending" ]; do
    state=$(gate_state)
    case "$state" in
        COMPLETED) say "gate $AFTEROK completed; submitting without a dependency"
                   AFTEROK="" ;;
        FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                   say "gate $AFTEROK ended $state; abandoning [$pending]"; exit 1 ;;
    esac
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
