#!/bin/bash
#SBATCH --job-name=alpha10-qmgr
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=23:50:00
#SBATCH --output=logs/validation/alphaproteo10/%x-%j.out
#SBATCH --error=logs/validation/alphaproteo10/%x-%j.err

# Scheduler-friendly FIFO for AlphaProteo scoring. At most MAX_IN_FLIGHT GPU
# scoring jobs are outstanding. A scheduler/QOS rejection leaves the task at
# the queue front. A failed task is moved back to the front after a backoff.
# CSV and JSON artifacts, rather than Slurm state alone, determine success.

set -u -o pipefail

REPO_ROOT="${REPO_ROOT:-/hai/users/s/h/shenjm/Proteo-AA}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT from generation}"
TASK_FILE="${TASK_FILE:-${RUN_ROOT}/score_tasks.tsv}"
PXDBENCH_DIR="${PXDBENCH_DIR:?set PXDBENCH_DIR}"
PXDBENCH_PYTHON="${PXDBENCH_PYTHON:?set PXDBENCH_PYTHON}"
TOOL_WEIGHTS_ROOT="${TOOL_WEIGHTS_ROOT:?set TOOL_WEIGHTS_ROOT}"
SCORE_SCRIPT="${SCORE_SCRIPT:-${REPO_ROOT}/scripts/evaluation/slurm_score_alphaproteo_designability.sh}"
SUMMARY_PYTHON="${SUMMARY_PYTHON:-/hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python}"

TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-4}"
INITIAL_JOB_ID="${INITIAL_JOB_ID:-}"
INITIAL_TASK="${INITIAL_TASK:-${TASK_START}}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SUBMIT_BACKOFF_SECONDS="${SUBMIT_BACKOFF_SECONDS:-120}"
FAILURE_BACKOFF_SECONDS="${FAILURE_BACKOFF_SECONDS:-300}"
MAX_TASK_RETRIES="${MAX_TASK_RETRIES:-3}"
AUTO_SUMMARIZE="${AUTO_SUMMARIZE:-1}"
MPNN_SEQUENCES="${MPNN_SEQUENCES:-1}"
MPNN_TEMPERATURE="${MPNN_TEMPERATURE:-0.0001}"

[[ -f "${TASK_FILE}" ]] || { echo "ERROR: missing ${TASK_FILE}" >&2; exit 2; }
[[ -x "${PXDBENCH_PYTHON}" ]] || { echo "ERROR: invalid PXDBENCH_PYTHON" >&2; exit 2; }
[[ -x "${SCORE_SCRIPT}" ]] || { echo "ERROR: missing executable ${SCORE_SCRIPT}" >&2; exit 2; }
(( MAX_IN_FLIGHT >= 1 )) || { echo "ERROR: MAX_IN_FLIGHT must be >= 1" >&2; exit 2; }

n_tasks="$(( $(wc -l < "${TASK_FILE}") - 1 ))"
if [[ -z "${TASK_END}" ]]; then TASK_END="$((n_tasks - 1))"; fi
(( TASK_START >= 0 && TASK_START < n_tasks )) || { echo "ERROR: invalid TASK_START=${TASK_START}" >&2; exit 2; }
(( TASK_END >= TASK_START && TASK_END < n_tasks )) || { echo "ERROR: invalid TASK_END=${TASK_END}" >&2; exit 2; }

mkdir -p "${RUN_ROOT}/queue" "${RUN_ROOT}/summary" "${REPO_ROOT}/logs/validation/alphaproteo10"
QUEUE_LOG="${QUEUE_LOG:-${RUN_ROOT}/queue/manager-${SLURM_JOB_ID:-manual}.tsv}"
export PATH="$(dirname "${PXDBENCH_PYTHON}"):${PATH}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-openfold}"
export USE_DEEPSPEED_EVO_ATTENTION="${USE_DEEPSPEED_EVO_ATTENTION:-false}"

log_event() {
  local task="$1" job="$2" state="$3" detail="${4:-}"
  local timestamp
  timestamp="$(date --iso-8601=seconds)"
  printf '%s\t%s\t%s\t%s\t%s\n' "${timestamp}" "${task}" "${job}" "${state}" "${detail}" | tee -a "${QUEUE_LOG}"
}

task_fields() {
  sed -n "$(( $1 + 2 ))p" "${TASK_FILE}"
}

artifact_complete() {
  local task="$1" line output_dir expected_sequences rows
  line="$(task_fields "${task}")"
  [[ -n "${line}" ]] || return 1
  IFS=$'\t' read -r _ _ _ _ _ output_dir _ _ expected_sequences <<<"${line}"
  expected_sequences="${expected_sequences//$'\r'/}"
  [[ -s "${output_dir}/sample_level_output.csv" ]] || return 1
  [[ -s "${output_dir}/summary_output.json" ]] || return 1
  head -n 1 "${output_dir}/sample_level_output.csv" | grep -q 'af2_opt_success' || return 1
  rows="$(( $(wc -l < "${output_dir}/sample_level_output.csv") - 1 ))"
  (( rows >= expected_sequences ))
}

slurm_state() {
  local job="$1" state
  state="$(squeue -h -j "${job}" -o '%T' 2>/dev/null | head -n 1 || true)"
  if [[ -z "${state}" ]]; then
    state="$(sacct -n -X -j "${job}" --format=State --parsable2 2>/dev/null | sed -n '1{s/[| ].*$//;p;}' || true)"
  fi
  state="${state%%+*}"
  printf '%s' "${state}"
}

sleep_for() {
  local remaining="$1" chunk
  while (( remaining > 0 )); do
    chunk=60
    if (( remaining < chunk )); then chunk="${remaining}"; fi
    sleep "${chunk}"
    remaining="$((remaining - chunk))"
  done
}

declare -a queue=()
declare -a blocked=()
declare -A job_to_task=()
declare -A retries=()
declare -A retry_ready_at=()

for ((task = TASK_START; task <= TASK_END; task++)); do
  if [[ -n "${INITIAL_JOB_ID}" && "${task}" == "${INITIAL_TASK}" ]]; then
    continue
  fi
  queue+=("${task}")
done

if [[ -n "${INITIAL_JOB_ID}" ]]; then
  job_to_task["${INITIAL_JOB_ID}"]="${INITIAL_TASK}"
  retries["${INITIAL_TASK}"]=0
  log_event "${INITIAL_TASK}" "${INITIAL_JOB_ID}" attached "monitoring pre-existing job"
fi

submission_blocked_until=0

while :; do
  # Reap completed/failed jobs. Jobs still pending or running continue to
  # occupy one of the MAX_IN_FLIGHT slots.
  for job in "${!job_to_task[@]}"; do
    task="${job_to_task[${job}]}"
    if artifact_complete "${task}"; then
      log_event "${task}" "${job}" completed "validated CSV and JSON"
      unset 'job_to_task[$job]'
      unset 'retry_ready_at[$task]'
      continue
    fi

    state="$(slurm_state "${job}")"
    case "${state}" in
      PENDING|RUNNING|CONFIGURING|COMPLETING|RESIZING|REQUEUED|SUSPENDED)
        continue
        ;;
      "")
        # Accounting records can lag briefly after a job leaves squeue.
        continue
        ;;
      COMPLETED)
        state="COMPLETED_MISSING_OUTPUT"
        ;;
    esac

    unset 'job_to_task[$job]'
    attempt="$(( ${retries[${task}]:-0} + 1 ))"
    retries["${task}"]="${attempt}"
    if (( attempt > MAX_TASK_RETRIES )); then
      # Park the task and keep draining the queue. One unresolvable task used to
      # abort the manager, which left every task behind it unrun for no reason;
      # the blocked set is reported at the end and fails the manager then.
      blocked+=("${task}")
      log_event "${task}" "${job}" blocked "${state}; retry limit reached, task parked and queue continues"
      continue
    fi
    retry_ready_at["${task}"]="$(( $(date +%s) + FAILURE_BACKOFF_SECONDS ))"
    queue=("${task}" "${queue[@]}")
    log_event "${task}" "${job}" "${state}" "returned to queue front; retry ${attempt}/${MAX_TASK_RETRIES} after backoff"
  done

  # Fill free slots from the FIFO. If its front task is backing off, leave the
  # slot empty so that task remains first rather than being bypassed.
  while (( ${#job_to_task[@]} < MAX_IN_FLIGHT && ${#queue[@]} > 0 )); do
    task="${queue[0]}"
    if artifact_complete "${task}"; then
      log_event "${task}" - completed "found existing validated CSV and JSON"
      queue=("${queue[@]:1}")
      unset 'retry_ready_at[$task]'
      continue
    fi

    now="$(date +%s)"
    ready_at="${retry_ready_at[${task}]:-0}"
    if (( now < ready_at || now < submission_blocked_until )); then
      break
    fi

    submit_output="$(sbatch --parsable \
      --array="${task}-${task}%1" \
      --export="ALL,TASK_FILE=${TASK_FILE},PXDBENCH_DIR=${PXDBENCH_DIR},PXDBENCH_PYTHON=${PXDBENCH_PYTHON},TOOL_WEIGHTS_ROOT=${TOOL_WEIGHTS_ROOT},MPNN_SEQUENCES=${MPNN_SEQUENCES},MPNN_TEMPERATURE=${MPNN_TEMPERATURE},LAYERNORM_TYPE=${LAYERNORM_TYPE},USE_DEEPSPEED_EVO_ATTENTION=${USE_DEEPSPEED_EVO_ATTENTION}" \
      "${SCORE_SCRIPT}" 2>&1)"
    submit_rc="$?"
    if (( submit_rc != 0 )); then
      submission_blocked_until="$((now + SUBMIT_BACKOFF_SECONDS))"
      log_event "${task}" - submit_rejected "task retained at queue front; ${submit_output//$'\n'/ }"
      break
    fi

    job_root="${submit_output%%;*}"
    job_root="${job_root//$'\n'/}"
    if [[ ! "${job_root}" =~ ^[0-9]+$ ]]; then
      submission_blocked_until="$((now + SUBMIT_BACKOFF_SECONDS))"
      log_event "${task}" - submit_parse_error "task retained at queue front; ${submit_output//$'\n'/ }"
      break
    fi

    queue=("${queue[@]:1}")
    current_job="${job_root}_${task}"
    job_to_task["${current_job}"]="${task}"
    retries["${task}"]="${retries[${task}]:-0}"
    unset 'retry_ready_at[$task]'
    log_event "${task}" "${current_job}" submitted "in_flight=${#job_to_task[@]}/${MAX_IN_FLIGHT}"
  done

  if (( ${#queue[@]} == 0 && ${#job_to_task[@]} == 0 )); then
    break
  fi
  sleep_for "${POLL_SECONDS}"
done

if (( ${#blocked[@]} > 0 )); then
  log_event - - queue_drained "tasks ${TASK_START}-${TASK_END}; blocked tasks: ${blocked[*]}"
else
  log_event - - queue_complete "tasks ${TASK_START}-${TASK_END} validated"
fi

# Summarize even with blocked tasks: the summary already counts missing scores
# as failures and reports coverage, so a partial run stays interpretable.
if [[ "${AUTO_SUMMARIZE}" == "1" ]]; then
  "${SUMMARY_PYTHON}" "${REPO_ROOT}/scripts/evaluation/summarize_alphaproteo_designability.py" \
    --task-file "${TASK_FILE}" \
    --output-dir "${RUN_ROOT}/summary"
  log_event - - summary_complete "${RUN_ROOT}/summary"
fi

if (( ${#blocked[@]} > 0 )); then
  echo "ERROR: ${#blocked[@]} task(s) never produced valid output: ${blocked[*]}" >&2
  exit 1
fi
