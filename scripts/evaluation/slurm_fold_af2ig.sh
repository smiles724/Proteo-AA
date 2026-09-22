#!/bin/bash
#SBATCH --job-name=proteo-aa-af2ig
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/evaluation/af2ig/%x-%j.out
#SBATCH --error=logs/evaluation/af2ig/%x-%j.err

# AF2 initial-guess scoring for the A-CODE ConditionalBinderDesignBenchmark.
#
# This is the SECOND half. slurm_eval_conditional_binder_benchmark.sh generates
# designs/*.pdb + designs.csv on a Stage III checkpoint; this folds them and
# writes af2ig_metrics.csv, which score_af2ig_designability.py turns into the
# Table 4 row.
#
#   mkdir -p logs/evaluation/af2ig     # once; #SBATCH --output cannot mkdir
#   RUN_DIR=/scratch/.../cbdb/12345 sbatch scripts/evaluation/slurm_fold_af2ig.sh
#   RUN_DIR=... VARIANTS="co_design pmpnn" sbatch ...
#   RUN_DIR=... DRY_RUN=1 bash scripts/evaluation/slurm_fold_af2ig.sh   # no GPU
#
# DRY_RUN resolves every design and its binder chain and exits before touching
# JAX, which is where a moved run directory or an ambiguous chain surfaces --
# cheaply, and on a login node.
#
# Note the environment: this job runs the af2ig venv, NOT the training one.
# They cannot be the same interpreter (JAX AlphaFold vs torch Protenix, and
# PXDesign's AF2 path pins a Protenix this repo does not train against). Build
# it once with scripts/utilities/bootstrap_af2ig.sh.
#
# Cost, measured, not guessed. On one H100, PDL1 at L=80 (196 tokens) with 3
# recycles: ~28 s for the first (design, variant) pair, then **0.6-0.7 s** each
# in steady state, covering both AlphaFold passes. The 28 s is JIT compilation
# and is paid once per distinct token count, i.e. once per (target, length)
# cell -- which is why the driver sorts designs into cells rather than folding
# them in CSV order.
#
# AF2 here runs single-sequence (no MSA), so the MSA stack is trivial and the
# cost is dominated by the pair stack, which grows roughly quadratically in
# tokens. PDL1 is the smallest target; TNFa (438 target residues + binder) is
# ~3x the tokens and should be expected to cost several times more per design.
# Even so the whole benchmark is hours, not days: 10 targets x 6 lengths x 64
# samples x 2 arms is 7,680 folds, plus 60 compilations.
#
# The unbound pass is cached by sequence, so the second variant of a backbone
# costs one pass, not two. af2ig_metrics.csv is append-only and the driver
# resumes from it, so requeueing loses nothing.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/users/yfsun/Proteo-AA}"
AF2IG_ENV="${AF2IG_ENV:-${HOME}/.venvs/af2ig}"
PYTHON_BIN="${PYTHON_BIN:-${AF2IG_ENV}/bin/python}"
export AF2_PARAMS_DIR="${AF2_PARAMS_DIR:-${HOME}/af2_params}"

RUN_DIR="${RUN_DIR:-}"
VARIANTS="${VARIANTS:-co_design}"
TARGETS="${TARGETS:-}"
NUM_RECYCLES="${NUM_RECYCLES:-3}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
# On by default for a batch job: one design that fails to fold should not cost
# the other several thousand. Failures write no row, so a requeue retries them.
SKIP_FAILURES="${SKIP_FAILURES:-1}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: set RUN_DIR=<generation run directory with designs.csv>" >&2
  exit 2
fi
if [[ ! -f "${RUN_DIR}/designs.csv" ]]; then
  echo "ERROR: ${RUN_DIR}/designs.csv not found -- run the generation half first" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: no af2ig interpreter at ${PYTHON_BIN}" >&2
  echo "       bash scripts/utilities/bootstrap_af2ig.sh" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs/evaluation/af2ig"
cd "${REPO_ROOT}"

if [[ "${DRY_RUN}" != "1" ]]; then
  module purge   2>/dev/null || true
  module load slurm nvhpc cudnn/cuda12/9.3.0.75 mps 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
fi

# AlphaFold at these lengths fits comfortably; letting XLA preallocate 90% of
# the card is what makes a long run stable rather than fragmenting over
# thousands of differently-shaped predictions.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export TF_FORCE_UNIFIED_MEMORY="${TF_FORCE_UNIFIED_MEMORY:-0}"
export PYTHONUNBUFFERED=1

ARGS=(
  --run-dir "${RUN_DIR}"
  --data-dir "${AF2_PARAMS_DIR}"
  --num-recycles "${NUM_RECYCLES}"
  --variants ${VARIANTS}
)
if [[ -n "${TARGETS}" ]]; then ARGS+=(--targets ${TARGETS}); fi
if [[ -n "${LIMIT}" ]]; then ARGS+=(--limit "${LIMIT}"); fi
if [[ "${SKIP_FAILURES}" == "1" ]]; then ARGS+=(--skip-failures); fi
if [[ "${DRY_RUN}" == "1" ]]; then ARGS+=(--dry-run); fi

"${PYTHON_BIN}" -u scripts/evaluation/fold_af2ig.py "${ARGS[@]}" "${@}"

if [[ "${DRY_RUN}" == "1" ]]; then exit 0; fi

echo
echo "folding done -> ${RUN_DIR}/af2ig_metrics.csv"
"${PYTHON_BIN}" scripts/evaluation/score_af2ig_designability.py \
    --metrics-csv "${RUN_DIR}/af2ig_metrics.csv" \
    --output-dir "${RUN_DIR}"
