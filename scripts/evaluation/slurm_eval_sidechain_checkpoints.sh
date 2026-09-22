#!/bin/bash
#SBATCH --job-name=proteo-aa-sc-ckpt-cmp
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/validation/sidechain_checkpoint_compare/%x-%j.out
#SBATCH --error=logs/validation/sidechain_checkpoint_compare/%x-%j.err

# Score several checkpoints under ONE eval configuration, so the differences are
# attributable to the checkpoints and not to the harness.
#
# Why this exists: the Stage III run reported val_sc_local ~3.3, against ~1.83 for
# the Stage II checkpoint it started from -- but those numbers came from different
# eval sets. Stage III passes --max-n-token = crop (384), which rebuilds the
# validation index with a 384-token cap and yields 308 proteins, while Stage II
# evaluated 491 at a 640 cap. A 1.83 -> 3.3 move across two different sets is not
# a measurement. This runs every checkpoint at 640/491 with GT frames and GT types
# (`sidechain_warmup`), which is the side-chain packing measurement the 1.83 came
# from.
#
# CHECKPOINTS: space-separated list. Each is scored into its own subdirectory.

set -euo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
OUT_ROOT="${OUT_ROOT:-/hai/scratch/yfsun/proteo_aa_runs/sidechain_checkpoint_compare}"
# Same for every checkpoint -- that is the point.
TRAINING_STAGE="${TRAINING_STAGE:-sidechain_warmup}"
CROP_SIZE="${CROP_SIZE:-640}"
MAX_N_TOKEN="${MAX_N_TOKEN:-640}"
NUM_SAMPLES="${NUM_SAMPLES:-491}"
DTYPE="${DTYPE:-bf16}"

if [[ -z "${CHECKPOINTS:-}" ]]; then
  echo "ERROR: set CHECKPOINTS to a space-separated list of .pt files." >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs/validation/sidechain_checkpoint_compare" "${OUT_ROOT}"
cd "${REPO_ROOT}"

export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== config held fixed: stage=${TRAINING_STAGE} crop=${CROP_SIZE} max_n_token=${MAX_N_TOKEN} n=${NUM_SAMPLES} ==="

for CK in ${CHECKPOINTS}; do
  if [[ ! -f "${CK}" ]]; then
    echo "SKIP (missing): ${CK}" >&2
    continue
  fi
  # Name the output by run + step so two runs' step6000 cannot collide.
  RUN_NAME="$(basename "$(dirname "$(dirname "${CK}")")")"
  TAG="${RUN_NAME}__$(basename "${CK}" .pt)"
  OUT="${OUT_ROOT}/${TAG}"
  mkdir -p "${OUT}"
  echo ""
  echo "=== scoring ${TAG} ==="
  "${PYTHON_BIN}" -u scripts/evaluation/eval_protenix_monomer.py \
    --checkpoint "${CK}" \
    --training-stage "${TRAINING_STAGE}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT}" \
    --crop-size "${CROP_SIZE}" \
    --max-n-token "${MAX_N_TOKEN}" \
    --num-samples "${NUM_SAMPLES}" \
    --dtype "${DTYPE}" \
    --device cuda \
    --protenix-code-dir "${PROTENIX_CODE_DIR}" \
    --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
    "${@}" 2>&1 | tail -25
done

echo ""
echo "=== summary ==="
for CK in ${CHECKPOINTS}; do
  RUN_NAME="$(basename "$(dirname "$(dirname "${CK}")")")"
  TAG="${RUN_NAME}__$(basename "${CK}" .pt)"
  J="${OUT_ROOT}/${TAG}/metrics.json"
  if [[ -f "${J}" ]]; then
    "${PYTHON_BIN}" - "${TAG}" "${J}" <<'PY'
import json, sys, math
tag, path = sys.argv[1], sys.argv[2]
d = json.load(open(path))
m = d.get("metrics", {})
sc = m.get("sc_local")
rmsd = f"{math.sqrt(sc):.3f}" if isinstance(sc, (int, float)) and sc >= 0 else "n/a"
print(f"  {tag:58s} n={d.get('evaluated_samples')} "
      f"sc_local={sc if sc is None else round(sc, 4)} rmsd={rmsd} "
      f"aa_acc={m.get('aa_acc')}")
PY
  else
    echo "  ${TAG}: no metrics.json"
  fi
done
