#!/bin/bash
# Score every finished side-chain arm on the same 491-protein set, each under its
# own inference protocol. Skips an arm whose checkpoint is absent rather than
# failing the whole job: this runs on `afterany`, so a crashed arm must not take
# the other two's numbers down with it.
#SBATCH --job-name=proteo-aa-sc-armeval
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/validation/sidechain_arms/%x-%j.out
#SBATCH --error=logs/validation/sidechain_arms/%x-%j.err

set -uo pipefail

source ~/.bashrc
conda activate ml

REPO_ROOT="${REPO_ROOT:-/hai/users/y/f/yfsun/Proteo-AA}"
DATA_ROOT="${DATA_ROOT:-/hai/scratch/yfsun/protenix_data}"
PROTENIX_CODE_DIR="${PROTENIX_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/Protenix}"
PXDESIGN_CODE_DIR="${PXDESIGN_CODE_DIR:-/hai/users/y/f/yfsun/Protein Project/11/PXDesign}"
PYTHON_BIN="${PYTHON_BIN:-/hai/users/y/f/yfsun/miniconda3/envs/ml/bin/python}"
RUNS="${RUNS:-/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_sidechain_warmup}"
OUT_DIR="${OUT_DIR:-$RUNS/arm_eval_491}"
STEP="${STEP:-step50000.pt}"

cd "${REPO_ROOT}"
mkdir -p logs/validation/sidechain_arms "${OUT_DIR}"
export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# label:run-directory
ARMS=(
  "edm_global:edm_global_from_bb_step96000"
  "fixed_global:fixed_global_from_bb_step96000"
  "fixed_local:fixed_local_from_bb_step96000"
)

rc=0
for entry in "${ARMS[@]}"; do
  label="${entry%%:*}"
  run="${entry#*:}"
  ckpt="${RUNS}/${run}/checkpoints/${STEP}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "SKIP ${label}: no checkpoint at ${ckpt}" >&2
    continue
  fi
  echo "=== ${label} <- ${ckpt} ==="
  "${PYTHON_BIN}" scripts/evaluation/eval_sidechain_arms.py \
    --checkpoint "${ckpt}" \
    --label "${label}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --num-samples "${NUM_SAMPLES:-491}" \
    --protenix-code-dir "${PROTENIX_CODE_DIR}" \
    --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
    || { echo "FAILED ${label}" >&2; rc=1; }
done

echo "=== summary ==="
for f in "${OUT_DIR}"/arm_eval_*.json; do
  [[ -f "$f" ]] || continue
  "${PYTHON_BIN}" -c "
import json,sys
d=json.load(open('$f'))
print(f\"{d['label']:<14} protocol={d['protocol']:<17} \"
      f\"atom_mse={d['atom_weighted_mse']:.4f} A^2  \"
      f\"rmsd={d['atom_weighted_rmsd_A']:.4f} A  \"
      f\"n={d['n_proteins']}\")
"
done
exit $rc
