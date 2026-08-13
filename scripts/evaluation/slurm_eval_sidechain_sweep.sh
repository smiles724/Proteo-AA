#!/bin/bash
# Score ONE arm at MANY checkpoints, so its progress can be read on the
# comparable metric rather than on the training loss.
#
# Separate from slurm_eval_sidechain_arms.sh on purpose: that script has a queued
# job depending on it, and editing a script a pending job will read at launch is
# a good way to change an experiment without meaning to.
#SBATCH --job-name=proteo-aa-sc-sweep
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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

ARM_LABEL="${ARM_LABEL:-edm_global}"
ARM_DIR="${ARM_DIR:-edm_global_from_bb_step96000}"
STEPS="${STEPS:-2000 10000 20000 30000 42000}"
OUT_DIR="${OUT_DIR:-$RUNS/arm_sweep_${ARM_LABEL}}"
# "gt" is the DIAGNOSTIC start (noised ground truth). Not an inference
# protocol; the eval script stamps the artifact accordingly.
EDM_INIT="${EDM_INIT:-template}"
SUFFIX=""
[[ "${EDM_INIT}" == "gt" ]] && SUFFIX="_gtstart"

cd "${REPO_ROOT}"
mkdir -p logs/validation/sidechain_arms "${OUT_DIR}"
export PROTENIX_ROOT_DIR="${DATA_ROOT}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
export PYTHONPATH="${REPO_ROOT}:${PXDESIGN_CODE_DIR}:${PROTENIX_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

rc=0
for step in ${STEPS}; do
  ckpt="${RUNS}/${ARM_DIR}/checkpoints/step${step}.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "SKIP step${step}: no checkpoint at ${ckpt}" >&2
    continue
  fi
  echo "=== ${ARM_LABEL} @ step${step} ==="
  "${PYTHON_BIN}" scripts/evaluation/eval_sidechain_arms.py \
    --checkpoint "${ckpt}" \
    --label "${ARM_LABEL}_step${step}${SUFFIX}" \
    --edm-init "${EDM_INIT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --num-samples "${NUM_SAMPLES:-491}" \
    --protenix-code-dir "${PROTENIX_CODE_DIR}" \
    --pxdesign-code-dir "${PXDESIGN_CODE_DIR}" \
    || { echo "FAILED step${step}" >&2; rc=1; }
done

echo "=== sweep summary: ${ARM_LABEL} ==="
"${PYTHON_BIN}" - "${OUT_DIR}" "${ARM_LABEL}" <<'PY'
import json, re, sys
from pathlib import Path
out_dir, label = Path(sys.argv[1]), sys.argv[2]
rows = []
for f in out_dir.glob(f"arm_eval_{label}_step*.json"):
    d = json.load(open(f))
    m = re.search(r"_step(\d+)$", d["label"])
    if m:
        rows.append((int(m.group(1)), d["atom_weighted_mse"], d["atom_weighted_rmsd_A"]))
rows.sort()
if rows:
    csv = out_dir / f"sweep_{label}.csv"
    with csv.open("w") as fh:
        fh.write("step,atom_weighted_mse,atom_weighted_rmsd_A\n")
        for s, mse, r in rows:
            fh.write(f"{s},{mse:.6f},{r:.6f}\n")
    for s, mse, r in rows:
        print(f"  step {s:>6}   {mse:8.4f} A^2   {r:7.4f} A")
    print(f"wrote_sweep={csv}")
else:
    print("  (no results)")
PY
exit $rc
