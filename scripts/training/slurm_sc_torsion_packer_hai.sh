#!/bin/bash
#SBATCH --job-name=sc-torsion-packer
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --array=0-3
#SBATCH --output=logs/training/sc_torsion_packer/%x-%A_%a.out
#SBATCH --error=logs/training/sc_torsion_packer/%x-%A_%a.err
#
# THE 2x2: what does Stage 1's a_token carry that a PLM does not?
#
#   task 0  none      BB + res_type                    (geometry-only reference)
#   task 1  a_token   + Stage 1's structure-aware token
#   task 2  plm       + frozen ESM-2 650M              (APM's own setting)
#   task 3  both
#
# The channel enters at exactly one point -- projected to c_node and ADDED to
# the node embedding, where APM does `init_node_embed += plm_s` -- and all four
# arms construct the same parameters, so they differ in information only.
#
# Both arms are the same scratch SC warm-up curriculum as
# `slurm_official_sc_scratch_hai.sh` -- monomer-only, native types, native
# frames, frozen backbone and FAMPNN -- with S_phi replaced by the one-step
# torsion packer. Nothing else differs between the arms: same seed, same data
# order, same parameter count, same initial weights. See
# docs/sc_torsion_packer_apm_zh.md for the preregistered acceptance criteria.
#
#   mkdir -p logs/training/sc_torsion_packer
#   bash scripts/training/slurm_sc_torsion_packer_hai.sh --dry-run   # login node, arm a_on
#   sbatch scripts/training/slurm_sc_torsion_packer_hai.sh
#
set -euo pipefail

# This worktree, not yfsun's: the packer only exists here. Everything downstream
# resolves PROTEOAA_REPO with ${...:-default}, so exporting it is enough.
export PROTEOAA_REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
export PYTHON_BIN=${PYTHON_BIN:-/hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python}
# Datasets are read-only shares; the donor is this account's own copy.
export PROTEOAA_DATA_ROOT=${PROTEOAA_DATA_ROOT:-/hai/scratch/yfsun}
export BACKBONE_CHECKPOINT=${BACKBONE_CHECKPOINT:-/hai/scratch/shenjm/pxdesign_official/pxdesign_v0.1.0.pt}
# No AA head at all. sc_warmup supervises side chains against NATIVE residue
# types and decodes no sequence, so FaMPNN was only ever loaded, frozen and
# never called here -- `--aa-backend sc_only` drops it, which is what makes this
# experiment runnable from an account without the released FaMPNN weights.
export AA_BACKEND=sc_only

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) ARM=none ;;
  1) ARM=a_token ;;
  2) ARM=plm ;;
  3) ARM=both ;;
  *) echo "Unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
# ESM-2 650M, the model APM configures (`PLM: faESM2-650M` in base.yaml). Only
# the plm/both arms load it; the driver refuses those arms without it rather
# than training something quieter than its name.
export PLM_CHECKPOINT=${PLM_CHECKPOINT:-/hai/scratch/shenjm/plm_weights/esm2_t33_650M_UR50D.pt}
PLM_ARGS=()
if [[ "$ARM" == plm || "$ARM" == both ]]; then
  [[ -f "$PLM_CHECKPOINT" ]] || { echo "Missing PLM weights: $PLM_CHECKPOINT" >&2; exit 2; }
  PLM_ARGS=(--sc-packer-plm-checkpoint "$PLM_CHECKPOINT")
fi

RUN_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-dry-run}}
export OUTPUT_DIR=${OUTPUT_DIR:-/hai/scratch/shenjm/proteo_aa_runs/sc_torsion_packer/${RUN_ID}/${ARM}}

RUN_OPTIONS=()
if [[ ${1:-} == --dry-run ]]; then
  shift
  RUN_OPTIONS=(--dry-run)
fi

# Reuse the validated scratch warm-up launcher rather than restating its
# curriculum: it owns the monomer-only data mixture, the frozen components, the
# crop, the LR and the evaluation cadence, and those must stay identical to the
# S_phi baseline for the numbers to be comparable.
exec bash "$PROTEOAA_REPO/scripts/training/slurm_official_sc_scratch_hai.sh" "${RUN_OPTIONS[@]}" \
  --sc-torsion-packer --sc-packer-seq-cond "$ARM" "${PLM_ARGS[@]}" \
  --stage4-weight-sc-chi "${WEIGHT_SC_CHI:-1.0}" \
  --seed "${SEED:-0}" \
  "$@"
