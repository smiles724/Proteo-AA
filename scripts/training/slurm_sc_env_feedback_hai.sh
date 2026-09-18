#!/bin/bash
#SBATCH --job-name=sc-env-feedback
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --array=0-1
#SBATCH --output=logs/training/env_feedback/%x-%A_%a.out
#SBATCH --error=logs/training/env_feedback/%x-%A_%a.err
#
# Side-chain -> backbone feedback through h_res', over a FROZEN APM packer.
# docs/sc_to_bb_hres_feedback_zh.md
#
# Both runs use arm C (h_res' + the residue-KNN coordinate neighbourhood) and
# differ ONLY in what is allowed to train:
#
#   0  fb_only   feedback only. Packer frozen at APM's released weights, backbone
#                frozen at official PXDesign. One thing moves, so an effect is
#                attributable to the feedback and to nothing else.
#   1  like_s3   feedback + packer + backbone, i.e. Stage III's degrees of
#                freedom (measured from 111408: bb_optimizer 770 tensors over
#                backbone/condition/feedback, sc_optimizer 389 over the packer).
#                Not attributable by construction -- it answers "how far does
#                this get end to end", which is a different question.
#
# Arm F (no neighbourhood) and C-only (atom-id features, no packer node_embed)
# remain the controls to run once one of these shows an effect; they are what
# separate "the coordinates helped" from "any extra capacity helped".
#
# The ordinary `mlp` residue-type head is BUILT AND NEVER TRAINED, rather than
# using --aa-backend sc_only. sc_only builds no head, and outside the supervised
# SC phases (which return early through adaptation_forward) the ordinary forward
# and its consumers all assume one exists -- chasing that produced four
# consecutive failures with no bound on the remainder. Here the head is
# unsupervised (--weight-aa 0, SC->AA off, not in feedback_adapt's trainable
# set) and its logits are discarded anyway because --sc-force-gt-type-logits
# feeds the packer the native types. A few million unread parameters is the
# cheaper correctness.
#
# The packer is APM's released checkpoint and it is FROZEN: it is an external
# fixed reference at 1.3469 symmetry_rmsd, so the feedback's effect does not get
# entangled with our own packer's quality.
#
# Composition, not warm start. `--warm-start-checkpoint pxdesign_v0.1.0.pt` is
# refused ("A complete integrated checkpoint is required", checkpoints.py:219)
# because the official release carries diffusion_module + design_condition_
# embedder and no sidechain_module at all. `--backbone-checkpoint` +
# `--sidechain-checkpoint` is the component-composition path built for exactly
# this: official backbone plus an external packer. `--sc-load-packer-from`
# stays on as well -- it overlays the same tensors after the load and raises if
# nothing matched, so the packer being in place is asserted twice rather than
# assumed once.
set -euo pipefail

REPO=${PROTEOAA_REPO:-/hai/scratch/shenjm/wt_torsion_packer}
export PYTHONPATH="$REPO:$REPO/PXDesign:$REPO/Protenix"
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
export PROTENIX_DATA_ROOT_DIR="$PROTENIX_ROOT_DIR/common"
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/hai/scratch/shenjm/triton_cache}
PYBIN=${PYBIN:-/hai/users/s/h/shenjm/miniconda3/envs/proteoaa/bin/python}
cd "$REPO"
mkdir -p logs/training/env_feedback

PACKER_OVERLAY=${PACKER_OVERLAY:-/hai/scratch/shenjm/apm_weights/apm_packer_overlay.pt}
BACKBONE=${BACKBONE:-/hai/scratch/shenjm/pxdesign_official/pxdesign_v0.1.0.pt}
RUNROOT=${RUNROOT:-/hai/scratch/shenjm/proteo_aa_runs/sc_env_feedback}

ENV_ARGS=(--sc-env-feedback "${ENV_SOURCE:-packer}" --sc-env-use-neighbourhood)
case "${SLURM_ARRAY_TASK_ID:-0}" in
  # arm 1 TRAINS the packer, so it must not also freeze it: APM's weights are
  # the initialisation there, not a fixed reference.
  0) ARM=fb_only; TRAIN_ARGS=(--no-train-sc --sc-freeze-packer) ;;
  1) ARM=like_s3; TRAIN_ARGS=(--train-sc --bb-trainable-prefixes diffusion_module. design_condition_embedder.) ;;
  *) echo "unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
[[ -n "${ARM_OVERRIDE:-}" ]] && ARM=$ARM_OVERRIDE

RUN_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-dry-run}}
OUT="$RUNROOT/$RUN_ID/$ARM"
mkdir -p "$OUT"
echo "arm=$ARM out=$OUT node=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec "$PYBIN" scripts/training/train_protenix_monomer.py \
  --training-stage coevolution \
  --data-root "$PROTENIX_ROOT_DIR" --output-dir "$OUT" \
  --sc-torsion-packer --sc-packer-seq-cond none \
  "${ENV_ARGS[@]}" \
  --sc-env-blocks "${ENV_BLOCKS:-2}" --sc-env-neighbors "${ENV_NEIGHBORS:-16}" \
  --sc-env-detach \
  --sc-load-packer-from "$PACKER_OVERLAY" \
  --backbone-checkpoint "$BACKBONE" --sidechain-checkpoint "$PACKER_OVERLAY" \
  --crop-size "${CROP_SIZE:-384}" --max-n-token "${CROP_SIZE:-384}" \
  --max-steps "${MAX_STEPS:-20000}" --warmup-steps "${WARMUP_STEPS:-1000}" \
  "${TRAIN_ARGS[@]}" \
  --stage4-phase feedback_adapt --no-stage4-sc-to-aa --weight-refine "${WEIGHT_REFINE:-1.0}" \
  --sc-force-gt-type-logits \
  --diffusion-batch-size 1 --iters-to-accumulate "${GRAD_ACCUM:-8}" \
  --cif-dir "${CIF_DIR:-/hai/scratch/yfsun/afdb_laproteina/cif_phase1}" \
  --data-mode monomer --no-ref-pos-augment \
  --seed "${SEED:-0}" "$@"
