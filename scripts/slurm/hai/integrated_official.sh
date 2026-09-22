#!/bin/bash
#SBATCH --job-name=pxf_ifb_official
#SBATCH --partition=yejin,yejin-lo
#SBATCH --account=yejin
#SBATCH --gres=gpu:1
#SBATCH --constraint=hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_ifb/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_ifb/%x-%j.out
#
# HAI port of scripts/slurm/marlowe/integrated_official.sh. Same contract:
# the OFFICIAL runtime (Protenix 0.5.0+pxd), and deliberately NOT the vendored
# trees -- $REPO/PXDesign and $REPO/Protenix stay off PYTHONPATH or they
# shadow the official install with c3bfc36, which pxf/official/require.py
# refuses.
#
# --constraint=hopper, not --gres=gpu:h200:1. This env pins torch
# 2.3.1+cu121, which has no sm100 kernels: on a b200 cuda.is_available()
# still returns True and the failure arrives at the first kernel launch. The
# feature excludes blackwell structurally while keeping both h100 and h200
# eligible, which matters because `yejin` reaches only the two h200 nodes and
# they are routinely held for a day at a time. yejin-lo adds the five h100
# nodes at PriorityTier 10 with PreemptMode=REQUEUE -- a preempted cell is
# requeued, and the matrix resumes per arm off a shared prefix.
set -uo pipefail
REPO="${PXF_REPO:-/hai/scratch/yfsun/proteo_aa_worktrees/bdm}"
BUNDLE="${BUNDLE:-/hai/scratch/yfsun/pxf_handoff/pxf_hai_bundle}"
PRISTINE="${PXF_PRISTINE:-/hai/scratch/yfsun/pxdesign_official/PXDesign}"
# FaMPNN comes from the main checkout's initialised submodule rather than from
# `git submodule update --init` here: initialising submodules in this worktree
# would also materialise PXDesign/Protenix next to the official install.
FAMPNN="${PXF_FAMPNN:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-pxdesign-fampnn-pack/fampnn}"
PROTEOAA="${PROTEOAA_ROOT:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-sc-adaptation-phases}"
OUTROOT="${OUTROOT:-/hai/scratch/yfsun/pxf_runs/integrated_feedback_v1}"
# configs.resolved/, not configs/: the staged YAMLs carry an @BUNDLE@
# placeholder for the target structure, and fetch_bundle_on_hai.sh renders
# them into this sibling directory (rendering in place would fight a
# re-pull) after asserting every file: actually opens.
PREPARED="${PREPARED:-$BUNDLE/targets/binder_bench_targets/configs.resolved}"

cd "$REPO"
source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate /hai/scratch/yfsun/envs/pxdesign_official
export PYTHONPATH="$REPO:$PRISTINE:$FAMPNN"
export PROTENIX_ROOT_DIR="$BUNDLE/official_release_data"
export PROTENIX_DATA_ROOT_DIR="$BUNDLE/official_release_data/ccd_cache"
export PROTEOAA_ROOT="$PROTEOAA"
export PROTEOAA_METRICS_ROOT="$PROTEOAA"
export LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TMPDIR=/hai/scratch/yfsun/tmp
export TRITON_CACHE_DIR="$TMPDIR/triton-${SLURM_JOB_ID:-local}"
mkdir -p "$TRITON_CACHE_DIR" "$OUTROOT"

echo "=== provenance ==="
echo "node      : $(hostname)  job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo "repo HEAD : $(git -C "$REPO" rev-parse --short HEAD)  [$(git -C "$REPO" rev-parse --abbrev-ref HEAD)]"
echo "bundle    : $(cat "$BUNDLE/repo/HEAD" 2>/dev/null | head -1)"
# The handoff asks for the digest of every checkpoint actually loaded, so the
# HAI run can be matched against the Marlowe provenance in $BUNDLE/reports/.
for f in "$BUNDLE/checkpoints/donors/pxdesign_v0.1.0.pt" \
         "$BUNDLE/checkpoints/donors/fampnn_0_3.pt" \
         "$BUNDLE/checkpoints/bs_seq_sc/J03_seed0_step00000500.pt" \
         "$BUNDLE/checkpoints/bs_seq_sc/J03_seed1_step00000500.pt"; do
  echo "  $(sha256sum "$f" | cut -c1-16)  $(basename "$f")"
done
python -c "from pxf.official.require import official_protenix_available as a; print('official:', a())" || exit 1

echo "=== 3a. one generation cell: PDL1, length 80, seed 101, seven outputs ==="
python scripts/run_integrated_binder_matrix.py \
  --targets-config   "$BUNDLE/targets/configs_binder_benchmark/targets.yaml" \
  --prepared-dir     "$PREPARED" \
  --checkpoint-dir   "$BUNDLE/checkpoints/donors" \
  --checkpoint-selection "$BUNDLE/selection/selected_checkpoints.json" \
  --bs-checkpoint 0="$BUNDLE/checkpoints/bs_seq_sc/J03_seed0_step00000500.pt" \
  --bs-checkpoint 1="$BUNDLE/checkpoints/bs_seq_sc/J03_seed1_step00000500.pt" \
  --fampnn-checkpoint "$BUNDLE/checkpoints/donors/fampnn_0_3.pt" \
  --fampnn-variant 0.3 \
  --targets PDL1 --lengths 80 --seeds 101 \
  --out "$OUTROOT/generation_cell"
echo "EXIT=$?"
