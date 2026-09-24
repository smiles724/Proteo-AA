#!/bin/bash
#SBATCH --job-name=unc_fold
#SBATCH --partition=yejin,yejin-lo
#SBATCH --constraint=hopper
#SBATCH --account=yejin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_uncond/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_uncond/%x-%j.out
# ESMFold shard. Caches by fold_id, so a requeue or a rerun costs nothing and
# shards may overlap safely. `ml` has torch but no transformers; the official
# env has both and ESMFold here touches neither Protenix nor PXDesign.
set -uo pipefail
cd /hai/scratch/yfsun/proteo_aa_worktrees/mev
export ESMFOLD_WEIGHTS=/hai/scratch/yfsun/tool_weights/facebook__esmfold_v1
export TMPDIR=/hai/scratch/yfsun/tmp HF_HOME=/hai/scratch/yfsun/tmp/hf
source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate /hai/scratch/yfsun/envs/pxdesign_official
python scripts/uncond/esmfold_refold.py \
  --sequences "${SEQS:?set SEQS}" \
  --out "${FOLD_OUT:?set FOLD_OUT}" \
  --shard-index "${SHARD_INDEX:-0}" --shard-count "${SHARD_COUNT:-1}" \
  ${CSV_NAME:+--csv-name "$CSV_NAME"}
status=$?; echo "EXIT=$status"; exit $status
