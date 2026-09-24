#!/bin/bash
#SBATCH --job-name=nov
#SBATCH --partition=yejin-lo
#SBATCH --account=yejin
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_uncond/%x-%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_uncond/%x-%j.out
# Novelty: max TM-score to the nearest PDB chain, via foldseek TMalign.
# No GPU -- foldseek is CPU-only, so this asks for cores and nothing else.
# PER_SAMPLE is optional: set it to restrict to designable samples (the
# published convention), leave it unset to score every sample.
set -uo pipefail
cd /hai/scratch/yfsun/proteo_aa_worktrees/mev
export TMPDIR=/hai/scratch/yfsun/tmp
source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate /hai/scratch/yfsun/envs/pxdesign_official
python scripts/uncond/novelty.py \
  --samples-dir "${SAMPLES:?set SAMPLES}" \
  --db /hai/scratch/yfsun/foldseek_db/pdb \
  --out "${NOV_OUT:?set NOV_OUT}" \
  --threads "${SLURM_CPUS_PER_TASK:-16}" \
  ${PER_SAMPLE:+--per-sample "$PER_SAMPLE"} \
  ${EXHAUSTIVE:+--exhaustive} ${REUSE_HITS:+--reuse-hits}
status=$?; echo "EXIT=$status"; exit $status
