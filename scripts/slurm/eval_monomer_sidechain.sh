#!/bin/bash
#SBATCH --job-name=pxf_sc_monomer
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_sc_monomer/%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_sc_monomer/%j.out
#
# Side-chain packing inference on monomer benchmarks, scored with Proteo-AA's own
# side-chain metrics. Backbone + native sequence in, side chains out; no design.
#
# h200 is requested explicitly: the yejin partition also has a b200 node, whose
# sm_100 this env's torch 2.7.1+cu126 has no kernels for (it stops at sm_90).
#
# Submit from an interactive job with SLURM_* cleared, or the job silently
# inherits this shell's allocation:
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       sbatch scripts/slurm/eval_monomer_sidechain.sh
set -euo pipefail

# Derived from this script's location so the repo can be moved or renamed
# (it already has been, into ~/"Proteo-AA old"/). Quote it everywhere: the
# current path contains a space.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=/hai/scratch/yfsun/proteo_aa_runs/pxf_sc_monomer/${SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-$SLURM_JOB_ID"

echo "node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'arch_list', torch.cuda.get_arch_list()[-3:])"

SAMPLES=${SAMPLES:-3}
for SET in casp14 casp15 casp13; do
    DATA="$ROOT/fampnn/data/$SET/pdbs"
    [ -d "$DATA" ] || { echo "skip $SET (no data)"; continue; }
    echo ""
    echo "################ $SET ################"
    python scripts/eval_monomer_sidechain.py \
        --pdb-dir "$DATA" \
        --label "$SET" \
        --out "$OUT/$SET" \
        --weights 0.0 \
        --samples "$SAMPLES" \
        --seed 0
done

echo ""
echo "outputs under $OUT"
