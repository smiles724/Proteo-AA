#!/bin/bash
#SBATCH --job-name=pxf_seqdes
#SBATCH --partition=yejin
#SBATCH --account=yejin
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/hai/scratch/yfsun/proteo_aa_runs/pxf_seqdes/%j.out
#SBATCH --error=/hai/scratch/yfsun/proteo_aa_runs/pxf_seqdes/%j.out
#
# PXDesign generates a backbone; FaMPNN then designs BOTH the sequence and the
# side chains on it (SeqDenoiser.sample -- upstream's seq_design path).
#
# h200 is requested explicitly: the yejin partition also has a b200 node whose
# sm_100 this env's torch has no kernels for, and the failure only surfaces on
# the first kernel launch.
#
# Submit from an interactive job with SLURM_* cleared, or the job silently
# inherits this shell's allocation:
#   env $(env | grep -o '^SLURM_[^=]*' | sed 's/^/-u /' | tr '\n' ' ') \
#       sbatch scripts/slurm/design_fullatom.sh
set -euo pipefail

# sbatch copies this script to /var/lib/slurm/scripts, so BASH_SOURCE does NOT
# point at the repo under SLURM. SLURM_SUBMIT_DIR is the submission cwd
# (the repo root); fall back to BASH_SOURCE only for direct execution.
if [ -n "${PXF_REPO:-}" ]; then ROOT="$PXF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
if [ ! -f "$ROOT/pxf/provenance.py" ]; then
    echo "ROOT=$ROOT is not the pxf repo; set PXF_REPO" >&2; exit 2
fi
OUT=${OUT:-/hai/scratch/yfsun/proteo_aa_runs/pxf_seqdes/${SLURM_JOB_ID}}
CKPT_DIR=${CKPT_DIR:-/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/runs/component_donors}
N_SAMPLE=${N_SAMPLE:-8}
N_STEP=${N_STEP:-200}
mkdir -p "$OUT"
cd "$ROOT"

source /hai/users/y/f/yfsun/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONPATH="$ROOT:$ROOT/PXDesign:$ROOT/Protenix:$ROOT/fampnn"
# Point Protenix at the CCD cache already on scratch so nothing is downloaded.
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/hai/scratch/yfsun/protenix_data}
export PROTENIX_DATA_ROOT_DIR=${PROTENIX_DATA_ROOT_DIR:-/hai/scratch/yfsun/protenix_data/common}
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-$SLURM_JOB_ID"
export TQDM_DISABLE=1

echo "node=$(hostname) job=$SLURM_JOB_ID out=$OUT"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'arch', torch.cuda.get_arch_list()[-2:])"

# The shipped target yaml uses paths relative to the PXDesign directory; rewrite
# them absolute so the run does not depend on cwd.
TARGET_SRC=${TARGET:-$ROOT/PXDesign/examples/PDL1_quick_start.yaml}
TARGET_RESOLVED="$OUT/target.yaml"
python - "$TARGET_SRC" "$ROOT/PXDesign" "$TARGET_RESOLVED" <<'PYEOF'
import sys, yaml
from pathlib import Path
src, base, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cfg = yaml.safe_load(src.read_text())
target = cfg["target"]
target["file"] = str((base / target["file"]).resolve())
for chain in (target.get("chains") or {}).values():
    if chain and chain.get("msa"):
        chain["msa"] = str((base / chain["msa"]).resolve())
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("resolved target ->", dst)
print(dst.read_text())
PYEOF

python scripts/design_fullatom.py \
    --input-json "$TARGET_RESOLVED" \
    --pxdesign-checkpoint-dir "$CKPT_DIR" \
    --out "$OUT" \
    --n-sample "$N_SAMPLE" \
    --n-step "$N_STEP" \
    --fampnn-weights 0.3 \
    --seed 0 \
    ${EXTRA_ARGS:-}

echo "outputs under $OUT"
