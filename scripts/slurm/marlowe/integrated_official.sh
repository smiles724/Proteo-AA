#!/bin/bash
#SBATCH --job-name=pxf_ifb_official
#SBATCH --partition=batch
#SBATCH --account=marlowe-m000137-pm06
#SBATCH --qos=medium
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_ifb/%x-%j.out
#SBATCH --error=/scratch/m000137-pm06/Proteo-AA/pxf/runs/logs/pxf_ifb/%x-%j.out
#
# The OFFICIAL runtime, for anything that must run under Protenix 0.5.0+pxd.
# Deliberately NOT the general wrapper: that one puts the vendored PXDesign
# and Protenix trees on PYTHONPATH, which would shadow the official install
# with c3bfc36 -- the exact pairing pxf/official/require.py refuses. Here the
# only repo path is the repo root, and PXDesign comes from the pristine
# worktree.
set -uo pipefail
ROOT="${PXF_REPO:-/users/yfsun/Proteo-AA-pxdesign-fampnn-pack}"
DATA_ROOT="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
PRISTINE="${PXF_PRISTINE:-/users/yfsun/pxdesign_pristine}"
CMD="${CMD:?set CMD}"
ARGS="${ARGS:?set ARGS}"
cd "$ROOT"
mkdir -p "$DATA_ROOT/runs/logs/pxf_ifb"
export PATH="/users/yfsun/.venvs/pxdesign_official/bin:$PATH"
export PYTHONPATH="$ROOT:$PRISTINE"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$DATA_ROOT/official_release_data}"
export PROTENIX_DATA_ROOT_DIR="${PROTENIX_DATA_ROOT_DIR:-$DATA_ROOT/official_release_data/ccd_cache}"
export PROTEOAA_ROOT="${PROTEOAA_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export PROTEOAA_METRICS_ROOT="${PROTEOAA_METRICS_ROOT:-/users/yfsun/proteo-aa-pxdesign-train}"
export LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} cmd=$CMD (OFFICIAL runtime)"
python -c "from pxf.official.require import official_protenix_available as a; print('official:', a())"
eval "python $CMD $ARGS"
echo "EXIT=$?"
