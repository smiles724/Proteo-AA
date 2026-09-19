#!/bin/bash
# AlphaProteo-10 designability for the two SC->BB feedback arms AND the official
# reference, submitted as ONE batch.
#
# Why all three together rather than reusing the official 8.26% already on
# record: that number came from a different run with its own sampler build and
# its own AF2/scoring environment. Designability is the measure this whole line
# of work is judged on, so the reference is regenerated alongside the arms and
# every number in the comparison comes from one batch. The existing
# submit_alphaproteo_generation.sh pairs official with exactly one Proteo-AA
# checkpoint, which cannot express a three-way comparison.
#
# Baselines to beat, from docs/alphaproteo10_designability_result_zh.md:
#   official PXDesign v0.1.0   8.26%   (271 / 3280 strict passes)
#   Proteo-AA 111408/step6000  0.0%    (0 / 3280, coverage 1.0 -- a real zero)
#
#   RUN=... STEP=final bash scripts/evaluation/submit_sc_env_feedback_alphaproteo.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/hai/scratch/shenjm/wt_torsion_packer}"
RUN="${RUN:-/hai/scratch/shenjm/proteo_aa_runs/sc_env_feedback/119143}"
STEP="${STEP:-final}"
OFFICIAL="${OFFICIAL:-/hai/scratch/shenjm/pxdesign_official/pxdesign_v0.1.0.pt}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/hai/scratch/shenjm/proteo_aa_runs/alphaproteo10_designability/sc_env_${RUN_TAG}}"
SCRIPT="${REPO_ROOT}/scripts/evaluation/slurm_generate_alphaproteo_designability.sh"

pick() {
  local d="$RUN/$1/checkpoints"
  if [[ "$STEP" == final ]]; then ls -t "$d"/step*.pt 2>/dev/null | head -1
  else echo "$d/step${STEP}.pt"; fi
}
FB_ONLY="$(pick fb_only)"
LIKE_S3="$(pick like_s3)"
for f in "$FB_ONLY" "$LIKE_S3" "$OFFICIAL"; do
  [[ -f "$f" ]] || { echo "missing checkpoint: '$f'" >&2; exit 2; }
done

mkdir -p "${REPO_ROOT}/logs/validation/alphaproteo10" "${RUN_ROOT}"
cd "${REPO_ROOT}"
export REPO_ROOT RUN_ROOT
export NUM_DESIGNS_PER_TARGET="${NUM_DESIGNS_PER_TARGET:-1}"
export FIXED_LENGTH="${FIXED_LENGTH:-}"
export LENGTH_MIN="${LENGTH_MIN:-80}" LENGTH_MAX="${LENGTH_MAX:-130}"
export SEED="${SEED:-42}" N_STEP="${N_STEP:-400}"
export TARGETS="${TARGETS:-bhrf1,h1,il17a,il7ra,ir,pdl1,sc2rbd,tnfa,trka,vegfa}"

echo "fb_only : $FB_ONLY"
echo "like_s3 : $LIKE_S3"
echo "official: $OFFICIAL"
echo "run root: $RUN_ROOT"

px_job="$(CHECKPOINT="$OFFICIAL" MODEL_LABEL=pxdesign_official MODEL_MODE=pxdesign \
  sbatch --parsable --job-name=alpha10-official "$SCRIPT")"
fb_job="$(CHECKPOINT="$FB_ONLY" MODEL_LABEL=sc_env_fb_only MODEL_MODE=proteoaa \
  AA_READOUTS="${AA_READOUTS:-final}" AA_READOUT_SIGMA="${AA_READOUT_SIGMA:-0.4}" \
  sbatch --parsable --job-name=alpha10-fb-only "$SCRIPT")"
s3_job="$(CHECKPOINT="$LIKE_S3" MODEL_LABEL=sc_env_like_s3 MODEL_MODE=proteoaa \
  AA_READOUTS="${AA_READOUTS:-final}" AA_READOUT_SIGMA="${AA_READOUT_SIGMA:-0.4}" \
  sbatch --parsable --job-name=alpha10-like-s3 "$SCRIPT")"

cat <<MSG

RUN_ROOT=$RUN_ROOT
generation jobs: official=$px_job  fb_only=$fb_job  like_s3=$s3_job

When all three finish, score them (one environment, all three):
  RUN_ROOT=$RUN_ROOT \\
  PXDBENCH_DIR=<PXDesignBench-v0.1.2> \\
  PXDBENCH_PYTHON=<pxdbench-env>/bin/python \\
  TOOL_WEIGHTS_ROOT=<tool_weights> \\
  bash scripts/evaluation/submit_alphaproteo_scoring.sh

COVERAGE IS NOT OPTIONAL. A previous round reported 0.0% for Proteo-AA when its
40 scoring tasks had never run, and the summariser counted missing scores as
failures. Check coverage == 1.0 before reading any designability number.
MSG
