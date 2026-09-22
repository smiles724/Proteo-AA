#!/bin/bash
# Push the repo state the pending integrated-feedback jobs need.
#
# Scope: the MAIN repo only. The vendored PXDesign submodule is dirty (a local
# `d_lm` bridging patch in pxdesign/model/embedders.py) and is deliberately NOT
# pushed -- the official runtime excludes $ROOT/PXDesign from PYTHONPATH on
# purpose, so that patch has no bearing on the pending jobs. It is captured as
# a patch file instead, so the state is recorded rather than lost.
set -euo pipefail

ROOT="${PXF_REPO:-/users/yfsun/Proteo-AA-pxdesign-fampnn-pack}"
BRANCH="${PXF_BRANCH:-exp/binder-design-matrix}"
cd "$ROOT"

test "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" \
  || { echo "refusing: HEAD is $(git rev-parse --abbrev-ref HEAD), not $BRANCH"; exit 1; }

# 1. Record the vendored-PXDesign patch as a file.
mkdir -p patches
git -C PXDesign diff > patches/pxdesign_vendored_embedders_d_lm.patch
if [ -s patches/pxdesign_vendored_embedders_d_lm.patch ]; then
  echo "captured $(wc -l < patches/pxdesign_vendored_embedders_d_lm.patch) lines of vendored-PXDesign patch"
else
  rm -f patches/pxdesign_vendored_embedders_d_lm.patch
fi

# 2. Stage by explicit path. Never `git add -A` here: that would commit the
#    dirty submodule pointer as `...-dirty`, which is not a resolvable commit.
PATHS=(
  scripts/utilities/install_pxdesign_official.sh
  scripts/handoff/
  docs/handoff/
  configs/phase1_structures_afdb.marlowe.txt
  configs/val_structures_afdb.marlowe.txt
)
[ -f patches/pxdesign_vendored_embedders_d_lm.patch ] && PATHS+=(patches/pxdesign_vendored_embedders_d_lm.patch)
for p in "${PATHS[@]}"; do [ -e "$p" ] && git add -- "$p"; done
git reset -q -- PXDesign 2>/dev/null || true

if git diff --cached --quiet; then
  echo "nothing new to commit"
else
  git commit -q -F - <<'MSG'
handoff: HAI transfer bundle for the pending generation jobs

The generation cell has been queued behind a fairshare of 0.003568 for
long enough that the work is better finished on HAI, which has the
official PXDesign runtime and no such backlog.

Adds scripts/handoff/{push_pending,stage_hai_bundle,fetch_bundle_on_hai}.sh
and docs/handoff/hai_run_prompt.md. The bundle carries only what the
pending jobs actually read: the two A_BS checkpoints, the four feedback
checkpoints at their selected step, the selection artifact with its
absolute paths replaced by a @BUNDLE@ placeholder, the prepared targets,
and the provenance JSON. It deliberately does NOT carry the 872M backbone
collection -- the matrix generates its own backbones per cell -- nor the
305M conditioning cache, which is only needed to retrain.

Also records the local d_lm bridging patch to the vendored PXDesign as a
patch file. The submodule pointer is not pushed: the official runtime
excludes $ROOT/PXDesign from PYTHONPATH by design, so the patch is
irrelevant to these jobs and committing a `-dirty` pointer would be
unresolvable.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
  echo "committed $(git rev-parse --short HEAD)"
fi

git push origin "$BRANCH"
echo "pushed $BRANCH -> $(git rev-parse --short HEAD)"
