#!/bin/bash
# RUN THIS ON HAI, not on Marlowe.
#
# Marlowe cannot reach HAI: /hai is not mounted on the Marlowe login nodes and
# the HAI hostname does not resolve there. So the transfer is a PULL from the
# HAI side. If your site routes the other way, skip step 1 and point
# --bundle at a copy you moved yourself; everything after step 1 is transport
# agnostic.
#
#   MARLOWE_HOST=login.marlowe.stanford.edu \
#   scripts/handoff/fetch_bundle_on_hai.sh --dest /hai/scratch/yfsun/pxf_handoff
#
#   scripts/handoff/fetch_bundle_on_hai.sh --bundle /path/already/copied --no-pull
set -euo pipefail

MARLOWE_HOST="${MARLOWE_HOST:-}"                     # e.g. login.marlowe.stanford.edu
MARLOWE_USER="${MARLOWE_USER:-yfsun}"
REMOTE_BUNDLE="${REMOTE_BUNDLE:-/scratch/m000137-pm06/Proteo-AA/pxf/handoff/pxf_hai_bundle}"
DEST="${PXF_HAI_ROOT:-/hai/scratch/yfsun/pxf_handoff}"
BUNDLE=""
PULL=1

while [ $# -gt 0 ]; do case "$1" in
  --dest) DEST="$2"; shift 2 ;;
  --bundle) BUNDLE="$2"; shift 2 ;;
  --no-pull) PULL=0; shift ;;
  --host) MARLOWE_HOST="$2"; shift 2 ;;
  *) echo "unknown flag: $1"; exit 2 ;;
esac; done

BUNDLE="${BUNDLE:-$DEST/pxf_hai_bundle}"

# ---------------------------------------------------------------- 1. pull
if [ "$PULL" = 1 ]; then
  [ -n "$MARLOWE_HOST" ] || { echo "set MARLOWE_HOST (or pass --no-pull)"; exit 2; }
  mkdir -p "$DEST"
  # -H preserves hardlinks, --partial survives a dropped VPN, and the
  # checksum pass at step 2 is what actually decides whether it arrived
  # intact -- rsync's own size/mtime check is not enough for weights.
  rsync -aHP --partial --info=progress2 \
    "$MARLOWE_USER@$MARLOWE_HOST:$REMOTE_BUNDLE/" "$BUNDLE/"
fi

[ -d "$BUNDLE" ] || { echo "no bundle at $BUNDLE"; exit 1; }

# ------------------------------------------------------------- 2. verify
echo "== verifying digests =="
( cd "$BUNDLE" && sha256sum -c --quiet SHA256SUMS ) \
  || { echo "TRANSFER CORRUPT: re-run the pull before doing anything else"; exit 1; }
echo "  $(wc -l < "$BUNDLE/SHA256SUMS") file(s) OK"

# Symlinks are not in SHA256SUMS (they have no content of their own). The
# staging side rewrote the absolute ones to be relative so they survive the
# move; confirm none of them dangles here, because a dangling CCD link shows
# up much later as an opaque Protenix featurisation failure.
if [ -f "$BUNDLE/SYMLINKS" ] && [ -s "$BUNDLE/SYMLINKS" ]; then
  BAD=0
  while IFS= read -r line; do
    rel="${line%% -> *}"
    [ -e "$BUNDLE/${rel#./}" ] || { echo "  DANGLING: $line"; BAD=1; }
  done < "$BUNDLE/SYMLINKS"
  [ "$BAD" = 0 ] || { echo "dangling symlink(s) after transfer; re-pull with rsync -aH"; exit 1; }
  echo "  $(wc -l < "$BUNDLE/SYMLINKS") symlink(s) resolve"
fi

# ---------------------------------------------- 3. rehydrate the selection
# The staged artifact carries @BUNDLE@ and @PXDESIGN_DONOR@ placeholders
# because the Marlowe absolute paths do not exist here. The sha256 fields are
# content digests and are NOT rewritten: check_policy compares against those,
# so a wrong donor on this side fails closed instead of silently producing a
# differently-conditioned arm.
DONOR="${PXDESIGN_DONOR:-}"
if [ -z "$DONOR" ]; then
  for c in "$BUNDLE/checkpoints/donors/pxdesign_v0.1.0.pt" \
           /hai/scratch/yfsun/pxdesign_official/release_data/checkpoint/pxdesign_v0.1.0.pt \
           /hai/scratch/yfsun/envs/pxdesign_official/release_data/checkpoint/pxdesign_v0.1.0.pt; do
    [ -f "$c" ] && { DONOR="$c"; break; }
  done
fi
[ -n "$DONOR" ] || { echo "cannot find pxdesign_v0.1.0.pt; set PXDESIGN_DONOR"; exit 1; }

GOT=$(sha256sum "$DONOR" | cut -c1-16)
if [ "$GOT" != "b075867bae942dc0" ]; then
  echo "DONOR MISMATCH: $DONOR -> $GOT, expected b075867bae942dc0"
  echo "  These are the weights every arm is conditioned on. A different"
  echo "  donor makes the arms incomparable to the Marlowe results. Stop."
  exit 1
fi
echo "== donor verified: $DONOR =="

# ------------------------------------------- 3a. prepared target YAMLs
# Rendered into a sibling directory rather than in place, so a re-pull does
# not have to fight rsync over a file it already rewrote. --prepared-dir
# points HERE, not at configs/.
PREPARED="$BUNDLE/targets/binder_bench_targets/configs.resolved"
rm -rf "$PREPARED"; mkdir -p "$PREPARED"
for y in "$BUNDLE/targets/binder_bench_targets/configs"/*.yaml; do
  sed "s|@BUNDLE@|$BUNDLE|g" "$y" > "$PREPARED/$(basename "$y")"
done
python3 - "$PREPARED" <<'CHECK'
import re, sys
from pathlib import Path
bad = []
for y in sorted(Path(sys.argv[1]).glob("*.yaml")):
    for m in re.finditer(r"^\s*file:\s*(\S+)", y.read_text(), flags=re.M):
        path = m.group(1)
        if path.startswith("@") or not Path(path).is_file():
            bad.append(f"{y.name}: {path}")
if bad:
    raise SystemExit("unresolvable target structure(s):\n  " + "\n  ".join(bad))
print(f"  {len(list(Path(sys.argv[1]).glob('*.yaml')))} prepared YAML(s) resolved")
CHECK

# ----------------------------------------------------- 3b. FaMPNN 0.3
FAMPNN="${FAMPNN_03:-}"
if [ -z "$FAMPNN" ]; then
  for c in "$BUNDLE/checkpoints/donors/fampnn_0_3.pt" \
           "${PXF_REPO:-$DEST/Proteo-AA-pxdesign-fampnn-pack}/fampnn/weights/fampnn_0_3.pt"; do
    [ -f "$c" ] && { FAMPNN="$c"; break; }
  done
fi
[ -n "$FAMPNN" ] || { echo "cannot find fampnn_0_3.pt; set FAMPNN_03"; exit 1; }
GOT=$(sha256sum "$FAMPNN" | cut -c1-16)
[ "$GOT" = "8969b3f1f3c94117" ] \
  || { echo "FAMPNN MISMATCH: $FAMPNN -> $GOT, expected 8969b3f1f3c94117"; exit 1; }
echo "== FaMPNN 0.3 verified: $FAMPNN =="

sed -e "s|@BUNDLE@|$BUNDLE|g" -e "s|@PXDESIGN_DONOR@|$DONOR|g" \
  -e "s|@FAMPNN@|$FAMPNN|g" \
  "$BUNDLE/selection/selected_checkpoints.template.json" \
  > "$BUNDLE/selection/selected_checkpoints.json"
python3 -c "
import json,sys
from pathlib import Path
d=json.load(open('$BUNDLE/selection/selected_checkpoints.json'))
bad=[k for k,v in d.items() if not Path(v['checkpoint']).is_file()]
if bad: raise SystemExit(f'rehydrated paths missing for {bad}')
left=[k for k,v in d.items() for x in v.values() if isinstance(x,str) and x.startswith('@')]
if left: raise SystemExit(f'unsubstituted placeholder in {sorted(set(left))}')
print(f'  {len(d)} selection entries rehydrated and resolvable')
"

cat <<EOS

bundle ready: $BUNDLE
  selection : $BUNDLE/selection/selected_checkpoints.json
  A_BS s0   : $BUNDLE/checkpoints/bs_seq_sc/J03_seed0_step00000500.pt
  A_BS s1   : $BUNDLE/checkpoints/bs_seq_sc/J03_seed1_step00000500.pt
  prepared  : $PREPARED        <-- pass this to --prepared-dir
  config    : $BUNDLE/targets/configs_binder_benchmark/targets.yaml

next: read $BUNDLE/RUN_HERE.md
EOS
