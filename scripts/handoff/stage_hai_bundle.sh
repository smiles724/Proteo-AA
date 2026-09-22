#!/bin/bash
# Stage everything the pending integrated-feedback jobs need into one bundle,
# on Marlowe, for transfer to HAI.
#
#   scripts/handoff/stage_hai_bundle.sh                 # core, ~90M
#   scripts/handoff/stage_hai_bundle.sh --with-donors   # + pxdesign/FaMPNN donors
#   scripts/handoff/stage_hai_bundle.sh --with-release-data  # + PROTENIX_ROOT_DIR
#   scripts/handoff/stage_hai_bundle.sh --with-cache    # + conditioning cache (retrain only)
#   scripts/handoff/stage_hai_bundle.sh --all --tar
#
# What is deliberately NOT here:
#
#   runs/binder_bench/backbones (872M)  The matrix generates its own backbone
#     per cell via PXDesign; it never reads the cached collection. That
#     collection is also NOT reproducible (0.55 A between identical
#     invocations), so it must be backed up rather than regenerated -- but
#     that is a separate concern from these jobs.
#   the conditioning cache (305M)  Only an input to training. The four runs
#     are finished; HAI generates, it does not retrain. Available behind
#     --with-cache if that changes.
#   pxdesign_pristine/release_data (2.4G)  HAI already has an official
#     PXDesign install. The bundle pins the donor sha256 instead so the HAI
#     side can prove its copy is the same weights, which is what actually
#     matters.
#
# The selection artifact is rewritten: its checkpoint paths are Marlowe
# absolutes, which do not exist on HAI. They become @BUNDLE@-relative and are
# rehydrated by fetch_bundle_on_hai.sh. The sha256 fields are content digests
# and survive the move untouched -- that is what check_policy verifies.
set -euo pipefail

ROOT="${PXF_REPO:-/users/yfsun/Proteo-AA-pxdesign-fampnn-pack}"
DATA="${PXF_DATA_ROOT:-/scratch/m000137-pm06/Proteo-AA/pxf}"
IFB="$DATA/runs/integrated_feedback_v1"
OUT="${PXF_BUNDLE_OUT:-$DATA/handoff/pxf_hai_bundle}"

WITH_DONORS=0 WITH_RELEASE=0 WITH_CACHE=0 MAKE_TAR=0
for a in "$@"; do case "$a" in
  --with-donors) WITH_DONORS=1 ;;
  --with-release-data) WITH_RELEASE=1 ;;
  --with-cache) WITH_CACHE=1 ;;
  --all) WITH_DONORS=1; WITH_RELEASE=1; WITH_CACHE=1 ;;
  --tar) MAKE_TAR=1 ;;
  *) echo "unknown flag: $a"; exit 2 ;;
esac; done

rm -rf "$OUT"
mkdir -p "$OUT"/{checkpoints/bs_seq_sc,checkpoints/feedback,checkpoints/donors,selection,targets,reports,repo}

copy () {  # copy <src> <dst-relative>; fail loudly on a missing input
  local src="$1" dst="$OUT/$2"
  [ -e "$src" ] || { echo "MISSING INPUT: $src"; exit 1; }
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}

echo "== A_BS (J03) checkpoints =="
copy "$DATA/runs/bs_seq_sc/J03_seed0/checkpoints/step00000500.pt" checkpoints/bs_seq_sc/J03_seed0_step00000500.pt
copy "$DATA/runs/bs_seq_sc/J03_seed1/checkpoints/step00000500.pt" checkpoints/bs_seq_sc/J03_seed1_step00000500.pt

echo "== feedback checkpoints, at the steps the selection artifact names =="
# Read the steps out of the artifact rather than hardcoding 2000: if selection
# is ever rerun with a repaired guardrail, this picks up whatever it chose.
python3 - "$IFB/evaluation/selected_checkpoints.json" "$IFB" "$OUT" <<'PY'
import json, shutil, sys
from pathlib import Path
sel, ifb, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
data = json.loads(sel.read_text())
for key, entry in data.items():
    src = Path(entry["checkpoint"])
    if not src.is_file():
        raise SystemExit(f"MISSING INPUT: {src}")
    run = src.parent.parent.name          # e.g. E1_full_s0
    dst = out / "checkpoints" / "feedback" / run / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {key:22} {run}/{src.name}")
    # history.json is small and is the only record of the loss curve.
    hist = src.parent.parent / "history.json"
    if hist.is_file():
        shutil.copy2(hist, dst.parent.parent / f"{run}_history.json")
PY

echo "== selection artifact, paths made relocatable =="
python3 - "$IFB/evaluation/selected_checkpoints.json" "$DATA" "$OUT" <<'PY'
import json, sys
from pathlib import Path
sel, data_root, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
entries = json.loads(sel.read_text())
for key, entry in entries.items():
    src = Path(entry["checkpoint"])
    run = src.parent.parent.name
    entry["checkpoint"] = f"@BUNDLE@/checkpoints/feedback/{run}/{src.name}"
    bs = entry.get("bs_checkpoint")
    if bs:
        seed = int(entry["bs_seed"])
        entry["bs_checkpoint"] = (
            f"@BUNDLE@/checkpoints/bs_seq_sc/J03_seed{seed}_step00000500.pt")
    if entry.get("pxdesign_donor"):
        entry["pxdesign_donor"] = "@PXDESIGN_DONOR@"
    if entry.get("fampnn_checkpoint"):
        entry["fampnn_checkpoint"] = "@FAMPNN@"
    entry["_relocated_from"] = data_root
(out / "selection" / "selected_checkpoints.template.json").write_text(
    json.dumps(entries, indent=2) + "\n")
print(f"  {len(entries)} entries templated")
PY
copy "$IFB/evaluation/selected_checkpoints.json" selection/selected_checkpoints.marlowe.json

echo "== prepared targets =="
copy "$DATA/runs/binder_bench/targets" targets/binder_bench_targets
copy "$ROOT/configs/binder_benchmark" targets/configs_binder_benchmark

echo "== provenance =="
copy "$IFB/acceptance/acceptance.json" reports/acceptance.json
copy "$IFB/data/audit.json" reports/data_audit.json
[ -f "$IFB/evaluation/evaluation.json" ] && copy "$IFB/evaluation/evaluation.json" reports/evaluation.json
for m in "$IFB/cache"/*/cache.json; do
  [ -f "$m" ] && copy "$m" "reports/cache_$(basename "$(dirname "$m")")_manifest.json"
done

echo "== repo pin =="
git -C "$ROOT" rev-parse HEAD > "$OUT/repo/HEAD"
git -C "$ROOT" rev-parse --abbrev-ref HEAD > "$OUT/repo/BRANCH"
git -C "$ROOT" remote get-url origin > "$OUT/repo/ORIGIN"
[ -f "$ROOT/patches/pxdesign_vendored_embedders_d_lm.patch" ] \
  && copy "$ROOT/patches/pxdesign_vendored_embedders_d_lm.patch" repo/pxdesign_vendored_embedders_d_lm.patch

if [ "$WITH_DONORS" = 1 ]; then
  echo "== donors (531M + 42M) =="
  copy "$DATA/component_donors/pxdesign_v0.1.0.pt" checkpoints/donors/pxdesign_v0.1.0.pt
  copy "$ROOT/fampnn/weights/fampnn_0_3.pt" checkpoints/donors/fampnn_0_3.pt
fi
if [ "$WITH_RELEASE" = 1 ]; then
  echo "== official_release_data (549M) =="
  copy "$DATA/official_release_data" official_release_data
  # common/components.cif and its rdkit pickle are ABSOLUTE symlinks into the
  # Marlowe scratch path. Copied verbatim they dangle on HAI and Protenix
  # fails to find the CCD -- so rewrite any symlink pointing inside the tree
  # to a relative target. Relative rather than dereferenced: the targets are
  # ~500M and duplicating them doubles the transfer for nothing.
  python3 - "$OUT/official_release_data" "$DATA/official_release_data" <<'RELINK'
import os, sys
from pathlib import Path
out, src_root = Path(sys.argv[1]), Path(sys.argv[2])
fixed = dangling = 0
for link in out.rglob("*"):
    if not link.is_symlink():
        continue
    target = Path(os.readlink(link))
    if target.is_absolute():
        try:
            inside = target.relative_to(src_root)
        except ValueError:
            print(f"  DANGLING (points outside the tree): {link} -> {target}")
            dangling += 1
            continue
        link.unlink()
        link.symlink_to(os.path.relpath(out / inside, link.parent))
        fixed += 1
    if not link.resolve().exists():
        print(f"  DANGLING after rewrite: {link}")
        dangling += 1
print(f"  {fixed} symlink(s) relativised")
if dangling:
    raise SystemExit(f"{dangling} dangling symlink(s); the bundle would be broken on HAI")
RELINK
fi
if [ "$WITH_CACHE" = 1 ]; then
  echo "== conditioning cache (305M) =="
  copy "$IFB/cache" cache
fi

cp "$ROOT/docs/handoff/hai_run_prompt.md" "$OUT/RUN_HERE.md" 2>/dev/null || true
cp "$ROOT/scripts/handoff/fetch_bundle_on_hai.sh" "$OUT/" 2>/dev/null || true

echo "== digests =="
( cd "$OUT" && find . -type f ! -type l ! -name SHA256SUMS ! -name MANIFEST.json -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS )
( cd "$OUT" && find . -type l -printf "%p -> %l\n" | sort > SYMLINKS )
echo "  $(wc -l < "$OUT/SHA256SUMS") file(s)"

echo "== manifest =="
python3 - "$OUT" "$DATA" "$WITH_DONORS$WITH_RELEASE$WITH_CACHE" <<'PY'
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
out, data_root, flags = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# The expected donor digests. These are what pxf.bench.integrated_checkpoints
# .check_policy compares against, so if the HAI copy of PXDesign or FaMPNN
# differs, the run fails closed rather than producing a quietly wrong arm.
expected = {
    "pxdesign_v0.1.0.pt": "b075867bae942dc0",
    "fampnn_0_3.pt": "8969b3f1f3c94117",
}
files = []
total = 0
symlinks = []
for path in sorted(out.rglob("*")):
    if path.is_symlink():
        symlinks.append({"path": str(path.relative_to(out)),
                         "target": os.readlink(path),
                         "resolves": path.resolve().exists()})
        continue
    if not path.is_file() or path.name in {"SHA256SUMS", "MANIFEST.json", "SYMLINKS"}:
        continue
    size = path.stat().st_size
    total += size
    rel = str(path.relative_to(out))
    rec = {"path": rel, "bytes": size}
    # Digest everything small, plus every checkpoint regardless of size.
    if size < 64 << 20 or rel.startswith("checkpoints/"):
        rec["sha256"] = digest(path)
        want = expected.get(path.name)
        if want:
            rec["expected_prefix"] = want
            rec["donor_ok"] = rec["sha256"].startswith(want)
    files.append(rec)

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_host": os.uname().nodename,
    "source_data_root": data_root,
    "options": {"with_donors": flags[0] == "1",
                "with_release_data": flags[1] == "1",
                "with_cache": flags[2] == "1"},
    "repo": {
        "origin": (out / "repo/ORIGIN").read_text().strip(),
        "branch": (out / "repo/BRANCH").read_text().strip(),
        "head": (out / "repo/HEAD").read_text().strip(),
    },
    "pending_jobs": [
        {"what": "generation cell", "cell": "PDL1 L80 gen-seed 101",
         "outputs": 7, "arms": ["U03",
                                "J03_s0", "E1_bb_only_s0", "E1_full_s0",
                                "J03_s1", "E1_bb_only_s1", "E1_full_s1"],
         "marlowe_job": 498855, "marlowe_state": "PENDING (fairshare)"},
        {"what": "28-design smoke", "cells": "2 targets x 2 gen-seeds x 7 arms"},
        {"what": "AF2-IG scoring", "note": "needs af2 params on the runner"},
    ],
    "total_bytes": total,
    "symlinks": symlinks,
    "files": files,
}
(out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
bad = [f for f in files if f.get("donor_ok") is False]
print(f"  {len(files)} file(s), {len(symlinks)} symlink(s), {total/1e6:.1f} MB")
for f in bad:
    print(f"  DONOR MISMATCH: {f['path']} -> {f['sha256'][:16]} != {f['expected_prefix']}")
if bad:
    raise SystemExit("donor digest mismatch; refusing to declare the bundle sound")
PY

echo
echo "bundle: $OUT  ($(du -sh "$OUT" | cut -f1))"
if [ "$MAKE_TAR" = 1 ]; then
  TAR="$OUT.tar.zst"
  command -v zstd >/dev/null && tar -C "$(dirname "$OUT")" -cf - "$(basename "$OUT")" | zstd -T0 -3 -o "$TAR" -f \
    || { TAR="$OUT.tar.gz"; tar -C "$(dirname "$OUT")" -czf "$TAR" "$(basename "$OUT")"; }
  sha256sum "$TAR" | tee "$TAR.sha256"
  echo "tarball: $TAR ($(du -sh "$TAR" | cut -f1))"
fi
