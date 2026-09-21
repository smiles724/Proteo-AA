#!/bin/bash
# PXDesign's OFFICIAL runtime (Protenix v0.5.0+pxd, d18aa1da) on Marlowe.
#
# This exists because the repo spent a long time believing the official
# runtime was HAI-only and that de novo generation therefore could not run
# here. That was wrong on both counts:
#
#   * v0.5.0+pxd is a public git tag. Installing it is using the CORRECT
#     version -- it is not the `protenix.data.parser` alias that
#     pxf/official/require.py refuses, and it does not touch the vendored
#     v2.0.0 tree, which stays exactly where it is for the pxf training path.
#   * Marlowe reaches PyPI and GitHub fine.
#
# TWO THINGS THE PUBLISHED INSTALLER GETS WRONG, both found by running it:
#
#   1. deepspeed is unpinned (`>=0.15.1`) and today resolves to 0.19.x, which
#      needs torch>=2.4 while the installer pins torch 2.3.1. Protenix probes
#      deepspeed with importlib.util.find_spec, which executes the package
#      __init__, so ANY protenix import dies. Pinned 0.15.4 and installed
#      LAST so nothing re-resolves it. (audit section 6)
#   2. PXDesign must be PRISTINE here. This repo's working copy of
#      pxdesign/model/embedders.py carries a patch bridging to Protenix
#      v2.0.0's newer AtomAttentionEncoder signature (explicit d_lm/v_lm/
#      pad_info). The official encoder takes the whole feature dict and
#      derives those internally, so the patched copy fails with
#      `KeyError: 'd_lm'`. A separate pristine worktree at f788441 is used:
#          git -C PXDesign worktree add /users/yfsun/pxdesign_pristine f788441
#      Keeping the two runtimes in separate trees is the point; sharing one
#      patched tree is how they would silently contaminate each other.
#
# VALIDATED, not assumed: audit section 7's positive control re-run here
# (PDL1 quick start, released checkpoint, 3 seeds x 4 samples) gave 12/12
# clash-free interfaces, min BB-BB 2.81-5.95 A, chain A0 464 atoms / B0 320 --
# matching the HAI numbers. A clean import proves nothing; that run does.
# See scripts/slurm/marlowe/pxdesign_positive_control.sh.
#
# Cost note: the one-time downloads (394 MB CCD cache + ~2.2 GB auxiliary
# Protenix checkpoints, from a Beijing endpoint) dominate. Actual inference
# was ~16 s per seed for 4 samples on an H100.

set -uo pipefail
BASE=/users/yfsun/.venvs/proteoaa-stage4/bin/python3.11
ENV=/users/yfsun/.venvs/pxdesign_official
PX=/users/yfsun/Proteo-AA-pxdesign-fampnn-pack/PXDesign
step(){ echo; echo "### $* ###"; }

step "create venv"
rm -rf "$ENV"; "$BASE" -m venv "$ENV" || exit 1
P="$ENV/bin/python"; PIP="$ENV/bin/pip"
$PIP install -q -U pip setuptools wheel || exit 1
$P -V

step "torch 2.3.1 cu121"
$PIP install -q --no-cache-dir torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121 || exit 1

step "protenix v0.5.0+pxd"
$PIP install --no-cache-dir "git+https://github.com/bytedance/Protenix.git@v0.5.0+pxd" 2>&1 | tail -5

step "pxdbench 0.1.2 (no deps)"
$PIP install -q --no-cache-dir --no-deps \
  "git+https://github.com/bytedance/PXDesignBench.git@v0.1.2" 2>&1 | tail -3

step "PXDesign (no deps, editable)"
$PIP install -q --no-cache-dir --no-deps -e "$PX" 2>&1 | tail -3

step "remaining runtime deps"
$PIP install -q --no-cache-dir \
  einops natsort dm-tree posix_ipc ml_collections optree rdkit icecream \
  "biopython==1.83" "modelcif==0.7" "biotite==1.0.1" scikit-learn scipy tqdm \
  pandas PyYaml "protobuf==3.20.2" "transformers==4.51.3" 2>&1 | tail -3

step "deepspeed 0.15.4 LAST (audit section 6: resolver otherwise picks 0.19.x,"
echo "  which needs torch>=2.4 and kills ANY protenix import via find_spec)"
$PIP install -q --no-cache-dir "deepspeed==0.15.4" 2>&1 | tail -5

step "numpy 1.26.3 LAST (torch/protenix both tolerate it; 2.x breaks biotite)"
$PIP install -q --no-cache-dir "numpy==1.26.3" 2>&1 | tail -3

step "VERSIONS"
$P - <<'PY'
import importlib.metadata as md
for p in ("torch","protenix","deepspeed","numpy","pxdbench","pxdesign","biotite"):
    try: print(f"  {p:12s} {md.version(p)}")
    except Exception as e: print(f"  {p:12s} MISSING ({type(e).__name__})")
PY

step "THE IMPORT THAT DEFINES THE BLOCKER"
$P - <<'PY'
ok=True
try:
    from protenix.data.parser import MMCIFParser, DistillationMMCIFParser
    print("  [OK] from protenix.data.parser import MMCIFParser, DistillationMMCIFParser")
except Exception as e:
    ok=False; print(f"  [FAIL] {type(e).__name__}: {e}")
for m in ("protenix","pxdbench","pxdesign"):
    try:
        __import__(m); print(f"  [OK] import {m}")
    except Exception as e:
        ok=False; print(f"  [FAIL] import {m}: {type(e).__name__}: {e}")
print("VERDICT:", "IMPORTS_CLEAN" if ok else "IMPORTS_BROKEN")
PY
echo "EXIT=$?"
