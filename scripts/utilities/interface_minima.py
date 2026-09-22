#!/usr/bin/env python3
"""Interface minima, ALL-ATOM vs BACKBONE-ONLY, over a design collection.

_geometry() in run_integrated_binder_matrix.py measures ALL-ATOM binder x
target and flags pairs < 2.6 A. The section-10 positive control measured
BACKBONE-ONLY. This prints both on the same structures so the two numbers
can be compared to the right baseline.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from biotite.structure.io.pdb import PDBFile

BB = ("N", "CA", "C", "O")
rows = []
for path in sys.argv[1:]:
    try:
        a = PDBFile.read(path).get_structure(model=1)
    except Exception:
        continue
    a = a[a.element != "H"]
    chains = sorted(set(a.chain_id))
    if len(chains) < 2: continue
    lens = {c: len(set(a[a.chain_id == c].res_id)) for c in chains}
    binder = min(lens, key=lens.get)
    bm, tm = a.chain_id == binder, a.chain_id != binder
    out = {}
    for tag, extra in (("all", None), ("bb", np.isin(a.atom_name, BB))):
        bsel, tsel = (bm, tm) if extra is None else (bm & extra, tm & extra)
        b, t = a[bsel].coord.astype(np.float64), a[tsel].coord.astype(np.float64)
        if not len(b) or not len(t): continue
        d = np.linalg.norm(b[:, None] - t[None, :], axis=-1)
        out[tag] = (float(d.min()), int((d < 2.6).sum()))
    if "all" in out and "bb" in out:
        rows.append((path.split("/")[-1], out["all"], out["bb"]))

def q(v, p): return float(np.percentile(v, p))
am = [r[1][0] for r in rows]; ac = [r[2][1] for r in rows]
bm_ = [r[2][0] for r in rows]; acn = [r[1][1] for r in rows]
print(f"n = {len(rows)} designs\n")
print(f"{'metric':<28} {'min':>7} {'p05':>7} {'median':>7} {'max':>7}")
print(f"{'ALL-ATOM iface min (A)':<28} {min(am):7.3f} {q(am,5):7.3f} {q(am,50):7.3f} {max(am):7.3f}")
print(f"{'BACKBONE-ONLY iface min (A)':<28} {min(bm_):7.3f} {q(bm_,5):7.3f} {q(bm_,50):7.3f} {max(bm_):7.3f}")
print(f"{'ALL-ATOM pairs < 2.6 A':<28} {min(acn):7d} {q(acn,5):7.1f} {q(acn,50):7.1f} {max(acn):7d}")
print(f"{'BACKBONE pairs < 2.6 A':<28} {min(ac):7d} {q(ac,5):7.1f} {q(ac,50):7.1f} {max(ac):7d}")
print(f"\nfraction with ZERO all-atom pairs < 2.6 A : {sum(1 for c in acn if c==0)}/{len(rows)}")
print(f"fraction with ZERO backbone pairs < 2.6 A: {sum(1 for c in ac if c==0)}/{len(rows)}")
print(f"designs with all-atom min below 0.90 A   : {sum(1 for m in am if m < 0.90)}/{len(rows)}")
print(f"designs with backbone min below 0.90 A   : {sum(1 for m in bm_ if m < 0.90)}/{len(rows)}")
