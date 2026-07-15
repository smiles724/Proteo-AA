#!/usr/bin/env python
"""Quantify the side-chain template against REAL structures.

What is being measured
----------------------
The template is the initialization mu_ideal of

    y_{T,ij} = mu_ideal[a_i, chi_i, j] + sigma_T * eps ,   x_T = F_hat_i y_T

so the thing that matters is how far mu_ideal sits from the residue's TRUE side chain
*in the residue-local frame*. Working in the local frame is the point: it holds the
backbone frame fixed and therefore isolates the template's own contribution, with no
global superposition or Kabsch alignment to launder the error through.

For every residue of every chain we take the real N/CA/C, build the same local frame
the model uses (``frames.build_frame``), express the real side-chain heavy atoms in it,
and compare against each construction:

  gaussian      mu = 0                     isotropic baseline.
  ccd           static CCD conformer       one arbitrary chi per residue type (pre-0714)
  dunbrack_mode BuildSC, chi = argmax p    0714 appendix, deterministic selection
  dunbrack      BuildSC, chi ~ Cat(p)      0714 appendix, sampled selection (the default);
                                           averaged over --draws draws, so this is the
                                           EXPECTED init error, not a lucky one
  oracle        BuildSC, chi = TRUE chi    lower bound: the best any rotamer library could
                                           do with this ideal covalent geometry. The gap
                                           from oracle to truth is pure bond-length/angle
                                           idealization; the gap from dunbrack to oracle is
                                           rotamer-selection error.

Reported per method: RMSD (A) over side-chain heavy atoms, and chi1 recovery
(|dchi1| < 40 deg, the standard rotamer-recovery criterion).

Usage
-----
    python scripts/eval_template_quality.py                 # default 30-chain set
    python scripts/eval_template_quality.py --pdb 1ubq 3nir --draws 8
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "Protenix", _REPO / "PXDesign"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pxdesign_train.sidechain import rotamers  # noqa: E402
from pxdesign_train.sidechain.buildsc import build_sidechain_local, chi_from_local  # noqa: E402
from pxdesign_train.sidechain.frames import build_frame, phi_psi_from_ncac, to_local  # noqa: E402
from pxdesign_train.sidechain.instantiate import STD_AA_3, sidechain_atoms  # noqa: E402
from pxdesign_train.sidechain.templates import ideal_template  # noqa: E402

# High-resolution, structurally diverse single-domain proteins.
DEFAULT_PDBS = [
    "1ubq", "3nir", "1crn", "1cse", "5pti", "1a2p", "1igd", "1shg", "2ci2", "1pga",
    "1csp", "1fkb", "3chy", "2ptl", "1aps", "1bdd", "1enh", "1poh", "1ycc", "1lz1",
    "2lzm", "1rop", "1ten", "1vqb", "1beo", "1opd", "1whi", "2acy", "1msi", "1cbs",
]
CACHE = Path(".pdb_cache")


def fetch(pdb: str) -> Path:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{pdb}.pdb"
    if not p.exists():
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb.upper()}.pdb", p)
    return p


def parse_chains(path: Path):
    """{chain: [(resseq, resname, {atom: (x,y,z)})]} — model 1, altloc blank/A, heavy atoms."""
    chains = defaultdict(dict)
    order = defaultdict(list)
    with path.open(errors="replace") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            if line[76:78].strip() == "H":
                continue
            name = line[12:16].strip()
            resname = line[17:20].strip()
            ch = line[21]
            resseq = line[22:27].strip()          # includes insertion code
            key = (resseq, resname)
            if key not in chains[ch]:
                chains[ch][key] = {}
                order[ch].append(key)
            chains[ch][key].setdefault(name, (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return {ch: [(k[0], k[1], chains[ch][k]) for k in order[ch]] for ch in chains}


def residues_of(chain):
    """Keep residues that are standard, have N/CA/C, and have a COMPLETE side chain."""
    keep = []
    for resseq, resname, atoms in chain:
        if resname not in STD_AA_3:
            continue
        if not all(a in atoms for a in ("N", "CA", "C")):
            continue
        sc = sidechain_atoms(resname)
        if not all(a in atoms for a in sc):
            continue                              # partially-resolved side chain: skip
        try:
            num = int("".join(c for c in resseq if c.isdigit() or c == "-"))
        except ValueError:
            continue
        keep.append((num, resname, atoms))
    return keep


def rmsd(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-residue RMSD over valid side-chain atoms. a, b: [L, MAX_SC, 3]."""
    d2 = ((a - b) ** 2).sum(-1) * mask
    n = mask.sum(-1).clamp_min(1)
    return torch.sqrt(d2.sum(-1) / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", nargs="*", default=DEFAULT_PDBS)
    ap.add_argument("--draws", type=int, default=8, help="draws for the sampled provider")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not rotamers.available():
        print("rotamer library missing: run scripts/build_rotamer_library.py --download")
        return 1

    T, MU, PHI, PSI = [], [], [], []
    nres = nchain = 0
    for pdb in args.pdb:
        try:
            chains = parse_chains(fetch(pdb))
        except Exception as e:                     # noqa: BLE001
            print(f"  skip {pdb}: {e}")
            continue
        for ch, chain in chains.items():
            res = residues_of(chain)
            if len(res) < 20:
                continue
            nchain += 1
            nres += len(res)

            n = torch.tensor([r[2]["N"] for r in res])
            ca = torch.tensor([r[2]["CA"] for r in res])
            c = torch.tensor([r[2]["C"] for r in res])
            ri = torch.tensor([r[0] for r in res])
            ai = torch.zeros(len(res), dtype=torch.long)

            phi, psi = phi_psi_from_ncac(n, ca, c, ri, ai)
            R, t = build_frame(n, ca, c)

            tix = torch.tensor([STD_AA_3.index(r[1]) for r in res])
            mu_true = torch.zeros(len(res), 10, 3)
            for i, (_, rn, atoms) in enumerate(res):
                names = sidechain_atoms(rn)
                if not names:
                    continue
                g = torch.tensor([atoms[a] for a in names])
                mu_true[i, : len(names)] = to_local(g, R[i], t[i])

            T.append(tix)
            MU.append(mu_true)
            PHI.append(phi)
            PSI.append(psi)

    tix = torch.cat(T)
    mu_true = torch.cat(MU)
    phi = torch.cat(PHI)
    psi = torch.cat(PSI)

    _, mask = ideal_template(tix)
    has_sc = mask.any(-1)

    # --- the five constructions ------------------------------------------------
    mu_ccd, _ = ideal_template(tix)
    chi_true = chi_from_local(tix, mu_true)
    mu_oracle, _ = build_sidechain_local(tix, chi_true)

    chi_mode = rotamers.select_chi(tix, phi, psi, mode="mode")
    mu_mode, _ = build_sidechain_local(tix, chi_mode)

    g = torch.Generator().manual_seed(args.seed)
    samp_rmsd, samp_chi1 = [], []
    for _ in range(args.draws):
        chi_s = rotamers.select_chi(tix, phi, psi, mode="sample", generator=g)
        mu_s, _ = build_sidechain_local(tix, chi_s)
        samp_rmsd.append(rmsd(mu_s, mu_true, mask))
        samp_chi1.append(chi_s[:, 0])

    methods = {
        "gaussian  (mu=0, isotropic)": (rmsd(torch.zeros_like(mu_true), mu_true, mask), None),
        "ccd       (static, pre-0714)": (rmsd(mu_ccd, mu_true, mask), chi_from_local(tix, mu_ccd)[:, 0]),
        "dunbrack_mode (0714, argmax p)": (rmsd(mu_mode, mu_true, mask), chi_mode[:, 0]),
        "dunbrack  (0714, sampled)": (torch.stack(samp_rmsd).mean(0), None),
        "oracle    (true chi; lower bound)": (rmsd(mu_oracle, mu_true, mask), chi_true[:, 0]),
    }

    def chi1_recovery(pred):
        ok = torch.isfinite(chi_true[:, 0]) & torch.isfinite(pred) & has_sc
        if ok.sum() == 0:
            return float("nan")
        d = torch.rad2deg(torch.atan2(torch.sin(pred[ok] - chi_true[ok, 0]),
                                      torch.cos(pred[ok] - chi_true[ok, 0]))).abs()
        return float((d < 40).float().mean() * 100)

    samp_rec = sum(chi1_recovery(c) for c in samp_chi1) / len(samp_chi1)

    print(f"\nchains={nchain}  residues with a complete side chain={int(has_sc.sum())}  "
          f"(of {nres} parsed)   draws={args.draws}\n")
    print(f"{'construction':34s} {'local-frame RMSD (A)':>21s}   {'chi1 recovery <40deg':>20s}")
    print("-" * 80)
    for name, (r, c1) in methods.items():
        v = r[has_sc]
        rec = samp_rec if name.startswith("dunbrack  ") else (chi1_recovery(c1) if c1 is not None else float("nan"))
        rec_s = "        n/a" if rec != rec else f"{rec:9.1f} %"
        print(f"{name:34s} {v.mean():10.3f} +/- {v.std():<8.3f}  {rec_s:>20s}")

    # --- per residue type ------------------------------------------------------
    print(f"\n{'res':4s} {'n':>5s}  {'gaussian':>9s} {'ccd':>9s} {'db_mode':>9s} {'db_samp':>9s} "
          f"{'oracle':>9s}   {'mode vs ccd':>12s} {'samp vs ccd':>12s}")
    print("-" * 96)
    r_g, r_c = methods["gaussian  (mu=0, isotropic)"][0], methods["ccd       (static, pre-0714)"][0]
    r_m = methods["dunbrack_mode (0714, argmax p)"][0]
    r_d = methods["dunbrack  (0714, sampled)"][0]
    r_o = methods["oracle    (true chi; lower bound)"][0]
    for a in STD_AA_3:
        m = (tix == STD_AA_3.index(a)) & has_sc
        if m.sum() == 0:
            continue
        gc, cc, mc, dc, oc = (x[m].mean() for x in (r_g, r_c, r_m, r_d, r_o))
        dm = (cc - mc) / cc * 100 if cc > 0 else 0.0
        dd = (cc - dc) / cc * 100 if cc > 0 else 0.0
        tag = ""
        if a == "PRO":
            tag = "  ring-closed"
        elif a == "ALA":
            tag = "  no chi"
        print(f"{a:4s} {int(m.sum()):5d}  {gc:9.3f} {cc:9.3f} {mc:9.3f} {dc:9.3f} {oc:9.3f}   "
              f"{dm:+11.1f}% {dd:+11.1f}%{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
