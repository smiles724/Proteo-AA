#!/usr/bin/env python3
"""APM's atom37 pickles -> mmCIFs that Protenix's parser accepts.

The a_token arm needs the frozen PXDesign trunk, and the trunk consumes a
Protenix feature dict built by `CifFileProvider` -> `DesignSourceDataset` from
an mmCIF. APM's pickles carry only raw arrays, so this is the step that gets
APM's structures into that pipeline.

Round-tripping through a file rather than constructing features directly is
deliberate. The CIF path is the one the CASP benchmark and the design runs
already use, so every naming, ordering and entity convention it relies on is
already exercised; a bespoke in-memory featuriser would be a second
implementation of the same contract with no test behind it.

The residue set written is APM's `modeled_idx` in APM's order, which is what
`sidechain/apm_dataset.py` also keeps, so `a_token[i]` and the packer's residue
`i` refer to the same residue. `check_a_token_bridge.py` asserts that rather
than trusting it.

    python scripts/data/apm_pkl_to_cif.py --ids-from-test-set --out DIR
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _sibling(rel, name):
    """Import a module from this repo by PATH, not through a `scripts` package.

    `Protenix/scripts/__init__.py` makes `scripts` a regular package, and a
    regular package beats this repo's namespace one -- so `from scripts.x
    import y` resolves against Protenix and raises ModuleNotFoundError as soon
    as Protenix is on PYTHONPATH, which it is in every GPU job.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(REPO / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


structure_to_cif = _sibling(
    "scripts/evaluation/casp_natives_to_cif.py", "_casp_cif").structure_to_cif

# openfold atom37 ordering and restype table, imported rather than retyped.
from openfold.np.residue_constants import (atom_types, restype_1to3,  # noqa: E402
                                           restypes)

ELEMENT = {"C": "C", "N": "N", "O": "O", "S": "S"}


def build_structure(d, name):
    """atom37 arrays -> a gemmi Structure over APM's residue span.

    Two details decide whether `a_token[i]` lines up with the packer's residue
    `i`, and both follow APM rather than convenience:

    * The residue set is the CONTIGUOUS SPAN `modeled_idx.min() .. .max()`, not
      `modeled_idx` itself. `_process_csv_row_FAESM` slices the span, so ~11% of
      chains carry interior residues with no coordinates; dropping them here
      would shorten a_token relative to every other tensor by a few positions
      and shift the tail of the chain by one residue per gap.
    * Those coordinate-less residues still have to become tokens. They are put
      in the entity's `full_sequence`, so gemmi writes them into
      `_entity_poly_seq` and Protenix tokenises them as unresolved -- which is
      what they are, and what `res_mask` already says about them.

    Coordinates are passed through APM's `parse_chain_feats` first (CA-centroid
    centering, unobserved atoms zeroed), so the trunk reads the same numbers
    the packer's frames were built from rather than a translate of them.
    """
    import gemmi
    from apm.data.utils import parse_chain_feats

    d = parse_chain_feats({k: np.asarray(v) for k, v in d.items()})
    mi = np.asarray(d["modeled_idx"], dtype=int)
    span = slice(int(mi.min()), int(mi.max()) + 1)
    aatype = np.asarray(d["aatype"])[span]
    pos = np.asarray(d["atom_positions"], dtype=float)[span]
    mask = np.asarray(d["atom_mask"])[span] > 0.5
    res_idx = np.asarray(d["residue_index"], dtype=int)[span]
    chain_idx = np.asarray(d["chain_index"], dtype=int)[span]

    st = gemmi.Structure()
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    # gemmi keys entities off the chain name, so name chains A, B, ... by the
    # order APM's chain_index first appears. APM's own values are absolute
    # alphabet codes (a monomer can be chain 26), which would make ugly and
    # occasionally invalid names.
    order = list(dict.fromkeys(int(c) for c in chain_idx))
    names = {c: (chr(ord("A") + i) if i < 26 else f"A{i}") for i, c in enumerate(order)}
    chains = {c: gemmi.Chain(names[c]) for c in order}
    full_seq = {c: [] for c in order}
    serial = 0
    for i in range(len(aatype)):
        cid = int(chain_idx[i])
        t = int(aatype[i])
        # aatype 20 is UNK; openfold has no atom set for it, so write it as the
        # unknown residue and let it be a token with no coordinates.
        rname = restype_1to3[restypes[t]] if 0 <= t < len(restypes) else "UNK"
        full_seq[cid].append(rname)
        if not mask[i].any() or rname == "UNK":
            continue           # unresolved: a sequence position, not an atom row
        res = gemmi.Residue()
        res.name = rname
        res.seqid = gemmi.SeqId(int(res_idx[i]) + 1, " ")
        res.het_flag = "A"
        for j, aname in enumerate(atom_types):
            if not mask[i, j]:
                continue
            at = gemmi.Atom()
            at.name = aname
            at.element = gemmi.Element(ELEMENT.get(aname[0], "C"))
            at.pos = gemmi.Position(*pos[i, j])
            at.occ = 1.0
            at.b_iso = 20.0
            serial += 1
            res.add_atom(at)
        chains[cid].add_residue(res)
    for c in order:
        model.add_chain(chains[c])
    st.add_model(model)
    st.setup_entities()
    # One entity per chain here (distinct sequences are not deduplicated by
    # setup_entities when the chains differ), so match by subchain membership.
    for ent in st.entities:
        for c in order:
            if names[c] in ent.subchains or names[c] == ent.name:
                ent.full_sequence = full_seq[c]
                break
    return st, int(len(aatype)), serial, {names[c]: len(full_seq[c]) for c in order}


def convert(pkl_path, out_dir, revision_date="2021-01-01"):
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    name = Path(pkl_path).stem
    st, n_span, n_atom, chain_lens = build_structure(d, name)
    rec = structure_to_cif(st, Path(out_dir) / f"{name}.cif", name, revision_date,
                           keep_entity_poly_seq=True)
    rec["apm_span_res"] = n_span
    rec["written_atoms"] = int(n_atom)
    rec["chain_lengths"] = chain_lens
    # `native_len` counts residues with coordinates, which is legitimately
    # smaller than the span whenever the chain has an interior gap. What must
    # never differ is the number of SEQUENCE positions, because that is what
    # becomes a token and therefore a row of a_token.
    if sum(chain_lens.values()) != n_span:
        rec["span_mismatch"] = True
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", nargs="*", default=[])
    ap.add_argument("--pkl-dir", default="")
    ap.add_argument("--ids-csv", default="", help="csv with a pdb_name column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default="")
    args = ap.parse_args()

    files = [Path(p) for p in args.pkl]
    if args.ids_csv:
        import pandas as pd
        ids = pd.read_csv(args.ids_csv)["pdb_name"].astype(str)
        files += [Path(args.pkl_dir) / f"{i}.pkl" for i in ids]
    elif args.pkl_dir:
        files += sorted(Path(args.pkl_dir).glob("*.pkl"))
    files = [f for f in files if f.is_file()]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("no input pickles")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs, failed = [], []
    for f in files:
        try:
            recs.append(convert(f, out))
        except Exception as e:                      # noqa: BLE001
            failed.append({"pkl": str(f), "error": f"{type(e).__name__}: {e}"})
    print(f"converted {len(recs)}/{len(files)}  failed {len(failed)}")
    for bad in failed[:10]:
        print("  FAIL", bad["pkl"], bad["error"])
    man = Path(args.manifest) if args.manifest else out / "manifest.json"
    man.write_text(json.dumps({"targets": recs, "failed": failed}, indent=1))
    print("manifest", man)


if __name__ == "__main__":
    main()
