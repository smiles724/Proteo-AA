#!/usr/bin/env python3
"""Convert CASP native target structures (.pdb) into mmCIFs Protenix can parse.

WHY THIS EXISTS. CASP publishes its natives as coordinate-only PDB files that
carry no PDB identifier and leave the chain-ID column blank. Two consequences:

  * You cannot look the target up in the PDB without sequence matching, and that
    heuristic is unreliable — on CASP14 a 25-mer probe matched ten unrelated
    targets to one 2194-residue polymerase, and the length filter that fixed the
    false positives then discarded ~9 legitimate domain targets.
  * A naive coordinate-only CIF will not parse: Protenix needs `entry`, `exptl`,
    `pdbx_audit_revision_history`, `pdbx_struct_assembly{,_gen}`,
    `pdbx_struct_oper_list`, a populated `label_seq_id`, AND `entity_poly_seq`
    (it rebuilds a reference chain from `poly_res_names[entity_id]`).

Converting the natives directly sidesteps the mapping problem altogether and
scores the exact coordinates CASP assessed — which is what "the experimentally
determined backbone" means in the packing protocol. It also works unchanged for
CASP14/15/16, whereas PDB lookup fails for recent editions (our local seqres and
mmCIF snapshots are 2022-vintage; CASP16 is 2024).

GOTCHA WORTH KNOWING. gemmi names the polymer subchain itself (e.g. `xp`), and
that string — not the chain name — is what lands in `label_asym_id` and therefore
what the parsed AtomArray uses as `chain_id`. The manifest records that asym id as
`chain`, so the caller selects the binder correctly instead of silently selecting
nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path

# Categories Protenix's parser requires but a coordinate-only CIF lacks.
IDENTITY_OPER = {
    "id": ["1"], "type": ["identity operation"], "name": ["1_555"],
    "matrix[1][1]": ["1.0"], "matrix[1][2]": ["0.0"], "matrix[1][3]": ["0.0"], "vector[1]": ["0.0"],
    "matrix[2][1]": ["0.0"], "matrix[2][2]": ["1.0"], "matrix[2][3]": ["0.0"], "vector[2]": ["0.0"],
    "matrix[3][1]": ["0.0"], "matrix[3][2]": ["0.0"], "matrix[3][3]": ["1.0"], "vector[3]": ["0.0"],
}


def convert_one(pdb_path: Path, out_path: Path, name: str, revision_date: str) -> dict:
    import gemmi

    st = gemmi.read_structure(str(pdb_path))
    st.remove_ligands_and_waters()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    # CASP natives leave the chain-ID column blank; setup_entities() keys off it,
    # so name the chains BEFORE calling it or the entity comes out malformed.
    for model in st:
        for chain in model:
            if not chain.name.strip():
                chain.name = "A"
    st.name = name
    st.setup_entities()
    # force=True — without it label_seq_id stays '.', and Protenix does int('.').
    st.assign_label_seq_id(True)

    doc = st.make_mmcif_document()
    blk = doc.sole_block()
    site = blk.get_mmcif_category("_atom_site.")
    if not site or not site.get("label_asym_id"):
        raise ValueError("no atom_site rows after filtering")
    asyms = sorted(set(site["label_asym_id"]))

    # entity_poly_seq: one row per (entity, seq position). Protenix rebuilds a
    # reference chain from this; without it you get KeyError on the entity id.
    ent, num, mon, seen = [], [], [], set()
    for eid, n, comp in zip(site["label_entity_id"], site["label_seq_id"], site["label_comp_id"]):
        key = (eid, n)
        if key in seen:
            continue
        seen.add(key)
        ent.append(eid)
        num.append(n)
        mon.append(comp)
    blk.set_mmcif_category("_entity_poly_seq", {"entity_id": ent, "num": num, "mon_id": mon})

    blk.set_mmcif_category("_entry", {"id": [name]})
    blk.set_mmcif_category("_exptl", {"entry_id": [name], "method": ["X-RAY DIFFRACTION"]})
    blk.set_mmcif_category("_pdbx_audit_revision_history", {
        "ordinal": ["1"], "data_content_type": ["Structure model"],
        "revision_date": [revision_date]})
    blk.set_mmcif_category("_pdbx_struct_assembly", {
        "id": ["1"], "details": ["author_defined_assembly"], "method_details": ["?"],
        "oligomeric_details": ["monomeric"], "oligomeric_count": [str(len(asyms))]})
    # asym_id_list must reference the REAL label_asym_id values, not chain names —
    # otherwise assembly expansion selects no atoms and the parse returns None.
    blk.set_mmcif_category("_pdbx_struct_assembly_gen", {
        "assembly_id": ["1"], "oper_expression": ["1"], "asym_id_list": [",".join(asyms)]})
    blk.set_mmcif_category("_pdbx_struct_oper_list", dict(IDENTITY_OPER))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.write_file(str(out_path))
    n_res = len({(a, b) for a, b in zip(site["label_asym_id"], site["label_seq_id"])})
    return {"target": name, "cif": str(out_path), "chain": asyms[0],
            "asym_ids": asyms, "native_len": n_res, "n_atoms": len(site["label_asym_id"])}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb-glob", required=True,
                   help="e.g. '/hai/scratch/yfsun/casp15/T/*.pdb'")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--revision-date", default="2022-01-01",
                   help="placeholder date for pdbx_audit_revision_history")
    p.add_argument("--max-residues", type=int, default=0,
                   help="skip targets longer than this (0 = keep all)")
    args = p.parse_args()

    # recursive=True so a '**' pattern reaches nested layouts (CASP15 keeps the
    # natives directly under T/, CASP16 under T/Targets_processed/).
    paths = sorted(glob.glob(args.pdb_glob, recursive=True))
    if not paths:
        raise SystemExit(f"no files matched {args.pdb_glob}")
    logging.info("converting %d CASP natives", len(paths))

    manifest, failed = [], []
    for path in paths:
        name = os.path.basename(path)
        name = name[:-4] if name.endswith(".pdb") else name
        try:
            rec = convert_one(Path(path), args.out_dir / f"{name}.cif", name, args.revision_date)
            if args.max_residues and rec["native_len"] > args.max_residues:
                failed.append({"target": name, "reason":
                               f"{rec['native_len']} residues > --max-residues {args.max_residues}"})
                continue
            manifest.append(rec)
        except Exception as exc:  # noqa: BLE001 — one bad native must not stop the set
            failed.append({"target": name, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"})
            logging.warning("  FAILED %-12s %s", name, str(exc)[:120])

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"targets": manifest, "failed": failed}, indent=1) + "\n")
    logging.info("converted %d, failed/skipped %d", len(manifest), len(failed))
    if manifest:
        lens = sorted(r["native_len"] for r in manifest)
        logging.info("residue counts: min=%d median=%d max=%d",
                     lens[0], lens[len(lens) // 2], lens[-1])
    logging.info("wrote %s", args.manifest)


if __name__ == "__main__":
    main()
