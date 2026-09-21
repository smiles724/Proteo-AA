#!/usr/bin/env python3
"""R0: the ProteinMPNN baseline row, on the same shared backbones.

    python scripts/design_binder_r0.py \
        --backbones runs/binder_bench/backbones \
        --out runs/binder_bench/designs_r0

TWO STAGES, because no single environment can do both. The payloads are torch
files and the af2ig venv has no torch; the af2ig venv is where ColabDesign and
ProteinMPNN's vanilla weights live, and it is pinned (jax 0.4.35, dm-haiku
0.0.13) tightly enough that installing torch into it to save a stage is not
worth the risk to a validated scorer.

    # stage 1, proteoaa-stage4: payload -> backbone PDB + sidecar
    python scripts/design_binder_r0.py --stage prep --backbones ... --out ...
    # stage 2, af2ig: PDB -> sequence, no torch imported anywhere
    python scripts/design_binder_r0.py --stage mpnn --out ...

Both stages read the same cached collection and stage 1 checks the same
digests, which is what keeps R0 paired with the FaMPNN arms.

### R0's initial guess is weaker, and that is measured rather than waved at

ProteinMPNN designs a sequence and nothing else -- there are no side chains to
write. ColabDesign's AF2 initial guess is
``prev_pos = batch["all_atom_positions"]`` (`colabdesign/af/design.py:166`), so
a backbone-only PDB is a *poorer* starting structure than the full-atom one the
FaMPNN arms produce. Ignoring that would quietly bias J03 - R0 in J03's favour.

Two things follow, both deliberate:

  * R0 is written backbone-only, because that is genuinely all ProteinMPNN
    produces. Packing it with FaMPNN would put FaMPNN inside the "ProteinMPNN
    baseline" and the row would no longer be the external reference it exists
    to be.
  * The scoring stage additionally runs U03 backbone-only. U03 full-atom minus
    U03 backbone-only is then a direct measurement of what the initial guess
    alone is worth, on the same sequences -- so J03 - R0 can be read against
    it instead of assuming the handicap is small.

R0 is the matrix's shared ABSOLUTE reference. It is not the uncoupled half of
any pair: it changes the designer and the context level at once, so a J03 - R0
difference is a designer comparison and `arms.yaml` says so.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ROW_COLUMNS = (
    "sample_id", "target", "binder_length", "sequence", "design_pdb",
    "binder_chain", "target_chains",
    "design_id", "arm", "context", "coupled", "source_sha256",
    "actual_sigma", "event_to_final_rmsd", "delta_h_norm", "hook_calls",
    "seq_steps", "temperature", "seconds",
)

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def write_backbone_pdb(inputs, sequence, path):
    """Backbone-only complex, one chain letter per asym_id.

    ``sequence`` is over ALL rows (target identities plus the binder's), so the
    residue names match what AF2 will be asked to fold.
    """
    from pxf import atom37

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = inputs.coords_af2[0].cpu().numpy()
    mask = inputs.atom_mask[0].cpu().numpy()
    chain = inputs.chain_index[0].cpu().numpy()
    resi = inputs.residue_index[0].cpu().numpy()
    from fampnn.data import residue_constants as rc

    slot = {n: atom37.ATOM37.index(n) for n in BACKBONE_ATOMS}
    lines, serial = [], 1
    for i, letter in enumerate(sequence):
        three = rc.restype_1to3.get(letter, "GLY")
        ch = CHAIN_LETTERS[int(chain[i])]
        for name in BACKBONE_ATOMS:
            s = slot[name]
            if mask[i, s] <= 0:
                continue
            x, y, z = coords[i, s]
            lines.append(
                f"ATOM  {serial:5d} {name:^4s}{three:>4s} {ch}{int(resi[i]) + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                f"{name[0]:>2s}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path





def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=("prep", "mpnn"), required=True)
    parser.add_argument("--backbones", default=None,
                        help="required for --stage prep")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mpnn-model", default="v_48_020",
                        help="ProteinMPNN vanilla weights; the standard choice")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="arms.yaml shared.temperature")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    if args.stage == "prep":
        prep(args, out)
    else:
        run_mpnn(args, out)


def prep(args, out: Path) -> None:
    """Stage 1 (torch): payload -> backbone PDB + the metadata stage 2 cannot see."""
    from pxf.bench.backbone_inputs import (binder_chain_of, chains_of,
                                           load_payload, target_chains_of,
                                           to_design_inputs)

    if not args.backbones:
        raise SystemExit("--stage prep needs --backbones")
    base = Path(args.backbones)
    manifest = json.loads((base / "backbones.json").read_text())
    records = manifest["designs"]
    if args.targets:
        records = [r for r in records if r["target"] in set(args.targets)]
    records.sort(key=lambda r: r["design_id"])
    if args.limit:
        records = records[: args.limit]

    (out / "inputs").mkdir(parents=True, exist_ok=True)
    sidecars = []
    for n, record in enumerate(records):
        path = base / "designs" / f"{record['design_id']}.pt"
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        if digest != record["sha256"]:
            raise SystemExit(
                f"{record['design_id']}: sha256 {digest[:12]} != manifest "
                f"{record['sha256'][:12]}"
            )
        payload = load_payload(path)
        inputs = to_design_inputs(payload, context="complex")

        from pxf import atom37

        # Binder rows carry a placeholder; fix_pos="A" leaves them free.
        seq_in = "".join(
            "G" if b else (atom37.AA_ORDER[a] if a < atom37.UNKNOWN_AA_INDEX else "G")
            for a, b in zip(inputs.aatype[0].tolist(),
                            inputs.binder_mask[0].bool().tolist())
        )
        write_backbone_pdb(inputs, seq_in,
                           out / "inputs" / f"{record['design_id']}.pdb")
        sidecars.append({
            "design_id": record["design_id"],
            "target": record["target"],
            "binder_length": record["binder_length"],
            "source_sha256": record["sha256"],
            "actual_sigma": float(payload["actual_sigma"]),
            "event_to_final_rmsd": record.get("event_to_final_rmsd"),
            "binder_mask": [int(v) for v in inputs.binder_mask[0].tolist()],
            "length": inputs.length,
            "chains": chains_of(inputs),
            "target_chains": target_chains_of(inputs),
            "binder_chain": binder_chain_of(inputs),
        })
        if (n + 1) % 20 == 0 or n + 1 == len(records):
            print(f"  prep {n + 1}/{len(records)}", flush=True)
    (out / "prep.json").write_text(json.dumps(
        {"designs": sidecars, "backbone_sources": manifest.get("sources")},
        indent=2, default=str))
    print(f"prepared {len(sidecars)} backbone PDB(s) in {out / 'inputs'}")


def thread_sequence(pdb_in: Path, pdb_out: Path, sequence: str) -> Path:
    """Rewrite residue names in a backbone PDB. Text only -- stage 2 has no torch.

    Coordinates are copied verbatim, so every arm's initial guess for a given
    design_id is the identical shared backbone rather than merely a similar one.
    """
    three = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    lines, index, seen = [], -1, None
    for line in pdb_in.read_text().splitlines():
        if not line.startswith("ATOM"):
            lines.append(line)
            continue
        key = (line[21], line[22:27])
        if key != seen:
            seen = key
            index += 1
        if index >= len(sequence):
            raise ValueError(
                f"{pdb_in.name}: more residues than the {len(sequence)}-letter "
                "sequence ProteinMPNN returned"
            )
        lines.append(line[:17] + f"{three[sequence[index]]:>3s}" + line[20:])
    if index + 1 != len(sequence):
        raise ValueError(
            f"{pdb_in.name}: {index + 1} residues but sequence has {len(sequence)}"
        )
    pdb_out.parent.mkdir(parents=True, exist_ok=True)
    pdb_out.write_text("\n".join(lines) + "\n")
    return pdb_out


def run_mpnn(args, out: Path) -> None:
    """Stage 2 (jax/ColabDesign): backbone PDB -> designed sequence. No torch."""
    from colabdesign.mpnn import mk_mpnn_model

    prep_path = out / "prep.json"
    if not prep_path.is_file():
        raise SystemExit(f"{prep_path} missing; run --stage prep first")
    sidecars = json.loads(prep_path.read_text())["designs"]
    (out / "designs").mkdir(parents=True, exist_ok=True)
    model = mk_mpnn_model(model_name=args.mpnn_model, seed=args.seed)
    print(f"R0: {len(sidecars)} design(s), ProteinMPNN {args.mpnn_model}")

    rows = []
    for n, side in enumerate(sidecars):
        pdb_in = out / "inputs" / f"{side['design_id']}.pdb"
        started = time.time()
        chains = side["chains"]          # every chain, in asym_id order
        fixed = side["target_chains"]    # all of them but the binder's
        model.prep_inputs(pdb_filename=str(pdb_in), chain=",".join(chains),
                          fix_pos=",".join(fixed))
        sampled = model.sample(num=1, batch=1, temperature=args.temperature)
        seconds = time.time() - started
        # ColabDesign joins chains with "/" in the returned string (here
        # 157 + 1 + 100). Strip it rather than slicing a fixed offset: the
        # separator count is the chain count, which varies by target.
        full = sampled["seq"][0].replace("/", "")
        if len(full) != side["length"]:
            raise SystemExit(
                f"{side['design_id']}: ProteinMPNN returned {len(full)} residues "
                f"for a {side['length']}-row complex"
            )
        binder = [bool(v) for v in side["binder_mask"]]
        binder_seq = "".join(c for c, keep in zip(full, binder) if keep)
        if len(binder_seq) != side["binder_length"]:
            raise SystemExit(f"{side['design_id']}: binder length mismatch")

        sample_id = f"{side['design_id']}__R0"
        pdb = thread_sequence(pdb_in, out / "designs" / f"{sample_id}.pdb", full)
        rows.append({
            "sample_id": sample_id,
            "target": side["target"],
            "binder_length": side["binder_length"],
            "sequence": binder_seq,
            "design_pdb": str(pdb),
            "binder_chain": side["binder_chain"],
            "target_chains": ",".join(side["target_chains"]),
            "design_id": side["design_id"],
            "arm": "R0",
            "context": "complex",
            "coupled": 0,
            "source_sha256": side["source_sha256"],
            "actual_sigma": side["actual_sigma"],
            "event_to_final_rmsd": side.get("event_to_final_rmsd"),
            "delta_h_norm": 0.0,
            "hook_calls": 0,
            "seq_steps": "",
            "temperature": args.temperature,
            "seconds": round(seconds, 2),
        })
        if (n + 1) % 20 == 0 or n + 1 == len(sidecars):
            print(f"  {n + 1}/{len(sidecars)}", flush=True)

    csv_path = out / "designs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    (out / "provenance.json").write_text(json.dumps({
        "arm": "R0", "designer": "proteinmpnn",
        "mpnn_model": args.mpnn_model, "temperature": args.temperature,
        "seed": args.seed, "n": len(rows),
        "initial_guess": "backbone-only; see module docstring",
    }, indent=2, default=str))
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
