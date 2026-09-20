"""ESMFold once per unique sequence, for the official-runtime pilot.

The three arms of a triplet share their sequence by construction -- it is
designed once, before the arms branch, and frozen. So a refold is a property
of the *sequence*, not of the arm: one prediction per (target, seed), at most
eight for a 4 x 2 pilot. The same refold is then compared against each arm's
own final backbone, which is where the arms actually differ.

That also means pLDDT is identical across the arms of a triplet. It is a
confidence descriptor for the sequence, and reporting it as an outcome of the
intervention would be a category error.

Configuration is fixed and recorded: no recycling sweep, no chunking changes
between runs, one dtype. Runs in the esmfold venv, which is a separate
environment from the official PXDesign one on purpose.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

WEIGHTS = "/hai/scratch/yfsun/tool_weights/facebook__esmfold_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True,
                    help="directory of <target>_s<seed>/ from the pilot")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--max-predictions", type=int, default=8)
    ap.add_argument("--num-recycles", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoTokenizer, EsmForProteinFolding

    base = Path(args.pilot_dir)
    jobs = sorted(base.glob("*/sequence.fasta"))
    if not jobs:
        print(f"no sequences under {base}", file=sys.stderr)
        return 1

    # Deduplicate by sequence: two triplets that happened to design the same
    # chain should not cost two predictions, and must not get two answers.
    by_sequence = {}
    for path in jobs:
        lines = path.read_text().splitlines()
        seq = "".join(l.strip() for l in lines if not l.startswith(">"))
        by_sequence.setdefault(seq, []).append(path.parent)
    print(f"{len(jobs)} triplet(s), {len(by_sequence)} unique sequence(s)")
    if len(by_sequence) > args.max_predictions:
        print(f"refusing: {len(by_sequence)} unique sequences exceeds "
              f"--max-predictions {args.max_predictions}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    model = EsmForProteinFolding.from_pretrained(args.weights).to(device).eval()
    model.trunk.set_chunk_size(64)

    config = dict(num_recycles=args.num_recycles, chunk_size=64,
                  weights=args.weights, dtype="fp32")
    print("esmfold config:", json.dumps(config))

    written = 0
    for seq, dirs in by_sequence.items():
        inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, num_recycles=args.num_recycles)
        # `positions` is [blocks, batch, L, n_atom, 3] and holds **atom14**,
        # not atom37 -- upstream converts it with atom14_to_atom37 before
        # writing a PDB. CA happens to be index 1 in both layouts, so the
        # index is stable, but it is asserted rather than assumed.
        pos = out["positions"][-1, 0]
        if pos.shape[-2] not in (14, 37):
            raise RuntimeError(f"unexpected atom layout {tuple(pos.shape)}")
        ca = pos[:, 1].float().cpu().numpy()

        plddt_atom = out["plddt"][0].float().cpu().numpy()
        plddt = plddt_atom[:, 1] if plddt_atom.ndim == 2 else plddt_atom
        # `categorical_lddt` returns a bin mean in [0, 1]; nothing upstream
        # rescales it. Normalise to the conventional 0-100 so the number is
        # comparable to published pLDDT, and record which branch was taken.
        plddt_scale = 100.0 if float(np.nanmax(plddt)) <= 1.0 else 1.0
        plddt = plddt * plddt_scale
        mean_plddt = float(np.mean(plddt))
        if len(ca) != len(seq):
            raise RuntimeError(
                f"refold returned {len(ca)} residues for a {len(seq)}-aa "
                "sequence; the scorer would pair the wrong residues")
        for d in dirs:
            np.savez(d / "refold.npz", ca=ca, plddt=plddt,
                     sequence=np.array(seq), mean_plddt=np.array(mean_plddt),
                     plddt_scale=np.array(plddt_scale))
            (d / "refold.json").write_text(json.dumps(
                dict(mean_plddt=round(mean_plddt, 3), length=len(seq),
                     plddt_scale=plddt_scale,
                     shared_with=[str(x.name) for x in dirs], **config),
                indent=2))
            written += 1
        print(f"  {len(seq)} aa -> mean pLDDT {mean_plddt:6.2f}  "
              f"({len(dirs)} triplet(s): {[d.name for d in dirs]})")

    print(f"\n{len(by_sequence)} prediction(s), written to {written} triplet dir(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
