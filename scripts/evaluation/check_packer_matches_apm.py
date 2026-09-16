#!/usr/bin/env python3
"""Does our port compute the same function as APM's SideChainModel?

`sidechain/packer.py` is a port, and every tensor of APM's released
`sidechain_model.ckpt` now matches ours by name and shape. That is necessary
and not sufficient: identical parameter names say nothing about the order of
concatenated features, the units of the translations, the sin/cos convention,
or whether a mask is applied before or after a projection. Any of those can be
wrong while the weights still load cleanly, and the result is a model that runs
and is quietly not the model whose weights it holds.

So: replay APM's own forward. `dump_apm_reference_forward.py` runs their module
on a fixed input in their environment and saves inputs and outputs; this script
loads the same weights into ours, feeds the same input, and compares the
predicted torsions.

    PYTHONPATH=$REPO:$REPO/PXDesign:$REPO/Protenix LAYERNORM_TYPE=torch \
    python scripts/evaluation/check_packer_matches_apm.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

DUMP = "/hai/scratch/shenjm/apm_weights/apm_reference_forward.pt"
SHAPES = "/hai/scratch/shenjm/apm_weights/sidechain_ckpt_shapes.json"
WEIGHTS = "/hai/scratch/shenjm/apm_weights/sidechain_state_dict.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    from pxdesign_train.sidechain.instantiate import instantiate_from_type_indices
    from pxdesign_train.sidechain.packer import TorsionPacker

    ref = torch.load(args.dump, map_location="cpu", weights_only=False)
    sd = torch.load(args.weights, map_location="cpu", weights_only=True)
    feats = ref["inputs"]

    # Released config: node 256 / edge 128 / 6 blocks / IPA 16-8-8-12 /
    # seq-tfmr 4x4 / 4 torsion blocks / use_mlp / embed_chain / embed_aatype.
    packer = TorsionPacker(
        c_res=768, c_node=256, c_pair=128, n_blocks=6, ipa_c_hidden=16,
        ipa_no_heads=8, no_qk_points=8, no_v_points=12, seq_tfmr_num_heads=4,
        seq_tfmr_num_layers=4, num_torsion_blocks=4, seq_cond="none",
        embed_aatype=True, embed_rotvecs=True, use_mlp=True, embed_chain=True,
        # APM draws a uniform random torsion input; the reference dump pinned it
        # to zero, and this is the switch that does the same on our side.
        random_torsion_input=False,
    ).eval()
    missing, unexpected = packer.load_state_dict(sd, strict=False)
    extra_ours = [k for k in missing]
    print(f"load_state_dict: {len(unexpected)} unexpected, {len(missing)} missing")
    if unexpected:
        raise SystemExit(f"checkpoint has tensors our module lacks: {unexpected[:5]}")
    print("  missing (ours only, expected):", sorted(extra_ours))

    B, L = feats["res_mask"].shape
    types = feats["aatypes_1"].long()
    ids, chem = instantiate_from_type_indices(types)
    logits = torch.nn.functional.one_hot(types, 20).float() * 40.0 - 20.0

    with torch.no_grad():
        _xyz, _sc, _bb = packer(
            torch.zeros(B, L, 768),          # h_res: seq_cond="none" zeroes it out
            logits, ids, chem,
            torch.zeros(B, L, chem.shape[-1], 3),   # noisy_coords: ignored
            torch.zeros(B),                          # t: ignored
            frame_R=feats["rotmats_1"], frame_t=feats["trans_1"],
            res_mask=feats["res_mask"].bool(),
            residue_index=feats["res_idx"], asym_id=feats["chain_idx"],
        )
    got = packer.last_torsions
    want_chi = ref["pred_torsions"]
    want_unit = ref["pred_sincos"]
    want_raw = ref["pred_sincos_unnorm"]

    def report(name, a, b):
        d = (a - b).abs().max().item()
        print(f"  {name:28s} max|diff| = {d:.3e}   {'OK' if d < args.tol else 'MISMATCH'}")
        return d

    print("\nagainst APM's own forward, same weights, same input:")
    worst = max(
        report("unnormalised sin/cos", got["raw"], want_raw),
        report("unit sin/cos", got["unit"], want_unit),
        report("chi (radians)", got["chi"], want_chi),
    )
    # chi compared modulo 2pi as well, in case only the wrapping differs.
    delta = got["chi"] - want_chi
    wrapped = torch.atan2(delta.sin(), delta.cos()).abs().max().item()
    print(f"  {'chi, wrapped to (-pi, pi]':28s} max|diff| = {wrapped:.3e}"
          f"  ({math.degrees(wrapped):.2e} deg)")

    if worst < args.tol:
        print(f"\nEQUIVALENT within {args.tol:g}: the port computes APM's function.")
    else:
        raise SystemExit(f"\nNOT EQUIVALENT (worst {worst:.3e}).")


if __name__ == "__main__":
    main()
