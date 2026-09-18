#!/usr/bin/env python3
"""APM's released packer state dict -> an overlay checkpoint for the trainer.

`PXDesignTrainer.overlay_module_from_checkpoint` matches on the model's own key
prefixes, and APM's released tensors are stored unprefixed (`node_feature_net.
linear.weight`, ...) because they came out of APM's own `SideChainModel`. This
re-keys them under `sidechain_module.` so the overlay matches, and writes the
`{"model": ...}` envelope the loader expects.

The port has been shown equivalent to APM's forward (546/546 tensors, chi to
1e-05 rad; `scripts/evaluation/check_packer_matches_apm.py`), so this is a
rename, not a conversion.

    python scripts/data/make_apm_packer_overlay.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

SRC = "/hai/scratch/shenjm/apm_weights/sidechain_state_dict.pt"
OUT = "/hai/scratch/shenjm/apm_weights/apm_packer_overlay.pt"
PREFIX = "sidechain_module."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seeds the initialisation of the tensors APM does not "
                         "supply, so the donor is reproducible.")
    ap.add_argument("--plm-checkpoint",
                    default="/hai/scratch/shenjm/plm_weights/esm2_t33_650M_UR50D.pt")
    ap.add_argument("--seq-cond", default="none", choices=["none", "a_token", "plm", "both"],
                    help="Must equal the run's --sc-packer-seq-cond: it is a "
                         "recorded layout key, so a mismatch is refused.")
    args = ap.parse_args()

    sd = torch.load(args.src, map_location="cpu", weights_only=True)
    if any(k.startswith(PREFIX) for k in sd):
        raise SystemExit(f"{args.src} is already prefixed; nothing to do")

    # The component loader requires a COMPLETE donor -- every
    # `sidechain_module.*` tensor the model builds -- because a partial
    # component would leave some of it at whatever the run happened to
    # initialise. APM's release has 546 tensors and our module has three more
    # that are ours, not APM's: `a_proj` (the a_token channel's projection,
    # zero-initialised by `init="final"`), `atom_embed` (the atom-name
    # embedding), and `plm_conditioner` (built on every arm so all four share
    # one parameter set). So start from OUR module's own fresh initialisation
    # and overwrite the 546 APM carries; the remainder then holds exactly what
    # a fresh run would have had.
    import torch.nn as nn

    from pxdesign_train.sidechain.packer import TorsionPacker

    torch.manual_seed(int(args.seed))
    packer = TorsionPacker(
        c_res=768, c_node=256, c_pair=128, n_blocks=6, ipa_c_hidden=16,
        ipa_no_heads=8, no_qk_points=8, no_v_points=12, seq_tfmr_num_heads=4,
        seq_tfmr_num_layers=4, num_torsion_blocks=4, seq_cond=args.seq_cond,
        embed_aatype=True, embed_rotvecs=True, use_mlp=True, embed_chain=True,
        random_torsion_input=True,
        plm_checkpoint=(args.plm_checkpoint if args.seq_cond in ("plm", "both") else ""),
    )
    out = {f"{PREFIX}{k}": v.clone() for k, v in packer.state_dict().items()}
    n_before = len(out)
    overwritten = 0
    for k, v in sd.items():
        key = f"{PREFIX}{k}"
        if key not in out:
            raise SystemExit(f"APM tensor {k} has no home in our packer")
        if out[key].shape != v.shape:
            raise SystemExit(f"{k}: APM {tuple(v.shape)} vs ours {tuple(out[key].shape)}")
        out[key] = v
        overwritten += 1
    ours_only = sorted(k[len(PREFIX):] for k in out if f"{PREFIX}" + k[len(PREFIX):] not in
                       {f"{PREFIX}{x}" for x in sd})
    print(f"donor: {n_before} tensors, {overwritten} from APM, "
          f"{n_before - overwritten} at our own init: {ours_only[:6]}")
    # `check_sc_layout` (checkpoints.py:109) refuses a side-chain donor that does
    # not record its complete `sidechain_arch`, and then refuses one whose record
    # disagrees with the run's `configs.sidechain`. The point of that guard is
    # that a layout difference means different parameter names and a different
    # input contract, which load_state_dict would accept silently. APM's release
    # carries no such record because it was never a donor for this codebase, so
    # it is stated here -- for the torsion packer, not for the old Cartesian
    # module: `torsion_packer` on, `chi_output` off (the packer supersedes that
    # read-out), `edm` off (the guard requires it, and the packer is one-step).
    # The three keys the guard named on the first attempt are set from what the
    # run actually configures, not from the old Cartesian module's defaults:
    # the torsion packer takes frames rather than centred coordinates, does not
    # use the a/bs concat fusion, and `packer_seq_cond` is a STRING key
    # (SC_LAYOUT_KEYS_STR) whose value is the ablation arm.
    arch = dict(
        bb_context=True, centre_coord_input=False, frame_aware_head=False,
        template_residual=False, type_logits_input=True, edm=False,
        a_bs_concat=False, q_bs=False, chi_output=False, torsion_packer=True,
        packer_seq_cond=args.seq_cond,
    )
    torch.save({"model": out, "sidechain_arch": arch}, args.out)
    print("sidechain_arch:", arch)
    n_par = sum(v.numel() for v in out.values())
    print(f"{len(out)} tensors, {n_par:,} parameters -> {args.out}")
    print("sample keys:", list(out)[:3])


if __name__ == "__main__":
    main()
