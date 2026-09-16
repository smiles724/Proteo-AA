#!/usr/bin/env python3
"""Precompute the a_token the packer's a_token arm conditions on, at zero noise.

WHY A CACHE. a_token comes out of a FROZEN PXDesign trunk, and at zero
coordinate noise the trunk's input is fully determined by the structure. So
a_token is a pure function of the chain: computing it once per chain and
reading it back costs a disk read instead of a trunk forward on every step, and
the a_token arm then trains at the same speed as the none arm.

WHAT "NOISE = 0" MEANS HERE, PRECISELY. Two separate things are called noise:

  1. The Gaussian perturbation added to the coordinates. This is set to EXACTLY
     ZERO -- `clean_coordinate_input=True`, the flag that already existed for
     the AA clean-coordinate diagnostic. The trunk sees the native backbone.

  2. The value written into the time channel. EDM conditions on
     `c_noise = ln(sigma / sigma_data) / 4`, which is -inf at sigma = 0, so
     sigma = 0 is not a representable input to this network -- not a policy
     choice, an arithmetic one. The conditioning sigma is therefore pinned at
     `--sigma-floor`, default 4e-4, which is Protenix's own `s_min`: the
     terminal noise level of its sampling schedule, i.e. the cleanest state the
     trunk was ever asked to condition on. The scale factor it induces,
     `c_in = 1/sqrt(sigma^2 + sigma_data^2)`, differs from its value at
     sigma = 0 by 3e-10 relative.

  The value actually used is recorded in the cache manifest, so no reader has
  to infer it.

    python scripts/data/build_a_token_cache.py --pkl-dir ... --out ... --check 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SIGMA_DATA = 16.0
PROTENIX_S_MIN = 4e-4          # Protenix/configs/configs_base.py :: s_min


def build_trainer(args, device, torch):
    """Stage II-A configuration with the packer's a_token arm selected.

    Built from `train_protenix_monomer.py`'s own argument parser so the trunk
    here cannot drift from the trunk the arms train against.
    """
    import importlib.util

    from pxdesign_train.runner import PXDesignTrainer, TrainerComponents
    from pxdesign_train.data.curriculum import CurriculumMultiDataset, CurriculumSchedule

    spec = importlib.util.spec_from_file_location(
        "tpm", str(REPO / "scripts" / "training" / "train_protenix_monomer.py"))
    tpm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpm)

    saved = sys.argv
    sys.argv = [
        "x", "--training-stage", "sidechain_warmup",
        "--data-root", args.data_root,
        "--output-dir", str(Path(args.out) / "_cfg"),
        "--crop-size", str(args.crop_size), "--max-n-token", str(args.crop_size),
        "--eval-interval", "0", "--eval-samples", "0", "--num-workers", "0",
        "--device", args.device, "--dtype", args.dtype,
        "--aa-backend", "sc_only", "--sc-torsion-packer",
        "--sc-packer-seq-cond", "a_token",
    ]
    cfg_args = tpm.parse_args()
    configs = tpm.build_configs(cfg_args, device)
    sys.argv = saved

    # One noise draw per item: there is nothing to average over once the noise
    # is fixed, and N_sample > 1 would just repeat the same forward.
    configs.training.diffusion_batch_size = 1
    # (1) no coordinate noise, (2) a representable conditioning sigma.
    configs.residue_type.clean_coordinate_input = True
    configs.residue_type.forced_sigmas = f"{args.sigma_floor:.10g}"
    configs.training.checkpoint_include_prefixes = []
    return tpm, configs, PXDesignTrainer, TrainerComponents, CurriculumMultiDataset, CurriculumSchedule


def make_dataset(cif_path, chain_id, crop_size):
    from pxdesign_train.runner import DesignSourceDataset
    from pxdesign_train.runner.cif_provider import CifFileProvider

    prov = CifFileProvider(cif_paths=[cif_path],
                           binder_chain_ids=[chain_id] if chain_id else None)
    return DesignSourceDataset(
        prov, source_name="apm", crop_size=int(crop_size),
        compute_sidechain=True, backbone_only_binder=True,
        inference_safe_binder=True, ref_pos_augment=False,
        hotspot_force_zero_prob=0.0,
        # `aa_mask_mode="none"` is what `--training-stage sidechain_warmup` sets
        # (train_protenix_monomer.py:1331), and therefore what the earlier
        # a_token arm conditioned on. The CASP benchmark uses "all", and copying
        # that here turns EVERY token into the [xpb] design token -- measured:
        # 154/154 on 101m -- so a_token would carry no sequence at all. Nothing
        # errors; the arm just silently stops being the same experiment.
        aa_mask_mode="none", aa_mask_prob=0.0,
        max_crop_retries=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", nargs="*", default=[])
    ap.add_argument("--pkl-dir", default="")
    ap.add_argument("--ids-csv", default="")
    ap.add_argument("--out", required=True, help="cache directory")
    ap.add_argument("--cif-dir", default="", help="where CIFs go (default <out>/cif)")
    ap.add_argument("--checkpoint",
                    default="/hai/scratch/shenjm/pxdesign_official/pxdesign_v0.1.0.pt")
    ap.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    ap.add_argument("--sigma-floor", type=float, default=PROTENIX_S_MIN)
    ap.add_argument("--compare-sigma", type=float, default=0.4,
                    help="second sigma to quantify against, in --check mode")
    ap.add_argument("--crop-size", type=int, default=768)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", type=int, default=0,
                    help="run the consistency check on the first N chains and exit")
    args = ap.parse_args()

    os.environ.setdefault("PROTENIX_ROOT_DIR", args.data_root)
    os.environ.setdefault("LAYERNORM_TYPE", "torch")
    sys.path.insert(0, str(REPO))
    import torch

    # By path: `scripts` resolves to Protenix's package whenever Protenix is on
    # PYTHONPATH, so a plain `from scripts.data...` import raises here.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "_apm_pkl_to_cif", str(REPO / "scripts" / "data" / "apm_pkl_to_cif.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    convert = _mod.convert
    from pxdesign_train.sidechain.apm_dataset import featurise

    files = [Path(p) for p in args.pkl]
    if args.ids_csv:
        import pandas as pd
        ids = pd.read_csv(args.ids_csv)["pdb_name"].astype(str)
        files += [Path(args.pkl_dir) / f"{i}.pkl" for i in ids]
    elif args.pkl_dir:
        files += sorted(Path(args.pkl_dir).glob("*.pkl"))
    files = [f for f in files if f.is_file()]
    if args.check:
        files = files[:args.check]
    elif args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("no input pickles")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cif_dir = Path(args.cif_dir or out / "cif"); cif_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    (tpm, configs, PXDesignTrainer, TrainerComponents,
     CurriculumMultiDataset, CurriculumSchedule) = build_trainer(args, device, torch)

    c_noise = math.log(args.sigma_floor / SIGMA_DATA) / 4
    c_in = 1.0 / math.sqrt(args.sigma_floor**2 + SIGMA_DATA**2)
    print(f"conditioning: coordinate noise = 0 exactly; sigma_floor="
          f"{args.sigma_floor:g} -> c_noise={c_noise:.6f}, c_in={c_in:.10f} "
          f"(c_in at sigma=0 is {1/SIGMA_DATA:.10f})", flush=True)

    # The trainer needs a dataset at construction; the first chain serves.
    first = convert(files[0], cif_dir)
    ds0 = make_dataset(first["cif"], first["chain"], args.crop_size)
    from torch.utils.data import DataLoader
    from pxdesign_train.runner.trainer import _identity_collate
    loader0 = DataLoader(ds0, batch_size=1, shuffle=False, num_workers=0,
                         collate_fn=_identity_collate)
    components = TrainerComponents(
        train_dataset=CurriculumMultiDataset(datasets=[ds0], source_names=["apm"],
                                             per_item_weights=[[1.0] * len(ds0)]),
        schedule=CurriculumSchedule(stage1={"apm": 1.0}, stage2={"apm": 1.0},
                                    stage1_end_step=1, stage2_start_step=2),
        train_samples_per_epoch=1, eval_dataloader=loader0)
    trainer = PXDesignTrainer(configs=configs, components=components, device=device,
                              checkpoint_dir=None,
                              load_checkpoint_path=str(Path(args.checkpoint).resolve()),
                              checkpoint_params_only=True)
    # `trainer.model` may be a DDP wrapper; the flags below are read off the
    # module itself, so setting them on a wrapper would do nothing and the
    # noise would quietly stay on. Always go through raw_model.
    net = trainer.raw_model
    net.eval()

    # Pinning sigma only works if the sampler is the forced variant, and that
    # is only built when `residue_type.forced_sigmas` was non-empty at model
    # construction. If it is the plain EDM sampler, assigning `.forced_sigmas`
    # silently does nothing: every chain would be cached at a RANDOM training
    # sigma with clean coordinates -- wrong, and invisible in the output.
    from pxdesign_train.model import ForcedSigmaNoiseSampler
    if not isinstance(net.training_noise_sampler, ForcedSigmaNoiseSampler):
        raise SystemExit(
            "training_noise_sampler is "
            f"{type(net.training_noise_sampler).__name__}, not "
            "ForcedSigmaNoiseSampler -- residue_type.forced_sigmas did not "
            "reach model construction, so the conditioning sigma is not pinned."
        )
    if not net.aa_clean_coordinate_input:
        raise SystemExit("aa_clean_coordinate_input is False after construction: "
                         "coordinates would be perturbed.")
    print("verified: clean_coordinate_input=True and sigma is pinned "
          f"({type(net.training_noise_sampler).__name__})", flush=True)

    def a_token_for(cif, chain, sigma=None, clean=True, seed=0):
        """One forward; returns (a_token [L, C], restype [L], sigma actually used).

        `clean=True` is the zero-coordinate-noise mode this cache is for.
        `clean=False` restores the Gaussian perturbation, which is what the
        earlier runs did -- the check uses it to measure the difference between
        the two settings instead of asserting it is small.

        The third return value is read back out of the forward (`out["sigma"]`),
        not echoed from the argument: it is what the network was conditioned on,
        which is the only version of this number worth reporting.
        """
        if sigma is not None:
            net.training_noise_sampler.forced_sigmas = (float(sigma),)
        net.aa_clean_coordinate_input = bool(clean)
        ds = make_dataset(cif, chain, args.crop_size)
        # Exactly what `forward_loss` does: collate, move, call. The model takes
        # the unbatched dict; adding a leading dim here trips an assertion deep
        # in the atom-attention encoder.
        batch = trainer._to_device(_identity_collate([ds[0]]))
        # The coordinate augmentation is a random rotation + translation of the
        # GT -- a rigid motion, not noise, but it makes the trunk's input differ
        # run to run. NUMPY has to be seeded too, not just torch: the rotation
        # comes from scipy's `Rotation.random` (Protenix/protenix/model/utils.py
        # :108), which draws on numpy's global RNG, so a torch-only seed leaves
        # it free and a "repeat" forward silently measures rotation sensitivity
        # instead of determinism.
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        with torch.no_grad():
            o = net(input_feature_dict=batch["input_feature_dict"],
                    label_dict=batch["label_dict"], mode="train")
        a = o["h_res_candidate"]
        fd = batch["input_feature_dict"]
        rt = fd["restype"].reshape(-1, fd["restype"].shape[-1]).argmax(-1).cpu()
        ri = fd["residue_index"].reshape(-1).cpu()
        used = o["sigma"].reshape(-1).float().cpu()
        return a.reshape(-1, a.shape[-1]).float().cpu(), (rt, ri), used

    if args.check:
        run_check(args, files, cif_dir, convert, featurise, a_token_for, torch)
        return

    manifest = {"sigma_floor": args.sigma_floor, "c_noise": c_noise, "c_in": c_in,
                "clean_coordinate_input": True, "checkpoint": args.checkpoint,
                "chains": []}
    for f in files:
        try:
            rec = convert(f, cif_dir)
            a, _ids, used = a_token_for(rec["cif"], rec["chain"], sigma=args.sigma_floor)
            if abs(float(used.max()) - args.sigma_floor) > 1e-9:
                raise RuntimeError(f"forward ran at sigma {float(used.max()):g}, "
                                   f"not the pinned {args.sigma_floor:g}")
            np.save(out / f"{rec['target']}.npy", a.numpy().astype(np.float16))
            manifest["chains"].append({"name": rec["target"], "L": int(a.shape[0]),
                                       "C": int(a.shape[1]),
                                       "apm_span_res": rec["apm_span_res"]})
        except Exception as e:                        # noqa: BLE001
            manifest.setdefault("failed", []).append(
                {"pkl": str(f), "error": f"{type(e).__name__}: {e}"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"cached {len(manifest['chains'])}/{len(files)}; "
          f"failed {len(manifest.get('failed', []))}", flush=True)


def run_check(args, files, cif_dir, convert, featurise, a_token_for, torch):
    """Does a_token line up with the packer's residues, and is it deterministic?

    Five questions, because each has its own way of being silently wrong:

      shape        one a_token row per residue the packer sees?
      alignment    is row i the SAME residue as the packer's residue i?
      sequence     did the trunk see the real residue types, or [xpb]?
      determinism  does a repeat forward reproduce the tensor?
      sigma        how much does dropping the noise actually change a_token?

    Alignment is checked twice over, by `residue_index` and by residue name.
    A length check alone would miss a chain where the two featurisers disagree
    by one insertion and one deletion, which is exactly the shape of the bug an
    interior unresolved residue would cause.

    The sequence question is here because it has already been wrong once: the
    CASP benchmark's `aa_mask_mode="all"` makes every token the [xpb] design
    token, and a_token then carries no sequence at all without anything failing.

    The sigma comparison is between two REGIMES, not two numbers in one:
    clean coordinates at the sigma floor, against noisy coordinates at
    sigma=0.4, which is what the earlier arms trained on.
    """
    from pxdesign.data.constants import STD_RESIDUES_WITH_GAP
    from openfold.np.residue_constants import restype_1to3, restypes

    px_name = {v: k for k, v in STD_RESIDUES_WITH_GAP.items()}
    apm_name = [restype_1to3[r] for r in restypes] + ["UNK"]

    print("\n=== a_token bridge consistency check ===", flush=True)
    print(f"  regime A: coordinate noise 0, conditioning sigma {args.sigma_floor:g}")
    print(f"  regime B: coordinate noise on, sigma {args.compare_sigma:g} "
          f"(what the earlier a_token arm trained under)", flush=True)
    ok = True
    for f in files:
        rec = convert(f, cif_dir)
        feats = featurise(f)                       # APM's own featurisation
        apm_aat = feats["aatypes_1"].cpu()
        apm_ri = feats["res_idx"].cpu()
        n_apm = int(apm_aat.shape[0])

        a, (rt, ri), s_used = a_token_for(rec["cif"], rec["chain"],
                                          sigma=args.sigma_floor, clean=True, seed=0)
        a2, _, _ = a_token_for(rec["cif"], rec["chain"], sigma=args.sigma_floor,
                               clean=True, seed=0)
        a_rot, _, _ = a_token_for(rec["cif"], rec["chain"], sigma=args.sigma_floor,
                                  clean=True, seed=7)
        a_hi, _, s_hi = a_token_for(rec["cif"], rec["chain"], sigma=args.compare_sigma,
                                    clean=False, seed=0)

        # Read back, not echoed: proves the pin took effect on this forward.
        if abs(float(s_used.max()) - args.sigma_floor) > 1e-9:
            print(f"  {rec['target']}: SIGMA NOT PINNED -- ran at {float(s_used.max()):g}")
            ok = False

        len_ok = a.shape[0] == n_apm
        n = min(len(rt), n_apm)
        # Informational only. The CIF has to renumber residues sequentially
        # because APM flattens insertion codes and its residue_index is not
        # unique (102l has two residues at index 40), so a disagreement here is
        # expected on those chains and says nothing about alignment.
        ri_ok = len_ok and bool((ri[:n] == apm_ri[:n]).all())
        # This is the alignment test that counts: the residue NAME at every
        # position, which an insertion paired with a deletion cannot survive.
        same = sum(1 for k in range(n)
                   if px_name[int(rt[k])] == apm_name[int(apm_aat[k])])
        design_tok = sum(1 for k in range(n) if px_name[int(rt[k])] in
                         ("xpb", "xpa", "rbb", "raa", "-"))
        det = (a - a2).abs().max().item()
        rot = (a - a_rot).abs().max().item()
        d_sigma = (a - a_hi).abs().max().item()
        scale = max(a.abs().max().item(), 1e-9)

        row_ok = len_ok and same == n and det < 1e-3
        ok = ok and row_ok
        print(f"  {rec['target']:6s} L_apm={n_apm:4d} L_atoken={a.shape[0]:4d} "
              f"{'len OK' if len_ok else 'LEN MISMATCH'}  "
              f"res_idx {'same' if ri_ok else 'renumbered'}  "
              f"restype {same}/{n}{'' if design_tok == 0 else f' ({design_tok} design tokens!)'}  "
              f"repeat={det:.2e}  rot-aug={rot:.2e}  "
              f"sigma A={float(s_used.max()):.2e} B={float(s_hi.max()):.3g}  "
              f"|A-B|max={d_sigma:.4f} ({100*d_sigma/scale:.1f}% of |a|max={scale:.3f})",
              flush=True)
    print("VERDICT:", "bridge aligned, sequence-carrying and deterministic" if ok
          else "BRIDGE NOT USABLE -- see mismatches above", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
