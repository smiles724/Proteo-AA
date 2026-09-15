#!/usr/bin/env python3
"""Train the PXDesign <-> FaMPNN coupling adapters (phases 1-3).

Only the adapters train; both donors stay frozen, and the adapters are
zero-initialized, so step 0 of any phase reproduces the two pretrained models
exactly.

    python scripts/train_couple.py --config configs/couple_phase1.yaml \
        --structures <dir of PDBs | file of paths> --out runs/phase1 \
        --backbone stub

**No data source is assumed or defaulted.** ``--structures`` is required and
there is no fallback path; a run that cannot find its data fails immediately
rather than quietly training on something else.

The backbone driver is selected explicitly:

``--backbone stub``
    Returns the input backbone unchanged with token features derived
    deterministically from its geometry. Enough for smoke tests and for
    exercising the graph; **not** a scientific configuration, because the
    "proposal" is the ground truth.

``--backbone pxdesign``
    The real driver. Not wired yet: PXDesign's official inference runner cannot
    generate monomers (it designs a binder against a target), so the backbone has
    to be driven through pxdesign_train. Raises with that explanation rather than
    silently substituting the stub.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

logger = logging.getLogger("pxf.train_couple")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/couple_phase{1,2,3}.yaml")
    p.add_argument("--structures", required=True,
                   help="directory of PDBs, or a text file of paths (no default)")
    p.add_argument("--out", required=True)
    p.add_argument("--backbone", required=True, choices=("stub", "pxdesign"),
                   help="which backbone driver to couple to")
    p.add_argument("--phase", default=None, help="override the config's phase")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--sigma", type=float, default=1.0,
                   help="backbone noise level fed to the cycle")
    p.add_argument("--fampnn-weights", default="0.0", choices=("0.0", "0.3", "0.3-cath"))
    p.add_argument("--fampnn-checkpoint", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--pack-steps", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--allow-unpinned-sources", action="store_true")
    return p.parse_args(argv)


def resolve_structures(spec):
    """A directory of PDBs or a file listing paths. No implicit default."""
    path = Path(spec)
    if path.is_dir():
        found = sorted(str(p) for p in path.glob("*.pdb"))
        if not found:
            raise SystemExit(f"--structures {path} contains no .pdb files")
        return found
    if path.is_file():
        found = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        missing = [p for p in found if not Path(p).is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} listed path(s) do not exist, e.g. {missing[:3]}")
        if not found:
            raise SystemExit(f"--structures {path} is empty")
        return found
    raise SystemExit(f"--structures {path} is neither a directory nor a file")


def make_stub_backbone(length, channels):
    """Token features derived from the backbone, so A_BS has real signal to read.

    Deterministic and frozen: a fixed random projection of per-residue CA
    displacements. It carries geometry, which is what a_token carries, without
    pretending to be PXDesign.
    """
    generator = torch.Generator().manual_seed(0)
    projection = torch.randn(9, channels, generator=generator) * (channels ** -0.5)

    def backbone(x_noisy, sigma, *, feedback=None):
        flat = x_noisy.reshape(-1, length, 4, 3) if x_noisy.dim() == 2 else x_noisy
        features = flat.reshape(flat.shape[0], length, 12)[..., :9] @ projection
        if feedback is not None:
            # The correction must actually reach the coordinates, or L_BB has no
            # gradient path to A_SB and phase 2 trains nothing while still
            # producing a loss curve. A crude scalar coupling is enough for a
            # smoke driver; the real driver injects it before the atom decoder.
            return x_noisy + feedback.mean(), features
        return x_noisy, features
    return backbone


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import yaml
    from fampnn.data import residue_constants as rc
    from fampnn.model.sd_model import SeqDenoiser
    from pxf import atom37, provenance
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.controller import CoupledDenoiser, Topology
    from pxf.couple.trainer import CoupledBatch, CoupledTrainer, CoupleSettings
    from pxf.device import select_device
    from pxf.train.data import StructureCropDataset, collate
    from pxf.train.trainer import OptimSettings

    if args.backbone == "pxdesign":
        raise SystemExit(
            "The PXDesign backbone driver is not wired yet. PXDesign's official "
            "inference runner designs a binder against a target and cannot generate "
            "monomers, so the backbone must be driven through pxdesign_train "
            "(Proteo-AA's featurizer plus its own sampler). Use --backbone stub for "
            "smoke tests until that driver lands.")

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    couple_cfg = dict(config.get("couple", {}))
    optim_cfg = dict(config.get("optim", {}))
    adapter_cfg = dict(config.get("adapters", {}))
    for key, value in (("phase", args.phase), ("max_steps", args.max_steps),
                       ("seed", args.seed), ("pack_steps", args.pack_steps)):
        if value is not None:
            couple_cfg[key] = value
    if args.lr is not None:
        optim_cfg["lr"] = args.lr

    structures = resolve_structures(args.structures)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger.info("%d structure(s); phase %s", len(structures), couple_cfg.get("phase"))

    checkpoint = (Path(args.fampnn_checkpoint) if args.fampnn_checkpoint
                  else provenance.fampnn_checkpoint(args.fampnn_weights))
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fampnn = SeqDenoiser(bundle["model_cfg"])
    fampnn.load_state_dict(bundle["state_dict"], strict=True)
    fampnn.eval()
    fampnn.requires_grad_(False)                 # donors frozen: the whole point
    device = select_device(args.device)
    fampnn.to(device)

    dataset = StructureCropDataset(structures, crop_size=args.crop_size, noise=0.0,
                                   seed=int(couple_cfg.get("seed", 0)))
    c_h_V = int(fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V)
    c_token = 384
    adapters = CouplingAdapters(c_token, c_h_V, **adapter_cfg).to(device)
    backbone = make_stub_backbone(args.crop_size, c_token)
    controller = CoupledDenoiser(backbone, fampnn, adapters,
                                 phase=couple_cfg.get("phase", "bb_to_sc"),
                                 pack_steps=couple_cfg.get("pack_steps"))
    frozen = dict(fampnn=provenance.weight_record(checkpoint, variant=args.fampnn_weights),
                  upstream=provenance.runtime_sources(
                      strict=not args.allow_unpinned_sources, components=("fampnn",)),
                  backbone_driver=args.backbone)
    trainer = CoupledTrainer(controller, out_dir=out,
                             optim=OptimSettings(**optim_cfg),
                             settings=CoupleSettings(**couple_cfg),
                             device=device, frozen_identity=frozen)
    if args.resume:
        logger.info("resumed at step %d", trainer.resume(args.resume))

    (out / "run_config.json").write_text(json.dumps(dict(
        couple=couple_cfg, optim=optim_cfg, adapters=adapter_cfg,
        n_structures=len(structures), backbone=args.backbone,
        device=str(device), frozen=frozen, arguments=vars(args)), indent=2, default=str))

    slots = list(atom37.BACKBONE_SLOTS)

    def batches():
        """Endless stream of coupled batches drawn from the supplied structures."""
        noise_gen = torch.Generator().manual_seed(int(couple_cfg.get("seed", 0)) + 1)
        epoch = 0
        while True:
            dataset.set_epoch(epoch)
            for index in range(len(dataset)):
                item = collate([dataset[index]])
                length = item["aatype"].shape[1]
                aatype = item["aatype"][0].long()
                flat = item["x"][0][:, slots, :].reshape(-1, 3).to(device)
                topology = Topology(
                    atom_names=[atom37.ATOM37[i] for i in slots] * length,
                    atom_to_token_idx=[r for r in range(length) for _ in slots],
                    num_tokens=length,
                    res_names=[rc.restype_1to3[atom37.AA_ORDER[int(a)]]
                               for a in aatype for _ in slots])
                # x_noisy must actually be noisy: with x_noisy == target the
                # stub's proposal is already perfect, L_BB is exactly zero at
                # zero-init, and that is a stationary point -- A_SB would receive
                # no gradient and phase 2 would "train" while learning nothing.
                noise = torch.randn(flat.shape, generator=noise_gen).to(device)
                yield CoupledBatch(
                    topology=topology, x_noisy=flat + noise * args.sigma,
                    sigma=torch.tensor([args.sigma], device=device),
                    aatype=aatype.to(device),
                    sidechain_batch={k: v.to(device) for k, v in item.items()
                                     if torch.is_tensor(v)},
                    backbone_target=flat, name=item.get("name", [None])[0])
            epoch += 1

    result = trainer.train(batches(), progress=lambda m: logger.info(m))
    logger.info("done: %s", result)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
