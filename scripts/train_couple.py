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
    The real driver: the official PXDesign network loaded from the published
    donor checkpoint, featurized through ``pxdesign_train`` (PXDesign's own
    inference runner designs a binder against a target and cannot denoise a
    monomer). Takes ``.cif`` inputs and requires ``--crop-size`` to be at least
    the structure length, so residue correspondence with the atom37 side-chain
    targets is exact and checkable.

**Backbone noise.** ``sigma_B`` is *sampled* per batch from the interval the
coupling is intended to run in, not held at one value -- both adapters are
conditioned on ``log sigma_B``, so a single training value leaves that
conditioning constant and only licenses deployment at that same value. The
default window is the late end of PXDesign's own 400-step trajectory; see
:mod:`pxf.couple.schedule`. ``--sigma-mode fixed --sigma X`` restores the old
single-value behaviour for a deployment-matched ablation.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from pxf.couple import schedule

logger = logging.getLogger("pxf.train_couple")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="configs/couple_phase{1,2,3}.yaml")
    p.add_argument(
        "--structures",
        required=True,
        help="directory of PDBs, or a text file of paths (no default)",
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--backbone",
        required=True,
        choices=("stub", "pxdesign"),
        help="which backbone driver to couple to",
    )
    p.add_argument("--phase", default=None, help="override the config's phase")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument(
        "--sigma-mode",
        default=None,
        choices=schedule.MODES,
        help=(
            "how sigma_B is drawn: trajectory (uniform over the backbone "
            "sampler's own steps inside the window, the default), loguniform, "
            "or fixed (one value; noise conditioning is then constant)"
        ),
    )
    p.add_argument(
        "--sigma-min",
        type=float,
        default=None,
        help=f"low edge of the coupling window, Angstroms (default {schedule.DEFAULT_SIGMA_MIN})",
    )
    p.add_argument(
        "--sigma-max",
        type=float,
        default=None,
        help=f"high edge of the coupling window, Angstroms (default {schedule.DEFAULT_SIGMA_MAX})",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="the single noise level used by --sigma-mode fixed",
    )
    p.add_argument(
        "--sigma-n-step",
        type=int,
        default=None,
        help=(
            "steps in the backbone sampler's schedule, which defines the discrete "
            f"sigma values trajectory mode draws from (default {schedule.PXDESIGN_N_STEP})"
        ),
    )
    p.add_argument(
        "--pxdesign-donor",
        default=None,
        help="PXDesign donor checkpoint (required for --backbone pxdesign)",
    )
    p.add_argument(
        "--proteoaa-root",
        default=None,
        help="Proteo-AA checkout supplying pxdesign_train (default: searched)",
    )
    p.add_argument("--fampnn-weights", default="0.0", choices=("0.0", "0.3", "0.3-cath"))
    p.add_argument("--fampnn-checkpoint", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--pack-steps", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--allow-unpinned-sources", action="store_true")
    return p.parse_args(argv)


def resolve_structures(spec, *, suffix=".pdb"):
    """A directory of structures or a file listing paths. No implicit default."""
    path = Path(spec)
    if path.is_dir():
        found = sorted(str(p) for p in path.glob(f"*{suffix}"))
        if not found:
            raise SystemExit(f"--structures {path} contains no *{suffix} files")
        return found
    if path.is_file():
        found = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        missing = [p for p in found if not Path(p).is_file()]
        if missing:
            raise SystemExit(
                f"{len(missing)} listed path(s) do not exist, e.g. {missing[:3]}"
            )
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
    projection = torch.randn(9, channels, generator=generator) * (channels**-0.5)

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


def _native_atom37(structures, sample_id, structure):
    """Side-chain targets in atom37, from the same file the backbone came from.

    L_SC needs ground-truth side chains, which the design-region featurization
    scrubs. They are read back from the original structure and checked against the
    featurized crop by sequence equality, so a mismatch is an error rather than a
    silently misaligned target.
    """
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    from pxf import atom37 as _atom37

    match = next((p for p in structures if Path(p).stem == sample_id), None)
    if match is None:
        raise ValueError(f"no source file for {sample_id}")
    single = process_single_pdb(load_feats_from_pdb(str(match)))
    length = single["aatype"].shape[0]
    if length != structure.num_tokens:
        raise ValueError(
            f"{sample_id}: featurized crop has {structure.num_tokens} residues but the "
            f"file has {length}. Coupling training needs --crop-size >= the structure "
            "length, because a crop breaks correspondence with the side-chain targets."
        )
    supplied = _atom37.sequence_from_aatype(structure.aatype)
    native = _atom37.sequence_from_aatype(single["aatype"].long())
    if supplied != native:
        raise ValueError(
            f"{sample_id}: featurized sequence differs from the file's; "
            "side-chain targets would be misaligned"
        )
    device = structure.aatype.device
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in single.items()
        if torch.is_tensor(value)
        and key
        in ("x", "aatype", "seq_mask", "missing_atom_mask", "residue_index", "chain_index")
    }


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import yaml
    from fampnn.model.sd_model import SeqDenoiser

    from fampnn.data import residue_constants as rc
    from pxf import atom37, provenance
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.controller import CoupledDenoiser, Topology
    from pxf.couple.trainer import CoupledBatch, CoupledTrainer, CoupleSettings
    from pxf.device import select_device
    from pxf.train.data import StructureCropDataset, collate
    from pxf.train.trainer import OptimSettings

    if args.backbone == "pxdesign" and not args.pxdesign_donor:
        raise SystemExit(
            "--backbone pxdesign requires --pxdesign-donor "
            "(the published pxdesign_v0.1.0.pt)"
        )

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    couple_cfg = dict(config.get("couple", {}))
    optim_cfg = dict(config.get("optim", {}))
    adapter_cfg = dict(config.get("adapters", {}))
    for key, value in (
        ("phase", args.phase),
        ("max_steps", args.max_steps),
        ("seed", args.seed),
        ("pack_steps", args.pack_steps),
    ):
        if value is not None:
            couple_cfg[key] = value
    if args.lr is not None:
        optim_cfg["lr"] = args.lr
    # The sigma_B distribution: config block, overridden by any flag that is set.
    sigma_schedule = schedule.from_config(
        config.get("sigma"),
        mode=args.sigma_mode,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma=args.sigma,
        n_step=args.sigma_n_step,
    )

    suffix = ".cif" if args.backbone == "pxdesign" else ".pdb"
    structures = resolve_structures(args.structures, suffix=suffix)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger.info("%d structure(s); phase %s", len(structures), couple_cfg.get("phase"))

    checkpoint = (
        Path(args.fampnn_checkpoint)
        if args.fampnn_checkpoint
        else provenance.fampnn_checkpoint(args.fampnn_weights)
    )
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fampnn = SeqDenoiser(bundle["model_cfg"])
    fampnn.load_state_dict(bundle["state_dict"], strict=True)
    fampnn.eval()
    fampnn.requires_grad_(False)  # donors frozen: the whole point
    device = select_device(args.device)
    fampnn.to(device)

    c_h_V = int(fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V)

    if args.backbone == "pxdesign":
        from pxf.backbone.driver import (
            PXDesignBackboneDriver,
            featurize_structures,
            load_backbone_model,
            to_featurized,
        )

        px_model, _, px_record = load_backbone_model(
            args.pxdesign_donor, device=device, proteoaa_root=args.proteoaa_root
        )
        px_driver = PXDesignBackboneDriver(px_model)
        c_token = px_driver.c_token
        sigma_data = px_driver.sigma_data
        logger.info(
            "PXDesign donor loaded: c_token=%d sigma_data=%.1f", c_token, sigma_data
        )
        featurized = featurize_structures(
            structures, crop_size=args.crop_size, proteoaa_root=args.proteoaa_root
        )
    else:
        c_token = 384
        sigma_data = 16.0
        px_driver = None
        dataset = StructureCropDataset(
            structures,
            crop_size=args.crop_size,
            noise=0.0,
            seed=int(couple_cfg.get("seed", 0)),
        )

    couple_cfg.setdefault("sigma_data_backbone", sigma_data)
    # The schedule's sigma_data must be the donor's, or the discrete sigma values
    # trajectory mode draws from are not the ones the sampler visits.
    if args.backbone == "pxdesign" and sigma_schedule.sigma_data != sigma_data:
        sigma_schedule = replace(sigma_schedule, sigma_data=sigma_data)
    couple_cfg["sigma_schedule"] = sigma_schedule.identity()
    logger.info("%s", sigma_schedule.describe())
    adapters = CouplingAdapters(c_token, c_h_V, **adapter_cfg).to(device)
    if px_driver is None:
        backbone = make_stub_backbone(args.crop_size, c_token)
    else:
        # The real driver is bound per target, because conditioning is per
        # structure. Until the first batch binds it, calling it is a bug.
        def backbone(*_args, **_kwargs):
            raise RuntimeError(
                "the PXDesign driver is bound per batch; "
                "the controller was called before any batch arrived"
            )

    controller = CoupledDenoiser(
        backbone,
        fampnn,
        adapters,
        phase=couple_cfg.get("phase", "bb_to_sc"),
        pack_steps=couple_cfg.get("pack_steps"),
    )
    frozen = dict(
        fampnn=provenance.weight_record(checkpoint, variant=args.fampnn_weights),
        upstream=provenance.runtime_sources(
            strict=not args.allow_unpinned_sources, components=("fampnn",)
        ),
        backbone_driver=args.backbone,
    )
    if px_driver is not None:
        frozen["pxdesign"] = px_record
    trainer = CoupledTrainer(
        controller,
        out_dir=out,
        optim=OptimSettings(**optim_cfg),
        settings=CoupleSettings(**couple_cfg),
        device=device,
        frozen_identity=frozen,
    )
    if args.resume:
        logger.info("resumed at step %d", trainer.resume(args.resume))

    (out / "run_config.json").write_text(
        json.dumps(
            dict(
                couple=couple_cfg,
                optim=optim_cfg,
                adapters=adapter_cfg,
                n_structures=len(structures),
                sigma_schedule=sigma_schedule.identity(),
                backbone=args.backbone,
                device=str(device),
                frozen=frozen,
                arguments=vars(args),
            ),
            indent=2,
            default=str,
        )
    )

    slots = list(atom37.BACKBONE_SLOTS)

    def pxdesign_batches():
        """Real PXDesign proposals, with atom37 side-chain targets from the same file.

        The backbone target and x_noisy live on PXDesign's flat atom axis; the
        side-chain target is the native structure in atom37. Correspondence is
        asserted by sequence equality rather than assumed, and cropping is refused
        because a crop would break it.
        """
        noise_gen = torch.Generator().manual_seed(int(couple_cfg.get("seed", 0)) + 1)
        epoch = 0
        while True:
            for sample_id, source in featurized:
                # The featurizer emits CPU tensors; the model is on `device`.
                structure = to_featurized(sample_id, source[0]).to(device)
                native = _native_atom37(structures, sample_id, structure)
                target = structure.backbone_target.float()
                # A fresh sigma_B per example, from the coupling window. The
                # generator is a CPU one, so draw on the CPU and then move.
                sigma = sigma_schedule.sample(1, generator=noise_gen).to(device)
                noise = torch.randn(target.shape, generator=noise_gen).to(device)
                # One denoiser evaluation per target needs its own conditioning.
                controller.backbone = px_driver.bind(
                    px_driver.conditioning(structure.feature_dict)
                )
                yield CoupledBatch(
                    topology=structure.topology,
                    x_noisy=(target + noise * sigma)[None],
                    sigma=sigma,
                    aatype=structure.aatype.to(device),
                    sidechain_batch=native,
                    backbone_target=target[None],
                    name=sample_id,
                )
            epoch += 1

    def stub_batches():
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
                    # Two-sided bound: aatype 20 is X/UNK and AA_ORDER holds
                    # only the canonical twenty, so an unresolved residue would
                    # index past the end of the string.
                    res_names=[
                        rc.restype_1to3[atom37.AA_ORDER[int(a)]]
                        if 0 <= int(a) < 20
                        else "UNK"
                        for a in aatype
                        for _ in slots
                    ],
                )
                sigma = sigma_schedule.sample(1, generator=noise_gen).to(device)
                noise = torch.randn(flat.shape, generator=noise_gen).to(device)
                yield CoupledBatch(
                    topology=topology,
                    x_noisy=flat + noise * sigma,
                    sigma=sigma,
                    aatype=aatype.to(device),
                    sidechain_batch={
                        k: v.to(device) for k, v in item.items() if torch.is_tensor(v)
                    },
                    backbone_target=flat,
                    name=item.get("name", [None])[0],
                )
            epoch += 1

    batches = pxdesign_batches if args.backbone == "pxdesign" else stub_batches

    result = trainer.train(batches(), progress=lambda m: logger.info(m))
    logger.info("done: %s", result)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
