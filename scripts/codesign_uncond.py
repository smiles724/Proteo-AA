#!/usr/bin/env python3
"""Feed PXDesign backbones to FaMPNN; get sequence AND side-chain coordinates.

The co-design branch of the unconditional benchmark: for each generated
backbone, FaMPNN's seq_design path invents a sequence and packs its side chains,
producing a full-atom structure. Those two outputs are what co-designability
scores -- the structure against ESMFold's prediction of the sequence FaMPNN
chose.

Inputs are backbone-only CIFs named ``L<length>_s<index>.cif`` (the convention
from Proteo-AA's unconditional sampler), so a directory of them is a complete
{100 samples} x {each length} draw.

    python scripts/codesign_uncond.py --samples-dir <dir of L*.cif> --out runs/codesign

Outputs, per sample: a full-atom PDB (psCE in the B-factor column), the designed
sequence as FASTA, and a manifest row. Residue numbering is carried over from the
input so the scorer's residue pairing joins correctly.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

logger = logging.getLogger("pxf.codesign")

SAMPLE_RE = re.compile(r"^L(?P<length>\d+)_s(?P<index>\d+)$")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--samples-dir", required=True, help="directory of backbone-only CIFs")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--lengths",
        type=int,
        nargs="*",
        default=None,
        help="restrict to these lengths (default: all present)",
    )
    p.add_argument("--max-per-length", type=int, default=0, help="0 = all")
    p.add_argument("--fampnn-weights", default="0.3", choices=("0.0", "0.3", "0.3-cath"))
    p.add_argument("--fampnn-checkpoint", default=None)
    p.add_argument("--seq-steps", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--psce-threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--adapters",
        default=None,
        help="coupling checkpoint; enables the COUPLED arm. The BB->SC residual "
        "is injected into h_V on every step of FaMPNN's iterative design loop "
        "(see pxf.couple.pack_hook) because model.sample() gives a caller no "
        "feature dict to substitute. Omit for the uncoupled arm",
    )
    p.add_argument(
        "--sigma-b",
        type=float,
        default=None,
        help="backbone noise level the adapter is conditioned on. REQUIRED with "
        "--adapters and deliberately not defaulted: a fully denoised sample has "
        "sigma=0, which is outside the trained window and log-undefined. The "
        "bottom of the training window is 0.010 A; whatever is passed is "
        "recorded in the output JSON",
    )
    p.add_argument(
        "--pxdesign-donor",
        default=None,
        help="PXDesign donor, required with --adapters: the residual is "
        "A_BS(a_token, sigma_B) and a_token comes from one denoiser call",
    )
    p.add_argument("--uncond-crop-size", type=int, default=640)
    p.add_argument("--allow-unpinned-sources", action="store_true")
    return p.parse_args(argv)


def backbone_from_record(record, atom37_order, backbone_slots):
    """Build ``[L, 37, 3]`` coordinates and mask from a flat atom record.

    Residues are ordered by first appearance so the output numbering matches the
    input file rather than being re-sorted.
    """
    names = {name: slot for slot, name in enumerate(atom37_order)}
    wanted = {atom37_order[i] for i in backbone_slots}
    residues, order = {}, []
    for res_id, atom_name, coord in zip(
        record["res_id"], record["atom_name"], record["coord"]
    ):
        if atom_name not in wanted:
            continue
        key = int(res_id)
        if key not in residues:
            residues[key] = {}
            order.append(key)
        residues[key][str(atom_name)] = coord
    length = len(order)
    coords = torch.zeros(length, len(atom37_order), 3, dtype=torch.float32)
    mask = torch.zeros(length, len(atom37_order), dtype=torch.float32)
    for position, key in enumerate(order):
        for atom_name, coord in residues[key].items():
            slot = names[atom_name]
            coords[position, slot] = torch.as_tensor(np.asarray(coord), dtype=torch.float32)
            mask[position, slot] = 1.0
    return coords, mask, torch.tensor(order, dtype=torch.long)



class _Coupling:
    """Supplies the per-sample BB->SC residual for the coupled arm.

    ``delta_h = A_BS(a_token, sigma_B)`` needs PXDesign's token features, so
    each sampled CIF is re-featurized through ``pxdesign_train`` (the bare CIF
    parse this script already does is not enough -- the diffusion module needs
    the featurizer's feature_dict) and the denoiser is evaluated once at the
    requested ``sigma_B``.

    One evaluation per sample, not per decoding step: ``a_token`` and
    ``sigma_B`` are both fixed within a sample, so the residual is constant.
    """

    def __init__(self, *, driver, adapters, sigma_b, datasets, device):
        self.driver = driver
        self.adapters = adapters
        self.sigma_b = float(sigma_b)
        self.datasets = datasets
        self.device = device
        self._cache = {}

    def delta_h_for(self, name, length):
        import torch

        if name in self._cache:
            return self._cache[name]
        entry = self.datasets.get(name)
        if entry is None:
            raise SystemExit(
                f"{name}: no featurized structure; cannot compute a_token for "
                "the coupled arm"
            )
        from pxf.backbone.driver import to_featurized

        _sample_id, dataset = entry
        structure = to_featurized(name, dataset[0]).to(self.device)
        sigma = torch.full((1,), self.sigma_b, device=self.device)
        cond = self.driver.conditioning(structure.feature_dict)
        bound = self.driver.bind(cond)
        with torch.no_grad():
            # x_noisy is the sample itself perturbed to sigma_B: the adapter was
            # trained on (denoised backbone, sigma_B) pairs from the trajectory,
            # so feeding the clean sample at a declared sigma_B is the closest
            # in-distribution query available for a finished structure.
            target = structure.backbone_target.float()
            noise = torch.randn(
                target.shape,
                generator=torch.Generator().manual_seed(hash(name) % (2**31)),
            ).to(self.device)
            x_noisy = (target + noise * self.sigma_b)[None]
            _bb0, a_token = bound(x_noisy, sigma)
        if a_token is None:
            raise SystemExit(f"{name}: driver returned no a_token")
        per_residue = a_token if a_token.dim() == 3 else a_token.reshape(
            -1, length, a_token.shape[-1]
        )
        with torch.no_grad():
            delta = self.adapters.delta_h(per_residue, sigma)
        if delta is None:
            raise SystemExit(
                f"{name}: adapters returned no residual; is enable_bb_to_sc off?"
            )
        self._cache[name] = delta
        return delta


def _build_coupling(args, designer, entries):
    """Load the donor and adapters, and featurize every sampled CIF once."""
    import torch

    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        featurize_structures,
        load_backbone_model,
        to_featurized,
    )
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.fampnn_iface import node_feature_dim

    device = designer.device
    # The adapter's FaMPNN side was fitted to one variant's h_V. Running the
    # coupled arm against a different donor would query the adapter far out of
    # distribution while appearing to work, because the widths can still match.
    state_peek = torch.load(args.adapters, map_location="cpu", weights_only=False)
    frozen_fampnn = (state_peek.get("frozen") or {}).get("fampnn")
    # The trainer records this as the bare variant string ("0.0"); tolerate a
    # dict in case an older checkpoint nested it.
    trained_variant = (
        frozen_fampnn.get("variant")
        if isinstance(frozen_fampnn, dict)
        else frozen_fampnn
    )
    del state_peek
    if trained_variant and str(trained_variant) != str(designer.variant):
        raise SystemExit(
            f"--adapters was trained against FaMPNN {trained_variant} but this "
            f"run uses {designer.variant}; pass --fampnn-weights "
            f"{trained_variant} so the coupled arm queries the adapter in "
            "distribution (and use the same variant for the uncoupled arm)"
        )
    px_model, _cfg, _rec = load_backbone_model(args.pxdesign_donor, device=device)
    driver = PXDesignBackboneDriver(px_model)

    adapters = CouplingAdapters(driver.c_token, node_feature_dim(designer.model)).to(
        device
    )
    state = torch.load(args.adapters, map_location="cpu", weights_only=False)
    if "adapters" not in state:
        raise SystemExit(f"{args.adapters} is not a coupling checkpoint")
    adapters.load_state_dict(state["adapters"])
    if state.get("ema"):
        # Match how eval_couple.py scores the adapter arm.
        from pxf.train.ema import EMA

        settings = state.get("settings") or {}
        ema = EMA(
            adapters,
            decay=settings.get("ema_decay"),
            relative_length=(
                None
                if settings.get("ema_decay") is not None
                else settings.get("ema_relative_length") or 0.25
            ),
        )
        ema.load_state_dict(state["ema"])
        ema.copy_to(adapters)
        logger.info("using the adapter EMA weights (step %s)", state.get("step"))
    adapters.eval().requires_grad_(False)
    adapters.enable_bb_to_sc = True
    adapters.enable_sc_to_bb = False  # phase 1 trains only A_BS

    paths = [str(path) for _, _, path in entries]
    names = [Path(path).stem for path in paths]
    # featurize_structures returns (sample_id, DesignSourceDataset) pairs and
    # the item comes from indexing the dataset. Datasets are held rather than
    # items so the structures are featurized one at a time instead of all 60
    # sitting on the GPU at once.
    datasets = dict(zip(names, featurize_structures(
        paths, crop_size=args.uncond_crop_size
    )))
    logger.info("prepared %d dataset(s) for a_token", len(datasets))
    return _Coupling(
        driver=driver,
        adapters=adapters,
        sigma_b=args.sigma_b,
        datasets=datasets,
        device=device,
    )


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from fampnn.model.sd_model import SeqDenoiser

    from pxf import atom37
    from pxf.device import select_device
    from pxf.eval.canonical import load as load_canonical
    from pxf.sidechain.design import FaMPNNFullAtomDesigner

    samples_dir = Path(args.samples_dir).resolve()
    out = Path(args.out).resolve()
    (out / "samples").mkdir(parents=True, exist_ok=True)

    entries = []
    for path in sorted(samples_dir.glob("L*.cif")):
        match = SAMPLE_RE.match(path.stem)
        if not match:
            continue
        length = int(match.group("length"))
        if args.lengths and length not in args.lengths:
            continue
        entries.append((length, int(match.group("index")), path))
    if not entries:
        raise SystemExit(f"no L<len>_s<idx>.cif samples under {samples_dir}")
    if args.max_per_length:
        kept, seen = [], {}
        for length, index, path in entries:
            if seen.get(length, 0) < args.max_per_length:
                kept.append((length, index, path))
                seen[length] = seen.get(length, 0) + 1
        entries = kept
    if args.shard_count > 1:
        entries = entries[args.shard_index :: args.shard_count]
    per_length = {}
    for length, _, _ in entries:
        per_length[length] = per_length.get(length, 0) + 1
    logger.info(
        "%d sample(s) to co-design: %s",
        len(entries),
        ", ".join(f"L{k}={v}" for k, v in sorted(per_length.items())),
    )

    canonical = load_canonical()  # for load_structure (cif reader)
    designer = FaMPNNFullAtomDesigner(
        args.fampnn_checkpoint,
        variant=args.fampnn_weights,
        seq_steps=args.seq_steps,
        temperature=args.temperature,
        psce_threshold=None if args.psce_threshold < 0 else args.psce_threshold,
        strict_sources=not args.allow_unpinned_sources,
    )
    designer = designer.to(select_device(args.device))
    logger.info(
        "FaMPNN %s (%s) on %s", designer.variant, designer.identity["mode"], designer.device
    )

    coupling = None
    if args.adapters:
        if args.sigma_b is None:
            raise SystemExit(
                "--adapters requires --sigma-b: A_BS is conditioned on log "
                "sigma_B and a denoised sample has sigma=0, which is outside "
                "the trained window and log-undefined. Pass 0.010 (the bottom "
                "of the training window) unless you mean something else"
            )
        if not args.pxdesign_donor:
            raise SystemExit("--adapters requires --pxdesign-donor for a_token")
        coupling = _build_coupling(args, designer, entries)
        logger.info(
            "COUPLED arm: adapters=%s sigma_b=%.4g A", args.adapters, args.sigma_b
        )
    else:
        logger.info("UNCOUPLED arm: no adapter applied")

    backbone_slots = list(atom37.BACKBONE_SLOTS)
    manifest, failures = [], []
    for position, (length, index, path) in enumerate(entries):
        name = path.stem
        try:
            record = canonical.uncond.load_structure(str(path))
            coords, mask, residue_index = backbone_from_record(
                record, atom37.ATOM37, backbone_slots
            )
            if coords.shape[0] != length:
                raise ValueError(
                    f"file holds {coords.shape[0]} residues, name says {length}"
                )
            present = mask[:, backbone_slots].sum(-1)
            if not bool((present == len(backbone_slots)).all()):
                raise ValueError(
                    f"{int((present < len(backbone_slots)).sum())} residue(s) "
                    "lack a complete N/CA/C/O backbone"
                )

            delta_h = None
            hook_stats = {}
            if coupling is not None:
                delta_h = coupling.delta_h_for(name, length)
            # The hook is installed for BOTH arms -- delta_h=None is a
            # traversing no-op -- so the two arms run identical code and any
            # difference between them is the residual, not the call path.
            from pxf.couple.pack_hook import residual_on_sidechain_diffusion

            with residual_on_sidechain_diffusion(
                designer.model, delta_h, counter=hook_stats
            ):
                result = designer.design(
                    coords_af2=coords[None],
                    atom_mask=mask[None],
                    residue_index=residue_index[None],
                    chain_index=torch.zeros(1, length, dtype=torch.long),
                    seed=args.seed + 1000 * index + length,
                )
            if coupling is not None and hook_stats.get("applied", 0) == 0:
                # An inert hook would make this arm a copy of the uncoupled one
                # and the comparison a null result dressed as a finding.
                raise SystemExit(
                    f"{name}: --adapters was given but the residual was never "
                    f"applied ({hook_stats}); the coupled arm would be a "
                    "duplicate of the uncoupled arm"
                )
            design = result["designs"][0]

            pdb_path = out / "samples" / f"{name}.pdb"
            samples = {
                "x_denoised": design.coords_af2.unsqueeze(0).cpu(),
                "seq_mask": torch.ones(1, length),
                "missing_atom_mask": torch.zeros(1, length, atom37.NUM_ATOM37),
                "residue_index": residue_index.unsqueeze(0).long(),
                "chain_index": torch.zeros(1, length, dtype=torch.long),
                "pred_aatype": design.aatype.unsqueeze(0).cpu(),
                "psce": design.psce.unsqueeze(0).cpu(),
            }
            SeqDenoiser.save_samples_to_pdb(samples, [str(pdb_path)])
            fasta_path = out / "samples" / f"{name}.fasta"
            fasta_path.write_text(f">{name}\n{design.sequence}\n")

            manifest.append(
                dict(
                    sample_id=name,
                    length=length,
                    index=index,
                    backbone_cif=str(path),
                    pdb=str(pdb_path.relative_to(out)),
                    fasta=str(fasta_path.relative_to(out)),
                    sequence=design.sequence,
                    mean_psce=float(design.psce.mean()),
                    n_atoms=int(design.atom_mask_af2.sum()),
                    backbone_shift_angstrom=result["backbone_shift"],
                )
            )
            if position % 10 == 0 or position == len(entries) - 1:
                logger.info(
                    "[%d/%d] %s L=%d psce %.3f shift %.4f A",
                    position + 1,
                    len(entries),
                    name,
                    length,
                    manifest[-1]["mean_psce"],
                    manifest[-1]["backbone_shift_angstrom"],
                )
        except Exception as error:  # one bad sample must not end the draw
            logger.warning("%s FAILED: %s: %s", name, type(error).__name__, error)
            failures.append(
                dict(
                    sample_id=name, length=length, error=f"{type(error).__name__}: {error}"
                )
            )

    combined = out / f"codesign_shard{args.shard_index:03d}of{args.shard_count:03d}.fasta"
    with combined.open("w") as stream:
        for row in manifest:
            stream.write(f">{row['sample_id']}\n{row['sequence']}\n")

    record = dict(
        samples_dir=str(samples_dir),
        n_requested=len(entries),
        n_designed=len(manifest),
        n_failed=len(failures),
        per_length=per_length,
        failures=failures,
        combined_fasta=str(combined.relative_to(out)),
        provenance=dict(
            task="pxdesign backbone -> fampnn sequence + side chains",
            sidechain=designer.identity,
            metrics=canonical.record(),
        ),
        arguments=vars(args),
        designs=manifest,
    )
    suffix = f"_shard{args.shard_index:03d}of{args.shard_count:03d}"
    (out / f"codesign{suffix}.json").write_text(json.dumps(record, indent=2, default=str))
    logger.info(
        "co-designed %d/%d sample(s) into %s (%d failed)",
        len(manifest),
        len(entries),
        out,
        len(failures),
    )
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
