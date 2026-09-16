#!/usr/bin/env python3
"""Unconditional PXDesign backbone sampling, optionally through the coupled cycle.

The trajectory is Protenix's published sampler -- ``generator.sample_diffusion``,
AF3 Algorithm 18, with its centre-random-augmentation, churn (``gamma0`` /
``gamma_min``), ``noise_scale_lambda`` and ``step_scale_eta`` -- driven through
its injectable ``denoise_net``. Nothing here re-derives Euler or Heun: a
hand-rolled loop drifts from PXDesign in ways that are invisible in the output.

    python scripts/sample_uncond.py --out runs/uncond --lengths 100 200 300 400 500 \\
        --num-samples 100 --pxdesign-donor .../pxdesign_v0.1.0.pt

Sample naming is ``L<length>_s<index>.cif`` because the downstream stages join
on it (``parse_sample_id``); do not change it.

THE SIGMA WINDOW. The adapters are conditioned on log sigma_B and were trained
only on trajectory steps 281-395 (sigma 0.010-4.881 A). The full schedule starts
at sigma = 2560 A, so ~70% of the steps are outside the trained window.
``--adapter-window gated`` (the default) runs the bare driver above
``sigma_max`` and engages the coupled cycle below it, with the cut taken from
``schedule.window_steps()`` rather than a literal. ``always`` is the ablation;
``off`` is the baseline arm.

PHASE 1 IS BACKBONE-INVARIANT. ``A_BS`` only modifies FaMPNN's ``h_V``; the
backbone is reached solely through ``A_SB`` (``delta_a`` -> the driver's
``feedback=`` port). In a phase-1 checkpoint ``A_SB`` is still zero-initialised,
so ``delta_a`` is exactly zero and zero feedback is a verified no-op
(tests/test_backbone_driver.py). This script therefore short-circuits to the
bare driver whenever ``enable_sc_to_bb`` is off, which is provably identical and
400x cheaper than running a full packing cycle per denoiser call. The gate is
still built, because it is what makes a phase-2/3 run defensible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from pxf.couple import schedule

logger = logging.getLogger("pxf.sample_uncond")

DEFAULT_LENGTHS = (100, 200, 300, 400, 500)
ADAPTER_WINDOWS = ("gated", "always", "off")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True, help="directory for the sampled CIFs")
    p.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    p.add_argument("--num-samples", type=int, default=100, help="samples per length")
    p.add_argument(
        "--crop-size",
        type=int,
        default=640,
        help="must be >= max(lengths); uncond_design_dataset refuses otherwise. "
        "Lowering it does NOT reduce memory -- the featurizer emits the "
        "structure's true token count, so cost follows length",
    )
    p.add_argument("--conformation", default="helix")
    p.add_argument("--n-step", type=int, default=schedule.PXDESIGN_N_STEP)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pxdesign-donor", required=True)
    p.add_argument("--proteoaa-root", default=None)
    p.add_argument("--adapters", default=None, help="coupling checkpoint (optional)")
    p.add_argument(
        "--adapter-window", default="gated", choices=ADAPTER_WINDOWS,
    )
    p.add_argument("--sigma-max", type=float, default=schedule.DEFAULT_SIGMA_MAX)
    p.add_argument("--sigma-min", type=float, default=schedule.DEFAULT_SIGMA_MIN)
    # Algorithm 18 knobs, defaulted to Protenix's own values.
    p.add_argument("--gamma0", type=float, default=0.8)
    p.add_argument("--gamma-min", type=float, default=1.0)
    p.add_argument("--noise-scale-lambda", type=float, default=1.003)
    p.add_argument("--step-scale-eta", type=float, default=1.5)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-unpinned-sources", action="store_true")
    return p.parse_args(argv)


def sample_tasks(lengths, num_samples, *, shard_index=0, num_shards=1):
    """``(length, index)`` pairs for this shard, strided so each shard spans lengths.

    Striding rather than blocking matters: a partial run then still covers all
    five lengths instead of finishing only the short ones.
    """
    tasks = [(int(L), int(i)) for L in lengths for i in range(int(num_samples))]
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"bad shard {shard_index}/{num_shards}")
    return tasks[shard_index::num_shards]


def sample_id(length, index):
    """The downstream naming contract. Stages B-G parse this."""
    return f"L{int(length)}_s{int(index)}"


def build_denoise_net(px_driver, conditioning, *, controller=None, window=None):
    """A ``denoise_net`` for ``sample_diffusion``.

    ``sample_diffusion`` calls it with the full conditioning as keywords and
    expects the x0 prediction back as a bare tensor. The driver is already
    closed over its own conditioning, so the forwarded copies are ignored
    rather than re-threaded.
    """
    bound = px_driver.bind(conditioning)
    engaged = {"steps": 0, "total": 0}

    def denoise_net(x_noisy, t_hat_noise_level, **_ignored):
        engaged["total"] += 1
        sigma = t_hat_noise_level
        use_cycle = controller is not None
        if use_cycle and window is not None:
            peak = float(torch.as_tensor(sigma).reshape(-1).max())
            use_cycle = peak <= window
        if use_cycle:
            engaged["steps"] += 1
            # Only reachable with A_SB active; see the module docstring.
            out = controller.forward(
                controller.topology, x_noisy, sigma, controller.aatype,
                run_feedback=True,
            )
            return out.bb1_flat if out.bb1_flat is not None else out.bb0_flat
        x_denoised, _a_token = bound(x_noisy, sigma)
        return x_denoised

    denoise_net.stats = engaged
    return denoise_net


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(name)s: %(message)s",
    )

    from protenix.model.generator import sample_diffusion

    from pxf import provenance
    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        load_backbone_model,
        to_featurized,
    )
    from pxf.device import select_device
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if not args.allow_unpinned_sources:
        sources = provenance.runtime_sources()
    else:
        sources = {"unpinned": True}

    if args.crop_size < max(args.lengths):
        raise SystemExit(
            f"--crop-size {args.crop_size} < max length {max(args.lengths)}; "
            "the samples would be silently cropped below the requested length"
        )

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    # The published 400-step schedule, descending 2560 -> 0.0064 A.
    sigmas = schedule.karras_sigmas(args.n_step).to(device=device, dtype=torch.float32)

    tasks = sample_tasks(
        args.lengths, args.num_samples,
        shard_index=args.shard_index, num_shards=args.num_shards,
    )
    logger.info(
        "%d task(s) on shard %d/%d; lengths %s; %d steps; sigma %.4g -> %.4g",
        len(tasks), args.shard_index, args.num_shards, sorted(set(args.lengths)),
        args.n_step, float(sigmas[0]), float(sigmas[-1]),
    )

    px_model, _configs, _record = load_backbone_model(
        args.pxdesign_donor, device=device, proteoaa_root=args.proteoaa_root
    )
    px_driver = PXDesignBackboneDriver(px_model)

    # pxdesign_train only becomes importable once pxf.backbone.proteoaa has
    # inserted PROTEOAA_ROOT on sys.path, which load_backbone_model does above.
    from pxdesign_train.runner.length_provider import uncond_design_dataset

    dataset = uncond_design_dataset(
        sorted(set(args.lengths)),
        crop_size=args.crop_size,
        conformation=args.conformation,
        seed=args.seed,
        compute_sidechain=True,
    )
    index_of_length = {int(L): i for i, L in enumerate(dataset.provider.lengths)}

    controller = None
    window = None
    if args.adapters:
        controller, window = _load_controller(args, px_driver, device)

    written, skipped = [], []
    started = time.time()
    for n, (length, index) in enumerate(tasks):
        name = sample_id(length, index)
        path = out / f"{name}.cif"
        if path.exists() and not args.overwrite:
            skipped.append(dict(sample=name, reason="exists"))
            continue

        item = dataset[index_of_length[length]]
        # A silent mismatch here produces samples at the wrong length.
        got = int(item["input_feature_dict"]["design_token_mask"].sum())
        if got != length:
            raise SystemExit(
                f"{name}: design_token_mask.sum()={got} but length={length}; "
                "the provider handed back the wrong monomer"
            )
        structure = to_featurized(name, item).to(device)
        conditioning = px_driver.conditioning(structure.feature_dict)
        if controller is not None:
            controller.topology = structure.topology
            controller.aatype = structure.aatype

        denoise_net = build_denoise_net(
            px_driver, conditioning, controller=controller, window=window
        )
        # Per-sample seed so a shard is reproducible independently of its slice.
        torch.manual_seed(abs(hash((args.seed, name))) % (2**31))
        with torch.no_grad():
            x0 = sample_diffusion(
                denoise_net=denoise_net,
                input_feature_dict=conditioning.input_feature_dict,
                s_inputs=conditioning.s_inputs,
                s_trunk=conditioning.s_trunk,
                z_trunk=conditioning.z_trunk,
                pair_z=None,
                p_lm=None,
                c_l=None,
                noise_schedule=sigmas,
                N_sample=1,
                gamma0=args.gamma0,
                gamma_min=args.gamma_min,
                noise_scale_lambda=args.noise_scale_lambda,
                step_scale_eta=args.step_scale_eta,
            )
        _write_sample(x0, structure, path, px_driver)
        written.append(name)
        if n % 5 == 0 or n == len(tasks) - 1:
            logger.info(
                "%d/%d written, %.1fs elapsed (adapter engaged %d/%d steps)",
                len(written), len(tasks), time.time() - started,
                denoise_net.stats["steps"], denoise_net.stats["total"],
            )

    manifest = out / f"samples_shard{args.shard_index}of{args.num_shards}.json"
    manifest.write_text(
        json.dumps(
            dict(
                written=written, skipped=skipped,
                lengths=sorted(set(args.lengths)), num_samples=args.num_samples,
                n_step=args.n_step, seed=args.seed,
                adapter_window=args.adapter_window,
                adapters=args.adapters,
                sigma_window=[args.sigma_min, args.sigma_max],
                sampler="protenix.generator.sample_diffusion",
                algorithm18=dict(
                    gamma0=args.gamma0, gamma_min=args.gamma_min,
                    noise_scale_lambda=args.noise_scale_lambda,
                    step_scale_eta=args.step_scale_eta,
                ),
                schedule_endpoints=[float(sigmas[0]), float(sigmas[-1])],
                provenance=sources,
            ),
            indent=2, default=str,
        )
        + "\n"
    )
    logger.info("wrote %d sample(s) -> %s", len(written), out)
    return 0


def _load_controller(args, px_driver, device):
    """Build the coupled cycle and resolve the sigma gate."""
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.controller import CoupledDenoiser
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.provenance import fampnn_checkpoint

    from fampnn.model.sd_model import SeqDenoiser

    bundle = torch.load(fampnn_checkpoint(), map_location="cpu", weights_only=False)
    fampnn = SeqDenoiser(bundle["model_cfg"])
    fampnn.load_state_dict(bundle["state_dict"], strict=True)
    fampnn = fampnn.to(device).eval().requires_grad_(False)

    adapters = CouplingAdapters(px_driver.c_token, node_feature_dim(fampnn)).to(device)
    state = torch.load(args.adapters, map_location="cpu", weights_only=False)
    if "adapters" not in state:
        raise SystemExit(f"{args.adapters} is not a coupling checkpoint")
    adapters.load_state_dict(state["adapters"])
    adapters.eval().requires_grad_(False)

    if not getattr(adapters, "enable_sc_to_bb", False):
        # Phase 1: A_SB is zero, so the cycle cannot move the backbone. Running
        # it per denoiser call would cost 400 packings per sample for a provably
        # identical result -- see the module docstring.
        logger.info(
            "adapters loaded but SC->BB is inactive (phase 1): the backbone is "
            "invariant, so sampling runs the bare driver"
        )
        return None, None

    if args.adapter_window == "off":
        return None, None
    if args.adapter_window == "always":
        return CoupledDenoiser(
            backbone=None, fampnn=fampnn, adapters=adapters
        ), float("inf")

    sched = schedule.CouplingNoiseSchedule(
        mode="trajectory", sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        n_step=args.n_step, sigma_data=px_driver.sigma_data,
    )
    first, last = sched.window_steps()
    logger.info(
        "adapter gate: engaged for sigma <= %.4g A (trajectory steps %s-%s of %d)",
        args.sigma_max, first, last, args.n_step,
    )
    return CoupledDenoiser(
        backbone=None, fampnn=fampnn, adapters=adapters
    ), float(args.sigma_max)


def _write_sample(x0, structure, path, px_driver):
    """Densify PXDesign's flat atom output to atom37 and emit the mmCIF."""
    from pxf.couple.converter import PXFaRepresentationConverter
    from pxf.train.afdb import AFDBRecord, write_cif

    converter = PXFaRepresentationConverter()
    topology = structure.topology
    inputs = converter.px_backbone_to_fampnn(
        x0.reshape(-1, x0.shape[-2], 3)[0],
        topology.atom_names,
        topology.atom_to_token_idx,
        topology.num_tokens,
        res_names=topology.res_names,
        residue_index=topology.residue_index,
        chain_index=topology.chain_index,
        aatype=structure.aatype,
    )
    length = topology.num_tokens
    coords = inputs.coords_af2[0].detach().cpu()
    from pxf import atom37 as atom37_module

    mask = torch.zeros(length, atom37_module.NUM_ATOM37)
    mask[:, list(atom37_module.BACKBONE_SLOTS)] = 1.0
    record = AFDBRecord(
        afid=structure.sample_id,
        aatype=structure.aatype.detach().cpu().reshape(-1).long(),
        x=coords,
        atom_mask=mask,
        residue_index=torch.arange(1, length + 1, dtype=torch.long),
        plddt=torch.zeros(length),
    )
    write_cif(record, path)


if __name__ == "__main__":
    raise SystemExit(main())
