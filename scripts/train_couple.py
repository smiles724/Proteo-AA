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


def readout_variants():
    """The SC->BB arms, imported lazily so ``--help`` does not load torch."""
    from pxf.couple.readout import VARIANTS

    return set(VARIANTS)


def conditioner_arms():
    """The named (architecture, variant) rows of E1 and E2."""
    from pxf.couple.conditioning import ARMS

    return set(ARMS)


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
    p.add_argument(
        "--resume",
        default=None,
        help="continue THIS experiment: adapters, optimizer, step, EMA and RNG. "
        "Not for starting a new phase from a previous one -- use "
        "--init-from for that",
    )
    p.add_argument(
        "--init-from",
        "--init-from-phase1",
        dest="init_from",
        default=None,
        help="weights-only initialization from another phase's checkpoint: "
        "inherit whichever adapter direction that phase trained, start this "
        "phase's direction fresh, and reset the step counter, the optimizer and "
        "the EMA. This is the correct way to chain phases; --resume is not, and "
        "used to be what the launcher recommended",
    )
    p.add_argument(
        "--sb-variant",
        default=None,
        choices=sorted(readout_variants()),
        help="which SC->BB arm to train. full: the candidate. bb_only: the "
        "trained BB/sequence-only control, reading the side-chain-masked "
        "encoding with every SC-derived group zeroed. generic: the trained "
        "sigma-only control, z identically zero. All three have the same "
        "parameter count, so they differ in information and not in capacity",
    )
    p.add_argument(
        "--sb-arm",
        default=None,
        choices=sorted(conditioner_arms()),
        help="which named SC->BB arm to train, selecting the architecture and "
        "the variant together. late_*: the existing decoder-input adapter. "
        "early_s_*: the same readout injected into s_single instead (E1). "
        "atom_*: predicted atoms encoded into s_single and z_pair (E2). "
        "Mutually exclusive with --sb-variant, which names a variant of the "
        "late architecture only",
    )
    p.add_argument(
        "--bs-policy",
        default=None,
        choices=("bypass", "matched"),
        help="the Phase-1 BB->SC policy, held fixed across every SC->BB arm",
    )
    p.add_argument(
        "--bs-gate",
        default=None,
        choices=("off", "A", "B", "C", "one"),
        help="sigma gate on the BB->SC residual; only read with --bs-policy matched",
    )
    p.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="how many (structure, sigma) examples the pilot trains on. A FIXED "
        "pool with known backbone targets, precomputed once, so the frozen half "
        "is paid for per example rather than per step",
    )
    p.add_argument(
        "--sigmas-per-structure",
        type=int,
        default=4,
        help="sigma draws per structure when building the pilot pool",
    )
    p.add_argument(
        "--val-structures",
        default=None,
        help="held-out structures for the in-loop bb0-vs-bb1 score, reported at "
        "initialization and at every couple.eval_steps",
    )
    p.add_argument("--val-pool-size", type=int, default=32)
    p.add_argument(
        "--cache-upstream",
        default=None,
        help="path to save/load the frozen upstream states. Refuses to load a "
        "cache built from different donors, packing length or BB->SC policy",
    )
    p.add_argument(
        "--conditioning-cache",
        type=int,
        default=8,
        help="how many structures' PXDesign conditioning to keep in memory. The "
        "corrective call needs it every step and it does not depend on sigma, so "
        "caching it is the difference between recomputing the trunk per step and "
        "per structure",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--train-fampnn",
        action="store_true",
        help="CONTROL ARM: train FaMPNN's own weights instead of the adapters, "
        "with no adapter applied. Same data, same steps, same objective and "
        "same optimiser settings as the adapter run, so the two are comparable "
        "and any difference is attributable to the mechanism rather than to the "
        "data. The checkpoint is written in --fampnn-checkpoint's own format",
    )
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
        # '#' lines are skipped so a manifest can record how it was produced --
        # which is what makes a run's data source reproducible rather than a
        # glob that happened to be evaluated at submit time.
        found = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
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


def _masked_rmsd(predicted, reference, keep):
    """RMSD over the supervised atoms, **un-superposed**.

    Matching the loss on purpose: a denoiser predicts in the frame it was given,
    so aligning first would hide exactly the error the correction is supposed to
    remove. The superposed numbers are the evaluator's job
    (:mod:`pxf.eval.backbone_metrics`), where they are the right quantity.
    """
    delta = predicted.reshape(-1, 3)[keep] - reference
    return float(delta.pow(2).sum(-1).mean().sqrt())


class ConditioningCache:
    """PXDesign conditioning per structure, least-recently-used.

    The corrective call needs ``(s_inputs, s_trunk, z_trunk)`` every step, and
    they are a function of the structure alone -- not of sigma, not of the
    feedback. Recomputing the trunk 2,000 times for a pool of a few dozen
    structures is the single largest avoidable cost in the pilot.

    ``z_trunk`` is the large one (``[1, L, L, c_z]``, ~33 MB at L = 256), so the
    cache is bounded and evicts rather than growing.
    """

    def __init__(self, driver, *, capacity=8):
        self.driver = driver
        self.capacity = max(0, int(capacity))
        self._items = {}
        self._order = []
        self.hits = self.misses = 0

    def get(self, sample_id, feature_dict):
        if self.capacity and sample_id in self._items:
            self.hits += 1
            self._order.remove(sample_id)
            self._order.append(sample_id)
            return self._items[sample_id]
        self.misses += 1
        conditioning = self.driver.conditioning(feature_dict)
        if self.capacity:
            self._items[sample_id] = conditioning
            self._order.append(sample_id)
            while len(self._order) > self.capacity:
                del self._items[self._order.pop(0)]
        return conditioning

    def stats(self):
        total = self.hits + self.misses
        return dict(
            capacity=self.capacity,
            entries=len(self._items),
            hits=self.hits,
            misses=self.misses,
            hit_rate=(self.hits / total if total else 0.0),
        )


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
    from pxf.couple import pilot
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
    sb_cfg = dict(config.get("sb_feedback", {}))
    for key, value in (
        ("phase", args.phase),
        ("max_steps", args.max_steps),
        ("seed", args.seed),
        ("pack_steps", args.pack_steps),
        ("bs_policy", args.bs_policy),
    ):
        if value is not None:
            couple_cfg[key] = value
    if args.sb_variant is not None and args.sb_arm is not None:
        raise SystemExit(
            "--sb-arm names an architecture and a variant together; --sb-variant "
            "names a variant of the late architecture only. Passing both leaves "
            "it ambiguous which one the run is"
        )
    if args.sb_variant is not None:
        sb_cfg["variant"] = args.sb_variant
    if args.sb_arm is not None:
        sb_cfg["arm"] = args.sb_arm
    run_pilot = bool(couple_cfg.get("corrective_event"))
    if run_pilot and args.backbone != "pxdesign":
        raise SystemExit(
            "the SC->BB pilot needs --backbone pxdesign: the whole question is "
            "whether the correction reaches the real decoder, and the stub's "
            "feedback path is a scalar shim that cannot answer it"
        )
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
    if getattr(args, "train_fampnn", False):
        # The control arm trains the donor. Keep it in train() so dropout and
        # any norm statistics behave as they would in a real fine-tune.
        fampnn.train()
        fampnn.requires_grad_(True)
    else:
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
        px_record = dict(px_record, driver_settings=px_driver.identity())
        logger.info(
            "PXDesign donor loaded: %s", json.dumps(px_driver.identity(), default=str)
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

    # The donor's sigma_data wins over the config's. `setdefault` let a stale
    # config value stand: the EDM weight is 1/c_out(sigma_data)^2, so a
    # mismatch silently reweights every noise level. They agree today (both
    # 16.0), which is exactly when a precedence bug is invisible.
    configured = couple_cfg.get("sigma_data_backbone")
    if configured is not None and abs(float(configured) - float(sigma_data)) > 1e-9:
        logger.warning(
            "config sigma_data_backbone=%s but the loaded donor reports %s; "
            "using the donor's, since the EDM loss weight is defined by it",
            configured,
            sigma_data,
        )
    couple_cfg["sigma_data_backbone"] = sigma_data
    # The schedule's sigma_data must be the donor's, or the discrete sigma values
    # trajectory mode draws from are not the ones the sampler visits.
    if args.backbone == "pxdesign" and sigma_schedule.sigma_data != sigma_data:
        sigma_schedule = replace(sigma_schedule, sigma_data=sigma_data)
    couple_cfg["sigma_schedule"] = sigma_schedule.identity()
    logger.info("%s", sigma_schedule.describe())
    sb_module = None
    if sb_cfg:
        from pxf.couple.conditioning import ARMS, build_conditioner
        from pxf.couple.readout import FeedbackPath, SigmaWindow

        gate_cfg = dict(sb_cfg.get("gate") or {})
        gate = SigmaWindow(**gate_cfg) if gate_cfg else None
        arm = sb_cfg.get("arm")
        if arm is None:
            # The legacy path, unchanged: a variant name alone means the late
            # decoder-input adapter. Existing configs and checkpoints keep
            # producing exactly the module they produced before.
            sb_module = FeedbackPath(
                c_h_V,
                c_token,
                d_hidden=int(adapter_cfg.get("d_hidden", 256)),
                d_noise=int(adapter_cfg.get("d_noise", 64)),
                variant=sb_cfg.get("variant", "full"),
                gate=gate,
            )
        else:
            if ARMS[arm]["arch"] != "late" and px_driver is None:
                raise SystemExit(
                    f"arm {arm!r} injects into the conditioning output, which "
                    "only the real PXDesign driver has. The stub backbone has no "
                    "DiffusionConditioning to hook, so this run would train "
                    "against a shim rather than the model"
                )
            sb_module = build_conditioner(
                arm,
                c_h_V=c_h_V,
                c_token=c_token,
                c_s=None if px_driver is None else px_driver.c_s,
                c_z=None if px_driver is None else px_driver.c_z,
                gate=gate,
                d_hidden=int(adapter_cfg.get("d_hidden", 256)),
                d_noise=int(adapter_cfg.get("d_noise", 64)),
            )
        sb_cfg["arm"] = arm
        logger.info("SC->BB arm: %s", json.dumps(sb_module.identity(), default=str))
    adapters = CouplingAdapters(c_token, c_h_V, sc_to_bb=sb_module, **adapter_cfg).to(
        device
    )
    if args.train_fampnn:
        # No adapter may train or be applied: the control has to isolate the
        # fine-tune. CoupledTrainer._assert_adapters_are_inert re-checks both.
        adapters.requires_grad_(False)
        adapters.enable_bb_to_sc = False
        adapters.enable_sc_to_bb = False
        logger.info("CONTROL ARM: training FaMPNN weights, adapters frozen and not applied")
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

    bs_gate = None
    if args.bs_gate and args.bs_gate != "off":
        from pxf.couple import bs_policy

        if couple_cfg.get("bs_policy", "bypass") != "matched":
            raise SystemExit(
                f"--bs-gate {args.bs_gate} is only read with --bs-policy matched; "
                "the bypass applies no BB->SC residual for a gate to scale"
            )
        bs_gate = bs_policy.gate_by_name(args.bs_gate)
    controller = CoupledDenoiser(
        backbone,
        fampnn,
        adapters,
        phase=couple_cfg.get("phase", "bb_to_sc"),
        pack_steps=couple_cfg.get("pack_steps"),
        bs_gate=bs_gate,
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
        fampnn_finetune=bool(args.train_fampnn),
        fampnn_model_cfg=bundle["model_cfg"] if args.train_fampnn else None,
    )
    if args.resume and args.init_from:
        raise SystemExit(
            "--resume and --init-from do different things and cannot be "
            "combined: resume continues this experiment from its own state "
            "(step counter, optimizer moments, EMA), initialization starts a "
            "new phase from a previous one's weights at step 0"
        )
    if args.init_from:
        record = trainer.initialize_from(args.init_from)
        logger.info("initialized from another phase: %s", json.dumps(record, default=str))
    if args.resume:
        logger.info("resumed at step %d", trainer.resume(args.resume))

    (out / "run_config.json").write_text(
        json.dumps(
            dict(
                couple=couple_cfg,
                optim=optim_cfg,
                adapters=adapter_cfg,
                sb_feedback=sb_cfg,
                sb_identity=(sb_module.identity() if sb_module is not None else None),
                bs_gate=(bs_gate.identity() if bs_gate is not None else None),
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
    # Whether the selected SC->BB arm reads the sequence-blind or
    # predicted-sequence encodings, which the cycle does not build by default.
    from pxf.couple.readout import needs_sequence_controls

    needs_controls = bool(
        sb_module is not None
        and needs_sequence_controls(getattr(sb_module, "variant", None))
    )
    if needs_controls:
        logger.info(
            "arm %s reads a sequence-control encoding; each frozen half gets two "
            "extra encoder passes (cache identity unchanged)",
            sb_cfg.get("arm") or sb_cfg.get("variant"),
        )

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

    # ---- the SC->BB pilot ------------------------------------------------

    def pilot_examples(paths, *, per_structure, limit, seed_offset):
        """A FIXED pool of ``(structure, sigma, seed)`` with known BB targets.

        Fixed, not streamed: every arm must train on the same examples, the
        frozen half is expensive and cacheable, and a reconstructible noisy
        state is what makes the cache key mean anything. The sigma draws are
        derived from the structure's name rather than from loop position, so
        ``--pool-size`` truncates the pool without changing which examples
        survive.
        """
        from pxf.eval.couple import target_seed

        base = int(couple_cfg.get("seed", 0)) + int(seed_offset)
        # Sampled across the manifest rather than taken from the front: a
        # prefix of a 2,000-entry export is whatever order the exporter used,
        # and `--pool-size 512` would then train every arm on the same
        # arbitrary corner of it. Seeded, so the pool is still reproducible and
        # every arm gets the identical examples.
        order = torch.randperm(
            len(paths), generator=torch.Generator().manual_seed(base)
        ).tolist()
        out = []
        for path in [paths[i] for i in order]:
            name = Path(path).stem
            generator = torch.Generator().manual_seed(target_seed(base, name, 0.0, 0))
            sigmas = sigma_schedule.sample(int(per_structure), generator=generator)
            for replicate, sigma in enumerate(sigmas.tolist()):
                out.append(
                    dict(
                        path=str(path),
                        name=name,
                        sigma=float(sigma),
                        seed=target_seed(base, name, sigma, replicate),
                    )
                )
        if limit:
            out = out[: int(limit)]
        return out

    def build_pool(paths, *, per_structure, limit, seed_offset, label):
        """The pool, with unfeaturizable structures dropped rather than fatal.

        ``DesignSourceDataset`` refuses a structure whose design region exceeds
        ``--crop-size`` -- it does not crop the binder, it raises ("binder has
        278 tokens but crop_size=256"). One such entry must not take a 2,000-step
        run with it, so each is tried, skipped and recorded. The pool is
        over-drawn first and truncated after, so the requested size survives a
        few skips.
        """
        # Ask for more than needed, so skips do not shrink the pool below the
        # requested size; `limit` is applied to what actually featurizes.
        examples = pilot_examples(
            paths,
            per_structure=per_structure,
            limit=int(limit) * 2 if limit else None,
            seed_offset=seed_offset,
        )
        featurized_by_name, dropped = {}, []
        kept = []
        for entry in examples:
            if limit and len(kept) >= int(limit):
                break
            name = entry["name"]
            if name in dropped:
                continue
            if name not in featurized_by_name:
                try:
                    _sample_id, source = featurize_structures(
                        [entry["path"]],
                        crop_size=args.crop_size,
                        proteoaa_root=args.proteoaa_root,
                    )[0]
                    # The refusal happens when the item is accessed, not when
                    # the dataset is built, so it has to be touched here or the
                    # failure surfaces mid-training instead.
                    _ = source[0]
                except Exception as error:  # noqa: BLE001 - upstream raises broadly
                    dropped.append(name)
                    logger.warning(
                        "%s pool: dropping %s (%s)", label, name, str(error)[:160]
                    )
                    continue
                featurized_by_name[name] = source
            kept.append(entry)
        logger.info(
            "%s pool: %d example(s) over %d structure(s), %d structure(s) dropped",
            label,
            len(kept),
            len(featurized_by_name),
            len(dropped),
        )
        if dropped and len(dropped) > 0.5 * (len(dropped) + len(featurized_by_name)):
            raise SystemExit(
                f"{len(dropped)} of {len(dropped) + len(featurized_by_name)} "
                f"structures could not be featurized at --crop-size "
                f"{args.crop_size}. The featurizer refuses a design region "
                "larger than the crop rather than cropping it, so raise "
                "--crop-size to at least the longest structure in the manifest."
            )
        return kept, featurized_by_name

    def prepared(entry, featurized_by_name):
        """Featurize, bind the denoiser, and build the supervised BB mask."""
        source = featurized_by_name[entry["name"]]
        structure = to_featurized(entry["name"], source[0]).to(device)
        target = structure.backbone_target.float()
        mask = pilot.backbone_supervision_mask(
            structure.topology.atom_names,
            coordinate_mask=structure.label_dict.get("coordinate_mask"),
            device=device,
        )
        generator = torch.Generator().manual_seed(int(entry["seed"]))
        noise = torch.randn(target.shape, generator=generator).to(device)
        x_noisy = (target + noise * entry["sigma"])[None]
        sigma = torch.full((1,), entry["sigma"], device=device)
        controller.backbone = px_driver.bind(
            conditioning_cache.get(entry["name"], structure.feature_dict)
        )
        return structure, target, mask, x_noisy, sigma

    def upstream_for(entry, featurized_by_name, cache):
        """The cached frozen half, computing it on a miss."""
        structure, target, mask, x_noisy, sigma = prepared(entry, featurized_by_name)
        state = cache.get(entry["name"], entry["sigma"], entry["seed"])
        if state is None:
            # The packing sampler draws from the global RNG, so the frozen half
            # has to be seeded from the example rather than from wherever the
            # loop happens to have left it. Without this, two arms -- or a rerun
            # of the same arm -- read a DIFFERENT sc0 for the same example, and
            # the comparison measures the side-chain sampler as well as the
            # adapter. It is also what makes the cache reproducible rather than
            # merely reusable.
            torch.manual_seed(int(entry["seed"]))
            state = controller.frozen_half(
                structure.topology,
                x_noisy,
                sigma,
                structure.aatype,
                bs_delta_h=trainer.bs_delta_h,
            )
            cache.put(entry["name"], entry["sigma"], entry["seed"], state.to("cpu"))
        state = state.to(device)
        if needs_controls:
            # Two extra encoder passes, and only for the arms that read them.
            # Deterministic from the cached frozen half, so this does NOT
            # invalidate the upstream cache -- which is what keeps these arms
            # comparable to every arm already trained on it.
            state = replace(
                state,
                packed=controller.encode_sequence_controls(state.inputs, state.packed),
            )
        return structure, target, mask, x_noisy, sigma, state

    def pilot_batches():
        examples, featurized_by_name = build_pool(
            structures,
            per_structure=args.sigmas_per_structure,
            limit=args.pool_size,
            seed_offset=1,
            label="train",
        )
        if not examples:
            raise SystemExit("the pilot pool is empty")
        order = torch.Generator().manual_seed(int(couple_cfg.get("seed", 0)) + 7)
        while True:
            # Reshuffled each epoch rather than cycled in order: with a pool
            # smaller than the step budget, a fixed order makes the gradient
            # sequence periodic and the loss curve reads as if it converged.
            for index in torch.randperm(len(examples), generator=order).tolist():
                entry = examples[index]
                structure, target, mask, x_noisy, sigma, state = upstream_for(
                    entry, featurized_by_name, upstream_cache
                )
                yield CoupledBatch(
                    topology=structure.topology,
                    x_noisy=x_noisy,
                    sigma=sigma,
                    aatype=structure.aatype,
                    backbone_target=target[None],
                    backbone_atom_mask=mask[None],
                    upstream=state,
                    name=entry["name"],
                )

    def install_validation():
        """Score bb0 against bb1 on held-out examples. The pilot's own signal."""
        if not args.val_structures:
            return None
        val_paths = resolve_structures(args.val_structures, suffix=".cif")
        examples, featurized_by_name = build_pool(
            val_paths,
            per_structure=1,
            limit=args.val_pool_size,
            seed_offset=101,
            label="val",
        )
        val_cache = pilot.UpstreamCache(identity=upstream_cache.identity)

        def validate(step):
            rows = []
            was_training = adapters.training
            adapters.eval()
            for entry in examples:
                structure, target, mask, x_noisy, sigma, state = upstream_for(
                    entry, featurized_by_name, val_cache
                )
                with torch.no_grad():
                    cycle = controller.corrective_event(
                        structure.topology,
                        x_noisy,
                        sigma,
                        structure.aatype,
                        bs_delta_h=trainer.bs_delta_h,
                        upstream=state,
                    )
                keep = mask.bool()
                reference = target[keep]
                row = dict(
                    sigma=entry["sigma"],
                    bb0=_masked_rmsd(cycle.bb0_flat, reference, keep),
                )
                row["bb1"] = (
                    _masked_rmsd(cycle.bb1_flat, reference, keep)
                    if cycle.bb1_flat is not None
                    else row["bb0"]
                )
                # How far the correction actually moved the coordinates. At
                # step 0 this is the equivalence check, and it has to be read
                # instead of `fraction_improved`: two invocations of the same
                # PXDesign forward on a GPU differ in the last bits, so a
                # strict `bb1 < bb0` count is a coin flip over ties even when
                # delta_a is exactly zero -- which is why step 0 reports 12.5%
                # improved with an improvement of 0.0000 and a delta_a norm of 0.
                row["max_abs_change"] = (
                    float((cycle.bb1_flat - cycle.bb0_flat).abs().max())
                    if cycle.bb1_flat is not None
                    else 0.0
                )
                # Whichever residual this architecture emits. The late adapter
                # reports delta_a_norm; an early conditioner reports
                # delta_s_norm (and delta_z_norm), so reading only the first
                # left the column a flat 0.0000 for every early arm -- a
                # diagnostic that silently says "no correction" while
                # val_max_abs_change shows 1.9 A of movement.
                stats = cycle.feedback_stats
                row["delta_a_norm"] = float(
                    stats.get("delta_a_norm", stats.get("delta_s_norm", 0.0))
                )
                row["delta_z_norm"] = float(stats.get("delta_z_norm", 0.0))
                row["relative_residual"] = float(stats.get("relative_residual", 0.0))
                rows.append(row)
            if was_training:
                adapters.train()
            if not rows:
                return None
            mean = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
            improved = sum(1 for r in rows if r["bb1"] < r["bb0"])
            return dict(
                val_examples=len(rows),
                val_bb0_rmsd=mean("bb0"),
                val_bb1_rmsd=mean("bb1"),
                val_improvement=mean("bb0") - mean("bb1"),
                val_fraction_improved=improved / len(rows),
                val_delta_a_norm=mean("delta_a_norm"),
                val_delta_z_norm=mean("delta_z_norm"),
                val_relative_residual=mean("relative_residual"),
                val_max_abs_change=max(r["max_abs_change"] for r in rows),
            )

        return validate

    if run_pilot:
        conditioning_cache = ConditioningCache(px_driver, capacity=args.conditioning_cache)
        cache_identity = pilot.cache_identity(
            frozen=frozen,
            pack_steps=couple_cfg.get("pack_steps"),
            bs_policy=couple_cfg.get("bs_policy", "bypass"),
            sigma_schedule=sigma_schedule.identity(),
            seed_base=int(couple_cfg.get("seed", 0)),
            crop_size=args.crop_size,
        )
        upstream_cache = pilot.UpstreamCache(identity=cache_identity)
        if args.cache_upstream and Path(args.cache_upstream).is_file():
            upstream_cache = pilot.UpstreamCache.load(
                args.cache_upstream, identity=cache_identity
            )
            logger.info("loaded upstream cache: %s", upstream_cache.stats())
        trainer.validate_fn = install_validation()
        batches = pilot_batches
    else:
        batches = pxdesign_batches if args.backbone == "pxdesign" else stub_batches

    result = trainer.train(batches(), progress=lambda m: logger.info(m))
    if run_pilot:
        result["upstream_cache"] = upstream_cache.stats()
        result["conditioning_cache"] = conditioning_cache.stats()
        logger.info("caches: %s", json.dumps(result["upstream_cache"], default=str))
        # Only the run that computed something writes. Arms that loaded a
        # complete cache have zero misses and nothing to add, and rewriting a
        # 500 MB file they did not change is both wasted work and a race: the
        # three arms share one cache path by design, and two of them running
        # concurrently would interleave torch.save on it. The previous suite got
        # away with that; it was luck, not design.
        if args.cache_upstream and upstream_cache.misses:
            upstream_cache.save(args.cache_upstream)
            logger.info(
                "wrote upstream cache (%d new state(s)) -> %s",
                upstream_cache.misses,
                args.cache_upstream,
            )
        elif args.cache_upstream:
            logger.info(
                "upstream cache unchanged (%d hits, 0 misses); not rewriting %s",
                upstream_cache.hits,
                args.cache_upstream,
            )
    logger.info("done: %s", result)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
