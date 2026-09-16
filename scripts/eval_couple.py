#!/usr/bin/env python
"""Does the coupling actually pack better? Held-out packing metrics, two arms.

``train_couple.py`` reports ``L_SC``, FaMPNN's diffusion loss at randomly drawn
noise levels, on the structures it is fitting. That curve cannot answer the
question the staged plan poses -- whether PXDesign's ``a_token`` improves
side-chain packing on structures the adapter has never seen. This script answers
it directly: pack held-out structures through the coupled cycle, pack the same
structures through the same cycle with the adapters switched off, and compare.

    # phase 1 checkpoint against its own zero-adapter baseline
    python scripts/eval_couple.py --checkpoint runs/phase1/checkpoints/final.pt \\
        --structures configs/val_structures_afdb.txt \\
        --pxdesign-donor .../pxdesign_v0.1.0.pt --out runs/phase1/eval

    # the delta table again later, without recomputing
    python scripts/eval_couple.py --compare runs/phase1/eval

**Why the two arms are the right control.** The adapters are zero-initialized,
so switching them off does not approximate the pretrained system -- it restores
it exactly. And in phase 1 the backbone proposal is computed before ``A_BS`` is
applied, so with the same seed and the same sigma_B both arms see a bit-identical
backbone. The only difference between them is the residual on ``h_V``, which is
what makes the delta attributable to the adapter rather than to sampler noise or
backbone error.

**Side chains are scored in their own backbone's frames.** The packing happens
on a diffusion proposal, in the featurizer's coordinate frame; the reference is
FaMPNN's parse of the same file, which is neither centered the same way nor the
same backbone. ``sidechain_metrics.score`` assumes one shared backbone serves
both structures -- true for ``eval_protenix_sidechain.py``, which packs onto the
deposited backbone, and false here. Each residue's predicted side chain is
therefore transferred through its own backbone frame onto the native backbone
before scoring (``pxf.eval.couple.place_on_native_backbone``). Skipping that step
reports ~20 A RMSD on a perfectly good packing, which is how the need for it was
found.

What that measures is side-chain conformation relative to its own backbone,
which is the standard side-chain accuracy quantity and the one that isolates
packing from backbone error. Backbone error is not swept under the rug: it is
reported per sigma as ``backbone_rmsd``, Kabsch-superposed, so a coupling that
improves packing while the proposal drifts is distinguishable from one that
improves both.

**Read the delta, not the level.** Even with frames handled, the task is harder
than the deposited-backbone one: at high sigma_B the proposal is a real
perturbation and the side chains are packed into it. ``--with-native-reference``
adds a third arm -- pretrained FaMPNN on the deposited backbone, exactly the
``eval_protenix_sidechain.py`` task -- as the ceiling.

**sigma_B is swept.** Both adapters take log sigma_B as input, so one evaluation
point would only license a claim at that point. The sweep covers the schedule's
own training window and the report is per-sigma as well as pooled, because an
adapter that helps at low noise and hurts at high noise is a real and otherwise
invisible outcome.

Exit status is 0 even when the coupling regresses: this script measures, it does
not gate. Pass ``--fail-on-regression`` to make a headline regression non-zero
for CI-style use.
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from pxf.couple import codesign, schedule
from pxf.couple.controller import CycleOutput
from pxf.eval import couple as ev

logger = logging.getLogger("pxf.eval_couple")

METRICS_FILE = "couple_metrics.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--compare",
        nargs=1,
        metavar="RUN_DIR",
        default=None,
        help=f"print the delta table from an existing {METRICS_FILE} and exit",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="coupling checkpoint; omit to evaluate the untrained adapters "
        "(both arms then agree, which is the sanity check)",
    )
    p.add_argument(
        "--structures",
        default=None,
        help="held-out .cif directory or manifest (no default -- the "
        "training manifest is not a validation set)",
    )
    p.add_argument("--out", default=None)
    p.add_argument(
        "--config",
        default="configs/couple_phase1.yaml",
        help="supplies the adapter dims and the sigma window",
    )
    p.add_argument("--pxdesign-donor", default=None, help="published pxdesign_v0.1.0.pt")
    p.add_argument("--proteoaa-root", default=None)
    p.add_argument("--fampnn-weights", default="0.0", choices=("0.0", "0.3", "0.3-cath"))
    p.add_argument("--fampnn-checkpoint", default=None)
    p.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="must be >= the longest structure; a crop breaks the "
        "positional correspondence with the side-chain targets",
    )
    p.add_argument(
        "--pack-steps",
        type=int,
        default=50,
        help="side-chain rollout length; 50 at eval time, 10 during training",
    )
    p.add_argument("--n-sigma", type=int, default=5, help="sweep points across the window")
    p.add_argument(
        "--sigma-mode", default=None, choices=("trajectory", "loguniform", "fixed")
    )
    p.add_argument("--sigma-min", type=float, default=None)
    p.add_argument("--sigma-max", type=float, default=None)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--sigma-n-step", type=int, default=None)
    p.add_argument("--max-targets", type=int, default=200, help="0 for all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--ema",
        dest="ema",
        action="store_true",
        default=True,
        help="evaluate the EMA weights when the checkpoint has them (default)",
    )
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument(
        "--with-native-reference",
        action="store_true",
        help="add pretrained FaMPNN on the native backbone as a ceiling",
    )
    p.add_argument(
        "--run-feedback",
        action="store_true",
        help="also run the SC->BB half; off by default because the "
        "metric scored here is side-chain packing",
    )
    p.add_argument(
        "--mode",
        default="denoised",
        choices=("denoised", "native"),
        help="denoised: pack the PXDesign proposal, coupled vs uncoupled arms. "
        "native: pack the deposited backbone with pretrained FaMPNN -- the "
        "ceiling, and the only measurement that needs no donor and no sigma",
    )
    p.add_argument(
        "--codesign",
        action="store_true",
        help="co-generate sequence and side chains instead of teacher-forcing "
        "the native sequence: the MPNN module predicts s_hat from a fully "
        "X-masked sequence and (s_hat, h_V) go into the side-chain diffusion "
        "MLP, per FaMPNN's published inference. Side-chain geometry is then "
        "scored only where s_hat matches the deposited identity -- elsewhere "
        "the atom sets differ -- and sequence recovery is reported alongside",
    )
    p.add_argument(
        "--seq-temperature",
        type=float,
        default=0.0,
        help="sampling temperature for the predicted sequence; 0.0 is argmax "
        "(only read with --codesign)",
    )
    p.add_argument(
        "--shuffle-a-token",
        action="store_true",
        help="information-content control: feed A_BS another protein's token "
        "features at the same sigma, leaving the backbone, the encoder and the "
        "uncoupled arm untouched. If the coupled gain survives this, the "
        "adapter is acting as a generic regularizer on h_V rather than using "
        "sample-specific information from PXDesign",
    )
    p.add_argument("--fail-on-regression", action="store_true")
    p.add_argument("--allow-unpinned-sources", action="store_true")
    return p.parse_args(argv)


# --- reporting -------------------------------------------------------------


def _print_arm_table(title, arms, *, indent="  "):
    """One metric per row, both arms and the signed improvement."""
    table = ev.delta_table(arms)
    if not table:
        return
    print(f"{indent}{title}")
    print(f"{indent}  {'metric':32s} {'uncoupled':>11s} {'coupled':>11s} {'delta':>11s}")
    for group in ev.REPORT.values():
        for key in group:
            if key not in table:
                continue
            lo, hi, delta, better = table[key]
            if abs(delta) < ev.REGRESSION_TOLERANCE:
                flag = ""
            elif better > 0:
                flag = "  better"
            else:
                flag = "  WORSE"
            print(f"{indent}  {key:32s} {lo:11.4f} {hi:11.4f} {delta:+11.4f}{flag}")
    print()


def _report_native(record):
    """One arm, so a delta table would have nothing to compare. Levels only.

    There is no verdict to reach here -- this run establishes the ceiling that
    the denoised run is read against, so it always returns zero regressions.
    """
    summary = record["native"]
    print(f"\n=== {record['label']} ===")
    print(
        f"  {record['n_targets']} target(s), {record['pack_steps']} pack steps, "
        f"FaMPNN {record['fampnn_weights']}"
    )
    print(f"  structures: {record['structures']}\n")
    for title, group in ev.REPORT.items():
        shown = [key for key in group if key in summary]
        if not shown:
            continue
        print(f"  [{title}]")
        for key in shown:
            print(f"    {key:32s} {float(summary[key]):11.4f}")
    print(
        "\n  This is the packing task with the backbone given. The denoised run's\n"
        "  gap to these numbers is what the diffusion proposal costs.\n"
    )
    return 0


def report(record):
    """Print the verdict; return the number of headline regressions."""
    if record.get("mode") == "native":
        return _report_native(record)
    pooled = record["pooled"]
    print(f"\n=== coupling evaluation: {record['label']} ===")
    print(
        f"  {record['n_targets']} target(s), {record['n_packings']} packing(s), "
        f"sigma_B in {record['sigma_values'][0]:.3f}..{record['sigma_values'][-1]:.3f} A"
    )
    if record.get("checkpoint"):
        print(f"  checkpoint: {record['checkpoint']}  (step {record.get('step', '?')})")
    else:
        print("  no checkpoint: untrained adapters, so the two arms should agree")
    print()

    _print_arm_table("[pooled over all sigma_B]", pooled)
    for entry in record["per_sigma"]:
        # The proposal's own error at this noise level, alongside the packing
        # numbers -- side chains are scored in local frames, so the packing
        # metrics deliberately do not reflect it and it has to be stated.
        bb = (entry.get("backbone_rmsd") or {}).get("coupled")
        suffix = f"   backbone RMSD {bb:.3f} A" if isinstance(bb, (int, float)) else ""
        _print_arm_table(f"[sigma_B = {entry['sigma']:.3f} A]{suffix}", entry["arms"])

    if "native_reference" in record:
        ref = record["native_reference"]
        print("  [native-backbone reference: pretrained FaMPNN, deposited backbone]")
        for key in ev.HEADLINE:
            if key in ref:
                print(f"    {key:32s} {float(ref[key]):11.4f}")
        print("    The gap from this row to the arms above is backbone error,")
        print("    not packing error -- the coupling cannot be blamed for it.\n")

    hurt = ev.regressions(pooled)
    helped = ev.improvements(pooled)
    if hurt:
        print("  REGRESSION on headline metrics (pooled):")
        for key, lo, hi in hurt:
            print(f"    {key}: {lo:.4f} -> {hi:.4f}")
    if helped:
        print("  improved on headline metrics (pooled):")
        for key, lo, hi in helped:
            print(f"    {key}: {lo:.4f} -> {hi:.4f}")
    if not hurt and not helped:
        print("  no headline metric moved beyond float noise: the coupling is inert here.")
        print("  For an untrained checkpoint that is the expected result. For a trained")
        print("  one it means the adapter learned nothing that survives to packing.")
    print()
    return len(hurt)


def _with_sequence_recovery(summary):
    """Add ``sequence_recovery`` to an aggregate that carries the seq counts.

    ``aggregate`` sums every count key it is given but only derives the ratios
    ``canonical.summarize_metrics`` knows about, so the co-design ratio is
    formed here -- once, over summed counts, the same way the other ratios are.
    """
    total = summary.get("seq_residues")
    if not total:
        return summary
    summary = dict(summary)
    summary["sequence_recovery"] = float(summary.get("seq_recovered", 0.0)) / float(total)
    return summary


def compare(run_dir):
    path = Path(run_dir) / METRICS_FILE
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; run the evaluation first")
    return report(json.loads(path.read_text()))


# --- evaluation ------------------------------------------------------------


def run_native(args):
    """Pack the deposited backbone with pretrained FaMPNN. The ceiling arm.

    No PXDesign, no adapters, no sigma_B: there is no diffusion in this path, so
    there is nothing for a noise level to mean and nothing for a coupling to
    couple. It exists to answer the question the denoised run cannot -- how much
    of the error there is the backbone's rather than the packer's.

    This is also the one arm where ``score``'s shared-backbone assumption is
    satisfied without help: the packer is handed the native backbone and does not
    move it, so no frame transfer is needed and none is done.
    """
    import sys as _sys

    from fampnn.data import residue_constants as rc
    from pxf import atom37
    from pxf.device import select_device
    from pxf.eval.canonical import load as load_metrics
    from pxf.eval.sidechain_metrics import aggregate, score
    from pxf.sidechain.fampnn import FaMPNNSideChainPacker

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_couple import resolve_structures

    structures = resolve_structures(args.structures, suffix=".cif")
    if args.max_targets:
        structures = structures[: args.max_targets]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger.info(
        "%d structure(s), native backbone, %d pack steps", len(structures), args.pack_steps
    )

    device = select_device(args.device)
    packer = FaMPNNSideChainPacker(
        args.fampnn_checkpoint,
        variant=args.fampnn_weights,
        num_steps=args.pack_steps,
        strict_sources=not args.allow_unpinned_sources,
    ).to(device)
    canonical = load_metrics()
    backbone_slots = list(atom37.BACKBONE_SLOTS)

    counts, rows, skipped = [], [], []
    started = time.time()
    for index, path in enumerate(structures):
        sample_id = Path(path).stem
        try:
            native = _native_parse(structures, sample_id)
        except (ValueError, KeyError, IndexError, FileNotFoundError) as error:
            skipped.append(dict(target=sample_id, reason=str(error)[:200]))
            continue
        native37, native_mask = ev.native_atom37(native, rc)
        aatype = native["aatype"].reshape(-1).long()
        if aatype.numel() == 0 or int(aatype.max()) >= 20:
            skipped.append(dict(target=sample_id, reason="non-canonical residue"))
            continue

        # Backbone slots only: the side chains are what the packer must produce.
        given = torch.zeros_like(native_mask, dtype=torch.float32)
        given[:, backbone_slots] = native_mask[:, backbone_slots].float()
        backbone_only = (native37 * given[..., None])[None].to(device)
        residue_mask, seq_counts = (None, None)
        with torch.no_grad():
            if args.codesign:
                # MPNN -> s_hat -> side-chain diffusion on the deposited
                # backbone. No adapters, so this is the co-design ceiling.
                torch.manual_seed(ev.target_seed(args.seed, sample_id, 0))
                s_hat, sidechains, _aux = codesign.codesign_native(
                    packer.model,
                    backbone_only,
                    residue_index=native["residue_index"].reshape(1, -1).to(device),
                    chain_index=native["chain_index"].reshape(1, -1).to(device),
                    pack_steps=args.pack_steps,
                    temperature=args.seq_temperature,
                )
                assembled = CycleOutput(
                    bb0_flat=None,
                    bb0_dense=backbone_only,
                    a_token=None,
                    sidechains=sidechains,
                )
                pred37, pred_mask = ev.predicted_atom37(assembled, s_hat.reshape(-1), rc)
                pred37, pred_mask = pred37.cpu(), pred_mask.cpu()
                residue_mask, seq_counts = codesign.sequence_recovery(s_hat, aatype)
                residue_mask = residue_mask.cpu()
                # The deposited backbone is passed through unchanged here, so
                # there is no packer-induced shift to report.
                backbone_shift = 0.0
            else:
                packed = packer(
                    coords_af2=backbone_only,
                    aatype=aatype[None].to(device),
                    atom_mask=given[None].to(device),
                    residue_index=native["residue_index"].reshape(1, -1).to(device),
                    chain_index=native["chain_index"].reshape(1, -1).to(device),
                    seed=ev.target_seed(args.seed, sample_id, 0),
                )
                pred37 = packed["coords_af2"][0].cpu()
                pred_mask = packed["atom_mask_af2"][0].cpu()
                backbone_shift = float(packed["backbone_shift"])
        target_counts, summary = score(
            pred37,
            pred_mask,
            native37.cpu(),
            native_mask.cpu(),
            aatype.cpu(),
            canonical=canonical,
            residue_mask=residue_mask,
        )
        if seq_counts is not None:
            target_counts = dict(target_counts)
            target_counts["seq_residues"] = torch.tensor(float(seq_counts["residues"]))
            target_counts["seq_recovered"] = torch.tensor(float(seq_counts["recovered"]))
            summary = dict(summary)
            summary["sequence_recovery"] = seq_counts["sequence_recovery"]
        counts.append(target_counts)
        rows.append(
            dict(
                target=sample_id,
                arm="native",
                length=int(aatype.shape[0]),
                backbone_shift=backbone_shift,
                **{
                    k: summary[k]
                    for group in ev.REPORT.values()
                    for k in group
                    if k in summary
                },
            )
        )
        if index % 25 == 0 or index == len(structures) - 1:
            logger.info(
                "%d/%d targets, %.1fs elapsed",
                index + 1,
                len(structures),
                time.time() - started,
            )

    if not counts:
        raise SystemExit(f"nothing was scored; {len(skipped)} skipped, e.g. {skipped[:3]}")

    record = dict(
        label="pretrained FaMPNN on the deposited backbone",
        mode="native",
        codesign=bool(args.codesign),
        seq_decode=(codesign.DECODE_SINGLE_PASS if args.codesign else None),
        seq_temperature=(args.seq_temperature if args.codesign else None),
        checkpoint=None,
        structures=str(args.structures),
        n_targets=len(counts),
        n_packings=len(counts),
        pack_steps=args.pack_steps,
        fampnn_weights=args.fampnn_weights,
        native=_with_sequence_recovery(aggregate(counts, canonical=canonical)),
        skipped=skipped,
        seconds=round(time.time() - started, 1),
    )
    (out / METRICS_FILE).write_text(json.dumps(record, indent=2))
    with (out / "per_target.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", out / METRICS_FILE)
    report(record)
    return 0


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.compare:
        return 1 if compare(args.compare[0]) and args.fail_on_regression else 0

    required = ["structures", "out"]
    # Native mode never builds a backbone, so the donor is not merely optional
    # there -- requiring it would make the cheap measurement load a 557 MB
    # checkpoint and the whole PXDesign stack for nothing.
    if args.mode == "denoised":
        required.append("pxdesign_donor")
    for name in required:
        if not getattr(args, name):
            raise SystemExit(f"--{name.replace('_', '-')} is required in {args.mode} mode")

    if args.mode == "native":
        return run_native(args)

    import yaml
    from fampnn.model.sd_model import SeqDenoiser

    from fampnn.data import residue_constants as rc
    from pxf import provenance
    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        featurize_structures,
        load_backbone_model,
        to_featurized,
    )
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.controller import CoupledDenoiser
    from pxf.device import select_device
    from pxf.eval.canonical import load as load_metrics
    from pxf.eval.sidechain_metrics import aggregate, score

    # resolve_structures lives in the trainer and is imported rather than copied:
    # a second implementation of "which files is this run reading" is exactly the
    # kind of divergence that makes two runs incomparable.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_couple import resolve_structures

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    adapter_cfg = dict(config.get("adapters", {}))
    sigma_schedule = schedule.from_config(
        config.get("sigma"),
        mode=args.sigma_mode,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma=args.sigma,
        n_step=args.sigma_n_step,
    )
    sigmas = ev.sweep_sigmas(sigma_schedule, args.n_sigma)

    structures = resolve_structures(args.structures, suffix=".cif")
    if args.max_targets:
        structures = structures[: args.max_targets]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger.info(
        "%d held-out structure(s); sigma_B sweep %s",
        len(structures),
        [round(s, 3) for s in sigmas],
    )

    # --- frozen components ---
    checkpoint = (
        Path(args.fampnn_checkpoint)
        if args.fampnn_checkpoint
        else provenance.fampnn_checkpoint(args.fampnn_weights)
    )
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fampnn = SeqDenoiser(bundle["model_cfg"])
    fampnn.load_state_dict(bundle["state_dict"], strict=True)
    fampnn.eval().requires_grad_(False)
    device = select_device(args.device)
    fampnn.to(device)
    c_h_V = int(fampnn.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V)

    px_model, _configs, _record = load_backbone_model(
        args.pxdesign_donor, device=device, proteoaa_root=args.proteoaa_root
    )
    px_driver = PXDesignBackboneDriver(px_model)

    adapters = CouplingAdapters(px_driver.c_token, c_h_V, **adapter_cfg).to(device)
    step = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "adapters" not in state:
            raise SystemExit(f"{args.checkpoint} is not a coupling checkpoint")
        adapters.load_state_dict(state["adapters"])
        step = int(state.get("step", 0))
        if args.ema and state.get("ema"):
            from pxf.train.ema import EMA

            # EMA requires exactly one of decay / relative_length, so the
            # checkpoint's own setting is what rebuilds it. The shadow weights
            # are overwritten by load_state_dict immediately afterwards; the
            # constructor argument only has to be consistent, not correct.
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
            logger.info("evaluating the EMA weights (step %s)", step)
    adapters.eval().requires_grad_(False)

    controller = CoupledDenoiser(
        backbone=None, fampnn=fampnn, adapters=adapters, pack_steps=args.pack_steps
    )
    canonical = load_metrics()

    featurized = featurize_structures(
        structures, crop_size=args.crop_size, proteoaa_root=args.proteoaa_root
    )

    packer = None
    if args.with_native_reference:
        from pxf.sidechain.fampnn import FaMPNNSideChainPacker

        packer = FaMPNNSideChainPacker(
            args.fampnn_checkpoint,
            variant=args.fampnn_weights,
            num_steps=args.pack_steps,
            strict_sources=not args.allow_unpinned_sources,
        ).to(device)

    # None = control off. {} = on but no donor seen yet (the first target).
    donor_bank = {} if args.shuffle_a_token else None
    counts = {(arm, i): [] for arm in ev.ARMS for i in range(len(sigmas))}
    backbone_rmsds = {(arm, i): [] for arm in ev.ARMS for i in range(len(sigmas))}
    pooled_counts = {arm: [] for arm in ev.ARMS}
    reference_counts = []
    rows, skipped = [], []
    started = time.time()

    for index, (sample_id, source) in enumerate(featurized):
        try:
            structure = to_featurized(sample_id, source[0]).to(device)
            native = _native_parse(structures, sample_id)
            ev.check_alignment(sample_id, native["aatype"], structure.aatype)
        except (ValueError, KeyError, IndexError, FileNotFoundError) as error:
            skipped.append(dict(target=sample_id, reason=str(error)[:200]))
            logger.warning("skipping %s: %s", sample_id, str(error)[:160])
            continue

        native37, native_mask = ev.native_atom37(native, rc)
        native37 = native37.to(device)
        native_mask = native_mask.to(device)
        aatype = structure.aatype.reshape(-1)
        if int(aatype.max()) >= 20:
            skipped.append(dict(target=sample_id, reason="non-canonical residue"))
            continue

        target = structure.backbone_target.float()
        controller.backbone = px_driver.bind(px_driver.conditioning(structure.feature_dict))

        if packer is not None:
            reference_counts.append(
                _native_reference(
                    packer,
                    native,
                    native37,
                    native_mask,
                    aatype,
                    canonical,
                    score,
                    args.seed,
                    device,
                )
            )

        # Shuffled control: this target's own a_token becomes the next target's
        # donor, so the donor is always a *different* protein at the same sigma.
        # The first target has no donor yet, so it only seeds the bank and is
        # not scored -- hence n is one lower than an unshuffled run.
        own_a_token = {}
        score_this_target = donor_bank is None or bool(donor_bank)

        for si, sigma_value in enumerate(sigmas):
            seed = ev.target_seed(args.seed, sample_id, si)
            sigma = torch.full((1,), float(sigma_value), device=device)
            # One noise draw per (target, sigma), reused by both arms: the arms
            # must differ only in the adapter, never in the input.
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(target.shape, generator=generator).to(device)
            x_noisy = (target + noise * float(sigma_value))[None]

            for arm in ev.ARMS:
                enabled = arm == "coupled"
                adapters.enable_bb_to_sc = enabled
                adapters.enable_sc_to_bb = enabled and args.run_feedback
                # The packing sampler draws from the global RNG, so both arms are
                # reseeded identically -- otherwise the delta measures the sampler.
                torch.manual_seed(seed)
                with torch.no_grad():
                    if args.codesign:
                        # MPNN -> s_hat -> side-chain diffusion, with the
                        # adapter residual still on h_V between the stages.
                        cycle, s_hat = codesign.codesign_cycle(
                            controller,
                            structure.topology,
                            x_noisy,
                            sigma,
                            aatype,
                            temperature=args.seq_temperature,
                        )
                    else:
                        s_hat = None
                        # The shuffled control substitutes a donor protein's
                        # token features into A_BS. Only the coupled arm is
                        # touched: the uncoupled arm has no residual to
                        # substitute into, and its packing must stay the shared
                        # baseline both runs are measured against.
                        override = None
                        if enabled and donor_bank:
                            override = ev.donor_a_token(
                                donor_bank[si], int(aatype.shape[0])
                            )
                        cycle = controller.forward(
                            structure.topology,
                            x_noisy,
                            sigma,
                            aatype,
                            run_feedback=args.run_feedback and enabled,
                            a_token_override=override,
                        )
                        # cycle.a_token is this structure's own even when the
                        # residual read a donor, so caching it here is correct
                        # whichever arm ran first.
                        if donor_bank is not None:
                            own_a_token.setdefault(si, cycle.a_token.detach())
                if not score_this_target:
                    continue
                # The packed atom set belongs to whatever sequence was packed
                # for, so atom37 assembly must use s_hat in codesign mode.
                scored_aatype = aatype if s_hat is None else s_hat.reshape(-1)
                pred37, pred_mask = ev.predicted_atom37(cycle, scored_aatype, rc)
                pred37, pred_mask = pred37.cpu(), pred_mask.cpu()
                native_cpu, native_mask_cpu = native37.cpu(), native_mask.cpu()
                # How far the proposal itself landed, before side chains are
                # re-framed onto the native backbone and that error is removed
                # from the packing numbers.
                bb_rmsd = ev.backbone_rmsd(pred37, native_cpu, native_mask_cpu)
                # See place_on_native_backbone: score() assumes one shared
                # backbone, which a diffusion proposal in the featurizer's frame
                # is not.
                placed = ev.place_on_native_backbone(pred37, native_cpu, canonical)
                # With a predicted sequence, geometry is only comparable where
                # the identity matches -- elsewhere the residue has a different
                # atom set and rotamer recovery has no referent.
                residue_mask, seq_counts = (None, None)
                if s_hat is not None:
                    residue_mask, seq_counts = codesign.sequence_recovery(s_hat, aatype)
                    residue_mask = residue_mask.cpu()
                target_counts, summary = score(
                    placed,
                    pred_mask,
                    native_cpu,
                    native_mask_cpu,
                    aatype.cpu(),
                    canonical=canonical,
                    residue_mask=residue_mask,
                )
                if seq_counts is not None:
                    target_counts = dict(target_counts)
                    target_counts["seq_residues"] = torch.tensor(
                        float(seq_counts["residues"])
                    )
                    target_counts["seq_recovered"] = torch.tensor(
                        float(seq_counts["recovered"])
                    )
                    summary = dict(summary)
                    summary["sequence_recovery"] = seq_counts["sequence_recovery"]
                counts[(arm, si)].append(target_counts)
                pooled_counts[arm].append(target_counts)
                rows.append(
                    dict(
                        target=sample_id,
                        arm=arm,
                        sigma=float(sigma_value),
                        length=int(aatype.shape[0]),
                        backbone_rmsd=bb_rmsd,
                        **{
                            k: summary[k]
                            for group in ev.REPORT.values()
                            for k in group
                            if k in summary
                        },
                    )
                )
                backbone_rmsds[(arm, si)].append(bb_rmsd)

        if donor_bank is not None and own_a_token:
            # Rotate: the next target's donor is this one. A cyclic shift by one
            # is a derangement for any n > 1, so no target ever donates to
            # itself.
            donor_bank = own_a_token

        if index % 10 == 0 or index == len(featurized) - 1:
            logger.info(
                "%d/%d targets, %.1fs elapsed",
                index + 1,
                len(featurized),
                time.time() - started,
            )

    if not pooled_counts["coupled"]:
        raise SystemExit(
            "no target survived featurization and alignment; nothing was scored. "
            f"{len(skipped)} skipped, e.g. {skipped[:3]}"
        )

    record = dict(
        label="coupling vs pretrained, held-out",
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        step=step,
        structures=str(args.structures),
        n_targets=len(pooled_counts["coupled"]) // max(1, len(sigmas)),
        n_packings=len(pooled_counts["coupled"]),
        sigma_values=[float(s) for s in sigmas],
        sigma_schedule=sigma_schedule.identity(),
        pack_steps=args.pack_steps,
        run_feedback=bool(args.run_feedback),
        a_token_source="shuffled-donor" if args.shuffle_a_token else "own",
        codesign=bool(args.codesign),
        seq_decode=(codesign.DECODE_SINGLE_PASS if args.codesign else None),
        seq_temperature=(args.seq_temperature if args.codesign else None),
        ema=bool(args.ema and args.checkpoint),
        pooled={
            arm: _with_sequence_recovery(aggregate(pooled_counts[arm], canonical=canonical))
            for arm in ev.ARMS
        },
        per_sigma=[
            dict(
                sigma=float(sigmas[i]),
                arms={
                    arm: _with_sequence_recovery(
                        aggregate(counts[(arm, i)], canonical=canonical)
                    )
                    for arm in ev.ARMS
                },
                backbone_rmsd={
                    arm: (
                        sum(backbone_rmsds[(arm, i)]) / len(backbone_rmsds[(arm, i)])
                        if backbone_rmsds[(arm, i)]
                        else None
                    )
                    for arm in ev.ARMS
                },
            )
            for i in range(len(sigmas))
        ],
        skipped=skipped,
        seconds=round(time.time() - started, 1),
    )
    if reference_counts:
        record["native_reference"] = aggregate(reference_counts, canonical=canonical)

    (out / METRICS_FILE).write_text(json.dumps(record, indent=2))
    with (out / "per_target.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", out / METRICS_FILE)

    hurt = report(record)
    return 1 if hurt and args.fail_on_regression else 0


def _native_parse(structures, sample_id):
    """FaMPNN's own parse of the source file -- the side-chain reference."""
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    match = next((p for p in structures if Path(p).stem == sample_id), None)
    if match is None:
        raise ValueError(f"no source file for {sample_id}")
    return process_single_pdb(load_feats_from_pdb(str(match)))


def _native_reference(
    packer, native, native37, native_mask, aatype, canonical, score, seed, device
):
    """Pretrained FaMPNN packing the *deposited* backbone: the ceiling arm.

    Backbone slots only are handed over, so this is the same packing task
    ``eval_protenix_sidechain.py`` measures -- the difference from the two
    coupled arms is the backbone, which is the point of having it.
    """
    from pxf import atom37

    backbone = list(atom37.BACKBONE_SLOTS)
    given = torch.zeros_like(native_mask, dtype=torch.float32)
    given[:, backbone] = native_mask[:, backbone].float()
    packed = packer(
        coords_af2=(native37 * given[..., None])[None],
        aatype=aatype[None],
        atom_mask=given[None],
        residue_index=native["residue_index"].reshape(1, -1).to(device),
        chain_index=native["chain_index"].reshape(1, -1).to(device),
        seed=seed,
    )
    target_counts, _summary = score(
        packed["coords_af2"][0].cpu(),
        packed["atom_mask_af2"][0].cpu(),
        native37.cpu(),
        native_mask.cpu(),
        aatype.cpu(),
        canonical=canonical,
    )
    return target_counts


if __name__ == "__main__":
    raise SystemExit(main())
