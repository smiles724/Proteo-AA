#!/usr/bin/env python3
"""Phase 0: establish that the iterative BB->SC coupling is live on real weights.

    python scripts/phase0_pack_hook_check.py \
        --adapters   <a_bs final.pt> \
        --pxdesign-donor <pxdesign_v0.1.0.pt> \
        --out runs/phase0

The pack hook (`pxf/couple/pack_hook.py`) is implemented and unit-tested, but
its tests drive a fake side-chain module, and its only caller
(`scripts/codesign_uncond.py`) is single-chain monomer co-design. Nothing has
ever run it on a two-chain target+binder complex with real weights. Every
check below exists because its failure mode is a *null result that looks like
a finding*: a coupled arm that silently equals the uncoupled arm reports "no
effect" just as convincingly as a coupling that genuinely does not help.

The checks, in the order a failure localises:

  1. ``load``        strict donor and adapter loading. Records module file
                     locations and the FaMPNN variant, and refuses when the
                     adapter was trained against a different donor. Widths
                     match across FaMPNN 0.0 and 0.3, so shape agreement is
                     not evidence.
  2. ``noop``        the hook installed with ``delta_h=None`` reproduces the
                     uncoupled sampler under a replayed seed. Measured, not
                     asserted approximately: coordinates and sequence.
  3. ``residual``    the trained residual is finite, non-zero on binder rows,
                     exactly zero on target rows. Records the gate value and
                     -- the number that matters -- the residual norm relative
                     to ``h_V``.
  4. ``counters``    with S unmasking steps and a final repack, side-chain
                     diffusion should be entered S+1 times and the residual
                     applied on every one.
  5. ``invariance``  target identities and target coordinates are unchanged
                     between the coupled and uncoupled arms.
  6. ``visibility``  the missing-atom mask, computed once from input
                     identities that are UNKNOWN at design positions, must not
                     permanently hide side chains the model then generates.
  7. ``logits``      with a single unmasking step the post-logit hook cannot
                     change the sequence. Identical sequence with different
                     coordinates is the signature of a correctly placed
                     post-logit intervention; a changed sequence means it is
                     not post-logit and every later interpretation shifts.

What is deliberately NOT here: AF2-IG scoring. That is the plan's eighth
Phase 0 item and it belongs with `score_binder_matrix.py`, in the other
environment, because it shares nothing with this one.

Every arm draws from a named generator so the coupled and uncoupled runs
consume matched randomness. Reusing one integer seed is not sufficient when
two arms can consume different numbers of draws.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import torch  # noqa: E402
import yaml  # noqa: E402

logger = logging.getLogger("phase0")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV = REPO_ROOT / "configs" / "binder_benchmark" / "dev_complexes.yaml"
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "binder_benchmark" / "checkpoint_manifest.json"

# Coordinates agreeing to this are the same trajectory. Float accumulation over
# ~100 iterative steps on GPU will not give bit-equality, so the tolerance is
# measured and reported rather than assumed.
NOOP_TOLERANCE_ANGSTROM = 1e-4


def _module_location(module) -> Optional[str]:
    """Where a package actually lives, for regular and namespace packages alike."""
    path = getattr(module, "__file__", None)
    if path:
        return str(Path(path).parent)
    paths = list(getattr(module, "__path__", []) or [])
    return str(paths[0]) if paths else None


def _manifest_path(manifest: dict, role: str) -> Optional[str]:
    for entry in manifest.get("entries", []):
        if entry["role"] == role and entry["status"] == "available":
            return entry["path"]
    return None


# --------------------------------------------------------------------- setup


def build(args) -> dict[str, Any]:
    """Load donor, designer and adapters, strictly. Check 1."""
    from pxf import provenance
    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.device import select_device
    from pxf.sidechain.design import FaMPNNFullAtomDesigner

    device = select_device(args.device)

    state = torch.load(args.adapters, map_location="cpu", weights_only=False)
    if "adapters" not in state:
        raise SystemExit(f"{args.adapters} is not a coupling checkpoint")
    frozen = (state.get("frozen") or {}).get("fampnn")
    trained_variant = (
        frozen.get("variant") if isinstance(frozen, dict) else frozen
    )
    recorded_donor_sha = frozen.get("sha256") if isinstance(frozen, dict) else None

    variant = args.fampnn_weights or trained_variant
    if trained_variant and str(variant) != str(trained_variant):
        raise SystemExit(
            f"the adapter was trained against FaMPNN {trained_variant} but this "
            f"run asks for {variant}. The widths match either way, which is "
            "exactly why this is refused rather than warned about."
        )

    designer = FaMPNNFullAtomDesigner(
        variant=variant,
        seq_steps=args.seq_steps,
        temperature=args.temperature,
        psce_threshold=args.psce_threshold,
        repack_last=not args.no_repack,
        strict_sources=not args.allow_unpinned_sources,
    ).to(device)
    # The donor the designer actually loaded, hashed now.
    donor_path = provenance.fampnn_checkpoint(variant)
    donor_sha = provenance.file_sha256(donor_path)
    if recorded_donor_sha and donor_sha != recorded_donor_sha:
        raise SystemExit(
            f"FaMPNN {variant} on disk hashes {donor_sha[:16]} but the adapter "
            f"records {recorded_donor_sha[:16]}. Same variant name, different "
            "bytes: the adapter would be queried out of distribution."
        )

    px_model, _cfg, _rec = load_backbone_model(args.pxdesign_donor, device=device)
    driver = PXDesignBackboneDriver(px_model)

    adapters = CouplingAdapters(driver.c_token, node_feature_dim(designer.model)).to(device)
    adapters.load_state_dict(state["adapters"])
    used_ema = False
    if state.get("ema") and not args.raw_weights:
        from pxf.train.ema import EMA

        settings = state.get("settings") or {}
        ema = EMA(
            adapters,
            decay=settings.get("ema_decay"),
            relative_length=(
                None if settings.get("ema_decay") is not None
                else settings.get("ema_relative_length") or 0.25
            ),
        )
        ema.load_state_dict(state["ema"])
        ema.copy_to(adapters)
        used_ema = True
    adapters.eval().requires_grad_(False)
    adapters.enable_bb_to_sc = True

    import fampnn
    import pxf

    record = {
        "device": str(device),
        "fampnn_variant": str(variant),
        "fampnn_trained_variant": str(trained_variant),
        "fampnn_donor_path": str(donor_path),
        "fampnn_donor_sha256": donor_sha,
        "adapter_path": str(args.adapters),
        "adapter_step": state.get("step"),
        "adapter_uses_ema": used_ema,
        "pxdesign_donor": str(args.pxdesign_donor),
        "c_token": int(driver.c_token),
        "c_h_V": int(node_feature_dim(designer.model)),
        # `fampnn` is a namespace package, so `__file__` is None and
        # Path(None) raises. __path__ is the portable answer for both kinds.
        "module_locations": {
            name: _module_location(module)
            for name, module in (("pxf", pxf), ("fampnn", fampnn))
        },
        "sources": provenance.runtime_sources(strict=not args.allow_unpinned_sources),
    }
    return {"designer": designer, "driver": driver, "adapters": adapters,
            "device": device, "record": record}


def featurize(cif_path: Path, binder_chain: str, crop_size: int):
    from pxf.backbone.driver import featurize_structures, to_featurized

    # Returns a list of (sample_id, DesignSourceDataset) pairs, one per path.
    items = featurize_structures(
        [str(cif_path)],
        crop_size=crop_size,
        binder_chain_ids=[binder_chain],
        parser_dataset="WeightedPDB",
    )
    sample_id, dataset = items[0]
    return to_featurized(sample_id, dataset[0])


def a_token_at(driver, structure, sigma_b: float, device, generator) -> torch.Tensor:
    """PXDesign's token features for this complex at a declared sigma_B.

    The adapter was trained on (noisy backbone, sigma_B) pairs drawn from the
    trajectory, so the in-distribution query for a finished structure is the
    structure perturbed to that sigma -- not the clean structure with a sigma
    label attached.
    """
    sigma = torch.full((1,), float(sigma_b), device=device)
    cond = driver.conditioning(structure.feature_dict)
    bound = driver.bind(cond)
    with torch.no_grad():
        target = structure.backbone_target.float()
        noise = torch.randn(target.shape, generator=generator).to(device)
        x_noisy = (target + noise * float(sigma_b))[None]
        _bb0, a_token = bound(x_noisy, sigma)
    if a_token is None:
        raise SystemExit("the backbone driver returned no a_token")
    return a_token if a_token.dim() == 3 else a_token[None]


# -------------------------------------------------------------------- checks


def _design(designer, inputs, delta_h, seed, counter=None):
    from pxf.couple.pack_hook import residual_on_sidechain_diffusion

    stats = counter if counter is not None else {}
    with residual_on_sidechain_diffusion(designer.model, delta_h, counter=stats):
        out = designer.design(seed=seed, **inputs)
    # design() returns dict(designs=[DesignResult, ...], backbone_shift, unbatched);
    # every call here is batch-of-one.
    return out["designs"][0], stats


def run_complex(entry, ctx, args) -> dict[str, Any]:
    from pxf.couple.binder_residual import (
        ChainRoles, binder_masked_residual, describe_residual,
    )
    from pxf.couple.bs_policy import gate_by_name

    designer, driver, adapters = ctx["designer"], ctx["driver"], ctx["adapters"]
    device = ctx["device"]
    cif = Path(args.mmcif_dir or entry["_mmcif_dir"]) / f"{entry['id']}.cif"

    out: dict[str, Any] = {"id": entry["id"], "cif": str(cif), "checks": {}}
    structure = featurize(cif, entry["binder_chain"], args.crop_size).to(device)

    roles = ChainRoles(binder=structure.design_mask.reshape(-1).bool().cpu())
    out["roles"] = roles.identity()
    length = roles.length

    # PXDesign emits flat atoms; FaMPNN wants a per-residue atom37 block. The
    # converter is the component that owns that mapping, and using it here
    # rather than re-densifying by hand is what keeps this check on the same
    # path the real arms will take.
    from pxf.couple.converter import PXFaRepresentationConverter

    converter = PXFaRepresentationConverter()
    topology = structure.topology
    coupled_inputs = converter.px_backbone_to_fampnn(
        structure.backbone_target,
        topology.atom_names,
        topology.atom_to_token_idx,
        topology.num_tokens,
        res_names=topology.res_names,
        residue_index=topology.residue_index,
        chain_index=topology.chain_index,
        aatype=structure.aatype,
    ).to(device)

    # The binder's native identities must not reach the designer. `aatype` from
    # the featurizer is the teacher-forced native sequence, which is the right
    # thing for the held-fixed target and exactly the wrong thing for the
    # designed chain: leaving it in would let a "de novo" design start from the
    # answer. Blank the binder rows explicitly.
    from pxf import atom37

    aatype = coupled_inputs.aatype.clone()
    aatype[:, roles.binder.to(aatype.device)] = atom37.UNKNOWN_AA_INDEX
    out["native_binder_identities_blanked"] = int(roles.n_binder)

    inputs = dict(coupled_inputs.fampnn_kwargs())
    inputs["aatype"] = aatype
    inputs["fixed_sequence_mask"] = roles.fixed_sequence_mask()[None].to(device)

    # --- 3. residual -------------------------------------------------------
    gen = torch.Generator().manual_seed(args.seed)
    a_token = a_token_at(driver, structure, args.sigma_b, device, gen)
    if a_token.shape[-2] != length:
        raise SystemExit(
            f"a_token covers {a_token.shape[-2]} rows, the complex has {length}"
        )
    gate = gate_by_name(args.gate)
    gate_value = 1.0 if gate is None else float(gate(args.sigma_b))
    sigma = torch.full((1,), float(args.sigma_b), device=device)
    with torch.no_grad():
        delta_h = binder_masked_residual(
            adapters, "matched", roles=roles, a_token=a_token, sigma=sigma, gate=gate,
        )
    h_v_probe = _probe_h_v(designer, inputs)
    residual_report = describe_residual(
        delta_h, roles, h_v=h_v_probe, gate_value=gate_value
    )
    residual_report["sigma_b"] = args.sigma_b
    residual_report["gate"] = args.gate
    out["checks"]["residual"] = {
        "pass": bool(
            delta_h is not None
            and residual_report["target_row_norm_max"] == 0.0
            and residual_report["binder_row_norm_mean"] > 0.0
            and residual_report["finite"]
        ),
        **residual_report,
    }

    # --- 2. no-op equivalence ---------------------------------------------
    bare, _ = _design(designer, inputs, None, seed=args.seed)
    hooked, noop_stats = _design(designer, inputs, None, seed=args.seed)
    coord_delta = float((bare.coords_af2 - hooked.coords_af2).abs().max())
    out["checks"]["noop"] = {
        "pass": bool(coord_delta <= NOOP_TOLERANCE_ANGSTROM
                     and bare.sequence == hooked.sequence),
        "max_coord_delta_angstrom": coord_delta,
        "tolerance": NOOP_TOLERANCE_ANGSTROM,
        "sequence_identical": bare.sequence == hooked.sequence,
        "hook_calls": noop_stats.get("calls"),
        "hook_applied": noop_stats.get("applied"),
        "note": "delta_h=None must traverse the wrapper and change nothing",
    }

    # --- 4. counters + coupled arm ----------------------------------------
    coupled, stats = _design(designer, inputs, delta_h, seed=args.seed)
    expected = args.seq_steps + (0 if args.no_repack else 1)
    out["checks"]["counters"] = {
        "pass": bool(stats.get("calls") == expected
                     and stats.get("applied") == expected),
        "calls": stats.get("calls"),
        "applied": stats.get("applied"),
        "expected": expected,
        "note": f"{args.seq_steps} unmasking steps"
                f"{'' if args.no_repack else ' + 1 final repack'}",
    }

    # --- 5. target invariance ---------------------------------------------
    target_rows = roles.target
    aatype_same = bool(
        torch.equal(bare.aatype[target_rows], coupled.aatype[target_rows])
    )
    target_coord_delta = float(
        (bare.coords_af2[target_rows] - coupled.coords_af2[target_rows]).abs().max()
    )
    binder_coord_delta = float(
        (bare.coords_af2[roles.binder] - coupled.coords_af2[roles.binder]).abs().max()
    )
    out["checks"]["invariance"] = {
        "pass": bool(aatype_same),
        "target_aatype_unchanged": aatype_same,
        "target_max_coord_delta": target_coord_delta,
        "binder_max_coord_delta": binder_coord_delta,
        "note": "the target is held fixed; the binder is allowed and expected "
                "to move between arms",
    }

    # --- 6. side-chain visibility -----------------------------------------
    out["checks"]["visibility"] = _visibility(
        designer, aatype, inputs["atom_mask"], roles, coupled
    )

    # --- 7. post-logit: one step cannot change the sequence ---------------
    if args.skip_single_step:
        out["checks"]["logits"] = {"pass": None, "note": "skipped"}
    else:
        out["checks"]["logits"] = _single_step_logits(
            designer, inputs, delta_h, args
        )

    out["sequences"] = {
        "uncoupled": bare.designed_sequence(),
        "coupled": coupled.designed_sequence(),
        "identical": bare.designed_sequence() == coupled.designed_sequence(),
    }
    return out


def _probe_h_v(designer, inputs) -> Optional[torch.Tensor]:
    """Capture one ``h_V`` so the residual has something to be relative to.

    Captured from the real model on the real input rather than estimated: the
    whole point of the number is to say whether the residual is large enough
    to matter against the signal it is actually added to.
    """
    captured: list[torch.Tensor] = []
    module = designer.model.denoiser.scn_diffusion_module
    original = module.sidechain_diffusion

    def probe(feature_dict, *a, **k):
        if not captured:
            captured.append(feature_dict["h_V"].detach().clone())
        raise _StopProbe

    module.sidechain_diffusion = probe
    try:
        designer.design(seed=0, **inputs)
    except _StopProbe:
        pass
    except Exception:  # noqa: BLE001 - probing must never fail the run
        logger.warning("h_V probe did not complete; relative_norm unavailable")
    finally:
        try:
            del module.sidechain_diffusion
        except AttributeError:
            module.sidechain_diffusion = original
    return captured[0] if captured else None


class _StopProbe(Exception):
    """Unwind out of `design` once h_V has been seen; nothing else is wanted."""


def _visibility(designer, aatype, atom_mask, roles, result) -> dict[str, Any]:
    """Can generated side chains reach the encoder at all?

    ``missing_atom_mask`` is computed once, before decoding, from identities
    that are UNKNOWN at every design position, and FaMPNN then reuses that one
    tensor for the whole loop. If UNKNOWN made a design position's side-chain
    slots read as "missing", those atoms would be masked out of the encoder
    permanently and the iterative context the coupling is supposed to enrich
    would be empty -- an inert sequence pathway with no error anywhere.
    """
    from pxf import atom37

    aatype_in = aatype.clamp_max(atom37.UNKNOWN_AA_INDEX)
    missing = designer.missing_atom_mask(aatype_in, atom_mask)
    binder = roles.binder.to(missing.device)
    sidechain_slots = [s for s in range(37) if s not in (0, 1, 2, 4)]
    hidden = float(missing[0][binder][:, sidechain_slots].sum())

    produced = float(
        result.atom_mask_af2[roles.binder.to(result.atom_mask_af2.device)][
            :, sidechain_slots
        ].sum()
    )
    return {
        "pass": bool(hidden == 0.0 and produced > 0.0),
        "binder_sidechain_slots_marked_missing": hidden,
        "binder_sidechain_atoms_produced": produced,
        "note": "missing_atom_mask is built once from UNKNOWN identities and "
                "reused for the whole loop; a non-zero count here would hide "
                "generated atoms from every later encoder call",
    }


def _single_step_logits(designer, inputs, delta_h, args) -> dict[str, Any]:
    """One unmasking step: a post-logit hook cannot change the sequence.

    The hook adds to ``h_V`` *inside* side-chain diffusion, which runs after
    that step's sequence logits are computed. With a single step there is no
    later step for the perturbed packing to feed into, so the sequence must be
    identical and only the coordinates may differ. A changed sequence here
    means the injection is not post-logit and every downstream interpretation
    of "sequence effects must come through side-chain context" is wrong.
    """
    previous = designer.seq_steps
    try:
        designer.seq_steps = 1
        bare, _ = _design(designer, inputs, None, seed=args.seed)
        coupled, stats = _design(designer, inputs, delta_h, seed=args.seed)
    finally:
        designer.seq_steps = previous

    same_sequence = bare.sequence == coupled.sequence
    coord_delta = float((bare.coords_af2 - coupled.coords_af2).abs().max())
    return {
        "pass": bool(same_sequence and stats.get("applied", 0) > 0),
        "sequence_identical": same_sequence,
        "max_coord_delta_angstrom": coord_delta,
        "hook_applied": stats.get("applied"),
        "note": "identical sequence with non-zero coordinate change is the "
                "signature of a correctly placed post-logit intervention",
    }


# ---------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dev-config", default=str(DEFAULT_DEV))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--adapters", default=None)
    parser.add_argument("--pxdesign-donor", default=None)
    parser.add_argument("--mmcif-dir", default=None)
    parser.add_argument("--fampnn-weights", default=None,
                        help="default: whatever the adapter was trained against")
    parser.add_argument("--sigma-b", type=float, default=0.429)
    parser.add_argument("--gate", default="one",
                        help="bs_policy gate name; 'one' is the adapter as trained")
    parser.add_argument("--seq-steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--psce-threshold", type=float, default=0.3)
    parser.add_argument("--no-repack", action="store_true")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw-weights", action="store_true",
                        help="use the adapter's raw weights instead of its EMA")
    parser.add_argument("--allow-unpinned-sources", action="store_true")
    parser.add_argument("--skip-single-step", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    manifest = {}
    if Path(args.manifest).is_file():
        manifest = json.loads(Path(args.manifest).read_text())
    args.adapters = args.adapters or _manifest_path(manifest, "a_bs")
    args.pxdesign_donor = args.pxdesign_donor or _manifest_path(manifest, "donor_pxdesign")
    if not args.adapters or not args.pxdesign_donor:
        raise SystemExit(
            "need --adapters and --pxdesign-donor, or a checkpoint manifest that "
            "resolves roles a_bs and donor_pxdesign "
            "(scripts/build_checkpoint_manifest.py)"
        )

    dev = yaml.safe_load(Path(args.dev_config).read_text())
    entries = dev["complexes"][: args.limit] if args.limit else dev["complexes"]
    for entry in entries:
        entry["_mmcif_dir"] = dev["mmcif_dir"]

    ctx = build(args)
    logger.info("loaded: %s", json.dumps(ctx["record"], indent=2, default=str))

    results = []
    for entry in entries:
        logger.info("=== %s", entry["id"])
        try:
            results.append(run_complex(entry, ctx, args))
        except Exception as exc:  # noqa: BLE001 - a failed complex is a finding
            logger.exception("%s failed", entry["id"])
            results.append({"id": entry["id"], "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "settings": {
            k: v for k, v in vars(args).items() if not k.startswith("_")
        },
        "environment": ctx["record"],
        "complexes": results,
    }

    names = ["load", "noop", "residual", "counters", "invariance", "visibility", "logits"]
    print("\nPhase 0")
    print(f"  load                 {'pass' if ctx['record'] else 'FAIL'}")
    failed = 0
    for result in results:
        if "error" in result:
            print(f"  {result['id']:<20} ERROR {result['error']}")
            failed += 1
            continue
        marks = []
        for name in names[1:]:
            check = result["checks"].get(name) or {}
            verdict = check.get("pass")
            marks.append(f"{name}={'pass' if verdict else ('skip' if verdict is None else 'FAIL')}")
            if verdict is False:
                failed += 1
        print(f"  {result['id']:<20} " + "  ".join(marks))
        residual = result["checks"].get("residual") or {}
        if residual.get("relative_norm") is not None:
            print(f"      residual/h_V = {residual['relative_norm']:.4g}  "
                  f"gate={residual.get('gate_value')}  "
                  f"binder rows touched {residual.get('n_rows_touched')}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "phase0_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(f"\nwrote {out / 'phase0_report.json'}")

    if failed:
        raise SystemExit(f"{failed} Phase 0 check(s) failed")
    print("\nPhase 0 clean: the coupling is live and the uncoupled path is reproduced")


if __name__ == "__main__":
    main()
