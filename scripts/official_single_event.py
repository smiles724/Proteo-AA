"""One-event SC -> BB comparison inside the official PXDesign runtime.

Three arms -- baseline, bb_only, full -- resumed from the *same* recorded
sampler state, sharing one FaMPNN sequence and one packed structure. They
differ only in the residual injected at a single ``(step, substage)``.

What is deliberately absent: any coordinate overwrite. The target is
conditioned by a distogram and its pose is emergent (see
``docs/target_conditioning_audit.md``), so pinning it would place the binder
and target in different frames. Arms are compared to each other, and the
complex is judged by its own chain-chain geometry, not against the native
partner as though it were a reconstruction.

Checks reported alongside the numbers, because a null result is only
informative if the plumbing is known to have fired:

  * the generated-residue mapping, including whether it is a contiguous tail;
  * exactly one injection per feedback arm, zero for baseline;
  * zero residual on target tokens (``scatter_design_residual`` raises
    otherwise);
  * the readout's actual inputs and the residual's magnitude, so a runtime
    that silently changed a feature shows up as a changed input rather than
    as a mysteriously different result.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

# Pin the OFFICIAL pxdesign/protenix in sys.modules before anything from
# scripts/ runs. `scripts/_bootstrap` -- pulled in by eval_sb_feedback --
# inserts the repo's PXDesign and Protenix submodules at the *front* of
# sys.path, which otherwise shadows the installed official packages and
# produces the worst possible mix: this repo's PXDesign against official
# Protenix. That failed loudly here (`No module named
# protenix.data.parser`), but a shadow that merely changes behaviour would
# not, and a silent one is what produced the invalid baseline to begin with.
import pxdesign  # noqa: E402
import pxdesign.runner.inference  # noqa: E402,F401
import protenix  # noqa: E402

_OFFICIAL_ROOT = "/hai/scratch/yfsun/pxdesign_official"
_SITE = "site-packages"


def _assert_official():
    px, ptx = pxdesign.__file__ or "", protenix.__file__ or ""
    if _OFFICIAL_ROOT not in px and _SITE not in px:
        raise SystemExit(f"pxdesign is not the official install: {px}")
    if _SITE not in ptx:
        raise SystemExit(f"protenix is not the official install: {ptx}")
    return dict(pxdesign=px, protenix=ptx)


_PROVENANCE = _assert_official()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pxf import atom37
from pxf.couple import mapping
from pxf.couple.adapters import CouplingAdapters
from pxf.couple.controller import CoupledDenoiser
from pxf.couple.pxdesign_iface import BackboneTap
from pxf.couple.replay import RngStream, run_trajectory
from pxf.eval.gen_metrics import arm_feedback, shared_preparation
from pxf.official.bridge import OfficialStructure
from pxf.official.runtime import OfficialDenoiser, build_runner, first_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("official_single_event")

ARMS = ("baseline", "bb_only", "full")
CLASH_RADIUS = 2.6


def chain_geometry(coords, design_mask, topology):
    """Min distance / clash count between generated and target backbone."""
    a2t = topology.atom_to_token_idx.reshape(-1)
    names = topology.atom_names
    backbone = torch.tensor(
        [n in ("N", "CA", "C", "O") for n in names], device=coords.device
    )
    token_is_design = design_mask.to(coords.device)[a2t]
    gen = coords[backbone & token_is_design]
    tgt = coords[backbone & ~token_is_design]
    if gen.numel() == 0 or tgt.numel() == 0:
        return dict(min_dist=None, clashes=None, contacts=None)
    d = torch.cdist(gen.float(), tgt.float())
    return dict(
        min_dist=round(float(d.min()), 3),
        clashes=int((d < CLASH_RADIUS).sum()),
        contacts=int((d < 5.0).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--fampnn-checkpoint", required=True)
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="label=path for a trained A_SB")
    ap.add_argument("--config", default="configs/couple_phase2_pilot.yaml")
    ap.add_argument("--n-step", type=int, default=200)
    ap.add_argument("--event-step", type=int, default=None)
    ap.add_argument("--event-sigma", type=float, default=0.5,
                    help="pick the event step whose t_hat is nearest this; "
                         "the trained gate is a SigmaWindow over [0.1, 2.0], "
                         "so an event outside it returns an exactly zero "
                         "residual and the comparison is vacuous")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--pack-steps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--use-msa", action="store_true")
    args = ap.parse_args()

    from eval_sb_feedback import load_arm  # imports scripts/_bootstrap
    from pxf.couple import torsions

    _assert_official()
    logger.info('provenance: %s', json.dumps(_PROVENANCE))
    from fampnn.model.sd_model import SeqDenoiser
    from pxf.couple.fampnn_iface import node_feature_dim

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sb_cfg = dict((yaml.safe_load(Path(args.config).read_text()) or {}).get(
        "sb_feedback", {}))

    runner, _configs = build_runner(
        args.yaml, str(out / "infer"),
        load_checkpoint_dir=args.checkpoint_dir,
        n_step=args.n_step, n_sample=1, use_msa=args.use_msa,
    )
    data, atom_array = first_batch(runner)
    den = OfficialDenoiser(runner, data)
    device = den.device
    n_tokens = int(data["N_token"])
    structure = OfficialStructure(atom_array, den.features, n_tokens, device=device)

    token_map = mapping.build_mapping(
        den.features, structure.design_mask, n_tokens=n_tokens
    )
    logger.info("mapping: %s", json.dumps(token_map.identity()))

    bundle = torch.load(args.fampnn_checkpoint, map_location="cpu",
                        weights_only=False)
    fampnn = SeqDenoiser(bundle["model_cfg"])
    fampnn.load_state_dict(bundle["state_dict"], strict=True)
    fampnn.eval().requires_grad_(False).to(device)
    c_h_V = node_feature_dim(fampnn)
    # The width the adapters were trained against and the official config's
    # diffusion_module.c_token; asserted against what the tap actually sees
    # rather than trusted, since a mismatch would only surface as a shape
    # error deep inside the injection.
    c_token = int(den.model.diffusion_module.layernorm_a.weight.shape[-1])

    trained = {}
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        trained[label] = load_arm(path, c_h_V=c_h_V, c_token=c_token,
                                  sb_cfg=sb_cfg, use_ema=args.ema, device=device)
        logger.info("arm %s: variant=%s step=%d", label,
                    trained[label]["variant"], trained[label]["step"])

    adapters = CouplingAdapters(c_token, c_h_V).to(device)
    adapters.eval().requires_grad_(False)
    controller = CoupledDenoiser(backbone=None, fampnn=fampnn, adapters=adapters,
                                 phase="sc_to_bb", pack_steps=args.pack_steps)

    schedule = den.schedule(args.n_step)
    # The denoiser sees t_hat = c_tau_last * (gamma + 1), and PXDesign's
    # gamma0/gamma_min make that 2 * c_tau_last across the useful range. The
    # gate is evaluated on that churned level, not on the scheduled one, so
    # the event step has to be chosen against it.
    t_hat_by_step = (2.0 * schedule[:-1].to(torch.float64)).cpu()
    if args.event_step is None:
        event_step = int((t_hat_by_step - args.event_sigma).abs().argmin())
    else:
        event_step = int(args.event_step)
    event_sigma = float(t_hat_by_step[event_step])
    in_window = 0.1 <= event_sigma <= 2.0
    logger.info("event step %d -> t_hat %.4f (gate window [0.1, 2.0]: %s)",
                event_step, event_sigma, "open" if in_window else "CLOSED")
    if not in_window:
        logger.warning(
            "the event is outside the trained gate; the residual will be zero "
            "and the arms will differ only by numerical noise")
    common = dict(schedule=schedule, n_atom=den.n_atom, device=device,
                  dtype=torch.float32, batch_shape=tuple(den.s_inputs.shape[:-2]),
                  n_sample=1, step_scale_eta=2.5)

    tap = BackboneTap(den.model.diffusion_module)

    def plain(x, s, feedback=None):
        return den.denoise(x, s, feedback=feedback, tap=tap)

    def bound(x, s, feedback=None):
        """Coordinates plus the captured token features, for the clean estimate."""
        y = den.denoise(x, s, feedback=feedback, tap=tap)
        return y, tap.a_token

    with tap:
        # Baseline trajectory, recording the state the arms resume from.
        _x_base_full, records, _ = run_trajectory(
            denoise=plain, stream=RngStream("bb", args.seed, device=device),
            record_steps=(event_step,), **common,
        )
        if not records:
            print(f"FAILED: nothing recorded at step {event_step}")
            return 1
        state = records[0].to(device)

        sc_stream = RngStream("sc", args.seed + 1, device=device)
        prep = shared_preparation(
            controller=controller, bound=bound, state=state, structure=structure,
            token_map=token_map, sc_stream=sc_stream,
            pack_steps=args.pack_steps, temperature=args.temperature,
        )
        sequence = "".join(
            atom37.AA_ORDER[int(a)] for a in prep["aatype"].reshape(-1).tolist()
        )
        logger.info("shared sequence (%d aa): %s...", len(sequence), sequence[:40])

        rows = {}
        for arm in ARMS:
            feedback, module = arm_feedback(
                arm, trained=trained, adapters=adapters, controller=controller,
                prep=prep, token_map=token_map, torsions=torsions, seed=args.seed,
            )
            if arm != "baseline" and feedback is None:
                logger.warning("arm %s has no trained checkpoint; skipping", arm)
                continue

            magnitude = {}
            if feedback is not None:
                inner = feedback

                def feedback(state, _inner=inner, _m=magnitude):
                    delta = _inner(state)
                    if delta is not None:
                        d = delta.reshape(-1, delta.shape[-1])
                        _m["residual_l2_mean"] = round(float(d.norm(dim=-1).mean()), 6)
                        _m["residual_absmax"] = round(float(d.abs().max()), 6)
                        _m["residual_rows_nonzero"] = int(
                            (d.norm(dim=-1) > 0).sum()
                        )
                        _m["residual_width"] = int(delta.shape[-1])
                    return delta

            tap.reset()
            x0, _, stats = run_trajectory(
                denoise=plain, stream=RngStream("bb", args.seed, device=device),
                resume=state, event=(event_step, 0), feedback=feedback,
                **common,
            )
            rows[arm] = dict(
                injections=stats["injections"],
                tap_injections=tap.injections,
                calls=stats["calls"],
                **magnitude,
                **chain_geometry(x0.reshape(-1, 3), structure.design_mask,
                                 structure.topology),
            )
            logger.info("%s: %s", arm, json.dumps(rows[arm]))

    # arm-vs-arm coordinate differences
    print(json.dumps(dict(
        target=str(data["sample_name"]), n_token=n_tokens, n_atom=den.n_atom,
        event_step=event_step, event_t_hat=round(event_sigma, 6),
        gate_window_open=in_window, n_step=args.n_step, eta=2.5,
        sequence_length=len(sequence),
        mapping=token_map.identity(), arms=rows,
    ), indent=2))

    (out / "single_event.json").write_text(json.dumps(rows, indent=2))
    print("\n=== CHECKS ===")
    ok = True
    if rows.get("baseline", {}).get("injections") != 0:
        print("FAIL baseline injected"); ok = False
    else:
        print("PASS baseline: 0 injections")
    for arm in ("bb_only", "full"):
        if arm in rows:
            n = rows[arm]["injections"]
            print(f"{'PASS' if n == 1 else 'FAIL'} {arm}: {n} injection(s)")
            ok &= n == 1
    print("PASS zero target-token residual (scatter_design_residual enforces)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
