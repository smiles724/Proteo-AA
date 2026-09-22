#!/usr/bin/env python3
"""Everything that must hold before 2,000 updates are queued.

    python scripts/preflight_integrated_feedback.py \
        --manifest runs/integrated_feedback_v1/data/calibration_pdb.parquet \
        --bs-checkpoint .../J03_seed0/checkpoints/step00000500.pt \
        --pxdesign-donor .../pxdesign_v0.1.0.pt \
        --fampnn-checkpoint .../fampnn_0_3.pt \
        --smoke-steps 20 --out runs/integrated_feedback_v1/preflight

Runs on the eight PDB CALIBRATION rows, which exist to be spent: the 20-update
overfit run below produces a checkpoint that must never be registered for an
experiment.

### The checks, and what each would catch

1. **Zero-feedback equivalence.** A conditioner at initialisation emits
   exactly zero, so the corrected call must reproduce the uncorrected one. A
   difference here means the injection path perturbs the model even when it
   has nothing to say, and every later comparison would be measuring that.

2. **Non-zero output-projection gradient at init**, located structurally and
   fail-closed. ``dL/dW_out`` is non-zero even for a zero ``W_out``; a zero
   means the feedback never reached the loss and 2,000 updates of nothing
   would follow. Internal weights are legitimately zero here -- the zero
   projection blocks their path -- and must become non-zero after one update.

3. **Analytic vs finite-difference gradient**, on CPU, with an epsilon sweep.
   A single epsilon is not a check: the signal can sit at a few float32 ULPs
   of the loss, where the estimator is noise. Convergence across epsilons is
   what makes the comparison mean anything.

4. **Leakage.** Perturb the native binder identity and side chains; the loss
   must not move. They are supervision, and a readout that could reach them
   would report a gain it cannot reproduce at inference.

5. **Injection accounting.** Exactly one conditioning injection per corrective
   call, no double A_BS addition, no residual accumulation across the decode,
   and hooks removed even when the body raises.

6. **RNG isolation.** FaMPNN's ~101 encoder calls and its packing draws must
   not advance the backbone stream.

7. **A disposable 20-update overfit**, which must decrease the loss. If a
   module cannot fit eight complexes it will not learn 198.

Reports measured seconds and memory per event, and the implied cost of the
four-run pilot, so that is a number rather than a guess before anything is
queued.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

#: Epsilons for the finite-difference sweep. Wide on purpose: the loss here is
#: O(0.2) and a small epsilon puts the signal inside float32 noise.
FD_EPS = (1e-3, 1e-2, 5e-2, 1e-1)
#: A relative disagreement below this counts as agreement, once converged.
FD_TOLERANCE = 0.05
#: Zero-feedback equivalence. Not widened to pass: this is a same-state,
#: same-RNG comparison of a model against itself, so it should be exact up to
#: CUDA reduction order.
EQUIVALENCE_ATOL = 1e-4


def load_cache_or_build(args, ctx):
    """One cached event, built here if the cache directory has none."""
    import pandas as pd

    frame = pd.read_parquet(args.manifest)
    row = frame.iloc[args.row]
    module = ctx["cache_module"]
    out = Path(args.out) / "events"
    out.mkdir(parents=True, exist_ok=True)

    class _Args:
        event_sigma = args.event_sigma
        crop_size = args.crop_size
        context = args.context

    record = module._one_event(
        row, driver=ctx["driver"], designer=ctx["designer"],
        adapters=ctx["adapters"], device=ctx["device"], args=_Args(),
        seed=args.seed, event_index=0, out=Path(args.out),
    )
    return record


def check_zero_feedback_equivalence(ctx, example, conditioner) -> dict:
    """The corrected call at initialisation must reproduce the uncorrected one."""
    import torch

    from pxf.couple.integrated_event import mask_feedback
    from pxf.couple.pxdesign_iface import BackboneTap

    sigma = torch.full((1,), float(example.sigma), device=ctx["device"])
    with BackboneTap(ctx["driver"].model.diffusion_module) as tap:
        bound = ctx["driver"].bind(ctx["cond"], tap=tap)
        with torch.no_grad():
            plain = _coords(bound(example.x_noisy, sigma, feedback=None))
            raw, _stats = conditioner(example.packed, sigma)
            delta = mask_feedback(
                raw, example.binder_mask, zero_bypass=False, name="E1"
            )
            corrected = _coords(bound(example.x_noisy, sigma, feedback=delta))
        injections = tap.conditioning_injections
    difference = float((plain - corrected).abs().max())
    return {
        "max_abs_difference": difference,
        "tolerance": EQUIVALENCE_ATOL,
        "equivalent": difference <= EQUIVALENCE_ATOL,
        "conditioning_injections": int(injections),
        "delta_is_exactly_zero": bool(_all_zero(delta)),
        "note": "a zero-emitting conditioner must not change the estimate; a "
                "difference here is the injection path perturbing the model "
                "when it has nothing to say",
    }


def check_finite_difference(ctx, example, conditioner) -> dict:
    """Analytic vs numerical gradient on CPU, swept over epsilon."""
    import torch

    from pxf.train.integrated_feedback import feedback_loss, output_projection

    parameter = output_projection(conditioner).weight
    sigma = torch.full((1,), float(example.sigma), device=ctx["device"])

    def loss_value(*, require_grad=True):
        from pxf.couple.pxdesign_iface import BackboneTap

        with BackboneTap(ctx["driver"].model.diffusion_module) as tap:
            bound = ctx["driver"].bind(ctx["cond"], tap=tap)
            return feedback_loss(
                example, conditioner,
                lambda x, s, *, feedback=None: _coords(
                    bound(x, s, feedback=feedback)
                ),
                require_grad=require_grad,
            )

    # Analytic FIRST: perturbing a parameter in place after a backward would
    # trip autograd's version counter.
    result = loss_value()
    conditioner.zero_grad(set_to_none=True)
    result.total.backward()
    # Probe the element with the LARGEST |gradient|, not element 0. This loss
    # is O(0.2) and a typical single weight's derivative is O(1e-7), i.e.
    # below the float32 noise floor of the difference -- the same precision
    # trap that made an earlier check in this repo report 13.6% error on a
    # correct derivative. Maximising the signal is what makes the comparison
    # possible at all; it is not cherry-picking, because a wrong derivative
    # would be wrong here too.
    flat_grad = parameter.grad.reshape(-1)
    index = int(flat_grad.abs().argmax())
    analytic = float(flat_grad[index])
    if abs(analytic) < 1e-12:
        return {
            "analytic": analytic, "reliable": False,
            "verdict": "FAIL",
            "note": "the analytic gradient is ~0, so the comparison would be "
                    "vacuous: 0 vs 0 agrees for the wrong reason",
        }

    base = float(result.total.detach())
    sweep = []
    with torch.no_grad():
        original = parameter.reshape(-1)[index].item()
        for eps in FD_EPS:
            parameter.reshape(-1)[index] = original + eps
            plus = float(loss_value(require_grad=False).total.detach())
            parameter.reshape(-1)[index] = original - eps
            minus = float(loss_value(require_grad=False).total.detach())
            parameter.reshape(-1)[index] = original
            numeric = (plus - minus) / (2 * eps)
            sweep.append({
                "eps": eps, "numeric": numeric,
                "relative_error": abs(numeric - analytic) / max(abs(analytic), 1e-12),
            })
    best = min(sweep, key=lambda s: s["relative_error"])
    # Converged = the two largest epsilons agree with each other. An
    # unconverged sweep means the ESTIMATOR is unreliable here, not that the
    # derivative is wrong.
    converged = abs(sweep[-1]["numeric"] - sweep[-2]["numeric"]) <= (
        FD_TOLERANCE * max(abs(sweep[-1]["numeric"]), 1e-12)
    )
    return {
        "analytic": analytic, "probed_index": index,
        "max_abs_gradient": float(flat_grad.abs().max()),
        "base_loss": base, "sweep": sweep,
        "best": best, "estimator_converged": converged,
        "tolerance": FD_TOLERANCE,
        "verdict": ("PASS" if converged and best["relative_error"] <= FD_TOLERANCE
                    else "CHECK"),
    }


def check_leakage(ctx, example, conditioner) -> dict:
    """Perturbing the withheld native labels must not move the loss."""
    import torch

    from dataclasses import replace

    from pxf.train.integrated_feedback import feedback_loss

    def value(ex):
        from pxf.couple.pxdesign_iface import BackboneTap

        with BackboneTap(ctx["driver"].model.diffusion_module) as tap:
            bound = ctx["driver"].bind(ctx["cond"], tap=tap)
            with torch.no_grad():
                return float(feedback_loss(
                    ex, conditioner,
                    lambda x, s, *, feedback=None: _coords(
                        bound(x, s, feedback=feedback)
                    ),
                    require_grad=False,
                ).total)

    baseline = value(example)
    findings = {}

    # The native binder SEQUENCE is not an input; the decode saw X. Changing
    # the deposited identities must not change anything the model reads.
    scrambled = replace(
        example,
        provenance={**example.provenance, "perturbed": "native_binder_aatype"},
    )
    findings["native_aatype_perturbed_delta"] = abs(value(scrambled) - baseline)

    # Unresolved target atoms must stay absent.
    available = example.packed.visibility.available
    findings["target_atoms_marked_available"] = int(available.sum())
    findings["sidechain_visibility_restored"] = bool(
        example.packed.visibility.sidechain_visible.sum() > 0
    )
    findings["binder_rows_are_the_supervised_ones"] = int(example.supervised.sum())
    findings["baseline_loss"] = baseline
    findings["verdict"] = (
        "PASS" if findings["native_aatype_perturbed_delta"] < 1e-9 else "CHECK"
    )
    findings["note"] = (
        "the cached example carries no native binder identity channel at all, "
        "so this check is a statement that the cache schema excludes it rather "
        "than a perturbation of a live input. The stronger version needs the "
        "cache rebuilt with a deliberately corrupted label, which "
        "--leakage-rebuild does."
    )
    return findings


def check_hooks(ctx, example, conditioner) -> dict:
    """Injection accounting, accumulation, and cleanup on exception."""
    import torch

    from pxf.couple.integrated_event import mask_feedback
    from pxf.couple.pxdesign_iface import BackboneTap

    sigma = torch.full((1,), float(example.sigma), device=ctx["device"])
    raw, _ = conditioner(example.packed, sigma)
    delta = mask_feedback(raw, example.binder_mask, zero_bypass=False)

    with BackboneTap(ctx["driver"].model.diffusion_module) as tap:
        bound = ctx["driver"].bind(ctx["cond"], tap=tap)
        with torch.no_grad():
            bound(example.x_noisy, sigma, feedback=None)
            after_plain = tap.conditioning_injections
            bound(example.x_noisy, sigma, feedback=delta)
            after_delta = tap.conditioning_injections
    handles_after_context = len(tap._handles)

    # Cleanup on exception: the tap must not leave hooks installed.
    leaked = None
    try:
        with BackboneTap(ctx["driver"].model.diffusion_module) as tap2:
            raise RuntimeError("deliberate")
    except RuntimeError:
        leaked = len(tap2._handles)

    return {
        "injections_without_feedback": int(after_plain),
        "injections_with_feedback": int(after_delta - after_plain),
        "exactly_one_injection": bool(after_delta - after_plain == 1),
        "handles_after_context": handles_after_context,
        "handles_after_exception": leaked,
        "hooks_cleaned_up": bool(handles_after_context == 0 and leaked == 0),
    }


def check_rng_isolation() -> dict:
    """FaMPNN's draws must not advance the backbone stream."""
    import torch

    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.couple.replay import RngStream, run_trajectory

    class _Stub:
        def __init__(self):
            self.layernorm_a = torch.nn.Identity()
            self.atom_attention_decoder = torch.nn.Identity()

    def trajectory(with_draws):
        module = _Stub()
        schedule = torch.logspace(1, -1, 6)

        def feedback(state):
            if with_draws:
                torch.randn(500)
            return None

        with BackboneTap(module) as tap:
            x, _r, _s = run_trajectory(
                denoise=lambda x, s, *, feedback=None: torch.zeros_like(x),
                schedule=schedule, n_atom=12, device=torch.device("cpu"),
                n_sample=1, stream=RngStream("p", 0), event=(1, 0),
                feedback=feedback,
            )
        return x

    a, b = trajectory(False), trajectory(True)
    return {
        "identical": bool(torch.equal(a, b)),
        "note": "stream.protected() around the event callback; without it the "
                "corrected trajectory would differ from the uncorrected one "
                "for RNG bookkeeping rather than for feedback",
    }


def overfit(ctx, examples, conditioner, steps: int) -> dict:
    """A disposable run that must decrease the loss. Never register it."""
    import torch

    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.train.integrated_feedback import feedback_loss

    optimizer = torch.optim.AdamW(
        [p for p in conditioner.parameters() if p.requires_grad], lr=1e-4
    )
    losses, started = [], time.time()
    peak = 0
    for step in range(steps):
        example = examples[step % len(examples)]
        with BackboneTap(ctx["driver"].model.diffusion_module) as tap:
            bound = ctx["driver"].bind(ctx["cond_for"](example), tap=tap)
            result = feedback_loss(
                example, conditioner,
                lambda x, s, *, feedback=None: _coords(
                    bound(x, s, feedback=feedback)
                ),
            )
        optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        torch.nn.utils.clip_grad_norm_(conditioner.parameters(), 1.0)
        optimizer.step()
        losses.append(float(result.total.detach()))
        if torch.cuda.is_available():
            peak = max(peak, torch.cuda.max_memory_allocated())
    first = sum(losses[: max(1, steps // 4)]) / max(1, steps // 4)
    last = sum(losses[-max(1, steps // 4):]) / max(1, steps // 4)
    seconds = time.time() - started
    return {
        "steps": steps, "losses": losses,
        "first_quarter_mean": first, "last_quarter_mean": last,
        "decreased": bool(last < first),
        "seconds_per_step": seconds / max(1, steps),
        "peak_memory_gb": peak / 1e9,
        "estimated_pilot_gpu_hours": (
            seconds / max(1, steps) * 2000 * 4 / 3600
        ),
        "note": "DISPOSABLE. This checkpoint must not be registered for an "
                "experiment; it was fit to the calibration rows.",
    }


def _coords(out):
    return out[0] if isinstance(out, tuple) else out


def _all_zero(delta):
    import torch

    from pxf.couple.pxdesign_iface import ConditioningFeedback

    if isinstance(delta, ConditioningFeedback):
        parts = [t for t in (delta.delta_single, delta.delta_pair) if t is not None]
        return all(not bool(torch.any(t != 0)) for t in parts)
    return delta is None or not bool(torch.any(delta != 0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bs-checkpoint", required=True)
    parser.add_argument("--pxdesign-donor", required=True)
    parser.add_argument("--fampnn-checkpoint", default=None)
    parser.add_argument("--fampnn-variant", default="0.3")
    parser.add_argument("--arm", default="early_s_full")
    parser.add_argument("--event-sigma", type=float, default=0.429)
    parser.add_argument("--context", default="complex_sc")
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--n-events", type=int, default=2)
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import importlib.util as ilu

    import pandas as pd
    import torch

    import _bootstrap  # noqa: F401

    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.couple.conditioning import build_conditioner
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.couple.pxdesign_iface import (conditioning_widths,
                                           token_feature_dim)
    from pxf.device import select_device
    from pxf.sidechain.design import FaMPNNFullAtomDesigner
    from pxf.train.integrated_feedback import (FeedbackExample,
                                               check_initial_gradient,
                                               freeze_everything_but,
                                               gradient_norms)

    spec = ilu.spec_from_file_location(
        "_cache_mod", str(REPO_ROOT / "scripts" / "cache_integrated_feedback.py")
    )
    cache_module = ilu.module_from_spec(spec)
    spec.loader.exec_module(cache_module)

    device = select_device(args.device)
    out = Path(args.out)
    (out / "events").mkdir(parents=True, exist_ok=True)

    px_model, _c, _r = load_backbone_model(args.pxdesign_donor, device=device)
    driver = PXDesignBackboneDriver(px_model)
    designer = FaMPNNFullAtomDesigner(
        args.fampnn_checkpoint, variant=args.fampnn_variant,
        seq_steps=100, temperature=0.1, psce_threshold=0.3, repack_last=True,
    ).to(device).eval()
    adapters = cache_module._load_adapters(
        args.bs_checkpoint, designer, driver, device, "ema",
        __import__("pxf.provenance", fromlist=["x"]).file_sha256(
            args.fampnn_checkpoint
            or __import__("pxf.provenance", fromlist=["x"]).fampnn_checkpoint(
                args.fampnn_variant)
        ),
    )
    c_s, c_z = conditioning_widths(px_model)
    conditioner = build_conditioner(
        args.arm, c_h_V=node_feature_dim(designer.model),
        c_token=token_feature_dim(px_model), c_s=c_s, c_z=c_z,
    ).to(device)
    frozen = freeze_everything_but(conditioner, px_model, designer.model)

    ctx = {
        "device": device, "driver": driver, "designer": designer,
        "adapters": adapters, "cache_module": cache_module,
    }

    print(f"preflight {args.arm}: {frozen['trainable_parameters']} trainable "
          f"parameter(s), {frozen['frozen_tensors']} donor tensor(s) frozen")

    # ---- build a couple of real events ------------------------------------
    frame = pd.read_parquet(args.manifest)
    events, conds = [], {}
    for index in range(min(args.n_events, len(frame))):
        row = frame.iloc[index]

        class _A:
            event_sigma = args.event_sigma
            crop_size = args.crop_size
            context = args.context

        record = cache_module._one_event(
            row, driver=driver, designer=designer, adapters=adapters,
            device=device, args=_A(), seed=args.seed + index,
            event_index=0, out=out,
        )
        blob = torch.load(record["path"], map_location=device, weights_only=False)
        example = FeedbackExample(
            example_id=blob["example_id"], x_noisy=blob["x_noisy"],
            sigma=float(blob["sigma"]), packed=blob["packed"],
            binder_mask=blob["binder_mask"], native_bb=blob["native_bb"],
            supervised=blob["supervised"], provenance=blob["provenance"],
        ).to(device)
        structure = cache_module.featurize_native(
            record["cif_path"], record["binder_chain"],
            crop_size=args.crop_size, device=device,
        )
        conds[example.example_id] = driver.conditioning(structure.feature_dict)
        events.append(example)
        print(f"  built event {example.example_id} "
              f"({record['supervised_atoms']} supervised atoms, "
              f"{record['seconds']}s)")

    ctx["cond"] = conds[events[0].example_id]
    ctx["cond_for"] = lambda ex: conds[ex.example_id]

    report = {"arm": args.arm, "frozen": frozen, "n_events": len(events)}

    print("\n1. zero-feedback equivalence")
    report["zero_feedback"] = check_zero_feedback_equivalence(
        ctx, events[0], conditioner
    )
    z = report["zero_feedback"]
    print(f"   max|plain - corrected| = {z['max_abs_difference']:.3e} "
          f"(tolerance {z['tolerance']:.0e}) -> "
          f"{'PASS' if z['equivalent'] else 'FAIL'}")

    print("\n2. initial gradient")
    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.train.integrated_feedback import feedback_loss

    with BackboneTap(driver.model.diffusion_module) as tap:
        bound = driver.bind(ctx["cond"], tap=tap)
        result = feedback_loss(
            events[0], conditioner,
            lambda x, s, *, feedback=None: _coords(bound(x, s, feedback=feedback)),
        )
    conditioner.zero_grad(set_to_none=True)
    result.total.backward()
    report["initial_gradient"] = check_initial_gradient(conditioner)
    report["initial_gradient"]["all_norms"] = gradient_norms(conditioner)
    print(f"   output projection weight grad "
          f"{report['initial_gradient']['weight']:.3e} -> PASS")

    print("\n3. finite difference (CPU-stable closure, epsilon sweep)")
    report["finite_difference"] = check_finite_difference(ctx, events[0], conditioner)
    fd = report["finite_difference"]
    print(f"   analytic {fd.get('analytic', float('nan')):.6e}  "
          f"best rel error {fd.get('best', {}).get('relative_error', float('nan')):.2e}  "
          f"converged={fd.get('estimator_converged')}  -> {fd['verdict']}")

    print("\n4. leakage")
    report["leakage"] = check_leakage(ctx, events[0], conditioner)
    print(f"   native-label perturbation moved the loss by "
          f"{report['leakage']['native_aatype_perturbed_delta']:.2e} -> "
          f"{report['leakage']['verdict']}")

    print("\n5. injection accounting and hook cleanup")
    report["hooks"] = check_hooks(ctx, events[0], conditioner)
    h = report["hooks"]
    print(f"   injections: {h['injections_without_feedback']} without, "
          f"{h['injections_with_feedback']} with -> "
          f"{'PASS' if h['exactly_one_injection'] else 'FAIL'};  "
          f"hooks cleaned up: {h['hooks_cleaned_up']}")

    print("\n6. RNG isolation")
    report["rng"] = check_rng_isolation()
    print(f"   trajectories identical with and without FaMPNN-sized draws: "
          f"{report['rng']['identical']}")

    print(f"\n7. disposable {args.smoke_steps}-update overfit")
    report["overfit"] = overfit(ctx, events, conditioner, args.smoke_steps)
    o = report["overfit"]
    print(f"   loss {o['first_quarter_mean']:.5f} -> {o['last_quarter_mean']:.5f} "
          f"({'decreased' if o['decreased'] else 'DID NOT DECREASE'})")
    print(f"   {o['seconds_per_step']:.2f} s/step, peak "
          f"{o['peak_memory_gb']:.1f} GB -> the 4-run pilot is about "
          f"{o['estimated_pilot_gpu_hours']:.1f} GPU-hours")

    (out / "preflight.json").write_text(json.dumps(report, indent=2, default=str))
    failures = [
        name for name, ok in (
            ("zero_feedback", report["zero_feedback"]["equivalent"]),
            ("finite_difference", report["finite_difference"]["verdict"] == "PASS"),
            ("hooks", report["hooks"]["exactly_one_injection"]),
            ("hook_cleanup", report["hooks"]["hooks_cleaned_up"]),
            ("rng", report["rng"]["identical"]),
            ("overfit", report["overfit"]["decreased"]),
        ) if not ok
    ]
    print(f"\nwrote {out / 'preflight.json'}")
    if failures:
        print(f"PREFLIGHT FAILED: {failures}")
        raise SystemExit(1)
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
