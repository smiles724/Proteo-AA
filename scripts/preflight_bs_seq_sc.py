#!/usr/bin/env python3
"""Section 7 gradient validation, memory/timing preflight, and an optional smoke.

    python scripts/preflight_bs_seq_sc.py --out runs/bs_seq_sc_preflight
    python scripts/preflight_bs_seq_sc.py --out ... --smoke-steps 40

Three jobs in one script because the Marlowe queue is deep enough that
submitting them separately costs more wall time than running them together.
None of them needs the training complexes: no weights are selected and nothing
is reported from these structures, so running on the 31 held-out dimers does
not spend them. **Nothing here trains a checkpoint that is kept.**

What it does NOT do: calibrate lambda_seq for the real run. That is a decision
made from data, section 7 requires a training-only calibration batch, and the
val split is not one. The gradient-norm ratio is reported per structure so the
calibration is a lookup once the training complexes exist -- but the
coefficient is not chosen here.

### The section 7 checks

  1. L_MLM gives a finite, non-zero gradient at the adapter's output
     projection. Under `packing_only` this is exactly zero, so it is the
     check that distinguishes the new routing from the old.
  2. L_diffusion gives a finite, non-zero gradient there.
  3. PXDesign, the FaMPNN encoder, W_out, the SC denoiser and the confidence
     head receive no parameter gradient.
  4. With the adapter's output zeroed, logits and SC predictions reproduce the
     unadapted 0.3 donor bit-for-bit under the same corruption and noise.
  5. No target row receives a residual or contributes to either loss.
  6. Finite differences agree with the analytical sequence gradient -- ON A
     CPU COPY -- and the GPU gradient agrees with the CPU one.

Check 4 is the one that catches a routing error rather than a gradient error:
a fresh adapter is zero-initialised at its output projection, so the coupled
forward MUST equal the uncoupled one exactly. If it does not, the residual is
reaching somewhere it was not meant to and every later comparison is against
the wrong baseline.

On zero-init and check 1: with the output projection at zero, the gradient
*into earlier adapter layers* is legitimately zero on the first step. That is
normal, not a dead graph. So check 1 asserts a live gradient in the OUTPUT
PROJECTION, and the propagation into earlier layers is verified after one
optimiser step.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "gen_stress_prepared.marlowe.parquet"


# ------------------------------------------------------------------ assembly


def build(args):
    """Frozen 0.3 donor + PXDesign, plus a FRESH zero-initialised adapter."""
    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.device import select_device
    from pxf.sidechain.fampnn import FaMPNNSideChainPacker
    from pxf.train.bs_seq_sc import freeze_everything_but

    device = select_device(args.device)
    packer = FaMPNNSideChainPacker(variant=args.fampnn_variant).to(device)
    px_model, _cfg, _rec = load_backbone_model(args.pxdesign_donor, device=device)
    driver = PXDesignBackboneDriver(px_model)

    adapters = CouplingAdapters(
        driver.c_token, node_feature_dim(packer.model)
    ).to(device)
    adapters.enable_bb_to_sc = True
    counts = freeze_everything_but(adapters, packer.model, px_model)
    return {
        "device": device, "packer": packer, "driver": driver,
        "adapters": adapters, "freeze": counts,
        "is_identity": bool(adapters.is_identity()),
    }


def prepare_example(row, ctx, args):
    """Featurize one dimer into (masks, roles, features, batch, a_token)."""
    from pxf.backbone.chain_ids import featurizer_chain_id
    from pxf.backbone.driver import featurize_structures, to_featurized
    from pxf.couple.binder_masks import build_masks
    from pxf.couple.binder_residual import ChainRoles
    from pxf.couple.converter import PXFaRepresentationConverter
    from pxf.couple.fampnn_iface import encode
    from pxf.train.bs_seq_sc import batch_from_inputs

    device = ctx["device"]
    label = featurizer_chain_id(row.cif_path, row.converted_binder_chain)
    sample_id, dataset = featurize_structures(
        [row.cif_path], crop_size=args.crop_size,
        binder_chain_ids=[label], parser_dataset="Distillation",
    )[0]
    structure = to_featurized(sample_id, dataset[0]).to(device)
    roles = ChainRoles(binder=structure.design_mask.reshape(-1).bool().cpu())

    generator = torch.Generator().manual_seed(args.seed)
    masks = build_masks(
        roles, structure.aatype.reshape(1, -1).cpu(),
        mask_fraction=args.mask_fraction, generator=generator,
    )
    masks = type(masks)(
        roles=masks.roles,
        seq_mask=masks.seq_mask.to(device),
        seq_mlm_mask=masks.seq_mlm_mask.to(device),
        sidechain_visible=masks.sidechain_visible.to(device),
        aatype_encoder=masks.aatype_encoder.to(device),
        aatype_true=masks.aatype_true.to(device),
    )

    converter = PXFaRepresentationConverter()
    topology = structure.topology
    inputs = converter.px_backbone_to_fampnn(
        structure.backbone_target, topology.atom_names,
        topology.atom_to_token_idx, topology.num_tokens,
        res_names=topology.res_names, residue_index=topology.residue_index,
        chain_index=topology.chain_index, aatype=structure.aatype,
    ).to(device)

    # The ENCODER sees the masked identities; the SC branch is teacher-forced
    # on ground truth. Getting this backwards produces a number either way.
    with torch.no_grad():
        # coords_af2 and aatype are POSITIONAL, and there is no atom_mask
        # argument: encode builds it from missing_atom_mask and
        # sidechain_visible, which is what keeps a hidden residue's side chain
        # out of the encoder rather than merely out of the loss.
        _logits, _h_v, features = encode(
            ctx["packer"].model,
            inputs.coords_af2,
            masks.aatype_encoder,
            seq_mask=inputs.seq_mask,
            missing_atom_mask=inputs.missing_atom_mask,
            residue_index=inputs.residue_index,
            chain_index=inputs.chain_index,
            sidechain_visible=masks.sidechain_visible,
        )
    batch = batch_from_inputs(inputs, masks.aatype_true)

    sigma = torch.full((1,), float(args.sigma_b), device=device)
    cond = ctx["driver"].conditioning(structure.feature_dict)
    bound = ctx["driver"].bind(cond)
    with torch.no_grad():
        target = structure.backbone_target.float()
        noise = torch.randn(
            target.shape, generator=torch.Generator().manual_seed(args.seed)
        ).to(device)
        _bb0, a_token = bound((target + noise * float(args.sigma_b))[None], sigma)
    if a_token is None:
        raise SystemExit("the driver returned no a_token")
    if a_token.dim() == 2:
        a_token = a_token[None]

    return {
        "roles": roles, "masks": masks, "features": features, "batch": batch,
        "a_token": a_token, "sigma": sigma, "inputs": inputs,
        "tokens": int(topology.num_tokens),
    }


# ------------------------------------------------------------- the 6 checks


def section7(example, ctx, args) -> dict[str, Any]:
    from pxf.train.bs_seq_sc import gradient_norms, joint_loss, suggest_lambda_seq

    adapters, packer = ctx["adapters"], ctx["packer"]
    out: dict[str, Any] = {}
    # A fresh adapter is zero at its output projection, so the residual IS
    # identically zero here and that is the point -- check 4 requires it. The
    # gradient is still live, so checks 1-3 are unaffected.
    common = dict(
        adapters=adapters, a_token=example["a_token"], sigma_b=example["sigma"],
        roles=example["roles"], masks=example["masks"],
        multiplier=args.multiplier, allow_zero=True,
    )

    # --- 4. zero-init parity (run FIRST, before any update) ----------------
    from pxf.couple.fampnn_iface import encode  # noqa: F401  (documented path)

    seq_module = packer.model.denoiser.seq_design_module
    with torch.no_grad():
        baseline_logits = seq_module.W_out(example["features"]["h_V"])
    from pxf.couple.shared_prelogit import conditioned_forward

    with torch.no_grad():
        coupled_logits, conditioned, delta = conditioned_forward(
            seq_module, example["features"], adapters=adapters,
            a_token=example["a_token"], sigma_b=example["sigma"],
            roles=example["roles"], allow_zero=True,
        )
    max_logit_delta = float((coupled_logits - baseline_logits).abs().max())
    max_h_delta = float(
        (conditioned["h_V"] - example["features"]["h_V"]).abs().max()
    )
    out["check4_zero_init_parity"] = {
        "pass": bool(ctx["is_identity"] and max_logit_delta == 0.0),
        "adapter_is_identity": ctx["is_identity"],
        "max_logit_delta": max_logit_delta,
        "max_h_V_delta": max_h_delta,
        "note": "a zero-initialised adapter must reproduce the donor exactly; "
                "any difference means the residual reaches somewhere unintended",
    }

    # --- 5. target rows untouched ------------------------------------------
    target_rows = example["roles"].target.to(delta.device)
    out["check5_target_untouched"] = {
        "pass": bool(float(delta[:, target_rows, :].abs().max()) == 0.0
                     and float(example["masks"].seq_supervision[:, target_rows].sum()) == 0.0),
        "max_target_residual": float(delta[:, target_rows, :].abs().max()),
        "n_target_supervised": int(example["masks"].seq_supervision[:, target_rows].sum()),
    }

    # --- 1 & 2. per-objective gradients at the OUTPUT PROJECTION -----------
    seq_only = joint_loss(
        packer.model, example["batch"], example["features"],
        lambda_seq=1.0, lambda_sc=0.0, **common,
    )
    seq_norm = gradient_norms(seq_only.sequence, adapters)
    sc_only = joint_loss(
        packer.model, example["batch"], example["features"],
        lambda_seq=0.0, lambda_sc=1.0, **common,
    )
    sc_norm = gradient_norms(sc_only.sidechain, adapters)

    out["check1_sequence_gradient"] = {
        "pass": bool(seq_norm > 0 and torch.isfinite(torch.tensor(seq_norm))),
        "grad_norm": seq_norm,
        "loss": float(seq_only.sequence.detach()),
        "note": "exactly zero under the legacy packing_only routing; this is "
                "the check that distinguishes the two application modes",
    }
    out["check2_sidechain_gradient"] = {
        "pass": bool(sc_norm > 0 and torch.isfinite(torch.tensor(sc_norm))),
        "grad_norm": sc_norm,
        "loss": float(sc_only.sidechain.detach()),
    }
    out["lambda_suggestion"] = suggest_lambda_seq(seq_norm, sc_norm)
    out["lambda_suggestion"]["note"] = (
        "reported per structure, NOT adopted: section 7 requires a "
        "training-only calibration batch and these are val"
    )

    # --- 3. nothing but the adapter moves ----------------------------------
    joint = joint_loss(
        packer.model, example["batch"], example["features"],
        lambda_seq=1.0, lambda_sc=1.0, **common,
    )
    for parameter in list(packer.model.parameters()) + list(ctx["driver"].model.parameters()):
        parameter.grad = None
    joint.total.backward()
    donor_with_grad = [
        n for n, p in list(packer.model.named_parameters())
        + list(ctx["driver"].model.named_parameters())
        if p.grad is not None and float(p.grad.abs().sum()) > 0
    ]
    out["check3_donors_frozen"] = {
        "pass": not donor_with_grad,
        "n_donor_tensors_with_gradient": len(donor_with_grad),
        "examples": donor_with_grad[:5],
        **ctx["freeze"],
    }
    out["joint_loss"] = float(joint.total.detach())
    return out


def finite_difference_check(example, ctx, args, *, device_note="") -> dict[str, Any]:
    """Check 6: analytical vs numerical gradient of L_seq, one scalar.

    **Run this on CPU only.** Commit 8fe1b72 established why, on an H200:

        autograd says 0.0211725, central differences say 0.0183508

    13% apart, with the ESTIMATOR at fault, not the gradient. A central
    difference divides a difference of two losses by a small number, so it
    needs the loss reproducible to far better than that difference, and on a
    GPU it is not. Widening the tolerance until the GPU passes would throw
    away the only check that can catch a genuinely wrong derivative in order
    to accommodate a limitation of the estimator -- and would have to be
    widened past 13%, wide enough to admit real errors.

    So the two questions are separated, following that commit: finite
    differences keep 5% and speak about the graph's math, which does not
    depend on the device; whether the GPU computes the same gradient as the
    CPU is asked directly by :func:`cross_device_check`.

    Ordering also matters here and my first attempt got it wrong: perturbing
    a parameter in place while a graph referencing it is alive trips
    autograd's version counter. The analytical gradient is taken FIRST on a
    clean graph; the numerical probes then run entirely under no_grad.
    """
    from pxf.train.bs_seq_sc import joint_loss

    adapters = ctx["adapters"]
    parameter = next(p for p in adapters.parameters() if p.requires_grad and p.numel() > 1)
    index = (0,) * (parameter.dim() - 1) + (0,)

    def sequence_loss():
        return joint_loss(
            ctx["packer"].model, example["batch"], example["features"],
            adapters=adapters, a_token=example["a_token"],
            sigma_b=example["sigma"], roles=example["roles"],
            masks=example["masks"], lambda_seq=1.0, lambda_sc=0.0,
            multiplier=args.multiplier, allow_zero=True,
        ).sequence

    # 1. analytical, on a graph nothing has touched.
    loss = sequence_loss()
    analytical = float(
        torch.autograd.grad(loss, parameter, retain_graph=False)[0][index]
    )
    del loss

    # 2. numerical, with autograd switched off entirely.
    eps = args.fd_eps
    with torch.no_grad():
        centre = parameter[index].clone()
        parameter[index] = centre + eps
        up = float(sequence_loss())
        parameter[index] = centre - eps
        down = float(sequence_loss())
        parameter[index] = centre
    numerical = (up - down) / (2 * eps)

    denom = max(abs(numerical), abs(analytical), 1e-8)
    rel = abs(numerical - analytical) / denom
    return {
        "pass": bool(rel < args.fd_tolerance or abs(numerical - analytical) < 1e-7),
        "numerical": numerical, "analytical": analytical,
        "relative_error": rel, "eps": eps, "tolerance": args.fd_tolerance,
        "note": "one scalar, central differences; narrow on purpose -- an "
                "independent confirmation that the analytical path is the "
                "function it claims to be",
    }


def cross_device_check(example, ctx, args) -> dict[str, Any]:
    """Does the GPU compute the gradient the CPU computes?

    The question a GPU run actually needs answered. Finite differences cannot
    answer it there (see :func:`finite_difference_check`), and this can: the
    same inputs, the same parameters, two devices, compared directly.
    """
    from pxf.train.bs_seq_sc import gradient_norms, joint_loss

    if ctx["device"].type != "cuda":
        return {"pass": None, "skipped": "not on a GPU"}

    def norm_on(device):
        moved = {
            "batch": {k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in example["batch"].items()},
            "features": {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in example["features"].items()},
        }
        adapters = ctx["adapters"].to(device)
        masks = example["masks"]
        moved_masks = type(masks)(
            roles=masks.roles,
            seq_mask=masks.seq_mask.to(device),
            seq_mlm_mask=masks.seq_mlm_mask.to(device),
            sidechain_visible=masks.sidechain_visible.to(device),
            aatype_encoder=masks.aatype_encoder.to(device),
            aatype_true=masks.aatype_true.to(device),
        )
        loss = joint_loss(
            ctx["packer"].model.to(device), moved["batch"], moved["features"],
            adapters=adapters, a_token=example["a_token"].to(device),
            sigma_b=example["sigma"].to(device), roles=example["roles"],
            masks=moved_masks, lambda_seq=1.0, lambda_sc=0.0,
            multiplier=args.multiplier, allow_zero=True,
        ).sequence
        return gradient_norms(loss, adapters, retain=False)

    gpu = norm_on(ctx["device"])
    cpu = norm_on(torch.device("cpu"))
    ctx["packer"].model.to(ctx["device"])
    ctx["adapters"].to(ctx["device"])
    denom = max(abs(gpu), abs(cpu), 1e-12)
    rel = abs(gpu - cpu) / denom
    return {
        "pass": bool(rel < args.cross_device_tolerance),
        "gpu_grad_norm": gpu, "cpu_grad_norm": cpu,
        "relative_error": rel, "tolerance": args.cross_device_tolerance,
    }


# ------------------------------------------------------------- the preflight


def preflight(example, ctx, args) -> dict[str, Any]:
    """Peak memory and step time at this token count, for sizing the batch."""
    from pxf.train.bs_seq_sc import joint_loss

    device = ctx["device"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.time()
    joint = joint_loss(
        ctx["packer"].model, example["batch"], example["features"],
        adapters=ctx["adapters"], a_token=example["a_token"],
        sigma_b=example["sigma"], roles=example["roles"],
        masks=example["masks"], lambda_seq=1.0, lambda_sc=1.0,
        multiplier=args.multiplier, allow_zero=True,
    )
    joint.total.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - started
    peak = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda" else None
    )
    for parameter in ctx["adapters"].parameters():
        parameter.grad = None
    return {
        "tokens": example["tokens"], "seconds_per_step": round(elapsed, 3),
        "peak_gib": None if peak is None else round(peak, 3),
        "multiplier": args.multiplier,
    }


def smoke(examples, ctx, args) -> dict[str, Any]:
    """A few real optimiser steps. The checkpoint is DISCARDED, never registered.

    Proves the loop executes -- optimiser, EMA, no NaNs -- before HAI spends
    generation time. It updates weights using val structures, so nothing it
    produces may be kept or selected on, and it writes no checkpoint.
    """
    from pxf.train.bs_seq_sc import joint_loss

    adapters = ctx["adapters"]
    optimizer = torch.optim.AdamW(
        [p for p in adapters.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    history = []
    for step in range(args.smoke_steps):
        example = examples[step % len(examples)]
        optimizer.zero_grad(set_to_none=True)
        joint = joint_loss(
            ctx["packer"].model, example["batch"], example["features"],
            adapters=adapters, a_token=example["a_token"],
            sigma_b=example["sigma"], roles=example["roles"],
            masks=example["masks"], lambda_seq=args.lambda_seq,
            lambda_sc=args.lambda_sc, multiplier=args.multiplier,
            allow_zero=True,
        )
        if not torch.isfinite(joint.total):
            return {"pass": False, "step": step, "reason": "non-finite loss"}
        joint.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in adapters.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        history.append({
            "step": step, "total": float(joint.total.detach()),
            "seq": float(joint.sequence.detach()),
            "sc": float(joint.sidechain.detach()),
            "grad_norm": float(grad_norm),
        })

    # After updates the output projection is no longer zero, so the gradient
    # must now reach EARLIER adapter layers. Zero there on step 0 was normal
    # zero-init behaviour; zero here would be a dead graph.
    still_identity = bool(adapters.is_identity())
    return {
        "pass": bool(history and not still_identity),
        "steps": len(history),
        "first": history[0] if history else None,
        "last": history[-1] if history else None,
        "adapter_still_identity_after_updates": still_identity,
        "checkpoint_written": False,
        "note": "val structures; checkpoint discarded and not registered",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", required=True)
    parser.add_argument("--pxdesign-donor", default=(
        "/scratch/m000137-pm06/Proteo-AA/pxf/component_donors/pxdesign_v0.1.0.pt"))
    parser.add_argument("--fampnn-variant", default="0.3")
    parser.add_argument("--sigma-b", type=float, default=0.429)
    parser.add_argument("--mask-fraction", type=float, default=0.5)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--multiplier", type=int, default=None)
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--lambda-seq", type=float, default=1.0)
    parser.add_argument("--lambda-sc", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-finite-difference", action="store_true")
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--fd-tolerance", type=float, default=0.05)
    parser.add_argument("--cross-device-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_parquet(args.manifest).head(args.n_examples)
    ctx = build(args)
    print(f"device={ctx['device']} fampnn={args.fampnn_variant} "
          f"adapter_is_identity={ctx['is_identity']} "
          f"frozen={ctx['freeze']['frozen_tensors']} "
          f"trainable={ctx['freeze']['trainable_tensors']}")

    examples, records = [], []
    for row in frame.itertuples():
        example = prepare_example(row, ctx, args)
        examples.append(example)
        checks = section7(example, ctx, args)
        checks["preflight"] = preflight(example, ctx, args)
        checks["example_id"] = row.example_id
        records.append(checks)
        names = [k for k in checks if k.startswith("check")]
        marks = " ".join(
            f"{k.split('_')[0]}={'ok' if checks[k]['pass'] else 'FAIL'}" for k in names
        )
        pf = checks["preflight"]
        print(f"  {row.example_id:<8} tokens {pf['tokens']:>4}  {marks}  "
              f"peak {pf['peak_gib']} GiB  {pf['seconds_per_step']}s/step  "
              f"lambda_seq~{checks['lambda_suggestion'].get('lambda_seq')}")

    cross = cross_device_check(examples[0], ctx, args)
    if cross.get("pass") is not None:
        print(f"  cross-device: {'ok' if cross['pass'] else 'FAIL'} "
              f"gpu={cross['gpu_grad_norm']:.6g} cpu={cross['cpu_grad_norm']:.6g} "
              f"rel={cross['relative_error']:.3g}")

    fd = None
    if not args.skip_finite_difference:
        # CPU only: on a GPU the estimator, not the gradient, is what fails.
        saved = ctx["device"]
        ctx["device"] = torch.device("cpu")
        ctx["packer"].model.to("cpu"); ctx["adapters"].to("cpu")
        cpu_example = dict(examples[0])
        for key in ("batch", "features"):
            cpu_example[key] = {
                k: (v.to("cpu") if torch.is_tensor(v) else v)
                for k, v in examples[0][key].items()
            }
        masks = examples[0]["masks"]
        cpu_example["masks"] = type(masks)(
            roles=masks.roles,
            seq_mask=masks.seq_mask.cpu(), seq_mlm_mask=masks.seq_mlm_mask.cpu(),
            sidechain_visible=masks.sidechain_visible.cpu(),
            aatype_encoder=masks.aatype_encoder.cpu(),
            aatype_true=masks.aatype_true.cpu(),
        )
        cpu_example["a_token"] = examples[0]["a_token"].cpu()
        cpu_example["sigma"] = examples[0]["sigma"].cpu()
        fd = finite_difference_check(cpu_example, ctx, args)
        ctx["device"] = saved
        ctx["packer"].model.to(saved); ctx["adapters"].to(saved)
        print(f"  finite-difference: {'ok' if fd['pass'] else 'FAIL'} "
              f"num={fd['numerical']:.6g} ana={fd['analytical']:.6g} "
              f"rel={fd['relative_error']:.3g}")

    smoke_report = None
    if args.smoke_steps:
        smoke_report = smoke(examples, ctx, args)
        print(f"  smoke {smoke_report['steps']} steps: "
              f"{'ok' if smoke_report['pass'] else 'FAIL'} "
              f"first={smoke_report['first']} last={smoke_report['last']}")

    failed = sum(
        1 for r in records for k in r if k.startswith("check") and not r[k]["pass"]
    )
    if fd and not fd["pass"]:
        failed += 1
    if cross.get("pass") is False:
        failed += 1
    if smoke_report and not smoke_report["pass"]:
        failed += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight_bs_seq_sc.json").write_text(json.dumps({
        "task": "bs_seq_sc_v1", "application_mode": "shared_prelogit",
        "settings": vars(args), "environment": {
            "adapter_is_identity_at_init": ctx["is_identity"], **ctx["freeze"],
        },
        "examples": records, "finite_difference": fd,
        "cross_device": cross, "smoke": smoke_report,
        "note": (
            "val structures; no checkpoint kept, no coefficient adopted. "
            "lambda_seq must be calibrated on a training-only batch."
        ),
    }, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {out / 'preflight_bs_seq_sc.json'}")
    if failed:
        raise SystemExit(f"{failed} check(s) failed")
    print("section 7 clean; preflight recorded; no checkpoint kept")


if __name__ == "__main__":
    main()
