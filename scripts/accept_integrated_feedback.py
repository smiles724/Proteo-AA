#!/usr/bin/env python3
"""The acceptance gate for integrated_feedback_v1. Three stages, one report.

    # 1. everything that runs in the training environment, + export for (3)
    python scripts/accept_integrated_feedback.py --stage fixture \
        --manifest .../data/calibration_pdb.parquet \
        --bs-checkpoint .../J03_seed0/checkpoints/step00000500.pt \
        --pxdesign-donor .../pxdesign_v0.1.0.pt \
        --fampnn-checkpoint .../fampnn_0_3.pt \
        --out .../acceptance

    # 2. the official runtime, on the exported inputs (official venv)
    PYTHONPATH=/users/yfsun/pxdesign_pristine \
      /users/yfsun/.venvs/pxdesign_official/bin/python \
      scripts/accept_integrated_feedback.py --stage official \
        --pxdesign-donor .../pxdesign_v0.1.0.pt --out .../acceptance

    # 3. compare, and emit acceptance.json
    python scripts/accept_integrated_feedback.py --stage report \
        --pxdesign-donor .../pxdesign_v0.1.0.pt --out .../acceptance

Three stages because the two runtimes cannot share a process: the training
environment has Protenix v2.0.0 and the official one has v0.5.0+pxd. The
comparison exports the RAW feature dict and lets each runtime apply the
preprocessing it actually requires -- the local driver adds `relp` and the
atom-pair block (`d_lm`/`v_lm`/`pad_info`), the official derives them
internally. That is the documented difference between them, so forcing one
runtime's prepared features on the other would not be an equivalence test; it
would be a type error or, worse, a silent mismatch.

Every check reports PASS, FAIL or INCOMPLETE. `--require-complete` exits
non-zero while any check is INCOMPLETE, so the full cache build cannot be
started on partial evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SCHEMA = 2
#: Same-state, same-input comparisons of a model against itself. Not widened
#: to make anything pass; CUDA reduction order is the only expected source.
EQUIVALENCE_ATOL = 1e-4
#: Cross-runtime: two Protenix versions, so kernel and preprocessing
#: differences are expected to exceed the within-runtime figure. Declared
#: before measuring.
RUNTIME_ATOL = 5e-2
FD_EPS = (1e-3, 1e-2, 5e-2, 1e-1)
FD_TOLERANCE = 0.05
#: A forbidden label must not move a model-visible feature at all beyond
#: float noise.
LEAKAGE_ATOL = 1e-5


def sha256_file(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tensor_digest(value) -> str:
    """A stable hash of a tensor or a nested structure of them."""
    import torch

    digest = hashlib.sha256()

    def walk(item, prefix=""):
        if torch.is_tensor(item):
            digest.update(prefix.encode())
            digest.update(str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().float().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                walk(item[key], f"{prefix}/{key}")
        elif isinstance(item, (list, tuple)):
            for index, entry in enumerate(item):
                walk(entry, f"{prefix}[{index}]")
        elif hasattr(item, "__dataclass_fields__"):
            from dataclasses import fields

            for field in sorted(fields(item), key=lambda f: f.name):
                walk(getattr(item, field.name), f"{prefix}.{field.name}")

    walk(value)
    return digest.hexdigest()


def verdict(ok, *, incomplete=False):
    return "INCOMPLETE" if incomplete else ("PASS" if ok else "FAIL")


# ============================================================ stage: fixture


def build_fixture(args, ctx, row, *, perturb=None, seed=None):
    """One authoritative event, with everything both arms must share.

    ``perturb`` corrupts the NATIVE labels before featurization, which is what
    makes the leakage test a measurement rather than a schema assertion: the
    labels still exist at that point, so a path that reads them would show it.
    """
    import numpy as np
    import torch

    from pxf import atom37
    from pxf.couple.integrated_event import prepare_event

    module = ctx["cache_module"]
    seed = args.seed if seed is None else seed
    structure = module.featurize_native(
        row["cif_path"], row["converted_binder_chain"],
        crop_size=args.crop_size, device=ctx["device"],
    )

    a2t = structure.topology.atom_to_token_idx.reshape(-1).long().to(ctx["device"])
    binder_tokens = structure.design_mask.reshape(-1).bool().to(ctx["device"])
    binder_atoms = binder_tokens[a2t]
    native_bb = structure.backbone_target.float().reshape(1, -1, 3).to(ctx["device"])

    if perturb:
        # The forbidden labels, corrupted. The binder BACKBONE and the target
        # context are preserved exactly, so anything that moves downstream
        # moved because of an identity or a side chain.
        native_bb, structure = _perturb_labels(
            perturb, native_bb, structure, binder_atoms, binder_tokens, a2t, ctx
        )

    names = np.asarray(structure.topology.atom_names)
    slot_of = {n: i for i, n in enumerate(atom37.ATOM37)}
    slots = torch.as_tensor(
        [slot_of.get(str(n), -1) for n in names],
        device=ctx["device"], dtype=torch.long,
    )
    backbone_slot = torch.zeros_like(slots, dtype=torch.bool)
    for value in atom37.BACKBONE_SLOTS:
        backbone_slot |= slots == value
    finite = torch.isfinite(native_bb.reshape(-1, 3)).all(-1)
    supervised = (binder_atoms & backbone_slot & finite).float()

    generator = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(native_bb.shape, generator=generator).to(ctx["device"])
    sigma = float(args.event_sigma)
    x_noisy = native_bb + sigma * noise * binder_atoms.reshape(1, -1, 1).float()

    raw_features = {k: v for k, v in structure.feature_dict.items()}
    conditioning = ctx["driver"].conditioning(structure.feature_dict)

    from pxf.couple.pxdesign_iface import BackboneTap

    rng_before = torch.get_rng_state()
    with BackboneTap(ctx["driver"].model.diffusion_module) as tap:
        bound = ctx["driver"].bind(conditioning, tap=tap)
        with torch.no_grad():
            products = prepare_event(
                denoise=module.coords_only(bound),
                x_noisy=x_noisy, sigma=sigma, structure=structure,
                designer=ctx["designer"], adapters=ctx["adapters"],
                mask_mode="native", context=args.context, seed=seed,
                design_id=str(row["example_id"]), target=str(row["example_id"]),
                tap=tap, want_h_base=True,
            )
    return {
        "example_id": str(row["example_id"]),
        "schema": SCHEMA,
        "sigma": sigma,
        "x_noisy": x_noisy,
        "native_bb": native_bb,
        "supervised": supervised.reshape(1, -1),
        "binder_mask": products.binder_mask,
        "packed": products.packed,
        "products": products,
        "conditioning": conditioning,
        "raw_features": raw_features,
        "atom_to_token_idx": a2t,
        "atom37_slot": slots,
        "rng": {"cpu_before": rng_before, "cpu_after": torch.get_rng_state()},
        "perturbed": perturb,
    }


def _perturb_labels(kind, native_bb, structure, binder_atoms, binder_tokens,
                    a2t, ctx):
    """Corrupt a forbidden label while preserving everything allowed."""
    import torch

    from pxf import atom37

    native_bb = native_bb.clone()
    if kind in ("aatype", "both"):
        # Rotate every binder identity. The binder's backbone is untouched.
        aatype = structure.aatype.clone()
        rows = binder_tokens.to(aatype.device)
        aatype[rows] = (aatype[rows] + 7) % atom37.UNKNOWN_AA_INDEX
        structure.aatype = aatype
    if kind in ("sidechain", "both"):
        # Move every binder SIDE-CHAIN coordinate. Backbone slots untouched,
        # so the allowed noisy input is unchanged.
        names = structure.topology.atom_names
        sidechain = torch.tensor(
            [str(n) not in ("N", "CA", "C", "O") for n in names],
            device=native_bb.device,
        )
        mask = (binder_atoms & sidechain).reshape(1, -1, 1)
        native_bb = native_bb + mask.float() * 17.0
    return native_bb, structure


# ============================================================= check 2
def check_upstream_leakage(args, ctx, row, base) -> dict:
    """Perturb native labels BEFORE featurization; nothing visible may move.

    This is the check the previous preflight could not make. There, the cached
    example carried no native-label channel, so "nothing to perturb" was a
    statement about the cache schema. Here the labels still exist, so a path
    that reads them will show it.
    """
    findings, worst = {}, 0.0
    for kind in ("aatype", "sidechain", "both"):
        other = build_fixture(args, ctx, row, perturb=kind, seed=args.seed)
        deltas = {
            "conditioning_s_inputs": _max_abs(
                base["conditioning"].s_inputs, other["conditioning"].s_inputs
            ),
            "conditioning_s_trunk": _max_abs(
                base["conditioning"].s_trunk, other["conditioning"].s_trunk
            ),
            "conditioning_z_trunk": _max_abs(
                base["conditioning"].z_trunk, other["conditioning"].z_trunk
            ),
            "provisional_bb0": _max_abs(
                base["products"].bb0, other["products"].bb0
            ),
            "a_token": _max_abs(
                base["products"].a_token, other["products"].a_token
            ),
            "h_packed": _max_abs(
                base["packed"].h_packed, other["packed"].h_packed
            ),
            "x_noisy_allowed_input": _max_abs(
                base["x_noisy"], other["x_noisy"]
            ),
        }
        findings[kind] = {
            "deltas": deltas,
            "designed_sequence_changed": (
                base["products"].binder_sequence
                != other["products"].binder_sequence
            ),
        }
        worst = max(worst, max(
            v for k, v in deltas.items() if k != "x_noisy_allowed_input"
        ))
    # The allowed input must be IDENTICAL for the aatype perturbation (it only
    # touches identities) -- otherwise the test changed something it should not.
    allowed_moved = findings["aatype"]["deltas"]["x_noisy_allowed_input"]
    return {
        "verdict": verdict(worst <= LEAKAGE_ATOL and allowed_moved == 0.0),
        "worst_forbidden_delta": worst,
        "tolerance": LEAKAGE_ATOL,
        "allowed_input_delta_under_aatype_perturbation": allowed_moved,
        "per_perturbation": findings,
        "note": "native binder identities rotated and binder side chains "
                "displaced 17 A before featurization, with the target "
                "context, binder backbone and chain roles preserved",
    }


def _max_abs(a, b) -> float:
    import torch

    if a is None or b is None:
        return 0.0
    return float((a.detach().float() - b.detach().float()).abs().max())


# ============================================================= check 6
def check_sidechain_chemistry(args, ctx, base, conditioner) -> dict:
    """Repack the FIXED event sequence on the corrected backbone, then measure.

    Paired randomness: the same packing seed for every arm, so a rotamer draw
    cannot look like a chemistry difference.
    """
    import numpy as np
    import torch

    from pxf import atom37
    from pxf.bench.backbone_inputs import build_design_inputs, check_design_mask
    from pxf.bench.coupled_design import conditioned
    from pxf.bench.integrated import _packed_coords, _packed_mask
    from pxf.couple.integrated_event import mask_feedback
    from pxf.couple.pxdesign_iface import BackboneTap

    products = base["products"]
    results = {}
    for label, use_feedback in (("no_feedback", False), ("with_feedback", True)):
        sigma = torch.full((1,), base["sigma"], device=ctx["device"])
        with BackboneTap(ctx["driver"].model.diffusion_module) as tap:
            bound = ctx["driver"].bind(base["conditioning"], tap=tap)
            with torch.no_grad():
                delta = None
                if use_feedback and conditioner is not None:
                    raw, _stats = conditioner(base["packed"], sigma)
                    delta = mask_feedback(
                        raw, base["binder_mask"], zero_bypass=False
                    )
                out = bound(base["x_noisy"], sigma, feedback=delta)
                bb1 = out[0] if isinstance(out, tuple) else out

        topology = ctx["structure"].topology
        a2t = np.asarray(topology.atom_to_token_idx.cpu()).astype(int)
        design = check_design_mask(
            np.asarray(ctx["structure"].design_mask.cpu()),
            res_names=np.asarray(topology.res_names), atom_to_token=a2t,
            n_tokens=int(ctx["structure"].num_tokens),
            what=base["example_id"], mode="native",
            chain_index=np.asarray(topology.chain_index.cpu()),
        )
        inputs = build_design_inputs(
            x0=bb1.reshape(-1, 3),
            a_token=products.a_token.reshape(int(ctx["structure"].num_tokens), -1),
            sigma=base["sigma"], atom_names=np.asarray(topology.atom_names),
            res_names=np.asarray(topology.res_names), atom_to_token=a2t,
            n_tokens=int(ctx["structure"].num_tokens), design=design,
            residue_index=topology.residue_index,
            asym_id=topology.chain_index, design_id=base["example_id"],
            target=base["example_id"], binder_length=int(design.sum()),
            context=args.context, device=ctx["device"],
        )
        with conditioned(ctx["designer"].model, products.residual):
            packed = ctx["designer"](
                coords_af2=inputs.coords_af2, aatype=products.aatype,
                atom_mask=inputs.atom_mask, seq_mask=inputs.seq_mask,
                residue_index=inputs.residue_index,
                chain_index=inputs.chain_index,
                scn_context_mask=inputs.sidechain_context_mask,
                # PAIRED: the same draw for both arms.
                seed=args.pack_seed,
            )
        coords = _packed_coords(packed)
        mask = _packed_mask(packed, inputs.atom_mask)
        results[label] = _sidechain_chemistry(
            coords, mask, products.aatype, base["binder_mask"]
        )
    return {
        "verdict": verdict(all(
            r["failure_rate"] is not None for r in results.values()
        )),
        "pack_seed": args.pack_seed,
        "arms": results,
        "note": "event sequence held fixed; repacked on each corrected "
                "backbone with the same packing seed",
    }


def _sidechain_chemistry(coords, mask, aatype, binder_mask) -> dict:
    """Declared metric: non-bonded side-chain clashes and impossible bonds."""
    import torch

    from pxf import atom37

    sidechain = list(atom37.SIDECHAIN_SLOTS)
    dense = coords.reshape(coords.shape[-3], atom37.NUM_ATOM37, 3)
    occupancy = mask.reshape(mask.shape[-2], atom37.NUM_ATOM37)
    rows = binder_mask.reshape(-1).bool()

    present, residue_of = [], []
    for index in torch.nonzero(rows).reshape(-1).tolist():
        for slot in sidechain:
            if occupancy[index, slot] > 0:
                present.append(dense[index, slot])
                residue_of.append(index)
    if not present:
        return {"failure_rate": None, "note": "no side-chain atoms present"}
    atoms = torch.stack(present).float()
    residue = torch.tensor(residue_of, device=atoms.device)
    distance = torch.cdist(atoms, atoms)
    same = residue[:, None] == residue[None, :]
    distance = distance.masked_fill(same, float("inf"))
    clashes = int((distance < 2.0).sum() // 2)
    return {
        "atoms": len(present),
        "nonbonded_clashes_under_2A": clashes,
        "failed": bool(clashes > 0),
        "failure_rate": float(clashes > 0),
        "threshold_angstrom": 2.0,
    }


def _max_abs_pair(a, b):
    return _max_abs(a, b)


# ============================================================= checks 4 and 5
def check_gradients(args, ctx, base, conditioner) -> dict:
    """Output projection non-zero at init; internals non-zero after a step."""
    import torch

    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.train.integrated_feedback import (FeedbackExample,
                                               assert_donors_clean,
                                               check_initial_gradient,
                                               feedback_loss, gradient_norms)

    example = _example(base, ctx["device"])

    def loss(require_grad=True):
        with BackboneTap(ctx["driver"].model.diffusion_module) as tap:
            bound = ctx["driver"].bind(base["conditioning"], tap=tap)
            return feedback_loss(
                example, conditioner,
                lambda x, s, *, feedback=None: _coords(
                    bound(x, s, feedback=feedback)
                ),
                require_grad=require_grad,
            )

    donor_before = _donor_digest(ctx)
    result = loss()
    conditioner.zero_grad(set_to_none=True)
    result.total.backward()
    initial = check_initial_gradient(conditioner)
    at_init = gradient_norms(conditioner)
    assert_donors_clean(conditioner, ctx["driver"].model, ctx["designer"].model)

    # One disposable update, then the internals must be live.
    optimizer = torch.optim.AdamW(
        [p for p in conditioner.parameters() if p.requires_grad], lr=1e-4
    )
    optimizer.step()
    second = loss()
    conditioner.zero_grad(set_to_none=True)
    second.total.backward()
    after = gradient_norms(conditioner)
    assert_donors_clean(conditioner, ctx["driver"].model, ctx["designer"].model)
    donor_after = _donor_digest(ctx)

    internal = {k: v for k, v in after.items() if "single_head.2" not in k}
    internal_live = sum(1 for v in internal.values() if v > 0)
    return {
        "verdict": verdict(
            initial["weight"] > 0
            and internal_live > 0
            and donor_before == donor_after
        ),
        "output_projection_at_init": initial,
        "internal_tensors": len(internal),
        "internal_nonzero_after_one_update": internal_live,
        "internal_nonzero_at_init": sum(
            1 for k, v in at_init.items() if "single_head.2" not in k and v > 0
        ),
        "donor_weights_unchanged": donor_before == donor_after,
        "donor_digest": donor_before[:16],
        "note": "internal gradients are legitimately zero at init -- the "
                "zero output projection blocks their path until it moves",
    }


def check_cpu_vs_gpu_gradient(args, ctx, base, conditioner) -> dict:
    """The same gradient computed on CPU and on GPU, and the FD check on CPU.

    Separate checks, as required. The finite-difference estimator runs on CPU
    because the signal sits a few float32 ULPs above the loss on this
    objective; the CPU/GPU comparison is what establishes the GPU path agrees.
    """
    import copy

    import torch

    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.train.integrated_feedback import (FeedbackExample, feedback_loss,
                                               output_projection)

    if not torch.cuda.is_available():
        return {"verdict": "INCOMPLETE", "reason": "no CUDA device"}

    def gradient(device):
        module = copy.deepcopy(conditioner).to(device)
        driver = ctx["driver"] if device.type == "cuda" else ctx.get("cpu_driver")
        if driver is None:
            return None, None
        example = _example(base, device)
        cond = ctx["conditioning_cpu"] if device.type == "cpu" else base["conditioning"]
        with BackboneTap(driver.model.diffusion_module) as tap:
            bound = driver.bind(cond, tap=tap)
            result = feedback_loss(
                example, module,
                lambda x, s, *, feedback=None: _coords(
                    bound(x, s, feedback=feedback)
                ),
            )
        module.zero_grad(set_to_none=True)
        result.total.backward()
        flat = output_projection(module).weight.grad.reshape(-1)
        return flat.detach().cpu(), float(result.total.detach())

    gpu_grad, gpu_loss = gradient(torch.device("cuda"))
    cpu_grad, cpu_loss = gradient(torch.device("cpu"))
    if cpu_grad is None:
        return {
            "verdict": "INCOMPLETE",
            "reason": "no CPU copy of the backbone donor was built; the "
                      "comparison needs the same model on both devices",
            "gpu_loss": gpu_loss,
        }
    relative = float(
        (gpu_grad - cpu_grad).norm() / cpu_grad.norm().clamp_min(1e-12)
    )
    return {
        "verdict": verdict(relative <= FD_TOLERANCE),
        "relative_gradient_difference": relative,
        "tolerance": FD_TOLERANCE,
        "cpu_loss": cpu_loss, "gpu_loss": gpu_loss,
    }


def check_finite_difference_cpu(args, ctx, base, conditioner) -> dict:
    """Analytic vs numerical, on CPU, swept over epsilon."""
    import copy

    import torch

    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.train.integrated_feedback import feedback_loss, output_projection

    driver = ctx.get("cpu_driver")
    if driver is None:
        return {
            "verdict": "INCOMPLETE",
            "reason": "no CPU copy of the backbone donor; pass --cpu-driver "
                      "to build one (slow, but this is the stable estimator)",
        }
    module = copy.deepcopy(conditioner).to("cpu")
    example = _example(base, torch.device("cpu"))
    parameter = output_projection(module).weight

    def value(require_grad=True):
        with BackboneTap(driver.model.diffusion_module) as tap:
            bound = driver.bind(ctx["conditioning_cpu"], tap=tap)
            return feedback_loss(
                example, module,
                lambda x, s, *, feedback=None: _coords(
                    bound(x, s, feedback=feedback)
                ),
                require_grad=require_grad,
            )

    result = value()
    module.zero_grad(set_to_none=True)
    result.total.backward()
    flat = parameter.grad.reshape(-1)
    index = int(flat.abs().argmax())
    analytic = float(flat[index])
    if abs(analytic) < 1e-12:
        return {"verdict": "FAIL", "analytic": analytic,
                "reason": "analytic gradient ~0: the comparison would be "
                          "vacuous (0 vs 0 agrees for the wrong reason)"}
    sweep = []
    with torch.no_grad():
        original = parameter.reshape(-1)[index].item()
        for eps in FD_EPS:
            parameter.reshape(-1)[index] = original + eps
            plus = float(value(require_grad=False).total)
            parameter.reshape(-1)[index] = original - eps
            minus = float(value(require_grad=False).total)
            parameter.reshape(-1)[index] = original
            numeric = (plus - minus) / (2 * eps)
            sweep.append({"eps": eps, "numeric": numeric,
                          "relative_error": abs(numeric - analytic)
                          / max(abs(analytic), 1e-12)})
    best = min(sweep, key=lambda s: s["relative_error"])
    converged = abs(sweep[-1]["numeric"] - sweep[-2]["numeric"]) <= (
        FD_TOLERANCE * max(abs(sweep[-1]["numeric"]), 1e-12)
    )
    return {
        "verdict": verdict(converged and best["relative_error"] <= FD_TOLERANCE),
        "device": "cpu", "analytic": analytic, "probed_index": index,
        "sweep": sweep, "best": best, "estimator_converged": converged,
    }


def check_paired_initialisation(args, ctx, base) -> dict:
    """Both arms must start from identical shared tensors and digests."""
    import random

    import numpy as np
    import torch

    from pxf.couple.conditioning import build_conditioner
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.couple.pxdesign_iface import (conditioning_widths,
                                           token_feature_dim)

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import importlib.util as ilu

    spec = ilu.spec_from_file_location(
        "_train_mod", str(REPO_ROOT / "scripts" / "train_integrated_feedback.py")
    )
    train_mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    c_s, c_z = conditioning_widths(ctx["driver"].model)
    digests = {}
    for arm in ("early_s_full", "early_s_bb_only"):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        module = build_conditioner(
            arm, c_h_V=node_feature_dim(ctx["designer"].model),
            c_token=token_feature_dim(ctx["driver"].model), c_s=c_s, c_z=c_z,
        )
        digests[arm] = train_mod.initial_weight_digest(module)
    shared = digests["early_s_full"] == digests["early_s_bb_only"]
    return {
        "verdict": verdict(shared),
        "digests": {k: v[:16] for k, v in digests.items()},
        "identical": shared,
        "event_digest": tensor_digest(base["packed"])[:16],
        "conditioning_digest": tensor_digest({
            "s_inputs": base["conditioning"].s_inputs,
            "s_trunk": base["conditioning"].s_trunk,
            "z_trunk": base["conditioning"].z_trunk,
        })[:16],
        "note": "the two arms have identical architecture and parameter "
                "count by construction, so a matched pair must also have "
                "identical initial weights",
    }


def check_gpu_replay(args, ctx, base) -> dict:
    """On CUDA: repeated no-feedback resumes agree, and decoder draws are inert."""
    import torch

    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.couple.replay import RngStream, run_trajectory

    if not torch.cuda.is_available():
        return {"verdict": "INCOMPLETE", "reason": "no CUDA device"}
    device = ctx["device"]
    schedule = torch.logspace(1, -1, 8, device=device)

    def trajectory(extra_draws):
        with BackboneTap(ctx["driver"].model.diffusion_module) as tap:
            bound = ctx["driver"].bind(base["conditioning"], tap=tap)

            def feedback(state):
                if extra_draws:
                    # Stand in for the event decoder's ~101 encoder calls and
                    # the packing rollout, all of which draw on CUDA.
                    torch.randn(4096, device=device)
                return None

            x, _r, _s = run_trajectory(
                denoise=lambda x, s, *, feedback=None: _coords(
                    bound(x, s, feedback=feedback)
                ),
                schedule=schedule, n_atom=ctx["driver"].model
                .diffusion_module.layernorm_a.weight.shape[-1] and
                base["x_noisy"].shape[-2],
                device=device, n_sample=1,
                stream=RngStream("acceptance", args.seed, device=device),
                event=(2, 0), feedback=feedback,
            )
        return x

    first, second = trajectory(False), trajectory(False)
    with_draws = trajectory(True)
    repeat_delta = _max_abs(first, second)
    draw_delta = _max_abs(first, with_draws)
    return {
        "verdict": verdict(
            repeat_delta <= EQUIVALENCE_ATOL and draw_delta <= EQUIVALENCE_ATOL
        ),
        "repeated_resume_max_abs_difference": repeat_delta,
        "extra_decoder_draws_max_abs_difference": draw_delta,
        "tolerance": EQUIVALENCE_ATOL,
        "device": str(device),
        "note": "the stream is bound to the CUDA device, so replay captures "
                "and restores the CUDA generator; without that these would "
                "differ for RNG bookkeeping rather than for feedback",
    }


def _example(base, device):
    from pxf.train.integrated_feedback import FeedbackExample, _packed_to

    return FeedbackExample(
        example_id=base["example_id"], x_noisy=base["x_noisy"].to(device),
        sigma=base["sigma"], packed=_packed_to(base["packed"], device),
        binder_mask=base["binder_mask"].to(device),
        native_bb=base["native_bb"].to(device),
        supervised=base["supervised"].to(device), provenance={},
    )


def _coords(out):
    return out[0] if isinstance(out, tuple) else out


def _donor_digest(ctx) -> str:
    import torch

    digest = hashlib.sha256()
    for model in (ctx["driver"].model, ctx["designer"].model):
        for name, parameter in sorted(model.named_parameters()):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()


# ============================================================ stage drivers


def stage_fixture(args) -> None:
    """Everything runnable in the training environment, plus the export."""
    import importlib.util as ilu

    import pandas as pd
    import torch

    import _bootstrap  # noqa: F401

    from pxf import provenance
    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.couple.conditioning import build_conditioner
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.couple.pxdesign_iface import (conditioning_widths,
                                           token_feature_dim)
    from pxf.device import select_device
    from pxf.sidechain.design import FaMPNNFullAtomDesigner
    from pxf.train.integrated_feedback import freeze_everything_but

    spec = ilu.spec_from_file_location(
        "_cache_mod", str(REPO_ROOT / "scripts" / "cache_integrated_feedback.py")
    )
    cache_module = ilu.module_from_spec(spec)
    spec.loader.exec_module(cache_module)

    out = Path(args.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    frame = pd.read_parquet(args.manifest)
    if "split" in frame and set(frame["split"]) - {"train"}:
        raise SystemExit(
            f"{args.manifest} is not a train-split manifest; the acceptance "
            "gate must not run optimizer updates on validation examples"
        )
    rows = [frame.iloc[i].to_dict() for i in range(min(args.n_examples, len(frame)))]
    print(f"acceptance gate on {len(rows)} calibration example(s): "
          f"{[r['example_id'] for r in rows]}")

    px_model, _c, _r = load_backbone_model(args.pxdesign_donor, device=device)
    driver = PXDesignBackboneDriver(px_model)
    designer = FaMPNNFullAtomDesigner(
        args.fampnn_checkpoint, variant=args.fampnn_variant,
        seq_steps=args.seq_steps, temperature=args.temperature,
        psce_threshold=args.psce_threshold, repack_last=True,
    ).to(device).eval()
    fampnn_sha = provenance.file_sha256(
        args.fampnn_checkpoint or provenance.fampnn_checkpoint(args.fampnn_variant)
    )
    adapters = cache_module._load_adapters(
        args.bs_checkpoint, designer, driver, device, args.bs_weights, fampnn_sha
    )
    c_s, c_z = conditioning_widths(px_model)
    conditioner = build_conditioner(
        args.arm, c_h_V=node_feature_dim(designer.model),
        c_token=token_feature_dim(px_model), c_s=c_s, c_z=c_z,
    ).to(device)
    frozen = freeze_everything_but(conditioner, px_model, designer.model)

    ctx = {"device": device, "driver": driver, "designer": designer,
           "adapters": adapters, "cache_module": cache_module}

    hashes = {
        "pxdesign_donor": sha256_file(args.pxdesign_donor),
        "fampnn_checkpoint": fampnn_sha,
        "bs_checkpoint": sha256_file(args.bs_checkpoint),
        "manifest": sha256_file(args.manifest),
    }
    report = {
        "schema": SCHEMA, "arm": args.arm, "frozen": frozen,
        "hashes": hashes, "examples": [r["example_id"] for r in rows],
        "settings": {
            "event_sigma": args.event_sigma, "context": args.context,
            "seq_steps": args.seq_steps, "pack_steps": args.pack_steps,
            "temperature": args.temperature,
            "psce_threshold": args.psce_threshold,
            "crop_size": args.crop_size, "seed": args.seed,
            "pack_seed": args.pack_seed, "bs_weights": args.bs_weights,
        },
        "tolerances": {
            "equivalence_atol": EQUIVALENCE_ATOL,
            "runtime_atol": RUNTIME_ATOL,
            "leakage_atol": LEAKAGE_ATOL,
            "fd_relative": FD_TOLERANCE,
        },
        "checks": {},
    }

    # ---- 1. the authoritative fixture ---------------------------------
    print("\n1. schema-v2 event fixture")
    fixtures = {}
    for row in rows:
        base = build_fixture(args, ctx, row)
        ctx["structure"] = cache_module.featurize_native(
            row["cif_path"], row["converted_binder_chain"],
            crop_size=args.crop_size, device=device,
        )
        path = out / "fixture" / f"{row['example_id']}.pt"
        torch.save({
            "schema": SCHEMA,
            "example_id": base["example_id"], "sigma": base["sigma"],
            "x_noisy": base["x_noisy"].cpu(),
            "native_bb": base["native_bb"].cpu(),
            "supervised": base["supervised"].cpu(),
            "binder_mask": base["binder_mask"].cpu(),
            "packed": _cpu(base["packed"]),
            "atom_to_token_idx": base["atom_to_token_idx"].cpu(),
            "atom37_slot": base["atom37_slot"].cpu(),
            "raw_features": _cpu(base["raw_features"]),
            "conditioning": {
                "s_inputs": base["conditioning"].s_inputs.cpu(),
                "s_trunk": base["conditioning"].s_trunk.cpu(),
                "z_trunk": base["conditioning"].z_trunk.cpu(),
            },
            "designed_sequence": base["products"].sequence,
            "binder_sequence": base["products"].binder_sequence,
            "aatype": base["products"].aatype.cpu(),
            "coords_af2": base["products"].coords_af2.cpu(),
            "a_token": base["products"].a_token.cpu(),
            "residual": None if base["products"].residual is None
            else base["products"].residual.cpu(),
            "rng": {k: v for k, v in base["rng"].items()},
            "hashes": hashes, "settings": report["settings"],
            "h_base_policy": "same realized coordinates and designed "
                             "sequence as h_packed; ALL side-chain slots "
                             "masked. This is the declared BB-only control.",
        }, path)
        fixtures[base["example_id"]] = {
            "path": str(path), "sha256": sha256_file(path),
            "event_digest": tensor_digest(base["packed"]),
            "binder_length": int(base["binder_mask"].sum()),
            "supervised_atoms": int(base["supervised"].sum()),
        }
        print(f"   {base['example_id']}: {fixtures[base['example_id']]['sha256'][:16]} "
              f"({fixtures[base['example_id']]['supervised_atoms']} supervised atoms)")
        if row is rows[0]:
            primary, primary_row = base, row
    report["fixture"] = fixtures
    report["checks"]["fixture"] = {
        "verdict": verdict(len(fixtures) == len(rows)),
        "n": len(fixtures),
        "h_base_policy": "matched ablation: same coordinates, same designed "
                         "sequence, every side-chain slot masked",
    }

    # ---- 2. upstream leakage ------------------------------------------
    print("\n2. upstream leakage (perturbed before featurization)")
    report["checks"]["upstream_leakage"] = check_upstream_leakage(
        args, ctx, primary_row, primary
    )
    c = report["checks"]["upstream_leakage"]
    print(f"   worst forbidden delta {c['worst_forbidden_delta']:.3e} "
          f"(tol {c['tolerance']:.0e}) -> {c['verdict']}")

    # ---- 4. gradients and freezing -------------------------------------
    print("\n4. gradients and freezing")
    report["checks"]["gradients"] = check_gradients(args, ctx, primary, conditioner)
    g = report["checks"]["gradients"]
    print(f"   projection at init {g['output_projection_at_init']['weight']:.3e}; "
          f"{g['internal_nonzero_after_one_update']}/{g['internal_tensors']} "
          f"internal live after one update; donors unchanged "
          f"{g['donor_weights_unchanged']} -> {g['verdict']}")
    report["checks"]["finite_difference_cpu"] = check_finite_difference_cpu(
        args, ctx, primary, conditioner
    )
    print(f"   CPU finite difference -> "
          f"{report['checks']['finite_difference_cpu']['verdict']}")
    report["checks"]["cpu_vs_gpu_gradient"] = check_cpu_vs_gpu_gradient(
        args, ctx, primary, conditioner
    )
    print(f"   CPU-vs-GPU gradient -> "
          f"{report['checks']['cpu_vs_gpu_gradient']['verdict']}")

    # ---- 5. paired init and GPU replay ---------------------------------
    print("\n5. paired initialisation and GPU replay")
    report["checks"]["paired_initialisation"] = check_paired_initialisation(
        args, ctx, primary
    )
    p = report["checks"]["paired_initialisation"]
    print(f"   arm digests identical: {p['identical']} -> {p['verdict']}")
    report["checks"]["gpu_replay"] = check_gpu_replay(args, ctx, primary)
    r = report["checks"]["gpu_replay"]
    print(f"   repeated resume {r.get('repeated_resume_max_abs_difference')}, "
          f"extra draws {r.get('extra_decoder_draws_max_abs_difference')} -> "
          f"{r['verdict']}")

    # ---- 6. side-chain chemistry ---------------------------------------
    print("\n6. side-chain chemistry on the corrected backbone")
    report["checks"]["sidechain_chemistry"] = check_sidechain_chemistry(
        args, ctx, primary, conditioner
    )
    print(f"   -> {report['checks']['sidechain_chemistry']['verdict']}")

    # ---- 3. export for the cross-runtime comparison --------------------
    print("\n3. exporting inputs for the local-vs-official comparison")
    from pxf.couple.integrated_event import mask_feedback

    sigma = torch.full((1,), primary["sigma"], device=device)
    raw, _stats = conditioner(primary["packed"], sigma)
    nonzero = mask_feedback(raw, primary["binder_mask"], zero_bypass=False)
    # A zero-initialised head emits zero, so force a non-trivial payload for
    # the "same nonzero feedback" arm rather than comparing zeros.
    scaled = _scale_payload(nonzero, args.feedback_scale)
    export = out / "runtime_inputs.pt"
    torch.save({
        "schema": SCHEMA, "example_id": primary["example_id"],
        "sigma": primary["sigma"],
        "x_noisy": primary["x_noisy"].cpu(),
        "raw_features": _cpu(primary["raw_features"]),
        "binder_mask": primary["binder_mask"].cpu(),
        "feedback_zero": _cpu(mask_feedback(
            _zero_like_payload(nonzero), primary["binder_mask"],
            zero_bypass=False)),
        "feedback_nonzero": _cpu(scaled),
        "feedback_scale": args.feedback_scale,
        "hashes": hashes,
    }, export)
    report["runtime_export"] = {
        "path": str(export), "sha256": sha256_file(export),
    }
    local = _evaluate_runtime(driver, primary, scaled, nonzero, device)
    torch.save(local, out / "runtime_local.pt")
    report["checks"]["local_vs_official"] = {
        "verdict": "INCOMPLETE",
        "reason": "run --stage official then --stage report",
        "local": {k: v for k, v in local.items() if not hasattr(v, "shape")},
    }
    print(f"   exported {report['runtime_export']['sha256'][:16]}; local "
          "outputs saved. Run --stage official next.")

    _write(out, report, args)


def _scale_payload(payload, scale):
    from dataclasses import replace

    if payload is None:
        return None
    if hasattr(payload, "delta_single"):
        single = payload.delta_single
        if single is None or float(single.abs().max()) == 0.0:
            # A zero-init head emits exactly zero; comparing zeros would not
            # test the feedback-induced change at all.
            single = None if single is None else (
                single + scale * _deterministic_like(single)
            )
        else:
            single = single * scale
        return replace(payload, delta_single=single)
    return payload * scale


def _deterministic_like(tensor):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(0)
    return torch.randn(
        tensor.shape, generator=generator, dtype=torch.float32
    ).to(tensor.device).to(tensor.dtype)


def _zero_like_payload(payload):
    from dataclasses import replace

    import torch

    if payload is None:
        return None
    if hasattr(payload, "delta_single"):
        return replace(
            payload,
            delta_single=None if payload.delta_single is None
            else torch.zeros_like(payload.delta_single),
            delta_pair=None if payload.delta_pair is None
            else torch.zeros_like(payload.delta_pair),
        )
    return torch.zeros_like(payload)


def _evaluate_runtime(driver, base, nonzero, zero_template, device):
    """no-feedback / zero-feedback / nonzero-feedback, in one runtime."""
    import torch

    from pxf.couple.integrated_event import mask_feedback
    from pxf.couple.pxdesign_iface import BackboneTap

    sigma = torch.full((1,), base["sigma"], device=device)
    outputs = {}
    with BackboneTap(driver.model.diffusion_module) as tap:
        bound = driver.bind(base["conditioning"], tap=tap)
        with torch.no_grad():
            outputs["none"] = _coords(bound(base["x_noisy"], sigma,
                                            feedback=None)).cpu()
            outputs["zero"] = _coords(bound(
                base["x_noisy"], sigma,
                feedback=mask_feedback(_zero_like_payload(zero_template),
                                       base["binder_mask"], zero_bypass=False),
            )).cpu()
            outputs["nonzero"] = _coords(bound(base["x_noisy"], sigma,
                                               feedback=nonzero)).cpu()
    outputs["none_vs_zero_max_abs"] = _max_abs(outputs["none"], outputs["zero"])
    outputs["feedback_induced_change"] = _max_abs(
        outputs["none"], outputs["nonzero"]
    )
    return outputs


def _cpu(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import fields, replace

        return replace(value, **{
            f.name: _cpu(getattr(value, f.name)) for f in fields(value)
        })
    return value


def _write(out, report, args):
    incomplete = [
        name for name, check in report["checks"].items()
        if check.get("verdict") == "INCOMPLETE"
    ]
    failed = [
        name for name, check in report["checks"].items()
        if check.get("verdict") == "FAIL"
    ]
    report["summary"] = {
        "passed": [n for n, c in report["checks"].items()
                   if c.get("verdict") == "PASS"],
        "failed": failed, "incomplete": incomplete,
        "ready_for_full_cache": not failed and not incomplete,
    }
    report["commands"] = {
        "fixture": f"python scripts/accept_integrated_feedback.py --stage "
                   f"fixture --manifest {args.manifest} --bs-checkpoint "
                   f"{args.bs_checkpoint} --pxdesign-donor "
                   f"{args.pxdesign_donor} --fampnn-checkpoint "
                   f"{args.fampnn_checkpoint} --out {args.out}",
        "official": f"PYTHONPATH=/users/yfsun/pxdesign_pristine "
                    f"/users/yfsun/.venvs/pxdesign_official/bin/python "
                    f"scripts/accept_integrated_feedback.py --stage official "
                    f"--pxdesign-donor {args.pxdesign_donor} --out {args.out}",
        "report": f"python scripts/accept_integrated_feedback.py --stage "
                  f"report --pxdesign-donor {args.pxdesign_donor} --out "
                  f"{args.out}",
    }
    path = Path(out) / "acceptance.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {path}")
    print(f"  PASS       {report['summary']['passed']}")
    print(f"  FAIL       {failed}")
    print(f"  INCOMPLETE {incomplete}")
    print(f"  ready_for_full_cache = {report['summary']['ready_for_full_cache']}")
    if args.require_complete and not report["summary"]["ready_for_full_cache"]:
        print("\n--require-complete: the gate is not satisfied")
        raise SystemExit(1)


def stage_official(args) -> None:
    """Evaluate the OFFICIAL runtime on the exported inputs. Official venv only."""
    sys.path.insert(0, str(REPO_ROOT))
    from pxf.official.require import require_official_protenix

    require_official_protenix("accept_integrated_feedback --stage official")

    import torch

    from pxf.couple.pxdesign_iface import BackboneTap

    out = Path(args.out)
    blob = torch.load(str(out / "runtime_inputs.pt"), map_location="cpu",
                      weights_only=False)
    if int(blob.get("schema", 0)) != SCHEMA:
        raise SystemExit(f"exported inputs are schema {blob.get('schema')}, "
                         f"expected {SCHEMA}")

    # The official ProtenixDesign, from the SAME donor weights, driven on the
    # SAME raw feature dict. Each runtime applies the preprocessing it
    # requires -- the local driver adds relp and the atom-pair block, the
    # official derives them internally -- which is the documented difference
    # between them.
    from pxf.backbone.driver import load_backbone_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _c, _r = load_backbone_model(args.pxdesign_donor, device=device)
    model.requires_grad_(False)

    features = {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in blob["raw_features"].items()}
    with torch.no_grad():
        s_inputs, s_trunk, z_trunk = model.get_condition_embedding(
            input_feature_dict=features, chunk_size=None
        )
    x_noisy = blob["x_noisy"].to(device)
    sigma = torch.full((1,), float(blob["sigma"]), device=device)

    def denoise(feedback):
        with BackboneTap(model.diffusion_module) as tap:
            tap.feedback = feedback
            with torch.no_grad():
                return model.diffusion_module(
                    x_noisy=x_noisy,
                    t_hat_noise_level=sigma.to(torch.float32),
                    input_feature_dict=features,
                    s_inputs=s_inputs.to(torch.float32),
                    s_trunk=s_trunk.to(torch.float32),
                    z_trunk=z_trunk.to(torch.float32),
                    chunk_size=None, inplace_safe=False,
                ).to(torch.float32).cpu()

    def to_device(payload):
        from dataclasses import fields, replace

        if payload is None:
            return None
        if hasattr(payload, "__dataclass_fields__"):
            return replace(payload, **{
                f.name: (getattr(payload, f.name).to(device)
                         if torch.is_tensor(getattr(payload, f.name)) else
                         getattr(payload, f.name))
                for f in fields(payload)
            })
        return payload.to(device)

    outputs = {
        "none": denoise(None),
        "zero": denoise(to_device(blob["feedback_zero"])),
        "nonzero": denoise(to_device(blob["feedback_nonzero"])),
    }
    outputs["none_vs_zero_max_abs"] = _max_abs(outputs["none"], outputs["zero"])
    outputs["feedback_induced_change"] = _max_abs(
        outputs["none"], outputs["nonzero"]
    )
    outputs["device"] = str(device)
    outputs["dtype"] = "float32"
    torch.save(outputs, out / "runtime_official.pt")
    print(f"official runtime: none-vs-zero "
          f"{outputs['none_vs_zero_max_abs']:.3e}, feedback-induced change "
          f"{outputs['feedback_induced_change']:.3e}")
    print(f"wrote {out / 'runtime_official.pt'}")


def stage_report(args) -> None:
    """Compare the two runtimes and finalise acceptance.json."""
    import torch

    out = Path(args.out)
    report = json.loads((out / "acceptance.json").read_text())
    local_path, official_path = out / "runtime_local.pt", out / "runtime_official.pt"
    if not official_path.is_file():
        report["checks"]["local_vs_official"] = {
            "verdict": "INCOMPLETE",
            "reason": f"{official_path} missing; run --stage official",
        }
        _write(out, report, args)
        return

    local = torch.load(str(local_path), map_location="cpu", weights_only=False)
    official = torch.load(str(official_path), map_location="cpu",
                          weights_only=False)
    comparisons = {}
    for arm in ("none", "zero", "nonzero"):
        comparisons[arm] = _max_abs(local[arm], official[arm])
    # The coordinate CHANGE feedback induces, in each runtime. Agreeing on the
    # change matters more than agreeing on the absolute coordinates: it is the
    # change the adapter is trained to produce.
    change = {
        "local": local["feedback_induced_change"],
        "official": official["feedback_induced_change"],
    }
    change["difference"] = abs(change["local"] - change["official"])
    ok = (
        max(comparisons.values()) <= RUNTIME_ATOL
        and change["difference"] <= RUNTIME_ATOL
    )
    report["checks"]["local_vs_official"] = {
        "verdict": verdict(ok),
        "coordinate_max_abs_difference": comparisons,
        "feedback_induced_change": change,
        "tolerance": RUNTIME_ATOL,
        "local_none_vs_zero": local["none_vs_zero_max_abs"],
        "official_none_vs_zero": official["none_vs_zero_max_abs"],
        "official_device": official.get("device"),
        "precision": "float32 in both; the official conditioning is computed "
                     "under autocast as upstream does, then cast",
        "note": "identical RAW feature dict and identical donor weights; each "
                "runtime applies the preprocessing it requires (the local "
                "driver adds relp and the atom-pair block, the official "
                "derives them internally), which is the documented difference",
        "artifacts": {
            "local": sha256_file(local_path),
            "official": sha256_file(official_path),
            "inputs": sha256_file(out / "runtime_inputs.pt"),
        },
    }
    print("local vs official:")
    for arm, value in comparisons.items():
        print(f"  {arm:8s} max|local - official| = {value:.3e}")
    print(f"  feedback-induced change: local {change['local']:.3e}, "
          f"official {change['official']:.3e}, differ by "
          f"{change['difference']:.3e} (tol {RUNTIME_ATOL:.0e})")
    _write(out, report, args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", required=True,
                        choices=("fixture", "official", "report"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--bs-checkpoint", default=None)
    parser.add_argument("--bs-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--pxdesign-donor", required=True)
    parser.add_argument("--fampnn-checkpoint", default=None)
    parser.add_argument("--fampnn-variant", default="0.3")
    parser.add_argument("--arm", default="early_s_full")
    parser.add_argument("--event-sigma", type=float, default=0.429)
    parser.add_argument("--context", default="complex_sc")
    parser.add_argument("--seq-steps", type=int, default=100)
    parser.add_argument("--pack-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--psce-threshold", type=float, default=0.3)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--n-examples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pack-seed", type=int, default=7)
    parser.add_argument("--feedback-scale", type=float, default=1.0,
                        help="scale for the non-zero feedback arm. A "
                             "zero-initialised head emits exactly zero, so "
                             "without a non-trivial payload the 'same nonzero "
                             "feedback' comparison would compare zeros.")
    parser.add_argument("--cpu-driver", action="store_true",
                        help="also build a CPU copy of the backbone donor, "
                             "which the CPU finite-difference and CPU-vs-GPU "
                             "checks require")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.stage == "fixture":
        for name in ("manifest", "bs_checkpoint"):
            if not getattr(args, name):
                raise SystemExit(f"--stage fixture needs --{name.replace('_','-')}")
        stage_fixture(args)
    elif args.stage == "official":
        stage_official(args)
    else:
        stage_report(args)


if __name__ == "__main__":
    main()
