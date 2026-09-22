#!/usr/bin/env python3
"""Paired integrated generation: one shared prefix, several arms branched from it.

    python scripts/run_integrated_binder_matrix.py \
        --targets-config configs/binder_benchmark/targets.yaml \
        --prepared-dir runs/binder_bench/targets/configs \
        --checkpoint-selection runs/.../evaluation/selected_checkpoints.json \
        --targets PDL1 BHRF1 --lengths 100 --seeds 101 102 \
        --out runs/integrated_feedback_v1/generation_smoke

`scripts/design_binder_integrated.py` runs ONE arm per process, which cannot
give a paired comparison: separate CUDA trajectories are not bit-identical
even at equal seeds, so two arms' prefixes would differ for reasons unrelated
to feedback. This generates the prefix once, records the sampler state at the
event, and resumes every arm from that same recorded state -- which is what
`pxf.couple.replay`'s record/resume exists for, and why its records carry the
RNG as well as the coordinates.

### What is shared and what is not

For a fixed A_BS seed, the three J03 arms -- no-feedback, E1-BB-only, E1-full
-- share the prefix AND the event's designed sequence and packing. Only the
feedback payload differs, so the comparison isolates it exactly.

U03 shares the prefix but runs its OWN unadapted event decode. Its different
sequence is part of the method contrast, not a confound: U03 is the claim that
the adapter is unnecessary, so it must be allowed to design its own sequence.

Each arm repacks on its own final backbone, with the final packing randomness
paired across arms so a rotamer draw cannot masquerade as an effect.

### Cost

The prefix costs one full trajectory. Each arm then resumes from the event,
which at the default sigma lands near step 338 of 400, so an arm costs about
60 solver calls plus its decode. Four arms is roughly 1.6 trajectories, not 4.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ROW_COLUMNS = (
    "sample_id", "target", "binder_length", "sequence", "design_pdb",
    "binder_chain", "target_chains", "arm", "feedback_arm", "bs_seed",
    "generation_seed", "shared_prefix_id",
    "requested_sigma", "actual_sigma", "event_step",
    "solver_calls", "conditioning_injections", "decode_hook_calls",
    "delta_h_norm", "feedback_norm",
    "event_to_final_aligned_rmsd", "min_bb_bb_distance", "interface_clashes",
    "seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--targets-config",
                        default="configs/binder_benchmark/targets.yaml")
    parser.add_argument("--prepared-dir", required=True,
                        help="directory of prepared per-target YAMLs")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-selection", default=None,
                        help="selected_checkpoints.json from the evaluation. "
                             "Without it only U03 and the A_BS-only arm run.")
    parser.add_argument("--bs-checkpoint", action="append", default=[],
                        metavar="SEED=PATH")
    parser.add_argument("--fampnn-checkpoint", default=None)
    parser.add_argument("--fampnn-variant", default="0.3")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--lengths", nargs="*", type=int, default=[100])
    parser.add_argument("--seeds", nargs="*", type=int, default=[101, 102])
    parser.add_argument("--event-sigma", type=float, default=0.429)
    parser.add_argument("--n-step", type=int, default=400)
    parser.add_argument("--step-scale-eta", type=float, default=2.5)
    parser.add_argument("--context", default="complex_sc")
    parser.add_argument("--seq-steps", type=int, default=100)
    parser.add_argument("--pack-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--psce-threshold", type=float, default=0.3)
    parser.add_argument("--use-msa", action="store_true")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--allow-feedback-policy-transfer", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from pxf.official.require import require_official_protenix

    require_official_protenix("run_integrated_binder_matrix")

    import torch
    import yaml

    from pxf.bench.integrated import select_event
    from pxf.bench.integrated_checkpoints import (expected_policy, file_sha256,
                                                  load_feedback)
    from pxf.couple.integrated_event import (assert_target_rows_untouched,
                                             mask_feedback, prepare_event)
    from pxf.couple.pxdesign_iface import (BackboneTap, conditioning_widths,
                                           token_feature_dim)
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.couple.replay import RngStream, run_trajectory
    from pxf.official.bridge import OfficialStructure
    from pxf.official.runtime import OfficialDenoiser, build_runner, first_batch
    from pxf.sidechain.design import FaMPNNFullAtomDesigner

    out = Path(args.out)
    (out / "designs").mkdir(parents=True, exist_ok=True)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)

    bs_by_seed = {}
    for entry in args.bs_checkpoint:
        seed, _, path = entry.partition("=")
        bs_by_seed[int(seed)] = path
    feedback_selection = {}
    if args.checkpoint_selection:
        feedback_selection = json.loads(
            Path(args.checkpoint_selection).read_text()
        )
        missing = [k for k, v in feedback_selection.items()
                   if not v.get("checkpoint")]
        if missing:
            raise SystemExit(
                f"the selection artifact has no checkpoint for {missing}; the "
                "matrix refuses implicit checkpoint defaults"
            )

    targets_cfg = yaml.safe_load(Path(args.targets_config).read_text())
    wanted = set(args.targets or [])
    rows = []

    for target in targets_cfg["targets"]:
        name = target.get("name")
        if wanted and name not in wanted:
            continue
        prepared = Path(args.prepared_dir) / f"{name}.yaml"
        if not prepared.is_file():
            print(f"SKIP {name}: no prepared YAML at {prepared}")
            continue
        for length in args.lengths:
            for gen_seed in args.seeds:
                rows.extend(_one_cell(
                    name=name, length=length, gen_seed=gen_seed,
                    prepared=prepared, args=args, out=out,
                    bs_by_seed=bs_by_seed,
                    feedback_selection=feedback_selection,
                    build_runner=build_runner, first_batch=first_batch,
                    OfficialDenoiser=OfficialDenoiser,
                    OfficialStructure=OfficialStructure,
                    FaMPNNFullAtomDesigner=FaMPNNFullAtomDesigner,
                    select_event=select_event, prepare_event=prepare_event,
                    mask_feedback=mask_feedback,
                    assert_target_rows_untouched=assert_target_rows_untouched,
                    BackboneTap=BackboneTap, RngStream=RngStream,
                    run_trajectory=run_trajectory,
                    load_feedback=load_feedback,
                    expected_policy=expected_policy,
                    conditioning_widths=conditioning_widths,
                    token_feature_dim=token_feature_dim,
                    node_feature_dim=node_feature_dim,
                ))

    with (out / "designs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    (out / "provenance.json").write_text(json.dumps({
        "protocol": "integrated_paired_matrix",
        "shared_prefix": "one trajectory per (target, length, generation seed); "
                         "every arm resumes from the recorded event state, so "
                         "the prefix is bit-identical rather than merely "
                         "similarly seeded",
        "arms": sorted({r["arm"] for r in rows}),
        "targets": sorted({r["target"] for r in rows}),
        "lengths": args.lengths, "generation_seeds": args.seeds,
        "bs_checkpoints": {str(k): file_sha256(v)
                           for k, v in bs_by_seed.items()},
        "feedback_selection": feedback_selection,
        "event_sigma_requested": args.event_sigma,
        "n_step": args.n_step, "step_scale_eta": args.step_scale_eta,
        "decoder": {"seq_steps": args.seq_steps, "pack_steps": args.pack_steps,
                    "temperature": args.temperature,
                    "context": args.context},
        "not_comparable_to": "scripts/design_binder_matrix.py's cached-backbone "
                             "arms: a different inference protocol, and A_BS is "
                             "queried at a different sigma there",
    }, indent=2, default=str))
    print(f"\nwrote {out / 'designs.csv'} ({len(rows)} row(s))")


def _one_cell(*, name, length, gen_seed, prepared, args, out, bs_by_seed,
              feedback_selection, **api):
    """One (target, length, generation seed): shared prefix, then every arm."""
    import torch

    print(f"\n=== {name} L{length} seed {gen_seed} ===")
    runner = api["build_runner"](
        str(prepared), str(out / "pxdesign" / f"{name}_L{length}_s{gen_seed}"),
        load_checkpoint_dir=args.checkpoint_dir, n_step=args.n_step,
        n_sample=1, use_msa=args.use_msa, dtype=args.dtype,
        eta_type="const", eta_min=args.step_scale_eta,
        eta_max=args.step_scale_eta,
    )
    data, atom_array = api["first_batch"](runner)
    denoiser = api["OfficialDenoiser"](runner, data)
    structure = api["OfficialStructure"](
        atom_array, denoiser.features,
        int(denoiser.features["residue_index"].reshape(-1).shape[0]),
        device=denoiser.device,
    )
    designer = api["FaMPNNFullAtomDesigner"](
        args.fampnn_checkpoint, variant=args.fampnn_variant,
        seq_steps=args.seq_steps, temperature=args.temperature,
        psce_threshold=args.psce_threshold, repack_last=True,
    ).to(denoiser.device).eval()
    designer.model.requires_grad_(False)

    schedule = denoiser.schedule(args.n_step)
    choice = api["select_event"](schedule, args.event_sigma)
    print(f"  event step {choice.step}: requested {choice.requested_sigma:.4f} "
          f"-> actual {choice.actual_sigma:.4f}")

    # ---- pass 1: the shared prefix, recorded at the event -----------------
    prefix_id = f"{name}_L{length}_s{gen_seed}"
    with api["BackboneTap"](denoiser.model.diffusion_module) as tap:
        stream = api["RngStream"](
            "integrated", gen_seed, device=_cuda(denoiser.device)
        )
        _x0, records, _stats = api["run_trajectory"](
            denoise=lambda x, s, *, feedback=None: denoiser.denoise(
                x, s, feedback=feedback, tap=tap
            ),
            schedule=schedule, n_atom=denoiser.n_atom,
            device=denoiser.device, n_sample=1,
            step_scale_eta=args.step_scale_eta, stream=stream,
            record_steps={choice.step},
        )
    if not records:
        raise SystemExit(
            f"{prefix_id}: the trajectory recorded no state at step "
            f"{choice.step}; every arm would start somewhere different"
        )
    realized = int(structure.design_mask.reshape(-1).sum())
    if realized != int(length):
        raise SystemExit(
            f"{prefix_id}: the prepared YAML produced a design mask of "
            f"{realized} token(s) but --lengths asked for {length}. The "
            "length is used in every label and must describe the structure, "
            "not the request; regenerate the YAML for this length."
        )
    recorded = records[0]
    torch.save(recorded, out / "diagnostics" / f"{prefix_id}.event.pt")
    print(f"  recorded the event state; {len(records)} record(s)")

    rows = []
    for bs_seed, bs_path in sorted(bs_by_seed.items()) or [(None, None)]:
        adapters = None
        if bs_path:
            adapters = _load_adapters(bs_path, designer, denoiser, api)
        # The J03 arms share ONE event decode; only the feedback differs.
        shared = None
        # R7.4: U03 is ALWAYS included, not only when the J03 mapping is
        # empty. It is the unadapted baseline and omitting it whenever both
        # A_BS seeds are supplied removed the very row the comparison needs.
        arms = [("J03", None, None)]
        for label, pick in sorted(feedback_selection.items()):
            # R7.2: read the arm and seed as STRUCTURED fields. Parsing
            # "early_s_full_s0" with split("_s")[0] yields "early", which then
            # fails the expected-arm check for a legitimate checkpoint.
            if int(pick.get("bs_seed", -1)) != int(bs_seed):
                continue
            arms.append((label, pick["checkpoint"], pick.get("arm")))
        for arm_label, feedback_path, conditioner_arm in arms:
            rows.append(_one_arm(
                arm_label=arm_label, feedback_path=feedback_path,
                conditioner_arm=conditioner_arm,
                shared=shared, adapters=adapters, recorded=recorded,
                denoiser=denoiser, structure=structure, designer=designer,
                schedule=schedule, choice=choice, prefix_id=prefix_id,
                name=name, length=length, gen_seed=gen_seed, bs_seed=bs_seed,
                args=args, out=out, api=api,
            ))
            if shared is None:
                shared = rows[-1].pop("_shared", None)
            else:
                rows[-1].pop("_shared", None)
    return rows


def _one_arm(*, arm_label, feedback_path, conditioner_arm, shared, adapters,
             recorded, denoiser,
             structure, designer, schedule, choice, prefix_id, name, length,
             gen_seed, bs_seed, args, out, api):
    """Resume from the recorded event and finish this arm's trajectory."""
    import torch

    started = time.time()
    conditioner = None
    if feedback_path:
        c_s, c_z = api["conditioning_widths"](denoiser.model)
        if not conditioner_arm:
            raise SystemExit(
                f"{arm_label}: the selection artifact records no `arm`, so the "
                "expected-arm check cannot run. E1's full and bb_only "
                "variants have identical parameter shapes, so the wrong one "
                "would load cleanly and be reported as the right one."
            )
        conditioner, _report, _ident = api["load_feedback"](
            feedback_path,
            expected=api["expected_policy"](
                # R7.1: the REAL A_BS path. Passing None made expected_policy
                # record bs_checkpoint_sha256=None, which then mismatched the
                # checkpoint's recorded hash and rejected a valid pair.
                bs_checkpoint=bs_path_for(bs_seed, args),
                fampnn_checkpoint=args.fampnn_checkpoint,
                pxdesign_donor=_donor_file(args.checkpoint_dir),
                bs_weights="ema", context=args.context,
                seq_steps=args.seq_steps, pack_steps=args.pack_steps,
                temperature=args.temperature,
            ),
            expected_arm=conditioner_arm,
            c_h_V=api["node_feature_dim"](designer.model),
            c_s=c_s, c_z=c_z,
            c_token=api["token_feature_dim"](denoiser.model),
            device=denoiser.device,
            allow_transfer=args.allow_feedback_policy_transfer,
        )

    state = {"products": shared, "feedback_norm": 0.0}

    with api["BackboneTap"](denoiser.model.diffusion_module) as tap:

        def feedback(sampler_state):
            if state["products"] is None:
                state["products"] = api["prepare_event"](
                    denoise=lambda x, s, **kw: denoiser.denoise(
                        x, s, feedback=None, tap=tap
                    ),
                    x_noisy=sampler_state.x_noisy,
                    sigma=float(sampler_state.sigma.reshape(-1)[0]),
                    structure=structure, designer=designer,
                    adapters=adapters, context=args.context, seed=gen_seed,
                    design_id=prefix_id, target=name, tap=tap,
                    # REQUIRED: the bb_only arm shares this event and reads
                    # h_base. Without it the control cannot run, and the
                    # cache-side fix alone did not cover this path.
                    want_h_base=True,
                )
            products = state["products"]
            if conditioner is None:
                return None
            raw, _stats = conditioner(
                products.packed,
                torch.full((1,), products.sigma, device=denoiser.device),
            )
            delta = api["mask_feedback"](
                raw, products.binder_mask, zero_bypass=True
            )
            if delta is not None:
                api["assert_target_rows_untouched"](delta, products.binder_mask)
                state["feedback_norm"] = float(
                    delta.delta_single.norm()
                    if getattr(delta, "delta_single", None) is not None
                    else 0.0
                )
            return delta

        stream = api["RngStream"](
            "integrated", gen_seed, device=_cuda(denoiser.device)
        )
        x0, _records, stats = api["run_trajectory"](
            denoise=lambda x, s, *, feedback=None: denoiser.denoise(
                x, s, feedback=feedback, tap=tap
            ),
            schedule=schedule, n_atom=denoiser.n_atom,
            device=denoiser.device, n_sample=1,
            step_scale_eta=args.step_scale_eta, stream=stream,
            resume=recorded, event=choice.key, feedback=feedback,
        )
        conditioning_injections = tap.conditioning_injections

    products = state["products"]
    sample_id = f"{prefix_id}__{arm_label}" + (
        f"_bs{bs_seed}" if bs_seed is not None else ""
    )
    # R7.5: repack the event's sequence on THIS arm's final backbone and write
    # the PDB. Returning design_pdb="" meant the matrix could not feed AF2-IG
    # at all, which is the whole point of generating.
    pdb, packed_coords = _finalise(
        x0=x0, products=products, structure=structure, designer=designer,
        adapters=adapters, args=args, out=out, sample_id=sample_id,
        pack_seed=gen_seed, api=api,
    )
    geometry = _geometry(x0, structure, products.binder_mask)
    row = {
        "sample_id": sample_id, "target": name, "binder_length": length,
        "sequence": products.binder_sequence,
        "design_pdb": str(pdb), "binder_chain": geometry["binder_chain"],
        "target_chains": ",".join(geometry["target_chains"]),
        "arm": arm_label, "feedback_arm": (arm_label if conditioner else ""),
        "bs_seed": bs_seed, "generation_seed": gen_seed,
        "shared_prefix_id": prefix_id,
        "requested_sigma": choice.requested_sigma,
        "actual_sigma": choice.actual_sigma, "event_step": choice.step,
        "solver_calls": int(stats["calls"]),
        "conditioning_injections": int(conditioning_injections),
        "decode_hook_calls": products.provenance.get("decode_hook_calls"),
        "delta_h_norm": products.provenance.get("delta_h_norm"),
        "feedback_norm": state["feedback_norm"],
        "event_to_final_aligned_rmsd": _aligned(products.bb0, x0),
        "min_bb_bb_distance": geometry["min_bb_bb"],
        "interface_clashes": geometry["clashes"],
        "seconds": round(time.time() - started, 2),
        "_shared": products,
    }
    print(f"  {arm_label}: injections={row['conditioning_injections']} "
          f"calls={row['solver_calls']} "
          f"event->final={row['event_to_final_aligned_rmsd']:.2f} A "
          f"minBB={geometry['min_bb_bb']:.2f} A clashes={geometry['clashes']}")
    torch.save({"x0": x0.detach().cpu()},
               out / "diagnostics" / f"{sample_id}.backbone.pt")
    return row


def _finalise(*, x0, products, structure, designer, adapters, args, out,
              sample_id, pack_seed, api):
    """Repack the event sequence on the final backbone and write a PDB."""
    import numpy as np
    import torch

    from pxf import atom37
    from pxf.bench.backbone_inputs import build_design_inputs, check_design_mask
    from pxf.bench.coupled_design import conditioned
    from pxf.bench.integrated import _packed_coords, _packed_mask
    from fampnn.model.sd_model import SeqDenoiser

    topology = structure.topology
    a2t = np.asarray(topology.atom_to_token_idx.cpu()).astype(int)
    res_names = np.asarray(topology.res_names)
    design = check_design_mask(
        np.asarray(structure.design_mask.cpu()), res_names=res_names,
        atom_to_token=a2t, n_tokens=int(structure.num_tokens),
        what=sample_id,
    )
    final_inputs = build_design_inputs(
        x0=x0.reshape(-1, 3),
        a_token=products.a_token.reshape(int(structure.num_tokens), -1),
        sigma=products.sigma, atom_names=np.asarray(topology.atom_names),
        res_names=res_names, atom_to_token=a2t,
        n_tokens=int(structure.num_tokens), design=design,
        residue_index=topology.residue_index, asym_id=topology.chain_index,
        design_id=sample_id, target=sample_id,
        binder_length=int(design.sum()), context=args.context,
        device=x0.device,
    )
    with conditioned(designer.model, products.residual):
        packed = designer(
            coords_af2=final_inputs.coords_af2, aatype=products.aatype,
            atom_mask=final_inputs.atom_mask, seq_mask=final_inputs.seq_mask,
            residue_index=final_inputs.residue_index,
            chain_index=final_inputs.chain_index,
            scn_context_mask=final_inputs.sidechain_context_mask,
            seed=pack_seed,
        )
    coords = _packed_coords(packed)
    mask = _packed_mask(packed, final_inputs.atom_mask)
    path = out / "designs" / f"{sample_id}.pdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    length = int(products.aatype.shape[-1])
    SeqDenoiser.save_samples_to_pdb({
        "x_denoised": coords.cpu(),
        "seq_mask": torch.ones(1, length),
        # The packer's OWN occupancy, not an all-zero mask: writing zeros
        # would claim every atom is present including ones it never built.
        "missing_atom_mask": (1.0 - mask).cpu(),
        "residue_index": topology.residue_index.reshape(1, -1).cpu().long(),
        "chain_index": topology.chain_index.reshape(1, -1).cpu().long(),
        "pred_aatype": products.aatype.cpu().long(),
        "psce": products.psce.cpu(),
    }, [str(path)])
    return path, coords


def _donor_file(checkpoint_dir):
    """The weight file inside the release directory; a directory has no hash."""
    from pathlib import Path

    candidate = Path(checkpoint_dir) / "pxdesign_v0.1.0.pt"
    return str(candidate) if candidate.is_file() else None


def bs_path_for(bs_seed, args):
    for entry in args.bs_checkpoint:
        seed, _, path = entry.partition("=")
        if int(seed) == int(bs_seed):
            return path
    raise SystemExit(f"no --bs-checkpoint supplied for seed {bs_seed}")


def _aligned(a, b):
    from pxf.bench.integrated import _aligned_rmsd_local

    return _aligned_rmsd_local(a.reshape(-1, 3), b.reshape(-1, 3))


def _cuda(device):
    """The CUDA device an RngStream must bind to, or None on CPU.

    Without it replay records cuda=None and never captures or restores the
    CUDA generator, so two resumes are not paired on GPU even though they
    share coordinates and the CPU stream.
    """
    import torch

    device = torch.device(device)
    return device if device.type == "cuda" else None


def _geometry(x0, structure, binder_mask):
    """Interface geometry, the cheap chemistry screen the handoff asks for."""
    import torch

    from pxf.bench.backbone_inputs import CHAIN_LETTERS

    asym = structure.topology.chain_index.reshape(-1)
    a2t = structure.topology.atom_to_token_idx.reshape(-1).long()
    binder_tokens = binder_mask.reshape(-1).bool()
    binder_atoms = binder_tokens[a2t]
    coords = x0.reshape(-1, 3)
    b, t = coords[binder_atoms], coords[~binder_atoms]
    if not len(b) or not len(t):
        return {"min_bb_bb": float("nan"), "clashes": -1,
                "binder_chain": "?", "target_chains": []}
    distance = torch.cdist(b.float(), t.float())
    ids = sorted({int(v) for v in asym.tolist()})
    bid = sorted({int(v) for v in asym[binder_tokens].tolist()})
    return {
        "min_bb_bb": float(distance.min()),
        "clashes": int((distance < 2.6).sum()),
        "binder_chain": CHAIN_LETTERS[bid[0]] if len(bid) == 1 else "?",
        "target_chains": [CHAIN_LETTERS[i] for i in ids if i not in bid],
    }


def _load_adapters(path, designer, denoiser, api):
    import torch

    from pxf.couple.adapters import CouplingAdapters

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    identity = state.get("identity") or {}
    for key, want in (("task", "bs_seq_sc_v1"),
                      ("application_mode", "shared_prelogit")):
        if identity.get(key) != want:
            raise SystemExit(f"{path}: {key}={identity.get(key)!r} != {want!r}")
    adapters = CouplingAdapters(
        api["token_feature_dim"](denoiser.model),
        api["node_feature_dim"](designer.model),
    ).to(denoiser.device)
    adapters.enable_bb_to_sc = True
    adapters.load_state_dict(state["adapters"])
    if state.get("ema"):
        from pxf.train.ema import EMA

        ema = EMA(adapters, relative_length=0.25)
        ema.load_state_dict(state["ema"])
        ema.copy_to(adapters)
    adapters.eval().requires_grad_(False)
    return adapters


if __name__ == "__main__":
    main()
