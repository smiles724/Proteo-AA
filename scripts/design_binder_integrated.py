#!/usr/bin/env python3
"""Integrated binder design: one PXDesign trajectory with one coupling event.

    python scripts/design_binder_integrated.py \
        --input runs/binder_bench/targets/configs/PDL1.yaml --target PDL1 \
        --binder-length 100 \
        --checkpoint-dir /path/to/pxdesign_release \
        --bs-checkpoint runs/bs_seq_sc/J03_seed0/checkpoints/step00000500.pt \
        --feedback-checkpoint /path/to/matching_feedback.pt \
        --feedback-arm early_s_full \
        --event-sigma 0.429 --seed 101 --n-samples 2 \
        --out runs/integrated/J03_early_full

Omit ``--feedback-checkpoint`` for the A_BS-only control; omit
``--bs-checkpoint`` as well for U03, the unadapted donor.

**This needs PXDesign's OFFICIAL runtime** (Protenix 0.5.0+pxd) and a PRISTINE
PXDesign checkout -- the repo's working copy patches ``embedders.py`` for the
vendored Protenix v2.0.0 and fails here with ``KeyError: 'd_lm'``. Build the
environment with ``scripts/utilities/install_pxdesign_official.sh``. The guard
below runs before anything bootstraps vendored sources, so the failure is one
line rather than an ImportError four frames inside PXDesign's pipeline.

### What this is, next to the cached-backbone matrix

`scripts/design_binder_matrix.py` generates all backbones first and designs
sequences afterwards: two-stage by construction, and its arms share one cached
backbone collection, which is what makes them strictly paired. This designs the
sequence INSIDE the trajectory and corrects the backbone with it, so the two
co-determine each other.

The two protocols are not comparable design-for-design and their outputs are
not part of the same paired collection. Two reasons, both structural: separate
CUDA trajectories are not bit-identical even at equal seeds, and A_BS is
queried at a different noise level (see below).

### The sigma difference from the matrix, stated because it is easy to miss

``--event-sigma`` is the CHURNED sigma the denoiser actually sees. Requesting
0.429 places the event where A_BS is queried at 0.429 -- its training noise.
`cache_binder_backbones.py` instead selected by the SCHEDULED level, landing on
0.4355 scheduled / **0.8711 actual**, so the cached-backbone arms query A_BS at
about twice its training sigma. Neither is a bug; they are different choices,
and the realised value is recorded in every row of both.

### Paired generation needs the other script

One arm per invocation cannot give a strictly paired comparison, because the
prefix before the event is not reproducible across processes.
`scripts/run_integrated_binder_matrix.py` generates the common prefix once and
branches the solver from that shared state. Use this script for single arms,
smoke runs, and diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ROW_COLUMNS = (
    "sample_id", "target", "binder_length", "sequence", "design_pdb",
    "binder_chain", "target_chains",
    "arm", "feedback_arm", "coupled", "seed",
    "requested_sigma", "actual_sigma", "scheduled_sigma", "sigma_churn_ratio",
    "event_step", "n_step", "step_scale_eta",
    "solver_calls", "model_calls_total", "provisional_calls",
    "solver_injections", "conditioning_injections", "late_injections",
    "decode_hook_calls", "final_pack_hook_calls",
    "delta_h_norm", "feedback_norm",
    "event_to_final_aligned_rmsd", "event_to_final_raw_rmsd",
    "seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True,
                        help="prepared target YAML, as PXDesign's CLI takes")
    parser.add_argument("--target", required=True)
    parser.add_argument("--binder-length", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True,
                        help="the PXDesign release checkpoint directory")
    parser.add_argument("--bs-checkpoint", default=None,
                        help="selected J03 A_BS; omit for U03 (unadapted donor)")
    parser.add_argument("--bs-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--fampnn-checkpoint", default=None,
                        help="default: the pinned FaMPNN 0.3 weights")
    parser.add_argument("--fampnn-variant", default="0.3")
    parser.add_argument("--feedback-checkpoint", default=None,
                        help="trained E1/A_SB checkpoint; omit for the "
                             "A_BS-only control")
    parser.add_argument("--feedback-arm", default="early_s_full")
    parser.add_argument("--feedback-weights", choices=("ema", "raw"),
                        default="ema")
    parser.add_argument("--allow-feedback-policy-transfer", action="store_true",
                        help="record policy differences and proceed. Never "
                             "permits a donor mismatch.")
    parser.add_argument("--event-sigma", type=float, default=0.429,
                        help="the CHURNED sigma to place the event at")
    parser.add_argument("--n-step", type=int, default=400)
    parser.add_argument("--step-scale-eta", type=float, default=2.5)
    parser.add_argument("--context", default="complex_sc",
                        choices=("complex_sc", "complex", "binder_only"))
    parser.add_argument("--seq-steps", type=int, default=100)
    parser.add_argument("--pack-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--psce-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--use-msa", action="store_true")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Before ANY vendored source can reach sys.path.
    sys.path.insert(0, str(REPO_ROOT))
    from pxf.official.require import require_official_protenix

    require_official_protenix("design_binder_integrated")

    import torch

    from pxf.bench.integrated import run_integrated, select_event
    from pxf.bench.integrated_checkpoints import (expected_policy,
                                                  file_sha256, load_feedback)
    from pxf.official.bridge import OfficialStructure
    from pxf.official.runtime import OfficialDenoiser, build_runner, first_batch
    from pxf.sidechain.design import FaMPNNFullAtomDesigner

    out = Path(args.out)
    (out / "designs").mkdir(parents=True, exist_ok=True)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)

    runner = build_runner(
        args.input, str(out / "pxdesign"),
        load_checkpoint_dir=args.checkpoint_dir,
        n_step=args.n_step, n_sample=1, use_msa=args.use_msa, dtype=args.dtype,
        eta_type="const", eta_min=args.step_scale_eta,
        eta_max=args.step_scale_eta,
    )
    data, atom_array = first_batch(runner)
    denoiser = OfficialDenoiser(runner, data)
    structure = OfficialStructure(
        atom_array, denoiser.features,
        int(denoiser.features["residue_index"].reshape(-1).shape[0]),
        device=denoiser.device,
    )

    designer = FaMPNNFullAtomDesigner(
        args.fampnn_checkpoint, variant=args.fampnn_variant,
        seq_steps=args.seq_steps, temperature=args.temperature,
        psce_threshold=args.psce_threshold, repack_last=True,
    ).to(denoiser.device).eval()
    designer.model.requires_grad_(False)

    adapters = None
    bs_identity = None
    if args.bs_checkpoint:
        adapters, bs_identity = _load_adapters(
            args.bs_checkpoint, designer, denoiser, args.bs_weights
        )

    conditioner, policy_report, fb_identity = None, None, None
    if args.feedback_checkpoint:
        if adapters is None:
            raise SystemExit(
                "--feedback-checkpoint without --bs-checkpoint: the feedback "
                "module was trained to correct states an A_BS-conditioned "
                "decode produced, so running it on an unadapted decode asks it "
                "a question it never saw."
            )
        widths = _conditioning_widths(denoiser.model)
        conditioner, policy_report, fb_identity = load_feedback(
            args.feedback_checkpoint,
            expected=expected_policy(
                bs_checkpoint=args.bs_checkpoint,
                bs_weights=args.bs_weights, context=args.context,
                seq_steps=args.seq_steps, pack_steps=args.pack_steps,
                temperature=args.temperature,
            ),
            expected_arm=args.feedback_arm,
            c_h_V=_node_dim(designer.model), c_token=widths["c_token"],
            c_s=widths["c_s"], c_z=widths["c_z"],
            device=denoiser.device,
            allow_transfer=args.allow_feedback_policy_transfer,
            weights=args.feedback_weights,
        )

    arm = ("U03" if adapters is None else
           ("J03" if conditioner is None else f"J03+{args.feedback_arm}"))
    choice = select_event(denoiser.schedule(args.n_step), args.event_sigma)
    print(f"arm={arm}  event step {choice.step}/{choice.n_levels - 1}  "
          f"requested sigma {choice.requested_sigma:.4f} -> actual "
          f"{choice.actual_sigma:.4f} (scheduled {choice.scheduled_sigma:.4f}, "
          f"churn {choice.churn_ratio:.2f}x)")

    rows = []
    for index in range(args.n_samples):
        seed = args.seed + index
        sample_id = f"{args.target}_L{args.binder_length}_s{seed}__{arm}"
        sample = run_integrated(
            denoiser=denoiser, structure=structure, designer=designer,
            adapters=adapters, conditioner=conditioner,
            event_sigma=args.event_sigma, n_step=args.n_step,
            step_scale_eta=args.step_scale_eta, context=args.context,
            seed=seed, design_id=sample_id, target=args.target,
        )
        pdb = _write_pdb(out / "designs" / f"{sample_id}.pdb", sample, structure)
        (out / "diagnostics" / f"{sample_id}.json").write_text(
            json.dumps(sample.diagnostics, indent=2, default=str)
        )
        torch.save(
            {"x0": sample.x0.cpu(), "aatype": sample.aatype.cpu(),
             "binder_mask": sample.binder_mask.cpu()},
            out / "diagnostics" / f"{sample_id}.backbone.pt",
        )
        binder_chain, target_chains = _chains(sample, structure)
        d = sample.diagnostics
        rows.append({
            "sample_id": sample_id, "target": args.target,
            "binder_length": args.binder_length,
            "sequence": sample.binder_sequence, "design_pdb": str(pdb),
            "binder_chain": binder_chain,
            "target_chains": ",".join(target_chains),
            "arm": arm, "feedback_arm": (args.feedback_arm if conditioner else ""),
            "coupled": int(adapters is not None), "seed": seed,
            **{k: d.get(k) for k in (
                "requested_sigma", "actual_sigma", "scheduled_sigma",
                "sigma_churn_ratio", "event_step", "n_step", "step_scale_eta",
                "solver_calls", "model_calls_total", "provisional_calls",
                "solver_injections", "conditioning_injections",
                "late_injections", "decode_hook_calls",
                "final_pack_hook_calls", "delta_h_norm", "feedback_norm",
                "event_to_final_aligned_rmsd", "event_to_final_raw_rmsd",
                "seconds")},
        })
        print(f"  {sample_id}: injections={d['conditioning_injections']} "
              f"calls={d['model_calls_total']} "
              f"event->final(aligned)={d['event_to_final_aligned_rmsd']:.2f} A "
              f"{d['seconds']:.0f}s")

    with (out / "designs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    (out / "provenance.json").write_text(json.dumps({
        "arm": arm,
        "protocol": "integrated_single_event",
        "event": choice.record(),
        "bs_checkpoint": args.bs_checkpoint,
        "bs_checkpoint_sha256": (
            file_sha256(args.bs_checkpoint) if args.bs_checkpoint else None
        ),
        "bs_weights": args.bs_weights, "bs_identity": bs_identity,
        "feedback_checkpoint": args.feedback_checkpoint,
        "feedback_arm": args.feedback_arm if conditioner else None,
        "feedback_identity": fb_identity,
        "feedback_policy": None if policy_report is None else policy_report.record(),
        "decoder": {"seq_steps": args.seq_steps, "pack_steps": args.pack_steps,
                    "temperature": args.temperature,
                    "psce_threshold": args.psce_threshold,
                    "context": args.context},
        "n_step": args.n_step, "step_scale_eta": args.step_scale_eta,
        "n_samples": args.n_samples, "seed": args.seed,
        "not_paired_with": (
            "designs produced by scripts/design_binder_matrix.py: separate CUDA "
            "trajectories are not bit-identical and A_BS is queried at a "
            "different sigma there (0.8711 vs this run's actual value)"
        ),
    }, indent=2, default=str))
    print(f"wrote {out / 'designs.csv'} ({len(rows)} row(s))")


def _load_adapters(path, designer, denoiser, weights):
    import torch

    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.fampnn_iface import node_feature_dim

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    identity = state.get("identity") or {}
    for key, want in (("task", "bs_seq_sc_v1"),
                      ("application_mode", "shared_prelogit")):
        if identity.get(key) != want:
            raise SystemExit(
                f"{path}: records {key}={identity.get(key)!r}, expected {want!r}"
            )
    adapters = CouplingAdapters(
        int(identity.get("c_token") or _conditioning_widths(denoiser.model)["c_token"]),
        node_feature_dim(designer.model),
    ).to(denoiser.device)
    adapters.enable_bb_to_sc = True
    adapters.load_state_dict(state["adapters"])
    if state.get("ema") and weights == "ema":
        from pxf.train.ema import EMA

        ema = EMA(adapters, relative_length=0.25)
        ema.load_state_dict(state["ema"])
        ema.copy_to(adapters)
    adapters.eval().requires_grad_(False)
    return adapters, {**identity, "weights": weights, "step": state.get("step")}


def _conditioning_widths(model):
    """``{c_s, c_z, c_token}``. ``conditioning_widths`` returns a TUPLE."""
    from pxf.couple.pxdesign_iface import conditioning_widths, token_feature_dim

    c_s, c_z = conditioning_widths(model)
    return {"c_s": c_s, "c_z": c_z, "c_token": token_feature_dim(model)}


def _node_dim(model):
    from pxf.couple.fampnn_iface import node_feature_dim

    return node_feature_dim(model)


def _chains(sample, structure):
    from pxf.bench.backbone_inputs import CHAIN_LETTERS

    asym = structure.topology.chain_index.reshape(-1).cpu()
    binder = sample.binder_mask.reshape(-1).bool().cpu()
    ids = sorted({int(v) for v in asym.tolist()})
    b = sorted({int(v) for v in asym[binder].tolist()})
    if len(b) != 1:
        raise SystemExit(f"the binder spans {len(b)} chain(s) ({b}); expected one")
    letter = CHAIN_LETTERS[b[0]]
    return letter, [CHAIN_LETTERS[i] for i in ids if i != b[0]]


def _write_pdb(path, sample, structure):
    import torch

    from fampnn.model.sd_model import SeqDenoiser
    from pxf import atom37

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    length = int(sample.aatype.shape[-1])
    SeqDenoiser.save_samples_to_pdb({
        "x_denoised": sample.coords_af2.cpu(),
        "seq_mask": torch.ones(1, length),
        "missing_atom_mask": torch.zeros(1, length, atom37.NUM_ATOM37),
        "residue_index": structure.topology.residue_index.reshape(1, -1).cpu().long(),
        "chain_index": structure.topology.chain_index.reshape(1, -1).cpu().long(),
        "pred_aatype": sample.aatype.cpu().long(),
        "psce": sample.psce.cpu(),
    }, [str(path)])
    return path


if __name__ == "__main__":
    main()
