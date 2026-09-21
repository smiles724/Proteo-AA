#!/usr/bin/env python3
"""Score `bs_seq_sc_v1` checkpoints on the held-out reconstruction panel.

    python scripts/eval_bs_seq_sc.py \
        --checkpoint S03=runs/S03_seed0/checkpoints/final.pt \
        --checkpoint J03=runs/J03_seed0/checkpoints/final.pt \
        --include-donor --out runs/bs_seq_sc_eval

Section 9's FIRST stage: held-out reconstruction, on the 31 val dimers that
no arm trained on. Masked-sequence NLL and recovery, the side-chain diffusion
loss, and both reported per sigma with paired noise across arms.

What this deliberately does NOT claim:

  * **Not designability.** That needs iterative generation on shared de novo
    backbones and then complex AF2-IG, which runs in the official-runtime
    environment on HAI. A reconstruction number is a statement about the
    objective, not about binder success, and calling it one would be the
    same error as reporting packing quality from a backbone-only evaluator.
  * **Not native side-chain RMSD under generated identities.** It is not
    defined across different amino-acid atom sets. Where identities differ
    from native the RMSD column is null, not zero.

### Pairing

Every arm sees the same structures, the same masks and the same side-chain
noise, drawn once per (structure, sigma) from a named stream and replayed.
Without that, J03 - S03 mixes the effect of supervision with the effect of a
different random draw, and at the effect sizes expected here the draw wins.

U03 -- the 0.3 donor with the adapter off -- is included by default, because
J03 - U03 is the total-benefit number and S03 - U03 says whether the
side-chain objective alone did anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = REPO_ROOT / "configs" / "gen_stress_prepared.marlowe.parquet"
DONOR_LABEL = "U03"
DEFAULT_SIGMAS = (0.105, 0.314, 0.429, 0.847)


def load_arm(label: str, path: Optional[str], ctx, args) -> dict[str, Any]:
    """An arm is a set of adapter weights, or the donor with none."""
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.fampnn_iface import node_feature_dim

    adapters = CouplingAdapters(
        ctx["driver"].c_token, node_feature_dim(ctx["packer"].model)
    ).to(ctx["device"])
    adapters.enable_bb_to_sc = True
    record: dict[str, Any] = {"arm": label, "checkpoint": path}

    if path is None:
        # U03: the residual is never applied at all. Not a zero adapter --
        # that is a different code path, and the uncoupled arm must be the
        # uncoupled path rather than an arithmetic equivalent of it.
        record.update({"weights": "none", "step": 0, "identity": None})
        return {"adapters": None, "record": record}

    state = torch.load(path, map_location="cpu", weights_only=False)
    identity = state.get("identity") or {}
    for key, want in (("task", "bs_seq_sc_v1"),
                      ("application_mode", "shared_prelogit")):
        if identity.get(key) != want:
            raise SystemExit(
                f"{label}: checkpoint records {key}={identity.get(key)!r}, "
                f"expected {want!r}. Scoring a checkpoint from a different "
                "task or application mode against these arms would compare "
                "two different experiments."
            )
    if identity.get("fampnn_sha256") != ctx["fampnn_sha256"]:
        raise SystemExit(
            f"{label}: trained against FaMPNN {identity.get('fampnn_sha256','?')[:12]}, "
            f"this environment has {ctx['fampnn_sha256'][:12]}. A donor change "
            "is not an adapter difference."
        )
    adapters.load_state_dict(state["adapters"])
    used_ema = False
    if state.get("ema") and args.weights == "ema":
        from pxf.train.ema import EMA

        ema = EMA(adapters, relative_length=0.25)
        ema.load_state_dict(state["ema"])
        ema.copy_to(adapters)
        used_ema = True
    adapters.eval().requires_grad_(False)
    record.update({
        "weights": "ema" if used_ema else "raw",
        "step": state.get("step"),
        "lambda_seq": identity.get("lambda_seq"),
        "lambda_sc": identity.get("lambda_sc"),
        "seed": identity.get("seed"),
        "identity": identity,
    })
    return {"adapters": adapters, "record": record}


def score_one(arm, example, ctx, args, *, sigma: float, noise_seed: int) -> dict[str, Any]:
    """Masked-sequence and side-chain numbers for one arm on one structure."""
    from pxf.couple.shared_prelogit import conditioned_forward
    from pxf.train import losses as loss_fns
    from pxf.train import step as train_step

    seq_module = ctx["packer"].model.denoiser.seq_design_module
    masks = example["masks"]

    with torch.no_grad():
        if arm["adapters"] is None:
            # The uncoupled path: logits straight off the donor's own h_V.
            logits = seq_module.W_out(example["features"]["h_V"])
            conditioned = example["features"]
            delta_norm = 0.0
        else:
            logits, conditioned, delta = conditioned_forward(
                seq_module, example["features"], adapters=arm["adapters"],
                a_token=example["a_token"],
                sigma_b=torch.full_like(example["sigma"], sigma),
                roles=example["roles"],
            )
            delta_norm = float(delta.norm(dim=-1).mean())

        sequence, seq_stats = loss_fns.sequence_mlm_loss(
            logits, masks.aatype_true, masks.seq_mlm_mask, masks.seq_mask,
            seq_unk_mask=train_step.batch_masks(example["batch"])["seq_unk_mask"],
        )
        # Paired noise: the same draw for every arm at this (structure, sigma).
        generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        sidechain, sc_stats = train_step.diffusion_loss(
            ctx["packer"].model, example["batch"], conditioned,
            multiplier=args.multiplier, generator=generator, scn_mlm_mask=None,
        )

    return {
        "sigma_b": sigma,
        "seq_nll": float(sequence),
        "seq_per_token_nll": float(seq_stats["mlm_per_token"]),
        "seq_recovery": float(seq_stats["sequence_accuracy"]),
        "n_scored": int(seq_stats["masked_residues"]),
        "sc_loss": float(sidechain),
        "delta_h_norm": delta_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--checkpoint", action="append", default=[],
                        metavar="LABEL=PATH")
    parser.add_argument("--include-donor", action="store_true", default=True)
    parser.add_argument("--no-donor", dest="include_donor", action="store_false")
    parser.add_argument("--pxdesign-donor", default=(
        "/scratch/m000137-pm06/Proteo-AA/pxf/component_donors/pxdesign_v0.1.0.pt"))
    parser.add_argument("--fampnn-variant", default="0.3")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    parser.add_argument("--mask-fraction", type=float, default=0.5)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--multiplier", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import pandas as pd

    from pxf import provenance
    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.device import select_device
    from pxf.sidechain.fampnn import FaMPNNSideChainPacker
    from pxf.train.bs_seq_sc import prepare_example

    device = select_device(args.device)
    packer = FaMPNNSideChainPacker(variant=args.fampnn_variant).to(device).eval()
    packer.model.requires_grad_(False)
    px_model, _c, _r = load_backbone_model(args.pxdesign_donor, device=device)
    px_model.requires_grad_(False)
    ctx = {
        "device": device, "packer": packer,
        "driver": PXDesignBackboneDriver(px_model),
        "fampnn_sha256": provenance.file_sha256(
            provenance.fampnn_checkpoint(args.fampnn_variant)
        ),
    }

    specs = [(DONOR_LABEL, None)] if args.include_donor else []
    for entry in args.checkpoint:
        label, _, path = entry.partition("=")
        specs.append((label, path))
    arms = [load_arm(label, path, ctx, args) for label, path in specs]
    print("arms: " + ", ".join(
        f"{a['record']['arm']}({a['record']['weights']}@{a['record']['step']})"
        for a in arms
    ))

    frame = pd.read_parquet(args.panel)
    if args.limit:
        frame = frame.head(args.limit)
    if "split" in frame and set(frame["split"]) != {"val"}:
        print(f"WARNING: panel splits are {sorted(set(frame['split']))}, not val")

    rows = []
    for position, row in enumerate(frame.itertuples()):
        try:
            example = prepare_example(
                row.cif_path, row.converted_binder_chain,
                packer=packer, driver=ctx["driver"], device=device,
                sigma_b=args.sigmas[0], mask_fraction=args.mask_fraction,
                crop_size=args.crop_size,
                # One mask draw per structure, shared by every arm.
                seed=args.seed * 100_003 + position,
            )
        except Exception as exc:  # noqa: BLE001 - a failure is a finding
            print(f"ERROR {row.example_id}: {type(exc).__name__}: {str(exc)[:110]}")
            rows.append({"example_id": row.example_id, "error": str(exc)[:300]})
            continue

        for sigma in args.sigmas:
            noise_seed = args.seed * 7_919 + position * 31 + int(sigma * 1e4)
            for arm in arms:
                score = score_one(arm, example, ctx, args,
                                  sigma=sigma, noise_seed=noise_seed)
                rows.append({
                    "example_id": row.example_id, "arm": arm["record"]["arm"],
                    "tokens": example["tokens"], **score,
                })
        done = [r for r in rows if r.get("example_id") == row.example_id and "error" not in r]
        print(f"  {row.example_id:<10} {len(done)} scores")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame_out = pd.DataFrame(rows)
    frame_out.to_csv(out / "scores.csv", index=False)

    summary: dict[str, Any] = {"arms": [a["record"] for a in arms],
                               "panel": args.panel, "sigmas": args.sigmas,
                               "weights": args.weights, "seed": args.seed}
    scored = frame_out[frame_out.get("error").isna()] if "error" in frame_out else frame_out
    if len(scored):
        agg = scored.groupby(["arm", "sigma_b"])[
            ["seq_per_token_nll", "seq_recovery", "sc_loss", "delta_h_norm"]
        ].median().round(6)
        summary["median_by_arm_sigma"] = json.loads(agg.reset_index().to_json(orient="records"))
        print("\nmedian by arm and sigma:")
        print(agg.to_string())
        # Paired differences, computed per structure and then aggregated --
        # not as a difference of medians, which is not the median difference.
        pivot = scored.pivot_table(
            index=["example_id", "sigma_b"], columns="arm",
            values=["seq_per_token_nll", "sc_loss"],
        )
        deltas = {}
        labels = [a["record"]["arm"] for a in arms]
        for metric in ("seq_per_token_nll", "sc_loss"):
            for a in labels:
                for b in labels:
                    if a >= b or (metric, a) not in pivot or (metric, b) not in pivot:
                        continue
                    diff = (pivot[(metric, a)] - pivot[(metric, b)]).dropna()
                    if len(diff):
                        deltas[f"{metric}:{a}-{b}"] = {
                            "median": float(diff.median()),
                            "n_pairs": int(len(diff)),
                            "frac_negative": float((diff < 0).mean()),
                        }
        summary["paired_deltas"] = deltas
        if deltas:
            print("\npaired differences (per structure, then aggregated):")
            for key, value in deltas.items():
                print(f"  {key:<34} median {value['median']:+.5f}  "
                      f"n={value['n_pairs']}  frac<0 {value['frac_negative']:.2f}")

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"\nwrote {out / 'scores.csv'} and {out / 'summary.json'}")
    print("reconstruction only -- this is not designability")


if __name__ == "__main__":
    main()
