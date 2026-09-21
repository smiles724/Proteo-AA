#!/usr/bin/env python3
"""Design sequences for the shared backbone collection, one row per (design, arm).

    python scripts/design_binder_matrix.py \
        --backbones runs/binder_bench/backbones \
        --out runs/binder_bench/designs_v1 \
        --arm U03 --arm J03=runs/bs_seq_sc/J03_seed0/checkpoints/step00000500.pt

Every arm consumes the SAME cached backbones, which is what makes the on/off
difference paired: coupled and uncoupled see the same geometry, the same
target, the same cached `a_token`. Nothing here regenerates a backbone -- it
cannot, the collection is not reproducible (0.55 A between identical
invocations), so the source digest is carried into every output row and
checked against the manifest on load.

### Three things this gets right on purpose

**sigma.** The adapter is conditioned on the payload's `actual_sigma` (0.8711),
never the scheduled value (0.4355). Churn is exactly 2.0 on this collection, so
the scheduled number would query A_BS at half the noise the denoiser saw.

**The iterative decoder.** `arms.yaml` specifies `seq_steps: 100`, so this
drives `FaMPNNFullAtomDesigner.design` (FaMPNN's shipped `SeqDenoiser.sample`)
and injects the residual with a forward hook. The single-pass `pxf.couple.
codesign` path would be easier and would answer a different question at lower
recovery.

**The uncoupled arm is the uncoupled path.** U03 runs with no hook installed
at all, not with a zero residual. An arithmetic equivalent is not the same
evidence.

Output is the layout `scripts/evaluation/fold_af2ig.py` consumes: `designs/`
holding one PDB per row and a `designs.csv` carrying `sample_id, target,
binder_length, sequence, design_pdb`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
DEFAULT_ARMS = REPO_ROOT / "configs" / "binder_benchmark" / "arms.yaml"

ROW_COLUMNS = (
    # what the AF2-IG scorer requires
    "sample_id", "target", "binder_length", "sequence", "design_pdb",
    # provenance: which backbone, which arm, under what conditioning
    "design_id", "arm", "context", "coupled", "source_sha256",
    "actual_sigma", "event_to_final_rmsd", "delta_h_norm", "hook_calls",
    "seq_steps", "temperature", "seconds",
)


def parse_arm(spec: str) -> tuple[str, str, str | None]:
    """``(label, arm_id, checkpoint)`` from one --arm value.

        U03                     uncoupled, arm_id U03
        J03=ckpt.pt             coupled, arm_id J03
        J03_s1:J03=ckpt.pt      coupled, arm_id J03, reported as J03_s1

    The alias form exists because the seed policy in
    `configs/bs_seq_sc/selection.yaml` is `report_both_and_agreement`: both
    seeds of an arm are separate ROWS but the same arms.yaml DEFINITION, and
    inventing a second definition per seed would let the two drift apart.
    """
    head, sep, path = spec.partition("=")
    label, _, arm_id = head.partition(":")
    return label, (arm_id or label), (path if sep else None)


def load_adapters(path, *, c_token, node_dim, device, fampnn_sha256, weights):
    """Adapters from a bs_seq_sc_v1 checkpoint, with the identity checks kept.

    The same refusals as `scripts/eval_bs_seq_sc.py`: a checkpoint from another
    task or application mode, or trained against a different FaMPNN, is a
    different experiment and is not silently comparable to these arms.
    """
    import torch

    from pxf.couple.adapters import CouplingAdapters

    state = torch.load(path, map_location="cpu", weights_only=False)
    identity = state.get("identity") or {}
    for key, want in (("task", "bs_seq_sc_v1"),
                      ("application_mode", "shared_prelogit")):
        if identity.get(key) != want:
            raise SystemExit(
                f"{path}: records {key}={identity.get(key)!r}, expected {want!r}"
            )
    if identity.get("fampnn_sha256") not in (None, fampnn_sha256):
        raise SystemExit(
            f"{path}: trained against FaMPNN {identity['fampnn_sha256'][:12]}, "
            f"this environment has {fampnn_sha256[:12]}. A donor change is not "
            "an adapter difference."
        )
    adapters = CouplingAdapters(c_token, node_dim).to(device)
    adapters.enable_bb_to_sc = True
    adapters.load_state_dict(state["adapters"])
    used_ema = False
    if state.get("ema") and weights == "ema":
        from pxf.train.ema import EMA

        ema = EMA(adapters, relative_length=0.25)
        ema.load_state_dict(state["ema"])
        ema.copy_to(adapters)
        used_ema = True
    adapters.eval().requires_grad_(False)
    return adapters, {**identity, "weights": "ema" if used_ema else "raw",
                      "step": state.get("step")}


def write_pdb(path, result, inputs):
    """FaMPNN's own writer, psce in the B-factor column."""
    import torch

    from fampnn.model.sd_model import SeqDenoiser
    from pxf import atom37

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    length = int(result.aatype.shape[-1])
    samples = {
        "x_denoised": result.coords_af2[None].cpu(),
        "seq_mask": torch.ones(1, length),
        "missing_atom_mask": torch.zeros(1, length, atom37.NUM_ATOM37),
        "residue_index": inputs.residue_index.cpu().long(),
        "chain_index": inputs.chain_index.cpu().long(),
        "pred_aatype": result.aatype[None].cpu().long(),
        "psce": result.psce[None].cpu(),
    }
    SeqDenoiser.save_samples_to_pdb(samples, [str(path)])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backbones", required=True,
                        help="the collection directory (holds backbones.json)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms-config", default=str(DEFAULT_ARMS))
    parser.add_argument("--arm", action="append", default=[], metavar="LABEL[=CKPT]",
                        help="repeatable; bare LABEL is uncoupled")
    parser.add_argument("--fampnn-variant", default="0.3")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--lengths", nargs="*", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-digests", action="store_true", default=True)
    parser.add_argument("--no-verify-digests", dest="verify_digests",
                        action="store_false")
    args = parser.parse_args()

    import hashlib

    import torch
    import yaml

    import _bootstrap  # noqa: F401  (puts the vendored trees on sys.path)
    from pxf import provenance
    from pxf.bench.backbone_inputs import load_payload, to_design_inputs
    from pxf.bench.coupled_design import build_residual, conditioned
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.device import select_device
    from pxf.sidechain.design import FaMPNNFullAtomDesigner

    base = Path(args.backbones)
    manifest = json.loads((base / "backbones.json").read_text())
    arms_cfg = yaml.safe_load(Path(args.arms_config).read_text())
    by_id = {a["id"]: a for a in arms_cfg["arms"]}
    shared = arms_cfg["shared"]

    device = select_device(args.device)
    designer = FaMPNNFullAtomDesigner(
        variant=args.fampnn_variant,
        seq_steps=shared.get("seq_steps", 100),
        temperature=shared.get("temperature", 0.1),
        psce_threshold=shared.get("psce_threshold", 0.3),
        repack_last=shared.get("repack_last", True),
    ).to(device).eval()
    designer.model.requires_grad_(False)
    fampnn_sha256 = provenance.file_sha256(
        provenance.fampnn_checkpoint(args.fampnn_variant)
    )
    node_dim = node_feature_dim(designer.model)

    # ---- resolve the arms -------------------------------------------------
    arms = []
    for spec in args.arm:
        label, arm_id, ckpt = parse_arm(spec)
        cfg = by_id.get(arm_id)
        if cfg is None:
            raise SystemExit(f"{arm_id}: not an arm in {args.arms_config}")
        context = cfg.get("context", "complex_sc")
        wants_residual = cfg.get("residual_source", "none") != "none"
        if wants_residual and ckpt is None:
            raise SystemExit(
                f"{label} ({arm_id}) declares "
                f"residual_source={cfg['residual_source']!r} "
                "but no checkpoint was given: --arm "
                f"{label}=/path/to/checkpoint.pt"
            )
        if ckpt is not None and not wants_residual:
            raise SystemExit(
                f"{label} declares residual_source: none but a checkpoint was "
                "given. The uncoupled arm runs the uncoupled path."
            )
        arms.append({"label": label, "arm_id": arm_id, "context": context,
                     "checkpoint": ckpt, "cfg": cfg, "adapters": None,
                     "identity": None})

    if not arms:
        raise SystemExit("no --arm given")

    # ---- select the designs ----------------------------------------------
    records = manifest["designs"]
    if args.targets:
        records = [r for r in records if r["target"] in set(args.targets)]
    if args.lengths:
        records = [r for r in records if r["binder_length"] in set(args.lengths)]
    records.sort(key=lambda r: r["design_id"])
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no designs selected")

    out = Path(args.out)
    (out / "designs").mkdir(parents=True, exist_ok=True)
    print(f"{len(records)} backbone(s) x {len(arms)} arm(s) "
          f"= {len(records) * len(arms)} design(s)")
    print("arms: " + ", ".join(
        f"{a['label']}({a['context']}, "
        f"{'coupled' if a['checkpoint'] else 'uncoupled'})" for a in arms))

    rows = []
    for n, record in enumerate(records):
        path = base / "designs" / f"{record['design_id']}.pt"
        if args.verify_digests:
            digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            if digest != record["sha256"]:
                raise SystemExit(
                    f"{record['design_id']}: sha256 {digest[:12]} != manifest "
                    f"{record['sha256'][:12]}. The collection on this disk is "
                    "not the one the manifest describes; every arm's claim to "
                    "have seen one collection rests on this check."
                )
        payload = load_payload(path)

        for arm in arms:
            inputs = to_design_inputs(payload, context=arm["context"],
                                      device=device)
            if arm["checkpoint"] and arm["adapters"] is None:
                arm["adapters"], arm["identity"] = load_adapters(
                    arm["checkpoint"], c_token=inputs.a_token.shape[-1],
                    node_dim=node_dim, device=device,
                    fampnn_sha256=fampnn_sha256, weights=args.weights,
                )
            delta = None
            if arm["adapters"] is not None:
                delta = build_residual(
                    arm["adapters"],
                    binder_mask=inputs.binder_mask,
                    a_token=inputs.a_token,
                    sigma=inputs.sigma,
                    source=arm["cfg"].get("residual_source", "matched"),
                )

            started = time.time()
            # One seed per (backbone, arm-context): the arms differ by the
            # residual, not by their sampling noise.
            seed = args.seed * 1_000_003 + n
            with conditioned(designer.model, delta) as hook:
                result = designer.design(
                    coords_af2=inputs.coords_af2,
                    atom_mask=inputs.atom_mask,
                    aatype=inputs.aatype,
                    seq_mask=inputs.seq_mask,
                    residue_index=inputs.residue_index,
                    chain_index=inputs.chain_index,
                    fixed_sequence_mask=inputs.fixed_sequence_mask,
                    sidechain_context_mask=inputs.sidechain_context_mask,
                    seed=seed,
                )["designs"][0]
            seconds = time.time() - started

            sample_id = f"{record['design_id']}__{arm['label']}"
            pdb = out / "designs" / f"{sample_id}.pdb"
            write_pdb(pdb, result, inputs)

            binder = inputs.binder_mask.reshape(-1).bool().cpu()
            binder_seq = "".join(
                c for c, keep in zip(result.sequence, binder.tolist()) if keep
            )
            if len(binder_seq) != inputs.binder_length:
                raise SystemExit(
                    f"{sample_id}: extracted {len(binder_seq)} binder residues, "
                    f"expected {inputs.binder_length}"
                )
            rows.append({
                "sample_id": sample_id,
                "target": record["target"],
                "binder_length": record["binder_length"],
                "sequence": binder_seq,
                "design_pdb": str(pdb),
                "design_id": record["design_id"],
                "arm": arm["label"],
                "context": arm["context"],
                "coupled": int(delta is not None),
                "source_sha256": record["sha256"],
                "actual_sigma": inputs.sigma,
                "event_to_final_rmsd": record.get("event_to_final_rmsd"),
                "delta_h_norm": (0.0 if delta is None
                                 else float(delta.norm(dim=-1).mean())),
                "hook_calls": hook.calls,
                "seq_steps": designer.seq_steps,
                "temperature": designer.temperature,
                "seconds": round(seconds, 2),
            })
        if (n + 1) % 10 == 0 or n + 1 == len(records):
            print(f"  {n + 1}/{len(records)} backbones", flush=True)

    csv_path = out / "designs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    (out / "provenance.json").write_text(json.dumps({
        "backbones": str(base),
        "backbone_sources": manifest.get("sources"),
        "n_backbones": len(records),
        "fampnn_variant": args.fampnn_variant,
        "fampnn_sha256": fampnn_sha256,
        "weights": args.weights,
        "seed": args.seed,
        "arms": [{"label": a["label"], "arm_id": a["arm_id"],
                  "context": a["context"],
                  "checkpoint": a["checkpoint"], "identity": a["identity"]}
                 for a in arms],
        "shared": shared,
    }, indent=2, default=str))
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
