#!/usr/bin/env python3
"""Score a side-chain checkpoint on the 491-protein validation set, comparably.

WHY THIS EXISTS. The training-time validation number is not comparable across
arms. `Trainer.evaluate()` calls the same `forward_loss` training uses, so an
EDM arm is scored at a RANDOM sigma per item and weighted by lambda(sigma),
while a one-step arm is scored at its fixed sigma_T with no weighting. Those two
numbers have different units and one of them is a draw from a distribution
rather than a measurement -- putting them in the same table would be the same
class of mistake as the 125 A^2 that turned out to be an unresolved-atom floor.

WHAT IT MEASURES. Each arm is run the way it will actually be used:

  one-step arm : template init at sigma_T -> ONE decode -> x0
  EDM arm      : template init -> + sigma_max -> reverse loop, carrying state

and then every arm is scored with the SAME unweighted masked MSE against the GT
side-chain geometry. That is the number a downstream consumer of this model
cares about; the training loss is an optimisation detail.

WHERE THE CONFIG COMES FROM. The checkpoint's own `sidechain_arch` record, not
the command line. Evaluating a checkpoint under a different architecture than it
was trained with produces shape-compatible garbage -- exactly what the trainer's
load-time guard exists to prevent -- and here there is no guard to lean on, so
the config is read from the artifact itself.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", required=True, help="arm name for the output files")
    p.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    p.add_argument("--filtered-index", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=491)
    p.add_argument("--min-n-token", type=int, default=16)
    p.add_argument("--max-n-token", type=int, default=640)
    p.add_argument("--crop-size", type=int, default=640)
    p.add_argument("--max-crop-retries", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--template-provider", default="dunbrack_mode")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--edm-init", choices=["template", "gt"], default="template",
        help="DIAGNOSTIC. 'gt' starts the EDM reverse loop from the noised TARGET "
             "instead of the noised template. Not a valid inference protocol -- it "
             "uses the answer -- and only tells you whether the arm's gap comes from "
             "the starting distribution or from the model.",
    )
    p.add_argument("--protenix-code-dir", default="")
    p.add_argument("--pxdesign-code-dir", default="")
    return p.parse_args()


def _default_training_args(training_module) -> argparse.Namespace:
    saved = sys.argv
    try:
        sys.argv = ["train_protenix_monomer.py"]
        return training_module.parse_args()
    finally:
        sys.argv = saved


def _quantile(values: list[float], q: float) -> float:
    v = sorted(values)
    if not v:
        return float("nan")
    if len(v) == 1:
        return v[0]
    i = q * (len(v) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    import train_protenix_monomer as training

    train_args = _default_training_args(training)
    train_args.training_stage = "sidechain_warmup"
    train_args.data_root = args.data_root
    train_args.output_dir = args.output_dir
    # build_eval_dataloader returns None unless BOTH of these are positive; it is
    # a training-loop gate ("evaluate every N steps") that this script has to
    # satisfy even though it evaluates exactly once.
    train_args.eval_interval = 1
    train_args.eval_samples = int(args.num_samples)
    train_args.eval_filtered_index = args.filtered_index
    train_args.min_n_token = int(args.min_n_token)
    train_args.max_n_token = int(args.max_n_token)
    train_args.crop_size = int(args.crop_size)
    train_args.max_crop_retries = int(args.max_crop_retries)
    train_args.eval_num_workers = int(args.num_workers)
    train_args.template_provider = args.template_provider
    train_args.protenix_code_dir = args.protenix_code_dir
    train_args.pxdesign_code_dir = args.pxdesign_code_dir

    os.environ.setdefault("PROTENIX_ROOT_DIR", str(Path(args.data_root).resolve()))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")

    import torch

    from pxdesign_train.sidechain.frames import to_global

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = ckpt.get("sidechain_arch") or {}
    if not arch:
        raise SystemExit(
            f"{args.checkpoint} records no sidechain_arch. Its S_phi was trained "
            "under an unknown configuration, and evaluating it under a guessed one "
            "silently produces a number for a model that does not exist."
        )
    print(f"checkpoint_arch={json.dumps(arch, sort_keys=True)}")

    # The architecture comes from the artifact, never from this script's flags.
    train_args.sc_frame_aware_head = bool(arch.get("frame_aware_head", False))
    train_args.sc_centre_coord_input = bool(arch.get("centre_coord_input", False))
    train_args.sc_template_residual = bool(arch.get("template_residual", False))
    train_args.sc_edm = bool(arch.get("edm", False))
    training.apply_training_stage_args(train_args)
    training._bootstrap_paths(train_args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    configs = training.build_configs(train_args, device)
    configs.sidechain.bb_context = bool(arch.get("bb_context", True))
    configs.sidechain.a_bs_concat = bool(arch.get("a_bs_concat", True))
    configs.sidechain.q_bs = bool(arch.get("q_bs", False))
    configs.sidechain.type_logits_input = bool(arch.get("type_logits_input", True))

    torch.manual_seed(int(args.seed))

    from pxdesign_train.model import ProtenixDesignTrain

    model = ProtenixDesignTrain(configs).to(device).eval()
    state = ckpt["model"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded missing={len(missing)} unexpected={len(unexpected)}")
    if any(k.startswith("sidechain_module.") for k in missing):
        raise SystemExit("checkpoint carries no S_phi weights; nothing to evaluate")

    # Deterministic inference protocol for the EDM arm (reverse loop from
    # sigma_max, no lambda weighting). A no-op on the one-step arms.
    model.sc_edm_eval = True
    model.sc_edm_eval_from_gt = (args.edm_init == "gt")
    if model.sc_edm_eval_from_gt:
        print("WARNING: --edm-init gt is a DIAGNOSTIC. The reverse loop starts from "
              "the noised ground truth, so this number is NOT achievable at "
              "inference and must not be reported as a result.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_loader, n_eval, eval_index = training.build_eval_dataloader(train_args, output_dir)
    if eval_loader is None:
        raise SystemExit(
            "validation loader was not constructed: build_eval_dataloader "
            f"got eval_interval={train_args.eval_interval} "
            f"eval_samples={train_args.eval_samples}; both must be > 0"
        )
    print(f"validation_rows={n_eval} index={eval_index}")

    rows: list[dict] = []
    total_se, total_atoms = 0.0, 0
    for i, batch in enumerate(eval_loader):
        feat = {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch["input_feature_dict"].items()}
        label = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch["label_dict"].items()}
        # One seed per ITEM, so the number is reproducible and every arm meets the
        # same backbone noise draw.
        torch.manual_seed(int(args.seed) + i)
        with torch.no_grad():
            out = model(feat, label, mode="train")

        pred = out.get("sc_pred_global")
        if pred is None:
            continue
        gt_local = feat["sc_gt_local"].float()
        mask = out["sc_atom_mask"].bool()
        fR, ft = out["sc_frame_R"].float(), out["sc_frame_t"].float()
        tgt = to_global(gt_local, fR, ft)
        while pred.dim() > tgt.dim():
            pred = pred.mean(dim=0)
        se = ((pred.float() - tgt) ** 2).sum(dim=-1)
        m = mask.to(se.dtype).expand_as(se)
        n = float(m.sum())
        if n == 0:
            continue
        mse = float((se * m).sum() / n)
        total_se += float((se * m).sum())
        total_atoms += int(n)
        rows.append({
            "sample_id": batch.get("source_name", [f"item{i}"])[0]
            if isinstance(batch.get("source_name"), list) else f"item{i}",
            "index": i, "n_atoms": int(n),
            "sc_mse": round(mse, 6), "sc_rmsd_A": round(math.sqrt(mse), 6),
        })
        if (i + 1) % 50 == 0:
            print(f"processed={i + 1}/{n_eval}", flush=True)

    if not rows:
        raise SystemExit("no evaluable items")

    per = [r["sc_mse"] for r in rows]
    summary = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "sidechain_arch": arch,
        "protocol": (
            ("edm_reverse_loop_from_noised_GT__DIAGNOSTIC"
             if args.edm_init == "gt" else "edm_reverse_loop")
            if arch.get("edm") else "one_step"
        ),
        # Stamped into the artifact so a number that cannot be reached at inference
        # cannot be mistaken for one that can, however it is later copied around.
        "diagnostic_uses_ground_truth_start": bool(
            arch.get("edm") and args.edm_init == "gt"
        ),
        "unit": "A^2 (unweighted masked mean squared displacement per atom)",
        "n_proteins": len(rows),
        "sidechain_atoms": total_atoms,
        # Atom-weighted: the number a downstream consumer feels. Sample-mean gives
        # every protein equal say regardless of size; they answer different
        # questions, so both are reported rather than one being picked silently.
        "atom_weighted_mse": total_se / max(1, total_atoms),
        "atom_weighted_rmsd_A": math.sqrt(total_se / max(1, total_atoms)),
        "sample_mean_mse": sum(per) / len(per),
        "sample_median_mse": _quantile(per, 0.5),
        "sample_p90_mse": _quantile(per, 0.9),
        "sample_max_mse": max(per),
    }
    (output_dir / f"arm_eval_{args.label}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (output_dir / f"arm_eval_{args.label}_per_protein.csv").open("w") as fh:
        fh.write("sample_id,index,n_atoms,sc_mse,sc_rmsd_A\n")
        for r in rows:
            fh.write(f"{r['sample_id']},{r['index']},{r['n_atoms']},"
                     f"{r['sc_mse']},{r['sc_rmsd_A']}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
