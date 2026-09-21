#!/usr/bin/env python3
"""Train A_BS jointly on masked sequence + side-chain diffusion (`bs_seq_sc_v1`).

    python scripts/train_bs_seq_sc.py --config configs/bs_seq_sc/J03.yaml \
        --train-manifest <train-split parquet> --out runs/J03_seed0

A FRESH adapter against a FROZEN FaMPNN 0.3 and a FROZEN PXDesign. Nothing
else trains. This is not fine-tuning FaMPNN.

    L_J = lambda_seq * L_MLM + lambda_SC * L_diffusion

S03 is this with `lambda_seq: 0`. It still builds the same masks and encodes
the same partially masked context, so J03 - S03 isolates sequence supervision
with routing held fixed. That is why the two configs differ in one number.

### Four things this refuses to do

**Train on the val split.** The manifest's `split` column is checked and a
run whose examples are not all `train` aborts. The 31 prepared dimers are
val; they validate the mask plumbing and section 7, and spending them on
2,000 updates costs sections 7 and 9 their held-out set. A post-hoc split
does not undo it, so the check is at startup.

**Start J03 without a calibrated lambda_seq.** Section 7 requires the
coefficient to come from a training-only calibration batch. The preflight
reports a suggestion per structure and deliberately does not adopt one, so
`lambda_seq: auto` is an error here rather than a silent default. The
measured ratio is ~1e-3 -- unbalanced, the sequence term is two to three
orders of magnitude larger and the side-chain loss RISES -- so the default
of 1.0 is actively wrong and must not be reachable by omission.

**Load the existing 0.0 adapter onto 0.3.** Identical widths (both donors are
10,513,177 parameters) and a clean `load_state_dict` establish nothing about
feature distributions. `--init-from` is available but records itself as a
declared transfer experiment and refuses to inherit optimizer or EMA state.

**Run the packing hook at the same time.** shared_prelogit plus the legacy
hook adds the residual twice on the side-chain side and once on the sequence
side, which is neither mode. `check_exclusive` enforces it.

### The checkpoint contract

Every checkpoint records what it would take to reproduce or invalidate it:
task and schema, application mode, both donors by SHA-256 (not by path, so
copying the weights to another cluster stays valid), the adapter's
architecture and zero-init policy, the sigma distribution and gate, the data
manifest digest and split, the objective coefficients, optimizer state, seed
and the EMA/raw choice. `--verify-only` re-checks a written checkpoint
against the live environment without training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import torch  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bs_seq_sc/1"


def _digest(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# ------------------------------------------------------------------ the data


def load_manifest(path: Path, *, require_split: str = "train") -> Any:
    """Rows to train on, refusing anything that is not the declared split."""
    import pandas as pd

    frame = pd.read_parquet(path)
    if "split" not in frame.columns:
        raise SystemExit(
            f"{path}: no `split` column, so this cannot be shown to be "
            f"{require_split}. Training refuses to guess."
        )
    present = sorted(set(frame["split"]))
    wrong = [s for s in present if s != require_split]
    if wrong:
        raise SystemExit(
            f"{path}: contains split(s) {wrong}, expected only "
            f"{require_split!r}.\n"
            "The prepared 31 dimers are the VAL split: they validate the mask "
            "plumbing and the section 7 checks. Training on them spends the "
            "held-out set that sections 7 and 9 need, and a post-hoc split "
            "does not undo it. Pass --allow-split to override, and say why in "
            "the run notes."
        )
    return frame


# ---------------------------------------------------------------- the model


def build(config: dict[str, Any], args) -> dict[str, Any]:
    from pxf import provenance
    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.couple.shared_prelogit import check_exclusive
    from pxf.device import select_device
    from pxf.sidechain.fampnn import FaMPNNSideChainPacker
    from pxf.train.bs_seq_sc import APPLICATION_MODE, TASK, freeze_everything_but

    variant = str(config["donor"]["fampnn_variant"])
    if variant != "0.3":
        raise SystemExit(
            f"bs_seq_sc_v1 is defined against FaMPNN 0.3; got {variant!r}. "
            "There is no fallback to 0.0 -- that is the legacy family, and a "
            "0.0/0.3 difference is a donor change, never an adapter gain."
        )
    check_exclusive(APPLICATION_MODE, packing_hook_active=False)

    device = select_device(args.device)
    packer = FaMPNNSideChainPacker(variant=variant).to(device)
    donor_path = provenance.fampnn_checkpoint(variant)
    donor_sha = provenance.file_sha256(donor_path)

    declared = (config["donor"].get("fampnn_sha256") or "").strip()
    if declared and declared != donor_sha:
        raise SystemExit(
            f"FaMPNN {variant} on disk hashes {donor_sha[:16]} but the config "
            f"declares {declared[:16]}. Same label, different bytes."
        )

    px_path = Path(config["donor"]["pxdesign"])
    px_model, _cfg, _rec = load_backbone_model(str(px_path), device=device)
    driver = PXDesignBackboneDriver(px_model)
    px_sha = _digest(px_path)

    adapters = CouplingAdapters(
        driver.c_token, node_feature_dim(packer.model)
    ).to(device)
    adapters.enable_bb_to_sc = True
    if not adapters.is_identity():
        raise SystemExit(
            "a fresh adapter must be the identity at initialisation (zero "
            "output projection); it is not, so the zero-init parity check "
            "would be meaningless"
        )

    initialised_from = None
    if args.init_from:
        # Declared transfer experiment. Weights only: inheriting the optimizer
        # or EMA state of a run against a DIFFERENT donor would carry momentum
        # fitted to features this adapter will never see.
        state = torch.load(args.init_from, map_location="cpu", weights_only=False)
        adapters.load_state_dict(state["adapters"])
        initialised_from = {
            "path": str(args.init_from), "sha256": _digest(Path(args.init_from)),
            "inherited": "adapters only; no optimizer, no EMA",
            "warning": (
                "cross-donor initialisation is a separately labelled transfer "
                "experiment, not the primary path"
            ),
        }

    counts = freeze_everything_but(adapters, packer.model, px_model)
    return {
        "device": device, "packer": packer, "driver": driver,
        "adapters": adapters,
        "identity": {
            "task": TASK, "schema": SCHEMA,
            "application_mode": APPLICATION_MODE,
            "fampnn_variant": variant,
            "fampnn_path": str(donor_path), "fampnn_sha256": donor_sha,
            "pxdesign_path": str(px_path), "pxdesign_sha256": px_sha,
            "c_token": int(driver.c_token),
            "c_h_V": int(node_feature_dim(packer.model)),
            "adapter_parameters": sum(p.numel() for p in adapters.parameters()),
            "zero_initialized": True,
            "initialised_from": initialised_from,
            "sources": provenance.runtime_sources(strict=not args.allow_unpinned_sources),
            **counts,
        },
    }


def verify_against(checkpoint: dict[str, Any], identity: dict[str, Any]) -> list[str]:
    """Content-based comparison. Paths may move; bytes and modes may not."""
    recorded = checkpoint.get("identity") or {}
    problems = []
    for key in ("task", "schema", "application_mode", "fampnn_variant",
                "fampnn_sha256", "pxdesign_sha256", "c_token", "c_h_V"):
        want, got = identity.get(key), recorded.get(key)
        if want != got:
            problems.append(f"{key}: checkpoint has {got!r}, environment has {want!r}")
    return problems


# --------------------------------------------------------------- the loop


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lambda-seq", type=float, default=None,
                        help="overrides the config; required when it is 'auto'")
    parser.add_argument("--init-from", default=None,
                        help="declared cross-donor transfer experiment")
    parser.add_argument("--allow-split", action="store_true")
    parser.add_argument("--cache-size", type=int, default=48,
                        help="featurized structures held on device; an "
                             "eviction costs a recomputation, nothing else")
    parser.add_argument("--allow-unpinned-sources", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-only", default=None,
                        help="re-check a checkpoint against this environment")
    parser.add_argument("--dry-run", action="store_true",
                        help="build everything, verify, write no checkpoint")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    ctx = build(config, args)

    if args.verify_only:
        state = torch.load(args.verify_only, map_location="cpu", weights_only=False)
        problems = verify_against(state, ctx["identity"])
        print(json.dumps({"checkpoint": args.verify_only,
                          "problems": problems}, indent=2))
        raise SystemExit(len(problems))

    objective = dict(config["objective"])
    lambda_seq = args.lambda_seq if args.lambda_seq is not None else objective["lambda_seq"]
    if lambda_seq == "auto" or lambda_seq is None:
        raise SystemExit(
            "lambda_seq is 'auto'. Section 7 requires it to be calibrated on a "
            "TRAINING-ONLY batch; scripts/preflight_bs_seq_sc.py reports a "
            "suggestion per structure and deliberately does not adopt one.\n"
            "Measured on val the ratio is ~1e-3: the sequence term is two to "
            "three orders of magnitude larger than the side-chain one, and "
            "over 40 steps the SC loss ROSE while the sequence loss fell. The "
            "default of 1.0 is actively wrong, so it is not reachable by "
            "omission. Pass --lambda-seq once you have calibrated it."
        )
    lambda_seq = float(lambda_seq)
    lambda_sc = float(objective["lambda_sc"])

    manifest_path = Path(args.train_manifest or config["data"]["train_manifest"])
    frame = load_manifest(
        manifest_path,
        require_split=("" if args.allow_split else config["data"].get("split", "train")),
    ) if not args.allow_split else __import__("pandas").read_parquet(manifest_path)

    seed = args.seed if args.seed is not None else int(config["run"]["seed"])
    max_steps = args.max_steps or int(config["run"]["max_steps"])
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    identity = dict(ctx["identity"])
    identity.update({
        "lambda_seq": lambda_seq, "lambda_sc": lambda_sc,
        "seed": seed, "max_steps": max_steps,
        "arm": config["run"]["arm"],
        "data": {
            "manifest": str(manifest_path),
            "manifest_sha256": _digest(manifest_path),
            "n_examples": int(len(frame)),
            "split": sorted(set(frame["split"])) if "split" in frame else None,
            "allow_split_override": bool(args.allow_split),
        },
        "sigma_b": config["noise"]["sigma_b"],
        "gate": config["noise"].get("gate"),
        "mask_fraction": config["masking"]["mask_fraction"],
        "binder_sidechain_dropout": config["masking"]["binder_sidechain_dropout"],
        "target_sidechain_dropout": config["masking"]["target_sidechain_dropout"],
        "optimizer": config["optimizer"],
        "ema": config["ema"],
    })

    print(json.dumps({"identity": identity}, indent=2, default=str))
    (out / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True, default=str) + "\n"
    )
    if args.dry_run:
        print(f"\ndry run: wrote {out / 'identity.json'}, trained nothing")
        return

    from collections import OrderedDict

    from pxf.train.bs_seq_sc import (
        example_from_structure, joint_loss, prepare_structure,
    )
    from pxf.train.ema import EMA

    opt_cfg = config["optimizer"]
    trainable = [p for p in ctx["adapters"].parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
    )
    warmup = int(opt_cfg.get("warmup_steps", 0))
    clip = float(opt_cfg.get("max_grad_norm", 1.0))
    ema = EMA(ctx["adapters"], relative_length=float(config["ema"]["relative_length"]))

    torch.manual_seed(seed)
    order = list(range(len(frame)))
    rng = torch.Generator().manual_seed(seed)
    checkpoints = {int(s) for s in config["run"]["checkpoint_steps"]}
    cache: "OrderedDict[int, Any]" = OrderedDict()
    log_path = out / "train_log.jsonl"
    handle = log_path.open("a")
    started = time.time()

    for step in range(1, max_steps + 1):
        if (step - 1) % len(order) == 0:
            order = torch.randperm(len(frame), generator=rng).tolist()
        index = order[(step - 1) % len(order)]
        row = frame.iloc[index]

        # Featurization and a_token are mask-independent, so a revisited
        # structure pays for them once. Bounded LRU rather than unbounded:
        # 512 featurized structures would not fit, and an eviction only costs
        # a recomputation. The MASKS are redrawn every visit -- that is the
        # corruption the objective is defined over, and caching it would
        # train against one fixed draw per structure.
        if index in cache:
            cache.move_to_end(index)
        else:
            cache[index] = prepare_structure(
                row.cif_path, row.converted_binder_chain,
                packer=ctx["packer"], driver=ctx["driver"], device=ctx["device"],
                sigma_b=float(config["noise"]["sigma_b"]),
                a_token_seed=seed * 1_000_003 + index,
            )
            while len(cache) > args.cache_size:
                cache.popitem(last=False)
        example = example_from_structure(
            cache[index], packer=ctx["packer"], device=ctx["device"],
            mask_fraction=float(config["masking"]["mask_fraction"]),
            binder_sidechain_dropout=float(config["masking"]["binder_sidechain_dropout"]),
            target_sidechain_dropout=float(config["masking"]["target_sidechain_dropout"]),
            seed=seed * 1_000_003 + step,
        )

        for group in optimizer.param_groups:
            group["lr"] = float(opt_cfg["lr"]) * (
                min(1.0, step / warmup) if warmup else 1.0
            )
        optimizer.zero_grad(set_to_none=True)
        joint = joint_loss(
            ctx["packer"].model, example["batch"], example["features"],
            adapters=ctx["adapters"], a_token=example["a_token"],
            sigma_b=example["sigma"], roles=example["roles"],
            masks=example["masks"],
            lambda_seq=lambda_seq, lambda_sc=lambda_sc,
            # Only step 1 may see a zero residual: the adapter is the identity
            # until the first update. After that a zero residual is the
            # inert-arm failure and must raise.
            allow_zero=(step == 1),
        )
        if not torch.isfinite(joint.total):
            raise SystemExit(f"step {step}: non-finite loss; refusing to continue")
        joint.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, clip)
        optimizer.step()
        ema.update(ctx["adapters"])

        record = {
            "step": step, "example": str(row.example_id),
            "total": float(joint.total.detach()),
            "seq": float(joint.sequence.detach()),
            "sc": float(joint.sidechain.detach()),
            "grad_norm": float(grad_norm),
            "lr": optimizer.param_groups[0]["lr"],
            "delta_h_norm": float(joint.stats["delta_h_norm"]),
            "seconds": round(time.time() - started, 1),
        }
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        if step % 50 == 0 or step == 1:
            print(f"  step {step:>5} total {record['total']:.5f} "
                  f"seq {record['seq']:.5f} sc {record['sc']:.5f} "
                  f"|g| {record['grad_norm']:.4f}")

        if step in checkpoints or step == max_steps:
            state = {
                "adapters": ctx["adapters"].state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "identity": identity,
                "schema": SCHEMA,
            }
            path = out / "checkpoints" / f"step{step:08d}.pt"
            torch.save(state, path)
            print(f"  wrote {path}")

    handle.close()
    final = out / "checkpoints" / "final.pt"
    torch.save({
        "adapters": ctx["adapters"].state_dict(), "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(), "step": max_steps,
        "identity": identity, "schema": SCHEMA,
    }, final)
    print(f"\ndone: {max_steps} steps -> {final}")


if __name__ == "__main__":
    main()
