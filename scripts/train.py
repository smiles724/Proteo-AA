#!/usr/bin/env python3
"""Train or continue training FaMPNN's full-atom modules.

FaMPNN ships inference only, so the objectives and loop here are written from the
preprint (bioRxiv 2025.02.13.637498); see pxf/train/ for the section references.

    # continue training from the released weights on a directory of PDBs
    python scripts/train.py --pdb-dir <dir> --out runs/ft \
        --init-weights 0.0 --config configs/train_cath.yaml --max-steps 2000

    # resume an interrupted run of this loop
    python scripts/train.py --pdb-dir <dir> --out runs/ft --resume runs/ft/checkpoints/step00002000.pt

The released checkpoints carry no optimizer state, so they can be warm-started
(--init-weights) but not resumed (--resume); checkpoints this loop writes can do both.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

logger = logging.getLogger("pxf.train")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb-dir", help="directory of training PDBs")
    src.add_argument("--cluster-csv", help="cluster_id,path rows; one sample per cluster per epoch")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None, help="YAML preset (configs/train_*.yaml)")
    p.add_argument("--init-weights", default="0.0",
                   help="FaMPNN variant or checkpoint path to start from; 'scratch' for none")
    p.add_argument("--resume", default=None, help="checkpoint from this loop to resume")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--noise", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--train-confidence", choices=("auto", "always", "never"), default="auto",
                   help="auto = the paper's 1-in-8 sampling")
    p.add_argument("--trainable", default=None,
                   help="comma-separated name substrings to train; default trains everything")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def load_config(path):
    if path is None:
        return {}
    import yaml
    return yaml.safe_load(Path(path).read_text()) or {}


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from natsort import natsorted
    from omegaconf import OmegaConf
    from fampnn.model.sd_model import SeqDenoiser
    from pxf import provenance
    from pxf.device import select_device
    from pxf.train.data import ClusterIndex, build_loader
    from pxf.train.trainer import OptimSettings, TrainSettings, Trainer

    cfg = load_config(args.config)
    data_cfg, train_cfg, optim_cfg = (dict(cfg.get(k, {})) for k in ("data", "train", "optim"))
    for key, value in (("batch_size", args.batch_size), ("crop_size", args.crop_size),
                       ("noise", args.noise), ("num_workers", args.num_workers)):
        if value is not None:
            data_cfg[key] = value
    for key, value in (("max_steps", args.max_steps), ("seed", args.seed),
                       ("grad_accum_steps", args.grad_accum_steps)):
        if value is not None:
            train_cfg[key] = value
    if args.lr is not None:
        optim_cfg["lr"] = args.lr
    train_cfg["train_confidence"] = {"auto": None, "always": True, "never": False}[args.train_confidence]

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    if args.cluster_csv:
        index = ClusterIndex.from_csv(args.cluster_csv)
        paths = index.sample()
        logger.info("%d clusters; sampling one member each per epoch", len(index))
    else:
        paths = [str(p) for p in natsorted(str(p) for p in Path(args.pdb_dir).glob("*.pdb"))]
        logger.info("%d structures from %s", len(paths), args.pdb_dir)
    if not paths:
        raise SystemExit("no training structures found")

    dataset, loader = build_loader(
        paths, batch_size=int(data_cfg.get("batch_size", 1)),
        crop_size=int(data_cfg.get("crop_size", 256)),
        noise=float(data_cfg.get("noise", 0.0)),
        noise_targets=bool(data_cfg.get("noise_targets", True)),
        spatial_crop_p=float(data_cfg.get("spatial_crop_p", 0.5)),
        seed=int(train_cfg.get("seed", 0)),
        num_workers=int(data_cfg.get("num_workers", 0)))

    # ---- model ----
    if args.init_weights == "scratch":
        raise SystemExit(
            "--init-weights scratch needs a model_cfg to build from; pass a released "
            "checkpoint path or variant so the architecture and sigma_data are defined")
    source = (Path(args.init_weights) if Path(args.init_weights).exists()
              else provenance.fampnn_checkpoint(args.init_weights))
    bundle = torch.load(source, map_location="cpu", weights_only=False)
    model = SeqDenoiser(bundle["model_cfg"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    logger.info("initialized from %s (%.2fM params)", source,
                sum(p.numel() for p in model.parameters()) / 1e6)

    if args.trainable:
        keywords = [k.strip() for k in args.trainable.split(",") if k.strip()]
        kept = 0
        for name, param in model.named_parameters():
            keep = any(k in name for k in keywords)
            param.requires_grad_(keep)
            kept += param.numel() if keep else 0
        if not kept:
            raise SystemExit(f"--trainable {keywords} matched no parameters")
        logger.info("training only %s (%.2fM params)", keywords, kept / 1e6)

    device = select_device(args.device)
    trainer = Trainer(model, bundle["model_cfg"], loader, out_dir=out, dataset=dataset,
                      optim=OptimSettings(**optim_cfg), train=TrainSettings(**train_cfg),
                      device=device)
    if args.resume:
        logger.info("resumed at step %d from %s", trainer.resume(args.resume), args.resume)

    (out / "run_config.json").write_text(json.dumps(dict(
        data=data_cfg, train=train_cfg, optim=optim_cfg,
        init_weights=str(source), resume=args.resume, n_structures=len(paths),
        device=str(device), trainable=args.trainable,
        upstream=provenance.runtime_sources(components=("fampnn",)),
        paper="bioRxiv 2025.02.13.637498"), indent=2, default=str))

    logger.info("training on %s for %d steps (batch %s x accum %s, crop %s)", device,
                trainer.settings.max_steps, data_cfg.get("batch_size", 1),
                trainer.settings.grad_accum_steps, data_cfg.get("crop_size", 256))
    result = trainer.train(progress=lambda m: logger.info(m))
    logger.info("done: %s", result)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
