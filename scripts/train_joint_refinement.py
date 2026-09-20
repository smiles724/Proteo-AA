#!/usr/bin/env python3
"""Train one arm of the side-chain-supervised backbone refinement experiment.

    python scripts/train_joint_refinement.py --arm B1 --out runs/B1 \
        --config configs/joint_refinement/B1.yaml --max-steps 2000

Every arm shares the donor, the trainable scope, the examples and their order,
and the noise draws; the arm decides only which losses reach the backbone. See
:mod:`pxf.joint.trainer` for the table.

The preflight is not optional and not advisory: ``--lambda-local`` /
``--lambda-place`` default to 0, and an arm that needs one and does not have it
is refused rather than run as a silent baseline. Point ``--preflight`` at the
JSON from ``scripts/preflight_joint.py`` to take the calibrated values, or pass
them explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

logger = logging.getLogger("pxf.joint.train")

DONOR = (
    "/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn"
    "/runs/component_donors/pxdesign_v0.1.0.pt"
)
# Which coefficient each arm cannot run without.
REQUIRED_COEFFICIENTS = {
    "B1": ("lambda_local",),
    "B2": ("lambda_local", "lambda_place"),
    "BF": ("lambda_place",),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arm", required=True, choices=("B0", "B1", "B2", "BF", "BS"))
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None, help="configs/joint_refinement/<arm>.yaml")
    p.add_argument("--structures", default=None, help="directory of training CIFs")
    p.add_argument("--donor", default=DONOR)
    p.add_argument("--fampnn-weights", default="0.0")
    p.add_argument(
        "--preflight",
        default=None,
        help="preflight.json to take the calibrated loss coefficients from",
    )
    p.add_argument("--lambda-local", type=float, default=None)
    p.add_argument("--lambda-place", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--multiplier", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="run an auxiliary arm with a zero coefficient (it is then B0 with "
        "extra compute); for debugging the loop only",
    )
    return p.parse_args(argv)


def load_config(path):
    """A config plus the ``defaults:`` file it names, shallow-merged per section."""
    if path is None:
        return {}
    import yaml

    path = Path(path)
    config = yaml.safe_load(path.read_text()) or {}
    parent = config.pop("defaults", None)
    if parent:
        base = yaml.safe_load((path.parent / parent).read_text()) or {}
        for section, values in config.items():
            merged = dict(base.get(section, {}))
            merged.update(values or {})
            base[section] = merged
        config = base
    return config


def coefficients_from_preflight(path):
    """``(lambda_local, lambda_place)`` as the preflight calibrated them."""
    report = json.loads(Path(path).read_text())
    coefficients = report["coefficients"]
    return (
        float(coefficients["L_local"]["coefficient"]),
        float(coefficients["L_place"]["coefficient"]),
    )


def structure_stream(directory, *, min_length=64, max_length=256, crop_size=1024,
                     seed=0, repeat=True):
    """Training entries in a fixed, seeded order, cycling until the loop stops.

    The order is a function of the seed alone, so two arms with the same seed
    see the same examples in the same sequence -- which is what makes their
    difference the objective rather than the data.
    """
    paths = sorted(Path(directory).glob("*.cif"))
    if not paths:
        raise SystemExit(f"no CIFs under {directory}")
    order = torch.randperm(
        len(paths), generator=torch.Generator().manual_seed(int(seed))
    ).tolist()
    ordered = [paths[i] for i in order]
    while True:
        for path in ordered:
            yield dict(
                sample_id=path.stem,
                path=str(path),
                crop_size=crop_size,
                split="train",
            )
        if not repeat:
            return


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.device import select_device
    from pxf.joint.trainer import JointSettings, JointTrainer
    from pxf.provenance import fampnn_checkpoint

    config = load_config(args.config)
    data_cfg = dict(config.get("data", {}))
    train_cfg = dict(config.get("train", {}))
    train_cfg["arm"] = args.arm

    if args.preflight:
        local, place = coefficients_from_preflight(args.preflight)
        train_cfg["lambda_local"], train_cfg["lambda_place"] = local, place
        logger.info(
            "preflight %s: lambda_local=%.6g lambda_place=%.6g",
            args.preflight, local, place,
        )
    for key, value in (
        ("lambda_local", args.lambda_local),
        ("lambda_place", args.lambda_place),
        ("max_steps", args.max_steps),
        ("grad_accum_steps", args.grad_accum_steps),
        ("multiplier", args.multiplier),
        ("lr", args.lr),
        ("seed", args.seed),
    ):
        if value is not None:
            train_cfg[key] = value

    missing = [
        name
        for name in REQUIRED_COEFFICIENTS.get(args.arm, ())
        if not float(train_cfg.get(name, 0.0)) > 0
    ]
    if missing and not args.allow_uncalibrated:
        raise SystemExit(
            f"arm {args.arm} needs {missing} and they are zero. Those coefficients "
            "come from scripts/preflight_joint.py -- the losses are normalized "
            "differently, so their values do not set their relative pull. Pass "
            "--preflight <preflight.json>, or --allow-uncalibrated to run this as "
            "B0 with extra compute on purpose."
        )

    settings = JointSettings(**train_cfg)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    backbone, _bundle, donor_record = load_backbone_model(args.donor, device=device)
    driver = PXDesignBackboneDriver(backbone)
    weights = torch.load(
        fampnn_checkpoint(args.fampnn_weights), map_location="cpu", weights_only=False
    )
    fampnn = SeqDenoiser(weights["model_cfg"])
    fampnn.load_state_dict(weights["state_dict"], strict=True)
    fampnn.to(device)

    structures = args.structures or data_cfg.get("structures")
    if not structures:
        raise SystemExit("no --structures and none in the config")
    source = structure_stream(
        structures,
        min_length=int(data_cfg.get("min_length", 64)),
        max_length=int(data_cfg.get("max_length", 256)),
        crop_size=int(data_cfg.get("crop_size", 1024)),
        seed=settings.seed,
    )

    trainer = JointTrainer(
        driver,
        fampnn,
        source,
        out_dir=out,
        settings=settings,
        donor_record=donor_record,
        device=device,
    )
    if args.resume:
        logger.info("resumed at step %d from %s", trainer.resume(args.resume), args.resume)

    identity = trainer.identity()
    identity.update(structures=str(structures), data=data_cfg)
    (out / "run_config.json").write_text(json.dumps(identity, indent=2, default=str))
    logger.info(
        "arm %s on %s: %d trainable parameters, lambda_local=%.6g lambda_place=%.6g",
        settings.arm, device, identity["trainable_parameters"],
        settings.lambda_local, settings.lambda_place,
    )

    result = trainer.train(progress=lambda message: logger.info(message))
    logger.info("done: %s", result)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
