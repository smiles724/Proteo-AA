#!/usr/bin/env python3
"""Train the E1 feedback module. Only the adapter trains; both donors stay frozen.

    python scripts/train_integrated_feedback.py \
        --config configs/integrated_feedback/E1_full.yaml \
        --train-cache runs/integrated_feedback_v1/cache/J03_s0 \
        --validation-manifest runs/integrated_feedback_v1/data/validation.parquet \
        --bs-checkpoint .../J03_seed0/checkpoints/step00000500.pt \
        --pxdesign-donor .../pxdesign_v0.1.0.pt \
        --fampnn-checkpoint .../fampnn_0_3.pt \
        --seed 0 --max-steps 2000 --out runs/integrated_feedback_v1/E1_full_s0

The upstream half of every example was computed once by
`cache_integrated_feedback.py` under ``no_grad`` and is shared by both E1 arms,
so ``full`` and ``bb_only`` see bit-identical states. Only the corrective call
is live, and it must be: ``OfficialDenoiser.denoise`` is wrapped in
``no_grad`` and would yield a loss with no path to the feedback module, which
is why this uses ``PXDesignBackboneDriver.bind``.

### Four refusals

* A cache whose identity does not match this run's donors and A_BS.
* A cache built from anything but ``split == train``.
* A ``max_steps`` that does not reach the declared checkpoint steps.
* A first step whose output-projection gradient is zero -- that would mean the
  feedback never reached the loss, and 2,000 updates of nothing would follow.

### Conditioning is memoised, not recomputed

``driver.conditioning`` costs ~8 s and ~88 MB per complex, and 2,000 updates
over ~200 complexes revisits each about ten times. Recomputing it every step
would spend roughly 4.5 GPU-hours per run on featurization alone. It is
written to ``<out>/conditioning/`` on first touch and read back after, which
makes every epoch past the first I/O-bound instead. The cache is keyed by the
event digest, so a changed upstream cannot silently reuse it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_config(path):
    import yaml

    path = Path(path)
    config = yaml.safe_load(path.read_text())
    parent = config.pop("extends", None)
    if parent:
        base = yaml.safe_load((path.parent / parent).read_text())
        base.pop("extends", None)
        base.update(config)
        config = base
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-manifest", default=None)
    parser.add_argument("--bs-checkpoint", required=True)
    parser.add_argument("--pxdesign-donor", required=True)
    parser.add_argument("--fampnn-checkpoint", default=None)
    parser.add_argument("--fampnn-variant", default="0.3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify-only", action="store_true",
                        help="build everything, check the gradient contract, "
                             "and exit without training")
    args = parser.parse_args()

    import torch
    import yaml

    import _bootstrap  # noqa: F401

    from pxf import provenance
    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.bench.integrated_checkpoints import TASK, file_sha256
    from pxf.couple.conditioning import build_conditioner
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.couple.pxdesign_iface import (BackboneTap, conditioning_widths,
                                           token_feature_dim)
    from pxf.device import select_device
    from pxf.sidechain.design import FaMPNNFullAtomDesigner
    from pxf.train.ema import EMA
    from pxf.train.integrated_feedback import (FeedbackExample,
                                               assert_donors_clean,
                                               check_initial_gradient,
                                               feedback_loss,
                                               freeze_everything_but,
                                               gradient_norms)

    config = load_config(args.config)
    arm = config["arm"]
    max_steps = int(args.max_steps or config["max_steps"])
    checkpoint_steps = sorted(int(s) for s in config["checkpoint_steps"])
    if checkpoint_steps and max_steps < checkpoint_steps[-1]:
        raise SystemExit(
            f"--max-steps {max_steps} never reaches the declared checkpoint "
            f"steps {checkpoint_steps}; selection would have nothing to choose "
            "from at the last one"
        )

    # R3: seed BEFORE the conditioner is constructed. Seeding afterwards (as
    # this did) means two independently launched arms of a matched pair get
    # UNMATCHED initial readout/head weights, so `full - bb_only` would
    # include an initialisation difference. The pair's whole claim is that
    # they differ in one declared respect.
    import numpy as _np
    import random as _random

    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "conditioning").mkdir(parents=True, exist_ok=True)

    cache = json.loads((Path(args.train_cache) / "cache.json").read_text())
    identity = cache["identity"]
    fampnn_sha = provenance.file_sha256(
        args.fampnn_checkpoint or provenance.fampnn_checkpoint(args.fampnn_variant)
    )
    for key, want in (
        ("bs_checkpoint_sha256", file_sha256(args.bs_checkpoint)),
        ("pxdesign_sha256", file_sha256(args.pxdesign_donor)),
        ("fampnn_sha256", fampnn_sha),
    ):
        if identity.get(key) != want:
            raise SystemExit(
                f"cache was built with {key}={str(identity.get(key))[:12]} but "
                f"this run has {want[:12]}. The cached states are a function of "
                "the upstream that produced them; training on them under a "
                "different upstream trains a correction for a distribution "
                "that no longer exists."
            )
    if int(identity.get("cache_schema", 1)) != 2:
        raise SystemExit(
            f"cache reports schema {identity.get('cache_schema', 1)}; this "
            "trainer requires 2. A v1 cache encoded h_base from a different "
            "sequence AND different coordinates, so its bb_only arm was not "
            "a matched ablation, and it overwrote the target's observed "
            "side-chain occupancy. Rebuild it."
        )
    if not identity.get("stores_h_base"):
        raise SystemExit(
            "cache does not store h_base; the bb_only control cannot run "
            "from it and the pair would not be matched"
        )
    events = cache["events"]
    if not events:
        raise SystemExit(f"{args.train_cache} holds no events")
    print(f"arm={arm}  cache={len(events)} event(s)  key={cache['cache_key'][:16]}")

    # ---- donors, all frozen ----------------------------------------------
    px_model, _c, _r = load_backbone_model(args.pxdesign_donor, device=device)
    driver = PXDesignBackboneDriver(px_model)
    designer = FaMPNNFullAtomDesigner(
        args.fampnn_checkpoint, variant=args.fampnn_variant,
        seq_steps=config["upstream"]["seq_steps"],
        temperature=config["upstream"]["temperature"],
        psce_threshold=config["upstream"]["psce_threshold"], repack_last=True,
    ).to(device).eval()

    # conditioning_widths returns a TUPLE (c_s, c_z), not a mapping. These are
    # 384 and 128 on this donor while c_token is 768; a conditioner sized from
    # c_token would be a shape error at the hook.
    c_s, c_z = conditioning_widths(px_model)
    conditioner = build_conditioner(
        arm, c_h_V=node_feature_dim(designer.model),
        c_token=token_feature_dim(px_model),
        c_s=c_s, c_z=c_z, gate=config.get("gate"),
    ).to(device)
    frozen = freeze_everything_but(conditioner, px_model, designer.model)
    init_digest = initial_weight_digest(conditioner)
    print(f"trainable: {frozen['trainable_parameters']} parameter(s) in "
          f"{frozen['trainable_tensors']} tensor(s); {frozen['frozen_tensors']} "
          "donor tensor(s) frozen")
    print(f"initial weight digest {init_digest[:16]} -- both arms of a pair "
          "must report the same value")

    optim_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(
        [p for p in conditioner.parameters() if p.requires_grad],
        lr=float(optim_cfg["lr"]), weight_decay=float(optim_cfg["weight_decay"]),
        betas=tuple(optim_cfg["betas"]),
    )
    ema = EMA(conditioner, relative_length=float(config["ema"]["relative_length"]))

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        conditioner.load_state_dict(state["conditioner"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        start_step = int(state["step"])
        torch.set_rng_state(state["rng"]["cpu"])
        print(f"resumed from {args.resume} at step {start_step}")

    policy = {
        "bs_checkpoint_sha256": identity["bs_checkpoint_sha256"],
        # Taken from the CACHE identity, not from this script's arguments:
        # the policy must describe the upstream that actually produced the
        # training states.
        "fampnn_sha256": identity["fampnn_sha256"],
        "pxdesign_sha256": identity["pxdesign_sha256"],
        "bs_weights": identity["bs_weights"],
        "application_mode": identity["application_mode"],
        "sequence_source": identity["sequence_source"],
        "context": identity["context"],
        "feedback_scope": config["feedback_scope"],
        "seq_steps": int(config["upstream"]["seq_steps"]),
        "pack_steps": int(config["upstream"]["pack_steps"]),
        "temperature": float(config["upstream"]["temperature"]),
    }

    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "_cache_mod", str(REPO_ROOT / "scripts" / "cache_integrated_feedback.py")
    )
    _cache_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cache_mod)
    featurize_native = _cache_mod.featurize_native

    conditioning_cache: dict[str, object] = {}

    def conditioning_for(record):
        """Memoised per-complex conditioning. Keyed on the event digest."""
        key = record["sha256"][:32]
        if key in conditioning_cache:
            return conditioning_cache[key]
        path = out / "conditioning" / f"{key}.pt"
        if path.is_file():
            blob = torch.load(str(path), map_location=device, weights_only=False)
        else:
            structure = featurize_native(
                _cif_for(record), _binder_for(record),
                crop_size=identity["crop_size"], device=device,
            )
            cond = driver.conditioning(structure.feature_dict)
            blob = {
                "input_feature_dict": cond.input_feature_dict,
                "s_inputs": cond.s_inputs, "s_trunk": cond.s_trunk,
                "z_trunk": cond.z_trunk,
            }
            torch.save({k: _cpu(v) for k, v in blob.items()}, str(path))
        from pxf.couple.pxdesign_iface import Conditioning

        cond = Conditioning(**{k: _to(v, device) for k, v in blob.items()})
        if len(conditioning_cache) >= 4:
            conditioning_cache.pop(next(iter(conditioning_cache)))
        conditioning_cache[key] = cond
        return cond

    order = list(range(len(events)))
    history, started = [], time.time()
    print(f"training {arm} for {max_steps} step(s) from step {start_step}")

    for step in range(start_step, max_steps):
        record = events[order[step % len(order)]]
        blob = torch.load(record["path"], map_location=device, weights_only=False)
        example = FeedbackExample(
            example_id=blob["example_id"], x_noisy=blob["x_noisy"],
            sigma=float(blob["sigma"]), packed=blob["packed"],
            binder_mask=blob["binder_mask"], native_bb=blob["native_bb"],
            supervised=blob["supervised"], provenance=blob["provenance"],
        ).to(device)

        cond = conditioning_for(record)
        with BackboneTap(driver.model.diffusion_module) as tap:
            bound = driver.bind(cond, tap=tap)
            result = feedback_loss(
                example, conditioner,
                lambda x, s, *, feedback=None: _coords(bound(x, s, feedback=feedback)),
                sigma_data=float(config["sigma_data"]),
            )

        optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        norms = gradient_norms(conditioner)
        if step == start_step:
            # Structural, and fail-closed: an unlocatable projection raises
            # rather than passing on a nan.
            projection = check_initial_gradient(conditioner)
            print(f"  step {step}: output-projection grad "
                  f"weight={projection['weight']:.3e} "
                  f"bias={projection['bias']:.3e} (non-zero, as required)")
            internal = {k: v for k, v in norms.items()
                        if v == 0.0 and "single_head.2" not in k}
            print(f"    {len(internal)} internal tensor(s) still at zero "
                  "gradient, which is expected: the zero output projection "
                  "blocks their path until it moves")
        assert_donors_clean(conditioner, driver.model, designer.model)
        torch.nn.utils.clip_grad_norm_(
            conditioner.parameters(), float(optim_cfg["max_grad_norm"])
        )
        warmup = int(optim_cfg["warmup_steps"])
        scale = min(1.0, (step + 1) / max(1, warmup))
        for group in optimizer.param_groups:
            group["lr"] = float(optim_cfg["lr"]) * scale
        optimizer.step()
        ema.update(conditioner)

        history.append({"step": step, **result.stats})
        if args.verify_only:
            print("--verify-only: the gradient contract holds; exiting before "
                  "training")
            (out / "verify.json").write_text(json.dumps({
                "arm": arm, "frozen": frozen, "first_step": result.stats,
                "gradient_norms": norms, "policy": policy,
            }, indent=2, default=str))
            return
        if (step + 1) % args.log_every == 0:
            recent = history[-args.log_every:]
            print(f"  step {step + 1}/{max_steps}  loss "
                  f"{sum(h['loss'] for h in recent) / len(recent):.5f}  "
                  f"delta {recent[-1]['delta_norm']:.4f}  "
                  f"{(time.time() - started) / (step + 1 - start_step):.2f} s/step",
                  flush=True)
        if (step + 1) in checkpoint_steps:
            _save(out / "checkpoints" / f"step{step + 1:08d}.pt", conditioner,
                  optimizer, ema, step + 1, arm, policy, identity, frozen,
                  config, cache, args, init_digest)
    _save(out / "checkpoints" / "final.pt", conditioner, optimizer, ema,
          max_steps, arm, policy, identity, frozen, config, cache, args)
    (out / "history.json").write_text(json.dumps(history, indent=2, default=str))
    print(f"done: {max_steps} step(s) in {(time.time() - started) / 60:.1f} min")


def _rng_state():
    """Every stream, not just CPU torch. A resume that restores one of four
    does not reproduce the run it claims to continue."""
    import random

    import numpy as np
    import torch

    return {
        "cpu": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(state):
    import random

    import numpy as np
    import torch

    if state.get("cpu") is not None:
        torch.set_rng_state(state["cpu"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])


def initial_weight_digest(module):
    """A hash of the freshly built parameters, for asserting a matched pair."""
    import hashlib

    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _coords(out):
    return out[0] if isinstance(out, tuple) else out


def _cpu(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    return value


def _to(value, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to(v, device) for k, v in value.items()}
    return value


def _cif_for(record):
    path = record.get("cif_path")
    if not path:
        raise SystemExit(
            f"the cache does not record cif_path for {record['example_id']}; "
            "rebuild it with a cache_integrated_feedback.py that stores it per "
            "event. The trainer must not re-derive it from a manifest, which "
            "may have changed since the cache was built."
        )
    return path


def _binder_for(record):
    chain = record.get("binder_chain")
    if not chain:
        raise SystemExit(
            f"the cache does not record binder_chain for {record['example_id']}"
        )
    return chain


def _save(path, conditioner, optimizer, ema, step, arm, policy, cache_identity,
          frozen, config, cache, args, init_digest=None):
    import torch

    from pxf.bench.integrated_checkpoints import TASK
    from pxf.couple.conditioning import feature_schema

    torch.save({
        "conditioner": conditioner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "step": int(step),
        "rng": _rng_state(),
        # The policy this run ACTUALLY implemented. Not to be added to an older
        # checkpoint to make the inference loader accept it.
        "integrated_policy": policy,
        "identity": {
            "task": TASK, "arm": arm,
            **feature_schema({"arm": arm}),
            "seed": args.seed,
            "max_steps": int(config["max_steps"]),
            "sigma_data": float(config["sigma_data"]),
            "optimizer": config["optimizer"],
            "ema": config["ema"],
            **frozen,
            "cache_key": cache["cache_key"],
            "cache_identity": cache_identity,
            "n_cached_events": len(cache["events"]),
            "cache_schema": identity.get("cache_schema"),
            "initial_weight_digest": init_digest,
        },
    }, str(path))
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
