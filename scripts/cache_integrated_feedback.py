#!/usr/bin/env python3
"""Cache the detached upstream event for each accepted training complex.

    python scripts/cache_integrated_feedback.py \
        --manifest runs/integrated_feedback_v1/data/train_pdb.parquet \
        --bs-checkpoint .../J03_seed0/checkpoints/step00000500.pt \
        --bs-weights ema --event-sigma 0.429 \
        --pxdesign-donor .../pxdesign_v0.1.0.pt \
        --fampnn-checkpoint .../fampnn_0_3.pt \
        --seq-steps 100 --pack-steps 50 --temperature 0.1 --seed 0 \
        --out runs/integrated_feedback_v1/cache/J03_s0

Everything upstream of the feedback module is frozen and under ``no_grad``, so
it is computed once and reused by BOTH E1 arms. That is not only cheaper: it is
what makes ``full`` and ``bb_only`` a matched pair, since they then see
bit-identical upstream states rather than two independently sampled ones.

### The cache key is the experiment's identity

A cached state is only valid for the upstream that produced it, so the key
covers both donor hashes, the A_BS file hash AND its ema/raw selection, the
actual sigma, every decoder setting, the manifest digest, and the source
revisions. Change any of them and the cache must be rebuilt; a run that reused
a stale one would train a feedback module to correct a distribution that no
longer exists, silently.

**Native binder labels are excluded from the key**, as they are from the
conditioning. A key containing them would be a channel through which the
identity of the thing being predicted could influence which cache entry is
read.

### What is and is not native here

The noisy state is derived from the deposited backbone -- that is what
denoising training is -- and the deposited coordinates are the loss target.
The binder's IDENTITY and SIDE CHAINS never enter: the decode sees X on binder
rows with backbone-only occupancy, exactly as generation does. The target's
sequence and resolved side chains are context, which is the task.

### This uses the LOCAL driver, and that is a debt

Training must be differentiable, so the corrective call cannot use the
official runtime's ``no_grad`` wrapper. The states cached here therefore come
from the local (Protenix v2.0.0) driver. Before a long run, its zero-feedback
output and its response to the same nonzero feedback must be compared against
the official runtime on exported identical inputs. ``--verified-against-official``
records that this was done; it does not perform the comparison.
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


def sha256_file(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cache_key(payload: dict) -> str:
    """A stable digest of everything the cached state depends on."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bs-checkpoint", required=True)
    parser.add_argument("--bs-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--pxdesign-donor", required=True)
    parser.add_argument("--fampnn-checkpoint", default=None)
    parser.add_argument("--fampnn-variant", default="0.3")
    parser.add_argument("--event-sigma", type=float, default=0.429)
    parser.add_argument("--seq-steps", type=int, default=100)
    parser.add_argument("--pack-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--psce-threshold", type=float, default=0.3)
    parser.add_argument("--context", default="complex_sc")
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--events-per-complex", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verified-against-official", action="store_true",
                        help="record that the local driver was compared with "
                             "the official runtime on identical inputs. This "
                             "flag RECORDS the claim; it does not check it.")
    args = parser.parse_args()

    import pandas as pd
    import torch

    import _bootstrap  # noqa: F401

    from pxf import provenance
    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.device import select_device
    from pxf.sidechain.design import FaMPNNFullAtomDesigner

    device = select_device(args.device)
    frame = pd.read_parquet(args.manifest)
    if "split" in frame and set(frame["split"]) - {"train"}:
        raise SystemExit(
            f"{args.manifest} contains split(s) {sorted(set(frame['split']))}. "
            "A cache that contains validation rows must never be trained from."
        )
    if args.limit:
        frame = frame.head(args.limit)

    px_model, _c, _r = load_backbone_model(args.pxdesign_donor, device=device)
    px_model.requires_grad_(False)
    driver = PXDesignBackboneDriver(px_model)
    designer = FaMPNNFullAtomDesigner(
        args.fampnn_checkpoint, variant=args.fampnn_variant,
        seq_steps=args.seq_steps, temperature=args.temperature,
        psce_threshold=args.psce_threshold, repack_last=True,
    ).to(device).eval()
    designer.model.requires_grad_(False)

    fampnn_sha = provenance.file_sha256(
        args.fampnn_checkpoint or provenance.fampnn_checkpoint(args.fampnn_variant)
    )
    adapters = _load_adapters(args.bs_checkpoint, designer, driver, device,
                             args.bs_weights, fampnn_sha)

    identity = {
        "task": "integrated_feedback_v1",
        "pxdesign_sha256": sha256_file(args.pxdesign_donor),
        "fampnn_sha256": fampnn_sha,
        "fampnn_variant": args.fampnn_variant,
        "bs_checkpoint_sha256": sha256_file(args.bs_checkpoint),
        "bs_weights": args.bs_weights,
        "application_mode": "shared_prelogit",
        "sequence_source": "predicted",
        "context": args.context,
        "event_sigma_requested": args.event_sigma,
        "seq_steps": args.seq_steps,
        "pack_steps": args.pack_steps,
        "temperature": args.temperature,
        "psce_threshold": args.psce_threshold,
        "crop_size": args.crop_size,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "sources": provenance.official_sources()
        if hasattr(provenance, "official_sources") else None,
        "verified_against_official": bool(args.verified_against_official),
        # Deliberately absent: anything derived from the native binder
        # identity or side chains.
        "excludes_native_binder_labels": True,
        # Part of the KEY: a cache without h_base cannot serve the bb_only
        # arm, and nothing else in this dict would have distinguished the two.
        # A stale cache would otherwise be silently reused.
        "stores_h_base": True,
    }
    key = cache_key(identity)
    out = Path(args.out)
    (out / "events").mkdir(parents=True, exist_ok=True)

    print(f"caching {len(frame)} complex(es) x {args.events_per_complex} event(s)")
    print(f"cache key {key[:16]}  sigma(requested) {args.event_sigma}")

    records, failures = [], []
    for n, row in enumerate(frame.itertuples()):
        for event_index in range(args.events_per_complex):
            seed = args.seed * 1_000_003 + n * 17 + event_index
            try:
                record = _one_event(
                    row, driver=driver, designer=designer, adapters=adapters,
                    device=device, args=args, seed=seed,
                    event_index=event_index, out=out,
                )
                records.append(record)
            except Exception as exc:  # noqa: BLE001 - a failure is a finding
                import traceback

                trace = traceback.format_exc()
                failures.append({
                    "example_id": row.example_id, "event": event_index,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    # The traceback is kept: a failure that only reports its
                    # message costs a whole GPU round-trip to locate.
                    "traceback": trace.splitlines()[-12:],
                })
                print(f"  ERROR {row.example_id}: {type(exc).__name__}: "
                      f"{str(exc)[:120]}", flush=True)
                if len(failures) == 1:
                    print("  --- first traceback ---", flush=True)
                    print("  " + "\n  ".join(trace.splitlines()[-14:]),
                          flush=True)
        if (n + 1) % 10 == 0 or n + 1 == len(frame):
            print(f"  {n + 1}/{len(frame)}  cached {len(records)}  "
                  f"failed {len(failures)}", flush=True)

    (out / "cache.json").write_text(json.dumps({
        "cache_key": key,
        "identity": identity,
        "n_events": len(records),
        "n_failures": len(failures),
        "events": records,
        "failures": failures,
    }, indent=2, default=str))
    print(f"\ncached {len(records)} event(s), {len(failures)} failure(s)")
    print(f"wrote {out / 'cache.json'}")
    if failures:
        print("failures are recorded, not dropped: they stay in the denominator")


def featurize_native(cif_path, binder_chain_author, *, crop_size, device):
    """Featurize a deposited complex. NOT ``bs_seq_sc.prepare_structure``.

    That function perturbs the native backbone to sigma and taps ``a_token``
    from it, then discards the provisional estimate -- right for its
    reconstruction task and wrong here, where the feedback module must read
    packing computed on the PROVISIONAL bb0, as inference does. The handoff
    flags this explicitly. Only the featurization is shared, and it is
    re-derived here from the same three calls rather than reached through a
    wrapper with a different purpose.
    """
    from pxf.backbone.chain_ids import featurizer_chain_id
    from pxf.backbone.driver import featurize_structures, to_featurized

    label = featurizer_chain_id(cif_path, binder_chain_author)
    sample_id, dataset = featurize_structures(
        [str(cif_path)], crop_size=crop_size,
        binder_chain_ids=[label], parser_dataset="Distillation",
    )[0]
    return to_featurized(sample_id, dataset[0]).to(device)


def coords_only(bound):
    """Adapt the local driver's closure to the official denoiser's calling shape.

    Two differences, both of which bit on the first real run:

    * ``OfficialDenoiser.denoise`` returns coordinates; ``driver.bind`` returns
      ``(x_denoised, a_token)``. Passing the pair through would make ``bb0`` a
      tuple and fail later in a shape check that names neither cause.
    * ``OfficialDenoiser.denoise`` takes ``tap=`` per call; ``driver.bind``
      CLOSES OVER its tap and has no such parameter. ``prepare_event`` always
      passes one, so it is dropped here -- the tap is already installed, and
      ``prepare_event`` reads ``a_token`` off the object it handed in, not off
      the return value.
    """

    def denoise(x_noisy, sigma, *, tap=None, **kwargs):
        out = bound(x_noisy, sigma, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    return denoise


def _one_event(row, *, driver, designer, adapters, device, args, seed,
               event_index, out, **_unused):
    import numpy as np
    import torch

    from pxf.couple.pxdesign_iface import BackboneTap

    started = time.time()
    structure = featurize_native(
        row.cif_path, row.converted_binder_chain,
        crop_size=args.crop_size, device=device,
    )
    native_bb = structure.backbone_target.float().reshape(1, -1, 3).to(device)

    # Binder rows -> binder ATOMS, via the topology's own map. The design mask
    # is per token; the loss and the corruption are per atom.
    a2t = structure.topology.atom_to_token_idx.reshape(-1).long().to(device)
    binder_tokens = structure.design_mask.reshape(-1).bool().to(device)
    binder_atoms = binder_tokens[a2t].float()

    # The noisy state: deposited backbone corrupted to the event sigma on
    # BINDER atoms only. The target stays clean -- that is what "target
    # context" means, and noising it would pose a different task.
    generator = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(native_bb.shape, generator=generator).to(device)
    sigma = float(args.event_sigma)
    x_noisy = native_bb + sigma * noise * binder_atoms.reshape(1, -1, 1)

    conditioning = driver.conditioning(structure.feature_dict)
    with BackboneTap(driver.model.diffusion_module) as tap:
        bound = driver.bind(conditioning, tap=tap)
        with torch.no_grad():
            products = prepare_event_fn()(
                denoise=coords_only(bound),
                x_noisy=x_noisy, sigma=sigma, structure=structure,
                designer=designer, adapters=adapters,
                mask_mode="native",   # deposited complex: no 'xpb' marker
                context=args.context,
                seed=seed, design_id=str(row.example_id),
                target=str(row.example_id), tap=tap,
                # REQUIRED, not optional. The bb_only control reads h_base --
                # the side-chain-masked encoding -- and that is exactly what
                # makes it a control. Caching without it to save an encoder
                # pass made the control refuse to run, which is the readout
                # declining to substitute another encoding and silently stop
                # being a control. It costs one extra encode per event.
                want_h_base=True,
            )

    path = out / "events" / f"{row.example_id}_e{event_index}.pt"
    # The whole PackedStructure, detached. Storing only h_packed would not be
    # enough: the readout also reads coords37, aatype, seq_mask, psce and the
    # Visibility, and it is that object -- not a tensor -- that defines what
    # the feedback module is allowed to see.
    torch.save({
        "example_id": str(row.example_id),
        "x_noisy": x_noisy.detach().cpu(),
        "sigma": sigma,
        "packed": _detach_cpu(products.packed),
        "binder_mask": products.binder_mask.detach().cpu(),
        "native_bb": native_bb.detach().cpu(),
        "supervised": binder_atoms.reshape(1, -1).detach().cpu(),
        "binder_sequence": products.binder_sequence,
        "provenance": {**products.provenance, "seed": seed,
                       "event_index": event_index},
    }, path)
    return {
        "example_id": str(row.example_id), "event": event_index,
        "path": str(path), "sigma": sigma,
        # Recorded per event so the trainer never re-derives them from a
        # manifest that may since have changed.
        "cif_path": str(row.cif_path),
        "binder_chain": str(row.converted_binder_chain),
        "sha256": sha256_file(path),
        "supervised_atoms": int(binder_atoms.sum()),
        "binder_length": products.provenance.get("binder_length"),
        "delta_h_norm": products.provenance.get("delta_h_norm"),
        "seconds": round(time.time() - started, 2),
        "leakage": leakage_report_stub(products, binder_atoms),
    }


def _detach_cpu(obj):
    """Deep-detach a dataclass of tensors onto the CPU, for the cache file."""
    from dataclasses import fields, replace

    import torch

    def move(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if hasattr(value, "__dataclass_fields__"):
            return replace(value, **{
                f.name: move(getattr(value, f.name)) for f in fields(value)
            })
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value

    return move(obj)


def prepare_event_fn():
    from pxf.couple.integrated_event import prepare_event

    return prepare_event


def leakage_report_stub(products, binder_atoms):
    return {
        "binder_identity_withheld": True,
        "binder_sidechains_withheld": True,
        "supervised_atoms": int(binder_atoms.sum()),
    }


def _load_adapters(path, designer, driver, device, weights, fampnn_sha):
    import torch

    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.fampnn_iface import node_feature_dim

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    identity = state.get("identity") or {}
    for key, want in (("task", "bs_seq_sc_v1"),
                      ("application_mode", "shared_prelogit")):
        if identity.get(key) != want:
            raise SystemExit(f"{path}: {key}={identity.get(key)!r} != {want!r}")
    if identity.get("fampnn_sha256") not in (None, fampnn_sha):
        raise SystemExit(
            f"{path}: trained against FaMPNN {identity['fampnn_sha256'][:12]}, "
            f"this run has {fampnn_sha[:12]}"
        )
    adapters = CouplingAdapters(
        driver.c_token, node_feature_dim(designer.model)
    ).to(device)
    adapters.enable_bb_to_sc = True
    adapters.load_state_dict(state["adapters"])
    if state.get("ema") and weights == "ema":
        from pxf.train.ema import EMA

        ema = EMA(adapters, relative_length=0.25)
        ema.load_state_dict(state["ema"])
        ema.copy_to(adapters)
    adapters.eval().requires_grad_(False)
    return adapters


if __name__ == "__main__":
    main()
