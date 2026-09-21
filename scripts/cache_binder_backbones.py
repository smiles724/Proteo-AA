#!/usr/bin/env python3
"""Generate the shared backbone collection and cache the a_token event.

    python scripts/cache_binder_backbones.py \
        --targets-dir runs/binder_targets \
        --out runs/binder_backbones \
        --checkpoint-dir <pxdesign release dir> \
        --targets PDL1 --lengths 100 --n-samples 2

Phase 1A's seven arms all consume ONE backbone collection, which is what makes
their on/off differences paired. This produces it, through the official
PXDesign runtime, and caches everything the later stages need so that nothing
downstream has to re-run the generator to recover a number.

Per design it writes:

  * the final backbone, target and binder in their shared GENERATED pose. The
    target's pose is emergent -- PXDesign conditions on it through binned pair
    distances, which are invariant to rotation and translation -- so its
    coordinates are a prediction, and `fixed_target` is never passed. Clamping
    the native frame back on would put binder and target in different frames.
  * `a_token` at one late denoising event, captured from `layernorm_a`, which
    is the representation A_BS was trained to read.
  * the **actual** sigma at that event, including churn. Not the scheduler
    index and not the scheduler's sigma: `run_trajectory` perturbs sigma by
    gamma before the call, and the adapter is conditioned on what the denoiser
    actually saw.
  * the token map and chain roles, so the design stage never re-derives them.

### The transfer assumption, measured rather than assumed

The features are cached at an intermediate event; the arms then design against
the FINAL backbone. Those are different geometries, and the gap is a real
assumption rather than a detail. This records it: `event_to_final_rmsd` is the
displacement between the event's clean estimate D(x_sigma, sigma) and the
finished backbone, per design, carried into the results so a later reader can
see how far the features were transported. A large value does not invalidate
the arm -- both arms see the same cached features -- but it does bound how
much "the residual describes this backbone" can mean.

### Why the event is re-denoised rather than tapped in place

`run_trajectory` records sampler state, not model internals, and the tap holds
only the most recent call, which the loop immediately overwrites. So the
trajectory runs once to completion (untouched, no feedback), and the recorded
state at the event is then denoised ONE more time with the tap installed. That
second call is bit-identical in its inputs to the one inside the loop -- same
x_noisy, same sigma, same RNG state -- so the captured `a_token` is the
event's, and it costs one denoiser evaluation per design rather than a second
trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import torch  # noqa: E402
import yaml  # noqa: E402

logger = logging.getLogger("cache_backbones")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARMS = REPO_ROOT / "configs" / "binder_benchmark" / "arms.yaml"

# PXDesign's own CLI constants, not Protenix's generic ones.
DEFAULT_N_STEP = 400
DEFAULT_ETA = 2.5

# Where the official install lives, as pxf/official/require.py names it.
OFFICIAL_ROOT = "/hai/scratch/yfsun/pxdesign_official"


def official_sources(checkpoint_dir: str) -> dict[str, Any]:
    """What actually generated these backbones -- the OFFICIAL install.

    Not `pxf.provenance.runtime_sources`, which records the revisions this repo
    *vendors* (Protenix c3bfc36). Nothing here runs against those: generation
    goes through the official install (Protenix 0.5.0+pxd), which is the entire
    reason this step lives on HAI. Writing the vendored pins into a manifest
    that ships to Marlowe would assert the exact pairing
    `pxf/official/require.py` refuses, in the one file a later reader would
    trust to tell them which code produced the backbones.

    The checkpoint is hashed rather than named: the directory path means
    nothing on the cluster this manifest is read on.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    import protenix
    import pxdesign

    record: dict[str, Any] = {}
    for name, module in (("pxdesign", pxdesign), ("protenix", protenix)):
        path = module.__file__ or ""
        if "site-packages" not in path and OFFICIAL_ROOT not in path:
            raise SystemExit(
                f"{name} resolves to {path}, which is neither the official "
                f"install nor {OFFICIAL_ROOT}. scripts/_bootstrap puts this "
                "repo's submodules at the front of sys.path; against the "
                "official runtime that is the shadow that produced the "
                "invalid baseline. Refusing to record it as official."
            )
        try:
            installed = package_version(name)
        except PackageNotFoundError:
            installed = None
        record[name] = {
            "component": name, "version": installed, "path": path,
            "install": "official",
        }

    weights = Path(checkpoint_dir) / "pxdesign_v0.1.0.pt"
    record["pxdesign_weights"] = {
        "component": "pxdesign_weights",
        "file": weights.name,
        "sha256": (
            hashlib.file_digest(weights.open("rb"), "sha256").hexdigest()
            if weights.is_file() else None
        ),
    }
    return record


def _write_single_length_yaml(target_config: Path, length: int, out: Path) -> Path:
    """PXDesign takes one binder length per run; the prepared config has the grid."""
    payload = yaml.safe_load(target_config.read_text())
    payload = dict(payload)
    payload.pop("binder_lengths", None)
    payload["binder_length"] = int(length)
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out


def pick_event(schedule: torch.Tensor, sigma_b: float) -> int:
    """The step whose scheduled sigma is closest to `sigma_b`.

    Chosen on the schedule because the choice has to be made before the run;
    the sigma actually recorded is the churned one, which is what gets written
    out and what the adapter is conditioned on.
    """
    values = schedule.detach().reshape(-1).float()
    # The last entry is the terminal 0; a feature event there is meaningless.
    values = values[:-1]
    return int(torch.argmin((values - float(sigma_b)).abs()).item())


def topology_record(atom_array, features, n_tokens: int) -> dict[str, Any]:
    """Everything the design stage needs in order to read ``x0``.

    ``x0`` is a flat ``[n_atom, 3]`` tensor: without the atom names, the
    ``atom_to_token_idx`` axis and the chain roles it is coordinates of
    nothing. `pxf.official.bridge` recovers those from the dataloader's
    AtomArray plus three feature keys -- and that dataloader is the official
    featurizer, which exists on HAI and not on Marlowe. So they travel with
    the design rather than being re-derived on the far side, which is what
    this module's docstring already promises ("the token map and chain roles,
    so the design stage never re-derives them").

    Every AtomArray annotation is kept rather than a chosen four: they are a
    few hundred kilobytes against a cache that costs a GPU-hour per target,
    and the one that turns out to be missing is the expensive kind of loss.
    """
    import numpy as np

    from pxf.official.bridge import design_mask

    annotations = {
        name: np.asarray(atom_array.get_annotation(name))
        for name in atom_array.get_annotation_categories()
    }
    return {
        "annotations": annotations,
        "atom_to_token_idx": features["atom_to_token_idx"].reshape(-1).long().cpu(),
        "residue_index": features["residue_index"].reshape(-1)[:n_tokens].long().cpu(),
        "asym_id": features["asym_id"].reshape(-1)[:n_tokens].long().cpu(),
        "n_tokens": int(n_tokens),
        "design_mask": design_mask(atom_array, features, n_tokens).cpu(),
    }


def _rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1, 3).float()
    b = b.reshape(-1, 3).float()
    return float(torch.sqrt(((a - b) ** 2).sum(-1).mean()))


def build_conditioning(*, runner, sigma_b: float, n_step: int) -> dict[str, Any]:
    """Everything a (target, length) pair shares across its samples.

    Hoisted out of :func:`generate_one` for two reasons. The cheap one is
    cost: the trunk forward and the checkpoint load dominate -- measured on
    PDL1 L100, 8 s of trajectory inside a 109 s job -- so rebuilding them per
    sample spends an order of magnitude more GPU time on setup than on the
    sampling the run exists for.

    The one that matters is correctness. ``OfficialDenoiser.__init__`` deletes
    the template and MSA keys from the feature dict it is handed, in place,
    because the conditioning has already consumed them. Constructing a second
    denoiser from the same batch therefore conditions on a dict those keys
    have already been removed from. One denoiser per batch makes that
    unrepresentable rather than merely unattempted.

    Nothing here is sample-dependent: every draw in the trajectory comes from
    a :class:`RngStream` seeded per design, so the order the samples run in
    does not enter any of them. That is not the same as making a design
    reproducible -- it is not; see ``bit_reproducible`` in the manifest -- but
    it does keep the seed the only thing separating one sample from the next.
    """
    from pxf.official.runtime import OfficialDenoiser, first_batch

    data, atom_array = first_batch(runner)
    denoiser = OfficialDenoiser(runner, data)
    schedule = denoiser.schedule(n_step)
    return {
        "denoiser": denoiser,
        "schedule": schedule,
        "event_step": pick_event(schedule, sigma_b),
        "topology": topology_record(
            atom_array, denoiser.features, int(data["N_token"])
        ),
    }


def generate_one(
    *, denoiser, schedule, event_step: int, n_step: int, eta: float, seed: int,
) -> dict[str, Any]:
    """One trajectory, plus the event's features and the transfer measurement."""
    from pxf.couple.pxdesign_iface import BackboneTap
    from pxf.couple.replay import RngStream, run_trajectory

    # A NAMED stream. RngStream's first argument is the subsystem name, and
    # naming it is the plan's requirement rather than decoration: backbone
    # generation, sequence decoding, side-chain sampling and scoring each draw
    # their own, so two arms stay matched even when they consume different
    # numbers of draws. Sharing one integer seed does not give that.
    stream = RngStream("backbone", seed, device=denoiser.device)
    started = time.time()
    x0, records, stats = run_trajectory(
        denoise=denoiser.denoise,
        schedule=schedule,
        n_atom=denoiser.n_atom,
        device=denoiser.device,
        stream=stream,
        step_scale_eta=eta,
        record_steps=(event_step,),
        # No feedback and no fixed_target: this is the untouched trajectory
        # every arm shares.
    )
    if not records:
        raise SystemExit(f"no state recorded at step {event_step}")
    # run_trajectory records states with `detach_cpu`, and its own resume path
    # moves them back before denoising. This is the same call by another route,
    # so it needs the same move -- without it the re-denoise dies inside the
    # Fourier embedding on a CPU sigma against CUDA weights.
    event = records[0].to(denoiser.device)

    # Re-denoise the recorded event with the tap installed. Same x_noisy, same
    # sigma as the in-loop call, so the captured a_token is the event's.
    with BackboneTap(denoiser.model.diffusion_module) as tap:
        clean_at_event = denoiser.denoise(event.x_noisy, event.sigma, tap=tap)
        a_token = tap.a_token
        tap_calls = tap.calls
    if a_token is None:
        raise SystemExit(
            "the tap captured no a_token; layernorm_a was not reached on this "
            "denoiser call"
        )

    actual_sigma = float(event.sigma.reshape(-1)[0])
    scheduled_sigma = float(schedule.reshape(-1)[event_step])
    return {
        "x0": x0.detach().to("cpu"),
        "a_token": a_token.detach().to("cpu"),
        "record": {
            "event_step": event_step,
            "n_step": int(n_step),
            "eta": float(eta),
            "seed": int(seed),
            "scheduled_sigma": scheduled_sigma,
            # The churned value the denoiser actually saw. These differ, and
            # the adapter is conditioned on this one.
            "actual_sigma": actual_sigma,
            "sigma_churn_ratio": (
                actual_sigma / scheduled_sigma if scheduled_sigma else None
            ),
            "event_to_final_rmsd": _rmsd(clean_at_event, x0),
            "denoiser_calls": int(stats.get("calls", 0)),
            "injections": int(stats.get("injections", 0)),
            "tap_calls": int(tap_calls),
            "seconds": round(time.time() - started, 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--targets-dir", required=True,
                        help="output of scripts/prepare_binder_targets.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-dir", required=True,
                        help="PXDesign release checkpoint directory")
    parser.add_argument("--arms", default=str(DEFAULT_ARMS))
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--lengths", nargs="*", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=DEFAULT_N_STEP)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--sigma-b", type=float, default=None,
                        help="default: arms.yaml residual.sigma_b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-msa", action="store_true",
                        help="off by default; the MSA policy is recorded either way")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    arms = yaml.safe_load(Path(args.arms).read_text())
    sigma_b = args.sigma_b if args.sigma_b is not None else arms["residual"]["sigma_b"]
    lengths = args.lengths or arms["shared"].get("lengths") or [80, 90, 100, 110, 120, 130]

    targets_dir = Path(args.targets_dir).expanduser().resolve()
    prepared = json.loads((targets_dir / "prepared.json").read_text())["targets"]
    if args.targets:
        wanted = {t.lower() for t in args.targets}
        prepared = [t for t in prepared if t["name"].lower() in wanted]
    if not prepared:
        raise SystemExit("no targets selected")

    out = Path(args.out).expanduser().resolve()
    (out / "designs").mkdir(parents=True, exist_ok=True)
    (out / "yaml").mkdir(parents=True, exist_ok=True)

    from pxf.official.require import require_official_protenix

    # Before anything expensive. Without this the failure is an ImportError
    # four frames inside PXDesign's data pipeline that says nothing about why.
    require_official_protenix("cache_binder_backbones")

    from pxf.official.runtime import build_runner

    manifest: dict[str, Any] = {
        "settings": {
            "n_step": args.n_step, "eta": args.eta, "sigma_b": sigma_b,
            "dtype": args.dtype, "use_msa": bool(args.use_msa),
            "seed": args.seed, "n_samples": args.n_samples,
            "lengths": list(lengths),
            "checkpoint_dir": str(args.checkpoint_dir),
            "fixed_target": False,
            "bit_reproducible": False,
            "bit_reproducible_note": (
                "measured: two identical invocations at the same seed on the "
                "same GPU give x0 up to 0.55 A apart, because the CUDA "
                "reductions inside 400 denoiser calls are not deterministic. "
                "This collection is an artifact to be copied, not a recipe to "
                "be re-run; `sha256` per design is how a reader proves the "
                "arms consumed the same one."
            ),
            "fixed_target_note": (
                "never passed: the target's pose is emergent from a distogram "
                "condition, so clamping the native frame would put binder and "
                "target in different frames"
            ),
        },
        "sources": official_sources(args.checkpoint_dir),
        "designs": [],
    }

    made = 0
    for target in prepared:
        name = target["name"]
        config_path = Path(target["config"])
        for length in lengths:
            single = _write_single_length_yaml(
                config_path, length, out / "yaml" / f"{name}_L{length}.yaml"
            )
            # build_runner returns (runner, configs), as the other two
            # official-runtime scripts unpack it.
            runner, _configs = build_runner(
                str(single), str(out / "work" / f"{name}_L{length}"),
                load_checkpoint_dir=args.checkpoint_dir,
                n_step=args.n_step, n_sample=1, use_msa=args.use_msa,
                dtype=args.dtype, eta_type="const",
                eta_min=args.eta, eta_max=args.eta,
            )
            shared = build_conditioning(
                runner=runner, sigma_b=sigma_b, n_step=args.n_step
            )
            for sample in range(args.n_samples):
                design_id = f"{name}_L{length}_s{sample:04d}"
                seed = args.seed + 1000 * length + sample
                logger.info("%s (seed %d)", design_id, seed)
                result = generate_one(
                    denoiser=shared["denoiser"], schedule=shared["schedule"],
                    event_step=shared["event_step"], n_step=args.n_step,
                    eta=args.eta, seed=seed,
                )
                payload = {
                    "design_id": design_id, "target": name,
                    "binder_length": int(length), "sample": sample,
                    "x0": result["x0"], "a_token": result["a_token"],
                    "topology": shared["topology"],
                    # Also in backbones.json; duplicated here so a design can
                    # be loaded without joining against the manifest to find
                    # the sigma its a_token was captured at.
                    "actual_sigma": result["record"]["actual_sigma"],
                }
                design_path = out / "designs" / f"{design_id}.pt"
                torch.save(payload, design_path)
                record = {"design_id": design_id, "target": name,
                          "binder_length": int(length), "sample": sample,
                          "path": str(design_path),
                          # Measured, not assumed: two identical invocations
                          # of this script -- same seed, same node, same GPU,
                          # back to back -- produce x0 differing by up to
                          # 0.55 A. Non-deterministic CUDA reductions amplify
                          # over 400 denoiser calls. So this collection cannot
                          # be regenerated, only copied, and "the arms shared
                          # one collection" has to be checkable rather than
                          # believed. Hence the digest.
                          "sha256": hashlib.file_digest(
                              design_path.open("rb"), "sha256"
                          ).hexdigest(),
                          **result["record"]}
                manifest["designs"].append(record)
                made += 1
                logger.info(
                    "  sigma sched=%.4f actual=%.4f  event->final RMSD %.2f A  %.1fs",
                    record["scheduled_sigma"], record["actual_sigma"],
                    record["event_to_final_rmsd"], record["seconds"],
                )
                if args.limit and made >= args.limit:
                    break
            if args.limit and made >= args.limit:
                break
        if args.limit and made >= args.limit:
            break

    (out / "backbones.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    rmsds = [d["event_to_final_rmsd"] for d in manifest["designs"]]
    print(f"\n{made} design(s) -> {out}")
    if rmsds:
        print(f"event -> final backbone RMSD: min {min(rmsds):.2f} "
              f"median {sorted(rmsds)[len(rmsds) // 2]:.2f} max {max(rmsds):.2f} A")
        print("  (the transfer assumption: features are cached at the event, "
              "the arms design against the final backbone)")
    print(f"wrote {out / 'backbones.json'}")


if __name__ == "__main__":
    main()
