#!/usr/bin/env python3
"""Development evaluation for integrated_feedback_v1, per the declared selection.

    python scripts/eval_integrated_feedback.py \
        --manifest runs/integrated_feedback_v1/data/validation.parquet \
        --runs runs/.../E1_full_s0 runs/.../E1_bb_only_s0 \
               runs/.../E1_full_s1 runs/.../E1_bb_only_s1 \
        --selection configs/integrated_feedback/selection.yaml \
        --out runs/integrated_feedback_v1/evaluation

Reconstruction on the 31 held-out dimers, with a no-feedback baseline computed
on the SAME events so every comparison is paired per complex. PDB and TED rows
are reported separately, as `selection.yaml` requires.

**A 31-complex reconstruction panel does not establish binder success.** It
establishes whether the correction reconstructs held-out backbones better,
which is a different and much narrower claim. The generation matrix is what
speaks to design quality, and even that is a pilot.

### The primary comparison is full minus bb_only

Both arms have identical architecture and parameter count; they differ only in
which feature groups the readout may see. So ``full - bb_only`` isolates the
side-chain contribution, and a gain that ``bb_only`` reproduces is a
BACKBONE-conditioning gain and must not be renamed a side-chain one. It is
reported even when null.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def wilcoxon(pairs):
    """Two-sided Wilcoxon signed-rank p, or None when scipy is unavailable."""
    try:
        from scipy import stats
    except Exception:  # noqa: BLE001
        return None
    diffs = [a - b for a, b in pairs if a is not None and b is not None]
    if len(diffs) < 5 or all(d == 0 for d in diffs):
        return None
    return float(stats.wilcoxon(diffs).pvalue)


def select_checkpoint(records, selection) -> dict:
    """Earliest checkpoint within the tie band of the best eligible primary."""
    guard = {g["metric"]: g for g in selection.get("guardrails", [])}
    band = float(selection.get("tie_band_relative", 0.01))

    baseline = {
        name: next((r for r in records if r["step"] == min(
            x["step"] for x in records)), None)
        for name in guard
    }
    eligible = []
    for record in records:
        problems = []
        for metric, rule in guard.items():
            value = record["metrics"].get(metric)
            reference = (baseline[metric] or {}).get("metrics", {}).get(metric)
            if value is None or reference is None:
                continue
            if reference == 0:
                # `zero_baseline_rule: require_zero` -- a relative regression
                # is undefined against zero, and dividing would either crash
                # or silently pass anything.
                if value > 0:
                    problems.append(f"{metric} {value} against a zero baseline")
            elif (value - reference) / reference > float(
                rule["max_relative_regression"]
            ):
                problems.append(
                    f"{metric} regressed "
                    f"{(value - reference) / reference:.1%} > "
                    f"{rule['max_relative_regression']:.0%}"
                )
        record = {**record, "ineligible_because": problems}
        if not problems:
            eligible.append(record)
    if not eligible:
        return {"selected": None, "reason": "no checkpoint passed the guardrails",
                "considered": records}
    primary = selection["primary"]
    best = min(r["metrics"][primary] for r in eligible)
    within = [r for r in eligible if r["metrics"][primary] <= best * (1 + band)]
    chosen = min(within, key=lambda r: r["step"])
    return {
        "selected": chosen["step"],
        "best_primary": best,
        "tie_band": [r["step"] for r in within],
        "rule": f"earliest step within {band:.0%} of the best eligible "
                f"{primary}; guardrails applied first",
        "considered": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--bs-checkpoint", action="append", default=[],
                        metavar="SEED=PATH",
                        help="repeatable, e.g. 0=/path/J03_seed0/...pt")
    parser.add_argument("--pxdesign-donor", required=True)
    parser.add_argument("--fampnn-checkpoint", default=None)
    parser.add_argument("--fampnn-variant", default="0.3")
    parser.add_argument("--event-sigma", type=float, default=0.429)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--context", default="complex_sc")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import importlib.util as ilu

    import pandas as pd
    import torch
    import yaml

    import _bootstrap  # noqa: F401

    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model
    from pxf.bench.integrated_checkpoints import load_feedback, expected_policy
    from pxf.couple.integrated_event import mask_feedback
    from pxf.couple.losses import backbone_denoising_loss
    from pxf.couple.pxdesign_iface import (BackboneTap, conditioning_widths,
                                           token_feature_dim)
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.device import select_device
    from pxf.sidechain.design import FaMPNNFullAtomDesigner

    spec = ilu.spec_from_file_location(
        "_cache_mod", str(REPO_ROOT / "scripts" / "cache_integrated_feedback.py")
    )
    cache_module = ilu.module_from_spec(spec)
    spec.loader.exec_module(cache_module)

    selection = yaml.safe_load(Path(args.selection).read_text())
    device = select_device(args.device)
    out = Path(args.out)
    (out / "events").mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.manifest)
    if "split" in frame and set(frame["split"]) != {"val"}:
        print(f"WARNING: panel splits are {sorted(set(frame['split']))}, not val")
    if args.limit:
        frame = frame.head(args.limit)

    bs_by_seed = {}
    for entry in args.bs_checkpoint:
        seed, _, path = entry.partition("=")
        bs_by_seed[int(seed)] = path

    px_model, _c, _r = load_backbone_model(args.pxdesign_donor, device=device)
    driver = PXDesignBackboneDriver(px_model)
    designer = FaMPNNFullAtomDesigner(
        args.fampnn_checkpoint, variant=args.fampnn_variant,
        seq_steps=100, temperature=0.1, psce_threshold=0.3, repack_last=True,
    ).to(device).eval()
    c_s, c_z = conditioning_widths(px_model)
    from pxf import provenance

    fampnn_sha = provenance.file_sha256(
        args.fampnn_checkpoint or provenance.fampnn_checkpoint(args.fampnn_variant)
    )

    # ---- one event per validation complex, per A_BS seed -------------------
    print(f"building events for {len(frame)} complex(es) x "
          f"{len(bs_by_seed)} A_BS seed(s)")
    events = {}
    for seed, bs_path in sorted(bs_by_seed.items()):
        adapters = cache_module._load_adapters(
            bs_path, designer, driver, device, "ema", fampnn_sha
        )
        for index, row in enumerate(frame.itertuples()):

            class _A:
                event_sigma = args.event_sigma
                crop_size = args.crop_size
                context = args.context

            try:
                record = cache_module._one_event(
                    row, driver=driver, designer=designer, adapters=adapters,
                    device=device, args=_A(), seed=args.seed + index,
                    event_index=0, out=out,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {row.example_id}: {type(exc).__name__}")
                continue
            structure = cache_module.featurize_native(
                record["cif_path"], record["binder_chain"],
                crop_size=args.crop_size, device=device,
            )
            events[(seed, row.example_id)] = {
                "record": record, "pool": row.pool,
                "cond": driver.conditioning(structure.feature_dict),
            }
            if (index + 1) % 10 == 0:
                print(f"  seed {seed}: {index + 1}/{len(frame)}", flush=True)

    # ---- evaluate each run's checkpoints ----------------------------------
    results = {}
    for run in args.runs:
        run_path = Path(run)
        for checkpoint in sorted((run_path / "checkpoints").glob("step*.pt")):
            step = int(checkpoint.stem.replace("step", ""))
            if step not in selection["evaluated_checkpoints"]:
                continue
            blob = torch.load(str(checkpoint), map_location="cpu",
                              weights_only=False)
            identity = blob.get("identity") or {}
            seed = int(identity.get("seed", 0))
            arm = identity.get("arm")
            conditioner, report, _ident = load_feedback(
                str(checkpoint),
                expected=expected_policy(
                    bs_checkpoint=bs_by_seed.get(seed),
                    bs_weights="ema", context=args.context,
                    seq_steps=100, pack_steps=50, temperature=0.1,
                ),
                expected_arm=arm,
                c_h_V=node_feature_dim(designer.model),
                c_s=c_s, c_z=c_z, c_token=token_feature_dim(px_model),
                device=device, weights=selection.get("weights", "ema"),
            )
            per_complex = _score(
                conditioner, events, seed, driver, device, mask_feedback,
                backbone_denoising_loss, BackboneTap,
            )
            results.setdefault(f"{arm}_s{seed}", []).append({
                "step": step, "checkpoint": str(checkpoint),
                "metrics": _aggregate(per_complex),
                "per_complex": per_complex,
                "policy": report.record(),
            })
            print(f"  {arm}_s{seed}@{step}: median resolved-binder BB RMSD "
                  f"{_aggregate(per_complex)['resolved_binder_bb_rmsd']:.4f} A")

    # no-feedback baseline on the same events
    baseline = {}
    for seed in sorted(bs_by_seed):
        baseline[seed] = _score(
            None, events, seed, driver, device, mask_feedback,
            backbone_denoising_loss, BackboneTap,
        )
        print(f"  no_feedback_s{seed}: median "
              f"{_aggregate(baseline[seed])['resolved_binder_bb_rmsd']:.4f} A")

    chosen = {name: select_checkpoint(records, selection)
              for name, records in results.items()}
    report = {
        "selection": selection,
        "runs": results,
        "no_feedback": {str(k): _aggregate(v) for k, v in baseline.items()},
        "selected": chosen,
        "comparisons": _compare(results, baseline, chosen, events),
        "n_complexes": len({k[1] for k in events}),
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "selected_checkpoints.json").write_text(json.dumps({
        name: {
            "step": pick["selected"],
            "checkpoint": next(
                (r["checkpoint"] for r in results[name]
                 if r["step"] == pick["selected"]), None),
            "rule": pick.get("rule"),
        } for name, pick in chosen.items()
    }, indent=2, default=str))
    print(f"\nwrote {out / 'evaluation.json'}")
    for name, pick in chosen.items():
        print(f"  {name}: selected step {pick['selected']} ({pick.get('rule')})")


def _score(conditioner, events, seed, driver, device, mask_feedback,
           backbone_denoising_loss, BackboneTap):
    """Per-complex resolved-binder backbone RMSD after the correction."""
    import torch

    out = {}
    for (event_seed, example_id), entry in events.items():
        if event_seed != seed:
            continue
        blob = torch.load(entry["record"]["path"], map_location=device,
                          weights_only=False)
        sigma = torch.full((1,), float(blob["sigma"]), device=device)
        with BackboneTap(driver.model.diffusion_module) as tap:
            bound = driver.bind(entry["cond"], tap=tap)
            with torch.no_grad():
                delta = None
                if conditioner is not None:
                    raw, _stats = conditioner(blob["packed"], sigma)
                    delta = mask_feedback(
                        raw, blob["binder_mask"], zero_bypass=True
                    )
                result = bound(blob["x_noisy"], sigma, feedback=delta)
                bb = result[0] if isinstance(result, tuple) else result
                coupled = backbone_denoising_loss(
                    bb, blob["native_bb"], sigma=sigma,
                    atom_mask=blob["supervised"],
                )
        out[example_id] = {
            "pool": entry["pool"],
            "resolved_binder_bb_rmsd": float(
                coupled.stats["backbone_rmsd_angstrom"]
            ),
            "loss": float(coupled.total),
        }
    return out


def _aggregate(per_complex):
    values = [v["resolved_binder_bb_rmsd"] for v in per_complex.values()]
    by_pool = {}
    for pool in {v["pool"] for v in per_complex.values()}:
        subset = [v["resolved_binder_bb_rmsd"] for v in per_complex.values()
                  if v["pool"] == pool]
        by_pool[pool] = {
            "n": len(subset),
            "median": statistics.median(subset) if subset else None,
        }
    return {
        "resolved_binder_bb_rmsd": statistics.median(values) if values else None,
        "n": len(values),
        "by_pool": by_pool,
        # Failure rates are placeholders until the chemistry guardrails are
        # computed; selection treats a missing metric as unevaluable rather
        # than as passing.
        "backbone_chemistry_failure_rate": None,
        "sidechain_chemistry_failure_rate": None,
    }


def _compare(results, baseline, chosen, events):
    """full - bb_only per seed, paired per complex, plus each against no-feedback."""
    out = []
    for seed in sorted(baseline):
        full = _selected(results, chosen, "early_s_full", seed)
        bb = _selected(results, chosen, "early_s_bb_only", seed)
        if full and bb:
            out.append(_pair(
                f"E1_full - E1_bb_only (seed {seed})", full, bb,
                "PRIMARY: the side-chain-specificity comparison. A gain that "
                "bb_only reproduces is a backbone-conditioning gain.",
            ))
        for name, record in (("E1_full", full), ("E1_bb_only", bb)):
            if record:
                out.append(_pair(
                    f"{name} - no_feedback (seed {seed})", record,
                    {"per_complex": baseline[seed]},
                    "secondary: the correction against no correction at all",
                ))
    return out


def _selected(results, chosen, arm, seed):
    name = f"{arm}_s{seed}"
    pick = chosen.get(name, {}).get("selected")
    if pick is None:
        return None
    return next((r for r in results[name] if r["step"] == pick), None)


def _pair(label, a, b, kind):
    keys = sorted(set(a["per_complex"]) & set(b["per_complex"]))
    pairs = [(a["per_complex"][k]["resolved_binder_bb_rmsd"],
              b["per_complex"][k]["resolved_binder_bb_rmsd"]) for k in keys]
    diffs = [x - y for x, y in pairs]
    return {
        "comparison": label, "kind": kind, "n": len(pairs),
        "median_difference": statistics.median(diffs) if diffs else None,
        "fraction_favouring_first": (
            sum(1 for d in diffs if d < 0) / len(diffs) if diffs else None
        ),
        "wilcoxon_p": wilcoxon(pairs),
        "note": "negative favours the first arm (lower RMSD is better)",
    }


if __name__ == "__main__":
    main()
