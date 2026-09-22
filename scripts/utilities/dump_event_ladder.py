#!/usr/bin/env python3
"""Dump the PXDesign sampling ladder with BOTH sigmas, as an artifact.

`select_event` chooses on the CHURNED sigma (`t_hat`), not the scheduled one
(`c_tau`). With PXDesign's `gamma0=1.0, gamma_min=0.01` the churned value is
2x the scheduled one everywhere above the tail, which has already caused one
misreading -- the cached-backbone matrix selects by scheduled sigma (0.4355,
actual 0.8711) and the integrated path by churned sigma (0.429). There is no
copy of this ladder in docs/, so this writes one.

Also reports, for a grid of requested sigmas, which step select_event picks --
that mapping is what --event-sigma actually controls.
"""
import argparse, json, sys
from pathlib import Path

REPO = Path("/hai/scratch/yfsun/proteo_aa_worktrees/bdm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True, help="single-length target YAML")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--n-step", type=int, default=400)
    ap.add_argument("--step-scale-eta", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from pxf.official.require import require_official_protenix
    require_official_protenix("dump_event_ladder")
    import torch
    from pxf.bench.integrated import GAMMA0, GAMMA_MIN, select_event
    from pxf.official.runtime import OfficialDenoiser, build_runner, first_batch

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runner, _ = build_runner(a.prepared, str(out / "pxdesign"),
                             load_checkpoint_dir=a.checkpoint_dir,
                             n_step=a.n_step, n_sample=1, use_msa=False,
                             dtype="bf16", eta_type="const",
                             eta_min=a.step_scale_eta, eta_max=a.step_scale_eta)
    data, _atom_array = first_batch(runner)
    denoiser = OfficialDenoiser(runner, data)
    levels = denoiser.schedule(a.n_step).reshape(-1).to(torch.float64)

    ladder = []
    for step in range(int(levels.numel()) - 1):
        c_tau_last, c_tau = float(levels[step]), float(levels[step + 1])
        gamma = GAMMA0 if c_tau > GAMMA_MIN else 0.0
        ladder.append({"step": step, "scheduled_sigma": c_tau_last,
                       "next_level": c_tau, "gamma": gamma,
                       "churned_sigma": c_tau_last * (gamma + 1.0)})

    # What --event-sigma resolves to, over a log grid spanning the ladder.
    churned = [row["churned_sigma"] for row in ladder]
    lo, hi = min(churned), max(churned)
    grid = [lo * (hi / lo) ** (i / 24) for i in range(25)]
    resolved = []
    for requested in grid:
        c = select_event(levels, requested)
        resolved.append({"requested": requested, "step": c.step,
                         "scheduled": c.scheduled_sigma,
                         "actual": c.actual_sigma,
                         "churn_ratio": c.churn_ratio,
                         "fraction_through": c.step / (len(ladder) or 1)})

    (out / "ladder.json").write_text(json.dumps(
        {"n_step": a.n_step, "gamma0": GAMMA0, "gamma_min": GAMMA_MIN,
         "n_levels": int(levels.numel()), "ladder": ladder,
         "requested_to_event": resolved}, indent=2))
    with open(out / "ladder.csv", "w") as fh:
        fh.write("step,scheduled_sigma,churned_sigma,gamma\n")
        for row in ladder:
            fh.write(f"{row['step']},{row['scheduled_sigma']:.6f},"
                     f"{row['churned_sigma']:.6f},{row['gamma']:.1f}\n")

    print(f"n_levels={int(levels.numel())}  gamma0={GAMMA0} gamma_min={GAMMA_MIN}")
    print(f"{'step':>5} {'scheduled':>11} {'churned':>9} {'gamma':>6}")
    for step in (0, 50, 100, 150, 200, 250, 300, 330, 350, 370, 390,
                 len(ladder) - 1):
        r = ladder[step]
        print(f"{r['step']:>5} {r['scheduled_sigma']:>11.4f} "
              f"{r['churned_sigma']:>9.4f} {r['gamma']:>6.1f}")
    print(f"\nchurned sigma spans {lo:.4f} .. {hi:.4f}")
    print(f"\n--event-sigma 0.429 resolves to step "
          f"{select_event(levels, 0.429).step} of {len(ladder)}")
    # Four sigmas spanning the ladder by STEP, which is what bounds the
    # remaining trajectory -- quartiles of the step axis, not of sigma.
    picks = [ladder[int(len(ladder) * f)]["churned_sigma"] for f in (0.25, 0.5, 0.75, 0.875)]
    (out / "sigma_sweep.txt").write_text("\n".join(f"{p:.4f}" for p in picks) + "\n")
    print("sweep sigmas (steps at 25/50/75/87.5% through the ladder): "
          + " ".join(f"{p:.4f}" for p in picks))


if __name__ == "__main__":
    raise SystemExit(main())
