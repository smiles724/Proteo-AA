"""Validate the replay harness inside the official runtime, adapters disabled.

Four comparisons, each of which must agree to the numerical floor. They are
ordered so that a failure localises: if (1) fails the transcription is wrong,
if (2) the recording is incomplete, if (3) hook installation perturbs, if (4)
the injection plumbing is not a no-op at zero.

  1. transcription -- upstream ``sample_diffusion`` vs ``run_trajectory`` from
     an identical RNG state.
  2. replay        -- one uninterrupted trajectory vs record-at-k then
     resume-from-k.
  3. hooks bypassed -- ``BackboneTap`` installed with ``feedback=None`` vs no
     tap at all.
  4. zero residual -- a single injected residual of exact zeros at one
     ``(step, substage)`` vs no injection. This is the only check that
     exercises ``_inject``/``_add``, since a bypassed tap never reaches them:
     it confirms the decoder's ``a`` argument is found under this runtime
     while remaining mathematically a no-op.

No ``fixed_target`` anywhere. Target handling is judged against the official
contract -- the target is conditioned by a distogram and its pose is emergent
-- so the check is that the *conditioning* is identical across arms, not that
target coordinates sit still.
"""

import argparse
import json
import sys

import torch

from pxf.couple.pxdesign_iface import BackboneTap
from pxf.couple.replay import RngStream, run_trajectory
from pxf.official.runtime import OfficialDenoiser, build_runner, first_batch

# The floor is measured here, not assumed. The 4.8e-6 figure recorded earlier
# was for a *single* denoiser call; over a 60-step trajectory the same
# computation run twice diverges far more, because each step's rounding feeds
# the next through a chaotic map. Check 0 runs the identical trajectory twice
# and every other check is judged against that, with FLOOR_MIN only as a lower
# bound in case check 0 comes out bitwise identical.
FLOOR_MIN = 1e-4


def summarize(a, b):
    a = a.reshape(-1, 3).to(torch.float64)
    b = b.reshape(-1, 3).to(torch.float64)
    n = min(len(a), len(b))
    d = (a[:n] - b[:n]).norm(dim=-1)
    return dict(
        atoms=int(n),
        max_dev=round(float(d.max()), 9),
        mean_dev=round(float(d.mean()), 9),
        rmsd=round(float((d**2).mean().sqrt()), 9),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--n-step", type=int, default=60)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--event-step", type=int, default=30)
    ap.add_argument("--use-msa", action="store_true")
    args = ap.parse_args()

    runner, configs = build_runner(
        args.yaml, args.out,
        load_checkpoint_dir=args.checkpoint_dir,
        n_step=args.n_step, n_sample=1, use_msa=args.use_msa,
    )
    data, _atom_array = first_batch(runner)
    den = OfficialDenoiser(runner, data)
    device = den.device
    schedule = den.schedule(args.n_step)
    batch_shape = tuple(den.s_inputs.shape[:-2])
    common = dict(
        schedule=schedule, n_atom=den.n_atom, device=device,
        dtype=torch.float32, batch_shape=batch_shape, n_sample=1,
        step_scale_eta=2.5,
    )
    print(json.dumps(dict(
        target=str(data["sample_name"]),
        n_token=int(data["N_token"]), n_atom=den.n_atom,
        batch_shape=list(batch_shape), n_step=args.n_step,
        eta=2.5, schedule_first=float(schedule[0]), schedule_last=float(schedule[-1]),
    ), indent=2), flush=True)

    results = {}

    # 0. null replicate -- the same trajectory twice, nothing changed. This
    # measures run-to-run nondeterminism over the full trajectory, which is
    # the only scale at which the other three comparisons mean anything.
    x_null_a, _, _ = run_trajectory(
        denoise=lambda x, s, feedback=None: den.denoise(x, s, feedback=feedback),
        stream=RngStream("bb", args.seed, device=device), **common,
    )
    x_null_b, _, _ = run_trajectory(
        denoise=lambda x, s, feedback=None: den.denoise(x, s, feedback=feedback),
        stream=RngStream("bb", args.seed, device=device), **common,
    )
    results["0_null_replicate"] = summarize(x_null_a, x_null_b)
    floor = max(FLOOR_MIN, results["0_null_replicate"]["max_dev"])

    # 1. transcription ----------------------------------------------------
    with RngStream("bb", args.seed, device=device).active():
        x_upstream = den.official_sample(n_sample=1, n_step=args.n_step)
    x_upstream = x_upstream.reshape(-1, den.n_atom, 3).to(torch.float32)

    x_tr, _, st_tr = run_trajectory(
        denoise=lambda x, s, feedback=None: den.denoise(x, s, feedback=feedback),
        stream=RngStream("bb", args.seed, device=device), **common,
    )
    results["1_transcription"] = summarize(x_upstream, x_tr)
    results["1_transcription"]["upstream_calls"] = args.n_step
    results["1_transcription"]["replay_calls"] = st_tr["calls"]

    # 2. record / resume --------------------------------------------------
    x_full, records, _ = run_trajectory(
        denoise=lambda x, s, feedback=None: den.denoise(x, s, feedback=feedback),
        stream=RngStream("bb", args.seed, device=device),
        record_steps=(args.event_step,), **common,
    )
    if not records:
        print(f"FAILED: no state recorded at step {args.event_step}")
        return 1
    state = records[0].to(device)
    x_resumed, _, st_res = run_trajectory(
        denoise=lambda x, s, feedback=None: den.denoise(x, s, feedback=feedback),
        stream=RngStream("bb", args.seed, device=device),
        resume=state, **common,
    )
    results["2_replay"] = summarize(x_full, x_resumed)
    results["2_replay"]["resumed_calls"] = st_res["calls"]
    results["2_replay"]["recorded_sigma"] = round(float(state.sigma.reshape(-1)[0]), 6)
    results["2_replay"]["recorded_c_tau_last"] = round(state.c_tau_last, 6)
    # t_hat must be the churned level, ~2x the scheduled one, not the schedule.
    results["2_replay"]["sigma_over_c_tau_last"] = round(
        float(state.sigma.reshape(-1)[0]) / max(state.c_tau_last, 1e-12), 4
    )

    # 3. hooks installed, feedback bypassed --------------------------------
    tap = BackboneTap(den.model.diffusion_module)
    with tap:
        x_tapped, _, _ = run_trajectory(
            denoise=lambda x, s, feedback=None: den.denoise(
                x, s, feedback=feedback, tap=tap
            ),
            stream=RngStream("bb", args.seed, device=device), **common,
        )
    results["3_hooks_bypassed"] = summarize(x_full, x_tapped)
    results["3_hooks_bypassed"]["tap_captures"] = tap.calls
    results["3_hooks_bypassed"]["tap_injections"] = tap.injections
    results["3_hooks_bypassed"]["a_token_width"] = (
        None if tap.a_token is None else int(tap.a_token.shape[-1])
    )

    # 4. one injected residual of zeros ------------------------------------
    tap2 = BackboneTap(den.model.diffusion_module)
    width = None if tap.a_token is None else int(tap.a_token.shape[-1])

    def zero_feedback(state):
        return torch.zeros(width, device=device, dtype=torch.float32)

    with tap2:
        x_zero, _, st_zero = run_trajectory(
            denoise=lambda x, s, feedback=None: den.denoise(
                x, s, feedback=feedback, tap=tap2
            ),
            stream=RngStream("bb", args.seed, device=device),
            event=(args.event_step, 0), feedback=zero_feedback, **common,
        )
    results["4_zero_residual"] = summarize(x_full, x_zero)
    results["4_zero_residual"]["injections"] = st_zero["injections"]
    results["4_zero_residual"]["tap_injections"] = tap2.injections

    print(json.dumps(results, indent=2), flush=True)

    print("\n=== VERDICT ===")
    print(f"measured floor (null replicate, same computation twice): "
          f"{results['0_null_replicate']['max_dev']:.3e} max, "
          f"{results['0_null_replicate']['rmsd']:.3e} rmsd")
    failures = []
    for name in ("1_transcription", "2_replay", "3_hooks_bypassed", "4_zero_residual"):
        dev = results[name]["max_dev"]
        ok = dev <= floor
        print(f"{'PASS' if ok else 'FAIL'}  {name:20s} max deviation {dev:.3e} "
              f"(floor {floor:.3e})")
        if not ok:
            failures.append(name)
    if results["4_zero_residual"].get("injections") != 1:
        failures.append("4_zero_residual: injection count != 1")
        print(f"FAIL  exactly-one-injection: {results['4_zero_residual']['injections']}")
    else:
        print("PASS  exactly-one-injection")
    if results["3_hooks_bypassed"]["tap_injections"] != 0:
        failures.append("3_hooks_bypassed: tap injected while bypassed")

    if failures:
        print(f"\nSTOPPING at: {', '.join(failures)}")
        return 1
    print("\nAll no-op checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
