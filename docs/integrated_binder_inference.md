# Integrated binder inference

One PXDesign trajectory with a single coupling event. The sequence is designed
*inside* the trajectory and the backbone is corrected with what the designer
produced, so backbone and sequence co-determine each other -- as opposed to
`scripts/design_binder_matrix.py`, which generates every backbone first and
designs sequences afterwards.

    steps 0 .. E-1   stock PXDesign
    step E           bb0, a_token = D(x_noisy, sigma)          provisional
                     seq, sc      = FaMPNN_iter(bb0; A_BS)     100 steps
                     h_packed     = E(bb0, seq, sc)            no A_BS hook
                     delta        = E1(h_packed, sigma)        feedback
                     bb1          = D(x_noisy, sigma, delta)   SAME state
                     advance the solver once, using bb1
    steps E+1 .. N   stock PXDesign
    finally          repack the event's sequence on the final backbone

With the shipped 400 steps that is **401 denoiser evaluations**: 400 solver
calls plus one provisional call at the event. No baseline trajectory runs
first.

## Files

| file | role |
|---|---|
| `pxf/couple/integrated_event.py` | the event, shared by inference and feedback training |
| `pxf/bench/integrated.py` | event selection, the trajectory, finalisation |
| `pxf/bench/integrated_checkpoints.py` | feedback checkpoint loading and policy refusals |
| `scripts/design_binder_integrated.py` | single-arm CLI |
| `tests/test_integrated_binder.py`, `tests/test_integrated_event.py` | 24 CPU tests |

## Contracts worth knowing before reading a number

**`--event-sigma` is the CHURNED sigma.** Protenix churns before denoising
(`gamma = gamma0 if c_tau > gamma_min else 0`, `t_hat = c_tau_last*(gamma+1)`),
so with PXDesign's `gamma0=1.0, gamma_min=0.01` the denoiser sees about twice
the scheduled level. `select_event` reproduces that rule *including the
next-level threshold* and picks the step whose actual `t_hat` is nearest the
request. Default 0.429 is J03's training noise.

**This differs from the cached-backbone matrix, deliberately.**
`cache_binder_backbones.py` selected by the SCHEDULED level and landed on
0.4355 scheduled / **0.8711 actual**, so those arms query A_BS at roughly twice
its training sigma. Neither choice is a bug; both record the realised value.
It is one of two reasons the two protocols' outputs are not one paired
collection -- the other being that separate CUDA trajectories are not
bit-identical at equal seeds.

**Feedback reaches the model only through a tap.**
`OfficialDenoiser.denoise` with `tap=None` increments the injection counter and
*discards* the residual, so a run could report injections having applied
nothing. One tap is held for the whole trajectory and every call goes through
it. `tests/test_integrated_binder.py` pins this.

**Early and late feedback are alternatives, never stacked.** `BackboneTap`
enforces it: a `ConditioningFeedback` goes to the conditioning site and the
`a_token` injector returns early.

**FaMPNN's draws cannot move the backbone RNG.** The event runs ~101 encoder
calls and a packing rollout; `run_trajectory` wraps the callback in
`stream.protected()`, so the corrected trajectory differs from the uncorrected
one because of feedback and not because of noise bookkeeping. Tested.

**No `FixedTarget` anywhere.** Audit §9 measured the coordinate overwrite as
the cause of the interpenetrating backbones; the generation path must not
re-impose the native target frame. eta stays 2.5 and the default is 400 steps.

**The final packing reuses the event's residual and its actual sigma**, rather
than re-querying A_BS at some arbitrary low noise level. That is a transfer
assumption, and it is measured rather than assumed:
`event_to_final_aligned_rmsd` reports the displacement after removing rigid
motion. Removing it matters -- every solver step re-augments, so the raw
distance is mostly a random rotation. The event's *side-chain coordinates* are
not carried over; only its sequence.

## Running it

Needs the official runtime and a **pristine** PXDesign checkout (the repo's
working copy patches `embedders.py` for the vendored Protenix v2.0.0 and fails
here with `KeyError: 'd_lm'`):

    PYTHONPATH=/users/yfsun/pxdesign_pristine \
      /users/yfsun/.venvs/pxdesign_official/bin/python \
      scripts/design_binder_integrated.py --help

Arms: omit `--feedback-checkpoint` for the A_BS-only control; omit
`--bs-checkpoint` too for U03, the unadapted donor. `--feedback-checkpoint`
without `--bs-checkpoint` is refused -- the feedback module was trained to
correct states an A_BS-conditioned decode produced.

## Status

Code complete and CPU-tested. **Not** established: real-donor execution, any
training, and any binder-quality claim. A trained feedback checkpoint does not
exist, so only the U03 and A_BS-only arms are runnable today. Paired
multi-arm generation needs `scripts/run_integrated_binder_matrix.py`, which is
not yet written: one arm per process cannot be paired, because the prefix
before the event is not reproducible across processes.
