# Several feedback events in one trajectory

`--event-sigmas S1 S2 ...` schedules one event per sigma. Each event is a
full exchange: FaMPNN decodes the current backbone, E1 reads the packed
state, and PXDesign takes **one corrective call** with that residual. The
solver then advances normally until the next event.

```
--event-sigmas 4.0 2.0 1.0 0.429      # four events, four corrective calls
--event-sigma 0.429                    # unchanged single-event default
```

`--event-sigma` remains the default and the documented single-event
spelling; `--event-sigmas` wins when both are given. A single-element list
is byte-identical to the single-event path, which is asserted by test.

## What happens at each event

1. `prepare_event` decodes **from the current noisy state**. It is a fresh
   decode, not a refinement of the previous one -- which is also what makes
   the events chain: the backbone at event *k+1* already carries every
   correction made before it, so no explicit plumbing between events is
   needed.
2. E1 reads that packed state and returns a binder-masked residual.
3. One corrective `denoise` call consumes it.

Under `--sequence-policy post_feedback_redesign` the output sequence comes
from a re-decode after the **last** event only. Re-decoding at an
intermediate event would be thrown away, since the next event decodes
afresh. Under `event_fixed` the output sequence is the **first** event's,
preserving that policy's meaning at any event count.

## Matched controls

Events after the first cost a decode each, so a control arm -- one with no
conditioner -- skips them. It cannot use them: there is nothing to feed, the
decode would be discarded, and the trajectory is identical either way
because every decode is RNG-protected. The first event is still decoded on
every arm, because that is the shared-prefix product.

The first event's products stay **shared** across the arms in a J03 seed
group; later events are decoded inside each arm. By then the arms have
diverged, and a shared decode would be a decode of another arm's backbone.

Decode seeds come from the event's **position in the schedule**, not a
running counter: `gen_seed + 1000003 * (index + 1)`. A counter would give
one event different seeds depending on which arm reached it first, and the
arms are supposed to differ only in their residual.

## Two refusals

**Two sigmas resolving to the same solver call are refused, not deduped.**
Collapsing them would label a run "four events" while it ran three, and the
injection count is the variable under study.

**A bare `(step, substage)` pair stays one key.** `(350, 0)` is two ints,
not two keys; read as a collection it would arm step 0 -- the first call of
the whole run.

## This is off-distribution, and the run says so

A_BS and the feedback adapters were trained against **one** event at sigma
0.429. Every additional injection queries them at a sigma they never saw,
and at a point in the trajectory whose noise statistics they were not fitted
to. The run prints `OFF-DISTRIBUTION` when more than one event is scheduled.

Treat a multi-event run as a **checkpoint-transfer probe**, not as the same
arm as a single-event run, and do not put the two in one table without
saying which is which. If multiple feedback looks promising, the honest
follow-up is retraining the adapters against the event schedule you intend
to use.

## What is recorded

`designs.csv` gains `n_events`, `event_steps`, `event_actual_sigmas` and
`per_event_feedback_norms` -- the residual norm at each event in trajectory
order. That last one is the first thing to look at: whether the correction
grows, shrinks or stays flat across events says more about whether repeated
feedback can accumulate than the final designability number does.
