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

**A control still decodes the terminal event.** It skips *intermediate*
events -- nothing to feed, and every decode is RNG-protected so the
trajectory is unaffected -- but under `post_feedback_redesign` the terminal
re-decode is validated against that event's own products. Skipping the last
decode left U03 and J03 re-decoding at the terminal sigma against a
first-event reference, which the sigma guard in `redecode_after_feedback`
correctly rejected: the controls died before any feedback arm ran.


Events after the first cost a decode each, so a control arm -- one with no
conditioner -- skips them. It cannot use them: there is nothing to feed, the
decode would be discarded, and the trajectory is identical either way
because every decode is RNG-protected. The first event is still decoded on
every arm, because that is the shared-prefix product.

The first event's products stay **shared** across the arms in a J03 seed
group; later events are decoded inside each arm. By then the arms have
diverged, and a shared decode would be a decode of another arm's backbone.

Decode seeds come from the **absolute solver step** and the decode's role,
never from the event's index in the selected list. An index makes a step's
seed depend on how many *other* events were requested, so the same step
would draw differently in a one-event and a four-event schedule, turning a
schedule ablation into a seed ablation as well. The first event's
provisional decode keeps `gen_seed` exactly, which is what the single-event
path used before multi-event existed.

Cross-schedule runs are still **not** paired at a shared step, and cannot
be: by the time a four-event run reaches the last event it has taken three
corrections, so it is not decoding the state a one-event run decodes there.
Equal seeds would imply a pairing that does not exist.

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

`designs.csv` gains `schedule_id`, `n_events`, `event_steps`,
`event_actual_sigmas` and `per_event_feedback_norms` -- one entry per
**scheduled** event, in schedule order, so position *k* is event *k* in
`event_steps`. Events an arm never decoded read `-`; a zero residual reads
`0.000000`. Recording only non-zero payloads made the two lists different
lengths and left the scalar `feedback_norm` showing the previous event's
value when the last was zero.

`events_scheduled`, `event_decodes` and `event_injections` are counted
separately, because they differ: a four-event feedback arm with terminal
redesign schedules 4, decodes 5 and injects 4.

`schedule_id` is part of the sample id and of the reporter's pairing key.
Without it a one-event and a four-event run write the same filenames, and
the reporter's `(prefix, arm, bs_seed)` pairing keeps whichever row it read
last. The reporter now refuses a CSV mixing schedules, as it already
refused one mixing sequence policies.

**`immediate_event_coordinate_rmsd` is provisional-vs-corrected at the SAME
event**, so one noisy state and one augmented frame. First-event-to-final
displacement is a different quantity and is reported separately as
`first_to_final_coordinate_displacement`; taking it as the "immediate"
correction folded in the whole intervening trajectory and its random
rotations, which cannot show whether a corrective call moved the backbone. That last one is the first thing to look at: whether the correction
grows, shrinks or stays flat across events says more about whether repeated
feedback can accumulate than the final designability number does.
