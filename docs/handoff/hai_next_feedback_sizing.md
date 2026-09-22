# Before the smoke: size the feedback, then re-aim it

Decision taken 2026-09-22. The 28-design smoke is **on hold**. Two cheap
measurements come first, in this order.

## Why

Your own numbers bound the effect. Event-to-final displacement is
**0.198-0.201 Å** -- that is the entire remaining trajectory after the event.
The feedback moves the backbone 0.005-0.006 Å, ~3% of all the motion left.

So this is not a discretisation problem and not an inert intervention. It is
an **injection point** problem: the feedback fires once, at
`--event-sigma 0.429`, late in the ladder where there is almost no trajectory
left to amplify it. The ceiling is structural, and it is set by a flag. Even
a perfect residual could not move the backbone more than ~0.2 Å from there.

## Measurement 1: the logit margin (your proposal -- do it)

On the existing cell, no new generation.

At each decode position, report the margin between the top-1 logit and the
runner-up, and the change in that margin induced by the feedback residual.
What the sizing decision needs:

- the distribution of |margin| over positions, not just its mean;
- the number of positions within the feedback-induced shift of flipping;
- the ratio (typical margin) / (feedback-induced shift), which is the factor
  by which the intervention is under-powered.

If that ratio is orders of magnitude, the feedback arms are duplicates at any
scale and the smoke should not run them. Say so in those terms.

## Measurement 2: where does the trajectory still have room?

The quantity that bounds the intervention is the remaining trajectory length,
so measure it directly rather than reasoning about the ladder.

1. Dump the schedule with both sigmas, since only the churned one is
   selected on:
   ```
   schedule = denoiser.schedule(400)
   # for each step: scheduled c_tau, churned t_hat, and select_event's choice
   ```
   Record it as an artifact -- `docs/` has no copy of this ladder and the
   scheduled/churned 2x factor has already caused one misreading.
2. Run **U03 alone** (cheapest arm, no adapter, no feedback) at four event
   sigmas spanning the ladder, one cell each, and record `ev->fin` for each.
   That is ~4 x 20 s of arm time.
3. Report `ev->fin` as a function of event sigma.

Pick the event sigma where `ev->fin` is on the order of **2-5 Å** -- enough
remaining trajectory that a residual can express itself -- and note what that
costs in the other direction: an earlier event means the packed side chains
the feedback reads are further from the final structure, so the readout it
was trained against degrades. That trade is the actual experimental design
question, and it should be stated with both numbers in hand.

## Constraint: the adapters were trained at sigma 0.429

`--event-sigma 0.429` is J03's training noise. Moving the event changes the
distribution the feedback adapter sees at inference from the one it was
trained on. Report any re-aimed cell as **off-distribution for the adapter**,
and do not quietly present it as the same arm. If re-aiming looks promising,
the honest follow-up is retraining the feedback at the new sigma, not
reusing these checkpoints at it.

## What not to conclude from the scored cell

J03_bs0's ipTM 0.774 against U03's 0.322 on a bit-identical backbone is a
genuinely clean contrast, and you were right to say so. It is also **one
sequence per arm**, and there is already a 480-design null on exactly this
comparison from the cached-backbone path: J03 26.9% vs U03 26.5%, +0.42 pp,
p = 0.86. One cell does not overturn n = 480. Carry that prior into the
write-up so the number is not read as an effect.
