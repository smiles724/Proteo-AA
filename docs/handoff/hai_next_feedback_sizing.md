# Before the smoke: measure the correction, not the sequence

Revised 2026-09-22 after review. **The previous version of this document was
wrong on its central point** and is superseded. What it got wrong is recorded
below, because the error is instructive.

## The correction

Identical sequences across the feedback arms are **enforced by the
protocol**, not produced by a perturbation too small to flip an argmax.

The three arms within one A_BS seed share **one event decode**
(`_one_cell`, the `shared` variable), and `_finalise` repacks *that* event
sequence onto each arm's final backbone -- its docstring says so, and it
reads `products.binder_sequence` and `products.aatype` from the shared
pre-feedback event products. **There is no sequence-design step after the
feedback.** The feedback changes the backbone; nothing re-decodes.

So a much larger backbone correction would still produce identical
sequences under this implementation. The fixed-sequence comparison is
deliberate -- it isolates the *geometric* contribution of feedback -- and
sequence equality is therefore not evidence of anything about the
intervention's size.

**What was wrong before.** This document claimed an "injection point
problem": that the feedback fires too late for a residual to express itself,
with 0.2 Å of event-to-final displacement as the ceiling. Two errors.
First, the sequence-equality observation it was built to explain has a
structural cause and needed no dynamical explanation at all. Second, 0.2 Å
is the displacement of the *observed, unintervened* trajectory. It is a
useful scale, not a bound: an intervened trajectory can differ in direction
and magnitude, and whether a correction is preserved or suppressed is a
measurement, not an inference.

## Sigma direction

Sigma **decreases** along the trajectory. The Marlowe run confirms the
geometry: `event step 350: requested 0.4290 -> actual 0.4453`, i.e. step 350
of 400 at sigma 0.429.

| change | consequence |
|---|---|
| **lower** than 0.429 | later, cleaner event; fewer remaining steps |
| **higher** than 0.429 | earlier, noisier event; more remaining steps |

An earlier event needs a **higher** sigma. Anything in the previous version
implying otherwise is wrong.

## The measurements, in order

Keep `--event-sigma 0.429` for the existing matched experiment. It is A_BS's
training noise, and both A_BS and the feedback adapter need support
appropriate to any new event distribution.

**1. Is the immediate correction meaningful?**
Compare the provisional `bb0` against the corrected `bb1` at the *identical*
noisy state and sigma. This is the intervention's own magnitude, before the
trajectory has a chance to do anything to it.

**2. Does the remaining trajectory preserve or suppress it?**
Compare feedback and no-feedback final backbones. Report binder-only and
interface geometry **alongside** whole-complex RMSD -- a whole-complex number
is dominated by the target, which does not move.

**3. Would the corrected geometry change sequence preferences?**
Evaluate FaMPNN logits on the two geometries under identical sequence masks,
context, adapter policy, and randomness. This is a **new diagnostic**:
current inference never makes this comparison, because the sequence is fixed
at the event.

Report *distributions* of logit changes and margins, and the change in
sampling probability. Do **not** report a single "margin / shift"
amplification factor -- the shift has to favour the specific competing
residue, not merely be large, and FaMPNN's decoder is iterative and
stochastic over 101 steps, so a per-position static margin does not compose
into a flip probability.

**4. Would an earlier intervention help?**
Record states at several actual sigmas -- an illustrative grid is
**0.429, 1, 2, 4** mapped onto the actual schedule -- and examine how the
correction propagates, plus the chemistry at each. Record all four points on
**one unadapted trajectory** rather than running four independent
trajectories: cheaper, and it removes between-trajectory variance from the
comparison.

Do not select a sigma because displacement reaches some target value.
Displacement alone cannot rank injection points.

## If the objective is for feedback to change the sequence

Moving the event cannot achieve that while the sequence stays fixed. It
requires a controlled **sequence re-decode after the corrective backbone
call, within the same trajectory** -- a protocol change, not a parameter
change. That is a design decision, not a diagnostic, and it should be taken
deliberately.

## Priors to carry

The 480-design cached-backbone null on J03 - U03 (+0.42 pp, p = 0.86) is
relevant prior evidence, but it ran a **different inference protocol**: that
path selects the event by *scheduled* sigma (0.4355, actual 0.8711 after
churn) while the integrated path selects by *actual* churned sigma (0.429).
Cite it as a prior, not as a directly comparable measurement.

And matched backbones alone do not isolate a sequence effect: different
sequences imply different side-chain packing, so a "bit-identical backbone"
contrast still confounds sequence with packing and with anything
sequence-dependent in scoring.
