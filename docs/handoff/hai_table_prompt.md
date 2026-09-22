# Producing the results table

The reference format is `docs/results/alphaproteo_table4_combined.md`. Match
it. The rules below are what make that table honest; most of them exist
because an earlier version got one of them wrong.

## Before you count anything: collapse identical sequences

Within one A_BS seed the three arms currently emit the **same sequence**, so
`J03`, `early_s_bb_only` and `early_s_full` are one observation, not three.
Counting them as three inflates n by 3x and makes correlated rows look like
independent ones.

So, first:

1. Group by `(target, binder_length, generation_seed, bs_seed)`.
2. Within each group, hash the designed sequence. Report how many **distinct**
   sequences each arm contributed.
3. Every rate must be computed over distinct sequences, and every table must
   print **both** the raw design count and the effective n.

If the arms are still emitting identical sequences, say so in one line at the
top of the table. That fact matters more than any number under it.

## Main table

One row per arm. Columns: one per target, then `Mean`.

| Type | Method | BHRF1 | H1 | IL17A | IL7RA | IR | PDL1 | SC2RBD | TNFa | TrkA | VEGFA | Mean |

- Cells are **designability %**: the four-way AF2-IG conjunction (ipAE <
  10.85 Å, ipTM > 0.5, pLDDT > 80%, bound/unbound RMSD < 3.5 Å), successes
  **summed (pooled)** across the length grid, lengths 80-130.
- **Bold = best on that target across all rows.** Ties bolded jointly.
- Append the published A-CODE Table 4 rows above a rule, verbatim, and this
  work's rows below it. Do not renormalise or adjust the published numbers.

Follow it with a component table -- per-arm pass rate for each of the four
criteria separately. In the cell just scored, pLDDT and bound/unbound RMSD
passed 7/7 and the conjunction was decided entirely by ipAE and ipTM. If that
holds, the two interface criteria are the whole result and the table should
make that visible rather than hiding it in a conjunction.

## Wilson intervals are not optional

A second table of 95% Wilson score intervals per target per arm, plus a
pooled `ALL` row. Tell the reader the finest non-zero rate the sample can
express (1/n), because a `0.0` at small n means "none of n", not "below
0.1%", and the published rows contain genuine sub-1% entries that have no
representable counterpart.

Any bold whose interval overlaps the row it beats gets called out in prose
underneath. Do not let a one-design margin read as a win.

## The caveats section is part of the deliverable

Carry these forward; they are still true:

1. **This is not an independent method.** It is the PXDesign row with the
   sequence stage swapped -- same released-PXDesign backbones. The nearest
   published comparator is the PXDesign row, and the delta from it is a
   sequence-designer difference, not a new generator.
2. **State n per target against A-CODE's 328-728.**
3. **The length grid matches** -- A-CODE samples 80-130 too. This is *not* a
   comparability gap; an earlier version of the doc wrongly said it was.
4. **Report the arm split by A_BS seed.** In the scored cell the entire
   spread was between bs0 and bs1, not between arms, and that spread was
   larger than any arm effect. If that survives, it is the finding.

## What the current data can and cannot support

The 28-design smoke is 2 targets x 2 generation seeds x 7 arms at **one
length**. After collapsing identical sequences that is ~12 distinct
sequences. It **cannot** produce a Table 4 row, and it must not be formatted
as one. For the smoke, produce the per-arm component table and the paired
per-design differences, and label it a pilot.

A real Table 4 row needs 10 targets x 6 lengths x ~8 generation seeds = 480
cells, ~3360 designs, to reach the n=48 per target per arm the existing rows
use. Measure one cell's wall time and quote the GPU-hours before launching
anything at that scale.

## Do not

- Do not pool across A_BS seeds to make n look bigger. Report per seed, and
  report agreement between seeds.
- Do not compute a difference of aggregates. Pair per design, then aggregate,
  and report the fraction of pairs favouring the candidate alongside.
- Do not drop a null. Four independent comparisons on this system have
  already come back null; a fifth is a result, not a failure.
