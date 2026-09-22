# The same cell on two clusters: the backbone decides

PDL1, L=80, generation seed 101, seven arms, scored with the four-way AF2-IG
conjunction (ipAE < 10.85 Å, ipTM > 0.5, pLDDT > 80%, bound/unbound RMSD
< 3.5 Å). Run independently on HAI and on Marlowe (job 498855 / 499047).

Same target, same length, same generation seed, same PXDesign donor
(`b075867bae942dc0`), same A_BS and feedback checkpoints. PXDesign backbone
generation is not reproducible across machines -- a known property of this
collection, measured at up to 0.55 Å between identical invocations -- so the
two runs sit on two different backbones.

## The two cells

| | HAI backbone (bb-only 4.14 Å) | Marlowe backbone (bb-only 4.63 Å) |
|---|---|---|
| U03 | ipAE 17.40, ipTM 0.322 — **fail** | ipAE 4.90, ipTM 0.797 — **PASS** |
| J03 @ bs0 | 5.68, 0.774 — PASS | 5.23, 0.778 — PASS |
| J03 @ bs1 | 16.77, 0.340 — **fail** | 5.45, 0.758 — **PASS** |
| early_s_bb_only @ bs0 | 5.62, 0.777 — PASS | 5.37, 0.772 — PASS |
| early_s_bb_only @ bs1 | 16.71, 0.345 — **fail** | 5.37, 0.764 — **PASS** |
| early_s_full @ bs0 | 5.65, 0.776 — PASS | 5.29, 0.775 — PASS |
| early_s_full @ bs1 | 16.78, 0.341 — **fail** | 5.40, 0.761 — **PASS** |
| **designs** | 3/7 | **7/7** |
| **distinct sequences** | 1/3 | **3/3** |

pLDDT and bound/unbound RMSD pass everywhere in both cells. The conjunction
is decided entirely by the two interface criteria, on both backbones.

## What reproduced, and what did not

**Reproduced.** Three distinct sequences from seven arms, on both clusters:
within one A_BS seed the three arms emit one sequence. Event-to-final
displacement 0.198-0.201 Å on both. Backbone-only interface 7/7 clash-free
with zero pairs under 2.6 Å on both.

**Did not reproduce.** The A_BS seed split. On HAI every bs1 design failed at
ipTM ≈ 0.34 and every bs0 design passed at ≈ 0.78, which read as a real
seed effect larger than any arm effect. On Marlowe's backbone the same two
sequence-generation procedures give 0.758 and 0.778 -- a gap of 0.02, not
0.44 -- and everything passes.

So the bs0/bs1 split was **backbone-specific**, not a property of the A_BS
seeds. The bs1 sequences moved from ipTM 0.34 to 0.76 on a backbone change
alone.

## The size of the backbone effect

The same three sequence procedures, on two backbones of the same target,
length and generation seed: **1/3 designable versus 3/3**. Nothing about the
sequence stage changed between those two columns.

This is the same ordering seen across the AlphaProteo Table 4 rows, where
results agree more closely by backbone generator (mean abs 11.8 pp) than by
sequence designer (22.9 pp). Here it is visible within a single cell.

## What this does not license

**Not a U03 < J03 result, and not a U03 > J03 result.** U03 is worst on one
backbone and best on the other. n = 2 distinct sequences per arm across both
cells. The existing 480-design comparison on the cached-backbone path is the
only powered estimate available and it is null: J03 26.9% vs U03 26.5%,
+0.42 pp, p = 0.86.

**Not a feedback result.** The feedback arms contributed zero distinct
sequences in either cell. Their ipTM differs from their matching J03 arm by
0.002-0.006, which is the AF2 run-to-run floor, not a measurement of the
intervention.

## Consequence for sizing

Between-backbone variance dominates both the arm effect and the seed effect
at this sample size. Any design of the 28-design smoke that varies arms
while holding the backbone fixed will measure the wrong thing. Generation
seeds have to be the outer loop and there have to be enough of them for the
backbone distribution to average out -- which the earlier n=48-per-target
work achieved with 8 seeds x 6 lengths.
