# Table 4, with this work's arms appended

Designability (%) on the ten AlphaProteo targets: the four-way AF2-IG
conjunction (ipAE < 10.85 A, ipTM > 0.5, pLDDT > 80%, binder bound/unbound
RMSD < 3.5 A), successes summed across the length grid per target.

Literature rows are A-CODE Table 4 as published. **The three rows below the
rule are not drop-in comparable to them**; the reasons are listed under the
table and they are not small.

| Type | Method | BHRF1 | H1 | IL17A | IL7RA | IR | PDL1 | SC2RBD | TNFa | TrkA | VEGFA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Two-Stage | BoltzGen | 14.56 | 19.21 | 0.31 | 12.41 | 22.34 | 14.26 | 0.69 | 0.88 | **28.65** | 7.69 | 12.1 |
| Two-Stage | ODesign | 42.23 | 7.29 | 0.10 | 7.61 | 16.82 | 10.26 | 3.13 | 0.00 | 9.99 | 3.71 | 10.1 |
| Two-Stage | RFDiffusion-3 | 33.38 | 0.89 | 0.82 | 6.47 | 18.17 | 16.81 | 4.34 | 0.00 | 14.44 | 2.95 | 9.8 |
| Two-Stage | PXDesign | 43.90 | 12.08 | 0.82 | 29.80 | 25.04 | 45.33 | 11.20 | 3.43 | 23.55 | **16.72** | 21.2 |
| Two-Stage | A-CODE (PMPNN) | 25.00 | **65.59** | 1.79 | 4.93 | 30.09 | 39.96 | 28.05 | 6.16 | 6.87 | 1.37 | 21.0 |
| One-Stage | Protpardelle-1c | 3.73 | 0.27 | 0.00 | 0.09 | 0.19 | 7.04 | 0.99 | 0.00 | 3.52 | 0.17 | 1.6 |
| One-Stage | A-CODE (Co-Design) | 22.87 | 55.71 | 1.24 | 4.05 | 41.67 | 28.70 | **37.50** | **7.57** | 3.70 | 0.96 | 20.4 |
| This work | PXD bb + FaMPNN 0.3 + A_BS (**J03**) | **66.7** | 6.2 | **4.2** | **47.9** | 50.0 | **62.5** | 4.2 | 4.2 | 8.3 | 14.6 | **26.9** |
| This work | PXD bb + FaMPNN 0.3 (**U03**) | **66.7** | 2.1 | **4.2** | **47.9** | **54.2** | 58.3 | 2.1 | 6.2 | 8.3 | 14.6 | 26.5 |
| This work | PXD bb + ProteinMPNN (**R0**) | 58.3 | 6.2 | 0.0 | 43.8 | 45.8 | 54.2 | 4.2 | 0.0 | 4.2 | 14.6 | 23.1 |

Bold = best on that target across ALL rows (not per model type, unlike the
published table). Ties are bolded jointly.

**Read the bolds on this work's rows with the n=48 caveat in front of you.**
Six of the ten targets are "won" here, but on IL17A that win is 2/48 = 4.2%
against 1.79%, i.e. one design either way, and the Wilson interval is
[1.2, 14.0]. The wins that survive their intervals are BHRF1, IL7RA and PDL1,
where the margins are 15-23 points. And all three of these rows share the
released-PXDesign backbones, so what they mostly show is that PXDesign's
generator is strong on exactly those targets -- see the backbone section.

Overall, pooled over all 480 designs: J03 **26.9%**, U03 **26.5%**, R0 **23.1%**.

95% Wilson intervals on this work's rows (n = 48 per target):

| target | J03 | U03 | R0 |
|---|---|---|---|
| BHRF1 | 66.7 [52.5, 78.3] | 66.7 [52.5, 78.3] | 58.3 [44.3, 71.2] |
| H1 | 6.2 [2.1, 16.8] | 2.1 [0.4, 10.9] | 6.2 [2.1, 16.8] |
| IL17A | 4.2 [1.2, 14.0] | 4.2 [1.2, 14.0] | 0.0 [0.0, 7.4] |
| IL7RA | 47.9 [34.5, 61.7] | 47.9 [34.5, 61.7] | 43.8 [30.7, 57.7] |
| IR | 50.0 [36.4, 63.6] | 54.2 [40.3, 67.4] | 45.8 [32.6, 59.7] |
| PDL1 | 62.5 [48.4, 74.8] | 58.3 [44.3, 71.2] | 54.2 [40.3, 67.4] |
| SC2RBD | 4.2 [1.2, 14.0] | 2.1 [0.4, 10.9] | 4.2 [1.2, 14.0] |
| TNFa | 4.2 [1.2, 14.0] | 6.2 [2.1, 16.8] | 0.0 [0.0, 7.4] |
| TrkA | 8.3 [3.3, 19.6] | 8.3 [3.3, 19.6] | 4.2 [1.2, 14.0] |
| VEGFA | 14.6 [7.2, 27.2] | 14.6 [7.2, 27.2] | 14.6 [7.2, 27.2] |
| **ALL** | **26.9 [23.1, 31.0]** | **26.5 [22.7, 30.6]** | **23.1 [19.6, 27.1]** |

## Why these rows are not a like-for-like extension of the table

**1. They are not an independent method; they are the PXDesign row with the
sequence stage swapped.** All three consume the SAME released-PXDesign
backbones. So the nearest published comparator is the PXDesign row (43.90 /
12.08 / ... ), and the difference from it is a sequence-designer and protocol
difference, not a new generator. A-CODE's own rows use A-CODE backbones, so
`R0` is *not* a reproduction of `A-CODE (PMPNN)`: same sequence designer,
different backbones.

**2. n = 48 per target, against A-CODE's 328-728.** This is the dominant
caveat. The finest non-zero rate expressible at n=48 is **2.08%**, so the
sub-1% entries in the published table (0.09, 0.10, 0.27, 0.31) have no
representable counterpart here -- a 0.0 in these rows means "none of 48", not
"below 0.1%". The Wilson intervals above are 25-30 points wide on the
mid-range targets.

**3. The length grid MATCHES.** A-CODE Table 4's protocol is "for each
different target, we sample 328-728 binders with lengths ranging from 80 to
130" -- the same {80, 90, 100, 110, 120, 130} used here.

CORRECTION: an earlier version of this document listed the length grid as a
comparability gap, on the basis of the per-target `alphaproteo_length_range`
in `configs/binder_benchmark/targets.yaml` (40-50 up to 120-140). Those are
AlphaProteo Table S1's ranges, not A-CODE's sampling protocol. The grid is
not a difference, and the H1 hypothesis that rested on unsampled short
lengths is WITHDRAWN -- A-CODE did not sample below 80 either. See the
backbone section below for what the H1 gap actually is.

**4. AF2-IG settings are this repo's**: 3 recycles, `model_1_ptm` for the
complex and `model_3_ptm` for the monomer, single-sequence (no MSA). R0
additionally gets a **backbone-only** initial guess, because ProteinMPNN emits
no side chains and ColabDesign feeds `all_atom_positions` into `prev_pos`
(`colabdesign/af/design.py:166`). That handicaps R0 by an unmeasured amount;
a U03-backbone-only calibration arm is the way to measure it and has not been
run yet.

## What the H1 gap actually points at: the backbones, not the sequence stage

With the length grid identical and the filter identical, the remaining large
differences are the BACKBONE SOURCE and n. Lining the rows up by backbone
provenance makes the pattern legible -- these rows and the PXDesign row share
the released-PXDesign generator; the A-CODE rows do not:

| target | J03 (this work) | PXDesign | J03 - PXD | A-CODE (PMPNN) | J03 - A-CODE |
|---|---|---|---|---|---|
| TrkA | 8.3 | 23.55 | -15.2 | 6.87 | +1.4 |
| SC2RBD | 4.2 | 11.20 | -7.0 | 28.05 | -23.9 |
| H1 | 6.2 | 12.08 | -5.9 | **65.59** | **-59.4** |
| VEGFA | 14.6 | 16.72 | -2.1 | 1.37 | +13.2 |
| TNFa | 4.2 | 3.43 | +0.8 | 6.16 | -2.0 |
| IL17A | 4.2 | 0.82 | +3.4 | 1.79 | +2.4 |
| PDL1 | 62.5 | 45.33 | +17.2 | 39.96 | +22.5 |
| IL7RA | 47.9 | 29.80 | +18.1 | 4.93 | +43.0 |
| BHRF1 | 66.7 | 43.90 | +22.8 | 25.00 | +41.7 |
| IR | 50.0 | 25.04 | +25.0 | 30.09 | +19.9 |
| | | | **mean abs 11.8** | | **mean abs 22.9** |

On H1 this work gets 6.2% and PXDesign -- the same backbone generator -- gets
12.08%, while A-CODE gets 65.59% with its own backbones. So H1 is a target
where A-CODE's generator is far better than PXDesign's, and the 59-point gap
is a BACKBONE result that no sequence stage on PXDesign backbones was going
to close. The same reading applies to SC2RBD (11.20 vs 28.05).

It cuts the other way on IL7RA and BHRF1, where PXDesign backbones are much
better than A-CODE's (29.80 vs 4.93, 43.90 vs 25.00) and these rows inherit
that advantage.

That is the main thing to take from the combined table: rows agree more
closely when they share a BACKBONE generator (mean absolute difference
11.8 pp) than when they share a SEQUENCE designer (22.9 pp). This benchmark
is dominated by the backbone stage -- the stage this work did not change.

## The filter is binding, and was checked

Marginal pass rates for J03 over all 480: ipAE 27.9%, ipTM 33.8%, pLDDT 94.8%,
RMSD 95.0%, all four 26.9%. Interface quality is the constraint and the two
interface criteria are nearly nested; pLDDT and RMSD almost never bind alone
(0 and 5 designs respectively fail only on them). Metric scales were confirmed
(pLDDT and ipTM on 0-1, ipAE in Angstrom), so the high numbers on BHRF1 /
IL7RA / PDL1 / IR are not a leniency bug.

## What is actually established

The only comparison here that controls everything except the thing being
tested is **J03 - U03**: identical backbones, identical donor, identical
context, residual on vs off, paired per design.

    J03 - U03   +0.42 pp   17 designs only-J03, 15 only-U03   McNemar p = 0.86

That is null, and it agrees with stage 1, where J03 also failed to beat the
uncoupled donor on masked-sequence NLL. `J03 - R0` (+3.75 pp, p = 0.057) and
`U03 - R0` (+3.33 pp, p = 0.072) are nearly equal, which is itself evidence
that the adapter is inert: the gap to ProteinMPNN is the same whether the
residual is on or off.

Still running: S03 (both seeds) and J03 seed 1, which give stage 1's PRIMARY
comparison J03 - S03 on designability, and the two-seed agreement
`configs/bs_seq_sc/selection.yaml` requires.
