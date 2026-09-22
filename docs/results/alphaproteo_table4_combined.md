# Table 4, with this work's arms appended

Designability (%) on the ten AlphaProteo targets: the four-way AF2-IG
conjunction (ipAE < 10.85 Å, ipTM > 0.5, pLDDT > 80%, binder bound/unbound
RMSD < 3.5 Å), successes summed across the length grid per target.

Literature rows are A-CODE Table 4 as published. **The six rows below the
rule are not drop-in comparable to them**; the reasons are listed under the
table and they are not small.

All six of this work's rows are now present. Earlier versions of this
document showed only J03 s0, U03 and R0, which omitted **S03 and the second
adapter seed** -- and `configs/bs_seq_sc/selection.yaml` names `J03 - S03`
as the *primary* declared comparison. The headline table was the one missing
it.

| Type | Method | BHRF1 | H1 | IL17A | IL7RA | IR | PDL1 | SC2RBD | TNFa | TrkA | VEGFA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Two-Stage | BoltzGen | 14.56 | 19.21 | 0.31 | 12.41 | 22.34 | 14.26 | 0.69 | 0.88 | **28.65** | 7.69 | 12.1 |
| Two-Stage | ODesign | 42.23 | 7.29 | 0.10 | 7.61 | 16.82 | 10.26 | 3.13 | 0.00 | 9.99 | 3.71 | 10.1 |
| Two-Stage | RFDiffusion-3 | 33.38 | 0.89 | 0.82 | 6.47 | 18.17 | 16.81 | 4.34 | 0.00 | 14.44 | 2.95 | 9.8 |
| Two-Stage | PXDesign | 43.90 | 12.08 | 0.82 | 29.80 | 25.04 | 45.33 | 11.20 | 3.43 | 23.55 | **16.72** | 21.2 |
| Two-Stage | A-CODE (PMPNN) | 25.00 | **65.59** | 1.79 | 4.93 | 30.09 | 39.96 | 28.05 | 6.16 | 6.87 | 1.37 | 21.0 |
| One-Stage | Protpardelle-1c | 3.73 | 0.27 | 0.00 | 0.09 | 0.19 | 7.04 | 0.99 | 0.00 | 3.52 | 0.17 | 1.6 |
| One-Stage | A-CODE (Co-Design) | 22.87 | 55.71 | 1.24 | 4.05 | 41.67 | 28.70 | **37.50** | **7.57** | 3.70 | 0.96 | 20.4 |
| This work | PXD bb + FaMPNN 0.3 + A_BS joint (**J03** s0) | 66.7 | 6.2 | **4.2** | **47.9** | 50.0 | **62.5** | 4.2 | 4.2 | 8.3 | 14.6 | **26.9** |
| This work | PXD bb + FaMPNN 0.3 + A_BS joint (**J03** s1) | **72.9** | 2.1 | **4.2** | 45.8 | 50.0 | 58.3 | 2.1 | 4.2 | 6.2 | 16.7 | 26.2 |
| This work | PXD bb + FaMPNN 0.3, unadapted (**U03**) | 66.7 | 2.1 | **4.2** | **47.9** | **54.2** | 58.3 | 2.1 | 6.2 | 8.3 | 14.6 | 26.5 |
| This work | PXD bb + FaMPNN 0.3 + A_BS sc-only (**S03** s1) | 66.7 | 6.2 | **4.2** | 39.6 | 41.7 | **62.5** | 2.1 | 2.1 | 6.2 | 10.4 | 24.2 |
| This work | PXD bb + FaMPNN 0.3 + A_BS sc-only (**S03** s0) | 62.5 | 6.2 | **4.2** | 41.7 | 37.5 | 50.0 | 0.0 | 2.1 | 10.4 | 16.7 | 23.1 |

Bold = best on that target across ALL rows (not per model type, unlike the
published table). Ties are bolded jointly.

## 95% Wilson intervals, n = 48 per target



| target | J03 | J03_s1 | U03 | S03_s1 | S03_s0 | R0 |
|---|---|---|---|---|---|---|
| BHRF1 | 66.7 [52.5, 78.3] | 72.9 [59.0, 83.4] | 66.7 [52.5, 78.3] | 66.7 [52.5, 78.3] | 62.5 [48.4, 74.8] | 58.3 [44.3, 71.2] |
| H1 | 6.2 [2.1, 16.8] | 2.1 [0.4, 10.9] | 2.1 [0.4, 10.9] | 6.2 [2.1, 16.8] | 6.2 [2.1, 16.8] | 6.2 [2.1, 16.8] |
| IL17A | 4.2 [1.2, 14.0] | 4.2 [1.2, 14.0] | 4.2 [1.2, 14.0] | 4.2 [1.2, 14.0] | 4.2 [1.2, 14.0] | 0.0 [0.0, 7.4] |
| IL7RA | 47.9 [34.5, 61.7] | 45.8 [32.6, 59.7] | 47.9 [34.5, 61.7] | 39.6 [27.0, 53.7] | 41.7 [28.8, 55.7] | 43.8 [30.7, 57.7] |
| IR | 50.0 [36.4, 63.6] | 50.0 [36.4, 63.6] | 54.2 [40.3, 67.4] | 41.7 [28.8, 55.7] | 37.5 [25.2, 51.6] | 45.8 [32.6, 59.7] |
| PDL1 | 62.5 [48.4, 74.8] | 58.3 [44.3, 71.2] | 58.3 [44.3, 71.2] | 62.5 [48.4, 74.8] | 50.0 [36.4, 63.6] | 54.2 [40.3, 67.4] |
| SC2RBD | 4.2 [1.2, 14.0] | 2.1 [0.4, 10.9] | 2.1 [0.4, 10.9] | 2.1 [0.4, 10.9] | 0.0 [0.0, 7.4] | 4.2 [1.2, 14.0] |
| TNFa | 4.2 [1.2, 14.0] | 4.2 [1.2, 14.0] | 6.2 [2.1, 16.8] | 2.1 [0.4, 10.9] | 2.1 [0.4, 10.9] | 0.0 [0.0, 7.4] |
| TrkA | 8.3 [3.3, 19.6] | 6.2 [2.1, 16.8] | 8.3 [3.3, 19.6] | 6.2 [2.1, 16.8] | 10.4 [4.5, 22.2] | 4.2 [1.2, 14.0] |
| VEGFA | 14.6 [7.2, 27.2] | 16.7 [8.7, 29.6] | 14.6 [7.2, 27.2] | 10.4 [4.5, 22.2] | 16.7 [8.7, 29.6] | 14.6 [7.2, 27.2] |
| **ALL** | **26.9 [23.1, 31.0]** | **26.2 [22.5, 30.4]** | **26.5 [22.7, 30.6]** | **24.2 [20.6, 28.2]** | **23.1 [19.6, 27.1]** | **23.1 [19.6, 27.1]** |

## What the added rows show

**The primary declared comparison, J03 - S03, favours J03 in all four
cells** -- +3.75 pp (60/96 favouring, p = 0.027, s0 vs s0) and +2.08 pp
(56/96, p = 0.229, s1 vs s1). Only the first clears p < 0.05. Paired per
design, then aggregated; see `bs_seq_sc_v1_designability.md`.

**The side-chain-only objective is below the unadapted baseline.** S03 s0
23.1% and S03 s1 24.2% against U03 26.5%: S03 - U03 is -3.33 pp (p = 0.068,
s0) and -2.29 pp (p = 0.169, s1). Whatever the joint adapter is doing, the
side-chain objective alone does not reproduce it, and points the wrong way.

**J03 - U03 remains undetected.** J03 s0 26.9% vs U03 26.5% is +0.42 pp at
p = 0.86; J03 s1 is 26.2%, below U03. Carry this as **no detected
improvement under this cached-backbone protocol**, not as equivalence -- a
non-significant p-value alone establishes neither equivalence nor adequate
power, and the two seeds straddle the baseline.

**Seed spread is comparable to the arm effect.** J03 s0 26.9% vs J03 s1
26.2%; S03 s0 23.1% vs S03 s1 24.2%. Roughly 1 pp between seeds of one arm,
against 0.4 pp between J03 and U03.

## Why these rows are not a like-for-like extension of the table

**1. They are not an independent method; they are the PXDesign row with the
sequence stage swapped.** All six consume the SAME released-PXDesign
backbones. The nearest published comparator is the PXDesign row, and the
difference from it is a sequence-designer and protocol difference, not a new
generator. A-CODE's own rows use A-CODE backbones, so `R0` is *not* a
reproduction of `A-CODE (PMPNN)`: same sequence designer, different
backbones.

**2. n = 48 per target, against A-CODE's 328-728.** The finest non-zero rate
expressible at n = 48 is **2.08%**, so the sub-1% entries in the published
table (0.09, 0.10, 0.27, 0.31) have no representable counterpart here -- a
0.0 in these rows means "none of 48", not "below 0.1%". The Wilson intervals
above are 25-30 points wide on the mid-range targets.

**3. The length grid MATCHES.** A-CODE Table 4's protocol is "for each
different target, we sample 328-728 binders with lengths ranging from 80 to
130" -- the same {80, 90, 100, 110, 120, 130} used here.

**4. Read the bolds with n = 48 in front of you.** The IL17A bold is a
five-way tie at 2/48 = 4.2% against 1.79%, i.e. one design either way, with
a Wilson interval of [1.2, 14.0]. The wins that survive their intervals are
BHRF1, IL7RA and PDL1, where the margins are 15-23 points. And every one of
these rows shares the released-PXDesign backbones, so what they mostly show
is that PXDesign's generator is strong on exactly those targets.
