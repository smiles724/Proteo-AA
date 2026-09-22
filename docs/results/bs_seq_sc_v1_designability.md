# bs_seq_sc_v1 on the AlphaProteo benchmark: all six arms

480 shared PXDesign backbones x 6 arms = 2,880 designs, every one folded with
AF2 initial guess. 480 complete six-way matched sets; the unit of comparison
is the design, so every comparison below is paired and tested with exact
McNemar.

Arms: **J03** = joint sequence + side-chain adapter (lambda_seq 0.0051),
**S03** = side-chain-only adapter (lambda_seq 0.0), both at step 500 EMA per
`configs/bs_seq_sc/selection.yaml`; **U03** = the same FaMPNN 0.3 donor with
no adapter and no hook installed; **R0** = ProteinMPNN. All arms see the same
backbones and the same frozen donors -- FaMPNN is never fine-tuned in any arm
(1,276 frozen tensors, 12 trainable, all in the adapter).

## Designability (%), pooled across lengths

| target | J03 s0 | J03 s1 | S03 s0 | S03 s1 | U03 | R0 |
|---|---|---|---|---|---|---|
| BHRF1 | 66.7 | 72.9 | 62.5 | 66.7 | 66.7 | 58.3 |
| H1 | 6.2 | 2.1 | 6.2 | 6.2 | 2.1 | 6.2 |
| IL17A | 4.2 | 4.2 | 4.2 | 4.2 | 4.2 | 0.0 |
| IL7RA | 47.9 | 45.8 | 41.7 | 39.6 | 47.9 | 43.8 |
| IR | 50.0 | 50.0 | 37.5 | 41.7 | 54.2 | 45.8 |
| PDL1 | 62.5 | 58.3 | 50.0 | 62.5 | 58.3 | 54.2 |
| SC2RBD | 4.2 | 2.1 | 0.0 | 2.1 | 2.1 | 4.2 |
| TNFa | 4.2 | 4.2 | 2.1 | 2.1 | 6.2 | 0.0 |
| TrkA | 8.3 | 6.2 | 10.4 | 6.2 | 8.3 | 4.2 |
| VEGFA | 14.6 | 16.7 | 16.7 | 10.4 | 14.6 | 14.6 |
| **ALL** | **26.9** | **26.2** | **23.1** | **24.2** | **26.5** | **23.1** |

|  | successes | rate | 95% CI |
|---|---|---|---|
| J03 s0 | 129/480 | 26.9% | [23.1, 31.0] |
| J03 s1 | 126/480 | 26.2% | [22.5, 30.4] |
| U03 | 127/480 | 26.5% | [22.7, 30.6] |
| S03 s1 | 116/480 | 24.2% | [20.6, 28.2] |
| S03 s0 | 111/480 | 23.1% | [19.6, 27.1] |
| R0 | 111/480 | 23.1% | [19.6, 27.1] |

## The seed noise floor, measured first

Before reading any effect, here is what the same arm does to itself across
seeds -- the floor any claimed difference has to clear:

| pair | rate diff | discordant | p |
|---|---|---|---|
| J03 s0 - J03 s1 | +0.63 pp | 33 | 0.73 |
| S03 s0 - S03 s1 | -1.04 pp | 29 | 0.46 |

So seed-to-seed wobble within an arm is ~0.6-1.0 pp over ~30 discordant
designs. Any effect at or below 1 pp is not distinguishable from reseeding.

## The three comparisons that matter

| comparison | rate diff | discordant | p | seed |
|---|---|---|---|---|
| **J03 - S03** (stage-1 primary) | **+3.75 pp** | 60 | **0.027** | s0 vs s0 |
| **J03 - S03** | **+2.08 pp** | 56 | 0.229 | s1 vs s1 |
| J03 - U03 (adapter effect) | +0.42 pp | 32 | 0.86 | s0 |
| J03 - U03 | +0.21 pp | 33 | 1.00 | s1 |
| S03 - U03 | -3.33 pp | 68 | 0.068 | s0 |
| S03 - U03 | -2.29 pp | 53 | 0.169 | s1 |
| J03 - R0 | +3.75 pp | 80 | 0.057 | s0 |
| U03 - R0 | +3.33 pp | 70 | 0.072 | - |

All four J03-vs-S03 cells (both seeds, crossed) favour J03, +2.1 to +3.8 pp,
which is 2-4x the seed floor. Per `selection.yaml`'s
`seed_policy: report_both_and_agreement`: **the sign and rough magnitude agree
across seeds; the significance does not** -- s0 reaches p=0.027, s1 does not
(p=0.229). This is reported as directionally consistent, not as significant in
both seeds.

## What the pattern actually says

Line the three FaMPNN arms up against the unadapted donor:

    S03  23.1 / 24.2      BELOW U03      the side-chain objective HURTS
    U03  26.5             the donor, unadapted
    J03  26.9 / 26.2      EQUAL to U03   the sequence objective repairs it

The stage-1 primary reproduces on designability: adding the sequence objective
beats the side-chain-only arm. But it reproduces together with the finding
that qualifies it -- **J03 does not beat the unadapted donor.** The sequence
term is repairing damage the side-chain term does, and the round trip lands
back where FaMPNN 0.3 already was.

This is the same shape as the reconstruction result in
`bs_seq_sc_v1_stage1.md` (S03 clearly worse than U03 on masked-sequence NLL,
J03 recovering most of it without crossing over), now reproduced on an
independent metric, an independent dataset, and a four-way structural filter
rather than a likelihood. Two independent evaluations agreeing on a null is
stronger evidence than either alone.

**Practical reading: the 494,720-parameter A_BS adapter, as trained, is worth
approximately zero on binder designability.** The honest headline of the
benchmark is J03 - U03 = +0.4 pp, p = 0.86.

## Caveats carried forward

n = 48 per target, against A-CODE's 328-728. The finest non-zero rate
expressible is 2.08%, so single-design moves dominate the low-rate targets.
R0's AF2 initial guess is backbone-only (ProteinMPNN emits no side chains),
which flatters the FaMPNN arms in the R0 comparisons by an unmeasured amount;
a U03-backbone-only calibration arm would measure it and has not been run.
