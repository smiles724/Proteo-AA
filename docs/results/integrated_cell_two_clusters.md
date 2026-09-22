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

## What varied, stated carefully

The same three sequence-generation procedures, on two complete runs of the
same target, length and generation seed: **1/3 designable versus 3/3**.

That is descriptive evidence of substantial instability across these two
cells. It is **not** an isolated backbone effect, and four things stop it
being one:

1. **"Backbone change alone" is not isolated.** The procedure was held
   fixed, but its input backbone changed and so did its output sequence.
   Cluster and runtime differences can affect generation and scoring too.
   Attributing the swing specifically to backbone geometry needs tighter
   controls -- at minimum, cross-scoring the identical saved structures and
   sequences on both clusters.
2. **Two backbones cannot establish a variance hierarchy.** U03 moving from
   ipTM 0.322 to 0.797 is a large descriptive swing. "Between-backbone
   variance dominates the arm effect" is a claim about two variance
   components and needs several independent backbones with paired arm
   differences inside each.
3. **Shared-backbone pairing remains essential, not discredited.** It is
   what measures the intervention while controlling upstream variability.
   The right design is **multiple independent backbone prefixes with every
   arm evaluated within each prefix** -- pairing and backbone diversity are
   complementary, not alternatives.
4. **Three sequences from seven arms is expected by construction.** The
   feedback arms inherit their matching J03 sequence (see below). That tells
   us nothing about feedback strength. They are neither seven independent
   sequence trials nor automatically redundant *structural* evaluations,
   since their geometry and packing can still differ.

What the cells do establish is narrower and still useful: **the apparent
A_BS seed separation in the HAI cell did not reproduce on Marlowe.**

## Consequence for sizing

**The generation seed is the independent experimental unit; shared-prefix
pairing belongs inside it.** One cell is one prefix, so two cells are two
prefixes -- far too few to separate arm effects from prefix-to-prefix
spread, whatever its source.

The revised next step:

| step | scope | purpose |
|---|---|---|
| scoring reproducibility | cross-score the *exact* saved structures and sequences on both clusters, matched settings and seeds | separate scoring variation from generation differences |
| expanded paired pilot | **2 targets x 8 generation seeds x 1 fixed length x 7 outputs = 112 designs** | 16 independent prefixes, with within-prefix comparisons preserved |
| analysis | report each target and adapter seed separately; compare arms within prefixes | estimate consistency and uncertainty without treating seven arms as independent samples |

Keep the current sigma and checkpoints fixed for this comparison, and
measure the immediate `bb0 -> bb1` correction and its surviving
final-backbone effect alongside the chemistry and AF2-IG results.
