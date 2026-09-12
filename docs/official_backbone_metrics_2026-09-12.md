# Frozen official backbone: bad-bond investigation

The backbone remains frozen in job **114897**. Its resolved phase is `sc_adapt`,
with zero revisions, SC-to-AA and SC-to-BB disabled, and backbone refinement
disabled. Every trainable parameter name starts with `sidechain_module.`.

Frozen weights do not make metrics constant across different proteins, noise
samples, or inference protocols. Training metrics below are from different
8-microbatch optimizer windows. They are not matched measurements of backbone
improvement or degradation:

| Training step | BB RMSD (Angstrom) | CA RMSD (Angstrom) | Coordinate MSE (Angstrom squared) | Fraction at low noise |
| --- | ---: | ---: | ---: | ---: |
| 50 | 1.074 | 1.066 | 2.680 | 75% |
| 100 | 0.7822 | 0.7734 | 1.039 | 37.5% |
| 150 | 2.736 | 2.757 | 25.78 | 50% |
| 200 | 3.663 | 3.704 | 20.18 | 0% |

The comparison below holds weights, source examples and per-example seeds fixed,
but deliberately compares two **sampling step counts**. The earlier “20 SC updates
then 20 feedback updates” pilot referred to optimizer updates, a separate quantity.
In feedback adaptation, changed feedback inputs can change predicted coordinates
through frozen backbone weights. The current SC-only job does not run that path.

## Measured free-generation geometry

Probe job **114901** used the integrated SC smoke checkpoint from job 114891 and
asserted exact tensor equality of all backbone and condition-encoder tensors
against official PXDesign v0.1.0. This represents the current run's frozen
backbone component; it is not a checkpoint of its adapting SC module.

Protocol: native PXDesign joint target/binder generation, FP32, seeds 17 + sample
index, no packing, no refinement. FAMPNN decodes after backbone sampling. Twelve
recentPDB monomers and eleven unique PINDER validation binders were measured.
A twelfth binder attempt returned a duplicate via crop retry and was excluded;
the probe exits nonzero to expose that incomplete coverage, while preserving all
46 valid source/step-count measurements. No duplicated sample enters the totals.

“坏键率” retains the historical **CA-spacing outlier** definition:

`100 * count(abs(distance(CA_i, CA_(i+1)) - 3.8) > 0.3) / valid CA pairs`.

Pairs must have both residues in the design region, the same chain, consecutive
residue indices, valid CA mappings and finite coordinates. Chain boundaries,
residue gaps and missing atoms are not bridged. This is not an all-chemical-bond
violation metric. Percentages below pool counts across samples; raw records also
provide the mean per-sample percentage.

| Source | Sampling steps | Unique samples | Bad CA pairs / measured pairs | Bad-bond rate |
| --- | ---: | ---: | ---: | ---: |
| Monomer | 20 | 12 | 957 / 1086 | **88.12%** |
| Monomer | 400 | 12 | 0 / 1086 | **0.00%** |
| Binder design region | 20 | 11 | 430 / 470 | **91.49%** |
| Binder design region | 400 | 11 | 0 / 470 | **0.00%** |

At 400 steps, additional bond distances are also consistent across these samples:

| Mean distance (Angstrom) | Monomer | Binder |
| --- | ---: | ---: |
| CA–CA | 3.8171 | 3.8096 |
| Peptide C–N | 1.3252 | 1.3277 |
| N–CA | 1.4633 | 1.4632 |
| CA–C | 1.5281 | 1.5252 |
| C–O | 1.2350 | 1.2350 |

The 20-step CA–CA means were 7.6575 and 8.6090 Angstrom respectively: the quick
sampling budget is insufficient for the official checkpoint. Integrated
`stage4.generate()` and `eval_stage4_codesign.py` now default to **400** steps.
Explicit short budgets remain available for engineering tests. The historical
MLP `cogenerate` API retains its defaults; pass its sampling budget explicitly.
Changing the free-generation default does not change the current training job's
single-pass denoising objective.

Native-coordinate controls had CA outliers in 2/1041 observed monomer pairs and
2/470 binder pairs. The generated and native denominators differ when native
atoms are unobserved. The old report's 8% official-monomer result used a different
probe/protocol and an adjacency calculation that did not check chain gaps; these
results are not an apples-to-apples improvement over that number.

This is a small geometry check: monomer design regions span 78–111 residues and
binder regions 40–48 residues. Zero observed spacing outliers does not establish
folding correctness, binding, or full AlphaProteo-10 designability. Cross-source
homology exclusion is not established. Reliable training comparisons need a
fixed validation set and controlled noise/sampling protocol.

## Code and evidence

- Metric/probe commit: `1b29c0d`.
- Full tests after sampling-default correction: job **114904**, **633 passed**.
- [Raw geometry summary](validation/official_backbone_metrics/summary.json).
- [Per-sample measurements](validation/official_backbone_metrics/rows.json).
- [Bond-distance aggregates](validation/official_backbone_metrics/geometry_aggregates.json).
- [Training log snapshot](validation/official_backbone_metrics/training_snapshot.json).
- Generated mmCIFs and source indices: `runs/official_backbone_metrics/114901/`.
- Probe log: `logs/official-bb-metrics-114901.out`.

The prior adaptation pilot did not establish a reliable benefit from our packer
or feedback. Keeping both pretrained networks frozen follows the requested
migration sequence; that decision is compatible with the healthy 400-step
backbone geometry measured here.
