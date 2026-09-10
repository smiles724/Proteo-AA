# AlphaProteo 10-target binder benchmark

Target specifications for the binder-design benchmark A-CODE reports in its
Table 4, so our model can be scored on the same set as the numbers we would be
compared against.

For why this benchmark, what the metric measures, current status and what is
still missing, see [`docs/binder_benchmark.md`](../../docs/binder_benchmark.md).
This file is the data reference: where each number came from and how to check it.

## Why this set

A-CODE (`arXiv:2605.03360`) benchmarks conditional binder design on ten targets
"as proposed in Zambaldi et al." — AlphaProteo (`arXiv:2409.08022`) — and states
it follows PXDesign's protocol. **PXDesign is a row in that table**, and our
backbone is PXDesign's, so its row is the reference this project is measured
against:

| | BHRF1 | H1 | IL17A | IL7RA | IR | PDL1 | SC2RBD | TNFa | TrkA | VEGFA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PXDesign** (two-stage) | 43.90 | 12.08 | 0.82 | 29.80 | 25.04 | 45.33 | 11.20 | 3.43 | 23.55 | 16.72 |
| A-CODE (PMPNN) | 25.00 | 65.59 | 1.79 | 4.93 | 30.09 | 39.96 | 28.05 | 6.16 | 6.87 | 1.37 |
| A-CODE (co-design) | 22.87 | 55.71 | 1.24 | 4.05 | 41.67 | 28.70 | 37.50 | 7.57 | 3.70 | 0.96 |
| Protpardelle-1c (one-stage) | 3.73 | 0.27 | 0.00 | 0.09 | 0.19 | 7.04 | 0.99 | 0.00 | 3.52 | 0.17 |

Designability, percent. The question this benchmark answers for us is whether
co-design moves PXDesign's row up or down.

## Layout

```
targets/*.yaml            ten target configs, PXDesign input format
download_structures.sh    fetch the ten structures from the RCSB
check_targets.py          verify every range and hotspot against the structures
structures/               downloaded (gitignored)
```

```bash
./benchmarks/alphaproteo10/download_structures.sh
python benchmarks/alphaproteo10/check_targets.py
```

## Where the numbers come from

Every PDB ID, chain, residue range and hotspot is transcribed from **AlphaProteo
Table S1** (p38), which PXDesign's technical report Table 3 reproduces verbatim
for the six targets it lists. Nothing here was chosen by us.

Two places where the sources disagree or the published form cannot be used
literally, both recorded in the config that carries them:

**TNFa hotspots.** AlphaProteo gives `A113, C73`. PXDesign's Table 3 gives
`A31, A32, A113, C73, C87` — three more. A-CODE follows PXDesign's protocol, so
PXDesign's wider set is what reproduces its row; `tnfa.yaml` ships AlphaProteo's
published set with PXDesign's noted alongside.

**H1 numbering.** Table S1 gives `A1-50, A76-80, A107-111, A258-322, B1-68,
B80-170` with hotspots `B21, B45, B52` — HA numbering, HA1 and HA2 each counted
from 1. 5vli does not deposit it that way: author numbering starts HA1 at 5 and
offsets HA2 by +500, so chain B spans 501-670 and a literal `B1-68` **matches no
residue at all**. Taken literally the published spec silently yields an empty
crop on chain B and a shifted one on chain A — no error, just the wrong epitope.
`h1.yaml` carries the converted ranges and the mapping they came from.

That conversion is why `check_targets.py` exists. It distinguishes a range that
matches *nothing* (a renumbering error) from one with internal gaps (ordinary
unresolved density — 4hsa's chain A is missing 30-40, a loop the other monomer
of the same homodimer resolves). Reverting `h1.yaml` to the published numbering
makes it fail with `crop 1-68 matches NOTHING`, which is the check working.

## Protocol, for whoever runs this

From A-CODE §4.2 and PXDesign's README:

| | |
|---|---|
| Designs per target | 328–728, binder length 80–130 |
| Metric | Designability = share passing the filter |
| Filter — **use this one to compare with Table 4** | pLDDT > 0.80, ipTM > 0.50, ipAE < 10.85 Å, binder bound/unbound RMSD < 3.5 Å |

That filter is A-CODE Appendix C.2, and it is the one ProtDBench ships as
`af2_easy` (`protdbench/protd_configs/eval.py`), where the same threshold is
written in normalized units as `i_pAE < 0.35` — ColabDesign divides PAE by 31.0,
so 0.35 × 31 = 10.85 Å. **Verified against ProtDBench's released per-design
scores: its `af2_easy` reproduces A-CODE Table 4's PXDesign row on all ten
targets to two decimal places** (mean absolute deviation 0.00, r = 1.000).

ProtDBench also ships an alternative, more stringent AF2-IG filter, `af2_opt` —
`pLDDT > 0.9`, `unscaled_i_pAE < 7.0`, `af2_binder_pred_design_rmsd < 1.5 Å`.
**That is not the Table 4 protocol**, and it is not `af2_easy` with tighter
numbers: it drops the ipTM criterion, and its RMSD measures something else.
`af2_easy`'s RMSD is the binder predicted alone against the binder chain of the
complex prediction; `af2_opt`'s is the binder predicted alone against the
original design (`protdbench/tools/af2/main_af2_monomer.py`). On the same
designs the two give a mean of 21.19% and 9.70%, and individual targets move by
up to 300× (H1 12.08 → 0.04, IL7RA 29.80 → 0.26). Reporting both is fine and
probably worth doing — but the column you place next to Table 4 has to be
`af2_easy`, or the percentages are not comparable.

`verify_filter_protocol.py` in this directory recomputes the comparison from
ProtDBench's released per-design scores; `filter_protocol_check.csv` is its
output.

A-CODE reports the co-designed sequence and a ProteinMPNN-redesigned variant
separately; for us those are the two halves of the same question, since the
PMPNN variant is close to a re-run of PXDesign's row.

Two caveats worth knowing before spending compute:

**MSA.** `msa` is unset in every config, and for reproducing Table 4 that is
fine. Neither the diffusion generation nor the AF2-IG filter uses an MSA:
ProtDBench's `af2` block sets `use_initial_guess` and `use_binder_template` and
has no `use_msa` key at all, while `use_msa: True` appears only under `ptx` and
`ptx_mini`. PXDesign's README says the same from the other side — MSA is
"Required for 'Extended' mode (Protenix evaluation)". So an MSA is needed if you
want to add the Protenix-based filters as a cross-check, not for the AF2-IG
column. Only PDL1 ships an example; `pxdesign prepare-msa` fills in the rest.

**Three different filters are in play; do not mix them.** AlphaProteo's own
in-silico benchmark uses pAE < 10, binder RMSD < 1 Å, pLDDT > 80. A-CODE Table 4
uses the PXDesign/`af2_easy` thresholds above. `af2_opt` is a third, stricter
set. Each produces a different percentage from the same designs, so every number
has to carry the name of the filter that produced it.

## Sources

- AlphaProteo — Zambaldi et al., `arXiv:2409.08022`, Table S1 (p38)
- A-CODE — `arXiv:2605.03360`, §4.2 and Table 4
- PXDesign technical report — `PXDesign/assets/technical_report.pdf`, Table 3
- Filter thresholds — `PXDesign/README.md` §3.2
