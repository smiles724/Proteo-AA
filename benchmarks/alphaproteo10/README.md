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
| Filter (**AF2-IG**, strict) | ipAE < 7.0, pLDDT > 0.9, binder RMSD < 1.5 Å |

A-CODE reports the co-designed sequence and a ProteinMPNN-redesigned variant
separately; for us those are the two halves of the same question, since the
PMPNN variant is close to a re-run of PXDesign's row.

Two caveats worth knowing before spending compute:

**MSA.** `msa` is unset in every config. PXDesign calls it optional but
recommended and produced its published numbers with one, so a run without MSA is
not comparable to the table above. Nine of the ten still need one; only PDL1
ships an example.

**AlphaProteo's own filter is not this one.** Its in-silico benchmark uses
pAE < 10, binder RMSD < 1 Å, pLDDT > 80 — looser on pAE, tighter on RMSD. Use
the PXDesign/AF2-IG thresholds above, or the numbers do not compare.

## Sources

- AlphaProteo — Zambaldi et al., `arXiv:2409.08022`, Table S1 (p38)
- A-CODE — `arXiv:2605.03360`, §4.2 and Table 4
- PXDesign technical report — `PXDesign/assets/technical_report.pdf`, Table 3
- Filter thresholds — `PXDesign/README.md` §3.2
