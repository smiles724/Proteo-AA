# Binder benchmark: reproducing A-CODE's Table 4

Status, provenance and handoff for the ten-target binder-design benchmark.
Data details live next to the data in
[`benchmarks/alphaproteo10/README.md`](../benchmarks/alphaproteo10/README.md);
this is the state and what is left.

## What is being reproduced

Three papers, one contribution each:

| | contributes |
|---|---|
| **AlphaProteo** (`arXiv:2409.08022`) | the ten targets — PDB IDs, chains, crops, hotspots, in Table S1 (p38) |
| **PXDesign** | the protocol and the filter thresholds — and **our own backbone**, so its row is our baseline |
| **A-CODE** (`arXiv:2605.03360`) | the table we want to appear in (§4.2, Table 4) |

A-CODE states both borrowings outright: the targets are "as proposed in Zambaldi
et al.", and "we follow PXDesign to use the filter from AF2-IG". It invented
neither, and neither do we.

Designability, percent, from A-CODE Table 4:

| | BHRF1 | H1 | IL17A | IL7RA | IR | PDL1 | SC2RBD | TNFa | TrkA | VEGFA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PXDesign** (two-stage) | 43.90 | 12.08 | 0.82 | 29.80 | 25.04 | 45.33 | 11.20 | 3.43 | 23.55 | 16.72 |
| A-CODE (PMPNN) | 25.00 | 65.59 | 1.79 | 4.93 | 30.09 | 39.96 | 28.05 | 6.16 | 6.87 | 1.37 |
| A-CODE (co-design) | 22.87 | 55.71 | 1.24 | 4.05 | 41.67 | 28.70 | 37.50 | 7.57 | 3.70 | 0.96 |
| Protpardelle-1c (one-stage) | 3.73 | 0.27 | 0.00 | 0.09 | 0.19 | 7.04 | 0.99 | 0.00 | 3.52 | 0.17 |

**PXDesign's row is our starting point** — our backbone is theirs — and A-CODE is
the direct competitor, doing the same one-stage co-design on the same substrate.
The question the benchmark answers is whether co-design moves that row up or
down.

Note that A-CODE reports **two** rows for itself: its own co-designed sequence,
and the same backbones with the sequence thrown away and redesigned by
ProteinMPNN. We need both for the same reason it did — the gap between them is
exactly "can our AA head replace ProteinMPNN". At 13% recovery against
ProteinMPNN's 33%, expect that row to look bad; it is also the number we most
need.

## What Designability measures

Generate N designs for a target, filter, report the share that passes.

The filter is a **self-consistency check**. Hand the designed sequence to
AlphaFold2, seeded with the design itself as the initial guess, and ask what
structure it predicts. If an independent predictor agrees the sequence folds
that way and docks there, the design is plausible; if it predicts something else
or is unconfident, it is not.

A-CODE Table 4 uses the PXDesign filter, all four required (A-CODE Appendix C.2;
the same thing ProtDBench ships as `af2_easy`):

```
pLDDT > 0.80     confident about the binder's own fold
ipTM  > 0.50     confident the two chains form an interface
ipAE  < 10.85 A  confident about how they sit relative to each other
RMSD  < 3.5 A    the binder takes the same shape predicted alone as predicted bound
```

The `10.85` looks arbitrary because it is a unit conversion: BindCraft and the
ColabDesign family write this threshold as `i_pAE < 0.35`, having divided PAE by
its 31.0 A ceiling. 0.35 x 31 = 10.85.

**There is a second, more stringent AF2-IG filter and it is easy to reach for by
mistake:** ProtDBench's `af2_opt` — `pLDDT > 0.9`, `unscaled_i_pAE < 7.0`,
`af2_binder_pred_design_rmsd < 1.5 A`. It is harder to pass, but it is not
`af2_easy` with the thresholds tightened: it drops the ipTM criterion, and its
RMSD is a different measurement. `af2_easy` compares the binder predicted alone
against the binder chain of the complex prediction; `af2_opt` compares the
binder predicted alone against the original design
(`protdbench/tools/af2/main_af2_monomer.py`). Reporting both is worthwhile, but
only `af2_easy` is the Table 4 protocol — on ProtDBench's released PXDesign
designs the two give 21.19% and 9.70% on average, and up to 300x apart on single
targets. Rerun the check yourself with
`benchmarks/alphaproteo10/verify_filter_protocol.py`.

This is a **prediction, not a measurement**. AlphaProteo has real experimental
hit rates (9–88% by target) but those need a wet lab. Designability is the
agreed computational surrogate, which is what makes cross-method comparison
possible at all.

## What is built

**The ten targets, as runnable configs.** A-CODE's text gives no
specifications — it names the ten and cites AlphaProteo — so they were
transcribed from Table S1 and cross-checked against PXDesign's technical report
Table 3, which reproduces six of them verbatim. `download_structures.sh` fetches
all ten from the RCSB; `check_targets.py` verifies every crop range and hotspot
exists in the structure it names. Ten of ten pass.

**A de novo inference path** (`scripts/evaluation/design_binder_from_target.py`).
This was the missing half, and its absence was not obvious: every inference entry
point in the repo requires the binder to already exist in the structure, because
training scrubs a real chain and asks the model to rebuild it. Real design has no
such chain — only a target and a length — so a placeholder has to be fabricated
before the model has anything to denoise. Verified end to end on all ten targets.

PDL1 is the correctness check on both at once: this repo's author-numbered
`17-132` with hotspots `56/115/123` becomes exactly PXDesign's own example
`crop: ["1-116"]`, `hotspots: [40, 99, 107]` after parsing.

## How to use it

```bash
./benchmarks/alphaproteo10/download_structures.sh
python benchmarks/alphaproteo10/check_targets.py          # optional, verifies the configs

python scripts/evaluation/design_binder_from_target.py \
    --target benchmarks/alphaproteo10/targets/pdl1.yaml \
    --checkpoint <stage III checkpoint> \
    --seed 1 --out designs/
```

One invocation produces one design. A full benchmark run loops it: A-CODE samples
328–728 binders per target at lengths 80–130. Roughly 60 GPU-hours all in, so
about 15 hours on four cards — compute is not the constraint here.

**`binder_length` in the configs is a smoke/default value, not the benchmark
protocol.** It is a single number — 105, the midpoint of A-CODE's 80–130 — which
is enough to check the pipeline runs end to end, but a run that leaves it there
samples one length ten times over. A production run sweeps a range, and there are
two conventions: A-CODE's uniform 80–130 across all ten targets, or the
per-target range each YAML records in its comments, which is what ProtDBench uses
(every integer in it — BHRF1 gets 80…120, IL-17A gets 50…140). Pick one and say
which; the length distribution changes the percentages.

**Which checkpoint.** One with all three components trained: backbone, AA head
and side-chain module — i.e. the *output* of Stage III / binder training, not the
Stage II and AA-head checkpoints it warm-starts from. Only after Stage III has
run has the co-evolution path been trained, and that is the thing under test.

Without `--checkpoint` the script runs untrained weights and says so loudly. That
checks the plumbing and nothing else.

The load is validated: the `module.` prefix a multi-GPU run writes is stripped,
and a checkpoint matching fewer than half the parameters raises instead of
loading. Both matter because `strict=False` is otherwise required — a Stage III
checkpoint legitimately lacks some buffers — and would turn a total mismatch into
a silent no-op, leaving the model at its random initialisation while reporting
success. If it refuses your checkpoint, that is the guard, not a bug: the message
names the keys that did not match.

## What is missing

**The scorer.** Generation produces CIF files; without AF2-IG they cannot become
a number. This is the one real blocker, and it is an environment build rather
than a task:

```
pip install git+https://github.com/bytedance/PXDesignBench.git@v0.1.2   # provides pxdbench
  ... which needs, per PXDesign's install.sh:
      Protenix v0.5.0+pxd          <-- NOT our version, see below
      ColabDesign (--no-deps)
      JAX with CUDA                <-- AF2 is JAX, not PyTorch
      einops, natsort, dm-tree, posix_ipc,
      transformers==4.51.3, dm-haiku==0.0.13, optax==0.2.5

bash PXDesign/download_tool_weights.sh <dir>    # AF2 params + ProteinMPNN weights
  then point pxdbench/globals.py at that directory
```

**The version conflict is the thing to plan around.** `pxdbench` and PXDesign's
own inference path expect **Protenix v0.5.0+pxd**; this repo trains against
**v2.0.0**, where modules moved (`protenix.data.ccd` →
`protenix.data.core.ccd`, `protenix.data.parser` → `protenix.data.core.parser`).
Importing PXDesign's pipeline here fails immediately on that. Scoring therefore
belongs in a **separate environment** from training — which is fine, since it
consumes CIF files and needs nothing from ours.

`pxdbench` ships `binder_eval_demo.sh`, which is close to what a scoring run
should look like.

**MSA.** Unset in all ten configs; only PDL1 has one in the repo. It does not
affect generation — PXDesign states MSAs are not required for the diffusion stage
— and AF2-IG runs single-sequence, so a first pass at Table 4 likely does not
need them. They *are* required for the Protenix-based filters, so getting them
unlocks a second, independent filter to cross-check with. `pxdesign prepare-msa`
is the intended route but goes through PXDesign, so it hits the same version wall;
Protenix's own MSA search is a remote MMseqs2 call
(`https://protenix-server.com/api/msa`, ColabFold as fallback), roughly fifteen
requests for all ten targets.

## Where results should go

Not in git: the designs themselves, 3,280–7,280 CIFs at ~266 KB each, 0.8–1.9 GB.
They are regenerable output and would bloat every clone.

In git: the ten-row summary, the per-design scores (small compressed, and enough
for anyone to re-derive the statistics or change a threshold without re-running),
and a note recording which checkpoint and commit produced them. A table nobody
can trace back to a checkpoint is not usable six months later.

## One trap worth knowing

**H1 cannot be transcribed literally.** Table S1 gives `B1-68, B80-170` with
hotspots `B21/B45/B52` — HA numbering, HA1 and HA2 each counted from 1. 5vli does
not deposit it that way: HA2 is offset by +500, so chain B spans 501-670 and a
literal `B1-68` **matches no residue at all**. No error, just an empty crop on
chain B and a shifted one on chain A, and a binder designed against an epitope
nobody chose. The config carries the converted ranges and the mapping they came
from; `check_targets.py` fails loudly on the published form, which is what it is
for.
