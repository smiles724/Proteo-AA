# HAI: the unconditional designability benchmark

Two arms over the Table 2-style metrics -- designability, codesignability,
diversity -- at lengths {100, 200, 300}. Marlowe has these queued behind a
Thursday fairshare estimate; they are held there as a backstop.

**The three scoring scripts are NEW and have never run against real data.**
Smoke each one before the batch.

## Code

```bash
git fetch origin feat/multi-event-feedback
git checkout feat/multi-event-feedback     # 0967fc2 or later
```

New in `scripts/uncond/`: `mpnn_designs.py`, `esmfold_refold.py`,
`score_uncond.py`. Plus `scripts/slurm/marlowe/codesign_uncond.sh` (the
existing `scripts/slurm/codesign_uncond.sh` is the HAI original and may
already suit you).

## The two arms, and why they share backbones

| arm | backbone | co-design |
|---|---|---|
| baseline | PXDesign, unconditional, to completion | FaMPNN 0.0, **no adapter** |
| **(a)** | *the same backbones* | FaMPNN 0.0 + **monomer A_BS** |

`sample_uncond.py`'s adapter path requires `enable_sc_to_bb`, and the
monomer A_BS is `bb_to_sc` only, so **A_BS provably cannot change the
backbone**. Both arms therefore score on one set of backbones and differ
only at co-design. That is a paired comparison and it halves the
generation cost -- do not generate twice.

## Checkpoints: use the MONOMER line, with FaMPNN 0.0

Newly staged in the bundle under `checkpoints/monomer/`:

| file | sha256 (first 16) |
|---|---|
| `A_BS_couple_phase1_final.pt` | `8065511457766df3` |
| `early_s_bb_only_119662_final.pt` | `3a4e872ba7f1d593` |
| `early_s_full_119661_final.pt` | `0bf37620b2abebf7` |

and `checkpoints/donors/fampnn_0_0.pt`, also new. Re-pull those two
directories; nothing else in the bundle changed.

**FaMPNN 0.0, not 0.3.** `couple_phase1` was fit against 0.0 (its
`run_config.json` says `fampnn_weights: 0.0`). Crossing an adapter with the
wrong donor loads cleanly and measures nothing -- that is the failure that
made an earlier packing smoke worthless.

Why this line rather than J03: all 512 J03/S03 training examples are
complexes (target 43-384 residues, median 145, zero monomers), and
unconditional generation produces monomers. `couple_phase1` was fit on 2000
AFDB structures, which are monomers. J03 here would be off-distribution
three ways at once.

**Honest caveat to carry:** 0.0 is the no-noise donor, recommended by
upstream for *packing*, while 0.3 is "recommended for sequence design"
because it tolerates imperfect backbones -- and diffused backbones are
exactly that. So arm (a) is matched on data and mismatched on backbone
regime. It is still the better of the two options, but do not present it
as fully matched.

## Tools to install

```bash
curl -fsSL -o foldseek.tar.gz https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
tar xzf foldseek.tar.gz          # -> foldseek/bin/foldseek
```

ESMFold: `transformers.EsmForProteinFolding`, NOT `esm.pretrained` -- the
latter needs openfold and its CUDA extensions, the former is a pure PyTorch
port of the same weights. `huggingface_hub.snapshot_download('facebook/esmfold_v1')`,
about 8 GB. Your `tool_weights` may already have it.

ProteinMPNN needs no download: ColabDesign bundles `v_48_020`.

## The chain

```bash
# 1. backbones, shared by both arms
for L in 100 200 300; do
  OUT=$R/uncond/baseline LENGTHS="$L" NUM_SAMPLES=128 sbatch .../sample_uncond.sh
done

# 2. co-design, two arms, same backbones
SAMPLES=$R/uncond/baseline OUT=$R/uncond/codesign_baseline FAMPNN=0.0 sbatch ...
SAMPLES=$R/uncond/baseline OUT=$R/uncond/codesign_arm_a   FAMPNN=0.0 \
  ADAPTERS=$BUNDLE/checkpoints/monomer/A_BS_couple_phase1_final.pt sbatch ...

# 3. ProteinMPNN@8 from the backbones (af2ig venv)
python scripts/uncond/mpnn_designs.py --samples-dir $R/uncond/baseline \
    --out $R/uncond/mpnn.csv --num-seqs 8

# 4. ESMFold everything: 8 MPNN + 1 co-designed per sample, per arm.
#    fold_id for a co-designed sequence MUST be "<sample_id>__codesign".
python scripts/uncond/esmfold_refold.py --sequences <all seqs>.csv \
    --out $R/uncond/refolds        # shardable, caches by fold_id

# 5. score, per arm
python scripts/uncond/score_uncond.py --samples-dir $R/uncond/baseline \
    --codesign-dir $R/uncond/codesign_arm_a --refolds-dir $R/uncond/refolds \
    --mpnn-designs $R/uncond/mpnn.csv --out $R/uncond/report_arm_a \
    --foldseek <path>/foldseek/bin/foldseek
```

Step 4 needs a small CSV assembling `fold_id,sequence` from the MPNN table
and each arm's `samples/*.fasta`. I have not written that joiner -- it is
five lines and depends on your layout.

## Three things in the scorer worth checking before trusting it

**Designability is CA-only, deliberately.** ProteinMPNN's sequence does not
match the generated one, so all-atom correspondence is undefined there.
Codesignability reports both CA and all-atom, where identities agree by
construction.

**The all-atom join is on residue ORDER, not `res_id`.** ESMFold renumbers
from 1; joining on `res_id` yields a shifted or empty correspondence that
still returns a plausible number.

**PMPNN@1 is the best-scoring of the eight, not a separate run.** Folds are
cached by `fold_id`, so @1 and @8 cost the same.

Unknown residues become glycine before folding, per the benchmark spec, and
the count is printed rather than done silently.

## What to report

Per arm: designability PMPNN@1 and @8, codesignability CA-only and
all-atom, diversity (foldseek Str / Seq / Str+Seq cluster counts), each
with n and the median scRMSD. Threshold is 2.0 A throughout.

Report **per length as well as pooled** -- 128 samples at each of three
lengths, and the spec averages across samples, so a pooled number hides
whether an effect is length-dependent.

The two arms are paired per backbone, so report the paired difference and
the fraction of backbones favouring arm (a), not just two independent rates.
