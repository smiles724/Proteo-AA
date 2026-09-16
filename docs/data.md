# Data: Protenix entries, supervision masks, and the eval split

## The two masks, and why both

The side-chain objective is gated by two independent masks that **compose**:

| | Granularity | Knows about | Built by |
|---|---|---|---|
| `missing_atom_mask` | per **atom** | atom presence in atom37: Ser has no CG; residue 41's OE1 was never resolved | FaMPNN, from `all_atom_mask` |
| `supervise_mask` | per **residue** | crystallographic ambiguity: zero/partial occupancy, altloc ties, side-chain mean B > 80, failed chirality or CA–CB checks | `/hai/scratch/yfsun/protenix_sidechain` |

FaMPNN derives `all_atom_mask` from presence alone and never reads occupancy or
altloc, so the second mask is genuinely additive rather than redundant. The
per-residue mask is multiplied *into* the per-atom one, never substituted for it:

```python
missing = missing_atom_mask_from_presence(aatype, atom_mask)
missing = veto_unsupervised_sidechains(missing, supervise, aatype=aatype)
```

`veto_unsupervised_sidechains` (`pxf/train/protenix.py`) enforces three
properties, each asserted:

- **Monotone.** The result is `>=` the input everywhere, so an atom the per-atom
  mask called missing is never revived.
- **Side chains only.** Just the 33 non-backbone slots. A residue with an
  unreliable side chain keeps its N/CA/C/O, hence its local frame and its
  `L_MLM` label — masking the backbone would drop the residue entirely.
- **Ghosts stay ghosts.** An atom that cannot exist for the residue type is not
  reported as a missing one.

Folding into `missing_atom_mask` (rather than adding a separate target-only mask)
is deliberate, and it reaches both consumers: `encoder_inputs` multiplies by
`1 - missing_atom_mask`, so the untrustworthy side chain stops being fed to the
encoder as context, and `sidechain_targets` multiplies by the same, so it stops
being a target. Both are wanted — a side chain too ambiguous to score is also too
ambiguous to condition on.

## Alignment is positional

Mask element `i` is the `i`-th standard amino-acid residue under `gemmi` model 0,
chains in file order, skipping waters, ligands, nucleotides and modified
residues. The iteration is **imported** from the builder's own
`sc_masks.iter_standard_residues` rather than reimplemented, and `read_entry`
asserts the parsed length against the recorded `mask_len`. A divergence of one
residue would shift every label while training still looked healthy, so it is an
error rather than a warning.

## The `keep_fampnn` column

`out_fampnn_strictB/entries.csv` carries both `keep` (inherited from the base
run) and `keep_fampnn`. **`keep_fampnn` is this variant's column.** The shipped
`SideChainMasks` loader auto-detects and lands on `keep`; they agree today
(161,537 entries each), which is exactly why `pxf/train/protenix.py` names the
column explicitly — a regenerated variant could diverge with no visible symptom.

`strictB` blocks `ZERO_OCC | PARTIAL_OCC | ALTLOC_TIE | EXTREME_B |
CHIRALITY_BAD | BOND_OUTLIER` = 876, i.e. `make_fampnn.py --drop-extreme-b`.
`MISSING_SC` is deliberately *not* a blocker: FaMPNN handles it per atom, and
vetoing the whole residue would discard the side-chain atoms that *are* present.

## Splits

| | Index | Entries | Released | Masks |
|---|---|---|---|---|
| train | `weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz` | 168,011 unique | 1976-05-19 → 2021-09-29 | `out_fampnn_strictB`, 161,537 |
| eval | `recentPDB_low_homology_maxtoken1536.csv` | 1,818 unique | 2022-05-04 → 2023-01-11 | `out_eval_fampnn_strictB`, 1,642 |

**The temporal split is clean:** `eval ∩ train = 0`, verified in
`test_ids_from_index_reads_both_splits`. The eval set is also already
low-homology (MMseqs2 + Foldseek clustered, one representative per cluster).

**The eval split needed its own masks.** All 165,470 training masks lie inside
the before-2021-09-30 index, so `eval ∩ masks` was also 0 — the shipped set has
no coverage for the eval entries by construction. `process.py` hard-coded that
index as its only ID source; it now takes `--ids-file`:

```bash
python process.py --ids-file .../recentPDB_low_homology_maxtoken1536.csv \
    --nproc 32 --out out_eval
python make_fampnn.py --in out_eval --out out_eval_fampnn_strictB --drop-extreme-b
```

That produced 1,642 usable entries from 1,818: 88 have no standard protein
residues (nucleic-acid-only entries in the index) and 88 fail the
`sc_atom_completeness >= 0.90` catastrophic cut. 1,124,882 residues, 764,641
supervised (68%).

## Measuring the fine-tune

`scripts/eval_protenix_sidechain.py` reports **two scopes in one pass**:

- `supervised` — canonical residues with a trustworthy deposited side chain. The
  verdict is taken here.
- `all_canonical` — every canonical residue. Context, not adjudicated.

The gap is not small. On three entries with the released 0.0 Å weights:

| | `supervised` | `all_canonical` |
|---|---|---|
| side-chain RMSD | 0.854 Å | 1.079 Å |
| rotamer recovery | 0.805 | 0.721 |

So roughly a quarter of the apparent RMSD on unfiltered residues is
crystallographic ambiguity rather than model error. Scoring against
zero-occupancy or B > 80 side chains measures noise, and noise moves in both
directions — which is why the verdict uses the filtered scope and the unfiltered
one is printed beside it rather than hidden.

`--compare BEFORE AFTER` prints the delta table and **exits non-zero on a
regression** in any of `symmetry_rmsd`, `rotamer_recovery`,
`chi_recovery_20deg`, `lddt_sc_sc` on the supervised scope. Direction conventions
differ between metrics (lower RMSD is better, higher recovery is better), so
`tests/test_eval_protenix.py` pins every one of them, including that an unchanged
model is not reported as a regression and that improving one metric cannot mask
regressing another.

## Running it

```bash
# 1. baseline, released weights
OUT=/hai/scratch/yfsun/proteo_aa_runs/eval_before \
    sbatch scripts/slurm/eval_protenix_sidechain.sh

# 2. fine-tune on the training split, masks applied
OUT=/hai/scratch/yfsun/proteo_aa_runs/ft_protenix \
    sbatch scripts/slurm/train_protenix.sh

# 3. same eval on the result
CHECKPOINT=/hai/scratch/yfsun/proteo_aa_runs/ft_protenix/checkpoints/final.pt \
    OUT=/hai/scratch/yfsun/proteo_aa_runs/eval_after \
    sbatch scripts/slurm/eval_protenix_sidechain.sh

# 4. the verdict
python scripts/eval_protenix_sidechain.py --compare \
    /hai/scratch/yfsun/proteo_aa_runs/eval_{before,after}
```

`NO_MASK=1` on step 2 runs the ablation that trains on FaMPNN's per-atom mask
alone — that is the run that says what the crystallographic filter was worth.

Every training run records the mask set it used under
`run_config.json:data_source.supervision_mask` (root, `keep_column`, blocker
bitmask, entry count), so a checkpoint is never ambiguous about its supervision.
