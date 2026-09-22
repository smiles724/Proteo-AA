# ConditionalBinderDesignBenchmark

The binder-design test set from **A-CODE: Fully Atomic Protein Co-Design with
Unified Multimodal Diffusion** (arXiv:2605.03360, §4.2 + Appendix C.2), wired up
to run against a Proteo-AA **Stage III (co-evolution)** checkpoint.

- Test set: [`pxdesign_train/benchmarks/targets/conditional_binder_design_v1.json`](../pxdesign_train/benchmarks/targets/conditional_binder_design_v1.json)
- Loader / metric: [`pxdesign_train/benchmarks/conditional_binder.py`](../pxdesign_train/benchmarks/conditional_binder.py)
- Input preparation: [`pxdesign_train/benchmarks/target_prep.py`](../pxdesign_train/benchmarks/target_prep.py)
- Generation: [`scripts/evaluation/eval_conditional_binder_benchmark.py`](../scripts/evaluation/eval_conditional_binder_benchmark.py)
- Folding (AF2-IG): [`scripts/evaluation/fold_af2ig.py`](../scripts/evaluation/fold_af2ig.py), [`pxdesign_train/benchmarks/af2ig.py`](../pxdesign_train/benchmarks/af2ig.py)
- Folding environment: [`scripts/utilities/bootstrap_af2ig.sh`](../scripts/utilities/bootstrap_af2ig.sh)
- Scoring: [`scripts/evaluation/score_af2ig_designability.py`](../scripts/evaluation/score_af2ig_designability.py)
- Smoke fixture: [`scripts/evaluation/make_smoke_design_run.py`](../scripts/evaluation/make_smoke_design_run.py)
- Tests: [`tests/test_conditional_binder_benchmark.py`](../tests/test_conditional_binder_benchmark.py), [`tests/test_af2ig_scoring.py`](../tests/test_af2ig_scoring.py)

## What the paper specifies

> We benchmark the conditional generation ability of A-CODE on a test set
> comprising 10 protein targets with diverse structural properties, as proposed
> in Zambaldi et al. In binder design, a target protein with known sequence and
> structure is fed in as the conditions for the generative model. Hotspot
> information of the binding site is often available as additional input. For a
> unified evaluation, we follow PXDesign to use the filter from AF2-IG. For each
> different target, we sample 328-728 binders with lengths ranging from 80 to
> 130, following the same protocol in PXDesign, and report the percentage of
> designable samples as Designability. We evaluate both the model-generated
> co-designed sequence and the PMPNN-redesigned single sequence as separate
> variants. — §4.2

> ... a designable binder is defined as: a) interface predicted absolute error
> (ipAE) less than 10.85 Å, and b) interface predicted TM score (ipTM) greater
> than 0.5, and c) predicted local distance difference test (pLDDT) greater than
> 80%, and d) binder bound/unbound RMSD less than 3.5 Å. AF2 is used to evaluate
> these metrics, in which residues in the binder chain are assigned a large
> offset to mimic multichain co-folding. Across different binder lengths for the
> same target, the success counts are summed to obtain the final result.
> — Appendix C.2

Those thresholds are the `AF2-IG-easy` row of PXDesign's technical report
Table 2 (originally BindCraft's). A-CODE reports it as "AF2-IG"; the numbers are
what matter and they are pinned in the manifest and in
`tests/test_conditional_binder_benchmark.py`.

## The test set

A-CODE gives the ten target *names* (Table 4) and defers their crops and hotspots
to PXDesign, which defers to AlphaProteo (Zambaldi et al.). All ten come from
**AlphaProteo Table S1**, "Binder design problem specifications for in silico
benchmarking and experimental testing"; six are independently corroborated by
PXDesign's Table 3, and every entry was re-verified against the local mmCIF
mirror.

| Target | PDB | Target chains / crop (author numbering) | Hotspots | AlphaProteo length range |
|---|---|---|---|---|
| BHRF1  | 2WH6 | A 2–158 | A65, A74, A77, A82, A85, A93 | 80–120 |
| H1     | 5VLI | A 1–50, 76–80, 107–111, 258–322; B 501–568, 580–670 | B521, B545, B552 | 40–120 |
| IL17A  | 4HSA | A 17–131, B 19–127 | A94, A116, B67 | 50–140 |
| IL7RA  | 3DI3 | B 17–209 | B58, B80, B139 | 50–120 |
| IR     | 4ZXB | E 6–155 | E64, E88, E96 | 40–120 |
| PDL1   | 5O45 | A 17–132 | A56, A115, A123 | 50–120 |
| SC2RBD | 6M0J | E 333–526 | E485, E489, E494, E500, E505 | 80–120 |
| TNFa   | 1TNF | A 12–157, B 12–157, C 12–157 | A113, C73 | 50–120 |
| TrkA   | 1WWW | X 282–382 | X294, X296, X333 | 50–120 |
| VEGFA  | 1BJ1 | V 14–107, W 14–107 | W81, W83, W91 | 50–140 |

The length-range column is AlphaProteo's own per-target range, recorded for
traceability. It is **not** what the harness samples: A-CODE overrides it with its
own 80–130 window (see [Sampling grid](#sampling-grid)).

### How "verified" was established

Every crop range and hotspot residue was checked against the deposited entry in
`/hai/scratch/yfsun/protenix_data/mmcif`:

* every hotspot exists at the stated **author** residue number;
* for the seven entries whose deposition has a partner bound at that site, each
  hotspot lies **2.2–7.1 Å** from it — i.e. they really are interface residues,
  which is what rules out an off-by-one or a label/author numbering mix-up;
* the exceptions are TNFa (1TNF is apo) and IR (nothing in 4ZXB comes within
  13 Å of the L1 patch). For those, the check is that the residues resolve — and
  for IR, that all three resolve to PHE, which is not a coincidence at three
  specified positions;
* PDL1 cross-checks a third way, against PXDesign's shipped
  `examples/PDL1_quick_start.yaml`: chain A cropped to `1-116` with hotspots
  `[40, 99, 107]` in 1-based within-chain indexing — 116 residues, and
  17+40−1 = 56, 17+99−1 = 115, 17+107−1 = 123.

Re-run it any time (no GPU needed); it fails loudly if a hotspot stops resolving:

```bash
VALIDATE=1 bash scripts/evaluation/slurm_eval_conditional_binder_benchmark.sh
```

### Two entries where the sources needed a judgement call

**TNFa — the sources disagree.** AlphaProteo Table S1 gives two hotspots, A113
and C73. PXDesign Table 3 gives five: A31, A32, A113, C73, C87. The crop is
identical in both. The manifest follows **AlphaProteo**, because A-CODE defines
its test set as the targets "proposed in Zambaldi et al.", and PXDesign's Table 3
describes its own wet-lab campaign rather than the in-silico benchmark spec.
Pinned by `test_tnfa_follows_alphaproteo_not_pxdesign`.

**H1 — a numbering offset, the one entry not transcribed literally.** AlphaProteo
quotes the HA2 chain in canonical 1–175 numbering (`B1-68, B80-170`, hotspots
`B21, B45, B52`), but 5VLI deposits HA2 as author residues 501–670. Since this
manifest is author-numbered throughout, chain B carries **+500** on both ranges
and hotspots; HA1 (chain A) needs no shift. With the shift, B521/B545/B552 resolve
to TRP/ILE/VAL, each 3.5–3.8 Å from a non-cropped chain — the HA stem epitope.
Without it, all three hotspots fall outside the crop, which is how the offset was
found in the first place. Pinned by
`test_h1_chain_b_carries_the_hemagglutinin_numbering_offset`.

### If a target ever goes unsourced again

The `pending_source` machinery is still live and still tested (against a synthetic
manifest, since the shipped one is complete). An entry with `"status":
"pending_source"` is skipped by `tasks()`, raises under
`tasks(include_pending=True)`, and is listed by `score_af2ig_designability.py`
under `targets_missing_from_this_run` — so a partial test set can never quietly
look like a full one.

## Sampling grid

The paper reports 328–728 samples per target over lengths 80–130 and says the
per-length success counts are summed. It does not publish the per-target length
lists (PXDesign: *"the length of generated binders strictly followed the previous
work"*), which is why the totals vary per target. The manifest therefore ships a
uniform grid — lengths `{80, 90, 100, 110, 120, 130}` × 64 samples = **384 per
target**, inside the paper's envelope — and honours per-target `lengths` /
`samples_per_length` overrides so AlphaProteo's exact lists can be dropped in
without touching code.

Diffusion sampling defaults to **1000 Euler steps** (A-CODE §4.1: *"we follow
PXDesign to use 1000 Euler steps for diffusion sampling"*). PXDesign's own report
used 400 for binders and 1000 for monomers; `--diffusion-steps` overrides.

## Running it

### 1. Validate inputs (no GPU)

```bash
VALIDATE=1 bash scripts/evaluation/slurm_eval_conditional_binder_benchmark.sh
```

Prepares every `(target, length)` input and exits. This is where a missing
structure or an unresolvable hotspot surfaces — before a GPU is allocated.

### 2. Generate designs

```bash
CHECKPOINT=/hai/scratch/yfsun/proteo_aa_runs/protenix_monomer_stage3_coevolution/<job>/checkpoints/step30000.pt \
sbatch scripts/evaluation/slurm_eval_conditional_binder_benchmark.sh
```

Smoke first (`SMOKE=1`, two targets × one length × two samples × 20 steps) to
prove the harness assembles. The paper-scale run is 6 lengths × 64 samples × 10
targets = 3,840 designs at 1000 steps; split it with `TARGETS=...` across jobs if
needed — `designs.csv` is append-only and the driver resumes from it.

Outputs, under `--output-dir`:

```
inputs/<target>_L<len>.cif        # cropped target + placeholder binder chain
inputs/<target>_L<len>.prep.json  # hotspot resolution + author-numbering map
designs/<sample_id>.pdb           # deposited target + designed binder
designs/<sample_id>.fasta         # designed binder sequence
designs.csv                       # one row per design
run_summary.json
```

### 3. Fold with AF2 initial-guess

Folding runs in its **own environment**. AF2 is JAX and training is
torch + Protenix, and PXDesign's own AF2 path pins Protenix `v0.5.0+pxd` while
this repo trains against `v2.0.0` (`protenix.data.ccd` moved to
`protenix.data.core.ccd`, `protenix.data.parser` to `protenix.data.core.parser`),
so the two cannot share an interpreter. They do not need to: scoring consumes
design PDBs and nothing else.

```bash
bash scripts/utilities/bootstrap_af2ig.sh        # venv + AlphaFold parameters
```

That installs JAX, ColabDesign pinned to a commit, and extracts exactly two
AlphaFold parameter files from DeepMind's 4.7 GB tar — `model_1_ptm` for the
complex pass and `model_3_ptm` for the unbound one. ProteinMPNN needs no
download: ColabDesign ships the original `v_48_020` weights inside the package,
which is what makes the PMPNN arm of Table 4 free.

```bash
RUN_DIR=<output-dir> VARIANTS="co_design pmpnn" \
  sbatch scripts/evaluation/slurm_fold_af2ig.sh
```

`DRY_RUN=1 bash scripts/evaluation/slurm_fold_af2ig.sh` resolves every design and
its binder chain and exits before touching JAX — that is where a moved run
directory or an ambiguous binder chain surfaces, on a login node rather than
twenty minutes into a GPU allocation.

Output is `af2ig_metrics.csv`, one row per (design, variant), append-only and
resumable: a requeued job re-reads it and skips what is already scored.

**Keep `inputs/` next to `designs/`.** The folder reads each
`inputs/<task>.prep.json` to learn which chain is the binder, and that is not
always `B`: `choose_binder_chain_id` takes the first letter the target does not
already use, so IL17A (A,B), TNFa (A,B,C) and H1 (A,B) all put the binder on
**Z**, and VEGFA (V,W) puts it on B. Without the sidecar the folder falls back
to "the one chain with exactly `binder_length` residues" and *aborts* when that
is ambiguous — which it is whenever a target chain happens to be 80-130
residues. Aborting is deliberate: scoring the wrong chain yields a complete,
believable, wrong row.

**Two AlphaFold passes per design**, and the second one is not optional:

| pass | protocol | gives |
|---|---|---|
| complex | `binder`, `initial_guess=True`, templates removed from the binder | ipAE, ipTM, binder pLDDT, predicted-vs-designed RMSD |
| unbound | `hallucination`, no template, no guess | the other half of criterion (d) |

A-CODE's criterion (d) is the *bound/unbound* RMSD — the binder as AF2 places it
in the complex against the same sequence folded alone. That is a property of the
sequence off its partner and cannot be read out of the complex prediction, which
is why most AF2-IG wrappers do not report it: dl_binder_design's and
ColabDesign's `rmsd` is predicted-vs-**designed**, aligned on the target, a
different question. Both are recorded — the filter uses
`binder_bound_unbound_rmsd`, and `binder_designed_rmsd` is carried alongside so a
run can be compared under either convention without refolding. The unbound pass
is cached by sequence, so the second variant of a backbone costs one pass, not
two.

### 4. Score

```bash
python scripts/evaluation/score_af2ig_designability.py \
    --metrics-csv <output-dir>/af2ig_metrics.csv --output-dir <output-dir>
```

Prints a Table 4-shaped row per variant and writes `designability.json` +
`designability_per_target.csv` (per-target and per-length). This step needs
neither a GPU nor Protenix — it is a published threshold applied to a CSV, and
it imports the manifest module directly for that reason, so re-deriving a
number after a threshold changes never requires the training stack.

### Testing the scoring half before a checkpoint exists

Generation needs a trained Stage III checkpoint; scoring needs AlphaFold
parameters and a second environment. The second is likelier to be broken on
arrival, so it can be exercised first:

```bash
python scripts/evaluation/make_smoke_design_run.py \
    --out /tmp/cbdb_smoke --mmcif-dir <mmcif> --targets PDL1 --n-samples 2
python scripts/evaluation/fold_af2ig.py --run-dir /tmp/cbdb_smoke --data-dir <params>
```

That writes a run directory of the right shape whose "designs" are the real
cropped target plus the inert placeholder helix carrying an arbitrary heptad
sequence. **They are not designs and every metric they produce is noise.** What
they verify is the plumbing: that the binder chain resolves, that ColabDesign
splits the complex where this harness thinks it does, that both passes run, that
ProteinMPNN redesigns the binder and leaves the target byte-identical, and that
the CSV the folder writes is the CSV the scorer reads.

Measured on PDL1 (2 designs × 2 arms, one H100, 3 recycles, 48 s wall): ipAE
≈ 26.7 Å, ipTM ≈ 0.09, binder pLDDT ≈ 0.89, bound/unbound RMSD ≈ 1.5–2.1 Å, and
Designability 0.00% on both arms — a well-formed helix docked nowhere, which is
exactly what a placeholder should score.

The same run is where the timing numbers in `slurm_fold_af2ig.sh` come from:
~28 s for the first pair (JIT, paid once per token count, i.e. once per
(target, length) cell) and 0.6–0.7 s per pair afterwards for both AlphaFold
passes together. AF2 runs single-sequence here, so the cost is the pair stack
and scales roughly quadratically in tokens — PDL1 at 196 tokens is the cheapest
target by some way.

Note the placeholder's backbone is an ideal CA trace with N/C/O hung off it and
is **not peptide-bonded** — C(i)–N(i+1) measures 0.81 Å against a real 1.33 Å.
That does not matter for its real job (during input preparation those
coordinates never reach the model) but ProteinMPNN reads backbone geometry and
nothing else, so on this fixture the PMPNN arm returns poly-serine. Read that
arm here as "it ran and held the target fixed", not as a sequence you would
look at.

## How the design input is built

Every training provider in this repo starts from a real complex and *masks* one
chain, so none of them can express "target + L design residues". Rather than add
a second featurization path — which is how the leakage bug in
`tests/test_data_contract_parity.py` happened — preparation writes a structure
file and reuses the repo's own loader:

```
deposited mmCIF -> crop to the manifest's ranges
                -> append a poly-GLY placeholder binder chain
                -> write prepared mmCIF
                -> CifFileProvider(dataset="Distillation")
                -> DesignSourceDataset(inference_safe_binder=True)
                -> overwrite `hotspot` with the published residues
                -> cogenerate()
```

Four decisions in there are load-bearing:

1. **`dataset="Distillation"`, not `"WeightedPDB"`.** The WeightedPDB parser runs
   PDB-curation filters. Two of them break a prepared input: assembly expansion
   needs deposition-style `pdbx_struct_assembly` records, and
   `remove_dissociation` deletes the placeholder chain outright — measured, it
   silently dropped all 80 binder residues. The distillation parser skips those
   while still tokenising polymers per residue.

2. **Polymer entities are declared explicitly.** A structure file carrying only
   `atom_site` is parsed as a bag of ligands and tokenised *per atom* — measured,
   a 116-residue target became 930 tokens. `write_prepared_cif` emits
   `entity` / `entity_poly` / `entity_poly_seq` / `struct_asym`.

3. **Numbering.** Published hotspots are author numbers; the parser reads
   `label_seq_id` into `res_id`, so author numbering does not survive it.
   Preparation resolves author → sequential once, while it still has both, and
   records the map in the `.prep.json` sidecar. Nothing downstream re-derives it.

4. **Hotspots are installed, not sampled.** Training samples hotspots from binder
   contacts; the benchmark conditions on a fixed published set, and the
   placeholder binder has no meaningful contacts anyway. The dataset is built
   with `hotspot_force_zero_prob=1.0` and `apply_hotspots` *raises* if the channel
   is non-empty before it writes, so a sampled hotspot can never union into the
   published set.

The placeholder binder's coordinates never reach the model: it is marked `[xpb]`,
rebuilt to exactly N/CA/C/O with residue-type-independent reference metadata,
excluded from `conditional_templ`, and `cogenerate` starts from
`x = sigma_0 * randn` rather than from any ground-truth coordinate. Its geometry
(an ideal α-helix placed 12 Å off the hotspot centroid) exists only so that
parsing accepts it as a polymer and so `DesignCropper`'s proximity ranking keeps
the binding site if a target ever needs cropping.

**Designs are re-anchored to the deposited target.** PXDesign-d soft-conditions
the target through binned pair distances and denoises *every* atom from noise,
target included, so a sample's target coordinates are a prediction. Each design is
superimposed back onto the deposited target on target CA atoms before it is
written, so every output PDB shares the real target frame — otherwise AF2-IG would
be scoring a binder posed against a predicted target.

## Deviations from the paper, and chosen values

Anything not pinned by the paper is listed here rather than left implicit.

| Item | Paper | Here |
|---|---|---|
| Per-length sample counts | unpublished; per-target totals 328–728 | uniform 6 × 64 = 384, overridable per target |
| Per-target length ranges | AlphaProteo's are 40–140 and vary per target; A-CODE overrides with 80–130 | A-CODE's 80–130; AlphaProteo's recorded per target as `alphaproteo_binder_length_range` but unused |
| Binder-chain residue offset for AF2 | "a large offset" | **50**, ColabDesign's binder protocol, which the folder does not override. The manifest's `filter.chain_break_offset = 200` is the AF2-multimer convention and is recorded, but AF2's monomer relative-position encoding clips at ±32, so 50 and 200 are the same input to the model |
| AF2 models | "AF2" | `model_1_ptm` for the complex (the binder protocol needs a template stack), `model_3_ptm` for the unbound fold (it does not). One model each, not an ensemble |
| AF2 recycles | unstated | **3**, the AF2-IG / dl_binder_design default |
| Unbound prediction | implied by criterion (d), not specified | single-sequence, no template, no initial guess; cached by sequence |
| PMPNN redesign | "the PMPNN-redesigned single sequence" | ColabDesign's bundled original ProteinMPNN (`v_48_020`), temperature **1e-4** (PXDesign's), one sequence per backbone, target held fixed, **no** amino acid excluded (`--mpnn-rm-aa C` restores BindCraft's cysteine ban) |
| TNFa hotspots | A-CODE cites Zambaldi et al. | AlphaProteo's A113 + C73, **not** PXDesign Table 3's five |
| H1 chain-B numbering | AlphaProteo quotes HA2 as 1–175 | +500 applied, to match 5VLI's author numbering |
| Diffusion steps | 1000 (A-CODE) | 1000 |
| Placeholder binder geometry | n/a (not part of the task) | ideal α-helix, 12 Å standoff; inert |

Note also that A-CODE's own Stage 3 is non-canonical-amino-acid finetuning, while
Proteo-AA's Stage III is co-evolutionary training — the benchmark is agnostic to
which, it only needs a checkpoint the `coevolution` config bundle can load.

## Known blocker: `--sidechain-cycle` crashes (pre-existing, not benchmark-side)

Sampling with the side-chain cycle live dies inside the model:

```
File "pxdesign_train/model.py", line 557, in _a_token_forward_hook
    fused = self.a_token_fusion(out, a_sc)
AttributeError: 'ProtenixDesignTrain' object has no attribute 'a_token_fusion'.
Did you mean: 'a_token_fusion_pre'?
```

Reading the source, the three pieces line up:

* `configs_train.py` defaults to `a_direct=False`, `a_direct_pre=True`;
* `model.py:428-434` builds `a_token_fusion` **only** when `sc_a_direct` is
  true, and `a_token_fusion_pre` only when `sc_a_direct_pre` is;
* `model.py:518-523` registers the `layernorm_a` forward hook when *either*
  `sc_a_direct` is true **or** the AA head reads `diffusion_internal` — which
  Stage III always does, since it needs that hook to cache `a_token`;
* the hook's fusion branch (`model.py:~556`) is guarded only by
  `_a_direct_active` and a populated `_a_sc_cache`, with no `sc_a_direct` check.

So on the default arm, as soon as the refinement pass runs with a side-chain
summary in hand, the hook reaches for a module that was never constructed. The
existing `tests/test_a_direct_pre.py` cannot catch it: it drives the pre-hook
against a stub that sets `sc_a_direct_pre` and `a_token_fusion_pre` only, and
never builds a real `ProtenixDesignTrain` on the default arm.

This is upstream of the benchmark and is left unfixed here — it is model /
ablation-semantics code, and the guard belongs wherever the intended arm
semantics are decided. The apparent one-line fix is to gate the branch on the
switch that owns it:

```python
if getattr(self, "sc_a_direct", False) and getattr(self, "_a_direct_active", False):
```

Until then `--sidechain-cycle` is off by default in the SLURM wrapper. **The
benchmark does not depend on it**: AF2-IG refolds each design from its sequence,
so designability is a function of the binder sequence and backbone; the
side-chain cycle only enriches the written PDBs with S_phi's full-atom output.
Every number this harness produces is unaffected.
