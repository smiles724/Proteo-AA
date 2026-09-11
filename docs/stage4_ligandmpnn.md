# Stage IV: the LigandMPNN sequence backend

Worktree `/hai/users/s/h/shenjm/Proteo-AA-ligandmpnn`, branch
`sjm/stage4-ligandmpnn`, based on `stage4-fampnn` at `17120fa`. Written
11 September 2026.

**Status: integration, unverified as science.** The backend loads released
weights, resolves configuration, passes 612 tests including exact parity with
upstream's own decoder, and clears a bounded real-model GPU smoke (HAI job
**114335**). No training result, no FaMPNN comparison and no binding-quality
claim exists. Read [Open problems](#open-problems) before reporting any number
from this path.

This is the sibling of [`stage4_fampnn.md`](stage4_fampnn.md), which owns the
shared machinery: the co-design cycle, the phases, the data sources, the
validation loaders. Only what differs is described here.

## Why a second backend

FaMPNN reads the fixed receptor through Atom37. It has no channel for anything
that is not a protein residue, which `stage4_fampnn.md` states plainly in its
last line: ligands and metals "are not encoded as generic entities by FaMPNN's
protein Atom37 interface". LigandMPNN has exactly that channel.

Note what this does **not** mean. A protein binder against a protein target is
not "the ligand case": in LigandMPNN the target is a second chain in the same
backbone graph, distinguished by `chain_labels` (`model_utils.py:1240-1242`
encodes same-chain vs cross-chain into the edge features), which plain
ProteinMPNN already handles. The ligand channel carries atoms that are not in
the residue graph — small molecules, metals, glycans, and (see below) fixed
side chains.

## The interface the head has to satisfy

`codesign.decode()` calls the head once per block and reads the logits of the
positions it is about to commit:

```python
logits, _ = head(**state.aa_input(sc_feedback=cfg.sc_to_aa))
aa = assign_aa(logits, cfg.temperature, generator)
state = state.commit(aa, selected)
```

So the head must return, from **one** forward, `p(s_i | visible)` for every
queried position, with the queried positions mutually invisible. `aa_input()`
supplies `denoised_coords [B,S,L,37,3]`, `aatype_noised` (20 = X), `seq_mask`,
`atom_mask_noised`, `residue_index`, `chain_encoding`; the head returns
`(logits[..., 20], features)` in Proteo-AA's canonical amino-acid order.

FaMPNN satisfies this natively — it is a masked denoiser, `fampnn_head.py:33`
rejects the autoregressive configuration, and a hidden position is literally X.

LigandMPNN does not, and the reason is worth stating precisely, because the
usual summary ("it is autoregressive, so block decoding is off-distribution")
is **wrong**. ProteinMPNN and LigandMPNN train with a *uniformly random
decoding order*, so predicting a position from an arbitrary subset of the
others is their native task; conditional decoding is not a hack here. The
problem is mechanical: `order_mask_backward[q, p] = 1` iff p is decoded before
q, and such a p contributes `W_s(S_true[p])` — p's **true residue**.

No official entry point produces the mask this loop needs:

| upstream call | what it gives | why it does not fit |
| --- | --- | --- |
| `score(use_sequence=True)` | order derived from `chain_mask` | designed positions see each other's native residue; the AA cross-entropy reads its own label |
| `score(use_sequence=False)` | no sequence context at all | discards the blocks the cycle just committed |
| `single_aa_score` | one position per forward | L encoder passes per block |

**The risk is label leakage, not distribution shift.** A leak makes recovery
look good and mean nothing, and nothing raises.

## What the head does

`pxdesign_train/aa/ligandmpnn_head.py`. Featurisation, encoder and decoder
layers are upstream's, called on upstream's feature dict; nothing is
reimplemented. `_decode_against_visible` is `ProteinMPNN.score()`'s decoder
body with one substitution:

> upstream's permutation-derived `order_mask_backward` is kept for the
> **visible** rows and replaced, on the **queried** rows only, by the visible
> set itself.

For a queried q that is the mask of the decoding order "all visible first,
then q" — a real order, inside the training distribution. What it is not is a
permutation: every queried position gets that same predecessor set, which is
what makes one forward enough and what keeps queried positions blind to each
other.

**Leaving the visible rows alone is load-bearing.** Overriding every row —
each node attending to all visible, which reads as the obviously equivalent
simplification — moves the queried logits by ~1.4e-2 on the released
architecture. The decoder is three rounds of message passing, so a queried
node reads its neighbours' representations, and those are computed under their
own predecessor sets; giving the context nodes a fuller view than any
permutation could produce is off-distribution. This was found by the parity
test, not by reading the code, which is the argument for having written it
first.

The visible prefix order comes from `decoding_randn`, defaulting to a constant
so `argsort` is stable and the order is residue order: reproducible across the
blocks of one decode and across the feedback arms of one seed. The parity test
passes upstream's own draw so both sides order the prefix identically.

### Mapping

| | upstream | Proteo-AA | handling |
| --- | --- | --- | --- |
| alphabet | `ACDEFGHIKLMNPQRSTVWYX` | `ARNDCQEGHILKMFPSTWYV` + X | remapped both ways, X pinned at 20 |
| Atom37 | `N CA C CB O CG …` | identical | no permutation needed |
| slot 36 | never written by their PDB reader (36 names) | `OXT` | their model types slot 36 as oxygen (`side_chain_atom_types[31] == 8`), so ours is compatible and strictly more informative |

`test_alphabet_and_atom37_match_upstream_source` pins both against upstream's
source **text**, because `data_utils` cannot be imported: it pulls in `prody`
for PDB parsing this head never performs.

### One adaptation upstream forces

`ProteinFeaturesLigand` keeps `side_chain_atom_types` and
`periodic_table_features` as **plain attributes, not buffers**.
`nn.Module.to()` walks parameters and buffers only, so both stay on the CPU
when the model moves to a GPU, and featurisation dies inside `torch.cat`.
Upstream never hits this: `run.py` constructs the model with `device=` already
set, while Proteo-AA builds the whole model and then moves it.

`_rehome_constant_tensors` re-registers them as **non-persistent** buffers —
they follow the device but stay out of `state_dict()`, because they are fixed
tables and extra keys would change the checkpoint shape and break the strict
load on resume.

No CPU test could have caught this; the GPU smoke did, on its first run. It is
now pinned by `test_constant_tables_follow_the_module_to_a_device`.

### Refusals

* `augment_eps != 0` is rejected. It perturbs backbone and ligand coordinates
  inside featurisation (`ProteinFeaturesLigand.forward`); under a
  coordinate-gradient objective that is noise injected between the backbone
  and its own loss, and it breaks same-seed arm comparisons. `run.py` sets 0.0
  for inference; a training head must too.
* Revision mismatch, a dirty upstream tree, or a state-dict that does not load
  strictly. `num_edges` and `atom_context_num` are read from the checkpoint
  exactly as `run.py` reads them, never from a config.

## Phases, including the new one

`stage4.py` gained **`IV-F`: freeze the sequence head, train the generator
against it.** No existing phase expressed this — IV-0 freezes everything and
dies in the trainer on `No trainable parameters` (`trainer.py:398`), IV-A is
its mirror image, and IV-B/IV-C train the head too.

| phase | `aa_head` | packer / feedback / bb subset | structural losses |
| --- | --- | --- | --- |
| IV-0 | frozen | frozen | (not a trainable phase) |
| IV-A | **trained** | frozen | zeroed |
| **IV-F** | frozen | **trained** | **kept** |
| IV-B / IV-C | trained | trained | kept |

Two things about IV-F are not obvious and are therefore asserted, not assumed:

1. **`requires_grad=False` is not `no_grad`.** The AA cross-entropy still
   reaches the backbone *through* the frozen head, via its coordinate inputs.
   That route is the entire objective — train backbones that a fixed,
   well-trained designer reads as native, in the spirit of MPNN-in-the-loop.
   If freezing ever severed it, the run would still train (the packer has its
   own losses) and the backbone would silently receive no sequence signal.
   `test_frozen_head_still_passes_gradient` and
   `test_coordinate_gradient_reaches_the_backbone` cover the two halves.
2. **IV-A's structural-loss zeroing must not apply.** In IV-A only the head
   moves, so `weight_mse`/`lddt`/`disto`/`bb_post` can only add noise to a
   gradient that cannot reach the structure. In IV-F the backbone is what is
   being trained, and an AA cross-entropy through a frozen head is not a
   geometry objective — dropping the structural losses would let the backbone
   chase the designer's opinion off physical structure. `build_configs` raises
   if every structural weight is zero under IV-F.

`--stage4-bb-trainable-prefixes` is newly reachable from the CLI; IV-F needs
it, since with the head frozen those are the only weights that move.

## One predicate, not twenty checks

`aa_backend == "fampnn"` appeared eleven times across `model.py`,
`cogenerate.py` and `trainer.py`, gating optimizer groups, phase
re-application, checkpoint identity and donor validation.
`training_stage == "stage4_fampnn"` appeared nine times in the launcher,
gating strict binder featurisation, cluster-disjoint PPI sampling, the binder
validation loaders and the provenance dump.

A second backend added by editing ten of eleven and eight of nine trains like
a Stage III run and reports nothing wrong. Both collapsed to one place:

* `pxdesign_train.aa.uses_codesign(model)` / `CODESIGN_BACKENDS`
* `train_protenix_monomer.STAGE4_STAGES` / `_is_stage4(args)`

`test_stage_four_backends_share_one_predicate` asserts that no site goes back
to testing a name directly.

## Where things live

Home (`/hai/users/s/h/shenjm`) is a **50 GiB** volume, already 54% full;
scratch (`/hai/scratch/shenjm`) has 5 TiB. Anything that grows goes to
scratch, matching the convention the rest of the project already follows
(`pxdesign_tool_weights` is 5.3 GiB there).

| | path | size |
| --- | --- | --- |
| released weights | `/hai/scratch/shenjm/ligandmpnn_weights/` | 21 MiB |
| run outputs, checkpoints, PINDER CIF cache | `/hai/scratch/shenjm/proteo_aa_runs/stage4_ligandmpnn_*/` | **1.9 GiB per checkpoint** |
| Triton compile cache | `/hai/scratch/shenjm/triton_cache` | — |
| upstream LigandMPNN source | `/hai/users/s/h/shenjm/tools/LigandMPNN` | 21 MiB |
| this worktree | `/hai/users/s/h/shenjm/Proteo-AA-ligandmpnn` | 12 MiB |
| Slurm logs | `<worktree>/logs/training/stage4_ligandmpnn/` | ~2 MiB per 24 h job |

A checkpoint is **1.9 GiB**, so at `CHECKPOINT_INTERVAL=500` and ~15k steps in
a 24-hour slot a single run writes roughly 57 GiB. That does not fit in home
at all, which is why `OUTPUT_DIR` defaults to scratch and why
`TRITON_CACHE_DIR` is overridden -- its default is `~/.triton`.

**PINDER is read from yfsun's tree but extracted into ours.** The manifest
and the 168 GiB `pdbs.zip` are readable and are used in place; `--pinder-root`
points at `/hai/scratch/shenjm/pinder/2024-02` instead, because his `pdbs/` is
both partly unreadable and not writable by us. Structures are materialised
from the archive at first use, ~200 KiB each, so a 30k-step run adds roughly
6 GiB. The launcher preflights all three: archive readable, manifest readable,
root writable.

Two things stay in home deliberately. The worktree is source only; its
`Protenix` and `PXDesign` submodules are symlinks to the main checkout rather
than second copies. The upstream LigandMPNN clone stays next to
`PXDesignBench` because the head verifies its revision and that its tree is
clean, so it is a provenance input, and scratch is the volume with a purge
policy. Its 21 MiB is mostly `outputs/` and `inputs/` example data shipped in
the repo -- do **not** delete those to save space, the clean-tree check
(`git status --porcelain`) would then refuse to build the head.

## Running it

Upstream checkout `~/tools/LigandMPNN` at `26ec57ac`; released weights in
`/hai/scratch/shenjm/ligandmpnn_weights/` (`ligandmpnn_v_32_010_25.pt`, SHA-256
`161cd264…`; the `_005_` noise variant is also present). The checkpoint
carries `atom_context_num=25` and `k_neighbors=32`.

```bash
cd /hai/users/s/h/shenjm/Proteo-AA-ligandmpnn
mkdir -p logs/training/stage4_ligandmpnn
bash scripts/training/slurm_stage4_ligandmpnn_binder_hai.sh --dry-run   # login node
sbatch scripts/training/slurm_stage4_ligandmpnn_binder_hai.sh
STAGE4_PHASE=IV-F CROP_SIZE=256 sbatch scripts/training/slurm_stage4_ligandmpnn_binder_hai.sh
```

One file, not the base/wrapper pair the FaMPNN launchers use: those are
Marlowe-native with a HAI wrapper on top, and nothing here runs on Marlowe.
The donor is the same Stage III checkpoint (`111408/step6000`) the FaMPNN runs
use, so the two backends differ in the sequence network and nothing else.

### Resuming

The launcher prefers a checkpoint in `$OUTPUT_DIR/checkpoints/` over the
donor, picking the highest step number (numerically — `step150` must not beat
`step1000`, and mtime is the wrong key because an interrupted save is newer).
Only if none exists does it warm-start from the Stage III donor with
`--warm-start-params-only`.

This matters because the jobs are `--requeue` and a requeued job keeps its job
ID, so `OUTPUT_DIR` — which embeds it — still holds everything written before
the preemption. The launcher used to hardcode the donor path, so a restart
silently began again at step 0.

A full resume restores step, optimizer, scheduler and RNG. **It is refused if
the Stage IV identity moved**, and `implementation_identity()` hashes git HEAD
plus every `pxdesign_train/**/*.py`, so *any* commit trips it — including one
that touched nothing the run reads. The refusal is deliberate and is left
strict here; what changed is that it now names the differing fields, because
"identity differs" alone cannot tell a changed objective from an edited
docstring, and the two call for opposite responses. Loud refusal is also the
right failure: silently warm-starting instead would discard the run without
saying so.

Practical consequence, and it is sharp: **editing the repo while runs are in
flight makes their checkpoints unresumable.** The checkpoints written by
114341/114342/114345 before commit `b4d5708` already cannot be fully resumed.
Either freeze the branch while runs are live, or accept that a preemption
costs the run rather than an interval. Narrowing the gate to the fields that
actually affect correctness (backend, mapping, cycle, phase, optimizer policy
— all separately checked a few lines below) is the obvious fix and is
yfsun's call, since he set the strictness deliberately.

### Failure modes, all of them observed

Two carried over from `stage4_fampnn.md`:

* **`sbatch` from inside an interactive allocation** silently inherits the
  shell's CPU and memory request over the script's own `#SBATCH`. Strip
  `SLURM_*` (keeping `SLURM_CONF`) before submitting and confirm with
  `scontrol show job <id> | grep ReqTRES` — it must read
  `cpu=8,mem=192G,gres/gpu:h200=1`.
* **`CHECKPOINT_INTERVAL` stays below `EVAL_INTERVAL`**, because validation
  runs before the checkpoint save. Jobs 113677 and 113714 both died before
  their first checkpoint and lost everything.

Four learned here. Every one of them is the same shape — **something that can
be found or stat'ed, but not used, was treated as usable** — and the first is
by far the most dangerous, because it is the only one that fails quietly.

* **`PROTEOAA_REPO` must be resolved, never hardcoded.** The default pointed
  at the main checkout while this script lives in a worktree, so a plain
  `sbatch` `cd`'d into the *other* tree and ran its code with this tree's
  arguments. Here it failed loudly — the main checkout has no
  `stage4_ligandmpnn` stage, so argparse rejected it. Two branches that merely
  drifted would produce wrong numbers with nothing to say so.
* **`BASH_SOURCE` alone does not fix that.** `sbatch` COPIES the script to
  `/var/lib/slurm/slurmd/job<ID>/slurm_script`, so under Slurm it resolves to
  a slurmd-owned directory and the job dies in a second on `mkdir: cannot
  create directory 'logs': Permission denied` (jobs 114336, 114337). The
  launcher now tries `SLURM_SUBMIT_DIR` first, then `BASH_SOURCE`, accepts a
  candidate only if it contains `pxdesign_train/aa/ligandmpnn_head.py`, and
  exits 2 rather than guessing.
* **A third of the shared PINDER tree is unreadable.** yfsun's runs extracted
  `/hai/scratch/yfsun/pinder/2024-02/pdbs` under a restrictive umask, leaving
  ~35% at mode 600. `_ensure_cif` selected candidates with `Path.is_file()`,
  which only stats, so an unopenable file passed, the archive fallback sitting
  right below it was skipped, and the run died later inside `pdb_to_cif` with
  `PermissionError` in a DataLoader worker (jobs 114339, 114340). All three
  decision points now use `_is_readable_file`. See
  [Where things live](#where-things-live) for the writable-root half of the
  fix.
* **The worktree must be on shared storage.** Compute nodes cannot see
  `/tmp`, which is node-local; a job submitted against a `/tmp` path fails
  instantly at `cd`. This worktree lives under `/hai/users/s/h/shenjm/`.

The smoke is backend-agnostic:

```bash
python scripts/utilities/smoke_stage4_fampnn.py --backend ligandmpnn \
  --ligandmpnn-checkpoint /hai/scratch/shenjm/ligandmpnn_weights/ligandmpnn_v_32_010_25.pt \
  --ligandmpnn-source ~/tools/LigandMPNN \
  --donor /hai/scratch/yfsun/proteo_aa_runs/stage3_binder_coevolution/111408/checkpoints/step6000.pt \
  --data-root /hai/scratch/yfsun --output runs/lmpnn-smoke
```

## Verified

Released weights load with hyper-parameters read from the checkpoint. The dry
run resolves 47,622 monomer and 1,178,236 eligible PINDER rows at a 0.25/0.75
mixture and adopts the donor's S_phi layout. IV-A/IV-F/IV-B each resolve to
the intended loss weights (IV-A zeroes `weight_mse`, IV-F and IV-B keep it);
the missing-checkpoint and missing-source guards fire.

Head tests run on a randomly initialised network and need no released weights,
because masking, parity and gradient flow are properties of the decode path:

* exact parity (`rtol=0, atol=0`) with `score(use_sequence=True)` on the
  single-query case
* queried logits do not move when native residues are planted under the mask,
  driving `_decode_against_visible` directly rather than through
  `visible_input`'s sanitisation
* non-zero coordinate gradient reaching N/CA/C/O
* alphabet and Atom37 pinned against upstream source text
* `augment_eps != 0` rejected
* upstream's two constant tables are re-registered as non-persistent buffers
  so they follow `.to(device)` — see below

612 tests pass.

**HAI job 114335** ran the real model on one 48-token PINDER complex
(`1aw8__A1_P0A790--1aw8__C1_P0A790`, 101 observed binder side-chain atoms
across 22 residues): donor load `unexpected=0`, all loss components finite,
the trained parameter moved and a sampled frozen one stayed bit-identical
across the optimizer step, all three feedback routes reached the packer
(`aa_to_sc` 2.7e4, `bb_to_sc` 3.1e5, `sc_aux_to_sc` 1.5e5), save/resume
`missing=0, unexpected=0`, and a three-round generation exported to mmCIF.

Its step-1 readings, for orientation only and not a measurement: `aa_pre`
2.77 against ln 20 = 3.00, `aa_revision` 3.27, `recovery_pre` 20.8%,
`recovery_revision` 12.0%, `sc_aux` 4.32, `phys` 6e-4. One item at step 1 with
an untrained cycle says nothing about either backend.

These are bounded engineering checks; none of them measures design quality.

## Runs in flight

Submitted 11 September 2026, 23:50 slots, same donor as the FaMPNN runs.

| job | phase | crop | what it tests |
| --- | --- | --- | --- |
| **114341** | IV-A | 384 | trains the LigandMPNN head only. Matched to yfsun's FaMPNN IV-A: same donor, same data, same mixture, so the two are directly comparable. Do not change this crop — the match is the point |
| **114342** | IV-F | 256 | trains packer + atom-attention decoder against the FROZEN head. The conservative control |
| **114345** | IV-F | 384 | the same, at IV-A's crop |

All three reached training.

**Crop 256 was over-cautious and costs nearly half the data.**
`max_binder_tokens = crop x 0.75` is a hard filter, so at 256 only 664,664 of
1,219,793 cluster-disjoint PINDER rows are eligible (54.5%), and the excluded
ones are systematically the larger binders — median binder tokens among the
survivors drops from 175 to 104. At 384 it is 96.6%, at 448 it is 100%.
Measured GPU use says there was never a reason to pay that: IV-A at 384 sits
at 38.0 GiB, IV-F at 256 at 47.2 GiB, IV-F at 384 at 78.8 GiB — all against
143.8 GiB on an H200. 114345 exists to retire 114342.

IV-A's step-50 readings, **for orientation only** --
one job, fifty steps, an untrained cycle: `stage4/aa_pre` 2.83–3.48 against
ln 20 = 3.00, `recovery_pre` 6.9–17%, `loss_bb` 0 as IV-A intends. One log
line per accumulation micro-batch, not per step. Nothing here is a
measurement and nothing should be compared to FaMPNN yet.

Per-step logs are the record; this table is only the index. They live in
`<worktree>/logs/training/stage4_ligandmpnn/<name>-<jobid>.err`, one line per
accumulation micro-batch, so roughly 150k lines per job per day -- grep them,
do not read them. Job-level events (FAILED, REQUEUED, preemption) are not in
there at all and come from `sacct -j <id>`. Both survive any session, so
there is nothing to carry forward by hand beyond this table.

Four earlier submissions died and are worth keeping straight, because each
one is a distinct trap now covered above: 114336/114337 on the sbatch script
copy, 114339/114340 on unreadable PINDER structures.

## Open problems

**The ligand channel is empty.** This is the big one: it is the reason to
prefer this backend, and it is not wired. `CoDesignState.aa_input()` returns a
protein-only Atom37 view, so the head passes an all-masked `Y/Y_t/Y_m`. That
is not a silent degradation of the protein path — upstream concatenates
side-chain atoms in front and keeps the `atom_context_num` closest to Cb,
pushing masked entries to distance 10000 (`Cb_Y_distances_adjusted`), so empty
slots lose every contest and are masked downstream regardless. But no ligand,
metal or glycan reaches the AA decision.

Wiring it needs: `aa_input()` extended to project non-protein atoms out of
`fixed_atom_xyz`/`fixed_atom_mask` (which already carry them — see the
`CoDesignState` docstring) with elements from `structure_element`
(`featurizer.py:497`); `FaMPNNHead.forward` given `**_unused` so the new keys
do not break it. The head's `ligand_xyz`/`ligand_element`/`ligand_mask` kwargs
are already the entry point. This is the one part of the plan that must touch
`codesign.py`.

**Fixed side chains, by contrast, ARE wired**, through
`ligand_mpnn_use_side_chain_context=True` and `chain_mask`
(`model_utils.py:1252` gates them to non-designed positions). So the two
backends see the same information today and a head-to-head comparison is
fair. Turning the flag off makes LigandMPNN see strictly less; the launcher
logs a warning.

**Two decisions are required before the ligand channel is worth building.**

*Crystallisation artifacts must be filtered.* Of the ten AlphaProteo-10
targets, eight carry non-protein entities:

| kind | entities | targets |
| --- | --- | --- |
| glycans | NAG, MAN, BMA, FUC | il7ra, il17a, ir, h1, sc2rbd |
| structural metal | ZN | sc2rbd (6m0j, ACE2) |
| bound ligand / non-canonical peptide | 9KK, CCS, MEA, SAR, NH2 (all chain 2) | pdl1 (5o45) |
| crystallisation & cryo agents | SO4, GOL, PEG, PGF, BR, CL | vegfa, bhrf1, h1, sc2rbd |
| none | — | tnfa, trka |

The last row of that table is the problem: those do not exist in solution, and
feeding them as context teaches the model to design around things that are not
there. An entity allowlist is needed, not "every non-protein atom".

*Adding ligands breaks the matched comparison.* AlphaProteo's and PXDesign's
protocols almost certainly did not include glycans. Letting Proteo-AA see them
while the baseline does not means that column can no longer be compared to
their table. Either both arms get them, or it is a separate arm.

**The benchmark and training paths disagree about non-protein atoms today.**
Training keeps them: `cif_provider.py:159` uses `mol_type == "protein"` only
to pick the smallest protein chain as binder, so non-protein atoms flow into
`fixed_atom_mask` (`featurizer.py:488`) and are already in the state. The
benchmark drops them: `design_binder_from_target.py:138` filters target chain
selection to `mol_type == "protein"`, so glycans and the zinc never enter the
structure at all. **Fix this inconsistency before wiring the channel**, or the
model will train with context it cannot have at evaluation.

**`decode_blocks` is the cheap experiment nobody has run.** It interpolates
between one-shot marginals (`1`) and fully autoregressive decoding
(`= |query|`, one position per block along a random order); the default is 4.
Raising it costs only extra forwards, and it directly answers whether
finer-grained sequential conditioning matters here — which is the evidence
needed before building an efficient native autoregressive path (encoder once,
decoder stepped per position, `~N` sequential steps in the autograd graph)
that would require changing `codesign.decode()`. The sweep applies to FaMPNN
too, and nobody has measured either.

**Frozen LigandMPNN recovery may read low for interface reasons.** It is being
driven through a mask no official entry point produces. The parity test bounds
that risk on the single-query case but says nothing about many-query blocks.
Do not read a low number as "LigandMPNN is worse than FaMPNN" without first
checking `decode_blocks`.

**Resume identity is as brittle here as for FaMPNN.**
`implementation_identity()` hashes git HEAD plus every
`pxdesign_train/**/*.py`, and `load_checkpoint` raises on any mismatch when
`params_only=False`. Any code edit makes every earlier checkpoint
non-resumable with optimizer state. Unchanged by this branch and worth
narrowing to the fields that actually affect correctness.

**IV-F and IV-B are not memory-proven at crop 384.** IV-A retains no autograd
graph through the frozen generator, which is why 384 fits. IV-F opens the
packer and the backbone subset. Start smaller and watch.
