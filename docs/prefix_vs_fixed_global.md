# Pre-fix global vs. fixed global

Two Stage II side-chain warm-up runs with **identical head parameterisation**,
separated by the single-frame refactor (`dae31c0`). Everything that differs is
in how coordinates reach S_φ and which mask drives its forward pass.

| | pre-fix global | fixed global |
|---|---|---|
| Job | 98133 | 98587 |
| Run dir | `global_head_from_bb_step96000` | `fixed_global_from_bb_step96000` |
| Started | 2026-08-10 16:36 | 2026-08-11 18:40 |
| Wall clock | 21:42:36 | 21:11:30 |
| Steps | 50 000 | 50 000 |
| Best val `sc_local` | **2.447 Å²** (1.564 Å) @ 50 k | **1.854 Å²** (1.362 Å) @ 48 k |
| Warm start | `bb_100k_val/step96000.pt` | same |

Both: 491 recent-PDB monomers, crop 640, lr 5e-5, `weight_sc_local=1.0`, every
other loss term zero, physical loss off, Dunbrack `template_init` on, GT frames
(`predicted_frame=False`), teacher-forced residue type.

## What is identical

The thing people assume is being compared here is **not** what differs:

```
frame_aware_head  = False    # head emits absolute local coords, anchored at CA
template_residual = False    # the head's output is NOT added to the template
bb_context        = True     # 14-slot atom axis (N/CA/C/O + 10 side-chain)
a_bs_concat       = True
q_bs              = False
```

Both arms are "global" in the head sense, and both use the ideal-rotamer
template as their coordinate **input**. Neither adds it to the output.

## What differs

### 1. The 14-slot coordinate tensor mixed two frames

`SideChainModule` concatenates the four backbone context slots onto the ten
side-chain slots and reads all fourteen with **one** `nn.Linear(3, c_atom)`.

Pre-fix, the two halves were built in different frames:

```python
# model.py, pre-fix
bb_local = to_local(bb4, fR, ft)            # UNCONDITIONAL -> residue-local
noisy    = (noisy_init if local_coord_input # this arm: False
            else to_global(noisy_init, fR, ft))   # -> raw global
coords   = cat([bb_local, noisy])           # local ⊕ global
```

So a residue's N/CA/C/O arrived as displacements of ~0–4 Å while its side-chain
atoms arrived as absolute positions of tens to hundreds of Å. One weight matrix
had to read both. The backbone rows are one to two orders of magnitude smaller,
so they contribute almost nothing to the embedding: **`bb_context` was close to
dead in this arm**, and it is the prerequisite for the `q_direct` and `q_bs`
channels.

Fixed: every slot is global. `bb_coords` and `noisy_coords` are the same frame
by construction, and the module's docstring now states the contract.

### 2. `w_xyz` was reading absolute position, not shape

With `local_coord_input=False` the side-chain slots were raw global coordinates:

```
CA  = (247.3, -88.1, 412.0)     <- where this residue sits in the assembly
CB  = (248.1, -87.4, 411.6)     <- 1.5 Å from CA
NH1 = (252.0, -84.9, 409.3)     <- ~6 Å from CA
```

The quantity S_φ must model is the ~4 Å of side-chain geometry; roughly 98% of
each input number is the residue's absolute position. A single linear layer
cannot separate the two, so `w_xyz` largely encoded *where* rather than *what
shape*.

Fixed: `centre_coord_input=True` subtracts the residue CA **for the embedding
only**:

```python
# module.py:328-334
if self.centre_coord_input and ca_coords is not None:
    embed_coords = coords - ca_coords[:, :, None, :]
xyz_feat = self.w_xyz(embed_coords)
```

This is the part of the old `local_coord_input` that was doing useful work. The
part that was not — rotating into per-residue frames — is gone. A pure
translation cannot desynchronise from the coordinates used for cross-residue
distances, because those keep using the un-centred tensor.

### 3. Part of the cross-residue distance bias was garbage

`_CrossAtomBlock` biases attention by inter-atomic distance:

```python
d_atom = torch.cdist(atom_coords, k_xyz)   # module.py:129
```

computed over the same mixed-frame `coords`, plus a virtual atom at each context
residue's (global) CA. Which pairs survived depends on the arm, and the two
pre-fix arms were **not** equally damaged:

| pair | pre-fix **global** | pre-fix **local** (97826, for contrast) |
|---|---|---|
| side-chain ↔ side-chain, across residues | ✅ both global | ❌ each in its own frame |
| virtual CA ↔ side-chain | ✅ | ❌ global vs local |
| backbone slot ↔ anything | ❌ | ❌ |

Pre-fix global kept roughly 10 of 14 slots carrying real spatial information;
pre-fix local kept none. Fixed: one frame, so every pair is meaningful.

### 4. `sc_atom_mask` gated the forward pass as well as the loss

Pre-fix, one `[L, MAX_SC]` bool drove the attention key set, the output zeroing,
the `h_res'` pooling, the template init **and** the coordinate loss. After the
unresolved-atom fix that mask means *chemistry AND the crystallographer resolved
it*, so ~9.8% of side-chain atoms — disproportionately the flexible surface
residues — were removed from S_φ's attention keys, not just from supervision.

There is no `is_resolved` at inference, so the model trained on a key set no
inference input can reproduce.

Fixed: split into `sc_slot_mask` (chemistry only → drives the forward) and
`sc_atom_mask` (chemistry ∧ resolved ∧ plausible → drives the losses).

### 5. Minor: the implausible-target backstop

`MAX_SC_LOCAL_RADIUS_A = 12` (commit `b3c4588`) landed after 98133 started, so
only the fixed arm has it. It drops ~7 atoms of 155 385 in 1 of 150 sampled
structures — about 0.005% — and cannot account for anything below.

## Results

```
best val sc_local     2.447 Å²  ->  1.854 Å²      -24.2%
best RMSD             1.564 Å   ->  1.362 Å
```

The convergence-rate change is the more telling number:

```
fixed global reaches 2.741 Å² at step  2 000
pre-fix global reaches 2.741 Å² at step 20 000        10x
```

The pre-fix curve starts at 6.827 Å² — worse than the 4.664 Å² ideal-rotamer
template baseline — and is still descending at step 50 000, i.e. it never
converged. That is the shape of a model fighting its input rather than one
short of capacity or data.

`runs/figures/val_comparison_four_arms.png` plots both, alongside the local arms.

## A synthesis worth keeping

The two pre-fix arms had **complementary** defects, which explains an otherwise
odd result — pre-fix global started far worse yet finished ahead of pre-fix
local:

| | coordinate embedding | cross-residue geometry | best |
|---|---|---|---|
| pre-fix local | ✅ bounded, shape-like | ❌ entirely dead | 2.805 |
| pre-fix global | ❌ swamped by absolute position | ⚠️ mostly intact | 2.447 |
| fixed (both) | ✅ | ✅ | 1.918 / 1.854 |

Pre-fix local read its coordinates cleanly, so it learned fast and then hit a
ceiling at ~12 k with no spatial channel to go further. Pre-fix global could not
read its coordinates, so it learned slowly, but its spatial channel was largely
alive and it kept improving to step 50 k.

With both halves working, the two arms converge to nearly the same place. The
remaining 3.3% (1.918 vs 1.854) is the head parameterisation's actual
contribution — an order of magnitude smaller than what the frame handling was
costing.

## Caveats

**The attribution above is inferred from the code, not measured.** Four things
changed at once; no arm isolates any one of them. In particular:

- "pre-fix global kept ~10/14 slots intact" follows from reading the concat and
  the `cdist`, not from an ablation.
- The mask split's contribution is entirely unquantified.
- Switching the embedding from residue-local (pre-fix *local*) to CA-centred
  global arguably makes the representation slightly *less* canonical — a CB is
  no longer always in the same place — so some of the fixed arms' gain may be
  offsetting a small loss elsewhere.

The minimum experiment to settle it is two arms of ~20 000 steps each: one that
only fixes the backbone-slot frame, one that only adds centring. They are cheap
and can run in parallel.

**Nothing here is a template ablation.** All four arms initialise from the
Dunbrack modal rotamer. A genuinely template-free control needs
`--disable-template-init`, which has not been run.

## Pointers

| | |
|---|---|
| Refactor commit | `dae31c0` "Give S_phi one coordinate frame, real packing supervision, and split masks" |
| Backstop commit | `b3c4588` |
| Coordinate contract | `pxdesign_train/sidechain/module.py`, `SideChainModule.forward` docstring |
| Centring | `module.py:326-334` |
| Frame guard | `pxdesign_train/runner/trainer.py`, `SIDECHAIN_LAYOUT_KEYS` — refuses a pre-refactor checkpoint outright |
| Equivariance tests | `tests/test_sidechain_module.py::test_centred_input_makes_the_module_translation_equivariant` and its uncentred control |
| Curves | `runs/metrics/sidechain_arm_{global_head,fixed_global}_val.csv` |
| Figure | `runs/figures/val_comparison_four_arms.png` |
