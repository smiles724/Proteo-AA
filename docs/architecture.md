# Architecture

Two published models, each used whole, with a tensor boundary between them.

```
                        PXDesign input JSON (unchanged)
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │ BACKBONE MODULE — official PXDesign                   │
        │   pxdesign.runner.inference.InferenceRunner           │
        │   official ProtenixDesign + pxdesign_v0.1.0 weights   │
        │   official featurizer, dataset and config             │
        └───────────────────────────┬───────────────────────────┘
                                    │ [N_sample, N_atom, 3]
                                    │ + atom_to_token_idx, atom names, res names
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ pxf/bridge.py — densify, in memory                    │
        │   ragged atom list → [N_sample, L, 37, 3] + mask       │
        │   design tokens (res_name == "xpb") → identity unknown │
        └───────────────────────────┬───────────────────────────┘
                                    │
                        sequence ───┤  native where known,
                        (an INPUT)  │  supplied for design tokens
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │ SIDE-CHAIN MODULE — official FaMPNN, packing mode     │
        │   SeqDenoiser.sidechain_pack, fampnn_0_0 weights      │
        │   aatype_override_mask = all → sequence is GIVEN      │
        │   scn_override_mask   = none → side chains INFERRED   │
        └───────────────────────────┬───────────────────────────┘
                                    │ [L, 37, 3] + psce [L, 33]
                                    ▼
                    PDB via FaMPNN's own writer (psce in B-factor)
```

## Why the boundary is a no-op

PXDesign/Protenix and FaMPNN both use AlphaFold2's atom37 layout **in the same
order**, and the same residue order (`ARNDCQEGHILKMFPSTWYV`). So the handoff is a
densification, not a translation — no permutation, no PDB round-trip. That is
convenient but fragile if assumed, so `pxf/atom37.py` pins it and
`assert_upstream_mapping()` fails at import if either upstream ever renumbered.
FaMPNN's `non_bb_idxs` is checked to be exactly the complement of the backbone
slots `(0, 1, 2, 4)` = `N, CA, C, O`, which is what makes "keep the backbone,
replace the side chains" well defined.

## Packing, not co-design

FaMPNN can co-design sequence and side chains. This pipeline does not use that.
Every position's identity is handed to the model through `aatype_override_mask`,
leaving it only the side-chain conformations. Two consequences:

1. **The sequence must come from somewhere explicit.** Non-design residues use
   the native identity parsed from PXDesign's own input. Design tokens (`xpb`)
   have none, so a sequence must be supplied via `--sequence`/`--sequence-fasta`.
   If one is missing the pipeline stops — filling it in would be co-design.
2. **The invariant is enforced, not trusted.** `sidechain_pack` echoes the aatype
   it used; that echo is compared against the input and any drift is refused.

## Ownership of coordinates

| Atoms | Owner |
|---|---|
| `N, CA, C, O` (slots 0, 1, 2, 4) | PXDesign — restored from its output after packing |
| the other 33 slots | FaMPNN |

FaMPNN is a packer and does not move the backbone; the pipeline restores it
regardless and reports `backbone_shift_angstrom`, which is `0.0` in practice.

## Packing is a sampler

Repacking one backbone twice gives different rotamers — around 0.5 Å RMS apart —
which is why generating several packings per backbone is useful. `seed` makes a
run reproducible; bitwise equality holds on CPU or with
`torch.use_deterministic_algorithms(True)`, while CUDA's default kernels leave
about 1e-5 A of jitter. `batch_size` changes how noise is drawn, so
reproducing a run means fixing both; both are recorded in the manifest.

## Devices

`torch.cuda.is_available()` only means a driver and device exist, not that the
installed torch has kernels for that device. On a mixed-GPU cluster those differ:
a build compiled through `sm_90` calls an `sm_100` card available and then fails
on the first kernel launch. `pxf/device.py` settles it by launching a real kernel
and falls back to CPU with an explanation. An explicit `--device` is honoured as
given, so you get the real error rather than a silent CPU run.

## Provenance

`pxf/provenance.py` pins all three submodule revisions and digests every weight
file. The only tolerated source change is the checked-in PXDesign embedders patch
(adapting `InputFeatureEmbedder` to the Protenix 2.0 `atom_attention_encoder`
signature); FaMPNN must be pristine. Every run writes a `manifest.json` carrying
those revisions and SHA-256s alongside the designs.
