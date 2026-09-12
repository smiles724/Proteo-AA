# Native SC rigid augmentation and incomplete backbone context

The new launcher is `scripts/training/slurm_sc_rigid_warmup_hai.sh`. It uses
the isolated `Proteo-AA-sc-rigid-augmentation` worktree; the source and job
114932 in `Proteo-AA-official-pxdesign-fampnn` are left untouched.

The run starts SC from scratch on monomers, with official PXDesign v0.1.0
(including its condition encoder) and pretrained FAMPNN frozen. It preserves
the existing SC architecture and warm-up settings: native identities/frames,
one-step packing, `edm=false`, `centre_coord_input=true`,
`frame_aware_head=false`, `template_residual=false`, `a_bs_concat=true`,
`q_bs=false`, crop 384, SC LR 5e-5, 2,000 warm-up steps, accumulation 8,
checkpoint every 500 steps, validation every 2,000 steps, maximum 50,000 steps.
No revision, backbone refinement or feedback is enabled.

## Coordinate contract

`--stage4-native-sc-augmentation` enables training-only augmentation in
`sc_warmup` and `sc_complex_adapt`. Before condition embedding and the native
backbone feature pass, sample one uniform SO(3) rotation using a normalized
Gaussian quaternion, center on observed backbone/context atoms, and add an
isotropic translation of standard deviation 1 angstrom. The whole example
uses the affine transform `x' = R x + t`:

- Transform label coordinates, N/CA/C/O coordinates, frame origins and fixed
  target coordinates together.
- Transform each local-to-global frame as `frame_R' = R @ frame_R`.
- Keep local SC targets and reference-space `ref_pos` unchanged. SC global
  initialization/targets are reconstructed through the transformed frames.
- Keep missing coordinate placeholders inactive and zero; preserve chemical,
  observation and frame masks. Do not mutate the caller's tensors.

Torch RNG is already persisted by integrated checkpoints. The effective
configuration records the opt-in; older checkpoints retain augmentation off.
Training reports `sc_rotation_augmented=1`; normal validation reports 0.
Augmentation improves orientation coverage; it does not impose architectural
rotation equivariance or establish a quality improvement.

## Mask contract

Commit 6d84def already passed per-atom N/CA/C/O observation masks to SC
attention. This change also excludes unobserved native atoms/representative
centers from physical context, masks B-to-S q features, and assigns index -1
to invalid S-to-B q-feedback slots. Generated packing removes native
observation metadata; generated coordinates do not inherit experimental
missing-atom masks. O absence does not invalidate an otherwise valid N/CA/C
frame.

## Validation

CPU regression job 114963: **656 tests passed**. Tests cover a shared rigid
transform of coordinates, frames, context and reconstructed SC targets;
local-loss invariance; deterministic RNG replay; masked NaNs; and incomplete
backbone/context examples.

Data audit job 114965 sampled 256 distinct crops from the same 47,622-row
filtered training index as job 114932, uniformly without replacement (seed
83; normal crop/retry behavior retained). All 256 succeeded:

| Count | Result |
| --- | ---: |
| Canonical design residues | 56,933 |
| Valid native frames | 53,973 |
| Residues missing observed O | 2,957 (5.1938%) |
| Missing O among valid N/CA/C frames | 0 / 53,973 |
| Missing O atom rows | 0 |
| Present but unobserved O rows | 2,957 |

This is an estimate from sampled crops, not a full-dataset or cluster-weighted
census. No O-only defect appeared in the valid-frame subset; synthetic tests
still cover that condition. Raw record: `runs/sc_input_audit/114965.json`.

GPU smoke job 114964 exercises rigid augmentation, a full-model incomplete-O
case, hidden-label and invalid-frame isolation, finite backward, frozen
pretrained tensor hashes, validation, save/resume and exact reconstruction.
It also measures rotation consistency on one real monomer and three rotations
with matched RNG, after two SC updates. These measurements are a diagnostic
of orientation sensitivity, not evidence of improved packing quality.
