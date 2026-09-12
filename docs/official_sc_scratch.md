# Scratch SC training with frozen pretrained networks

`--sidechain-init scratch` explicitly initializes the one-step SC network through
its module constructors. It does not open an SC donor or import any donor SC
weights. It is mutually exclusive with SC-donor, warm-start and resume options.
Omitting an SC donor without explicitly selecting scratch still fails.

The scratch preset uses the same SC architecture as the validated donor:
`edm=false`, `centre_coord_input=true`, `frame_aware_head=false`,
`template_residual=false`, `a_bs_concat=true`, `q_bs=false`, `bb_context=true`,
`type_logits_input=true`. Existing rotamer input initialization is retained;
that is an input prior rather than loaded network weights. Provenance records
`sidechain.origin=scratch`, constructor initialization, seed and architecture.

Official PXDesign v0.1.0 supplies the full backbone and condition encoder;
pretrained FAMPNN supplies its sequence network. Their tensors and optimizer
exclusion are checked by the same component and phase machinery as the
SC-donor experiment.

Submit the initial phase on HAI:

```bash
sbatch scripts/training/slurm_official_sc_scratch_hai.sh
```

This submits a separate experiment with the matched donor-run settings:

- Phase `sc_adapt`: only `sidechain_module.*` trains (116.56M parameters).
- No revisions, no backbone refinement, no SC-to-AA or SC-to-BB feedback.
- 25% PDB monomer / 75% PINDER complex sampling, native SC supervision.
- Crop 384, gradient accumulation 8, SC learning rate 1e-5, warmup 500.
- Up to 30,000 optimizer steps; seed 0, BF16 training, one H200.
- Checkpoints every 500 steps, validation every 2,000 steps.

SC training comes first. The launcher does not schedule an automatic feedback
transition. A subsequent feedback phase warm-starts from an integrated SC
checkpoint, retaining trained SC weights and leaving PXDesign/FAMPNN frozen;
do not pass `--sidechain-init scratch` when transitioning or resuming.

Validation: scratch-only GPU smoke **114908** passed. It checked an SC optimizer
update, exact official donor tensors, bitwise frozen pretrained tensor hashes,
native-sampler parity, validation, checkpoint resume, self-contained evaluation
reconstruction and mmCIF export, with feedback disabled throughout.
[Machine-readable evidence](validation/official_sc_scratch/smoke_114908.json).
