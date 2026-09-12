# Scratch SC curriculum with frozen pretrained networks

The initial scratch experiment now uses **monomer-only native-frame SC warm-up**.
The previous 25% monomer / 75% PINDER recipe was carried over from donor adaptation;
it was not established for a randomly initialized packer. Runs **114897** (SC donor)
and **114910** (scratch SC) were canceled on 2026-09-12 after this correction.

## Initial warm-up

```bash
sbatch --parsable scripts/training/slurm_official_sc_scratch_hai.sh
```

- Phase `sc_warmup`; only `sidechain_module.*` trains (116.56M parameters).
- SC weights initialize from module constructors; no SC donor or prior run is loaded.
- Official PXDesign v0.1.0 backbone **and condition encoder**, plus pretrained
  FAMPNN, are frozen and remain in evaluation mode.
- Existing Protenix monomer dataset: pre-2021-09-30 weighted-PDB chain index,
  filtered to whole monomers of at most 384 tokens. No PINDER training items.
- Native residue types, native backbone frames, observed native SC coordinate
  targets. Backbone identity inputs remain XPB masked; native types enter SC only.
- The frozen PXDesign feature pass observes the native backbone with conditioning
  sigma 0.4 and discards its coordinate output. It adds no backbone corruption and
  does not call the training denoising sampler. FAMPNN decoding is not called.
- One-step SC packing uses the existing shared packer. Chemical slot masks and
  observed-atom supervision masks remain separate. The objective is native SC
  coordinate MSE, calculated in FP32; logs report SC error, not a GT-as-prediction
  backbone score.
- Zero revisions, no refinement, no SC-to-AA/BB feedback; no AA, physical or
  backbone denoising objective during this initial phase.
- Crop 384, BF16, accumulation 8, LR 5e-5, warmup 2,000, gradient clip 1,
  maximum 50,000 optimizer steps. One H200, seed 0, 23h50 allocation.
- Save every 500 updates; evaluate every 2,000 on up to 491 recent-PDB monomers
  that pass the same token filter. These are starting settings, not a validated
  optimum or a promise that all 50,000 steps fit in one allocation.

Architecture remains `edm=false`, `centre_coord_input=true`,
`frame_aware_head=false`, `template_residual=false`, `a_bs_concat=true`,
`q_bs=false`, `bb_context=true`, `type_logits_input=true`. Dunbrack template
initialization is an input prior; network weights are random. Native frames,
inventories and GT type logits are explicit runtime settings saved in the
integrated checkpoint. The checkpoint records component hashes, initialization
origins, optimizer state and effective configuration.

Training job **114920**, code **320b5af**, is the corrected submission. Resolved
sources: 47,622 training monomers and 308 eligible recent-PDB validation monomers
(the requested cap is 491). Component preflight confirmed only SC is trainable,
no SC donor, native frame/inventory flags and disabled feedback/refinement.
[Submission record](validation/official_sc_scratch/run_114920.json).

## Subsequent phases

1. `sc_complex_adapt`: warm-start the trained SC weights and introduce PINDER
   with a monomer-heavy mixture. Continue native-type/native-frame supervision;
   receptor/interface context uses the same native coordinate frame.
2. `sc_adapt`: generated-input adaptation with predicted frames and FAMPNN
   assignments, retaining isolated native SC auxiliary targets. Consider 25/75
   only if held-out validation supports the shift.
3. `feedback_adapt`: enable feedback routes progressively after SC validation.
   Updates to either pretrained network require a later explicit phase choice.

Later phases are not submitted automatically or started from scratch alongside
warm-up. Use `--warm-start-checkpoint` for phase transitions; use
`--resume-checkpoint` for exact continuation. Do not pass `--sidechain-init scratch`
with either operation. The scratch launcher intentionally starts only fresh runs.

## Engineering validation

Regression job **114919**: **639 passed**, including native-frame masking, empty
supervision, frozen phase groups and native-to-generated phase transitions.

GPU job **114916** passed at crop 384: native input/frame equality, no FAMPNN call,
no backbone training sampler, finite SC updates, exact official donor coverage,
bitwise frozen pretrained tensor hashes, held-out monomer evaluation, save/resume,
and reconstruction from saved configuration. The two-update smoke does not
establish packing quality. Evidence: [smoke_114916.json](validation/official_sc_scratch/smoke_114916.json).

Earlier smoke **114908** checked component loading and generated-input integration;
it did not validate this new curriculum. Its evidence remains available as
[smoke_114908.json](validation/official_sc_scratch/smoke_114908.json).
