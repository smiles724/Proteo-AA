# AlphaProteo designability evaluation — status: backbones generated on HAI

Asked for: AlphaProteo-panel designability for the newly trained adapter,
compared against R0. Step 1 of four — the shared backbone collection — was the
blocker and is now done. This records what was generated, what had to be fixed
to generate it, and the two things a reader has to know before consuming it.

## What now exists

| | |
| --- | --- |
| targets | 10/10 prepared and validated, `runs/binder_bench/targets` |
| backbones | 10 targets x 6 lengths x 8 samples = 480, `runs/binder_bench/backbones` |
| per design | `x0`, `a_token` at the event, the token map and chain roles, `actual_sigma`, sha256 |
| runtime | Protenix `0.5.0+pxd`, pxdesign `0.1.0`, `pxdesign_v0.1.0.pt` `b075867b...` |
| generated on | HAI, `haic-hgx-{6,9}`, H200, slurm array 121282 |

Still to write, unchanged: `design_binder_matrix.py` and
`report_binder_matrix.py`. AF2-IG scoring
(`scripts/evaluation/fold_af2ig.py`) is ready and was validated on job 494737.

## Two things to know before consuming it

**1. It cannot be regenerated.** Measured on one node, one GPU, back to back:
two identical invocations at the same seed give `x0` up to **0.55 A** apart,
and across separate jobs 5.9 A. The RNG is not the cause --
`pxf.couple.replay.RngStream` seeds torch (hence CUDA) and numpy, which is
both sources `centre_random_augmentation` draws from. What is left is
non-deterministic CUDA reductions compounding over 400 denoiser calls. The
summary statistics barely move (`event -> final` RMSD 23.7485 vs 23.7565 A)
while the structures do not match, which is the combination that makes this
easy to miss.

Consequences: back the collection up rather than re-deriving it; re-running one
target to fill a gap gives backbones its siblings do not share; and
`backbones.json` carries a `sha256` per design so "every arm consumed the same
collection" is checkable instead of assumed.

**2. 8 samples per length is not A-CODE's sampling depth.** 48 backbones per
target against Section 4.2's 328-728. The four-way conjunction is a rate, so a
thin denominator widens every arm's interval and the paired `J03 - U03`
difference is what suffers first. This is the number the runbook specified and
it is enough to exercise the whole pipeline end to end; it is not enough to
reproduce Table 4. Scaling to ~330/target is ~3,300 designs, roughly 8 GPU-hours
per target on an H200 at the measured 8.2 s/design plus one model load per
length.

## What had to be fixed

`scripts/cache_binder_backbones.py` had never been run -- it could not be,
since the official runtime is only on HAI. Six defects, each found by running
it and each fixed:

1. `build_runner` returns `(runner, configs)`; the script bound the tuple to
   `runner`. `official_single_event.py` and `official_replay_check.py` both
   unpack it.
2. `RandomStream` does not exist in `pxf.couple.replay`; the class is
   `RngStream`.
3. `provenance.runtime_sources(strict=False)` raises on uninitialized
   submodules — and initializing them is not the fix. `scripts/_bootstrap`
   puts `<repo>/PXDesign` and `<repo>/Protenix` at the *front* of `sys.path`,
   so populating them shadows the official install with the vendored c3bfc36:
   precisely the pairing `pxf/official/require.py` refuses and §5/§10 of the
   conditioning audit measured. It also records the wrong thing — the manifest
   would name a runtime that did not run. Replaced with `official_sources()`,
   which records the installed versions, asserts they resolve to the official
   install, and hashes the weights.
4. `run_trajectory` records states with `detach_cpu`; the re-denoise fed the
   CPU sigma to CUDA weights and died inside the Fourier embedding.
5. The payload dropped the topology. `x0` is a flat `[n_atom, 3]` tensor, and
   `pxf.official.bridge` recovers atom names, `atom_to_token_idx` and chain
   roles from the dataloader's AtomArray — the official featurizer, which
   Marlowe does not have. The module docstring already promised to cache "the
   token map and chain roles"; now it does.
6. `scripts/prepare_binder_targets.py` used `gemmi.Model.name`, removed in
   gemmi 0.7. HAI's official env carries 0.7.5.

Two additions: `scripts/merge_binder_backbones.py`, because the ten targets run
as ten array tasks and the design stage expects one collection — it refuses to
merge shards whose `n_step`, `eta`, `sigma_b`, `dtype`, `use_msa`, `seed` or
runtime disagree, since those would be two experiments sharing a directory. And
the runner is now built once per (target, length) rather than once per sample:
measured, 8 s of trajectory sat inside a 109 s job. That also removes a hazard,
because `OfficialDenoiser.__init__` deletes the template keys from the feature
dict in place, so a second denoiser built from the same batch would condition
on a dict they had already been removed from.

## Target preparation

All ten validate against the structures (`scripts/validate_binder_targets.py`):
every crop selects residues, every hotspot resolves to a real residue, and the
seven depositions with a partner put their hotspots 1.95-6.25 A from it. IR and
TNFa are apo and are reported as such rather than silently passed.

The end-to-end check on the whole numbering conversion holds: PDL1's published
crop `A 17-132` with hotspots 56/115/123 converts to `1-116` with hotspots
40/99/107, which is character-for-character PXDesign's shipped
`examples/PDL1_quick_start.yaml`.

One judgement call is still open and is recorded in `targets.yaml` rather than
resolved: H1's chain A has three surviving readings of Table S1's numbering.
The `+3` reading is used. All three give identical hotspots, and the hotspots
are what steer the design, so this affects target context only.

## The comparison itself

Unchanged from the previous revision of this file, and it is the part of the
original request that should not be served as written. R0 is ProteinMPNN under
`context: complex`; the trained adapter is J03 (FaMPNN 0.3, `shared_prelogit`,
`context: complex_sc`). `J03 - R0` moves the designer, the context level and
the adapter at once.

* `J03 - U03` — the adapter effect. Primary for the 0.3 family, two seeds,
  reported per seed with agreement.
* `J03 - R0` and `U03 - R0` — absolute standing against ProteinMPNN, labelled
  as a designer comparison, not an adapter gain.

Registered as `comparisons.new_main_family` in
`configs/binder_benchmark/arms.yaml`.

Stage-1 reconstruction on the 31 val dimers (`bs_seq_sc_v1_stage1.md`, job
495864) showed J03 beating the side-chain-only arm on masked-sequence NLL in
both seeds but not beating the uncoupled 0.3 donor. That is a reason to expect
a small-to-nil `J03 - U03` designability effect — and a reason to run the
benchmark rather than assume one. With 48 backbones per target, an effect that
small is unlikely to clear the noise; see point 2 above.
