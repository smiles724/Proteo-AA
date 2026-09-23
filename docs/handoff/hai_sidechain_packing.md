# HAI: the side-chain packing benchmark

Does A_BS actually improve side-chain packing? Table 4 structurally cannot
answer this -- `fold_af2ig.py` feeds AF2 `prev_pos`, which `pseudo_beta_fn`
reduces to CA (glycine) or CB (everything else), so **designability is blind
to every atom past CB**. All three integrated pilot cells score 100 %
designable while their clash density spans a 12x range.

Marlowe has these queued behind a two-day fairshare wall. Running on HAI
for speed; the Marlowe jobs stay queued as a backstop.

## Code

```bash
cd <Proteo-AA-pxdesign-fampnn-pack>
git fetch origin feat/multi-event-feedback
git checkout feat/multi-event-feedback     # 6cb9010 or later
```

This runs in the **vendored** environment, not the official PXDesign one --
`load_backbone_model` pulls `pxdesign_train`, which the official venv does
not have. Use whatever env you run `train_couple.py` / `eval_couple.py`
under. No Protenix v0.5.0 involved, so it will not contend with the
integrated-matrix queue.

## Protocol: GT sequence, side chains masked, packer rebuilds them

`scripts/eval_couple.py`'s default path supplies the **native** aatype and
masks the side-chain slots:

```python
aatype = native["aatype"].reshape(-1).long()      # GT sequence
given[:, backbone_slots] = native_mask[:, backbone_slots].float()
packed = packer(coords_af2=backbone_only, aatype=aatype[None], atom_mask=given[None], ...)
```

That is what makes RMSD, chi-recovery and rotamer-recovery defined: they
compare per-residue against a native, which needs the identity to match.
`ev.check_alignment` asserts the correspondence rather than assuming it, and
non-canonical residues are skipped rather than coerced.

**Keep the GT sequence.** Do NOT pass `--codesign` for the headline numbers:
that designs the sequence and then scores side chains only at positions
where the designed residue happens to match the native, which is a
selection toward easy buried positions and is not comparable to these RMSDs.
It is a useful separate reference; keep it in its own column with its
recovery rate beside it.

## Two corrections learned the hard way on Marlowe

**1. `MODE=native` cannot test adapters.** Its docstring: *"No PXDesign, no
adapters, no sigma_B: there is no diffusion in this path"*, and it reads
`args.checkpoint` zero times. A_BS maps PXDesign's `a_token` to FaMPNN's
`h_V`; a deposited backbone produces no `a_token`, so there is nothing to
condition on. Passing `CHECKPOINT=` to a native run is **silently ignored**
-- it ran, wrote metrics, and reported `"checkpoint": null`.

So `MODE=native` is a reference point only. **Only `MODE=denoised` tests the
adapters.**

**2. `--fampnn-weights` defaults to `0.0`, and J03/S03 were trained against
`0.3`.** Both checkpoints carry `identity.fampnn_variant == '0.3'`. Running
the default silently evaluates an adapter against the wrong donor. Always
pass `--fampnn-weights 0.3`.

**3.** (Marlowe-specific, already fixed in the wrapper there.)
`pxf/eval/canonical.py`'s `default_root()` resolves to a stale `/hai` path.
On HAI it may resolve correctly -- but check, because the failure lands 13
seconds in, *after* GPU allocation and *after* the held-out guard prints
success, so it looks like a healthy start. Set `PROTEOAA_METRICS_ROOT`
explicitly.

## Jobs

```bash
# reference: GT backbone, no adapters possible
MODE=native OUT=$OUT/native_reference_f03 \
  EXTRA_ARGS="--fampnn-weights 0.3" \
  sbatch scripts/slurm/eval_couple.sh

# the adapter comparison; each run also produces its own adapters-off arm
for ARM in J03_seed0 J03_seed1 S03_seed0 S03_seed1; do
  MODE=denoised CHECKPOINT=$CKPT/$ARM/checkpoints/step00000500.pt \
    OUT=$OUT/denoised_${ARM}_f03 \
    EXTRA_ARGS="--ema --fampnn-weights 0.3" \
    sbatch scripts/slurm/eval_couple.sh
done
```

`--ema` because `configs/bs_seq_sc/selection.yaml` declares `weights: ema`.

Zero-initialised adapters mean the adapters-off arm restores the pretrained
system **exactly**, so every denoised run carries its own matched baseline.
Four runs therefore give four adapter arms against one shared control.

**Smoke first** with `--max-targets 2`: two of the three problems above
surfaced in a 13-second and a 40-second smoke rather than at hour three.

## Data and leakage

The 31 held-out val dimers (`configs/val_structures_afdb.marlowe.txt`, or
the HAI original). Two independent checks both pass:

- cluster overlap between the 512 J03/S03 training clusters and the 31 val
  clusters is **zero**
- the script's own guard prints `256 eval ids, 0 overlap with 2000
  training ids`

Do **not** substitute random PINDER draws. PINDER train is cluster-redundant
-- 1.4M rows over 40k clusters, largest cluster 82k rows -- so uniform
sampling both oversamples large clusters and risks training overlap. Extend
by held-out `cluster_id` if 31 is too few.

## Metrics

`symmetry_rmsd`, `rotamer_recovery`, `chi_recovery_20deg`,
`chi1_accuracy_20deg`, `lddt_sc_sc`, `lddt_sc_env`, `bad_bond_fraction`,
`rotamer_outlier_fraction_40deg`. `scripts/eval_couple.py --compare <dirs>`
prints the delta table without recomputing.

The two-target native smoke on Marlowe gives a sense of scale (FaMPNN 0.0,
so not the number you want, but the shape): symmetry_rmsd 1.267,
rotamer_recovery 0.606, chi_recovery_20deg 0.750, lddt_sc_sc 0.837.

## What this tests that nothing else does

S03 is the side-chain-only adapter, and on Table 4 it sits **below** the
unadapted U03 (23.1 / 24.2 % against 26.5 %). That was measured in a channel
S03 does not act on. This benchmark measures it where it should win. If S03
does not beat the adapters-off arm here either, that is a far stronger
negative than the Table 4 row -- and it is the cheapest remaining
experiment that could still return a positive.
