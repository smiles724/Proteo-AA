# AlphaProteo designability evaluation — status: BLOCKED, and on what

Asked for: AlphaProteo-panel designability for the newly trained adapter,
compared against R0. This records why that number does not exist yet, what
exactly is missing, and the one thing that unblocks it.

## What is missing

Table 4 designability needs, in order:

1. **The shared backbone collection.** `scripts/cache_binder_backbones.py`,
   10 targets x 6 binder lengths.
2. Sequence design per arm (`design_binder_matrix.py`, not yet written).
3. AF2-IG scoring — **this part is ready**: `scripts/evaluation/fold_af2ig.py`
   in the Proteo-AA repo, validated end to end on H100 job 494737.
4. The pooled four-way conjunction (ipAE < 10.85 A, ipTM > 0.5, pLDDT > 80%,
   bound/unbound RMSD < 3.5 A), summed across lengths.

Step 1 is the blocker, and it blocks everything after it. Verified today on
Marlowe:

```
official available: False | found: protenix 2.0.0 (protenix.data.parser absent)
/hai                                       not mounted
runs/binder_bench/backbones_smoke/designs  empty
runs/binder_bench/targets/prepared.json    1 of 10 targets (PDL1)
```

No designability result exists for **any** arm. `runs/eval_joint/run1_vs_R0` is
the *joint-refinement* R0 (backbone RMSD in A) — a different experiment with a
coincidentally identical arm name. The only AF2-IG artifacts on disk are from
`proteo_aa_runs/cbdb_smoke`, which is the scorer's own smoke test.

## Why there is no workaround

The backbones are generated, not downloadable, and the `a_token` every coupled
arm reads is tapped from `layernorm_a` *during* denoising — so the coupled arms'
inputs cannot be reconstructed from a finished structure. Generation needs
PXDesign's official runtime (Protenix 0.5.0+pxd, `d18aa1da`); this repo vendors
`c3bfc36` (v2.0.0), where `protenix.data.parser` has moved.

The two-line alias that makes that import succeed is the pairing that produced
the interpenetrating backbones in `docs/target_conditioning_audit.md` §5/§10
(282 atom pairs under 2.6 A, 0.207 A minimum). It is not on the table, and
rebuilding the environment on Marlowe is explicitly not the plan.

## What unblocks it

On HAI, in `/hai/scratch/yfsun/envs/pxdesign_official`:

```bash
python scripts/prepare_binder_targets.py --out runs/binder_bench/targets   # all 10
python scripts/cache_binder_backbones.py \
    --targets-dir runs/binder_bench/targets \
    --out runs/binder_bench/backbones \
    --checkpoint-dir <pxdesign release dir> \
    --lengths 80 90 100 110 120 130 --n-samples 8
```

Ship `runs/binder_bench/backbones/` back to
`/scratch/m000137-pm06/Proteo-AA/pxf/runs/binder_bench/`. Everything downstream
runs on Marlowe.

## One correction to the comparison itself

**R0 is not the right baseline for this adapter, and the request should not be
served as written.** R0 is ProteinMPNN under `context: complex`. The trained
adapter is J03: FaMPNN 0.3, `shared_prelogit`, `context: complex_sc`. A
J03 − R0 difference therefore moves the designer, the context level and the
adapter simultaneously.

The adapter's uncoupled half is **U03** — the same 0.3 donor with the residual
off. That pair isolates the residual, and it is now registered as
`comparisons.new_main_family` in `configs/binder_benchmark/arms.yaml`.

R0 remains worth reporting as the shared *absolute* external reference for the
whole matrix. So when this runs, the table should carry both:

* `J03 − U03` — the adapter effect (primary for the 0.3 family, two seeds,
  reported per seed with agreement).
* `J03 − R0` and `U03 − R0` — absolute standing against ProteinMPNN, labelled
  as a designer comparison, not an adapter gain.

Reporting J03 − R0 alone as the adapter's benefit would be the same error
`arms.yaml`'s `donor_policy` already forbids for 0.0 vs 0.3.

## What the adapter's number is, on the evidence that does exist

Stage-1 reconstruction on the 31 val dimers (`bs_seq_sc_v1_stage1.md`, job
495864) is measured and is *not* designability — `selection.yaml` puts
designability explicitly out of scope. It showed J03 beating the side-chain-only
arm on masked-sequence NLL in both seeds, but **not** beating the uncoupled 0.3
donor. That is a reason to expect a small-to-nil J03 − U03 designability effect,
and a reason to run the benchmark rather than to assume one.
