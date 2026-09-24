# Retrain the feedback adapters at three event sigmas

Self-contained: assumes no prior context.

## What this project is, in one paragraph

Binder design by coupling two frozen models. **PXDesign** (a Protenix
diffusion model) generates a protein backbone; **FaMPNN** designs a sequence
and packs side chains onto it. Small trainable **adapters** bridge them.
`A_BS` maps PXDesign's token features (`a_token`) into FaMPNN's hidden state
`h_V`. The **feedback** adapters (called E1) go the other way: mid-diffusion,
FaMPNN packs side chains on a partially-denoised backbone, and a residual
from that packed state is injected back into one PXDesign denoiser call,
correcting the backbone. The point in the trajectory where that exchange
happens is the **event**, identified by its noise level **sigma**.

## The task

The E1 adapters were trained at one event at **sigma 0.429** (step 350 of
400). We want to know whether an earlier event works better. The adapters
are explicitly conditioned on log sigma, so simply running inference at a
different sigma extrapolates that conditioning and cannot produce an
adoptable checkpoint. **So: retrain at three earlier sigmas.**

| requested sigma | solver step | position in trajectory |
|---|---|---|
| 1.0469 | 334 | 84 % |
| 2.0625 | 319 | 80 % |
| 4.1250 | 303 | 76 % |

Sigma **decreases** along the trajectory, so a higher sigma is an earlier
event. 0.429 stays as the already-trained baseline; do not redo it.

Each sigma needs: two conditioning caches (one per A_BS seed) and four
trained arms (`E1_full` and `E1_bb_only` x seeds 0 and 1). **18 checkpoints
total.**

## Code

```bash
git clone https://github.com/smiles724/Proteo-AA.git   # or reuse a checkout
cd <the Proteo-AA-pxdesign-fampnn-pack worktree>
git fetch origin feat/multi-event-feedback
git checkout feat/multi-event-feedback      # a202d4f or later
git submodule update --init --recursive     # fampnn must be on PYTHONPATH
```

`scripts/slurm/marlowe/retrain_event_sigma.sh` does one whole sigma --
both caches then all four arms, chained. Adapt its paths; the body is what
you want. Each stage skips work already on disk, so a resubmission resumes.

## THE RUNTIME WALL -- read before touching PYTHONPATH

Two incompatible PXDesign/Protenix pairings exist in this project:

- **official** -- Protenix `v0.5.0+pxd`, module `protenix.data.parser`. This
  is what the retrain runs under.
- **vendored** -- `c3bfc36` (v2.0.0), module `protenix.data.core.parser`.

Rules, all load-bearing:

- `PYTHONPATH` must be `$REPO:$PRISTINE_PXDESIGN:$REPO/fampnn` and must
  **not** include `$REPO/PXDesign` or `$REPO/Protenix` -- those shadow the
  official install with the vendored tree, which `pxf/official/require.py`
  refuses.
- **Never alias `protenix.data.parser` to make v2.0.0 import.**
- Do not `pip install torch_geometric` or `timm` without `--no-deps`. On
  another cluster that pulled torch 2.14 over the pinned 2.3.1 and broke a
  validated environment. Pins: `torch==2.3.1+cu121`,
  `torchvision==0.18.1+cu121`.

Sanity check before submitting anything:

```bash
python -c "from pxf.official.require import official_protenix_available as a; print(a())"
# must print (True, ...)
```

## Data: everything is in the bundle

```bash
export MARLOWE_HOST=<marlowe login fqdn>     # ask; not discoverable from here
export BUNDLE=/hai/scratch/yfsun/pxf_handoff/pxf_hai_bundle
scripts/handoff/fetch_bundle_on_hai.sh --dest $(dirname $BUNDLE)
```

That verifies every digest, checks the donors, and rehydrates the
`@BUNDLE@` placeholders in the manifests and target YAMLs. **If a donor
digest mismatches, stop** -- those are the weights every arm is conditioned
on, and a different donor makes the results incomparable.

What the retrain reads:

| what | path in bundle | notes |
|---|---|---|
| training manifest | `training_data/resolved/train_pdb.parquet` | 198 complexes |
| structures | `training_data/structures/` | 237 CIFs, ~44 MB |
| A_BS checkpoints | `checkpoints/bs_seq_sc/J03_seed{0,1}_step00000500.pt` | `c506c7e1c43104dd`, `ee673b982c4f5dd7` |
| PXDesign donor | `checkpoints/donors/pxdesign_v0.1.0.pt` | `b075867bae942dc0` |
| FaMPNN 0.3 | `checkpoints/donors/fampnn_0_3.pt` | `8969b3f1f3c94117` |
| Protenix data | `official_release_data/` | `PROTENIX_ROOT_DIR` |

Arm configs are in the repo: `configs/integrated_feedback/E1_full.yaml`,
`E1_bb_only.yaml`.

**Use the `resolved/` manifests**, not the templated ones beside them --
the raw ones still contain `@BUNDLE@`.

## Run it

Per sigma, adapting the Marlowe wrapper:

```
cache:  scripts/cache_integrated_feedback.py --manifest <train_pdb.parquet> \
          --out <out>/cache/J03_s<S> \
          --bs-checkpoint <J03_seed<S>> --bs-weights ema \
          --pxdesign-donor <donor> --fampnn-checkpoint <fampnn_0_3> \
          --fampnn-variant 0.3 --event-sigma <SIGMA> --seed <S>

train:  scripts/train_integrated_feedback.py \
          --config configs/integrated_feedback/<ARM>.yaml \
          --train-cache <out>/cache/J03_s<S> \
          --bs-checkpoint <J03_seed<S>> --pxdesign-donor <donor> \
          --fampnn-checkpoint <fampnn_0_3> --fampnn-variant 0.3 \
          --seed <S> --max-steps 2000 --out <out>/<ARM>_s<S>
```

**`--fampnn-variant 0.3` everywhere.** J03 was fit against FaMPNN 0.3; an
adapter crossed with the wrong donor loads cleanly and measures nothing.
That mistake has already cost a run here.

## Cost

Measured on the original 0.429 training: caches 39 s and 61 s; training
step 500 to 1000 took 63 s, 1000 to 2000 took 126 s -- so about 5 minutes
per arm. Six sequential stages plus model loading is roughly **40 minutes
per sigma**, about 2 hours for all three. Request ~1.5 h per job, not 6 --
a smaller reservation backfills far sooner on a busy partition.

## Deliverable

For each sigma, under `integrated_feedback_sigma/sig<sigma>/`:

```
cache/J03_s0/  cache/J03_s1/
E1_full_s0/checkpoints/final.pt      E1_full_s1/checkpoints/final.pt
E1_bb_only_s0/checkpoints/final.pt   E1_bb_only_s1/checkpoints/final.pt
```

Report the sha256 of all 18 `final.pt` files and each run's final loss.

## Two things to check, and one not to assume

**Check that the caches differ between sigmas.** Each cache is built by
running the event at its sigma; if two sigmas produced byte-identical
caches, the `--event-sigma` flag is not reaching the event selection and
every downstream number would be a duplicate. Compare the `cache_key` in
each `cache.json`.

**Check the mean-residual control.** `MeanResidual` interpolates in log
sigma and **clamps at its knots**, and its own docstring says the schedule
"only ever visits noise levels inside the training window". If it is not
re-estimated against the new caches, that control arm silently returns an
edge value at the new sigmas and must not be quoted. I have not verified
whether the retrain re-estimates it.

**Do not assume A_BS transfers.** Only the E1 feedback adapters are
retrained here; `A_BS` (J03) is reused unchanged at every sigma, so its own
sigma-conditioning is still extrapolated at 1.05, 2.06 and 4.125. If the
sweep shows movement, part of it may be J03 straining rather than E1
improving. Retraining A_BS per sigma is a separate, larger job.

## Do not

- Do not train on the 31 validation complexes (`validation.parquet`).
- Do not mix FaMPNN 0.0 and 0.3 with an adapter fit against the other.
- Do not regenerate the cached backbone collection if you encounter it
  elsewhere: two identical invocations differ by up to 0.55 A, so it is not
  reproducible and must be copied rather than rebuilt.
