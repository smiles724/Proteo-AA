# Handoff to Claude on HAI: finish the integrated-feedback generation matrix

Paste everything from "## Task" down into a Claude Code session on HAI.

---

## Task

Finish the integrated-feedback binder-generation experiment. Training,
caching, and the acceptance gate are **done on Marlowe**; what remains is
generation and scoring, which stalled there behind a project fairshare of
0.003568 (the queued cell had a ~6 h estimated start and the queue is not
improving). Everything you need has been staged into a bundle.

### 0. Ground truth you should not re-derive

These are measured, not assumed. Do not spend time reproving them.

- The acceptance gate **PASSED** with `ready_for_full_cache = true` and
  `--require-complete` exit 0. Nine required checks PASS. `gpu_replay` is
  INCOMPLETE and is *not* a required check — it is resolution-limited.
- Schema-v3 caches: **198 events, 0 failures**, both A_BS seeds
  (`f56f95213c1ea38e`, `f816a269433b615d`).
- Four feedback runs are complete, and **matched-pair verification PASSED**
  (seed-0 init digest `8ccb34d6a25350f0`, seed-1 `095d44549d66789d`).
- Checkpoint selection ran and was **blocked by a mis-specified guardrail**
  (see §5). The artifact you were given therefore has every entry marked
  `SELECTION_STATUS: "OVERRIDE -- NOT SELECTED"`. That string is intentional
  and must survive into the results — the step-2000 checkpoints are being
  used by override, not by a passing criterion.

### 1. Set up

```bash
# a) the repo
git clone https://github.com/smiles724/Proteo-AA.git  # or reuse an existing clone
cd Proteo-AA-pxdesign-fampnn-pack
git fetch origin exp/binder-design-matrix
git checkout exp/binder-design-matrix
git submodule update --init --recursive        # fampnn is needed on PYTHONPATH
# confirm you match the bundle:
cat $BUNDLE/repo/HEAD

# b) the bundle
export MARLOWE_HOST=<marlowe login fqdn>       # ask the user; it is not in the bundle
export BUNDLE=/hai/scratch/yfsun/pxf_handoff/pxf_hai_bundle
scripts/handoff/fetch_bundle_on_hai.sh --dest $(dirname $BUNDLE)
```

`fetch_bundle_on_hai.sh` verifies every digest, checks the PXDesign donor is
`b075867bae942dc0` and FaMPNN 0.3 is `8969b3f1f3c94117`, and rehydrates the
`@BUNDLE@` placeholders in the selection artifact. **If a donor digest
mismatches, stop.** Those are the weights every arm is conditioned on; a
different donor makes the HAI arms incomparable to the Marlowe results, and
that is not something you can correct after the fact.

### 1b. Environment

The bundle ships the PXDesign donor, FaMPNN 0.3, and the whole
`official_release_data` tree, so nothing below needs a second fetch. The two
CCD symlinks inside it were rewritten relative at staging time and
`fetch_bundle_on_hai.sh` re-checks that they resolve here.

```bash
export REPO=<your checkout of Proteo-AA-pxdesign-fampnn-pack>
export PRISTINE=<your pristine PXDesign v0.5.0+pxd tree>   # e.g. /hai/scratch/yfsun/pxdesign_official
export PYTHONPATH="$REPO:$PRISTINE:$REPO/fampnn"
export PROTENIX_ROOT_DIR="$BUNDLE/official_release_data"
export PROTENIX_DATA_ROOT_DIR="$BUNDLE/official_release_data/ccd_cache"
export PROTEOAA_ROOT=<your Proteo-AA checkout>
export PROTEOAA_METRICS_ROOT="$PROTEOAA_ROOT"
export LAYERNORM_TYPE=torch PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-$$"     # not NFS
```

`scripts/slurm/marlowe/integrated_official.sh` is the Marlowe equivalent and
is worth reading as the reference for what the job actually needs; adapt its
body to HAI's scheduler rather than rewriting the environment from scratch.

### 2. The runtime wall — read this before editing any path

There are two PXDesign/Protenix pairings in this project and they are not
interchangeable:

- **official** — Protenix `v0.5.0+pxd` (`d18aa1da`), module
  `protenix.data.parser`. This is what generation runs under.
- **vendored** — `c3bfc36` (v2.0.0), module `protenix.data.core.parser`.

Constraints, all of which are load-bearing:

- **Never alias `protenix.data.parser` to make v2.0.0 import.** This
  prohibition stands on version-skew grounds. (A historical note in case you
  read older comments: the interpenetrating-backbone failure was *measured*
  to come from `FixedTarget` + eta, not from aliasing. The prohibition is
  still correct; the old causal attribution was not.)
- `PYTHONPATH` must be `$REPO:$PRISTINE_PXDESIGN:$REPO/fampnn` and must
  **not** include `$REPO/PXDesign` or `$REPO/Protenix` — those shadow the
  official install with the vendored tree, which `pxf/official/require.py`
  refuses.
- Do **not** `pip install torch_geometric` or `timm` without `--no-deps`.
  On Marlowe that pulled torch 2.14 over the pinned 2.3.1 and damaged the
  validated venv. Pins: `torch==2.3.1+cu121`, `torchvision==0.18.1+cu121`.
  See `scripts/utilities/install_pxdesign_official.sh`.
- Sanity check before submitting anything:
  ```bash
  python -c "from pxf.official.require import official_protenix_available as a; print(a())"
  ```
  must print `(True, ...)`.

### 3. The pending work

**3a. One generation cell — do this first, alone.**

One target, one generation seed, all seven outputs. Seven = `U03` (unadapted
donor, its own event decode, no A_BS and no feedback) plus
`{J03, E1_bb_only, E1_full}` for each of the two A_BS seeds.

```bash
export OUTROOT=/hai/scratch/yfsun/pxf_runs/integrated_feedback_v1
python scripts/run_integrated_binder_matrix.py \
  --targets-config   $BUNDLE/targets/configs_binder_benchmark/targets.yaml \
  --prepared-dir     $BUNDLE/targets/binder_bench_targets/configs.resolved \
  --checkpoint-dir   $BUNDLE/checkpoints/donors \
  --checkpoint-selection $BUNDLE/selection/selected_checkpoints.json \
  --bs-checkpoint 0=$BUNDLE/checkpoints/bs_seq_sc/J03_seed0_step00000500.pt \
  --bs-checkpoint 1=$BUNDLE/checkpoints/bs_seq_sc/J03_seed1_step00000500.pt \
  --fampnn-checkpoint $BUNDLE/checkpoints/donors/fampnn_0_3.pt \
  --fampnn-variant 0.3 \
  --targets PDL1 --lengths 80 --seeds 101 \
  --out $OUTROOT/generation_cell
```

`--checkpoint-dir` is only read to locate `pxdesign_v0.1.0.pt`; point it at
whatever directory holds the verified donor.

Then **validate the 7 PDBs before scoring anything**: chain assignment,
binder length exactly 80, `resolve_binder_chain` agrees, no missing CA atoms.
A design that fails this is a plumbing bug, not a weak design.

**3b. Report the cell.**

```bash
python scripts/report_integrated_binder_matrix.py \
  --designs $OUTROOT/generation_cell/designs.csv \
  --out     $OUTROOT/generation_cell/report
```

`--designs` wants the **CSV**, not the run directory, and `--out` is a
**directory** the script creates, not a filename. Once §3c has written
`metrics/af2ig.csv`, re-run this with `--metrics-dir
$OUTROOT/generation_cell/metrics` — the report globs `*.csv` from a
directory, while `fold_af2ig.py` takes `--metrics-csv` for a single file, so
the two compose only through that directory.

The report runs plumbing → chemistry → designability in that order, and
emits designability **only if** the AF2-IG metrics are present.

**3c. AF2-IG scoring.**

The driver is `scripts/evaluation/fold_af2ig.py` in the **Proteo-AA** repo,
branch **`binder-design-training`**, at `c0c1732` or later. It was untracked
until 2026-09-22 — an earlier version of this handoff named it without it
existing in any ref, which was wrong. It is pushed now, along with
`pxdesign_train/benchmarks/af2ig.py`, `score_af2ig_designability.py`,
`slurm_fold_af2ig.sh`, `bootstrap_af2ig.sh`, and `tests/test_af2ig_scoring.py`.

It is **not** the `origin/sjm/alphaproteo10_eval` path
(`slurm_score_alphaproteo_designability.sh` et al.). That scorer is a
different harness with different weight requirements — do not mix them.

*Environment.* Scoring runs in its own venv, and must: AF2 is JAX, this
project is torch+Protenix, and PXDesign's scoring stack pins Protenix
v0.5.0+pxd against the training repo's v2.0.0. They cannot share an
interpreter. Build it with `bash scripts/utilities/bootstrap_af2ig.sh`
(read the header first — ColabDesign installs `--no-deps` on purpose).

*Weights — your existing files are correct.* The harness reads **exactly
two** AlphaFold parameter files: `params_model_1_ptm.npz` (complex pass,
needs the template stack) and `params_model_3_ptm.npz` (unbound pass, does
not). `/hai/scratch/yfsun/af2_params/params` already holds both, so pass
`--data-dir /hai/scratch/yfsun/af2_params` — the flag wants the directory
*containing* `params/`, not `params/` itself.

**No ProteinMPNN download is needed.** ColabDesign ships the original
`v_48_020` weights inside the wheel at `colabdesign/mpnn/weights/`; the code
calls `mk_mpnn_model(weights="original")` and never reads
`tool_weights/mpnn/vanilla_model_weights`. And MPNN is only touched by
`--variants pmpnn` at all — the §3c default `co_design` does not use it.
The non-ptm `params_model_1.npz` requirement you found belongs to the
alphaproteo10_eval scorer, not this one.

```bash
python <proteo-aa>/scripts/evaluation/fold_af2ig.py \
  --designs-csv $OUTROOT/generation_cell/designs.csv \
  --designs-dir $OUTROOT/generation_cell/designs \
  --metrics-csv $OUTROOT/generation_cell/metrics/af2ig.csv \
  --data-dir    /hai/scratch/yfsun/af2_params \
  --variants co_design --targets PDL1
```

Designability is the **four-way conjunction**, all four required:
ipAE < 10.85 Å, ipTM > 0.5, pLDDT > 80%, binder bound/unbound RMSD < 3.5 Å.
For A-CODE Table 4 comparability successes are **summed (pooled)** across the
length grid, lengths 80–130. Note `AF2IGFilter.is_designable` takes a **dict**,
not kwargs.

**3d. Only after 3a–3c are clean: the 28-design smoke.**

2 targets × 2 generation seeds × 7 arms. Add `--seeds 101 102` and a second
target. The matrix resumes per arm off a shared prefix, so a partial rerun is
safe; `U03` is computed once per cell.

**3e. Report the table.**

One row per arm, with per-arm designability and the four component rates, and
the `SELECTION_STATUS` override carried through. Paired differences per
design, not differences of aggregates.

### 4. What you must not do

- **Do not train on the 31 validation complexes.**
- **Do not** load the existing FaMPNN 0.0 adapter on top of 0.3 and present
  it as a trained 0.3 coupling.
- **Do not regenerate** `runs/binder_bench/backbones`. That 480-backbone
  collection is **not reproducible** — two identical invocations of
  `cache_binder_backbones.py` at the same seed on the same GPU differ by up to
  0.55 Å. It is not in the bundle because the matrix generates its own
  backbone per cell and never reads it, but if you encounter it elsewhere:
  back it up, never re-derive it.
- If binder **side-chain** coordinates ever appear in the noisy state, that is
  a leak — fix the input construction with explicit binder backbone atom
  selection (`N, CA, C, O`; see `pxf/bench/native_event_inputs.py`) and
  invalidate the affected fixtures. **Do not explain it away as a legitimate
  perturbation.** This exact leak was already found and fixed once.

### 5. Two open issues — surface them, do not quietly patch them

**The side-chain guardrail is mis-specified.** `configs/bs_seq_sc/selection.yaml`
sets `max_relative_regression: 0.05` on median `sc_loss`. With 15 complexes,
one complex flipping is 6.7 pp ≈ 11% relative — the threshold is finer than
the panel can resolve, and the no-feedback baseline itself fails it (60%,
47–53%). It was **deliberately not retuned**, because retuning a predeclared
criterion after seeing results is how you select the curve you already liked.
The redefinition is the user's call: chemistry evaluated on a *finished*
backbone, and a threshold in units 15 complexes can actually resolve. Report
against the override; flag it; do not silently change the number.

**The two protocols select the event sigma differently.** The cached-backbone
matrix selects by *scheduled* sigma (0.4355, actual 0.8711 after churn); the
integrated path selects by *actual churned* sigma (0.429, which is J03's
training noise). Churn is `gamma = gamma0 if c_tau > gamma_min else 0`,
`t_hat = c_tau_last * (gamma + 1)`, and PXDesign uses `gamma0=1.0,
gamma_min=0.01`, so the actual sigma is 2× the scheduled one. The integrated
default `--event-sigma 0.429` is the right one for this experiment. Just be
aware the two number families are not directly comparable.

### 6. Calibrate your expectations

Four independent comparisons on this system have come back **null**:
J03 − U03 on NLL; J03 − U03 on designability (+0.42 pp, p = 0.86); the A_SB
feedback arm; and E1_full − E1_bb_only (±0.0004 Å). A null result here is the
most likely outcome and is a perfectly good one. Report it as such — do not
go looking for a configuration that makes a difference appear.

One finding that *is* real and worth preserving: across the AlphaProteo Table
4 rows, results agree far more closely by **backbone generator** (mean abs
11.8 pp) than by sequence designer (22.9 pp). H1's 59-point gap is a backbone
result, not a sequence-design result.

### 7. Report back

- The 7-row table for the cell, then the 28-row table for the smoke.
- Per-arm designability plus the four component pass rates.
- Digests of every checkpoint actually loaded, so the HAI run can be matched
  against the Marlowe provenance in `$BUNDLE/reports/`.
- Anything you had to change to make it run on HAI, stated plainly.
