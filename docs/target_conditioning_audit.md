# Target-conditioned PXDesign: runtime, checkpoint, and conditioning audit

Status: step 1 (official positive control) built and staged; steps 2-3 partly
answered by static and feature-level evidence below.

## 0. The comparison that must not be reused

The three-arm comparison from generation stress-test stage 1 --

| arm | clashes |
|---|---|
| bb0 | 1092 |
| full | 1093 |
| bb_only | 1094 |

-- **is feedback evaluated on an invalid generation baseline.** The generated
backbone interpenetrated the fixed target (BB x BB minimum 0.207 A, 282 pairs
under 2.6 A, against 0 clashes at 2.792 A for the native control). Whatever
those three numbers differ by, they differ inside a complex that should not
exist. They are recorded here only so they are not quoted later as evidence
about feedback. No conclusion about the SC->BB adapter follows from them.

The harness itself did verify: replay drift 0.000 A through the real decoder,
mapping 228/456, target movement 0.00 A, injections 0/1/1, one unique
sequence, 31 calls per arm. The harness is not what failed.

## 1. How PXDesign actually conditions on a target

Three channels, established by reading the released code
(`PXDesign/pxdesign/model/embedders.py`, `pxdesign/data/featurizer.py`):

* **`conditional_templ` / `conditional_templ_mask`** -- pairwise distances
  between resolved non-design tokens, binned into 64 bins over 2..22 A
  (`torch.linspace(2.0, 22.0, 63)` boundaries), embedded by
  `ConditionTemplateEmbedder` as `nn.Embedding(64 + 1, c_z)` into the pair
  representation `z`. The embedder computes `pair_mask * (1 + conditional_templ)`,
  so masked pairs read bin 0 and bins 1..64 carry distance.
* **`restype`** -- 32 + 4 classes. The extra classes mark "needs to be
  designed"; design residues are additionally renamed to `xpb`.
* **`hotspot`** (1) and **`plddt`** (1), defaulted to zeros when absent.

### What this implies

`ProtenixDesign.sample_diffusion` forwards only sampler configuration. There is
no fixed-atom channel, no `x_gt`, and no coordinate overwrite anywhere in
`pxdesign/model/pxdesign.py` or `pxdesign/runner/inference.py`.

The target therefore enters **only as a pairwise distogram**, which is
invariant to global rotation and translation. The sampler denoises target and
binder tokens alike; the target's internal geometry is pinned by the
distogram, but **its absolute pose is emergent, not given**.

This is the conditioning contract, and it explains the interpenetration
directly. Overwriting target coordinates with the native pose at every solver
step forces the target into a frame the model never chose, while the binder is
being generated relative to the model's own placement of the target. The two
halves end up in different frames, so they overlap. The fix is not a better
overwrite; it is to stop overwriting and superpose afterwards.

## 2. The conditioning inputs are correct locally (measured, not assumed)

`scripts/audit_conditioning.py` featurizes a prepared dimer exactly as
`scripts/gen_stress_test.py` does and inspects the tensors:

| | 1jfl | 1za5 |
|---|---|---|
| design / condition tokens | 100 / 156 | 192 / 192 |
| design chain / condition chain | 0 / 1 | 0 / 1 |
| `conditional_templ_mask` pairs | 24336 | 36864 |
| target x target pairs available | 24336 | 36864 |
| mask on design x design | 0 | 0 |
| mask on cross terms | 0 | 0 |
| bin range | 0..63 | 0..63 |
| `restype` dim / design class | 36 / 32 | 36 / 32 |
| hotspot sum | 4.0 | 2.0 |

The template covers the target block exactly and nothing else. So the earlier
suspicion -- that `MONOMER_DATASET`'s whole-chain design defaults left no
condition tokens -- is **false for these runs**: `featurize_structures` is
called with `binder_chain_ids`, which restricts the design region to one chain.

`inference_safe_binder` is likewise not a guessed flag: it is defined in
`pxdesign_train/runner/data.py:113` and documented as the leak-free four-atom
rebuild. Its downstream effect is real and local to the featurizer.

**Conclusion for the decision table: the failure is not missing or malformed
conditioning features.**

## 3. Runtime divergence is real and independently established

`PXDesign/install.sh` pins Protenix to `v0.5.0+pxd`; the published
`requirements.txt` says only `protenix>=0.1.0`, which identifies nothing. The
resolved commit is:

    protenix @ git+https://github.com/bytedance/Protenix.git@d18aa1da

This repo instead vendors Protenix at **`c3bfc36`** (`v2.0.0-11-gc3bfc36`),
deliberately, because `ATOM14` postdates `v0.5.0+pxd`.

The two revisions do not share an API. The working copy of the PXDesign
submodule carries an uncommitted patch to `pxdesign/model/embedders.py` that
converts

    self.atom_attention_encoder(input_feature_dict=input_feature_dict, ...)

into nine positional arguments, because `c3bfc36`'s `AtomAttentionEncoder.forward`
takes explicit tensors. That specific patch looks harmless -- the omitted
parameters (`r_l, s, z, p_lm, c_l`) are genuinely unused by
`InputFeatureEmbedder`. The concern is structural rather than local: only the
call site that *crashed* was adapted. A call site that silently accepts
defaults would not have announced itself.

## 4. Decision table

| | official runtime (`d18aa1da`) | local runtime (`c3bfc36`) |
|---|---|---|
| **released checkpoint** | loads strictly, 732/732; generation running | produced the invalid baseline |
| **adapted weights** | not attempted | not attempted |

The checkpoint cell is now measured rather than assumed
(`/hai/scratch/yfsun/pxdesign_official/check_ckpt_load.py`): against the
official `ProtenixDesign`, `pxdesign_v0.1.0.pt` gives 732 model tensors and 732
checkpoint tensors, **0 missing, 0 unexpected, 0 shape mismatches**, and
`load_state_dict(strict=True)` succeeds. A clean strict load is weak evidence
on its own, which is why the key sets and shapes are reported instead of a
boolean -- but there is nothing here to explain a broken baseline.

Checkpoint identity is settled: the donor at
`Proteo-AA-official-pxdesign-fampnn/runs/component_donors/pxdesign_v0.1.0.pt`
is 556,554,618 bytes, byte-for-byte the size the release CDN reports for
`release_model/pxdesign_v0.1.0.pt`. So the checkpoint is the released one, and
the remaining axis is the runtime.

## 5. Isolated official environment

Built at `/hai/scratch/yfsun/envs/pxdesign_official` (scratch, because home is
a hard 50 G quota), following `install.sh --cuda-version 12.1`. Deviations are
confined to paths: prefix env, conda/pip caches, `TMPDIR`, and `CUTLASS_PATH`
all on scratch, and the PXDesign source is a pristine clone of upstream
`f788441` rather than the patched working copy.

cu121 rather than cu124: torch 2.3.1 has no cu124 wheel (cu124 starts at 2.4),
and jax 0.4.29 requires cudnn < 9, which matches torch 2.3.1+cu121's
`nvidia-cudnn-cu12==8.9.2.26`.

Resolved versions (full list in
`/hai/scratch/yfsun/pxdesign_official/pip_freeze.txt`):

    torch==2.3.1+cu121          jax==0.4.29
    jaxlib==0.4.29+cuda12.cudnn91   numpy==1.26.3
    protenix @ ...@d18aa1da     pxdbench @ ...@f6d0d72
    transformers==4.51.3        dm-haiku==0.0.13
    optax==0.2.5                deepspeed==0.19.7
    biotite==1.0.1              colabdesign @ ...@e31a56fe
    nvidia-cudnn-cu12==8.9.2.26

All seven sanity imports pass on CPU. `PROTENIX_DATA_ROOT_DIR` points at a
cache holding only the released `components.v20240608.*` files, so
`configs_data`'s "prefer `components.cif`" branch falls through to the official
data rather than a newer locally built cache.


## 6. The official installer does not reproduce today

`install.sh` never pins deepspeed, and `requirements.txt` only bounds it below
(`deepspeed>=0.15.1`). It arrives as a Protenix dependency, and pip resolves it
today to **0.19.7**, which requires torch >= 2.4:

    deepspeed/compile/custom_ops/all_to_all.py:12
      @torch.library.custom_op("autosp::all_to_all", mutates_args=())
    AttributeError: module 'torch.library' has no attribute 'custom_op'

`torch.library.custom_op` was added in torch 2.4; the installer pins torch
**2.3.1**. This is not an optional path -- Protenix probes deepspeed with
`importlib.util.find_spec("deepspeed.ops.deepspeed4science")` in
`openfold_local/model/primitives.py`, and `find_spec` executes the package's
`__init__`, so *any* import of protenix dies.

Pinned **`deepspeed==0.15.4`** (satisfies the documented floor, predates the
`compile/custom_ops` module). This is the second under-specified dependency in
the published contract, after `protenix>=0.1.0`, and it is the kind of thing
that makes "the official installer" and "a working environment" different
objects. Recorded in `pip_freeze.txt`.

## 7. Official positive control: passes

`pxdesign infer` on the shipped `examples/PDL1_quick_start.yaml`, released
checkpoint, pristine `f788441`, no FaMPNN, no adapters, no refolding. Three
seeds (101/102/103) x 4 samples on an H200.

The checkpoint loaded `strict: True` inside the real runner, and every sample
came out as a plausible two-chain complex: chain A0 = 464 backbone atoms (the
116-residue cropped PD-L1), chain B0 = 320 (the 80-residue binder).

| | min BB-BB | clashes < 2.6 A | contacts < 5 A | centroid sep |
|---|---|---|---|---|
| seed 101 sample 0 | 2.968 A | 0 | 48 | 21.4 A |
| seed 101 sample 1 | 5.323 A | 0 | 0 | 19.3 A |
| seed 101 sample 2 | 5.054 A | 0 | 0 | 19.4 A |
| seed 101 sample 3 | 4.790 A | 0 | 1 | 21.0 A |
| ... 12 interfaces total | | **0** | | |

**12 of 12 interfaces show no interpenetration.** The local baseline, same
checkpoint, gave 0.207 A and 282 clashing pairs. So the released weights are
capable of producing plausible complexes, and the failure is on the runtime /
sampling side rather than the checkpoint.

(The `OSError: [Errno 16] Device or resource busy: '.nfs...'` after each seed
is NFS silly-rename cleanup in the dumper, after `succeeded` is logged and the
CIFs are written. It does not affect output.)

## 8. Sampler contract divergences found so far

| | official `pxdesign infer` | local `pxf/couple/replay.py` |
|---|---|---|
| `step_scale_eta` | **2.5** (`const`, from the CLI defaults) | **1.5** |
| `N_step` | 400 | 200 (`--n-step` default) |
| `gamma0` / `gamma_min` | 1.0 / 0.01 | 1.0 / 0.01 (match) |
| `noise_scale_lambda` | 1.003 | 1.003 (match) |
| `sigma_data`, `s_max`, `s_min`, `rho` | 16.0, 160.0, 0.0004, 7 | same |
| target coordinates | never touched | overwritten at 3 points per step |

`step_scale_eta` is the substantive one. `configs_base.py` defaults the design
model to `eta_schedule = {type: piecewise_65, min: 1.0, max: 2.5}`, and the
`pxdesign` CLI overrides that to a constant 2.5. The local sampler instead
carries 1.5, which is Protenix's generic default, not PXDesign's design
setting. The Euler step is scaled by this factor at every one of the steps, so
it is not a small discrepancy.

None of this yet proves which divergence produced the interpenetration; the
coordinate overwrite remains the leading candidate on mechanism. But eta is a
second, independent departure from the official contract and has to be
corrected before any local sampler output is trusted.

## 9. Root cause, measured

`check_target_pose.py` compares the target chain the official runner *emitted*
against the target chain it was *given* (PDL1 seed 101 sample 0, chain A0 vs
`5o45.cif` chain A, 116 paired CA):

| | |
|---|---|
| raw RMSD | **26.417 A** |
| superposed RMSD | **0.107 A** |
| centroid distance | 18.099 A |

The target's internal geometry is reproduced to 0.107 A, in a global frame
18 A and 26 A away from the input. This is exactly what section 1 predicts
from a rotation- and translation-invariant distogram, and it rules out the
alternative that preprocessing pins coordinates somewhere out of sight.

**So the generated target's pose is chosen by the model, not given to it.**
Pinning target coordinates to the native frame -- which the local sampler did
at three points per solver step -- forces the target into a frame the binder
was never generated against. The two halves are each internally sensible and
mutually misplaced, which is what 0.207 A minimum distance and 282 clashing
pairs look like.

The correction is to remove the overwrite, let the sampler place both chains,
and superpose onto the native target afterwards if a native-frame comparison
is wanted. `FixedTarget` should not be used in the generation path at all.

## 10. The PINDER target through the official runtime: passes

`pxdesign infer` on `1jfl` (target chain B, `binder_length: 228`,
`--use_msa false`), two seeds x 4 samples. The YAML validates under the
official `check-input`, and `parse-target` renders the task as
`{"condition": {"chain_id": ["B"]}, "generation": [{"type": "protein",
"length": 228, "count": 1}]}`.

| | min BB-BB | clashes < 2.6 A | contacts < 5 A | centroid sep |
|---|---|---|---|---|
| seed 101 s0 | 4.606 A | 0 | 3 | 28.0 A |
| seed 101 s1 | 2.675 A | 0 | 32 | 24.8 A |
| seed 101 s2 | 2.722 A | 0 | 121 | 23.4 A |
| seed 101 s3 | 2.854 A | 0 | 57 | 26.7 A |
| seed 102 s0 | 2.696 A | 0 | 127 | 24.2 A |
| seed 102 s1 | 2.645 A | 0 | 119 | 27.0 A |
| seed 102 s2 | 2.440 A | 1 | 153 | 23.2 A |
| seed 102 s3 | 3.211 A | 0 | 57 | 25.1 A |
| **native 1jfl** | **2.792 A** | **0** | **140** | **30.5 A** |

Seven of eight interfaces are clash-free, and the eighth has a *single* pair
at 2.44 A alongside 153 contacts -- a tight interface, not interpenetration.
The contrast with the local baseline is not marginal: 282 pairs under 2.6 A
with a 0.207 A minimum. Contact counts up to 153 bracket the native 140, so
these are real interfaces rather than chains parked next to each other.

Target pose on `1jfl` seed 101 sample 2 (228 paired CA against native chain B):
raw RMSD **83.567 A**, superposed RMSD **0.133 A**, centroid distance 79.9 A.
Same conclusion as section 9, more starkly.

## 11. Decision table, resolved

| | official runtime (`d18aa1da`) | local runtime (`c3bfc36`) |
|---|---|---|
| **released checkpoint** | **plausible complexes** (12/12 PDL1, 7/8 1jfl clash-free) | interpenetration (0.207 A, 282 pairs) |
| **adapted weights** | not attempted | not attempted |

The released checkpoint produces good complexes when driven by the official
runtime and the official sampling contract. It is therefore **not a checkpoint
failure**. The conditioning features built locally are also correct
(section 2). What remains is the sampling path, where two concrete departures
are now documented: the coordinate overwrite (section 9, the mechanism) and
`step_scale_eta` 1.5 vs 2.5 (section 8).

### What this does and does not license

It licenses removing `FixedTarget` from the generation path and correcting
eta. It does not yet license re-running the feedback comparison: neither
change has been shown *sufficient* to fix the local sampler, and the adapted
weights have never been tested for architecture/feature-contract compatibility
against this runtime. The next step is to port the recorder, residue mapping,
single-event injector, and paired randomness into this working runtime and
re-run the no-feedback replay and target-input/output checks there.

## 12. Reproducing

    # environment (once)
    bash /hai/scratch/yfsun/pxdesign_official/install_official.sh
    pip install "deepspeed==0.15.4"       # see section 6

    # positive control + target
    sbatch /hai/scratch/yfsun/pxdesign_official/run_example.sh
    TARGET=1jfl sbatch /hai/scratch/yfsun/pxdesign_official/run_target.sh

    # checks
    python /hai/scratch/yfsun/pxdesign_official/check_ckpt_load.py
    python /hai/scratch/yfsun/pxdesign_official/check_overlaps.py '<out>/**/*.cif'
    python /hai/scratch/yfsun/pxdesign_official/check_target_pose.py \
        <out.cif> <input.cif> <out_chain> <in_chain>
    python scripts/audit_conditioning.py --prepared configs/gen_stress_prepared.parquet

## 13. Porting assessment (step 4 feasibility)

`pxf` touches Protenix through seven symbols. Against the official
`v0.5.0+pxd` (`d18aa1da`), five import cleanly and two do not:

| symbol | in `v0.5.0+pxd` | consequence |
|---|---|---|
| `config.config.parse_configs` | yes | - |
| `data.utils.pdb_to_cif` | yes | - |
| `model.generator.sample_diffusion` | yes | the hook point, unchanged |
| `model.utils.centre_random_augmentation` | yes | - |
| `utils.seed.seed_everything` | yes | - |
| `data.constants.ATOM14` | **no** | side-chain *metrics* only (`pxf/eval/canonical.py`); not on the generation path |
| `model.protenix.update_input_feature_dict` | **no** | not needed there -- see below |

`update_input_feature_dict` exists only to precompute `d_lm`, `v_lm` and
`pad_info` for `c3bfc36`, whose `AtomAttentionEncoder.forward` takes them as
explicit tensors. The official encoder has the older signature:

    forward(self, input_feature_dict, r_l=None, s=None, z=None,
            inplace_safe=False, chunk_size=None)

and derives the atom-pair features internally -- `d_lm` and `pad_info` appear
nowhere in `v0.5.0+pxd`. So under the official runtime `prepare_features` is
dropped rather than reimplemented, and the working-copy `embedders.py` patch
becomes unnecessary (it exists solely to bridge to the newer signature).

`sample_diffusion`'s loop in `v0.5.0+pxd` is structurally identical to what
`pxf/couple/replay.py` transcribed -- augmentation, churn, denoise, Euler --
so the single-event injector has the same hook point. Two adjustments are
required: eta 1.5 -> 2.5, and deletion of the `FixedTarget` calls.

**The port is therefore tractable**, and the pieces that do not carry over are
either evaluation-only or artefacts of the newer Protenix. This says nothing
yet about whether the *adapted weights* are compatible with this runtime,
which remains untested and is a separate question from the code port.

## 14. Adapted-weights compatibility (the precondition, not the test)

Checked before any attempt to run adapted weights, since a clean load is not
evidence.

**Provenance chain closed by hash.** The adapter checkpoint
(`pxf_sb_pilot/full_118215`) records the donor it was trained against:

    sha256 b075867bae942dc0c6487173736922b0e2913308c1ba542d227418b6e176478d
    bytes  556554618

That is identical to the donor file on disk *and* to the copy staged into the
official environment. So the adapters were trained against the exact weight
file the official runtime just used to produce plausible complexes -- not a
same-sized lookalike.

**Dimensions agree with the official config.** The checkpoint records
`c_token: 768`, `sigma_data: 16.0`; the official run's `config.yaml` has
`diffusion_module.c_token: 768`, `sigma_data: 16.0`. The adapter tensors are
consistent with that: `sc_to_bb.project_out` is `(768, 256)` and
`bb_to_sc.norm` is `(768,)`.

**The tap points exist.** `BackboneTap` hooks `diffusion_module.layernorm_a`
and `diffusion_module.atom_attention_decoder`. Against the official
`ProtenixDesign`, all of `layernorm_a`, `atom_attention_decoder`,
`atom_attention_encoder`, `diffusion_transformer`, `sigma_data` (= 16.0) and
`diffusion_conditioning.relpe` are present.

**Why the module graphs are very likely identical, not merely similar.** The
released donor loads into the official model with 732 model tensors and 732
checkpoint tensors, 0 missing, 0 unexpected, 0 shape mismatches (section 5),
and the same file loads into the local runtime. Two runtimes accepting an
identical tensor set at identical shapes constrains the graphs tightly.

### What this still does not establish

Structural compatibility is not behavioural equivalence. None of the above
shows that `a_token` carries the same *semantics* under `d18aa1da` as under
`c3bfc36`, and the readout's inputs (186-dim readout, 250-dim projection) come
from the FaMPNN side, which this analysis does not touch. The precondition for
testing adapted weights is met; the test itself has not been run.

## 15. The port, and its no-op validation

`pxf/official/runtime.py` drives the official runtime with a recordable
sampler; `pxf/official/bridge.py` recovers `Topology`, the design mask and
per-token `aatype` from the official `AtomArray` plus feature dict, since the
official path produces neither `pxdesign_train`'s featurizer object nor its
keys. Design tokens are identified by PXDesign's own `xpb` marker rather than
by a chain index or a fraction, because that predicate *is* the featurizer's
definition of the condition region.

Three departures from the local path, all forced by section 9 and section 13:
no `fixed_target` anywhere, eta 2.5, and no `prepare_features`.

### Two things the first run got wrong

**The official baseline is entry-point dependent.** `configs_base` gives the
design model `eta_schedule = {type: piecewise_65, min: 1.0, max: 2.5}`, and
the `pxdesign` *CLI* overrides it to a constant 2.5. Calling `get_configs`
directly -- as the harness did -- silently inherits `piecewise_65`, so the
transcription check compared a constant-eta replay against a piecewise-eta
upstream and diverged by **49.65 A**. The runs that produced plausible
complexes (sections 7, 10) went through the CLI and therefore used const 2.5,
so that is the baseline the harness now sets explicitly. A `run_trajectory`
that took a *schedule* rather than a scalar would be needed to reproduce
`piecewise_65`; it currently cannot, and that is a real limitation.

**The numerical floor had never been measured at trajectory scale.** The
4.8e-6 figure on record was for a single denoiser call. Over 60 steps the same
computation run twice diverges by 1.3e-3, because each step's rounding feeds
the next through a chaotic map. Judging against the single-call figure made
all four checks "fail", including one -- a bypassed tap -- that is a
mathematical no-op and therefore *cannot* fail for a real reason. The harness
now measures the floor with a null replicate (check 0) instead of assuming it.

### Results, adapters disabled

| check | max deviation | verdict |
|---|---|---|
| 0 null replicate (identical computation twice) | **1.313e-03** | defines the floor |
| 1 transcription: upstream `sample_diffusion` vs `run_trajectory` | 2.683e-04 | PASS |
| 2 replay: uninterrupted vs record-at-30 then resume | 3.619e-04 | PASS |
| 3 hooks installed, feedback bypassed | 1.299e-03 | PASS |
| 4 one injected residual of exact zeros | 5.712e-04 | PASS |
| exactly one injection | 1 | PASS |

Recorded alongside: `sigma / c_tau_last = 2.0` exactly at the event step,
confirming the churn behaves as the module docstring claims and that the
*churned* level is what gets recorded; and `a_token` width 768, matching both
the official `diffusion_module.c_token` and the adapter tensors.

Check 1 is the load-bearing one. It says this repo's transcription of the
sampler loop reproduces upstream's own `sample_diffusion` to well below the
floor -- so the loop substituted for the official one is the official one, and
`fixed_target`'s removal did not quietly change the trajectory.
