# Backbone module

**Sequence–structure co-design by adding a residue-type masked discrete diffusion
on top of PXDesign-d.**

PXDesign's original pipeline is decoupled: **PXDesign-d** generates a backbone,
**ProteinMPNN** does inverse folding to a sequence, and **AF2** refolds to filter.
This repo turns the structure-only generator into a **co-design model** that, in a
single reverse-diffusion pass, generates a backbone **and** a residue-type sequence
together.

---

## Lineage & attribution

This repository is a **derivative/extension**. It is built on, and re-uses, prior work:

| Layer | Source | Role here |
|---|---|---|
| Structure generator **PXDesign-d** + inference | [bytedance/PXDesign](https://github.com/bytedance/PXDesign) | git submodule (pointer only) |
| Structure-prediction backbone **Protenix** | [bytedance/Protenix](https://github.com/bytedance/Protenix) | git submodule (pointer only) |
| **Training-pipeline reproduction** of PXDesign-d | [guanlueli/PXDesign-train](https://github.com/guanlueli/PXDesign-train) | **base repo** — this fork is built on it |
| **This work** | pxdesign_train/ additions + tests | audit + residue-type masked diffusion (below) |

The upstream reproduction's own README is preserved verbatim as
[`PXDESIGN_TRAIN_README.md`](PXDESIGN_TRAIN_README.md). Submodules are referenced as
commit pointers — no ByteDance code or weights are re-hosted here.

---

## What this work contributes

### 1. Audit of the reproduction (verify before extending)
Before building on it, the `guanlueli/PXDesign-train` reproduction was audited and
run: the coordinate-diffusion pipeline is structurally faithful and **generates
coordinates**, and the official checkpoint loads cleanly
(`load_strict=False` → missing keys = exactly our new head params, unexpected = 0).
**Conclusion: usable as the backbone-module base for co-design.**

### 2. Residue-type masked discrete diffusion → co-design
On top of PXDesign-d's coordinate diffusion, a **masked (absorbing) discrete
diffusion for residue type** is added and integrated across
data / model / loss / trainer / config (66 tests):

- absorbing masking over residue type (design positions become `[xpb]`),
- a sinusoidal timestep (`aa_t`) embedding in the AA head,
- an MDLM time-weighted masked cross-entropy loss,
- **joint co-generation** ([`pxdesign_train/cogenerate.py`](pxdesign_train/cogenerate.py)):
  an EDM reverse loop that each step denoises coordinates, reads a structure-aware
  token representation, predicts residue types, and progressively unmasks —
  **co-generating `(backbone, sequence)` from noise in one pass**, without an
  external inverse-folding step.

### 3. Side-Chain Module (S_φ) — engineering prototype (smoke-tested)

A residue-aware side-chain branch (`pxdesign_train/sidechain/`) that reads the
shared residue representation + predicted AA logits, decodes side-chain atoms in a
residue-local frame from Gaussian init (one-step, leakage-free), pools them back to
an updated `h_res′`, and (in the cycle) runs a **second denoise pass with `h_res′`
injection** (not a closed refinement of the first pass — see below).

**Side-chain warmup and cycle wiring are implemented / being updated. The per-σ
aligned side-chain/cycle path is the intended joint-training design. Current
results are smoke tests / single-structure overfit, not method validation.** (No
claim of: co-evolution fully validated · Stage III complete · leakage-free fully
solved · full-atom design quality proven.)

Status, graded honestly (see `reports/` for the full audit):

| Piece | Status |
|---|---|
| One-step Gaussian init + local-frame `L_sc` | implemented, tested |
| Dynamic per-residue atom instantiation (no ghost atoms) | implemented, tested |
| `h_res′` atom→residue feedback + injection into `B_θ` | implemented, smoke-tested |
| **B_θ backbone-only / S_φ owns side chains** (M1) | binder side chains excluded from `L_bb` **and** scrubbed (→ Cα) from the diffusion input, so B_θ never sees GT side-chain geometry |
| **Per-σ aligned side-chain path** | S_φ reads a per-σ `h_res`/`aa_logits`/`sigma` (flattened `[B·N_sample, L, C]`), **not** a mean/low-σ-reduced `h_res`; each S_φ row = one specific σ, and its real σ feeds the time embedding. This is the intended joint-training design. |
| Stage II-A warmup | single reduced-`h_res` baseline + `trunk_grad_scale=0` (side-chain loss can't update the backbone). Labeled as warmup/completion — **not** per-σ co-evolution. |
| Cycle feedback (`h_res′` → B_θ) | **wiring implemented / being updated.** `s_trunk` is sample-shared in the Protenix diffusion, so `h_res′` is injected **σ-reduced** (true per-σ feedback needs a per-sample `s_trunk` = submodule change). Smoke-tested; `post_aa` stays gated (M2). |
| physical loss: clash + contact | implemented **and activated** |
| physical loss: bond / angle / rotamer | implemented, **not activated** — needs a vetted residue-specific ideal-geometry table (bond lengths / angles / χ definitions) |
| **Predicted-backbone frames in II-B** (paper Stage II-B) | side-chain frames `F̂` built from `x̂₀` (`x_denoised`) via gathered N/CA/C, **plus** a stop-grad global pseudo-target aux loss (`weight_sc_global`). Warmup (II-A) still uses GT frames. |
| **Stage III predicted-mask branch** | atom set instantiated from the **predicted** residue type; coord loss routed to type-matched residues, physical elsewhere; this is what makes `post_aa` safe to re-enable |
| type-mismatch loss routing (`route_sidechain_loss` module) | still a skeleton; the model routes coord/physical directly via `sc_type_match` |
| Stage III `L_SC-AA` candidate ranking | core implemented + unit-tested, **not integrated** (needs per-candidate S_φ orchestration) |
| iterative termination on sequence stability (paper) | implemented as an optional `cogenerate` early-stop (`stop_on_seq_stable`) |
| Full-atom side-chain output at inference (M3) | `cogenerate` returns S_φ side-chain coords |

**Boundaries that must stay explicit:**
1. **The cycle *feedback* is a conservative smoke-test, not fully per-σ co-evolution.**
   `h_res′` must be σ-averaged before injecting into Protenix's sample-shared
   `s_trunk`, so the feedback is not per-σ (true per-σ feedback needs a submodule
   change). The S_φ *forward* and frames ARE per-σ / predicted-backbone.
2. **Physical bond/angle/rotamer are not activated** (need a vetted geometry table),
   and **`L_SC-AA` candidate ranking is not integrated**.

**Deliberately NOT claimed:** method validated · co-evolution verified · fully per-σ
cycle feedback · Stage III complete · generalization / design quality. `post_aa` is
supervised **only** when `sidechain.predicted_mask=True` (atom set from predicted
type); it stays off by default so GT atom composition cannot leak identity into the
AA head. Predicted-backbone-frame training is implemented but **only smoke-tested**,
not validated.

---

## Design: where the AA head reads its information, and why

An AA head predicting masked residue types needs a **structure-aware** input. Three
candidate representations exist in this codebase:

| Representation | Cross-token? | Sees the binder's own noisy backbone (`r_noisy`)? | Verdict |
|---|---|---|---|
| `s_inputs` (449-d) | ❌ per-token linear, no attention | ❌ | baseline / ablation |
| learned `s_trunk` (Pairformer on `s_inputs + z`) | ✅ | ❌ (`z` is only target pairwise; `r_noisy` absent) | structure-blind |
| **`a_token`** (after full token-level self-attention) | ✅ | ✅ | **used** |

**We read `a_token`.** It has across-token context **and** is conditioned on the
binder's own noisy backbone and the target, so residue prediction carries the
conformational information of the design site itself (the way ProteinMPNN takes a
backbone as input for inverse folding).

### Where `a_token` is
`a_token` lives **inside `DiffusionModule`**
(`Protenix/protenix/model/modules/diffusion.py`). The flow:

```
atom_attention_encoder(r_noisy)  →  a_token
   → + s conditioning
   → DiffusionTransformer         # full token-level self-attention, conditioned on s and z
   → layernorm_a                  # ← we read a_token HERE (via a forward hook, no submodule edit)
   → atom_attention_decoder       → coordinate update
```

We capture `a_token` **after** the token self-attention (at `layernorm_a`), before it
is consumed by the coordinate decoder. Selected via
`residue_type.input_source=diffusion_internal` (default); `s_inputs` remains as an
ablation switch.

### h_res handoff
`a_token` is a **structure-aware, residue-level state** — exactly the kind of
representation a shared `h_res` (backbone ↔ side-chain interface) will need. The AA
head's actual input is exposed as `h_res_candidate`, pre-wiring the handoff for a
future side-chain branch.

### Gradient coupling — measured, not assumed
Reading `a_token` means the AA head's gradient flows back into the whole diffusion
backbone, which could perturb structure training. A **fixed-σ, multi-seed (n=16)**
coordinate evaluation (which removes the noise that dominates single-run coord MSE)
shows the coordinate error at σ∈{1,4,16} is **statistically identical** with full
coupling (`trunk_grad_scale=1.0`), stop-grad (`0.0`), and the `s_inputs` baseline —
i.e. **structure-aware AA does not cost coordinate quality**. A
`trunk_grad_scale` knob is kept for re-checking at the multi-structure stage.

### Per-sigma AA loss (not reduce-then-predict)
Training draws `N_sample` structure-noise levels (σ) per item, so `a_token` is
`[…, N_sample, N_token, c]`. The AA loss is computed **per sample (per σ) and then
averaged** — it does **not** average/select the representation before the head.
Averaging representations across σ would blur the conditioning; selecting the
lowest-σ sample would bias the head toward clean-structure inverse folding and
mismatch inference, where `cogenerate` queries the head across the whole σ
trajectory. Per-σ loss is a Monte-Carlo estimate over the EDM σ distribution and
mirrors how the coordinate MSE is already averaged over samples. `internal_reduce`
(`mean`/`low_sigma`) now only picks the single reduced state that `h_res` / the
side-chain module consume — it no longer affects the AA training target.

---

## Results (official checkpoint, GPU, 5o45 chain B, crop 256)

- **Load + fine-tune** (500 steps): `aa_ce 3 → ~0.05`, `aa_acc → 1.0`; coordinate
  losses keep training; official backbone loads cleanly.
- **Co-generation** after fine-tune: coordinates `(1170, 3)` + a **61-residue
  sequence**, **recovery vs GT = 0.375** (vs 0.0 with an untrained head), with a
  **confidence trajectory that rises as σ decreases** (0.36 → peak 0.51) — the
  intended co-design dynamic (sequence sharpens as structure clarifies).
- **Fixed-σ coord eval**: coupling does not degrade structure (see above).

> **Status / limitations.** All positive numbers are on a **single structure with a
> tiny binder** (`aa_mask_frac ≈ 0.016`) — this validates *wiring + end-to-end
> generation + learnability*, i.e. **memorization, not generalization**. Multi-structure
> held-out evaluation is the next step. `cogenerate` is a minimal deterministic sampler.

---

## What's this work's, vs inherited

**Added / substantially changed here** (`git diff 7cfd3e7..HEAD`):

```
pxdesign_train/cogenerate.py            # joint sequence–structure co-generation (new)
pxdesign_train/sampler.py               # iterative-unmask residue sampler (new)
pxdesign_train/heads.py                 # DesignResidueTypeHead + aa_t time embedding
pxdesign_train/loss.py                  # MDLM time-weighted masked-CE AA term
pxdesign_train/model.py                 # a_token hook, h_res_candidate, predict_aa, grad-scale
pxdesign_train/configs/configs_train.py # residue_type + AA-loss config
pxdesign_train/runner/{trainer,train,data,cif_provider}.py
pxdesign_train/data/featurizer.py       # residue corruption / [xpb] / no leakage
scripts/{finetune_mini,ckpt_load_check,smoke_test_gpu}.py
tests/test_aa_masked_diffusion.py       # + existing reproduction tests
```

Everything else — the coordinate-diffusion training pipeline and the `Protenix` /
`PXDesign` submodules — is inherited (see the lineage table).

---

## Setup

```bash
git clone --recursive <this-repo-url>
cd proteoaa
# apply the PXDesign↔Protenix-2.0 embedders patch (see PXDESIGN_TRAIN_README.md)
bash scripts/setup.sh   # or the manual patch step documented upstream
pip install -r requirements.txt   # torch, pytest, ...
```

Submodule pins: `Protenix @ c3bfc36`, `PXDesign @ f78844` + embedders patch. See
[`PXDESIGN_TRAIN_README.md`](PXDESIGN_TRAIN_README.md) for the reproduction-side setup
details, CCD cache, and server notes.

## Usage

```bash
# CPU unit tests
LAYERNORM_TYPE=torch PYTHONPATH="Protenix:PXDesign:." python -m pytest tests/ -q

# mini fine-tune from the official checkpoint, then co-generate
python scripts/finetune_mini.py --cogenerate --max_steps 500

# fixed-σ coordinate evaluation (does gradient coupling hurt structure?)
python scripts/finetune_mini.py eval_coord_fixed_sigma --n_seed 16
```

---

## License & citation

This is a research extension of third-party work. **`PXDesign` and `Protenix` are
ByteDance's** (see each submodule's `LICENSE`); the coordinate-diffusion
reproduction is **guanlueli/PXDesign-train**. Please cite ByteDance's PXDesign and
Protenix, and credit the reproduction, when using this code. Only the additions
listed above are contributed by this repository's authors.
