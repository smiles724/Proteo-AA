# ProteoAA

**Full-atom protein co-design** — generate backbone coordinates, residue sequence,
and side-chain atoms *together* in one residue-aware diffusion process.

Two coupled modules communicate through a shared per-residue representation `h_res`:

- **Backbone–AA module** — denoises backbone coordinates and predicts residue type
  via masked (absorbing) discrete diffusion, co-generating `(backbone, sequence)`
  from noise in a single reverse-diffusion pass (no external inverse-folding step).
- **Side-Chain module (S_φ)** — reads the predicted backbone / AA logits / `h_res`,
  dynamically instantiates the residue-specific atom set (no ghost atoms), and
  predicts side-chain coordinates in residue-local frames from a **one-step,
  leakage-free Gaussian init**; it then pools a side-chain-aware `h_res′` back to the
  Backbone module for backbone/type refinement (co-evolution).

Built on **PXDesign-d** (structure diffusion) and **Protenix** (ByteDance;
git submodules), extending the [`guanlueli/PXDesign-train`](https://github.com/guanlueli/PXDesign-train)
reproduction. Our training layer lives in [`pxdesign_train/`](pxdesign_train/).

---

## Method → code (mapped to the paper's stages)

| Stage | What | Status |
|---|---|---|
| **I — Backbone-AA** | coord diffusion + masked-diffusion residue type; AA head reads the structure-aware `a_token` | runnable |
| **· per-σ AA loss** | AA cross-entropy computed **per noise level (σ) then averaged**, not reduce-then-predict | done |
| **II-A — side-chain warmup** | one-step Gaussian `S_φ` on GT frames; `L_sc^local` + physical; backbone frozen (`trunk_grad_scale=0`) | done |
| **II-B — co-evolution** | `S_φ` on **predicted-backbone** frames `F̂` (from `x̂₀`) + stop-grad global pseudo-target; `h_res′` → reuse `B_θ` to refine | wiring done (smoke) |
| **III — predicted-mask** | atom set from the **predicted** residue type; coord loss on type-matched residues, physical elsewhere; makes `post_aa` safe | done |

**Status: engineering prototype, single-structure GPU smoke — not method-validated.**
Both `--sidechain_warmup` and `--coevolution` run end-to-end on GPU (`sc_local` drops,
losses finite, no shape/leakage issues). See [`reports/`](reports/) for the full audit
and honest per-piece grading.

**Leakage safeguards.** Side chains initialise from Gaussian noise (never noised GT);
binder side chains are excluded from `L_bb` *and* scrubbed (→ Cα) from the diffusion
input, so the backbone never sees GT side-chain geometry; `post_aa` is supervised only
under predicted-mask, so GT atom composition cannot leak identity into the AA head.

**Not yet done** (left for the training phase): physical bond/angle/rotamer activation
(needs a residue-specific geometry table); Stage III `L_SC-AA` candidate ranking (core
only, not orchestrated); *strict* per-σ cycle feedback (`h_res′` is σ-averaged before
injection because Protenix's `s_trunk` is sample-shared); multi-structure / generalization.

---

## Why the AA head reads `a_token`

The residue-type head needs a **structure-aware** input. It reads `a_token` — captured
(via a forward hook, no submodule edit) *after* the DiffusionModule's token
self-attention — because it has cross-token context **and** is conditioned on the
binder's own noisy backbone, unlike the structure-blind `s_inputs`/`s_trunk`. This same
representation is the shared `h_res` handed to `S_φ`. Because training draws `N_sample`
noise levels per item, the AA loss is computed **per σ and averaged** (a Monte-Carlo
estimate over the σ distribution), matching how the coordinate MSE and inference-time
sampling treat the σ axis.

---

## Setup

```bash
git clone --recursive <this-repo-url>   # Protenix + PXDesign submodules
# apply the PXDesign↔Protenix embedders patch (see PXDESIGN_TRAIN_README.md)
pip install -r requirements.txt
```

## Usage

```bash
# CPU unit tests
LAYERNORM_TYPE=torch PYTHONPATH="Protenix:PXDesign:." python -m pytest tests/ -q

# Stage II-A side-chain warmup (one-step Gaussian completion)
python scripts/finetune_mini.py --sidechain_warmup \
  --cif PXDesign/examples/5o45.cif --binder_chain B --ckpt <pxdesign_v0.1.0.pt>

# Stage II-B co-evolution (per-σ + predicted-frame + h_res′ cycle)
python scripts/finetune_mini.py --coevolution \
  --cif PXDesign/examples/5o45.cif --binder_chain B --ckpt <pxdesign_v0.1.0.pt>

# Joint co-generation (backbone + sequence + full-atom side chains) from noise
python scripts/finetune_mini.py --cogenerate --sc_cycle \
  --cif PXDesign/examples/5o45.cif --binder_chain B --ckpt <pxdesign_v0.1.0.pt>
```

## Previous version

The backbone-AA-only version (masked-diffusion co-design, before the side-chain module)
is preserved on the **`backbone`** branch and tag **`baseline-aa-masked-diffusion`**.

## Attribution & license

`PXDesign` and `Protenix` are **ByteDance's** (see each submodule's `LICENSE`);
the coordinate-diffusion reproduction is **guanlueli/PXDesign-train**. Submodules are
referenced as commit pointers — no ByteDance code or weights are re-hosted. Please cite
PXDesign and Protenix, and credit the reproduction, when using this code.
