# ProteoAA

**Full-atom protein co-design** — jointly model backbone coordinates, residue
sequence, and side-chain geometry in one residue-aware diffusion process.
Inference returns backbone + sequence + `S_φ` side-chain coordinates (full assembled
PDB/tensor is still pending).

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
| **II-A — side-chain warmup** | one-step Gaussian `S_φ`; GT frames + GT atom masks; `L_sc^local` + physical; gradient-isolated (`trunk_grad_scale=0`) | implemented |
| **II-B — co-evolution** | `S_φ` on **predicted-backbone** frames `F̂` (from `x̂₀`) + stop-grad global pseudo-target; `h_res′` → reuse `B_θ` to refine | wiring / smoke done; full recurrent feedback pending |
| **III — predicted-mask** | atom set from the **predicted** residue type; coord/physical routing; makes `post_aa` safe | partial — core implemented, **default off** |

**Status: engineering prototype, single-structure GPU smoke — not method-validated.**
Both `--sidechain_warmup` and `--coevolution` run end-to-end on GPU (`sc_local` drops,
losses finite, no shape/leakage issues). See [`docs/method_status.md`](docs/method_status.md)
for the honest per-stage grading.

**Leakage safeguards.** Side chains initialise from Gaussian noise (never noised GT);
binder side chains are excluded from `L_bb` *and* scrubbed (→ Cα) from the diffusion
input, so the backbone never sees GT side-chain geometry; `post_aa` is supervised only
under predicted-mask, so GT atom composition cannot leak identity into the AA head.

**Not yet done** (left for the training phase): physical bond/angle/rotamer activation
(needs a residue-specific geometry table); Stage III `L_SC-AA` candidate ranking (core
only, not orchestrated); *strict* per-σ cycle feedback (`h_res′` is σ-averaged before
injection because Protenix's `s_trunk` is sample-shared); multi-structure / generalization.

---


## Setup

```bash
git clone --recursive <this-repo-url>          # pulls Protenix + PXDesign submodules
pip install -e .                               # this package (torch, numpy)
pip install -r Protenix/requirements.txt       # Protenix deps
pip install -r PXDesign/requirements.txt       # PXDesign deps
bash scripts/setup.sh                          # applies the PXDesign↔Protenix embedders patch (required)
```

The `scripts/setup.sh` patch step is **required** — without it the PXDesign↔Protenix-2.0
embedder shapes mismatch. [`PXDESIGN_TRAIN_README.md`](PXDESIGN_TRAIN_README.md) is the
**upstream `guanlueli/PXDesign-train` reproduction note** (its clone URLs point upstream),
kept for the manual patch, CCD-cache, and server details.

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

# Joint co-generation (backbone + sequence + S_φ side-chain coordinates) from noise
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
