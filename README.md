# ProteoAA

**Full-atom protein co-design** — jointly model backbone coordinates, residue sequence,
and side-chain geometry in one residue-aware diffusion process. Two modules communicate
through a shared per-residue representation `h_res`:

- **Backbone–AA module** co-generates `(backbone, sequence)` from noise in one
  reverse-diffusion pass (masked discrete diffusion for residue type; no external
  inverse-folding step).
- **Side-Chain module (S_φ)** instantiates the residue-specific atom set (no ghost atoms)
  and predicts global side-chain coordinates from a leakage-free, template-anchored init,
  then feeds a side-chain-aware `h_res′` back for backbone/type refinement (co-evolution).

Built on **PXDesign-d** + **Protenix** (ByteDance; git submodules), extending the
[`guanlueli/PXDesign-train`](https://github.com/guanlueli/PXDesign-train) reproduction.

## What's new

Over the reproduction baseline this branch adds the full side-chain co-design layer:

- **Global side-chain output + template-anchored init** — S_φ emits global coordinates
  (Overleaf par.204/256) and starts from a residue-type ideal template posed at a
  **backbone-dependent rotamer** (Dunbrack BBDEP2010), not from Gaussian noise (par.221).
- **Interconnection between Backbone and Side-Chain modules** — side-chain → backbone
  feedback at two levels: token-level `a'_bb = a_bb + MLP([a_bb, a_sc])` and atom-level
  `q'_bb = q_bb + MLP([q_bb, q_sc])`, with a **six-arm ablation harness**
  (`--sc_ablation_arm`) to isolate each channel.
- **Receptor-aware S_φ** — cross-residue attention and the physical terms now see
  receptor / motif / ligand atoms, not just the binder.
- **Correctness** — Overleaf par.221/256 conformance, plus 16 blocking bugs fixed that had
  left the side-chain training path unable to run at all.

**Status:** engineering prototype — runs end-to-end on single-structure GPU smoke, not yet
method-validated. Per-stage grading and design rationale in
[`docs/`](docs/) ([`method_status.md`](docs/method_status.md),
[`sidechain_config_notes.md`](docs/sidechain_config_notes.md)).

## Where to look

| What changed | Main code |
|---|---|
| **New framing loss** — predicted-frame-aligned side-chain coordinate supervision | [`pxdesign_train/sidechain/losses.py`](pxdesign_train/sidechain/losses.py), [`pxdesign_train/loss.py`](pxdesign_train/loss.py), [`tests/test_sidechain_losses.py`](tests/test_sidechain_losses.py) |
| **Ideal-template init** — `mu_ideal` + Dunbrack/BuildSC template instead of pure Gaussian init | [`pxdesign_train/sidechain/init.py`](pxdesign_train/sidechain/init.py), [`templates.py`](pxdesign_train/sidechain/templates.py), [`rotamers.py`](pxdesign_train/sidechain/rotamers.py), [`buildsc.py`](pxdesign_train/sidechain/buildsc.py) |
| **Backbone-SideChain interconnection** — direct `a`/`q` side-chain feedback into the backbone module | [`pxdesign_train/sidechain/coevolution.py`](pxdesign_train/sidechain/coevolution.py), [`pxdesign_train/sidechain/module.py`](pxdesign_train/sidechain/module.py), [`pxdesign_train/model.py`](pxdesign_train/model.py) |
| **Ablation arms** — `no`, `a-indirect`, `a-direct`, `bbctx`, `q`, `a-direct+q` | [`pxdesign_train/configs/configs_train.py`](pxdesign_train/configs/configs_train.py), [`scripts/finetune_mini.py`](scripts/finetune_mini.py), [`tests/test_ablation_arms.py`](tests/test_ablation_arms.py) |
| **Status / caveats** — what is smoke-tested vs. still waiting for real-data comparison | [`docs/method_status.md`](docs/method_status.md), [`docs/sidechain_config_notes.md`](docs/sidechain_config_notes.md) |

## Repository layout

| Path | What |
|---|---|
| [`pxdesign_train/`](pxdesign_train/) | training layer — model, losses, data, side-chain module, configs, runner |
| [`scripts/`](scripts/) | entry points — fine-tune driver, rotamer/template builders, eval |
| [`tests/`](tests/) | unit + regression tests (CPU-runnable) |
| [`docs/`](docs/) | design notes |
| [`patches/`](patches/) | PXDesign↔Protenix embedders patch (applied by `scripts/setup.sh`) |
| `Protenix/`, `PXDesign/` | ByteDance submodules (commit pointers only) |

## Setup

```bash
git clone --recursive <this-repo-url>          # Protenix + PXDesign submodules
pip install -e . && pip install -r Protenix/requirements.txt -r PXDesign/requirements.txt
bash scripts/setup.sh                          # PXDesign↔Protenix embedders patch (required)
python scripts/build_rotamer_library.py --download   # Dunbrack table for the side-chain template
```

## Usage

```bash
# CPU tests
LAYERNORM_TYPE=torch PYTHONPATH="Protenix:PXDesign:." python -m pytest tests/ -q

# Side-chain warmup / co-evolution / joint co-generation
python scripts/finetune_mini.py --sidechain_warmup --cif <x.cif> --binder_chain B --ckpt <ckpt.pt>
python scripts/finetune_mini.py --coevolution      --cif <x.cif> --binder_chain B --ckpt <ckpt.pt>
python scripts/finetune_mini.py --cogenerate --sc_cycle --cif <x.cif> --binder_chain B --ckpt <ckpt.pt>

# Ablation arms: no | a-indirect | a-direct | bbctx | q | a-direct+q
python scripts/finetune_mini.py --coevolution --sc_ablation_arm q --cif <x.cif> --binder_chain B --ckpt <ckpt.pt>
```

## Attribution & license

`PXDesign` and `Protenix` are **ByteDance's** (see each submodule's `LICENSE`); the
coordinate-diffusion reproduction is **guanlueli/PXDesign-train**. Submodules are commit
pointers — no ByteDance code or weights are re-hosted.

The side-chain template uses the **Dunbrack BBDEP2010** rotamer library (redistributed
under ODC-By). If you publish results computed with it, cite: Shapovalov, M.V. & Dunbrack,
R.L. Jr. (2011), *A smoothed backbone-dependent rotamer library…*, **Structure** 19, 844–858.
