# ProteoAA

**Full-atom protein co-design** built on **PXDesign-d** + **Protenix**. This branch
adds side-chain co-design on top of the
[`guanlueli/PXDesign-train`](https://github.com/guanlueli/PXDesign-train)
reproduction: global side-chain denoising, template-anchored initialization, and
side-chain feedback into the backbone module.

**Status:** engineering prototype. Single-structure GPU smoke runs; real-data method
validation is still pending. Details and caveats live in [`docs/`](docs/).

## Change map

| Change | Start here |
|---|---|
| **New framing loss** — predicted-frame-aligned side-chain coordinate supervision | [`pxdesign_train/sidechain/losses.py`](pxdesign_train/sidechain/losses.py), [`pxdesign_train/loss.py`](pxdesign_train/loss.py), [`tests/test_sidechain_losses.py`](tests/test_sidechain_losses.py) |
| **Add ideal-template init** — `mu_ideal` + Dunbrack/BuildSC instead of pure Gaussian init | [`pxdesign_train/sidechain/init.py`](pxdesign_train/sidechain/init.py), [`templates.py`](pxdesign_train/sidechain/templates.py), [`rotamers.py`](pxdesign_train/sidechain/rotamers.py), [`buildsc.py`](pxdesign_train/sidechain/buildsc.py) |
| **Add a/q interconnection** — side-chain feedback into backbone token/atom streams | [`pxdesign_train/sidechain/coevolution.py`](pxdesign_train/sidechain/coevolution.py), [`pxdesign_train/sidechain/module.py`](pxdesign_train/sidechain/module.py), [`pxdesign_train/model.py`](pxdesign_train/model.py) |
| **Add ablation arms** — `no`, `a-indirect`, `a-direct`, `bbctx`, `q`, `a-direct+q` | [`pxdesign_train/configs/configs_train.py`](pxdesign_train/configs/configs_train.py), [`scripts/finetune_mini.py`](scripts/finetune_mini.py), [`tests/test_ablation_arms.py`](tests/test_ablation_arms.py) |

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
