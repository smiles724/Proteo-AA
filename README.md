# PXDesign-train

Training code for **PXDesign-d**, the diffusion generator at the core of ByteDance's
[PXDesign](https://github.com/bytedance/PXDesign). ByteDance released inference code
only; this repo reconstructs the training pipeline from
[`PXDesign/assets/technical_report.pdf`](../PXDesign/assets/technical_report.pdf)
(Appendix C, pp. 23–24).

This is the reproduction version, not the original training code. Numbers are taken verbatim from the report where stated;
gaps (e.g. exact featurizer behavior for hotspots, target-pair-distance binning) are
filled in with reasonable choices that the comments call out explicitly.

## Repo layout & setup

Protenix and PXDesign live inside this repo as **git submodules** pinned to
known-good commits:

```
PXDesign-train/
├── Protenix/        # bytedance/Protenix @ c3bfc36 (v2.0.0)        — submodule
├── PXDesign/        # bytedance/PXDesign @ f78844 + embedders patch — submodule + patch
├── pxdesign_train/  # this package
├── patches/
└── ...
```

The patch in [`patches/pxdesign-embedders-protenix-2.0.patch`](patches/pxdesign-embedders-protenix-2.0.patch)
adapts PXDesign's `InputFeatureEmbedder` to Protenix 2.0's new
`AtomAttentionEncoder(*tensors)` signature (the released PXDesign was built
against an older Protenix where the encoder accepted `input_feature_dict=...`).

One-shot setup:

```bash
git clone --recursive https://github.com/guanlueli/PXDesign-train.git
cd PXDesign-train
bash scripts/setup.sh        # applies the PXDesign patch
```

If you forgot `--recursive` on clone:

```bash
git submodule update --init --recursive
bash scripts/setup.sh
```

## Quick start: fine-tune on your own CIF files

```bash
cd PXDesign-train/
LAYERNORM_TYPE=torch \
PYTHONPATH="Protenix:PXDesign:." \
python scripts/smoke_test_gpu.py \
    --cif PXDesign/examples/5o45.cif \
    --binder_chain B \
    --crop_size 200 \
    --device cuda        # or "cpu" for a slow test
```

For a custom fine-tuning driver, see the example in [`scripts/finetune_demo.sh`](scripts/finetune_demo.sh).

## Training recipe (from PXDesign technical report Appendix C)

| Item | Value | Source |
|---|---|---|
| Loss | `L = 4·MSE + 1{σ<4Å}·(1.0·LDDT + 0.03·distogram)` | Eq. 4, p. 24 |
| Optimizer | Adam, lr=5e-4 | p. 24 |
| Crop size | 640 residues | p. 24 |
| Batch size | 64 (diffusion batch 8 per macro-batch) | p. 24 |
| Noise sampler | EDM log-normal, sigma_data=16, "same as Protenix" | p. 24 |
| Curriculum | Stage 1: AFDB/MGnify monomers; Stage 2: PDB complexes | p. 24 |
| From scratch | Not warm-started from Protenix base | p. 24 |
| Conditioning | Target pairwise distance bins (64+1 mask bin) + hotspot mask + restype | p. 23 |
| Target coords | **Noised and denoised like the rest** — *not* frozen | p. 23 |

## Layout

```
pxdesign_train/
├── generator.py            # TrainingNoiseSampler + sample_diffusion_training
├── heads.py                # Distogram heads (conditioning-z + diffusion-token)
├── model.py                # ProtenixDesignTrain: adds mode="train" forward
├── loss.py                 # PXDesignLoss: eq. 4 composite loss
├── configs/
│   └── configs_train.py    # crop 640, lr 5e-4, batch 64, diff_batch 8
├── data/
│   ├── _helpers.py         # Vendored PXDesign helpers (import-shim workaround)
│   ├── featurizer.py       # DesignFeaturizer: xpb, conditional_templ, hotspot
│   ├── cropper.py          # DesignCropper: binder-anchored 640-token crop
│   └── curriculum.py       # CurriculumSchedule + CurriculumSampler (stage 1→2)
└── runner/
    ├── data.py             # DesignSourceDataset (provider → crop → featurize)
    ├── providers.py        # ProtenixComplexProvider + binder selectors
    ├── cif_provider.py     # CifFileProvider: raw CIF → ComplexProvider
    ├── trainer.py          # PXDesignTrainer (EMA, DDP, AMP, grad accum)
    ├── finetune.py         # make_finetune_configs + finetune_from_components
    └── train.py            # train_from_components entry point

scripts/
├── setup.sh                # Clone Protenix + PXDesign at the right commits, apply patch
├── smoke_test_gpu.py       # One-step GPU end-to-end test with real model
├── train_demo.sh           # Training driver template
└── finetune_demo.sh        # Fine-tuning driver template

patches/
└── pxdesign-embedders-protenix-2.0.patch   # Adapts PXDesign to Protenix 2.0 API

tests/                      # 47 tests, all CPU-only
├── test_train_forward.py   # Loss + heads + noise sampler
├── test_design_featurizer.py
├── test_design_cropper.py
├── test_curriculum.py
├── test_trainer_integration.py
└── test_providers_and_finetune.py
```


## Data providers

Two providers ship with this repo:

### `CifFileProvider` — for small fine-tuning sets

Takes a list of CIF file paths. Parses each via Protenix's `MMCIFParser`,
featurizes with the Protenix base `Featurizer` (no MSA, no templates — dummy
zeros). Requires the CCD components file
(`download_tool_weights.sh` or `$PROTENIX_DATA_ROOT_DIR`).

```python
from pxdesign_train.runner.cif_provider import CifFileProvider
provider = CifFileProvider(
    cif_paths=["target1.cif", "target2.cif"],
    binder_chain_ids=["B", "C"],
)
```

### `ProtenixComplexProvider` — for large-scale training

Wraps a Protenix `BaseSingleDataset`. Construct it with `crop_size=0` in
`cropping_configs` so Protenix returns the full bioassembly — our
`DesignCropper` does the design-aware crop.

```python
from protenix.data.pipeline.dataset import BaseSingleDataset
from pxdesign_train.runner import ProtenixComplexProvider, select_protenix_chain_2

base = BaseSingleDataset(..., cropping_configs={"crop_size": 0, ...})
provider = ProtenixComplexProvider(base, binder_selector_fn=select_protenix_chain_2())
```

Binder selectors: `select_chain_by_id("B")`, `select_protenix_chain_2()`,
`select_smallest_protein_chain()`, `select_random_protein_chain(seed=42)`.

## Fine-tuning

```python
from pxdesign_train.runner import finetune_from_components, make_finetune_configs, TrainerComponents

configs = make_finetune_configs(base_configs, lr=1e-5, warmup_steps=200, max_steps=5_000)
finetune_from_components(
    configs, components,
    load_checkpoint_path="pxdesign_v0.1.0.pt",  # load_strict=False auto-set
    checkpoint_dir="./finetune_ckpts",
)
```

## Tests

```bash
cd PXDesign-train/
LAYERNORM_TYPE=torch \
PYTHONPATH="Protenix:PXDesign:." \
python -m pytest tests/ -v
# 47 passed in ~5s
```

## Environment notes

- Protenix's fused LayerNorm JIT-compiles CUDA kernels at import. Set
  `LAYERNORM_TYPE=torch` to fall back to `torch.nn.LayerNorm` (slower but
  works on CPU and machines without `ninja`).
- PXDesign's released data modules import `from protenix.data import ccd`,
  but the local Protenix exposes that at `protenix.data.core.ccd`. We work
  around this with vendored helpers in `pxdesign_train/data/_helpers.py`.
  No monkey-patching needed in user code.
