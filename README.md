# PXF — PXDesign backbone × FaMPNN side-chain packing

Full-atom protein structures from two published models, each used whole:

| Module | Component | What it owns |
|---|---|---|
| **Backbone** | [PXDesign](https://github.com/bytedance/PXDesign) `pxdesign_v0.1.0` | every backbone atom (N, CA, C, O) |
| **Side chain** | [FaMPNN](https://github.com/richardshuai/fampnn) `fampnn_0_0` | every side-chain atom |

The sequence is an **input**, not an output. FaMPNN is run through its own
`sidechain_pack` path — the one behind `fampnn/inference/pack.py` — with every
residue identity supplied, so its only job is to infer side-chain conformations.
It does not design sequence, and the pipeline verifies that it didn't.

Neither network is reimplemented. PXDesign runs through its own
`InferenceRunner` (official model, weights, featurizer and config) and FaMPNN
through its own `SeqDenoiser`. The code here is the boundary between them, plus
the contract checks that keep that boundary honest.

## How the two modules meet

PXDesign emits a flat atom list (`[N_sample, N_atom, 3]` plus
`atom_to_token_idx`); FaMPNN wants a dense `[B, L, 37, 3]` block per residue.
Both use the same AF2 atom37 order, so the handoff is a densification in
tensors — no reordering and no PDB round-trip.

```
PXDesign ──► ragged atoms ──► dense atom37 ──► FaMPNN ──► full-atom PDB
 (backbone)                    pxf/bridge.py    (packing)   psce in B-factor
                                     ▲
                     sequence ───────┘  native where known,
                                        supplied for design tokens
```

See [`docs/architecture.md`](docs/architecture.md) for the detail.

## Setup

```bash
git clone --recursive <this-repo>
bash scripts/setup.sh          # submodules + the one PXDesign patch + contract check
pip install -e . -r Protenix/requirements.txt -r PXDesign/requirements.txt
```

FaMPNN's weights ship inside its own submodule (`fampnn/weights/`), so there is
nothing to download for the side-chain module. PXDesign fetches
`pxdesign_v0.1.0.pt` on first run, or point `--pxdesign-checkpoint-dir` at an
existing copy.

## Usage

**Pack side chains onto existing backbones**, using each structure's own
sequence — the side-chain module alone:

```bash
python scripts/pack.py --pdb-dir fampnn/data/casp15/pdbs --out packed/
```

**Generate backbones and pack them** — the full pipeline. Any residue PXDesign
generates needs an identity, so supply one for the designed region:

```bash
python scripts/design.py \
    --input-json <pxdesign_input.json> \
    --pxdesign-checkpoint-dir <dir holding pxdesign_v0.1.0.pt> \
    --out designs/ --n-sample 8 --seed 0 \
    --sequence-fasta binder.fasta
```

Output is one PDB per structure (FaMPNN's per-side-chain-atom confidence `psce`
in the B-factor column) plus a `manifest.json` recording the revision and
SHA-256 of every weight and source tree that produced it.

**Train, or continue training, the side-chain modules.** FaMPNN ships inference
only ([upstream issue #9](https://github.com/richardshuai/fampnn/issues/9) is
unanswered), so the objectives and loop are implemented here from the preprint:

```bash
python scripts/train.py --pdb-dir <dir> --out runs/ft \
    --init-weights 0.0 --config configs/train_cath.yaml
```

`L_total = L_MLM + L_diff` unweighted (Appendix C.1), `t = sqrt(u)` masking, 8
noise clones per example, teacher-forced sequence, and a confidence head on a
stop-gradient rollout. See [`docs/training.md`](docs/training.md) for what the
paper specifies, what it leaves unspecified (optimizer and learning rate — chosen
here and recorded in every checkpoint), and the two upstream gaps it works around.

```bash
PYTHONPATH="$PWD:$PWD/PXDesign:$PWD/Protenix:$PWD/fampnn" python -m pytest tests/ -q
```

The side-chain tests run against the real weights, including an accuracy
regression guard: packing a CASP15 backbone from its own sequence reproduces the
withheld native side chains to ~1.1 Å all-atom RMSD.

## What this code guarantees

Composing two frozen models has a small number of ways to go quietly wrong, and
each is a loud failure here rather than a silent one:

- **The sequence is never designed.** Every identity is supplied to FaMPNN, and
  the aatype it echoes back is compared against the input. If a position lacks an
  identity the pipeline stops instead of letting the model invent one.
- **Atom order.** The shared AF2 atom37 order is pinned and checked against the
  live upstream constants, so a renumbering fails instead of scrambling coordinates.
- **Strict weights.** Checkpoints load with `strict=True`; a partial load would
  leave randomly initialized tensors in a model that still emits plausible coordinates.
- **The backbone is not moved.** Only the 33 side-chain slots may change; the
  backbone is restored from PXDesign and the deviation reported (`0.0` in practice).
- **Pristine upstreams.** FaMPNN must be unmodified; PXDesign carries exactly one
  recorded patch (the Protenix 2.0 embedder signature). Anything else is rejected.
- **Devices.** A GPU newer than the installed torch build is detected by probing a
  real kernel launch, not by trusting `torch.cuda.is_available()`.

Packing is a sampler: repeated runs on one backbone give rotamers ~0.5 Å RMS
apart. A seed makes a run reproducible -- bitwise on CPU or with deterministic
algorithms; CUDA's default kernels leave about 1e-5 A of jitter.

## Attribution & license

`PXDesign` and `Protenix` are **ByteDance's**; `FaMPNN` is **Richard Shuai et
al.'s**. All three are submodule commit pointers — no upstream code or weights
are re-hosted here. See each submodule's `LICENSE`, and cite the original work:

- PXDesign / Protenix — ByteDance.
- FaMPNN — *Sidechain conditioning and modeling for full-atom protein sequence
  design with FAMPNN*, [bioRxiv 2025.02.13.637498](https://www.biorxiv.org/content/10.1101/2025.02.13.637498v1).
