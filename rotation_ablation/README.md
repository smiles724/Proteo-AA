# Paired side-chain rotation ablation

This folder implements the paired SE(3) diagnostic at the `SideChainModule`
boundary. It deliberately does not edit the training forward. One captured
protein/noise/sigma row is reused for every intervention, so coordinate
sensitivity is not confounded with a new diffusion sample.

## What it runs

For each of 20 rotations, the runner evaluates the coordinate-path 2-by-2:

| `local_coord_input` | `frame_aware_head` | Diagnostic |
|---:|---:|---|
| false | false | current global-coordinate/CA-offset baseline |
| true | false | local input only |
| false | true | frame-aware decoder only |
| true | true | consistent local input and local decoder |

Within each arm it runs:

- A: original coordinates and `h_res`;
- C: rigidly transformed coordinates with original `h_res`;
- B: transformed coordinates with recomputed `h_res_q`;
- D: original coordinates with recomputed `h_res_q`.

B/D are emitted only when a captured bundle contains `h_res_q`. C is always
exact and is sufficient to isolate the current S-phi coordinate path.

The report contains relative Frobenius error and per-residue cosine similarity
for:

- projected `h_res`, AA, time, and coordinates;
- the decomposed initial atom feature `u`;
- every intra-residue and cross-residue block;
- final `atom_feats`;
- pooled `a_sc`, projected `g`, `delta_h_res`, and `h_res_prime`;
- head output before frame reconstruction;
- global coordinate equivariance RMSD after reconstruction.

Translation-only and +60-degree chi-like local torsion controls are included.
When targets/context are present in the bundle, the report also includes local
SC RMSD, chi1/chi2 20-degree accuracy, clash rate, and errors split by residue
category and AA correctness.

## Quick architecture check

This uses a deterministic same-protein fixture. With no checkpoint, weights are
random; this checks the mechanism and installation:

```bash
LAYERNORM_TYPE=torch PYTHONPATH="Protenix:PXDesign:." \
python -m rotation_ablation \
  --n-rotations 20 \
  --output rotation_ablation/results_smoke.json
```

To test the currently trained S-phi weights:

```bash
LAYERNORM_TYPE=torch PYTHONPATH="Protenix:PXDesign:." \
python -m rotation_ablation \
  --checkpoint /path/to/checkpoint.pt \
  --n-heads 16 \
  --n-rotations 20 \
  --device cuda \
  --output rotation_ablation/results_checkpoint.json
```

The checkpoint loader reconstructs `SideChainModule` and `HResFeedback` from
tensor shapes. Attention head count cannot be recovered from a state dict, so
`--n-heads` must match training (the repository default is 16).

## Real-protein bundle

Capture the actual call after the backbone has produced a row-aligned
`h_res`/AA/sigma/noise state:

```python
from rotation_ablation.bundle import (
    SidechainCallCapture,
    SidechainInputBundle,
    save_bundle,
)

model.eval()
with SidechainCallCapture(model.sidechain_module) as capture:
    output = model(
        input_feature_dict=batch["input_feature_dict"],
        label_dict=batch["label_dict"],
        mode="train",
    )

args, kwargs = capture.call
bundle = SidechainInputBundle.from_module_call(
    args,
    kwargs,
    local_coord_input=model.sc_local_coord_input,
    # The default CA-anchored head does not pass frames into S_phi, but the
    # enclosing model still returns the active predicted/GT frame.
    frame_R=output["sc_frame_R"],
    frame_t=output["sc_frame_t"],
    metadata={
        "protein": sample_name,
        "sigma_row": sigma_row,
        "noise_seed": noise_seed,
    },
)
bundle.gt_local = batch["input_feature_dict"].get("sc_gt_local")
bundle.residue_type_idx = batch["input_feature_dict"].get("aa_clean")
save_bundle(bundle, "rotation_ablation/my_protein.pt")
```

Then run:

```bash
LAYERNORM_TYPE=torch PYTHONPATH="Protenix:PXDesign:." \
python -m rotation_ablation \
  --checkpoint /path/to/checkpoint.pt \
  --bundle rotation_ablation/my_protein.pt \
  --device cuda \
  --output rotation_ablation/my_protein_results.json
```

For the full A/B/C/D test, repeat the backbone calculation on the rigidly
transformed same sample with the same sigma/noise row. Store its aligned token
state as `bundle.h_res_q`, its aligned logits as `bundle.aa_logits_q`, and the
exact transform as `bundle.paired_rotation` / `bundle.paired_translation`.
These values are evaluated once under `paired_abcd`; they are never incorrectly
reused for the other 20 random coordinate-only rotations.
The S-phi runner itself reconstructs the transformed geometry from the original
bundle, ensuring that local initialization noise is reused exactly.

## Reading the verdict

`rotation_sensitive=false` requires both:

- maximum `atom_feats` relative error <= `--invariant-tolerance` (default
  `1e-4`);
- maximum global coordinate equivariance RMSD <=
  `--coordinate-tolerance` (default `1e-3` Angstrom).

The most useful detailed signal is the first intermediate whose
`relative_frobenius.max` rises sharply.

Run the focused tests with:

```bash
LAYERNORM_TYPE=torch PYTHONPATH="Protenix:PXDesign:." \
python -m pytest rotation_ablation/test_rotation_ablation.py -q
```

## Slurm

One Slurm job runs all four coordinate arms, the 20 rotations, translation,
torsion, and the paired A/B/C/D intervention when the bundle contains
`h_res_q`:

```bash
CHECKPOINT=/path/to/checkpoint.pt \
BUNDLE=/path/to/my_protein.pt \
RUN_ROOT=/hai/scratch/yfsun/proteo_aa_runs/rotation_ablation/my_run \
sbatch rotation_ablation/slurm_rotation_ablation.sh
```

The real-protein bundle is optional. Without it, the checkpoint is evaluated
on the deterministic same-protein synthetic fixture:

```bash
CHECKPOINT=/path/to/checkpoint.pt \
sbatch rotation_ablation/slurm_rotation_ablation.sh
```

Installation/architecture smoke without a checkpoint:

```bash
SMOKE=1 sbatch rotation_ablation/slurm_rotation_ablation.sh
```

Useful overrides are `N_ROTATIONS`, `SEED`, `N_HEADS`,
`TRANSLATION_SCALE`, `INVARIANT_TOLERANCE`, `COORDINATE_TOLERANCE`,
`RUN_ROOT`, `PYTHON_BIN`, and `CONDA_ENV`. Additional CLI arguments supplied
after the script name are forwarded to `python -m rotation_ablation`.
