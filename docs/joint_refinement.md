# Side-chain-supervised backbone fine-tuning

Does a side-chain objective improve a backbone, when the side-chain model is
frozen and only sees what the backbone predicted?

    B0:  D_theta(noisy backbone, sigma)  trained with  L_BB
    B1:  the same, trained with          L_BB + l_local L_local

and B1 - B0, at the same budget, the same examples and the same noise, is the
answer. Everything else here exists to make that subtraction mean something.

## Status

Implemented and tested: the data contract, the one-step side-chain helper, the
live backbone forward, the gradient controls, the losses, and the trainer.
**No arm has been run.** The evaluator (`scripts/eval_joint_refinement.py`) is
not written, so nothing here reports a backbone metric yet.

| | |
|---|---|
| `pxf/joint/data.py` | residue correspondence, the preprocessing pose, the three masks |
| `pxf/joint/randomness.py` | named, replayable noise draws |
| `pxf/joint/model.py` | the one-pass BB -> SC graph and its detach policy |
| `pxf/joint/losses.py` | placement, the frame-only control, the combined objective |
| `pxf/joint/trainer.py` | arms, allowlist, optimizer, EMA, checkpoints, resume |
| `scripts/preflight_joint.py` | memory, timing, and the loss coefficients |
| `scripts/train_joint_refinement.py` | one arm |

## The arms

| Arm | Trains | Objective reaching the backbone | Purpose |
|---|---|---|---|
| R0 | nothing | — | the donor, unchanged |
| B0 | last 4 blocks + `layernorm_a` + decoder | `L_BB` | the matched baseline |
| B1 | same | `+ l_local L_local` | encoder-mediated supervision |
| B2 | same | `+ l_place L_place` | the candidate |
| BF | same | `L_BB + l_place L_frame` | geometric supervision, no FaMPNN prediction |
| BS | same | `L_BB`, side chains computed with the backbone detached | the compute-matched control |

`pxf/joint/trainer.py::ARMS` is the only place an arm is defined. Adding one
anywhere else is how two arms quietly stop being comparable.

## Three masks, not one

Conflating any pair of these is silent, so they are separate objects:

| Mask | What it is | What it is for |
|---|---|---|
| encoder | predicted backbone atoms; every side-chain slot zero | the encoder's **input** |
| local | `x_mask * frames_exist`, **ghosts included** | `L_local`, the source objective |
| physical | `atom_mask * frames_exist`, real atoms only | placement, chemistry, any atom count |

`physical` is a strict subset of `local`, and the gap is not only glycine: a
quality-vetoed residue loses its real atoms from `physical` while keeping
ghost-zero supervision in `local`, because the veto travels through
`missing_atom_mask` and a ghost is not "missing".

**87% of the local objective is ghosts.** On T1031, 2,728 of 3,135 local
targets are slots the residue does not have, whose target is the origin. That
is what the source objective supervises, and it is why the physical/ghost split
is logged per example: a backbone gain driven by predicting that atoms do not
exist is different evidence from one driven by packing.

## What has been measured

**The side-chain loss reaches the backbone.** With 34.4M backbone parameters
trainable and every FaMPNN parameter frozen, on T1031: `L_local` reaches them
through FaMPNN's geometry, `L_place` reaches them by both the encoder and the
frame route, and the detached arm reaches nothing. Central finite differences
agree with autograd on `d(L_local)/d(B_hat)` to 5%, which is the check that the
derivative is real rather than self-consistently wrong.

**The two parses already share a frame.** The recovered preprocessing transform
is the identity to 0.000000 A on T1031, so the recovery is a per-example check
rather than a fix. It is still run: a configuration that centred or reposed
would otherwise put side-chain targets several Angstroms from their backbone
with nothing downstream to say so.

**A degenerate match set defeats the residual.** A collinear backbone superposes
at 2e-6 A and still places every CB 0.46 A wrong, because rotation about the
line is unconstrained. `recover_preprocessing_transform` refuses an extent ratio
below 1e-3.

**CPU threading, not the arms, is the reproducibility limit.** At torch's
default 16 intra-op threads, two runs of B0 from identical weights and seed
diverge by **4.2e-05** on a weight after two steps -- about 40% of an optimizer
step at lr 1e-5..1e-4 -- purely from reduction order. At one thread B0
reproduces itself bitwise, and B0 and BS then agree bitwise too. Any CPU
comparison of two arms must pin `torch.set_num_threads(1)` or it is measuring
thread scheduling. On a GPU the equivalent statement needs a declared tolerance
instead, and that tolerance has not been measured yet.

## The coefficients are not a guess

`L_BB`, `L_local` and `L_place` are normalized differently -- over backbone
atoms, over side-chain coordinate components with an EDM weight, and over
physical components with none -- so their values say nothing about their
relative pull. `scripts/preflight_joint.py` sets each coefficient from the
term's **gradient at `B_hat`**, to a median ratio of 0.1 against the anchor,
and `calibrate()` refuses a dead or non-finite auxiliary gradient rather than
returning an enormous coefficient for a term that trains nothing.

Per band, not pooled: a CPU rehearsal on one AFDB chain gave `L_place` gradient
ratios of 2.0 / 2.1 / 10.6 / 14.8 across the four noise bands and `L_frame`
0.44 to 6.1. One median would have been wrong at both ends.

`train_joint_refinement.py` **refuses** an auxiliary arm whose coefficient is
zero, because that arm is B0 with extra compute and would otherwise be compared
against B0 as though it differed.

## Placement, and why it is its own term

`L_local` scores side chains in each residue's own backbone frame, so it is
blind to where that frame is: a backbone error rotates the whole residue and
leaves the local error untouched. `L_place` is the term that can see it, and it
reaches the backbone through the frames as well as through the encoder.

It scores physical atoms only, resolves ASP/GLU/PHE/TYR naming symmetry once
and detached (and only where both atoms are observed), and carries **no EDM
weight**: `1/c_out^2` diverges as the side-chain noise goes to zero, which is
right for an error that vanishes with the noise and wrong for a frame error,
which does not.

`rho(u) = u^2 / (1 + sqrt(1 + u^2))`, not the algebraically identical
`sqrt(1+u^2) - 1`: the literal form cancels in float32 and reads 5% low at
`u = 1e-3`, which is where a converging run spends its time.

## Running

```bash
# 1. the gates: GPU suite, memory, timing, and the coefficients
OUT=.../pxf_preflight_joint/run1 sbatch scripts/slurm/preflight_joint.sh

# 2. the baseline
ARM=B0 OUT=.../B0 sbatch --job-name=pxf_joint_B0 scripts/slurm/train_joint_refinement.sh

# 3. the control, against B0 -- cheapest check that the arms are paired
ARM=BS OUT=.../BS sbatch --job-name=pxf_joint_BS scripts/slurm/train_joint_refinement.sh

# 4. the candidates, once the preflight has produced their coefficients
ARM=B1 OUT=.../B1 PREFLIGHT=.../preflight.json \
    sbatch --job-name=pxf_joint_B1 scripts/slurm/train_joint_refinement.sh
```

Marlowe copies of both launchers are under `scripts/slurm/marlowe/`. They differ
in the header (untyped gres, `--qos=medium`), the python (a relocated prefix on
`PATH`, no conda), the data roots, and in namespacing `JOINT_STRUCTURES` /
`JOINT_EXTRA_ARGS` away from the bare `STRUCTURES` / `EXTRA_ARGS` that
`marlowe_env.sh` exports for other jobs.

## Not done

- `scripts/eval_joint_refinement.py`: matched backbone evaluation, a common
  frozen packer, paired intervals. Until it exists no arm can be scored.
- The joint arms J0/J1 (trainable side-chain denoiser) need a second optimizer
  group; `freeze_sidechain=False` is refused rather than silently training
  nothing.
- Activation checkpointing, whose on/off gradient parity has not been checked
  on the hook-free forward.
- The experimental-structure cohort; only AFDB is wired.
