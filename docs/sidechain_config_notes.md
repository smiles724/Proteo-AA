# Side-chain config: rationale and measurements

Detail behind the one-line comments in `pxdesign_train/configs/configs_train.py`
(`training_configs["sidechain"]` / `["loss"]`). Kept here so the config stays readable.

## `context_aware` (default `True`) — spec, not an option

Overleaf requires it in six places, e.g. *"Operating in the global frame allows
side-chain atoms to directly attend to neighboring residues, receptor atoms, fixed
motifs, ligands, and other spatial context"*; clash covers side-chain↔backbone /
↔side-chain / ↔context pairs; the appendix says our default S_φ's context attention
captures side-chain↔receptor interactions.

Before this, S_φ saw none of it: every non-binder token was an all-masked key in the
cross-residue attention, clash scored side-chain↔side-chain only, and contact's
reference was binder-only **and unmasked**, so phantom rows silently zeroed the
penalty. Emitting global coordinates was necessary for receptor awareness but never
sufficient — no path carried the receptor in.

It is a switch so it can be ablated and so Stage II-A warmup can run without it:
warmup uses GT frames (raw coordinate frame), while `x_denoised` (our only source of
receptor atoms) lives in the augmented frame — scoring one against the other would be
a frame bug, not a conservative approximation.

`context_radius=10.0` Å (drop atoms farther than this from any binder CA),
`context_max_atoms=4096` (memory cap).

## `template_init` (default `True`) — Overleaf par.221

Start side-chain denoising from the type-conditioned ideal template perturbed by
`sigma_T`, not isotropic Gaussian noise (`y_T = mu_ideal[a, chi, j] + sigma_T·eps`,
then `x_T = F_hat·y_T`, mirrored in training and sampling). An isotropic Gaussian is
rotation-invariant, so pushing it through `F_hat` carries no backbone orientation; the
anisotropic template does. `mu_ideal == 0` is exactly the old Gaussian init.

The 0714 appendix ("Residue-Specific Side-Chain Template Construction") specifies how
`mu_ideal` is built, in three steps:

- **Step 1** `chi_constants.py` — `A_sc`, `K_i`, `G_ideal` (connectivity, bond
  lengths/angles, rigid groups) from the CCD + Protenix's AF chi tables.
- **Step 2** `rotamers.py` — `chi ~ p(r | a_hat, phi_hat, psi_hat)` from Dunbrack
  BBDEP2010; phi/psi are dihedrals of the **predicted** backbone.
- **Step 3** `buildsc.py` — `BuildSC`: pose `G_ideal` at those torsions in the local
  frame, preserving bond lengths/angles exactly.

Set `False` to restore the isotropic Gaussian A/B baseline.

## `template_provider` (default `"dunbrack_mode"`)

- `"dunbrack_mode"` — BuildSC, `chi = argmax_r p(r | a, phi, psi)` (default)
- `"dunbrack"` — BuildSC, `chi ~ Categorical(p(r | a, phi, psi))`
- `"ccd"` — static one-conformer CCD table (pre-0714 baseline)

Mean local-frame RMSD of the template from the true side chain, on 2790 residues of 33
real chains (`scripts/eval_template_quality.py` regenerates it):

| provider | RMSD (Å) | chi1 recovery |
|---|---|---|
| gaussian (`mu=0`, pre-0714 default) | 2.887 | n/a |
| ccd (static, one arbitrary chi) | 1.662 | 46.3 % |
| dunbrack (sampled) | 1.487 | 61.0 % |
| **dunbrack_mode (argmax)** | **1.277** | **68.7 %** |
| oracle (true chi; lower bound) | 0.333 | 100 % |

Mode wins on 18 of 19 side-chain types. Sampling is *worse* than the static CCD table
on high-entropy residues (GLN −19%, GLU −16%, LYS −6%, ARG −4%) because their rotamer
distributions are nearly flat (GLN's modal rotamer holds only 12% of the mass over 108
rotamers), so a draw is usually far from the truth. As an *initialization*, lower error
is the point, so mode is the default.

**Open question:** sampling is the more natural prior for a *generative* model — it
makes the init distribution match `p(r | a, phi, psi)` instead of collapsing to the
mode. That is about sample diversity, which this geometric benchmark cannot measure; it
needs a training A/B. Nothing forecloses it — flip the string.

## `frame_aware_head` (default `False`) — ablation candidate

- OFF (default): `x0_global = MLP(atom_feats) + ca_coords` (CA-anchored global head)
- ON: `x0_global = F_hat·MLP(atom_feats)` (regress local offsets, let the known
  stop-grad frame rotate them)

Output space is global either way, so both satisfy par.204; the paper does not mandate
a head parameterisation (its appendix allows equivariant nets), so this is a
training-stability assumption, not a spec requirement — hence default OFF.

Measured (1cse chain B, single-structure memorization, 400 steps, `sc_local`; old
local-output baseline = 0.51):

| config | sc_local |
|---|---|
| OFF (CA-anchored) + gaussian init | 4.05 |
| OFF + template init | 3.84 |
| ON + template init | 2.22 |
| ON + template init + local_coord_input | 0.557 |

Hypothesis: with random rotation augmentation the CA-anchored MLP must infer R from its
input *and* apply it to its output — a bilinear op an MLP approximates poorly. A
single-structure memorization run cannot settle this; open it as an ablation arm if
real-data training stalls.

## `local_coord_input` (default `False`) — ablation candidate

- OFF (default): S_φ's own noisy side-chain atoms fed as raw global coords,
  `x = F_hat·(mu + sigma·eps)`
- ON: fed in the residue-local frame (translation-free)

The appendix calls the global-coordinate atom feature "optional", so neither violates
spec. Concern (untested on real data): the global form carries `t_CA`, the residue's
absolute position (tens of Å), on top of ~4 Å side-chain geometry, so the linear coord
embedding `W_xyz` sees mostly "where is this residue" rather than "what shape is this
side chain". Measured contribution on the smoke (with `frame_aware_head` ON): 2.22 →
0.557.

## Ablation arms

See `SC_ABLATION_ARMS` and `tests/test_ablation_arms.py`. Six arms:
`no / a-indirect / a-direct / bbctx / q / a-direct+q`, one channel isolated each.
`hres_inject=False` is the true no-feedback control (refinement pass still runs);
`bbctx` is q's control (14-slot S_φ without the q write-back), so `q − bbctx` isolates
the atom channel. Single-structure numbers cannot rank the arms — that waits for real
data.
