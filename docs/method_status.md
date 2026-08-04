# ProteoAA — method → implementation status

Honest per-stage status of the current `main`. **"done" means the mechanism is
implemented + covered by CPU unit tests + (where noted) a single-structure GPU
smoke.** It does **not** mean trained, multi-structure, or method-validated — every
result so far is single-structure overfit.

| Stage / piece | Status | Notes |
|---|---|---|
| **Stage I — Backbone-AA** | runnable | coordinate diffusion + masked (absorbing) discrete diffusion for residue type; AA head reads the structure-aware `a_token` |
| **per-σ AA loss** | done | AA cross-entropy computed per noise level (σ) then averaged — not reduce-then-predict |
| **Stage II-A — side-chain warmup** | implemented | one-step Gaussian `S_φ`; reads **GT frames + GT atom masks**; gradient-isolated (`trunk_grad_scale=0` — side-chain loss can't update the backbone; optimizer/backbone losses are not otherwise frozen) |
| **Stage II-B — co-evolution** | wiring / smoke done | `S_φ` on predicted-backbone frames `F̂` (from `x̂₀`) + stop-grad global pseudo-target; `h_res′` → reuse `B_θ` to refine. **Full recurrent/per-σ feedback pending**: `h_res′` is σ-averaged before injection because Protenix's `s_trunk` is sample-shared. |
| **Stage III — predicted-mask** | partial, default off | `sidechain.predicted_mask=False` by default. When enabled: atom set from the **predicted** residue type + coord/physical routing, which makes `post_aa` safe to supervise. |
| **L_SC-AA candidate ranking** | core only | ranking loss + compatibility energy implemented and unit-tested; per-candidate `S_φ` orchestration not wired into training |
| **physical loss** | clash + contact active | bond / angle / rotamer implemented but **not activated**. The blocker — "need a residue-specific ideal-geometry table" — is **now gone**: `sidechain/chi_constants.py` has the bond/angle geometry and rigid groups, and `data/dunbrack2010_bbdep.npz` has rotamer targets. Activation is a separate task. |
| **side-chain template** | done | Overleaf 0714 appendix, all 3 steps: residue constants (`chi_constants`) → backbone-dependent rotamer lookup (`rotamers`, Dunbrack BBDEP2010, conditioned on φ̂/ψ̂ of the **predicted** backbone) → `BuildSC` (`buildsc`). Default **on** (`template_init=True`, `template_provider=dunbrack_mode`). Cuts the initialisation's distance from the true side chain from **2.89 Å** (Gaussian) to **1.28 Å**; χ₁ recovery 68.7%. GPU: backbone-conditioned for ~97% of side-chain-bearing tokens. |
| **full-atom output** | side-chain coords returned | `cogenerate` returns backbone + sequence + `S_φ` side-chain coordinates; **full assembled PDB/tensor pending** |

**Leakage safeguards.** Side chains initialise from a residue-type + **predicted**-backbone
template (never noised GT); the template's rotamer is conditioned on φ̂/ψ̂ of the predicted
backbone, which is inference-available, and the provider contract has no parameter through
which GT side-chain coordinates could arrive (asserted by test). Binder side chains are
excluded from `L_bb` and scrubbed (→ Cα) from the diffusion input; `post_aa` is supervised
only under predicted-mask (so GT atom composition can't leak identity into the AA head).

**Known future-batch limitation.** The trainer runs `batch_size=1` (macro-batching is
done over the `N_sample` diffusion axis). Two side-chain paths currently assume that:
the predicted-mask branch instantiates the atom set / routing from item 0 (warns if
`batch>1`), and the predicted-frame pseudo-target is tiled over the σ axis but not over
a `batch>1` axis. Both are fine at `batch_size=1`; revisit before enabling true
macro-batch training.

**Validated so far.** `--sidechain_warmup` and `--coevolution` run end-to-end on a
single-structure GPU smoke: `sc_local` drops, losses finite, no shape/leakage issues.
**Not yet:** multi-structure / held-out / design-quality benchmarks (need real-data
training), physical geometry tables, `L_SC-AA` orchestration, strict per-σ feedback.
