"""Inference half of Overleaf par.221: the co-generation sampler must initialize
side chains from the residue-type-specific IDEAL TEMPLATE of the PREDICTED residue
type (a_hat), perturbed by sigma_T, around the PREDICTED backbone frame — exactly
what model.py's training path does. Feeding S_phi isotropic N(0, sigma^2) at
sampling time while training fed it F_hat(mu_ideal + sigma_T eps) is a hard
train/inference mismatch on the one input channel this whole change is about.

These tests drive `cogenerate` with a stub model (no DiffusionModule / CUDA) and
spy on the two initializers.
"""
import torch
import torch.nn as nn

from pxdesign_train.cogenerate import cogenerate, _AA3
from pxdesign_train.sampler import build_aa20_to_restype36
from pxdesign_train.sidechain import init as sc_init
from pxdesign_train.sidechain.instantiate import MAX_SC, sidechain_atoms

C = 8                      # token/trunk channel width
C_ATOM = 4
N_TOKEN = 3                # tokens 0,1 are design; token 2 is context
ATOMS_PER_TOKEN = 4
N_ATOM = N_TOKEN * ATOMS_PER_TOKEN

# Predicted types we force the AA head to emit: TRP (10 side-chain atoms) and
# ALA (1). Distinct atom counts => a template init cannot be confused with noise.
PRED = {0: _AA3.index("TRP"), 1: _AA3.index("ALA")}


class _FakeDiffusion(nn.Module):
    class _Relpe:
        def generate_relp(self, feat):
            return feat

    class _Cond:
        def __init__(self):
            self.relpe = _FakeDiffusion._Relpe()

    def __init__(self):
        super().__init__()
        self.diffusion_conditioning = _FakeDiffusion._Cond()

    def forward(self, x_noisy=None, **kw):
        return x_noisy * 0.5           # any denoiser; coords stay finite


class _SpySideChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, h_res, restype_logits, atom_name_ids, atom_mask, noisy_local,
                t, ca_coords=None, frame_R=None, frame_t=None, bb_local=None,
                res_mask=None, ctx_mask=None):
        self.calls.append({
            "noisy": noisy_local.detach().clone(),
            "mask": atom_mask.detach().clone(),
            "frame_R": None if frame_R is None else frame_R.detach().clone(),
            "frame_t": None if frame_t is None else frame_t.detach().clone(),
            "ctx_mask": None if ctx_mask is None else ctx_mask.detach().clone(),
        })
        B, L, A = atom_name_ids.shape
        return (torch.zeros(B, L, A, 3), torch.zeros(B, L, A, C_ATOM))


class _FakeModel(nn.Module):
    def __init__(self, template_init=True, sigma_T=0.3,
                 local_coord_input=True, frame_aware_head=True,
                 context_aware=False):
        super().__init__()
        self.aa_input_source = "diffusion_internal"
        self.enable_sidechain = True
        self.enable_coevolution = True
        self.sc_init_sigma = 1.0
        self.sc_template_init = template_init
        self.sc_init_sigma_T = sigma_T
        self.sc_local_coord_input = local_coord_input
        self.sc_frame_aware_head = frame_aware_head
        self.sc_context_aware = context_aware
        self.diffusion_module = _FakeDiffusion()
        self.sidechain_module = _SpySideChain()
        self._a_token_cache = torch.randn(1, N_TOKEN, C)

    def get_condition_embedding(self, feat, chunk_size=None):
        s = torch.zeros(1, N_TOKEN, C)
        return s, s.clone(), None

    def inference_noise_scheduler(self, N_step=20, device=None, dtype=None):
        # step 0 sigma=1.0 (> 0.5*sigma_max -> no side chain), step 1 sigma=0.4
        # (<= 0.5*sigma_max -> side-chain step, both design tokens committed).
        return torch.tensor([1.0, 0.4, 0.1], device=device, dtype=dtype)

    def _reduce_a_token(self, a, sigma):
        return a

    def design_residue_type_head(self, a_red, aa_t=None):
        logits = torch.zeros(N_TOKEN, 20)
        for tok, aa in PRED.items():
            logits[tok, aa] = 10.0
        return logits

    def sidechain_feedback(self, atom_feats, mask, h_res, **kw):
        return h_res

    def hres_injector(self, h):
        return torch.zeros(1, N_TOKEN, C)


def _feat(with_bb_index=False):
    aa20_to_36, xpb = build_aa20_to_restype36()
    n_ch = int(max(int(aa20_to_36.max()), xpb)) + 1
    restype = torch.zeros(N_TOKEN, n_ch)
    restype[:, 0] = 1.0
    a2t = torch.arange(N_TOKEN).repeat_interleave(ATOMS_PER_TOKEN)
    dtm = torch.zeros(N_TOKEN, dtype=torch.bool)
    dtm[0] = dtm[1] = True
    feat = {
        "restype": restype,
        "atom_to_token_idx": a2t,
        "design_token_mask": dtm,
    }
    if with_bb_index:
        # Binder tokens 0,1 own N/CA/C/O; token 2 is context (receptor) -> -1.
        bb = torch.full((N_TOKEN, 4), -1, dtype=torch.long)
        for tok in (0, 1):
            for k in range(4):
                bb[tok, k] = ATOMS_PER_TOKEN * tok + k
        feat["sc_bb_atom_idx"] = bb
        # Every token's representative (CA) atom — including the receptor's.
        feat["sc_token_center_idx"] = torch.tensor(
            [ATOMS_PER_TOKEN * t + 1 for t in range(N_TOKEN)], dtype=torch.long
        )
    return feat


def _run(monkeypatch, with_bb_index=False, **model_kw):
    """Run cogenerate with both initializers spied; returns (model, calls)."""
    import protenix.model.protenix as pxm
    monkeypatch.setattr(pxm, "update_input_feature_dict", lambda f: f, raising=False)

    calls = {"template": [], "gaussian": []}
    real_tpl, real_gauss = sc_init.template_init_local, sc_init.gaussian_init_local

    def spy_tpl(type_idx, mask, sigma_T=sc_init.DEFAULT_SIGMA_T, generator=None,
                backbone=None, phi=None, psi=None):
        calls["template"].append({"type_idx": type_idx.clone(), "mask": mask.clone(),
                                  "sigma_T": sigma_T,
                                  "phi": None if phi is None else phi.clone(),
                                  "psi": None if psi is None else psi.clone()})
        return real_tpl(type_idx, mask, sigma_T=sigma_T, generator=generator,
                        backbone=backbone, phi=phi, psi=psi)

    def spy_gauss(mask, sigma=1.0, generator=None):
        calls["gaussian"].append({"mask": mask.clone(), "sigma": sigma})
        return real_gauss(mask, sigma=sigma, generator=generator)

    monkeypatch.setattr(sc_init, "template_init_local", spy_tpl)
    monkeypatch.setattr(sc_init, "gaussian_init_local", spy_gauss)

    torch.manual_seed(0)
    model = _FakeModel(**model_kw)
    out = cogenerate(model, _feat(with_bb_index), N_step=2,
                     sidechain_cycle=True, sc_start_frac=0.5)
    return model, calls, out


def test_sampler_gives_sphi_the_receptor_as_context_keys(monkeypatch):
    """sidechain.context_aware is mirrored at SAMPLING, behaviourally.

    Training lets S_phi's cross-residue attention key on the receptor. If the sampler
    does not, the trained module runs blind to the thing it packs against — the exact
    train/inference mismatch the parity test exists to prevent, and one a source-scrape
    alone cannot catch.
    """
    model, _, _ = _run(monkeypatch, with_bb_index=True, context_aware=True)
    call = model.sidechain_module.calls[-1]

    ctx = call["ctx_mask"]
    assert ctx is not None, "sampler never handed S_phi a context mask"
    # Token axis = [committed residues 0,1] + [context token 2], key-only.
    assert ctx.shape == (1, 3)
    assert ctx.tolist() == [[False, False, True]]
    # The context row owns NO side-chain slot, so it decodes nothing.
    assert not call["mask"][0, 2].any()


def test_sampler_without_context_aware_is_unchanged(monkeypatch):
    model, _, _ = _run(monkeypatch, with_bb_index=True, context_aware=False)
    call = model.sidechain_module.calls[-1]
    assert call["ctx_mask"] is None
    assert call["mask"].shape[1] == 2, "only the committed residues on the token axis"


def test_sampler_uses_template_init_with_predicted_type(monkeypatch):
    """template_init=True (the training default) -> the sampler must call
    template_init_local with the PREDICTED residue types, not gaussian_init_local."""
    model, calls, out = _run(monkeypatch, template_init=True, sigma_T=0.3)

    assert calls["gaussian"] == [], "sampler still used Gaussian init under template_init=True"
    assert len(calls["template"]) >= 1, "sampler never called template_init_local"

    c = calls["template"][0]
    # a_hat: the types the unmasking loop committed (== what produced the atom mask).
    expected = torch.tensor([PRED[0], PRED[1]], dtype=torch.long)
    assert torch.equal(c["type_idx"].cpu().long(), expected), (
        f"template init got types {c['type_idx'].tolist()}, expected predicted {expected.tolist()}"
    )
    assert torch.equal(out["sequence"][:2].cpu(), expected)  # same source as the mask
    assert c["sigma_T"] == 0.3                               # config switch honoured
    # Atom mask agrees with the predicted types (TRP=10 atoms, ALA=1).
    assert c["mask"].shape == (2, MAX_SC)
    assert int(c["mask"][0].sum()) == len(sidechain_atoms("TRP"))
    assert int(c["mask"][1].sum()) == len(sidechain_atoms("ALA"))


def test_sampler_falls_back_to_gaussian_when_disabled(monkeypatch):
    """template_init=False keeps the old Gaussian init (A/B control intact)."""
    model, calls, _ = _run(monkeypatch, template_init=False)
    assert calls["template"] == [], "template init used although template_init=False"
    assert len(calls["gaussian"]) >= 1
    assert calls["gaussian"][0]["sigma"] == 1.0   # model.sc_init_sigma


def test_sampler_sigma_T_switch_is_plumbed(monkeypatch):
    _, calls, _ = _run(monkeypatch, template_init=True, sigma_T=0.05)
    assert calls["template"][0]["sigma_T"] == 0.05


def test_sampler_coord_input_and_head_match_training(monkeypatch):
    """The other two halves of the same distribution contract: with
    local_coord_input=True S_phi's coordinate channel is the LOCAL init (not the
    to_global one), and with frame_aware_head=True it receives the predicted frame —
    both as model.py's training block does."""
    model, calls, _ = _run(monkeypatch, template_init=True,
                           local_coord_input=True, frame_aware_head=True)
    call = model.sidechain_module.calls[0]
    assert call["frame_R"] is not None and call["frame_t"] is not None, (
        "frame_aware_head=True in training but the sampler called S_phi without a frame"
    )
    # Local init: coords sit within a few Angstrom of the frame origin. A global
    # init would carry t_CA (absolute position of the residue) instead.
    valid = call["mask"][0]
    coords = call["noisy"][0][valid]
    assert coords.abs().max() < 12.0
    assert torch.allclose(coords.mean(0), torch.zeros(3), atol=8.0)
    # And it is NOT the global mapping of the same init (which would add t_CA).
    from pxdesign_train.sidechain.frames import to_global
    glob = to_global(call["noisy"].float(), call["frame_R"].float(), call["frame_t"].float())
    assert not torch.allclose(glob[0][valid], coords.float(), atol=1e-4)


def test_sampler_calls_sphi_without_frame_when_head_not_frame_aware(monkeypatch):
    model, _, _ = _run(monkeypatch, template_init=True,
                       local_coord_input=False, frame_aware_head=False)
    call = model.sidechain_module.calls[0]
    assert call["frame_R"] is None and call["frame_t"] is None


def test_cogenerate_honours_hres_inject_switch():
    """Sampling must respect sidechain.hres_inject, or the ablation is contaminated.

    Training (model.py) gates the INDIRECT h_res' -> s_trunk channel on
    `sidechain.hres_inject`. If cogenerate ignored it, then every arm trained WITHOUT the
    indirect channel — the true `no` control, `a-direct`, `bbctx`, `q` — would silently get
    it back at sampling time, and the information-flow ablation would be evaluating a model
    that was never trained that way. This pins the gate at the source level (the sampler is
    a long integration loop; a source assertion is the cheap, non-vacuous check that the
    condition is actually consulted at the injection site).
    """
    import inspect

    from pxdesign_train import cogenerate as cg

    src = inspect.getsource(cg.cogenerate)
    inj = src.index("model.hres_injector(")
    guard = src.rindex('getattr(model, "sc_hres_inject", True)', 0, inj)
    assert guard < inj, (
        "cogenerate applies hres_injector without consulting model.sc_hres_inject — "
        "training can disable the indirect channel but sampling would re-enable it."
    )


def test_cogenerate_clears_the_q_call_key_registry_each_step():
    """The q call-key registry is a training-only device; it must not leak at inference.

    It exists so a checkpointed decoder call's backward RECOMPUTE reaches the same decision
    as its forward. Inference runs under no_grad (no recompute), so leaving it uncleared
    just strong-references every sampling step's q_skip for the whole run.
    """
    import inspect

    from pxdesign_train import cogenerate as cg

    src = inspect.getsource(cg.cogenerate)
    assert "model._q_inject_calls = {}" in src, (
        "cogenerate never clears model._q_inject_calls — every step's q_skip stays alive."
    )
