"""Tests for the M1-M5 review fixes.

M1 backbone-only L_bb, M2 post-AA leak gate (loss side), M3 side-chain global
assembly math, M5 config loss-weight plumbing. The full-model cycle (M2 model
gating, M3 cogenerate integration) needs the real DiffusionModule / CUDA and is
exercised on GPU; here we lock the CPU-testable contracts.
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))


def _ala_atom_array():
    biotite = pytest.importorskip("biotite.structure")
    names = ["N", "CA", "C", "O", "CB"]
    coords = np.array(
        [[-1.0, 0, 0], [0, 0, 0], [1, 0.5, 0], [1.2, 1.6, 0], [0.2, -0.8, 0.9]],
        dtype=np.float32,
    )
    aa = biotite.AtomArray(length=5)
    aa.coord = coords
    aa.chain_id = np.array(["A"] * 5)
    aa.res_id = np.array([1] * 5)
    aa.res_name = np.array(["ALA"] * 5)
    aa.atom_name = np.array(names)
    aa.set_annotation("distogram_rep_atom_mask", np.array([0, 1, 0, 0, 0]))
    return aa


# ---------------- M1: backbone-only loss mask ----------------

def test_backbone_loss_mask_drops_binder_sidechain():
    from pxdesign_train.data.featurizer import DesignFeaturizer

    aa = _ala_atom_array()
    binder = np.ones(5, dtype=bool)  # whole residue is binder
    mask = DesignFeaturizer._backbone_loss_mask(aa, binder)
    # N, CA, C, O kept; CB (side chain) dropped.
    assert mask.tolist() == [True, True, True, True, False]


def test_backbone_loss_mask_keeps_nonbinder_sidechain():
    from pxdesign_train.data.featurizer import DesignFeaturizer

    aa = _ala_atom_array()
    binder = np.zeros(5, dtype=bool)  # nothing is binder -> everything kept
    mask = DesignFeaturizer._backbone_loss_mask(aa, binder)
    assert bool(mask.all())


def test_backbone_atom_mask_removes_sidechain_from_mse():
    from pxdesign_train.loss import PXDesignLoss

    n_atom = 5
    pred = torch.zeros(1, n_atom, 3)                    # [N_sample, N_atom, 3]
    gt = torch.zeros(1, n_atom, 3)
    gt[0, 4] = 10.0                                     # big error on the CB atom
    sigma = torch.tensor([100.0])                       # gate off lddt/disto
    cmask = torch.ones(n_atom)
    rep = torch.tensor([0, 1, 0, 0, 0], dtype=torch.bool)
    loss = PXDesignLoss(align_before_mse=False, weight_lddt=0.0, weight_disto=0.0)

    with_cb = loss(pred, gt, sigma, cmask, rep)["mse"]
    bb_mask = torch.tensor([1, 1, 1, 1, 0], dtype=torch.float32)   # drop CB
    without_cb = loss(pred, gt, sigma, cmask, rep, backbone_atom_mask=bb_mask)["mse"]
    assert with_cb.item() > 0.0
    assert without_cb.item() == 0.0     # side-chain error no longer supervised


# ---------------- P1: scrub GT side chains from the diffusion INPUT ----------------

def test_scrub_design_sidechain_coords_moves_cb_to_ca():
    from pxdesign_train.data.featurizer import DesignFeaturizer

    aa = _ala_atom_array()
    label = {"coordinate": torch.from_numpy(np.asarray(aa.coord).copy()),
             "coordinate_mask": torch.ones(5)}
    binder = np.ones(5, dtype=bool)
    out = DesignFeaturizer._scrub_design_sidechain_coords(aa, label, binder)
    coord = out["coordinate"]
    ca = coord[1]                       # CA
    assert torch.allclose(coord[4], ca)          # CB collapsed onto CA (no side-chain geom)
    for i in range(4):                           # backbone atoms untouched
        assert torch.allclose(coord[i], torch.from_numpy(np.asarray(aa.coord)[i]))


def test_scrub_leaves_nonbinder_untouched():
    from pxdesign_train.data.featurizer import DesignFeaturizer

    aa = _ala_atom_array()
    orig = torch.from_numpy(np.asarray(aa.coord).copy())
    label = {"coordinate": orig.clone(), "coordinate_mask": torch.ones(5)}
    out = DesignFeaturizer._scrub_design_sidechain_coords(aa, label, np.zeros(5, bool))
    assert torch.allclose(out["coordinate"], orig)   # nothing is binder -> no change


def test_backbone_only_binder_couples_to_compute_sidechain():
    """P3: turning on S_phi target extraction forces backbone-only binder."""
    from pxdesign_train.runner.data import DesignSourceDataset

    class _P:  # minimal provider stub; never indexed here
        def __len__(self): return 0
    ds = DesignSourceDataset(_P(), source_name="x", compute_sidechain=True)
    assert ds.backbone_only_binder is True


# ---------------- M5: config loss-weight plumbing ----------------

def test_sc_local_weight_scales_total():
    from pxdesign_train.loss import PXDesignLoss
    from pxdesign_train.sidechain.instantiate import MAX_SC, sidechain_mask

    def _args():
        pred = torch.zeros(1, 4, 3)
        gt = torch.zeros(1, 4, 3)
        sigma = torch.tensor([100.0])
        return pred, gt, sigma, torch.ones(4), torch.tensor([0, 1, 0, 1], dtype=torch.bool)

    amask = sidechain_mask(["PHE", "LYS"])[None]
    sc_pred = torch.ones(1, 2, MAX_SC, 3)
    sc_gt = torch.zeros(1, 2, MAX_SC, 3)

    base = PXDesignLoss(align_before_mse=False, weight_lddt=0.0, weight_disto=0.0,
                        weight_sc_local=1.0, weight_sc_phys=0.0)
    dbl = PXDesignLoss(align_before_mse=False, weight_lddt=0.0, weight_disto=0.0,
                       weight_sc_local=2.0, weight_sc_phys=0.0)
    kw = dict(sc_pred_local=sc_pred, sc_gt_local=sc_gt, sc_atom_mask=amask)
    o1 = base(*_args(), **kw)
    o2 = dbl(*_args(), **kw)
    # sc_local (raw) identical; total differs by exactly one more sc_local unit.
    assert torch.isclose(o1["sc_local"], o2["sc_local"])
    assert torch.isclose(o2["loss"] - o1["loss"], o1["sc_local"], atol=1e-5)


def test_config_defaults_backbone_only_and_predicted_mask():
    """The shipped config keeps B_theta backbone-only and the post-AA leak closed."""
    from pxdesign_train.configs.configs_train import training_configs
    sc = training_configs["sidechain"]
    assert sc["backbone_only_binder"] is True
    assert sc["predicted_mask"] is False       # M2: post_aa not supervised by default
    loss = training_configs["loss"]
    for k in ("weight_sc_local", "weight_sc_phys", "weight_bb_post", "weight_aa_post"):
        assert k in loss


# ---------------- M2: loss side of the post-AA gate ----------------

def test_post_aa_absent_gives_zero_and_no_leak_term():
    from pxdesign_train.loss import PXDesignLoss

    pred = torch.randn(1, 6, 3, requires_grad=True)
    gt = torch.randn(1, 6, 3)
    out = PXDesignLoss(align_before_mse=False, weight_lddt=0.0, weight_disto=0.0)(
        pred, gt, torch.tensor([10.0]), torch.ones(6), torch.ones(6, dtype=torch.bool),
        aa_clean=torch.randint(0, 20, (6,)), aa_loss_mask=torch.ones(6),
        post_aa_logits=None,   # model gates this off under GT composition
    )
    assert out["aa_post"].item() == 0.0


# ---------------- Per-sample (per-sigma) AA loss ----------------

def test_aa_loss_per_sample_equals_mean_of_per_sample_ce():
    """AA loss over [B, N_sample, L, 20] == mean of per-sample cross-entropy."""
    import torch.nn.functional as F
    from pxdesign_train.loss import PXDesignLoss

    torch.manual_seed(0)
    B, S, L = 1, 4, 5
    logits = torch.randn(B, S, L, 20, requires_grad=True)
    aa_clean = torch.randint(0, 20, (B, L))
    aa_mask = torch.ones(B, L)

    ce, acc, frac = PXDesignLoss(align_before_mse=False)._aa_term(
        logits, aa_clean, aa_mask, aa_t=None
    )
    ref = torch.stack([F.cross_entropy(logits[0, s], aa_clean[0]) for s in range(S)]).mean()
    assert torch.allclose(ce, ref, atol=1e-5)      # per-sigma CE, then averaged
    assert 0.0 <= acc.item() <= 1.0
    ce.backward()
    assert torch.isfinite(logits.grad).all()
    # Every sample receives gradient (not just the least-noisy one).
    assert (logits.grad.abs().sum(dim=(0, 2, 3)) > 0).all()


def test_aa_loss_per_sample_batch_two_matches_reference():
    """Batched [B, S, L, 20] logits broadcast [B, L] labels over S."""
    import torch.nn.functional as F
    from pxdesign_train.loss import PXDesignLoss

    torch.manual_seed(4)
    B, S, L = 2, 3, 5
    logits = torch.randn(B, S, L, 20, requires_grad=True)
    aa_clean = torch.randint(0, 20, (B, L))
    aa_mask = torch.ones(B, L)

    ce, _, _ = PXDesignLoss(align_before_mse=False)._aa_term(
        logits, aa_clean, aa_mask, aa_t=None
    )
    ref = torch.stack(
        [F.cross_entropy(logits[b, s], aa_clean[b]) for b in range(B) for s in range(S)]
    ).mean()
    assert torch.allclose(ce, ref, atol=1e-5)
    ce.backward()
    assert torch.isfinite(logits.grad).all()


def test_aa_head_broadcasts_over_sample_axis():
    """The model now calls the AA head on [B, N_sample, L, c] (per-sigma). The
    head must broadcast over the sample axis and over per-item aa_t."""
    from pxdesign_train.heads import DesignResidueTypeHead

    head = DesignResidueTypeHead(c_s=16, no_bins=20)
    # batched [B, S, L, c] with per-item aa_t [B]
    a_full = torch.randn(2, 4, 5, 16)
    o = head(a_full, aa_t=torch.rand(2))
    assert o.shape == (2, 4, 5, 20) and torch.isfinite(o).all()
    # reduced [B, L, c] (what S_phi/h_res consume)
    o_red = head(torch.randn(2, 5, 16), aa_t=torch.rand(2))
    assert o_red.shape == (2, 5, 20)
    # collapsed-batch per-sample [S, L, c] with scalar aa_t (trainer path)
    o_c = head(torch.randn(4, 5, 16), aa_t=torch.tensor(0.3))
    assert o_c.shape == (4, 5, 20) and torch.isfinite(o_c).all()


def test_aa_loss_single_sample_matches_no_sample_axis():
    from pxdesign_train.loss import PXDesignLoss

    torch.manual_seed(1)
    L = 6
    logits_ns = torch.randn(1, L, 20)
    aa_clean = torch.randint(0, 20, (1, L))
    aa_mask = torch.ones(1, L)
    term = PXDesignLoss(align_before_mse=False)._aa_term
    ce_ns = term(logits_ns, aa_clean, aa_mask, aa_t=None)[0]
    ce_s = term(logits_ns.unsqueeze(1), aa_clean, aa_mask, aa_t=None)[0]  # [1,1,L,20]
    assert torch.allclose(ce_ns, ce_s, atol=1e-6)


# ---------- side-chain path is unaffected by the per-sample AA change ----------

def test_sidechain_consumes_reduced_logits_not_per_sample():
    """S_phi must receive the REDUCED AA logits (no sample axis) — exactly what it
    got before the per-sample change. Reproduces the model's dual-output split
    (aa_logits per-sample for the loss, aa_logits_reduced for S_phi) at the tensor
    level and checks S_phi runs on the reduced one and rejects the per-sample one."""
    import pytest
    from pxdesign_train.heads import DesignResidueTypeHead
    from pxdesign_train.sidechain.module import SideChainModule
    from pxdesign_train.sidechain.feedback import HResFeedback
    from pxdesign_train.sidechain.instantiate import MAX_SC, sidechain_atom_name_ids, sidechain_mask

    torch.manual_seed(0)
    S, L, c = 4, 3, 16
    a_cache = torch.randn(S, L, c)            # [N_sample, N_token, c] (collapsed batch)
    head = DesignResidueTypeHead(c_s=c, no_bins=20)

    token_repr = a_cache.mean(0)              # model's reduced repr -> h_res / S_phi
    aa_logits_reduced = head(token_repr)      # [L, 20]  (what S_phi consumes)
    aa_logits_persample = head(a_cache)       # [S, L, 20] (what the AA loss consumes)
    assert aa_logits_reduced.shape == (L, 20)
    assert aa_logits_persample.shape == (S, L, 20)

    restypes = ["PHE", "LYS", "ALA"]
    ids = sidechain_atom_name_ids(restypes)[None]
    m = sidechain_mask(restypes)[None]
    noisy = torch.randn(1, L, MAX_SC, 3)
    scm = SideChainModule(c_res=c, c_atom=16, n_type=20)
    fb = HResFeedback(c_atom=16, c_res=c)

    # Reduced logits (unsqueezed to batch) run cleanly — identical contract to
    # before the per-sample change.
    y0, feats = scm(token_repr[None], aa_logits_reduced[None], ids, m, noisy,
                    torch.ones(1), ca_coords=torch.randn(1, L, 3))
    assert y0.shape == (1, L, MAX_SC, 3)
    hp = fb(feats, m, token_repr[None])
    assert hp.shape == (1, L, c)

    # Routing the per-sample logits into S_phi is a shape error — this is why the
    # model keeps them separate.
    with pytest.raises(Exception):
        scm(token_repr[None], aa_logits_persample[None], ids, m, noisy,
            torch.ones(1), ca_coords=torch.randn(1, L, 3))


# ---------- per-sigma side-chain path (simulates model._train_forward) ----------

def test_per_sigma_sidechain_forward_and_loss():
    """Reproduces the model's per-sigma side-chain flow with real modules:
    flatten [N_sample, L, C] -> S_phi batch, tile per-token frames, per-sigma
    physical loss, and a per-sigma-averaged local loss. Locks the shape logic the
    GPU-only model path relies on."""
    from pxdesign_train.sidechain.module import SideChainModule
    from pxdesign_train.sidechain.feedback import HResFeedback
    from pxdesign_train.sidechain.losses import sidechain_local_loss
    from pxdesign_train.sidechain.physical import physical_loss
    from pxdesign_train.sidechain.frames import to_global
    from pxdesign_train.sidechain.instantiate import MAX_SC, sidechain_atom_name_ids, sidechain_mask

    torch.manual_seed(0)
    S, L, C, A = 4, 3, 16, MAX_SC          # N_sample=4 different sigmas
    a_full = torch.randn(S, L, C)          # h_res_sigma [N_sample, L, C]
    sigma = torch.tensor([50.0, 8.0, 2.0, 0.5])   # one sigma per row
    aa_logits_sigma = torch.randn(S, L, 20)

    restypes = ["PHE", "LYS", "ALA"]
    ids = sidechain_atom_name_ids(restypes)[None].expand(S, -1, -1)   # tile to batch S
    m = sidechain_mask(restypes)[None].expand(S, -1, -1)
    noisy = torch.randn(S, L, A, 3)
    t = 0.25 * sigma.clamp_min(1e-4).log()                # real per-sigma time
    fR = torch.eye(3)[None, None].expand(S, L, 3, 3)      # GT frames tiled
    ft = torch.zeros(S, L, 3)

    scm = SideChainModule(c_res=C, c_atom=16, n_type=20)
    fb = HResFeedback(c_atom=16, c_res=C)
    y0, feats = scm(a_full, aa_logits_sigma, ids, m, noisy, t, ca_coords=ft)
    assert y0.shape == (S, L, A, 3)                       # per-sigma predictions
    hprime = fb(feats, m, a_full)
    assert hprime.shape == (S, L, C)                      # per-sigma h_res'

    # per-sigma-averaged local loss (gt broadcasts over the sigma axis)
    gt_local = torch.randn(L, A, 3)
    loss = sidechain_local_loss(y0, gt_local, m)
    assert loss.dim() == 0 and torch.isfinite(loss)

    # per-sigma physical loss (frames/bb broadcast to batch S)
    y_g = to_global(y0, fR, ft)
    bb = torch.randn(S, L, 4, 3)
    phys = physical_loss(y_g.reshape(S, L * A, 3),
                         context_coords=bb.reshape(S, L * 4, 3),
                         context_mask=torch.ones(S, L * 4, dtype=torch.bool),
                         valid_mask=m.reshape(S, L * A))
    assert torch.isfinite(phys["total"])

    # h_res' reduced over sigma for the cycle injection (substrate limit)
    reduced = hprime.reshape(1, S, L, C).mean(dim=1).squeeze(0)
    assert reduced.shape == (L, C)


def test_loss_scale_invariant_to_n_sample():
    """Smoke-check #6: per-sigma AA loss and sc_local are AVERAGES over the sample
    axis, so their scale must NOT grow with N_sample. Tiling identical rows across
    N_sample must reproduce the single-sample value (mean semantics, not sum)."""
    from pxdesign_train.loss import PXDesignLoss
    from pxdesign_train.sidechain.losses import sidechain_local_loss
    from pxdesign_train.sidechain.instantiate import MAX_SC, sidechain_mask

    torch.manual_seed(0)
    L = 5
    term = PXDesignLoss(align_before_mse=False)._aa_term
    logits1 = torch.randn(1, L, 20)
    aa_clean = torch.randint(0, 20, (1, L))
    aa_mask = torch.ones(1, L)
    ce1 = term(logits1, aa_clean, aa_mask, aa_t=None)[0]
    for S in (2, 8, 32):
        logits_s = logits1.unsqueeze(1).expand(1, S, L, 20).reshape(S, L, 20)
        ce_s = term(logits_s, aa_clean.expand(S, L), aa_mask.expand(S, L), aa_t=None)[0]
        assert torch.allclose(ce_s, ce1, atol=1e-5), f"AA loss scaled with N_sample={S}"

    # sc_local: masked mean, same invariance.
    amask = sidechain_mask(["PHE", "LYS", "ALA"])           # [L, MAX_SC]
    pred1 = torch.randn(3, MAX_SC, 3)
    gt = torch.randn(3, MAX_SC, 3)
    scl1 = sidechain_local_loss(pred1, gt, amask)
    for S in (2, 8):
        pred_s = pred1.unsqueeze(0).expand(S, 3, MAX_SC, 3)
        scl_s = sidechain_local_loss(pred_s, gt, amask)      # gt/mask broadcast
        assert torch.allclose(scl_s, scl1, atol=1e-5), f"sc_local scaled with N_sample={S}"


def test_sidechain_loss_tiles_batch_targets_to_per_sigma_predictions():
    """Loss accepts sc_pred [B*S,L,A,3] with GT/routing masks still [B,L,...]."""
    from pxdesign_train.loss import PXDesignLoss
    from pxdesign_train.sidechain.instantiate import MAX_SC

    torch.manual_seed(5)
    B, S, L, A, N = 2, 3, 4, MAX_SC, 12
    loss_fn = PXDesignLoss(align_before_mse=False)
    out = loss_fn(
        pred_coordinate=torch.zeros(1, N, 3),
        gt_coordinate_aug=torch.zeros(1, N, 3),
        sigma=torch.ones(1),
        coordinate_mask=torch.ones(N),
        rep_atom_mask=torch.zeros(N, dtype=torch.bool),
        sc_pred_local=torch.randn(B * S, L, A, 3, requires_grad=True),
        sc_gt_local=torch.randn(B, L, A, 3),
        sc_atom_mask=torch.ones(B * S, L, A, dtype=torch.bool),
        sc_type_match=torch.ones(B, L, dtype=torch.bool),
    )
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["sc_local"])


def test_h_res_sigma_row_alignment():
    """h_res_sigma / aa_logits_sigma / sigma must stay row-aligned after flatten."""
    S, L, C = 3, 2, 8
    h = torch.arange(S * L * C, dtype=torch.float32).reshape(S, L, C)
    sig = torch.tensor([10.0, 1.0, 0.1])
    flat_h = h.reshape(-1, L, C)
    flat_sig = sig.reshape(-1)
    assert flat_h.shape == (S, L, C) and flat_sig.shape == (S,)
    for i in range(S):
        assert torch.equal(flat_h[i], h[i]) and flat_sig[i] == sig[i]


# ---------- Stage III: instantiate side-chain atom set from PREDICTED type ----------

def test_instantiate_from_type_indices():
    from pxdesign_train.sidechain.instantiate import (
        instantiate_from_type_indices, sidechain_mask, MAX_SC,
    )

    # STD_AA_3 order: ALA=0 (CB only), GLY=7 (no side chain).
    ids, mask = instantiate_from_type_indices(torch.tensor([0, 7]))
    assert ids.shape == (2, MAX_SC) and mask.shape == (2, MAX_SC)
    assert int(mask[0].sum()) == 1        # ALA -> CB
    assert int(mask[1].sum()) == 0        # GLY -> empty
    # matches the string-based instantiation
    assert torch.equal(mask, sidechain_mask(["ALA", "GLY"]))


# ---------- predicted-backbone frames + stop-grad pseudo-target (paper II-B) ----------

def test_frames_from_backbone_index_matches_direct_build():
    from pxdesign_train.sidechain.frames import frames_from_backbone_index, build_frame

    torch.manual_seed(0)
    N_atom = 12
    coords = torch.randn(2, N_atom, 3)                 # [B, N_atom, 3]
    bb_idx = torch.tensor([[0, 1, 2], [4, 5, 6], [-1, -1, -1]])  # 3 tokens; last invalid
    R, t, valid = frames_from_backbone_index(coords, bb_idx)
    assert R.shape == (2, 3, 3, 3) and t.shape == (2, 3, 3)
    assert valid.tolist() == [True, True, False]
    R0, t0 = build_frame(coords[:, 0], coords[:, 1], coords[:, 2])
    assert torch.allclose(R[:, 0], R0) and torch.allclose(t[:, 0], t0)


def test_predicted_frame_pseudo_target_stopgrad():
    """Global pseudo-target: attaching y^GT to the predicted frame with the frame
    stop-gradded gives zero loss at y_pred==y^GT, and the target side gets NO grad."""
    from pxdesign_train.sidechain.frames import to_global

    torch.manual_seed(0)
    L, A = 2, 3
    R = torch.randn(L, 3, 3, requires_grad=True)
    t = torch.randn(L, 3, requires_grad=True)
    y_gt = torch.randn(L, A, 3)
    y_pred = y_gt.clone().requires_grad_(True)          # prediction == target
    y_g = to_global(y_pred, R, t)
    pseudo = to_global(y_gt, R.detach(), t.detach())    # stop-grad target frame
    loss = ((y_g - pseudo) ** 2).sum()
    assert torch.allclose(loss, torch.zeros(()), atol=1e-5)
    loss.backward()
    assert R.grad is not None                            # backbone still learns via pred side
    assert torch.allclose(y_pred.grad, torch.zeros_like(y_pred.grad), atol=1e-5)


# ---------------- M3: side-chain global assembly math ----------------

def test_local_global_roundtrip_matches_assembly():
    from pxdesign_train.sidechain.frames import build_frame, to_global, to_local

    n = torch.tensor([[-1.0, 0.2, 0.1]])
    ca = torch.tensor([[0.0, 0.0, 0.0]])
    c = torch.tensor([[1.0, 0.3, -0.2]])
    R, t = build_frame(n, ca, c)
    gt_global = torch.tensor([[[0.2, -0.8, 0.9], [1.5, 1.0, -0.3]]])  # [1,2,3]
    local = to_local(gt_global, R, t)
    back = to_global(local, R, t)              # the exact op cogenerate uses
    assert torch.allclose(back, gt_global, atol=1e-5)
