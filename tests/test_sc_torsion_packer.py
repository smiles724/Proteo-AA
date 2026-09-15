"""The four claims the APM-style V0 packer is allowed to make.

1. It reads NO side-chain coordinate. Not "the template is a weak prior" -- the
   output is bit-identical under an arbitrary change of `noisy_coords`.
2. Its covalent geometry is a constant, because BuildSC places the atoms.
3. Its geometric channel is SE(3)-invariant: rotate and translate the whole
   input and the predicted torsions do not move, while the coordinates move with
   the structure. This holds with the a-token channel OFF; with it ON, the
   invariance is only as good as the trunk's, which is why the A/B exists.
4. The a-token ablation changes information, not architecture: same parameter
   names, same shapes, same initial values; the OFF arm's output does not depend
   on h_res and the ON arm's does.
"""
import math

import pytest
import torch

import pxdesign_train.sidechain.buildsc as buildsc
from pxdesign_train.sidechain.chemistry import canonical_registry
from pxdesign_train.sidechain.chi_constants import MAX_CHI
from pxdesign_train.sidechain.frames import build_frame
from pxdesign_train.sidechain.instantiate import MAX_SC, sidechain_atoms
from pxdesign_train.sidechain.packer import TorsionPacker
from pxdesign_train.sidechain.torsion_loss import (
    CHI_PI_PERIODIC, native_chi_targets, torsion_angle_loss,
)

STD = buildsc.STD_AA_3
RESIDUES = ("ARG", "PHE", "LYS", "SER", "TRP", "GLU", "GLY", "PRO", "ASP", "TYR")


def _example(seed=0, L=None):
    torch.manual_seed(seed)
    from pxdesign_train.sidechain.chi_constants import IDEAL_BB_LOCAL
    types = torch.tensor([[STD.index(a) for a in RESIDUES]])
    L = types.shape[1]
    ids = torch.zeros(1, L, MAX_SC, dtype=torch.long)
    mask = torch.zeros(1, L, MAX_SC, dtype=torch.bool)
    for j, name in enumerate(RESIDUES):
        n = len(sidechain_atoms(name))
        mask[0, j, :n] = True
        ids[0, j, :n] = torch.arange(1, n + 1)
    bb = torch.randn(1, L, 4, 3) * 0.8
    bb = bb + torch.arange(L, dtype=torch.float)[None, :, None, None] * 3.8
    frame_R, frame_t = build_frame(bb[:, :, 0], bb[:, :, 1], bb[:, :, 2])
    logits = torch.nn.functional.one_hot(types, 20).float() * 40.0 - 20.0
    return dict(types=types, ids=ids, mask=mask, bb=bb, frame_R=frame_R,
                frame_t=frame_t, logits=logits, h_res=torch.randn(1, L, 32),
                res_mask=torch.ones(1, L, dtype=torch.bool),
                residue_index=torch.arange(L)[None])


def _packer(seq_cond="a_token", seed=0, scramble=False, **kwargs):
    """A small TorsionPacker. Same architecture as the real one, tiny sizes.

    `random_torsion_input` defaults to False here: APM feeds a uniform random
    torsion, which makes the packer's output stochastic and every determinism
    assertion below meaningless. Tests that care about APM's default say so.
    """
    # numpy too: the AF2-style Linear initializers draw through scipy's
    # truncnorm, i.e. numpy's global RNG. Seeding only torch leaves half the
    # weights different between two "identically seeded" modules -- which is the
    # bug this mirrors in the training entry point (see
    # test_training_entry_seeds_before_building_the_model).
    from pxdesign_train.runner.sc_stream import seed_all
    seed_all(seed)
    opts = dict(c_res=32, c_node=32, c_pair=16, n_blocks=2, ipa_c_hidden=4,
                ipa_no_heads=2, no_qk_points=2, no_v_points=3,
                seq_tfmr_num_heads=2, seq_tfmr_num_layers=1,
                transformer_dropout=0.0, num_torsion_blocks=2,
                c_pos_emb=16, c_timestep_emb=16, edge_feat_dim=8, edge_num_bins=6,
                random_torsion_input=False)
    opts.update(kwargs)
    m = TorsionPacker(seq_cond=seq_cond, **opts).eval()
    if scramble:
        with torch.no_grad():
            for name, q in m.named_parameters():
                if q.dim() > 1 and "plm" not in name:
                    q.normal_(0, 0.5)
    return m


def _run(m, ex, *, noisy=None, h_res=None):
    L = ex["types"].shape[1]
    if noisy is None:
        noisy = torch.zeros(1, L, MAX_SC, 3)
    return m(
        ex["h_res"] if h_res is None else h_res,
        ex["logits"], ex["ids"], ex["mask"], noisy, torch.zeros(1),
        ca_coords=ex["bb"][:, :, 1], frame_R=ex["frame_R"], frame_t=ex["frame_t"],
        bb_coords=ex["bb"], res_mask=ex["res_mask"], residue_index=ex["residue_index"],
    )


def test_output_does_not_depend_on_any_side_chain_coordinate():
    """The V0 contract: p(chi | BB, res_type, seq channel). No x_t^SC, at all."""
    ex = _example()
    m = _packer(scramble=True)
    L = ex["types"].shape[1]
    a = _run(m, ex, noisy=torch.zeros(1, L, MAX_SC, 3))[0]
    b = _run(m, ex, noisy=torch.randn(1, L, MAX_SC, 3) * 10.0)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_arbitrary_weights_cannot_break_covalent_geometry():
    ex = _example()
    xyz = _run(_packer(scramble=True), ex)[0]
    registry = canonical_registry()
    worst = 0.0
    for j, name in enumerate(RESIDUES):
        record = registry.get(name)
        if record is None or not getattr(record, "bonds", None):
            continue
        slots = {n: k for k, n in enumerate(sidechain_atoms(name)[:MAX_SC])}
        ideal, _ = buildsc.build_sidechain_local(ex["types"][0, j])
        for u, v in record.bonds:
            if u in slots and v in slots:
                iu, iv = slots[u], slots[v]
                worst = max(worst, abs(float(
                    (xyz[0, j, iu] - xyz[0, j, iv]).norm() - (ideal[iu] - ideal[iv]).norm()
                )))
    assert worst < 1e-4


def _rotate(ex):
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.randn(3) * 25.0
    moved = dict(ex)
    moved["bb"] = ex["bb"] @ q.T + shift
    moved["frame_R"], moved["frame_t"] = build_frame(
        moved["bb"][:, :, 0], moved["bb"][:, :, 1], moved["bb"][:, :, 2])
    return moved, q, shift


def test_se3_invariant_without_rotvecs():
    """IPA is invariant, so with the rotvec node feature off the packer is too."""
    ex = _example()
    m = _packer(seq_cond="none", scramble=True, embed_rotvecs=False)
    xyz, _, _ = _run(m, ex)
    chi = m.last_torsions["chi"].clone()
    moved, q, shift = _rotate(ex)
    xyz2, _, _ = _run(m, moved)
    # Exact in principle (see tests/test_sc_ipa.py, where IPA alone matches to
    # 0.0); this tolerance is float32 accumulation through a trunk whose weights
    # were scrambled to std 0.5, which is far outside the trained regime.
    torch.testing.assert_close(chi, m.last_torsions["chi"], atol=2e-3, rtol=0)
    expected = torch.where(ex["mask"][..., None], xyz @ q.T + shift, torch.zeros(()))
    torch.testing.assert_close(xyz2, expected, atol=5e-3, rtol=0)


def test_apm_rotvec_feature_breaks_that_invariance():
    """Not a bug -- APM's own node features include the global rotation vector.

    Pinned so nobody reads the invariance test above as a property of the
    default configuration. `embed_rotvecs=True` is APM's setting and is the
    default; the price is that the packer must LEARN invariance from the rigid
    augmentation instead of having it by construction.
    """
    ex = _example()
    m = _packer(seq_cond="none", scramble=True, embed_rotvecs=True)
    _run(m, ex)
    chi = m.last_torsions["chi"].clone()
    moved, _, _ = _rotate(ex)
    _run(m, moved)
    assert (chi - m.last_torsions["chi"]).abs().max() > 1e-3


def test_the_four_arms_share_one_parameter_set():
    """The ablation must change information, not capacity.

    Both projections are always constructed, so `none`/`a_token`/`plm`/`both`
    have identical parameter names, shapes and initial values. Only which one is
    fed differs.
    """
    arms = {}
    for cond in ("none", "a_token"):
        arms[cond] = _packer(cond, seed=3)
    keys = [list(m.state_dict()) for m in arms.values()]
    assert keys[0] == keys[1]
    for key, value in arms["none"].state_dict().items():
        torch.testing.assert_close(value, arms["a_token"].state_dict()[key],
                                   atol=0, rtol=0)
    # plm/both need a checkpoint path, but the parameter set is the same one.
    with pytest.raises(ValueError, match="plm_checkpoint"):
        _packer("plm", seed=3)
    plm_arm = _packer("both", seed=3, plm_checkpoint="/nonexistent/esm2.pt")
    assert list(plm_arm.state_dict()) == keys[0]


def test_only_the_selected_arm_reads_its_channel():
    ex = _example()
    other = torch.randn_like(ex["h_res"]) * 5.0

    off = _packer("none", seed=4, scramble=True)
    torch.testing.assert_close(_run(off, ex)[0], _run(off, ex, h_res=other)[0],
                               atol=0, rtol=0)
    on = _packer("a_token", seed=4, scramble=True)
    assert (_run(on, ex)[0] - _run(on, ex, h_res=other)[0]).abs().max() > 1e-4

    # The unused projection still receives a (zero) gradient, so the two arms
    # have the same optimizer state rather than a silently different one.
    out = _run(off, ex)[0]
    out.square().sum().backward()
    assert off.a_proj.weight.grad is not None
    assert float(off.a_proj.weight.grad.abs().sum()) == 0.0


def test_frozen_esm_is_not_part_of_the_model():
    """A registered ESM would become 651M TRAINABLE parameters under apply_phase."""
    m = _packer("both", plm_checkpoint="/nonexistent/esm2.pt")
    assert not any("plm_runner" in k or "_model" in k for k in m.state_dict())
    trainable = {n for n, q in m.named_parameters() if q.requires_grad}
    assert any(n.startswith("plm_conditioner.") for n in trainable)
    assert all("esm" not in n.lower() for n in trainable)
    # plm_s_combine is a learned softmax over layers, zero-init like APM's.
    assert torch.equal(m.plm_conditioner.plm_s_combine,
                       torch.zeros_like(m.plm_conditioner.plm_s_combine))


def test_random_torsion_input_is_apm_default_and_is_stochastic():
    """APM feeds a uniform random torsion; that makes the packer non-deterministic.

    Pinned because it is surprising for something called a one-step packer, and
    because the switch that turns it off is the one thing standing between this
    and a reproducible evaluation.
    """
    ex = _example()
    m = _packer(seq_cond="none", scramble=True, random_torsion_input=True)
    torch.manual_seed(11)
    a = _run(m, ex)[0]
    torch.manual_seed(12)
    b = _run(m, ex)[0]
    assert (a - b).abs().max() > 1e-5

    det = _packer(seq_cond="none", scramble=True, random_torsion_input=False)
    torch.manual_seed(11)
    c = _run(det, ex)[0]
    torch.manual_seed(12)
    d = _run(det, ex)[0]
    torch.testing.assert_close(c, d, atol=0, rtol=0)


def test_missing_frames_or_backbone_are_refused():
    ex = _example()
    m = _packer()
    L = ex["types"].shape[1]
    with pytest.raises(ValueError, match="frames"):
        m(ex["h_res"], ex["logits"], ex["ids"], ex["mask"],
          torch.zeros(1, L, MAX_SC, 3), torch.zeros(1), bb_coords=ex["bb"])


def test_masked_residues_are_finite_and_zero():
    ex = _example()
    ex["res_mask"] = torch.zeros_like(ex["res_mask"])
    ex["res_mask"][0, 0] = True
    xyz, feats, bb_feats = _run(_packer(scramble=True), ex)
    assert torch.isfinite(xyz).all() and torch.isfinite(feats).all()
    assert torch.isfinite(bb_feats).all()
    xyz.square().sum().backward()


def test_torsion_loss_is_zero_on_a_perfect_prediction():
    chi = torch.tensor([[[0.4, -1.2, 2.5, 0.0]]])
    valid = torch.tensor([[[True, True, True, False]]])
    types = torch.tensor([[STD.index("LYS")]])
    raw = torch.stack([chi.sin(), chi.cos()], dim=-1)
    loss, metrics = torsion_angle_loss(raw, chi, valid, types)
    assert float(loss) < 1e-6
    assert float(metrics["chi_mae_deg"]) < 1e-3
    assert float(metrics["chi_supervised"]) == 3.0


def test_pi_periodic_torsions_are_not_penalised_for_the_flip():
    """ASP chi2 flipped by 180 degrees is the same side chain."""
    gt = torch.tensor([[[0.4, 1.0, 0.0, 0.0]]])
    valid = torch.tensor([[[True, True, False, False]]])
    asp = torch.tensor([[STD.index("ASP")]])
    asn = torch.tensor([[STD.index("ASN")]])
    flipped = gt.clone()
    flipped[..., 1] = flipped[..., 1] + math.pi
    raw = torch.stack([flipped.sin(), flipped.cos()], dim=-1)
    assert float(torsion_angle_loss(raw, gt, valid, asp)[0]) < 1e-6
    # ASN chi2 is NOT pi-periodic (the amide has distinguishable O and N), so the
    # same flip is a full error there. This is the line between the two tables.
    assert not bool(CHI_PI_PERIODIC[STD.index("ASN"), 1])
    assert float(torsion_angle_loss(raw, gt, valid, asn)[0]) > 1.0


def test_angle_norm_term_penalises_an_unnormalised_head():
    gt = torch.zeros(1, 1, MAX_CHI)
    valid = torch.zeros(1, 1, MAX_CHI, dtype=torch.bool)
    valid[..., 0] = True
    types = torch.tensor([[STD.index("SER")]])
    unit = torch.zeros(1, 1, MAX_CHI, 2)
    unit[..., 0, 0] = 1.0
    short = unit * 0.1
    long_ = unit.clone()
    long_[..., 0, 0] = 1.0
    assert float(torsion_angle_loss(short, gt, valid, types)[0]) > float(
        torsion_angle_loss(long_, gt, valid, types)[0]
    )


def test_native_chi_targets_round_trip_through_buildsc():
    """Measured chi of a structure built at known chi is that chi."""
    types = torch.tensor([STD.index(a) for a in ("ARG", "PHE", "SER", "GLY")])
    torch.manual_seed(1)
    chi = torch.rand(4, MAX_CHI) * 2 * math.pi - math.pi
    from pxdesign_train.sidechain.chi_constants import CHI_MASK, IDEAL_BB_LOCAL
    built, slots = buildsc.build_sidechain_local(
        types, torch.where(CHI_MASK[types], chi, torch.full_like(chi, float("nan")))
    )
    gt, valid = native_chi_targets(
        types, built, IDEAL_BB_LOCAL[types], slots,
        bb_observed=torch.ones(4, 3, dtype=torch.bool),
    )
    delta = torch.atan2((gt - chi).sin(), (gt - chi).cos()).abs()
    assert float(delta[valid].max()) < 1e-4
    # GLY owns no torsion; PRO-style ring closure and absent chis stay masked.
    assert not bool(valid[3].any())
    assert bool(valid[0].all())


def test_unobserved_atoms_remove_their_torsion_from_supervision():
    types = torch.tensor([STD.index("ARG")])
    from pxdesign_train.sidechain.chi_constants import IDEAL_BB_LOCAL
    built, slots = buildsc.build_sidechain_local(types, torch.zeros(1, MAX_CHI))
    partial = slots.clone()
    partial[0, 3:] = False            # CB, CG, CD present; NE onwards unresolved
    _, valid = native_chi_targets(types, built, IDEAL_BB_LOCAL[types], partial)
    assert bool(valid[0, 0]) and bool(valid[0, 1])
    assert not bool(valid[0, 2]) and not bool(valid[0, 3])


def _phase_features(seed=0):
    """The subset of the SC-only phase's feature dict that L_chi consumes.

    Shapes follow the real contract: per-token features are UNBATCHED [L, ...],
    while the packer's output carries the flattened (item x sigma) row axis.

    The backbone is IDEAL N/CA/C placed by a random rigid transform per residue.
    That is not cosmetic: L_chi's target is measured against the residue's OWN
    backbone (the convention `metrics.diagnose_packing` reports in), while BuildSC
    poses the side chain against the IDEAL N. The two agree exactly only when the
    backbone has ideal N-CA-C geometry -- see
    `test_target_convention_matches_the_builder_on_ideal_backbones`, which pins
    the size of the disagreement when it does not.
    """
    torch.manual_seed(seed)
    from pxdesign_train.sidechain.chi_constants import CHI_MASK, IDEAL_BB_LOCAL
    types = torch.tensor([STD.index(a) for a in RESIDUES])
    L = types.shape[0]
    q = torch.linalg.qr(torch.randn(L, 3, 3))[0]
    q = q * torch.sign(torch.linalg.det(q))[:, None, None]
    shift = torch.randn(L, 3) * 20.0
    bb3 = torch.einsum("lij,laj->lai", q, IDEAL_BB_LOCAL[types]) + shift[:, None]
    # O is carried in the feature dict but never used by the frame or by L_chi.
    bb = torch.cat([bb3, bb3[:, 2:3] + 1.23], dim=1)
    frame_R, frame_t = build_frame(bb[:, 0], bb[:, 1], bb[:, 2])
    chi = torch.rand(L, MAX_CHI) * 2 * math.pi - math.pi
    gt_local, slots = buildsc.build_sidechain_local(
        types, torch.where(CHI_MASK[types], chi, torch.full_like(chi, float("nan")))
    )
    feat = dict(sc_gt_local=gt_local, sc_frame_R=frame_R, sc_frame_t=frame_t,
                sc_bb_coords=bb, sc_bb_observed_mask=torch.ones(L, 4, dtype=torch.bool))
    return types, feat, slots, chi


def test_packing_chi_loss_recovers_the_torsions_it_was_built_from():
    from pxdesign_train.sidechain.chi_constants import CHI_MASK, CHI_ROTATABLE
    from pxdesign_train.sidechain.torsion_loss import packing_chi_loss
    types, feat, slots, chi = _phase_features()
    L = types.shape[0]
    perfect = torch.stack([chi.sin(), chi.cos()], dim=-1)[None]     # [1, L, 4, 2]
    loss, metrics = packing_chi_loss(
        feat, {"sc_pred_chi_raw": perfect}, types, slots[None].expand(1, L, MAX_SC)
    )
    assert float(loss) < 1e-4
    assert float(metrics["torsion/chi_mae_deg"]) < 0.1
    # PRO's two chis are counted by CHI_MASK but are NOT supervised: the ring is
    # not rotatable, so BuildSC cannot realise a predicted value there.
    assert float(metrics["torsion/chi_supervised"]) == float(
        (CHI_MASK & CHI_ROTATABLE)[types].sum())
    assert float(CHI_MASK[types].sum()) - float((CHI_MASK & CHI_ROTATABLE)[types].sum()) == 2.0


def test_packing_chi_loss_is_large_for_a_wrong_prediction():
    from pxdesign_train.sidechain.torsion_loss import packing_chi_loss
    types, feat, slots, chi = _phase_features()
    L = types.shape[0]
    wrong = chi + math.pi / 2
    raw = torch.stack([wrong.sin(), wrong.cos()], dim=-1)[None]
    loss, metrics = packing_chi_loss(
        feat, {"sc_pred_chi_raw": raw}, types, slots[None].expand(1, L, MAX_SC)
    )
    assert float(loss) > 1.0
    # 90 degrees everywhere except the pi-periodic torsions, which fold to 90 too.
    assert 85.0 < float(metrics["torsion/chi_mae_deg"]) <= 90.5


def test_target_convention_matches_the_builder_on_ideal_backbones():
    """How far apart the two chi conventions are, measured rather than assumed.

    L_chi's target is chi measured against the residue's own N; BuildSC poses the
    side chain against the ideal N. On an ideal backbone they coincide exactly. On
    a perturbed one they do not, and the gap is the systematic offset between the
    torsion target and the coordinates the same chi would produce. Native N-CA
    geometry is tight (~0.02 A), so this stays well under a degree in real data --
    but it is a real term, so it is pinned here rather than assumed away.
    """
    from pxdesign_train.sidechain.chi_constants import CHI_MASK, IDEAL_BB_LOCAL
    types, feat, slots, chi = _phase_features(seed=2)
    measured, valid = native_chi_targets(
        types, feat["sc_gt_local"], IDEAL_BB_LOCAL[types], slots,
        bb_observed=torch.ones(types.shape[0], 3, dtype=torch.bool))
    ideal_gap = torch.atan2((measured - chi).sin(), (measured - chi).cos()).abs()
    assert float(ideal_gap[valid].max()) < 1e-4

    # Now perturb N by a realistic 0.02 A and re-measure against that backbone.
    bb = feat["sc_bb_coords"].clone()
    bb[:, 0] = bb[:, 0] + torch.randn_like(bb[:, 0]) * 0.02
    local = torch.einsum("lij,laj->lai", feat["sc_frame_R"].transpose(-1, -2),
                         bb[:, :3] - feat["sc_frame_t"][:, None])
    perturbed, _ = native_chi_targets(
        types, feat["sc_gt_local"], local, slots,
        bb_observed=torch.ones(types.shape[0], 3, dtype=torch.bool))
    gap = torch.atan2((perturbed - chi).sin(), (perturbed - chi).cos()).abs()
    assert float(gap[valid].max()) < math.radians(2.0)


def _model_configs(**sidechain):
    import copy
    from protenix.config.config import parse_configs
    from pxdesign_train.configs.configs_train import training_configs
    t = copy.deepcopy(training_configs)
    t["enable_sidechain"] = True
    t["sidechain"].update(sidechain)
    cfg = parse_configs(t, arg_str="")
    cfg.load_strict = False
    return cfg


def test_model_builds_the_packer_and_rewires_the_feedback_width():
    """The feedback contract must follow the module that is actually built.

    HResFeedback / the a- and q-fusions are all constructed from one `c_atom`
    variable. The packer is residue-level and has its own width, so if that
    variable is not rebound the first feedback call dies on a shape mismatch --
    at step 1 of a 24 h allocation.
    """
    from pxdesign_train.model import ProtenixDesignTrain
    cfg = _model_configs(torsion_packer=True, packer_c_node=64, packer_c_pair=32,
                         packer_n_blocks=1, packer_ipa_no_heads=4,
                         packer_seq_tfmr_num_heads=2, a_bs_concat=False, q_bs=False)
    m = ProtenixDesignTrain(cfg)
    assert isinstance(m.sidechain_module, TorsionPacker)
    assert m.sc_torsion_packer is True and m.sc_packer_seq_cond == "a_token"
    assert m.sidechain_feedback.pool_proj.in_features == 64
    assert m.sidechain_module.c_atom == 64


def test_model_refuses_packer_plus_coordinate_module_switches():
    """Those switches describe a module that is not being built."""
    from pxdesign_train.model import ProtenixDesignTrain
    for conflicting in ({"chi_output": True}, {"edm": True},
                        {"template_residual": True, "frame_aware_head": True}):
        cfg = _model_configs(torsion_packer=True, packer_c_node=32, packer_c_pair=16,
                             packer_n_blocks=1, packer_ipa_no_heads=4,
                             packer_seq_tfmr_num_heads=2, **conflicting)
        with pytest.raises(ValueError, match="torsion_packer"):
            ProtenixDesignTrain(cfg)


def test_layout_keys_record_the_packer_and_its_arm():
    """A donor trained with another arm must be refused, not silently loaded."""
    from pxdesign_train.checkpoints import (
        SC_LAYOUT_KEYS, SC_LAYOUT_KEYS_OPTIONAL, SC_LAYOUT_KEYS_STR, check_sc_layout)
    assert set(SC_LAYOUT_KEYS_OPTIONAL) == {"torsion_packer"}
    assert set(SC_LAYOUT_KEYS_STR) == {"packer_seq_cond"}

    class _Stub:
        def __init__(self, **kw):
            self.configs = type("C", (), {"sidechain": type("S", (), kw)()})()

    base = {k: False for k in SC_LAYOUT_KEYS}
    base.update(bb_context=True, type_logits_input=True)
    saved = dict(base, torsion_packer=True, packer_seq_cond="a_token")
    assert check_sc_layout(_Stub(**saved), {"sidechain_arch": saved})["torsion_packer"] is True
    for other in ("none", "plm", "both"):
        wrong = _Stub(**dict(saved, packer_seq_cond=other))
        with pytest.raises(ValueError, match="packer_seq_cond"):
            check_sc_layout(wrong, {"sidechain_arch": saved})
    # A pre-packer donor has no opinion about these keys and must still load.
    assert check_sc_layout(_Stub(**base), {"sidechain_arch": dict(base)})


def test_training_entry_seeds_before_building_the_model():
    """Two arms differing by one flag must not also differ by their init.

    `--seed` reached only the data sampler here, while `train_sc_adaptation.py`
    seeded everything. Pinned at source level because the failure is invisible:
    both runs train fine, and the A/B silently measures init variance too.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "training" / "train_protenix_monomer.py").read_text()
    assert "seed_all(int(configs.seed))" in src
    assert src.index("seed_all(int(configs.seed))") < src.index("train_from_components(\n")


def test_masked_rows_do_not_poison_the_gradient():
    """Two NaN sources that `torch.where` cannot mask, both found on real data.

    A masked residue's `node_embed` is exactly zero, and AngleResnet's residual
    branch is zero-initialised with zero biases, so the head emits exactly
    (0, 0) there. Then:

      * `atan2(0, 0)` is 0 forward and NaN backward (packer.py), and
      * `vector_norm` at 0 is NaN backward (torsion_loss.py).

    Neither is removed by masking afterwards -- NaN * 0 is NaN -- so both reach
    the parameters. Forward checks and small-example gradient checks both pass;
    the symptom is "Nonfinite gradient before optimizer update 0" on the first
    real batch (jobs 117014/117023). Anomaly detection is the point of the test.
    """
    ex = _example()
    ex["res_mask"] = ex["res_mask"].clone()
    ex["res_mask"][0, 3:6] = False
    m = _packer("a_token", seed=1)
    m.train()
    torch.autograd.set_detect_anomaly(True)
    try:
        xyz, feats, _ = _run(m, ex)
        (xyz.square().sum() + feats.square().sum()).backward()
    finally:
        torch.autograd.set_detect_anomaly(False)
    assert all(torch.isfinite(q.grad).all()
               for _, q in m.named_parameters() if q.grad is not None)


def test_chi_loss_gradient_is_finite_at_an_exactly_zero_prediction():
    raw = torch.zeros(1, 3, MAX_CHI, 2, requires_grad=True)
    gt = torch.zeros(1, 3, MAX_CHI)
    valid = torch.zeros(1, 3, MAX_CHI, dtype=torch.bool)
    valid[0, 0, 0] = True
    loss, _ = torsion_angle_loss(raw, gt, valid, torch.zeros(1, 3, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(raw.grad).all()
