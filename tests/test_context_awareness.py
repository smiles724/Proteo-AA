"""Side-chain <-> context (receptor / motif / ligand) awareness.

Two defects this locks down, both found by auditing the code against the paper:

1. `contact_loss` reduced over an UNMASKED context axis. Callers build that axis
   from a per-TOKEN table, whose non-binder rows are phantoms — (0,0,0) on the
   GT-frame path (`sc_bb_coords` is zero-filled) and a duplicate of atom 0 on the
   predicted path (`bb_idx.clamp_min(0)`). `min(dim=-1)` selected them, so the
   runaway penalty the loss exists to impose was silently zeroed.

2. S_phi could not see the receptor AT ALL: the cross-residue attention masked
   every non-binder token out of its keys, and clash scored side-chain <-> side-chain
   pairs only. The paper requires the opposite in six places, e.g. clash covers
   "side-chain--backbone, side-chain--side-chain, and side-chain--context atom
   pairs", and the appendix says our own default S_phi's "inter-residue and context
   attention capture ... side-chain--receptor interactions".
"""
import torch

from pxdesign_train.sidechain.module import SideChainModule
from pxdesign_train.sidechain.physical import (
    build_sidechain_context,
    clash_loss,
    contact_loss,
    select_context_atoms,
)


# --------------------------------------------------- the model's assembly step
#
# This is the block model._train_forward runs. It is the one piece of the change
# that only executes on the real (GPU) training path, so it is tested here
# directly, on the shapes it actually sees.

def _complex(B=2):
    """3 binder tokens (atoms 0..11, N/CA/C/O each) + 2 receptor tokens (atoms 12..15)."""
    L, N = 5, 16
    bb = torch.full((B, L, 4), -1, dtype=torch.long)
    for t in range(3):                                   # binder tokens 0,1,2
        for k in range(4):
            bb[:, t, k] = 4 * t + k
    center = torch.tensor([[1, 5, 9, 12, 14]]).expand(B, -1).contiguous()  # CA / rep atom
    a2t = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4]]).expand(B, -1).contiguous()

    xyz = torch.zeros(B, N, 3)
    xyz[:, :12, 0] = torch.arange(12).float()            # binder strung along x
    xyz[:, 12:, 0] = torch.tensor([4.0, 4.5, 60.0, 60.5])  # 2 receptor atoms near, 2 far
    return xyz, center, a2t, bb


def test_build_sidechain_context_separates_binder_from_receptor():
    xyz, center, a2t, bb = _complex()
    ca_in = torch.zeros(2, 5, 3)                          # binder frame origins (dummy)

    ca, ctx_tok, (cxyz, cmask, cgrp) = build_sidechain_context(
        xyz=xyz, center_idx=center, atom_to_token=a2t, bb_atom_idx=bb,
        ca=ca_in, radius=10.0, max_atoms=16,
    )
    # Tokens 3,4 have no N/CA/C -> context. Tokens 0,1,2 are the binder.
    assert ctx_tok.tolist() == [[False, False, False, True, True]] * 2
    # Binder CA rows are untouched; context CA rows are filled from their rep atom.
    assert torch.equal(ca[:, :3], ca_in[:, :3])
    assert torch.allclose(ca[0, 3], torch.tensor([4.0, 0.0, 0.0]))    # near receptor
    assert torch.allclose(ca[0, 4], torch.tensor([60.0, 0.0, 0.0]))   # far receptor
    # The far receptor atoms are masked out by the radius; the near ones survive.
    kept = {int(g) for g, m in zip(cgrp[0], cmask[0]) if m}
    assert 3 in kept, "the receptor atoms 4-4.5 A from the binder must be context"
    assert 4 not in kept, "the receptor atoms 60 A away must be dropped by the radius"
    assert not cxyz.requires_grad


def test_scrubbed_binder_sidechain_rows_are_excluded_from_context():
    """The binder's own side-chain rows survive as tokens with coordinates pinned to
    the residue Cα (featurizer `_scrub_design_sidechain_coords`). Passing
    `design_sidechain_atom_mask` as `exclude_atom_mask` must drop them, so those
    phantom Cα-piled atoms never reach the mismatch clash/contact context set.
    The real side-chain geometry comes from S_phi."""
    # token 0 = binder (N/CA/C at atoms 0,1,2) with a scrubbed side-chain row at atom 3;
    # token 1 = receptor with atom 4. All 5 atoms sit within the radius of the binder Cα.
    bb = torch.tensor([[[0, 1, 2, -1], [-1, -1, -1, -1]]])
    center = torch.tensor([[1, 4]])                       # Cα of token 0, rep atom of token 1
    a2t = torch.tensor([[0, 0, 0, 0, 1]])                 # atom 3 belongs to binder token 0
    xyz = torch.zeros(1, 5, 3)
    xyz[0, :3, 0] = torch.tensor([0.0, 1.0, 2.0])         # binder N/CA/C along x
    xyz[0, 3] = xyz[0, 1]                                 # atom 3 scrubbed ONTO the Cα (atom 1)
    xyz[0, 4, 0] = 3.0                                    # receptor atom, within radius
    kw = dict(xyz=xyz, center_idx=center, atom_to_token=a2t, bb_atom_idx=bb,
              ca=torch.zeros(1, 2, 3), radius=10.0, max_atoms=16)

    _, _, (_, m_no, _) = build_sidechain_context(**kw)
    assert int(m_no.sum()) == 5, "precondition: all 5 atoms are in-radius candidates"

    excl = torch.tensor([[False, False, False, True, False]])   # design_sidechain_atom_mask
    _, _, (_, m_ex, _) = build_sidechain_context(**kw, exclude_atom_mask=excl)
    assert int(m_ex.sum()) == 4, "the scrubbed binder side-chain row must be dropped"


# ------------------------------------------------------------------ contact (bug 2)

def test_phantom_context_rows_do_not_zero_the_runaway_penalty():
    """A side-chain atom 40 A from any REAL context atom must still be penalised,
    even when the context tensor carries padded rows near it."""
    sc = torch.tensor([[[5.0, 0.0, 0.0]]])          # [B=1, A=1, 3]
    ok = torch.ones(1, 1, dtype=torch.bool)

    real = torch.tensor([[[45.0, 0.0, 0.0]]])       # the only real context atom
    alone = contact_loss(sc, real, ok, torch.ones(1, 1, dtype=torch.bool))
    assert alone.item() > 0, "runaway atom should be penalised"

    # Same structure, but a receptor token contributed 4 phantom rows at the origin.
    padded = torch.cat([real, torch.zeros(1, 4, 3)], dim=1)          # [1, 5, 3]
    mask = torch.tensor([[True, False, False, False, False]])

    unmasked = contact_loss(sc, padded, ok, None)
    assert unmasked.item() == 0.0, "precondition: phantoms DO zero the penalty"

    masked = contact_loss(sc, padded, ok, mask)
    assert torch.isclose(masked, alone), "masking must restore the real penalty"


def test_contact_with_no_valid_context_is_zero_not_nan():
    sc = torch.tensor([[[5.0, 0.0, 0.0]]])
    out = contact_loss(
        sc, torch.zeros(1, 3, 3),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.bool),      # nothing valid -> min() is +inf
    )
    assert torch.isfinite(out) and out.item() == 0.0


# -------------------------------------------------------------------- clash (bug 1)

def test_clash_scores_side_chain_against_context_atoms():
    """The paper's side-chain--backbone / side-chain--context pair classes."""
    sc = torch.tensor([[[0.0, 0.0, 0.0]]])                 # one side-chain atom
    ok = torch.ones(1, 1, dtype=torch.bool)
    grp = torch.zeros(1, 1, dtype=torch.long)              # residue 0

    far = torch.tensor([[[10.0, 0.0, 0.0]]])
    near = torch.tensor([[[0.5, 0.0, 0.0]]])               # overlapping a receptor atom
    cmask = torch.ones(1, 1, dtype=torch.bool)
    cgrp = torch.full((1, 1), 7, dtype=torch.long)         # a DIFFERENT residue

    base = clash_loss(sc, valid_mask=ok)                   # sc<->sc only: nothing to clash
    hit = clash_loss(sc, valid_mask=ok, group_id=grp,
                     context_coords=near, context_mask=cmask, context_group_id=cgrp)
    miss = clash_loss(sc, valid_mask=ok, group_id=grp,
                      context_coords=far, context_mask=cmask, context_group_id=cgrp)

    assert base.item() == 0.0
    assert hit.item() > 0.0, "an overlapping context atom must be a clash"
    assert miss.item() == 0.0


def test_clash_excludes_bonded_same_residue_pairs():
    """CB sits ~1.53 A from its own CA — below clash_dist. Scoring that as a clash
    would fight the bond loss, so same-residue pairs must be dropped."""
    cb = torch.tensor([[[1.53, 0.0, 0.0]]])                # this residue's CB
    ok = torch.ones(1, 1, dtype=torch.bool)
    own_ca = torch.tensor([[[0.0, 0.0, 0.0]]])
    cmask = torch.ones(1, 1, dtype=torch.bool)

    same = clash_loss(cb, valid_mask=ok, group_id=torch.zeros(1, 1, dtype=torch.long),
                      context_coords=own_ca, context_mask=cmask,
                      context_group_id=torch.zeros(1, 1, dtype=torch.long))
    other = clash_loss(cb, valid_mask=ok, group_id=torch.zeros(1, 1, dtype=torch.long),
                       context_coords=own_ca, context_mask=cmask,
                       context_group_id=torch.ones(1, 1, dtype=torch.long))

    assert same.item() == 0.0, "own backbone is BONDED, not clashing"
    assert other.item() > 0.0, "another residue's backbone at 1.53 A IS a clash"


# ------------------------------------------------------------- context selection

def test_select_context_atoms_is_radius_bounded_capped_and_detached():
    ref = torch.zeros(1, 2, 3)                      # two binder CAs at the origin
    ref_mask = torch.ones(1, 2, dtype=torch.bool)
    atoms = torch.stack([
        torch.tensor([1.0, 0.0, 0.0]),              # inside radius
        torch.tensor([5.0, 0.0, 0.0]),              # inside radius
        torch.tensor([99.0, 0.0, 0.0]),             # far away
    ])[None].requires_grad_(True)                   # [1, 3, 3]
    amask = torch.ones(1, 3, dtype=torch.bool)
    grp = torch.tensor([[0, 1, 2]])

    xyz, mask, group = select_context_atoms(
        ref, ref_mask, atoms, amask, grp, radius=10.0, max_atoms=2,
    )
    assert xyz.shape == (1, 2, 3) and mask.shape == (1, 2)   # capped at max_atoms
    assert mask.all(), "the two nearest atoms are both inside the radius"
    assert set(group[0].tolist()) == {0, 1}, "kept the NEAREST atoms"
    assert not xyz.requires_grad, "context is fixed conditioning — must be stop-grad"

    # The far atom survives the cap but must be masked OUT by the radius.
    xyz2, mask2, _ = select_context_atoms(
        ref, ref_mask, atoms, amask, grp, radius=10.0, max_atoms=3,
    )
    assert mask2.tolist() == [[True, True, False]]


# ---------------------------------------------------- S_phi cross-residue keys

def _sphi_inputs(B=1, L=3, A=4, C=8):
    torch.manual_seed(0)
    h_res = torch.randn(B, L, C)
    logits = torch.randn(B, L, 20)
    ids = torch.ones(B, L, A, dtype=torch.long)
    mask = torch.zeros(B, L, A, dtype=torch.bool)
    mask[:, 0] = True                      # ONLY token 0 is a binder residue
    noisy = torch.randn(B, L, A, 3)
    ca = torch.randn(B, L, 3)
    return h_res, logits, ids, mask, noisy, ca


def test_context_tokens_reach_the_side_chain_through_cross_res_attention():
    """Without ctx_mask a receptor token is an all-masked key and CANNOT influence
    the side chain. With it, the receptor's h_res changes the prediction."""
    mod = SideChainModule(c_res=8, c_atom=16, n_type=20).eval()
    h_res, logits, ids, mask, noisy, ca = _sphi_inputs()
    ctx = torch.tensor([[False, True, True]])      # tokens 1,2 are receptor

    with torch.no_grad():
        base, _ = mod(h_res, logits, ids, mask, noisy, torch.ones(1), ca_coords=ca)
        withctx, _ = mod(h_res, logits, ids, mask, noisy, torch.ones(1),
                         ca_coords=ca, ctx_mask=ctx)
        # Perturb ONLY the receptor tokens' representation.
        h2 = h_res.clone()
        h2[:, 1:] += 5.0
        base2, _ = mod(h2, logits, ids, mask, noisy, torch.ones(1), ca_coords=ca)
        withctx2, _ = mod(h2, logits, ids, mask, noisy, torch.ones(1),
                          ca_coords=ca, ctx_mask=ctx)

    sc = mask[..., None].expand_as(base)
    # Old behaviour: the receptor is invisible, so its features change nothing.
    assert torch.allclose(base[sc], base2[sc], atol=1e-6), (
        "precondition: without ctx_mask the receptor cannot reach the side chain"
    )
    # New behaviour: it does.
    assert not torch.allclose(withctx[sc], withctx2[sc], atol=1e-5), (
        "with ctx_mask the receptor MUST be able to influence the side chain"
    )


def test_ctx_mask_is_backward_compatible_when_absent():
    """Omitting ctx_mask must reproduce the pre-change module exactly."""
    mod = SideChainModule(c_res=8, c_atom=16, n_type=20).eval()
    h_res, logits, ids, mask, noisy, ca = _sphi_inputs()
    with torch.no_grad():
        a, _ = mod(h_res, logits, ids, mask, noisy, torch.ones(1), ca_coords=ca)
        b, _ = mod(h_res, logits, ids, mask, noisy, torch.ones(1),
                   ca_coords=ca, ctx_mask=None)
    assert torch.equal(a, b)


def test_gradient_reaches_sphi_through_the_context_path():
    mod = SideChainModule(c_res=8, c_atom=16, n_type=20)
    h_res, logits, ids, mask, noisy, ca = _sphi_inputs()
    h_res.requires_grad_(True)
    ctx = torch.tensor([[False, True, True]])
    y, _ = mod(h_res, logits, ids, mask, noisy, torch.ones(1),
               ca_coords=ca, ctx_mask=ctx)
    y.sum().backward()
    # The receptor tokens own no side-chain slot, so any gradient on their h_res row
    # can only have arrived through the cross-residue context attention.
    assert h_res.grad[:, 1:].abs().sum() > 0
