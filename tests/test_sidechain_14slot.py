"""S_phi's INTERNAL 14-slot atom axis (4 backbone context + 10 side-chain).

The external side-chain contract is unchanged (10 slots everywhere: features,
templates, loss). The 14 slots live only inside SideChainModule: it takes the 4
backbone atoms as CONTEXT, attends over all 14, decodes coordinates for the 10
side-chain slots only, and additionally returns the updated backbone-slot
features so the Backbone Module can later be given an atom-level (q-level)
side-chain signal.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))

from pxdesign_train.heads import sinusoidal_time_embedding
from pxdesign_train.sidechain.frames import to_global
from pxdesign_train.sidechain.instantiate import (
    ATOM_NAME_TO_ID,
    ATOM_VOCAB_SIZE,
    BACKBONE_ATOM_NAME_IDS,
    BACKBONE_ATOMS,
    MAX_SC,
    N_ATOM14,
    N_BB,
    SC_VOCAB_SIZE,
    sidechain_atom_name_ids,
    sidechain_mask,
)
from pxdesign_train.sidechain.module import SideChainModule

C_RES = 16
C_ATOM = 32


# --- the frozen side-chain atom-name ids, snapshotted BEFORE the vocab extension.
# A trained atom-name embedding indexes these rows; if any id moves, every
# checkpoint silently re-labels its atoms.
OLD_SC_IDS = {
    "CB": 1, "CD": 2, "CD1": 3, "CD2": 4, "CE": 5, "CE1": 6, "CE2": 7, "CE3": 8,
    "CG": 9, "CG1": 10, "CG2": 11, "CH2": 12, "CZ": 13, "CZ2": 14, "CZ3": 15,
    "ND1": 16, "ND2": 17, "NE": 18, "NE1": 19, "NE2": 20, "NH1": 21, "NH2": 22,
    "NZ": 23, "OD1": 24, "OD2": 25, "OE1": 26, "OE2": 27, "OG": 28, "OG1": 29,
    "OH": 30, "SD": 31, "SG": 32,
}


def _module(seed=0, scale=1.0):
    torch.manual_seed(seed)
    return SideChainModule(c_res=C_RES, c_atom=C_ATOM, c_time=16, n_blocks=2,
                           n_heads=4, trunk_grad_scale=scale)


def _batch(restypes, seed=1, requires_grad=False):
    torch.manual_seed(seed)
    L = len(restypes)
    mask = sidechain_mask(restypes)[None]                 # [1, L, 10]
    ids = sidechain_atom_name_ids(restypes)[None]         # [1, L, 10]
    h_res = torch.randn(1, L, C_RES)
    logits = torch.randn(1, L, 20)
    noisy = torch.randn(1, L, MAX_SC, 3, requires_grad=requires_grad)
    bb = torch.randn(1, L, N_BB, 3)                       # known backbone coords (local frame)
    ca = torch.randn(1, L, 3)
    t = torch.tensor([0.5])
    return dict(mask=mask, ids=ids, h_res=h_res, logits=logits, noisy=noisy,
                bb=bb, ca=ca, t=t)


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

def test_existing_sidechain_ids_are_unchanged():
    """APPEND-only vocab extension: no side-chain id may be renumbered."""
    for name, old in OLD_SC_IDS.items():
        assert ATOM_NAME_TO_ID[name] == old, f"{name} moved {old} -> {ATOM_NAME_TO_ID[name]}"
    # and nothing else squeezed into the frozen 1..32 block
    frozen = {n: i for n, i in ATOM_NAME_TO_ID.items() if i < SC_VOCAB_SIZE}
    assert frozen == OLD_SC_IDS
    assert SC_VOCAB_SIZE == 33          # padding + 32 side-chain names


def test_backbone_names_appended_after_the_frozen_block():
    assert [ATOM_NAME_TO_ID[n] for n in BACKBONE_ATOMS] == [33, 34, 35, 36]
    assert BACKBONE_ATOM_NAME_IDS.tolist() == [33, 34, 35, 36]
    assert ATOM_VOCAB_SIZE == 37        # 0 pad + 32 side-chain + 4 backbone
    # ids are unique and dense
    assert len(set(ATOM_NAME_TO_ID.values())) == len(ATOM_NAME_TO_ID)
    assert sorted(ATOM_NAME_TO_ID.values()) == list(range(1, ATOM_VOCAB_SIZE))


def test_layout_constants():
    assert BACKBONE_ATOMS == ("N", "CA", "C", "O")
    assert N_BB == 4 and MAX_SC == 10 and N_ATOM14 == 14


# --------------------------------------------------------------------------
# bb_local=None must be bit-identical to the pre-14-slot module
# --------------------------------------------------------------------------

def _legacy_forward(m, h_res, logits, ids, mask, noisy, t, ca=None, R=None, tt=None):
    """The 10-slot forward exactly as it was before the 14-slot axis existed."""
    B, L, A = ids.shape
    h = m._scale_grad(h_res)
    te = sinusoidal_time_embedding(torch.as_tensor(t, device=h.device).float(), m.c_time)
    te = m.w_t(te)
    if te.dim() == 1:
        te = te[None]
    res_feat = m.w_res(h)[:, :, None, :]
    type_feat = m.w_aa(torch.softmax(logits, dim=-1))[:, :, None, :]
    u = m.atom_embed(ids) + res_feat + type_feat + m.w_xyz(noisy) + te[:, None, None, :]
    x = u.reshape(B * L, A, m.c_atom)
    kpm = ~mask.reshape(B * L, A)
    fully_pad = kpm.all(dim=1)
    kpm = kpm & ~fully_pad[:, None]
    for blk in m.blocks:
        x = blk(x, key_padding_mask=kpm)
    af = x.reshape(B, L, A, m.c_atom)
    if ca is not None:
        am = mask.to(af.dtype)[..., None]
        rf = (af * am).sum(2) / (am.sum(2) + 1e-6)
        af = af + m.cross_res(rf, ca, mask.any(dim=-1))[:, :, None, :]
    y0 = m.out(m.out_ln(af))
    if R is not None and tt is not None:
        x0 = to_global(y0, R, tt)
    else:
        x0 = y0 + (ca[:, :, None, :] if ca is not None else 0.0)
    return x0 * mask[..., None].to(x0.dtype), af


def test_bb_local_none_is_bit_identical_to_legacy():
    m = _module().eval()
    b = _batch(["ALA", "PHE", "GLY", "LYS"])
    with torch.no_grad():
        out = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"], b["t"],
                ca_coords=b["ca"])
        ref = _legacy_forward(m, b["h_res"], b["logits"], b["ids"], b["mask"],
                              b["noisy"], b["t"], ca=b["ca"])
    assert len(out) == 2, "legacy call sites unpack exactly two values"
    y0, feats = out
    assert torch.equal(y0, ref[0])
    assert torch.equal(feats, ref[1])
    assert y0.shape == (1, 4, MAX_SC, 3)
    assert feats.shape == (1, 4, MAX_SC, C_ATOM)


def test_bb_local_none_bit_identical_with_frame_aware_head():
    m = _module(seed=3).eval()
    b = _batch(["TRP", "SER"], seed=7)
    torch.manual_seed(11)
    R = torch.linalg.qr(torch.randn(1, 2, 3, 3))[0]
    tt = torch.randn(1, 2, 3)
    with torch.no_grad():
        y0, feats = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"], b["t"],
                      ca_coords=b["ca"], frame_R=R, frame_t=tt)
        ry0, rf = _legacy_forward(m, b["h_res"], b["logits"], b["ids"], b["mask"],
                                  b["noisy"], b["t"], ca=b["ca"], R=R, tt=tt)
    assert torch.equal(y0, ry0) and torch.equal(feats, rf)


# --------------------------------------------------------------------------
# the 14-slot path
# --------------------------------------------------------------------------

def test_shapes_with_backbone_context():
    m = _module()
    b = _batch(["ALA", "PHE", "LYS"])
    y0, feats, bb_feats = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"],
                            b["t"], ca_coords=b["ca"], bb_local=b["bb"])
    # coordinate output stays 10-slot: backbone slots decode NO coordinates
    assert y0.shape == (1, 3, MAX_SC, 3)
    assert feats.shape == (1, 3, MAX_SC, C_ATOM)
    assert bb_feats.shape == (1, 3, N_BB, C_ATOM)
    assert torch.isfinite(y0).all() and torch.isfinite(bb_feats).all()
    # padded side-chain slots still zeroed (ALA has only CB)
    assert torch.count_nonzero(y0[0, 0, 1:]) == 0


def test_backbone_context_changes_the_sidechain_prediction():
    """The 4 backbone atoms must actually condition the side chain."""
    m = _module().eval()
    b = _batch(["PHE", "LYS"])
    with torch.no_grad():
        y_no, _ = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"], b["t"],
                    ca_coords=b["ca"])
        y_bb, _, _ = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"], b["t"],
                       ca_coords=b["ca"], bb_local=b["bb"])
    assert not torch.allclose(y_no, y_bb)


def test_sidechain_gradient_reaches_backbone_slot_features():
    """THE q-CHANNEL PRECONDITION. If d(bb_feats)/d(side-chain input) == 0 the
    whole atom-level feedback channel is a no-op."""
    m = _module()
    b = _batch(["PHE", "LYS", "TRP"], requires_grad=True)
    _, _, bb_feats = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"], b["t"],
                       ca_coords=b["ca"], bb_local=b["bb"])
    bb_feats.sum().backward()
    g = b["noisy"].grad
    assert g is not None
    # only VALID side-chain atoms may push on the backbone slots
    gv = g[b["mask"]]
    assert torch.count_nonzero(gv) > 0, "side chain does not move the backbone slots"
    assert float(gv.abs().max()) > 1e-8
    # padded slots are masked out of attention -> no gradient
    assert torch.count_nonzero(g[~b["mask"]]) == 0


def test_backbone_slot_features_differ_per_atom():
    """N, CA, C, O must not collapse to one shared feature (they carry distinct
    atom-name embeddings and distinct coordinates)."""
    m = _module().eval()
    b = _batch(["PHE"])
    with torch.no_grad():
        _, _, bb_feats = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"],
                           b["t"], ca_coords=b["ca"], bb_local=b["bb"])
    v = bb_feats[0, 0]                       # [4, c_atom]
    for i in range(N_BB):
        for j in range(i + 1, N_BB):
            assert not torch.allclose(v[i], v[j])


def test_gly_zero_sidechain_atoms_does_not_nan():
    """GLY: 4 valid backbone slots, 0 valid side-chain slots. The all-masked
    side-chain row must not blow up attention."""
    m = _module()
    b = _batch(["GLY", "GLY", "ALA"], requires_grad=True)
    assert int(b["mask"][0, 0].sum()) == 0            # GLY really has no side chain
    y0, feats, bb_feats = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"],
                            b["t"], ca_coords=b["ca"], bb_local=b["bb"])
    assert torch.isfinite(y0).all()
    assert torch.isfinite(feats).all()
    assert torch.isfinite(bb_feats).all()
    assert torch.count_nonzero(y0[0, 0]) == 0         # GLY emits no side-chain coords
    (y0.sum() + bb_feats.sum()).backward()
    assert torch.isfinite(b["noisy"].grad).all()
    # GLY still gets real backbone features (it is a residue, it just has no side chain)
    assert torch.count_nonzero(bb_feats[0, 0]) > 0


def test_gly_only_batch_does_not_nan():
    m = _module()
    b = _batch(["GLY", "GLY"])
    y0, feats, bb_feats = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"],
                            b["t"], ca_coords=b["ca"], bb_local=b["bb"])
    assert torch.isfinite(y0).all() and torch.isfinite(bb_feats).all()


def test_res_mask_false_row_does_not_nan():
    """A padded / non-protein token (res_mask False, no side chain) is fully
    masked on all 14 slots -- the fully_pad guard must still hold."""
    m = _module()
    b = _batch(["PHE", "GLY", "ALA"])
    res_mask = torch.tensor([[True, True, False]])
    y0, feats, bb_feats = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"],
                            b["t"], ca_coords=b["ca"], bb_local=b["bb"], res_mask=res_mask)
    assert torch.isfinite(y0).all() and torch.isfinite(bb_feats).all()


def test_frame_aware_head_still_only_decodes_sidechain_slots():
    m = _module()
    b = _batch(["TRP", "GLY"])
    torch.manual_seed(5)
    R = torch.linalg.qr(torch.randn(1, 2, 3, 3))[0]
    tt = torch.randn(1, 2, 3)
    y0, feats, bb_feats = m(b["h_res"], b["logits"], b["ids"], b["mask"], b["noisy"],
                            b["t"], ca_coords=b["ca"], frame_R=R, frame_t=tt,
                            bb_local=b["bb"])
    assert y0.shape == (1, 2, MAX_SC, 3)
    assert bb_feats.shape == (1, 2, N_BB, C_ATOM)
    assert torch.isfinite(y0).all()
