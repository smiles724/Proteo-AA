"""Tests for SideChainModule (S_phi) and h_res feedback."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))

from pxdesign_train.sidechain.instantiate import (
    MAX_SC,
    sidechain_mask,
    sidechain_atom_name_ids,
)
from pxdesign_train.sidechain.module import SideChainModule
from pxdesign_train.sidechain.feedback import HResFeedback
from pxdesign_train.sidechain.frames import to_global

C_RES = 16


def _toy_batch():
    restypes = ["ALA", "PHE", "LYS"]           # 1, 7, 5 side-chain atoms
    L = len(restypes)
    atom_mask = sidechain_mask(restypes)[None]        # [1, L, MAX_SC]
    atom_ids = sidechain_atom_name_ids(restypes)[None]  # [1, L, MAX_SC]
    h_res = torch.randn(1, L, C_RES, requires_grad=True)
    logits = torch.randn(1, L, 20)
    noisy = torch.randn(1, L, MAX_SC, 3)
    t = torch.tensor([0.5])
    return restypes, atom_mask, atom_ids, h_res, logits, noisy, t


def _module(scale=1.0):
    return SideChainModule(c_res=C_RES, c_atom=32, c_time=16, n_blocks=2,
                           n_heads=4, trunk_grad_scale=scale)


def test_forward_shape_and_padding():
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    y0, feats = _module().forward(h_res, logits, atom_ids, atom_mask, noisy, t)
    assert y0.shape == (1, 3, MAX_SC, 3)
    assert feats.shape == (1, 3, MAX_SC, 32)
    assert y0.requires_grad
    # padded atoms (beyond each residue's side-chain count) are exactly zero
    assert torch.count_nonzero(y0[0, 0, 1:]) == 0     # ALA: only CB valid
    assert torch.count_nonzero(y0[0, 2, 5:]) == 0     # LYS: 5 valid


def test_cross_residue_attention_runs():
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    ca = torch.randn(1, 3, 3)  # [B, L, 3] residue CA coords
    y0, feats = _module().forward(h_res, logits, atom_ids, atom_mask, noisy, t, ca_coords=ca)
    assert y0.shape == (1, 3, MAX_SC, 3)
    assert torch.isfinite(y0).all()
    y0.sum().backward()
    assert h_res.grad is not None and torch.count_nonzero(h_res.grad) > 0


def test_centred_input_makes_the_module_translation_equivariant():
    """The single-frame contract's whole justification, stated as a property.

    With `centre_coord_input` on, nothing inside the module reads an ABSOLUTE
    position: the embedding sees coords - CA, and every geometric term is a
    distance. So translating the entire input (atoms, CAs, frame origins) by a
    constant must translate the output by exactly that constant -- no drift, no
    change in the attention pattern. If some tensor were still residue-local while
    another was global, this would not hold.
    """
    torch.manual_seed(0)
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    h_res = h_res.detach()
    ca = torch.randn(1, 3, 3)
    frame_R = torch.linalg.qr(torch.randn(1, 3, 3, 3))[0]
    mod = SideChainModule(c_res=C_RES, c_atom=32, c_time=16, n_blocks=2,
                          n_heads=4, centre_coord_input=True).eval()
    shift = torch.tensor([137.0, -42.0, 8.5])

    with torch.no_grad():
        a, _ = mod.forward(h_res, logits, atom_ids, atom_mask, noisy, t,
                           ca_coords=ca, frame_R=frame_R, frame_t=ca)
        b, _ = mod.forward(h_res, logits, atom_ids, atom_mask, noisy + shift, t,
                           ca_coords=ca + shift, frame_R=frame_R, frame_t=ca + shift)
    expected = (a + shift) * atom_mask[..., None].to(a.dtype)
    assert torch.allclose(b, expected, atol=1e-3), (
        "centred input is not translation-equivariant — some tensor is still being "
        "read as an absolute position"
    )


def test_uncentred_input_is_not_translation_equivariant():
    """The control for the test above, and the reason `centre_coord_input` exists:
    fed raw global coordinates, `w_xyz` encodes where the residue is in the
    assembly, so moving the assembly changes the prediction."""
    torch.manual_seed(0)
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    h_res = h_res.detach()
    ca = torch.randn(1, 3, 3)
    frame_R = torch.linalg.qr(torch.randn(1, 3, 3, 3))[0]
    mod = SideChainModule(c_res=C_RES, c_atom=32, c_time=16, n_blocks=2,
                          n_heads=4, centre_coord_input=False).eval()
    shift = torch.tensor([137.0, -42.0, 8.5])

    with torch.no_grad():
        a, _ = mod.forward(h_res, logits, atom_ids, atom_mask, noisy, t,
                           ca_coords=ca, frame_R=frame_R, frame_t=ca)
        b, _ = mod.forward(h_res, logits, atom_ids, atom_mask, noisy + shift, t,
                           ca_coords=ca + shift, frame_R=frame_R, frame_t=ca + shift)
    assert not torch.allclose(b, (a + shift) * atom_mask[..., None].to(a.dtype), atol=1e-3)


def test_cross_residue_bias_reads_the_same_tensor_the_atoms_do():
    """One coordinate tensor, one frame: moving the residues apart has to change
    the cross-residue attention, because the distance bias is computed off exactly
    the coordinates that were handed in."""
    torch.manual_seed(0)
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    h_res = h_res.detach()
    ca = torch.zeros(1, 3, 3)
    mod = SideChainModule(c_res=C_RES, c_atom=32, c_time=16, n_blocks=2,
                          n_heads=4, centre_coord_input=True).eval()

    spread = torch.arange(3, dtype=torch.float32).view(1, 3, 1) * 60.0
    ca_far = ca + spread
    with torch.no_grad():
        y_near, _ = mod.forward(h_res, logits, atom_ids, atom_mask, noisy, t,
                                ca_coords=ca)
        y_far, _ = mod.forward(h_res, logits, atom_ids, atom_mask,
                               noisy + spread[..., None, :], t, ca_coords=ca_far)
    # Subtract the trivial translation each residue underwent; what is left is the
    # change in the attention pattern.
    delta = (y_far - spread[..., None, :]) - y_near
    assert delta.abs().max() > 1e-5, (
        "pulling the residues 60 A apart changed nothing — the distance bias is "
        "not reading the coordinates"
    )


def test_multiple_cross_residue_blocks_run():
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    ca = torch.randn(1, 3, 3)
    mod = SideChainModule(
        c_res=C_RES, c_atom=32, c_time=16, n_blocks=2, n_heads=4, n_cross_blocks=3
    )
    y0, feats = mod.forward(h_res, logits, atom_ids, atom_mask, noisy, t, ca_coords=ca)
    assert len(mod.cross_res_blocks) == 3
    assert y0.shape == (1, 3, MAX_SC, 3)
    assert feats.shape == (1, 3, MAX_SC, 32)
    assert torch.isfinite(y0).all()


def test_gly_no_nan():
    restypes = ["GLY"]  # zero side-chain atoms -> fully padded residue
    atom_mask = sidechain_mask(restypes)[None]
    atom_ids = sidechain_atom_name_ids(restypes)[None]
    h = torch.randn(1, 1, C_RES)
    y0, _ = _module().forward(h, torch.randn(1, 1, 20), atom_ids, atom_mask,
                              torch.randn(1, 1, MAX_SC, 3), torch.tensor([0.3]))
    assert torch.isfinite(y0).all()
    assert torch.count_nonzero(y0) == 0


def test_grad_reaches_hres_when_coupled():
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    y0, _ = _module(scale=1.0).forward(h_res, logits, atom_ids, atom_mask, noisy, t)
    y0.sum().backward()
    assert h_res.grad is not None and torch.count_nonzero(h_res.grad) > 0


def test_grad_cut_when_readonly():
    _, atom_mask, atom_ids, h_res, logits, noisy, t = _toy_batch()
    y0, _ = _module(scale=0.0).forward(h_res, logits, atom_ids, atom_mask, noisy, t)
    y0.sum().backward()
    assert h_res.grad is None or torch.count_nonzero(h_res.grad) == 0


def test_template_residual_starts_at_noisy_template_in_active_frame():
    _, atom_mask, atom_ids, h_res, logits, noisy_local, t = _toy_batch()
    mod = SideChainModule(
        c_res=C_RES,
        c_atom=32,
        c_time=16,
        n_blocks=2,
        n_heads=4,
        template_residual=True,
    )
    frame_R = torch.linalg.qr(torch.randn(1, 3, 3, 3))[0]
    frame_t = torch.randn(1, 3, 3)
    y0, _ = mod(
        h_res,
        logits,
        atom_ids,
        atom_mask,
        noisy_local,
        t,
        frame_R=frame_R,
        frame_t=frame_t,
    )
    # Input is GLOBAL under the single-frame contract; the module maps it into the
    # frame to form the residual base and the zero-init head maps it straight back,
    # so a template-residual module starts as the identity on its own input.
    expected = noisy_local * atom_mask[..., None].to(noisy_local.dtype)
    assert torch.allclose(y0, expected, atol=1e-4)
    assert torch.count_nonzero(mod.out.weight) == 0
    assert torch.count_nonzero(mod.out.bias) == 0


def test_template_residual_requires_frame_aware_output():
    _, atom_mask, atom_ids, h_res, logits, noisy_local, t = _toy_batch()
    mod = SideChainModule(
        c_res=C_RES,
        c_atom=32,
        c_time=16,
        n_blocks=2,
        n_heads=4,
        template_residual=True,
    )
    try:
        mod(h_res, logits, atom_ids, atom_mask, noisy_local, t)
    except ValueError as exc:
        assert "frame-aware" in str(exc)
    else:
        raise AssertionError("template residual accepted a missing residue frame")


# --- feedback ---
def test_feedback_shape_and_grad_flow():
    fb = HResFeedback(c_atom=32, c_res=C_RES)
    atom_feats = torch.randn(1, 3, MAX_SC, 32, requires_grad=True)
    atom_mask = sidechain_mask(["ALA", "PHE", "LYS"])[None]
    h_res = torch.randn(1, 3, C_RES)
    hp = fb(atom_feats, atom_mask, h_res, detach=False)
    assert hp.shape == (1, 3, C_RES)
    hp.sum().backward()
    assert atom_feats.grad is not None and torch.count_nonzero(atom_feats.grad) > 0


def test_feedback_detach_cuts_sidechain_grad():
    fb = HResFeedback(c_atom=32, c_res=C_RES)
    atom_feats = torch.randn(1, 3, MAX_SC, 32, requires_grad=True)
    atom_mask = sidechain_mask(["ALA", "PHE", "LYS"])[None]
    h_res = torch.randn(1, 3, C_RES)
    hp = fb(atom_feats, atom_mask, h_res, detach=True)
    hp.sum().backward()
    assert atom_feats.grad is None or torch.count_nonzero(atom_feats.grad) == 0
