"""The ported IPA blocks, checked against the property the port could break.

`sidechain/ipa.py` is APM's `ipa_pytorch.py` with one change: frames arrive as
`(R, t)` tensors instead of an openfold `Rigid`, so `r[..., None].apply(...)` and
`r[..., None, None].invert_apply(...)` are written out by hand. Those two lines
are the entire risk surface of the port, and getting either wrong produces a
model that trains -- just not an invariant one. So the test is the invariance.
"""
import math

import torch

from pxdesign_train.sidechain.ipa import (
    AngleResnet,
    EdgeTransition,
    InvariantPointAttention,
    StructureModuleTransition,
)
from pxdesign_train.sidechain.packer import rotmat_to_rotvec


def _frames(B, L, seed=0):
    torch.manual_seed(seed)
    R = torch.linalg.qr(torch.randn(B, L, 3, 3))[0]
    R = R * torch.sign(torch.linalg.det(R))[..., None, None]
    return R, torch.randn(B, L, 3) * 2.0


def _global_motion():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q, torch.randn(3) * 30.0


def test_ipa_output_is_invariant_to_a_global_rigid_motion():
    B, L, c_s, c_z = 1, 9, 32, 16
    ipa = InvariantPointAttention(c_s=c_s, c_z=c_z, c_hidden=8, no_heads=4,
                                  no_qk_points=4, no_v_points=6).eval()
    s = torch.randn(B, L, c_s)
    z = torch.randn(B, L, L, c_z)
    mask = torch.ones(B, L)
    R, t = _frames(B, L)
    q, shift = _global_motion()
    with torch.no_grad():
        out = ipa(s, z, R, t, mask)
        moved = ipa(s, z, torch.einsum("ij,bljk->blik", q, R),
                    torch.einsum("ij,blj->bli", q, t) + shift, mask)
    torch.testing.assert_close(out, moved, atol=1e-5, rtol=0)


def test_masked_keys_do_not_contribute():
    """The mask must remove keys, not merely down-weight them."""
    B, L, c_s, c_z = 1, 6, 16, 8
    ipa = InvariantPointAttention(c_s=c_s, c_z=c_z, c_hidden=4, no_heads=2,
                                  no_qk_points=2, no_v_points=3).eval()
    s = torch.randn(B, L, c_s)
    z = torch.randn(B, L, L, c_z)
    R, t = _frames(B, L)
    mask = torch.ones(B, L)
    mask[0, -2:] = 0.0
    with torch.no_grad():
        out = ipa(s, z, R, t, mask)
        # Change everything about the masked rows; the kept rows must not move.
        s2 = s.clone()
        s2[0, -2:] = torch.randn(2, c_s) * 5
        z2 = z.clone()
        z2[0, -2:] = torch.randn(2, L, c_z) * 5
        z2[0, :, -2:] = torch.randn(L, 2, c_z) * 5
        out2 = ipa(s2, z2, R, t, mask)
    torch.testing.assert_close(out[0, :-2], out2[0, :-2], atol=1e-5, rtol=0)


def test_angle_resnet_returns_unit_vectors_and_their_norms():
    ar = AngleResnet(c_in=16, c_hidden=16, no_blocks=2, no_angles=4, epsilon=1e-4)
    s = torch.randn(1, 5, 16)
    unnormalized, unit = ar(s, s)
    assert unnormalized.shape == (1, 5, 4, 2)
    torch.testing.assert_close(unit.norm(dim=-1), torch.ones(1, 5, 4), atol=1e-5, rtol=0)
    # The unnormalised output is what the angle-norm term regularises, so it must
    # NOT already be unit length.
    assert (unnormalized.norm(dim=-1) - 1.0).abs().max() > 1e-3


def test_structure_module_transition_and_edge_transition_shapes():
    s = torch.randn(2, 5, 16)
    z = torch.randn(2, 5, 5, 8)
    assert StructureModuleTransition(16)(s).shape == s.shape
    assert EdgeTransition(node_embed_size=16, edge_embed_in=8,
                          edge_embed_out=8)(s, z).shape == z.shape


def test_rotvec_round_trips_through_the_exponential_map():
    """APM feeds the frame's rotation vector as a node feature; check the log map.

    Includes the two cases the implementation special-cases -- theta ~ 0 and
    theta ~ pi -- because the generic formula divides by sin(theta) at both.
    """
    axes = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    angles = torch.cat([
        torch.rand(60) * (math.pi - 0.2) + 0.1,      # generic
        torch.tensor([1e-7, 1e-6]),                   # theta ~ 0
        torch.tensor([math.pi - 1e-4, math.pi]),      # theta ~ pi
    ])
    v = axes * angles[:, None]
    # Rodrigues: exp of the skew matrix.
    K = torch.zeros(64, 3, 3)
    K[:, 0, 1], K[:, 0, 2] = -axes[:, 2], axes[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axes[:, 2], -axes[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axes[:, 1], axes[:, 0]
    a = angles[:, None, None]
    R = torch.eye(3) + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)

    recovered = rotmat_to_rotvec(R)
    # At exactly pi the axis sign is unrecoverable (R(v) == R(-v)), so compare
    # the rotations, not the vectors.
    same = torch.minimum((recovered - v).norm(dim=-1), (recovered + v).norm(dim=-1))
    assert float(same.max()) < 1e-3


def test_module_can_be_moved_between_devices():
    """`.to()` calls `nn.Module._apply` -- a helper named `_apply` shadows it.

    Cheap to check on CPU, and the only CPU-side symptom of a bug that otherwise
    surfaces as a TypeError one minute into a GPU allocation (job 116989).
    """
    ipa = InvariantPointAttention(c_s=8, c_z=4, c_hidden=2, no_heads=2,
                                  no_qk_points=2, no_v_points=2)
    ipa.to("cpu")
    ipa.to(torch.float32)
    from pxdesign_train.sidechain.packer import TorsionPacker
    packer = TorsionPacker(c_res=8, c_node=8, c_pair=4, n_blocks=1, ipa_c_hidden=2,
                           ipa_no_heads=2, no_qk_points=2, no_v_points=2,
                           seq_tfmr_num_heads=2, seq_tfmr_num_layers=1,
                           c_pos_emb=4, c_timestep_emb=4, edge_feat_dim=4,
                           edge_num_bins=4, seq_cond="none")
    packer.to("cpu")


def test_buildsc_gradients_are_finite_under_anomaly_detection():
    """The NaN that `torch.where` cannot mask.

    A residue without chi_k indexes its four dihedral atoms at slot 0, so they
    coincide and atan2 gets (0, 0): value 0, gradient NaN. Multiplying by the
    active mask afterwards does NOT clear it, because NaN * 0 is NaN. Forward
    checks pass, small-example gradient checks pass (the NaN is masked before it
    reaches a parameter), and then the first real batch dies with "Nonfinite
    gradient before optimizer update 0" -- which is how this was found.

    Anomaly detection is the point of the test: it fails on the NaN itself
    rather than on whether this particular loss happens to propagate it.
    """
    import pxdesign_train.sidechain.buildsc as buildsc
    from pxdesign_train.sidechain.chi_constants import CHI_MASK
    from pxdesign_train.sidechain.instantiate import STD_AA_3

    # GLY and ALA own no torsion; ARG owns four. The mixture is what matters.
    types = torch.tensor([STD_AA_3.index(n) for n in ("ARG", "GLY", "ALA", "SER")])
    chi = torch.zeros(4, 4, requires_grad=True)
    target = torch.where(CHI_MASK[types], chi, torch.full_like(chi, float("nan")))
    torch.autograd.set_detect_anomaly(True)
    try:
        built, _ = buildsc.build_sidechain_local(types, target)
        built.square().sum().backward()
    finally:
        torch.autograd.set_detect_anomaly(False)
    assert torch.isfinite(chi.grad).all()
