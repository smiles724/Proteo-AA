"""Acceptance criteria 0-2 for the side-chain -> backbone h_res' feedback.

These are written before any training run, because each one is a way the
feedback could look like it works while measuring something else:

  0  zero-init no-op    if an untrained module perturbs the second pass at all,
                        every later difference is contaminated by the
                        perturbation rather than caused by the feedback.
  1  rigid invariance   the module reads coordinates, and a feature that moves
                        when the whole protein is rotated is encoding the frame
                        of reference, not the structure.
  2  read-only inputs   the feedback must not change the packed side chains. If
                        it did, an improvement could come from silently
                        repacking rather than from informing the backbone.
"""
from __future__ import annotations

import math

import pytest
import torch

from pxdesign_train.sidechain.env_feedback import SidechainEnvFeedback

B, L, A, C_ATOM, C_TRUNK = 2, 12, 10, 256, 384


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    sc_feats = torch.randn(B, L, A, C_ATOM, generator=g)
    sc_coords = torch.randn(B, L, A, 3, generator=g) * 4.0
    ca = torch.randn(B, L, 3, generator=g) * 8.0
    chem = torch.rand(B, L, A, generator=g) > 0.3
    res_mask = torch.ones(B, L, dtype=torch.bool)
    res_mask[1, -3:] = False                       # some padding, as in a real crop
    chem = chem & res_mask[..., None]
    ids = torch.randint(1, 20, (B, L, A), generator=g)
    return dict(sc_feats=sc_feats, sc_coords=sc_coords, chem_mask=chem, ca=ca,
                res_mask=res_mask, atom_name_ids=ids)


def _rotate(x, R, t):
    return x @ R.T + t


@pytest.mark.parametrize("use_env", [True, False])
@pytest.mark.parametrize("feature_source", ["packer", "atom_id"])
def test_zero_init_returns_exact_zeros(use_env, feature_source):
    """Criterion 0. Exactly zero, not merely small: the injector's output layer
    is zero-initialised, so the second pass must be bit-identical to the first.
    """
    m = SidechainEnvFeedback(c_atom=C_ATOM, c_trunk=C_TRUNK, use_env=use_env,
                             n_blocks=2 if use_env else 0,
                             feature_source=feature_source).eval()
    out = m(**_batch())
    assert out.shape == (B, L, C_TRUNK)
    assert torch.count_nonzero(out) == 0, f"max|out| = {out.abs().max().item():.3e}"


def _perturbed(feature_source, dtype, R, t, seed=2):
    """Same module and inputs, once as given and once rigidly moved."""
    torch.manual_seed(1)
    m = SidechainEnvFeedback(c_atom=C_ATOM, c_trunk=C_TRUNK, n_blocks=2,
                             feature_source=feature_source).to(dtype).eval()
    # Break the zero-init, or the test passes on zeros and proves nothing.
    with torch.no_grad():
        m.injector.proj[-1].weight.normal_(0, 0.02)
        m.injector.proj[-1].bias.normal_(0, 0.02)
    b = _batch(seed)
    b = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in b.items()}
    ref = m(**b)
    moved = dict(b)
    moved["sc_coords"] = _rotate(b["sc_coords"], R.to(dtype), t.to(dtype))
    moved["ca"] = _rotate(b["ca"], R.to(dtype), t.to(dtype))
    return (ref - m(**moved)).abs().max().item(), ref.abs().max().item()


def _rand_rotation(seed=7):
    q = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed)))[0]
    return q * torch.sign(torch.linalg.det(q))


@pytest.mark.parametrize("feature_source", ["packer", "atom_id"])
def test_rigid_motion_invariance_exact_in_float64(feature_source):
    """Criterion 1, the real claim. Coordinates enter only through `cdist`
    (atom-atom distances) and the CA KNN, both of which are rigid invariants,
    so the output is invariant BY CONSTRUCTION. In float64 that shows up as
    ~1e-8; the residue is the rotation matmul and cdist, not the module.

    Asserted in float64 rather than float32 deliberately: `cdist` expands
    ||a||^2 + ||b||^2 - 2a.b and loses precision when coordinate magnitudes
    dwarf the differences, so a float32 threshold would be measuring torch's
    arithmetic and would have to be loosened until it stopped testing anything.
    """
    d, _ = _perturbed(feature_source, torch.float64,
                      _rand_rotation(), torch.tensor([13.0, -5.0, 2.5]))
    assert d < 1e-6, f"float64 max|diff| under rigid motion = {d:.3e}"


def test_float32_invariance_error_does_not_grow_with_position():
    """What the centring in `forward` buys, locked in.

    Without centring the float32 error tracked the absolute coordinate
    magnitude -- measured 2.0e-04 at a 13 A offset and 2.1e-03 at 1000 A,
    because `cdist`'s expansion cancels badly when |a| >> |a-b|. Centring on the
    masked CA centroid leaves the distances untouched and makes the error
    independent of where the protein sits, which is the property a feature fed
    to the backbone needs: two identical structures at different places in the
    box must not get measurably different feedback.
    """
    R = _rand_rotation()
    near, scale = _perturbed("packer", torch.float32, R, torch.tensor([13.0, -5.0, 2.5]))
    far, _ = _perturbed("packer", torch.float32, R, torch.tensor([1000.0, 0.0, 0.0]))
    assert near < 1e-3, f"float32 error at 13 A = {near:.3e} (output scale {scale:.3f})"
    # Position-independence, with slack for the different rotation-induced terms.
    assert far < 3 * max(near, 1e-6), (
        f"float32 error grew with position: {near:.3e} at 13 A vs {far:.3e} at 1000 A")


def test_inputs_are_not_modified():
    """Criterion 2. The feedback informs the backbone; it must not repack."""
    torch.manual_seed(3)
    m = SidechainEnvFeedback(c_atom=C_ATOM, c_trunk=C_TRUNK, n_blocks=2)
    b = _batch(4)
    before = {k: v.clone() for k, v in b.items()}
    m(**b)
    for k, v in b.items():
        assert torch.equal(v, before[k]), f"{k} was modified in place"


def test_detach_controls_the_gradient_path():
    """The detach/flow arms must actually differ: APM detaches the side chains
    it consumes, Stage II-B did not, and that is the comparison being set up.
    """
    for detach, expect_grad in ((True, False), (False, True)):
        torch.manual_seed(5)
        m = SidechainEnvFeedback(c_atom=C_ATOM, c_trunk=C_TRUNK, n_blocks=2,
                                 detach_inputs=detach)
        with torch.no_grad():
            m.injector.proj[-1].weight.normal_(0, 0.02)
        b = _batch(6)
        b["sc_coords"] = b["sc_coords"].clone().requires_grad_(True)
        m(**b).square().sum().backward()
        has = b["sc_coords"].grad is not None and bool(
            b["sc_coords"].grad.abs().sum() > 0)
        assert has is expect_grad, (
            f"detach_inputs={detach}: gradient reaching sc_coords was {has}")


def test_neighbourhood_is_shared_across_blocks():
    """One KNN for every block: the neighbourhood is a property of the backbone,
    and blocks disagreeing about it would make the stack order-dependent in a
    way nothing else records.
    """
    torch.manual_seed(7)
    m = SidechainEnvFeedback(c_atom=C_ATOM, c_trunk=C_TRUNK, n_blocks=3)
    seen = []
    for blk in m.env_blocks:
        original = blk.__class__.residue_neighbours
        seen.append(original)
    calls = {"n": 0}
    orig = type(m.env_blocks[0]).residue_neighbours

    def counting(ca, res_mask, n):
        calls["n"] += 1
        return orig(ca, res_mask, n)

    type(m.env_blocks[0]).residue_neighbours = staticmethod(counting)
    try:
        m(**_batch(8))
    finally:
        type(m.env_blocks[0]).residue_neighbours = staticmethod(orig)
    assert calls["n"] == 1, f"residue_neighbours called {calls['n']} times for 3 blocks"


def test_width_mismatch_is_refused():
    """Stage III's injector is 768-wide and the APM packer emits 256. Loading
    one into the other is a shape error; feeding one to the other must be too.
    """
    m = SidechainEnvFeedback(c_atom=C_ATOM, c_trunk=C_TRUNK, n_blocks=1)
    b = _batch(9)
    b["sc_feats"] = torch.randn(B, L, A, 768)
    with pytest.raises(ValueError, match="768"):
        m(**b)


# ---------------------------------------------------------------------------
# Criterion 0 at the integration level
# ---------------------------------------------------------------------------
# The unit test above proves the MODULE returns zeros. That is necessary and not
# sufficient: the term still has to reach `s_trunk_refine` as an addition of
# zeros rather than, say, replacing it, and the cache has to be cleared between
# forwards. These check the wiring, not the module.

def test_injection_is_additive_and_zero_at_init():
    """`s_trunk_refine + env_term` with env_term==0 must leave s_trunk_refine
    bit-identical, including dtype. A `.to(s.dtype)` on a zero tensor is still
    exactly zero, but the ADDITION is what the model does, so add it here too.
    """
    m = SidechainEnvFeedback(c_atom=C_ATOM, c_trunk=C_TRUNK, n_blocks=2).eval()
    env_term = m(**_batch(11))
    s_trunk = torch.randn(B, L, C_TRUNK)
    assert torch.equal(s_trunk + env_term.to(s_trunk.dtype), s_trunk)


def test_cache_is_reset_between_forwards():
    """A stale `_sc_env_cache` would silently feed the PREVIOUS item's side
    chains into this item's refinement pass, and nothing downstream would
    notice. The model clears it per forward; this pins the attribute name that
    contract depends on, so a rename cannot quietly break it.
    """
    import inspect

    from pxdesign_train import model as model_mod

    src = inspect.getsource(model_mod.ProtenixDesignTrain)
    # Set in the packer, consumed in the cycle, and cleared once per forward.
    assert src.count("self._sc_env_cache = None") >= 2, (
        "_sc_env_cache must be initialised in __init__ AND cleared per forward; "
        "found fewer than two assignments to None")
    assert "self._sc_env_cache = env_term" in src
