"""chi_output makes covalent geometry a constant, not a regression target.

The failure this replaces: a deterministic Cartesian head trained on coordinate
MSE converges to the conditional mean, and ||E[a-b]|| <= E[||a-b||] (Jensen),
so every bond whose direction varies across rotamers comes out short. Measured
on the step46000 warmup donor: 92.7% of internal side-chain bonds short by a
mean of 0.41 A, while CB-CA -- pinned by the GT frame, direction invariant --
was clean at 53% (native control: 53%).
"""
import torch
import pytest

import pxdesign_train.sidechain.buildsc as buildsc
from pxdesign_train.sidechain.chemistry import canonical_registry
from pxdesign_train.sidechain.frames import to_global
from pxdesign_train.sidechain.instantiate import MAX_SC, sidechain_atoms
from pxdesign_train.sidechain.module import SideChainModule

STD = buildsc.STD_AA_3
RESIDUES = ("ARG", "PHE", "LYS", "SER", "TRP", "GLU")


def _example():
    types = torch.tensor([[STD.index(a) for a in RESIDUES]])
    L = types.shape[1]
    ids = torch.zeros(1, L, MAX_SC, dtype=torch.long)
    mask = torch.zeros(1, L, MAX_SC, dtype=torch.bool)
    for j in range(L):
        n = len(sidechain_atoms(STD[int(types[0, j])])[:MAX_SC])
        mask[0, j, :n] = True
        ids[0, j, :n] = torch.arange(1, n + 1)
    h = torch.randn(1, L, 32)
    logits = torch.nn.functional.one_hot(types, 20).float() * 40.0 - 20.0
    frame_R = torch.eye(3).expand(1, L, 3, 3).contiguous()
    frame_t = torch.randn(1, L, 3)
    template, _ = buildsc.build_sidechain_local(types)
    return types, ids, mask, h, logits, frame_R, frame_t, to_global(template, frame_R, frame_t)


def _module(chi_output, scramble=False):
    m = SideChainModule(c_res=32, c_atom=32, c_time=16, n_type=20, n_blocks=2,
                        n_heads=4, chi_output=chi_output)
    if scramble:
        with torch.no_grad():
            for p in m.parameters():
                if p.dim() > 1:
                    p.normal_(0, 0.5)
    return m


def _bond_errors(xyz, types):
    registry = canonical_registry()
    errors = []
    for j in range(types.shape[1]):
        aa = STD[int(types[0, j])]
        slots = {n: k for k, n in enumerate(sidechain_atoms(aa)[:MAX_SC])}
        ideal, _ = buildsc.build_sidechain_local(types[0, j])
        for u, v in registry[aa].bonds:
            if u in slots and v in slots:
                iu, iv = slots[u], slots[v]
                errors.append(((xyz[0, j, iu] - xyz[0, j, iv]).norm()
                               - (ideal[iu] - ideal[iv]).norm()).item())
    return torch.tensor(errors)


def test_arbitrary_weights_cannot_break_bond_lengths():
    torch.manual_seed(0)
    types, ids, mask, h, logits, fR, ft, noisy = _example()
    t = torch.tensor([0.5])
    free = _module(False, scramble=True)(h, logits, ids, mask, noisy, t, frame_R=fR, frame_t=ft)[0]
    chi = _module(True, scramble=True)(h, logits, ids, mask, noisy, t, frame_R=fR, frame_t=ft)[0]
    free_err, chi_err = _bond_errors(free, types), _bond_errors(chi, types)
    # The free head contracts essentially every bond; that is the whole failure.
    assert free_err.mean() < -0.1 and (free_err < 0).float().mean() > 0.9
    # The chi head is exact regardless of what the weights do.
    assert chi_err.abs().max() < 1e-4


def test_zero_initialised_chi_head_reproduces_the_template():
    torch.manual_seed(0)
    types, ids, mask, h, logits, fR, ft, noisy = _example()
    out = _module(True)(h, logits, ids, mask, noisy, torch.tensor([0.5]), frame_R=fR, frame_t=ft)[0]
    torch.testing.assert_close(out, torch.where(mask[..., None], noisy, 0.), atol=1e-5, rtol=0)


def test_chi_head_is_differentiable_and_frame_required():
    torch.manual_seed(0)
    types, ids, mask, h, logits, fR, ft, noisy = _example()
    m = _module(True)
    out = m(h, logits, ids, mask, noisy, torch.tensor([0.5]), frame_R=fR, frame_t=ft)[0]
    out.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.chi_out.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="frame-aware"):
        _module(True)(h, logits, ids, mask, noisy, torch.tensor([0.5]))


def test_chi_output_and_template_residual_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        SideChainModule(c_res=32, c_atom=32, c_time=16, n_type=20, n_blocks=2,
                        n_heads=4, chi_output=True, template_residual=True)


def test_builder_preserves_every_bond_under_arbitrary_torsions():
    torch.manual_seed(0)
    registry = canonical_registry()
    worst = 0.0
    for aa in STD:
        record = registry.get(aa)
        if record is None or not getattr(record, "bonds", None):
            continue
        idx = torch.tensor(STD.index(aa))
        slots = {n: k for k, n in enumerate(sidechain_atoms(aa)[:MAX_SC])}
        ideal, _ = buildsc.build_sidechain_local(idx)
        for _ in range(8):
            chi = torch.rand(4) * 2 * torch.pi - torch.pi
            xyz, _ = buildsc.build_sidechain_local(idx, chi)
            for u, v in record.bonds:
                if u in slots and v in slots:
                    iu, iv = slots[u], slots[v]
                    worst = max(worst, abs(float((xyz[iu] - xyz[iv]).norm()
                                                 - (ideal[iu] - ideal[iv]).norm())))
    assert worst < 1e-4
