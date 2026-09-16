"""PXDesign's ragged atom list must densify into atom37 without losing an atom."""

import pytest
import torch

from pxf import atom37, bridge


def _two_residues():
    """One native LEU (7 atoms) and one PXDesign design token (5 atoms)."""
    names = ["N", "CA", "C", "O", "CB", "CG", "CD1"] + ["N", "CA", "C", "O", "OXT"]
    tokens = [0] * 7 + [1] * 5
    res_names = ["LEU"] * 7 + ["xpb"] * 5
    coords = torch.arange(len(names) * 3, dtype=torch.float64).reshape(len(names), 3)
    return names, tokens, res_names, coords


def test_every_atom_reaches_its_slot():
    names, tokens, _, coords = _two_residues()
    dense, mask, dropped = bridge.atoms_to_atom37(coords, names, tokens, 2)
    assert dense.shape == (2, 37, 3)
    assert dropped == []
    assert mask.sum(-1).tolist() == [7, 5]
    for atom, (name, token) in enumerate(zip(names, tokens)):
        slot = atom37.ATOM37.index(name)
        assert torch.equal(dense[token, slot], coords[atom])
        assert bool(mask[token, slot])


def test_empty_slots_stay_zero():
    names, tokens, _, coords = _two_residues()
    dense, mask, _ = bridge.atoms_to_atom37(coords, names, tokens, 2)
    assert torch.all(dense[~mask] == 0)


def test_leading_dimensions_are_preserved():
    names, tokens, _, coords = _two_residues()
    dense, mask, _ = bridge.atoms_to_atom37(
        coords.expand(3, len(names), 3), names, tokens, 2
    )
    assert dense.shape == (3, 2, 37, 3)
    assert mask.shape == (3, 2, 37)


def test_duplicate_atom_in_a_residue_is_rejected():
    with pytest.raises(ValueError, match="Duplicate atom"):
        bridge.atoms_to_atom37(torch.zeros(2, 3), ["CA", "CA"], [0, 0], 1)


def test_unknown_atom_names_are_dropped_or_rejected():
    coords = torch.zeros(3, 3)
    _, mask, dropped = bridge.atoms_to_atom37(coords, ["CA", "H1", "ZN"], [0, 0, 0], 1)
    assert dropped == ["H1", "ZN"] and int(mask.sum()) == 1
    with pytest.raises(ValueError, match="outside the atom37 vocabulary"):
        bridge.atoms_to_atom37(coords, ["CA", "H1", "ZN"], [0, 0, 0], 1, strict=True)


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError, match="atom names"):
        bridge.atoms_to_atom37(torch.zeros(3, 3), ["CA"], [0, 0, 0], 1)
    with pytest.raises(ValueError, match="atom_to_token_idx has"):
        bridge.atoms_to_atom37(torch.zeros(3, 3), ["N", "CA", "C"], [0, 0], 1)
    with pytest.raises(ValueError, match="exceeds num_tokens"):
        bridge.atoms_to_atom37(torch.zeros(2, 3), ["N", "CA"], [0, 5], 1)


def test_autograd_flows_through_the_bridge():
    names, tokens, _, coords = _two_residues()
    coords = coords.clone().requires_grad_(True)
    bridge.atoms_to_atom37(coords, names, tokens, 2)[0].sum().backward()
    assert torch.equal(coords.grad, torch.ones_like(coords))


def test_design_mask_marks_only_xpb_tokens():
    names, tokens, res_names, _ = _two_residues()
    assert bridge.design_mask_from_res_names(res_names, tokens, 2).tolist() == [False, True]


def test_native_sequence_reports_unknown_rather_than_guessing():
    names, tokens, res_names, _ = _two_residues()
    sequence, known = bridge.native_sequence(res_names, tokens, 2)
    assert sequence == "LX"
    assert known.tolist() == [True, False]


def test_non_standard_residues_count_as_unknown():
    sequence, known = bridge.native_sequence(["MSE", "ALA"], [0, 1], 2)
    assert sequence == "XA" and known.tolist() == [False, True]


def test_overrides_complete_the_sequence():
    sequence, known = bridge.native_sequence(["LEU", "xpb"], [0, 1], 2)
    assert bridge.apply_sequence_overrides(sequence, known, {1: "W"}) == "LW"
    # A full-length string supplies only the unknown positions.
    assert bridge.apply_sequence_overrides(sequence, known, "QW") == "LW"


def test_a_missing_identity_is_refused_not_invented():
    sequence, known = bridge.native_sequence(["LEU", "xpb"], [0, 1], 2)
    with pytest.raises(ValueError, match="will not design one"):
        bridge.apply_sequence_overrides(sequence, known, {})


def test_bad_overrides_are_rejected():
    sequence, known = bridge.native_sequence(["LEU", "xpb"], [0, 1], 2)
    with pytest.raises(ValueError, match="not canonical"):
        bridge.apply_sequence_overrides(sequence, known, {1: "B"})
    with pytest.raises(ValueError, match="outside 0"):
        bridge.apply_sequence_overrides(sequence, known, {9: "W"})
    with pytest.raises(ValueError, match="length"):
        bridge.apply_sequence_overrides(sequence, known, "WWW")


def test_token_with_no_atoms_is_rejected():
    with pytest.raises(ValueError, match="no atoms"):
        bridge.token_reduce(["ALA"], [0], 2, how="first")


def test_token_reduce_any_is_an_or_over_each_tokens_atoms():
    """Pinned as an OR rather than as a particular scatter call.

    The reduction is written as an int64 count followed by ``> 0``, not as
    ``scatter_reduce_(amax)`` on a bool tensor: the latter works on the CPU and
    raises ``"cuda_scatter_gather_base_kernel_func" not implemented for 'Bool'``
    on CUDA, so no CPU-only run can reach the failure. This test fixes the
    semantics so the implementation stays free to use a portable op.
    """
    tokens = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    cases = {
        (False, False, False, True, False, False, False, False, False): [
            False,
            True,
            False,
        ],
        (True, False, False, False, False, False, False, False, True): [True, False, True],
        (False,) * 9: [False, False, False],
        (True,) * 9: [True, True, True],
    }
    for flags, expected in cases.items():
        out = bridge.token_reduce(list(flags), tokens, 3, how="any")
        assert out.dtype == torch.bool
        assert out.tolist() == expected


def test_token_reduce_any_handles_a_token_with_many_true_atoms():
    """A count-based OR must not overflow or saturate into the wrong answer."""
    out = bridge.token_reduce([True] * 500, [0] * 500, 1, how="any")
    assert out.dtype == torch.bool and out.tolist() == [True]
