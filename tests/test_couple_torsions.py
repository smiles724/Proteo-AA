"""Chi torsions must be measured right and perturbed without breaking anything.

The perturbation is the load-bearing part: the whole point of a torsion probe,
as against Gaussian noise on coordinates, is that it lands on a structure the
packer could actually have produced. If it also shifts a bond length or drags
the backbone, it is just noise with extra steps and the sensitivity number it
produces means nothing.
"""

import math

import pytest
import torch

from pxf import atom37
from pxf.couple import torsions


@pytest.fixture(scope="module")
def rc():
    from fampnn.data import residue_constants as rc

    return rc


@pytest.fixture(scope="module")
def structure():
    """One real protein in atom37, with its native side chains."""
    from pxf.provenance import repo_root
    from pxf.train.data import StructureCropDataset, collate

    dataset = StructureCropDataset(
        [str(repo_root() / "fampnn/data/casp14/pdbs/T1031.pdb")],
        crop_size=48,
        noise=0.0,
        seed=0,
    )
    item = collate([dataset[0]])
    available = (
        torch.as_tensor(
            __import__(
                "fampnn.data.residue_constants", fromlist=["x"]
            ).STANDARD_ATOM_MASK_WITH_X
        )[item["aatype"].long()]
        * (1.0 - item["missing_atom_mask"])
    )
    return dict(coords=item["x"], aatype=item["aatype"].long(), available=available)


def test_the_downstream_table_matches_the_rigid_group_rule(rc):
    """chi_k carries exactly the atoms in AF2 rigid groups >= 4 + k."""
    tables = torsions.chi_tables(rc=rc)
    groups = torch.as_tensor(rc.restype_atom37_to_rigid_group)
    exists = torch.as_tensor(rc.STANDARD_ATOM_MASK_WITH_X).bool()
    for index in range(20):
        for k in range(torsions.MAX_CHI):
            expected = (groups[index] >= 4 + k) & exists[index]
            assert torch.equal(tables.downstream[index, k], expected)
    # No chi ever carries a backbone atom.
    for slot in atom37.BACKBONE_SLOTS:
        assert not bool(tables.downstream[..., slot].any())


def test_proline_is_measurable_but_not_rotatable(rc):
    tables = torsions.chi_tables(rc=rc)
    pro = atom37.AA_ORDER.index("P")
    assert bool(tables.mask[pro].any()), "PRO does have chis"
    assert not bool(tables.rotatable[pro].any()), (
        "PRO's ring closes onto the backbone N, so no rigid rotation about its "
        "chi axes keeps the pyrrolidine intact"
    )
    gly = atom37.AA_ORDER.index("G")
    assert not bool(tables.mask[gly].any())


def test_chi_is_recovered_after_a_known_rotation(structure, rc):
    """Rotate by a known angle; the measured chi must move by that angle."""
    coords, aatype = structure["coords"], structure["aatype"]
    available = structure["available"]
    tables = torsions.chi_tables(rc=rc)
    before, valid = torsions.chi_angles(coords, aatype, available=available, tables=tables)
    delta = math.radians(37.0)
    deltas = torch.zeros(*aatype.shape, torsions.MAX_CHI)
    deltas[..., 0] = delta
    moved = torsions.perturb_chi(
        coords, aatype, deltas, available=available, tables=tables
    )
    after, valid_after = torsions.chi_angles(
        moved, aatype, available=available, tables=tables
    )
    rotatable = tables.rotatable[aatype][..., 0] & valid[..., 0] & valid_after[..., 0]
    assert bool(rotatable.any()), "no residue had a rotatable, measurable chi1"
    shift = (after[..., 0] - before[..., 0])[rotatable]
    wrapped = torch.atan2(torch.sin(shift), torch.cos(shift))
    assert torch.allclose(wrapped, torch.full_like(wrapped, delta), atol=1e-4)


def test_a_rotation_preserves_the_backbone_and_the_covalent_geometry(structure, rc):
    coords, aatype = structure["coords"], structure["aatype"]
    available = structure["available"]
    tables = torsions.chi_tables(rc=rc)
    deltas = torsions.random_chi_deltas(
        aatype,
        math.radians(120.0),
        generator=torch.Generator().manual_seed(0),
        tables=tables,
    )
    moved = torsions.perturb_chi(
        coords, aatype, deltas, available=available, tables=tables
    )
    backbone = list(atom37.BACKBONE_SLOTS)
    assert torch.allclose(moved[..., backbone, :], coords[..., backbone, :], atol=1e-5)

    # Every intra-residue distance between available atoms is a bond length, a
    # bond angle or a torsion-independent constraint of the rigid groups the
    # rotation did not separate. Checking all pairs is stronger than checking a
    # bond list and needs no bond table: distances that a chi rotation is
    # *allowed* to change are exactly those spanning the rotated axis, so
    # compare only within each rigid group.
    groups = torch.as_tensor(rc.restype_atom37_to_rigid_group)[aatype]  # [B, L, 37]
    keep = available.bool()
    for group in range(8):
        member = keep & (groups == group)
        if not bool(member.any()):
            continue
        for b in range(coords.shape[0]):
            for i in range(coords.shape[1]):
                slots = torch.nonzero(member[b, i], as_tuple=True)[0]
                if slots.numel() < 2:
                    continue
                d0 = torch.cdist(coords[b, i, slots], coords[b, i, slots])
                d1 = torch.cdist(moved[b, i, slots], moved[b, i, slots])
                assert torch.allclose(d0, d1, atol=1e-4), (
                    f"rigid group {group} of residue {i} was distorted"
                )


def test_a_perturbation_actually_moves_the_side_chains(structure, rc):
    coords, aatype = structure["coords"], structure["aatype"]
    available = structure["available"]
    tables = torsions.chi_tables(rc=rc)
    deltas = torsions.random_chi_deltas(
        aatype,
        math.radians(60.0),
        generator=torch.Generator().manual_seed(1),
        tables=tables,
    )
    moved = torsions.perturb_chi(
        coords, aatype, deltas, available=available, tables=tables
    )
    sidechain = list(atom37.SIDECHAIN_SLOTS)
    shift = (moved[..., sidechain, :] - coords[..., sidechain, :]).norm(dim=-1)
    assert float(shift.max()) > 0.5, "a 60 degree chi rotation moved nothing"


def test_only_one_chi_per_residue_is_perturbed_by_default(structure, rc):
    tables = torsions.chi_tables(rc=rc)
    aatype = structure["aatype"]
    deltas = torsions.random_chi_deltas(
        aatype, 1.0, generator=torch.Generator().manual_seed(2), tables=tables
    )
    moved_count = (deltas != 0).sum(dim=-1)
    rotatable_any = tables.rotatable[aatype].any(dim=-1)
    assert torch.all(moved_count[rotatable_any] == 1)
    assert torch.all(moved_count[~rotatable_any] == 0)


def test_sin_cos_zeroes_invalid_chis(rc):
    chi = torch.tensor([[[0.5, 1.0, 2.0, 3.0]]])
    valid = torch.tensor([[[True, True, False, False]]])
    features = torsions.chi_sin_cos(chi, valid)
    assert features.shape[-1] == 2 * torsions.MAX_CHI
    assert float(features[0, 0, 2]) == 0.0 and float(features[0, 0, 3]) == 0.0
    assert float(features[0, 0, 6]) == 0.0 and float(features[0, 0, 7]) == 0.0
    assert float(features[0, 0, 0]) == pytest.approx(math.sin(0.5))


def test_chi_is_invariant_to_a_rigid_motion(structure, rc):
    coords, aatype = structure["coords"], structure["aatype"]
    available = structure["available"]
    before, valid = torsions.chi_angles(coords, aatype, available=available)
    generator = torch.Generator().manual_seed(3)
    a = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))[None, :]
    if float(torch.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    moved = (coords.double() @ q + torch.tensor([3.0, -7.0, 11.0])).to(coords.dtype)
    after, _ = torsions.chi_angles(moved, aatype, available=available)
    assert torch.allclose(before[valid], after[valid], atol=1e-3)
