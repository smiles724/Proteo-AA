"""Chi torsions in atom37: measured for the readout, perturbed for the probes.

Two uses, one set of tables.

**Features.** The feedback readout needs local side-chain conformation, and chi
torsions are the right coordinate for it: they are the degrees of freedom the
packer actually chose, they are invariant to the global pose, and sine/cosine of
them is smooth across the +-pi wrap that the raw angle is not.

**Perturbations.** The decisive diagnostic for SC -> BB feedback is whether
re-encoding responds to a *plausible* change of packing. Gaussian noise on
side-chain coordinates and collapsing every side chain onto CA both change the
representation, but neither is a structure the packer could emit: they break bond
lengths and bond angles, so a response to them says only that the encoder reads
coordinates at all. Rotating about a chi axis is different -- it preserves every
bond length, every bond angle, the backbone and the sequence, and lands on
another *valid* rotamer of the same residue. That is the intervention the
feedback has to be sensitive to.

Everything is derived from FaMPNN's own ``residue_constants``, which is a pinned
submodule, so nothing here depends on an external checkout:

* ``chi_angles_atoms`` gives the four atoms defining each chi;
* ``chi_angles_mask`` gives which chis a residue type has;
* ``restype_atom37_to_rigid_group`` gives the AF2 rigid group of every atom, and
  the atoms carried by chi_k are exactly those in groups ``>= 4 + k`` -- groups
  0-3 are backbone/pre-omega/phi/psi and 4-7 are chi1-chi4.

**PRO is excluded from rotation.** Its ring closes back onto the backbone N, so
no rigid rotation about CA-CB or CB-CG keeps the pyrrolidine intact. Its chis are
still *measured*; they are just never perturbed, which ``rotatable`` records.
"""

from dataclasses import dataclass

import torch

from pxf import atom37

MAX_CHI = 4
# AF2 rigid groups: 0 backbone, 1 pre-omega, 2 phi, 3 psi, 4..7 chi1..chi4.
CHI_GROUP_OFFSET = 4
RING_CLOSED = ("PRO",)


@dataclass
class ChiTables:
    """Per-residue-type chi geometry, on one device. Built once, reused."""

    atom_index: torch.Tensor  # [21, 4, 4] long, atom37 slots of each chi's atoms
    mask: torch.Tensor  # [21, 4] bool, the residue type has this chi
    rotatable: torch.Tensor  # [21, 4] bool, a rigid rotation about it exists
    downstream: torch.Tensor  # [21, 4, 37] bool, atoms carried by the rotation

    def to(self, device):
        from dataclasses import replace

        return replace(
            self,
            atom_index=self.atom_index.to(device),
            mask=self.mask.to(device),
            rotatable=self.rotatable.to(device),
            downstream=self.downstream.to(device),
        )


_CACHE = {}


def chi_tables(*, rc=None, device=None):
    """The chi tables, cached per device."""
    key = str(device)
    if key in _CACHE:
        return _CACHE[key]
    if rc is None:
        from fampnn.data import residue_constants as rc

    n_types = atom37.UNKNOWN_AA_INDEX + 1  # 20 canonical plus X
    atom_index = torch.zeros(n_types, MAX_CHI, 4, dtype=torch.long)
    mask = torch.zeros(n_types, MAX_CHI, dtype=torch.bool)
    rotatable = torch.zeros(n_types, MAX_CHI, dtype=torch.bool)
    downstream = torch.zeros(n_types, MAX_CHI, atom37.NUM_ATOM37, dtype=torch.bool)

    groups = torch.as_tensor(rc.restype_atom37_to_rigid_group)
    exists = torch.as_tensor(rc.STANDARD_ATOM_MASK_WITH_X).bool()
    for index, letter in enumerate(atom37.AA_ORDER):
        name3 = rc.restype_1to3[letter]
        for k, quadruple in enumerate(rc.chi_angles_atoms[name3]):
            atom_index[index, k] = torch.tensor(
                [atom37.ATOM37.index(name) for name in quadruple], dtype=torch.long
            )
            mask[index, k] = bool(rc.chi_angles_mask[index][k])
            rotatable[index, k] = mask[index, k] and name3 not in RING_CLOSED
            downstream[index, k] = (groups[index] >= CHI_GROUP_OFFSET + k) & exists[index]
    tables = ChiTables(
        atom_index=atom_index, mask=mask, rotatable=rotatable, downstream=downstream
    )
    if device is not None:
        tables = tables.to(device)
    _CACHE[key] = tables
    return tables


def dihedral(p0, p1, p2, p3):
    """Signed torsion p0-p1-p2-p3 in radians, ``[...]``.

    The standard four-point formula. Degenerate configurations (collinear atoms,
    coincident atoms) come out as 0 rather than nan, and the caller's validity
    mask is what excludes them -- a nan here would propagate into the readout.
    """
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


def _gather(coords37, index):
    """``coords37[..., index, :]`` with ``index`` broadcast per residue."""
    return torch.gather(coords37, -2, index[..., None].expand(*index.shape, 3))


def chi_angles(coords37, aatype, *, available=None, tables=None):
    """``(chi, valid)`` -- ``[..., 4]`` radians and the mask of measurable chis.

    A chi is valid only when the residue type has it **and** all four of its
    atoms are available with finite coordinates, so a partially packed residue
    contributes its measurable chis and nothing more.
    """
    tables = tables or chi_tables(device=coords37.device)
    aatype = aatype.clamp(min=0, max=atom37.UNKNOWN_AA_INDEX).long()
    index = tables.atom_index[aatype]  # [..., 4, 4]
    valid = tables.mask[aatype].clone()  # [..., 4]

    flat = index.reshape(*index.shape[:-2], MAX_CHI * 4)
    points = _gather(coords37, flat).reshape(*index.shape, 3)
    if available is not None:
        got = torch.gather(available, -1, flat).reshape(*index.shape)
        valid = valid & (got > 0).all(dim=-1)
    valid = valid & torch.isfinite(points).all(dim=-1).all(dim=-1)

    chi = dihedral(
        points[..., 0, :], points[..., 1, :], points[..., 2, :], points[..., 3, :]
    )
    return torch.where(valid, chi, torch.zeros_like(chi)), valid


def chi_sin_cos(chi, valid):
    """``[..., 8]`` sine/cosine of each chi, zeroed where the chi is invalid.

    Zero rather than a sentinel: the readout is a linear map over these, and the
    validity flags travel alongside, so an absent chi contributes nothing to the
    sum instead of contributing a magic number the network has to learn to
    ignore.
    """
    keep = valid.to(chi.dtype)
    return torch.cat([torch.sin(chi) * keep, torch.cos(chi) * keep], dim=-1)


def _rodrigues(v, axis, angle):
    """Rotate ``v`` ``[..., A, 3]`` about unit ``axis`` ``[..., 3]`` by ``angle``."""
    k = axis[..., None, :]
    a = angle[..., None, None]
    return (
        v * torch.cos(a)
        + torch.cross(k.expand_as(v), v, dim=-1) * torch.sin(a)
        + k * (k * v).sum(-1, keepdim=True) * (1.0 - torch.cos(a))
    )


def perturb_chi(coords37, aatype, deltas, *, available=None, tables=None):
    """Rotate about chi axes by ``deltas`` radians. Backbone and sequence untouched.

    ``deltas`` is ``[..., 4]``; a zero or non-finite entry leaves that chi alone.
    Each rotation is applied in turn, with the axis recomputed from the current
    coordinates, because chi_k's defining atoms have already been moved by
    chi_1..chi_{k-1}.

    Returns a new ``[..., 37, 3]`` tensor. Every bond length and bond angle in
    the residue is preserved exactly -- a rotation about a bond cannot change
    them -- so the result is a different rotamer of the same residue rather than
    a distorted one.
    """
    tables = tables or chi_tables(device=coords37.device)
    aatype = aatype.clamp(min=0, max=atom37.UNKNOWN_AA_INDEX).long()
    coords = coords37.clone()
    index = tables.atom_index[aatype]
    rotatable = tables.rotatable[aatype]
    downstream = tables.downstream[aatype]
    backbone = list(atom37.BACKBONE_SLOTS)

    for k in range(MAX_CHI):
        delta = deltas[..., k]
        active = rotatable[..., k] & torch.isfinite(delta) & (delta != 0)
        quad = index[..., k, :]
        if available is not None:
            got = torch.gather(available, -1, quad)
            active = active & (got > 0).all(dim=-1)
        points = _gather(coords, quad)
        active = active & torch.isfinite(points).all(dim=-1).all(dim=-1)
        if not bool(active.any()):
            continue
        p1, p2 = points[..., 1, :], points[..., 2, :]
        axis = p2 - p1
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        angle = torch.where(active, delta, torch.zeros_like(delta))
        moved = downstream[..., k, :] & active[..., None]
        # The backbone is never carried: groups >= 4 exclude N/CA/C/O by
        # construction, and this makes that a guarantee rather than a property
        # of the upstream table.
        moved[..., backbone] = False
        rotated = _rodrigues(coords - p2[..., None, :], axis, angle) + p2[..., None, :]
        coords = torch.where(moved[..., None], rotated, coords)
    return coords


def random_chi_deltas(
    aatype, magnitude, *, generator=None, tables=None, one_per_residue=True
):
    """``[..., 4]`` perturbations of ``magnitude`` radians with random signs.

    ``one_per_residue`` perturbs a single randomly chosen rotatable chi per
    residue, which is the closer analogue of a rotamer flip than moving all four
    at once; the latter compounds into a large displacement that no packer would
    produce.
    """
    tables = tables or chi_tables(device=aatype.device)
    aatype_c = aatype.clamp(min=0, max=atom37.UNKNOWN_AA_INDEX).long()
    rotatable = tables.rotatable[aatype_c]  # [..., 4]
    sign = (
        torch.randint(
            0, 2, rotatable.shape, generator=generator, device=aatype.device
        ).float()
        * 2.0
        - 1.0
    )
    deltas = sign * float(magnitude) * rotatable.float()
    if not one_per_residue:
        return deltas
    weights = rotatable.float() + 1e-9
    flat = weights.reshape(-1, MAX_CHI)
    pick = torch.multinomial(flat, 1, generator=generator).reshape(*rotatable.shape[:-1], 1)
    chosen = torch.zeros_like(rotatable).scatter_(-1, pick, True) & rotatable
    return deltas * chosen.float()
