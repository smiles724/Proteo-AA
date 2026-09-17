"""Diagnostics for whether the coupling paths can carry signal at all.

The decisive question for SC -> BB feedback is whether

    h_packed = E_phi(X_BB, X_SC, S)

actually depends on *which* side chains were realized. If re-encoding is
insensitive to the side-chain conformation, then no adapter on top of h_packed
can transmit anything about packing back to the backbone, and the feedback
direction is dead regardless of how it is trained.

:func:`sidechain_sensitivity` measures that against two references that make the
number interpretable:

* a **floor** -- re-encoding the identical input twice, which is 0 for a
  deterministic encoder and bounds the noise;
* a **ceiling** -- the change from fully masked side chains to visible ones,
  which is the largest effect side-chain input can have on the encoder.

A perturbation response near the floor means no signal; one comparable to the
ceiling means the path is informative.

**Which perturbation is asked matters more than the margin.** Gaussian noise on
side-chain coordinates and collapsing every side chain onto CA are *wiring*
diagnostics: they establish that the encoder reads the side-chain block at all.
Neither is a structure the packer could emit -- both break bond lengths and bond
angles -- so a response to them does not establish sensitivity to the thing the
feedback would actually vary, which is the rotamer the packer chose. The
torsion probes do: rotating about a chi axis preserves every bond length, every
bond angle, the backbone and the sequence, and lands on another valid rotamer of
the same residue. :meth:`SensitivityReport.verdict` therefore reads the torsion
family by default.

**Invariances are checked alongside, not instead.** A response that came from
padded or nonexistent atoms, or one that would vanish under a rigid motion of
the whole structure, is an artefact of the plumbing rather than a signal. Three
are measured, with the tolerance each actually deserves
(:data:`INVARIANT_TOLERANCE`):

* **nonexistent slots** -- scrambling atom37 slots the residue type does not
  have must change nothing, exactly. Availability excludes them.
* **padded atom coordinates** -- with padding rows present, changing what is
  *in* them should change nothing. This is the leakage question, and it is asked
  by comparing two padded runs rather than a padded run against an unpadded one.
  It does **not** come out at zero: FaMPNN's encoder leaks on the order of
  1e-3 relative, measured from 2.6e-4 to 1.2e-3 across lengths 40 to 64 and over
  both native and predicted packings, and not a function of length -- so it is
  not the ``min(top_k, L)`` neighbour-count effect one would first suspect. It is upstream and small; the tolerance is set above it and
  says so, rather than being quietly widened. The mitigation is structural --
  the coupling path never pads -- and
  :meth:`~pxf.couple.controller.CoupledDenoiser.encode_predicted_packing` warns
  if it ever does.
* **rigid motion** -- rotating and translating the complete input must not move
  the representation beyond float32 precision. Measured at ~2e-5 relative, which
  is why this one carries an absolute tolerance instead of a multiple of the
  floor: the floor is exactly 0 for a deterministic encoder, and no rotation in
  float32 will reproduce that.

**Appending padding rows at all is a separate diagnostic.** It moves the
representation by ~2e-3 to 5e-3 on average -- several times the coordinate leak
above --
concentrated almost entirely on the *last real residue*: FaMPNN's encoder reads
an index-neighbour feature, so the C-terminus stops looking like a terminus once
a row follows it. Recorded as ``terminus_shift_from_padding`` and not counted as
a failure, because it is a property of where the chain ends rather than of what
the padding contains.

Neither matters for the pilot, and for a structural reason rather than a lucky
one: the converter gives every structure ``seq_mask = 1`` at its own length and
the cycle runs one structure per forward, so no padded row exists to leak from.
Batching the coupled cycle would reintroduce both effects at the ~1e-3 scale,
and would need this re-measured first.
"""

from dataclasses import dataclass, field

import torch

from pxf import atom37
from pxf.couple import fampnn_iface as iface
from pxf.couple import torsions
from pxf.couple import visibility as vis

# Rotamer-scale torsion perturbations, in degrees. 10 is within a rotamer well,
# 30 is its edge, 120 is a different well -- the range a packer's choices span.
DEFAULT_TORSIONS = (10.0, 30.0, 60.0, 120.0)
# Deliberately not the default any more; kept because they are the cheapest way
# to tell "the encoder ignores the block" from "the encoder is subtle".
DEFAULT_GAUSSIAN = (0.1, 0.5, 1.0)
INVARIANCE_MARGIN = 10.0
# The relative change each invariant is allowed. Zero means "at the encoder's
# own floor"; the rigid budget is float32 precision for a rotation, measured at
# ~2e-5 and allowed an order of magnitude of headroom.
INVARIANT_TOLERANCE = {
    # Exact: availability excludes these slots, so nothing can read them.
    "nonexistent_atoms_scrambled": 0.0,
    # NOT exact. An upstream leak in FaMPNN's encoder, measured from 2.6e-4 to
    # 1.2e-3 relative, independent of length, and larger on a predicted packing
    # than on a native one. Set a few times above the largest measurement and
    # documented in the module docstring rather than widened silently.
    "padded_atom_coordinates": 5e-3,
    # float32 precision for a rotation; measured at ~2e-5.
    "rigid_rotation_translation": 1e-3,
}


def _relative_change(a, b, mask=None):
    """Mean per-residue relative L2 change between two feature tensors."""
    delta = (a - b).float()
    scale = torch.maximum(a.float().norm(dim=-1), b.float().norm(dim=-1))
    per_residue = delta.norm(dim=-1) / scale.clamp_min(1e-8)
    if mask is not None:
        keep = mask.bool()
        if not bool(keep.any()):
            return float("nan")
        return float(per_residue[keep].mean())
    return float(per_residue.mean())


@dataclass
class SensitivityReport:
    """How much re-encoding responds to the side chains it is shown."""

    floor: float  # identical input re-encoded (noise floor)
    ceiling: float  # masked -> visible side chains
    responses: dict = field(default_factory=dict)  # perturbation -> relative change
    invariants: dict = field(default_factory=dict)  # transformation -> relative change
    stats: dict = field(default_factory=dict)

    def family(self, prefix):
        return {k: v for k, v in self.responses.items() if k.startswith(prefix)}

    def verdict(self, *, margin=10.0, prefix="chi_"):
        """Whether the strongest *plausible* response clears the floor.

        ``prefix`` selects the family. The default is the torsion probes: a
        model that only responds to bond-breaking perturbations has not been
        shown to respond to packing.
        """
        family = self.family(prefix) or self.responses
        best = max(family.values()) if family else 0.0
        if best <= max(self.floor, 1e-9) * margin:
            return "insensitive"
        if best >= 0.1 * self.ceiling:
            return "informative"
        return "weak"

    def invariance_failures(self, *, margin=INVARIANCE_MARGIN):
        """Invariants that moved the representation more than they are allowed to.

        Non-empty means a measured response may be an artefact: the feature the
        adapter reads is responding to padding, to nonexistent atoms, or to the
        global pose, none of which carry information about the packing.
        """
        out = {}
        for key, value in self.invariants.items():
            limit = max(self.floor * margin, INVARIANT_TOLERANCE.get(key, 0.0), 1e-9)
            if value > limit:
                out[key] = value
        return out

    def summary(self):
        return dict(
            floor=self.floor,
            ceiling=self.ceiling,
            responses=dict(self.responses),
            invariants=dict(self.invariants),
            invariance_failures=self.invariance_failures(),
            verdict=self.verdict(),
            verdict_gaussian=self.verdict(prefix="gaussian_"),
            stats=dict(self.stats),
        )


def _availability(aatype, seq_mask, supplied, coords, sidechains):
    return vis.predicted_availability(
        aatype, seq_mask, supplied, coords, sidechains=sidechains
    ).available


def _rigid_motion(coords, *, generator=None):
    """A random rotation and translation of a whole structure.

    Built from a QR decomposition so the rotation is exactly orthogonal; a
    composed-Euler-angle rotation accumulates enough float error to sit above
    the encoder's own noise floor and would make the invariance check fail for
    the wrong reason.
    """
    a = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))[None, :]
    if float(torch.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.randn(3, generator=generator, dtype=torch.float64) * 10.0
    moved = coords.double() @ q + shift
    return moved.to(coords.dtype), q, shift


@torch.no_grad()
def sidechain_sensitivity(
    model,
    coords_af2,
    aatype,
    *,
    seq_mask=None,
    missing_atom_mask=None,
    residue_index=None,
    chain_index=None,
    sidechains=None,
    supplied_atom_mask=None,
    perturbations=DEFAULT_GAUSSIAN,
    torsion_perturbations=DEFAULT_TORSIONS,
    generator=None,
):
    """Measure how ``h_packed`` responds to perturbing the side chains shown.

    ``sidechains`` defaults to the side chains already in ``coords_af2`` (i.e. the
    native ones); pass a packed prediction to probe the realization the coupled
    system would actually feed back -- which is what the pilot cares about, since
    a predicted packing is the only thing the feedback ever sees.

    Every encode goes through :func:`pxf.couple.visibility.predicted_availability`
    rather than the caller's ``missing_atom_mask``, so the probe measures the
    same path the controller takes. ``missing_atom_mask`` is accepted and used
    only for the masked-side-chain reference, where it is the correct mask.
    """
    from fampnn.data import residue_constants as rc

    device = coords_af2.device
    batch, length = coords_af2.shape[0], coords_af2.shape[1]
    if seq_mask is None:
        seq_mask = torch.ones(batch, length, device=device)
    scored = seq_mask
    backbone = list(atom37.BACKBONE_SLOTS)
    if supplied_atom_mask is None:
        # What a PXDesign proposal supplies: the four backbone atoms.
        supplied_atom_mask = torch.zeros(batch, length, atom37.NUM_ATOM37, device=device)
        supplied_atom_mask[..., backbone] = 1.0

    block = (
        coords_af2[..., rc.non_bb_idxs, :]
        if sidechains is None
        else (
            sidechains[..., rc.non_bb_idxs, :]
            if sidechains.shape[-2] == atom37.NUM_ATOM37
            else sidechains
        )
    )

    index = dict(residue_index=residue_index, chain_index=chain_index)

    def encode(coords, sc, *, aa=None, mask=None, supplied=None, numbering=None):
        aa = aatype if aa is None else aa
        mask = seq_mask if mask is None else mask
        supplied = supplied_atom_mask if supplied is None else supplied
        available = _availability(aa, mask, supplied, coords, sc)
        _logits, h, _features = iface.encode(
            model,
            coords,
            aa,
            sidechains=sc,
            atom_availability=available,
            seq_mask=mask,
            **(index if numbering is None else numbering),
        )
        return h, available

    # The masked reference keeps the caller's observation mask: with no side
    # chains supplied, "was this atom in the input" is exactly the right
    # question and availability would answer the same thing.
    _logits, h_masked, _f = iface.encode(
        model,
        coords_af2,
        aatype,
        seq_mask=seq_mask,
        missing_atom_mask=missing_atom_mask,
        **index,
    )
    h_visible, available = encode(coords_af2, block)
    h_repeat, _ = encode(coords_af2, block)

    report = SensitivityReport(
        floor=_relative_change(h_visible, h_repeat, scored),
        ceiling=_relative_change(h_visible, h_masked, scored),
    )
    report.stats = dict(
        residues=int(seq_mask.sum()),
        sidechain_atoms_available=int(available[..., rc.non_bb_idxs].sum()),
        sidechains_supplied=sidechains is not None,
    )

    # --- the meaningful probe: valid rotamer changes ---------------------
    # Assembled into atom37 once, so a rotation moves the chi-defining backbone
    # atoms and the side chain consistently.
    full = coords_af2.clone()
    full[..., rc.non_bb_idxs, :] = block.to(full.dtype)
    tables = torsions.chi_tables(device=device)
    chi, chi_valid = torsions.chi_angles(full, aatype, available=available, tables=tables)
    report.stats["chis_measurable"] = int(chi_valid.sum())
    report.stats["residues_with_a_rotatable_chi"] = int(
        (tables.rotatable[aatype.clamp(0, atom37.UNKNOWN_AA_INDEX).long()].any(-1)).sum()
    )
    for degrees in torsion_perturbations:
        deltas = torsions.random_chi_deltas(
            aatype,
            float(degrees) * torch.pi / 180.0,
            generator=generator,
            tables=tables,
        )
        moved_full = torsions.perturb_chi(
            full, aatype, deltas, available=available, tables=tables
        )
        h_moved, _ = encode(moved_full, moved_full[..., rc.non_bb_idxs, :])
        report.responses[f"chi_{degrees:g}deg"] = _relative_change(
            h_visible, h_moved, scored
        )
        # The backbone must be untouched by construction; assert it here so a
        # regression in perturb_chi cannot masquerade as side-chain sensitivity.
        assert torch.allclose(
            moved_full[..., backbone, :], full[..., backbone, :], atol=1e-5
        ), "the torsion perturbation moved the backbone"

    # --- wiring diagnostics: perturbations no packer could emit -----------
    exists = torch.as_tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=device)
    exists = exists[aatype.clamp(0, atom37.UNKNOWN_AA_INDEX).long()][..., rc.non_bb_idxs]
    for sigma in perturbations:
        noise = torch.randn(
            block.shape, generator=generator, dtype=block.dtype, device=block.device
        )
        moved = block + noise * float(sigma) * exists[..., None]
        h_moved, _ = encode(coords_af2, moved)
        report.responses[f"gaussian_{sigma}A"] = _relative_change(
            h_visible, h_moved, scored
        )

    ca = coords_af2[..., rc.atom_order["CA"], :]
    collapsed = ca[..., None, :].expand_as(block).contiguous()
    h_collapsed, _ = encode(coords_af2, collapsed)
    report.responses["collapsed_to_CA"] = _relative_change(h_visible, h_collapsed, scored)

    # --- invariants: these must NOT move the representation ---------------
    # Nonexistent slots: scramble every atom37 slot the residue type does not
    # have. Availability excludes them, so the encoder must not notice.
    scrambled = coords_af2.clone()
    scrambled_block = block.clone()
    noise = torch.randn(
        block.shape, generator=generator, dtype=block.dtype, device=block.device
    )
    scrambled_block = scrambled_block + noise * 50.0 * (1.0 - exists[..., None])
    h_scrambled, _ = encode(scrambled, scrambled_block)
    report.invariants["nonexistent_atoms_scrambled"] = _relative_change(
        h_visible, h_scrambled, scored
    )

    # Padding. The invariant is that the *contents* of a padded row cannot
    # reach a real residue's features, so it is asked by holding the padding
    # present and changing only what is in it. Appending the rows at all does
    # move the last real residue -- an index-neighbour feature in FaMPNN's
    # encoder, unrelated to the padded coordinates -- and that is reported
    # separately as a diagnostic rather than as a leak.
    pad = 4
    padded_coords = torch.cat(
        [coords_af2, torch.randn(batch, pad, atom37.NUM_ATOM37, 3, device=device) * 50.0],
        dim=1,
    )
    padded_block = torch.cat(
        [block, torch.randn(batch, pad, block.shape[-2], 3, device=device) * 50.0], dim=1
    )
    padded_aatype = torch.cat(
        [aatype, torch.zeros(batch, pad, dtype=aatype.dtype, device=device)], dim=1
    )
    padded_mask = torch.cat([seq_mask, torch.zeros(batch, pad, device=device)], dim=1)
    padded_supplied = torch.cat(
        [supplied_atom_mask, torch.zeros(batch, pad, atom37.NUM_ATOM37, device=device)],
        dim=1,
    )
    padded_index = dict(index)
    for key, value in list(padded_index.items()):
        if value is not None:
            filler = (
                torch.arange(pad, device=device) + int(value.max()) + 1
                if key == "residue_index"
                else torch.zeros(pad, dtype=value.dtype, device=device)
            )
            padded_index[key] = torch.cat(
                [value, filler.reshape(1, pad).expand(batch, pad).to(value.dtype)], dim=1
            )

    def padded_encode(scale, seed):
        junk = torch.Generator().manual_seed(seed)
        coords_pad = padded_coords.clone()
        block_pad = padded_block.clone()
        coords_pad[:, length:] = (
            torch.randn(batch, pad, atom37.NUM_ATOM37, 3, generator=junk) * scale
        ).to(coords_pad.dtype)
        block_pad[:, length:] = (
            torch.randn(batch, pad, block.shape[-2], 3, generator=junk) * scale
        ).to(block_pad.dtype)
        h, _ = encode(
            coords_pad,
            block_pad,
            aa=padded_aatype,
            mask=padded_mask,
            supplied=padded_supplied,
            numbering=padded_index,
        )
        return h[:, :length]

    h_pad_a = padded_encode(1.0, 11)
    h_pad_b = padded_encode(50.0, 12)
    report.invariants["padded_atom_coordinates"] = _relative_change(
        h_pad_a, h_pad_b, scored
    )
    # Diagnostic: how much appending the rows at all moves things, and where.
    shift = (h_pad_a - h_visible).float().norm(dim=-1) / h_visible.float().norm(
        dim=-1
    ).clamp_min(1e-8)
    report.stats["terminus_shift_from_padding"] = _relative_change(
        h_visible, h_pad_a, scored
    )
    report.stats["terminus_shift_worst_residue"] = int(shift.reshape(-1).argmax())
    report.stats["terminus_shift_worst_value"] = float(shift.max())

    # Rigid motion of the complete input.
    moved_full, rotation, shift = _rigid_motion(full, generator=generator)
    h_rigid, _ = encode(moved_full, moved_full[..., rc.non_bb_idxs, :])
    report.invariants["rigid_rotation_translation"] = _relative_change(
        h_visible, h_rigid, scored
    )
    report.stats["rigid_determinant"] = float(torch.det(rotation))
    report.stats["rigid_shift_norm"] = float(shift.norm())
    return report


@torch.no_grad()
def packing_response(
    model,
    coords_af2,
    aatype,
    *,
    seq_mask=None,
    missing_atom_mask=None,
    residue_index=None,
    chain_index=None,
    delta_scale=1.0,
    num_steps=None,
    generator=None,
):
    """Does a residual on ``h_V`` change the packing it produces? (BB -> SC path.)

    The mirror question for the forward direction: an adapter writing into the
    encoder features is only useful if packing responds to it.
    """
    kwargs = dict(
        seq_mask=seq_mask,
        missing_atom_mask=missing_atom_mask,
        residue_index=residue_index,
        chain_index=chain_index,
    )
    _, h_base, features = iface.encode(model, coords_af2, aatype, **kwargs)
    pack_kwargs = dict(
        seq_mask=seq_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        num_steps=num_steps,
    )
    torch.manual_seed(0)
    reference, _ = iface.pack_from_features(model, features, aatype, **pack_kwargs)
    torch.manual_seed(0)
    repeat, _ = iface.pack_from_features(model, features, aatype, **pack_kwargs)
    delta = torch.randn(
        h_base.shape, generator=generator, dtype=h_base.dtype, device=h_base.device
    )
    delta = (
        delta
        * float(delta_scale)
        * h_base.norm(dim=-1, keepdim=True)
        / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    )
    torch.manual_seed(0)
    moved, _ = iface.pack_from_features(
        model, features, aatype, delta_h=delta, **pack_kwargs
    )

    def rmsd(a, b):
        return float((a - b).float().pow(2).sum(-1).mean().sqrt())

    return dict(
        floor_angstrom=rmsd(reference, repeat),
        response_angstrom=rmsd(reference, moved),
        delta_relative_norm=float(delta_scale),
    )
