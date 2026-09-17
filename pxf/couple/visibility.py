"""Which atoms the re-encoder may *see*, kept separate from which it may be *scored* on.

The two are different masks and conflating them breaks the coupling contract in
opposite directions.

**Observation** (``missing_atom_mask`` as the converter builds it) answers "was
this atom supplied in the input?". For a backbone-only PXDesign proposal the
answer for every side-chain slot is no, so the converter marks all 33 of them
missing -- correctly, because at that point they do not exist.

**Availability after packing** answers "does this atom now have a usable
coordinate?". Once FaMPNN has generated the side chains the answer flips for
every slot the residue type has. Re-using the observation mask there is what
:func:`pxf.couple.fampnn_iface.build_atom_mask` turns into

    mask = exists * seq_mask * (1 - missing_atom_mask) * sidechain_visible

so a generated atom stays masked *even with* ``sidechain_visible = 1``: the
``(1 - missing_atom_mask)`` factor is zero. The re-encode then sees the same
backbone-only structure it started from, ``h_packed == h_base`` up to float
noise, and the whole SC -> BB path carries nothing -- while every shape checks
out and no error is raised.

So availability is computed here, from scratch, and never derived from the
observation mask:

* backbone slots keep their *actual* availability -- what PXDesign supplied, with
  a finite coordinate. A proposal that dropped an atom must not become one that
  has it.
* side-chain slots are available where the atom exists for the fixed sequence,
  the packer produced a finite coordinate, and the residue's own frame is valid.
* padding (``seq_mask = 0``), slots the residue type does not have, and residues
  whose N/CA/C frame is unusable are excluded everywhere.

Native observation masks stay out of this entirely. They are the supervision
signal -- which native atoms exist to compare against -- and letting them reach
the encoder input would leak the target into the feature the adapter reads.
"""

from dataclasses import dataclass

import torch

from pxf import atom37

# N, CA, C: the three atoms the residue frame is built from. Distinct from
# BACKBONE_SLOTS, which also holds O -- O is not needed for a frame, so a
# residue missing only its O still packs and still re-encodes.
FRAME_SLOTS = (0, 1, 2)


def _rc(rc=None):
    if rc is not None:
        return rc
    from fampnn.data import residue_constants as rc

    return rc


def atom_exists(aatype, *, rc=None, device=None):
    """``[..., 37]`` 1 where that atom37 slot exists for the residue type.

    Unknown residues (aatype 20, and anything above it after clamping) get the X
    row, which has the backbone only -- so an unresolved identity can still
    contribute a frame but never a side chain.
    """
    rc = _rc(rc)
    table = torch.as_tensor(
        rc.STANDARD_ATOM_MASK_WITH_X,
        dtype=torch.float32,
        device=device if device is not None else aatype.device,
    )
    return table[aatype.clamp(min=0, max=atom37.UNKNOWN_AA_INDEX).long()]


def finite(coords):
    """``[...]`` true where every component of a coordinate is finite."""
    return torch.isfinite(coords).all(dim=-1)


@dataclass
class Visibility:
    """Per-atom availability for one re-encoding, plus what it was derived from.

    ``available`` is the authoritative mask: 1 exactly where the encoder may use
    the coordinate in ``coords_af2``. ``missing_atom_mask`` is the same fact in
    FaMPNN's convention, provided so the legacy ``encode`` path can be driven
    from it; ``atom_availability=`` is the direct route and the one the
    controller uses.
    """

    available: torch.Tensor  # [B, L, 37] float, 1 = usable coordinate
    missing_atom_mask: torch.Tensor  # [B, L, 37] float, 1 = exists but unusable
    frame_valid: torch.Tensor  # [B, L] bool
    sidechain_visible: torch.Tensor  # [B, L] float, 1 = at least one SC atom usable
    exists: torch.Tensor  # [B, L, 37] float, residue-type atom set
    stats: dict

    @property
    def backbone_available(self):
        return self.available[..., list(atom37.BACKBONE_SLOTS)]

    @property
    def sidechain_available(self):
        return self.available[..., list(atom37.SIDECHAIN_SLOTS)]

    def record(self):
        """JSON-safe summary, for a run's diagnostics."""
        return dict(self.stats)


def predicted_availability(
    aatype,
    seq_mask,
    supplied_atom_mask,
    coords_af2,
    *,
    sidechains=None,
    rc=None,
):
    """Availability after packing: the mask the full-atom re-encode must use.

    Args:
        aatype: ``[B, L]`` the fixed residue identities the packer built for.
        seq_mask: ``[B, L]`` 1 for real residues, 0 for padding.
        supplied_atom_mask: ``[B, L, 37]`` what the *input* actually carried --
            the converter's ``atom_mask``. Only its backbone columns are read;
            its side-chain columns are zero by construction for a backbone-only
            proposal and must not be allowed to veto generated atoms.
        coords_af2: ``[B, L, 37, 3]`` the backbone the packing sits on.
        sidechains: ``[B, L, 33, 3]`` (or a full ``[B, L, 37, 3]``) generated
            side chains. ``None`` means none were generated, which yields the
            backbone-only availability -- the correct mask for the first encode
            and for the side-chain-masked control.

    Returns:
        :class:`Visibility`.
    """
    rc = _rc(rc)
    device = coords_af2.device
    aatype = aatype.long()
    seq_mask = seq_mask.float()
    backbone = list(atom37.BACKBONE_SLOTS)
    sidechain = list(atom37.SIDECHAIN_SLOTS)

    exists = atom_exists(aatype, rc=rc, device=device)
    # A slot that the residue type does not have, or that belongs to padding, is
    # not available for any reason and cannot be made so.
    real = exists * seq_mask[..., None]

    supplied = supplied_atom_mask.to(device=device, dtype=torch.float32)
    bb_finite = finite(coords_af2[..., backbone, :]).float()
    bb_available = real[..., backbone] * supplied[..., backbone] * bb_finite

    # The frame is what every downstream geometric feature is expressed in, and
    # a side chain hanging off an unusable frame is not evidence about anything.
    frame_slots = [backbone.index(slot) for slot in FRAME_SLOTS]
    frame_ok = bb_available[..., frame_slots].prod(dim=-1) > 0

    if sidechains is None:
        sc_available = torch.zeros_like(real[..., sidechain])
    else:
        block = sidechains
        if block.shape[-2] == atom37.NUM_ATOM37:
            block = block[..., sidechain, :]
        if block.shape[-2] != len(sidechain):
            raise ValueError(
                f"expected {len(sidechain)} side-chain slots, got {block.shape[-2]}"
            )
        sc_available = (
            real[..., sidechain] * finite(block).float() * frame_ok[..., None].float()
        )

    available = torch.zeros_like(real)
    available[..., backbone] = bb_available
    available[..., sidechain] = sc_available
    missing = (real * (1.0 - available)).clamp(0.0, 1.0)

    n_real = real.sum().clamp_min(1.0)
    stats = dict(
        residues=int(seq_mask.sum()),
        frames_valid=int(frame_ok.sum()),
        frames_invalid=int((seq_mask > 0).sum() - frame_ok.sum()),
        atoms_expected=int(real.sum()),
        atoms_available=int(available.sum()),
        sidechain_atoms_expected=int(real[..., sidechain].sum()),
        sidechain_atoms_available=int(sc_available.sum()),
        available_fraction=float(available.sum() / n_real),
        sidechains_supplied=sidechains is not None,
    )
    return Visibility(
        available=available,
        missing_atom_mask=missing,
        frame_valid=frame_ok,
        sidechain_visible=(sc_available.sum(dim=-1) > 0).float(),
        exists=exists,
        stats=stats,
    )


def native_observation_mask(aatype, seq_mask, missing_atom_mask, *, rc=None):
    """Which *native* atoms exist and were observed -- the supervision mask.

    Kept in this module next to :func:`predicted_availability` so the contrast is
    visible: this one may be used to decide what a loss or a metric scores, and
    never to decide what the encoder sees.
    """
    exists = atom_exists(aatype, rc=rc, device=missing_atom_mask.device)
    return (exists * seq_mask[..., None].float() * (1.0 - missing_atom_mask.float())).clamp(
        0.0, 1.0
    )


@dataclass
class PackedStructure:
    """The full-atom state one corrective event reads: ``bb0 + sc0``, re-encoded.

    Everything the feedback readout is allowed to see, gathered in one object so
    the boundary is checkable. Note what is *absent*: no native coordinates, no
    native side-chain error, no residual backbone error. Those are supervision,
    and a readout that could reach them would report a gain it cannot reproduce
    at inference.

    ``h_packed`` is FaMPNN's final node readout for the re-encoded structure --
    the encoder's own invariant summary -- rather than raw geometric vectors.
    """

    h_packed: torch.Tensor  # [B, L, c_h_V]
    coords37: torch.Tensor  # [B, L, 37, 3]  bb0 with sc0 written into the SC slots
    aatype: torch.Tensor  # [B, L]  the fixed sequence
    seq_mask: torch.Tensor  # [B, L]
    visibility: Visibility
    psce: torch.Tensor | None = None  # [B, L, 33] predicted per-atom SC error, A
    h_base: torch.Tensor | None = None  # [B, L, c_h_V] side-chain-masked encoding

    @property
    def available(self):
        return self.visibility.available

    @property
    def frame_valid(self):
        return self.visibility.frame_valid

    @property
    def valid_residues(self):
        """``[B, L]`` float: residues a feedback residual may be written to."""
        return (self.seq_mask > 0).float() * self.frame_valid.float()

    def detach(self):
        """The same state with every tensor cut from the graph.

        The pilot's upstream half is frozen and its gradient is not wanted, so
        detaching is explicit here rather than relying on a ``no_grad`` context
        that a later refactor could widen.
        """
        from dataclasses import replace

        def cut(value):
            return value.detach() if torch.is_tensor(value) else value

        return replace(
            self,
            h_packed=cut(self.h_packed),
            coords37=cut(self.coords37),
            psce=cut(self.psce),
            h_base=cut(self.h_base),
        )
