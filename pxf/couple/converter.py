"""The representation conversion layer between PXDesign and FaMPNN.

A first-class component rather than conversions scattered through the
controller, because the two modules disagree on more than atom order:

======================  ===========================  =========================
concern                 PXDesign / Protenix          FaMPNN
======================  ===========================  =========================
atom layout             flat ``[N_atom, 3]`` plus     dense ``[L, 37, 3]``
                        ``atom_to_token_idx``
side-chain block        interleaved in the flat list  ``rc.non_bb_idxs`` (33)
backbone slots          atom37 ``(0, 1, 2, 4)``       same, ``rc.bb_idxs``
residue identity        ``xpb`` for design tokens     integer aatype, X = 20
chain id                ``asym_id`` (integer)         ``chain_index``
residue numbering       ``residue_index``             ``residue_index``
padding                 no padding (ragged)           ``seq_mask``
missing atoms           absent from the flat list     ``missing_atom_mask``
======================  ===========================  =========================

The atom37 *ordering* happens to agree (see :mod:`pxf.atom37`), which is checked
rather than assumed. What does not agree is everything else in that table, and
each row is a place where a silent mistake produces plausible-looking
coordinates. So the conversion is done once, here, and the controller consumes
only its output.
"""

from dataclasses import dataclass, field

import torch

from pxf import atom37, bridge


@dataclass
class CoupledInputs:
    """Everything both modules need for one structure, in FaMPNN's conventions."""

    coords_af2: torch.Tensor  # [B, L, 37, 3]
    atom_mask: torch.Tensor  # [B, L, 37]  1 where an atom was supplied
    aatype: torch.Tensor  # [B, L]      X (=20) where undetermined
    seq_mask: torch.Tensor  # [B, L]      1 for real residues
    missing_atom_mask: torch.Tensor  # [B, L, 37]  1 where an atom should exist but does not
    residue_index: torch.Tensor  # [B, L]
    chain_index: torch.Tensor  # [B, L]
    design_mask: torch.Tensor  # [B, L] bool, PXDesign's design tokens
    sequence_known: torch.Tensor  # [B, L] bool, identity available
    num_tokens: int = 0
    dropped_atoms: list = field(default_factory=list)

    @property
    def batch(self):
        return int(self.coords_af2.shape[0])

    @property
    def length(self):
        return int(self.coords_af2.shape[1])

    def fampnn_kwargs(self):
        """The keyword set FaMPNN's packer and designer both take."""
        return dict(
            coords_af2=self.coords_af2,
            atom_mask=self.atom_mask,
            seq_mask=self.seq_mask,
            residue_index=self.residue_index,
            chain_index=self.chain_index,
        )


class PXFaRepresentationConverter:
    """Owns every PXDesign <-> FaMPNN representation mapping.

    Stateless with respect to coordinates; it holds only the conventions, so one
    instance serves a whole run and can be interrogated in tests.
    """

    def __init__(self, *, verify_contract=True):
        self._rc = atom37.assert_upstream_mapping() if verify_contract else None

    # ---- constants -------------------------------------------------------

    @property
    def rc(self):
        if self._rc is None:
            from fampnn.data import residue_constants as rc

            self._rc = rc
        return self._rc

    @property
    def backbone_slots(self):
        return list(atom37.BACKBONE_SLOTS)

    @property
    def sidechain_slots(self):
        return list(atom37.SIDECHAIN_SLOTS)

    # ---- PXDesign -> FaMPNN ---------------------------------------------

    def px_backbone_to_fampnn(
        self,
        coords,
        atom_names,
        atom_to_token_idx,
        num_tokens,
        *,
        res_names=None,
        residue_index=None,
        chain_index=None,
        aatype=None,
        strict=False,
    ):
        """Densify PXDesign's flat atom output into FaMPNN's per-residue block.

        ``coords`` is ``[..., N_atom, 3]``; leading dimensions (PXDesign's
        ``N_sample``) are preserved and folded into the batch axis. Returns
        :class:`CoupledInputs`.
        """
        dense, mask, dropped = bridge.atoms_to_atom37(
            coords, atom_names, atom_to_token_idx, num_tokens, strict=strict
        )
        if dense.dim() == 3:
            dense, mask = dense[None], mask[None]
        dense = dense.reshape(-1, num_tokens, atom37.NUM_ATOM37, 3)
        mask = mask.reshape(-1, num_tokens, atom37.NUM_ATOM37).float()
        batch = dense.shape[0]

        if res_names is not None:
            design = bridge.design_mask_from_res_names(
                res_names, atom_to_token_idx, num_tokens
            )
            sequence, known = bridge.native_sequence(
                res_names, atom_to_token_idx, num_tokens
            )
            derived = atom37.aatype_from_sequence(sequence, allow_unknown=True)
        else:
            design = torch.zeros(num_tokens, dtype=torch.bool)
            known = torch.zeros(num_tokens, dtype=torch.bool)
            derived = torch.full((num_tokens,), atom37.UNKNOWN_AA_INDEX, dtype=torch.long)
        aatype = derived if aatype is None else torch.as_tensor(aatype).long().reshape(-1)

        residue_index, chain_index = self.map_chain_and_residue_indices(
            num_tokens, residue_index=residue_index, chain_index=chain_index
        )

        # One device normalization at the boundary rather than a device= on every
        # constructor above. The per-token annotations are all derived from Python
        # lists or from the topology, so several of them land on the CPU whatever
        # the coordinates are on; `dense` is authoritative because it came from
        # the backbone module's own output.
        device = dense.device
        design, known = design.to(device), known.to(device)
        aatype = aatype.to(device)
        residue_index = residue_index.to(device)
        chain_index = chain_index.to(device)

        def expand(tensor):
            return tensor.reshape(1, num_tokens).expand(batch, num_tokens).contiguous()

        seq_mask = torch.ones(batch, num_tokens, device=device)
        return CoupledInputs(
            coords_af2=dense,
            atom_mask=mask,
            aatype=expand(aatype),
            seq_mask=seq_mask,
            missing_atom_mask=self.missing_atom_mask(expand(aatype), mask),
            residue_index=expand(residue_index),
            chain_index=expand(chain_index),
            design_mask=design,
            sequence_known=known,
            num_tokens=int(num_tokens),
            dropped_atoms=dropped,
        )

    def px_residue_mask_to_fampnn(self, mask, num_tokens, *, batch=1, dtype=torch.float32):
        """Broadcast a per-token PXDesign mask to FaMPNN's ``[B, L]`` convention."""
        tensor = torch.as_tensor(mask).reshape(-1)
        if tensor.numel() != num_tokens:
            raise ValueError(f"mask has {tensor.numel()} entries for {num_tokens} tokens")
        return (
            tensor.reshape(1, num_tokens).expand(batch, num_tokens).to(dtype).contiguous()
        )

    def map_chain_and_residue_indices(
        self, num_tokens, *, residue_index=None, chain_index=None
    ):
        """Normalize numbering to ``[L]`` longs, defaulting to a single chain.

        PXDesign's ``asym_id`` may be non-contiguous after cropping; FaMPNN only
        needs chains to be distinguishable, so ids are compacted to 0..n-1 while
        preserving grouping and order.

        Device-agnostic on purpose: the chain compaction round-trips through a
        Python dict, so the result is a CPU tensor regardless of the input.
        ``px_backbone_to_fampnn`` moves it onto the coordinates' device.
        """
        if residue_index is None:
            residue = torch.arange(num_tokens, dtype=torch.long)
        else:
            residue = torch.as_tensor(residue_index).reshape(-1)[:num_tokens].long()
        if chain_index is None:
            chain = torch.zeros(num_tokens, dtype=torch.long)
        else:
            raw = torch.as_tensor(chain_index).reshape(-1)[:num_tokens]
            order = {
                value: position
                for position, value in enumerate(dict.fromkeys(raw.tolist()))
            }
            chain = torch.tensor([order[value] for value in raw.tolist()], dtype=torch.long)
        if residue.numel() != num_tokens or chain.numel() != num_tokens:
            raise ValueError("numbering does not cover every token")
        return residue, chain

    def missing_atom_mask(self, aatype, atom_mask):
        """FaMPNN's convention: 1 where an atom should exist but was not supplied."""
        from fampnn.data.data import get_rc_tensor

        exists = get_rc_tensor(
            self.rc.STANDARD_ATOM_MASK_WITH_X,
            aatype.clamp_max(atom37.UNKNOWN_AA_INDEX).long(),
        )
        return (exists * (1.0 - atom_mask.float())).clamp(0.0, 1.0)

    # ---- FaMPNN -> PXDesign ---------------------------------------------

    def fampnn_sidechains_to_px(self, sidechains, coords_af2=None):
        """Place FaMPNN's 33-slot side-chain block back into atom37.

        Accepts either the bare ``[..., 33, 3]`` block or a full
        ``[..., 37, 3]`` tensor. When ``coords_af2`` is given its backbone is
        preserved, which is the assembly policy: PXDesign owns the backbone.
        """
        block = sidechains
        if block.shape[-2] == atom37.NUM_ATOM37:
            block = block[..., self.sidechain_slots, :]
        if block.shape[-2] != len(self.sidechain_slots):
            raise ValueError(
                f"expected {len(self.sidechain_slots)} side-chain atoms, "
                f"got {block.shape[-2]}"
            )
        if coords_af2 is None:
            out = block.new_zeros(*block.shape[:-2], atom37.NUM_ATOM37, 3)
        else:
            out = coords_af2.clone()
        out[..., self.sidechain_slots, :] = block.to(out.dtype)
        return out

    def scatter_to_px_atoms(self, coords_af2, atom_names, atom_to_token_idx):
        """Gather a dense atom37 tensor back onto PXDesign's flat atom axis.

        The inverse of :meth:`px_backbone_to_fampnn`'s densification, for writing
        coupled output through PXDesign's own structure writers. Atoms outside the
        atom37 vocabulary keep whatever the caller had (they were never densified).
        """
        lookup = {name: slot for slot, name in enumerate(atom37.ATOM37)}
        slots = torch.tensor(
            [lookup.get(str(name), -1) for name in atom_names], dtype=torch.long
        )
        token = torch.as_tensor(atom_to_token_idx, dtype=torch.long).reshape(-1)
        if slots.numel() != token.numel():
            raise ValueError(f"{slots.numel()} atom names for {token.numel()} atoms")
        valid = slots >= 0
        flat = coords_af2.reshape(-1, atom37.NUM_ATOM37, 3)
        out = coords_af2.new_zeros(token.numel(), 3)
        index = torch.nonzero(valid, as_tuple=True)[0]
        out[index] = flat[token[index], slots[index]]
        return out, valid

    def identity(self):
        return dict(
            component="PXFaRepresentationConverter",
            atom_mapping=atom37.mapping_record(),
            backbone_slots=self.backbone_slots,
            n_sidechain_slots=len(self.sidechain_slots),
        )
