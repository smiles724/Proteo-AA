"""Full-atom sequence design with FaMPNN: sequence *and* side chains at once.

This is upstream's other inference mode -- ``SeqDenoiser.sample``, the path behind
``fampnn/inference/seq_design.py`` -- as opposed to ``sidechain_pack``, which
takes the sequence as an input. Here FaMPNN iteratively unmasks residue
identities and packs their side chains, so a PXDesign backbone with no sequence
at all becomes a full-atom structure.

Defaults follow upstream's ``configs/seq_design.yaml`` and the preprint's
self-consistency protocol (Appendix E.2): the 0.3 A model, 100 iterative
unmasking steps, temperature 0.1, a final repack, and conditioning only on
previously generated side chains whose predicted error is under 0.3 A.

Positions whose identity is already known -- a binder-design target, say -- are
held fixed through ``aatype_override_mask``, so only the design positions get new
residues.
"""
from dataclasses import dataclass
from typing import Optional
import torch

from pxf import atom37, provenance
from pxf.sidechain.fampnn import FaMPNNSideChainPacker

# configs/seq_design.yaml, kept in one place so drift from upstream is visible.
DESIGN_VARIANT = "0.3"          # README: "Recommended for sequence design"
DEFAULT_SEQ_STEPS = 100         # Appendix E.2: 100 iterative unmasking steps
DEFAULT_TEMPERATURE = 0.1       # ProteinBench convention, used by the paper
DEFAULT_PSCE_THRESHOLD = 0.3    # condition only on side chains better than 0.3 A
DEFAULT_REPACK_LAST = True


@dataclass
class DesignResult:
    """One full-atom design produced from a backbone."""
    coords_af2: torch.Tensor      # [L, 37, 3]
    atom_mask_af2: torch.Tensor   # [L, 37]
    aatype: torch.Tensor          # [L]
    sequence: str
    psce: torch.Tensor            # [L, 33]
    seq_probs: Optional[torch.Tensor] = None
    fixed_mask: Optional[torch.Tensor] = None   # [L] True where identity was held

    def designed_sequence(self):
        if self.fixed_mask is None:
            return self.sequence
        return "".join(a for a, fixed in zip(self.sequence, self.fixed_mask.tolist())
                       if not fixed)


class FaMPNNFullAtomDesigner(FaMPNNSideChainPacker):
    """FaMPNN designing sequence and side chains together.

    Inherits the packer's strict loading, provenance record and input
    normalization; adds :meth:`design`. The inherited ``forward`` is still the
    packing path, and still refuses to let the model change a supplied sequence --
    that invariant belongs to packing, not to design.
    """

    def __init__(self, checkpoint=None, *, variant=DESIGN_VARIANT,
                 seq_steps=DEFAULT_SEQ_STEPS, temperature=DEFAULT_TEMPERATURE,
                 psce_threshold=DEFAULT_PSCE_THRESHOLD,
                 repack_last=DEFAULT_REPACK_LAST, seq_timestep_mode="linear",
                 **kwargs):
        super().__init__(checkpoint, variant=variant, **kwargs)
        self.seq_steps = int(seq_steps)
        self.temperature = float(temperature)
        self.psce_threshold = None if psce_threshold is None else float(psce_threshold)
        self.repack_last = bool(repack_last)
        self.seq_timestep_mode = str(seq_timestep_mode)
        self.identity = dict(self.identity)
        self.identity.update(
            mode="seq_design", designs_sequence=True,
            seq_steps=self.seq_steps, temperature=self.temperature,
            psce_threshold=self.psce_threshold, repack_last=self.repack_last,
            seq_timestep_mode=self.seq_timestep_mode)

    def _seq_timesteps(self, batch):
        from fampnn import sampling_utils
        steps = sampling_utils.get_timesteps_from_schedule(
            mode=self.seq_timestep_mode, num_steps=self.seq_steps,
            t_start=0.0, t_end=1.0)
        return steps[None].expand(batch, -1).to(self.device)

    @torch.no_grad()
    def design(self, *, coords_af2, atom_mask, aatype=None, seq_mask=None,
               residue_index=None, chain_index=None, fixed_sequence_mask=None,
               sidechain_context_mask=None, seed=None, batch_size=None,
               keep_input_backbone=True):
        """Design sequence and side chains for AF2-ordered backbone input.

        ``coords_af2`` is ``[B, L, 37, 3]``; ``atom_mask`` marks the slots the
        backbone module supplied. ``aatype`` supplies identities for positions
        held fixed by ``fixed_sequence_mask`` (both default to designing
        everything). ``sidechain_context_mask`` may only be a subset of the fixed
        positions -- upstream asserts that known side chains imply known sequence.
        """
        if seed is not None:
            torch.manual_seed(int(seed))
        if aatype is None:
            # Unknown identities enter as X; the sampler replaces them.
            length = coords_af2.shape[-3]
            lead = coords_af2.shape[0] if coords_af2.dim() == 4 else 1
            aatype = torch.full((lead, length), atom37.UNKNOWN_AA_INDEX, dtype=torch.long)
        coords, aatype, atom_mask, seq_mask, residue_index, chain_index, unbatched = \
            self._normalize_for_design(coords_af2, aatype, atom_mask, seq_mask,
                                       residue_index, chain_index)
        batch, length = aatype.shape

        fixed = (torch.zeros(batch, length, dtype=torch.long, device=self.device)
                 if fixed_sequence_mask is None
                 else self._as_batched(fixed_sequence_mask, "fixed_sequence_mask",
                                       batch, length).long().to(self.device))
        context = (torch.zeros_like(fixed) if sidechain_context_mask is None
                   else self._as_batched(sidechain_context_mask, "sidechain_context_mask",
                                         batch, length).long().to(self.device))
        if bool(((context - fixed) > 0).any()):
            raise ValueError(
                "sidechain_context_mask must be a subset of fixed_sequence_mask: "
                "a known side chain implies a known residue identity")
        # A position held fixed must actually carry an identity to hold.
        unknown_fixed = (fixed.bool() & (aatype >= atom37.UNKNOWN_AA_INDEX))
        if bool(unknown_fixed.any()):
            raise ValueError(
                f"{int(unknown_fixed.sum())} fixed position(s) have no residue identity; "
                "supply aatype there or leave them free to design")

        chunk = int(batch_size) if batch_size else batch
        coords_out, aatype_out, psce_out, probs_out = [], [], [], []
        for start in range(0, batch, chunk):
            piece = slice(start, min(start + chunk, batch))
            size = coords[piece].shape[0]
            missing = self.missing_atom_mask(aatype[piece].clamp_max(
                atom37.UNKNOWN_AA_INDEX), atom_mask[piece])
            designed, aatype_new, aux = self.model.sample(
                coords[piece], aatype=aatype[piece], seq_mask=seq_mask[piece],
                missing_atom_mask=missing, residue_index=residue_index[piece],
                chain_index=chain_index[piece],
                timesteps=self._seq_timesteps(size),
                temperature=self.temperature, seq_only=False,
                repack_last=self.repack_last, psce_threshold=self.psce_threshold,
                aatype_override_mask=fixed[piece], scn_override_mask=context[piece],
                scd_inputs=self._scd_inputs(size))
            coords_out.append(designed)
            aatype_out.append(aatype_new)
            psce_out.append(aux["psce"])
            probs_out.append(aux.get("seq_probs"))

        designed = torch.cat(coords_out, 0)
        aatype_new = torch.cat(aatype_out, 0)
        psce = torch.cat(psce_out, 0)
        probs = torch.cat(probs_out, 0) if probs_out[0] is not None else None

        # Fixed positions must come back unchanged, or the mask did not hold.
        if bool(fixed.any()):
            held = fixed.bool()
            if not torch.equal(aatype_new[held].long(), aatype[held].long()):
                changed = int((aatype_new[held].long() != aatype[held].long()).sum())
                raise ValueError(f"FaMPNN changed {changed} position(s) marked fixed")

        shift = self._backbone_shift(coords, designed, atom_mask)
        if keep_input_backbone:
            designed = self._restore_backbone(designed, coords, atom_mask)
        results = [
            DesignResult(coords_af2=designed[i], atom_mask_af2=self._output_atom_mask(
                             aatype_new[i][None], atom_mask[i][None])[0],
                         aatype=aatype_new[i],
                         sequence=atom37.sequence_from_aatype(aatype_new[i]),
                         psce=psce[i],
                         seq_probs=None if probs is None else probs[i],
                         fixed_mask=fixed[i].bool())
            for i in range(batch)]
        return dict(designs=results, backbone_shift=shift, unbatched=unbatched)

    def _normalize_for_design(self, coords, aatype, atom_mask, seq_mask,
                              residue_index, chain_index):
        """Like the packer's normalizer, but tolerant of unknown (X) identities."""
        unbatched = coords.dim() == 3
        if unbatched:
            coords = coords.unsqueeze(0)
        if coords.dim() != 4 or coords.shape[-2:] != (atom37.NUM_ATOM37, 3):
            raise ValueError(f"Expected [B, L, 37, 3] coordinates, got {tuple(coords.shape)}")
        batch, length = coords.shape[0], coords.shape[1]
        aatype = self._as_batched(aatype, "aatype", batch, length, dtype=torch.long)
        if int(aatype.max()) > atom37.UNKNOWN_AA_INDEX:
            raise ValueError("aatype contains indices beyond the unknown token")
        atom_mask = self._as_batched(atom_mask, "atom_mask", batch, length,
                                     trailing=(atom37.NUM_ATOM37,))
        seq_mask = (torch.ones(batch, length) if seq_mask is None
                    else self._as_batched(seq_mask, "seq_mask", batch, length))
        residue_index = (torch.arange(length).expand(batch, length) if residue_index is None
                         else self._as_batched(residue_index, "residue_index", batch,
                                               length, dtype=torch.long))
        chain_index = (torch.zeros(batch, length, dtype=torch.long) if chain_index is None
                       else self._as_batched(chain_index, "chain_index", batch, length,
                                             dtype=torch.long))
        move = lambda t: t.contiguous().to(self.device)
        return (move(coords.float()), move(aatype), move(atom_mask), move(seq_mask),
                move(residue_index), move(chain_index), unbatched)
