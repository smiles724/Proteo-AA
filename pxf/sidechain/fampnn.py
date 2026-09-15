"""Full official FaMPNN as the side-chain module, in pure packing mode.

FaMPNN can co-design sequence and side chains; this pipeline deliberately does
not use that. The sequence is an **input**: every position is handed to the model
through ``aatype_override_mask``, so the network's only remaining job is to infer
side-chain conformations for the residues it was told about. This is upstream's
own ``sidechain_pack`` path -- the one behind ``fampnn/inference/pack.py`` -- with
no reimplemented layers.

Two invariants are enforced rather than trusted:

* **The sequence is never designed.** ``sidechain_pack`` echoes back the aatype it
  used; that echo is compared against the input, so any drift into sequence
  design is caught instead of silently changing the design.
* **The backbone is never moved.** Only the 33 side-chain slots may change; the
  backbone is restored from the input and the deviation is reported.
"""
from pathlib import Path
import torch
from torch import nn

from pxf import atom37, provenance

# Upstream configs/pack.yaml defaults, kept in one place so drift is visible.
DEFAULT_DIFFUSION = dict(num_steps=50, step_scale=1.5, timestep_mode="linear",
                         t_start=0.0, t_end=1.0)
DEFAULT_CHURN = dict(s_churn=0, s_noise=1.0, s_t_min=0.01, s_t_max=50.0)


class FaMPNNSideChainPacker(nn.Module):
    """Published FaMPNN, frozen, packing side chains onto a supplied sequence."""

    def __init__(self, checkpoint=None, *, variant=provenance.DEFAULT_FAMPNN_WEIGHTS,
                 num_steps=None, step_scale=None, timestep_mode=None,
                 t_start=None, t_end=None, churn=None, strict_sources=True):
        super().__init__()
        self.source = provenance.component_record("fampnn", strict=strict_sources)
        rc = atom37.assert_upstream_mapping()
        self._rc = rc
        self.variant = variant
        path = Path(checkpoint) if checkpoint else provenance.fampnn_checkpoint(variant)
        if not path.is_file():
            raise ValueError(f"FaMPNN checkpoint not found: {path}")
        self.checkpoint_path = path

        from fampnn.model.sd_model import SeqDenoiser
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("state_dict", "model_cfg"):
            if key not in bundle:
                raise ValueError(f"{path} is not a FaMPNN checkpoint (missing {key!r})")
        self.model = SeqDenoiser(bundle["model_cfg"])
        # Strict: a silently partial load would leave randomly initialized tensors
        # in a model that still emits plausible-looking coordinates.
        self.model.load_state_dict(bundle["state_dict"], strict=True)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model_cfg = bundle["model_cfg"]

        self.diffusion = dict(DEFAULT_DIFFUSION)
        for key, value in dict(num_steps=num_steps, step_scale=step_scale,
                               timestep_mode=timestep_mode, t_start=t_start,
                               t_end=t_end).items():
            if value is not None:
                self.diffusion[key] = value
        self.churn = dict(DEFAULT_CHURN, **(churn or {}))

        from omegaconf import OmegaConf
        self.identity = dict(
            backend="fampnn", mode="sidechain_pack", designs_sequence=False,
            variant=variant, weights=provenance.weight_record(path, variant=variant),
            upstream=self.source, diffusion=dict(self.diffusion),
            churn=dict(self.churn), stochastic=True,
            atom_mapping=atom37.mapping_record(),
            model_config=OmegaConf.to_container(self.model_cfg, resolve=True)
            if OmegaConf.is_config(self.model_cfg) else dict(self.model_cfg))

    @property
    def device(self):
        return next(self.model.parameters()).device

    # ---- input assembly ---------------------------------------------------

    def ghost_atom_mask(self, aatype):
        """``1`` where an atom37 slot does not exist for that residue type."""
        table = torch.as_tensor(self._rc.restype_atom37_mask, device=aatype.device)
        return 1.0 - table[aatype.long()]

    def missing_atom_mask(self, aatype, atom_mask):
        """Upstream's formula: an atom is missing if it should exist but is absent."""
        ghost = self.ghost_atom_mask(aatype)
        return (1.0 - atom_mask.float()) * (1.0 - ghost)

    def _timesteps(self, batch):
        from fampnn import sampling_utils
        steps = sampling_utils.get_timesteps_from_schedule(
            mode=self.diffusion["timestep_mode"], num_steps=self.diffusion["num_steps"],
            t_start=self.diffusion["t_start"], t_end=self.diffusion["t_end"])
        return steps[None].expand(batch, -1).to(self.device)

    def _scd_inputs(self, batch):
        return {"num_steps": self.diffusion["num_steps"],
                "timesteps": self._timesteps(batch),
                "step_scale": self.diffusion["step_scale"],
                "churn_cfg": dict(self.churn, num_steps=self.diffusion["num_steps"])}

    # ---- packing ----------------------------------------------------------

    @torch.no_grad()
    def forward(self, *, coords_af2, aatype, atom_mask, seq_mask=None,
                residue_index=None, chain_index=None, scn_context_mask=None,
                batch_size=None, keep_input_backbone=True, seed=None):
        """Pack side chains onto ``coords_af2`` for the supplied ``aatype``.

        ``coords_af2`` is ``[B, L, 37, 3]`` in the shared AF2 atom37 order and
        ``atom_mask`` ``[B, L, 37]`` marks which slots the backbone module
        supplied. ``aatype`` ``[B, L]`` is the sequence to pack -- required, and
        returned unchanged. ``scn_context_mask`` optionally marks positions whose
        *input* side chains should be kept as context instead of repacked.

        Packing is a sampling procedure: repeated calls on one backbone yield
        different rotamers (order 0.5 A RMS apart), which is what makes several
        packings per backbone worth generating.

        ``seed`` makes a run reproducible, but *bitwise* equality only holds on
        CPU or under ``torch.use_deterministic_algorithms(True)``. On CUDA the
        default kernels leave about 1e-5 A of run-to-run jitter, which is
        nondeterminism in the kernels, not in the sampling -- for scale, a
        different seed moves atoms by several Angstrom. ``batch_size`` also
        changes how noise is drawn, so reproducing a run means fixing both.
        """
        if seed is not None:
            torch.manual_seed(int(seed))
        (coords_af2, aatype, atom_mask, seq_mask, residue_index, chain_index,
         unbatched) = self._normalize(coords_af2, aatype, atom_mask, seq_mask,
                                      residue_index, chain_index)
        batch, length = aatype.shape
        if scn_context_mask is None:
            context = torch.zeros_like(seq_mask)
        else:
            context = self._as_batched(scn_context_mask, "scn_context_mask",
                                       batch, length).to(self.device)

        chunk = int(batch_size) if batch_size else batch
        coords_out, psce_out = [], []
        for start in range(0, batch, chunk):
            stop = min(start + chunk, batch)
            piece = slice(start, stop)
            size = stop - start
            missing = self.missing_atom_mask(aatype[piece], atom_mask[piece])
            # Chunking changes how noise is drawn, so a seeded run must fix the
            # grouping too; batch_size is recorded in the identity for that reason.
            packed, echoed, aux = self.model.sidechain_pack(
                coords_af2[piece], aatype[piece],
                seq_mask=seq_mask[piece], missing_atom_mask=missing,
                residue_index=residue_index[piece], chain_index=chain_index[piece],
                # Every position's identity is supplied: the model designs nothing.
                aatype_override_mask=seq_mask[piece].long(),
                scn_override_mask=context[piece].long(),
                scd_inputs=self._scd_inputs(size))
            self._assert_sequence_untouched(aatype[piece], echoed, seq_mask[piece])
            coords_out.append(packed)
            psce_out.append(aux["psce"])

        packed = torch.cat(coords_out, dim=0)
        psce = torch.cat(psce_out, dim=0)
        backbone_shift = self._backbone_shift(coords_af2, packed, atom_mask)
        if keep_input_backbone:
            packed = self._restore_backbone(packed, coords_af2, atom_mask)
        result = dict(coords_af2=packed, aatype=aatype, psce=psce,
                      atom_mask_af2=self._output_atom_mask(aatype, atom_mask),
                      backbone_shift=backbone_shift,
                      sequence=[atom37.sequence_from_aatype(row) for row in aatype])
        if unbatched:
            # Mirror the caller's layout: a single structure in, a single one out.
            for key in ("coords_af2", "aatype", "psce", "atom_mask_af2"):
                result[key] = result[key].squeeze(0)
        return result

    @staticmethod
    def _as_batched(value, name, batch, length, *, trailing=(), dtype=torch.float32):
        """Reshape ``value`` to ``[batch, length, *trailing]``, broadcasting one row.

        The element count is checked before any reshape so a mismatched input
        produces an explanatory error instead of a bare reshape failure.
        """
        tensor = torch.as_tensor(value)
        row = length
        for size in trailing:
            row *= size
        rows = tensor.numel() // row if row and tensor.numel() % row == 0 else None
        if rows not in (1, batch):
            raise ValueError(
                f"{name} with shape {tuple(tensor.shape)} does not match coordinates "
                f"[{batch}, {length}{''.join(f', {s}' for s in trailing)}]")
        tensor = tensor.reshape(rows, length, *trailing)
        if rows == 1 and batch > 1:
            tensor = tensor.expand(batch, length, *trailing)
        return tensor.to(dtype)

    def _normalize(self, coords, aatype, atom_mask, seq_mask, residue_index, chain_index):
        """Bring every input to ``[B, ...]``; also report whether B was implicit."""
        unbatched = coords.dim() == 3
        if unbatched:
            coords = coords.unsqueeze(0)
        if coords.dim() != 4 or coords.shape[-2:] != (atom37.NUM_ATOM37, 3):
            raise ValueError(f"Expected [B, L, 37, 3] coordinates, got {tuple(coords.shape)}")
        batch, length = coords.shape[0], coords.shape[1]

        aatype = self._as_batched(aatype, "aatype", batch, length, dtype=torch.long)
        if int(aatype.max()) >= atom37.UNKNOWN_AA_INDEX:
            offenders = (aatype >= atom37.UNKNOWN_AA_INDEX).nonzero()[:8].tolist()
            raise ValueError(
                "aatype contains unknown residues at (batch, position) "
                f"{offenders}. FaMPNN packs a *given* sequence and will not design one; "
                "supply a canonical identity at every position.")
        atom_mask = self._as_batched(atom_mask, "atom_mask", batch, length,
                                     trailing=(atom37.NUM_ATOM37,))
        seq_mask = (torch.ones(batch, length) if seq_mask is None
                    else self._as_batched(seq_mask, "seq_mask", batch, length))
        residue_index = (torch.arange(length).expand(batch, length) if residue_index is None
                         else self._as_batched(residue_index, "residue_index", batch, length,
                                               dtype=torch.long))
        chain_index = (torch.zeros(batch, length, dtype=torch.long) if chain_index is None
                       else self._as_batched(chain_index, "chain_index", batch, length,
                                             dtype=torch.long))
        move = lambda t: t.contiguous().to(self.device)
        return (move(coords.float()), move(aatype), move(atom_mask), move(seq_mask),
                move(residue_index), move(chain_index), unbatched)

    @staticmethod
    def _assert_sequence_untouched(supplied, echoed, seq_mask):
        """The packing path must return the sequence it was given, position for position."""
        real = seq_mask.bool()
        if not torch.equal(supplied[real].long(), echoed[real].long()):
            changed = int((supplied[real].long() != echoed[real].long()).sum())
            raise ValueError(
                f"FaMPNN altered the sequence at {changed} position(s); this pipeline packs "
                "a given sequence and must not design one. Refusing the result.")

    @staticmethod
    def _backbone_shift(before, after, atom_mask):
        slots = list(atom37.BACKBONE_SLOTS)
        present = atom_mask[..., slots].bool()
        if not bool(present.any()):
            return None
        delta = (before[..., slots, :] - after[..., slots, :])[present]
        return float(torch.sqrt((delta ** 2).sum(-1).mean()))

    @staticmethod
    def _restore_backbone(packed, original, atom_mask):
        slots = list(atom37.BACKBONE_SLOTS)
        present = atom_mask[..., slots].bool()
        packed = packed.clone()
        packed[..., slots, :] = torch.where(present[..., None], original[..., slots, :],
                                            packed[..., slots, :])
        return packed

    def _output_atom_mask(self, aatype, atom_mask):
        """Backbone slots the generator supplied, plus every real slot for the sequence."""
        exists = 1.0 - self.ghost_atom_mask(aatype)
        mask = exists.clone()
        slots = list(atom37.BACKBONE_SLOTS)
        mask[..., slots] = torch.maximum(mask[..., slots], atom_mask[..., slots])
        return mask.bool()
