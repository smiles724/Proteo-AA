"""Compose the two modules into one full-atom pipeline.

Division of labour:

* **Backbone** -- official PXDesign owns every backbone atom (N, CA, C, O).
* **Side chain** -- official FaMPNN owns every side-chain atom, packed onto a
  sequence that is **given to it**, never designed by it.

That second point is the whole point of this pipeline, so the sequence has to
come from somewhere explicit. Non-design residues supply their native identity
from PXDesign's own input structure; design tokens have no native identity, so a
sequence must be provided for them. If one is missing the pipeline stops rather
than letting the side-chain module fill the gap -- that would be co-design.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch

from pxf import atom37, bridge

SCN_CONTEXT_MODES = ("none", "keep_context")


@dataclass
class PackedDesign:
    """One backbone with its side chains packed onto a supplied sequence."""
    sample_name: str
    sample_index: int
    sequence: str
    coords_af2: torch.Tensor      # [L, 37, 3]
    atom_mask_af2: torch.Tensor   # [L, 37]
    design_mask: torch.Tensor     # [L]
    residue_index: torch.Tensor   # [L]
    chain_index: torch.Tensor     # [L]
    psce: torch.Tensor            # [L, 33] per-side-chain-atom confidence error
    backbone_shift: Optional[float] = None
    metrics: dict = field(default_factory=dict)

    @property
    def length(self):
        return int(self.coords_af2.shape[0])

    def designed_sequence(self):
        """Only the positions PXDesign generated, in order."""
        return "".join(a for a, keep in zip(self.sequence, self.design_mask.tolist()) if keep)


class FullAtomPipeline:
    """PXDesign backbone module -> FaMPNN side-chain packing module."""

    def __init__(self, backbone, sidechain, *, sequence=None, scn_context="none",
                 batch_size=None):
        if scn_context not in SCN_CONTEXT_MODES:
            raise ValueError(f"Unknown scn_context {scn_context!r}; choose from {SCN_CONTEXT_MODES}")
        self.backbone = backbone
        self.sidechain = sidechain
        # Full-length string, or {token_index: letter} for the design positions.
        self.sequence = sequence
        self.scn_context = scn_context
        self.batch_size = batch_size

    @property
    def identity(self):
        return dict(pipeline="pxdesign->fampnn(pack)", designs_sequence=False,
                    scn_context=self.scn_context,
                    backbone=self.backbone.identity, sidechain=self.sidechain.identity)

    def resolve_sequence(self, batch):
        """Complete the sequence FaMPNN will pack, or explain what is missing."""
        overrides = self.sequence
        if overrides is None:
            overrides = {}
        elif isinstance(overrides, str):
            if len(overrides) != batch.length:
                raise ValueError(
                    f"{batch.sample_name}: supplied sequence has length {len(overrides)} "
                    f"but the structure has {batch.length} residues")
        try:
            return bridge.apply_sequence_overrides(
                batch.native_sequence, batch.sequence_known, overrides)
        except ValueError as error:
            designed = int(batch.design_mask.sum())
            raise ValueError(
                f"{batch.sample_name}: {error} PXDesign produced {designed} design "
                f"token(s) of {batch.length} residues, which have no native identity.") from error

    def run_batch(self, batch):
        """Pack every backbone sample in ``batch`` onto the resolved sequence."""
        sequence = self.resolve_sequence(batch)
        aatype = atom37.aatype_from_sequence(sequence)
        if self.scn_context == "keep_context":
            # Only repack the designed positions; the target keeps its input rotamers.
            context = (~batch.design_mask).float()
        else:
            context = torch.zeros(batch.length)

        packed = self.sidechain(
            coords_af2=batch.coords_af2, aatype=aatype.unsqueeze(0).expand(batch.num_samples, -1),
            atom_mask=batch.atom_mask_af2,
            residue_index=batch.residue_index.unsqueeze(0).expand(batch.num_samples, -1),
            chain_index=batch.chain_index.unsqueeze(0).expand(batch.num_samples, -1),
            scn_context_mask=context.unsqueeze(0).expand(batch.num_samples, -1),
            batch_size=self.batch_size)

        results = []
        for index in range(batch.num_samples):
            psce = packed["psce"][index].detach().cpu()
            designed = batch.design_mask.bool()
            results.append(PackedDesign(
                sample_name=batch.sample_name, sample_index=index, sequence=sequence,
                coords_af2=packed["coords_af2"][index].detach().cpu(),
                atom_mask_af2=packed["atom_mask_af2"][index].detach().cpu(),
                design_mask=batch.design_mask, residue_index=batch.residue_index,
                chain_index=batch.chain_index, psce=psce,
                backbone_shift=packed["backbone_shift"],
                metrics=dict(num_designed=int(designed.sum()),
                             mean_psce=float(psce.mean()),
                             mean_psce_designed=float(psce[designed].mean()) if bool(designed.any()) else None,
                             dropped_backbone_atoms=list(batch.dropped_atoms))))
        return results

    def run(self, *, seed=None):
        """Generate backbones and pack them; returns every design produced."""
        designs = []
        for batch in self.backbone.generate(seed=seed):
            designs.extend(self.run_batch(batch))
        return designs

    # ---- output ----------------------------------------------------------

    def write_pdb(self, result, path):
        """Write one design with FaMPNN's own writer (psce in the B-factor column)."""
        from fampnn.model.sd_model import SeqDenoiser
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        length = result.length
        samples = {
            "x_denoised": result.coords_af2.unsqueeze(0),
            "seq_mask": torch.ones(1, length),
            "missing_atom_mask": torch.zeros(1, length, atom37.NUM_ATOM37),
            "residue_index": result.residue_index.unsqueeze(0).long(),
            "chain_index": result.chain_index.unsqueeze(0).long(),
            "pred_aatype": atom37.aatype_from_sequence(result.sequence).unsqueeze(0),
            "psce": result.psce.unsqueeze(0),
        }
        SeqDenoiser.save_samples_to_pdb(samples, [str(path)])
        return path
