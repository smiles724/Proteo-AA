"""Side-chain -> backbone feedback through h_res', at the packer's width.

The Stage II-B cycle fed `h_res'` into the backbone's token trunk `s_trunk`
(which PXDesign leaves as zeros) and then reused B_theta for a second denoise
pass. That injection point is kept here; what changes is where h_res' comes
from -- a FROZEN APM packer instead of our own side-chain module.

WHY THE NEIGHBOURHOOD AND NOT THE RESIDUE. Encoding a residue's own side-chain
coordinates in its own frame carries nothing beyond its chi angles: bond lengths
and angles are constants in `build_sidechain_local` and Rodrigues rotation
preserves length, so `coord_local(i)` is a bijection of `(restype(i), chi(i))`.
What coordinates add over torsions appears only ACROSS residues -- where a
neighbour's side-chain atoms sit relative to mine -- and that is also the only
thing this feedback can supply that IPA reading (R, t) cannot. `_CrossAtomBlock`
already is that encoder (its own docstring: "mean-pooling is direction-blind"),
so this module is the wiring, not a new mechanism.

WIDTHS, AND WHY STAGE III's WEIGHTS CANNOT BE REUSED. The old side-chain
module's per-atom features are 768 wide, so Stage III's `hres_injector` is
LayerNorm(768) -> Linear(768, 384). The APM packer's are 256 (`packer.py`:
`self.c_atom = int(c_node)`). That is a shape incompatibility rather than a
distribution-shift concern, so the injector here is a fresh zero-initialised
instance and `111408/step6000`'s weights are not loadable into it.

ZERO INITIALISATION IS THE FIRST TEST, NOT A DETAIL. `HResInjector`'s output
layer starts at zero, so an untrained module returns exactly zeros and the
second pass is bit-identical to the first. `_CrossAtomBlock`'s own output
projection is NOT zero-initialised, so the no-op property rests entirely on the
injector -- which is why `tests/test_sc_env_feedback.py` asserts it rather than
assuming it.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from pxdesign_train.sidechain.coevolution import HResInjector, pool_side_chain_atoms
from pxdesign_train.sidechain.instantiate import ATOM_VOCAB_SIZE
from pxdesign_train.sidechain.module import _CrossAtomBlock

FEATURE_SOURCES = ("packer", "atom_id")


class SidechainEnvFeedback(nn.Module):
    """Packed side chains -> an additive update for the backbone's `s_trunk`.

    Args:
        c_atom: per-atom feature width. 256 for our APM packer port.
        c_trunk: backbone token width (`c_s`), 384 on this substrate.
        n_blocks: cross-atom neighbourhood blocks. 0 is the same as
            `use_env=False`.
        n_neighbors: residues per neighbourhood, by CA distance. The only
            memory knob: the gather is [B, L, M, A, *], so M=16 with A=10 and
            L=640 is 3.1e6 coordinates.
        feature_source: which per-atom features enter the blocks.
            "packer"  -- the packer's `sc_feats` (atom_embed + node_embed).
            "atom_id" -- this module's own atom-name embedding ONLY, which
                         removes the packer's learned representation and leaves
                         atom identity plus geometry. That is the control that
                         separates "the coordinates helped" from "APM's
                         node_embed helped"; without it a win for the
                         neighbourhood arm is not attributable.
        use_env: False reduces this to pooling the features straight into the
            injector -- the feature-only arm -- so one class covers both.
        detach_inputs: cut the gradient at the module's inputs. True reproduces
            APM's `RefineModel`, which detaches the torsions it consumes
            (`refine_model.py`). False lets the backbone learn to emit frames
            whose side chains pack well, which is what Stage II-B did and what
            APM did not try.
    """

    def __init__(
        self,
        c_atom: int = 256,
        c_trunk: int = 384,
        n_blocks: int = 2,
        n_heads: int = 8,
        n_neighbors: int = 16,
        feature_source: str = "packer",
        use_env: bool = True,
        detach_inputs: bool = True,
    ) -> None:
        super().__init__()
        if feature_source not in FEATURE_SOURCES:
            raise ValueError(
                f"feature_source must be one of {FEATURE_SOURCES}, got {feature_source!r}")
        if use_env and n_blocks < 1:
            raise ValueError(
                "use_env=True with n_blocks=0 would silently be the feature-only "
                "arm under a name that says otherwise; pass use_env=False instead")
        self.c_atom = int(c_atom)
        self.feature_source = str(feature_source)
        self.use_env = bool(use_env)
        self.detach_inputs = bool(detach_inputs)
        self.n_neighbors = int(n_neighbors)

        self.env_blocks = nn.ModuleList(
            [_CrossAtomBlock(c_atom, n_heads, n_neighbors) for _ in range(n_blocks)]
            if self.use_env else []
        )
        # Only the "atom_id" control owns an embedding; the "packer" source must
        # not carry one, or the control and the treatment would differ by a
        # parameter as well as by an input.
        self.atom_embed = (nn.Embedding(ATOM_VOCAB_SIZE, c_atom, padding_idx=0)
                           if self.feature_source == "atom_id" else None)
        self.injector = HResInjector(c_hres=c_atom, c_trunk=c_trunk)

    def forward(
        self,
        sc_feats: Optional[torch.Tensor],   # [B, L, A, c_atom] or None for atom_id
        sc_coords: torch.Tensor,            # [B, L, A, 3] GLOBAL side-chain coords
        chem_mask: torch.Tensor,            # [B, L, A] bool -- chemistry, not resolution
        ca: torch.Tensor,                   # [B, L, 3] GLOBAL CA, for the KNN
        res_mask: torch.Tensor,             # [B, L] bool
        atom_name_ids: Optional[torch.Tensor] = None,   # [B, L, A] long
    ) -> torch.Tensor:
        """Returns [B, L, c_trunk] to be ADDED to `s_trunk`. Zero at init."""
        if self.feature_source == "atom_id":
            if atom_name_ids is None:
                raise ValueError("feature_source='atom_id' needs atom_name_ids")
            feats = self.atom_embed(atom_name_ids.long())
        else:
            if sc_feats is None:
                raise ValueError("feature_source='packer' needs sc_feats")
            feats = sc_feats
        if feats.shape[-1] != self.c_atom:
            raise ValueError(
                f"per-atom features are {feats.shape[-1]} wide but this module was "
                f"built for {self.c_atom}. The APM packer emits c_node (256); the "
                f"old side-chain module emitted 768.")

        coords, ca_in = sc_coords, ca
        if self.detach_inputs:
            feats, coords, ca_in = feats.detach(), sc_coords.detach(), ca.detach()

        # Centre on the masked CA centroid before the blocks. Distances are
        # unchanged, so this is mathematically a no-op -- but `cdist` expands
        # ||a||^2 + ||b||^2 - 2a.b, which loses precision when the coordinate
        # magnitudes dwarf the differences. Measured over a rigid motion: fp64
        # agrees to 1.3e-08 (so the module IS invariant by construction) while
        # fp32 drifts 2.0e-04 at a 13 A offset and 2.1e-03 at 1000 A. Centring
        # makes the fp32 error independent of where the protein sits in space.
        m = res_mask.bool()[..., None].to(ca_in.dtype)
        centre = (ca_in * m).sum(-2, keepdim=True) / m.sum(-2, keepdim=True).clamp(min=1.0)
        ca_in = ca_in - centre
        coords = coords - centre[..., None, :]

        chem = chem_mask.bool()
        nbr_idx = None
        for blk in self.env_blocks:
            if nbr_idx is None:
                # Computed once and shared: the neighbourhood is a property of
                # the backbone, not of the block, and recomputing it per block
                # would also let the blocks disagree about it.
                nbr_idx = blk.residue_neighbours(ca_in, res_mask.bool(),
                                                 self.n_neighbors)
            feats = blk(feats, coords, chem, ca_in, res_mask.bool(), nbr_idx=nbr_idx)

        h_res_sc = pool_side_chain_atoms(feats, chem)
        return self.injector(h_res_sc)


__all__ = ["SidechainEnvFeedback", "FEATURE_SOURCES"]
