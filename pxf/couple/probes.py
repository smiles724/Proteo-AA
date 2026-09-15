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
"""
from dataclasses import dataclass, field
import torch

from pxf import atom37
from pxf.couple import fampnn_iface as iface


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
    floor: float                  # identical input re-encoded (noise floor)
    ceiling: float                # masked -> visible side chains
    responses: dict = field(default_factory=dict)   # perturbation -> relative change

    def verdict(self, *, margin=10.0):
        """Whether the strongest response clears the floor by a clear margin."""
        best = max(self.responses.values()) if self.responses else 0.0
        if best <= max(self.floor, 1e-9) * margin:
            return "insensitive"
        if best >= 0.1 * self.ceiling:
            return "informative"
        return "weak"

    def summary(self):
        return dict(floor=self.floor, ceiling=self.ceiling,
                    responses=dict(self.responses), verdict=self.verdict())


@torch.no_grad()
def sidechain_sensitivity(model, coords_af2, aatype, *, seq_mask=None,
                          missing_atom_mask=None, residue_index=None,
                          chain_index=None, sidechains=None,
                          perturbations=(0.1, 0.5, 1.0), generator=None):
    """Measure how ``h_packed`` responds to perturbing the side chains shown.

    ``sidechains`` defaults to the side chains already in ``coords_af2`` (i.e. the
    native ones); pass a packed prediction to probe the realization the coupled
    system would actually feed back.
    """
    from fampnn.data import residue_constants as rc
    kwargs = dict(seq_mask=seq_mask, missing_atom_mask=missing_atom_mask,
                  residue_index=residue_index, chain_index=chain_index)
    scored = seq_mask if seq_mask is not None else None

    block = (coords_af2[..., rc.non_bb_idxs, :] if sidechains is None
             else (sidechains[..., rc.non_bb_idxs, :]
                   if sidechains.shape[-2] == atom37.NUM_ATOM37 else sidechains))

    _, h_masked, _ = iface.encode(model, coords_af2, aatype, **kwargs)
    _, h_visible, _ = iface.encode(model, coords_af2, aatype, sidechains=block, **kwargs)
    _, h_repeat, _ = iface.encode(model, coords_af2, aatype, sidechains=block, **kwargs)

    report = SensitivityReport(
        floor=_relative_change(h_visible, h_repeat, scored),
        ceiling=_relative_change(h_visible, h_masked, scored))

    # Only perturb atoms that exist for the true residue type.
    exists = torch.as_tensor(rc.STANDARD_ATOM_MASK_WITH_X, device=coords_af2.device)
    exists = exists[aatype.long()][..., rc.non_bb_idxs]
    for sigma in perturbations:
        noise = torch.randn(block.shape, generator=generator,
                            dtype=block.dtype, device=block.device)
        moved = block + noise * float(sigma) * exists[..., None]
        _, h_moved, _ = iface.encode(model, coords_af2, aatype, sidechains=moved, **kwargs)
        report.responses[f"gaussian_{sigma}A"] = _relative_change(h_visible, h_moved, scored)

    # A maximally wrong realization: every side chain collapsed onto CA.
    ca = coords_af2[..., rc.atom_order["CA"], :]
    collapsed = ca[..., None, :].expand_as(block).contiguous()
    _, h_collapsed, _ = iface.encode(model, coords_af2, aatype, sidechains=collapsed, **kwargs)
    report.responses["collapsed_to_CA"] = _relative_change(h_visible, h_collapsed, scored)
    return report


@torch.no_grad()
def packing_response(model, coords_af2, aatype, *, seq_mask=None,
                     missing_atom_mask=None, residue_index=None, chain_index=None,
                     delta_scale=1.0, num_steps=None, generator=None):
    """Does a residual on ``h_V`` change the packing it produces? (BB -> SC path.)

    The mirror question for the forward direction: an adapter writing into the
    encoder features is only useful if packing responds to it.
    """
    kwargs = dict(seq_mask=seq_mask, missing_atom_mask=missing_atom_mask,
                  residue_index=residue_index, chain_index=chain_index)
    _, h_base, features = iface.encode(model, coords_af2, aatype, **kwargs)
    pack_kwargs = dict(seq_mask=seq_mask, residue_index=residue_index,
                       chain_index=chain_index, num_steps=num_steps)
    torch.manual_seed(0)
    reference, _ = iface.pack_from_features(model, features, aatype, **pack_kwargs)
    torch.manual_seed(0)
    repeat, _ = iface.pack_from_features(model, features, aatype, **pack_kwargs)
    delta = torch.randn(h_base.shape, generator=generator, dtype=h_base.dtype,
                        device=h_base.device)
    delta = delta * float(delta_scale) * h_base.norm(dim=-1, keepdim=True) / \
        delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    torch.manual_seed(0)
    moved, _ = iface.pack_from_features(model, features, aatype, delta_h=delta,
                                        **pack_kwargs)
    rmsd = lambda a, b: float((a - b).float().pow(2).sum(-1).mean().sqrt())
    return dict(floor_angstrom=rmsd(reference, repeat),
                response_angstrom=rmsd(reference, moved),
                delta_relative_norm=float(delta_scale))
