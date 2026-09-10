"""Mask before encoding: a query exposes neither identity nor SC inventory."""
import math
import torch
from .atom_mapping import BB37


def visible_input(xyz, atom_mask, assigned_aa, residue_mask, design_mask,
                  query_mask, seq_visible, sc_visible):
    if (query_mask & ~design_mask).any():
        raise ValueError("Queries must be a subset of design ownership")
    if (sc_visible & ~seq_visible).any():
        raise ValueError("Visible side chains require visible identities")
    seq = seq_visible.bool() & ~query_mask & residue_mask.bool()
    sc = sc_visible.bool() & seq
    aa = torch.where(seq, assigned_aa, 20)
    if ((aa < 0) | (aa > 20)).any():
        raise ValueError("Visible identities must be canonical AA or X")
    bb = torch.zeros(37, dtype=torch.bool, device=xyz.device)
    bb[list(BB37)] = True
    mask = atom_mask.bool() & residue_mask[..., None].bool()
    mask = mask & (bb | sc[..., None])
    clean = torch.where(mask[..., None], xyz, 0.0)
    return dict(denoised_coords=clean, aatype_noised=aa.long(),
                seq_mask=residue_mask.to(xyz.dtype), atom_mask_noised=mask.to(xyz.dtype))


def assign_aa(logits, temperature=0.0, generator=None):
    """Canonical assignment. X never participates in the 20-class distribution."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    if logits.shape[-1] != 20:
        raise ValueError("Assignment requires explicitly mapped canonical logits")
    if not torch.isfinite(logits).all():
        raise ValueError("AA assignment received non-finite logits")
    if temperature == 0:
        return logits.argmax(-1)
    p = (logits.float() / temperature).softmax(-1)
    return torch.multinomial(p.reshape(-1, 20), 1, generator=generator).reshape(logits.shape[:-1])
