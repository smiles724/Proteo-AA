"""Phase 0: expose FaMPNN's internals as three callable pieces.

Upstream only offers whole-pipeline entry points (``sidechain_pack``, ``sample``).
Coupling needs the seam *inside*: encode structure to node features, optionally
add a residual to those features, pack side chains from them, and re-encode the
result. Nothing here reimplements a layer -- the full-atom encoder and the
side-chain diffusion module are called as shipped, with ``h_V`` substituted in
the feature dict, which is the one hook the upstream code already reads.

    h_base       = encode(model, bb, seq)                  # side chains masked
    x_scn, aux   = pack_from_features(model, feats, seq)    # h_V may be modified
    h_packed     = encode(model, bb, seq, sidechains=x_scn) # predicted SC visible

The third call is the load-bearing one for SC -> BB feedback: if ``h_packed`` is
insensitive to which side chains were realized, the feedback path carries no
signal no matter how good the adapter is. :mod:`pxf.couple.probes` measures that.
"""
from typing import Optional
import torch

from pxf import atom37


def _rc():
    from fampnn.data import residue_constants as rc
    return rc


def build_atom_mask(model, aatype, seq_mask, missing_atom_mask, sidechain_visible):
    """The encoder's input atom mask, exactly as ``FAMPNNDenoiser.forward`` builds it.

    ``sidechain_visible`` is per-residue: 1 where the encoder may see side-chain
    atoms, 0 where they are masked out.
    """
    from fampnn.data.data import get_rc_tensor
    rc = _rc()
    mask = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)
    mask = mask * seq_mask.unsqueeze(-1)
    mask = mask * (1 - missing_atom_mask)
    mask[..., rc.non_bb_idxs] = (mask[..., rc.non_bb_idxs]
                                 * sidechain_visible.unsqueeze(-1))
    return mask


def encode(model, coords_af2, aatype, *, seq_mask=None, missing_atom_mask=None,
           residue_index=None, chain_index=None, sidechains=None,
           sidechain_visible=None):
    """Run the full-atom encoder; returns ``(seq_logits, h_V, feature_dict)``.

    ``sidechains`` optionally supplies side-chain coordinates to make visible --
    the ``[..., 33, 3]`` block in atom37 side-chain order, or a full
    ``[..., 37, 3]`` tensor. With neither ``sidechains`` nor
    ``sidechain_visible``, side chains are fully masked, which is the
    backbone-only encoding the packer starts from.
    """
    rc = _rc()
    coords = coords_af2.clone()
    batch, length = coords.shape[0], coords.shape[1]
    device = coords.device
    aatype = aatype.long()
    seq_mask = torch.ones(batch, length, device=device) if seq_mask is None else seq_mask
    missing_atom_mask = (torch.zeros(batch, length, atom37.NUM_ATOM37, device=device)
                         if missing_atom_mask is None else missing_atom_mask)
    if residue_index is None:
        residue_index = torch.arange(length, device=device).expand(batch, length)
    if chain_index is None:
        chain_index = torch.zeros(batch, length, dtype=torch.long, device=device)

    if sidechains is not None:
        block = sidechains
        if block.shape[-2] == atom37.NUM_ATOM37:
            block = block[..., rc.non_bb_idxs, :]
        coords[..., rc.non_bb_idxs, :] = block.to(coords.dtype)
        if sidechain_visible is None:
            sidechain_visible = torch.ones(batch, length, device=device)
    if sidechain_visible is None:
        sidechain_visible = torch.zeros(batch, length, device=device)
    # Masked side chains must not leak coordinates through the mask.
    coords[..., rc.non_bb_idxs, :] = (coords[..., rc.non_bb_idxs, :]
                                      * sidechain_visible[..., None, None])

    atom_mask = build_atom_mask(model, aatype, seq_mask, missing_atom_mask,
                                sidechain_visible)
    seq_logits, feature_dict = model.denoiser.seq_design_module(
        coords, aatype, seq_mask, atom_mask, residue_index, chain_index)
    return seq_logits, feature_dict["h_V"], feature_dict


def with_residual(feature_dict, delta):
    """A feature dict whose ``h_V`` carries an additive residual.

    Substituting ``h_V`` is the whole coupling mechanism on this side: the
    side-chain diffusion module reads ``feature_dict["h_V"]`` and nothing else
    from the encoder, so an additive residual here conditions packing without
    touching upstream code.
    """
    if delta is None:
        return feature_dict
    updated = dict(feature_dict)
    updated["h_V"] = feature_dict["h_V"] + delta
    return updated


def scd_inputs(model, batch, *, aatype=None, seq_mask=None, num_steps=None,
               step_scale=None, churn=None, device=None):
    """Side-chain diffusion sampling inputs, defaulted from the released config."""
    from fampnn import sampling_utils
    module = model.denoiser.scn_diffusion_module
    cfg = module.cfg.confidence_module.scn_diffusion
    steps = int(num_steps if num_steps is not None else cfg.num_steps)
    schedule = dict(cfg.timestep_schedule)
    schedule["num_steps"] = steps
    timesteps = sampling_utils.get_timesteps_from_schedule(**schedule)
    device = device or next(model.parameters()).device
    inputs = {
        "num_steps": steps,
        "timesteps": timesteps[None].expand(batch, -1).to(device),
        "step_scale": float(step_scale if step_scale is not None
                            else cfg.noise_schedule.c),
        "churn_cfg": dict(churn or dict(cfg.churn_cfg), num_steps=steps),
    }
    if aatype is not None:
        # Teacher-force the identities the packer must build for.
        inputs["aatype_override"] = aatype.long()
        inputs["aatype_override_mask"] = (torch.ones_like(aatype, dtype=torch.long)
                                          if seq_mask is None else seq_mask.long())
    return inputs


def pack_from_features(model, feature_dict, aatype, *, seq_mask=None,
                       residue_index=None, chain_index=None, num_steps=None,
                       step_scale=None, churn=None, delta_h=None):
    """Pack side chains from (optionally residual-modified) encoder features.

    Returns ``(x_scn_global, aux)`` where ``x_scn_global`` is ``[..., 33, 3]`` in
    global coordinates and ``aux`` carries ``psce`` and the local-frame packing.
    """
    module = model.denoiser.scn_diffusion_module
    features = with_residual(feature_dict, delta_h)
    h_V = features["h_V"]
    batch, length = h_V.shape[0], h_V.shape[1]
    device = h_V.device
    aatype = aatype.long()
    seq_mask = torch.ones(batch, length, device=device) if seq_mask is None else seq_mask
    if residue_index is None:
        residue_index = torch.arange(length, device=device).expand(batch, length)
    if chain_index is None:
        chain_index = torch.zeros(batch, length, dtype=torch.long, device=device)
    inputs = {"scd": scd_inputs(model, batch, aatype=aatype, seq_mask=seq_mask,
                                num_steps=num_steps, step_scale=step_scale,
                                churn=churn, device=device)}
    x_scn, aux = module.sidechain_diffusion(
        features, aatype, seq_mask=seq_mask, residue_index=residue_index,
        chain_index=chain_index, aux_inputs=inputs, is_sampling=True)
    return x_scn, aux


def node_feature_dim(model):
    """Width of ``h_V``, i.e. the adapter's FaMPNN-side dimension."""
    return int(model.denoiser.scn_diffusion_module.cfg.scn_denoiser.c_h_V)
