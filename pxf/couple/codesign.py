"""Sequence + side-chain co-design, following FaMPNN's published inference.

FaMPNN generates side chains for residue ``i`` by first predicting the identity
``s_hat_i`` with the MPNN module and then passing ``(s_hat_i, v_i)`` into the
side-chain diffusion MLP. ``FAMPNNDenoiser.forward`` is exactly that:

    seq_logits, mpnn_feature_dict = seq_design_module(x, aatype_noised, ...)
    aatype_pred, _                = sample_aatype(seq_logits, aux, is_sampling)
    x_scn, _                      = sidechain_diffusion(mpnn_feature_dict,
                                                        aatype_pred, ...)

This module reproduces that on pxf's own seams rather than calling
``FAMPNNDenoiser.forward``, because the coupling residual has to land on ``h_V``
between the two stages -- and ``h_V`` (inside ``mpnn_feature_dict``) is the only
encoder output the side-chain module reads. Nothing in FaMPNN is modified, so
``pxf.provenance`` still pins it unpatched.

The sequence is **not** teacher-forced. The encoder is handed the ``X`` mask
token, whose ``STANDARD_ATOM_MASK_WITH_X`` row exposes only the four backbone
atoms, so neither the identity nor the deposited side chain can leak into the
prediction. Contrast ``pxf.couple.fampnn_iface.scd_inputs``, which teacher-forces
identities for the pure packing task.

DECODING: this is the **single-pass** variant -- one MPNN evaluation over a fully
masked sequence, which is the literal reading of the paper's sentence. FaMPNN's
shipped ``SeqDenoiser.sample()`` instead decodes iteratively (MAR: ``S`` steps,
unmasking ``K`` residues per step) so later positions condition on earlier ones.
Single-pass keeps the two coupling arms strictly comparable and needs no hook
inside FaMPNN, but absolute sequence recovery is expected to be lower than the
iterative scheme; do not compare these recovery numbers against published
FaMPNN design results.

SCORING NOTE: with a predicted sequence, side-chain geometry metrics are only
defined where ``s_hat == aatype`` -- elsewhere the predicted residue has a
different atom set than the deposited one, so "rotamer recovery" has no referent.
:func:`sequence_recovery` reports the identity match, and callers should pass
``agreement_mask`` as ``score``'s ``residue_mask``.
"""

from __future__ import annotations

import torch

from pxf.couple import fampnn_iface as iface
from pxf.couple.controller import CycleOutput

DECODE_SINGLE_PASS = "single_pass"


def _rc():
    from fampnn.data import residue_constants as rc

    return rc


def mask_token():
    """FaMPNN's unknown/mask residue index (``X``)."""
    return int(_rc().restype_order_with_x["X"])


def masked_aatype(aatype):
    """``aatype``-shaped tensor of the ``X`` mask token."""
    return torch.full_like(aatype.long(), mask_token())


def predict_sequence(model, seq_logits, *, temperature=0.0):
    """``s_hat`` from the MPNN logits, via FaMPNN's own ``sample_aatype``.

    ``temperature=0.0`` is argmax/greedy. The call is delegated rather than
    reimplemented so the ``X``-suppression and temperature semantics match the
    released sampler exactly. ``sample_aatype`` writes ``-1e9`` into the logits
    in place, so a clone is passed.
    """
    aux = {"temperature": float(temperature)}
    s_hat, _ = model.denoiser.sample_aatype(seq_logits.clone(), aux, True)
    return s_hat.long()


def sequence_recovery(s_hat, aatype, *, seq_mask=None):
    """Identity agreement between prediction and reference.

    Returns ``(agreement_mask, counts)``; ``agreement_mask`` is the per-residue
    bool of ``s_hat == aatype``, suitable as ``score``'s ``residue_mask``.
    """
    ref = aatype.reshape(-1).long()
    # Call sites differ: the denoised loop holds aatype on the GPU, run_native
    # parses it on the CPU. Align to the reference rather than assuming either.
    s_hat = s_hat.reshape(-1).long().to(ref.device)
    if s_hat.shape != ref.shape:
        raise ValueError(
            f"predicted sequence {tuple(s_hat.shape)} does not match "
            f"reference {tuple(ref.shape)}"
        )
    valid = (
        torch.ones_like(ref, dtype=torch.bool)
        if seq_mask is None
        else seq_mask.reshape(-1).bool().to(ref.device)
    )
    agree = (s_hat == ref) & valid
    return agree, dict(
        residues=int(valid.sum()),
        recovered=int(agree.sum()),
        sequence_recovery=float(agree.sum() / valid.sum().clamp(min=1)),
    )


def codesign_native(
    model,
    coords_af2,
    *,
    seq_mask=None,
    missing_atom_mask=None,
    residue_index=None,
    chain_index=None,
    pack_steps=None,
    temperature=0.0,
):
    """Co-design on a given backbone: MPNN -> ``s_hat`` -> side-chain diffusion.

    No adapters and no backbone denoiser are involved, so this is the co-design
    counterpart of the ``native`` packing reference.
    """
    seq_logits, _h_V, features = iface.encode(
        model,
        coords_af2,
        masked_aatype(
            torch.zeros(
                coords_af2.shape[0], coords_af2.shape[1],
                dtype=torch.long, device=coords_af2.device,
            )
        ),
        seq_mask=seq_mask,
        missing_atom_mask=missing_atom_mask,
        residue_index=residue_index,
        chain_index=chain_index,
    )
    s_hat = predict_sequence(model, seq_logits, temperature=temperature)
    sidechains, aux = iface.pack_from_features(
        model,
        features,
        s_hat,
        seq_mask=seq_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        num_steps=pack_steps,
    )
    return s_hat, sidechains, dict(pack=aux, seq_logits=seq_logits)


def codesign_cycle(controller, topology, x_noisy, sigma, aatype, *, temperature=0.0):
    """One coupling cycle in co-design mode. Returns ``(CycleOutput, s_hat)``.

    Mirrors :meth:`pxf.couple.controller.CoupledDenoiser.forward` up to the
    packing step, substituting the masked encode and the predicted sequence. The
    adapter residual is obtained through the controller's own ``_delta_h``, so
    the coupled/uncoupled switch (``adapters.enable_bb_to_sc``) behaves exactly
    as in the packing-task cycle and the two arms differ only in that residual.

    The SC->BB feedback branch is not run: phase 1 trains only ``A_BS``, and
    re-encoding a *predicted* sequence's side chains would confound the
    measurement with the sequence error.
    """
    bb0_flat, a_token = controller.backbone(x_noisy, sigma)
    if a_token is None:
        raise ValueError(
            "The backbone denoiser returned no token features; "
            "coupling needs a_token (see pxf.couple.pxdesign_iface)"
        )

    inputs = controller.converter.px_backbone_to_fampnn(
        bb0_flat,
        topology.atom_names,
        topology.atom_to_token_idx,
        topology.num_tokens,
        res_names=topology.res_names,
        residue_index=topology.residue_index,
        chain_index=topology.chain_index,
        aatype=aatype,
    )

    # Masked encode: the MPNN must not see the identities it is predicting.
    seq_logits, h_base, features = iface.encode(
        controller.fampnn,
        inputs.coords_af2,
        masked_aatype(inputs.aatype),
        seq_mask=inputs.seq_mask,
        missing_atom_mask=inputs.missing_atom_mask,
        residue_index=inputs.residue_index,
        chain_index=inputs.chain_index,
    )
    s_hat = predict_sequence(controller.fampnn, seq_logits, temperature=temperature)

    delta_h = controller._delta_h(a_token, sigma, h_base)
    sidechains, pack_aux = iface.pack_from_features(
        controller.fampnn,
        iface.with_residual(features, delta_h),
        s_hat,
        seq_mask=inputs.seq_mask,
        residue_index=inputs.residue_index,
        chain_index=inputs.chain_index,
        num_steps=controller.pack_steps,
    )

    out = CycleOutput(
        bb0_flat=bb0_flat,
        bb0_dense=inputs.coords_af2,
        a_token=a_token,
        h_base=h_base,
        delta_h=delta_h,
        h_cond=h_base if delta_h is None else h_base + delta_h,
        sidechains=sidechains,
        aux=dict(
            inputs=inputs,
            pack=pack_aux,
            seq_logits=seq_logits,
            decode=DECODE_SINGLE_PASS,
        ),
    )
    return out, s_hat
