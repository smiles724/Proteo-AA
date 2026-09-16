"""The FaMPNN training forward pass.

Why this file exists: upstream's ``SidechainDiffusionModule.sidechain_diffusion``
accepts an ``is_sampling`` flag but ignores it -- the only path it implements is
the multi-step sampling integrator, which needs ``aux_inputs["scd"]`` and runs 50
denoising steps. Training needs the opposite: one denoising step at a randomly
sampled noise level. So the sequence encoder and the side-chain denoiser MLP are
driven directly here, reusing upstream modules unchanged.

Everything else follows the preprint:

* the MAR interpolant supplies the sequence/side-chain masking (Appendix C,
  ``t = sqrt(u)`` keep-probability; shipped as ``uniform_sqrt_t``);
* the side-chain denoiser is teacher-forced on the ground-truth sequence rather
  than its own prediction (Section 4.3.1);
* the conditioning is cloned ``training_batch_size_mult`` times (8 in the
  released config) and a different noise level drawn for each, because the MLP is
  lightweight relative to the encoder (Section 4.3.1);
* the confidence head trains on a stop-gradient diffusion rollout, on roughly one
  step in ``subsample_train_iter_mult`` (8) (Appendix D.4);
* the side-chain objective is scored only where the interpolant *hid* the side
  chain. The encoder receives visible side chains as input (``encoder_inputs``
  gates atom37's non-backbone slots by ``scn_mlm_mask``), so supervising those
  residues would ask the denoiser to reproduce coordinates it was just shown --
  a shortcut, not masked modeling. The paper's objective is ``p(Y_M | Y_M-bar)``,
  so the target mask is ``(1 - scn_mlm_mask) * seq_mask``.

  Note the regime this matches: at inference ``sidechain_pack`` hides *every*
  side chain, so restricting training to hidden residues is also the only setting
  the module is ever deployed in. ``scn_mlm_mask=None`` means "nothing was
  visible" and supervises everything, which is the correct mask for the packing
  and coupling paths.
"""

from dataclasses import dataclass, field

import torch

from pxf.train import losses as loss_fns

# Keys a training batch must provide, all [b, n, ...] with padding marked by seq_mask.
REQUIRED_KEYS = (
    "x",
    "aatype",
    "seq_mask",
    "missing_atom_mask",
    "residue_index",
    "chain_index",
)


@dataclass
class StepOutput:
    """Losses and diagnostics for one training forward."""

    total: torch.Tensor
    mlm: torch.Tensor
    diffusion: torch.Tensor
    confidence: torch.Tensor | None = None
    stats: dict = field(default_factory=dict)

    def scalars(self):
        # loss_main is L_MLM + L_diff, which every step has. The total only
        # includes the confidence term on the ~1-in-8 steps it trains on, so
        # comparing totals across logging windows is misleading; compare loss_main.
        out = {
            "loss": float(self.total),
            "loss_main": float(self.mlm + self.diffusion),
            "loss_mlm": float(self.mlm),
            "loss_diffusion": float(self.diffusion),
        }
        if self.confidence is not None:
            out["loss_confidence"] = float(self.confidence)
        # Stats carry a few strings (the active reduction), so coerce only what
        # is actually numeric instead of float()-ing everything.
        out.update(
            {
                k: (float(v) if torch.is_tensor(v) or isinstance(v, (int, float)) else v)
                for k, v in self.stats.items()
            }
        )
        return out


def _constants():
    from fampnn.data import residue_constants as rc

    return rc


def encoder_inputs(model, batch, mar_out):
    """Reproduce ``FAMPNNDenoiser.forward``'s atom mask construction exactly.

    A mask built differently here than at inference would train the encoder on
    inputs it never sees again, so this mirrors upstream line for line.
    """
    rc = _constants()
    from fampnn.data.data import get_rc_tensor

    aatype_noised = mar_out["aatype_noised"]
    seq_mask = batch["seq_mask"]
    atom_mask = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype_noised)
    atom_mask = atom_mask * seq_mask.unsqueeze(-1)
    atom_mask = atom_mask * (1 - batch["missing_atom_mask"])
    atom_mask[..., rc.non_bb_idxs] = atom_mask[..., rc.non_bb_idxs] * mar_out[
        "scn_mlm_mask"
    ].unsqueeze(-1)
    return atom_mask


def sidechain_targets(model, batch, *, scn_mlm_mask=None):
    """Ground-truth side chains in the local backbone frame, plus their mask.

    Uses upstream's own frame construction (AF2 Algorithm 21 via OpenFold), so
    training targets live in exactly the space inference denoises in.

    ``scn_mlm_mask`` is the interpolant's ``[b, n]`` visibility mask -- 1 where the
    residue's side chain was given to the encoder, 0 where it was hidden. When
    supplied, the target mask is restricted to the hidden residues, which is what
    makes this a masked-modeling objective rather than a partial copy. ``None``
    means nothing was visible (packing and coupling), so everything is supervised.
    """
    rc = _constants()
    from fampnn.data.data import get_rc_tensor, transform_sidechain_frame

    x = batch["x"]
    aatype = batch["aatype"].long()
    x_scn = x[..., rc.non_bb_idxs, :]
    x_bb = x[..., rc.bb_idxs, :]
    # An atom is supervised when it exists for the true residue type, is present
    # in the structure, and is not padding.
    exists = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)[..., rc.non_bb_idxs]
    present = 1 - batch["missing_atom_mask"][..., rc.non_bb_idxs]
    atom_mask_scn = exists * present * batch["seq_mask"].unsqueeze(-1)
    atom_mask_bb = (1 - batch["missing_atom_mask"][..., rc.bb_idxs]) * batch[
        "seq_mask"
    ].unsqueeze(-1)
    # The frame transform is run on the *unrestricted* mask: it needs the real
    # atom set to build frames, and restricting the target set is a separate
    # decision applied afterwards.
    x_scn_local, bb_frames_exist = transform_sidechain_frame(
        x_scn, x_bb, atom_mask_scn, atom_mask_bb, to_local=True
    )
    # No frame means no defined local target for that residue.
    atom_mask_scn = atom_mask_scn * bb_frames_exist.unsqueeze(-1)
    if scn_mlm_mask is not None:
        # Score only what was hidden: predict the masked side chains from the
        # visible context, never from themselves.
        hidden = 1.0 - scn_mlm_mask.to(atom_mask_scn.dtype)
        atom_mask_scn = atom_mask_scn * hidden.unsqueeze(-1)
    return x_scn_local, atom_mask_scn


def _repeat(tensor, times):
    """Tile along a new leading axis then fold it into the batch axis."""
    return tensor.repeat_interleave(times, dim=0) if times > 1 else tensor


def diffusion_loss(
    model,
    batch,
    mpnn_feature_dict,
    *,
    multiplier=None,
    self_cond_p=None,
    generator=None,
    scn_mlm_mask=None,
    reduction=loss_fns.DEFAULT_SIDECHAIN_REDUCTION,
):
    """One denoising step per (example, noise level), teacher-forced on GT sequence.

    ``scn_mlm_mask`` restricts the target set to the residues whose side chain the
    interpolant hid (see the module docstring); ``None`` supervises every
    supervisable atom, which is the right mask when no side chain was visible.
    ``reduction`` is forwarded to :func:`pxf.train.losses.sidechain_diffusion_loss`.
    """
    module = model.denoiser.scn_diffusion_module
    interpolant = module.scn_interpolant
    denoiser = module.scn_denoiser
    multiplier = int(
        getattr(module.cfg, "training_batch_size_mult", 1)
        if multiplier is None
        else multiplier
    )
    self_cond_p = (
        float(getattr(module.cfg, "self_cond_p", 0.0))
        if self_cond_p is None
        else self_cond_p
    )

    x1_local, atom_mask = sidechain_targets(model, batch, scn_mlm_mask=scn_mlm_mask)
    h_V = mpnn_feature_dict["h_V"]
    seq_mask = batch["seq_mask"]
    aatype = batch["aatype"].long()

    # Clone the conditioning and draw an independent noise level per clone.
    x1_rep = _repeat(x1_local, multiplier)
    mask_rep = _repeat(atom_mask, multiplier)
    h_V_rep = _repeat(h_V, multiplier)
    seq_mask_rep = _repeat(seq_mask, multiplier)
    aatype_rep = _repeat(aatype, multiplier)

    t = interpolant.sample_timestep(x1_rep.shape[0], device=x1_rep.device)
    xt = interpolant.noise_x(x1_rep, t)

    self_cond = None
    if module.use_self_conditioning and self_cond_p > 0:
        # Draw on the generator's own device (CPU); the value is only compared.
        if float(torch.rand((), generator=generator)) < self_cond_p:
            # Standard self-conditioning: a detached first pass supplies the
            # estimate the graded pass conditions on.
            with torch.no_grad():
                primed, _ = denoiser(xt, aatype_rep, t, h_V_rep, seq_mask_rep)
            self_cond = primed.detach()

    x1_pred, _ = denoiser(
        xt, aatype_rep, t, h_V_rep, seq_mask_rep, x_scn_self_cond=self_cond
    )
    weight = interpolant.get_loss_weight(t)
    loss, stats = loss_fns.sidechain_diffusion_loss(
        x1_pred, x1_rep, weight, mask_rep, reduction=reduction
    )
    stats["noise_clones"] = torch.tensor(float(multiplier))
    stats["self_conditioned"] = torch.tensor(float(self_cond is not None))
    if scn_mlm_mask is not None:
        with torch.no_grad():
            visible = (scn_mlm_mask * seq_mask).sum()
            stats["hidden_sidechain_fraction"] = 1.0 - visible / seq_mask.sum().clamp_min(1)
    return loss, stats


@torch.no_grad()
def _rollout_packed_sidechains(model, batch, mpnn_feature_dict, aatype):
    """Run the shipped sampling integrator to get packed local side chains.

    Appendix D.4 trains the confidence head on the output of a diffusion rollout.
    This is the one place the sampling path is the right path, so it is reused as
    shipped -- under ``no_grad``, which is also the required stop gradient.
    """
    from fampnn import sampling_utils

    module = model.denoiser.scn_diffusion_module
    cfg = module.cfg.confidence_module.scn_diffusion
    steps = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)
    batch_size = batch["seq_mask"].shape[0]
    scd_inputs = {
        "num_steps": cfg.num_steps,
        "timesteps": steps[None].expand(batch_size, -1).to(batch["seq_mask"].device),
        "step_scale": cfg.noise_schedule.c,
        "churn_cfg": dict(cfg.churn_cfg),
        "aatype_override": aatype,
        "aatype_override_mask": batch["seq_mask"].long(),
    }
    _, aux = module.sidechain_diffusion(
        mpnn_feature_dict,
        aatype,
        seq_mask=batch["seq_mask"],
        residue_index=batch["residue_index"],
        chain_index=batch["chain_index"],
        aux_inputs={"scd": scd_inputs},
        is_sampling=True,
    )
    return aux


def confidence_loss(model, batch, mpnn_feature_dict, aatype, *, scn_mlm_mask=None):
    """Train the confidence head on a stop-gradient rollout (Appendix D.4).

    ``scn_mlm_mask`` restricts what is *scored*, not what the head is *shown*, and
    the distinction is load-bearing. The head is a network over the whole packed
    structure, so its input has to look the way it does at deployment: the
    rollout packs every residue and ``sidechain_pack`` hides every side chain, so
    the input is fully populated. Zeroing the visible residues' coordinates
    before the head sees them would shift its input distribution away from the
    only regime it is ever used in. The *scored* set is restricted for the same
    reason the diffusion target is -- a psCE head trained to call visible side
    chains "zero error" would be calibrated for a case that never arises.
    """
    module = model.denoiser.scn_diffusion_module
    # One frame transform, on the unrestricted mask; the restriction is a
    # reduction over the result, applied below.
    x1_local, atom_mask = sidechain_targets(model, batch)
    detached = {
        k: (v.detach() if torch.is_tensor(v) else v) for k, v in mpnn_feature_dict.items()
    }
    aux = _rollout_packed_sidechains(model, batch, detached, aatype)
    packed_global = aux["scn_pred"].detach()

    # The head scores local-frame coordinates, so bring the rollout back in.
    from fampnn.data.data import transform_sidechain_frame

    from fampnn.data import residue_constants as rc

    atom_mask_bb = (1 - batch["missing_atom_mask"][..., rc.bb_idxs]) * batch[
        "seq_mask"
    ].unsqueeze(-1)
    packed_local, _ = transform_sidechain_frame(
        packed_global,
        batch["x"][..., rc.bb_idxs, :],
        atom_mask,
        atom_mask_bb,
        to_local=True,
    )
    packed_local = packed_local.detach()

    psce_logits, _ = module.confidence_module(
        packed_local,
        detached,
        aatype,
        batch["seq_mask"],
        batch["residue_index"],
        batch["chain_index"],
    )
    scored = atom_mask
    if scn_mlm_mask is not None:
        scored = scored * (1.0 - scn_mlm_mask.to(scored.dtype)).unsqueeze(-1)
    spec = loss_fns.psce_bin_spec(module.confidence_module)
    return loss_fns.confidence_loss(
        psce_logits, packed_local, x1_local, scored, bin_spec=spec
    )


def training_forward(
    model,
    batch,
    *,
    train_confidence=None,
    generator=None,
    multiplier=None,
    self_cond_p=None,
    reduction=loss_fns.DEFAULT_SIDECHAIN_REDUCTION,
    supervise_visible_sidechains=False,
):
    """Compute ``L_total = L_MLM + L_diff`` (+ confidence) for one batch.

    ``supervise_visible_sidechains=True`` restores the unrestricted target set --
    every existing atom scored, including residues whose side chain the encoder
    was shown. Kept only as an ablation switch; it is not masked modeling.
    """
    missing = [key for key in REQUIRED_KEYS if key not in batch]
    if missing:
        raise ValueError(f"training batch is missing {missing}")
    module = model.denoiser.scn_diffusion_module

    # 1. Mask sequence and side chains (MAR); drop_sidechains needs train mode.
    #
    # Upstream's MAR is declared as a plain class while being written like an
    # nn.Module: it has `forward` and `super().__init__()` and reads
    # `self.training`, but is not callable and never receives `model.train()`.
    # Two consequences, both handled here: call `forward` directly, and mirror the
    # model's mode across. (Inference never touches model.interpolant -- it goes
    # through sidechain_pack/sample -- which is why this went unnoticed upstream.)
    model.interpolant.training = bool(model.training)
    mar_out = model.interpolant.forward(batch)

    # 2. Full-atom encoder: sequence logits and node embeddings.
    atom_mask_noised = encoder_inputs(model, batch, mar_out)
    seq_logits, mpnn_feature_dict = model.denoiser.seq_design_module(
        mar_out["x_noised"],
        mar_out["aatype_noised"],
        batch["seq_mask"],
        atom_mask_noised,
        batch["residue_index"],
        batch["chain_index"],
    )

    # 3. L_MLM on the positions the interpolant masked.
    loss_mlm, stats = loss_fns.sequence_mlm_loss(
        seq_logits, batch["aatype"], mar_out["seq_mlm_mask"], batch["seq_mask"]
    )

    # 4. L_diff on the residues whose side chain was hidden, teacher-forced on
    #    the ground-truth sequence.
    scn_target_mask = None if supervise_visible_sidechains else mar_out["scn_mlm_mask"]
    loss_diff, diff_stats = diffusion_loss(
        model,
        batch,
        mpnn_feature_dict,
        multiplier=multiplier,
        self_cond_p=self_cond_p,
        generator=generator,
        scn_mlm_mask=scn_target_mask,
        reduction=reduction,
    )
    stats.update(diff_stats)

    # 5. Confidence head, on a stop-gradient rollout, ~1 step in 8.
    loss_conf = None
    if train_confidence is None:
        train_confidence = (
            module.use_confidence_module
            and float(torch.rand((), generator=generator))
            < module.confidence_module_train_p
        )
    if train_confidence:
        if not module.use_confidence_module:
            raise ValueError("Confidence training requested but the module is disabled")
        loss_conf, conf_stats = confidence_loss(
            model,
            batch,
            mpnn_feature_dict,
            batch["aatype"].long(),
            scn_mlm_mask=scn_target_mask,
        )
        stats.update(conf_stats)

    with torch.no_grad():
        stats["sequence_accuracy"] = _masked_accuracy(
            seq_logits, batch["aatype"], (1 - mar_out["seq_mlm_mask"]) * batch["seq_mask"]
        )
        stats["keep_fraction"] = mar_out["seq_mlm_mask"].sum() / batch[
            "seq_mask"
        ].sum().clamp_min(1)
    return StepOutput(
        total=loss_fns.total_loss(loss_mlm, loss_diff, loss_conf),
        mlm=loss_mlm,
        diffusion=loss_diff,
        confidence=loss_conf,
        stats=stats,
    )


def _masked_accuracy(seq_logits, aatype, mask):
    correct = (seq_logits.argmax(-1) == aatype.long()).float() * mask
    return correct.sum() / mask.sum().clamp_min(1)
