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
  step in ``subsample_train_iter_mult`` (8) (Appendix D.4).
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
        out.update({k: float(v) for k, v in self.stats.items()})
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


def sidechain_targets(model, batch):
    """Ground-truth side chains in the local backbone frame, plus their mask.

    Uses upstream's own frame construction (AF2 Algorithm 21 via OpenFold), so
    training targets live in exactly the space inference denoises in.
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
    x_scn_local, bb_frames_exist = transform_sidechain_frame(
        x_scn, x_bb, atom_mask_scn, atom_mask_bb, to_local=True
    )
    # No frame means no defined local target for that residue.
    atom_mask_scn = atom_mask_scn * bb_frames_exist.unsqueeze(-1)
    return x_scn_local, atom_mask_scn


def _repeat(tensor, times):
    """Tile along a new leading axis then fold it into the batch axis."""
    return tensor.repeat_interleave(times, dim=0) if times > 1 else tensor


def diffusion_loss(
    model, batch, mpnn_feature_dict, *, multiplier=None, self_cond_p=None, generator=None
):
    """One denoising step per (example, noise level), teacher-forced on GT sequence."""
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

    x1_local, atom_mask = sidechain_targets(model, batch)
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
    loss, stats = loss_fns.sidechain_diffusion_loss(x1_pred, x1_rep, weight, mask_rep)
    stats["noise_clones"] = torch.tensor(float(multiplier))
    stats["self_conditioned"] = torch.tensor(float(self_cond is not None))
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


def confidence_loss(model, batch, mpnn_feature_dict, aatype):
    """Train the confidence head on a stop-gradient rollout (Appendix D.4)."""
    module = model.denoiser.scn_diffusion_module
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
    spec = loss_fns.psce_bin_spec(module.confidence_module)
    return loss_fns.confidence_loss(
        psce_logits, packed_local, x1_local, atom_mask, bin_spec=spec
    )


def training_forward(
    model,
    batch,
    *,
    train_confidence=None,
    generator=None,
    multiplier=None,
    self_cond_p=None,
):
    """Compute ``L_total = L_MLM + L_diff`` (+ confidence) for one batch."""
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

    # 4. L_diff, teacher-forced on the ground-truth sequence.
    loss_diff, diff_stats = diffusion_loss(
        model,
        batch,
        mpnn_feature_dict,
        multiplier=multiplier,
        self_cond_p=self_cond_p,
        generator=generator,
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
            model, batch, mpnn_feature_dict, batch["aatype"].long()
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
