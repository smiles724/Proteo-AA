"""The FaMPNN training forward pass, as the original training code runs it.

Why this file exists: the released ``fampnn`` package is inference only.
``SidechainDiffusionModule.sidechain_diffusion`` still takes an ``is_sampling``
flag but the training branch was deleted from it, and the loss module, the
LightningModule and the whole loop were never shipped. What is reconstructed
here is the deleted branch, transcribed from the code the released weights were
trained by -- ``allatom_design`` (commit ``51c9d53``), the research tree
``fampnn`` was factored out of:

    allatom_design/model/seq_denoiser/sd_model.py                SeqDenoiser.forward
    .../denoisers/fampnn_denoiser.py                             FAMPNNDenoiser.forward
    .../denoisers/sidechain_diffusion/scn_diffusion_mlp.py       the ``not is_sampling``
                                                                 branch + mini_rollout
    allatom_design/model/seq_denoiser/sd_loss.py                 SDLoss

Every module that *is* shipped -- the MAR and EDM interpolants, the encoder, the
denoising MLP, the confidence head, the local-frame transforms -- is driven
unchanged; only the orchestration around them is written here.

The shape of one step:

1. **MAR** draws a keep probability ``t = sqrt(u)`` and hides sequence and side
   chains. Sequence and side chains are masked *separately*: ``drop_sidechains``
   hides a further random fraction of the side chains at positions whose
   identity is still visible, so the encoder routinely sees "identity known,
   conformation unknown" -- which is exactly what packing is.
2. **The encoder** runs on the masked structure and predicts the sequence.
3. **The side-chain denoiser** takes one EDM step per noise level on clean
   local-frame targets, conditioned on ``h_V`` and teacher-forced on the
   *ground-truth* sequence, with the conditioning cloned
   ``training_batch_size_mult`` (8) times so 8 noise levels are seen per example.
4. **The confidence head**, on roughly one step in ``subsample_train_iter_mult``
   (8), scores a full 50-step rollout taken under ``no_grad``.

Three things a reimplementation from the preprint gets wrong, all fixed here
against the original:

* **The diffusion term supervises every resolved side chain, not only the hidden
  ones.** Its mask is ``x_mask`` (missing atoms and padding removed) times
  "the backbone frame exists" -- ``scn_mlm_mask`` does not appear. The visible
  side chains are supervision too: the denoiser starts from pure noise in the
  local frame regardless, so reproducing a conformation whose atoms the *encoder*
  was shown is a real prediction problem, not a copy. (The confidence term is
  the one that *is* restricted to hidden side chains, and it is restricted here.)
* **Ghost slots are supervised to zero.** ``x_mask`` removes missing atoms but
  keeps the slots a residue type does not have; their local-frame target is
  exactly ``0``. The MLP always emits 33 atoms, and this is what teaches it to
  put the nonexistent ones at the origin.
* **Structural noise ("the 0.3 A model") is the model's own ``augment_eps``,**
  applied inside ``ProteinFeatures`` to the encoder's atom14 input in train mode.
  It never touches the diffusion target. Adding noise in the data pipeline
  instead both corrupts the target and double-counts against this.

One upstream wart is worked around: ``MAR`` is a plain class written like an
``nn.Module`` -- it has ``forward``, calls ``super().__init__()`` and reads
``self.training``, but is not callable and never receives ``model.train()``. So
``forward`` is called directly and the mode mirrored across. Inference never
touches ``model.interpolant``, which is why this went unnoticed.
"""

from dataclasses import dataclass, field
from functools import partial

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
        # loss_main is L_seq + L_scn, which every step has. The total only
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


# ---- the masks the original loss reads off its dataset ---------------------


def batch_masks(batch):
    """Rebuild the three dataset-level masks ``SDLoss`` indexes into.

    ``process_single_pdb`` computes these and the original dataset carries them
    through cropping and padding; a ``pxf`` batch carries the smaller set in
    :data:`REQUIRED_KEYS`, so they are derived here instead. The derivations are
    equalities, not approximations, because padding is zero-filled:

    ``atom_mask``     (upstream ``all_atom_mask``) -- the atom exists for this
        residue type and is present in the structure. Used to *build* the local
        frames and to score the confidence head.
    ``x_mask``        ``1 - missing_atom_mask``, zeroed at padding -- the loss
        mask for the diffusion term. Note it keeps **ghost** slots: an atom the
        residue type does not have is not "missing", it is a slot whose target
        is the origin.
    ``seq_unk_mask``  the residue's identity is ``X``. Excluded from the sequence
        and confidence terms so the model is never trained to emit its own mask
        token.

    One deliberate divergence. Upstream's ``all_atom_mask`` comes from the parse,
    so a *non-standard* residue -- read as ``X``, side-chain atoms kept -- has
    side-chain bits set that ``STANDARD_ATOM_MASK_WITH_X[X]`` does not, and the
    original therefore gives those atoms their real local coordinates as a
    target. Derived here, they are ghost slots instead and their target is the
    origin. That is the better behaviour: the encoder is never shown them (its
    mask is built from the same table) and the denoiser is conditioned on a
    one-hot ``X``, so the original is asking it to predict a side chain from an
    identity that does not name one. The residue is already excluded from
    ``L_seq`` and ``L_psce`` either way.
    """
    rc = _constants()
    from fampnn.data.data import get_rc_tensor

    aatype = batch["aatype"].long()
    seq_mask = batch["seq_mask"]
    missing = batch["missing_atom_mask"]
    exists = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)
    present = (1.0 - missing) * seq_mask.unsqueeze(-1)
    return {
        "atom_mask": exists * present,
        "x_mask": present,
        "seq_unk_mask": (aatype == rc.restype_order_with_x["X"]).to(seq_mask.dtype),
    }


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


def frame_targets(model, batch, *, scn_mlm_mask=None):
    """Clean side chains in the per-residue backbone frame, plus every mask.

    Uses upstream's own frame construction (AF2 Algorithm 21 via OpenFold), so
    the targets live in exactly the space inference denoises in. Returns a dict:

    ``local``        ``[b, n, 33, 3]`` the diffusion target.
    ``loss_mask``    ``[b, n, 33]`` what the diffusion term scores: resolved or
        ghost, not padding, and the residue has a backbone frame.
    ``atom_mask``    ``[b, n, 33]`` what actually exists in the structure, which
        is the narrower set the confidence head is scored on.
    ``frames_exist`` ``[b, n]``.

    ``scn_mlm_mask`` restricts ``loss_mask`` to the residues whose side chain the
    interpolant hid. The original does **not** do this for the diffusion term --
    it is an ablation switch, off by default.
    """
    rc = _constants()
    from fampnn.data.data import transform_sidechain_frame

    masks = batch_masks(batch)
    atom_mask = masks["atom_mask"]
    x = batch["x"]
    local, frames_exist = transform_sidechain_frame(
        x[..., rc.non_bb_idxs, :],
        x[..., rc.bb_idxs, :],
        atom_mask[..., rc.non_bb_idxs],
        atom_mask[..., rc.bb_idxs],
        to_local=True,
    )
    loss_mask = masks["x_mask"][..., rc.non_bb_idxs] * frames_exist.unsqueeze(-1)
    if scn_mlm_mask is not None:
        loss_mask = loss_mask * (1.0 - scn_mlm_mask.to(loss_mask.dtype)).unsqueeze(-1)
    return {
        "local": local,
        "loss_mask": loss_mask,
        "atom_mask": atom_mask[..., rc.non_bb_idxs],
        "frames_exist": frames_exist,
        "seq_unk_mask": masks["seq_unk_mask"],
    }


def sidechain_targets(model, batch, *, scn_mlm_mask=None):
    """``(local target, diffusion loss mask)`` -- the two-value view of
    :func:`frame_targets`."""
    targets = frame_targets(model, batch, scn_mlm_mask=scn_mlm_mask)
    return targets["local"], targets["loss_mask"]


def _clone(tensor, times):
    """Tile along a new leading axis and fold it into the batch axis.

    ``repeat(x, "b ... -> (m b) ...")``, the original's ordering: clone *blocks*
    of the batch, not interleaved copies of each example.
    """
    if times <= 1:
        return tensor
    return tensor.unsqueeze(0).expand(times, *tensor.shape).reshape(
        times * tensor.shape[0], *tensor.shape[1:]
    )


# ---- the diffusion term ----------------------------------------------------


def diffusion_loss(
    model,
    batch,
    mpnn_feature_dict,
    *,
    multiplier=None,
    self_cond_p=None,
    generator=None,
    scn_mlm_mask=None,
    t_scd=None,
    reduction=loss_fns.DEFAULT_SIDECHAIN_REDUCTION,
):
    """One denoising step per (example, noise level), teacher-forced on GT sequence.

    ``multiplier`` clones of the conditioning are drawn, each with its own noise
    level from the EDM interpolant's lognormal schedule -- the MLP is cheap next
    to the encoder, so the encoder's output is reused across all of them.

    ``t_scd`` pins the diffusion time instead of sampling it, which is how the
    original evaluates a validation curve at fixed noise levels.

    ``scn_mlm_mask`` is the ablation described in :func:`frame_targets`; leave it
    ``None`` for the original objective.
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

    targets = frame_targets(model, batch, scn_mlm_mask=scn_mlm_mask)
    h_V = mpnn_feature_dict["h_V"]
    seq_mask = batch["seq_mask"]
    aatype = batch["aatype"].long()

    x1_rep = _clone(targets["local"], multiplier)
    mask_rep = _clone(targets["loss_mask"], multiplier)
    h_V_rep = _clone(h_V, multiplier)
    seq_mask_rep = _clone(seq_mask, multiplier)
    aatype_rep = _clone(aatype, multiplier)

    t = None
    if t_scd is not None:
        t = torch.full((x1_rep.shape[0],), float(t_scd), device=x1_rep.device)
    # The interpolant's own forward, so the noise draw, the x1 target and the
    # loss weight all come from one place and stay consistent with sampling.
    interpolated = interpolant({"x": x1_rep, "aatype": aatype_rep}, t=t)
    xt = interpolated["x_noised"]
    x_target = interpolated["x_target"]
    t = interpolated["t"]
    weight = interpolated["loss_weight_t"]

    denoiser_fn = denoiser
    self_conditioned = False
    if module.use_self_conditioning and self_cond_p > 0:
        # Draw on the generator's own device (CPU); the value is only compared.
        if float(torch.rand((), generator=generator)) < self_cond_p:
            # Standard self-conditioning: a detached first pass supplies the
            # estimate the graded pass conditions on.
            with torch.no_grad():
                primed, _ = denoiser_fn(xt, aatype_rep, t, h_V_rep, seq_mask=seq_mask_rep)
            # Sidestep an AMP cache bug (PyTorch issue #65766), as upstream does.
            torch.clear_autocast_cache()
            denoiser_fn = partial(denoiser_fn, x_scn_self_cond=primed)
            self_conditioned = True

    x1_pred, _ = denoiser_fn(xt, aatype_rep, t, h_V_rep, seq_mask=seq_mask_rep)
    loss, stats = loss_fns.sidechain_diffusion_loss(
        x1_pred, x_target, weight, mask_rep, reduction=reduction
    )
    stats["noise_clones"] = torch.tensor(float(multiplier))
    stats["self_conditioned"] = torch.tensor(float(self_conditioned))
    stats["sigma_scn_mean"] = interpolant.sigma(t).mean().detach()
    if scn_mlm_mask is not None:
        with torch.no_grad():
            visible = (scn_mlm_mask * seq_mask).sum()
            stats["hidden_sidechain_fraction"] = 1.0 - visible / seq_mask.sum().clamp_min(1)
    return loss, stats


# ---- the confidence term ---------------------------------------------------


def _step_scale(cfg):
    """The constant the rollout scales its vector field by.

    The original carries a ``NoiseSchedule`` config (``name: step_scale, c: 1.5``)
    where the released integrator takes a bare ``step_scale`` float. Only the
    constant-scale schedule has an equivalent, so anything else is refused rather
    than silently ignored.
    """
    schedule = getattr(cfg, "noise_schedule", None)
    if schedule is None:
        return float(getattr(cfg, "step_scale", 1.5))
    name = getattr(schedule, "name", "step_scale")
    if name != "step_scale":
        raise ValueError(
            f"the released euler_step implements a constant step scale only, but the "
            f"confidence rollout is configured with noise schedule {name!r}"
        )
    return float(schedule.c)


@torch.no_grad()
def mini_rollout(module, h_V, aatype, seq_mask):
    """The 50-step rollout the confidence head is trained to score.

    Reconstructs ``SidechainDiffusionModule.mini_rollout``, which the released
    package dropped along with the rest of the training branch. It stays in the
    **local frame** throughout: the head scores local coordinates, and a round
    trip through global coordinates would need a backbone that the training
    batch and the encoder's (possibly noise-augmented) view do not share.

    Dropout is disabled for the rollout and the previous mode restored
    afterwards -- upstream calls ``self.train()`` unconditionally at the end,
    which silently leaves dropout on if the rollout ran during validation.
    """
    rc = _constants()
    from fampnn import sampling_utils

    cfg = module.cfg.confidence_module.scn_diffusion
    batch_size, length, _ = h_V.shape
    timesteps = sampling_utils.get_timesteps_from_schedule(**cfg.timestep_schedule)
    timesteps = timesteps[None].expand(batch_size, -1).to(h_V.device)
    churn_cfg = dict(cfg.churn_cfg)
    step_scale = _step_scale(cfg)

    was_training = module.training
    module.eval()
    try:
        xt = module.scn_interpolant.sample_prior(
            (batch_size, length, len(rc.non_bb_idxs), 3), h_V.device
        )
        denoiser_fn = partial(
            module.scn_denoiser, aatype=aatype, h_V=h_V, seq_mask=seq_mask
        )
        for i in range(cfg.num_steps):
            t, t_next = timesteps[:, i], timesteps[:, i + 1]
            xt, t = module.scn_interpolant.churn(xt, t, churn_cfg=churn_cfg)
            xt, aux = module.scn_interpolant.euler_step(
                denoiser_fn, xt, t=t, t_next=t_next, step_scale=step_scale, cfg_cfg=None
            )
            if module.use_self_conditioning:
                denoiser_fn = partial(denoiser_fn, x_scn_self_cond=aux["x1_pred"])
    finally:
        module.train(was_training)
    return xt


def confidence_loss(model, batch, mpnn_feature_dict, aatype, *, scn_mlm_mask=None):
    """Train the confidence head on a stop-gradient rollout.

    Unlike the diffusion term, this one **is** restricted to the side chains the
    interpolant hid, and to residues with a known identity and a backbone frame:
    a psCE head trained to call a side chain it was handed "zero error" would be
    calibrated for a case that never arises at deployment, where
    ``sidechain_pack`` hides every side chain.

    The restriction applies to what is *scored*, never to what the head is
    *shown*. The rollout packs every residue, which is what the head's input
    looks like at deployment.
    """
    module = model.denoiser.scn_diffusion_module
    targets = frame_targets(model, batch)
    seq_mask = batch["seq_mask"]
    detached = {
        k: (v.detach() if torch.is_tensor(v) else v) for k, v in mpnn_feature_dict.items()
    }
    rollout = mini_rollout(module, detached["h_V"], aatype, seq_mask).detach()

    psce_logits, psce = module.confidence_module(
        rollout,
        detached,
        aatype.detach(),
        seq_mask.detach(),
        batch["residue_index"].detach(),
        batch["chain_index"].detach(),
    )

    # Residue-level: hidden side chain, known identity, not padding, has a frame.
    residues = seq_mask * targets["frames_exist"] * (1.0 - targets["seq_unk_mask"])
    if scn_mlm_mask is not None:
        residues = residues * (1.0 - scn_mlm_mask.to(residues.dtype))
    # Atom-level: only atoms that really exist -- a ghost slot has no error to
    # be confident about.
    scored = residues.unsqueeze(-1) * targets["atom_mask"]

    spec = loss_fns.psce_bin_spec(module.confidence_module)
    loss, stats = loss_fns.confidence_loss(
        psce_logits, rollout, targets["local"], scored, bin_spec=spec
    )
    with torch.no_grad():
        # Per-residue RMSD of the rollout, averaged over the scored residues:
        # the quantity the head is trying to predict, in Angstroms.
        squared = (scored.unsqueeze(-1) * (targets["local"] - rollout)).pow(2)
        per_residue = squared.sum(dim=(-1, -2)) / scored.sum(dim=-1).clamp(min=1)
        rmsd = (per_residue.sqrt() * residues).sum(-1) / residues.sum(-1).clamp(min=1)
        stats["rollout_scn_rmsd"] = rmsd.mean()
        # ... and whether the head's own number tracks it. A falling L_psce with
        # a flat correlation means the head has learned the error distribution
        # rather than which atoms are wrong, which the loss alone will not show.
        correlation = _correlation(
            psce, torch.norm(rollout - targets["local"], dim=-1), scored
        )
        if correlation is not None:
            stats["sce_vs_psce_r"] = correlation
    return loss, stats


def _correlation(x, y, mask):
    """Pearson r over the masked entries, or None when it is undefined.

    Returning None rather than NaN keeps the stat out of the log entirely on a
    degenerate batch, instead of poisoning the window average it feeds.
    """
    keep = mask.bool()
    a, b = x[keep].float(), y[keep].float()
    if a.numel() < 2:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    if float(denominator) == 0.0:
        return None
    return (a * b).sum() / denominator


# ---- one training step -----------------------------------------------------


def training_forward(
    model,
    batch,
    *,
    train_confidence=None,
    generator=None,
    multiplier=None,
    self_cond_p=None,
    t_scd=None,
    settings=loss_fns.DEFAULT_LOSS_SETTINGS,
    hidden_sidechains_only=False,
):
    """Compute ``L_seq + L_scn`` (+ ``L_psce``) for one batch.

    ``hidden_sidechains_only=True`` restricts the diffusion target to the side
    chains the interpolant hid. The original supervises all of them; this is an
    ablation, not the objective.
    """
    missing = [key for key in REQUIRED_KEYS if key not in batch]
    if missing:
        raise ValueError(f"training batch is missing {missing}")
    module = model.denoiser.scn_diffusion_module

    # 1. Mask sequence and side chains (MAR); drop_sidechains needs train mode.
    model.interpolant.training = bool(model.training)
    mar_out = model.interpolant.forward(batch)

    # 2. Full-atom encoder: sequence logits and node embeddings. Structural noise
    #    ("the 0.3 A model") is applied inside ProteinFeatures on the way in,
    #    gated by train mode and the model's own augment_eps.
    atom_mask_noised = encoder_inputs(model, batch, mar_out)
    seq_logits, mpnn_feature_dict = model.denoiser.seq_design_module(
        mar_out["x_noised"],
        mar_out["aatype_noised"],
        batch["seq_mask"],
        atom_mask_noised,
        batch["residue_index"],
        batch["chain_index"],
    )

    # 3. L_seq on the positions the interpolant masked.
    masks = batch_masks(batch)
    loss_mlm, stats = loss_fns.sequence_mlm_loss(
        seq_logits,
        batch["aatype"],
        mar_out["seq_mlm_mask"],
        batch["seq_mask"],
        seq_unk_mask=masks["seq_unk_mask"],
        settings=settings,
    )

    # 4. L_scn on every resolved side chain, teacher-forced on the true sequence.
    scn_target_mask = mar_out["scn_mlm_mask"] if hidden_sidechains_only else None
    loss_diff, diff_stats = diffusion_loss(
        model,
        batch,
        mpnn_feature_dict,
        multiplier=multiplier,
        self_cond_p=self_cond_p,
        generator=generator,
        scn_mlm_mask=scn_target_mask,
        t_scd=t_scd,
        reduction=settings.sidechain_reduction,
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
            scn_mlm_mask=mar_out["scn_mlm_mask"],
        )
        stats.update(conf_stats)

    with torch.no_grad():
        stats["keep_fraction"] = mar_out["seq_mlm_mask"].sum() / batch[
            "seq_mask"
        ].sum().clamp_min(1)
        stats["sidechain_keep_fraction"] = mar_out["scn_mlm_mask"].sum() / batch[
            "seq_mask"
        ].sum().clamp_min(1)
    return StepOutput(
        total=loss_fns.total_loss(loss_mlm, loss_diff, loss_conf, settings=settings),
        mlm=loss_mlm,
        diffusion=loss_diff,
        confidence=loss_conf,
        stats=stats,
    )
