"""
PXDesign-d composite training loss.

Equation 4 from the PXDesign technical report (p. 24):

    L = (0.03 · L_disto + 1.0 · L_LDDT) · 1{σ̂ < 4 Å}  +  4.0 · L_MSE

- L_MSE is over all heavy atoms (target included — the report explicitly notes
  target coords are NOT frozen during training).
- L_LDDT and L_disto are gated by the per-sample σ being < 4 Å — only at low
  noise do we ask the model to be geometrically tight.

We reuse Protenix's `SmoothLDDTLoss` (Algorithm 27 in AF3) and use the
distogram heads from `heads.py`. We do NOT use Protenix's `MSELoss` directly:
that class applies a `weighted_rigid_align` and per-type weights (DNA/RNA/ligand)
which the PXDesign report does not mention. We write a plain heavy-atom MSE
matching the report's wording.
"""
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.model.loss import SmoothLDDTLoss
from protenix.metrics.rmsd import weighted_rigid_align

from pxdesign_train.sidechain.losses import (
    sidechain_global_frame_aligned_loss,
    sidechain_local_loss,
)


class PXDesignLoss(nn.Module):
    """Composite loss for PXDesign-d training.

    Args:
        weight_mse:    coefficient on MSE term (4.0 per report eq. 4)
        weight_lddt:   coefficient on smooth-LDDT term (1.0 per report)
        weight_disto:  coefficient on distogram term (0.03 per report)
        sigma_low_threshold: σ-mask cutoff in Å (4.0 per report).
            LDDT and distogram terms are zeroed when σ ≥ this value.
        no_bins:       number of distogram bins (64).
        min_bin:       distogram lower edge in Å (matches Protenix default).
        max_bin:       distogram upper edge in Å.
        lddt_radius:   neighbour radius used for the LDDT mask (15 Å for protein).
        align_before_mse: rigid-align GT to prediction before MSE (AF3 standard).
    """

    def __init__(
        self,
        weight_mse: float = 4.0,
        weight_lddt: float = 1.0,
        weight_disto: float = 0.03,
        sigma_low_threshold: float = 4.0,
        no_bins: int = 64,
        min_bin: float = 2.3125,
        max_bin: float = 21.6875,
        lddt_radius: float = 15.0,
        align_before_mse: bool = True,
        weight_aa: float = 0.0,
        aa_ignore_index: int = -100,
        aa_time_weighting: bool = False,
        aa_time_eps: float = 1e-2,
        weight_sc_local: float = 1.0,
        weight_sc_phys: float = 0.1,
        weight_sc_global: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight_sc_local = weight_sc_local
        self.weight_sc_phys = weight_sc_phys
        self.weight_sc_global = weight_sc_global
        self.weight_mse = weight_mse
        self.weight_lddt = weight_lddt
        self.weight_disto = weight_disto
        self.weight_aa = weight_aa
        self.aa_ignore_index = aa_ignore_index
        self.aa_time_weighting = aa_time_weighting
        self.aa_time_eps = aa_time_eps
        self.sigma_low_threshold = sigma_low_threshold
        self.no_bins = no_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.lddt_radius = lddt_radius
        self.align_before_mse = align_before_mse
        self.eps = eps

        # Protenix's SmoothLDDTLoss takes Python None to mean "no reduction".
        self.smooth_lddt = SmoothLDDTLoss(reduction=None)

    @staticmethod
    def _build_lddt_mask(
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        """Returns [..., N_atom, N_atom] mask of atom pairs within `radius` Å in GT."""
        d = torch.cdist(true_coordinate, true_coordinate)  # [..., N_atom, N_atom]
        within = (d < radius).to(d.dtype)
        pair_valid = coordinate_mask[..., :, None] * coordinate_mask[..., None, :]
        # Exclude self-pairs.
        n = within.shape[-1]
        eye = torch.eye(n, device=d.device, dtype=d.dtype)
        return within * pair_valid * (1 - eye)

    @staticmethod
    def _bin_distances(
        coords: torch.Tensor,
        rep_atom_mask: torch.Tensor,
        no_bins: int,
        min_bin: float,
        max_bin: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one-hot distogram labels and pair-valid mask on representative atoms."""
        rep = rep_atom_mask.bool()
        tok_coords = coords[..., rep, :]                # [..., N_token, 3]
        d = torch.cdist(tok_coords, tok_coords)         # [..., N_token, N_token]
        boundaries = torch.linspace(min_bin, max_bin, steps=no_bins - 1, device=d.device)
        bins = torch.sum(d.unsqueeze(-1) > boundaries, dim=-1)  # [..., N_token, N_token]
        return F.one_hot(bins, no_bins).to(coords.dtype), torch.ones_like(d, dtype=coords.dtype)

    def _mse_term(
        self,
        pred: torch.Tensor,                  # [..., N_sample, N_atom, 3]
        gt_aug: torch.Tensor,                # [..., N_sample, N_atom, 3]
        coordinate_mask: torch.Tensor,       # [..., N_atom]
    ) -> torch.Tensor:
        """Heavy-atom MSE, mean over atoms, mean over samples. Returns [...]."""
        if self.align_before_mse:
            # AF3-style rigid-align GT to prediction with uniform weights.
            with torch.no_grad():
                w = coordinate_mask.float()
                w_sample = w[..., None, :].expand_as(pred[..., 0]).contiguous()
                with torch.amp.autocast("cuda", enabled=False):
                    gt_aligned = weighted_rigid_align(
                        x=gt_aug.float(),
                        x_target=pred.float(),
                        atom_weight=w_sample.float(),
                        stop_gradient=True,
                    ).to(pred.dtype).detach()
        else:
            gt_aligned = gt_aug

        se = ((pred - gt_aligned) ** 2).sum(dim=-1)             # [..., N_sample, N_atom]
        mask = coordinate_mask[..., None, :]                    # [..., 1, N_atom]
        per_sample = (se * mask).sum(dim=-1) / (mask.sum(dim=-1) + self.eps)  # [..., N_sample]
        return per_sample.mean(dim=-1)                          # [...]

    def _aligned_rmsd_and_tm(
        self,
        pred: torch.Tensor,                  # [..., N_sample, N_atom, 3]
        gt_aug: torch.Tensor,                # [..., N_sample, N_atom, 3]
        coordinate_mask: torch.Tensor,       # [..., N_atom]
        metric_atom_mask: Optional[torch.Tensor],
        compute_tm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Kabsch-aligned RMSD, plus optional TM-score, over a selected atom mask."""
        if metric_atom_mask is None:
            zero = pred.sum() * 0.0
            return zero.detach(), zero.detach()

        n_atom = pred.shape[-2]
        n_sample = pred.shape[-3]
        item_shape = pred.shape[:-3]
        item_count = int(torch.tensor(item_shape).prod().item()) if item_shape else 1

        p = pred.reshape(item_count, n_sample, n_atom, 3).reshape(-1, n_atom, 3).float()
        g = gt_aug.reshape(item_count, n_sample, n_atom, 3).reshape(-1, n_atom, 3).float()

        def _prep_mask(m: torch.Tensor) -> torch.Tensor:
            m = m.to(device=pred.device).bool()
            if m.dim() == 1:
                m = m.reshape(1, n_atom).expand(item_count, n_atom)
            else:
                if item_shape:
                    m = m.expand(*item_shape, n_atom).reshape(item_count, n_atom)
                else:
                    m = m.reshape(-1, n_atom)[:1].expand(item_count, n_atom)
            return m[:, None, :].expand(item_count, n_sample, n_atom).reshape(-1, n_atom)

        m = _prep_mask(coordinate_mask) & _prep_mask(metric_atom_mask)
        w = m.float()
        count = w.sum(dim=-1)                                  # [B*S]
        valid = count > 0
        if not valid.any():
            zero = pred.sum() * 0.0
            return zero.detach(), zero.detach()

        denom = count.clamp_min(1.0)[:, None, None]
        gc = (g * w[..., None]).sum(dim=1, keepdim=True) / denom
        pc = (p * w[..., None]).sum(dim=1, keepdim=True) / denom
        g0 = (g - gc) * w[..., None]
        p0 = (p - pc) * w[..., None]

        cov = (g0.transpose(-1, -2) @ p0).float()
        autocast_device = pred.device.type
        with torch.autocast(device_type=autocast_device, enabled=False):
            U, _, Vh = torch.linalg.svd(cov)
            det = torch.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2))
        D = torch.eye(3, device=pred.device, dtype=p.dtype).expand(cov.shape[0], 3, 3).clone()
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)
        R = Vh.transpose(-1, -2) @ D @ U.transpose(-1, -2)
        g_aligned = (g - gc) @ R + pc

        d2 = ((g_aligned - p) ** 2).sum(dim=-1)
        rmsd_per = torch.sqrt((d2 * w).sum(dim=-1) / count.clamp_min(1.0))
        rmsd = rmsd_per[valid].mean()

        if not compute_tm:
            return rmsd.detach(), (pred.sum() * 0.0).detach()

        # Standard protein TM-score length scale; clamp for short crops/fragments.
        d0 = 1.24 * torch.clamp(count.float() - 15.0, min=1.0).pow(1.0 / 3.0) - 1.8
        d0 = d0.clamp_min(0.5)
        tm_per_atom = 1.0 / (1.0 + d2 / (d0[:, None] ** 2))
        tm_per = (tm_per_atom * w).sum(dim=-1) / count.clamp_min(1.0)
        return rmsd.detach(), tm_per[valid].mean().detach()

    def _distogram_term(
        self,
        logits: torch.Tensor,                # [..., N_token, N_token, no_bins]
        true_coord: torch.Tensor,            # [..., N_atom, 3]
        coordinate_mask: torch.Tensor,       # [..., N_atom]
        rep_atom_mask: torch.Tensor,         # [N_atom]
    ) -> torch.Tensor:
        with torch.no_grad():
            true_bins, _ = self._bin_distances(
                true_coord, rep_atom_mask, self.no_bins, self.min_bin, self.max_bin,
            )
            tok_valid = coordinate_mask[..., rep_atom_mask.bool()]      # [..., N_token]
            pair_valid = tok_valid[..., :, None] * tok_valid[..., None, :]  # [..., N_token, N_token]

        # Softmax CE per pair, masked.
        log_probs = F.log_softmax(logits.float(), dim=-1)
        per_pair_ce = -(true_bins * log_probs).sum(dim=-1)  # [..., N_token, N_token]
        per_pair_ce = per_pair_ce * pair_valid
        denom = pair_valid.sum(dim=(-1, -2)) + self.eps
        return per_pair_ce.sum(dim=(-1, -2)) / denom  # [...]

    def _aa_term(
        self,
        aa_logits: torch.Tensor,
        aa_clean: torch.Tensor,
        aa_loss_mask: torch.Tensor,
        aa_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masked cross-entropy + accuracy over design tokens.

        Baseline (``aa_t is None`` or ``aa_time_weighting=False``): plain mean
        CE over masked tokens — a masked-LM objective.

        Masked-diffusion (``aa_time_weighting=True`` and ``aa_t`` given): each
        token's CE is importance-weighted by ``1 / max(aa_t, eps)``, the MDLM /
        absorbing-diffusion weight for a linear masking schedule (mask prob = t).
        This turns the plain masked-LM into the discrete-diffusion ELBO surrogate:
        low-``t`` steps (few masked, high per-token information) get up-weighted.
        We normalise by the summed weights so the scalar stays CE-comparable.
        """
        if aa_clean.dim() == aa_logits.dim():
            aa_clean = aa_clean.squeeze(-1)
        aa_clean = aa_clean.long().to(aa_logits.device)
        aa_loss_mask = aa_loss_mask.bool().to(aa_logits.device)

        # Per-sample (per-sigma) AA logits carry an extra N_sample axis at -3
        # ([..., N_sample, N_token, 20]) while the labels/mask are per-token
        # ([..., N_token]). `torch.gather` does NOT broadcast, so explicitly
        # expand the labels/mask over the sample axis; the CE then averages over
        # samples (== a Monte-Carlo estimate over the EDM sigma distribution) as
        # well as tokens. When there is no sample axis this is a no-op.
        if aa_logits.dim() == aa_clean.dim() + 2:
            n_sample = aa_logits.shape[-3]
            pre, n_tok = aa_clean.shape[:-1], aa_clean.shape[-1]
            aa_clean = aa_clean.reshape(*pre, 1, n_tok).expand(*pre, n_sample, n_tok)
            aa_loss_mask = aa_loss_mask.reshape(*pre, 1, n_tok).expand(*pre, n_sample, n_tok)

        valid = aa_loss_mask & (aa_clean != self.aa_ignore_index)  # [..., (N_sample,) N_token]
        mask_frac = valid.float().mean()
        if not valid.any():
            zero = aa_logits.sum() * 0.0
            return zero, zero.detach(), mask_frac.detach()

        # Per-token CE (keep batch/token structure so we can time-weight).
        logp = torch.log_softmax(aa_logits.float(), dim=-1)
        tgt = aa_clean.clamp_min(0)  # -100 sites are excluded by `valid`
        ce_tok = -logp.gather(-1, tgt[..., None]).squeeze(-1)  # [..., N_token]
        vmask = valid.float()

        if self.aa_time_weighting and aa_t is not None:
            t = torch.as_tensor(aa_t, device=aa_logits.device).float()
            w = 1.0 / t.clamp_min(self.aa_time_eps)  # per-example weight [...]
            while w.dim() < ce_tok.dim():  # broadcast to token axis
                w = w[..., None]
            w = w.expand_as(ce_tok)
            denom = (vmask * w).sum().clamp_min(self.eps)
            ce = (ce_tok * vmask * w).sum() / denom
        else:
            ce = (ce_tok * vmask).sum() / vmask.sum().clamp_min(self.eps)

        correct = (aa_logits.argmax(dim=-1) == aa_clean) & valid
        acc = correct.float().sum() / vmask.sum().clamp_min(self.eps)
        return ce, acc.detach(), mask_frac.detach()

    def forward(
        self,
        pred_coordinate: torch.Tensor,       # [..., N_sample, N_atom, 3]
        gt_coordinate_aug: torch.Tensor,     # [..., N_sample, N_atom, 3]
        sigma: torch.Tensor,                 # [..., N_sample]
        coordinate_mask: torch.Tensor,       # [..., N_atom]
        rep_atom_mask: torch.Tensor,         # [N_atom]
        backbone_atom_mask: Optional[torch.Tensor] = None,  # [..., N_atom] M1: exclude binder side chains from L_bb
        distogram_logits: Optional[torch.Tensor] = None,  # [..., N_token, N_token, no_bins]
        aa_logits: Optional[torch.Tensor] = None,          # [..., N_token, 20]
        aa_clean: Optional[torch.Tensor] = None,           # [..., N_token]
        aa_loss_mask: Optional[torch.Tensor] = None,       # [..., N_token]
        aa_t: Optional[torch.Tensor] = None,               # [...] masked-diffusion time
        sc_pred_local: Optional[torch.Tensor] = None,      # [..., L, A, 3]
        sc_pred_global: Optional[torch.Tensor] = None,     # [..., L, A, 3]
        sc_gt_local: Optional[torch.Tensor] = None,        # [..., L, A, 3]
        sc_frame_R: Optional[torch.Tensor] = None,         # [..., L, 3, 3]
        sc_frame_t: Optional[torch.Tensor] = None,         # [..., L, 3]
        sc_atom_mask: Optional[torch.Tensor] = None,       # [..., L, A] bool
        sc_type_match: Optional[torch.Tensor] = None,      # [..., L] bool (pred==gt type)
        sc_phys: Optional[torch.Tensor] = None,            # precomputed physical loss scalar
        sc_global: Optional[torch.Tensor] = None,          # predicted-frame pseudo-target aux (scalar)
        post_pred_coordinate: Optional[torch.Tensor] = None,   # [..., N_sample, N_atom, 3]
        post_gt_coordinate_aug: Optional[torch.Tensor] = None, # [..., N_sample, N_atom, 3]
        post_aa_logits: Optional[torch.Tensor] = None,         # [..., N_token, 20]
        eval_ca_atom_mask: Optional[torch.Tensor] = None,       # [..., N_atom] bool
        eval_backbone_atom_mask: Optional[torch.Tensor] = None, # [..., N_atom] bool
        weight_bb_post: float = 1.0,
        weight_aa_post: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute the composite loss.

        Returns a dict with keys: "loss", "mse", "lddt", "distogram", "sigma_low_frac".
        Each component is a scalar (mean over batch).
        """
        # M1: backbone-only target — zero out binder side-chain atoms so L_bb
        # (MSE + LDDT + distogram share coordinate_mask) never supervises them.
        # S_phi is the sole side-chain generator.
        if backbone_atom_mask is not None:
            coordinate_mask = coordinate_mask * backbone_atom_mask.to(
                device=coordinate_mask.device, dtype=coordinate_mask.dtype
            )

        # σ-mask: 1 where sigma < threshold, else 0. Per (batch, sample).
        sigma_low = (sigma < self.sigma_low_threshold).to(pred_coordinate.dtype)

        # --- MSE (always on) ---
        mse = self._mse_term(pred_coordinate, gt_coordinate_aug, coordinate_mask)  # [...]
        with torch.no_grad():
            ca_rmsd, tm_score = self._aligned_rmsd_and_tm(
                pred_coordinate,
                gt_coordinate_aug,
                coordinate_mask,
                eval_ca_atom_mask,
                compute_tm=True,
            )
            bb_rmsd, _ = self._aligned_rmsd_and_tm(
                pred_coordinate,
                gt_coordinate_aug,
                coordinate_mask,
                eval_backbone_atom_mask,
                compute_tm=False,
            )

        # --- Smooth LDDT (gated) ---
        # SmoothLDDTLoss takes [..., N_sample, N_atom, 3] and returns per-sample lddt loss;
        # we use dense_forward + reduction='none' to get a per-batch scalar after averaging.
        # We compute LDDT under σ-mask by multiplying loss by mean σ-mask over samples.
        gt_single = gt_coordinate_aug[..., 0, :, :]  # use first-sample GT for the mask
        lddt_mask = self._build_lddt_mask(gt_single, coordinate_mask, self.lddt_radius)
        lddt_per_batch = self.smooth_lddt.dense_forward(
            pred_coordinate=pred_coordinate,
            true_coordinate=gt_single,
            lddt_mask=lddt_mask,
        )  # smooth_lddt with reduction='none' returns [...]
        # Apply σ-mask: average over samples where σ < threshold.
        gate_lddt = sigma_low.mean(dim=-1)  # [...]
        lddt = lddt_per_batch * gate_lddt

        # --- Distogram (gated) ---
        if distogram_logits is not None:
            disto = self._distogram_term(
                distogram_logits, gt_single, coordinate_mask, rep_atom_mask,
            )
            disto = disto * gate_lddt
        else:
            disto = torch.zeros_like(mse)

        total = (
            self.weight_mse * mse
            + self.weight_lddt * lddt
            + self.weight_disto * disto
        )

        if aa_logits is not None and aa_clean is not None and aa_loss_mask is not None:
            aa_ce, aa_acc, aa_mask_frac = self._aa_term(
                aa_logits=aa_logits,
                aa_clean=aa_clean,
                aa_loss_mask=aa_loss_mask,
                aa_t=aa_t,
            )
            total = total + self.weight_aa * aa_ce
        else:
            aa_ce = total.sum() * 0.0
            aa_acc = torch.zeros_like(aa_ce).detach()
            aa_mask_frac = torch.zeros_like(aa_ce).detach()

        # --- Side-chain terms (Stage II-A onwards) ---
        # Main coordinate loss. Current S_phi emits GLOBAL coordinates, so the
        # preferred path compares against stopgrad(F_hat) y_gt_local directly in
        # the global frame. The legacy local-frame path remains for older callers.
        # The physical loss (global-frame, needs ideal-geometry tables) is
        # computed upstream and passed in as `sc_phys`.
        has_global_sc = (
            sc_pred_global is not None
            and sc_gt_local is not None
            and sc_frame_R is not None
            and sc_frame_t is not None
            and sc_atom_mask is not None
        )
        has_local_sc = sc_pred_local is not None and sc_gt_local is not None and sc_atom_mask is not None
        if has_global_sc or has_local_sc:
            pred_ref = sc_pred_global if has_global_sc else sc_pred_local

            def _match_pred_leading(x: torch.Tensor, trailing_ndim: int) -> torch.Tensor:
                """Match per-item side-chain labels to per-sigma predictions.

                In per-sigma training side-chain predictions may be flattened from
                [B, S, L, A, 3] to [B*S, L, A, 3], while GT targets remain
                [B, L, A, 3]. Repeat each batch item over S so loss terms are
                row-aligned with the predictions.
                """
                # pred_ref is ALWAYS [..., L, A, 3] (4 dims), so its batch lead is
                # shape[:-3] regardless of x's own trailing rank. Slicing it with x's
                # trailing_ndim instead folded the TOKEN axis into the lead: for
                # sc_frame_t [B, L, 3] (trailing_ndim=2) that gave pred_lead=[B_, L] and
                # then expand() to [B_, L, L, 3] -> RuntimeError. It crashed every
                # per-sigma / co-evolution run and reddened 4 unit tests.
                pred_lead = pred_ref.shape[:-3]
                x_lead = x.shape[:-trailing_ndim]
                if x_lead == pred_lead:
                    return x
                if x.dim() == trailing_ndim:
                    return x
                if (
                    len(pred_lead) == 1
                    and len(x_lead) == 1
                    and x_lead[0] > 0
                    and pred_lead[0] % x_lead[0] == 0
                ):
                    return x.repeat_interleave(pred_lead[0] // x_lead[0], dim=0)
                return x.expand(*pred_lead, *x.shape[-trailing_ndim:])

            sc_gt_local = _match_pred_leading(sc_gt_local, trailing_ndim=3)
            if has_global_sc:
                sc_frame_R = _match_pred_leading(sc_frame_R, trailing_ndim=3)
                sc_frame_t = _match_pred_leading(sc_frame_t, trailing_ndim=2)
            coord_mask = _match_pred_leading(sc_atom_mask, trailing_ndim=2).bool()
            if sc_type_match is not None:
                sc_type_match = sc_type_match.bool()
                target_lead = coord_mask.shape[:-1]  # [..., L]
                if sc_type_match.shape != target_lead:
                    if sc_type_match.dim() == 1:
                        sc_type_match = sc_type_match.expand(*target_lead)
                    elif (
                        len(target_lead) == 2
                        and sc_type_match.dim() == 2
                        and sc_type_match.shape[0] > 0
                        and target_lead[0] % sc_type_match.shape[0] == 0
                        and target_lead[1] == sc_type_match.shape[1]
                    ):
                        sc_type_match = sc_type_match.repeat_interleave(
                            target_lead[0] // sc_type_match.shape[0], dim=0
                        )
                    else:
                        sc_type_match = sc_type_match.expand(*target_lead)
                coord_mask = coord_mask & sc_type_match.bool()[..., None]
            if has_global_sc:
                sc_local = sidechain_global_frame_aligned_loss(
                    sc_pred_global,
                    sc_gt_local,
                    sc_frame_R,
                    sc_frame_t,
                    coord_mask,
                    eps=self.eps,
                )
            else:
                sc_local = sidechain_local_loss(sc_pred_local, sc_gt_local, coord_mask, eps=self.eps)
            sc_phys_val = sc_phys if sc_phys is not None else total.sum() * 0.0
            if has_global_sc:
                # In the global-output path, the predicted-frame-aligned loss is
                # the primary coordinate loss, not a second auxiliary term.
                sc_global_val = sc_local
                total = total + self.weight_sc_local * sc_local + self.weight_sc_phys * sc_phys_val
            else:
                # Backward-compatible local path: keep the old optional aux.
                sc_global_val = sc_global if sc_global is not None else total.sum() * 0.0
                total = (total + self.weight_sc_local * sc_local
                         + self.weight_sc_phys * sc_phys_val
                         + self.weight_sc_global * sc_global_val)
        else:
            sc_local = total.sum() * 0.0
            sc_phys_val = total.sum() * 0.0
            sc_global_val = total.sum() * 0.0

        # --- Post-refinement terms (Stage II-B cycle closure) ---
        # Side-chain-informed backbone refinement: reuse the coord / AA loss on
        # the refined (post) outputs (Overleaf L_bb^post, L_aa^post).
        if post_pred_coordinate is not None and post_gt_coordinate_aug is not None:
            bb_post = self._mse_term(post_pred_coordinate, post_gt_coordinate_aug, coordinate_mask)
            total = total + weight_bb_post * bb_post
        else:
            bb_post = total.sum() * 0.0
        if post_aa_logits is not None and aa_clean is not None and aa_loss_mask is not None:
            aa_post_ce, _, _ = self._aa_term(post_aa_logits, aa_clean, aa_loss_mask, aa_t)
            total = total + weight_aa_post * aa_post_ce
        else:
            aa_post_ce = total.sum() * 0.0

        return {
            "loss": total.mean(),
            "mse": mse.mean().detach(),
            "ca_rmsd": ca_rmsd,
            "bb_rmsd": bb_rmsd,
            "tm_score": tm_score,
            "lddt": lddt.mean().detach(),
            "distogram": disto.mean().detach(),
            "sigma_low_frac": sigma_low.mean().detach(),
            "aa_ce": aa_ce.detach(),
            "aa_acc": aa_acc,
            "aa_mask_frac": aa_mask_frac,
            "sc_local": sc_local.detach(),
            "sc_phys": sc_phys_val.detach(),
            "sc_global": sc_global_val.detach(),
            "bb_post": bb_post.detach(),
            "aa_post": aa_post_ce.detach(),
        }
