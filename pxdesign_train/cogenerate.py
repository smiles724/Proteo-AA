"""
Joint sequence-structure co-generation.

Merges PXDesign-d (structure) and the ProteinMPNN stage (sequence) into ONE
reverse process: at every denoising step we run the DiffusionModule once (which
denoises coordinates AND, via the forward hook, exposes the structure-aware
`a_token`), take an EDM step on the coordinates, and use `a_token` to predict +
progressively unmask residue identities. Because the AA head reads the SAME
a_token that sees the binder's own noisy backbone, sequence is generated
conditioned on the structure being generated — the co-design the author asked for.

This is a MINIMAL, correctness-first sampler (deterministic EDM Euler step, no
predictor-corrector / churn). Quality tuning is out of scope here.
"""
from typing import Any, Optional

import logging

import torch

from pxdesign_train.sampler import build_aa20_to_restype36, _unmask_counts


# 20-AA index -> 3-letter (matches PRO_STD order used elsewhere).
_AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


@torch.no_grad()
def cogenerate(
    model,
    input_feature_dict: dict[str, Any],
    N_step: int = 20,
    temperature: float = 0.0,
    chunk_size: Optional[int] = None,
    sidechain_cycle: bool = False,
    sc_start_frac: float = 0.5,
    stop_on_seq_stable: bool = False,
    seq_patience: int = 3,
) -> dict[str, Any]:
    """Co-generate (backbone coordinates, residue sequence) from noise.

    Requires model.aa_input_source == "diffusion_internal" (needs a_token).
    `input_feature_dict` must be a real featurized input (for N_atom,
    atom_to_token_idx, design_token_mask, restype template, ...); its GT
    coordinates are NOT used — structure starts from noise.

    Returns {coordinate, sequence (aa20 per design token, -1 elsewhere),
             trajectory}.
    """
    from protenix.model.protenix import update_input_feature_dict

    assert model.aa_input_source == "diffusion_internal", (
        "cogenerate needs input_source='diffusion_internal' (a_token)."
    )
    model.eval()

    feat = dict(input_feature_dict)
    feat = model.diffusion_module.diffusion_conditioning.relpe.generate_relp(feat)
    feat = update_input_feature_dict(feat)
    s_inputs, s_trunk, z_trunk = model.get_condition_embedding(feat, chunk_size=chunk_size)

    device = s_inputs.device
    dtype = s_inputs.dtype
    N_atom = feat["atom_to_token_idx"].shape[-1]

    dtm = feat["design_token_mask"].bool()
    while dtm.dim() > 1:
        dtm = dtm.squeeze(0)
    positions = dtm.nonzero(as_tuple=False).squeeze(-1)
    N_token = dtm.shape[-1]

    aa20_to_36, xpb = build_aa20_to_restype36()
    aa20_to_36 = aa20_to_36.to(device)

    restype = feat["restype"].clone()
    while restype.dim() > 2:
        restype = restype.squeeze(0)
    n_ch = restype.shape[-1]

    def one_hot(ch):
        v = torch.zeros(n_ch, device=device, dtype=restype.dtype)
        v[ch] = 1.0
        return v

    for i in positions.tolist():  # design region starts fully masked (xpb)
        restype[i] = one_hot(xpb)

    sampled = torch.full((N_token,), -1, dtype=torch.long, device=device)
    still = torch.zeros(N_token, dtype=torch.bool, device=device)
    still[positions] = True  # True = still masked

    noise_schedule = model.inference_noise_scheduler(
        N_step=N_step, device=device, dtype=dtype
    )
    x = noise_schedule[0] * torch.randn(1, N_atom, 3, device=device, dtype=dtype)

    counts = _unmask_counts(int(positions.numel()), N_step)
    counts = counts + [0] * (max(0, len(noise_schedule) - 1 - len(counts)))
    trajectory = []

    # --- inference-side cycle setup (Overleaf iterative co-evolution) ---
    sc_enabled = (sidechain_cycle and getattr(model, "enable_sidechain", False)
                  and getattr(model, "enable_coevolution", False))
    bb_atom_idx = {}
    # 4-wide (N, CA, C, O) atom indices — the atom rows the ATOM-level (q_direct)
    # feedback writes q'_bb back into. Only available from the featurizer's by-name
    # resolution; the positional fallback below cannot identify O, so q_direct simply
    # stays off there rather than guessing.
    bb_atom_idx4 = {}
    if sc_enabled:
        a2t = feat["atom_to_token_idx"]
        while a2t.dim() > 1:
            a2t = a2t.squeeze(0)
        # Prefer the featurizer's BY-NAME resolution (sc_bb_atom_idx: [N_token, 4] =
        # (N, CA, C, O), -1 where absent) — the same tensor training uses. Falling back to
        # the positional aot[:3] assumes every token's atoms start (N, CA, C), which the
        # featurizer explicitly does NOT assume; two sources of truth here silently yield a
        # wrong frame on any token whose atom order differs.
        _bbidx = feat.get("sc_bb_atom_idx")
        if _bbidx is not None:
            while _bbidx.dim() > 2:
                _bbidx = _bbidx.squeeze(0)
            # Frame atoms only: the O column is allowed to be -1 on a token whose
            # N/CA/C are all present, so validity must be tested over 0:3 alone.
            for tok in positions.tolist():
                quad = _bbidx[tok].to(torch.long)      # (N, CA, C, O), -1 = absent
                tri = quad[:3]
                if int(tri.min()) >= 0:
                    bb_atom_idx[tok] = tri
                    bb_atom_idx4[tok] = quad
        else:
            logging.getLogger(__name__).warning(
                "cogenerate: sc_bb_atom_idx absent; falling back to positional N/CA/C "
                "(aot[:3]). This assumes backbone-first atom order — featurize with "
                "compute_sidechain=True to get the by-name resolution training uses."
            )
            for tok in positions.tolist():
                aot = (a2t == tok).nonzero(as_tuple=False).squeeze(-1)
                if aot.numel() >= 3:
                    bb_atom_idx[tok] = aot[:3]
    h_res_prime_inject = None  # persistent side-chain-aware h_res across steps
    a_sc_inject = None         # persistent per-token side-chain summary (a_direct)
    # persistent ATOM-level side-chain features for the 4 backbone atoms (q_direct):
    # q_sc_inject [Nc, 4, c_atom] paired row-for-row with q_idx_inject [Nc, 4].
    q_sc_inject = None
    q_idx_inject = None
    # M3: keep the latest decoded side-chain global coords per committed token so
    # the final result carries a full-atom (backbone + S_phi side-chain) output.
    sidechain_out: dict[int, dict[str, Any]] = {}
    _prev_seq: tuple = ()          # sequence-stabilization tracking (paper termination)
    _seq_stable = 0

    for step, (sig_t, sig_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
        feat["restype"] = restype.unsqueeze(0) if input_feature_dict["restype"].dim() == 3 else restype
        sigma = sig_t.reshape(1).to(dtype)

        # Persist side-chain-informed h_res' into the trunk for this step — but ONLY when
        # the INDIRECT channel is enabled. Training honours sidechain.hres_inject
        # (model.py: the refinement pass still runs, it just carries no side-chain info);
        # if sampling ignored it, then every arm trained WITHOUT the indirect channel
        # (no / a-direct / bbctx / q) would silently get it back at inference, and the
        # information-flow ablation would be measuring a model it never trained.
        s_trunk_step = s_trunk
        if (
            sc_enabled
            and getattr(model, "sc_hres_inject", True)
            and h_res_prime_inject is not None
        ):
            s_trunk_step = s_trunk + model.hres_injector(h_res_prime_inject).to(s_trunk.dtype)

        # Arm the direct a-level injection for THIS backbone call only. a_sc_inject is the
        # side-chain summary from the PREVIOUS step, mirroring how h_res_prime_inject
        # is persisted. The finally-clause
        # disarms even on exception, so step 0 can never inherit a live flag.
        # q_direct's call-key registry exists ONLY to make the forward and the
        # activation-checkpoint RECOMPUTE agree during training. Inference runs under
        # no_grad -> there is no recompute, so the registry is dead weight; left
        # uncleared it strong-references every step's q_skip for the whole sampling run.
        model._q_inject_calls = {}
        model._a_sc_cache = a_sc_inject
        model._a_direct_active = bool(
            getattr(model, "sc_a_direct", False) and a_sc_inject is not None
        )
        # Same for the ATOM-level channel (sidechain.q_direct): the decoder pre-hook
        # rewrites the 4 backbone atom rows of q_skip with q'_bb, using the q_sc_bb
        # S_phi produced at the PREVIOUS step ("only available after the first-round").
        # Without this the trained QAtomFusion would be dead weight at sampling — the
        # exact train/inference mismatch a_direct was just fixed for.
        model._q_sc_cache = q_sc_inject
        model._q_bb_idx_cache = q_idx_inject
        model._q_direct_active = bool(
            getattr(model, "sc_q_direct", False) and q_sc_inject is not None
        )
        try:
            x_denoised = model.diffusion_module(
                x_noisy=x, t_hat_noise_level=sigma, input_feature_dict=feat,
                s_inputs=s_inputs, s_trunk=s_trunk_step, z_trunk=z_trunk,
                pair_z=None, p_lm=None, c_l=None,
            )
        finally:
            model._a_direct_active = False
            model._q_direct_active = False
        # EDM Euler reverse step on coordinates.
        d = (x - x_denoised) / sig_t
        x = x + (sig_next - sig_t) * d

        # Structure-aware sequence step from the captured a_token.
        a = model._a_token_cache
        if a is None:
            continue
        a_red = model._reduce_a_token(a, sigma).to(dtype)  # [.., N_token, c_token]
        frac = float(still[positions].float().mean()) if positions.numel() else 0.0
        logits = model.design_residue_type_head(
            a_red, aa_t=torch.tensor(frac, device=device)
        ).float()
        if logits.dim() == 3:
            logits = logits.squeeze(0)  # [N_token, 20]
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)

        k = counts[step] if step < len(counts) else 0
        masked_idx = still.nonzero(as_tuple=False).squeeze(-1)
        if k > 0 and masked_idx.numel() > 0:
            k = min(k, int(masked_idx.numel()))
            top = torch.topk(conf[masked_idx], k).indices
            reveal = masked_idx[top]
            sampled[reveal] = pred[reveal]
            still[reveal] = False
            for j in reveal.tolist():
                restype[j] = one_hot(int(aa20_to_36[sampled[j]]))

        # --- side-chain step (late sigma): decode side chains for committed
        #     residues, pool -> h_res', persist for the next backbone step. ---
        # Gate on STEP fraction, not on a linear fraction of sigma_max: the Karras (rho=7)
        # schedule collapses sigma so fast that `sig_t <= 0.5*sigma_max` is already true at
        # step 3 of 20 (sigma ~995 A) — i.e. "late sigma" was firing for 17 of 20 steps, most
        # of them on near-pure noise. Last `sc_start_frac` of the trajectory is what the
        # comment always meant.
        _n_steps = len(noise_schedule) - 1
        _sc_from = int(round((1.0 - sc_start_frac) * _n_steps))
        if sc_enabled and step >= _sc_from:
            committed = [int(j) for j in positions.tolist()
                         if (not bool(still[j])) and (j in bb_atom_idx) and int(sampled[j]) >= 0]
            if committed:
                from pxdesign_train.sidechain.frames import build_frame, to_global, to_local
                from pxdesign_train.sidechain.instantiate import (
                    sidechain_atom_name_ids, sidechain_atoms, sidechain_mask,
                )
                from pxdesign_train.sidechain import init as sc_init
                from pxdesign_train.sidechain.coevolution import pool_side_chain_atoms

                # a_hat: the residue types already committed by the unmasking loop.
                # This is the SINGLE type source for this block — it produces the
                # atom set (ids / m) AND the ideal template below, exactly as the
                # training path derives both from one type (GT under teacher forcing,
                # predicted under sidechain.predicted_mask). `sampled` is in the
                # 20-class AA index space == _AA3 == instantiate.STD_AA_3 order,
                # which is the index space template_init_local expects.
                a_hat = sampled[torch.as_tensor(committed, device=device)].long()  # [Nc]
                restypes3 = [_AA3[int(i)] for i in a_hat.tolist()]
                ids = sidechain_atom_name_ids(restypes3).to(device)
                m = sidechain_mask(restypes3).to(device)
                # F_hat MUST come from x_denoised (x_hat_0), NOT from the noisy sample x.
                # Training builds it with frames_from_backbone_index(out["x_denoised"], ...).
                # At the first side-chain step sigma is still ~10^3 A, so a frame built from
                # x would be an essentially random rotation with |t| ~ 10^3 A — and since the
                # frame-aware head routes ALL of S_phi's output through it, that is not a
                # rounding error, it is garbage propagated into h_res' and every later step.
                xc = x_denoised.squeeze(0)
                Ns = torch.stack([xc[bb_atom_idx[j][0]] for j in committed]).float()
                CAs = torch.stack([xc[bb_atom_idx[j][1]] for j in committed]).float()
                Cs = torch.stack([xc[bb_atom_idx[j][2]] for j in committed]).float()
                R, t = build_frame(Ns, CAs, Cs)  # F_hat from x_hat_0, as in training
                # Overleaf par.221, inference half: "side-chain atoms are initialized
                # from residue-specific ideal templates around the predicted backbone
                # frames", with the residue type = a_hat. Must mirror model.py's
                # training block exactly (same switches, same frame), otherwise S_phi
                # is sampled off the input distribution it was trained on.
                if getattr(model, "sc_template_init", False):
                    noisy_local = sc_init.template_init_local(
                        a_hat.cpu(), m.cpu(),
                        sigma_T=getattr(model, "sc_init_sigma_T", sc_init.DEFAULT_SIGMA_T),
                    )
                else:
                    noisy_local = sc_init.gaussian_init_local(
                        m.cpu(), sigma=model.sc_init_sigma
                    )
                noisy_local = noisy_local.to(device).to(dtype)
                a_full = a_red.squeeze(0) if a_red.dim() == 3 else a_red   # [N_token, c]
                h_c = a_full[committed]
                l_c = logits[committed]
                # Sigma-embedding = this step's real noise level (EDM c_noise),
                # matching per-sigma training — not a constant.
                # Honour sidechain.per_sigma: training feeds S_phi a CONSTANT t=1 whenever
                # per_sigma is off (the Stage II-A warmup config), not the sigma embedding.
                if getattr(model, "sc_per_sigma", True):
                    sc_t = (0.25 * sig_t.reshape(1).clamp_min(1e-4).log()).to(device)
                else:
                    sc_t = torch.ones(1, device=device)
                # S_phi emits global side-chain coordinates. Its coordinate INPUT
                # channel follows sidechain.local_coord_input. With the default switch
                # off, we map the same init to global through F_hat, as training does;
                # turning it on feeds the residue-LOCAL frame, translation-free.
                if getattr(model, "sc_local_coord_input", False):
                    noisy_in = noisy_local[None]
                else:
                    noisy_in = to_global(
                        noisy_local[None].float(), R[None].float(), t[None].float()
                    ).to(dtype)
                # Frame-aware head (sidechain.frame_aware_head): hand S_phi the same
                # rigid frame training gives it, so it regresses local offsets and the
                # known transform maps them to global. Output space stays global.
                _fa = getattr(model, "sc_frame_aware_head", False)
                # ATOM-level channel (sidechain.q_direct): hand S_phi the residue's 4
                # backbone atoms (N, CA, C, O) in its own LOCAL frame — the SAME 14-slot
                # axis training builds, from the SAME source (the predicted backbone
                # x_denoised, gathered by name at sc_bb_atom_idx, mapped through F_hat).
                # S_phi returns their post-attention features (q_sc_bb), which the next
                # backbone step fuses into the Backbone Module's own 4 atom rows.
                sc_kwargs = {}
                q_idx_c = None
                # Gate on bb_context, NOT on q_direct. The 14-slot axis is its OWN switch:
                # the `bbctx` control arm trains with bb_context=True and q_direct=False, so
                # gating here on q_direct would run S_phi 10-slot at sampling after training
                # it 14-slot — sampling a model in a mode it was never trained in, and
                # corrupting the very control (q - bbctx) that isolates the atom channel.
                # (q_direct implies bb_context, so this still covers the q arms.)
                if getattr(model, "sc_bb_context", False) and all(
                    j in bb_atom_idx4 for j in committed
                ):
                    q_idx_c = torch.stack([bb_atom_idx4[j] for j in committed]).to(device)
                    v4 = (q_idx_c >= 0)                                   # [Nc, 4]
                    bb4 = xc[q_idx_c.clamp_min(0)].float()                # [Nc, 4, 3]
                    bb4 = bb4 * v4[..., None].to(bb4.dtype)
                    bb_local = to_local(bb4, R.float(), t.float()) * v4[..., None]
                    sc_kwargs = {"bb_local": bb_local[None].to(dtype)}
                sc_out = model.sidechain_module(
                    h_c[None], l_c[None], ids[None], m[None], noisy_in,
                    sc_t, ca_coords=t[None].float(),
                    frame_R=(R[None].float() if _fa else None),
                    frame_t=(t[None].float() if _fa else None),
                    **sc_kwargs,
                )
                bb_feats = None
                if len(sc_out) == 3:
                    y0_global, atom_feats, bb_feats = sc_out
                else:
                    y0_global, atom_feats = sc_out
                y0_global = y0_global.float()[0]  # [Nc, A, 3]
                for ci, j in enumerate(committed):
                    names = sidechain_atoms(restypes3[ci])
                    k = len(names)
                    sidechain_out[int(j)] = {
                        "restype3": restypes3[ci],
                        "atom_names": names,
                        "coords": y0_global[ci, :k].detach().cpu(),  # [k, 3] global
                    }
                h_prime = model.sidechain_feedback(atom_feats, m[None], h_c[None]).squeeze(0)
                full = a_full.clone()
                full[committed] = h_prime.to(full.dtype)
                h_res_prime_inject = full.unsqueeze(0) if s_trunk.dim() == 3 else full
                # DIRECT a-level feedback (sidechain.a_direct): cache the SAME per-token
                # side-chain summary training caches, so the next backbone step consumes the
                # fused token a'_bb = a_bb + MLP(concat(a_bb, W a_sc)). Without this the
                # trained ATokenFusion is silently dropped at sampling and every one of its
                # parameters is dead weight — a hard train/inference mismatch.
                if getattr(model, "sc_a_direct", False):
                    a_sc_c = pool_side_chain_atoms(atom_feats, m[None]).squeeze(0)  # [Nc, c_atom]
                    a_sc_full = a_sc_c.new_zeros((a_full.shape[0], a_sc_c.shape[-1]))
                    a_sc_full[committed] = a_sc_c
                    a_sc_inject = a_sc_full
                # DIRECT q-level (atom) feedback: cache q_sc_bb + the atom indices it
                # belongs to. The rows are a SUBSET of tokens (the committed residues) —
                # the hook writes back by ATOM INDEX, so it needs no full-token axis.
                if bb_feats is not None and q_idx_c is not None:
                    q_sc_inject = bb_feats[0]                             # [Nc, 4, c_atom]
                    q_idx_inject = q_idx_c                                # [Nc, 4]

        trajectory.append({
            "step": step,
            "sigma": float(sig_t),
            "mask_frac": float(still[positions].float().mean()) if positions.numel() else 0.0,
            "mean_conf": float(conf[positions].mean()) if positions.numel() else 0.0,
            "sc_committed": len(committed) if (sc_enabled and step >= _sc_from) else 0,
        })

        # Paper: terminate the iterative refinement when the predicted sequence
        # stabilizes (all positions committed and unchanged for `seq_patience`
        # steps). Off by default -> keep the full fixed EDM schedule.
        if stop_on_seq_stable and not bool(still[positions].any()):
            cur = tuple(sampled[positions].cpu().tolist())
            if step > 0 and cur == _prev_seq:
                _seq_stable += 1
                if _seq_stable >= seq_patience:
                    trajectory[-1]["early_stop"] = True
                    break
            else:
                _seq_stable = 0
            _prev_seq = cur

    # M3: full-atom assembly — backbone coords from diffusion + S_phi side-chain
    # global coords per committed design residue (empty dict if the cycle was off
    # or nothing committed). Each entry: {restype3, atom_names, coords[k,3]}.
    return {
        "coordinate": x.squeeze(0),
        "sequence": sampled,
        "trajectory": trajectory,
        "sidechain": sidechain_out,
        "has_full_atom_sidechain": bool(sidechain_out),
    }
