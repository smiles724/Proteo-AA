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
    if sc_enabled:
        a2t = feat["atom_to_token_idx"]
        while a2t.dim() > 1:
            a2t = a2t.squeeze(0)
        for tok in positions.tolist():
            aot = (a2t == tok).nonzero(as_tuple=False).squeeze(-1)
            if aot.numel() >= 3:
                bb_atom_idx[tok] = aot[:3]  # N, CA, C (backbone-first ordering)
    h_res_prime_inject = None  # persistent side-chain-aware h_res across steps
    # M3: keep the latest decoded side-chain global coords per committed token so
    # the final result carries a full-atom (backbone + S_phi side-chain) output.
    sidechain_out: dict[int, dict[str, Any]] = {}
    _prev_seq: tuple = ()          # sequence-stabilization tracking (paper termination)
    _seq_stable = 0

    for step, (sig_t, sig_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
        feat["restype"] = restype.unsqueeze(0) if input_feature_dict["restype"].dim() == 3 else restype
        sigma = sig_t.reshape(1).to(dtype)

        # Persist side-chain-informed h_res' into the trunk for this step.
        s_trunk_step = s_trunk
        if sc_enabled and h_res_prime_inject is not None:
            s_trunk_step = s_trunk + model.hres_injector(h_res_prime_inject).to(s_trunk.dtype)

        x_denoised = model.diffusion_module(
            x_noisy=x, t_hat_noise_level=sigma, input_feature_dict=feat,
            s_inputs=s_inputs, s_trunk=s_trunk_step, z_trunk=z_trunk,
            pair_z=None, p_lm=None, c_l=None,
        )
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
        if sc_enabled and float(sig_t) <= sc_start_frac * float(noise_schedule[0]):
            committed = [int(j) for j in positions.tolist()
                         if (not bool(still[j])) and (j in bb_atom_idx) and int(sampled[j]) >= 0]
            if committed:
                from pxdesign_train.sidechain.frames import build_frame, to_global
                from pxdesign_train.sidechain.instantiate import (
                    sidechain_atom_name_ids, sidechain_atoms, sidechain_mask,
                )
                from pxdesign_train.sidechain.init import gaussian_init_local

                restypes3 = [_AA3[int(sampled[j])] for j in committed]
                ids = sidechain_atom_name_ids(restypes3).to(device)
                m = sidechain_mask(restypes3).to(device)
                xc = x.squeeze(0)
                Ns = torch.stack([xc[bb_atom_idx[j][0]] for j in committed]).float()
                CAs = torch.stack([xc[bb_atom_idx[j][1]] for j in committed]).float()
                Cs = torch.stack([xc[bb_atom_idx[j][2]] for j in committed]).float()
                R, t = build_frame(Ns, CAs, Cs)
                noisy = gaussian_init_local(m.cpu(), sigma=model.sc_init_sigma).to(device).to(dtype)
                a_full = a_red.squeeze(0) if a_red.dim() == 3 else a_red   # [N_token, c]
                h_c = a_full[committed]
                l_c = logits[committed]
                # Sigma-embedding = this step's real noise level (EDM c_noise),
                # matching per-sigma training — not a constant.
                sc_t = (0.25 * sig_t.reshape(1).clamp_min(1e-4).log()).to(device)
                y0_local, atom_feats = model.sidechain_module(
                    h_c[None], l_c[None], ids[None], m[None], noisy[None],
                    sc_t, ca_coords=t[None].float(),
                )
                # M3: map predicted local side-chain coords -> global via the
                # predicted-backbone frame, and store per committed residue.
                y0_global = to_global(y0_local.float(), R[None].float(), t[None].float())[0]  # [Nc, A, 3]
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

        trajectory.append({
            "step": step,
            "sigma": float(sig_t),
            "mask_frac": float(still[positions].float().mean()) if positions.numel() else 0.0,
            "mean_conf": float(conf[positions].mean()) if positions.numel() else 0.0,
            "sc_committed": len(committed) if sc_enabled and float(sig_t) <= sc_start_frac * float(noise_schedule[0]) else 0,
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
