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


@torch.no_grad()
def cogenerate(
    model,
    input_feature_dict: dict[str, Any],
    N_step: int = 20,
    temperature: float = 0.0,
    chunk_size: Optional[int] = None,
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

    for step, (sig_t, sig_next) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
        feat["restype"] = restype.unsqueeze(0) if input_feature_dict["restype"].dim() == 3 else restype
        sigma = sig_t.reshape(1).to(dtype)

        x_denoised = model.diffusion_module(
            x_noisy=x, t_hat_noise_level=sigma, input_feature_dict=feat,
            s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
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

        trajectory.append({
            "step": step,
            "sigma": float(sig_t),
            "mask_frac": float(still[positions].float().mean()) if positions.numel() else 0.0,
            "mean_conf": float(conf[positions].mean()) if positions.numel() else 0.0,
        })

    return {"coordinate": x.squeeze(0), "sequence": sampled, "trajectory": trajectory}
