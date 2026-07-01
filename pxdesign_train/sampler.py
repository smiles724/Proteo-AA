"""
V3: inference-time iterative unmasking sampler for the residue-type
masked-diffusion head.

The coordinate side generates structure by running the reverse EDM loop
(`InferenceNoiseScheduler`). The discrete residue-type side needs its own
reverse process: start from an all-masked (`xpb`) design region and iteratively
un-mask the highest-confidence positions, re-conditioning on the partially
revealed sequence each step. This is the absorbing-state analogue of ancestral
sampling (cf. MaskGIT / MDLM confidence-based decoding).

Two layers:
  * `iterative_unmask` — the pure algorithm. Takes a `logits_fn(sampled, mask)`
    callback and is fully unit-testable on CPU with a stub.
  * `sample_residue_types` — wires the algorithm to a `ProtenixDesignTrain`
    model: it writes the revealed identities back into `restype` (so the next
    forward conditions on them) and updates the diffusion time `aa_t`.

Vocab note: the head's 20-class AA index equals the design `restype` channel
index for the 20 standard amino acids (both dicts share ALA=0..TYR=19), and the
mask token `xpb` is channel 32. `build_aa20_to_restype36` verifies this.
"""
from typing import Any, Callable, Optional

import torch


def build_aa20_to_restype36() -> tuple[torch.Tensor, int]:
    """Map 20-class AA index -> 36-channel design-restype index, plus the xpb
    (mask) channel. Built from PXDesign's constants; asserts the identity
    overlap holds so a silent vocab drift can't corrupt sampled sequences.
    """
    from pxdesign.data.constants import (
        PRO_STD_RESIDUES_NATURAL as NAT,
        STD_RESIDUES_WITH_GAP as GAP,
    )

    mapping = torch.full((20,), -1, dtype=torch.long)
    for resname, idx20 in NAT.items():
        if 0 <= idx20 < 20 and resname in GAP:
            mapping[idx20] = int(GAP[resname])
    assert (mapping >= 0).all(), "20-class AA vocab does not embed in design restype vocab"
    xpb = int(GAP["xpb"])
    return mapping, xpb


def _unmask_counts(n: int, n_steps: int) -> list[int]:
    """How many positions to reveal at each step (sums to n)."""
    if n_steps <= 0 or n <= 0:
        return [n] if n > 0 else []
    n_steps = min(n_steps, n)
    base, rem = divmod(n, n_steps)
    return [base + (1 if i < rem else 0) for i in range(n_steps)]


def iterative_unmask(
    logits_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    positions: torch.Tensor,
    n_steps: int = 8,
    temperature: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Confidence-based iterative unmasking over `positions` (long indices into
    the token axis of whatever `logits_fn` consumes).

    Args:
        logits_fn(sampled, mask) -> logits[N_token, 20]: recomputes AA logits
            given the current sampled ids (`-1` where still masked) and a bool
            `mask` (True where still masked). Called once per step.
        positions: 1-D long tensor of design-token indices to fill in.
        n_steps: number of reveal rounds.
        temperature: 0 -> greedy argmax; >0 -> sample from softmax(logits/T).

    Returns:
        sampled_full: [N_token] long, predicted AA (0..19) at `positions`,
            `-1` elsewhere (and at any position never reached).
        trajectory: per-step dicts with mask_frac / mean_conf / mean_entropy.
    """
    positions = positions.long()
    # Infer N_token from a probe call so the caller need not pass it.
    probe = logits_fn(None, None)
    n_token = probe.shape[-2]
    device = probe.device

    sampled = torch.full((n_token,), -1, dtype=torch.long, device=device)
    mask = torch.zeros(n_token, dtype=torch.bool, device=device)
    mask[positions] = True  # True = still masked

    counts = _unmask_counts(int(positions.numel()), n_steps)
    trajectory: list[dict] = []

    for k in counts:
        logits = logits_fn(sampled, mask).float()  # [N_token, 20]
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1)  # [N_token]

        if temperature and temperature > 0:
            scaled = torch.softmax(logits / temperature, dim=-1)
            pred = torch.multinomial(scaled, 1, generator=generator).squeeze(-1)
            conf = probs.gather(-1, pred[..., None]).squeeze(-1)
        else:
            conf, pred = probs.max(dim=-1)

        # Restrict to still-masked positions; reveal the top-k by confidence.
        masked_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if masked_idx.numel() == 0:
            break
        k = min(k, int(masked_idx.numel()))
        masked_conf = conf[masked_idx]
        top = torch.topk(masked_conf, k).indices
        reveal = masked_idx[top]

        sampled[reveal] = pred[reveal]
        mask[reveal] = False

        trajectory.append(
            {
                "mask_frac": float(mask[positions].float().mean()),
                "mean_conf": float(conf[positions].mean()),
                "mean_entropy": float(entropy[positions].mean()),
                "revealed": int(k),
            }
        )

    return sampled, trajectory


@torch.no_grad()
def sample_residue_types(
    model,
    input_feature_dict: dict[str, Any],
    design_token_mask: torch.Tensor,
    n_steps: int = 8,
    temperature: float = 0.0,
    chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Generate residue types for the design region via iterative unmasking.

    Conditions on inference-available information only: the design region starts
    fully masked (`xpb`), and each revealed identity is written back into
    `restype` so subsequent steps self-condition on the partial sequence. The
    diffusion time `aa_t` is set to the current masked fraction each step.

    Returns (sampled_aa20 [N_token], trajectory).
    """
    model.eval()
    aa20_to_36, xpb = build_aa20_to_restype36()

    design_token_mask = design_token_mask.bool()
    positions = design_token_mask.nonzero(as_tuple=False).squeeze(-1)
    feat = dict(input_feature_dict)  # shallow copy; we rewrite `restype`
    restype = feat["restype"].clone()
    n_ch = restype.shape[-1]
    device = restype.device
    aa20_to_36 = aa20_to_36.to(device)

    def _one_hot(ch_idx: int) -> torch.Tensor:
        v = torch.zeros(n_ch, device=device, dtype=restype.dtype)
        v[ch_idx] = 1.0
        return v

    # Start: whole design region = xpb.
    for i in positions.tolist():
        restype[i] = _one_hot(xpb)

    def logits_fn(sampled, mask):
        if sampled is not None:
            # Write revealed identities into restype; keep masked ones as xpb.
            for i in positions.tolist():
                if sampled[i] >= 0:
                    restype[i] = _one_hot(int(aa20_to_36[sampled[i]]))
            frac = float(mask[positions].float().mean()) if mask is not None else 1.0
            feat["aa_t"] = torch.tensor(frac, device=device)
        else:
            feat["aa_t"] = torch.tensor(1.0, device=device)
        feat["restype"] = restype
        logits = model.predict_aa(feat, chunk_size=chunk_size)  # [.., N_token, 20]
        if logits.dim() == 3:  # drop a leading batch axis of size 1
            logits = logits.squeeze(0)
        return logits

    return iterative_unmask(
        logits_fn, positions, n_steps=n_steps, temperature=temperature
    )
