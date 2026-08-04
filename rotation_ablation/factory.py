"""Build the diagnostic modules and deterministic input bundles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from pxdesign_train.sidechain.feedback import HResFeedback
from pxdesign_train.sidechain.frames import to_global
from pxdesign_train.sidechain.instantiate import (
    MAX_SC,
    STD_AA_3,
    sidechain_atom_name_ids,
    sidechain_mask,
)
from pxdesign_train.sidechain.module import SideChainModule

from .bundle import SidechainInputBundle
from .geometry import random_rotation


def _checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a model state dictionary")
    return {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def modules_from_checkpoint(
    path: str | Path,
    *,
    n_heads: int = 16,
) -> tuple[SideChainModule, HResFeedback, dict[str, Any]]:
    """Recreate S_phi/HResFeedback from their checkpoint tensor shapes."""
    state = _checkpoint_state(path)
    sc_prefix = "sidechain_module."
    fb_prefix = "sidechain_feedback."
    sc = {key[len(sc_prefix) :]: value for key, value in state.items() if key.startswith(sc_prefix)}
    fb = {key[len(fb_prefix) :]: value for key, value in state.items() if key.startswith(fb_prefix)}
    if not sc:
        raise ValueError(f"{path} contains no {sc_prefix} tensors")
    if not fb:
        raise ValueError(f"{path} contains no {fb_prefix} tensors")

    c_atom, c_res = sc["w_res.weight"].shape
    n_type = sc["w_aa.weight"].shape[1]
    c_time = sc["w_t.0.weight"].shape[1]
    block_ids = {
        int(match.group(1))
        for key in sc
        if (match := re.match(r"blocks\.(\d+)\.", key))
    }
    cross_ids = {
        int(match.group(1))
        for key in sc
        if (match := re.match(r"cross_res_blocks\.(\d+)\.", key))
    }
    n_blocks = max(block_ids) + 1
    n_cross_blocks = max(cross_ids) + 1
    ff_mult = sc["blocks.0.ff.0.weight"].shape[0] // c_atom
    if c_atom % int(n_heads):
        raise ValueError(f"checkpoint c_atom={c_atom} is not divisible by n_heads={n_heads}")

    module = SideChainModule(
        c_res=c_res,
        c_atom=c_atom,
        n_type=n_type,
        c_time=c_time,
        n_blocks=n_blocks,
        n_heads=int(n_heads),
        n_cross_blocks=n_cross_blocks,
        ff_mult=ff_mult,
    )
    feedback = HResFeedback(c_atom=c_atom, c_res=c_res)
    module.load_state_dict(sc, strict=True)
    feedback.load_state_dict(fb, strict=True)
    metadata = {
        "checkpoint": str(Path(path).resolve()),
        "c_res": c_res,
        "c_atom": c_atom,
        "n_type": n_type,
        "c_time": c_time,
        "n_blocks": n_blocks,
        "n_cross_blocks": n_cross_blocks,
        "n_heads": int(n_heads),
        "ff_mult": ff_mult,
    }
    return module, feedback, metadata


def random_modules(
    *,
    c_res: int = 32,
    c_atom: int = 64,
    n_type: int = 20,
    c_time: int = 32,
    n_blocks: int = 2,
    n_heads: int = 4,
    n_cross_blocks: int = 2,
    seed: int = 0,
) -> tuple[SideChainModule, HResFeedback, dict[str, Any]]:
    torch.manual_seed(int(seed))
    module = SideChainModule(
        c_res=c_res,
        c_atom=c_atom,
        n_type=n_type,
        c_time=c_time,
        n_blocks=n_blocks,
        n_heads=n_heads,
        n_cross_blocks=n_cross_blocks,
    )
    feedback = HResFeedback(c_atom=c_atom, c_res=c_res)
    return module, feedback, {"checkpoint": None, "random_weight_seed": int(seed)}


def synthetic_bundle(
    *,
    c_res: int,
    n_type: int,
    length: int = 6,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> SidechainInputBundle:
    """Deterministic same-protein fixture for installation/sensitivity checks."""
    if n_type != len(STD_AA_3):
        raise ValueError(
            f"synthetic fixture assumes {len(STD_AA_3)} residue types, got {n_type}; "
            "capture a real bundle for this checkpoint"
        )
    sequence = ["LYS", "PHE", "ARG", "GLU", "TRP", "ASN"]
    sequence = [sequence[i % len(sequence)] for i in range(int(length))]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    dtype = torch.float32

    atom_ids = sidechain_atom_name_ids(sequence).unsqueeze(0).to(device)
    atom_mask = sidechain_mask(sequence).unsqueeze(0).to(device)
    type_idx = torch.tensor([STD_AA_3.index(name) for name in sequence], device=device)
    h_res = torch.randn(1, length, c_res, generator=generator, device=device)
    aa_logits = torch.full((1, length, n_type), -4.0, device=device)
    aa_logits[0, torch.arange(length, device=device), type_idx] = 4.0

    local = torch.randn(
        1, length, MAX_SC, 3, generator=generator, device=device, dtype=dtype
    )
    local[..., 0] += 2.0
    local = local * atom_mask[..., None]
    frame_r = torch.stack(
        [
            random_rotation(generator=generator, device=device, dtype=dtype)
            for _ in range(length)
        ]
    ).unsqueeze(0)
    frame_t = torch.zeros(1, length, 3, device=device)
    frame_t[0, :, 0] = torch.arange(length, device=device, dtype=dtype) * 3.8
    frame_t[0, :, 1] = torch.sin(torch.arange(length, device=device, dtype=dtype)) * 2.0
    noisy_global = to_global(local, frame_r, frame_t)

    return SidechainInputBundle(
        h_res=h_res,
        aa_logits=aa_logits,
        atom_name_ids=atom_ids,
        atom_mask=atom_mask,
        noisy_local=local,
        noisy_global=noisy_global,
        time=torch.tensor([0.25], device=device),
        ca_coords=frame_t,
        frame_R=frame_r,
        frame_t=frame_t,
        gt_local=local.clone(),
        gt_global=noisy_global.clone(),
        residue_type_idx=type_idx.unsqueeze(0),
        metadata={"source": "deterministic synthetic same-protein fixture", "sequence": sequence},
    ).validate()
