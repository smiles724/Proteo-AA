"""Backbone-dependent rotamer lookup — Overleaf 0714 appendix, Step 2.

    (a_hat_i, phi_hat_i, psi_hat_i)  ->  { p_{i,r}, chibar_{i,r} }_{r=1..R_i}

    r_i = argmax_r p_{i,r}                       ("deterministic")
    r_i ~ Categorical(p_{i,1}, ..., p_{i,R_i})   ("sampled")

The library is the Dunbrack BBDEP2010 table built by
``scripts/build_rotamer_library.py`` into ``data/dunbrack2010_bbdep.npz``
(ODC-By; cite Shapovalov & Dunbrack 2011, Structure 19:844-858). It is stored
full-fidelity: every rotamer of every (residue, phi, psi) cell on the source's
10-degree grid, nothing truncated.

The appendix also names MEDFORD as an alternative -- a real library (Mortensen et al.,
"A backbone-dependent rotamer library with high (phi, psi) coverage using metadynamics
simulations", Protein Science 31:e4491, 2022; MEtadynamics of Dipeptides FOr Rotamer
Distribution). It is NOT implemented here. Its selling point is coverage of (phi, psi)
regions where the PDB, and therefore Dunbrack, is sparse -- which is exactly the regime
a *generated* backbone can wander into, so it is worth revisiting once backbones are
good. Dunbrack is the appendix's stated primary choice and is what ships.

Everything below is library-agnostic (a path to an npz with this schema), so a MEDFORD
table drops in without touching a single caller.

phi/psi are NaN at the first/last residue of a chain and across chain breaks
(``frames.backbone_phi_psi``). Those residues use the backbone-INDEPENDENT
marginal, obtained by averaging the library uniformly over the phi/psi grid.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

DATA = Path(__file__).parent / "data" / "dunbrack2010_bbdep.npz"

CITATION = (
    "Shapovalov, M.V. & Dunbrack, R.L. Jr. (2011) A smoothed backbone-dependent "
    "rotamer library for proteins derived from adaptive kernel density estimates "
    "and regressions. Structure 19, 844-858. (BBDEP2010, ODC-By)"
)

N_BIN = 36
BIN_RAD = torch.pi * 2 / N_BIN     # 10 degrees
MAX_CHI = 4

_LIB = None            # lazily loaded dict of tensors
_LOAD_FAILED = False


def _load(path: Path = DATA):
    global _LIB, _LOAD_FAILED
    if _LIB is not None or _LOAD_FAILED:
        return _LIB
    if not path.exists():
        _LOAD_FAILED = True
        logger.warning(
            "rotamer library not found at %s — build it with "
            "`python scripts/build_rotamer_library.py --download`. "
            "Falling back to the static CCD template.",
            path,
        )
        return None
    import numpy as np

    z = np.load(path, allow_pickle=False)
    _LIB = {
        "counts": torch.from_numpy(z["counts"]).long(),          # [20, 36, 36]
        "offsets": torch.from_numpy(z["offsets"]).long(),        # [20, 36, 36]
        "probs": torch.from_numpy(z["probs"]).float(),           # [Ntot]
        "chis": torch.from_numpy(z["chis"]).float() / 10.0,      # [Ntot, 4] degrees
        "marg_counts": torch.from_numpy(z["marg_counts"]).long(),
        "marg_offsets": torch.from_numpy(z["marg_offsets"]).long(),
        "marg_probs": torch.from_numpy(z["marg_probs"]).float(),
        "marg_chis": torch.from_numpy(z["marg_chis"]).float() / 10.0,
    }
    # Within-cell cumulative probability, for O(1) categorical sampling.
    _LIB["cumprobs"] = _cumsum_ragged(_LIB["probs"], _LIB["counts"].reshape(-1), _LIB["offsets"].reshape(-1))
    _LIB["marg_cumprobs"] = _cumsum_ragged(_LIB["marg_probs"], _LIB["marg_counts"], _LIB["marg_offsets"])
    return _LIB


def _cumsum_ragged(probs: torch.Tensor, counts: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Cumulative probability that restarts at every cell boundary.

    Cells are contiguous and in offset order, so the start of the cell owning flat
    index t is the largest offset <= t. Marking the starts and cumsum-ing the marker
    gives that lookup in one pass.
    """
    total = probs.cumsum(0)
    starts = offsets[counts > 0]                       # strictly increasing
    marker = torch.zeros(probs.shape[0], dtype=torch.long)
    marker[starts] = 1
    start_of = starts[marker.cumsum(0) - 1]            # [Ntot] start index of each element's cell
    prior = torch.where(start_of > 0, total[(start_of - 1).clamp_min(0)], torch.zeros_like(total))
    return total - prior


def available(path: Path = DATA) -> bool:
    return _load(path) is not None


def _bin(angle: torch.Tensor) -> torch.Tensor:
    """phi/psi in radians -> grid index 0..35, periodic. NaN -> 0 (masked by caller)."""
    a = torch.nan_to_num(angle, nan=0.0)
    return (torch.round((a + torch.pi) / BIN_RAD).long() % N_BIN)


def _pick(cum: torch.Tensor, offs: torch.Tensor, cnts: torch.Tensor,
          chis: torch.Tensor, u: Optional[torch.Tensor]) -> torch.Tensor:
    """Select one rotamer per row. u=None -> mode (rotamers are stored p-descending)."""
    if u is None:
        sel = offs                                       # first == most probable
    else:
        maxc = int(cnts.max().item()) if cnts.numel() else 1
        ar = torch.arange(max(maxc, 1), device=offs.device)
        idx = offs[:, None] + ar[None, :]                                  # [n, maxc]
        inb = ar[None, :] < cnts[:, None]
        c = cum[idx.clamp(max=cum.shape[0] - 1)]
        c = torch.where(inb, c, torch.full_like(c, 2.0))                   # past the end -> never chosen
        m = (c < u[:, None]).sum(-1)                                       # first index with cum >= u
        sel = offs + m.clamp(max=(cnts - 1).clamp_min(0))
    return chis[sel.clamp(min=0, max=chis.shape[0] - 1)]


def select_chi(
    type_idx: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
    psi: Optional[torch.Tensor] = None,
    mode: str = "sample",
    generator: Optional[torch.Generator] = None,
    path: Path = DATA,
) -> Optional[torch.Tensor]:
    """Step 2: look up and select a rotamer per residue.

    Args:
        type_idx: [...] long, STD_AA_3 order.
        phi, psi: [...] float radians (NaN where undefined). None => use the
            backbone-independent marginal everywhere.
        mode: "sample" (r ~ Categorical(p)) or "mode" (argmax p).
        generator: torch.Generator, so a sampled run stays reproducible.
    Returns:
        chi: [..., MAX_CHI] float RADIANS, or None if the library is unavailable.
        Torsions the residue does not have come back as 0 and are ignored by BuildSC
        (they are gated by chi_constants.CHI_MASK).
    """
    lib = _load(path)
    if lib is None:
        return None
    if mode not in ("sample", "mode"):
        raise ValueError(f"rotamer select mode must be 'sample' or 'mode', got {mode!r}")

    shape = type_idx.shape
    tix = type_idx.reshape(-1).long().clamp(0, 19)
    n = tix.numel()

    if phi is None or psi is None:
        pb = torch.zeros(n, dtype=torch.long)
        sb = torch.zeros(n, dtype=torch.long)
        bb_known = torch.zeros(n, dtype=torch.bool)
    else:
        p = phi.reshape(-1).float()
        s = psi.reshape(-1).float()
        bb_known = torch.isfinite(p) & torch.isfinite(s)
        pb, sb = _bin(p), _bin(s)

    counts = lib["counts"][tix, pb, sb]
    offs = lib["offsets"][tix, pb, sb]
    # Residues with no phi/psi (chain termini / breaks) use the marginal table.
    counts = torch.where(bb_known, counts, lib["marg_counts"][tix])
    offs = torch.where(bb_known, offs, lib["marg_offsets"][tix])

    u = None
    if mode == "sample":
        u = torch.rand(n, generator=generator, dtype=torch.float32)

    chi_bb = _pick(lib["cumprobs"], lib["offsets"][tix, pb, sb], lib["counts"][tix, pb, sb], lib["chis"], u)
    chi_mg = _pick(lib["marg_cumprobs"], lib["marg_offsets"][tix], lib["marg_counts"][tix], lib["marg_chis"], u)
    chi = torch.where(bb_known[:, None], chi_bb, chi_mg)

    # ALA / GLY (and anything with an empty cell) have no rotamers at all.
    chi = torch.where((counts > 0)[:, None], chi, torch.zeros_like(chi))

    return torch.deg2rad(chi).reshape(*shape, MAX_CHI)
