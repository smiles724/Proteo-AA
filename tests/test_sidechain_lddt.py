"""Tests for the side-chain lDDT score.

A metric that is wrong is worse than a metric that is missing, so the properties
checked here are the ones that pin the definition rather than the plumbing: the
value at a perfect prediction, the value at a known uniform distortion (where
the four thresholds give an exact answer), same-residue exclusion, and
superposition invariance -- which is the whole reason to prefer lDDT to RMSD.
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from pxdesign_train.sidechain.lddt import (
    DEFAULT_THRESHOLDS,
    lddt_score,
    sidechain_lddt,
)


def _toy(L=6, A=4, spread=6.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(L, 1, 3, generator=g) * spread
    sc = centres + torch.randn(L, A, 3, generator=g)
    bb = centres + torch.randn(L, 4, 3, generator=g) * 0.5
    mask = torch.ones(L, A, dtype=torch.bool)
    return sc, bb, mask


def test_perfect_prediction_scores_one():
    sc, bb, mask = _toy()
    out = sidechain_lddt(sc, sc, mask, bb_coords=bb)
    assert out["lddt_sc_sc"] == 1.0
    assert out["lddt_sc_env"] == 1.0
    assert out["n_pairs_sc_env"] > out["n_pairs_sc_sc"]


def test_uniform_distance_error_hits_the_thresholds_exactly():
    """Every scored distance off by exactly 1.5 A passes the 2 and 4 thresholds
    and fails 0.5 and 1, so the score must be exactly 0.5. This pins both the
    threshold set and the averaging."""
    n = 40
    ref = torch.zeros(n, 3)
    ref[:, 0] = torch.arange(n, dtype=torch.float32)          # a line, spacing 1
    res = torch.arange(n)
    # Scale the whole line so every pairwise distance grows; pick the factor so
    # the NEAREST pair (d=1) is off by 1.5. Distances further out are off by more
    # and would fail, so instead shift only ONE atom far from the rest.
    pred = ref.clone()
    # Two atoms, one pair, distance 2 -> 3.5: |delta| = 1.5.
    a = torch.tensor([[0.0, 0, 0], [2.0, 0, 0]])
    b = torch.tensor([[0.0, 0, 0], [3.5, 0, 0]])
    r2 = torch.tensor([0, 1])
    score, npair = lddt_score(b, a, b, a, r2, r2)
    assert npair == 2                      # (0,1) and (1,0)
    assert score == 0.5, f"expected exactly 0.5, got {score}"


def test_thresholds_are_the_standard_four():
    assert DEFAULT_THRESHOLDS == (0.5, 1.0, 2.0, 4.0)


def test_same_residue_pairs_are_excluded():
    """Intra-residue distances are fixed by the residue's own chemistry. Counting
    them would inflate the score, and more for long side chains than short ones.
    """
    sc, bb, mask = _toy(L=1, A=5)          # a single residue: no cross-residue pair
    out = sidechain_lddt(sc, sc, mask, bb_coords=bb)
    assert out["n_pairs_sc_sc"] == 0
    assert math.isnan(out["lddt_sc_sc"]), "no scorable pair must be nan, not 1.0"
    assert out["n_pairs_sc_env"] == 0


def test_superposition_invariance():
    """The property RMSD does not have. Rotating and translating BOTH structures
    must not move the score; rotating only the prediction must."""
    sc, bb, mask = _toy()
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(1)))
    shift = torch.tensor([100.0, -30.0, 7.0])
    pred = sc + torch.randn(sc.shape, generator=torch.Generator().manual_seed(2)) * 0.4

    base = sidechain_lddt(pred, sc, mask, bb_coords=bb)
    moved = sidechain_lddt(pred @ q + shift, sc @ q + shift, mask,
                           bb_coords=bb @ q + shift)
    assert abs(base["lddt_sc_env"] - moved["lddt_sc_env"]) < 1e-5
    assert abs(base["lddt_sc_sc"] - moved["lddt_sc_sc"]) < 1e-5


def test_worse_prediction_scores_lower():
    sc, bb, mask = _toy()
    g = torch.Generator().manual_seed(3)
    noise = torch.randn(sc.shape, generator=g)
    scores = [sidechain_lddt(sc + noise * s, sc, mask, bb_coords=bb)["lddt_sc_env"]
              for s in (0.0, 0.25, 1.0, 3.0)]
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] == 1.0


def test_env_convention_is_the_stricter_one_here():
    """The backbone is shared between prediction and reference, so adding it as a
    PARTNER adds pairs whose reference distance the side chain must reproduce --
    it does not add free credit. Worth pinning because the opposite is a natural
    assumption and would make sc_env look better than sc_sc for the wrong reason.
    """
    sc, bb, mask = _toy()
    pred = sc + torch.randn(sc.shape, generator=torch.Generator().manual_seed(4)) * 0.8
    out = sidechain_lddt(pred, sc, mask, bb_coords=bb)
    assert out["n_pairs_sc_env"] > out["n_pairs_sc_sc"]
    assert 0.0 < out["lddt_sc_env"] < 1.0
    assert 0.0 < out["lddt_sc_sc"] < 1.0


def test_masked_slots_are_not_scored():
    sc, bb, mask = _toy(L=5, A=4)
    mask[:, 2:] = False
    full = sidechain_lddt(sc, sc, torch.ones_like(mask), bb_coords=bb)
    part = sidechain_lddt(sc, sc, mask, bb_coords=bb)
    assert part["n_pairs_sc_sc"] < full["n_pairs_sc_sc"]
