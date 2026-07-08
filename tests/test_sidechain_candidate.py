"""Tests for Stage III L_SC-AA candidate compatibility ranking."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from pxdesign_train.sidechain.candidate import candidate_energy, sc_aa_ranking_loss


def test_ranking_prefers_low_energy_candidate():
    cand_types = torch.tensor([[3, 7]])            # candidate AA indices
    energies = torch.tensor([[0.1, 5.0]])          # candidate 0 (type 3) is best
    # logit(best type 3) high -> low ranking loss
    good = torch.zeros(1, 20); good[0, 3] = 5.0
    # logit(best type 3) low, worse type 7 high -> high ranking loss
    bad = torch.zeros(1, 20); bad[0, 7] = 5.0
    assert (sc_aa_ranking_loss(bad, cand_types, energies).item()
            > sc_aa_ranking_loss(good, cand_types, energies).item())


def test_candidate_energy_clash():
    clash = torch.zeros(1, 3, 3)                     # 3 atoms overlapping -> clash
    spread = torch.tensor([[[0.0, 0, 0], [5, 0, 0], [10, 0, 0]]])
    sc = torch.cat([clash, spread], dim=0)          # [2, 3, 3]
    mask = torch.ones(2, 3, dtype=torch.bool)
    e = candidate_energy(sc, mask)
    assert e.shape == (2,)
    assert e[0].item() > e[1].item()                # clashing candidate = worse


def test_ranking_backprop():
    aa_logits = torch.randn(2, 20, requires_grad=True)
    cand = torch.randint(0, 20, (2, 3))
    en = torch.rand(2, 3)
    sc_aa_ranking_loss(aa_logits, cand, en).backward()
    assert torch.isfinite(aa_logits.grad).all()


def test_position_mask_zeros_unselected():
    aa_logits = torch.zeros(2, 20)
    cand = torch.tensor([[1, 2], [3, 4]])
    en = torch.tensor([[0.0, 9.0], [0.0, 9.0]])
    pm = torch.tensor([False, False])
    assert sc_aa_ranking_loss(aa_logits, cand, en, position_mask=pm).item() == 0.0
