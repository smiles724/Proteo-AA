import torch

from pxdesign_train.loss import PXDesignLoss


def test_backbone_eval_metrics_are_alignment_invariant():
    n_atom, n_sample = 8, 2
    gt = torch.randn(n_sample, n_atom, 3)
    pred = gt + torch.tensor([12.0, -4.0, 3.0])
    sigma = torch.ones(n_sample)
    coord_mask = torch.ones(n_atom)
    ca_mask = torch.zeros(n_atom, dtype=torch.bool)
    ca_mask[1::4] = True
    bb_mask = torch.ones(n_atom, dtype=torch.bool)

    out = PXDesignLoss(
        align_before_mse=False,
        weight_lddt=0.0,
        weight_disto=0.0,
    )(
        pred,
        gt,
        sigma,
        coord_mask,
        ca_mask,
        eval_ca_atom_mask=ca_mask,
        eval_backbone_atom_mask=bb_mask,
    )

    assert out["ca_rmsd"].item() < 1e-4
    assert out["bb_rmsd"].item() < 1e-4
    assert abs(out["tm_score"].item() - 1.0) < 1e-5


def test_backbone_eval_metrics_default_to_zero_without_masks():
    n_atom, n_sample = 5, 1
    gt = torch.randn(n_sample, n_atom, 3)
    pred = gt + 1.0
    sigma = torch.ones(n_sample)
    coord_mask = torch.ones(n_atom)
    rep_mask = torch.ones(n_atom, dtype=torch.bool)

    out = PXDesignLoss(
        align_before_mse=False,
        weight_lddt=0.0,
        weight_disto=0.0,
    )(pred, gt, sigma, coord_mask, rep_mask)

    assert out["ca_rmsd"].item() == 0.0
    assert out["bb_rmsd"].item() == 0.0
    assert out["tm_score"].item() == 0.0
