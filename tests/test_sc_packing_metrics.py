import torch
from pxdesign_train.sidechain.metrics import packing_metrics, summarize_metrics
from pxdesign_train.sidechain.templates import IDEAL_SC_LOCAL
from pxdesign_train.sidechain.chi_constants import IDEAL_BB_LOCAL
from pxdesign_train.sidechain.instantiate import instantiate_from_type_indices, sidechain_atoms


def test_ideal_geometry_and_symmetry_exchange_with_missing_observations():
    types = torch.arange(20)
    _, mask = instantiate_from_type_indices(types)
    pred = IDEAL_SC_LOCAL.clone()
    asp = sidechain_atoms("ASP")
    a,b = asp.index("OD1"),asp.index("OD2")
    pred[3,[a,b]] = pred[3,[b,a]]
    observed = mask.clone(); observed[3,b] = False
    target = IDEAL_SC_LOCAL.clone(); target[3,b] = float("nan")
    design = torch.ones(20, dtype=torch.bool)
    old = design.clone()
    counts = packing_metrics(types, pred, IDEAL_BB_LOCAL, mask, design,
        target=target, observed=observed)
    metrics = summarize_metrics(counts)
    assert metrics["symmetry_rmsd"] == 0
    assert metrics["completeness"] == 1
    assert metrics["bad_bond_count"] == 0
    assert metrics["chi_recovery"] == 1
    assert torch.equal(old, design)


def test_displaced_sidechains_cannot_pass_covalent_geometry_with_low_clash():
    types = torch.tensor([0])
    _, mask = instantiate_from_type_indices(types)
    pred = IDEAL_SC_LOCAL[:1].clone() + 30
    metrics = packing_metrics(types, pred, IDEAL_BB_LOCAL[:1], mask, torch.tensor([True]))
    assert metrics["bad_bond_count"] == metrics["bond_count"] == 1
    assert "observed_atoms" not in metrics  # no native metrics on unlabeled input
