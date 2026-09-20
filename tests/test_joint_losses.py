"""The placement term: its symmetry policy, its weighting, and its refusals.

None of this needs the backbone donor -- these are properties of the loss
itself, and the point of separating them from the gradient tests is that they
can be run in a second.
"""

import math

import pytest
import torch

from pxf.joint import losses as JL


def _aa(one_letter):
    from fampnn.data import residue_constants as rc

    return rc.restype_order_with_x[one_letter]


def _slot(name):
    from fampnn.data import residue_constants as rc
    from pxf import atom37

    return list(rc.non_bb_idxs).index(atom37.ATOM37.index(name))


# ---- the robust function ----------------------------------------------------


def test_robust_is_quadratic_near_zero_and_linear_far_out():
    small = torch.tensor([1e-3])
    assert float(JL.robust(small)) == pytest.approx(0.5e-6, rel=1e-4)
    big = torch.tensor([100.0])
    assert float(JL.robust(big)) == pytest.approx(99.0, rel=1e-3)
    assert float(JL.robust(torch.zeros(1))) == 0.0


def test_robust_does_not_lose_precision_near_zero():
    """The naive ``sqrt(1+u^2) - 1`` cancels in float32 exactly where it matters.

    A converging run spends its time at small errors, so a 5% bias there is a
    5% bias on most of the gradient the term ever contributes.
    """
    u = torch.tensor([1e-3])
    naive = float(torch.sqrt(1.0 + u.pow(2)) - 1.0)
    exact = 0.5e-6
    assert abs(naive - exact) / exact > 0.01, "the naive form really is inaccurate"
    assert abs(float(JL.robust(u)) - exact) / exact < 1e-4


# ---- symmetry ---------------------------------------------------------------


def test_the_symmetry_table_is_upstreams():
    """Four residues, no invented entries: a swap the target does not license
    would silently relabel supervision."""
    pairs = JL.symmetry_pairs()
    residues = {int(row[0]) for row in pairs}
    assert residues == {_aa("D"), _aa("E"), _aa("F"), _aa("Y")}
    assert pairs.shape[1] == 3


def _aspartate(prediction_swapped=False):
    """One ASP with OD1/OD2 either matching the target or swapped."""
    length = 1
    predicted = torch.zeros(1, length, 33, 3)
    native = torch.zeros(1, length, 33, 3)
    mask = torch.zeros(1, length, 33)
    od1, od2, cg = _slot("OD1"), _slot("OD2"), _slot("CG")
    for slot in (od1, od2, cg):
        mask[0, 0, slot] = 1.0
    native[0, 0, cg] = torch.tensor([1.0, 0.0, 0.0])
    native[0, 0, od1] = torch.tensor([2.0, 1.0, 0.0])
    native[0, 0, od2] = torch.tensor([2.0, -1.0, 0.0])
    predicted[0, 0, cg] = native[0, 0, cg]
    if prediction_swapped:
        predicted[0, 0, od1] = native[0, 0, od2]
        predicted[0, 0, od2] = native[0, 0, od1]
    else:
        predicted[0, 0, od1] = native[0, 0, od1]
        predicted[0, 0, od2] = native[0, 0, od2]
    aatype = torch.full((1, length), _aa("D"), dtype=torch.long)
    return predicted, native, mask, aatype


def test_an_equivalent_swap_costs_nothing():
    """ASP's two carboxyl oxygens are interchangeable; naming is not an error."""
    swapped = _aspartate(prediction_swapped=True)
    straight = _aspartate(prediction_swapped=False)
    with_symmetry = float(JL.placement_loss(*swapped[:3], aatype=swapped[3])[0])
    perfect = float(JL.placement_loss(*straight[:3], aatype=straight[3])[0])
    assert with_symmetry == pytest.approx(perfect, abs=1e-9) == 0.0

    without = float(JL.placement_loss(*swapped[:3], aatype=swapped[3], symmetry=False)[0])
    assert without > 0.1, "the swap is only free because it was resolved"


def test_a_non_equivalent_swap_still_costs():
    """CG is not interchangeable with OD1, and must not be quietly matched."""
    predicted, native, mask, aatype = _aspartate()
    cg, od1 = _slot("CG"), _slot("OD1")
    predicted[0, 0, cg], predicted[0, 0, od1] = (
        native[0, 0, od1].clone(),
        native[0, 0, cg].clone(),
    )
    loss = float(JL.placement_loss(predicted, native, mask, aatype=aatype)[0])
    assert loss > 0.1


def test_a_swap_is_only_considered_when_both_atoms_are_observed():
    """Swapping a present atom onto a missing one relabels the supervision."""
    predicted, native, mask, aatype = _aspartate(prediction_swapped=True)
    mask[0, 0, _slot("OD2")] = 0.0  # OD2 unobserved
    resolved, _mask = JL.resolve_symmetry(predicted, native, mask, aatype)
    assert torch.equal(resolved, native), "no swap may be applied"
    assert float(JL.placement_loss(predicted, native, mask, aatype=aatype)[0]) > 0.1


def test_symmetry_is_resolved_once_and_detached():
    predicted, native, mask, aatype = _aspartate(prediction_swapped=True)
    predicted = predicted.requires_grad_(True)
    resolved, _mask = JL.resolve_symmetry(predicted, native, mask, aatype)
    assert not resolved.requires_grad


# ---- weighting and reduction -------------------------------------------------


def test_placement_carries_no_edm_weight(tmp_path):
    """A frame error must not be amplified as the side-chain noise vanishes.

    The local term's ``1/c_out^2`` goes to infinity at the clean end, which is
    correct for an error that vanishes with the noise. Placement error does not
    vanish with it -- at near-clean side-chain noise what is left is backbone --
    so the same weight would multiply a backbone error by an arbitrary number.
    """
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.provenance import fampnn_checkpoint

    bundle = torch.load(fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False)
    model = SeqDenoiser(bundle["model_cfg"])
    interpolant = model.denoiser.scn_diffusion_module.scn_interpolant

    noisy = interpolant.get_loss_weight(torch.tensor([0.1]))
    clean = interpolant.get_loss_weight(torch.tensor([0.999]))
    assert float(clean) > 100 * float(noisy), "the EDM weight really does diverge"

    predicted = torch.zeros(1, 2, 33, 3)
    native = torch.full((1, 2, 33, 3), 0.5)
    mask = torch.ones(1, 2, 33)
    # placement_loss takes no sigma at all: there is nothing for it to diverge by.
    loss, _stats = JL.placement_loss(predicted, native, mask)
    assert float(loss) == pytest.approx(float(JL.robust(torch.tensor(0.5))), rel=1e-6)


def test_placement_reduces_per_example_then_over_clones():
    predicted = torch.zeros(2, 1, 33, 3)
    native = torch.zeros(2, 1, 33, 3)
    native[0, 0, 0] = 1.0  # all three components of clone 0's first atom
    mask = torch.zeros(2, 1, 33)
    mask[:, 0, 0] = 1.0
    loss, stats = JL.placement_loss(predicted, native, mask)
    # Clone 0: rho(1) on each of its three scored components, over three
    # components. Clone 1: nothing wrong. The mean is over clones, so one bad
    # clone in two is half the error, not a third of it.
    expected = (float(JL.robust(torch.tensor(1.0))) + 0.0) / 2
    assert float(loss) == pytest.approx(expected, rel=1e-6)
    assert int(stats["placement_atoms"]) == 2


def test_placement_reports_an_rmsd_in_angstroms():
    predicted = torch.zeros(1, 1, 33, 3)
    native = torch.zeros(1, 1, 33, 3)
    native[0, 0, 0, 0] = 3.0
    mask = torch.zeros(1, 1, 33)
    mask[0, 0, 0] = 1.0
    _loss, stats = JL.placement_loss(predicted, native, mask)
    assert float(stats["placement_rmsd"]) == pytest.approx(3.0)


def test_nothing_scored_is_a_finite_zero():
    predicted = torch.zeros(1, 2, 33, 3, requires_grad=True)
    native = torch.ones(1, 2, 33, 3)
    loss, stats = JL.placement_loss(predicted, native, torch.zeros(1, 2, 33))
    assert float(loss) == 0.0 and int(stats["placement_atoms"]) == 0
    loss.backward()
    assert torch.isfinite(predicted.grad).all()


# ---- the combined objective --------------------------------------------------


def test_the_combined_objective_sums_with_its_coefficients():
    combined = JL.combined_loss(
        torch.tensor(1.0),
        local=torch.tensor(2.0),
        placement=torch.tensor(4.0),
        lambda_local=0.5,
        lambda_place=0.25,
    )
    assert float(combined.total) == pytest.approx(1.0 + 1.0 + 1.0)
    scalars = combined.scalars()
    assert scalars["loss_bb"] == 1.0 and scalars["lambda_local"] == 0.5


def test_a_non_finite_term_is_refused_rather_than_zeroed():
    """Deliberately unlike the source-parity total_loss, and for a reason.

    Replacing a NaN with zero is right for an unattended training run. Here it
    would turn the candidate arm into the baseline arm for that step, and the
    two are then compared as though they had differed.
    """
    with pytest.raises(ValueError, match="L_local is nan"):
        JL.combined_loss(
            torch.tensor(1.0), local=torch.tensor(float("nan")), sample_id="T1031"
        )
    with pytest.raises(ValueError, match="T1031"):
        JL.combined_loss(
            torch.tensor(1.0), placement=torch.tensor(float("inf")), sample_id="T1031"
        )


def test_the_generic_loss_still_zeroes_a_nan():
    """The parity implementation is unchanged; only the experiment is strict."""
    from pxf.train import losses as loss_fns

    total = loss_fns.total_loss(torch.tensor(float("nan")), torch.tensor(1.0))
    assert float(total) == pytest.approx(1.0)


# ---- calibration -------------------------------------------------------------


def test_calibrate_targets_a_gradient_ratio():
    assert JL.calibrate(10.0, 5.0, ratio=0.1) == pytest.approx(0.2)
    assert JL.calibrate(1.0, 1.0, ratio=0.3) == pytest.approx(0.3)


def test_calibrating_against_a_dead_gradient_is_refused():
    """A zero auxiliary gradient means the term does not reach the backbone.

    Dividing around it with an epsilon would produce an enormous coefficient for
    a term that cannot train anything.
    """
    for dead in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="does not reach the backbone"):
            JL.calibrate(1.0, dead)


def test_gradient_norm_needs_a_differentiable_target():
    target = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="does not require grad"):
        JL.gradient_norm(torch.tensor(1.0), target)


def test_gradient_norm_measures_the_backbone_derivative():
    target = torch.zeros(4, 3, requires_grad=True)
    loss = (target - 2.0).pow(2).sum()
    # d/dx sum (x-2)^2 = 2(x-2) = -4 everywhere, so the RMS is 4.
    assert JL.gradient_norm(loss, target) == pytest.approx(4.0)
    assert math.isfinite(JL.gradient_norm(loss, target))
