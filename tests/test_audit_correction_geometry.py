"""The displacement diagnostic, against motions whose answer is known in advance.

Written because the first version of this script crashed on its first target,
in a GPU job, after waiting two hours in a queue: it called
``backbone_metrics._kabsch`` as though it were batched and returned aligned
coordinates. It takes UNBATCHED ``[L, 3]`` and returns post-superposition
DISTANCES, and ``superposed_rmsd`` is the wrapper that gets both right.

The lesson is not "read the signature" -- it is that a diagnostic reporting a
number nobody can check by eye should be run on inputs whose answer is known.
A pure rigid motion must come back as entirely rigid; independent jitter must
come back as entirely local. Neither assertion would have survived the bug, and
both run in milliseconds on the CPU.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from audit_correction_geometry import (  # noqa: E402
    SLOT_C,
    SLOT_CA,
    SLOT_N,
    SLOT_O,
    backbone_chemistry,
    displacement_stats,
    pearson,
)


def rotation_about_z(radians):
    c, s = torch.cos(torch.tensor(radians)), torch.sin(torch.tensor(radians))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def cloud():
    torch.manual_seed(0)
    return torch.randn(200, 3) * 10.0


def test_a_pure_rigid_motion_is_reported_as_entirely_rigid(cloud):
    moved = cloud @ rotation_about_z(0.3).T + torch.tensor([1.0, 2.0, 3.0])
    stats = displacement_stats(cloud, moved, torch.ones(200, dtype=torch.bool))
    assert stats["rms_displacement"] > 1.0, "the test motion should be large"
    assert stats["rms_after_superposition"] < 1e-4
    assert stats["rigid_fraction"] > 0.999


def test_independent_jitter_is_reported_as_entirely_local(cloud):
    torch.manual_seed(1)
    moved = cloud + torch.randn(200, 3) * 0.1
    stats = displacement_stats(cloud, moved, torch.ones(200, dtype=torch.bool))
    assert stats["rigid_fraction"] < 0.1
    assert stats["rms_after_superposition"] == pytest.approx(
        stats["rms_displacement"], rel=0.05
    )


def test_no_motion_reports_zero_rather_than_dividing_by_it(cloud):
    stats = displacement_stats(cloud, cloud.clone(), torch.ones(200, dtype=torch.bool))
    assert stats["rms_displacement"] == 0.0
    assert stats["rigid_fraction"] == 0.0
    assert stats["displacement_max"] == 0.0


def test_the_mask_selects_which_atoms_are_measured(cloud):
    moved = cloud.clone()
    moved[0] += torch.tensor([100.0, 0.0, 0.0])
    keep = torch.ones(200, dtype=torch.bool)
    assert displacement_stats(cloud, moved, keep)["displacement_max"] > 99
    keep[0] = False
    assert displacement_stats(cloud, moved, keep)["displacement_max"] == 0.0


# A straight chain laid along x with every bond at its ideal length, so the
# baseline deviation is zero and a perturbation is the entire signal. The
# spacing is the sum of the three bonds that span one residue-to-residue step.
SPACING = 1.458 + 1.525 + 1.329


def ideal_backbone(length=8, spacing=SPACING):
    """An extended chain whose bond LENGTHS are all exactly ideal.

    Angles are not: a straight chain has N-CA-C at 180 degrees, not 111. That is
    deliberate -- these tests exercise the bond and clash terms, and an angle
    baseline of zero would need real geometry rather than a fixture.
    """
    coords = torch.zeros(length, 37, 3)
    for i in range(length):
        origin = torch.tensor([i * spacing, 0.0, 0.0])
        coords[i, SLOT_N] = origin
        coords[i, SLOT_CA] = origin + torch.tensor([1.458, 0.0, 0.0])
        coords[i, SLOT_C] = origin + torch.tensor([1.458 + 1.525, 0.0, 0.0])
        coords[i, SLOT_O] = coords[i, SLOT_C] + torch.tensor([0.0, 1.231, 0.0])
    return coords


def test_chemistry_reports_zero_on_ideal_bond_lengths():
    """The baseline has to be zero, or a perturbation cannot be read against it."""
    report = backbone_chemistry(ideal_backbone(), torch.arange(8), torch.ones(8))
    assert report["bond_rms_deviation"] < 1e-5
    assert report["bond_max_deviation"] < 1e-5
    assert report["backbone_clashes"] == 0


def test_chemistry_notices_a_stretched_bond():
    coords = ideal_backbone()
    clean = backbone_chemistry(coords, torch.arange(8), torch.ones(8))
    # Displacing CA by 1.5 A perpendicular to an ideal 1.458 A N-CA bond
    # lengthens it to sqrt(1.458^2 + 1.5^2) = 2.09, a deviation of 0.63.
    coords[3, SLOT_CA] += torch.tensor([0.0, 0.0, 1.5])
    broken = backbone_chemistry(coords, torch.arange(8), torch.ones(8))
    assert clean["bond_max_deviation"] < 1e-5
    assert broken["bond_max_deviation"] == pytest.approx(0.634, abs=0.01)


def test_chemistry_notices_a_clash():
    coords = ideal_backbone(length=8)
    clean = backbone_chemistry(coords, torch.arange(8), torch.ones(8))
    # Fold residue 7 back on top of residue 0.
    coords[7] = coords[0] + torch.tensor([0.0, 0.0, 0.5])
    clashed = backbone_chemistry(coords, torch.arange(8), torch.ones(8))
    assert clashed["backbone_clashes"] > clean["backbone_clashes"]


def test_a_chain_break_is_not_counted_as_a_peptide_bond():
    """Non-consecutive residue indices must not register a 40 A bond."""
    coords = ideal_backbone(length=6)
    contiguous = backbone_chemistry(coords, torch.arange(6), torch.ones(6))
    broken_index = torch.tensor([0, 1, 2, 50, 51, 52])
    split = backbone_chemistry(coords, broken_index, torch.ones(6))
    assert split["bond_max_deviation"] <= contiguous["bond_max_deviation"] + 1e-6


def test_pearson_is_nan_when_one_side_does_not_vary():
    assert pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) != pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
