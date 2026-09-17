"""The backbone instruments the before/after comparison is read from.

All three have to be invariant to the coordinate frame, because bb0 and bb1 come
out of the featurizer's frame while the reference comes from FaMPNN's parse of
the same file, and the two are not even centred the same way. RMSD and TM-score
get that from superposing; lDDT gets it from using distances only. A metric that
quietly was not frame-invariant would report the centring offset as backbone
error, which is exactly the mistake ``place_on_native_backbone`` exists to fix on
the side-chain side.
"""

import math

import pytest
import torch

from pxf import atom37
from pxf.eval import backbone_metrics as bb


def helix(length=40):
    """An ideal alpha helix's N, CA, C, O in atom37, as a plausible backbone."""
    coords = torch.zeros(length, atom37.NUM_ATOM37, 3)
    for i in range(length):
        angle = i * 100.0 * math.pi / 180.0
        rise = i * 1.5
        ca = torch.tensor([2.3 * math.cos(angle), 2.3 * math.sin(angle), rise])
        coords[i, 1] = ca
        coords[i, 0] = ca + torch.tensor([0.5, -0.8, -0.9])
        coords[i, 2] = ca + torch.tensor([-0.4, 0.9, 0.9])
        coords[i, 4] = ca + torch.tensor([-1.1, 1.6, 1.5])
    return coords


def rigid(coords, *, seed=0):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))[None, :]
    if float(torch.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([17.0, -9.0, 4.0], dtype=torch.float64)
    return (coords.double() @ q + shift).to(coords.dtype)


def full_mask(length):
    mask = torch.zeros(length, atom37.NUM_ATOM37, dtype=torch.bool)
    mask[:, list(atom37.BACKBONE_SLOTS)] = True
    return mask


# --- RMSD -------------------------------------------------------------------


def test_rmsd_is_zero_for_identical_structures():
    coords = helix()[:, 1]
    assert bb.superposed_rmsd(coords, coords) == pytest.approx(0.0, abs=1e-6)


def test_rmsd_ignores_a_rigid_motion():
    coords = helix()[:, 1]
    assert bb.superposed_rmsd(rigid(coords), coords) == pytest.approx(0.0, abs=1e-4)


def test_rmsd_sees_a_real_deformation():
    coords = helix()[:, 1]
    moved = coords.clone()
    moved[20:] += torch.tensor([1.0, 0.0, 0.0])
    value = bb.superposed_rmsd(moved, coords)
    assert 0.1 < value < 1.0, value


def test_rmsd_respects_the_mask():
    coords = helix()[:, 1]
    moved = coords.clone()
    moved[0] += torch.tensor([50.0, 0.0, 0.0])
    keep = torch.ones(coords.shape[0], dtype=torch.bool)
    keep[0] = False
    assert bb.superposed_rmsd(moved, coords, keep) < bb.superposed_rmsd(moved, coords)


def test_too_few_atoms_gives_nan_not_a_crash():
    assert math.isnan(bb.superposed_rmsd(torch.zeros(2, 3), torch.zeros(2, 3)))


# --- TM-score ---------------------------------------------------------------


def test_tm_score_is_one_for_identical_structures():
    coords = helix()[:, 1]
    assert bb.tm_score(coords, coords) == pytest.approx(1.0, abs=1e-6)


def test_tm_score_ignores_a_rigid_motion():
    coords = helix()[:, 1]
    assert bb.tm_score(rigid(coords, seed=3), coords) == pytest.approx(1.0, abs=1e-3)


def test_tm_score_falls_as_the_structure_is_perturbed():
    coords = helix(60)[:, 1]
    generator = torch.Generator().manual_seed(0)
    scores = []
    for scale in (0.0, 0.5, 2.0, 6.0):
        noise = torch.randn(coords.shape, generator=generator) * scale
        scores.append(bb.tm_score(coords + noise, coords))
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < scores[-1] < 0.5


def test_tm_score_is_bounded():
    coords = helix(50)[:, 1]
    far = coords + 100.0
    value = bb.tm_score(far, coords)
    assert 0.0 <= value <= 1.0


def test_a_short_chain_gives_nan():
    assert math.isnan(bb.tm_score(torch.zeros(3, 3), torch.zeros(3, 3)))


# --- lDDT -------------------------------------------------------------------


def test_lddt_is_one_for_identical_structures():
    coords = helix()[:, 1]
    residue = torch.arange(coords.shape[0])
    score, pairs = bb.lddt(coords, coords, subject_residue=residue)
    assert score == pytest.approx(1.0)
    assert pairs > 0


def test_lddt_ignores_a_rigid_motion():
    coords = helix()[:, 1]
    residue = torch.arange(coords.shape[0])
    score, _pairs = bb.lddt(rigid(coords, seed=5), coords, subject_residue=residue)
    assert score == pytest.approx(1.0, abs=1e-3)


def test_lddt_falls_with_local_distortion():
    coords = helix()[:, 1]
    residue = torch.arange(coords.shape[0])
    generator = torch.Generator().manual_seed(1)
    moved = coords + torch.randn(coords.shape, generator=generator) * 1.5
    score, _pairs = bb.lddt(moved, coords, subject_residue=residue)
    assert 0.0 < score < 0.9


# --- the report -------------------------------------------------------------


def test_the_report_carries_every_headline_metric():
    coords = helix()
    report = bb.backbone_report(coords, coords, full_mask(coords.shape[0]))
    for key in bb.HEADLINE:
        assert key in report, key
    assert report["ca_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert report["tm_score"] == pytest.approx(1.0, abs=1e-6)
    assert report["lddt_ca"] == pytest.approx(1.0)
    assert report["scored_residues"] == coords.shape[0]
    assert report["scored_backbone_atoms"] == 4 * coords.shape[0]


def test_the_report_scores_only_masked_atoms():
    coords = helix()
    mask = full_mask(coords.shape[0])
    mask[5:] = False
    report = bb.backbone_report(coords, coords, mask)
    assert report["scored_residues"] == 5
    assert report["scored_backbone_atoms"] == 20


def test_improvement_is_sign_corrected():
    # Lower is better for RMSD, higher for lDDT and TM-score.
    assert bb.improvement("ca_rmsd", 1.0, 0.8) == pytest.approx(0.2)
    assert bb.improvement("backbone_rmsd", 0.8, 1.0) == pytest.approx(-0.2)
    assert bb.improvement("lddt_ca", 0.8, 0.9) == pytest.approx(0.1)
    assert bb.improvement("tm_score", 0.9, 0.8) == pytest.approx(-0.1)
    assert set(bb.LOWER_IS_BETTER) <= set(bb.HEADLINE)
