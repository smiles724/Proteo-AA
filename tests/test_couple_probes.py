"""The sensitivity probe has to establish four things, or its number is noise.

* the floor: re-encoding the identical input twice;
* a valid chi perturbation clears that floor;
* padded and nonexistent atoms do not move the representation at all;
* rigidly rotating and translating the whole input does not either.

The last two are the ones that make the second meaningful. A response that came
from junk in a padded slot, or one that would disappear under a change of
coordinate frame, is not information about the packing.
"""

import pytest
import torch

from pxf import atom37, provenance
from pxf.couple import fampnn_iface as iface
from pxf.couple import probes


@pytest.fixture(scope="module")
def fampnn():
    from fampnn.model.sd_model import SeqDenoiser

    bundle = torch.load(
        provenance.fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False
    )
    model = SeqDenoiser(bundle["model_cfg"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def item():
    from pxf.provenance import repo_root
    from pxf.train.data import StructureCropDataset, collate

    dataset = StructureCropDataset(
        [str(repo_root() / "fampnn/data/casp14/pdbs/T1031.pdb")],
        crop_size=40,
        noise=0.0,
        seed=0,
    )
    return collate([dataset[0]])


@pytest.fixture(scope="module")
def report(fampnn, item):
    """One probe run over a real structure, reused by every assertion."""
    return probes.sidechain_sensitivity(
        fampnn,
        item["x"],
        item["aatype"].long(),
        seq_mask=item["seq_mask"],
        missing_atom_mask=item["missing_atom_mask"],
        residue_index=item["residue_index"],
        chain_index=item["chain_index"],
        torsion_perturbations=(10.0, 60.0, 120.0),
        perturbations=(0.5,),
        generator=torch.Generator().manual_seed(0),
    )


def test_the_floor_is_essentially_zero(report):
    """A deterministic encoder re-encoding identical input must not move."""
    assert report.floor < 1e-5, f"the encoder is not deterministic: {report.floor}"


def test_the_ceiling_is_a_real_effect(report):
    """Masked -> visible side chains is the largest the input can do."""
    assert report.ceiling > 1e-2, (
        "showing the encoder side chains barely changed h_V, so there is no "
        f"headroom for any perturbation to register in (ceiling {report.ceiling})"
    )


def test_a_valid_chi_perturbation_clears_the_floor(report):
    family = report.family("chi_")
    assert family, "no torsion probe ran"
    best = max(family.values())
    assert best > max(report.floor, 1e-9) * probes.INVARIANCE_MARGIN, (
        "rotating about chi axes -- the only perturbation a packer could "
        f"actually produce -- did not move h_packed above the noise floor: "
        f"{family} against floor {report.floor}"
    )
    assert report.verdict() in ("weak", "informative")


def test_bigger_rotations_move_the_representation_more(report):
    small = report.responses["chi_10deg"]
    large = report.responses["chi_120deg"]
    assert large > small, (
        "a 120 degree rotamer flip should register more than a 10 degree "
        f"wiggle; got {small} and {large}"
    )


def test_nonexistent_atoms_cannot_reach_a_real_residue(report):
    """Exact: availability excludes the slots, so nothing can read them."""
    assert report.invariants["nonexistent_atoms_scrambled"] == 0.0, (
        "scrambling atom37 slots the residue type does not have moved the "
        "representation, so the adapter is reading coordinates that do not "
        "correspond to an atom"
    )


def test_padded_coordinates_leak_only_at_the_measured_upstream_scale(report):
    """Not exact, and pinned rather than hidden.

    FaMPNN's encoder lets 2.6e-4 to 1.2e-3 of a padded row's coordinates into
    real residues' features -- measured across lengths 40 to 64, on native and
    predicted packings, and not a neighbour-count effect. The coupling path never pads, which is the actual
    mitigation; this bounds the size in case it ever does.
    """
    value = report.invariants["padded_atom_coordinates"]
    assert value > 0.0, (
        "the upstream leak has gone away, which is good news but means this "
        "test and the notes in pxf.couple.probes are now stale"
    )
    assert value < probes.INVARIANT_TOLERANCE["padded_atom_coordinates"], (
        f"the leak grew to {value}, an order of magnitude past what was "
        "measured; the feedback features are no longer a function of one "
        "structure"
    )


def test_appending_padding_rows_only_moves_the_terminus(report):
    """A length effect in FaMPNN's encoder, documented rather than asserted away.

    Appending rows shifts the last real residue because the encoder reads an
    index-neighbour feature and the C-terminus stops looking like one. It is not
    leakage -- the invariant above pins that -- and the coupling path never pads.
    """
    assert report.stats["terminus_shift_from_padding"] < 1e-2
    assert (
        report.stats["terminus_shift_worst_value"]
        > 10 * report.stats["terminus_shift_from_padding"]
    ), "the shift should be concentrated, not spread across the structure"


def test_a_rigid_motion_does_not_affect_it(report):
    value = report.invariants["rigid_rotation_translation"]
    assert value < probes.INVARIANT_TOLERANCE["rigid_rotation_translation"], (
        "rotating and translating the whole structure changed h_packed by "
        f"{value}; the readout's features are supposed to be invariant"
    )
    assert report.stats["rigid_determinant"] == pytest.approx(1.0, abs=1e-6)
    assert report.stats["rigid_shift_norm"] > 1.0, "the probe barely moved anything"


def test_no_invariance_failures(report):
    assert report.invariance_failures() == {}, f"summary: {report.summary()}"


def test_the_probe_runs_on_a_predicted_packing(fampnn, item):
    """The realization the feedback would really see, not the native one."""
    backbone = list(atom37.BACKBONE_SLOTS)
    given = torch.zeros_like(item["missing_atom_mask"])
    given[..., backbone] = 1.0
    coords = item["x"] * given[..., None]
    _logits, _h, features = iface.encode(
        fampnn,
        coords,
        item["aatype"].long(),
        seq_mask=item["seq_mask"],
        residue_index=item["residue_index"],
        chain_index=item["chain_index"],
    )
    torch.manual_seed(0)
    packed, _aux = iface.pack_from_features(
        fampnn,
        features,
        item["aatype"].long(),
        seq_mask=item["seq_mask"],
        residue_index=item["residue_index"],
        chain_index=item["chain_index"],
        num_steps=3,
    )
    report = probes.sidechain_sensitivity(
        fampnn,
        coords,
        item["aatype"].long(),
        seq_mask=item["seq_mask"],
        missing_atom_mask=item["missing_atom_mask"],
        residue_index=item["residue_index"],
        chain_index=item["chain_index"],
        sidechains=packed,
        torsion_perturbations=(60.0,),
        perturbations=(),
        generator=torch.Generator().manual_seed(0),
    )
    assert report.stats["sidechains_supplied"] is True
    assert report.responses["chi_60deg"] > max(report.floor, 1e-9) * 10
    assert report.invariance_failures() == {}, report.summary()
