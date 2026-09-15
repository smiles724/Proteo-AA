"""Real-weights tests for the side-chain module.

FaMPNN's checkpoints ship inside its submodule, so these run without any
download. They cover the invariants that make this a *packing* pipeline rather
than a co-design one, plus an accuracy regression guard: packing a CASP15
backbone with its own sequence should reproduce the withheld native side chains
to roughly the accuracy FaMPNN reports, which no amount of correct-looking
plumbing achieves if the atom order or masks are wrong.
"""
import pytest
import torch

from pxf import atom37
from pxf.sidechain.fampnn import FaMPNNSideChainPacker

TARGET = "fampnn/data/casp15/pdbs/T1104-D1.pdb"
BB = list(atom37.BACKBONE_SLOTS)
SC = list(atom37.SIDECHAIN_SLOTS)


@pytest.fixture(scope="module")
def packer():
    # select_device probes a real kernel launch: this cluster mixes GPU
    # generations, and some are newer than the installed torch build supports.
    from pxf.device import select_device
    return FaMPNNSideChainPacker().to(select_device())


@pytest.fixture(scope="module")
def native():
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb
    from pxf.provenance import repo_root
    single = process_single_pdb(load_feats_from_pdb(str(repo_root() / TARGET)))
    return {k: v[None] for k, v in single.items() if torch.is_tensor(v) and v.ndim >= 1}


def _backbone_only(native):
    mask = torch.zeros_like(native["atom_mask"])
    mask[..., BB] = native["atom_mask"][..., BB]
    return native["x"] * mask[..., None], mask


def test_weights_load_strictly_and_are_frozen(packer):
    assert packer.variant == "0.0"
    assert not any(p.requires_grad for p in packer.model.parameters())
    assert packer.identity["designs_sequence"] is False
    assert packer.identity["mode"] == "sidechain_pack"


def test_ghost_mask_knows_which_slots_a_residue_has(packer):
    # Glycine has no side chain; tryptophan has the most atoms of the twenty.
    gly, trp = atom37.AA_ORDER.index("G"), atom37.AA_ORDER.index("W")
    aatype = torch.tensor([[gly, trp]], device=packer.device)
    exists = 1.0 - packer.ghost_atom_mask(aatype)
    assert exists[0, 0, BB].tolist() == [1.0] * 4
    assert exists[0, 0, SC].sum() == 0, "glycine should have no side-chain slots"
    assert exists[0, 1, SC].sum() == 10, "tryptophan side chain has ten atoms"


def test_missing_atom_mask_matches_the_upstream_formula(packer, native):
    from fampnn.data import residue_constants as rc
    _, mask = _backbone_only(native)
    aatype = native["aatype"].to(packer.device)
    ours = packer.missing_atom_mask(aatype, mask.to(packer.device))
    ghost = 1 - torch.as_tensor(rc.restype_atom37_mask, device=packer.device)[aatype.long()]
    theirs = (1 - mask.to(packer.device).float()) * (1 - ghost)
    assert torch.equal(ours, theirs)


def test_packing_returns_the_supplied_sequence_unchanged(packer, native):
    coords, mask = _backbone_only(native)
    out = packer(coords_af2=coords, aatype=native["aatype"], atom_mask=mask,
                 residue_index=native["residue_index"], chain_index=native["chain_index"])
    assert out["sequence"][0] == atom37.sequence_from_aatype(native["aatype"][0])
    assert torch.equal(out["aatype"].cpu(), native["aatype"].long())


def test_packing_does_not_move_the_backbone(packer, native):
    coords, mask = _backbone_only(native)
    out = packer(coords_af2=coords, aatype=native["aatype"], atom_mask=mask,
                 residue_index=native["residue_index"], chain_index=native["chain_index"])
    assert out["backbone_shift"] == pytest.approx(0.0, abs=1e-5)
    present = mask[..., BB].bool().to(packer.device)
    assert torch.allclose(out["coords_af2"][..., BB, :][present],
                          coords.to(packer.device)[..., BB, :][present])


def test_packed_sidechains_reproduce_the_withheld_native(packer, native):
    coords, mask = _backbone_only(native)
    out = packer(coords_af2=coords, aatype=native["aatype"], atom_mask=mask,
                 residue_index=native["residue_index"], chain_index=native["chain_index"])
    compare = (native["atom_mask"][..., SC].bool() & ~mask[..., SC].bool()).to(packer.device)
    assert int(compare.sum()) > 300, "the target should have side chains to compare against"
    delta = (out["coords_af2"][..., SC, :] - native["x"][..., SC, :].to(packer.device))[compare]
    rmsd = float(torch.sqrt((delta ** 2).sum(-1).mean()))
    # FaMPNN reports ~1 A all-sidechain-atom RMSD; a scrambled atom order or a
    # mis-built mask lands far above this bound rather than slightly above it.
    assert rmsd < 2.0, f"side-chain RMSD {rmsd:.3f} A is too high for a correct integration"


def test_output_mask_covers_every_slot_the_sequence_needs(packer, native):
    coords, mask = _backbone_only(native)
    out = packer(coords_af2=coords, aatype=native["aatype"], atom_mask=mask,
                 residue_index=native["residue_index"], chain_index=native["chain_index"])
    exists = (1.0 - packer.ghost_atom_mask(native["aatype"].to(packer.device))).bool()
    assert torch.all(out["atom_mask_af2"] >= exists)


def test_psce_is_reported_per_sidechain_slot(packer, native):
    coords, mask = _backbone_only(native)
    out = packer(coords_af2=coords, aatype=native["aatype"], atom_mask=mask,
                 residue_index=native["residue_index"], chain_index=native["chain_index"])
    assert out["psce"].shape == (1, native["x"].shape[1], len(SC))
    assert torch.isfinite(out["psce"]).all()


def test_batching_packs_every_sample(packer, native):
    coords, mask = _backbone_only(native)
    out = packer(coords_af2=coords.expand(3, -1, -1, -1), atom_mask=mask.expand(3, -1, -1),
                 aatype=native["aatype"], residue_index=native["residue_index"],
                 chain_index=native["chain_index"])
    length = native["x"].shape[1]
    assert out["coords_af2"].shape == (3, length, 37, 3)
    assert out["psce"].shape == (3, length, len(SC))
    assert len(out["sequence"]) == 3 and len(set(out["sequence"])) == 1


def test_packing_is_a_sampler_so_repeats_differ_but_stay_close(packer, native):
    """Repacking one backbone twice explores rotamers; that is the point of it."""
    coords, mask = _backbone_only(native)
    out = packer(coords_af2=coords.expand(2, -1, -1, -1), atom_mask=mask.expand(2, -1, -1),
                 aatype=native["aatype"], residue_index=native["residue_index"],
                 chain_index=native["chain_index"])
    present = native["atom_mask"][..., SC].bool()[0].to(packer.device)
    delta = (out["coords_af2"][0][..., SC, :] - out["coords_af2"][1][..., SC, :])[present]
    spread = float(torch.sqrt((delta ** 2).sum(-1).mean()))
    assert spread > 1e-4, "identical output would mean the sampler is not sampling"
    assert spread < 2.0, f"same backbone and sequence should give similar packings, got {spread:.3f} A"


def test_seeding_makes_packing_reproducible(packer, native):
    """A seed pins the sampling; bitwise equality is a separate, stricter claim.

    On CUDA the default kernels contribute ~1e-5 A of jitter between identical
    runs, so reproducibility is asserted at float tolerance here and bitwise only
    where the kernels are deterministic. The seed is clearly doing its job either
    way: a different seed moves atoms by Angstroms, not microns.
    """
    coords, mask = _backbone_only(native)
    kwargs = dict(coords_af2=coords, atom_mask=mask, aatype=native["aatype"],
                  residue_index=native["residue_index"], chain_index=native["chain_index"])
    first = packer(seed=7, **kwargs)["coords_af2"]
    second = packer(seed=7, **kwargs)["coords_af2"]
    assert torch.allclose(first, second, atol=1e-3)
    if packer.device.type == "cpu":
        assert torch.equal(first, second), "CPU kernels should be bitwise deterministic"
    third = packer(seed=8, **kwargs)["coords_af2"]
    # Orders of magnitude larger than the kernel jitter, so the seed matters.
    assert float((first - third).abs().max()) > 0.1


def test_a_single_structure_may_omit_the_batch_axis(packer, native):
    coords, mask = _backbone_only(native)
    out = packer(coords_af2=coords[0], aatype=native["aatype"][0], atom_mask=mask[0],
                 residue_index=native["residue_index"][0], chain_index=native["chain_index"][0])
    assert out["coords_af2"].shape == (native["x"].shape[1], 37, 3)


def test_unknown_residues_are_refused_rather_than_designed(packer, native):
    coords, mask = _backbone_only(native)
    aatype = native["aatype"].clone().long()
    aatype[0, 5] = atom37.UNKNOWN_AA_INDEX
    with pytest.raises(ValueError, match="will not design one"):
        packer(coords_af2=coords, aatype=aatype, atom_mask=mask,
               residue_index=native["residue_index"], chain_index=native["chain_index"])


def test_mismatched_shapes_are_refused(packer, native):
    coords, mask = _backbone_only(native)
    with pytest.raises(ValueError, match="does not match coordinates"):
        packer(coords_af2=coords, aatype=native["aatype"][:, :-2], atom_mask=mask)
    with pytest.raises(ValueError, match=r"Expected \[B, L, 37, 3\]"):
        packer(coords_af2=coords[..., :14, :], aatype=native["aatype"], atom_mask=mask)


def test_keeping_input_sidechains_as_context_reproduces_them(packer, native):
    """With full side-chain context the model has nothing left to infer."""
    out = packer(coords_af2=native["x"], aatype=native["aatype"],
                 atom_mask=native["atom_mask"],
                 residue_index=native["residue_index"], chain_index=native["chain_index"],
                 scn_context_mask=torch.ones(1, native["x"].shape[1]))
    present = native["atom_mask"][..., SC].bool().to(packer.device)
    delta = (out["coords_af2"][..., SC, :] - native["x"][..., SC, :].to(packer.device))[present]
    rmsd = float(torch.sqrt((delta ** 2).sum(-1).mean()))
    assert rmsd < 0.5, f"context side chains should be preserved, got {rmsd:.3f} A"
