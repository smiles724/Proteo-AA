"""The pipeline must source the sequence explicitly and never let FaMPNN design it."""
import pytest
import torch

from pxf import atom37
from pxf.backbone.pxdesign import BackboneBatch
from pxf.pipeline import FullAtomPipeline, PackedDesign


class StubPacker:
    """Stands in for FaMPNN: records its inputs, returns recognisable output."""

    def __init__(self):
        self.identity = dict(backend="stub")
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        coords = kwargs["coords_af2"]
        batch, length = coords.shape[0], coords.shape[1]
        return dict(coords_af2=torch.full_like(coords, 9.0),
                    atom_mask_af2=torch.ones(batch, length, 37, dtype=torch.bool),
                    psce=torch.full((batch, length, 33), 0.25),
                    aatype=kwargs["aatype"], backbone_shift=0.0,
                    sequence=["A" * length] * batch)


def _batch(length=4, design=(False, False, True, True), samples=2, native="LQXX"):
    coords = torch.zeros(samples, length, 37, 3)
    mask = torch.zeros(samples, length, 37, dtype=torch.bool)
    coords[..., list(atom37.BACKBONE_SLOTS), :] = 1.0
    mask[..., list(atom37.BACKBONE_SLOTS)] = True
    known = torch.tensor([letter in atom37.AA_ORDER for letter in native])
    return BackboneBatch(
        sample_name="target", coords_af2=coords, atom_mask_af2=mask,
        design_mask=torch.tensor(design), native_sequence=native, sequence_known=known,
        residue_index=torch.arange(length), chain_index=torch.zeros(length, dtype=torch.long))


def test_native_positions_need_no_sequence_argument():
    batch = _batch(length=2, design=(False, False), native="LQ")
    assert FullAtomPipeline(None, StubPacker()).resolve_sequence(batch) == "LQ"


def test_design_positions_require_a_supplied_sequence():
    pipeline = FullAtomPipeline(None, StubPacker())
    with pytest.raises(ValueError, match="will not design one"):
        pipeline.resolve_sequence(_batch())


def test_supplied_sequence_fills_only_the_design_positions():
    pipeline = FullAtomPipeline(None, StubPacker(), sequence="QQWY")
    # Native L,Q are kept; only the two design tokens take W,Y.
    assert pipeline.resolve_sequence(_batch()) == "LQWY"


def test_supplied_sequence_length_is_checked():
    pipeline = FullAtomPipeline(None, StubPacker(), sequence="WY")
    with pytest.raises(ValueError, match="length 2"):
        pipeline.resolve_sequence(_batch())


def test_positional_overrides_are_accepted():
    pipeline = FullAtomPipeline(None, StubPacker(), sequence={2: "W", 3: "Y"})
    assert pipeline.resolve_sequence(_batch()) == "LQWY"


def test_the_packer_receives_the_resolved_sequence_as_aatype():
    stub = StubPacker()
    FullAtomPipeline(None, stub, sequence="QQWY").run_batch(_batch())
    aatype = stub.calls[0]["aatype"]
    assert aatype.shape == (2, 4)
    assert atom37.sequence_from_aatype(aatype[0]) == "LQWY"


def test_default_gives_no_sidechain_context():
    stub = StubPacker()
    FullAtomPipeline(None, stub, sequence="QQWY").run_batch(_batch())
    assert torch.all(stub.calls[0]["scn_context_mask"] == 0)


def test_keep_context_preserves_target_rotamers_only():
    stub = StubPacker()
    FullAtomPipeline(None, stub, sequence="QQWY",
                     scn_context="keep_context").run_batch(_batch())
    # Context where NOT designed: the two native positions.
    assert stub.calls[0]["scn_context_mask"][0].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_unknown_context_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown scn_context"):
        FullAtomPipeline(None, StubPacker(), scn_context="sometimes")


def test_results_carry_one_entry_per_backbone_sample():
    results = FullAtomPipeline(None, StubPacker(), sequence="QQWY").run_batch(_batch())
    assert len(results) == 2
    assert [r.sample_index for r in results] == [0, 1]
    assert all(isinstance(r, PackedDesign) for r in results)
    assert all(r.sequence == "LQWY" for r in results)


def test_designed_sequence_selects_only_design_positions():
    result = FullAtomPipeline(None, StubPacker(), sequence="QQWY").run_batch(_batch())[0]
    assert result.designed_sequence() == "WY"


def test_metrics_report_psce_overall_and_on_designed_positions():
    result = FullAtomPipeline(None, StubPacker(), sequence="QQWY").run_batch(_batch())[0]
    assert result.metrics["num_designed"] == 2
    assert result.metrics["mean_psce"] == pytest.approx(0.25)
    assert result.metrics["mean_psce_designed"] == pytest.approx(0.25)


def test_identity_records_that_nothing_is_designed():
    class StubBackbone:
        identity = dict(backend="pxdesign")
    identity = FullAtomPipeline(StubBackbone(), StubPacker()).identity
    assert identity["pipeline"] == "pxdesign->fampnn(pack)"
    assert identity["designs_sequence"] is False
