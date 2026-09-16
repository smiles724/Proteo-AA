"""Adapters must start as an exact no-op and gate cleanly by phase."""

import pytest
import torch

from pxf.couple.adapters import CouplingAdapters, NoiseEmbedding, ResidualAdapter


def test_zero_init_is_an_exact_no_op():
    """Phase-0 equivalence has to be exact, not approximate."""
    adapter = ResidualAdapter(384, 128)
    assert adapter.is_identity()
    out = adapter(torch.randn(2, 10, 384), torch.tensor([0.5, 5.0]))
    assert out.shape == (2, 10, 128)
    assert bool((out == 0).all())


def test_a_nudge_breaks_identity_and_produces_signal():
    adapter = ResidualAdapter(64, 32)
    with torch.no_grad():
        adapter.project_out.weight.normal_(0, 0.02)
    assert not adapter.is_identity()
    assert float(adapter(torch.randn(1, 5, 64), torch.tensor([1.0])).abs().sum()) > 0


def test_noise_level_changes_the_residual():
    """A_BS must be able to weight side-chain evidence differently per sigma."""
    adapter = ResidualAdapter(64, 32)
    with torch.no_grad():
        adapter.project_out.weight.normal_(0, 0.05)
    source = torch.randn(2, 6, 64)
    low = adapter(source, torch.tensor([0.05, 0.05]))
    high = adapter(source, torch.tensor([80.0, 80.0]))
    assert not torch.allclose(low, high)


def test_noise_embedding_is_finite_across_the_edm_range():
    embedding = NoiseEmbedding(64)
    values = embedding(torch.tensor([1e-8, 0.01, 1.0, 80.0, 1e4]))
    assert values.shape == (5, 64)
    assert torch.isfinite(values).all()


def test_noise_embedding_rejects_odd_width():
    with pytest.raises(ValueError, match="even"):
        NoiseEmbedding(63)


def test_scalar_and_broadcast_sigma():
    adapter = ResidualAdapter(16, 8)
    assert adapter(torch.randn(3, 4, 16), 1.0).shape == (3, 4, 8)
    assert adapter(torch.randn(3, 4, 16), torch.tensor([2.0])).shape == (3, 4, 8)
    with pytest.raises(ValueError, match="sigma values"):
        adapter(torch.randn(3, 4, 16), torch.tensor([1.0, 2.0]))


def test_wrong_input_width_is_refused():
    with pytest.raises(ValueError, match="last dim"):
        ResidualAdapter(16, 8)(torch.randn(1, 4, 9), 1.0)


def test_phase_gating_selects_the_right_adapter():
    pair = CouplingAdapters(384, 128)
    assert pair.is_identity()
    expected = {
        "frozen": (False, False),
        "bb_to_sc": (True, False),
        "sc_to_bb": (False, True),
        "joint": (True, True),
    }
    for phase, (bs, sb) in expected.items():
        record = pair.set_phase(phase)
        assert record["bb_to_sc"] is bs and record["sc_to_bb"] is sb
        assert all(p.requires_grad == bs for p in pair.bb_to_sc.parameters())
        assert all(p.requires_grad == sb for p in pair.sc_to_bb.parameters())
    with pytest.raises(ValueError, match="Unknown phase"):
        pair.set_phase("phase4")


def test_frozen_phase_leaves_nothing_trainable():
    pair = CouplingAdapters(64, 32)
    assert pair.set_phase("frozen")["trainable_parameters"] == 0


def test_a_disabled_direction_returns_none():
    """Ablations are a runtime switch, so the A_SB=0 control is free to run."""
    pair = CouplingAdapters(64, 32, enable_sc_to_bb=False)
    assert pair.delta_a(torch.randn(1, 5, 32), torch.tensor([1.0])) is None
    assert pair.delta_h(torch.randn(1, 5, 64), torch.tensor([1.0])) is not None
    other = CouplingAdapters(64, 32, enable_bb_to_sc=False)
    assert other.delta_h(torch.randn(1, 5, 64), torch.tensor([1.0])) is None


def test_identity_record_is_serializable():
    import json

    json.loads(json.dumps(CouplingAdapters(64, 32).identity()))
