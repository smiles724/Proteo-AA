"""The BB->SC residual must reach the binder and nothing else.

The target's identity and side chains are held fixed for the whole decode, so a
residual on a target row cannot change what that row is. It can still change
the messages that row sends through the encoder, which turns the arm from
"A_BS informs the binder's packing" into "A_BS perturbs the whole complex".
Both could help; only one is the hypothesis, so the mask is load-bearing and is
tested rather than trusted.

The failures worth catching are the quiet ones. A ``[L]`` mask and a ``[B]``
mask both broadcast cleanly against ``[B, L, C]`` when ``B == L``; a transposed
mask produces a finite, plausible, wrong residual; and a residual that is
arithmetically applied but three orders of magnitude below ``h_V`` is inert in
a way that looks exactly like "coupling does not help".
"""
import pytest
import torch

from pxf.couple.binder_residual import (
    ChainRoles,
    binder_masked_residual,
    describe_residual,
)
from pxf.couple.bs_policy import Gate


class StubAdapters:
    """Returns a per-row constant so masking is visible by inspection."""

    def __init__(self, width=8, value=2.0):
        self.width, self.value = width, value

    def delta_h(self, a_token, sigma):
        batch, length = a_token.shape[0], a_token.shape[-2]
        return torch.full((batch, length, self.width), self.value)


def _a_token(length, width=16):
    return torch.ones(1, length, width)


# ------------------------------------------------------------------- roles


def test_roles_from_lengths_puts_the_binder_last():
    roles = ChainRoles.from_lengths(n_target=5, n_binder=3)
    assert roles.length == 8
    assert roles.n_target == 5 and roles.n_binder == 3
    assert roles.binder.tolist() == [False] * 5 + [True] * 3
    assert roles.is_contiguous_tail()


def test_roles_from_chain_index_need_not_be_a_tail():
    """A multi-chain target can interleave; downstream [-n:] slicing would break."""
    chain_index = torch.tensor([0, 1, 0, 1, 0])
    roles = ChainRoles.from_chain_index(chain_index, binder_chain=1)
    assert roles.n_binder == 2
    assert not roles.is_contiguous_tail()
    assert roles.identity()["binder_is_contiguous_tail"] is False


def test_fixed_sequence_mask_is_the_complement_of_the_binder():
    roles = ChainRoles.from_lengths(4, 2)
    assert roles.fixed_sequence_mask().tolist() == [1, 1, 1, 1, 0, 0]


@pytest.mark.parametrize(
    "mask, message",
    [
        (torch.zeros(6, dtype=torch.bool), "selects no rows"),
        (torch.ones(6, dtype=torch.bool), "selects every row"),
    ],
)
def test_degenerate_masks_are_refused(mask, message):
    with pytest.raises(ValueError, match=message):
        ChainRoles(binder=mask)


def test_a_non_bool_mask_is_refused():
    """An int mask would still index, and would select row 1 instead of row 0."""
    with pytest.raises(TypeError, match="must be bool"):
        ChainRoles(binder=torch.tensor([0, 1, 1]))


# ---------------------------------------------------------------- masking


def test_target_rows_are_exactly_zero_and_binder_rows_are_not():
    roles = ChainRoles.from_lengths(4, 3)
    delta = binder_masked_residual(
        StubAdapters(), "matched", roles=roles,
        a_token=_a_token(7), sigma=torch.tensor([0.4]),
    )
    assert delta.shape == (1, 7, 8)
    assert float(delta[:, :4, :].abs().sum()) == 0.0
    assert float(delta[:, 4:, :].abs().min()) > 0.0


def test_a_residual_sized_for_another_structure_is_refused():
    roles = ChainRoles.from_lengths(4, 3)
    with pytest.raises(ValueError, match="different structure"):
        binder_masked_residual(
            StubAdapters(), "matched", roles=roles,
            a_token=_a_token(9), sigma=torch.tensor([0.4]),
        )


def test_bypass_returns_none_rather_than_zeros():
    """None keeps the uncoupled arm on the *same* path, not an equivalent one."""
    roles = ChainRoles.from_lengths(4, 3)
    assert binder_masked_residual(
        StubAdapters(), "none", roles=roles,
        a_token=_a_token(7), sigma=torch.tensor([0.4]),
    ) is None


def test_a_gate_closed_to_zero_is_bypass_not_a_zero_tensor():
    roles = ChainRoles.from_lengths(4, 3)
    gate = Gate("A", 0.010, 0.429)
    assert gate(0.005) == 0.0
    assert binder_masked_residual(
        StubAdapters(), "matched", roles=roles, a_token=_a_token(7),
        sigma=torch.tensor([0.005]), gate=gate,
    ) is None


def test_the_gate_scales_the_binder_rows():
    roles = ChainRoles.from_lengths(2, 2)
    gate = Gate("A", 0.010, 0.429)
    full = binder_masked_residual(
        StubAdapters(), "matched", roles=roles, a_token=_a_token(4),
        sigma=torch.tensor([0.429]),
    )
    part = binder_masked_residual(
        StubAdapters(), "matched", roles=roles, a_token=_a_token(4),
        sigma=torch.tensor([0.1]), gate=gate,
    )
    scale = gate(0.1)
    assert 0.0 < scale < 1.0
    assert torch.allclose(part, full * scale, atol=1e-6)


def test_an_identically_zero_residual_is_refused_not_reported_as_null():
    """A silently inert coupled arm is a duplicate of the uncoupled arm."""
    roles = ChainRoles.from_lengths(3, 3)
    with pytest.raises(AssertionError, match="identically zero"):
        binder_masked_residual(
            StubAdapters(value=0.0), "matched", roles=roles,
            a_token=_a_token(6), sigma=torch.tensor([0.4]),
        )


def test_a_non_finite_residual_is_refused():
    roles = ChainRoles.from_lengths(3, 3)
    # NaN * 0 is NaN, so this must be caught before masking or the run aborts
    # blaming the mask for a numerically broken adapter.
    with pytest.raises(AssertionError, match="non-finite values before masking"):
        binder_masked_residual(
            StubAdapters(value=float("nan")), "matched", roles=roles,
            a_token=_a_token(6), sigma=torch.tensor([0.4]),
        )


def test_masking_survives_a_square_shape_where_broadcasting_would_hide_a_bug():
    """B == L is where a wrong-axis mask broadcasts without complaining."""
    roles = ChainRoles.from_lengths(4, 4)
    delta = binder_masked_residual(
        StubAdapters(width=8), "matched", roles=roles,
        a_token=_a_token(8), sigma=torch.tensor([0.4]),
    )
    assert float(delta[:, :4, :].abs().sum()) == 0.0
    assert delta.shape == (1, 8, 8)


# -------------------------------------------------------------- reporting


def test_describe_reports_the_relative_norm_that_explains_an_inert_arm():
    roles = ChainRoles.from_lengths(4, 4)
    delta = binder_masked_residual(
        StubAdapters(width=8, value=0.001), "matched", roles=roles,
        a_token=_a_token(8), sigma=torch.tensor([0.4]),
    )
    h_v = torch.full((1, 8, 8), 10.0)
    report = describe_residual(delta, roles, h_v=h_v, gate_value=1.0)
    assert report["applied"] is True
    assert report["target_row_norm_max"] == 0.0
    assert report["n_rows_touched"] == 4
    # 0.001 against 10.0 per element: present in arithmetic, inert in effect.
    assert report["relative_norm"] == pytest.approx(1e-4, rel=1e-3)


def test_describe_handles_the_bypass_arm():
    roles = ChainRoles.from_lengths(2, 2)
    report = describe_residual(None, roles, gate_value=0.0)
    assert report["applied"] is False
    assert report["n_binder"] == 2
