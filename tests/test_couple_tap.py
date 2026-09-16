"""The PXDesign token-feature tap, including both decoder call conventions.

Protenix calls ``AtomAttentionDecoder`` positionally when it routes through
``checkpoint_fn`` and with ``a=`` as a keyword otherwise. A hook that handles
only one form silently stops injecting under the other, which does not raise --
it presents as an adapter that trains but learns nothing. Both forms are pinned
here, and so is the refusal when neither is found.

No model is needed: the tap is pure argument plumbing, so a bare module suffices
and these run everywhere.
"""

import pytest
import torch
from torch import nn

from pxf.couple.pxdesign_iface import (
    DECODER_A_ARG,
    DECODER_A_KWARG,
    BackboneTap,
    token_feature_dim,
)

WIDTH = 16


class FakeDecoder(nn.Module):
    def forward(self, atom_to_token_idx=None, a=None, **kwargs):
        return a


class FakeDiffusionModule(nn.Module):
    """Just enough surface for BackboneTap: a layernorm and a decoder."""

    def __init__(self, width=WIDTH):
        super().__init__()
        self.c_token = width
        self.layernorm_a = nn.LayerNorm(width)
        self.atom_attention_decoder = FakeDecoder()


@pytest.fixture
def module():
    return FakeDiffusionModule()


def test_keyword_call_is_injected(module):
    """The form Protenix uses with activation checkpointing off -- the live path."""
    tap = BackboneTap(module)
    with tap:
        tap.feedback = torch.full((1, 4, WIDTH), 0.5)
        args, kwargs = tap._inject(
            module.atom_attention_decoder, (), {DECODER_A_KWARG: torch.zeros(1, 4, WIDTH)}
        )
        assert args == ()
        assert float(kwargs[DECODER_A_KWARG].abs().mean()) == pytest.approx(0.5)
    assert tap.injections == 1


def test_positional_call_is_injected(module):
    """The form checkpoint_fn uses. Defensive today, live if checkpointing returns."""
    tap = BackboneTap(module)
    with tap:
        tap.feedback = torch.full((1, 4, WIDTH), 0.5)
        args, kwargs = tap._inject(
            module.atom_attention_decoder, (None, torch.zeros(1, 4, WIDTH), None), {}
        )
        assert kwargs == {}
        assert float(args[DECODER_A_ARG].abs().mean()) == pytest.approx(0.5)
    assert tap.injections == 1


def test_neither_form_is_refused_loudly(module):
    """Silently not injecting is the failure mode this guards against."""
    tap = BackboneTap(module)
    with tap:
        tap.feedback = torch.zeros(1, 4, WIDTH)
        with pytest.raises(ValueError, match="neither index"):
            tap._inject(module.atom_attention_decoder, (), {"q_skip": None})


def test_no_feedback_leaves_the_call_untouched(module):
    tap = BackboneTap(module)
    with tap:
        assert tap._inject(module.atom_attention_decoder, (), {"a": torch.zeros(1)}) is None
        assert tap.injections == 0


def test_a_width_mismatch_is_refused(module):
    tap = BackboneTap(module)
    with tap:
        tap.feedback = torch.zeros(1, 4, WIDTH + 1)
        with pytest.raises(ValueError, match="Feedback width"):
            tap._inject(module.atom_attention_decoder, (), {"a": torch.zeros(1, 4, WIDTH)})


def test_capture_records_the_layernorm_output(module):
    tap = BackboneTap(module)
    with tap:
        out = module.layernorm_a(torch.randn(1, 4, WIDTH))
        assert tap.a_token is not None
        assert torch.equal(tap.a_token, out)
        assert tap.calls == 1


def test_hooks_are_removed_on_exit(module):
    before = len(module.layernorm_a._forward_hooks)
    with BackboneTap(module):
        assert len(module.layernorm_a._forward_hooks) == before + 1
    assert len(module.layernorm_a._forward_hooks) == before


def test_hooks_are_removed_even_when_the_body_raises(module):
    before = len(module.layernorm_a._forward_hooks)
    with pytest.raises(RuntimeError):
        with BackboneTap(module):
            raise RuntimeError("boom")
    assert len(module.layernorm_a._forward_hooks) == before, (
        "a leaked hook would perturb every later call, including evaluation"
    )


def test_double_install_is_refused(module):
    tap = BackboneTap(module)
    with tap:
        with pytest.raises(RuntimeError, match="already installed"):
            tap.install()


def test_reset_clears_capture_and_counters(module):
    tap = BackboneTap(module)
    with tap:
        module.layernorm_a(torch.randn(1, 4, WIDTH))
        tap.feedback = torch.zeros(1, 4, WIDTH)
        tap.reset()
        assert tap.a_token is None and tap.feedback is None
        assert tap.calls == 0 and tap.injections == 0


def test_token_width_is_discoverable(module):
    assert token_feature_dim(module) == WIDTH
