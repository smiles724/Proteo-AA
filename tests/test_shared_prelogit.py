"""The shared pre-logit path must condition both heads, once, on binder rows only.

This mode exists because the legacy packing hook fires *after* the sequence
logits of its pass, so the same-pass sequence loss has no gradient into the
adapter. The tests that matter here are therefore about the gradient reaching
W_out's input, the residual not compounding over a ~101-step decode, and the
two application modes being mutually exclusive.
"""
import pytest
import torch

from pxf.couple.binder_residual import ChainRoles
from pxf.couple.shared_prelogit import (
    APPLICATION_MODES,
    assert_not_accumulating,
    check_exclusive,
    condition,
    conditioned_forward,
    sequence_head,
)


class StubSeqModule(torch.nn.Module):
    """Stands in for FaMPNN's seq_design_module: a pretrained W_out over h_V."""

    def __init__(self, width=8, n_aatype=21, no_aatype_pred=False):
        super().__init__()
        self.W_out = torch.nn.Linear(width, n_aatype)
        self.no_aatype_pred = no_aatype_pred


class StubAdapters(torch.nn.Module):
    """A trainable stand-in for CouplingAdapters.

    It has real parameters on purpose. An earlier version was a plain object
    computing a function of a_token, and the gradient test passed vacuously --
    with a_token detached there was nothing in the graph requiring grad, so
    `.backward()` raised rather than proving anything. The claim under test is
    that the ADAPTER'S PARAMETERS receive a gradient from the sequence loss,
    which needs parameters to exist.
    """

    def __init__(self, in_width=5, width=8):
        super().__init__()
        self.project = torch.nn.Linear(in_width, width)

    def delta_h(self, a_token, sigma):
        return self.project(a_token)


def _features(batch=1, length=6, width=8):
    return {"h_V": torch.randn(batch, length, width), "other": torch.ones(3)}


# ------------------------------------------------------------------- the head


def test_a_model_with_no_sequence_head_is_refused():
    """no_aatype_pred=True returns logits=None upstream; conditioning it is a no-op."""
    with pytest.raises(ValueError, match="no sequence head"):
        sequence_head(StubSeqModule(no_aatype_pred=True))


def test_a_module_without_w_out_is_refused_rather_than_invented():
    class Bare:
        no_aatype_pred = False

    with pytest.raises(AttributeError, match="will not invent one"):
        sequence_head(Bare())


# -------------------------------------------------------------- conditioning


def test_logits_are_recomputed_from_the_conditioned_h_v():
    module = StubSeqModule()
    features = _features()
    delta = torch.full_like(features["h_V"], 0.25)
    logits, conditioned = condition(module, features, delta)
    expected = module.W_out(features["h_V"] + delta)
    assert torch.allclose(logits, expected, atol=1e-6)
    assert torch.allclose(conditioned["h_V"], features["h_V"] + delta, atol=1e-6)


def test_a_none_residual_returns_the_original_dict_untouched():
    """The uncoupled arm takes the SAME path, not an equivalent one."""
    module = StubSeqModule()
    features = _features()
    logits, conditioned = condition(module, features, None)
    assert logits is None
    assert conditioned is features


def test_the_input_feature_dict_is_never_mutated():
    module = StubSeqModule()
    features = _features()
    before = features["h_V"].clone()
    condition(module, features, torch.full_like(features["h_V"], 0.3))
    assert torch.equal(features["h_V"], before)
    assert_not_accumulating(features, before)


def test_repeated_calls_do_not_compound_the_residual():
    """A decode calls the encoder ~101 times with one constant residual.

    If h' were written back into the caller's dict, the residual would grow
    linearly with step count -- and the symptom would be a coupled arm that
    drifts further from the uncoupled one the longer the decode runs, which
    looks exactly like a real effect.
    """
    module = StubSeqModule()
    features = _features()
    delta = torch.full_like(features["h_V"], 0.1)
    first = condition(module, features, delta)[1]["h_V"].clone()
    for _ in range(100):
        condition(module, features, delta)
    last = condition(module, features, delta)[1]["h_V"]
    assert torch.allclose(first, last, atol=1e-6)


def test_dimension_mismatches_are_refused():
    module = StubSeqModule()
    features = _features(length=6, width=8)
    with pytest.raises(ValueError, match="width"):
        condition(module, features, torch.zeros(1, 6, 4))
    with pytest.raises(ValueError, match="length"):
        condition(module, features, torch.zeros(1, 9, 8))


# ------------------------------------------------------------- the gradient


def test_the_sequence_loss_reaches_the_adapter():
    """The whole point of this mode. Under packing_only this gradient is zero.

    `conditioned_forward` must leave autograd enabled through the frozen
    W_out and back into the residual, so a masked-sequence loss can train
    A_BS in the same pass.
    """
    module = StubSeqModule()
    for p in module.parameters():
        p.requires_grad_(False)

    roles = ChainRoles.from_lengths(3, 3)
    a_token = torch.randn(1, 6, 5, requires_grad=True)
    adapters = StubAdapters()
    features = _features(length=6)

    logits, conditioned, delta = conditioned_forward(
        module, features, adapters=adapters, a_token=a_token,
        sigma_b=torch.tensor([0.429]), roles=roles,
    )
    loss = logits[:, roles.binder].square().mean()
    loss.backward()

    assert delta.grad_fn is not None, "the residual was detached from the graph"
    grads = [p.grad for p in adapters.parameters()]
    assert all(g is not None for g in grads), "no gradient reached the adapter"
    assert any(float(g.abs().sum()) > 0 for g in grads), (
        "the adapter's gradient is identically zero: the sequence loss is not "
        "reaching it, which is the failure this mode exists to fix"
    )
    assert all(p.grad is None for p in module.parameters()), (
        "the frozen sequence head received a parameter gradient"
    )


def test_a_token_is_detached_by_default_but_can_be_kept():
    """PXDesign is frozen, so the default must not send gradients into it."""
    module = StubSeqModule()
    roles = ChainRoles.from_lengths(3, 3)
    features = _features(length=6)

    a_token = torch.randn(1, 6, 5, requires_grad=True)
    logits, _, _ = conditioned_forward(
        module, features, adapters=StubAdapters(), a_token=a_token,
        sigma_b=torch.tensor([0.429]), roles=roles,
    )
    logits.square().mean().backward()
    assert a_token.grad is None

    a_token2 = torch.randn(1, 6, 5, requires_grad=True)
    logits2, _, _ = conditioned_forward(
        module, features, adapters=StubAdapters(), a_token=a_token2,
        sigma_b=torch.tensor([0.429]), roles=roles, detach_a_token=False,
    )
    logits2.square().mean().backward()
    assert a_token2.grad is not None


def test_target_rows_are_unconditioned_so_their_logits_are_unchanged():
    module = StubSeqModule()
    roles = ChainRoles.from_lengths(4, 2)
    features = _features(length=6)
    baseline = module.W_out(features["h_V"])

    logits, _, _ = conditioned_forward(
        module, features, adapters=StubAdapters(), a_token=torch.randn(1, 6, 5),
        sigma_b=torch.tensor([0.429]), roles=roles,
    )
    assert torch.allclose(logits[:, roles.target], baseline[:, roles.target], atol=1e-6)
    assert not torch.allclose(logits[:, roles.binder], baseline[:, roles.binder], atol=1e-4)


# ----------------------------------------------------------- mode exclusivity


def test_the_two_application_modes_cannot_both_be_active():
    check_exclusive("shared_prelogit", packing_hook_active=False)
    check_exclusive("packing_only", packing_hook_active=True)
    with pytest.raises(AssertionError, match="added twice"):
        check_exclusive("shared_prelogit", packing_hook_active=True)


def test_an_unknown_application_mode_is_refused():
    with pytest.raises(ValueError, match="unknown application_mode"):
        check_exclusive("both", packing_hook_active=False)
    assert APPLICATION_MODES == ("packing_only", "shared_prelogit")
