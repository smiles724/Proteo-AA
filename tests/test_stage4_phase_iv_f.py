"""IV-F: frozen sequence head, generator trained against it.

The phase exists because none of IV-0/IV-A/IV-B expresses it. IV-0 freezes
everything and the trainer rejects it (`No trainable parameters`); IV-A trains
the head and nothing else, which is the mirror image of what is wanted here;
IV-B/IV-C train the head too. The gates below are the two ways IV-F can look
like it is working while doing nothing useful.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pxdesign_train.aa import CODESIGN_BACKENDS, uses_codesign
from pxdesign_train.stage4 import GENERATOR_PREFIXES, PHASES, apply_phase, optimizer_groups


def model_with(phase, bb_prefixes=()):
    model = nn.Module()
    model.aa_head = nn.Linear(4, 4)
    model.sidechain_module = nn.Linear(4, 4)
    model.hres_injector = nn.Linear(4, 4)
    model.diffusion_module = nn.Module()
    model.diffusion_module.atom_attention_decoder = nn.Linear(4, 4)
    model.diffusion_module.other_block = nn.Linear(4, 4)
    model.configs = SimpleNamespace(stage4=SimpleNamespace(
        phase=phase, bb_trainable_prefixes=list(bb_prefixes),
        aa_lr=1e-5, sc_lr=1e-5, bb_lr=1e-6))
    return model


def trainable(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_iv_f_freezes_only_the_head():
    prefixes = ("diffusion_module.atom_attention_decoder.",)
    model = model_with("IV-F", prefixes)
    apply_phase(model)
    names = trainable(model)
    assert not any(n.startswith("aa_head.") for n in names), "IV-F must freeze the sequence head"
    assert any(n.startswith("sidechain_module.") for n in names)
    assert any(n.startswith("diffusion_module.atom_attention_decoder.") for n in names)
    # The bb subset is a subset: an unlisted diffusion block stays frozen.
    assert not any(n.startswith("diffusion_module.other_block.") for n in names)


def test_iv_f_is_the_complement_of_iv_a():
    prefixes = ("diffusion_module.atom_attention_decoder.",)
    a, f, b = (model_with(p, prefixes) for p in ("IV-A", "IV-F", "IV-B"))
    for model in (a, f, b):
        apply_phase(model)
    assert trainable(a).isdisjoint(trainable(f))
    assert trainable(a) | trainable(f) == trainable(b)


def test_iv_f_yields_a_usable_optimizer_without_an_aa_group():
    """IV-0 dies in the trainer on `No trainable parameters`; IV-F must not."""
    model = model_with("IV-F", ("diffusion_module.atom_attention_decoder.",))
    apply_phase(model)
    groups = optimizer_groups(model)
    assert groups, "IV-F must produce parameters to optimise"
    assert {g["name"] for g in groups} == {"sc", "bb"}
    torch.optim.Adam(groups, lr=1e-4)  # param-group dicts must be acceptable as-is

    frozen = model_with("IV-0")
    apply_phase(frozen)
    assert optimizer_groups(frozen) == []


def test_frozen_head_still_passes_gradient():
    """`requires_grad=False` on the head is not `no_grad`.

    IV-F's entire objective is the AA cross-entropy reaching the backbone
    THROUGH the frozen head. If freezing ever severed that route the run would
    still train -- the packer has its own losses -- and the backbone would
    simply receive no sequence signal, with nothing in the logs to say so.
    """
    model = model_with("IV-F", ("diffusion_module.atom_attention_decoder.",))
    apply_phase(model)
    coords = torch.randn(2, 4, requires_grad=True)
    hidden = model.diffusion_module.atom_attention_decoder(coords)
    logits = model.aa_head(hidden)                      # frozen module, live graph
    grad, = torch.autograd.grad(logits.square().sum(), coords)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert model.aa_head.weight.grad is None, "a frozen head must not accumulate its own grad"


def test_phase_names_are_closed():
    with pytest.raises(ValueError, match="Unknown Stage IV phase"):
        apply_phase(model_with("IV-Z"))
    assert set(PHASES) == {"IV-0", "IV-A", "IV-F", "IV-B", "IV-C"}


def test_codesign_backends_cover_both_heads():
    """The eleven equality checks became one predicate; keep it honest."""
    assert set(CODESIGN_BACKENDS) == {"fampnn", "ligandmpnn"}
    assert not uses_codesign(SimpleNamespace(aa_backend="mlp"))
    assert not uses_codesign(SimpleNamespace())
    for backend in CODESIGN_BACKENDS:
        assert uses_codesign(SimpleNamespace(aa_backend=backend))


def test_generator_prefixes_do_not_include_the_head():
    assert not any(p.startswith("aa_head") for p in GENERATOR_PREFIXES)
