"""The residual hook must fire on every step, then leave FaMPNN untouched."""

import pytest
import torch

from pxf.couple.pack_hook import residual_on_sidechain_diffusion


class FakeScn:
    """Stands in for scn_diffusion_module; records the h_V it was handed."""

    def __init__(self):
        self.seen = []

    def sidechain_diffusion(self, feature_dict, *args, **kwargs):
        self.seen.append(feature_dict["h_V"].clone())
        return feature_dict["h_V"], {"aux": True}


class FakeDenoiser:
    def __init__(self):
        self.scn_diffusion_module = FakeScn()


class FakeModel:
    def __init__(self):
        self.denoiser = FakeDenoiser()


def _feats(b=1, l=4, c=8, value=1.0):
    return {"h_V": torch.full((b, l, c), value)}


def test_residual_is_added_on_every_call():
    model = FakeModel()
    delta = torch.full((1, 4, 8), 0.25)
    with residual_on_sidechain_diffusion(model, delta) as stats:
        for _ in range(5):  # a MAR decode runs many steps
            model.denoiser.scn_diffusion_module.sidechain_diffusion(_feats())
    assert stats["calls"] == 5 and stats["applied"] == 5
    for seen in model.denoiser.scn_diffusion_module.seen:
        assert torch.allclose(seen, torch.full((1, 4, 8), 1.25))


def test_none_is_a_no_op_but_still_traverses_the_wrapper():
    """Arm 2 must take the same code path as arm 3, minus the residual."""
    model = FakeModel()
    with residual_on_sidechain_diffusion(model, None) as stats:
        model.denoiser.scn_diffusion_module.sidechain_diffusion(_feats())
    assert stats["calls"] == 1 and stats["applied"] == 0
    assert torch.allclose(
        model.denoiser.scn_diffusion_module.seen[0], torch.full((1, 4, 8), 1.0)
    )


def test_the_module_is_restored_on_exit():
    model = FakeModel()
    module = model.denoiser.scn_diffusion_module
    before = module.sidechain_diffusion
    with residual_on_sidechain_diffusion(model, torch.zeros(1, 4, 8)):
        assert module.sidechain_diffusion is not before
    assert "sidechain_diffusion" not in vars(module)
    assert module.sidechain_diffusion.__func__ is before.__func__


def test_restored_even_if_the_block_raises():
    model = FakeModel()
    module = model.denoiser.scn_diffusion_module
    with pytest.raises(RuntimeError):
        with residual_on_sidechain_diffusion(model, torch.zeros(1, 4, 8)):
            raise RuntimeError("decode blew up")
    assert "sidechain_diffusion" not in vars(module)


def test_width_mismatch_is_refused_rather_than_broadcast():
    model = FakeModel()
    with residual_on_sidechain_diffusion(model, torch.zeros(1, 4, 16)):
        with pytest.raises(ValueError, match="does not match h_V"):
            model.denoiser.scn_diffusion_module.sidechain_diffusion(_feats(c=8))


def test_length_mismatch_is_refused():
    model = FakeModel()
    with residual_on_sidechain_diffusion(model, torch.zeros(1, 9, 8)):
        with pytest.raises(ValueError, match="different structure"):
            model.denoiser.scn_diffusion_module.sidechain_diffusion(_feats(l=4))


def test_batch_of_one_is_expanded_to_the_feature_batch():
    model = FakeModel()
    with residual_on_sidechain_diffusion(model, torch.full((1, 4, 8), 0.5)):
        model.denoiser.scn_diffusion_module.sidechain_diffusion(_feats(b=3))
    assert model.denoiser.scn_diffusion_module.seen[0].shape == (3, 4, 8)


def test_the_original_feature_dict_is_not_mutated():
    """A mutated caller dict would leak the residual into the uncoupled arm."""
    model = FakeModel()
    feats = _feats()
    original = feats["h_V"].clone()
    with residual_on_sidechain_diffusion(model, torch.full((1, 4, 8), 1.0)):
        model.denoiser.scn_diffusion_module.sidechain_diffusion(feats)
    assert torch.allclose(feats["h_V"], original)
