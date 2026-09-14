"""Tests for the template-centred sampled-sigma arm (`sidechain.template_sigma`).

This arm sits between the two that already exist, and the properties worth
pinning are the ones that distinguish it from them:

  fixed 0.3      S_phi(template + 0.3*eps,     const) -> GT
  template_sigma S_phi(template + sigma*eps,   sigma) -> GT     <- here
  edm            S_phi(GT       + sigma*eps,   sigma) -> GT

Against the fixed arm the difference is that sigma varies per row and reaches
the time channel; against EDM it is that the corruption stays centred on the
TEMPLATE. Both are invisible in tensor shapes -- a checkpoint from any arm loads
cleanly into any other -- so the guard registration is tested too.
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from pxdesign_train.sidechain.init import (
    DEFAULT_SIGMA_T,
    _broadcast_sigma,
    template_init_local,
)
from pxdesign_train.sidechain.instantiate import STD_AA_3


# ---------------------------------------------------------------- broadcasting

def test_float_sigma_passes_through_unchanged():
    """The historical scalar path must stay byte-identical, not merely similar."""
    like = torch.zeros(4, 3, 10, 3)
    assert _broadcast_sigma(0.3, like) == 0.3


def test_per_row_sigma_reshapes_to_broadcast_over_atoms_and_coords():
    like = torch.zeros(4, 3, 10, 3)
    out = _broadcast_sigma(torch.tensor([0.1, 0.2, 0.3, 0.4]), like)
    # One sigma per ROW, then singleton everywhere else, so every atom and every
    # coordinate of a row shares that row's draw.
    assert out.shape == (4, 1, 1, 1)
    assert torch.allclose(out.reshape(-1), torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_single_element_tensor_is_treated_as_a_scalar():
    like = torch.zeros(4, 3, 10, 3)
    out = _broadcast_sigma(torch.tensor([0.25]), like)
    assert out.shape == ()
    assert float(out) == pytest.approx(0.25)


def test_wrong_row_count_raises_rather_than_broadcasting_creatively():
    """Silently pairing the wrong sigma with a row corrupts both the corruption
    scale and the time embedding that is supposed to report it."""
    like = torch.zeros(4, 3, 10, 3)
    with pytest.raises(ValueError, match="one sigma per row"):
        _broadcast_sigma(torch.tensor([0.1, 0.2, 0.3]), like)


# ------------------------------------------------------- template_init_local

def _types_and_mask(rows: int, tokens: int):
    # TRP: the largest side chain, so the atom mask is non-trivial.
    tix = torch.full((rows, tokens), STD_AA_3.index("TRP"), dtype=torch.long)
    from pxdesign_train.sidechain.instantiate import sidechain_mask

    m = sidechain_mask([["TRP"] * tokens for _ in range(rows)][0])
    mask = m.unsqueeze(0).expand(rows, tokens, m.shape[-1]).clone()
    return tix, mask


def test_per_row_sigma_scales_each_row_independently():
    """The point of the arm: row 0 sees a near-clean template, row 3 a washed-out
    one, from a single call. With one fixed sigma every row is corrupted alike."""
    rows, tokens = 4, 5
    tix, mask = _types_and_mask(rows, tokens)

    g = torch.Generator().manual_seed(0)
    clean = template_init_local(tix, mask, sigma_T=0.0, generator=g)

    sigmas = torch.tensor([0.01, 0.1, 1.0, 3.0])
    g = torch.Generator().manual_seed(0)
    noisy = template_init_local(tix, mask, sigma_T=sigmas, generator=g)

    valid = mask.bool()
    devs = [
        float((noisy[r][valid[r]] - clean[r][valid[r]]).abs().mean())
        for r in range(rows)
    ]
    # Monotone in sigma, and spanning two orders of magnitude -- not four draws
    # that happen to differ by noise.
    assert devs == sorted(devs), devs
    assert devs[-1] > 20 * devs[0], devs


def test_scalar_and_equivalent_per_row_tensor_agree_exactly():
    """A per-row tensor of a constant must reproduce the scalar path, so turning
    the arm on at its default sigma changes the objective and nothing else."""
    rows, tokens = 3, 4
    tix, mask = _types_and_mask(rows, tokens)

    g = torch.Generator().manual_seed(7)
    a = template_init_local(tix, mask, sigma_T=DEFAULT_SIGMA_T, generator=g)
    g = torch.Generator().manual_seed(7)
    b = template_init_local(
        tix, mask, sigma_T=torch.full((rows,), DEFAULT_SIGMA_T), generator=g
    )
    assert torch.equal(a, b)


def test_invalid_atoms_stay_zero_whatever_the_sigma():
    """Masked slots carry no template, so a large draw must not leak noise into
    them -- they are read downstream as 'this residue has no such atom'."""
    rows, tokens = 2, 3
    tix, mask = _types_and_mask(rows, tokens)
    mask[:, :, -2:] = False

    out = template_init_local(
        tix, mask, sigma_T=torch.tensor([3.0, 3.0]),
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.count_nonzero(out[:, :, -2:]) == 0


# ------------------------------------------------------------------- plumbing

def test_config_exposes_the_arm_and_its_range():
    from pxdesign_train.configs.configs_train import training_configs

    sc = training_configs["sidechain"]
    assert sc["template_sigma"] is False, "must stay opt-in"
    for key in (
        "template_sigma_p_mean", "template_sigma_p_std",
        "template_sigma_min", "template_sigma_max", "template_sigma_infer",
    ):
        assert key in sc, key
    assert sc["template_sigma_min"] < sc["template_sigma_max"]
    # The default deployment point reproduces the historical fixed input, which
    # is what makes "same input, live time channel" the first comparison to run.
    assert sc["template_sigma_infer"] == pytest.approx(DEFAULT_SIGMA_T)


def test_switch_is_registered_with_the_checkpoint_guard():
    """It adds no parameters and changes no shape, so without this a checkpoint
    from this arm warm-starts into a fixed-sigma run and quietly means something
    else -- the same hazard `edm` is registered for."""
    from pxdesign_train.runner.trainer import PXDesignTrainer

    assert "template_sigma" in PXDesignTrainer.SIDECHAIN_LAYOUT_KEYS
    assert "template_sigma" in PXDesignTrainer.SIDECHAIN_ARCH_KEYS


def test_numeric_range_is_recorded_like_the_edm_range():
    """Booleans cannot describe a range: two runs differing only in sigma_max
    record identically, and an evaluator rebuilding from defaults would score a
    checkpoint on a range it never saw."""
    from pxdesign_train.runner.trainer import PXDesignTrainer

    for key in (
        "template_sigma_p_mean", "template_sigma_p_std",
        "template_sigma_min", "template_sigma_max", "template_sigma_infer",
    ):
        assert key in PXDesignTrainer.TEMPLATE_SIGMA_HPARAM_KEYS, key


# ------------------------------------------------------------------ the guards

def _cfg(**sidechain):
    import copy

    from protenix.config.config import parse_configs
    from pxdesign_train.configs.configs_train import training_configs

    t = copy.deepcopy(training_configs)
    t["enable_sidechain"] = True
    t["sidechain"].update(sidechain)
    cfg = parse_configs(t, arg_str="")
    cfg.load_strict = False
    return cfg


def test_template_sigma_and_edm_are_mutually_exclusive():
    """They are two objectives, not two knobs: enabling both would perturb the
    template and then throw it away in favour of a noised target."""
    from pxdesign_train.model import ProtenixDesignTrain

    with pytest.raises(ValueError, match="two\\s+different objectives"):
        ProtenixDesignTrain(_cfg(template_sigma=True, edm=True))


def test_template_sigma_requires_a_template_to_centre_on():
    from pxdesign_train.model import ProtenixDesignTrain

    with pytest.raises(ValueError, match="template_init=True"):
        ProtenixDesignTrain(_cfg(template_sigma=True, template_init=False))


def test_enabling_the_arm_builds_a_sampler_over_the_configured_range():
    """The draw has to exist and respect the clamp; a sampler that silently fell
    back to a constant would reproduce the fixed-sigma arm under a new name."""
    from pxdesign_train.model import ProtenixDesignTrain

    m = ProtenixDesignTrain(
        _cfg(template_sigma=True, template_sigma_min=0.2, template_sigma_max=1.5)
    )
    assert m.sc_template_sigma is True
    s = m.sc_template_noise_sampler((4096,), device=torch.device("cpu"),
                                    dtype=torch.float32)
    assert float(s.min()) >= 0.2 - 1e-6
    assert float(s.max()) <= 1.5 + 1e-6
    # Actually varying, not a constant dressed up as a draw.
    assert float(s.std()) > 1e-3
