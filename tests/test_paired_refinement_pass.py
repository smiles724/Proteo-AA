"""Stage III refinement must continue from the first backbone prediction.

`sample_diffusion_training` draws the augmentation, the sigma and the Gaussian
noise inside itself for an ordinary training call. B_post must bypass that sampling
path and receive B_pre's `x_denoised` as its coordinate input. It retains the first
pass's augmented target, uses the explicit fixed refinement sigma shared with
inference, and does not add noise to the prediction before refining it.

The expected flow is therefore `x_noisy -> B_pre -> x_denoised -> B_post`, with
side-chain feedback additionally active in B_post.
"""
import os
import sys

import pathlib
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))


class _Recorder:
    """Stands in for DiffusionModule; keeps what it was asked to denoise."""

    def __init__(self):
        self.calls = []

    def __call__(self, x_noisy, t_hat_noise_level, **_kw):
        self.calls.append((x_noisy.clone(), t_hat_noise_level.clone()))
        return x_noisy * 0.5


def _args(n_atom=7):
    from pxdesign_train.generator import TrainingNoiseSampler

    label = {
        "coordinate": torch.randn(n_atom, 3),
        "coordinate_mask": torch.ones(n_atom),
    }
    return dict(
        noise_sampler=TrainingNoiseSampler(),
        label_dict=label,
        input_feature_dict={},
        s_inputs=torch.zeros(n_atom, 1),
        z_trunk=torch.zeros(n_atom, n_atom, 1),
        N_sample=2,
    )


def test_ordinary_calls_draw_independently_without_precomputed_input():
    """Ordinary one-step training calls still sample their own training states."""
    from pxdesign_train.generator import sample_diffusion_training

    net = _Recorder()
    kw = _args()
    torch.manual_seed(0)
    aug_a, _, sigma_a, _ = sample_diffusion_training(
        denoise_net=net, s_trunk=torch.zeros(7, 1), **kw
    )
    aug_b, _, sigma_b, _ = sample_diffusion_training(
        denoise_net=net, s_trunk=torch.ones(7, 1), **kw
    )
    assert not torch.allclose(sigma_a, sigma_b), "sigma happened to repeat; reseed"
    assert not torch.allclose(aug_a, aug_b), "augmentation happened to repeat"


def test_precomputed_input_refines_the_first_pass_prediction():
    """B_post sees B_pre's output, not B_pre's original noisy coordinate input."""
    from pxdesign_train.generator import sample_diffusion_training

    net = _Recorder()
    kw = _args()
    torch.manual_seed(0)
    aug, pred, sigma, noisy = sample_diffusion_training(
        denoise_net=net, s_trunk=torch.zeros(7, 1), **kw
    )
    aug2, _, sigma2, refinement_input = sample_diffusion_training(
        denoise_net=net,
        s_trunk=torch.ones(7, 1),
        precomputed_input=(aug, sigma, pred),
        **kw,
    )

    assert torch.equal(aug, aug2)
    assert torch.equal(sigma, sigma2)
    assert torch.equal(pred, refinement_input)
    assert not torch.equal(noisy, refinement_input)
    # The denoiser API calls its coordinate argument x_noisy, but B_post receives
    # the already-denoised prediction in that slot.
    assert torch.equal(net.calls[1][0], pred)
    assert torch.equal(net.calls[0][1], net.calls[1][1]), "sigma differed"


def test_precomputed_input_does_not_consume_randomness():
    """The refinement path must not draw -- otherwise it perturbs the RNG stream
    and two runs that differ only in this flag stop being comparable."""
    from pxdesign_train.generator import sample_diffusion_training

    kw = _args()
    torch.manual_seed(0)
    aug, pred, sigma, _ = sample_diffusion_training(
        denoise_net=_Recorder(), s_trunk=torch.zeros(7, 1), **kw
    )
    state = torch.get_rng_state()
    sample_diffusion_training(
        denoise_net=_Recorder(), s_trunk=torch.ones(7, 1),
        precomputed_input=(aug, sigma, pred), **kw
    )
    assert torch.equal(state, torch.get_rng_state())


def test_precomputed_input_keeps_the_coordinate_gradient_path():
    """B_post input stays differentiable; the helper must not detach B_pre output."""
    from pxdesign_train.generator import sample_diffusion_training

    refinement_input = torch.randn(2, 7, 3, requires_grad=True)
    target = torch.randn_like(refinement_input)
    sigma = torch.ones(2)
    _, refined, _, _ = sample_diffusion_training(
        denoise_net=_Recorder(),
        s_trunk=torch.ones(7, 1),
        precomputed_input=(target, sigma, refinement_input),
        **_args(),
    )
    refined.sum().backward()
    assert refinement_input.grad is not None
    assert torch.count_nonzero(refinement_input.grad) == refinement_input.numel()


def test_stage_iii_wires_refinement_to_the_first_prediction():
    """Wiring guard: the capability is useless if the model does not pass it.

    Source-level because importing the model pulls in the whole Protenix stack;
    what is being asserted is the call site, not runtime behaviour.
    """
    model = (pathlib.Path(__file__).resolve().parents[1]
             / "pxdesign_train" / "model.py").read_text()
    start = model.index("x_gt_aug_post, x_denoised_post, sigma_post")
    block = model[start:start + 1200]
    assert "precomputed_input=(x_gt_aug, refinement_sigma, x_denoised)" in block, (
        "the refinement pass must receive B_pre's x_denoised directly instead "
        "of drawing noise or reprocessing B_pre's original x_noisy"
    )
    assert "refinement_sigma = torch.full_like" in model
    assert "sigma, self.sc_refinement_sigma" in model
    assert "s + self.refinement_pass_embedding" in model, (
        "B_post needs an explicit pass identity because x_hat_0 and x_sigma have "
        "different semantics and conditioning"
    )


def test_refinement_pass_embedding_is_zero_initialized_and_in_the_bb_group():
    """Warm starts remain a no-op, and alternating training updates it with BB."""
    model = (pathlib.Path(__file__).resolve().parents[1]
             / "pxdesign_train" / "model.py").read_text()
    trainer = (pathlib.Path(__file__).resolve().parents[1]
               / "pxdesign_train" / "runner" / "trainer.py").read_text()
    assert "self.refinement_pass_embedding = torch.nn.Parameter(" in model
    assert "torch.zeros(c_trunk)" in model
    assert 'sc if "sidechain_module" in name else bb' in trainer
