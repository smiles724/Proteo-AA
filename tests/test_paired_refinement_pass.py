"""The Stage III refinement pass must denoise the SAME draw as the first pass.

`sample_diffusion_training` draws the augmentation, the sigma and the Gaussian
noise INSIDE itself. Stage III calls it twice -- once for B_pre and once for the
side-chain-informed B_post -- so the second call re-randomised all three. The two
passes therefore denoised different rotations of the structure at different noise
levels, which breaks the refinement pass in two separate ways:

  * `h'_res` is not rotation-invariant (it descends from `q = c_l + W r_noisy`,
    a linear map of the NOISY GLOBAL coordinates). Computed under rotation A and
    injected into a forward that lives under rotation B, its orientation-carrying
    components are noise from the second pass's point of view, and the fusion's
    best move is to learn to ignore them. That silently caps how much the
    feedback channel can carry.

  * `mse` (first pass) and `bb_post` (second pass) land on different sigmas, so
    their difference is dominated by which noise level each happened to draw
    rather than by whether the side-chain feedback helped. The one number Stage
    III exists to produce is unreadable.

Reusing the draw fixes both, and turns the pair into a controlled comparison:
same structure, same rotation, same sigma, same noise -- the only difference is
whether the backbone saw the side chain.

Inference does not have this problem: it carries `h_res_prime_inject` across
steps of ONE sampling trajectory, where the frame never changes.
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


def test_the_two_passes_are_independent_draws_without_reuse():
    """Characterisation: this is the behaviour the reuse path exists to replace."""
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


def test_reuse_gives_the_second_pass_the_identical_draw():
    """Same structure, same rotation, same sigma, same noise -- only s_trunk differs."""
    from pxdesign_train.generator import sample_diffusion_training

    net = _Recorder()
    kw = _args()
    torch.manual_seed(0)
    aug, _, sigma, noisy = sample_diffusion_training(
        denoise_net=net, s_trunk=torch.zeros(7, 1), **kw
    )
    aug2, _, sigma2, noisy2 = sample_diffusion_training(
        denoise_net=net, s_trunk=torch.ones(7, 1), reuse_draw=(aug, sigma, noisy), **kw
    )

    assert torch.equal(aug, aug2)
    assert torch.equal(sigma, sigma2)
    assert torch.equal(noisy, noisy2)
    # And the network really was handed the same tensors, not just told about them.
    assert torch.equal(net.calls[0][0], net.calls[1][0]), "x_noisy differed"
    assert torch.equal(net.calls[0][1], net.calls[1][1]), "sigma differed"


def test_reuse_does_not_consume_randomness():
    """The reuse path must not draw at all -- otherwise it perturbs the RNG stream
    and two runs that differ only in this flag stop being comparable."""
    from pxdesign_train.generator import sample_diffusion_training

    kw = _args()
    torch.manual_seed(0)
    aug, _, sigma, noisy = sample_diffusion_training(
        denoise_net=_Recorder(), s_trunk=torch.zeros(7, 1), **kw
    )
    state = torch.get_rng_state()
    sample_diffusion_training(
        denoise_net=_Recorder(), s_trunk=torch.ones(7, 1),
        reuse_draw=(aug, sigma, noisy), **kw
    )
    assert torch.equal(state, torch.get_rng_state())


def test_stage_iii_wires_the_refinement_pass_to_the_first_draw():
    """Wiring guard: the capability is useless if the model does not pass it.

    Source-level because importing the model pulls in the whole Protenix stack;
    what is being asserted is the call site, not runtime behaviour.
    """
    model = (pathlib.Path(__file__).resolve().parents[1]
             / "pxdesign_train" / "model.py").read_text()
    start = model.index("x_gt_aug_post, x_denoised_post, sigma_post")
    block = model[start:model.index(")", model.index("sample_diffusion_training(", start))]
    assert "reuse_draw=" in block, (
        "the refinement pass still re-randomises: h'_res would be computed under "
        "one rotation and consumed under another, and bb_post would land on a "
        "different sigma than mse"
    )
