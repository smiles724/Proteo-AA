"""The sigma_B distribution the coupling adapters train against.

Both adapters are conditioned on ``log sigma_B``. Training at a single value
leaves that conditioning constant, so the adapter is only licensed at that one
noise level -- and nothing about the code would look wrong. These tests pin the
distribution and, at the end, pin the property the distribution exists for: that
the adapter's output is genuinely a function of sigma_B.
"""

import pytest
import torch

from pxf.couple import schedule as S
from pxf.couple.adapters import CouplingAdapters, ResidualAdapter


def test_the_trajectory_matches_pxdesigns_published_scheduler():
    """A transcription check against InferenceNoiseScheduler's own formula."""
    sigmas = S.karras_sigmas(400)
    assert sigmas.numel() == 401
    assert float(sigmas[0]) == pytest.approx(S.PXDESIGN_SIGMA_DATA * S.PXDESIGN_S_MAX)
    assert float(sigmas[-1]) == pytest.approx(S.PXDESIGN_SIGMA_DATA * S.PXDESIGN_S_MIN)
    # Strictly descending, which every window calculation assumes.
    assert bool((sigmas[1:] < sigmas[:-1]).all())


def test_the_default_window_is_the_late_end_of_the_trajectory():
    sched = S.CouplingNoiseSchedule()
    first, last = sched.window_steps()
    assert (first, last) == (281, 395)
    # Late means late: the window starts past three quarters of the way through.
    assert first / sched.n_step > 0.7
    assert sched.identity()["distinct_values"] == 115


def test_trajectory_draws_are_values_the_sampler_actually_visits():
    """The point of trajectory mode: no interpolated noise levels."""
    sched = S.CouplingNoiseSchedule(mode="trajectory")
    on_schedule = set(round(float(x), 5) for x in sched.trajectory())
    draws = sched.sample(200, generator=torch.Generator().manual_seed(0))
    assert all(round(float(x), 5) in on_schedule for x in draws)
    assert bool(((draws >= sched.sigma_min) & (draws <= sched.sigma_max)).all())


def test_trajectory_draws_actually_span_the_window():
    """A distribution that collapsed to one value is the bug being fixed."""
    sched = S.CouplingNoiseSchedule()
    draws = sched.sample(500, generator=torch.Generator().manual_seed(0))
    assert len(set(float(x) for x in draws)) > 50
    # Both ends get real mass, not just the middle.
    assert float(draws.min()) < 0.1
    assert float(draws.max()) > 2.0


def test_loguniform_covers_between_step_values():
    sched = S.CouplingNoiseSchedule(mode="loguniform", sigma_min=0.01, sigma_max=5.0)
    draws = sched.sample(500, generator=torch.Generator().manual_seed(0))
    assert bool(((draws >= 0.01) & (draws <= 5.0)).all())
    on_schedule = set(round(float(x), 5) for x in sched.trajectory())
    # Continuous, so essentially nothing lands exactly on a scheduled step.
    assert sum(round(float(x), 5) in on_schedule for x in draws) < 5


def test_fixed_mode_is_available_but_has_to_be_asked_for():
    sched = S.CouplingNoiseSchedule(mode="fixed", sigma=1.0)
    draws = sched.sample(8)
    assert bool((draws == 1.0).all())
    record = sched.identity()
    assert record["distinct_values"] == 1
    assert "constant" in sched.describe()


def test_a_window_too_narrow_to_learn_from_is_refused():
    """Silently degenerate is the failure mode; an error is the fix."""
    with pytest.raises(ValueError, match="effectively constant"):
        S.CouplingNoiseSchedule(sigma_min=0.999, sigma_max=1.001)


def test_an_inverted_or_nonpositive_window_is_refused():
    with pytest.raises(ValueError, match="sigma_min < sigma_max"):
        S.CouplingNoiseSchedule(sigma_min=5.0, sigma_max=0.01)
    with pytest.raises(ValueError, match="sigma_min < sigma_max"):
        S.CouplingNoiseSchedule(sigma_min=0.0, sigma_max=5.0)
    with pytest.raises(ValueError, match="Unknown sigma mode"):
        S.CouplingNoiseSchedule(mode="gaussian")
    with pytest.raises(ValueError, match="must be positive"):
        S.CouplingNoiseSchedule(mode="fixed", sigma=0.0)


def test_sampling_is_reproducible_from_the_generator():
    sched = S.CouplingNoiseSchedule()
    a = sched.sample(16, generator=torch.Generator().manual_seed(7))
    b = sched.sample(16, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_from_config_lets_flags_override_but_never_shadow_with_none():
    spec = {"mode": "loguniform", "sigma_min": 0.02, "sigma_max": 4.0}
    unchanged = S.from_config(spec, mode=None, sigma_min=None)
    assert (unchanged.mode, unchanged.sigma_min) == ("loguniform", 0.02)
    overridden = S.from_config(spec, sigma_max=2.0)
    assert overridden.sigma_max == 2.0 and overridden.mode == "loguniform"
    with pytest.raises(ValueError, match="Unknown sigma schedule keys"):
        S.from_config({"sigma_mean": 1.0})


def test_identity_is_json_serialisable():
    import json

    for mode in S.MODES:
        record = S.CouplingNoiseSchedule(mode=mode).identity()
        assert json.loads(json.dumps(record))["mode"] == mode


def test_the_donors_sigma_data_changes_which_values_are_drawn():
    """The schedule is only on-trajectory if sigma_data matches the donor."""
    a = S.CouplingNoiseSchedule(sigma_data=16.0).trajectory()
    b = S.CouplingNoiseSchedule(sigma_data=1.0).trajectory()
    assert not torch.allclose(a, b)
    assert float(a[0]) == pytest.approx(16.0 * float(b[0]))


# ---- what the distribution is for -----------------------------------------


def test_the_adapter_response_depends_on_sigma():
    """Without this, conditioning on sigma_B would be decoration.

    Zero-init makes the *output* zero at step 0, so the adapter is given a
    non-zero output projection first; the check is that two noise levels produce
    two different residuals from identical features.
    """
    adapter = ResidualAdapter(8, 6, d_hidden=16, d_noise=8)
    torch.nn.init.normal_(adapter.project_out.weight, std=0.5)
    features = torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(0))
    low = adapter(features, torch.tensor([0.02]))
    high = adapter(features, torch.tensor([4.0]))
    assert not torch.allclose(low, high, atol=1e-6)


def test_a_fixed_sigma_gives_the_noise_embedding_one_input():
    """Stated as a test so the regression is visible if the default ever reverts."""
    sched = S.CouplingNoiseSchedule(mode="fixed", sigma=1.0)
    embedding = CouplingAdapters(8, 6, d_hidden=16, d_noise=8).bb_to_sc.noise
    draws = sched.sample(32)
    embedded = embedding(draws)
    assert embedded.unique(dim=0).shape[0] == 1  # exactly one distinct input

    varied = embedding(S.CouplingNoiseSchedule().sample(32, generator=torch.Generator()))
    assert varied.unique(dim=0).shape[0] > 1


def test_sampled_sigmas_reach_the_adapter_as_distinct_conditioning():
    """End to end: draw sigma_B, condition A_BS on it, get distinct residuals."""
    adapters = CouplingAdapters(12, 10, d_hidden=16, d_noise=8)
    torch.nn.init.normal_(adapters.bb_to_sc.project_out.weight, std=0.5)
    a_token = torch.randn(1, 4, 12, generator=torch.Generator().manual_seed(1))
    sched = S.CouplingNoiseSchedule()
    generator = torch.Generator().manual_seed(3)
    deltas = [
        adapters.delta_h(a_token, sched.sample(1, generator=generator)) for _ in range(6)
    ]
    norms = {round(float(d.norm()), 6) for d in deltas}
    assert len(norms) == len(deltas), "sigma_B did not change the residual"


def test_a_stale_sigma_flag_is_refused_rather_than_ignored():
    """``--sigma`` predates this module; accepting it silently would mislead."""
    with pytest.raises(ValueError, match="never reads it"):
        S.from_config({"mode": "trajectory"}, sigma=5.0)
    with pytest.raises(ValueError, match="never reads it"):
        S.from_config(None, sigma=5.0)  # the default mode is not fixed
    # Fixed mode reads it, and drops the window it cannot use.
    sched = S.from_config({"mode": "fixed", "sigma_min": 0.01}, sigma=5.0)
    assert sched.sigma == 5.0 and sched.mode == "fixed"
