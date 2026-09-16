"""Invariants for the unconditional sampler that do not need a GPU.

The two model-level parity checks from the plan (zero-init trajectory parity,
and backbone invariance under a phase-1 checkpoint) need the donor weights and
live in the GPU smoke path; these are the cheap ones that guard the contract
every downstream stage joins on.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "sample_uncond", REPO / "scripts" / "sample_uncond.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["sample_uncond"] = module
    spec.loader.exec_module(module)
    return module


su = _load()


def test_schedule_endpoints_are_pxdesigns_published_range():
    from pxf.couple import schedule

    sigmas = schedule.karras_sigmas(400)
    assert sigmas.numel() == 401
    assert float(sigmas[0]) == pytest.approx(2560.0, rel=1e-6)
    assert float(sigmas[-1]) == pytest.approx(0.0064, rel=1e-3)
    # Strictly descending: the sampler zips consecutive pairs.
    assert bool((sigmas[1:] < sigmas[:-1]).all())


def test_sample_id_matches_the_downstream_contract():
    assert su.sample_id(100, 0) == "L100_s0"
    assert su.sample_id(500, 42) == "L500_s42"


def test_sample_id_round_trips():
    for length in (100, 200, 300, 400, 500):
        for index in (0, 7, 99):
            name = su.sample_id(length, index)
            body = name[1:]
            got_length, got_index = body.split("_s")
            assert int(got_length) == length
            assert int(got_index) == index


def test_shards_partition_every_task_exactly_once():
    lengths, num = [100, 200, 300, 400, 500], 20
    expected = {(L, i) for L in lengths for i in range(num)}
    for num_shards in (1, 2, 3, 4, 7, 8):
        seen = []
        for shard in range(num_shards):
            seen.extend(
                su.sample_tasks(lengths, num, shard_index=shard, num_shards=num_shards)
            )
        assert len(seen) == len(expected), num_shards
        assert set(seen) == expected, num_shards


def test_every_shard_spans_all_lengths_so_a_partial_run_is_usable():
    lengths, num, num_shards = [100, 200, 300, 400, 500], 20, 5
    for shard in range(num_shards):
        got = su.sample_tasks(lengths, num, shard_index=shard, num_shards=num_shards)
        assert {L for L, _ in got} == set(lengths), shard


def test_bad_shard_arguments_are_refused():
    for bad in ((0, 0), (3, 3), (-1, 4)):
        with pytest.raises(ValueError):
            su.sample_tasks([100], 4, shard_index=bad[0], num_shards=bad[1])


def test_adapter_window_choices_are_the_documented_three():
    assert su.ADAPTER_WINDOWS == ("gated", "always", "off")


def test_denoise_net_returns_a_bare_tensor_and_ignores_forwarded_conditioning():
    """``sample_diffusion`` passes the full conditioning and wants x0 back."""

    class FakeDriver:
        def bind(self, conditioning):
            def denoise(x_noisy, sigma, *, feedback=None):
                return x_noisy * 0.5, "a_token"

            return denoise

    net = su.build_denoise_net(FakeDriver(), object())
    x = torch.ones(1, 4, 3)
    out = net(
        x_noisy=x,
        t_hat_noise_level=torch.tensor([1.0]),
        input_feature_dict={"unused": 1},
        s_inputs=None,
        s_trunk=None,
        z_trunk=None,
        pair_z=None,
        p_lm=None,
        c_l=None,
        chunk_size=None,
        inplace_safe=False,
        enable_efficient_fusion=False,
    )
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, x * 0.5)
    assert net.stats["total"] == 1
    assert net.stats["steps"] == 0  # no controller -> never engaged


def test_the_sigma_gate_engages_only_inside_the_trained_window():
    """Above sigma_max the bare driver runs; below it the cycle does."""

    class FakeDriver:
        def bind(self, conditioning):
            return lambda x_noisy, sigma, **kw: (x_noisy, None)

    class FakeCycle:
        topology = None
        aatype = None

        def forward(self, topology, x_noisy, sigma, aatype, run_feedback=None):
            class Out:
                bb0_flat = x_noisy + 1.0
                bb1_flat = None

            return Out()

    net = su.build_denoise_net(
        FakeDriver(), object(), controller=FakeCycle(), window=5.0
    )
    x = torch.zeros(1, 2, 3)
    # sigma far above the window: bare driver, unchanged
    out_hi = net(x_noisy=x, t_hat_noise_level=torch.tensor([2560.0]))
    assert torch.allclose(out_hi, x)
    # sigma inside the window: cycle engaged
    out_lo = net(x_noisy=x, t_hat_noise_level=torch.tensor([0.5]))
    assert torch.allclose(out_lo, x + 1.0)
    assert net.stats["steps"] == 1 and net.stats["total"] == 2
