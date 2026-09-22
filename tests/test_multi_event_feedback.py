"""Several feedback events in one trajectory, each with its own corrective call.

These are contract tests on the event plumbing -- how many injections fire,
where, with what seeds, and whether the extra decodes disturb the solver.
They say nothing about whether multiple feedback helps.
"""
import pytest
import torch

from pxf.bench.integrated import select_event, select_events
from pxf.couple.replay import RngStream, _event_key_set, run_trajectory


def ladder(n=41):
    return torch.linspace(16.0, 0.004, n)


# ---------------------------------------------------------------- selection

def test_one_event_per_sigma_in_trajectory_order():
    events = select_events(ladder(), [0.5, 2.0, 8.0])
    assert [e.step for e in events] == sorted(e.step for e in events)
    # ascending step is DESCENDING sigma: the trajectory starts noisy
    sigmas = [e.actual_sigma for e in events]
    assert sigmas == sorted(sigmas, reverse=True)


def test_single_sigma_matches_select_event():
    assert select_events(ladder(), [0.5])[0].key == select_event(ladder(), 0.5).key


def test_two_sigmas_on_one_step_are_refused_not_deduped():
    # Silently collapsing these would label a run "two events" while it ran
    # one, and the injection count is the variable under study.
    with pytest.raises(ValueError, match="same injection"):
        select_events(ladder(), [0.5, 0.5001])


def test_empty_request_is_refused():
    with pytest.raises(ValueError, match="no event sigmas"):
        select_events(ladder(), [])


# ------------------------------------------------------------- key parsing

def test_a_bare_pair_is_one_key_not_two():
    # (350, 0) is a single (step, substage), NOT steps 350 and 0. Reading it
    # as a collection would arm step 0 -- the first solver call of the run.
    assert _event_key_set((350, 0)) == frozenset({(350, 0)})


def test_a_list_of_pairs_is_many_keys():
    assert _event_key_set([(3, 0), (7, 0)]) == frozenset({(3, 0), (7, 0)})


def test_none_arms_nothing():
    assert _event_key_set(None) == frozenset()


def test_a_malformed_key_is_refused():
    with pytest.raises(ValueError):
        _event_key_set((1, 2, 3))


# ------------------------------------------------------------- trajectory

def _run(event, *, feedback=None, after=None, n=9, seed=0):
    schedule = ladder(n)
    seen = []

    def denoise(x, sigma, feedback=None):
        seen.append(None if feedback is None else float(feedback))
        return x * 0.5

    stream = RngStream("t", seed, device=torch.device("cpu"))
    x0, _records, stats = run_trajectory(
        denoise=denoise, schedule=schedule, n_atom=4,
        device=torch.device("cpu"), n_sample=1, stream=stream,
        event=event, feedback=feedback, after_event=after,
    )
    return x0, stats, seen


def test_n_events_give_n_injections():
    keys = [(2, 0), (4, 0), (6, 0)]
    fired = []

    def feedback(state):
        fired.append(state.key)
        return torch.tensor(1.0)

    _x0, stats, _seen = _run(keys, feedback=feedback)
    assert stats["injections"] == 3
    assert stats["events"] == 3
    assert fired == keys          # in trajectory order, not request order


def test_after_event_fires_at_every_event():
    keys = [(2, 0), (5, 0)]
    observed = []
    _x0, _stats, _seen = _run(
        keys, feedback=lambda s: None,
        after=lambda state, x: observed.append(state.key),
    )
    assert observed == keys


def test_a_control_with_no_residual_takes_the_same_calls():
    keys = [(2, 0), (4, 0), (6, 0)]
    _x0a, treated, _ = _run(keys, feedback=lambda s: torch.tensor(1.0))
    _x0b, control, _ = _run(keys, feedback=lambda s: None)
    assert treated["calls"] == control["calls"]
    assert control["injections"] == 0


def test_extra_decodes_do_not_advance_the_solver_stream():
    # The callback draws from the global RNG. If that leaked into the
    # trajectory stream, every step after event 1 would shift and the arms
    # would differ for a reason unrelated to feedback.
    keys = [(2, 0), (4, 0)]

    def greedy(state):
        torch.randn(32)        # a decode's worth of draws
        return None

    quiet, _s1, _ = _run(keys, feedback=lambda s: None)
    noisy, _s2, _ = _run(keys, feedback=greedy)
    assert torch.equal(quiet, noisy)


def test_single_event_is_unchanged_by_the_generalisation():
    one = select_event(ladder(9), 2.0).key
    a, sa, _ = _run(one, feedback=lambda s: None)
    b, sb, _ = _run([one], feedback=lambda s: None)
    assert torch.equal(a, b)
    assert sa["calls"] == sb["calls"]


def test_residual_reaches_the_denoiser_at_each_event():
    keys = [(2, 0), (5, 0)]
    _x0, _stats, seen = _run(keys, feedback=lambda s: torch.tensor(2.0))
    assert [v for v in seen if v is not None] == [2.0, 2.0]
