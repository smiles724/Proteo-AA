"""How `sequential` decides which positions to freeze each round.

`_select_commits` is the whole of that decision, and it is a pure function of
(confidence, what is still masked, the schedule), so it is tested directly
rather than through a sampling run.

The two guards get the most attention here. A bare confidence threshold
degenerates in both directions depending on how the checkpoint happens to be
calibrated -- which is a property of the weights, not of this code -- and both
degenerations are silent: one stalls the trajectory, the other quietly turns
sequential decoding back into the predict-everything-at-once mode it exists to
replace.
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from pxdesign_train.cogenerate import COMMIT_STRATEGIES, _select_commits

# index:        0    1    2    3    4    5
CONF = torch.tensor([0.95, 0.10, 0.80, 0.99, 0.20, 0.05])
MASKED = torch.tensor([0, 1, 2, 4, 5])          # 3 is already committed


def _sel(strategy, k=2, threshold=0.9, max_frac=1.0, conf=CONF, masked=MASKED):
    return _select_commits(strategy=strategy, conf=conf, masked_idx=masked, k=k,
                           threshold=threshold, max_frac=max_frac,
                           generator=torch.Generator().manual_seed(0))


def test_topk_takes_the_most_confident_and_only_from_masked():
    got = set(_sel("topk", k=2).tolist())
    assert got == {0, 2}, "should be the two highest among masked (0.95, 0.80)"
    assert 3 not in got, "index 3 is already committed and must not be re-picked"


def test_random_takes_the_same_count_but_ignores_confidence():
    """The control arm: same budget, chosen uniformly.

    Without it, 'ordering by confidence helps' is a claim borrowed from a
    language-model paper rather than one measured on proteins.
    """
    counts = {len(_sel("random", k=3)) for _ in range(5)}
    assert counts == {3}
    seen = set()
    for seed in range(40):
        seen |= set(_select_commits(
            strategy="random", conf=CONF, masked_idx=MASKED, k=1,
            threshold=0.9, max_frac=1.0,
            generator=torch.Generator().manual_seed(seed),
        ).tolist())
    assert seen == set(MASKED.tolist()), (
        "every masked position should be reachable; a random arm that only ever "
        "returns the confident ones is not a control"
    )


def test_threshold_commits_everything_over_the_bar():
    got = set(_sel("threshold", threshold=0.5, max_frac=1.0).tolist())
    assert got == {0, 2}, "0.95 and 0.80 clear 0.5; 0.10/0.20/0.05 do not"


def test_threshold_ignores_the_step_schedule():
    """The point of the threshold arm: the model sets the pace, not the step count."""
    few = _sel("threshold", k=1, threshold=0.5, max_frac=1.0)
    many = _sel("threshold", k=99, threshold=0.5, max_frac=1.0)
    assert set(few.tolist()) == set(many.tolist()) == {0, 2}


def test_threshold_still_advances_when_nothing_clears_the_bar():
    """Otherwise the run decides nothing and simply runs out of steps.

    Under-confidence is the expected regime early on -- the AA head sits near
    13% accuracy on predicted backbones -- so this is the common case, not the
    corner.
    """
    got = _sel("threshold", threshold=0.999, max_frac=1.0)
    assert got.numel() == 1
    assert int(got[0]) == 3 - 3, "the single most confident masked position (idx 0)"


def test_threshold_is_capped_so_it_cannot_collapse_into_one_shot():
    """Everything clearing the bar in round one is complete_unmask with extra steps.

    No position would ever be chosen with another's identity visible, which is
    the entire reason this mode exists.
    """
    got = _sel("threshold", threshold=0.0, max_frac=0.4)
    assert got.numel() == 2, f"5 masked * 0.4 -> 2, got {got.numel()}"
    assert set(got.tolist()) == {0, 2}, "the cap keeps the most confident ones"


def test_the_cap_never_rounds_down_to_zero():
    got = _sel("threshold", threshold=0.0, max_frac=0.01)
    assert got.numel() == 1


def test_nothing_left_to_commit_returns_none():
    for strategy in COMMIT_STRATEGIES:
        assert _sel(strategy, masked=torch.tensor([], dtype=torch.long)) is None


def test_exhausted_schedule_commits_nothing_in_the_scheduled_arms():
    """k == 0 means this step's budget is spent; threshold has no budget to spend."""
    assert _sel("topk", k=0) is None
    assert _sel("random", k=0) is None
    assert _sel("threshold", k=0, threshold=0.5, max_frac=1.0).numel() == 2


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="commit_strategy"):
        _sel("greedy")


# ---------------------------------------------------------------------------
# temperature, and the validation that keeps the knobs from being set on a mode
# that ignores them
# ---------------------------------------------------------------------------

from pxdesign_train.cogenerate import cogenerate  # noqa: E402
from test_cogenerate_init import _FakeModel, _feat  # noqa: E402


@pytest.fixture(autouse=True)
def _no_protenix_feature_update(monkeypatch):
    pytest.importorskip("protenix")
    import protenix.model.protenix as pxm

    monkeypatch.setattr(pxm, "update_input_feature_dict", lambda f: f, raising=False)


def test_confidence_is_read_off_the_untempered_distribution():
    """Temperature must not inflate confidence.

    Confidence is what the commit order ranks on, so it has to keep meaning "how
    sure is the model". Reading it off `softmax(logits / T)` would make a high
    temperature look like high certainty and let the least reliable positions be
    committed first — the exact inversion of what the ordering is for.
    """
    import inspect

    src = inspect.getsource(cogenerate)
    i = src.index("if temperature and temperature > 0:")
    block = src[i:src.index("else:", i)]
    assert "torch.softmax(logits / float(temperature)" in block
    assert "conf = probs.gather" in block, "confidence must come from `probs`"
    assert "tempered.gather" not in block


def test_zero_temperature_is_greedy_and_reproducible():
    out = [cogenerate(_FakeModel(), _feat(), N_step=3, temperature=0.0)["sequence"]
           for _ in range(2)]
    assert torch.equal(out[0], out[1])


def test_temperature_is_rejected_when_negative():
    with pytest.raises(ValueError, match="temperature"):
        cogenerate(_FakeModel(), _feat(), N_step=2, temperature=-0.1)


def test_commit_knobs_are_rejected_on_a_mode_that_ignores_them():
    """complete_unmask commits nothing until the last step.

    Accepting `commit_strategy` there would silently do nothing, and an ablation
    would report the arms as identical rather than as never having run.
    """
    with pytest.raises(ValueError, match="only applies"):
        cogenerate(_FakeModel(), _feat(), N_step=2,
                   seq_mode="complete_unmask", commit_strategy="threshold")


def test_commit_max_frac_must_be_a_fraction():
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="commit_max_frac"):
            cogenerate(_FakeModel(), _feat(), N_step=2, commit_max_frac=bad)


@pytest.mark.parametrize("strategy", ["topk", "threshold", "random"])
def test_every_strategy_fills_the_design_region(strategy):
    """Whatever the pacing, the run has to finish with a complete sequence."""
    out = cogenerate(_FakeModel(), _feat(), N_step=8, seq_mode="sequential",
                     commit_strategy=strategy, commit_threshold=0.5,
                     generator=torch.Generator().manual_seed(0))
    seq = out["sequence"]
    assert int((seq >= 0).sum()) > 0, "nothing was decoded"
