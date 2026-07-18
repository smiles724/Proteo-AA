"""Behavioral spec for `cogenerate` sequence prediction (seq_mode).

complete_unmask (DEFAULT): the design region stays fully masked as input the whole
trajectory (mask_frac==1), the full sequence is re-predicted every step, the trunk
is encoded once, and the final sequence is the argmax of the last step's logits.

sequential (ABLATION, LLaDA-style): progressively commits/freezes residues
(mask_frac decreases), and re-encodes the trunk each step so committed residues
feed back (the P3 fix — otherwise the commit loop is open-loop).

Driven with the same stub model as test_cogenerate_init (no DiffusionModule/CUDA).
"""
import torch

from pxdesign_train.cogenerate import cogenerate
from pxdesign_train.sampler import build_aa20_to_restype36
from test_cogenerate_init import _FakeModel, _feat, PRED


def test_complete_unmask_stays_fully_masked_and_reads_out_last_step(monkeypatch):
    import protenix.model.protenix as pxm
    monkeypatch.setattr(pxm, "update_input_feature_dict", lambda f: f, raising=False)

    model = _FakeModel()
    torch.manual_seed(0)
    out = cogenerate(model, _feat(), N_step=3)  # default seq_mode

    assert out["trajectory"], "no steps ran"
    # Fully masked input every step — never partially revealed (predict-all).
    assert all(abs(s["mask_frac"] - 1.0) < 1e-6 for s in out["trajectory"]), (
        f"complete_unmask must keep mask_frac==1, got "
        f"{[s['mask_frac'] for s in out['trajectory']]}"
    )
    seq = out["sequence"]
    for tok, aa in PRED.items():
        assert int(seq[tok]) == aa, f"token {tok}: expected {aa}, got {int(seq[tok])}"


def test_complete_unmask_encodes_trunk_once(monkeypatch):
    """Default path re-uses the single outer trunk encode — restype never changes."""
    import protenix.model.protenix as pxm
    monkeypatch.setattr(pxm, "update_input_feature_dict", lambda f: f, raising=False)

    model = _FakeModel()
    calls = {"n": 0}
    seen = []
    real = model.get_condition_embedding

    def spy(feat, chunk_size=None):
        calls["n"] += 1
        seen.append(feat["restype"].detach().clone())
        return real(feat, chunk_size=chunk_size)

    model.get_condition_embedding = spy
    torch.manual_seed(0)
    cogenerate(model, _feat(), N_step=4)  # complete_unmask
    assert calls["n"] == 1, f"complete_unmask must encode the trunk once, got {calls['n']}"
    _, xpb = build_aa20_to_restype36()
    assert seen[0][0].argmax().item() == xpb
    assert seen[0][1].argmax().item() == xpb


def test_sequential_commits_progressively_and_reencodes_trunk(monkeypatch):
    """Ablation: mask_frac shrinks (commits) AND the trunk is re-encoded each step so
    committed residues re-enter the model (P3 fix)."""
    import protenix.model.protenix as pxm
    monkeypatch.setattr(pxm, "update_input_feature_dict", lambda f: f, raising=False)

    model = _FakeModel()
    calls = {"n": 0}
    real = model.get_condition_embedding

    def spy(feat, chunk_size=None):
        calls["n"] += 1
        return real(feat, chunk_size=chunk_size)

    model.get_condition_embedding = spy
    torch.manual_seed(0)
    N_step = 4
    out = cogenerate(model, _feat(), N_step=N_step, seq_mode="sequential")

    # Progressive commit: last step's mask_frac is below the first step's.
    fracs = [s["mask_frac"] for s in out["trajectory"]]
    assert fracs[-1] < fracs[0] or fracs[-1] == 0.0, (
        f"sequential must progressively commit, mask_frac trajectory={fracs}"
    )
    # P3: trunk re-encoded — once outside the loop + once per step>0.
    assert calls["n"] > 1, (
        f"sequential must re-encode the trunk each step (P3 fix); got {calls['n']} encodes"
    )
    seq = out["sequence"]
    for tok, aa in PRED.items():
        assert int(seq[tok]) == aa


def test_invalid_seq_mode_rejected(monkeypatch):
    import protenix.model.protenix as pxm
    monkeypatch.setattr(pxm, "update_input_feature_dict", lambda f: f, raising=False)
    model = _FakeModel()
    try:
        cogenerate(model, _feat(), N_step=2, seq_mode="bogus")
    except ValueError as e:
        assert "seq_mode" in str(e)
    else:
        raise AssertionError("invalid seq_mode should raise ValueError")
