"""`L_MLM` in the joint objective must score the same rows `SDLoss` scores.

The sequence term is the one the joint task adds, so it is the one that can
diverge from the objective the released weights were trained by. The specific
way it diverges silently is by scoring a residue whose true identity is ``X``:
the loss still computes, the gradient still flows, and the model is taught to
emit the token the encoder uses as its own mask. `allatom_design`'s ``SDLoss``
excludes those rows (``seq_loss_mask * (1 - seq_unk_mask)``), and so does this
repository's single-task path in `pxf.train.step`.
"""
import pytest
import torch

from pxf.couple.binder_masks import BinderMaskSet
from pxf.couple.binder_residual import ChainRoles
from pxf.train import bs_seq_sc


X = 20  # restype_order_with_x["X"]; asserted against upstream below
WIDTH = 8
LENGTH = 6


class StubSeqModule(torch.nn.Module):
    """FaMPNN's seq_design_module, reduced to the head the residual reaches."""

    def __init__(self):
        super().__init__()
        self.W_out = torch.nn.Linear(WIDTH, 21)
        self.no_aatype_pred = False


class StubAdapters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project = torch.nn.Linear(WIDTH, WIDTH)

    def delta_h(self, a_token, sigma):
        return self.project(a_token)


class StubModel:
    def __init__(self):
        self.denoiser = type("D", (), {"seq_design_module": StubSeqModule()})()


def _case(unknown_rows=(), visible_rows=()):
    """Two target rows, four binder rows, all four scored by `L_MLM`."""
    binder = torch.tensor([False, False, True, True, True, True])
    roles = ChainRoles(binder=binder)
    aatype = torch.full((1, LENGTH), 5, dtype=torch.long)
    for row in unknown_rows:
        aatype[0, row] = X
    visible = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    for row in visible_rows:
        visible[0, row] = 1.0
    masks = BinderMaskSet(
        roles=roles,
        seq_mask=torch.ones(1, LENGTH),
        # 1 = identity VISIBLE; binder rows hidden, target rows kept.
        seq_mlm_mask=visible,
        sidechain_visible=torch.ones(1, LENGTH),
        aatype_encoder=aatype.clone(),
        aatype_true=aatype,
    )
    batch = {
        "x": torch.zeros(1, LENGTH, 37, 3),
        "aatype": aatype,
        "seq_mask": torch.ones(1, LENGTH),
        "missing_atom_mask": torch.zeros(1, LENGTH, 37),
        "residue_index": torch.arange(LENGTH).unsqueeze(0),
        "chain_index": torch.zeros(1, LENGTH, dtype=torch.long),
    }
    features = {"h_V": torch.randn(1, LENGTH, WIDTH)}
    return roles, masks, batch, features


def _joint(monkeypatch, masks, batch, features, roles):
    """`joint_loss` with the side-chain branch stubbed out.

    Only the sequence branch is under test; `diffusion_loss` needs the real
    denoiser and would decide nothing here.
    """
    monkeypatch.setattr(
        bs_seq_sc.train_step,
        "diffusion_loss",
        lambda *a, **k: (torch.zeros(()), {}),
    )
    return bs_seq_sc.joint_loss(
        StubModel(),
        batch,
        features,
        adapters=StubAdapters(),
        a_token=torch.randn(1, LENGTH, WIDTH),
        sigma_b=torch.tensor([1.0]),
        roles=roles,
        masks=masks,
    )


def test_x_label_is_upstreams_unknown_token():
    """The constant this test masks on is upstream's, not a local guess."""
    from fampnn.data import residue_constants as rc

    assert rc.restype_order_with_x["X"] == X


def test_unknown_identities_are_not_scored(monkeypatch):
    """A masked binder row whose true residue is X drops out of `L_MLM`."""
    torch.manual_seed(0)
    roles, masks, batch, features = _case(unknown_rows=(3,))
    out = _joint(monkeypatch, masks, batch, features, roles)
    # Four binder rows are hidden; one of them is X, so three are labels.
    assert float(out.stats["seq/masked_residues"]) == 3.0


def test_all_known_scores_every_hidden_row(monkeypatch):
    """Without an X the count is the whole hidden set, so the test above is
    measuring the exclusion and not an unrelated off-by-one."""
    torch.manual_seed(0)
    roles, masks, batch, features = _case()
    out = _joint(monkeypatch, masks, batch, features, roles)
    assert float(out.stats["seq/masked_residues"]) == 4.0


def test_a_hidden_x_costs_exactly_what_a_visible_row_costs(monkeypatch):
    """Not just the bookkeeping: the scalar being optimised is the same one.

    Hiding a row whose label is X and leaving that row visible remove it from
    the scored set by different routes, and the term is normalised by the crop
    length either way, so the two losses must be *equal*. If the unknown mask
    were dropped the first case would carry one extra cross entropy and they
    would not be. Both cases run from the same seed, so the logits and the
    stub's weights are identical and the scored set is the only difference.
    """
    torch.manual_seed(0)
    roles, masks, batch, features = _case(unknown_rows=(3,))
    hidden_x = _joint(monkeypatch, masks, batch, features, roles)
    torch.manual_seed(0)
    roles, masks, batch, features = _case(visible_rows=(3,))
    visible = _joint(monkeypatch, masks, batch, features, roles)
    assert float(hidden_x.stats["seq/masked_residues"]) == 3.0
    assert float(visible.stats["seq/masked_residues"]) == 3.0
    assert torch.isclose(hidden_x.sequence, visible.sequence)


def test_labels_and_mask_must_describe_the_same_residues(monkeypatch):
    """A batch built around different identities than the labels is refused."""
    roles, masks, batch, features = _case(unknown_rows=(3,))
    batch = dict(batch, aatype=torch.full((1, LENGTH), 7, dtype=torch.long))
    with pytest.raises(ValueError, match="not masks.aatype_true"):
        _joint(monkeypatch, masks, batch, features, roles)


def test_target_rows_are_never_labels(monkeypatch):
    """An X on a target row changes nothing: those rows are visible, not scored.

    Guards the derivation as much as the exclusion -- a `seq_unk_mask` built
    over the wrong axis, or applied before the visibility mask, would make the
    target rows move the count.
    """
    torch.manual_seed(0)
    roles, masks, batch, features = _case(unknown_rows=(0,))
    out = _joint(monkeypatch, masks, batch, features, roles)
    assert float(out.stats["seq/masked_residues"]) == 4.0
