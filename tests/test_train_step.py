"""The training forward, against the real model and real structures.

The load-bearing checks are the ones the paper is specific about and the code
could get silently wrong: the 8-way noise cloning, teacher forcing on the true
sequence, and the confidence head's stop gradient.
"""

import pytest
import torch

from pxf.train import step as S

TARGETS = ("T1031", "T1033")
KEYS = list(S.REQUIRED_KEYS)


@pytest.fixture(scope="module")
def model():
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.provenance import fampnn_checkpoint

    bundle = torch.load(fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False)
    net = SeqDenoiser(bundle["model_cfg"])
    net.load_state_dict(bundle["state_dict"], strict=True)
    net.train()
    return net


@pytest.fixture(scope="module")
def batch():
    """Built through the real training data path, so the test exercises it too."""
    from pxf.provenance import repo_root
    from pxf.train.data import StructureCropDataset, collate

    paths = [str(repo_root() / f"fampnn/data/casp14/pdbs/{name}.pdb") for name in TARGETS]
    dataset = StructureCropDataset(paths, crop_size=64, noise=0.0, seed=0)
    return collate([dataset[0], dataset[1]])


def test_forward_produces_both_objectives(model, batch):
    torch.manual_seed(0)
    out = S.training_forward(model, batch, train_confidence=False)
    assert torch.isfinite(out.total) and out.total > 0
    assert torch.isfinite(out.mlm) and torch.isfinite(out.diffusion)
    assert out.confidence is None
    assert float(out.total) == pytest.approx(
        float(out.mlm) + float(out.diffusion), rel=1e-5
    )


def test_missing_batch_keys_are_reported(model, batch):
    partial = {k: v for k, v in batch.items() if k != "residue_index"}
    with pytest.raises(ValueError, match="missing"):
        S.training_forward(model, partial)


def test_noise_clones_match_the_configured_multiplier(model, batch):
    """Section 4.3.1: conditioning is cloned and a noise level drawn per clone."""
    expected = int(model.denoiser.scn_diffusion_module.cfg.training_batch_size_mult)
    assert expected == 8, "the released config uses 8"
    torch.manual_seed(0)
    # Unrestricted, so the atom count is a fixed property of the batch rather
    # than of the interpolant's draw; the restriction has its own tests below.
    out = S.training_forward(
        model, batch, train_confidence=False, supervise_visible_sidechains=True
    )
    assert int(out.stats["noise_clones"]) == expected
    # Atoms scored = clones x per-batch supervised atoms.
    _, atom_mask = S.sidechain_targets(model, batch)
    assert int(out.stats["scored_atoms"]) == expected * int(atom_mask.sum())


def _mar_draw(model, batch, seed=0):
    """Reproduce the interpolant draw ``training_forward`` makes at this seed.

    MAR is the first RNG consumer in ``training_forward``, so seeding and calling
    it directly lands on the same masks.
    """
    torch.manual_seed(seed)
    model.interpolant.training = bool(model.training)
    return model.interpolant.forward(batch)


def test_diffusion_loss_scores_only_the_hidden_sidechains(model, batch):
    """The objective is p(Y_M | Y_M-bar), so a visible side chain is not a target.

    ``encoder_inputs`` hands the encoder every side chain where
    ``scn_mlm_mask == 1``. Scoring those residues would ask the denoiser to
    reproduce coordinates it was just shown -- a shortcut past the actual task.
    """
    mar_out = _mar_draw(model, batch)
    visible = mar_out["scn_mlm_mask"]
    assert 0 < float((visible * batch["seq_mask"]).sum()), (
        "this batch hid every side chain, so the test cannot distinguish the masks"
    )

    _, unrestricted = S.sidechain_targets(model, batch)
    _, restricted = S.sidechain_targets(model, batch, scn_mlm_mask=visible)
    assert int(restricted.sum()) < int(unrestricted.sum())

    # No atom survives in a residue whose side chain the encoder received.
    kept_in_visible_rows = (restricted.sum(-1) * visible).sum()
    assert float(kept_in_visible_rows) == 0.0
    # Everything else is untouched: restriction only drops rows.
    hidden = 1.0 - visible
    assert torch.equal(restricted, unrestricted * hidden.unsqueeze(-1))

    clones = int(model.denoiser.scn_diffusion_module.cfg.training_batch_size_mult)
    torch.manual_seed(0)
    out = S.training_forward(model, batch, train_confidence=False)
    assert int(out.stats["scored_atoms"]) == clones * int(restricted.sum())


def test_the_encoder_input_and_the_loss_target_never_overlap(model, batch):
    """The load-bearing invariant: no side chain is both an input and a target."""
    mar_out = _mar_draw(model, batch)
    from fampnn.data import residue_constants as rc

    encoder_mask = S.encoder_inputs(model, batch, mar_out)
    given = encoder_mask[..., rc.non_bb_idxs]  # side-chain atoms the encoder saw
    _, scored = S.sidechain_targets(model, batch, scn_mlm_mask=mar_out["scn_mlm_mask"])
    assert float((given * scored).sum()) == 0.0
    # And the restriction is not vacuous -- both sets are non-empty.
    assert float(given.sum()) > 0 and float(scored.sum()) > 0


def test_supervising_visible_sidechains_is_an_explicit_opt_in(model, batch):
    """The unrestricted objective stays reachable, but only by asking for it."""
    torch.manual_seed(0)
    masked = S.training_forward(model, batch, train_confidence=False)
    torch.manual_seed(0)
    everything = S.training_forward(
        model, batch, train_confidence=False, supervise_visible_sidechains=True
    )
    assert int(everything.stats["scored_atoms"]) > int(masked.stats["scored_atoms"])
    assert "hidden_sidechain_fraction" in masked.stats
    assert "hidden_sidechain_fraction" not in everything.stats
    fraction = float(masked.stats["hidden_sidechain_fraction"])
    assert 0.0 < fraction < 1.0


def test_no_hidden_sidechains_means_nothing_is_scored(model, batch):
    """Everything visible is the degenerate case, and it must not silently train.

    ``_masked_mean`` returns a differentiable zero rather than a NaN, so the loop
    survives such a batch; what matters is that it contributes no gradient.
    """
    all_visible = batch["seq_mask"].clone()
    _, restricted = S.sidechain_targets(model, batch, scn_mlm_mask=all_visible)
    assert float(restricted.sum()) == 0.0


def test_a_packing_batch_supervises_every_atom(model, batch):
    """``scn_mlm_mask=None`` is the packing and coupling case: nothing was visible."""
    _, unrestricted = S.sidechain_targets(model, batch, scn_mlm_mask=None)
    _, default = S.sidechain_targets(model, batch)
    assert torch.equal(unrestricted, default)
    assert float(unrestricted.sum()) > 0


def test_multiplier_can_be_overridden(model, batch):
    torch.manual_seed(0)
    out = S.training_forward(model, batch, train_confidence=False, multiplier=2)
    assert int(out.stats["noise_clones"]) == 2


def test_mar_interpolant_mode_is_mirrored_onto_the_plain_class(model, batch):
    """Upstream's MAR is not an nn.Module, so model.train() cannot reach it."""
    import torch.nn as nn

    assert not isinstance(model.interpolant, nn.Module)
    model.train()
    S.training_forward(model, batch, train_confidence=False)
    assert model.interpolant.training is True
    model.eval()
    S.training_forward(model, batch, train_confidence=False)
    assert model.interpolant.training is False
    model.train()


def test_encoder_atom_mask_matches_the_inference_construction(model, batch):
    """A mask built differently here would train on inputs inference never shows."""
    from fampnn.data.data import get_rc_tensor

    from fampnn.data import residue_constants as rc

    torch.manual_seed(0)
    model.interpolant.training = True
    mar_out = model.interpolant.forward(batch)
    mask = S.encoder_inputs(model, batch, mar_out)
    expected = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, mar_out["aatype_noised"])
    expected = expected * batch["seq_mask"].unsqueeze(-1) * (1 - batch["missing_atom_mask"])
    expected[..., rc.non_bb_idxs] = expected[..., rc.non_bb_idxs] * mar_out[
        "scn_mlm_mask"
    ].unsqueeze(-1)
    assert torch.equal(mask, expected)


def test_sidechain_targets_live_in_the_local_frame(model, batch):
    """Local-frame targets must be small -- they are offsets from CA, not positions."""
    x_local, atom_mask = S.sidechain_targets(model, batch)
    from fampnn.data import residue_constants as rc

    assert x_local.shape == (*batch["aatype"].shape, len(rc.non_bb_idxs), 3)
    scored = x_local[atom_mask.bool()]
    assert scored.numel() > 0
    assert float(scored.abs().max()) < 15.0, "local coordinates should be near the origin"
    # Global coordinates are far from the origin, so this really is a transform.
    assert float(batch["x"].abs().max()) > 15.0


def test_targets_exclude_missing_and_padded_atoms(model, batch):
    _, atom_mask = S.sidechain_targets(model, batch)
    padding = batch["seq_mask"] == 0
    assert float(atom_mask[padding].sum()) == 0.0


def test_gradients_reach_encoder_and_denoiser_but_not_confidence(model, batch):
    model.zero_grad()
    torch.manual_seed(0)
    S.training_forward(model, batch, train_confidence=False).total.backward()
    named = {n: p for n, p in model.named_parameters() if p.grad is not None}
    assert any("seq_design_module" in n for n in named)
    assert any("scn_denoiser" in n for n in named)
    assert not any("confidence" in n for n in named)
    model.zero_grad()


def test_confidence_term_has_a_real_stop_gradient(model, batch):
    """Appendix D.4: the confidence loss must not affect the main model."""

    def encoder_grads(train_confidence):
        model.zero_grad()
        torch.manual_seed(1234)
        out = S.training_forward(model, batch, train_confidence=train_confidence)
        out.total.backward()
        return (
            {
                n: p.grad.clone()
                for n, p in model.named_parameters()
                if p.grad is not None and "seq_design_module" in n
            },
            out,
        )

    without, _ = encoder_grads(False)
    with_conf, out = encoder_grads(True)
    assert out.confidence is not None and torch.isfinite(out.confidence)
    assert set(without) == set(with_conf) and without
    for name in without:
        assert torch.allclose(without[name], with_conf[name], atol=1e-6), name
    model.zero_grad()


def test_confidence_training_reaches_the_confidence_head(model, batch):
    model.zero_grad()
    torch.manual_seed(0)
    S.training_forward(model, batch, train_confidence=True).total.backward()
    assert any("confidence" in n for n, p in model.named_parameters() if p.grad is not None)
    model.zero_grad()


def test_requesting_confidence_when_disabled_is_refused(model, batch):
    module = model.denoiser.scn_diffusion_module
    module.use_confidence_module = False
    try:
        with pytest.raises(ValueError, match="Confidence training requested"):
            S.training_forward(model, batch, train_confidence=True)
    finally:
        module.use_confidence_module = True


def test_stats_report_the_masking_rate_and_accuracy(model, batch):
    torch.manual_seed(0)
    out = S.training_forward(model, batch, train_confidence=False)
    scalars = out.scalars()
    assert 0.0 <= scalars["keep_fraction"] <= 1.0
    assert 0.0 <= scalars["sequence_accuracy"] <= 1.0
    assert "loss_main" in scalars, "the always-comparable total must be logged"


def test_the_confidence_head_sees_an_unrestricted_input(model, batch):
    """The restriction must reach the psCE head's *score*, never its *input*.

    The head is a network over the whole packed structure. At deployment
    ``sidechain_pack`` hides every side chain and the rollout packs every
    residue, so the input is fully populated. Zeroing the visible residues'
    coordinates before the head sees them would train it on an input
    distribution that never occurs -- and the loss curve would look fine.
    """
    mar_out = _mar_draw(model, batch)
    visible = mar_out["scn_mlm_mask"]
    assert float((visible * batch["seq_mask"]).sum()) > 0

    captured = {}

    def capture(_module, args):
        # Must return None: a forward pre-hook's return value *replaces* the args.
        captured["x"] = args[0].detach().clone()

    head = model.denoiser.scn_diffusion_module.confidence_module
    handle = head.register_forward_pre_hook(capture)
    try:
        torch.manual_seed(0)
        features = _encode(model, batch, mar_out)
        _, stats = S.confidence_loss(
            model, batch, features, batch["aatype"].long(), scn_mlm_mask=visible
        )
    finally:
        handle.remove()

    _, unrestricted = S.sidechain_targets(model, batch)
    populated = (captured["x"].abs().sum(-1) > 0).float() * unrestricted
    # Every residue the rollout packed is present in the head's input, including
    # the ones whose side chain the encoder was shown.
    in_visible_rows = (populated.sum(-1) * visible).sum()
    assert float(in_visible_rows) > 0

    # But only the hidden ones are scored.
    _, restricted = S.sidechain_targets(model, batch, scn_mlm_mask=visible)
    assert int(stats["confidence_atoms"]) == int(restricted.sum())
    assert int(stats["confidence_atoms"]) < int(unrestricted.sum())


def _encode(model, batch, mar_out):
    """The encoder feature dict, for the rollout the confidence head scores."""
    atom_mask = S.encoder_inputs(model, batch, mar_out)
    _, features = model.denoiser.seq_design_module(
        mar_out["x_noised"],
        mar_out["aatype_noised"],
        batch["seq_mask"],
        atom_mask,
        batch["residue_index"],
        batch["chain_index"],
    )
    return features
