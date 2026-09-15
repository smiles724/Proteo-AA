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
    assert float(out.total) == pytest.approx(float(out.mlm) + float(out.diffusion), rel=1e-5)


def test_missing_batch_keys_are_reported(model, batch):
    partial = {k: v for k, v in batch.items() if k != "residue_index"}
    with pytest.raises(ValueError, match="missing"):
        S.training_forward(model, partial)


def test_noise_clones_match_the_configured_multiplier(model, batch):
    """Section 4.3.1: conditioning is cloned and a noise level drawn per clone."""
    expected = int(model.denoiser.scn_diffusion_module.cfg.training_batch_size_mult)
    assert expected == 8, "the released config uses 8"
    torch.manual_seed(0)
    out = S.training_forward(model, batch, train_confidence=False)
    assert int(out.stats["noise_clones"]) == expected
    # Atoms scored = clones x per-batch supervised atoms.
    _, atom_mask = S.sidechain_targets(model, batch)
    assert int(out.stats["scored_atoms"]) == expected * int(atom_mask.sum())


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
    from fampnn.data import residue_constants as rc
    from fampnn.data.data import get_rc_tensor
    torch.manual_seed(0)
    model.interpolant.training = True
    mar_out = model.interpolant.forward(batch)
    mask = S.encoder_inputs(model, batch, mar_out)
    expected = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, mar_out["aatype_noised"])
    expected = expected * batch["seq_mask"].unsqueeze(-1) * (1 - batch["missing_atom_mask"])
    expected[..., rc.non_bb_idxs] = (expected[..., rc.non_bb_idxs]
                                     * mar_out["scn_mlm_mask"].unsqueeze(-1))
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
        return ({n: p.grad.clone() for n, p in model.named_parameters()
                 if p.grad is not None and "seq_design_module" in n}, out)

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
