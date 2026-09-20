"""The training forward, against the real model and real structures.

The load-bearing checks are the ones the original training code is specific
about and a reimplementation gets silently wrong: which side chains are targets,
what a ghost slot is supervised to, the 8-way noise cloning, teacher forcing on
the true sequence, and the confidence head's stop gradient.
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
    dataset = StructureCropDataset(paths, crop_size=64, seed=0)
    return collate([dataset[0], dataset[1]])


def _mar_draw(model, batch, seed=0):
    """Reproduce the interpolant draw ``training_forward`` makes at this seed.

    MAR is the first RNG consumer in ``training_forward``, so seeding and calling
    it directly lands on the same masks.
    """
    torch.manual_seed(seed)
    model.interpolant.training = bool(model.training)
    return model.interpolant.forward(batch)


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


# ---- the masks the original loss reads off its dataset ---------------------


def test_batch_masks_reproduce_the_datasets_own(model, batch):
    """``atom_mask``, ``x_mask`` and ``seq_unk_mask``, derived rather than carried.

    ``process_single_pdb`` computes all three; a pxf batch carries a smaller set,
    so :func:`pxf.train.step.batch_masks` rebuilds them. Compare against the
    upstream featurizer's own output for the same structure.
    """
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    from pxf.provenance import repo_root

    example = process_single_pdb(
        load_feats_from_pdb(str(repo_root() / f"fampnn/data/casp14/pdbs/{TARGETS[0]}.pdb"))
    )
    length = int(example["seq_mask"].sum())
    single = {k: v[None, :length] for k, v in example.items() if torch.is_tensor(v)}
    masks = S.batch_masks(single)
    assert torch.equal(masks["atom_mask"], single["atom_mask"])
    assert torch.equal(masks["x_mask"], single["x_mask"][..., 0])
    assert torch.equal(masks["seq_unk_mask"], single["seq_unk_mask"])


def test_padding_is_masked_out_even_though_its_aatype_reads_as_alanine(model):
    """Padding pads ``aatype`` with 0, which is a real residue type.

    Deriving ``atom_mask`` from the residue type alone would therefore mark five
    atoms present at every padded position, and the loss would score them.
    """
    from pxf.provenance import repo_root
    from pxf.train.data import StructureCropDataset, collate

    # T1031 is 95 residues, so a 128-residue crop is a third padding.
    path = str(repo_root() / "fampnn/data/casp14/pdbs/T1031.pdb")
    padded = collate([StructureCropDataset([path], crop_size=128, seed=0)[0]])
    masks = S.batch_masks(padded)
    padding = padded["seq_mask"] == 0
    assert bool(padding.any())
    assert float(masks["atom_mask"][padding].sum()) == 0.0
    assert float(masks["x_mask"][padding].sum()) == 0.0
    _, loss_mask = S.sidechain_targets(model, padded)
    assert float(loss_mask[padding].sum()) == 0.0


# ---- which side chains are targets -----------------------------------------


def test_every_resolved_sidechain_is_a_target_including_the_visible_ones(model, batch):
    """The diffusion mask is ``x_mask * frames_exist``; ``scn_mlm_mask`` is absent.

    The denoiser starts from pure noise in the local frame whatever the encoder
    was shown, so a residue whose side chain was visible as *context* is still a
    real prediction. Restricting the target set to the hidden residues -- the
    natural reading of a masked-modelling objective -- trains against a different
    loss from the one the released weights were fitted with.
    """
    mar_out = _mar_draw(model, batch)
    visible = mar_out["scn_mlm_mask"]
    assert float((visible * batch["seq_mask"]).sum()) > 0, (
        "this batch hid every side chain, so the test cannot distinguish the masks"
    )

    _, mask = S.sidechain_targets(model, batch)
    scored_in_visible_rows = (mask.sum(-1) * visible).sum()
    assert float(scored_in_visible_rows) > 0

    # And the restriction is reachable, as an ablation.
    _, restricted = S.sidechain_targets(model, batch, scn_mlm_mask=visible)
    assert int(restricted.sum()) < int(mask.sum())
    assert float((restricted.sum(-1) * visible).sum()) == 0.0


def test_ghost_slots_are_supervised_to_the_origin(model, batch):
    """``x_mask`` drops missing atoms but keeps slots the residue type lacks.

    The MLP always emits 33 atoms; this is what teaches it to put the
    nonexistent ones at 0. Excluding them -- the natural reading of "supervise
    the atoms that exist" -- leaves those outputs untrained.
    """
    from fampnn.data.data import get_rc_tensor

    from fampnn.data import residue_constants as rc

    targets = S.frame_targets(model, batch)
    exists = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, batch["aatype"].long())
    ghost = (1 - exists)[..., rc.non_bb_idxs] * batch["seq_mask"].unsqueeze(-1)
    ghost = ghost * targets["frames_exist"].unsqueeze(-1)
    assert float(ghost.sum()) > 0

    scored_ghosts = ghost * targets["loss_mask"]
    assert float(scored_ghosts.sum()) == pytest.approx(float(ghost.sum()))
    assert float(targets["local"][ghost.bool()].abs().max()) == 0.0
    # The confidence head is scored on the narrower set: a ghost slot has no
    # error to be confident about.
    assert float((ghost * targets["atom_mask"]).sum()) == 0.0


def test_targets_exclude_missing_atoms_and_padding(model, batch):
    _, mask = S.sidechain_targets(model, batch)
    from fampnn.data import residue_constants as rc

    padding = batch["seq_mask"] == 0
    assert float(mask[padding].sum()) == 0.0
    missing = batch["missing_atom_mask"][..., rc.non_bb_idxs].bool()
    assert float(mask[missing].sum()) == 0.0


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


# ---- the diffusion step ----------------------------------------------------


def test_noise_clones_match_the_configured_multiplier(model, batch):
    """The conditioning is cloned and a noise level drawn per clone."""
    expected = int(model.denoiser.scn_diffusion_module.cfg.training_batch_size_mult)
    assert expected == 8, "the released config uses 8"
    torch.manual_seed(0)
    out = S.training_forward(model, batch, train_confidence=False)
    assert int(out.stats["noise_clones"]) == expected
    _, atom_mask = S.sidechain_targets(model, batch)
    assert int(out.stats["scored_atoms"]) == expected * int(atom_mask.sum())


def test_multiplier_can_be_overridden(model, batch):
    torch.manual_seed(0)
    out = S.training_forward(model, batch, train_confidence=False, multiplier=2)
    assert int(out.stats["noise_clones"]) == 2


def test_a_pinned_diffusion_time_fixes_the_noise_level(model, batch):
    """``t_scd`` is how the original evaluates a curve at fixed noise levels."""
    features = _encode(model, batch, _mar_draw(model, batch))
    interpolant = model.denoiser.scn_diffusion_module.scn_interpolant
    for t in (0.2, 0.9):
        _, stats = S.diffusion_loss(model, batch, features, multiplier=2, t_scd=t)
        expected = float(interpolant.sigma(torch.tensor([t])))
        assert float(stats["sigma_scn_mean"]) == pytest.approx(expected, rel=1e-5)
    # Higher t is less noise, and the loss should reflect that ordering.
    _, easy = S.diffusion_loss(model, batch, features, multiplier=4, t_scd=0.95)
    _, hard = S.diffusion_loss(model, batch, features, multiplier=4, t_scd=0.2)
    assert float(easy["sidechain_mse_local"]) < float(hard["sidechain_mse_local"])


def test_the_ablation_switch_restricts_the_target_set(model, batch):
    """``hidden_sidechains_only`` is reachable, but it is not the objective."""
    torch.manual_seed(0)
    original = S.training_forward(model, batch, train_confidence=False)
    torch.manual_seed(0)
    ablated = S.training_forward(
        model, batch, train_confidence=False, hidden_sidechains_only=True
    )
    assert int(ablated.stats["scored_atoms"]) < int(original.stats["scored_atoms"])
    assert "hidden_sidechain_fraction" in ablated.stats
    assert "hidden_sidechain_fraction" not in original.stats


def test_encoder_atom_mask_matches_the_inference_construction(model, batch):
    """A mask built differently here would train on inputs inference never shows."""
    from fampnn.data.data import get_rc_tensor

    from fampnn.data import residue_constants as rc

    mar_out = _mar_draw(model, batch)
    mask = S.encoder_inputs(model, batch, mar_out)
    expected = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, mar_out["aatype_noised"])
    expected = expected * batch["seq_mask"].unsqueeze(-1) * (1 - batch["missing_atom_mask"])
    expected[..., rc.non_bb_idxs] = expected[..., rc.non_bb_idxs] * mar_out[
        "scn_mlm_mask"
    ].unsqueeze(-1)
    assert torch.equal(mask, expected)


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


def test_sequence_and_sidechains_are_masked_separately(model, batch):
    """``drop_sidechains`` hides side chains at positions whose identity is kept.

    That is the regime packing runs in -- identity known, conformation unknown --
    and it only happens because MAR is driven in train mode.
    """
    mar_out = _mar_draw(model, batch)
    seq_keep, scn_keep = mar_out["seq_mlm_mask"], mar_out["scn_mlm_mask"]
    assert float((scn_keep * (1 - seq_keep)).sum()) == 0.0, "scn kept must imply seq kept"
    assert float(scn_keep.sum()) < float(seq_keep.sum()), "some kept identities lost theirs"
    torch.manual_seed(0)
    out = S.training_forward(model, batch, train_confidence=False)
    assert float(out.stats["sidechain_keep_fraction"]) < float(out.stats["keep_fraction"])


# ---- gradients -------------------------------------------------------------


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
    """The confidence loss must not affect the main model."""

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


# ---- the confidence rollout ------------------------------------------------


def test_the_rollout_stays_in_the_local_frame_and_packs_well(model, batch):
    """``mini_rollout`` is the shipped sampler, minus the trip through global.

    The head scores local coordinates, and the training batch's backbone is not
    the one the encoder saw once augment_eps is on, so a round trip would not
    close. That it reproduces FaMPNN's published packing accuracy (~1.1 A on
    CASP backbones) is what says the reconstruction is the right integrator.
    """
    mar_out = _mar_draw(model, batch)
    features = _encode(model, batch, mar_out)
    torch.manual_seed(0)
    _, stats = S.confidence_loss(
        model, batch, features, batch["aatype"].long(), scn_mlm_mask=mar_out["scn_mlm_mask"]
    )
    assert 0.3 < float(stats["rollout_scn_rmsd"]) < 2.5


def test_the_rollout_restores_the_modules_training_mode(model, batch):
    """Upstream ends with an unconditional ``self.train()``; that leaks into eval."""
    module = model.denoiser.scn_diffusion_module
    features = _encode(model, batch, _mar_draw(model, batch))
    for mode in (True, False):
        module.train(mode)
        S.mini_rollout(module, features["h_V"], batch["aatype"].long(), batch["seq_mask"])
        assert module.training is mode
    module.train(True)


def test_the_confidence_head_scores_only_hidden_sidechains_on_a_full_input(model, batch):
    """The restriction reaches the psCE head's *score*, never its *input*.

    A head trained to call a side chain it was handed "zero error" would be
    calibrated for a case that never arises: ``sidechain_pack`` hides every side
    chain. But the head is a network over the whole packed structure, so its
    input has to stay fully populated, as it is at deployment.
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

    targets = S.frame_targets(model, batch)
    populated = (captured["x"].abs().sum(-1) > 0).float() * targets["atom_mask"]
    in_visible_rows = (populated.sum(-1) * visible).sum()
    assert float(in_visible_rows) > 0, "the head must see the residues it will not score"

    hidden = (1.0 - visible) * batch["seq_mask"] * targets["frames_exist"]
    expected = (hidden.unsqueeze(-1) * targets["atom_mask"]).sum()
    assert int(stats["confidence_atoms"]) == int(expected)
    assert int(stats["confidence_atoms"]) < int(targets["atom_mask"].sum())


def test_stats_report_the_masking_rate_and_accuracy(model, batch):
    torch.manual_seed(0)
    out = S.training_forward(model, batch, train_confidence=False)
    scalars = out.scalars()
    assert 0.0 <= scalars["keep_fraction"] <= 1.0
    assert 0.0 <= scalars["sequence_accuracy"] <= 1.0
    assert "loss_main" in scalars, "the always-comparable total must be logged"


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
