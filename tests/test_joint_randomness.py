"""Explicit noise draws, and the one-step prediction they feed.

Two things are pinned here. That the extracted prediction record scores
identically to the loss helper it came out of -- otherwise the refactor changed
the objective. And that a named key reproduces a draw exactly, independently of
the global RNG and of which arm asked for it, because that is what makes two
training arms a paired comparison rather than two different experiments.
"""

import pytest
import torch

from pxf.joint import randomness as R
from pxf.train import losses as loss_fns
from pxf.train import step as S


@pytest.fixture(scope="module")
def model():
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.provenance import fampnn_checkpoint

    bundle = torch.load(fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False)
    net = SeqDenoiser(bundle["model_cfg"])
    net.load_state_dict(bundle["state_dict"], strict=True)
    net.eval()  # no dropout, so the global RNG has exactly one consumer
    return net


@pytest.fixture(scope="module")
def batch():
    from pxf.provenance import repo_root
    from pxf.train.data import StructureCropDataset, collate

    paths = [str(repo_root() / f"fampnn/data/casp14/pdbs/{n}.pdb") for n in ("T1031", "T1033")]
    dataset = StructureCropDataset(paths, crop_size=48, seed=0)
    return collate([dataset[0], dataset[1]])


@pytest.fixture(scope="module")
def features(model, batch):
    from pxf.couple import fampnn_iface as iface

    _logits, _h_V, feature_dict = iface.encode(
        model,
        batch["x"],
        batch["aatype"],
        seq_mask=batch["seq_mask"],
        missing_atom_mask=batch["missing_atom_mask"],
        residue_index=batch["residue_index"],
        chain_index=batch["chain_index"],
    )
    return feature_dict


def _interpolant(model):
    return model.denoiser.scn_diffusion_module.scn_interpolant


# ---- the refactor did not move the objective -------------------------------


def test_the_prediction_record_scores_exactly_as_the_loss_helper(model, batch, features):
    """``diffusion_loss`` is now a wrapper; it must still return what it did."""
    torch.manual_seed(0)
    reference, ref_stats = S.diffusion_loss(model, batch, features, multiplier=3)

    torch.manual_seed(0)
    prediction = S.sidechain_training_pass(model, batch, features, multiplier=3)
    loss, stats = loss_fns.sidechain_diffusion_loss(
        prediction.q_pred, prediction.q_target, prediction.weight, prediction.loss_mask
    )
    assert float(loss) == float(reference)
    assert int(stats["scored_atoms"]) == int(ref_stats["scored_atoms"])
    assert float(ref_stats["sigma_scn_mean"]) == pytest.approx(
        float(prediction.sigma.mean())
    )


def test_the_wrapper_is_reproducible_under_a_seed(model, batch, features):
    torch.manual_seed(7)
    first, _ = S.diffusion_loss(model, batch, features, multiplier=2)
    torch.manual_seed(7)
    second, _ = S.diffusion_loss(model, batch, features, multiplier=2)
    assert float(first) == float(second)


def test_the_clone_axis_is_block_ordered(model, batch, features):
    """``_clone`` tiles blocks, so row ``i`` is clone ``i // b``, example ``i % b``."""
    torch.manual_seed(0)
    prediction = S.sidechain_training_pass(model, batch, features, multiplier=4)
    rows = prediction.q_pred.shape[0]
    assert rows == 4 * prediction.batch_size
    assert prediction.clone_index.tolist() == [i // 2 for i in range(rows)]
    assert prediction.example_index.tolist() == [i % 2 for i in range(rows)]
    # Cloning a per-example tensor lands each row on its own example.
    marker = torch.tensor([10.0, 20.0])
    assert prediction.clone(marker).tolist() == [10.0, 20.0] * 4
    # And the target really is the same structure repeated, not four draws.
    first, second = prediction.q_target[:2], prediction.q_target[2:4]
    assert torch.equal(first, second)


# ---- the draws reproduce the donor's own -----------------------------------


def test_the_timestep_draw_matches_the_donors_own_sampler(model):
    """Term for term against ``EDM.sample_timestep``, not merely similar.

    A CPU generator seeded with ``s`` produces the same stream as the default
    generator after ``manual_seed(s)``, so the two are directly comparable.
    """
    interpolant = _interpolant(model)
    assert interpolant.training_noise_schedule == "lognormal"
    torch.manual_seed(1234)
    reference = interpolant.sample_timestep(16, device=torch.device("cpu"))
    ours = R.draw_sidechain_time(
        interpolant, 16, torch.Generator().manual_seed(1234)
    )
    assert torch.allclose(reference, ours, atol=1e-6)


def test_the_gaussian_matches_randn_like(model):
    shape = (4, 6, 33, 3)
    torch.manual_seed(99)
    reference = torch.randn(*shape)
    ours = R.draw_backbone_noise(shape, torch.Generator().manual_seed(99))
    assert torch.equal(reference, ours)


def test_applying_the_noise_reproduces_the_interpolants_forward(model):
    """``x + sigma(t) eps``, the EDM target, and ``1/c_out^2`` -- the same three."""
    interpolant = _interpolant(model)
    x1 = torch.randn(3, 5, 33, 3)
    noise = R.draw_sidechain_noise(
        interpolant, x1.shape, torch.Generator().manual_seed(3)
    )
    noised, target, t, weight = noise.apply(interpolant, x1)

    expected = x1 + noise.epsilon * interpolant.sigma(t).reshape(-1, 1, 1, 1)
    assert torch.allclose(noised, expected, atol=1e-6)
    assert torch.equal(target, x1), "EDM predicts the clean sample directly"
    assert torch.allclose(weight, interpolant.get_loss_weight(t), atol=1e-6)


def test_an_unreproducible_schedule_is_refused(model):
    interpolant = _interpolant(model)
    original = interpolant.training_noise_schedule
    interpolant.training_noise_schedule = "trunc_normal_t"
    try:
        with pytest.raises(R.UnsupportedSchedule, match="trunc_normal_t"):
            R.draw_sidechain_time(interpolant, 4, torch.Generator().manual_seed(0))
    finally:
        interpolant.training_noise_schedule = original


# ---- keys ------------------------------------------------------------------


def test_a_key_reproduces_its_draw_and_separates_occurrences():
    first = R.generator_for(0, "T1031", "sidechain_noise", occurrence=3)
    again = R.generator_for(0, "T1031", "sidechain_noise", occurrence=3)
    later = R.generator_for(0, "T1031", "sidechain_noise", occurrence=4)
    other = R.generator_for(0, "T1033", "sidechain_noise", occurrence=3)
    def draw(generator):
        return R.draw_backbone_noise((8, 3), generator)

    baseline = draw(first)
    assert torch.equal(baseline, draw(again))
    assert not torch.equal(baseline, draw(later))
    assert not torch.equal(baseline, draw(other))


def test_streams_are_independent():
    """A diagnostic that consumes one stream cannot shift another."""
    seeds = {
        stream: R.stream_seed(0, "T1031", stream, occurrence=1) for stream in R.STREAMS
    }
    assert len(set(seeds.values())) == len(R.STREAMS)


def test_an_unknown_stream_is_refused():
    with pytest.raises(ValueError, match="Unknown stream"):
        R.stream_seed(0, "T1031", "whatever")


def test_a_key_does_not_depend_on_which_arm_asked():
    """The load-bearing property: two arms at the same occurrence draw the same.

    If the key carried arm identity or loop position, B0 and B1 would see
    different backbones and their difference would no longer be the objective.
    """
    b0 = R.stream_seed(0, "T1031", "backbone_noise", occurrence=2)
    b1 = R.stream_seed(0, "T1031", "backbone_noise", occurrence=2)
    assert b0 == b1
    assert "arm" not in R.identity(base=0)["key"]


def test_explicit_noise_makes_the_step_independent_of_the_global_rng(
    model, batch, features
):
    """Two arms must be able to share a draw whatever else they each do."""
    interpolant = _interpolant(model)
    targets = S.frame_targets(model, batch)
    shape = (2 * targets["local"].shape[0], *targets["local"].shape[1:])
    noise = R.sidechain_noise_for(
        interpolant, shape, base=0, sample_id="T1031", occurrence=0
    )

    torch.manual_seed(11)
    first, _ = S.diffusion_loss(
        model, batch, features, multiplier=2, self_cond_p=0.0, noise=noise
    )
    torch.manual_seed(2024)  # a different global stream entirely
    second, _ = S.diffusion_loss(
        model, batch, features, multiplier=2, self_cond_p=0.0, noise=noise
    )
    assert float(first) == float(second)

    # ... and it really is the supplied noise doing it.
    other = R.sidechain_noise_for(
        interpolant, shape, base=0, sample_id="T1031", occurrence=1
    )
    third, _ = S.diffusion_loss(
        model, batch, features, multiplier=2, self_cond_p=0.0, noise=other
    )
    assert float(third) != float(first)


def test_pinning_the_time_twice_is_refused(model, batch, features):
    interpolant = _interpolant(model)
    targets = S.frame_targets(model, batch)
    noise = R.sidechain_noise_for(
        interpolant,
        (targets["local"].shape[0], *targets["local"].shape[1:]),
        base=0,
        sample_id="T1031",
    )
    with pytest.raises(ValueError, match="both pin the diffusion time"):
        S.diffusion_loss(model, batch, features, multiplier=1, noise=noise, t_scd=0.5)
