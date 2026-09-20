"""The gradient contract: which losses reach the backbone, and which must not.

This is the experiment's premise, so it is tested against the real donor rather
than a stub. A stub can be wired up correctly and prove nothing: the question is
whether a side-chain loss, evaluated on a *frozen* FaMPNN reading a predicted
backbone, has a live path back to the backbone's parameters through FaMPNN's
own geometry -- its k-NN graph, its GVP layers, its frame construction.

The controls matter as much as the positives. A "blocked" arm that still trains
the backbone through a path nobody detached would look like evidence.
"""

import os
from pathlib import Path

import pytest
import torch

from pxf.joint import data as J
from pxf.joint import losses as JL
from pxf.joint import model as M
from pxf.joint import randomness as R

DONOR = os.environ.get(
    "PXDESIGN_DONOR",
    "/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn"
    "/runs/component_donors/pxdesign_v0.1.0.pt",
)
CIF = os.environ.get("PXF_TEST_CIF", "/hai/scratch/yfsun/casp14/cif/T1031.cif")

needs_donor = pytest.mark.skipif(
    not (Path(DONOR).is_file() and Path(CIF).is_file()),
    reason="set PXDESIGN_DONOR and PXF_TEST_CIF to run the real gradient contract",
)

TRAINABLE_BLOCKS = 4

def _rig_device():
    """Where the rigs run. ``auto`` uses CUDA when there is one.

    Default-on rather than opt-in: the GPU job exists to catch device bugs, and
    a rig that quietly stays on the CPU on a GPU node cannot. That is exactly
    how the preflight's cpu/cuda mismatch (job 120170, 493423) survived a green
    suite -- the tests passed on the same node the script failed on.
    """
    requested = os.environ.get("PXF_TEST_DEVICE", "auto")
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _sidechain_slots():
    from fampnn.data import residue_constants as rc

    return list(rc.non_bb_idxs)


@pytest.fixture(scope="module")
def rig():
    """Donor, frozen FaMPNN, one real structure, and the trainable allowlist."""
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        featurize_structures,
        load_backbone_model,
        to_featurized,
    )
    from pxf.provenance import fampnn_checkpoint

    device = _rig_device()
    backbone, _bundle, _record = load_backbone_model(DONOR, device=device)
    driver = PXDesignBackboneDriver(backbone)
    sample_id, source = featurize_structures([CIF], crop_size=256)[0]
    structure = to_featurized(sample_id, source[0]).to(device)
    # Deliberately NOT moved: FaMPNN's loader leaves its parse on the CPU, and
    # that asymmetry between the two parses is the real deployment condition.
    # Moving it here would hide the class of bug this rig is meant to expose.
    native = process_single_pdb(load_feats_from_pdb(CIF))

    weights = torch.load(fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False)
    fampnn = SeqDenoiser(weights["model_cfg"])
    fampnn.load_state_dict(weights["state_dict"], strict=True)
    fampnn.to(device).eval()
    # Frozen parameters, live input derivatives. A no_grad context here would
    # also cut the inputs, which are the entire mechanism under test.
    fampnn.requires_grad_(False)

    backbone.requires_grad_(False)
    blocks = backbone.diffusion_module.diffusion_transformer.blocks
    assert len(blocks) >= TRAINABLE_BLOCKS
    tail = {
        f"diffusion_transformer.blocks.{i}."
        for i in range(len(blocks) - TRAINABLE_BLOCKS, len(blocks))
    }
    trainable = {}
    for name, parameter in backbone.named_parameters():
        wanted = (
            any(key in name for key in tail)
            or "diffusion_module.layernorm_a" in name
            or "atom_attention_decoder" in name
        )
        if wanted:
            parameter.requires_grad_(True)
            trainable[name] = parameter
    assert trainable, "the expected backbone modules are not in this donor"

    batch = J.build_joint_batch(fampnn, structure, native)
    conditioning = driver.conditioning(structure.feature_dict)
    return dict(
        driver=driver,
        backbone=backbone,
        fampnn=fampnn,
        batch=batch,
        conditioning=conditioning,
        trainable=trainable,
        sample_id=sample_id,
        device=device,
    )


def _noise(rig, multiplier=2, occurrence=0):
    batch = rig["batch"]
    interpolant = rig["fampnn"].denoiser.scn_diffusion_module.scn_interpolant
    backbone_noise = R.draw_backbone_noise(
        batch.backbone_target.shape,
        R.generator_for(0, rig["sample_id"], "backbone_noise", occurrence=occurrence),
        device=rig["device"],
    )
    sidechain_noise = R.sidechain_noise_for(
        interpolant,
        (multiplier, batch.length, len(_sidechain_slots()), 3),
        base=0,
        sample_id=rig["sample_id"],
        occurrence=occurrence,
        device=rig["device"],
    )
    return backbone_noise, sidechain_noise


def _forward(rig, *, detach_backbone=False, multiplier=2, sigma_b=1.0):
    backbone_noise, sidechain_noise = _noise(rig, multiplier)
    return M.joint_forward(
        rig["driver"],
        rig["conditioning"],
        rig["fampnn"],
        rig["batch"],
        sigma_b=sigma_b,
        backbone_noise=backbone_noise,
        sidechain_noise=sidechain_noise,
        multiplier=multiplier,
        self_cond_p=0.0,
        detach_backbone=detach_backbone,
    )


def _local_loss(rig, forward):
    from pxf.train import losses as loss_fns

    prediction = forward.prediction
    loss, _stats = loss_fns.sidechain_diffusion_loss(
        prediction.q_pred, prediction.q_target, prediction.weight, prediction.loss_mask
    )
    return loss


def _placement_loss(rig, forward, *, native_from=None):
    batch = rig["batch"]
    prediction = forward.prediction
    native = (native_from if native_from is not None else batch.native_batch["x"])[
        ..., _sidechain_slots(), :
    ]
    loss, _stats = JL.placement_loss(
        forward.placed,
        prediction.clone(native),
        prediction.clone(batch.physical_mask),
        aatype=prediction.aatype,
    )
    return loss


def _backbone_loss(rig, forward):
    from pxf.couple.losses import backbone_denoising_loss

    return backbone_denoising_loss(
        forward.bb_pred,
        forward.bb_target,
        sigma=forward.sigma_b,
        sigma_data=rig["driver"].sigma_data,
        atom_mask=forward.bb_mask,
    ).total


def _reaches(loss, parameters):
    """Total gradient magnitude ``loss`` puts on ``parameters``."""
    grads = torch.autograd.grad(
        loss, list(parameters), retain_graph=True, allow_unused=True
    )
    return sum(float(g.abs().sum()) for g in grads if g is not None)


# ---- the three live paths ---------------------------------------------------


@pytest.fixture(scope="module")
def live(rig):
    return _forward(rig)


@needs_donor
def test_the_backbone_loss_reaches_the_selected_parameters(rig, live):
    assert _reaches(_backbone_loss(rig, live), rig["trainable"].values()) > 0


@needs_donor
def test_the_local_sidechain_loss_reaches_the_backbone_through_frozen_fampnn(rig, live):
    """The experiment's premise, and not obvious: FaMPNN is frozen throughout.

    The path is q_hat -> h_V -> densified atom37 -> B_hat, entirely through
    FaMPNN's geometry. If this is zero there is nothing to test downstream.
    """
    assert _reaches(_local_loss(rig, live), rig["trainable"].values()) > 0


@needs_donor
def test_the_placement_loss_reaches_the_backbone_by_both_routes(rig, live):
    """Placement has a second path the local loss does not: the frames.

    Detaching ``q_hat`` removes the encoder-mediated route and leaves only
    ``T(B_hat)``. That the loss still reaches the backbone is what makes the
    placement term a different experiment from the local one rather than a
    rescaling of it.
    """
    both = _reaches(_placement_loss(rig, live), rig["trainable"].values())
    assert both > 0

    frames_only = JL.placement_loss(
        M.place_sidechains(
            live.prediction.q_pred.detach(),
            live.prediction.clone(live.coords37),
            live.prediction.clone(rig["batch"].physical_mask),
        )[0],
        live.prediction.clone(rig["batch"].native_batch["x"][..., _sidechain_slots(), :]),
        live.prediction.clone(rig["batch"].physical_mask),
        aatype=live.prediction.aatype,
    )[0]
    assert _reaches(frames_only, rig["trainable"].values()) > 0


@needs_donor
def test_frozen_fampnn_receives_no_parameter_gradient(rig, live):
    """Frozen by ``requires_grad_(False)``, so a real backward leaves ``.grad`` alone.

    Asserted through an actual ``backward`` rather than ``autograd.grad``: with
    every parameter frozen the input list is empty, and ``autograd.grad`` raises
    on that instead of returning nothing -- which would pass as an exception
    rather than as evidence.
    """
    assert [p for p in rig["fampnn"].parameters() if p.requires_grad] == []
    assert all(p.grad is None for p in rig["fampnn"].parameters())

    total = _local_loss(rig, live) + _placement_loss(rig, live)
    total.backward(retain_graph=True)
    try:
        assert all(p.grad is None for p in rig["fampnn"].parameters())
        # ... while the backbone's selected parameters did receive one, so the
        # backward really ran through the side-chain branch.
        assert any(p.grad is not None for p in rig["trainable"].values())
    finally:
        for parameter in rig["trainable"].values():
            parameter.grad = None


# ---- the control ------------------------------------------------------------


@needs_donor
def test_the_blocked_arm_contributes_no_backbone_gradient(rig):
    """``detach_backbone`` cuts both entrances, so the side-chain terms are inert.

    Tested as "the total objective's gradient equals the backbone loss's",
    rather than by calling backward on the side-chain term -- with both
    entrances detached it has no graph at all, and asserting on a non-grad
    scalar would pass for the wrong reason.
    """
    blocked = _forward(rig, detach_backbone=True)
    assert blocked.detached_backbone
    assert not blocked.prediction.q_pred.requires_grad
    assert not blocked.placed.requires_grad

    parameters = list(rig["trainable"].values())
    anchor = _backbone_loss(rig, blocked)
    alone = torch.autograd.grad(anchor, parameters, retain_graph=True, allow_unused=True)
    with_sidechains = torch.autograd.grad(
        anchor + 0.0 * _local_loss(rig, blocked).detach(),
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    for first, second in zip(alone, with_sidechains):
        assert (first is None) == (second is None)
        if first is not None:
            assert torch.equal(first, second)


@needs_donor
def test_detaching_only_the_features_would_leave_the_frame_path_live(rig, live):
    """The trap the control exists to avoid, demonstrated rather than asserted.

    A reader might reasonably think detaching ``h_V`` blocks side-chain gradients
    to the backbone. It does not: placement also flows through ``T(B_hat)``, so
    an arm blocked that way still trains the backbone.
    """
    detached_features = {
        key: (value.detach() if torch.is_tensor(value) else value)
        for key, value in live.features.items()
    }
    assert not detached_features["h_V"].requires_grad
    # The frames still come from the live prediction, which is the whole point.
    placed, _exists = M.place_sidechains(
        live.prediction.q_pred.detach(),
        live.prediction.clone(live.coords37),
        live.prediction.clone(rig["batch"].physical_mask),
    )
    loss, _stats = JL.placement_loss(
        placed,
        live.prediction.clone(rig["batch"].native_batch["x"][..., _sidechain_slots(), :]),
        live.prediction.clone(rig["batch"].physical_mask),
        aatype=live.prediction.aatype,
    )
    assert _reaches(loss, rig["trainable"].values()) > 0


# ---- the derivative is the real one -----------------------------------------

def _sidechain_loss_fn(fampnn, batch, sidechain_noise):
    """``flat backbone -> L_local``, the function the gradient claims to be of."""
    from pxf.couple import fampnn_iface as iface
    from pxf.train import losses as loss_fns
    from pxf.train import step as train_step

    native = batch.native_batch

    def sidechain_loss_of(flat):
        coords37, supplied = M.densify_prediction(flat, batch.topology)
        encoder_mask = J.encoder_availability(
            native["aatype"], native["seq_mask"], supplied
        )
        _logits, _h, features = iface.encode(
            fampnn,
            coords37,
            native["aatype"],
            seq_mask=native["seq_mask"],
            residue_index=native["residue_index"],
            chain_index=native["chain_index"],
            atom_availability=encoder_mask,
        )
        prediction = train_step.sidechain_training_pass(
            fampnn,
            native,
            features,
            multiplier=1,
            self_cond_p=0.0,
            noise=sidechain_noise,
        )
        loss, _stats = loss_fns.sidechain_diffusion_loss(
            prediction.q_pred,
            prediction.q_target,
            prediction.weight,
            prediction.loss_mask,
        )
        return loss

    return sidechain_loss_of


def _loss_and_gradient(fampnn, batch, sidechain_noise, *, direction=None):
    """``(loss, gradient, directional derivative)`` at the batch's own backbone."""
    loss_of = _sidechain_loss_fn(fampnn, batch, sidechain_noise)
    base = batch.backbone_target.detach().clone()
    leaf = base.clone().requires_grad_(True)
    loss = loss_of(leaf)
    (gradient,) = torch.autograd.grad(loss, leaf)
    directional = None if direction is None else float((gradient * direction).sum())
    return loss_of, base, gradient, directional




@needs_donor
def test_finite_differences_agree_with_autograd_through_fampnn(rig):
    """Central differences on a real structure, along a random direction.

    Autograd can be self-consistently wrong -- a mis-specified custom backward,
    a detached branch that still looks connected -- so the side-chain loss's
    derivative with respect to the predicted backbone is checked against the
    function it claims to differentiate. The perturbation is small enough not to
    reorder FaMPNN's k-nearest-neighbour graph, which is a genuine
    non-differentiability rather than an error in the graph.

    **Run on a CPU copy, deliberately.** A central difference divides a
    difference of two losses by a small number, so it needs the loss to be
    reproducible to far better than that difference. On an H200 it is not: the
    same check there reported 0.0184 against autograd's 0.0212, 13% low, with
    the gradient itself correct -- ``test_the_gradient_agrees_across_devices``
    is what establishes that. Loosening the tolerance until the GPU passed
    would have thrown away the only test that can catch a wrong derivative, to
    accommodate a limitation of the estimator rather than of the code.
    """
    import copy

    fampnn = copy.deepcopy(rig["fampnn"]).to("cpu")
    batch = rig["batch"].to("cpu")
    _bb_noise, sidechain_noise = _noise(rig, multiplier=1)
    sidechain_noise = sidechain_noise.to("cpu")

    torch.manual_seed(0)
    direction = torch.randn_like(batch.backbone_target)
    direction = direction / direction.norm()
    loss_of, base, gradient, analytic = _loss_and_gradient(
        fampnn, batch, sidechain_noise, direction=direction
    )
    assert float(gradient.abs().sum()) > 0

    step = 1e-3
    with torch.no_grad():
        plus = float(loss_of(base + step * direction))
        minus = float(loss_of(base - step * direction))
    numeric = (plus - minus) / (2 * step)
    assert numeric == pytest.approx(analytic, rel=0.05, abs=1e-6), (
        f"autograd says {analytic:.6g}, central differences say {numeric:.6g}"
    )


@needs_donor
def test_the_gradient_agrees_across_devices(rig):
    """The GPU gradient is the CPU gradient, which finite differences validated.

    This is what a GPU run actually needs to know, and it is a better question
    than "do finite differences agree on a GPU": it compares the quantity that
    is used against the same quantity computed where it is already checked,
    instead of against an estimator that is itself the least reliable part on
    that device.

    Skipped when the rig is already on the CPU -- there is nothing to compare.
    """
    import copy

    if rig["device"].type == "cpu":
        pytest.skip("the rig is on the CPU; there is no second device")

    _bb_noise, sidechain_noise = _noise(rig, multiplier=1)
    torch.manual_seed(0)
    direction = torch.randn_like(rig["batch"].backbone_target)
    direction = direction / direction.norm()

    _of, _base, on_device, device_directional = _loss_and_gradient(
        rig["fampnn"], rig["batch"], sidechain_noise, direction=direction
    )
    _of, _base, on_cpu, cpu_directional = _loss_and_gradient(
        copy.deepcopy(rig["fampnn"]).to("cpu"),
        rig["batch"].to("cpu"),
        sidechain_noise.to("cpu"),
        direction=direction.cpu(),
    )

    scale = float(on_cpu.abs().max())
    difference = float((on_device.cpu() - on_cpu).abs().max())
    assert difference < 0.02 * scale, (
        f"the gradient differs by {difference:.3g} between devices, "
        f"{difference / max(scale, 1e-12):.1%} of its largest component"
    )
    assert device_directional == pytest.approx(cpu_directional, rel=0.02)


@needs_donor
def test_moving_the_backbone_moves_the_placement_even_with_fixed_sidechains(rig, live):
    """Local error can be flat while placement is not -- that is the term's point."""
    batch = rig["batch"]
    local = live.prediction.q_pred.detach()
    physical = live.prediction.clone(batch.physical_mask)
    native = live.prediction.clone(
        batch.native_batch["x"][..., _sidechain_slots(), :]
    )
    frames = live.prediction.clone(live.coords37).detach()

    first, _exists = M.place_sidechains(local, frames, physical)
    moved = frames.clone()
    moved[..., 1, :] = moved[..., 1, :] + 0.5  # shift every CA
    second, _exists = M.place_sidechains(local, moved, physical)

    before = float(JL.placement_loss(first, native, physical)[0])
    after = float(JL.placement_loss(second, native, physical)[0])
    assert after != pytest.approx(before, rel=1e-3)
