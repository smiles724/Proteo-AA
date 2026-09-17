"""The SC -> BB pilot against the real models, not a stub.

``tests/test_couple_phases.py`` drives a stub backbone whose feedback path is
``x_noisy + feedback.mean()``. That is enough to prove a gradient *exists*, and
it cannot prove anything about the real decoder: the hook injects after
``layernorm_a`` and before ``atom_attention_decoder``, Protenix calls that
submodule positionally under activation checkpointing and by keyword otherwise,
and the whole question of the pilot is whether a residual placed there reaches
the coordinates at all.

So these run the published donor. What they pin:

* zero feedback reproduces the uncorrected denoiser **exactly**;
* non-zero feedback moves the coordinates;
* L_BB produces finite, non-zero gradients in A_SB's output projection;
* PXDesign, FaMPNN and A_BS are unchanged by the step.

**Set ``LAYERNORM_TYPE=torch``**, as the launchers do; see
``tests/test_backbone_driver.py`` for why that is not a GPU requirement.
"""

import os
from pathlib import Path

import pytest
import torch

DONOR = os.environ.get(
    "PXDESIGN_DONOR",
    "/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn"
    "/runs/component_donors/pxdesign_v0.1.0.pt",
)
CIF = os.environ.get("PXF_TEST_CIF", "/hai/scratch/yfsun/casp14/cif/T1031.cif")

needs_donor = pytest.mark.skipif(
    not (Path(DONOR).is_file() and Path(CIF).is_file()),
    reason="set PXDESIGN_DONOR and PXF_TEST_CIF to run the real-model pilot checks",
)

pytestmark = needs_donor

SIGMA = 0.8  # inside the pilot's [0.1, 2.0] window, so the gate is exactly 1


@pytest.fixture(scope="module")
def rig():
    """Both real donors, one real structure, and a controller wired as the pilot."""
    from fampnn.model.sd_model import SeqDenoiser

    from pxf import provenance
    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        featurize_structures,
        load_backbone_model,
        to_featurized,
    )
    from pxf.couple import pilot
    from pxf.couple.adapters import CouplingAdapters
    from pxf.couple.controller import CoupledDenoiser
    from pxf.couple.fampnn_iface import node_feature_dim
    from pxf.couple.readout import FeedbackPath, SigmaWindow

    bundle = torch.load(
        provenance.fampnn_checkpoint("0.0"), map_location="cpu", weights_only=False
    )
    fampnn = SeqDenoiser(bundle["model_cfg"])
    fampnn.load_state_dict(bundle["state_dict"], strict=True)
    fampnn.eval().requires_grad_(False)

    model, _configs, _record = load_backbone_model(DONOR)
    driver = PXDesignBackboneDriver(model)
    sample_id, source = featurize_structures([CIF], crop_size=128)[0]
    structure = to_featurized(sample_id, source[0])

    sb = FeedbackPath(
        node_feature_dim(fampnn),
        driver.c_token,
        variant="full",
        gate=SigmaWindow(0.1, 2.0),
    )
    adapters = CouplingAdapters(driver.c_token, node_feature_dim(fampnn), sc_to_bb=sb)
    adapters.set_phase("sc_to_bb")
    controller = CoupledDenoiser(
        driver.bind(driver.conditioning(structure.feature_dict)),
        fampnn,
        adapters,
        phase="sc_to_bb",
        pack_steps=3,  # the pilot uses 50; this is about the gradient path
    )
    target = structure.backbone_target.float()
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(target.shape, generator=generator)
    mask = pilot.backbone_supervision_mask(
        structure.topology.atom_names,
        coordinate_mask=structure.label_dict.get("coordinate_mask"),
    )
    return dict(
        controller=controller,
        adapters=adapters,
        sb=sb,
        fampnn=fampnn,
        model=model,
        structure=structure,
        target=target,
        mask=mask,
        x_noisy=(target + noise * SIGMA)[None],
        sigma=torch.full((1,), SIGMA),
        sigma_data=driver.sigma_data,
    )


@pytest.fixture(scope="module")
def upstream(rig):
    """The frozen half once; every check below reuses it."""
    return rig["controller"].frozen_half(
        rig["structure"].topology,
        rig["x_noisy"],
        rig["sigma"],
        rig["structure"].aatype,
        bs_delta_h=None,  # the bypass, as the pilot config selects
    )


def test_the_supervised_mask_covers_the_whole_backbone_only_axis(rig):
    """Records what the monomer configuration actually emits.

    ``backbone_only_binder=True`` over a fully designed chain gives a
    backbone-only flat axis, so the mask is all ones here and an unmasked L_BB
    was not silently scoring side chains. Asserted rather than assumed, because
    the mask's other job -- keeping CA-collapsed design side chains out of the
    supervised set -- only matters if that ever changes, and this is where the
    change would show up.
    """
    mask = rig["mask"]
    names = [str(n) for n in rig["structure"].topology.atom_names]
    assert float(mask.sum()) > 0
    kept = {names[i] for i in torch.nonzero(mask, as_tuple=True)[0].tolist()}
    assert kept <= {"N", "CA", "C", "O"}, sorted(kept)
    assert set(names) == {"N", "CA", "C", "O"}, (
        "the featurizer now puts non-backbone atoms on the flat axis. Those "
        "carry CA-collapsed coordinates for the design region, so the mask has "
        f"stopped being a no-op -- which is fine, but check it: {sorted(set(names))}"
    )
    assert float(mask.sum()) == mask.numel()
    assert mask.numel() == 4 * rig["structure"].num_tokens


def test_the_frozen_half_is_reproducible_from_its_seed(rig):
    """Every arm must read the SAME sc0, or the comparison includes the sampler.

    The packing sampler draws from the global RNG, so two runs of the same
    example give different side chains unless the seed is set first. That would
    make the bb_only and generic controls read a different packing than the
    candidate -- a difference in the sampler wearing the adapter's name.
    """
    controller = rig["controller"]
    states = []
    for _ in range(2):
        torch.manual_seed(1234)
        states.append(
            controller.frozen_half(
                rig["structure"].topology,
                rig["x_noisy"],
                rig["sigma"],
                rig["structure"].aatype,
                bs_delta_h=None,
            )
        )
    assert torch.equal(states[0].sidechains, states[1].sidechains)
    assert torch.equal(states[0].packed.h_packed, states[1].packed.h_packed)
    # And the seed is load-bearing: a different one gives a different packing.
    torch.manual_seed(4321)
    other = controller.frozen_half(
        rig["structure"].topology,
        rig["x_noisy"],
        rig["sigma"],
        rig["structure"].aatype,
        bs_delta_h=None,
    )
    assert not torch.equal(states[0].sidechains, other.sidechains), (
        "the packing does not depend on the RNG, so seeding it is pointless -- "
        "check that pack_steps > 1 and that the sampler is stochastic"
    )


def test_the_frozen_half_sees_the_packing(rig, upstream):
    packed = upstream.packed
    relative = float(
        (packed.h_packed - packed.h_base).norm() / packed.h_base.norm().clamp_min(1e-8)
    )
    assert relative > 1e-3, f"h_packed == h_base through the real path ({relative:.2e})"
    assert upstream.packed.visibility.stats["sidechain_atoms_available"] > 0
    assert not packed.h_packed.requires_grad, "the frozen half was not detached"


def test_zero_feedback_reproduces_the_original_denoiser(rig, upstream):
    """Bit-for-bit, not approximately. A_SB is zero-initialized."""
    controller = rig["controller"]
    assert rig["sb"].is_identity()
    with torch.no_grad():
        plain, _a = controller.backbone(rig["x_noisy"], rig["sigma"])
        cycle = controller.corrective_event(
            rig["structure"].topology,
            rig["x_noisy"],
            rig["sigma"],
            rig["structure"].aatype,
            bs_delta_h=None,
            upstream=upstream,
        )
    assert float(cycle.delta_a.abs().max()) == 0.0
    assert torch.equal(plain, cycle.bb1_flat), (
        "a zero correction changed the coordinates, so the feedback hook is "
        "perturbing the pass by something other than delta_a"
    )


def test_nonzero_feedback_changes_the_coordinates(rig, upstream):
    controller, sb = rig["controller"], rig["sb"]
    saved = sb.project_out.weight.detach().clone()
    try:
        torch.nn.init.normal_(sb.project_out.weight, std=0.02)
        with torch.no_grad():
            plain, _a = controller.backbone(rig["x_noisy"], rig["sigma"])
            cycle = controller.corrective_event(
                rig["structure"].topology,
                rig["x_noisy"],
                rig["sigma"],
                rig["structure"].aatype,
                bs_delta_h=None,
                upstream=upstream,
            )
        assert float(cycle.delta_a.abs().max()) > 0
        shift = float((cycle.bb1_flat - plain).norm(dim=-1).mean())
        assert shift > 1e-4, (
            f"a non-zero residual at the decoder input moved the coordinates by "
            f"{shift:.2e} A on average; the hook is not reaching the output"
        )
        assert cycle.feedback_stats["relative_residual"] > 0
    finally:
        with torch.no_grad():
            sb.project_out.weight.copy_(saved)


def test_l_bb_gives_finite_nonzero_gradients_to_the_output_projection(rig, upstream):
    """The check the stub cannot make: through the real atom decoder."""
    from pxf.couple import losses as couple_losses

    controller, sb = rig["controller"], rig["sb"]
    sb.zero_grad(set_to_none=True)
    cycle = controller.corrective_event(
        rig["structure"].topology,
        rig["x_noisy"],
        rig["sigma"],
        rig["structure"].aatype,
        bs_delta_h=None,
        upstream=upstream,
    )
    assert cycle.delta_a.requires_grad, "the trainable half was captured by no_grad"
    loss = couple_losses.backbone_feedback_loss(
        cycle,
        rig["target"][None],
        sigma=rig["sigma"],
        sigma_data=rig["sigma_data"],
        atom_mask=rig["mask"][None],
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    grad = sb.project_out.weight.grad
    assert grad is not None, "no gradient reached A_SB's output projection"
    assert torch.isfinite(grad).all(), "the gradient is not finite"
    assert float(grad.abs().sum()) > 0, (
        "L_BB produced a zero gradient in W2, so the pilot would train nothing "
        "while logging a loss curve"
    )
    # At zero-init the gradient stops at W2, and that is correct rather than a
    # problem: dL/dhidden = W2^T dL/ddelta, and W2 is zero. The readout starts
    # receiving gradient on the step after W2 first moves. Pinned in both
    # directions so a future refactor cannot make step 0 look healthy while
    # leaving the readout permanently unreachable.
    assert float(sb.readout.norm.weight.grad.abs().sum()) == 0.0
    sb.zero_grad(set_to_none=True)
    with torch.no_grad():
        torch.nn.init.normal_(sb.project_out.weight, std=0.02)
    try:
        cycle = controller.corrective_event(
            rig["structure"].topology,
            rig["x_noisy"],
            rig["sigma"],
            rig["structure"].aatype,
            bs_delta_h=None,
            upstream=upstream,
        )
        couple_losses.backbone_feedback_loss(
            cycle,
            rig["target"][None],
            sigma=rig["sigma"],
            sigma_data=rig["sigma_data"],
            atom_mask=rig["mask"][None],
        ).total.backward()
        for name, parameter in (
            ("readout.norm.weight", sb.readout.norm.weight),
            ("readout.sequence.weight", sb.readout.sequence.weight),
            ("project_in.weight", sb.project_in.weight),
        ):
            assert parameter.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(parameter.grad).all(), f"{name} gradient is not finite"
            assert float(parameter.grad.abs().sum()) > 0, (
                f"once W2 is non-zero, {name} must receive gradient or the "
                "readout's content can never be used"
            )
    finally:
        with torch.no_grad():
            torch.nn.init.zeros_(sb.project_out.weight)
        sb.zero_grad(set_to_none=True)


def test_the_frozen_components_do_not_move(rig, upstream):
    """PXDesign, FaMPNN and A_BS must be untouched by a pilot step."""
    from pxf.couple import losses as couple_losses

    controller, sb, adapters = rig["controller"], rig["sb"], rig["adapters"]
    before = {
        "pxdesign": {
            k: v.detach().clone() for k, v in list(rig["model"].state_dict().items())[:40]
        },
        "fampnn": {
            k: v.detach().clone() for k, v in list(rig["fampnn"].state_dict().items())[:40]
        },
        "bb_to_sc": {
            k: v.detach().clone() for k, v in adapters.bb_to_sc.state_dict().items()
        },
    }
    assert not any(p.requires_grad for p in rig["model"].parameters())
    assert not any(p.requires_grad for p in rig["fampnn"].parameters())
    assert not any(p.requires_grad for p in adapters.bb_to_sc.parameters())

    optimizer = torch.optim.AdamW(
        [p for p in adapters.parameters() if p.requires_grad], lr=1e-3
    )
    cycle = controller.corrective_event(
        rig["structure"].topology,
        rig["x_noisy"],
        rig["sigma"],
        rig["structure"].aatype,
        bs_delta_h=None,
        upstream=upstream,
    )
    couple_losses.backbone_feedback_loss(
        cycle,
        rig["target"][None],
        sigma=rig["sigma"],
        sigma_data=rig["sigma_data"],
        atom_mask=rig["mask"][None],
    ).total.backward()
    optimizer.step()

    for name, saved in before.items():
        live = (
            rig["model"].state_dict()
            if name == "pxdesign"
            else rig["fampnn"].state_dict()
            if name == "fampnn"
            else adapters.bb_to_sc.state_dict()
        )
        for key, value in saved.items():
            assert torch.equal(live[key], value), f"{name}.{key} moved"
    # And A_SB did move, or the step proved nothing.
    assert not sb.is_identity(), "the optimizer step left A_SB at zero"
