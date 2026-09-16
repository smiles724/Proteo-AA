"""The real PXDesign backbone driver.

Gated on the donor checkpoint and a Proteo-AA checkout, since both are large and
external. What these pin are the two failures that made the driver look broken
in ways that would not have been obvious from the loss curve:

* Protenix's activation checkpointing recomputes the forward during backward, and
  the feedback-injection hook makes the recomputation diverge (``CheckpointError:
  A different number of tensors was saved``). The driver disables it.
* Protenix calls ``AtomAttentionDecoder`` positionally through ``checkpoint_fn``
  but with ``a=`` as a keyword otherwise. A hook that handles only one form stops
  injecting under the other, which presents as an adapter that learned nothing.
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
    reason="set PXDESIGN_DONOR and PXF_TEST_CIF to run the real backbone driver",
)


@pytest.fixture(scope="module")
def driver_and_structure():
    from pxf.backbone.driver import (
        PXDesignBackboneDriver,
        featurize_structures,
        load_backbone_model,
        to_featurized,
    )

    model, _, record = load_backbone_model(DONOR)
    sample_id, source = featurize_structures([CIF], crop_size=128)[0]
    structure = to_featurized(sample_id, source[0])
    return PXDesignBackboneDriver(model), structure, record


@needs_donor
def test_donor_loads_with_no_missing_or_unexpected_keys(driver_and_structure):
    """A partially loaded backbone still emits plausible coordinates."""
    driver, _, record = driver_and_structure
    assert record["model_name"] == "pxdesign_v0.1.0"
    assert record["driver"] == "pxdesign_train"
    assert driver.c_token == 768
    assert driver.sigma_data == pytest.approx(16.0)


@needs_donor
def test_a_mismatched_donor_is_refused(tmp_path):
    from pxf.backbone.driver import load_backbone_model

    broken = tmp_path / "broken.pt"
    torch.save({"model": {"not_a_real_parameter": torch.zeros(2)}}, broken)
    with pytest.raises(ValueError, match="does not match the backbone exactly"):
        load_backbone_model(broken)


@needs_donor
def test_activation_checkpointing_is_off_by_default(driver_and_structure):
    driver, _, _ = driver_and_structure
    assert not any(getattr(m, "blocks_per_ckpt", None) for m in driver.model.modules())


@needs_donor
def test_denoising_improves_a_noised_backbone(driver_and_structure):
    driver, structure, _ = driver_and_structure
    denoise = driver.bind(driver.conditioning(structure.feature_dict))
    target = structure.backbone_target.float()
    torch.manual_seed(0)
    sigma = 5.0
    noisy = (target + torch.randn_like(target) * sigma)[None]
    denoised, a_token = denoise(noisy, torch.tensor([sigma]))
    assert denoised.shape == noisy.shape
    assert a_token.shape[-1] == driver.c_token
    assert a_token.shape[-2] == structure.num_tokens

    def rmsd(a, b):
        return float((a - b).pow(2).sum(-1).mean().sqrt())

    assert rmsd(denoised[0], target) < rmsd(noisy[0], target)


@needs_donor
def test_feedback_reaches_the_output_and_zero_is_a_no_op(driver_and_structure):
    """Both halves matter: injection must work, and must vanish at zero."""
    driver, structure, _ = driver_and_structure
    denoise = driver.bind(driver.conditioning(structure.feature_dict))
    target = structure.backbone_target.float()
    torch.manual_seed(0)
    noisy = (target + torch.randn_like(target) * 5.0)[None]
    baseline, _ = denoise(noisy, torch.tensor([5.0]))
    delta = torch.full((1, structure.num_tokens, driver.c_token), 0.05)
    moved, _ = denoise(noisy, torch.tensor([5.0]), feedback=delta)
    assert not torch.equal(baseline, moved)
    zeroed, _ = denoise(noisy, torch.tensor([5.0]), feedback=torch.zeros_like(delta))
    # Exact, not approximate: this is what makes phase-0 equivalence hold through
    # the real driver.
    assert torch.equal(baseline, zeroed)


@needs_donor
def test_featurization_makes_the_whole_monomer_the_design_region(driver_and_structure):
    _, structure, _ = driver_and_structure
    assert int(structure.design_mask.sum()) == structure.num_tokens
    # Identities come from aa_clean, not the scrubbed structure_res_name, which
    # would teacher-force glycine everywhere.
    assert set(structure.aatype.tolist()) != {7}
    assert structure.backbone_target.shape[0] == structure.num_tokens * 4


@needs_donor
def test_the_gradient_path_from_l_bb_reaches_the_feedback(driver_and_structure):
    """The end-to-end check that phase 2 can train at all on the real model."""
    driver, structure, _ = driver_and_structure
    denoise = driver.bind(driver.conditioning(structure.feature_dict))
    target = structure.backbone_target.float()
    torch.manual_seed(0)
    noisy = (target + torch.randn_like(target) * 5.0)[None]
    delta = torch.zeros(1, structure.num_tokens, driver.c_token, requires_grad=True)
    denoised, _ = denoise(noisy, torch.tensor([5.0]), feedback=delta)
    (denoised - target[None]).pow(2).sum().backward()
    assert delta.grad is not None and float(delta.grad.abs().sum()) > 0
