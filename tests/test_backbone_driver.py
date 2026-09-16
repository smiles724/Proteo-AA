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
* ``pxdesign_train`` emits CPU tensors while the model sits wherever it was
  loaded, and the mismatch surfaces at the condition embedder's first
  ``F.linear`` rather than anywhere informative, because every step before the
  first weight multiply is indexing, which tolerates a CPU index.

**Set ``LAYERNORM_TYPE=torch``** to run these, as the launchers do. Without it
Protenix selects its fused LayerNorm kernel, which raises ``RuntimeError: input
must be a CUDA tensor`` on a CPU model -- a message that invites the wrong
conclusion that these tests need a GPU. They do not; they need the non-fused
kernel.
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


@needs_donor
def test_the_structure_moves_every_tensor_and_keeps_the_string_columns(
    driver_and_structure,
):
    """``pxdesign_train`` emits CPU tensors; the model is wherever it was loaded.

    Without this the run dies inside the condition embedder's first ``F.linear``
    on mixed devices -- and not at the obvious place, because everything up to
    the first weight multiply is pure indexing, which tolerates a CPU index.
    """
    _, structure, _ = driver_and_structure
    moved = structure.to("cpu")  # a no-op device, so this runs anywhere

    strings = [k for k, v in structure.feature_dict.items() if not torch.is_tensor(v)]
    assert strings, "the featurizer is expected to emit string columns"
    for key in strings:
        assert moved.feature_dict[key] is structure.feature_dict[key]
    tensors = [k for k, v in structure.feature_dict.items() if torch.is_tensor(v)]
    assert len(tensors) > 50
    for key in tensors:
        assert moved.feature_dict[key].device == torch.device("cpu")

    # The topology travels too: indexing tolerates a CPU index, arithmetic does not.
    assert moved.topology.atom_to_token_idx.device == torch.device("cpu")
    assert moved.topology.num_tokens == structure.topology.num_tokens
    assert moved.topology.atom_names is structure.topology.atom_names
    for field in ("aatype", "design_mask", "backbone_target"):
        assert getattr(moved, field).device == torch.device("cpu")


@needs_donor
def test_conditioning_moves_the_dict_itself(driver_and_structure):
    """A caller that forgets ``.to()`` should still get a run, not a traceback."""
    driver, structure, _ = driver_and_structure
    stale = {
        key: (value.cpu() if torch.is_tensor(value) else value)
        for key, value in structure.feature_dict.items()
    }
    conditioning = driver.conditioning(stale)
    assert conditioning.s_inputs.device == driver.device
