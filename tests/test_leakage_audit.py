"""Ground truth must reach ``a_token`` only through the noised backbone.

The coupling's claim is that ``A_BS`` exploits PXDesign's learned denoising
representation. That claim is void if native coordinates reach ``a_token`` by
any second path, because the adapter could then be reading the answer instead
of inferring it:

    X_gt -> x_t -> PXDesign -> a_token -> A_BS -> FaMPNN      permitted
    X_gt -> template / ref / frame features -> a_token        forbidden

Two candidate paths exist in this featurization and both are checked here.

``conditional_templ`` is the real one to watch: PXDesign's
``get_condition_embedding`` documents ``z`` as encoding ``conditional_templ``
and ``conditional_templ_mask``, and ``z_trunk`` feeds the diffusion module that
emits ``a_token``. It is empty here only because the whole monomer is the
design region (``max_binder_fraction=1.0``, ``aa_mask_mode="all"``), so there is
nothing to template on. Change those dataset settings and the path opens.

``sc_bb_coords`` is the surprising one: it holds the native backbone *exactly*
(0.000 A per-atom), because Proteo-AA's own side-chain head reads it to build
frames. This pipeline packs with FaMPNN instead and never uses that head, so the
tensor is inert -- but it sits in the same feature dict, and "inert" is a
property of the consumer, not of the data. It is asserted rather than assumed.

``ref_pos`` legitimately reaches ``a_token``. It is the idealized reference
conformer -- centroid at the origin, ~1.8 A radius, ~9 A from the native
coordinates -- so it carries residue chemistry, not native geometry.
"""

import copy

import pytest
import torch

CIF = "/hai/scratch/yfsun/afdb_laproteina/cif_val/AF-P81613-F1-model_v4.cif"
DONOR = (
    "/hai/users/y/f/yfsun/Proteo-AA old/Proteo-AA-official-pxdesign-fampnn/"
    "runs/component_donors/pxdesign_v0.1.0.pt"
)
# Features that hold, or could hold, native geometry. Perturbing any of them
# must leave a_token untouched.
GT_BEARING = ("sc_bb_coords", "sc_frame_R")


def featurized():
    import pathlib

    if not pathlib.Path(CIF).is_file():
        pytest.skip(f"{CIF} not present")
    # pxdesign_train is not importable directly -- `pxf.backbone.proteoaa` stubs
    # its parent packages and inserts the worktree path, so go through the
    # driver rather than importorskip on the bare module name.
    from pxf.backbone.driver import featurize_structures, to_featurized

    ((sid, dataset),) = featurize_structures([CIF], crop_size=512)
    item = dataset[0]
    return sid, item, to_featurized(sid, item)


# --- cheap checks: no model needed ------------------------------------------


def test_the_template_path_carries_nothing():
    """``z`` encodes conditional_templ; here it must be entirely empty."""
    _sid, item, _st = featurized()
    fd = item["input_feature_dict"]
    for key in (
        "conditional_templ",
        "conditional_templ_mask",
        "template_all_atom_positions",
        "template_all_atom_mask",
    ):
        if key not in fd:
            continue
        value = fd[key]
        assert int((value != 0).sum()) == 0, (
            f"{key} is non-zero, so GT geometry can reach a_token through the "
            "template branch of the pair representation"
        )


def test_ref_pos_is_a_reference_conformer_not_the_native_structure():
    """ref_pos does reach a_token, so it must not be native coordinates."""
    _sid, item, _st = featurized()
    ref = item["input_feature_dict"]["ref_pos"].reshape(-1, 3)
    native = item["label_dict"]["coordinate"].reshape(-1, 3)
    assert ref.shape == native.shape
    # Centred on the origin and small: a per-residue idealized conformer.
    assert float(ref.mean(0).abs().max()) < 1e-3
    assert float(ref.norm(dim=-1).max()) < 5.0
    # And nowhere near the deposited coordinates.
    assert float((ref - native).norm(dim=-1).mean()) > 2.0


def test_sc_bb_coords_really_is_the_native_backbone():
    """Pins *why* the invariance test below matters, rather than assuming it."""
    _sid, item, _st = featurized()
    fd = item["input_feature_dict"]
    if "sc_bb_coords" not in fd:
        pytest.skip("no sc_bb_coords in this featurization")
    native = item["label_dict"]["coordinate"].reshape(-1, 3)
    flat = fd["sc_bb_coords"].reshape(-1, 3)
    assert flat.shape[0] <= native.shape[0]
    # Same centroid to float precision: this is the deposited backbone.
    assert torch.allclose(flat.mean(0), native.mean(0), atol=1e-3)


# --- the real check: perturb GT features, a_token must not move --------------


@pytest.mark.slow
def test_a_token_is_invariant_to_every_gt_bearing_feature():
    """The audit. Requires the 557 MB donor, so it is marked slow.

    A positive control is included deliberately: a test that cannot detect a
    change would pass this vacuously.
    """
    import pathlib

    if not pathlib.Path(DONOR).is_file():
        pytest.skip("PXDesign donor not present")
    from pxf.backbone.driver import PXDesignBackboneDriver, load_backbone_model

    _sid, _item, structure = featurized()
    model, _cfg, _rec = load_backbone_model(DONOR, device="cpu")
    driver = PXDesignBackboneDriver(model)
    sigma = torch.tensor([1.642])
    target = structure.backbone_target.float()
    noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(1))
    x_noisy = (target + noise * 1.642)[None]

    def a_token(feature_dict, x):
        denoise = driver.bind(driver.conditioning(feature_dict))
        with torch.no_grad():
            _x0, a = denoise(x, sigma)
        return a

    base = a_token(copy.deepcopy(structure.feature_dict), x_noisy)

    for key in GT_BEARING:
        if key not in structure.feature_dict:
            continue
        for label, make in (
            ("shifted", lambda v: v + 50.0),
            ("randomized", lambda v: torch.randn_like(v) * 10.0),
            ("zeroed", torch.zeros_like),
        ):
            fd = copy.deepcopy(structure.feature_dict)
            fd[key] = make(fd[key])
            moved = float((a_token(fd, x_noisy) - base).abs().max())
            assert moved < 1e-6, (
                f"a_token moved by {moved:.3e} when {key} was {label}: native "
                "geometry is reaching the token features outside x_t"
            )

    # Positive control: the noised backbone must matter, or the above is vacuous.
    shifted = float(
        (a_token(copy.deepcopy(structure.feature_dict), x_noisy + 5.0) - base).abs().max()
    )
    assert shifted > 1e-3, "a_token ignores x_noisy; the invariance test proves nothing"
