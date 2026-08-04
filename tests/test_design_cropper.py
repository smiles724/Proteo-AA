"""
Synthetic tests for `DesignCropper`.

We construct a 3-chain protein complex (binder + 2 target chains) where the
binder sits closer to chain A than to chain B. After cropping we expect:
  - all binder tokens to be retained
  - if the crop budget forces dropping target residues, those dropped should
    be the farthest ones (chain B residues drop before chain A)
  - the rebuilt `binder_atom_mask` is consistent with the cropped array
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "PXDesign")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "Protenix")))


BACKBONE = ("N", "CA", "C", "O")
ATOMS_PER_RES = len(BACKBONE)


def _make_synthetic_complex():
    """Build a 3-chain backbone-only protein.

    Geometry (Cα positions only — N/C/O are small offsets from Cα):
        Chain A (target): residues laid out along x at y=0   (close to binder)
        Chain B (target): residues laid out along x at y=30  (far from binder)
        Chain C (binder): residues laid out along x at y=6   (close to A)
    """
    biotite = pytest.importorskip("biotite.structure")
    pytest.importorskip("protenix")

    from protenix.data.tokenizer import Token, TokenArray

    n_a, n_b, n_c = 12, 12, 6  # target_close, target_far, binder
    n_res = n_a + n_b + n_c
    n_atom = n_res * ATOMS_PER_RES

    aa = biotite.AtomArray(length=n_atom)
    aa.coord = np.zeros((n_atom, 3), dtype=np.float32)

    def fill_chain(start_atom_idx, chain_id, n, y):
        for r in range(n):
            for ai, name in enumerate(BACKBONE):
                i = start_atom_idx + r * ATOMS_PER_RES + ai
                aa.chain_id[i] = chain_id
                aa.res_id[i] = r + 1
                aa.res_name[i] = "GLY"
                aa.atom_name[i] = name
                aa.element[i] = "N" if name == "N" else ("O" if name == "O" else "C")
                cax = r * 3.8
                if name == "N":
                    aa.coord[i] = (cax - 1.0, y, 0.0)
                elif name == "CA":
                    aa.coord[i] = (cax, y, 0.0)
                elif name == "C":
                    aa.coord[i] = (cax + 1.0, y, 0.0)
                else:  # O
                    aa.coord[i] = (cax + 1.2, y + 1.0, 0.0)

    fill_chain(0, "A", n_a, y=0.0)
    fill_chain(n_a * ATOMS_PER_RES, "B", n_b, y=30.0)
    fill_chain((n_a + n_b) * ATOMS_PER_RES, "C", n_c, y=6.0)

    is_ca = aa.atom_name == "CA"
    aa.set_annotation("distogram_rep_atom_mask", is_ca.astype(int))
    aa.set_annotation("is_resolved", np.ones(n_atom, dtype=bool))
    aa.set_annotation("mol_type", np.array(["protein"] * n_atom))
    # asym_id_int: needed by Protenix's CropData but not by DesignCropper itself.
    chain_to_int = {"A": 0, "B": 1, "C": 2}
    aa.set_annotation(
        "asym_id_int",
        np.array([chain_to_int[c] for c in aa.chain_id], dtype=np.int64),
    )

    # Build a minimal TokenArray: one token per residue, centre atom = Cα.
    tokens = []
    for r in range(n_res):
        atom_indices = list(range(r * ATOMS_PER_RES, (r + 1) * ATOMS_PER_RES))
        centre = atom_indices[BACKBONE.index("CA")]
        tokens.append(Token(r, atom_indices=atom_indices, centre_atom_index=centre))
    token_array = TokenArray(tokens)

    return aa, token_array, n_a, n_b, n_c


def test_cropper_keeps_whole_binder_under_budget():
    from pxdesign_train.data import DesignCropper

    aa, tokens, n_a, n_b, n_c = _make_synthetic_complex()
    total = n_a + n_b + n_c

    # Budget larger than total -> nothing dropped.
    result = DesignCropper(crop_size=total + 10).crop(aa, tokens, binder_chain_id="C")
    assert len(result.token_array) == total
    assert result.n_binder_tokens == n_c
    assert result.n_target_tokens == n_a + n_b


def test_cropper_drops_far_target_first():
    """Crop budget = 18: keep all 6 binder + 12 target. Chain A (close) survives,
    chain B (far) is fully dropped."""
    from pxdesign_train.data import DesignCropper

    aa, tokens, n_a, n_b, n_c = _make_synthetic_complex()
    crop_size = n_c + n_a  # 18
    result = DesignCropper(crop_size=crop_size).crop(aa, tokens, binder_chain_id="C")

    assert len(result.token_array) == crop_size
    assert result.n_binder_tokens == n_c
    assert result.n_target_tokens == n_a

    # Chain B should be entirely dropped.
    chain_ids_kept = set(np.unique(result.atom_array.chain_id))
    assert chain_ids_kept == {"A", "C"}


def test_cropper_partial_target_drop():
    """Crop budget = 12: keep all 6 binder + 6 closest target tokens (all A's)."""
    from pxdesign_train.data import DesignCropper

    aa, tokens, n_a, n_b, n_c = _make_synthetic_complex()
    crop_size = n_c + 6
    result = DesignCropper(crop_size=crop_size).crop(aa, tokens, binder_chain_id="C")

    assert len(result.token_array) == crop_size
    assert result.n_binder_tokens == n_c
    assert result.n_target_tokens == 6

    # The 6 surviving target tokens should all come from chain A.
    target_atoms_mask = ~result.binder_atom_mask
    target_chain_ids = set(np.unique(result.atom_array.chain_id[target_atoms_mask]))
    assert target_chain_ids == {"A"}, target_chain_ids


def test_cropper_rejects_oversize_binder():
    from pxdesign_train.data import DesignCropper

    aa, tokens, _, _, n_c = _make_synthetic_complex()
    with pytest.raises(ValueError, match="binder has"):
        DesignCropper(crop_size=n_c - 1).crop(aa, tokens, binder_chain_id="C")


def test_cropper_rejects_huge_binder_fraction():
    from pxdesign_train.data import DesignCropper

    aa, tokens, _, _, n_c = _make_synthetic_complex()
    # Binder is 6 tokens; max_binder_fraction=0.2 means budget needs >= 30.
    with pytest.raises(ValueError, match="more than"):
        DesignCropper(crop_size=20, max_binder_fraction=0.2).crop(
            aa, tokens, binder_chain_id="C",
        )


def test_cropper_via_atom_mask_matches_chain_id():
    """Passing an explicit atom mask should give the same result as chain_id."""
    from pxdesign_train.data import DesignCropper

    aa, tokens, _, _, _ = _make_synthetic_complex()
    mask = aa.chain_id == "C"

    r_id = DesignCropper(crop_size=18).crop(aa, tokens, binder_chain_id="C")
    r_mask = DesignCropper(crop_size=18).crop(aa, tokens, binder_atom_mask=mask)

    assert r_id.n_binder_tokens == r_mask.n_binder_tokens
    assert r_id.n_target_tokens == r_mask.n_target_tokens
    np.testing.assert_array_equal(
        r_id.binder_atom_mask, r_mask.binder_atom_mask,
    )


def test_slice_feature_dict_crops_pair_tensor_on_both_axes():
    """Regression: a [N_token, N_token] pair tensor must be cropped on BOTH
    token axes, not just axis 0. The old length-dispatch matched axis 0 first
    and left axis 1 uncropped -> [crop, N_token_orig] size mismatch downstream.
    """
    from pxdesign_train.data import DesignCropper
    from pxdesign_train.runner.data import _slice_feature_dict

    aa, tokens, n_a, n_b, n_c = _make_synthetic_complex()
    n_token_orig = len(tokens)          # 30
    n_atom_orig = len(aa)               # 120
    crop_size = n_c + n_a               # 18 (real crop: 30 -> 18)
    crop = DesignCropper(crop_size=crop_size).crop(aa, tokens, binder_chain_id="C")
    n_token_new = len(crop.token_array)
    n_atom_new = len(crop.atom_array)

    feat = {
        # [N_token, N_token] pair tensor (the bug target)
        "z_pair": torch.arange(n_token_orig * n_token_orig).float().reshape(
            n_token_orig, n_token_orig
        ),
        # per-token tensor (must still crop on axis 0)
        "restype": torch.zeros(n_token_orig, 32),
        # msa [depth, N_token] (must still crop on axis 1 only)
        "msa": torch.ones(1, n_token_orig),
        # per-atom tensor
        "rep": torch.zeros(n_atom_orig, dtype=torch.long),
    }
    sliced = _slice_feature_dict(feat, aa, tokens, crop)
    assert sliced["z_pair"].shape == (n_token_new, n_token_new)
    assert sliced["restype"].shape == (n_token_new, 32)
    assert sliced["msa"].shape == (1, n_token_new)
    assert sliced["rep"].shape == (n_atom_new,)


def test_cropper_then_featurizer_end_to_end():
    """Cropped output should feed cleanly into DesignFeaturizer."""
    from pxdesign_train.data import DesignCropper, DesignFeaturizer, DesignSelection

    aa, tokens, n_a, n_b, n_c = _make_synthetic_complex()

    cropped = DesignCropper(crop_size=n_c + 6).crop(aa, tokens, binder_chain_id="C")

    # The featurizer expects a Protenix-style feature_dict with at least
    # distogram_rep_atom_mask and the sequence-side keys it may mask.
    n_atom_post = len(cropped.atom_array)
    n_token_post = len(cropped.token_array)
    feature_dict = {
        "distogram_rep_atom_mask": torch.from_numpy(
            cropped.atom_array.distogram_rep_atom_mask.astype(np.int64),
        ).long(),
        "restype": torch.zeros(n_token_post, 32),  # widened by featurizer
        "deletion_mean": torch.ones(n_token_post),
        "profile": torch.ones(n_token_post, 32),
        "msa": torch.ones(1, n_token_post),
    }
    label_dict = {
        "coordinate": torch.from_numpy(cropped.atom_array.coord),
        "coordinate_mask": torch.ones(n_atom_post, dtype=torch.long),
    }
    selection = DesignSelection(
        binder_atom_mask=cropped.binder_atom_mask,
        hotspot_force_zero_prob=0.0,
        rng=np.random.default_rng(0),
    )
    new_feat, _, _ = DesignFeaturizer(selection).transform(
        cropped.atom_array, feature_dict, label_dict,
    )

    # Sanity: feature shapes match the cropped sizes, binder restype hits xpb.
    assert new_feat["restype"].shape == (n_token_post, 36)
    assert new_feat["design_token_mask"].sum().item() == cropped.n_binder_tokens
    assert new_feat["conditional_templ"].shape == (n_token_post, n_token_post)
