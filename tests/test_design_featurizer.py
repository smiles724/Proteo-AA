"""
Synthetic-AtomArray test for `DesignFeaturizer`.

We build a small fake complex by hand (no PDB I/O, no Protenix parser) just
enough to satisfy the annotations `DesignFeaturizer` reads:
  - res_name, chain_id, atom_name, coord
  - distogram_rep_atom_mask, is_resolved, mol_type
  - the matching per-token feature dict produced by Protenix's featurizer
    is mocked with just `restype` (we don't need the rest for this test)

The goal is to verify shapes, dtypes, and a few correctness invariants:
  - the binder chain's restype lands on the xpb one-hot slot (index 32)
  - conditional_templ is zero on design-token pairs and non-zero on target pairs
  - the design_token_mask aligns with the binder chain selection
  - hotspot is zero on design tokens
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


@pytest.fixture
def synthetic_complex():
    """Build a tiny 2-chain protein complex:
      - chain A: 6 residues of GLY (target) — backbone-only (N, CA, C, O)
      - chain B: 4 residues of GLY (binder) — backbone-only

    Place chain B close enough to chain A that ~half of A's residues are
    within 8 Å of B. That gives the hotspot sampler something to find.
    """
    biotite = pytest.importorskip("biotite.structure")
    pytest.importorskip("protenix")  # ensures pxdesign annotation conventions are loadable

    AtomArray = biotite.AtomArray

    n_a, n_b = 6, 4
    backbone_atoms = ("N", "CA", "C", "O")
    atoms_per_res = len(backbone_atoms)
    n_atom = (n_a + n_b) * atoms_per_res

    aa = AtomArray(length=n_atom)
    aa.coord = np.zeros((n_atom, 3), dtype=np.float32)

    def fill_chain(start_atom_idx, chain_id, n_res, base_offset):
        # Lay residues out along x-axis at base_offset y/z
        for r in range(n_res):
            for a_idx, name in enumerate(backbone_atoms):
                i = start_atom_idx + r * atoms_per_res + a_idx
                aa.chain_id[i] = chain_id
                aa.res_id[i] = r + 1
                aa.res_name[i] = "GLY"
                aa.atom_name[i] = name
                aa.element[i] = "N" if name == "N" else ("O" if name == "O" else "C")
                # Cα at (r*3.8, base_offset, 0); N/C/O small offsets
                cax, cay, caz = r * 3.8, base_offset, 0.0
                if name == "N":
                    aa.coord[i] = (cax - 1.0, cay, caz)
                elif name == "CA":
                    aa.coord[i] = (cax, cay, caz)
                elif name == "C":
                    aa.coord[i] = (cax + 1.0, cay, caz)
                else:  # O
                    aa.coord[i] = (cax + 1.2, cay + 1.0, caz)

    fill_chain(0, "A", n_a, base_offset=0.0)
    # Place chain B 6 Å away on the y-axis from chain A → all 4 B residues
    # are within the 8 Å hotspot radius of at least one A residue.
    fill_chain(n_a * atoms_per_res, "B", n_b, base_offset=6.0)

    # Annotations that Protenix + PXDesign expect.
    #   distogram_rep_atom_mask: Cα for protein (1 per residue).
    is_ca = aa.atom_name == "CA"
    aa.set_annotation("distogram_rep_atom_mask", is_ca.astype(int))
    aa.set_annotation("is_resolved", np.ones(n_atom, dtype=bool))
    aa.set_annotation("mol_type", np.array(["protein"] * n_atom))
    # Other annotations consumed by `cano_seq_resname_with_mask` are residue-
    # level via `res_name` — already set above.

    # Build a "Protenix featurizer output" feature_dict minimally: only the
    # bits that DesignFeaturizer reads.
    feature_dict = {
        "distogram_rep_atom_mask": torch.from_numpy(is_ca.astype(np.int64)).long(),
        # A stand-in `restype` of the right shape; DesignFeaturizer will replace
        # it with the 36-channel design one-hot computed from atom_array.
        "restype": torch.zeros((n_a + n_b, 32)),
        # Sequence-side features that the leakage-masker may touch. PXDesign's
        # `json_to_feature.py:353-361` multiplies `msa`/`has_deletion`/
        # `deletion_value` by `condi[None, :]`, so those are `[N_msa, N_token]`
        # (2D); `profile` is `[N_token, c_profile]`; `deletion_mean` is
        # `[N_token]`.
        "deletion_mean": torch.ones(n_a + n_b),
        "profile": torch.ones(n_a + n_b, 32),
        "msa": torch.ones(1, n_a + n_b),
        "has_deletion": torch.ones(1, n_a + n_b),
        "deletion_value": torch.ones(1, n_a + n_b),
    }
    label_dict = {
        "coordinate": torch.from_numpy(aa.coord),
        "coordinate_mask": torch.ones(n_atom, dtype=torch.long),
    }
    return aa, feature_dict, label_dict, n_a, n_b


def test_featurizer_basic_shapes(synthetic_complex):
    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    aa, feat, lbl, n_a, n_b = synthetic_complex
    selection = DesignSelection(
        binder_chain_id="B",
        hotspot_force_zero_prob=0.0,  # always sample hotspots in the test
        rng=np.random.default_rng(42),
    )
    new_feat, new_lbl, new_aa = DesignFeaturizer(selection).transform(aa, feat, lbl)

    n_token = n_a + n_b
    # restype widened to 36-channel design vocabulary.
    assert new_feat["restype"].shape == (n_token, 36)
    # 32 = xpb index. All binder tokens (last n_b) must one-hot at 32.
    binder_one_hot = new_feat["restype"][n_a:].argmax(dim=-1)
    assert torch.all(binder_one_hot == 32), binder_one_hot.tolist()
    # Target tokens stay on GLY (index 7 in PRO_STD_RESIDUES_NATURAL).
    target_one_hot = new_feat["restype"][:n_a].argmax(dim=-1)
    assert torch.all(target_one_hot == 7), target_one_hot.tolist()

    # Token masks.
    assert new_feat["design_token_mask"].sum().item() == n_b
    assert new_feat["condition_token_mask"].sum().item() == n_a
    assert new_feat["design_token_mask"].shape == (n_token,)
    assert new_feat["aa_clean"].shape == (n_token,)
    assert new_feat["aa_loss_mask"].shape == (n_token,)
    assert new_feat["aa_corrupted"].shape == (n_token,)
    assert new_feat["aa_corruption_mask"].shape == (n_token,)
    assert new_feat["aa_t"].shape == ()
    assert new_feat["aa_mask_prob"].shape == ()


def test_featurizer_conditional_templ(synthetic_complex):
    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    aa, feat, lbl, n_a, n_b = synthetic_complex
    selection = DesignSelection(binder_chain_id="B", hotspot_force_zero_prob=0.0,
                                rng=np.random.default_rng(0))
    new_feat, _, _ = DesignFeaturizer(selection).transform(aa, feat, lbl)

    n_token = n_a + n_b
    templ = new_feat["conditional_templ"]
    mask = new_feat["conditional_templ_mask"]

    assert templ.shape == (n_token, n_token)
    assert mask.shape == (n_token, n_token)
    # The mask must be zero on every pair touching a design token.
    design = new_feat["design_token_mask"].bool()
    pair_touch_design = design[:, None] | design[None, :]
    assert (mask[pair_touch_design] == 0).all()
    # And present on at least some target-target pairs.
    target_pair = (~design)[:, None] & (~design)[None, :]
    # Exclude self-pairs from this check (still get bin 0).
    n = n_token
    eye = torch.eye(n).bool()
    off_diag_target = target_pair & ~eye
    assert (mask[off_diag_target] == 1).any()

    # Distance bins on target pairs should be > 0 (the Cα-Cα spacing is 3.8 Å,
    # which sits well above the first bin's 2.0 Å boundary).
    target_bins = templ[off_diag_target]
    assert (target_bins > 0).any()


def test_featurizer_hotspot_only_on_target(synthetic_complex):
    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    aa, feat, lbl, n_a, n_b = synthetic_complex
    # Force a hotspot to be selected by setting max_frac=1.0 and disabling
    # the all-zero short-circuit.
    selection = DesignSelection(
        binder_chain_id="B",
        hotspot_force_zero_prob=0.0,
        hotspot_max_frac=1.0,
        rng=np.random.default_rng(0),
    )
    new_feat, _, _ = DesignFeaturizer(selection).transform(aa, feat, lbl)

    hotspot = new_feat["hotspot"]
    assert hotspot.shape == (n_a + n_b,)
    # No design token should ever be a hotspot.
    design = new_feat["design_token_mask"].bool()
    assert (hotspot[design] == 0).all()
    # At least one target residue (any of A's 6) should fire since chain B is
    # 6 Å from chain A.
    assert hotspot.sum().item() > 0


def test_featurizer_msa_leakage_masked(synthetic_complex):
    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    aa, feat, lbl, n_a, n_b = synthetic_complex
    selection = DesignSelection(binder_chain_id="B", hotspot_force_zero_prob=0.0,
                                rng=np.random.default_rng(0))
    new_feat, _, _ = DesignFeaturizer(selection).transform(aa, feat, lbl)

    # `deletion_mean` was all-ones; after masking, design-token entries are 0.
    design = new_feat["design_token_mask"].bool()
    assert (new_feat["deletion_mean"][design] == 0).all()
    assert (new_feat["deletion_mean"][~design] == 1).all()
    # profile is [N_token, 32] and masked along the token axis.
    assert (new_feat["profile"][design] == 0).all()
    assert (new_feat["profile"][~design] == 1).all()


def test_featurizer_preserves_clean_aa_targets_without_leakage(synthetic_complex):
    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    aa, feat, lbl, n_a, n_b = synthetic_complex
    selection = DesignSelection(
        binder_chain_id="B",
        hotspot_force_zero_prob=0.0,
        rng=np.random.default_rng(0),
    )
    new_feat, _, _ = DesignFeaturizer(selection).transform(aa, feat, lbl)

    design = new_feat["design_token_mask"].bool()
    # Synthetic binder residues are GLY, index 7 in the 20-AA vocabulary.
    assert torch.all(new_feat["aa_clean"][design] == 7)
    assert torch.all(new_feat["aa_loss_mask"][design] == 1)
    assert torch.all(new_feat["aa_loss_mask"][~design] == 0)
    assert torch.all(new_feat["aa_corruption_mask"][design] == 1)
    assert torch.all(new_feat["aa_corruption_mask"][~design] == 0)

    # Model input still sees xpb for binder/design residues, not clean GLY.
    assert torch.all(new_feat["restype"][design].argmax(dim=-1) == 32)


def test_featurizer_partial_aa_corruption_masks_only_selected_design_tokens(synthetic_complex):
    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    aa, feat, lbl, n_a, n_b = synthetic_complex
    selection = DesignSelection(
        binder_chain_id="B",
        hotspot_force_zero_prob=0.0,
        aa_mask_mode="fixed",
        aa_mask_prob=0.5,
        rng=np.random.default_rng(1),
    )
    new_feat, _, _ = DesignFeaturizer(selection).transform(aa, feat, lbl)

    design = new_feat["design_token_mask"].bool()
    corrupt = new_feat["aa_corruption_mask"].bool()
    uncorrupt_design = design & ~corrupt

    assert torch.all(corrupt <= design)
    assert 0 < corrupt.sum().item() < n_b
    assert torch.equal(new_feat["aa_loss_mask"].bool(), corrupt)
    assert torch.isclose(new_feat["aa_t"], torch.tensor(0.5))
    assert torch.isclose(new_feat["aa_mask_prob"], torch.tensor(0.5))

    # Corrupted design tokens are xpb; uncorrupted design tokens may condition on
    # their clean AA. Synthetic residues are all GLY, index 7.
    restype_idx = new_feat["restype"].argmax(dim=-1)
    assert torch.all(restype_idx[corrupt] == 32)
    assert torch.all(restype_idx[uncorrupt_design] == 7)
    assert torch.all(restype_idx[~design] == 7)
