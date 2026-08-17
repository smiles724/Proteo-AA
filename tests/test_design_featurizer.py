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


def test_native_aa_override_is_label_only_when_design_tokens_are_masked(
    synthetic_complex,
):
    """Strict pre-featurization canonicalises binder residues to GLY.

    The native labels must remain available for the loss without replacing xpb
    in the model input.
    """
    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    aa, feat, lbl, n_a, n_b = synthetic_complex
    native = torch.tensor([7] * n_a + [0, 1, 2, 3], dtype=torch.long)
    selection = DesignSelection(
        binder_chain_id="B",
        aa_mask_mode="all",
        aa_clean_override=native,
        rng=np.random.default_rng(0),
    )
    new_feat, _, _ = DesignFeaturizer(selection).transform(aa, feat, lbl)
    design = new_feat["design_token_mask"].bool()
    assert torch.equal(new_feat["aa_clean"], native)
    assert torch.all(new_feat["restype"][design].argmax(dim=-1) == 32)


def test_design_tokens_without_a_usable_label_still_show_xpb(synthetic_complex):
    """A design residue whose native type has no 20-AA label must still be xpb.

    `_sample_aa_corruption_mask` only draws from tokens carrying a valid label,
    so an UNK / unmapped design residue is never corrupted. Before the fix it
    fell through to `clean_restype` and the model was shown a concrete residue
    at a position that is always xpb at inference.
    """
    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    aa, feat, lbl, n_a, n_b = synthetic_complex
    # Second binder residue has no usable label; the rest are ordinary.
    native = torch.tensor([7] * n_a + [0, -100, 2, 3], dtype=torch.long)
    selection = DesignSelection(
        binder_chain_id="B",
        aa_mask_mode="all",
        aa_clean_override=native,
        rng=np.random.default_rng(0),
    )
    new_feat, _, _ = DesignFeaturizer(selection).transform(aa, feat, lbl)

    design = new_feat["design_token_mask"].bool()
    # EVERY design token is xpb in the model input, labelled or not.
    assert torch.all(new_feat["restype"][design].argmax(dim=-1) == 32)
    # ...but the unlabelled one is still excluded from the AA loss.
    assert new_feat["aa_loss_mask"].bool()[n_a + 1].item() is False
    assert new_feat["aa_loss_mask"].bool()[n_a].item() is True
    # Target tokens are untouched.
    assert torch.all(new_feat["restype"][~design].argmax(dim=-1) == 7)


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


def _ser_complex(offset_from_origin: float, unresolved_slots=()):
    """Two SER residues (N, CA, C, O, CB, OG) placed `offset` A from the origin.

    `unresolved_slots` marks (res_index, atom_name) pairs the way a real mmCIF
    does: the atom row EXISTS but its coordinate is the (0,0,0) placeholder and
    `is_resolved` is False.
    """
    biotite = pytest.importorskip("biotite.structure")
    pytest.importorskip("protenix")

    names = ("N", "CA", "C", "O", "CB", "OG")
    n_res = 2
    n_atom = n_res * len(names)
    aa = biotite.AtomArray(length=n_atom)
    aa.coord = np.zeros((n_atom, 3), dtype=np.float32)
    resolved = np.ones(n_atom, dtype=bool)

    i = 0
    for r in range(n_res):
        for nm in names:
            aa.chain_id[i] = "B"
            aa.res_id[i] = r + 1
            aa.res_name[i] = "SER"
            aa.atom_name[i] = nm
            aa.element[i] = nm[0]
            if (r, nm) in unresolved_slots:
                aa.coord[i] = (0.0, 0.0, 0.0)     # placeholder
                resolved[i] = False
            else:
                # A plausible little residue, translated far from the origin.
                local = {
                    "N": (-1.2, 0.0, 0.0), "CA": (0.0, 0.0, 0.0), "C": (1.2, 0.0, 0.0),
                    "O": (1.5, 1.0, 0.0), "CB": (0.0, 1.3, 0.6), "OG": (0.6, 2.2, 1.1),
                }[nm]
                aa.coord[i] = (
                    local[0] + offset_from_origin + 4.0 * r,
                    local[1] + offset_from_origin,
                    local[2] + offset_from_origin,
                )
            i += 1

    is_ca = aa.atom_name == "CA"
    aa.set_annotation("distogram_rep_atom_mask", is_ca.astype(int))
    aa.set_annotation("is_resolved", resolved)
    aa.set_annotation("mol_type", np.array(["protein"] * n_atom))
    feature_dict = {
        "distogram_rep_atom_mask": torch.from_numpy(is_ca.astype(np.int64)).long(),
        "restype": torch.zeros((n_res, 32)),
    }
    return aa, feature_dict, np.ones(n_atom, dtype=bool)


def _sc_targets(aa, feature_dict, binder):
    from pxdesign_train.data.featurizer import DesignFeaturizer, DesignSelection

    return DesignFeaturizer(
        DesignSelection(binder_atom_mask=binder)
    )._compute_sidechain_targets(aa, feature_dict, binder)


def test_unresolved_sidechain_atoms_are_not_supervised():
    """An atom with no real coordinate must not become a training target.

    Unresolved atoms keep their row but carry the (0,0,0) placeholder. Masking
    them in makes the loss demand the atom be placed at the coordinate ORIGIN,
    which is wherever the assembly happens to be centred — hundreds of Angstrom
    away. Measured on the 491-protein validation set before this guard: 9.84% of
    all supervised side-chain atoms across 461/491 structures were such targets,
    which flat-lined Stage II-A training (7us9 alone read 21860 A^2).
    """
    aa, feat, binder = _ser_complex(400.0, unresolved_slots={(0, "OG")})
    out = _sc_targets(aa, feat, binder)
    mask = out["sc_atom_mask"]

    # SER slots are [CB, OG]; residue 0's OG is unresolved and must be masked OUT.
    assert bool(mask[0, 0]) is True, "resolved CB must stay supervised"
    assert bool(mask[0, 1]) is False, "unresolved OG must NOT be supervised"
    assert bool(mask[1, 0]) is True and bool(mask[1, 1]) is True, "residue 1 is fully resolved"

    # The slot IDENTITY is type-derived and must survive (it names the slot).
    assert int(out["sc_atom_name_ids"][0, 1]) != 0, "atom-name id must not be cleared"


def test_no_supervised_target_sits_at_the_coordinate_origin():
    """The invariant that actually matters, checked in global space.

    Guarding the mask alone is not enough: what broke training was a supervised
    target whose GLOBAL position was (0,0,0). Reconstruct every supervised target
    and assert none of them lands on the origin, and that all are within side-chain
    reach of their own residue frame.
    """
    from pxdesign_train.sidechain.frames import to_global

    for offset in (37.0, 400.0):     # 7lmv-like and 7us9-like placements
        aa, feat, binder = _ser_complex(offset, unresolved_slots={(0, "OG"), (1, "CB")})
        out = _sc_targets(aa, feat, binder)
        gt, mask = out["sc_gt_local"].float(), out["sc_atom_mask"].bool()
        R, t = out["sc_frame_R"].float(), out["sc_frame_t"].float()

        assert mask.any(), "some atoms should still be supervised"
        glob = to_global(gt, R, t)[mask]
        assert (glob.abs().amax(dim=-1) > 1.0).all(), (
            f"offset={offset}: a supervised target sits at the coordinate origin"
        )
        # And every supervised target must be physically reachable from its frame.
        assert (gt[mask].norm(dim=-1) < 10.0).all(), (
            f"offset={offset}: supervised target further than 10 A from its own frame"
        )


def test_missing_is_resolved_annotation_still_rejects_placeholder_coords():
    """Fallback path: arrays without `is_resolved` must not regress to the bug."""
    aa, feat, binder = _ser_complex(400.0, unresolved_slots={(0, "OG")})
    aa.del_annotation("is_resolved")
    out = _sc_targets(aa, feat, binder)
    assert bool(out["sc_atom_mask"][0, 1]) is False, (
        "zero-coordinate atom must be rejected even with no is_resolved annotation"
    )


@pytest.mark.parametrize("frame_atom", ["N", "CA", "C"])
def test_residue_with_an_unresolved_frame_atom_is_excluded(frame_atom):
    """A residue whose N/CA/C was never measured has no usable local frame.

    Same root cause as the side-chain case above -- name presence was checked,
    `is_resolved` was not -- but it does NOT show up as a masked atom, because the
    side-chain atoms themselves may be perfectly well resolved. What is broken is
    the frame they hang off: `t` is CA and the basis comes from N/CA/C, so a
    (0,0,0) placeholder among those three puts the whole residue's local geometry
    in a frame belonging to nothing.

    Stage II hides this completely. The coordinate loss rebuilds the target with
    the SAME frame it was stored under, and `to_global(to_local(x, R, t), R, t)`
    is exactly x for any orthonormal R -- the bad frame cancels. Stage III does
    not cancel: there the target is rebuilt through the PREDICTED frame, so the
    round trip no longer closes and `gt_local` acts as a lever arm. With a
    placeholder CA it grows from ~3 A to |x_true| (hundreds of A), which
    multiplies the backbone's own angular error by two orders of magnitude.

    Excluding the residue is the same treatment a missing atom NAME already gets,
    so this is a widened guard rather than new behaviour.
    """
    aa, feat, binder = _ser_complex(400.0, unresolved_slots={(0, frame_atom)})
    out = _sc_targets(aa, feat, binder)
    mask = out["sc_atom_mask"].bool()

    assert not bool(mask[0].any()), (
        f"residue 0's {frame_atom} is unresolved, so it has no usable frame and "
        "must not be supervised at all"
    )
    # SER owns exactly two side-chain slots (CB, OG); the rest of MAX_SC is padding.
    assert bool(mask[1, :2].all()), "residue 1 is fully resolved and must be untouched"


def test_no_supervised_local_target_becomes_a_lever_arm():
    """`gt_local` must stay within side-chain reach for every supervised slot.

    States the invariant without naming which atom went missing: a local
    coordinate of hundreds of Angstrom means the frame origin is not on this
    residue, and in Stage III that magnitude multiplies the predicted backbone's
    angular error.
    """
    aa, feat, binder = _ser_complex(400.0, unresolved_slots={(0, "CA")})
    out = _sc_targets(aa, feat, binder)
    gt, mask = out["sc_gt_local"].float(), out["sc_atom_mask"].bool()

    assert mask.any(), "residue 1 is intact and should still be supervised"
    assert (gt[mask].norm(dim=-1) < 10.0).all(), (
        "a supervised target sits further than 10 A from its own frame origin -- "
        "the frame was built from a placeholder coordinate"
    )


def test_an_unresolved_backbone_O_takes_the_absent_atom_path():
    """An unresolved O must be marked ABSENT (index -1), not merely zero-valued.

    The 14-slot axis carries four backbone slots (N, CA, C, O) as context, and
    every consumer decides their validity from `sc_bb_atom_idx >= 0` -- the atom
    INDEX, which exists whenever the atom name does. An unresolved-but-named O
    therefore passed as valid while carrying the (0,0,0) placeholder, and the code
    comment describing the bounded fallback ("an absent atom sits at the frame
    origin") already conflated the two cases.

    The two stages are harmed differently, and neither through a loss target:

      * Stage II reads sc_bb_coords directly. Those are the raw deposited
        coordinates, never augmented, so the slot is the literal (0,0,0) and
        to_local puts it at -R^T.t -- the full distance from the deposition
        origin to this residue, hundreds of Angstrom for a 7us9-like placement.
        Its w_xyz activation is enormous.
      * Stage III gathers x_denoised, which is NOT the same failure.
        centre_random_augmentation re-applies coordinate_mask after the rotation
        and translation (Protenix utils.py:91-93), so the row is zeroed in the
        augmented frame and x_noisy is pure sigma-scale noise sitting near the
        centred protein -- bounded, not a large excursion. But coordinate_mask
        also excludes it from the coordinate loss, so nothing ever supervises it:
        the slot carries an unconstrained noise coordinate.

    The intra-residue block has no distance bias to suppress either of them (the
    cross-residue block does, by construction), so both mix into all fourteen of
    that residue's atom features.

    Marking it absent routes it to the fallback the comment already promised:
    v4 zeroes it, bb_local becomes 0 == the frame origin == CA. It also keeps
    q_bs's gather and the physical loss's bb_valid consistent, since both key off
    the same index.

    After the N/CA/C guard this can only ever be O -- a residue missing any frame
    atom is dropped whole.
    """
    aa, feat, binder = _ser_complex(400.0, unresolved_slots={(0, "O")})
    out = _sc_targets(aa, feat, binder)
    idx = out["sc_bb_atom_idx"]          # [L, 4] = (N, CA, C, O)

    assert int(idx[0, 3]) == -1, (
        "an unresolved O is still indexed, so every consumer keying off "
        "`sc_bb_atom_idx >= 0` treats a placeholder coordinate as real"
    )
    assert float(out["sc_bb_coords"][0, 3].abs().max()) == 0.0

    # The residue is otherwise intact: its frame atoms are resolved, so it keeps
    # its frame and its side chain stays supervised. Only the O slot drops out.
    assert (idx[0, :3] >= 0).all(), "resolved frame atoms must survive"
    assert bool(out["sc_atom_mask"][0, :2].all()), "SER's CB/OG stay supervised"
    assert int(idx[1, 3]) >= 0, "residue 1 is fully resolved and must be untouched"
