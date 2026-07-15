"""`sc_bb_atom_idx`: per-token flat atom indices of (N, CA, C, O).

These indices are the bridge between the token axis and the N_atom axis that
Protenix's per-atom tensors (coordinates, and the AtomAttentionEncoder's `q`
features) live on. Two consumers rely on them:

  * frames: `frames_from_backbone_index(x_denoised, bb_idx)` -> F_hat  (cols 0:3)
  * atom-level side-chain -> backbone feedback: gather/scatter q_bb  (all 4 cols)

The failure mode is silent: a wrong index gathers a real coordinate belonging to
some *other* atom, and nothing raises. So every test here re-derives the answer
BY NAME from the AtomArray and compares against the gather.
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

BB = ("N", "CA", "C", "O")


def _residue(res_name, extra_atoms):
    """(atom_names, coords) for one residue: backbone + `extra_atoms` side chain."""
    names = list(BB) + list(extra_atoms)
    return res_name, names


# A deliberately heterogeneous chain:
#  - GLY  : backbone only (no side chain)
#  - ALA  : one side-chain atom
#  - SER  : two
#  - a residue whose O is UNRESOLVED (dropped from the AtomArray) -> col 3 == -1
_SPEC = [
    ("ALA", ["N", "CA", "C", "O", "CB"]),
    ("GLY", ["N", "CA", "C", "O"]),
    ("SER", ["N", "CA", "C", "O", "CB", "OG"]),
    ("ALA", ["N", "CA", "C", "CB"]),           # <-- no O
    ("SER", ["N", "CA", "C", "O", "CB", "OG"]),
]


def _make_chain(chain_id="C", res_offset=0, x_offset=0.0, y=0.0, spec=None):
    """Build an AtomArray + TokenArray for a chain, with scrambled intra-residue
    atom order so that nothing works by accident from positional assumptions."""
    biotite = pytest.importorskip("biotite.structure")
    pytest.importorskip("protenix")
    from protenix.data.tokenizer import Token, TokenArray

    spec = spec if spec is not None else _SPEC
    rng = np.random.default_rng(0)

    names, res_names, res_ids, coords = [], [], [], []
    tokens_atom_idx, centre_idx = [], []
    cursor = 0
    for r, (rn, atom_names) in enumerate(spec):
        an = list(atom_names)
        # Scramble atom order within the residue: the featurizer must resolve by
        # NAME, never by position. (Keeps CA somewhere in the middle sometimes.)
        rng.shuffle(an)
        idxs = []
        cax = x_offset + r * 3.8
        for a in an:
            # A distinct, deterministic coordinate per (residue, atom name).
            jitter = (hash(a) % 7) * 0.1
            coords.append([cax + jitter, y + 0.37 * len(a), 0.11 * (r + 1)])
            names.append(a)
            res_names.append(rn)
            res_ids.append(res_offset + r + 1)
            idxs.append(cursor)
            if a == "CA":
                centre_idx.append(cursor)
            cursor += 1
        tokens_atom_idx.append(idxs)

    n_atom = cursor
    aa = biotite.AtomArray(length=n_atom)
    aa.coord = np.asarray(coords, dtype=np.float32)
    aa.chain_id = np.array([chain_id] * n_atom)
    aa.res_id = np.array(res_ids)
    aa.res_name = np.array(res_names)
    aa.atom_name = np.array(names)
    aa.element = np.array([n[0] for n in names])
    rep = np.zeros(n_atom, dtype=int)
    rep[np.asarray(centre_idx)] = 1
    aa.set_annotation("distogram_rep_atom_mask", rep)
    aa.set_annotation("is_resolved", np.ones(n_atom, dtype=bool))
    aa.set_annotation("mol_type", np.array(["protein"] * n_atom))
    aa.set_annotation("asym_id_int", np.zeros(n_atom, dtype=np.int64))

    tokens = [
        Token(i, atom_indices=idxs, centre_atom_index=centre_idx[i])
        for i, idxs in enumerate(tokens_atom_idx)
    ]
    return aa, TokenArray(tokens)


def _by_name_lookup(atom_array):
    """{(chain, res_id): {atom_name: flat_atom_index}} -- the independent oracle."""
    out = {}
    for i in range(len(atom_array)):
        key = (str(atom_array.chain_id[i]), int(atom_array.res_id[i]))
        out.setdefault(key, {})[str(atom_array.atom_name[i])] = i
    return out


def _token_keys(atom_array):
    rep = atom_array.distogram_rep_atom_mask.astype(bool)
    return [
        (str(c), int(r))
        for c, r in zip(atom_array.chain_id[rep], atom_array.res_id[rep])
    ]


def _sidechain_targets(atom_array, binder_mask):
    from pxdesign_train.data import DesignFeaturizer, DesignSelection

    feat = {
        "distogram_rep_atom_mask": torch.from_numpy(
            atom_array.distogram_rep_atom_mask.astype(np.int64)
        ),
    }
    fz = DesignFeaturizer(
        DesignSelection(binder_atom_mask=binder_mask, compute_sidechain=True)
    )
    return fz._compute_sidechain_targets(atom_array, feat, binder_mask)


# --------------------------------------------------------------------------
# 1. shape + the -1 sentinel
# --------------------------------------------------------------------------
def test_sc_bb_atom_idx_is_four_wide_with_minus_one_exactly_where_missing():
    aa, _ = _make_chain()
    binder = np.ones(len(aa), dtype=bool)
    out = _sidechain_targets(aa, binder)

    idx = out["sc_bb_atom_idx"]
    n_token = int(aa.distogram_rep_atom_mask.sum())
    assert idx.shape == (n_token, 4), idx.shape
    assert idx.dtype == torch.int64

    lookup = _by_name_lookup(aa)
    keys = _token_keys(aa)
    for ti, key in enumerate(keys):
        for bi, name in enumerate(BB):
            present = name in lookup[key]
            got = int(idx[ti, bi])
            assert (got >= 0) == present, (
                f"token {ti} ({key}) atom {name}: present={present} but idx={got}"
            )

    # The one residue with no O must be -1 in col 3 and VALID in cols 0:3 --
    # a `.all(-1)` validity test over four columns would wrongly kill its frame.
    no_o = [ti for ti, k in enumerate(keys) if "O" not in lookup[k]]
    assert len(no_o) == 1, no_o
    ti = no_o[0]
    assert int(idx[ti, 3]) == -1
    assert bool((idx[ti, :3] >= 0).all())


# --------------------------------------------------------------------------
# 2. the indices really point at N/CA/C/O -- gather vs. the by-name coords
# --------------------------------------------------------------------------
def test_gather_at_indices_reproduces_sc_bb_coords():
    """sc_bb_coords is built BY NAME; sc_bb_atom_idx is an index into N_atom.
    Gathering the atom array at those indices must reproduce sc_bb_coords.
    This is the check that catches an ordering / off-by-one / stale-index bug."""
    aa, _ = _make_chain()
    binder = np.ones(len(aa), dtype=bool)
    out = _sidechain_targets(aa, binder)

    idx = out["sc_bb_atom_idx"]           # [L, 4]
    ref = out["sc_bb_coords"]             # [L, 4, 3], by-name
    coords = torch.from_numpy(np.asarray(aa.coord, dtype=np.float32))  # [N_atom, 3]

    gathered = coords[idx.clamp_min(0)]   # [L, 4, 3]
    valid = idx >= 0                      # [L, 4]

    assert torch.allclose(gathered[valid], ref[valid], atol=0)
    # Missing atoms: sc_bb_coords is zero there (and the gather is meaningless).
    assert torch.all(ref[~valid] == 0)

    # And the indices agree with a fully independent by-name resolution.
    lookup = _by_name_lookup(aa)
    for ti, key in enumerate(_token_keys(aa)):
        for bi, name in enumerate(BB):
            if name in lookup[key]:
                assert int(idx[ti, bi]) == lookup[key][name], (ti, name)


def test_frames_from_backbone_index_accepts_four_wide():
    """The existing frame consumer must keep working on a 4-wide tensor: it uses
    cols 0:3 and must NOT be fooled by a -1 in the O column."""
    from pxdesign_train.sidechain.frames import build_frame, frames_from_backbone_index

    aa, _ = _make_chain()
    binder = np.ones(len(aa), dtype=bool)
    out = _sidechain_targets(aa, binder)
    idx = out["sc_bb_atom_idx"]
    coords = torch.from_numpy(np.asarray(aa.coord, dtype=np.float32))

    R, t, valid = frames_from_backbone_index(coords, idx)
    # Every token here has N/CA/C -> every frame is valid, including the no-O one.
    assert bool(valid.all()), valid

    R3, t3, valid3 = frames_from_backbone_index(coords, idx[:, :3])
    assert torch.equal(R, R3) and torch.equal(t, t3) and torch.equal(valid, valid3)

    # Frames match a direct build from the by-name N/CA/C coords.
    Rd, td = build_frame(
        out["sc_bb_coords"][:, 0], out["sc_bb_coords"][:, 1], out["sc_bb_coords"][:, 2]
    )
    assert torch.allclose(R, Rd, atol=1e-5)
    assert torch.allclose(t, td, atol=1e-5)
    # ... and the frames the featurizer stored.
    assert torch.allclose(R, out["sc_frame_R"], atol=1e-5)
    assert torch.allclose(t, out["sc_frame_t"], atol=1e-5)


def test_non_binder_tokens_are_all_minus_one():
    aa, _ = _make_chain()
    binder = np.zeros(len(aa), dtype=bool)
    binder[aa.res_id >= 4] = True     # only the last two residues are binder
    out = _sidechain_targets(aa, binder)
    idx = out["sc_bb_atom_idx"]

    keys = _token_keys(aa)
    for ti, (_, rid) in enumerate(keys):
        if rid >= 4:
            assert bool((idx[ti, :3] >= 0).all()), ti
        else:
            assert bool((idx[ti] == -1).all()), ti


# --------------------------------------------------------------------------
# 3. THE CROP. Both orders: featurize-after-crop (the training path) and
#    featurize-then-crop (the remap path in _slice_feature_dict).
# --------------------------------------------------------------------------
def _two_chain_complex():
    """Binder chain C (close) + target chain A (close) + target chain B (far)."""
    a, ta = _make_chain(chain_id="A", x_offset=0.0, y=0.0)
    b, tb = _make_chain(chain_id="B", x_offset=0.0, y=40.0)
    c, tc = _make_chain(chain_id="C", x_offset=0.0, y=5.0)

    biotite = pytest.importorskip("biotite.structure")
    from protenix.data.tokenizer import Token, TokenArray

    aa = a + b + c  # biotite concatenation preserves annotations
    # asym_id_int must differ per chain for Protenix's CropData.
    asym = np.concatenate([
        np.zeros(len(a), dtype=np.int64),
        np.ones(len(b), dtype=np.int64),
        np.full(len(c), 2, dtype=np.int64),
    ])
    aa.set_annotation("asym_id_int", asym)

    tokens, offset = [], 0
    tix = 0
    for sub, sub_tokens in ((a, ta), (b, tb), (c, tc)):
        for tok in sub_tokens:
            tokens.append(
                Token(
                    tix,
                    atom_indices=[int(i) + offset for i in tok.atom_indices],
                    centre_atom_index=int(tok.centre_atom_index) + offset,
                )
            )
            tix += 1
        offset += len(sub)
    return aa, TokenArray(tokens)


def test_indices_are_correct_after_crop_training_path():
    """The training path crops FIRST and featurizes the cropped AtomArray. The
    indices must point into the CROPPED atom axis."""
    from pxdesign_train.data import DesignCropper

    aa, tokens = _two_chain_complex()
    n_binder = int(sum(1 for t in _token_keys(aa) if t[0] == "C"))
    crop = DesignCropper(
        crop_size=n_binder + 5, max_binder_fraction=0.9
    ).crop(aa, tokens, binder_chain_id="C")

    # Chain B (far) must have been dropped -- otherwise the test proves nothing.
    kept_chains = set(np.unique(crop.atom_array.chain_id))
    assert "B" not in kept_chains, kept_chains
    assert len(crop.atom_array) < len(aa)

    out = _sidechain_targets(crop.atom_array, crop.binder_atom_mask)
    idx = out["sc_bb_atom_idx"]

    n_atom_new = len(crop.atom_array)
    assert int(idx.max()) < n_atom_new, "index points past the cropped atom axis"

    lookup = _by_name_lookup(crop.atom_array)      # oracle: BY NAME, cropped array
    keys = _token_keys(crop.atom_array)
    binder_tok = crop.binder_token_mask.astype(bool)
    coords = torch.from_numpy(np.asarray(crop.atom_array.coord, dtype=np.float32))

    checked = 0
    for ti, key in enumerate(keys):
        if not binder_tok[ti]:
            assert bool((idx[ti] == -1).all())
            continue
        for bi, name in enumerate(BB):
            if name not in lookup[key]:
                assert int(idx[ti, bi]) == -1
                continue
            got = int(idx[ti, bi])
            assert got == lookup[key][name], (ti, key, name, got)
            # The gathered coordinate is the right atom's coordinate.
            assert torch.allclose(coords[got], out["sc_bb_coords"][ti, bi], atol=0)
            checked += 1
    assert checked > 0
    assert int(binder_tok.sum()) == n_binder


def test_indices_are_remapped_when_feature_dict_is_cropped():
    """The other order: a feature dict that ALREADY carries sc_bb_atom_idx (built
    on the uncropped array) goes through `_slice_feature_dict`. Its VALUES must be
    remapped into the new atom numbering -- slicing rows alone leaves indices
    pointing into the uncropped atom axis, which gathers the wrong atom silently."""
    from pxdesign_train.data import DesignCropper
    from pxdesign_train.runner.data import _slice_feature_dict

    aa, tokens = _two_chain_complex()
    binder = aa.chain_id == "C"
    pre = _sidechain_targets(aa, binder)          # indices into the UNCROPPED axis

    n_binder = int(binder[aa.distogram_rep_atom_mask.astype(bool)].sum())
    crop = DesignCropper(
        crop_size=n_binder + 5, max_binder_fraction=0.9
    ).crop(aa, tokens, binder_chain_id="C")
    assert len(crop.atom_array) < len(aa)

    feat = {
        "sc_bb_atom_idx": pre["sc_bb_atom_idx"],
        "distogram_rep_atom_mask": torch.from_numpy(
            aa.distogram_rep_atom_mask.astype(np.int64)
        ),
    }
    sliced = _slice_feature_dict(feat, aa, tokens, crop)
    idx = sliced["sc_bb_atom_idx"]

    n_token_new = len(crop.token_array)
    n_atom_new = len(crop.atom_array)
    assert idx.shape == (n_token_new, 4)
    assert int(idx.max()) < n_atom_new, "stale (uncropped) atom index survived the crop"

    lookup = _by_name_lookup(crop.atom_array)
    keys = _token_keys(crop.atom_array)
    coords = torch.from_numpy(np.asarray(crop.atom_array.coord, dtype=np.float32))
    binder_tok = crop.binder_token_mask.astype(bool)

    checked = 0
    for ti, key in enumerate(keys):
        if not binder_tok[ti]:
            continue
        for bi, name in enumerate(BB):
            got = int(idx[ti, bi])
            if name not in lookup[key]:
                assert got == -1
                continue
            assert got == lookup[key][name], (ti, key, name, got)
            assert torch.allclose(coords[got], pre["sc_bb_coords"][:, bi][
                _uncropped_token_of(aa, key)
            ], atol=0)
            checked += 1
    assert checked > 0

    # An UNREMAPPED tensor (row-slice only) would have failed: prove the crop
    # really did renumber the atoms, so the test above is not vacuous.
    rows = torch.from_numpy(np.asarray(crop.original_token_indices, dtype=np.int64))
    naive = pre["sc_bb_atom_idx"].index_select(0, rows)
    assert not torch.equal(naive, idx), (
        "crop did not renumber atoms -- pick a crop that actually drops atoms"
    )


def _uncropped_token_of(atom_array, key):
    return _token_keys(atom_array).index(key)
