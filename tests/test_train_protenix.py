"""The Protenix mmCIF source and its per-residue supervision masks.

Two things here can fail silently and ruin a fine-tune, so both are pinned:

* **Alignment.** The masks are positional. If the residue iteration diverges from
  the builder's by even one residue, every label shifts and training still looks
  healthy.
* **Composition.** The per-residue veto must *add* to FaMPNN's per-atom mask, not
  replace it, and must never reach a backbone slot.
"""

import pytest
import torch

from pxf.train import protenix as P

pytest.importorskip("gemmi")

# Two entries with different properties: 1ubq has partial-occupancy residues,
# 101m's only unsupervised residues are glycines (nothing to veto).
TARGETS = ("1ubq", "101m")

needs_masks = pytest.mark.skipif(
    not (P.DEFAULT_MASK_ROOT / "entries.csv").is_file(),
    reason=f"no mask set at {P.DEFAULT_MASK_ROOT}",
)
needs_mmcif = pytest.mark.skipif(
    not P.DEFAULT_MMCIF_DIR.is_dir(), reason=f"no mmCIF at {P.DEFAULT_MMCIF_DIR}"
)


@pytest.fixture(scope="module")
def masks():
    return P.SideChainMaskSet()


@needs_masks
def test_the_keep_column_is_the_fampnn_one(masks):
    """The shipped loader auto-detects and picks `keep`; this variant's is not that.

    They agree in count today, which is exactly why naming it explicitly matters:
    a regenerated variant could diverge without any visible symptom.
    """
    assert masks.keep_column == "keep_fampnn" == P.KEEP_COLUMN
    assert len(masks) == masks.meta["fampnn_variant"]["entries_kept"]


@needs_masks
def test_a_variant_without_the_column_is_refused(tmp_path):
    (tmp_path / "meta.json").write_text("{}")
    (tmp_path / "entries.csv").write_text("pdb_id,error,keep,mask_len,mask_offset\n")
    (tmp_path / "residue_masks.u16").write_bytes(b"")
    with pytest.raises(ValueError, match="keep_fampnn"):
        P.SideChainMaskSet(tmp_path)


@needs_masks
@needs_mmcif
@pytest.mark.parametrize("pdb_id", TARGETS)
def test_the_parsed_residue_count_matches_the_mask(masks, pdb_id):
    """The alignment contract, checked against the builder's own record."""
    entry = P.read_entry(pdb_id, masks=masks)
    assert len(entry) == masks.lengths[pdb_id] == entry.supervise.shape[0]


@needs_masks
@needs_mmcif
def test_a_length_mismatch_is_an_error_not_a_shift(masks, monkeypatch):
    """Rather than silently mislabelling every residue."""
    monkeypatch.setitem(masks.lengths, "1ubq", 75)
    with pytest.raises(ValueError, match="positional"):
        P.read_entry("1ubq", masks=masks)


@needs_masks
@needs_mmcif
def test_reading_without_a_mask_set_is_allowed(masks):
    entry = P.read_entry("1ubq")
    assert entry.supervise is None and len(entry) == 76


@needs_masks
@needs_mmcif
@pytest.mark.parametrize("pdb_id", TARGETS)
def test_the_veto_composes_and_never_clears(masks, pdb_id):
    from fampnn.data import residue_constants as rc

    entry = P.read_entry(pdb_id, masks=masks)
    base = P.missing_atom_mask_from_presence(entry.aatype, entry.atom_mask)
    vetoed = P.veto_unsupervised_sidechains(base, entry.supervise, aatype=entry.aatype)

    # Composition: never revive an atom the per-atom mask called missing.
    assert bool((vetoed >= base - 1e-6).all())
    # Backbone untouched: the residue keeps its frame and its L_MLM label.
    backbone = list(rc.bb_idxs)
    assert torch.equal(vetoed[:, backbone], base[:, backbone])
    # Ghost slots stay ghosts: an atom that cannot exist is not "missing".
    from fampnn.data.data import get_rc_tensor

    exists = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, entry.aatype)
    assert float((vetoed * (1 - exists)).sum()) == 0.0


@needs_masks
@needs_mmcif
def test_the_veto_removes_exactly_the_unsupervised_sidechains(masks):
    """1ubq: two partial-occupancy residues, and their side chains are present.

    The per-atom mask sees nothing wrong with them -- that is the whole point of
    the per-residue mask.
    """
    from fampnn.data.data import get_rc_tensor

    from fampnn.data import residue_constants as rc

    entry = P.read_entry("1ubq", masks=masks)
    base = P.missing_atom_mask_from_presence(entry.aatype, entry.atom_mask)
    sidechain = list(rc.non_bb_idxs)
    assert float(base[:, sidechain].sum()) == 0.0, "1ubq has no absent side-chain atoms"

    vetoed = P.veto_unsupervised_sidechains(base, entry.supervise, aatype=entry.aatype)
    exists = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, entry.aatype)[:, sidechain]
    expected = (exists * (~entry.supervise).float()[:, None]).sum()
    assert float(vetoed[:, sidechain].sum()) == pytest.approx(float(expected))
    assert float(expected) > 0, "the veto must actually remove something here"


@needs_masks
@needs_mmcif
def test_glycine_costs_nothing(masks):
    """101m's unsupervised residues are all Gly, which has no side chain to lose."""
    from fampnn.data import residue_constants as rc

    entry = P.read_entry("101m", masks=masks)
    reasons = masks.reasons("101m")
    assert reasons["NO_SIDECHAIN"] == int((~entry.supervise).sum())
    base = P.missing_atom_mask_from_presence(entry.aatype, entry.atom_mask)
    vetoed = P.veto_unsupervised_sidechains(base, entry.supervise, aatype=entry.aatype)
    assert torch.equal(vetoed[:, list(rc.non_bb_idxs)], base[:, list(rc.non_bb_idxs)])


@needs_masks
@needs_mmcif
def test_training_batch_carries_the_required_keys(masks):
    from pxf.train.step import REQUIRED_KEYS

    entry = P.read_entry("1ubq", masks=masks)
    item = P.training_batch(entry)
    for key in REQUIRED_KEYS:
        assert key in item, key
    assert item["x"].shape == (76, 37, 3)
    assert float(item["seq_mask"].sum()) == 76


@needs_masks
@needs_mmcif
def test_supervision_can_be_ablated_but_not_by_accident(masks):
    entry = P.read_entry("1ubq", masks=masks)
    with_veto = P.training_batch(entry, apply_supervision=True)
    without = P.training_batch(entry, apply_supervision=False)
    assert float(with_veto["missing_atom_mask"].sum()) > float(
        without["missing_atom_mask"].sum()
    )
    unmasked = P.read_entry("1ubq")
    with pytest.raises(ValueError, match="without a mask set"):
        P.training_batch(unmasked, apply_supervision=True)


@needs_masks
@needs_mmcif
def test_the_dataset_yields_fixed_size_examples_the_step_accepts(masks):
    from pxf.train.data import collate
    from pxf.train.step import REQUIRED_KEYS

    dataset = P.ProtenixSideChainDataset(list(TARGETS), masks=masks, crop_size=64, seed=0)
    batch = collate([dataset[0], dataset[1]])
    for key in REQUIRED_KEYS:
        assert batch[key].shape[:2] == (2, 64), key
    assert batch["name"] == ["1ubq", "101m"]


@needs_masks
def test_the_dataset_refuses_ids_with_no_mask(masks):
    """The eval split has none, so this is the mistake most likely to be made."""
    with pytest.raises(ValueError, match="no mask"):
        P.ProtenixSideChainDataset(["1ubq", "7w6z"], masks=masks)


@needs_masks
def test_supervision_without_a_mask_set_is_refused():
    with pytest.raises(ValueError, match="needs a SideChainMaskSet"):
        P.ProtenixSideChainDataset(["1ubq"], masks=None, apply_supervision=True)


@needs_masks
def test_identity_records_what_the_run_trained_against(masks):
    import json

    record = masks.identity()
    assert json.loads(json.dumps(record))["keep_column"] == "keep_fampnn"
    # strictB: ZERO_OCC|PARTIAL_OCC|ALTLOC_TIE|EXTREME_B|CHIRALITY_BAD|BOND_OUTLIER
    assert record["blockers"] == 4 | 8 | 32 | 64 | 256 | 512 == 876
    assert record["dropped_extreme_b"] is True


def test_ids_from_index_reads_both_splits():
    """The eval and training indices, and the disjointness that makes the split."""
    indices = P.DEFAULT_MMCIF_DIR.parent / "indices"
    eval_csv = indices / "recentPDB_low_homology_maxtoken1536.csv"
    if not eval_csv.is_file():
        pytest.skip(f"no {eval_csv}")
    eval_ids = P.ids_from_index(eval_csv)
    assert len(eval_ids) == 1818

    train_gz = indices / (
        "weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz"
    )
    if not train_gz.is_file():
        pytest.skip(f"no {train_gz}")
    train_ids = P.ids_from_index(train_gz)
    assert not set(eval_ids) & set(train_ids), "temporal split must be disjoint"
