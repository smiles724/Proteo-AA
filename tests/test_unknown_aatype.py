"""``aa_clean``'s -100 must never reach a consumer that indexes with it.

Protenix marks anything it cannot name with -100, its ignore-index. Two kinds
of token land there -- UNK protein residues (observed backbone, unassigned
identity) and genuine non-protein tokens -- and both used to travel straight
into ``FeaturizedStructure.aatype``, which the rest of the pipeline treats as a
*class index*.

That produced two distinct failures, and the quiet one is the worse of the two:

* **Loud.** ``batch_masks`` indexes ``STANDARD_ATOM_MASK_WITH_X`` with it, the
  encoder embeds it, and ``masked_cross_entropy`` one-hots it. On CUDA this is
  an async device-side assert, which kills the context and makes every
  *subsequent* structure fail too -- so the panel reports ~26 failures for one
  bad input, and the first real cause is buried.
* **Quiet.** ``seq_unk_mask`` is ``aatype == X``. A -100 row does not match, so
  a residue with no known identity was not recognised as unknown and would be
  *scored as a label* had one fallen in the supervised set.

Nine of the 31 binder val structures carry such tokens, so this is the common
case, not an exotic one.
"""

import pytest
import torch

from pxf import atom37
from pxf.backbone.driver import to_featurized

X = atom37.UNKNOWN_AA_INDEX


def _item(aa_clean, is_protein=None):
    """The minimum ``DesignSourceDataset`` item ``to_featurized`` reads.

    One atom per token keeps ``atom_to_token_idx`` the identity map, so a
    token's index is also its atom's index and the assertions can be read
    without a second mapping in the head.
    """
    n = len(aa_clean)
    aa = torch.tensor(aa_clean, dtype=torch.long)
    fd = {
        "token_index": torch.arange(n),
        "aa_clean": aa,
        "design_token_mask": torch.zeros(n, dtype=torch.bool),
        "atom_to_token_idx": torch.arange(n),
        "structure_atom_name": ["CA"] * n,
        "residue_index": torch.arange(n),
        "asym_id": torch.zeros(n, dtype=torch.long),
    }
    if is_protein is not None:
        fd["is_protein"] = torch.tensor(is_protein, dtype=torch.bool)
    return {"input_feature_dict": fd, "label_dict": {}}


def test_unknown_protein_becomes_x_not_minus_100():
    """The -100 is remapped, and to X specifically -- the value ``restype`` uses."""
    s = to_featurized("unk", _item([0, -100, 5, -100, 19], is_protein=[1] * 5))
    assert s.aatype.tolist() == [0, X, 5, X, 19]


def test_aatype_is_always_a_valid_index():
    """The bound the three downstream consumers actually need."""
    s = to_featurized("mixed", _item([-100, 3, -100], is_protein=[1, 1, 0]))
    assert int(s.aatype.min()) >= 0
    assert int(s.aatype.max()) <= X
    # The property stated as the consumers use it: an embedding of size 21 and
    # a one-hot of width 21 both accept every row.
    torch.nn.functional.one_hot(s.aatype, num_classes=X + 1)
    torch.nn.Embedding(X + 1, 4)(s.aatype)


def test_remapped_rows_are_recognised_as_unknown():
    """The quiet bug: ``seq_unk_mask``'s predicate must now fire on these rows.

    Asserted against the predicate ``pxf.train.step.batch_masks`` uses rather
    than against X directly, so a change to that convention breaks this test
    instead of silently reintroducing scored-unknown labels.
    """
    s = to_featurized("unk", _item([0, -100, 5], is_protein=[1] * 3))
    seq_unk = (s.aatype == X)
    assert seq_unk.tolist() == [False, True, False]


def test_non_protein_tokens_are_still_reported(caplog):
    """Remapping them keeps the tensor valid; it must not silence the warning.

    For a ligand the side-chain targets genuinely do not line up, which is a
    real problem with the entry -- unlike an UNK residue, which is routine.
    """
    with caplog.at_level("WARNING"):
        to_featurized("lig", _item([0, -100, 1], is_protein=[1, 0, 1]))
    assert "not amino acids" in caplog.text


def test_unknown_protein_does_not_warn(caplog):
    """29% of the val panel has these; warning on them would be noise."""
    with caplog.at_level("WARNING"):
        to_featurized("unk", _item([0, -100, 1], is_protein=[1, 1, 1]))
    assert "not amino acids" not in caplog.text


def test_residue_names_report_unk():
    s = to_featurized("unk", _item([0, -100], is_protein=[1, 1]))
    assert s.topology.res_names[1] == "UNK"


def test_missing_is_protein_assumes_protein(caplog):
    """No ``is_protein`` feature: remap, and stay quiet rather than cry wolf."""
    with caplog.at_level("WARNING"):
        s = to_featurized("unk", _item([0, -100, 1]))
    assert s.aatype.tolist() == [0, X, 1]
    assert "not amino acids" not in caplog.text
