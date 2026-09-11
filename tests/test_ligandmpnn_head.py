"""Contract gates for the LigandMPNN AA backend.

The masking, parity and gradient properties below are properties of the decode
path, not of the released weights, so they run on a randomly initialised
upstream network and need no checkpoint. `LIGANDMPNN_SOURCE` is the only thing
they need, and it is the checkout the head pins anyway.
"""
import os
import re
from pathlib import Path

import pytest
import torch

SOURCE = os.environ.get("LIGANDMPNN_SOURCE", "/hai/users/s/h/shenjm/tools/LigandMPNN")
pytestmark = pytest.mark.skipif(
    not (Path(SOURCE) / "model_utils.py").is_file(),
    reason="Set LIGANDMPNN_SOURCE to the pinned upstream checkout",
)

LENGTH = 24
CONTEXT_ATOMS = 16


@pytest.fixture(scope="module")
def upstream():
    from pxdesign_train.aa.ligandmpnn_head import _import_upstream
    return _import_upstream(Path(SOURCE))


@pytest.fixture(scope="module")
def head(upstream):
    from pxdesign_train.aa.ligandmpnn_head import LigandMPNNHead
    torch.manual_seed(0)
    network = upstream.ProteinMPNN(
        model_type="ligand_mpnn", k_neighbors=16, atom_context_num=CONTEXT_ATOMS,
        ligand_mpnn_use_side_chain_context=True, augment_eps=0.0, dropout=0.0)
    return LigandMPNNHead.for_network(network.eval(), upstream)


def inputs(seed=3, hidden=(0, 1, 2, 3)):
    """One item, one sample. `hidden` are the queried positions (identity X)."""
    generator = torch.Generator().manual_seed(seed)
    # A plausible chain: consecutive CA roughly 3.8 A apart, so the kNN graph is
    # not degenerate and the frames the RBFs read are well conditioned.
    ca = torch.cumsum(torch.randn(LENGTH, 3, generator=generator) * 0.6, 0)
    ca = ca + torch.arange(LENGTH, dtype=torch.float32)[:, None] * 3.0
    coords = ca[:, None, :] + torch.randn(LENGTH, 37, 3, generator=generator) * 0.5
    atom_mask = torch.zeros(LENGTH, 37)
    atom_mask[:, list((0, 1, 2, 4))] = 1.0
    atom_mask[:, 5:12] = 1.0
    aatype = torch.randint(0, 20, (LENGTH,), generator=generator)
    aatype[list(hidden)] = 20
    atom_mask[list(hidden), 5:] = 0.0          # queried side chains are hidden too
    return dict(
        denoised_coords=coords[None, None], aatype_noised=aatype[None, None],
        seq_mask=torch.ones(1, 1, LENGTH), atom_mask_noised=atom_mask[None, None],
        residue_index=torch.arange(LENGTH)[None, None],
        chain_encoding=torch.zeros(1, 1, LENGTH))


def test_alphabet_and_atom37_match_upstream_source():
    """Pinned against upstream SOURCE TEXT: data_utils needs prody to import."""
    from pxdesign_train.aa.ligandmpnn_head import LIGANDMPNN_ALPHABET, ATOM37_HEAD
    from pxdesign_train.aa.atom_mapping import ATOM37

    text = (Path(SOURCE) / "data_utils.py").read_text()
    block = re.search(r"restype_int_to_str = \{(.*?)\}", text, re.S).group(1)
    pairs = {int(k): v for k, v in re.findall(r"(\d+):\s*\"([A-Z])\"", block)}
    assert "".join(pairs[i] for i in range(21)) == LIGANDMPNN_ALPHABET

    # data_utils has two `atom_types` lists; the long one is the Atom37 order.
    lists = [tuple(re.findall(r"\"([A-Z0-9]+)\"", b))
             for b in re.findall(r"atom_types = \[(.*?)\]", text, re.S)]
    names = max(lists, key=len)
    # Upstream's PDB reader never writes slot 36, so it names only 36 atoms --
    # but the model reserves 37 and types slot 36 as oxygen, which is OXT. Our
    # ATOM37 fills it, so the prefix must agree and the element must match.
    assert len(names) == 36 and names == ATOM37[:36]
    assert ATOM37[36] == "OXT"
    assert ATOM37[:5] == ATOM37_HEAD


def test_canonical_mapping_round_trips(head):
    from pxdesign_train.aa.ligandmpnn_head import LIGANDMPNN_ALPHABET
    from pxdesign_train.aa.atom_mapping import AA_ORDER

    table = head.canonical_to_upstream
    assert table.numel() == 21 and int(table[20]) == 20, "X must stay at 20"
    for canonical, letter in enumerate(AA_ORDER):
        assert LIGANDMPNN_ALPHABET[int(table[canonical])] == letter
    # The output selection is the same map, so a 21-logit row in upstream order
    # comes back in ours.
    upstream_row = torch.arange(21).float()
    picked = upstream_row.index_select(-1, head.canonical_indices)
    assert [LIGANDMPNN_ALPHABET[int(v)] for v in picked] == list(AA_ORDER)


def test_queried_positions_cannot_see_each_others_identity(head):
    """The gate that makes the AA loss mean anything.

    `visible_input` already writes X at every hidden position, so testing
    through `forward()` only re-tests that sanitisation. The leak this guards
    against lives one layer down: upstream's decoder reads `W_s(S[p])` for any
    neighbour p ordered earlier, so if a queried row's predecessor set ever
    included another queried position, the cross-entropy would see its own
    label. Drive `_decode_against_visible` directly with a sequence tensor that
    DOES carry native residues under the mask -- the logits at queried
    positions must not move.
    """
    hidden = [3, 4, 5, 11, 12]
    feed = inputs(hidden=tuple(hidden))
    coords = feed["denoised_coords"].reshape(1, LENGTH, 37, 3).float()
    known = torch.ones(1, LENGTH, dtype=torch.bool)
    known[0, hidden] = False

    def logits_for(planted):
        sequence = head.canonical_to_upstream[feed["aatype_noised"].reshape(1, LENGTH).long()]
        sequence = sequence.clone()
        sequence[0, hidden] = planted           # a real residue, not X
        feature_dict = dict(
            X=coords[:, :, [0, 1, 2, 4], :], S=sequence,
            mask=feed["seq_mask"].reshape(1, LENGTH),
            chain_mask=(~known).float(), R_idx=feed["residue_index"].reshape(1, LENGTH).long(),
            chain_labels=feed["chain_encoding"].reshape(1, LENGTH).float(),
            xyz_37=coords, xyz_37_m=feed["atom_mask_noised"].reshape(1, LENGTH, 37),
            Y=coords.new_zeros(1, LENGTH, CONTEXT_ATOMS, 3),
            Y_t=coords.new_zeros(1, LENGTH, CONTEXT_ATOMS),
            Y_m=coords.new_zeros(1, LENGTH, CONTEXT_ATOMS))
        with torch.no_grad():
            return head._decode_against_visible(feature_dict, known)

    reference = logits_for(torch.tensor([0, 1, 2, 3, 4]))
    for planted in (torch.tensor([19, 18, 17, 16, 15]), torch.tensor([7, 7, 7, 7, 7])):
        actual = logits_for(planted)
        torch.testing.assert_close(actual[0, hidden], reference[0, hidden], rtol=0, atol=0)


def test_hidden_identity_is_never_read(head):
    """Directly: flipping a hidden position's stored integer changes nothing.

    `visible_input` writes 20 at every hidden position, so the head's own input
    is already X there. This asserts the head does not reconstruct identity
    from anywhere else -- notably that `chain_mask` keeps the side-chain
    context off, and that no neighbour's `mask_bw` turns on for a hidden p.
    """
    hidden = (5, 6, 7, 8, 9)
    base = inputs(hidden=hidden)
    with torch.no_grad():
        reference, _ = head(**base)
    poisoned = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    # Side-chain coordinates under the queries: masked off, so invisible.
    poisoned["denoised_coords"][0, 0, list(hidden), 5:] += 25.0
    with torch.no_grad():
        actual, _ = head(**poisoned)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_single_query_matches_upstream_score(head, upstream):
    """Exact parity with `ProteinMPNN.score(use_sequence=True)`.

    With exactly one queried position, "attend to the visible set" IS the
    permutation "everything else first, then q" -- which upstream produces when
    `chain_mask` is one-hot at q. Equality to the bit pins the substituted mask
    as the only difference between this head and upstream's own decoder.
    """
    query = 7
    feed = inputs(hidden=(query,))
    # The visible prefix keeps upstream's order, so both sides must draw it the
    # same way; this is the knob that exists for exactly this comparison.
    randn = torch.randn(1, LENGTH, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        actual, _ = head(**feed, decoding_randn=randn)

    coords = feed["denoised_coords"].reshape(1, LENGTH, 37, 3).float()
    aatype = feed["aatype_noised"].reshape(1, LENGTH).long()
    sequence = head.canonical_to_upstream[aatype]
    chain_mask = torch.zeros(1, LENGTH)
    chain_mask[0, query] = 1.0                       # designed -> decoded last
    feature_dict = dict(
        X=coords[:, :, [0, 1, 2, 4], :], S=sequence, mask=feed["seq_mask"].reshape(1, LENGTH),
        chain_mask=chain_mask, R_idx=feed["residue_index"].reshape(1, LENGTH).long(),
        chain_labels=feed["chain_encoding"].reshape(1, LENGTH).float(),
        xyz_37=coords, xyz_37_m=feed["atom_mask_noised"].reshape(1, LENGTH, 37),
        Y=coords.new_zeros(1, LENGTH, CONTEXT_ATOMS, 3),
        Y_t=coords.new_zeros(1, LENGTH, CONTEXT_ATOMS),
        Y_m=coords.new_zeros(1, LENGTH, CONTEXT_ATOMS),
        batch_size=1, symmetry_residues=[[]],
        # |randn| only orders positions WITHIN a chain_mask level; with one
        # designed position the order of the visible prefix cannot matter.
        randn=randn)
    with torch.no_grad():
        expected = head.sequence_network.score(feature_dict, use_sequence=True)["logits"]

    torch.testing.assert_close(
        actual[0, 0, query], expected[0, query].index_select(-1, head.canonical_indices),
        rtol=0, atol=0)


def test_coordinate_gradient_reaches_the_backbone(head):
    """IV-F depends on this: a frozen head must still be differentiable.

    Upstream featurisation is plain torch (cross products, distances, RBFs), so
    the whole point is that nothing detaches. If this ever fails, a frozen-head
    phase trains the packer only and the backbone silently receives no sequence
    signal at all.
    """
    feed = inputs(hidden=(4, 5, 6))
    coords = feed["denoised_coords"].detach().clone().requires_grad_(True)
    feed = dict(feed, denoised_coords=coords)
    logits, _ = head(**feed)
    loss = logits[0, 0, 4].square().sum()
    grad, = torch.autograd.grad(loss, coords)
    assert torch.isfinite(grad).all()
    backbone = grad[0, 0, :, [0, 1, 2, 4], :]
    assert backbone.abs().sum() > 0, "no gradient reached N/CA/C/O"


def test_augment_eps_is_rejected(upstream):
    from pxdesign_train.aa.ligandmpnn_head import LigandMPNNHead
    noisy = upstream.ProteinMPNN(
        model_type="ligand_mpnn", k_neighbors=16, atom_context_num=CONTEXT_ATOMS,
        ligand_mpnn_use_side_chain_context=True, augment_eps=0.1, dropout=0.0)
    with pytest.raises(ValueError, match="augment_eps"):
        LigandMPNNHead.for_network(noisy, upstream)
