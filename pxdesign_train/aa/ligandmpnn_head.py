"""Strictly loaded official LigandMPNN sequence network; no packer calls.

The co-design cycle needs p(s_i | visible) for EVERY queried position out of one
forward, with queried positions mutually invisible -- that is the contract
`codesign.decode()` relies on when it reads a block of logits and commits it.

FaMPNN satisfies that contract natively: it is a masked denoiser, so a hidden
position is literally X and cannot see another hidden position. LigandMPNN
cannot. Its decoder is ORDER-conditioned: `order_mask_backward[q, p] = 1` iff p
is decoded before q, and such a p contributes `W_s(S_true[p])` -- its TRUE
residue. No official entry point produces the mask this loop needs:

  * `score(use_sequence=True)` derives the order from `chain_mask`, so every
    designed position also sees the native residue of the designed positions
    that happen to precede it. Used as an AA head, the cross-entropy would read
    its own label off its neighbours and recovery would be meaningless.
  * `score(use_sequence=False)` drops sequence context entirely, which throws
    away the committed blocks the cycle just decoded.
  * `single_aa_score` is one position per forward -- L encoder passes per block.

So `forward()` below is `ProteinMPNN.score()`'s decoder body verbatim, with one
substitution: upstream's permutation-derived `order_mask_backward` is kept for
the VISIBLE rows and replaced, on the queried rows only, by the visible set.
For a queried q that is the mask of the decoding order "all visible first, then
q" -- inside the distribution LigandMPNN was trained on, since training draws a
uniformly random order and so teaches every position to be predicted from an
arbitrary subset of the others. What it is not is a permutation: every queried
position gets that same predecessor set, which is what makes one forward enough
and what keeps queried positions from seeing each other.

Leaving the visible rows alone matters and is not cosmetic. The decoder is
three rounds of message passing, so a queried node reads its neighbours'
representations; overriding every row (each node attending to all visible)
gives those context nodes a view no permutation could produce and shifts the
queried logits by ~1e-2 on the released architecture. `test_single_query_
matches_upstream_score` is exact only with upstream's prefix order intact.

Nothing here is a reimplementation of the network. Featurisation, encoder and
decoder layers are upstream's, called on upstream's feature dict.
"""
from pathlib import Path
import hashlib
import subprocess
import torch
from torch import nn
from .atom_mapping import AA_ORDER, BB37, MAPPING_VERSION

UPSTREAM_REVISION = "26ec57ac976ade5379920dbd43c7f97a91cf82de"
# data_utils.restype_int_to_str, which we cannot import: it pulls in `prody`
# for PDB parsing that this head never performs. Pinned here and checked
# against the upstream source text by test_ligandmpnn_alphabet_matches_upstream.
LIGANDMPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
# Upstream Atom37 (data_utils.py, the `else` branch of `atom_types`) is the same
# order as ours, so `denoised_coords` needs no permutation -- only this check.
ATOM37_HEAD = ("N", "CA", "C", "CB", "O")


def _canonical_to_upstream() -> torch.Tensor:
    """Proteo-AA canonical index -> LigandMPNN index, with 20 = X fixed."""
    return torch.tensor([LIGANDMPNN_ALPHABET.index(a) for a in AA_ORDER] + [20])


class LigandMPNNHead(nn.Module):
    def __init__(self, checkpoint_path, source_root, source_revision=UPSTREAM_REVISION,
                 use_side_chain_context=True):
        super().__init__()
        root = Path(source_root).resolve()
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if actual != source_revision or actual != UPSTREAM_REVISION:
            raise ValueError(f"LigandMPNN source revision mismatch: {actual}")
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
            raise ValueError("LigandMPNN tracked source has local modifications")
        model_utils = _import_upstream(root)
        path = Path(checkpoint_path).resolve()
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # run.py: every hyper-parameter but the side-chain switch comes from the
        # checkpoint. Reading them from a config instead is how a silently
        # mis-shaped model gets built from weights that happen to load.
        config = dict(node_features=128, edge_features=128, hidden_dim=128,
            num_encoder_layers=3, num_decoder_layers=3,
            k_neighbors=int(checkpoint["num_edges"]),
            atom_context_num=int(checkpoint["atom_context_num"]),
            model_type="ligand_mpnn",
            ligand_mpnn_use_side_chain_context=bool(use_side_chain_context))
        network = model_utils.ProteinMPNN(**config, augment_eps=0.0, dropout=0.0)
        network.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self._adopt(network, model_utils, config,
                    dict(backend="ligandmpnn", upstream_revision=actual,
                         checkpoint_sha256=checksum, model_config=config,
                         mapping_version=MAPPING_VERSION,
                         decode_policy="visible-set-attend-one-forward-v1"))

    @classmethod
    def for_network(cls, network, model_utils, identity=None):
        """Wrap an already-built upstream network.

        For tests that check masking, parity and gradient -- properties of the
        decode path, not of the released weights. It deliberately skips the
        revision/clean-tree/checksum gates that `__init__` exists to enforce,
        so it must never be reachable from a training or evaluation entry point.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        config = dict(k_neighbors=None, atom_context_num=network.features.atom_context_num,
                      model_type="ligand_mpnn",
                      ligand_mpnn_use_side_chain_context=network.features.use_side_chains)
        self._adopt(network, model_utils, config, identity or dict(backend="ligandmpnn-unverified"))
        return self

    def _adopt(self, network, model_utils, config, identity):
        # augment_eps perturbs BOTH backbone and ligand coordinates inside
        # featurisation (model_utils.py, ProteinFeaturesLigand.forward). Under a
        # coordinate-gradient objective that is noise injected between the
        # backbone and its own loss, and it breaks same-seed arm comparisons.
        # run.py sets 0.0 for inference; a training head must too.
        if float(network.features.augment_eps) != 0.0:
            raise ValueError("LigandMPNN head requires augment_eps=0")
        self.sequence_network = network
        self.gather_nodes = model_utils.gather_nodes
        self.cat_neighbors_nodes = model_utils.cat_neighbors_nodes
        self.atom_context_num = int(config["atom_context_num"])
        self.use_side_chain_context = bool(config["ligand_mpnn_use_side_chain_context"])
        self.register_buffer("canonical_to_upstream", _canonical_to_upstream())
        self.register_buffer("canonical_indices", _canonical_to_upstream()[:20])
        self.identity = identity

    def forward(self, *, denoised_coords, aatype_noised, seq_mask,
                atom_mask_noised, residue_index, chain_encoding, decoding_randn=None,
                ligand_xyz=None, ligand_element=None, ligand_mask=None, **_unused):
        lead = denoised_coords.shape[:-3]
        length = denoised_coords.shape[-3]
        if len(lead) != 2:
            raise ValueError("LigandMPNN boundary requires [batch,sample,residue,37,3]")
        flat = lambda x, *trailing: x.reshape(-1, length, *trailing)
        coords = flat(denoised_coords.float(), 37, 3)
        atom_mask = flat(atom_mask_noised.float(), 37)
        aatype = flat(aatype_noised.long())
        mask = flat(seq_mask.float())
        rows = coords.shape[0]

        # `aatype_noised` is already X wherever `masking.visible_input` hid an
        # identity, so this is the whole visible set -- including a fixed
        # non-standard residue, whose identity we genuinely do not know either.
        known = (aatype != 20) & mask.bool()
        sequence = self.canonical_to_upstream.to(aatype.device)[aatype]
        # chain_mask is upstream's "this position is being designed": it gates
        # the side-chain context (model_utils.py, `xyz_37_m * (1 - chain_mask)`).
        # `atom_mask_noised` has already cleared hidden side chains, so this is
        # the same selection stated in upstream's own terms, not a second one.
        chain_mask = (~known).to(coords.dtype)

        feature_dict = dict(
            X=coords[:, :, list(BB37), :], S=sequence, mask=mask,
            chain_mask=chain_mask, R_idx=flat(residue_index.long()),
            chain_labels=flat(chain_encoding).float(),
            xyz_37=coords, xyz_37_m=atom_mask,
            **self._ligand_context(rows, length, coords, ligand_xyz,
                                   ligand_element, ligand_mask))

        # Geometry, RBF and the periodic-table one-hots are fp32 upstream;
        # autocast would silently halve the distance features the whole
        # coordinate gradient rides on.
        with torch.autocast(device_type=coords.device.type, enabled=False):
            logits = self._decode_against_visible(feature_dict, known, decoding_randn)
        logits = logits.reshape(*lead, length, 21)
        return logits.index_select(-1, self.canonical_indices), None

    def _ligand_context(self, rows, length, coords, xyz, element, atom_mask):
        """Per-residue ligand atom context.

        `CoDesignState.aa_input()` does not carry ligand atoms yet, so the
        default is an all-masked channel. That is not a silent degradation of
        the protein path: upstream concatenates side-chain atoms in FRONT of
        these and keeps the `atom_context_num` closest to Cb, pushing masked
        entries to distance 10000 first (model_utils.py, `Cb_Y_distances_adjusted`),
        so an all-masked channel is only selected when there is nothing else and
        is masked out downstream regardless.
        """
        if xyz is None:
            shape = (rows, length, self.atom_context_num)
            return dict(Y=coords.new_zeros(*shape, 3),
                        Y_t=coords.new_zeros(shape),
                        Y_m=coords.new_zeros(shape))
        return dict(Y=xyz.reshape(rows, length, -1, 3).float(),
                    Y_t=element.reshape(rows, length, -1).float(),
                    Y_m=atom_mask.reshape(rows, length, -1).float())

    def _decode_against_visible(self, feature_dict, known, randn=None):
        """`ProteinMPNN.score()`'s decoder, with queried rows made mutually blind.

        Upstream builds `mask_attend` by gathering a permutation-derived
        [B,L,L] matrix whose (q, p) entry says "p decoded before q". Only the
        QUERIED rows are wrong for this loop, so only those are replaced -- by
        the visible set itself. Visible rows keep upstream's own permutation.

        Overriding every row instead (each node attending to all visible) looks
        equivalent and is not: the decoder is three rounds of message passing,
        so a queried node reads its neighbours' representations, and those are
        computed under their own predecessor sets. Giving the context nodes a
        fuller view than any permutation would moves them off the distribution
        the network was trained on, and measurably changes the queried logits
        (~1e-2 on the released architecture). Keeping upstream's prefix order
        is what makes `test_single_query_matches_upstream_score` exact.
        """
        network = self.sequence_network
        h_V, h_E, E_idx = network.encode(feature_dict)
        mask = feature_dict["mask"]
        rows, length = known.shape
        device = known.device
        if randn is None:
            # Upstream draws |randn| to shuffle within a chain_mask level. A
            # constant leaves argsort stable, i.e. residue order among visible
            # positions: reproducible across the blocks of one decode and
            # across the feedback arms of one seed.
            randn = torch.ones(rows, length, device=device)
        # score(): "numbers will be smaller for places where chain_M = 0.0".
        decoding_order = torch.argsort((feature_dict["chain_mask"] + 0.0001) * randn.abs())
        permutation = torch.nn.functional.one_hot(decoding_order, num_classes=length).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(length, length, device=device))),
            permutation, permutation)
        # Queried q: predecessors are exactly the visible set, nothing else.
        order_mask_backward = torch.where(
            known[..., None], order_mask_backward, known[:, None, :].to(order_mask_backward.dtype))
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view(*mask.shape, 1, 1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_S = network.W_s(feature_dict["S"])
        h_ES = self.cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = self.cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = self.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in network.decoder_layers:
            h_ESV = self.cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
        return network.W_out(h_V)


def _import_upstream(root: Path):
    """Import upstream `model_utils` from the pinned checkout, by path.

    `data_utils` is deliberately NOT imported: it pulls in `prody` for PDB
    parsing this head never does, and nothing it defines is used here.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("ligandmpnn_model_utils",
                                                  root / "model_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
